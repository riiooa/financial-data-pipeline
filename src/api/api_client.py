import os
import time
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from dotenv import load_dotenv

# Import utilitas kustom kita
from src.utils.logging_utils import retry, timer, StructuredLogger, setup_logging

# Load environment variables
load_dotenv()

# Inisialisasi Logging Kustom
setup_logging()
logger = StructuredLogger('api')

class AlphaVantageClient:
    """
    Client for Alpha Vantage API dengan built-in rate limiting, 
    custom error handling, dan structured logging.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/raw"):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            logger.error("API Key missing")
            raise ValueError("API key required: Set ALPHA_VANTAGE_API_KEY in .env file")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting tracking
        self.last_request_time = 0
        self.min_request_interval = 12  # 5 requests per minute = 1 request tiap 12 detik
        self.request_count_today = 0
        self.daily_limit = 500

        logger.info(f"Alpha Vantage client initialized", key_prefix=self.api_key[:8])

    def _respect_rate_limit(self):
        """Pastikan kita tidak melebihi batas request per menit"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            # Menggunakan structured log
            logger.info("Rate limiting in effect", sleep_seconds=round(sleep_time, 2))
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _get_cache_path(self, function: str, symbol: str, **kwargs) -> Path:
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
        filename = f"{function}_{symbol}_{param_str}.json".replace(" ", "_").replace("/", "_")
        return self.cache_dir / filename 

    def _load_from_cache(self, cache_path: Path, max_age_hours: int = 24) -> Optional[Dict]:
        if cache_path.exists():
            file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if file_age < timedelta(hours=max_age_hours):
                with open(cache_path, "r") as f:
                    logger.info("Cache hit", path=str(cache_path))
                    return json.load(f)
        return None

    def _save_to_cache(self, cache_path: Path, data: Dict):
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Cache saved", path=str(cache_path))

    # MENGGUNAKAN DECORATOR KUSTOM BARU
    @retry(max_attempts=3, delay=2.0, backoff=2.0, exceptions=(requests.exceptions.RequestException,))
    def _make_request(self, params: Dict) -> Dict:
        """Enhanced dengan custom retry decorator dan timer"""
        self._respect_rate_limit()

        params['apikey'] = self.api_key
        params['datatype'] = 'json'

        # MENGGUNAKAN TIMER UNTUK MENGUKUR KECEPATAN API
        with timer(f"API request: {params.get('symbol')}"):
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Handle API-specific messages
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if "Note" in data:  # Alpha Vantage specific rate limit message
                logger.warning("API Rate Limit reached (Note detected)", note=data['Note'])
                time.sleep(60) 
                return self._make_request(params)
            
            return data

    def get_daily_data(self, symbol: str, outputsize: str = 'compact', use_cache: bool = True) -> pd.DataFrame:
        params = {'function': 'TIME_SERIES_DAILY', 'symbol': symbol, 'outputsize': outputsize}
        cache_path = self._get_cache_path('daily', symbol, outputsize=outputsize)

        if use_cache:
            data = self._load_from_cache(cache_path) or self._make_request(params)
            if data and not cache_path.exists(): self._save_to_cache(cache_path, data)
        else:
            data = self._make_request(params)

        # Parsing DataFrame (Logika tetap sama)
        time_series_key = 'Time Series (Daily)'
        if time_series_key not in data:
            api_message = data.get('Note') or data.get('Information') or data.get('Error Message')
            logger.error(f"API Reject: {api_message}", symbol=symbol)
            raise ValueError(f"No time series data found for {symbol}. Reason: {api_message}")
        
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df.columns = [col.split('. ')[1] for col in df.columns]
        df = df.apply(pd.to_numeric).sort_index()
        df['symbol'] = symbol
        
        logger.info("Data retrieval successful", symbol=symbol, rows=len(df))
        return df

    def get_quote(self, symbol: str, use_cache: bool = True) -> Dict:
        """
        Fetch current quote for a symbol

        Args:
            symbol: Stock symbol
            use_cache: Whether to use cached data

        Returns:
            Dictionary with current quote data
        """
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol
        }

        cache_path = self._get_cache_path('quote', symbol)

        if use_cache:
            cached_data = self._load_from_cache(cache_path, max_age_hours=1) # 1 hour cache for quotes

            if cached_data:
                data = cached_data
            else:
                data = self._make_request(params)
                self._save_to_cache(cache_path, data)
        else:
            data = self._make_request(params)

        quote_key = 'Global Quote'
        if quote_key not in data:
            raise ValueError(f"No quete data found for {symbol}")
        
        return data[quote_key]
    
    def get_multiple_symbols(self, symbols: List[str], outputsize: str = 'compact') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols

        Args:
            symbols: List of stock symbols
            outputsize: 'compact' or 'full'

        Returns:
            Dictonary mapping symbol to DataFrame
        """
        results = {}
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})")
                df = self.get_daily_data(symbol, outputsize)
                results[symbol] = df

                # Small delay between symbols to be nice to the API
                if i < len(symbols) - 1:
                    time.sleep(5)

            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = None

        return results
    

def test_client():
    """Test function to verify API client works"""
    client = AlphaVantageClient()

    # Test single symbol
    print("\n Testing single symbol (AAPL)...")
    aapl_data = client.get_daily_data('AAPL', outputsize='compact')
    print(f"Retrieved {len(aapl_data)} rows")
    print(aapl_data.head())

    # Test multiple symbols
    print("\n Testing multiple symbols")
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    results = client.get_multiple_symbols(symbols, outputsize='compact')

    for symbol, df in results.items():
        if df is not None:
            print(f"\n{symbol}: {len(df)} rows")
            print(f"Data range: {df.index.min()} to {df.index.max}")

    return results


if __name__ == "__main__":
    data = test_client()