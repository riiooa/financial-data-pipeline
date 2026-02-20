"""
API Enhancements - New features for API client
Extends the base AlphaVantageClient with batch processing and historical data.
"""

from typing import List, Dict
import pandas as pd
from src.api.api_client import AlphaVantageClient, logger

class EnhancedAlphaVantageClient(AlphaVantageClient):
    """Enhanced version with additional features"""

    def get_batch_quotes(self, symbols: List[str]) -> Dict:
        """
        Get quotes for multiple symbols efficiently.
        """
        results = {}
        logger.info(f"Starting batch quotes retrieval for {len(symbols)} symbols")

        for symbol in symbols:
            try:
                quote = self.get_quote(symbol)
                results[symbol] = quote
            except Exception as e:
                logger.error(f"Failed to get quote for {symbol}", error=str(e))
                results[symbol] = None

        return results
    
    def get_historical_data(self, symbol: str, years: int = 5) -> pd.DataFrame:
        """
        Get historical data using 'compact' for free tier compatibility.
        Note: Free tier only provides the last 100 days (~4 months).
        """
        logger.info(f"Fetching historical data (compact) for {symbol}")

        df = self.get_daily_data(symbol, outputsize='compact')

        cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=years)
        df_filtered = df[df.index >= cutoff_date]

        logger.info(f"Filtered historical data", symbol=symbol, rows=len(df_filtered))
        return df_filtered
    
def test_enhancements():
    """Simple test for enhanced features"""
    client = EnhancedAlphaVantageClient()

    # Test Batch
    symbols = ['AAPL', 'MSFT']
    print(f"\nTesting Batch Quotes for: {symbols}")
    quotes = client.get_batch_quotes(symbols)
    print(f"Retrieved {len(quotes)} quotes.")

    # Test Historical
    print(f"\nTesting 2 years Historical for TSLA...")
    try:
        df = client.get_historical_data('TSLA', years=2)
        print(df.tail())
    except Exception as e:
        print(f"Historical test failed: {e}")

if __name__ == "__main__":
    test_enhancements()