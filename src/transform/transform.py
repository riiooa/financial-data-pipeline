"""
Data Tranformation Layer for Financial Data Pipeline
Converts raw OHLC data to analytics-ready format
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'    
)
logger = logging.getLogger(__name__)


class FinancialDataTranfromer:
    """
    Transforms raw OHLC data into analytics-ready format
    Calculates financial metrics and handles data quality issues
    """

    def __init__(self):
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw OHLC data
        Args:  
            df: Raw DataFrame with OHLC data

        Returns:
            Cleaned DataFrame
        """
        logger.info("Strating data Cleaning...") # Menjaga typo 'Strating' sesuai gaya asli
        df_clean = df.copy()

        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df_clean.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Initial shape
        intial_rows = len(df_clean) # Menjaga variabel 'intial' sesuai gaya asli
        logger.info(f"Initial row: {intial_rows}")

        # Remove duplicates (keep last)
        df_clean = df_clean[~df_clean.index.duplicated(keep='last')]

        # Sort by date
        df_clean = df_clean.sort_index()

        # Check for missing dates
        date_range = pd.date_range(start=df_clean.index.min(), end=df_clean.index.max(), freq='D')
        missing_dates = date_range.difference(df_clean.index.tolist())
        if len(missing_dates) > 0:
            logger.warning(f"Missing: {len(missing_dates)} dates")
            logger.debug(f"Missing dates sample: {missing_dates[:5]}")

        # Handle missing values
        for col in df_clean.columns:
            if col in ['open', 'high', 'low', 'close']:
                # Forward fill untuk harga (mengatasi hari libur/missing data)
                df_clean[col] = df_clean[col].ffill()
            elif col == 'volume':
                # Fill missing volume with 0
                df_clean[col] = df_clean[col].fillna(0)
            
        # Check for outlier using IQR method 
        for col in ['close', 'volume']:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR # Using 3 * IQR for extreme outlier
                upper_bound = Q3 + 3 * IQR 

                outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
                if len(outliers) > 0:
                    logger.warning(f"Found {len(outliers)} outliers in {col}")
                    # Cap outliers
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        # Ensure OHLC consistency (low <= close <= high, etc)
        df_clean['low'] = df_clean[['low', 'close']].min(axis=1)
        df_clean['high'] = df_clean[['high', 'close']].max(axis=1)
        df_clean['open'] = df_clean[['open', 'close']].min(axis=1)

        final_row = len(df_clean)
        logger.info(f"Cleaning complete. Rows: {final_row} (remove {intial_rows - final_row} duplicates)")

        return df_clean

    def calculate_returns(self, df: pd.DataFrame) ->  pd.DataFrame:
        """
        Calculate various return metrics

        Args:
            df: DataFrame with OHLC data

        Returns:
            DataFrame with additional return columns
        """
        logger.info("Calculating returns ...")
        df_with_returns = df.copy()

        # Daily returns (percentage change)
        df_with_returns['daily_return'] = df_with_returns['close'].pct_change() * 100

        # Log returns (continuous compounding)
        df_with_returns['log_return'] = np.log(df_with_returns['close'] / df_with_returns['close'].shift(1)) * 100

        # Cumulative returns
        df_with_returns['cumulative_return'] = (1 + df_with_returns['daily_return'] / 100).cumprod() - 1

        # Returns by period
        df_with_returns['return_1d'] = df_with_returns['daily_return']
        df_with_returns['return_5d'] = df_with_returns['close'].pct_change(5) * 100
        df_with_returns['return_20d'] = df_with_returns['close'].pct_change(20) * 100
        df_with_returns['return_252d'] = df_with_returns['close'].pct_change(252) * 100


        logger.info("Returns calculated")
        return df_with_returns
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate rolling volatility

        Args:
            df: DataFrame with returns
            window: Rolling window size

        Returns:
            DataFrame with volatility columns
        """

        logger.info(f" Calculating {window}-day rolling volatility...")
        df_with_vol = df.copy()

        # Historical volatility (standart deviation of returns)
        df_with_vol['volatility'] = df_with_vol['daily_return'].rolling(window=window).std()                
        
        # Annualized volatility (assuming 252 trading days)
        df_with_vol['volatility_annualized'] = df_with_vol['volatility'] * np.sqrt(252)

        # True Range and Average True Range (ATR)
        df_with_vol['tru_range'] = np.maximum(
                df_with_vol['high'] - df_with_vol['low'],
                np.maximum(
                    abs(df_with_vol['high'] - df_with_vol['close'].shift()),
                    abs(df_with_vol['low'] - df_with_vol['close'].shift())
                )         
        )
        df_with_vol['atr'] = df_with_vol['tru_range'].rolling(window=14).mean()

        logger.info("Volatility calculated")
        return df_with_vol

    def calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various moving averages
        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with MA columns
        """
        logger.info("Calculating moving averages...")
        df_with_ma = df.copy()

        # Simple Moving Averages
        df_with_ma['sma_20'] = df_with_ma['close'].rolling(window=20).mean()
        df_with_ma['sma_50'] = df_with_ma['close'].rolling(window=50).mean()
        df_with_ma['sma_200'] = df_with_ma['close'].rolling(window=200).mean()

        # Exponential Moving Averages
        df_with_ma['ema_12'] = df_with_ma['close'].ewm(span=12, adjust=False).mean()
        df_with_ma['ema_26'] = df_with_ma['close'].ewm(span=26, adjust=False).mean()


        # MACD (Moving Average Convergence Divergence)
        df_with_ma['macd'] = df_with_ma['ema_12'] - df_with_ma['ema_26']
        df_with_ma['macd_signal'] = df_with_ma['macd'].ewm(span=9, adjust=False).mean()
        df_with_ma['macd_histogram'] = df_with_ma['macd'] - df_with_ma['macd_signal']

        logger.info("Moving Averages calculated")
        return df_with_ma
    
    def calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        calculate Relative Strength Index(RSI)
        Args:
            df: DataFrame with price data
            window: RSI window period
        
        Returns:
            DataFrame with RSI column
        """
        logger.info(f"Calculating {window}-day RSI")
        df_with_rsi = df.copy()

        # Calculate price changes
        delta = df_with_rsi['close'].diff()

        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        # Calculate RS and RSI
        rs = gain / loss
        df_with_rsi['rsi'] = 100 - (100 / (1 + rs))

        logger.info("RSI calculated")
        return df_with_rsi
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate support and resistance levels
        Args:
            df: DataFrame with OHLC data
            window: Rolling window for levels
        
        Returns:
            DataFrame with support/resistance columns
        """

        logger.info("Calculating support and resistance levels...")
        df_with_sr = df.copy()

        # Rolling highs and lows
        df_with_sr['rolling_high'] = df_with_sr['high'].rolling(window=window).max()
        df_with_sr['rolling_low'] = df_with_sr['low'].rolling(window=window).min()

        # Pivot points (classic formula)
        df_with_sr['pivot'] = (df_with_sr['high'] + df_with_sr['low'] + df_with_sr['close']) / 3
        df_with_sr['r1'] = 2 * df_with_sr['pivot'] - df_with_sr['low']
        df_with_sr['s1'] = 2 * df_with_sr['pivot'] - df_with_sr['high']
        df_with_sr['r2'] = df_with_sr['pivot'] + (df_with_sr['high'] - df_with_sr['low'])
        df_with_sr['s2'] = df_with_sr['pivot'] - (df_with_sr['high'] - df_with_sr['low'])

        logger.info("Support / Resistance calculated")
        return df_with_sr
    
    def transform_symbol_data(self, symbol: str, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Complete transformation pipeline for a single symbol

        Args:
            symbol: stock symbol
            df_raw: Raw DataFrame from API

        Returns:
            Fully transformed DataFrame ready for database
        """
        logger.info(f"Starting transformation pipeline for {symbol}")

        # Step 1: Clean data
        df_clean = self.clean_data(df_raw)

        # Step 2: Calculate returns
        df_returns = self.calculate_returns(df_clean)

        # Step 3: Calculate Volatility
        df_vol = self.calculate_volatility(df_returns)

        # Step 4: Calculate moving average
        df_ma = self.calculate_moving_averages(df_vol)

        # Step 5: Calculate RSI
        df_rsi = self.calculate_rsi(df_ma)

        # Step 6: Calculate support/resistance
        df_final = self.calculate_support_resistance(df_rsi)

        # Add metadata
        df_final['symbol'] = symbol
        df_final['last_updated'] = datetime.now()

        # Reorder columns for clarity
        column_order = [
            'open', 'high', 'low', 'close', 'volume',
            'daily_return', 'log_return', 'cumulative_return',
            'return_5d', 'return_20d', 'return_252d',
            'volatility', 'volatility_annualized', 'atr',
            'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
            'macd', 'macd_signal', 'macd_histogram',
            'rsi', 'rolling_high', 'rolling_low', 'pivot', 'r1', 's1', 'r2', 's2',
            'symbol', 'last_updated'
        ]

        # Only include columns that exist
        final_column = [col for col in column_order if col in df_final.columns]
        df_final = df_final[final_column]

        logger.info(f"Transformation complate for {symbol}. Shape: {df_final.shape}")
        return df_final
    
    def transform_multiple_symbols(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Transform multiple symbols
        Args:
            data_dict: Dictionary mapping symbol to raw DataFrame

        Returns:
            Dictionary mapping symbol to transformed DataFrame
        """
        results = {}
        for symbol, df in data_dict.items():
            if df is not None:
                try:
                    results[symbol] = self.transform_symbol_data(symbol, df)
                except Exception as e:
                    logger.error(f"Failed to transform {symbol}: {e}")
                    results[symbol] = None
            else:
                results[symbol] = None

        return results
    
    def save_to_csv(self, data_dict: Dict[str, pd.DataFrame], output_dir: str = "data/processed"):
        """
        Save transformed data to CSV files
        Args:
            data_dict: Dictionary transformed of DataFrame
            output_dir: Output dictionary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for symbol, df in data_dict.items():
            if df is not None:
                filename = output_path / f"{symbol}_transformed.csv"
                df.to_csv(filename)
                logger.info(f"Saved {symbol} data to  {filename}")

def test_transformer():
    """ Test the transformer with sample data """
    from src.api.api_client import AlphaVantageClient

    # Get sample data
    client = AlphaVantageClient()
    transformer = FinancialDataTranfromer()

    # Test single symbol
    print("\n Testing tranformation for AAPL...")
    aapl_raw = client.get_daily_data('AAPL', outputsize='compact')
    aapl_transformed = transformer.transform_symbol_data('AAPL', aapl_raw)

    print(f"Tranformed data shape: {aapl_transformed.shape}")
    print("\nSample of transformed data:")
    print(aapl_transformed[['close', 'daily_return', 'sma_20', 'rsi']].tail())

    # Test multiple symbols
    print("\n Testing multiple symbols tranformation...")
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    raw_data = client.get_multiple_symbols(symbols, outputsize='compact')
    transformed_data = transformer.transform_multiple_symbols(raw_data)

    for symbol, df in transformed_data.items():
        if df is not None:
            print(f"\n{symbol}: {df.shape}")

    # Save to CSV
    transformer.save_to_csv(transformed_data)

    return transformed_data


if __name__ == "__main__":
    transformed_data = test_transformer()