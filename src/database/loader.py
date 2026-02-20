"""
Database Loader - Local CSV Edition
Loads pre-transformed CSV files into PostgreSQL
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseLoader:
    def __init__(self):
        """Initialize database connection from .env"""
        self.conn_string = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','12345')}@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5433')}/{os.getenv('DB_NAME','financial_db')}"
        self.conn = None
    
    def connect(self):
        self.conn = psycopg2.connect(self.conn_string)
    
    def disconnect(self):
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.disconnect()
    
    def get_or_create_stock(self, symbol: str) -> int:
        """Find existing stock_id or create a new one with strict symbol matching"""
        with self.conn.cursor() as cur:
           
            cur.execute("SELECT stock_id FROM financial.stocks WHERE symbol = %s", (symbol.strip().upper(),))
            result = cur.fetchone()
            
            if result:
                return result[0]
            
            # Jika tidak ada, buat baru
            logger.info(f"Registering new ticker: {symbol}")
            cur.execute(
                "INSERT INTO financial.stocks (symbol, company_name) VALUES (%s, %s) RETURNING stock_id",
                (symbol.strip().upper(), f"{symbol} Inc.")
            )
            new_id = cur.fetchone()[0]
            self.conn.commit() 
            return new_id

    def insert_data(self, df: pd.DataFrame, symbol: str) -> int:
        stock_id = self.get_or_create_stock(symbol)
        df_insert = df.copy()
        
        if 'date' not in df_insert.columns:
            df_insert.rename(columns={df_insert.columns[0]: 'date'}, inplace=True)

        df_insert['stock_id'] = stock_id
        
        # Daftar kolom yang sesuai dengan skema SQL Anda
        target_columns = [
            'stock_id', 'date', 'open', 'high', 'low', 'close', 'volume',
            'daily_return', 'log_return', 'cumulative_return', 'sma_20', 
            'sma_50', 'sma_200', 'ema_12', 'ema_26', 'macd', 'macd_signal', 
            'macd_histogram', 'rsi', 'atr', 'volatility', 'volatility_annualized',
            'pivot', 'r1', 's1', 'r2', 's2'
        ]
        
        available_cols = [c for c in target_columns if c in df_insert.columns]
        df_final = df_insert[available_cols].copy()
        df_final['date'] = pd.to_datetime(df_final['date']).dt.date
        
        # Pembersihan nilai non-standar
        df_final = df_final.replace([np.inf, -np.inf], None)
        df_final = df_final.where(pd.notnull(df_final), None)
        
        with self.conn.cursor() as cur:
            cols_str = ', '.join(available_cols)
            insert_sql = f"INSERT INTO financial.daily_prices ({cols_str}) VALUES %s ON CONFLICT (stock_id, date) DO NOTHING"
            values = [tuple(x) for x in df_final.to_numpy()]
            
            try:
                execute_values(cur, insert_sql, values)
                self.conn.commit()
                return cur.rowcount
            except Exception as e:
                self.conn.rollback()
                raise e

def main():
    """Main execution to load local processed CSVs"""
    print("\n" + "="*40)
    print("OFFLINE LOAD: PROCESSED DATA TO DB")
    print("="*40)

    # List of symbols to process based on your files
    symbols = ['AAPL', 'GOOGL', 'TSLA']
    processed_dir = "data/processed"
    
    with DatabaseLoader() as loader:
        for symbol in symbols:
            file_path = os.path.join(processed_dir, f"{symbol}_transformed.csv")
            
            if os.path.exists(file_path):
                try:
                    print(f"[*] Reading {file_path}...")
                    df = pd.read_csv(file_path)
                    
                    count = loader.insert_data(df, symbol)
                    print(f" SUCCESS: {symbol} | {count} rows synchronized.")
                except Exception as e:
                    print(f" FAILED to load {symbol}: {e}")
            else:
                print(f" SKIPPED: File {file_path} not found.")

    print("\n" + "="*40)
    print("LOAD COMPLETED")
    print("="*40)

if __name__ == "__main__":
    main()