"""
Financial Data Pipeline DAG
Orchestrates the daily ETL process for stock market data.
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Import custom modules
from src.api.api_client import AlphaVantageClient
from src.transform.transform import FinancialDataTranfromer

# Configuration
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META', 'NVDA']
DB_CONN_ID = 'financial_postgres'
PROCESSED_PATH = '/opt/airflow/data/processed'

default_args = {
    'owner': 'financial_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 20),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def extract_task(**kwargs):
    """Fetch daily stock data from API and cache it."""
    client = AlphaVantageClient()
    results = {}

    for symbol in SYMBOLS:
        try:
            # Kita fokus ambil datanya saja, biarkan client yang mengurus cache
            df = client.get_daily_data(symbol, outputsize='compact')
            
            # Kita simpan status sukses tanpa memanggil _get_cache_path secara manual
            results[symbol] = {
                'status': 'success',
                'rows': len(df)
            }
            logging.info(f"Extracted {symbol}: {len(df)} rows")
        except Exception as e:
            logging.error(f"Extraction failed for {symbol}: {e}")
            results[symbol] = {'status': 'failed', 'error': str(e)}
    return results

def transform_task(ti, **kwargs):
    """Process raw data and calculate technical indicators."""
    extract_results = ti.xcom_pull(task_ids='extract')
    if not extract_results:
        raise ValueError("No data received from extract task")
        
    transformer = FinancialDataTranfromer()
    client = AlphaVantageClient()
    results = {}

    os.makedirs(PROCESSED_PATH, exist_ok=True)

    for symbol, meta in extract_results.items():
        if meta['status'] != 'success': 
            continue

        try:
            # Load raw, apply transformations
            df_raw = client.get_daily_data(symbol, use_cache=True)
            df_clean = transformer.transform_symbol_data(symbol, df_raw)

            output_file = f"{PROCESSED_PATH}/{symbol}_clean.csv"
            df_clean.to_csv(output_file)


            results[symbol] = {'status': 'success', 'csv_path': output_file}
            logging.info(f"Transformed {symbol}")
        except Exception as e:
            logging.error(f"Transformation failed for {symbol}: {e}")

    return results

def load_task(ti, **kwargs):
    """Bulk load transformed data into Postgres."""
    transform_result = ti.xcom_pull(task_ids='transform')
    if not transform_result:
        raise ValueError("No data received from transform task")
        
    pg_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)

    for symbol, meta in transform_result.items():
        if meta.get('status') != 'success':
            continue

        df = pd.read_csv(meta['csv_path'])
        
        # 1. Pastikan kolom 'date' tersedia (handle jika dia tersimpan sebagai index)
        if 'date' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'date'})
            
        # 2. Pastikan tipe data date adalah string/datetime agar Postgres tidak bingung
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # 3. Ambil stock_id
        stock_id = pg_hook.get_first(
            "INSERT INTO financial.stocks (symbol) VALUES (%s) "
            "ON CONFLICT (symbol) DO UPDATE SET symbol = EXCLUDED.symbol "
            "RETURNING stock_id", (symbol,)
        )[0]   

        # 4. Tambahkan stock_id ke DataFrame
        df['stock_id'] = stock_id

        # 5. DEFINISIKAN ULANG URUTAN KOLOM SECARA EKSPLISIT
        target_columns = [
            'stock_id', 'date', 'open', 'high', 'low', 'close',
            'volume', 'daily_return', 'sma_20', 'sma_50', 'rsi'
        ]
        
        # Susun ulang DataFrame berdasarkan urutan target_columns di atas
        # Ini memastikan 'date' selalu berada di posisi ke-2 (index 1)
        df_final = df[target_columns].copy()

        # 6. Konversi ke list of lists
        data_to_load = df_final.values.tolist()

        # 7. Insert ke Postgres
        pg_hook.insert_rows(
            table='financial.daily_prices',
            rows=data_to_load,
            target_fields=target_columns,
            replace=True,
            replace_index=['stock_id', 'date']
        )
        logging.info(f"Loaded {symbol} to database successfully.")

# DAG Definition
with DAG(
    'financial_data_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['finance']
) as dag:
    
    start = EmptyOperator(task_id='start')

    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_task
    )

    transform = PythonOperator(
        task_id='transform', 
        python_callable=transform_task
    )

    load = PythonOperator(
        task_id='load', 
        python_callable=load_task
    )

    report = BashOperator(
        task_id='generate_report',
        bash_command='echo "Pipeline finished at $(date)" > /tmp/pipeline_log.txt'
    )

    cleanup = BashOperator(
        task_id='cleanup',
        bash_command=f'rm -f {PROCESSED_PATH}/*.csv'
    )

    end = EmptyOperator(task_id='end') 

    # Workflow
    start >> extract >> transform >> load >> report >> cleanup >> end