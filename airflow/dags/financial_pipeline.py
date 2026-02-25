"""
Financial Data Pipeline DAG
Orchestrates the daily ETL process for stock market data.
"""

import os
import sys
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

from src.reporting.report_generator import FinancialReportGenerator
from src.analysis.analysis import FinancialAnalyzer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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

# --- ETL Tasks Functions ---

def extract_task(**kwargs):
    """Fetch daily stock data from API and cache it."""
    client = AlphaVantageClient()
    results = {}
    for symbol in SYMBOLS:
        try:
            df = client.get_daily_data(symbol, outputsize='compact')
            results[symbol] = {'status': 'success', 'rows': len(df)}
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
        if meta['status'] != 'success': continue
        try:
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
        if meta.get('status') != 'success': continue
        df = pd.read_csv(meta['csv_path'])
        if 'date' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        stock_id = pg_hook.get_first(
            "INSERT INTO financial.stocks (symbol) VALUES (%s) "
            "ON CONFLICT (symbol) DO UPDATE SET symbol = EXCLUDED.symbol "
            "RETURNING stock_id", (symbol,)
        )[0]   

        df['stock_id'] = stock_id
        target_columns = [
            'stock_id', 'date', 'open', 'high', 'low', 'close',
            'volume', 'daily_return', 'sma_20', 'sma_50', 'rsi'
        ]
        df_final = df[target_columns].copy()
        data_to_load = df_final.values.tolist()

        pg_hook.insert_rows(
            table='financial.daily_prices',
            rows=data_to_load,
            target_fields=target_columns,
            replace=True,
            replace_index=['stock_id', 'date']
        )
        logging.info(f"Loaded {symbol} to database successfully.")

# --- New Reporting Task Functions ---

def generate_report_task(**context):
    """Generate PDF report"""
    logging.info("Generating report...")
    analyzer = FinancialAnalyzer()
    reporter = FinancialReportGenerator()
    
    analysis = analyzer.generate_report(SYMBOLS)
    
    # Generate PDF
    pdf_path = reporter.generate_daily_report(analysis)
    
    context['ti'].xcom_push(key='report_path', value=pdf_path)
    return pdf_path

def send_report_task(**context):
    """Send report via email"""
    ti = context['ti']
    pdf_path = ti.xcom_pull(task_ids='generate_report', key='report_path')
    
    if not pdf_path:
        raise ValueError("No report path found in XCom")
        
    reporter = FinancialReportGenerator()
    reporter.send_email_report(pdf_path)

# --- DAG Definition ---

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

    # Generate Report
    generate_report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report_task,
        provide_context=True
    )

    # Send Report
    send_report = PythonOperator(
        task_id='send_report',
        python_callable=send_report_task,
        provide_context=True
    )

    cleanup = BashOperator(
        task_id='cleanup',
        bash_command=f'rm -f {PROCESSED_PATH}/*.csv'
    )

    end = EmptyOperator(task_id='end') 

    # --- Workflow / Dependencies ---
    # Start -> ETL -> Generate Report -> Send Report -> Cleanup -> End
    start >> extract >> transform >> load >> generate_report >> send_report >> cleanup >> end