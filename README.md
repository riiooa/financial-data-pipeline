# financial Data Pipeline

End-to-end data pipeline for financial market data using Alpha Vantage API, PostgreSQL, and Apache Airflow

## Project Overview

This project builds a complete financial data pipeline that:
- Fetches real-time and historical stock data from Alpha Vantage API
- Transforms raw OHLC data into analytic-ready format
- Stores data in optimized PostgreSQL schema with partitioning
- Performs advanced financial calculation using window functions
- Orchestrates daily ETL with Apache Airflow
- Implements robust error handling and logging
  
## Features

### Day 1 
- Alpha Vantage API integration with rate limiting
- Data transformation layer (OHLC to database format)
- Optimized PostgreSQL schema with monthly partitioning
- Advanced SQL analytics (moving averages, volatility)
- Airflow DAG with multi-task orchestration
- Comprehensive error handling and loggings
- Git workflow simulation (feature branches)

### Day 2
- Financial metric calculation (Sharpe ratio, max drawdown)
- Data visualization with Matplotlib/Seaborn
- Interactive Streamlit dashboard
- Automated PDF reporting
- Unit testing with pytest
- Performance optimization
  
## Quick Start

### Prerequisites
- Docker & Docker compose
- Python 3.10+
- Alpha Vantage API key (free from https://www.alphavantage.co/support/#api-key)
