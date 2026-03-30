# 📈 Financial Data Pipeline

An end-to-end automated pipeline for collecting, processing, analyzing, and visualizing stock market data. Built with Python, Apache Airflow, PostgreSQL (TimescaleDB), and Streamlit.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Features](#features)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Dashboard Preview](#dashboard-preview)
- [Database Schema](#database-schema)

---

## Overview

This project implements a fully automated financial data pipeline that:

1. **Extracts** daily OHLCV (Open, High, Low, Close, Volume) stock data from the [Alpha Vantage API](https://www.alphavantage.co/) for 7 major tech stocks: `AAPL`, `GOOGL`, `TSLA`, `MSFT`, `AMZN`, `META`, and `NVDA`.
2. **Transforms** raw data by computing technical indicators such as SMA, EMA, RSI, MACD, ATR, and daily returns.
3. **Loads** the processed data into a time-series-optimized PostgreSQL database with monthly partitioning.
4. **Analyzes** portfolio performance using advanced financial metrics (Sharpe Ratio, Maximum Drawdown, Annualized Volatility).
5. **Reports** automatically generated PDF reports and an interactive Streamlit dashboard.
6. **Orchestrates** the entire pipeline daily via Apache Airflow.

---

## Architecture

```
Alpha Vantage API
       │
       ▼
 ┌─────────────┐     ┌─────────────────┐     ┌──────────────────────┐
 │  API Client │────▶│  Transformer    │────▶│  PostgreSQL Database  │
 │ (rate-limit │     │ (indicators,    │     │  (TimescaleDB,        │
 │  + caching) │     │  cleaning)      │     │  monthly partitioned) │
 └─────────────┘     └─────────────────┘     └──────────────────────┘
                                                         │
                              ┌──────────────────────────┘
                              │
               ┌──────────────▼──────────────┐
               │       FinancialAnalyzer      │
               │  (Sharpe, MDD, Volatility,   │
               │   Correlation Matrix)        │
               └──────────────┬──────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
   ┌──────────▼──────────┐       ┌────────────▼────────────┐
   │  PDF Report          │       │  Streamlit Dashboard     │
   │  (ReportLab)         │       │  (Interactive Charts)    │
   └─────────────────────┘       └─────────────────────────┘
              │
   ┌──────────▼──────────┐
   │  Apache Airflow DAG  │
   │  (Daily Schedule)    │
   └─────────────────────┘
```

**Airflow DAG Pipeline:**
```
start → extract → transform → load → generate_report → send_report → cleanup → end
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Orchestration | Apache Airflow 2.7.3 |
| Database | PostgreSQL 15 (TimescaleDB) |
| Dashboard | Streamlit + Plotly |
| Analytics | Pandas, NumPy, SciPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| PDF Reports | ReportLab |
| Containerization | Docker & Docker Compose |
| Data Source | Alpha Vantage API |

---

## Project Structure

```
financial-data-pipeline/
├── airflow/
│   └── dags/
│       └── financial_pipeline.py    # Main Airflow DAG definition
├── config/
│   └── logging.conf                 # Logging configuration
├── docker/
│   └── docker-compose.yml           # Docker services (Airflow, Postgres, PGAdmin)
├── docs/
│   ├── API_DOCS.md
│   ├── day1_completed.md
│   ├── day2_completed.md
│   └── screenshots/                 # Dashboard & chart screenshots
├── plots/                           # Generated chart outputs (PNG, HTML)
├── report/
│   └── analysis/                    # JSON analysis reports
├── reports/                         # Generated PDF reports
├── src/
│   ├── analysis/
│   │   └── analysis.py              # Financial metrics engine (Sharpe, MDD, etc.)
│   ├── api/
│   │   ├── api_client.py            # Alpha Vantage API client with caching
│   │   └── api_enhancements.py      # Retry/backoff decorators
│   ├── dashboard/
│   │   └── dashboard.py             # Streamlit interactive dashboard
│   ├── database/
│   │   ├── loader.py                # Bulk data loader to PostgreSQL
│   │   └── schema.sql               # Database schema with partitioning
│   ├── queries/
│   │   └── analytics_queries.sql    # Advanced SQL analytics (window functions)
│   ├── reporting/
│   │   └── report_generator.py      # Automated PDF report generator
│   └── tests/
│       └── conftest.py
├── scripts/
│   └── setup.sh
├── requirements.txt
└── README.md
```

---

## Features

### 🔌 API Client (`src/api/api_client.py`)

- **OOP-based** `AlphaVantageClient` class
- **Rate limiting**: Automatically enforces 5 requests/minute (1 request every 12 seconds)
- **File-based caching**: Avoids redundant API calls by caching responses as JSON (24-hour TTL for daily data, 1-hour for quotes)
- **Exponential backoff retry**: Up to 3 retries with configurable delay and backoff multiplier
- **Daily request limit tracking**: Tracks usage against the 500-call daily limit
- **Structured JSON logging**: Compatible with ELK Stack or Datadog

Supported endpoints:
- `get_daily_data(symbol)` — historical OHLCV data
- `get_quote(symbol)` — real-time quote
- `get_multiple_symbols(symbols)` — batch fetch with politeness delay

### ⚙️ Data Transformation

Calculates the following technical indicators per symbol:

| Indicator | Description |
|---|---|
| `daily_return` | Percentage daily price change |
| `log_return` | Logarithmic daily return |
| `cumulative_return` | Cumulative return from start |
| `sma_20`, `sma_50`, `sma_200` | Simple Moving Averages |
| `ema_12`, `ema_26` | Exponential Moving Averages |
| `macd`, `macd_signal`, `macd_histogram` | MACD indicator |
| `rsi` | Relative Strength Index (14-day) |
| `atr` | Average True Range |
| `volatility`, `volatility_annualized` | Rolling volatility |
| `pivot`, `r1`, `r2`, `s1`, `s2` | Pivot points and support/resistance levels |

### 🗄️ Database (`src/database/schema.sql`)

- **`financial.stocks`** — Dimension table storing stock metadata
- **`financial.daily_prices`** — Fact table with full technical indicator columns, partitioned by month from 2025–2026 with a default partition for future data
- **Automatic triggers**: Updates `first_data_date` and `last_data_date` on the `stocks` table after every insert
- **Analytical views**:
  - `vw_latest_prices` — Most recent price per symbol
  - `vw_monthly_performance` — Monthly OHLC aggregation using window functions

### 📊 Financial Analytics (`src/analysis/analysis.py`)

The `FinancialAnalyzer` class computes:

- **Sharpe Ratio** — Risk-adjusted return using 2% annual risk-free rate
- **Maximum Drawdown (MDD)** — Largest peak-to-trough decline
- **Annualized Volatility** — Standard deviation scaled to annual basis
- **Correlation Matrix** — Cross-asset price correlation for portfolio diversification analysis
- **Market Signals** — Overbought/oversold detection using moving average crossovers

### 📄 PDF Reporting (`src/reporting/report_generator.py`)

Automatically generates professional PDF reports containing:

- **Executive Summary** — List of analyzed stocks and data coverage period
- **Market Signals Table** — Latest prices with technical signal labels
- **Portfolio Health** — Aggregated metrics including Portfolio Sharpe Ratio
- **Correlation Matrix Heatmap** — Visual cross-asset correlation

Reports are saved to `/reports/` with a timestamp in the filename and can be emailed via SMTP.

### 📡 Streamlit Dashboard (`src/dashboard/dashboard.py`)

Interactive web dashboard powered by Streamlit and Plotly, featuring:

- Real-time stock price charts with candlestick view
- RSI and volume overlays
- Relative performance comparison
- Rolling volatility visualization
- Correlation matrix heatmap
- Portfolio performance metrics display

### 🔄 Airflow Orchestration (`airflow/dags/financial_pipeline.py`)

A single DAG (`financial_data_pipeline`) runs on a `@daily` schedule:

```
start → extract → transform → load → generate_report → send_report → cleanup → end
```

- **extract**: Fetches data for all 7 symbols from Alpha Vantage
- **transform**: Cleans data and computes indicators, saves to CSV
- **load**: Bulk loads transformed CSVs into PostgreSQL via `PostgresHook`
- **generate_report**: Calls `FinancialAnalyzer` and `FinancialReportGenerator` to produce a PDF
- **send_report**: Emails the PDF report via SMTP
- **cleanup**: Removes temporary CSV files from the processed data directory

Data is passed between tasks via Airflow XCom.

---

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/) and Docker Compose installed
- [Alpha Vantage API Key](https://www.alphavantage.co/support/#api-key) (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/financial-data-pipeline.git
cd financial-data-pipeline
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Alpha Vantage
ALPHA_VANTAGE_API_KEY=your_api_key_here

# Database
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_NAME=financial_db
DB_HOST=postgres      # Use 'postgres' for Docker, 'localhost' for local Streamlit
DB_PORT=5432

# Email (optional, for report delivery)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
REPORT_RECIPIENTS=recipient@example.com
```

### 3. Start the Full Stack

```bash
cd docker
docker-compose up -d
```

This will start:

| Service | URL | Credentials |
|---|---|---|
| Airflow Webserver | http://localhost:8080 | `admin` / `admin` |
| PGAdmin | http://localhost:5050 | `admin@financial.com` / `admin` |
| PostgreSQL | `localhost:5433` | `postgres` / `<DB_PASSWORD>` |

### 4. Trigger the Pipeline

In the Airflow UI:
1. Find the `financial_data_pipeline` DAG
2. Toggle it **On**
3. Click **Trigger DAG** to run it immediately

### 5. Launch the Streamlit Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run src/dashboard/dashboard.py
```

The dashboard will be available at http://localhost:8501.

---

## Configuration

### Tracked Symbols

Symbols are defined in `airflow/dags/financial_pipeline.py`:

```python
SYMBOLS = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META', 'NVDA']
```

Edit this list to track different stocks.

### Rate Limiting

The Alpha Vantage free tier allows **5 requests/minute** and **500 requests/day**. The API client automatically enforces a 12-second minimum interval between requests. To use a premium key with higher limits, adjust `min_request_interval` in `AlphaVantageClient.__init__()`.

### Risk-Free Rate

The Sharpe Ratio calculation uses a **2% annual risk-free rate** (proxy for 10-year US Treasury). This can be changed in `src/analysis/analysis.py`:

```python
self.risk_free_rate = 0.02  # Change to your preferred rate
```

---

## Usage

### Run Analysis Manually

```python
from src.analysis.analysis import FinancialAnalyzer

analyzer = FinancialAnalyzer()
symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META', 'NVDA']
report = analyzer.generate_report(symbols)
print(report)
```

### Generate a PDF Report Manually

```python
from src.reporting.report_generator import FinancialReportGenerator
from src.analysis.analysis import FinancialAnalyzer

analyzer = FinancialAnalyzer()
reporter = FinancialReportGenerator()

analysis = analyzer.generate_report(['AAPL', 'MSFT', 'NVDA'])
pdf_path = reporter.generate_daily_report(analysis)
print(f"Report saved to: {pdf_path}")
```

### Fetch Stock Data via API Client

```python
from src.api.api_client import AlphaVantageClient

client = AlphaVantageClient()

# Single symbol
df = client.get_daily_data('AAPL', outputsize='compact')
print(df.head())

# Multiple symbols
results = client.get_multiple_symbols(['AAPL', 'GOOGL', 'TSLA'])
```

---

## Dashboard Preview

The Streamlit dashboard provides:

- **Price Chart** with candlestick and volume bars
- **RSI Indicator** with overbought/oversold markers
- **Correlation Heatmap** across all tracked symbols
- **Performance Metrics** per symbol

Sample charts are available in the [`docs/screenshots/`](docs/screenshots/) and [`plots/`](plots/) directories.

---

## Database Schema

```
financial
├── stocks                    # Dimension table (symbol, company_name, sector, ...)
└── daily_prices              # Fact table — partitioned by month
    ├── daily_prices_2025_01
    ├── daily_prices_2025_02
    ├── ...
    ├── daily_prices_2026_12
    └── daily_prices_future   # Default partition for future data
```

**Key analytical views:**

| View | Description |
|---|---|
| `vw_latest_prices` | Most recent price record per symbol |
| `vw_monthly_performance` | Monthly OHLC with return calculation |

---

## License

This project was developed as part of a Fintech & Digital Banking portfolio.
