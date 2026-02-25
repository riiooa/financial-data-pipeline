# 📈 Financial Data Pipeline (End-to-End)

This project is a comprehensive financial data pipeline designed to automatically extract, process, and present stock market analytics using the Alpha Vantage API, PostgreSQL, and Apache Airflow.

---

## 📋 Project Overview

The system automates the full financial data lifecycle:

**Extraction**  
Retrieves historical and real-time OHLCV data with built-in rate-limiting mechanisms.

**Transformation**  
Cleans raw data and computes technical indicators (Moving Averages, Daily Returns).

**Storage**  
Persists data into an optimized PostgreSQL schema with monthly partitioning.

**Analytics**  
Calculates advanced risk-return metrics (Sharpe Ratio, Maximum Drawdown, Annualized Volatility).

**Orchestration**  
Executes the entire workflow on a schedule using Apache Airflow.

**Reporting**  
Generates interactive dashboards (Streamlit) and professional PDF reports.

---

## 🚀 Key Features

### Day 1: Foundation & ETL Basics

**Robust API Client**  
OOP-based architecture with automatic rate-limit handling (5 requests per minute).

**Exponential Backoff**  
Resilient retry mechanism to handle transient network failures.

**Structured Logging**  
JSON-based logging system for seamless monitoring and integration with ELK Stack or Datadog.

**Dockerized Environment**  
Full orchestration using Docker Compose for:
- Apache Airflow  
- PostgreSQL  
- PGAdmin  

---

### Day 2: Advanced Analytics & Professional Reporting

#### Financial Metrics Engine

Automatic computation of key investment metrics:

- **Sharpe Ratio & Annualized Volatility** with properly normalized return scaling  
- **Maximum Drawdown (MDD)** for downside risk profiling  
- **Market Signals**: Automated detection of Overbought and Oversold conditions using Moving Average indicators  

---

#### Automated Intelligence Reporting

Integrated PDF report generation as an Airflow task, producing professional reports containing:

- Performance tables  
- Correlation matrix visualization  

---

#### Correlation Analytics

Heatmap visualization of cross-asset correlation matrices to support portfolio diversification strategies.

---

#### Interactive Dashboard

Streamlit-based dashboard for real-time data exploration and dynamic stock price visualization.

---

## 🛠 Tech Stack

- **Language**: Python 3.10+  
- **Orchestration**: Apache Airflow  
- **Database**: PostgreSQL (SQLAlchemy ORM)  
- **Dashboard**: Streamlit  
- **Analytics**: Pandas, NumPy, SciPy  
- **Visualization**: Matplotlib, Seaborn  
- **Containerization**: Docker & Docker Compose  

---

## 🚦 Quick Start

### 1. Prerequisites

- Docker & Docker Compose installed  
- Alpha Vantage API Key (free at Alpha Vantage)  
- `.env` file in the root directory with the following configuration:

```bash
ALPHA_VANTAGE_API_KEY=your_api_key
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=financial_db
DB_HOST=postgres   # for Docker
# or
DB_HOST=localhost  # for Streamlit

### 2. Running the Pipeline

```bash
# Start the full stack (Airflow, PostgreSQL)
docker-compose up -d

- Access the Airflow UI at:

```bash
http://localhost:8080
Default credentials: airflow / airflow

### 3. Launching the Dashboard

```bash
# Ensure virtual environment is activated and dependencies are installed
pip install -r requirements.txt

streamlit run src/dashboard/dashboard.py


## 📊 Sample Output

The generated PDF report includes:

### Executive Summary
Overview of the analyzed stocks and the historical data coverage period.

### Market Signals
Table of the latest closing prices with automatically generated technical signals.

### Portfolio Health
Aggregated portfolio metrics, including Portfolio Sharpe Ratio, for comprehensive investment performance evaluation.

---

This project was developed as part of a Fintech & Digital Banking portfolio.