# 📈 Financial Data Pipeline (Alpha Vantage)

This repository contains an advanced ETL (Extract, Transform, Load) pipeline for financial market data using the Alpha Vantage API, Python, and Apache Airflow.

---

## 🛠 Progress & Technical Implementation

The project currently implements the following core features:

---

### 1. Robust API Client Architecture

**Encapsulation**  
Built an OOP-based `AlphaVantageClient` to centralize and manage API communication.

**Custom Rate Limiting**  
Implemented a `_respect_rate_limit` mechanism to comply with the Alpha Vantage Free Tier limit (5 requests per minute).  
The system automatically calculates and enforces sleep intervals between requests.

**Response Caching**  
Implemented a JSON-based file caching system under `data/raw/` to:
- Reduce API quota consumption  
- Accelerate development and debugging cycles  

---

### 2. Structured Logging System

**JSON Logging**  
Configured a structured logger to produce JSON-formatted logs (via `StructuredLogger`), enabling seamless integration with log aggregation systems such as ELK Stack or Datadog.

**Centralized Configuration**  
Separated logging configuration from application logic using `config/logging.conf`, allowing flexible control of log levels (INFO, DEBUG, ERROR).

**Performance Monitoring**  
Implemented a `timer` decorator to precisely measure and log API request execution durations.

---

### 3. Error Handling & Resiliency

**Exponential Backoff Retry**  
Developed a custom `@retry` decorator that automatically retries failed requests caused by network-related exceptions (`RequestException`).

**Smart Backoff Strategy**  
Retry intervals increase exponentially (e.g., 1 second, 2 seconds, 4 seconds), allowing the server time to recover from transient failures.

**Validation & Error Detection**  
Explicit handling for Alpha Vantage-specific responses, including:
- Rate limit notifications (`Note` response)  
- Premium endpoint restriction messages  

---

### 4. Enhancements & Additional Features

**Batch Quote Retrieval**  
Added support for safely retrieving the latest stock quotes for multiple symbols in an iterative process.

**Historical Data Filtering**  
Implemented daily historical data retrieval with configurable date-range filtering using `pandas`.

**Free-Tier Compatibility Optimization**  
Optimized the `outputsize` parameter (using `compact` instead of `full`) to remain compliant with free-tier limitations without disrupting the pipeline flow.

---

### 5. DevOps & Workflow Development

**Environment Management**  
Managed sensitive credentials using `.env` files isolated in both the `docker/` directory and project root.

**Git Feature Branching Strategy**  
Adopted a structured Git workflow using feature branches (e.g., `feature/api-enhancements`) for isolated development.

**Docker Integration**  
Prepared a `docker-compose.yml` configuration to orchestrate the complete ecosystem:
- Apache Airflow  
- PostgreSQL  
- Redis  

---
