-- FINANCIAL DATA PIPELINE - DATABASE SCHEMA
-- Time-series optimized with monthly partitioning

-- Drop schema if exists 
DROP SCHEMA IF EXISTS financial CASCADE;
CREATE SCHEMA financial;
SET search_path TO financial;


-- 1. STOCK TABLE (Dimension)
CREATE TABLE IF NOT EXISTS stocks (
    stock_id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    company_name VARCHAR(255),
    exchange VARCHAR(50),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap DECIMAL(15, 2),
    is_active BOOLEAN DEFAULT TRUE,
    first_data_date DATE,
    last_data_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Index for quick lookups
    CONSTRAINT valid_symbol CHECK (symbol ~ '^[A-Z.]+$')
);

COMMENT ON TABLE stocks IS 'Stock is master data';
COMMENT ON COLUMN stocks.symbol IS 'Stock ticker symbol (e.g., AAPL)';


-- 2. DAILY_PRICES TABLE (Fact Table - Time Series)
-- Partitioned by date for performance

CREATE TABLE IF NOT EXISTS daily_prices (
    price_id BIGSERIAL,
    stock_id INT NOT NULL,
    date DATE NOT NULL,

    -- OHLC data
    open DECIMAL (12, 4) NOT NULL,
    high DECIMAL (12, 4) NOT NULL,
    low DECIMAL (12, 4) NOT NULL,
    close DECIMAL (12, 4) NOT NULL,
    adjusted_close DECIMAL (12, 4),
    volume BIGINT NOT NULL,

    -- Calculated metrics 
    daily_return DECIMAL (12, 4),
    log_return DECIMAL (12, 4),
    cumulative_return DECIMAL (12, 4),

    -- Moving averages
    sma_20 DECIMAL (12, 4),
    sma_50 DECIMAL (12, 4),
    sma_200 DECIMAL (12, 4),
    sma_12 DECIMAL (12, 4),
    ema_26 DECIMAL (12, 4),

    -- Technical indicators
    macd DECIMAL (12, 4),
    macd_signal DECIMAL (12, 4),
    macd_histogram DECIMAL (12, 4),
    rsi DECIMAL (12, 4),
    atr DECIMAL (12, 4),
    volatility DECIMAL (12, 4),
    volatility_annualized DECIMAL (12, 4),

    -- Support/Resistance
    pivot DECIMAL(12, 4),
    r1 DECIMAL (12, 4),
    s1 DECIMAL (12, 4),
    r2 DECIMAL (12, 4),
    s2 DECIMAL (12, 4),

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    PRIMARY KEY (stock_id, date), 
    CONSTRAINT valid_ohlc CHECK (
        open IS NOT NULL AND
        high >= low AND
        close BETWEEN low AND high
    ),
    CONSTRAINT fk_stock
        FOREIGN KEY (stock_id) REFERENCES stocks(stock_id)
) PARTITION BY RANGE (date); 


DO $$
DECLARE
    year_month DATE;
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    FOR year_month IN
        SELECT generate_series('2025-01-01'::DATE, '2026-12-01'::DATE, '1 month'::INTERVAL)::DATE
    LOOP
        partition_name := 'daily_prices_' || TO_CHAR(year_month, 'YYYY_MM');
        start_date := year_month;
        end_date := year_month + INTERVAL '1 month';

        EXECUTE format('
            CREATE TABLE IF NOT EXISTS %I PARTITION OF daily_prices
            FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
    END LOOP;
END $$;

-- Default partition for future dates
CREATE TABLE IF NOT EXISTS financial.daily_prices_future PARTITION OF daily_prices DEFAULT;


-- 3. INDEXES FOR PERFORMANCE
CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_stock_date ON daily_prices(stock_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_daily_prices_returns ON daily_prices(stock_id, date DESC) WHERE daily_return IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_daily_prices_volume ON daily_prices(date, volume DESC);


-- 4. VIEWS FOR COMMON ANALYTICS
CREATE OR REPLACE VIEW vw_latest_prices AS
SELECT DISTINCT ON (s.symbol)
    s.symbol,
    s.company_name,
    dp.date,
    dp.close,
    dp.daily_return,
    dp.volume,
    dp.rsi,
    dp.volatility_annualized
FROM stocks s
JOIN daily_prices dp ON s.stock_id = dp.stock_id
WHERE dp.date >= CURRENT_DATE - INTERVAL '5 days'
ORDER BY s.symbol, dp.date DESC;

-- FIX: Menggunakan DISTINCT ON untuk menggabungkan Window Function agar hasil ringkas (1 baris per bulan)
CREATE OR REPLACE VIEW vw_monthly_performance AS
SELECT DISTINCT ON (s.symbol, month)
    s.symbol,
    DATE_TRUNC('month', dp.date) as month,
    COUNT(dp.date) OVER w as trading_days,
    FIRST_VALUE(dp.close) OVER w as month_open,
    LAST_VALUE(dp.close) OVER w as month_close,
    MAX(dp.high) OVER w as month_high,
    MIN(dp.low) OVER w as month_low,
    SUM(dp.volume) OVER w as total_volume,
    (LAST_VALUE(dp.close) OVER w / FIRST_VALUE(dp.close) OVER w - 1) * 100 as monthly_return
FROM stocks s
JOIN daily_prices dp ON s.stock_id = dp.stock_id
WINDOW w AS (
    PARTITION BY s.symbol, DATE_TRUNC('month', dp.date) 
    ORDER BY dp.date 
    RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
)
ORDER BY s.symbol, month DESC;


-- 5. FUNCTION FOR MAINTENANCE
CREATE OR REPLACE FUNCTION update_stock_date_range()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE stocks
    SET
        first_data_date = (SELECT MIN(date) FROM daily_prices WHERE stock_id = NEW.stock_id),
        last_data_date = (SELECT MAX(date) FROM daily_prices WHERE stock_id = NEW.stock_id),
        updated_at = CURRENT_TIMESTAMP
    WHERE stock_id = NEW.stock_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_stock_dates
    AFTER INSERT OR UPDATE ON daily_prices
    FOR EACH ROW
    EXECUTE FUNCTION update_stock_date_range();

CREATE OR REPLACE FUNCTION create_next_month_partition()
RETURNS VOID AS $$
DECLARE
    next_month DATE;
    partition_name TEXT;
BEGIN 
    next_month := DATE_TRUNC('month', CURRENT_DATE + INTERVAL '1 month');
    partition_name := 'daily_prices_' || TO_CHAR(next_month, 'YYYY_MM');

    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I PARTITION OF daily_prices
        FOR VALUES FROM (%L) TO (%L)',
        partition_name, next_month, next_month + INTERVAL '1 month'
    );
END;
$$ LANGUAGE plpgsql;

-- 6. COMMENTS FOR DOCUMENTATION
COMMENT ON TABLE daily_prices IS 'Daily stock prices with technical indicators, partitioned by month';
COMMENT ON COLUMN daily_prices.daily_return IS 'Daily return as percentage';
COMMENT ON COLUMN daily_prices.rsi IS 'Relative Strength Index (14-day)';
COMMENT ON COLUMN daily_prices.atr IS 'Average True Range';
COMMENT ON COLUMN daily_prices.macd IS 'Moving Average Convergence Divergence';

-- END OF SCHEMA
