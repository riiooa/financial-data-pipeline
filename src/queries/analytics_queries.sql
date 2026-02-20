-- ADVANCED SQL ANALYTICS FOR FINANCIAL DATA
-- Window Function, Moving Averages, Rankings, 

-- 1. 7 DAY MOVING AVERAGES
-- Simple moving average using window functions
WITH base AS (
	SELECT
		s.symbol,
		dp.date,
		dp.close
	FROM financial.stocks s 
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
	WHERE s.symbol IN ('AAPL', 'GOOGL', 'TSLA')
		AND dp.date >= '2024-12-01'
)
SELECT
	symbol,
	date,
	close,
	AVG(close) OVER (
		PARTITION BY symbol
		ORDER BY date 
		ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
	) AS sma_7,
	AVG(close) OVER (
		PARTITION BY symbol
		ORDER BY date
		ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
	) AS sma_20
FROM base 
WHERE date >= '2025-01-01'
ORDER BY symbol, date DESC 
LIMIT 30;
	
	
-- Exponential moving average using window functions
WITH RECURSIVE ordered AS (
    SELECT
        s.symbol,
        dp.date,
        dp.close,
        ROW_NUMBER() OVER (
            PARTITION BY s.symbol
            ORDER BY dp.date
        ) AS rn
    FROM financial.stocks s
    JOIN financial.daily_prices dp
        ON s.stock_id = dp.stock_id
    WHERE s.symbol = 'AAPL'
      AND dp.date >= DATE '2025-01-01'
),

ema_calc AS (

    SELECT
        symbol,
        date,
        close,
        rn,
        close::numeric AS ema_10
    FROM ordered
    WHERE rn = 1

    UNION ALL


    SELECT
        o.symbol,
        o.date,
        o.close,
        o.rn,
        (2.0/11) * o.close +
        (1 - 2.0/11) * e.ema_10
    FROM ordered o
    JOIN ema_calc e
        ON o.rn = e.rn + 1
)

SELECT
    symbol,
    date,
    close,
    ROUND(ema_10, 4) AS ema_10
FROM ema_calc
ORDER BY date;

-- 2. MONTH-TO-DATE RETURNS
WITH monthly_prices AS (
	SELECT 
		s.symbol,
		dp.date,
		dp.close,
		FIRST_VALUE(dp.close) OVER (
			PARTITION BY s.symbol, DATE_TRUNC('month', dp.date)
			ORDER BY dp.date
			ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
		) AS month_open,
		LAST_VALUE(dp.close) OVER (
			PARTITION BY s.symbol, DATE_TRUNC('month', dp.date)
			ORDER BY dp.date
			ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING			
		) AS month_close
	FROM financial.stocks s 
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
	WHERE s.symbol IN ('AAPL', 'GOOGL', 'TSLA')
		AND dp.date >= '2025-01-01'
)
SELECT
	symbol,
	DATE_TRUNC('month', date) AS month,
	MIN(date) AS first_trading_day,
	MAX(date) AS last_trading_day,
	ROUND(
		(
			(MAX(month_close) / NULLIF(MIN(month_open), 0) -1) * 100
	) ::NUMERIC, 2 ) AS mtd_return_percentage,
	COUNT(*) AS trading_days_count
FROM monthly_prices 
GROUP BY symbol, DATE_TRUNC('month', date)
ORDER BY symbol, month DESC;




-- 3. VOLATILITY CALCULATION
-- Historical volatility (20-day rolling)
SELECT 
	symbol,
	date,
	close,
	daily_return,
	ROUND(
		(
			STDDEV(daily_return) OVER (
				PARTITION BY symbol
				ORDER BY date
				ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
			) * SQRT(252)
		)::NUMERIC, 2
	) AS volatility_20d_annualized
FROM (
	SELECT
		s.symbol,
		dp.date,
		dp.close,
		dp.daily_return
	FROM financial.stocks s 
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
	WHERE s.symbol = 'AAPL'
		AND dp.date >= '2025-01-01'
) t
WHERE daily_return IS NOT NULL
ORDER BY date DESC
LIMIT 30;
	
-- Volatility comparison across stocks
SELECT 
    symbol,
    ROUND(AVG(daily_return)::NUMERIC, 4) AS avg_daily_return,
    ROUND(STDDEV(daily_return)::NUMERIC, 4) AS daily_volatility_std,
    ROUND((STDDEV(daily_return) * SQRT(252))::NUMERIC, 2) AS annualized_volatility,
    ROUND(MIN(daily_return)::NUMERIC, 2) AS max_day_loss,
    ROUND(MAX(daily_return)::NUMERIC, 2) AS max_day_gain
FROM (
    SELECT
        s.symbol,
        dp.daily_return
    FROM financial.stocks s 
    JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
    WHERE s.symbol IN ('AAPL', 'GOOGL', 'TSLA')
        AND dp.date >= '2025-01-01'
        AND dp.daily_return IS NOT NULL
        AND dp.daily_return::text != 'NaN'
        AND dp.daily_return::text != 'Infinity'
) t
GROUP BY symbol 
ORDER BY annualized_volatility DESC;


-- 4. RANKING FUNCTIONS
-- Top Gainers/Losers by Day
SELECT
	date, 
	symbol,
	daily_return,
	RANK() OVER (PARTITION BY date ORDER BY daily_return DESC) AS best_performer_rank,
	RANK() OVER (PARTITION BY date ORDER BY daily_return ASC) AS worst_performer_rank
FROM (
	SELECT
		s.symbol,
		dp.date,
		dp.daily_return
	FROM financial.stocks s 
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
	WHERE s.symbol IN ('AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN')
		AND dp.date >= '2025-01-01'
		AND dp.daily_return IS NOT NULL
		AND dp.daily_return::text != 'NaN'
) t
ORDER BY date DESC, best_performer_rank 
LIMIT 30;

-- Top 3 Stocks by Volume Each Day
WITH ranked_volume AS (
	SELECT
		dp.date,
		s.symbol,
		dp.volume,
		ROW_NUMBER() OVER (PARTITION BY dp.date ORDER BY dp.volume DESC) AS volume_rank
	FROM financial.daily_prices dp 
	JOIN financial.stocks s ON dp.stock_id = s.stock_id 
	WHERE dp.date >= CURRENT_DATE - INTERVAL '30 days'
		AND dp.volume IS NOT NULL
)
SELECT
	date, 
	symbol,
	volume,
	volume_rank
FROM ranked_volume 
WHERE volume_rank <= 3
ORDER BY date DESC, volume_rank;

-- 5. CUMULATIVE SUMS (Year-to-Date)
-- YTD Cumulative Return (AAPL)
WITH ytd_data AS (
	SELECT
		s.symbol,
		dp.date,
		dp.close,
		FIRST_VALUE(dp.close) OVER (
			PARTITION BY s.symbol, EXTRACT(YEAR FROM dp.date) 
			ORDER BY dp.date
			ROWS BETWEEN  UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
		) AS year_start_price
	FROM financial.stocks s
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id
	WHERE s.symbol = 'AAPL'
		AND EXTRACT(YEAR FROM dp.date) = 2025
)
SELECT
	symbol,
	date,
	close,
	year_start_price,
	ROUND(
		(
			(close / NULLIF(year_start_price, 0) -1 ) * 100
		)::NUMERIC, 2
	) AS ytd_return_percentage
FROM ytd_data
ORDER BY date;

-- Cumulative Volume & Moving Average Volume
SELECT 
	symbol,
	date,
	volume,
	SUM(volume) OVER (
		PARTITION BY symbol, EXTRACT(YEAR FROM date)
		ORDER BY date
		ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
	) AS ytd_cumulative_volume,
	ROUND(
		AVG(volume) OVER (
			PARTITION BY symbol
			ORDER BY date
			ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
		)::NUMERIC, 0
	) AS avg_volume_20d
FROM (
	SELECT
		s.symbol,
		dp.date,
		dp.volume
	FROM financial.stocks s
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id
	WHERE s.symbol = 'AAPL'
		AND EXTRACT(YEAR FROM dp.date) = 2025
) t
ORDER BY date DESC
LIMIT 30;



-- 6. LAG/LEAD FOR COMPARISONS
-- Day-over-day comparison
WITH price_data AS (
	SELECT
		s.symbol,
		dp.date,
		dp.close,
		LAG(dp.close, 1) OVER (PARTITION BY s.symbol ORDER BY dp.date) AS p1_close,
		LAG(dp.close, 5) OVER (PARTITION BY s.symbol ORDER BY dp.date) AS p5_close,
		LAG(dp.close, 20) OVER (PARTITION BY s.symbol ORDER BY dp.date) AS p20_close
	FROM financial.stocks s 
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
	WHERE s.symbol = 'AAPL'
		AND dp.date >= '2025-01-01'
)
SELECT
	symbol,
	date,
	close,
	p1_close AS prev_day_close,
	p5_close AS prev_week_close,
	p20_close AS prev_month_close,
	ROUND( ((close / NULLIF(p1_close, 0) - 1) * 100)::NUMERIC, 2 ) AS daily_change_pct,
	ROUND( ((close / NULLIF(p5_close, 0) - 1) * 100)::NUMERIC, 2 ) AS weekly_change_pct,
	ROUND( ((close / NULLIF(p20_close, 0) - 1) * 100)::NUMERIC, 2 ) AS monthly_change_pct
FROM price_data 
ORDER BY date DESC 
LIMIT 30;

-- 7. PERCENTILE AND DISTRIBUTION ANALYSIS
-- Price percentiles
SELECT 
	symbol,
	ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY close)::NUMERIC, 2) AS q1_25th,
	ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY close)::NUMERIC, 2) AS median_50th,
	ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY close)::NUMERIC, 2) AS q3_75th,
	ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY close)::NUMERIC, 2) AS p90_90th,
	MAX(close) AS max_price,
	MIN(close) AS min_price
FROM (
	SELECT
		s.symbol,
		dp.close
	FROM financial.stocks s 
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
	WHERE s.symbol IN ('AAPL', 'GOOGL', 'TSLA')
		AND dp.date >= '2025-01-01'
		AND dp.close IS NOT NULL
) t
GROUP BY symbol; 


-- 8. COMPLEX QUERY: COMBINING ALL TECHNIQUES
-- Comprehensive stock analysis report
WITH stock_metrics AS (
	SELECT
		s.symbol,
		dp.date,
		dp.close,
		dp.volume,
		dp.daily_return,
		AVG(dp.close) OVER w_20d AS sma_20,
		STDDEV(dp.daily_return) OVER w_20d * SQRT(252) AS vol_20d,
		RANK() OVER (PARTITION BY dp.date ORDER BY dp.volume DESC) AS volume_rank,
		SUM(dp.volume) OVER (PARTITION BY s.symbol, EXTRACT(YEAR FROM dp.date) ORDER BY dp.date) AS cumulative_volume_ytd
	FROM financial.stocks s 
	JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
	WHERE dp.date >= '2025-01-01'
		AND dp.daily_return IS NOT NULL 
		AND dp.daily_return::text != 'NaN'
	WINDOW w_20d AS (
		PARTITION BY s.symbol
		ORDER BY dp.date
		ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
	)
),
ranked_metrics AS (
	SELECT
		symbol,
		date,
		close,
		ROUND(sma_20::NUMERIC, 2) AS sma_20,
		ROUND(vol_20d::NUMERIC, 2) AS volatility_20d,
		CASE
			WHEN close > (sma_20 * 1.05 ) THEN 'OVERBOUGHT'
			WHEN close < (sma_20 * 0.95 ) THEN 'OVERSOLD'
			ELSE 'NEUTRAL'
		END AS market_position,
		volume_rank,
		cumulative_volume_ytd
	FROM stock_metrics
)
SELECT * 
FROM ranked_metrics
WHERE date = (SELECT MAX(date) FROM financial.daily_prices ) 
ORDER BY symbol;


-- 9. EXPORT QUERY FOR REPORTING
-- Export last 30 days of metrics for all stocks
SELECT 
	s.symbol,
	dp.date,
	dp.close,
	dp.daily_return,
	dp.sma_20,
	dp.sma_50,
	dp.rsi,
	dp.volatility_annualized,
	dp.volume
FROM financial.stocks s 
JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id 
WHERE dp.date >= (SELECT MAX(date) FROM financial.daily_prices) - INTERVAL '30 days'
ORDER BY s.symbol, dp.date DESC;
