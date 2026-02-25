"""
Financial Metrics Analysis
Calculate advanced financial metrics: Sharpe ratio, max drawdown, correlation, etc.
"""

from venv import logger
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """
    Advanced financial metrics calculator
    """

    def __init__(self, db_config: Dict = None):
        """ Initialize database connection """
        
        is_docker = os.path.exists('/.dockerenv')
        
        env_host = os.getenv('DB_HOST')
        if env_host:
            default_host = env_host
        else:
            default_host = 'postgres' if is_docker else 'localhost'

        default_port = int(os.getenv('DB_PORT', 5432))

        if db_config is None:
            db_config = {
                'host': default_host,
                'port': default_port,
                'database': 'financial_db',
                'user': 'postgres',
                'password': '12345'
            }
            
        logger.info(f"Connecting to database at {db_config['host']}:{db_config['port']}")
        
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # Risk-Free rate (using 10-year Treasury as proxy)
        self.risk_free_rate = 0.02 # 2% annual

        # Output directory 
        self.output_dir = Path('report/analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_price_data(self, symbols: List[str] = None, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load price data from database

        Args:
            symbols: List of symbols (None for all)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with prices
        """    
        query = """

            SELECT
                s.symbol,
                dp.date,
                dp.close,
                dp.daily_return,
                dp.volume
            FROM financial.stocks s
            JOIN financial.daily_prices dp ON s.stock_id = dp.stock_id
            WHERE 1=1
        """
        if symbols:
            symbols_str = "','".join(symbols)
            query += f" AND s.symbol IN ('{symbols_str}')"

        if start_date:
            query += f" AND dp.date >= '{start_date}'"

        if end_date:
            query += f" AND dp.date <= '{end_date}'"
        
        query += " ORDER BY s.symbol, dp.date"

        df = pd.read_sql(query, self.engine, parse_dates=['date'])
        logger.info(f"Loaded {len(df)} rows for {df['symbol'].unique()} symbols")

        return df
    
    def calculate_returns_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create returns matrix for analysis
        Args:
            df: DateFrame with price data

        Returns:
            Pivot table of returns
        """

        # Pivot to get symbols as columns
        returns_matrix = df.pivot(index='date', columns='symbol', values='daily_return')

        # Forward fill any missing values
        returns_matrix = returns_matrix.ffill()
        # Drop any remaining NaN (first rows)
        returns_matrix = returns_matrix.dropna()

        logger.info(f"Return matrix shape: {returns_matrix.shape}")
        return returns_matrix
    
    def calculate_sharpe_ratio(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate sharpe ratio
        Args:
            returns: Series of daily returns
            periods_per_year: Number of periods in a year (252 for daily) 
        Returns:
            Sharpe ratio
        """
        # Excess returns over risk-free rate
        daily_rf = self.risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf

        # Annualized Sharpe ratio
        sharpe =np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

        return sharpe
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Dict:
        """
        Calculate maximum drawdown and relate metrics
        Args:
            Prices : Series of prices

        Returns:
            Dictionary with drawdown metrics
        """
        # Calculate cumulative maximum
        rolling_max = prices.expanding().max()

        # Calculate drawdown
        drawdown = (prices - rolling_max) / rolling_max * 100

        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()

        # Find recovery data
        recovery_mask = prices[prices.index > max_dd_date] >= rolling_max[max_dd_date]
        recovery_date = recovery_mask[recovery_mask].index.min() if any(recovery_mask) else None

        # Calculate drawdown duration
        if recovery_date:
            dd_duration = (recovery_date - max_dd_date).days
        else:
            dd_duration = (prices.index[-1] - max_dd_date).days

        return {
            'max_drawdown': float(max_dd),
            'max_drawdown_date': max_dd_date.strftime('%Y-%m-%d'),
            'recovery_date': recovery_date.strftime('%Y-%m-%d') if recovery_date else None,
            'drawdown_duration_days': dd_duration,
            'current_drawdown' : float(drawdown.iloc[-1])
        }
    
    def calculate_var_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> Dict:
        """
        Calculate Value at Risk (VaR) and calculate VaR
        Args:
            returns: Series of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
        Returns:
            Dictionary with VaR and CVaR
        """
        # Historical VaR
        var = np.percentile(returns,  (1 - confidence_level) * 100)

        # Conditional VaR (Expected Shortfall)
        cvar = returns[returns <= var].mean()

        return {
            'var': float(var),
            'cvar': float(cvar),
            'confidence_level': confidence_level
        }
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta relative to market
        Args:
            stock_returns: Stock daily returns
            market_returns: Market Index daily returns
        
        Returns:
            Beta ceofficient
        """
        # Align dates
        combined = pd.concat([stock_returns, market_returns], axis=1).dropna()

        if len(combined) < 30:
            return np.nan
        
        # Calculate coveriance and variance
        covariance = combined.iloc[:, 0].cov(combined.iloc[:, 1])
        variance = combined.iloc[:, 1].var()

        beta = covariance / variance if variance != 0 else np.nan

        return beta

    def calculate_alpha(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate alpha  (Jensen's alpha)
        Args:
            stock_returns: Stock daily returns
            market_returns: Market index daily raturns

        Returns:
            Alpha (annualized)
        """
        beta = self.calculate_beta(stock_returns, market_returns)

        if np.isnan(beta):
            return np.nan
        
        #Align dates
        combined = pd.concat([stock_returns, market_returns], axis=1).dropna()

        # Expected return based on CAPM
        daily_rf = self.risk_free_rate / 252
        expected_return = daily_rf + beta * (combined.iloc[:, 1].mean() - daily_rf)


        # Actual return
        actual_return = combined.iloc[:, 0].mean()

        # Annualized alpha
        alpha = (actual_return - expected_return) * 252

        return alpha
    
    def calculate_sortino_ratio(self, returns: pd.Series, target_return: float = 0, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (uses downside deviation)
        Args:
            returns: Series of returns
            target_returns: Minimum acceptable return
            periods_per_year: Number of period in a year
        
        Returns:
            Sortino ratio
        """
        # Calculate downside deviation
        downside_returns = returns[returns < target_return]
        downside_dev = np.sqrt(np.mean(downside_returns**2))

        if downside_dev == 0:
            return np.nan
        
        # Annualized excess return
        excess_return = (returns.mean() - target_return / periods_per_year) * periods_per_year

        sortino = excess_return / (downside_dev * np.sqrt(periods_per_year))

        return sortino
    
    def calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate calmar ratio (return / max drawdown)
        Args:
            returns: Series of returns
            prices: Series of prices
            periods_per_year: Number of periods in a year
        
        Returns:
            Calmar ratio
        """
        # Annualized return
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        years = len(returns) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1

        # Max drawdown
        max_dd = self.calculate_max_drawdown(prices)['max_drawdown'] / 100 # Convert to decimal

        if max_dd == 0:
            return np.nan
        
        calmar = annualized_return / abs(max_dd)

        return calmar
    
    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict:
        """
        Comprehensive analysis for a single symbol
        """
        logger.info(f"Analyzing {symbol}")

        symbol_data = df[df['symbol'] == symbol].copy()
        symbol_data = symbol_data.set_index('date').sort_index()

        if len(symbol_data) < 30:
            logger.warning(f"Insufficient data for {symbol}")
            return {'symbol': symbol, 'error': 'Insufficient data'}

        returns = symbol_data['daily_return'].dropna()
        prices = symbol_data['close']
        volumes = symbol_data['volume']

        # Normalisasi: Jika return dalam persen (ex: 1.5), ubah ke desimal (0.015)
        adj_returns = returns / 100 if returns.abs().mean() > 0.1 else returns

        # Moving Averages
        sma_7 = prices.rolling(window=7).mean()
        sma_20 = prices.rolling(window=20).mean()
        ema_10 = prices.ewm(span=10, adjust=False).mean()
        
        # Market Position Signal
        latest_close = prices.iloc[-1]
        latest_sma20 = sma_20.iloc[-1]
        market_position = "NEUTRAL"
        if latest_close > (latest_sma20 * 1.05): market_position = "OVERBOUGHT"
        elif latest_close < (latest_sma20 * 0.95): market_position = "OVERSOLD"

        # YTD Calculation
        current_year = datetime.now().year
        ytd_data = prices[prices.index.year == current_year]
        ytd_return = 0.0
        if not ytd_data.empty:
            year_start_price = ytd_data.iloc[0]
            ytd_return = ((latest_close / year_start_price) - 1) * 100

        # Rolling Volatility 20d (Annualized)
        vol_20d = adj_returns.rolling(window=20).std() * np.sqrt(252) * 100

        # Perbaikan Max Drawdown: Ambil nilai numerik dari dict
        mdd_data = self.calculate_max_drawdown(prices)
        mdd_value = mdd_data.get('max_drawdown', mdd_data.get('drawdown', 0))

        metrics = {
            'symbol': symbol,
            'last_close': float(latest_close),
            'last_volume': float(volumes.iloc[-1]),
            'sma_7': float(sma_7.iloc[-1]) if not np.isnan(sma_7.iloc[-1]) else None,
            'sma_20': float(latest_sma20) if not np.isnan(latest_sma20) else None,
            'ema_10': float(ema_10.iloc[-1]),
            'market_position': market_position,
            'volatility_20d': float(vol_20d.iloc[-1]) if not np.isnan(vol_20d.iloc[-1]) else 0.0,
            'start_date': symbol_data.index.min().strftime('%Y-%m-%d'),
            'end_date': symbol_data.index.max().strftime('%Y-%m-%d'),
            'total_days': len(symbol_data),
            'total_return': float((latest_close / prices.iloc[0] - 1) * 100),
            'ytd_return_pct': float(ytd_return),
            'annualized_return': float(((1 + adj_returns.mean()) ** 252 - 1) * 100),
            'annualized_volatility': float(adj_returns.std() * np.sqrt(252) * 100),
            'sharpe_ratio': float(self.calculate_sharpe_ratio(adj_returns)),
            'sortino_ratio': float(self.calculate_sortino_ratio(adj_returns)),
            'calmar_ratio': float(self.calculate_calmar_ratio(adj_returns, prices)),
            'max_drawdown': float(mdd_value), # Sudah dipastikan float
            'positive_days': int((returns > 0).sum()),
            'negative_days': int((returns < 0).sum()),
            'win_rate': float((returns > 0).sum() / len(returns) * 100),
            'avg_win': float(returns[returns > 0].mean()) if not returns[returns > 0].empty else 0.0,
            'avg_loss': float(returns[returns < 0].mean()) if not returns[returns < 0].empty else 0.0,
            'profit_factor': float(abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.inf),
            'best_day': float(returns.max()),
            'best_day_date': returns.idxmax().strftime('%Y-%m-%d'),
            'worst_day': float(returns.min()),
            'worst_day_date': returns.idxmin().strftime('%Y-%m-%d'),
        }

        return metrics

    def calculate_correlation_matrix(self, returns_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix between symbols
        Args:
            returns_matrix: Pivot table or returns
        Returns:
            Correlation matrix
        """
        corr_matrix = returns_matrix.corr()
        logger.info(f"Calculated correlation matrix: {corr_matrix.shape}")
        return corr_matrix
    
    def calculate_rolling_correlation(self, returns_matrix: pd.DataFrame, window: int = 60) -> Dict[str, pd.Series]:
        """
        Calculate rolling correlations
        
        Args:
            returns_matrix: Pivot table of returns
            window: Rolling window size
        
        Returns:
            Dictionary of rolling correlations
        """
        rolling_corr = {}
        symbols = returns_matrix.columns

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                pair = f"{sym1}-{sym2}"
                rolling_corr[pair] = returns_matrix[sym1].rolling(window).corr(
                    returns_matrix[sym2]
                )

        logger.info(f"Calculated Rolling Correlation for {len(rolling_corr)} pairs")
        return rolling_corr
    
    def generate_report(self, symbols: List[str] = None) -> Dict:
        """
        Generate comprehensive financial analysis report
        """
        logger.info("Generating comprehensive financial analysis...")

        # Load data
        df = self.load_price_data(symbols)

        if df.empty:
            logger.error("No Data Found")
            return {}
        
        # Create return matrix
        returns_matrix = self.calculate_returns_matrix(df)

        # Analyze each symbol
        symbol_metrics = {} 
        for symbol in returns_matrix.columns:
            try:
                metrics = self.analyze_symbol(symbol, df)
                if 'error' not in metrics:
                    symbol_metrics[symbol] = metrics
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                symbol_metrics[symbol] = {'error': str(e)}

        # Calculate correlations
        correlation_matrix = self.calculate_correlation_matrix(returns_matrix)

        # Calculate portfolio metrics (equal weight)
        adj_returns_matrix = returns_matrix / 100 if returns_matrix.abs().mean().mean() > 0.1 else returns_matrix
        portfolio_returns = adj_returns_matrix.mean(axis=1)
        portfolio_prices = (1 + portfolio_returns).cumprod() * 100

        # Perbaikan Max Drawdown Portofolio: Ambil nilai numerik dari dict
        p_mdd_data = self.calculate_max_drawdown(portfolio_prices)
        p_mdd_value = p_mdd_data.get('max_drawdown', p_mdd_data.get('drawdown', 0))

        portfolio_metrics = {
            'symbols': list(returns_matrix.columns),
            'avg_correlation': float(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()) if not correlation_matrix.empty else 0.0,
            'portfolio_volatility': float(portfolio_returns.std() * np.sqrt(252) * 100),
            'portfolio_sharpe': float(self.calculate_sharpe_ratio(portfolio_returns)),
            'portfolio_max_drawdown': float(p_mdd_value) # Perbaikan di sini
        }

        # Compile full report
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'trading_days': len(returns_matrix)
            },
            'symbols_analyzed': list(returns_matrix.columns),
            'symbol_metrics': symbol_metrics, 
            'symbols_metrics': symbol_metrics,
            'correlation_matrix': correlation_matrix.round(4).to_dict(),
            'portfolio_metrics': portfolio_metrics,
            'risk_free_rate': self.risk_free_rate
        }

        # Save to JSON
        output_file = self.output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        def stringify_keys(d):
            if isinstance(d, dict):
                return {str(k): stringify_keys(v) for k, v in d.items()}
            return d

        cleaned_report = stringify_keys(report)
        with open(output_file, "w") as f:
            json.dump(cleaned_report, f, indent=2, default=str)

        logger.info(f"Analysis saved to {output_file}")
        return report

    def print_summary(self, report: Dict):
        """Print summary of analysis"""
        print("\n" + "="*60)
        print(" FINANCIAL ANALYSIS SUMMARY")
        print("="*60)

        print(f"\nPeriod {report['data_range']['start']} to {report['data_range']['end']}")
        print(f"Trading Days: {report['data_range']['trading_days']}")

        print("\nSymbol Performance:")
        print("-"*40)
        for symbol, metrics in report['symbols_metrics'].items():
            if 'error' not in metrics:
                print(f"\n{symbol}")
                print(f" Total Returns: {metrics['total_return']:.2f}%")
                print(f" Sharpe Ratio: {metrics['sharpe_ratio']:.2f}%")
                print(f" Max Drawdown: {metrics['max_drawdown']['max_drawdown']:.2f}%")
                print(f" Win Rate: {metrics['win_rate']:.1f}%")
                
        print("\n Correlation Matrix:")
        print("-"*40)
        corr_df = pd.DataFrame(report['correlation_matrix'])
        print(corr_df.round(2))

        print("\n Portofolio Metrics:")
        print("-"*40)
        print(f"  Avg Correlation: {report['portofolio_metrics']['avg_correlation']:.3f}")
        print(f"  Portpfolio Sharpe: {report['portofolio_metrics']['portofolio_sharpe']:.2f}")
        print(f"  Portofolio Max DD: {report['portofolio_metrics']['portofolio_max_drawdown']['max_drawdown']:.2f}%")
        
        print("\n" + "="*60)

def main():
    """Main function to run analysis"""
    analyzer = FinancialAnalyzer()

    # Analyze specific symbols
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MFST', 'AMZN']

    report = analyzer.generate_report(symbols)
    analyzer.print_summary(report)

    return report


if __name__=="__main__":
    report = main()
