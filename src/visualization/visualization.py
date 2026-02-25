"""
Financial Data Visualization
Creates prefessional charts for financial analysis
"""

from ast import Dict, List
import enum
from fileinput import filename
from locale import normalize

from click import style
from matplotlib import axes
from matplotlib.lines import lineStyles
from matplotlib.patches import bbox_artist
from narwhals import col
import pandas as pd
import numpy as np
from pyparsing import alphas
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mplfinance as mpf
from datetime import date, datetime, timedelta
from pathlib import Path
import logging
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FinancialVisualizer:
    """
    Create professional financial visualizations
    """

    def __init__(self, db_config: Dict = None):
        """Initialize visualized"""
        if db_config is None:
            db_config = {
                'host': 'localhost',
                'port': 5433,
                'database': 'financial_db',
                'user': 'postgres',
                'password': '12345'
            }
        
        self.engine = create_engine(
            f"postgresql://{db_config['user']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        # Output directory
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(parents=True, exist_ok=True)


        # Color schema
        self.colors = plt.cm.Set2(np.linspace(0, 1, 10))

    def load_data(self, symbols: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load data from database"""
        
        query = """
            SELECT 
                s.symbol, 
                dp.date, 
                dp.open, 
                dp.high, 
                dp.low, 
                dp.close, 
                dp.volume, 
                dp.sma_20, 
                dp.sma_50, 
                dp.rsi
            FROM financial.daily_prices dp
            JOIN financial.stocks s ON dp.stock_id = s.stock_id
            WHERE s.symbol = ANY(%s)
        """
        
        params = [symbols] 

        if start_date:
            query += " AND dp.date >= %s"
            params.append(start_date)

        if end_date:
            query += " AND dp.date <= %s"
            params.append(end_date)

        query += " ORDER BY s.symbol, dp.date"

        df = pd.read_sql(query, self.engine, params=tuple(params), parse_dates=['date'])
        logger.info(f"Loaded {len(df)} rows for {len(symbols)} symbols")

        return df

    def plot_prices_trends(self, df: pd.DataFrame, symbols: List[str] = None, save: bool = True):
        """
        Plot price trends with moving averages
        
        Args:
            df: DataFrame with price data
            symbols: List of symbols to plot
            save: Whether to save plot
        """
        if symbols is None:
            symbols = df['symbol'].unique()

        fig, axes = plt.subplots(len(symbols), 1, figsize=(14, 4*len(symbols)))
        if len(symbols) == 1:
            axes = [axes]

        for idx, symbol in enumerate(symbols):
            symbol_data = df[df['symbol'] == symbol].sort_values('date')

            ax = axes[idx]

            # plot prices
            ax.plot(symbol_data['date'], symbol_data['close'], label='Close Price', linewidth=2, color='#2E86AB')

            # Plot moving averages
            ax.plot(symbol_data['date'], symbol_data['sma_20'], label='SMA 20', linewidth=1.5, linestyle='--', color='#A23B72')
            ax.plot(symbol_data['date'], symbol_data['sma_50'], label='SMA 50', linewidth=1.5, linestyle='--', color='#F18F01')

            ax.set_title(f'{symbol} - Price Trend with Moving Averages', fontsize=14, fontweight='bold')
            ax.set_ylabel('Price ($)')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            # Add volume as subplot
            ax2 = ax.twinx()
            ax2.bar(symbol_data['date'], symbol_data['volume'], alpha=0.2, color='gray', label='volume')
            ax2.set_ylabel('Volume')

        plt.tight_layout()

        if save:
            filename = self.plots_dir / f"prices trends_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved price trends  to {filename}")

        plt.show()

    def plot_returns_distribution(self, df: pd.DataFrame, symbols: List[str] = None, save: bool = True):
        """
        Plot distribution of returns
        
        Args:
            df: DataFrame with price data
            symbols: List of symbols
            save: Whether to save plot
        """
        if symbols is None:
            symbols = df['symbol'].unique()

        # Calculate daily returns
        fig, axes = plt.subplots(1, len(symbols), figsize=(6*len(symbols), 5))
        if len(symbols) == 1:
            axes = [axes]

        for idx, symbol in enumerate(symbols):
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            returns = symbol_data['close'].pct_change().dropna() * 100

            ax = axes[idx]

            # Histogram with KDE
            sns.histplot(returns, kde=True, ax=ax, color=self.colors[idx], bins=50)

            # Add statistics
            mean_ret = returns.mean()
            std_ret = returns.std()

            ax.axvline(mean_ret, color='red', linestyle='--', label=f'Mean: {mean_ret:.2f}%')
            ax.axvline(mean_ret - std_ret, color='orange', linestyle=':', alpha=0.5)
            ax.axvline(mean_ret + std_ret, color='orange', linestyle=':', alpha=0.5)

            ax.set_title(f'{symbol} - Daily Returns Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Daily Return (%)')
            ax.set_ylabel('Frequency')
            ax.legend()

            # Add stats box
            stats_text = f"Skewness: {returns.skew():.2f}\nKurtosis: {returns.kurtosis():.2f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()

        if save:
            filename = self.plots_dir / f"returns_dist_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved returns distribution to {filename}")

        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame, symbols: List[str] = None, save: bool = True):
        """
        Plot correlation heatmap
        
        Args:
            df: DataFrame with price data
            symbols: List of symbols
            save: Whether to save plot
        """
        if symbols is None:
            symbols = df['symbol'].unique()

        # Calculate returns matrix
        returns_matrix = pd.DataFrame()
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            returns = symbol_data['close'].pct_change()
            returns_matrix[symbol] = returns

        # Calculate correlation
        corr_matrix = returns_matrix.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix), k=1)

        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn', center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})

        ax.set_title('Stock Returns Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            filename = self.plots_dir / f"correlation_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved correlation heatmep to {filename}")

        plt.show()

        return corr_matrix
    
    def plot_rolling_volatility(self, df: pd.DataFrame, window: int = 20, symbols: List[str] = None, save: bool = True):
        """
        Plot rolling volatility
        
        Args:
            df: DataFrame with price data
            window: Rolling window
            symbols: List of symbols
            save: Whether to save plot
        """
        if symbols is None:
            symbols = df['symbol'].unique()

        fig, ax = plt.subplots(figsize=(14, 6))

        for idx, symbol in enumerate(symbols):
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            returns = symbol_data['close'].pct_change()
            rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100 # Annualized %

            ax.plot(symbol_data['date'].iloc[window:], rolling_vol.iloc[window:], label=symbol, linewidth=2, color=self.colors[idx])

        ax.set_title(f'{window}-Day Rolling Volatility (Annualized)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility %')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = self.plots_dir / f"rolling_vol_{datetime.now().strftime('%Y%m%d')}.png" 
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved rolling volatility to {filename}")

        plt.show()

    def plot_relative_performance(self, df: pd.DataFrame, base_symbol: str = 'SPY', symbols: List[str] = None, save: bool = True):
        """Plot relative performance (normalized to 100)"""
        if symbols is None:
            symbols = [s for s in df['symbol'].unique() if s != base_symbol]

        fig, ax = plt.subplots(figsize=(14, 6))

        # Kita gunakan list baru untuk memastikan base_symbol diproses duluan
        for idx, symbol in enumerate([base_symbol] + symbols):
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            
            # --- PROTEKSI DI SINI ---
            if len(symbol_data) == 0:
                logger.warning(f"Data untuk {symbol} kosong, melewati visualisasi ini.")
                continue
            # ------------------------

            normalized = symbol_data['close'] / symbol_data['close'].iloc[0] * 100

            # Perbaikan typo: linewidth (bukan linewidht)
            linewidth = 3 if symbol == base_symbol else 1.5
            alpha = 1.0 if symbol == base_symbol else 0.7

            ax.plot(
                symbol_data['date'], 
                normalized, 
                label=symbol, 
                linewidth=linewidth, 
                alpha=alpha, 
                color='black' if symbol == base_symbol else self.colors[idx % len(self.colors)]
            )  

        ax.set_title('Relative Performance (Base 100)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price (Base 100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5) # Pakai axhline lebih stabil

        plt.tight_layout()

        if save:
            filename = self.plots_dir / f"relative_perf_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved relative performance to {filename}")

        plt.show()


    def plot_rsi(self, df: pd.DataFrame, symbols: List[str] = None, save: bool = True):
        """
        Plot RSI with overbought/oversold levels
        
        Args:
            df: DataFrame with price data
            symbols: List of symbols
            save: Whether to save plot
        """
        if symbols is None:
            symbols = df['symbol'].unique()[:2]

        fig, axes = plt.subplots(len(symbols), 1, figsize=(14, 5*len(symbols)))
        if len(symbols) == 1:
            axes = [axes]

        for idx, symbol in enumerate(symbols):
            symbol_data = df[df['symbol'] == symbol].sort_values('date')

            ax = axes[idx]

            # Plot rsi
            ax.plot(symbol_data['date'], symbol_data['rsi'], linewidth=2, color='purple')

            # Overbought/Oversold levels
            ax.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
            ax.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
            ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

            # Fill between
            ax.fill_between(symbol_data['date'], 70, 100, alpha=0.2, color='red')
            ax.fill_between(symbol_data['date'], 0, 30, alpha=0.2, color='green')

            ax.set_title(f'{symbol} - Relative Strength Index (RSI)', fontsize=12, fontweight='bold')
            ax.set_ylabel('RSI')
            ax.set_ylim(0, 100)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = self.plots_dir / f"rsi_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved RSI Chart to {filename}")

        plt.show()

    def plot_candlestick(self, symbol: str, df: pd.DataFrame, days: int = 90, save : bool = True):
        """
        Plot candlestick chart using mplfinance
        
        Args:
            symbol: Stock symbol
            df: DataFrame with OHLC data
            days: Number of days to plot
            save: Whether to save plot
        """
        # Filter for symbol and last N days
        symbol_data =  df[df['symbol'] == symbol].sort_values('date').tail(days)

        # Prepare data for mplfinance
        ohlc_data = symbol_data.set_index('date')
        ohlc_data = ohlc_data[['open', 'high', 'low', 'close', 'volume']]

        # Create style
        mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)

        # Plot
        fig, axes = mpf.plot(ohlc_data, type='candle', style='charles', volume=True, figsize=(14, 8), title=f'\n{symbol}- Candlestick Chart (Last {days} days)', returnfig=True)

        if save:
            filename = self.plots_dir / f"candlestick_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved Candlestick Chart to {filename}")

        plt.show()

    def create_interactive_plot(self, df: pd.DataFrame, symbols: List[str] = None, save: bool = True):
        """
        Create interactive plot using Plotly
        
        Args:
            df: DataFrame with price data
            symbols: List of symbols
            save: Whether to save as HTML
        """
        if symbols is None:
            symbols = df['symbol'].unique()

        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Add price traces
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date')

            fig.add_trace(
                go.Scatter(x=symbol_data['date'], y=symbol_data['close'], mode='lines', name=symbol, line=dict(width=2)),
                row=1, col=1
            )
        
        # Add volume bars
        for symbol in symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date')

            fig.add_trace(  
                go.Bar(x=symbol_data['date'], y=symbol_data['volume'], name=f'{symbol} volume', opacity=0.3),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Interactive Stock Prices',
            xaxis_title='Date',
            hovermode='x unified',
            template='plotly_white',
            height=800
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        if save:
            filename = self.plots_dir / f"interactive_{datetime.now().strftime('%Y%m%d')}.html"
            fig.write_html(str(filename))
            logger.info(f"Saved interactive plot to {filename}")
        
        fig.show()
    
    def generate_all_plots(self, symbols: List[str] = None):
        """
        Generate all visualization plots
        
        Args:
            symbols: List of symbols to plot
        """
        if symbols is None:
            symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
        
        logger.info("Generating all visualizations...")
        
        # Load data
        df = self.load_data(symbols, start_date='2023-01-01')
        
        # Generate plots
        self.plot_prices_trends(df, symbols)
        self.plot_returns_distribution(df, symbols)
        self.plot_correlation_heatmap(df, symbols)
        self.plot_rolling_volatility(df, symbols=symbols)
        self.plot_relative_performance(df, symbols=symbols)
        self.plot_rsi(df, symbols)
        
        # Candlestick for each symbol
        for symbol in symbols[:3]:  # Limit to 3 for candlestick
            self.plot_candlestick(symbol, df)
        
        # Interactive plot
        self.create_interactive_plot(df, symbols)
        
        logger.info(f"All plots saved to {self.plots_dir}")


def main():
    """Main function to generate visualizations"""
    visualizer = FinancialVisualizer()
    
    symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
    
    visualizer.generate_all_plots(symbols)
    
    print(f"\nVisualizations saved to 'plots/' directory")
    print(f"Check the following files:")
    for f in Path('plots').glob('*.png'):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()