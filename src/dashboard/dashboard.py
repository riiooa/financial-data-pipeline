"""
Steamlit Dashboard for Financial Data
Intractive dashboard with real-time updates
"""

from turtle import title, width
import os
from dotenv import load_dotenv
from altair import Opacity
from git import refresh
from matplotlib import legend
from matplotlib.pyplot import show
from sqlalchemy.util import symbol
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.analysis import FinancialAnalyzer

load_dotenv()
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Page config
st.set_page_config(
    page_title="Financial Data Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ffffff; 
        padding: 1.2rem;
        border-radius: 0.8rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #1a1a1a !important;
        margin-top: 0;
        margin-bottom: 5px;
    }
    .metrics-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E88E5 !important; 
    }
    .metric-info {
        color: #555555 !important;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

class FinancialDashboard:
    """
    Interactive dashboard for financial data
    """

    def __init__(self):
        """Initialize dashboard"""
        self.analyzer = FinancialAnalyzer()
        self.engine = self.analyzer.engine

        # Avaiable symbols
        self.symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'META', 'NVDA']

        # Color scheme
        self.colors = px.colors.qualitative.Set2

    def load_data(self, symbols, start_date, end_date):
        """Load data for selected symbols"""
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
                dp.rsi,
                dp.daily_return
            FROM financial.daily_prices dp
            JOIN financial.stocks s ON dp.stock_id = s.stock_id
            WHERE s.symbol = ANY(%s)
                AND dp.date BETWEEN %s AND %s
            ORDER BY dp.date
        """

        df = pd.read_sql(
            query,
            self.engine,
            params=(list(symbols), str(start_date), str(end_date)), 
            parse_dates=['date']
        )

        return df

    def get_latest_metrics(self, symbols):
        """Get latest metrics for selected symbols"""
        query = """
            WITH latest_dates AS (
                SELECT
                    s.symbol,
                    MAX(dp.date) as latest_date
                FROM financial.daily_prices dp
                JOIN financial.stocks s ON dp.stock_id = s.stock_id
                WHERE s.symbol = ANY(%s)
                GROUP BY s.symbol
            )
            SELECT
                s.symbol,
                dp.date,
                dp.close,
                dp.daily_return,
                dp.rsi,
                dp.volatility_annualized
            FROM financial.daily_prices dp
            JOIN financial.stocks s ON dp.stock_id = s.stock_id
            JOIN latest_dates ld ON s.symbol = ld.symbol AND dp.date = ld.latest_date
            WHERE s.symbol = ANY(%s)
            ORDER BY s.symbol
        """

        df = pd.read_sql(
            query, 
            self.engine, 
            params=(list(symbols), list(symbols))
        )
        return df
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">Financial Data Dashboard</h1>', unsafe_allow_html=True)

        st.markdown("""
        Welcome to the interactive financial data dashboard. This dashboard providers
        real-time analytics for major tech stocks including price trends, technical indicator,
        and performance metrics.
        """)
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.header("Controls")

            # Symbol selection
            selected_symbols = st.multiselect(
                "Select stocks",
                self.symbols,
                default=['AAPL', 'GOOGL', 'TSLA']
            )

            if not selected_symbols:
                st.warning("Please select at leats one stock")
                return None, None, None
            
            # Date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    datetime.now() - timedelta(days=180)
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    datetime.now()
                )

            if start_date > end_date:
                st.error("Start date must be before end date")
                return None, None, None
            
            # Charts type
            chart_type = st.selectbox(
                "Chart Type",
                ["Line", "Candlestick", "Area"]
            )

            # Indicators
            show_ma = st.checkbox("Show Moving Averages", value=True)
            show_rsi = st.checkbox("Show RSI", value=True)

            # Refresh rate
            refresh_rate = st.select_slider(
                "Auto-refresh (seconds)",
                options=[0, 30, 60, 300],
                value=0,
                format_func=lambda x: "Off" if x == 0 else f"{x}s"
            )

            return selected_symbols, start_date, end_date, chart_type, show_ma
        
    def render_metrics(self, df_latest):
        """Render metrics cards"""
        st.header("Live Metrics")

        cols = st.columns(len(df_latest))
        for idx, (_, row) in enumerate(df_latest.iterrows()):
            with cols[idx]:
                # Determine color based on daily return
                color = "#2e7d32" if row['daily_return'] > 0 else "#d32f2f" 

                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['symbol']}</h3>
                    <div class="metrics-value">${row['close']:.2f}</div>
                    <div style="color: {color}; font-weight: bold; font-size: 1.1rem; margin: 8px 0;">
                        {row['daily_return']:+.2f}%
                    </div>
                    <div class="metric-info">
                        RSI: {row['rsi']:.1f} • Vol: {row['volatility_annualized']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

    def render_price_chart(self, df, selected_symbols, chart_type, show_ma):
        """Render Price Chart"""
        st.header("Price Chart")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price History", "Volume")
        )

        # Add price traces
        for i, symbol in enumerate(selected_symbols):
            symbol_data = df[df['symbol'] == symbol]

            if chart_type == "Candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=symbol_data['date'],
                        open=symbol_data['open'],
                        high=symbol_data['high'],
                        low=symbol_data['low'],
                        close=symbol_data['close'],
                        name=symbol,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            elif chart_type == "Line":
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'],
                        y=symbol_data['close'],
                        mode='lines',
                        name=symbol,
                        line=dict(color=self.colors[i], width=2)
                    ),
                    row=1, col=1
                )
            else: # Area
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'],
                        y=symbol_data['close'],
                        fill='tozeroy',
                        mode='lines',
                        name=symbol,
                        line=dict(color=self.colors[i], width=2)
                    ),
                    row=1, col=1
                )

            # Add Moving Averages
            if show_ma:
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'],
                        y=symbol_data['sma_20'],
                        mode='lines',
                        name=f'{symbol} SMA20',
                        line=dict(color=self.colors[i], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data['date'],
                        y=symbol_data['sma_20'],
                        mode='lines',
                        name=f'{symbol} SMA50',
                        line=dict(color=self.colors[i], width=1, dash='dot')
                    ),
                    row=1, col=1
                )

        # Add volume bars
        for i, symbol in enumerate(selected_symbols):
            symbol_data = df[df['symbol'] == symbol]

            fig.add_trace(
                go.Bar(
                    x=symbol_data['date'],
                    y=symbol_data['volume'],
                    name=f'{symbol} Volume',
                    marker_color=self.colors[i],
                    opacity=0.5,
                    showlegend=False
                ),
                row=2, col=1
            )
            
        # Update layout
        fig.update_layout(
            height=600,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        st.plotly_chart(fig, width='stretch')


    def render_rsi_chart(self, df, selected_symbols):
        """Render RSI Chart"""
        st.header("Relative Strength Index (RSI)")

        fig = go.Figure()

        for i, symbol in enumerate(selected_symbols):
            symbol_data = df[df['symbol'] == symbol]

            fig.add_trace(
                go.Scatter(
                    x=symbol_data['date'],
                    y=symbol_data['rsi'],
                    mode='lines',
                    name=symbol,
                    line=dict(color=self.colors[i], width=2)
                )
            )

        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash='dash', line_color='red', annotation_text="Overbought")
        fig.add_hline(y=30, line_dash='dash', line_color='green', annotation_text="Oversold")
        fig.add_hline(y=70, line_dash='dot', line_color='gray', opacity=0.5)

        fig.update_layout(
            height=400,
            hovermode='x unified',
            template='plotly_white',
            yaxis_range=[0, 100]
        )

        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="RSI")

        st.plotly_chart(fig, width='stretch')
    
    def render_correlation_matrix(self, df, selected_symbols):
        """Render correlation matrix"""
        st.header("Correlation Matrix")

        # Calculate returns matrix
        returns_matrix = pd.DataFrame()
        for symbol in selected_symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date')
            returns = symbol_data['close'].pct_change()
            returns_matrix[symbol] = returns

        corr_matrix = returns_matrix.corr()

        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect="auto",
            color_continuous_scale='RdYlGn',
            title="Returns Correlation"
        )

        fig.update_layout(height=500)

        st.plotly_chart(fig, width='stretch')

    def render_performance_table(self, df, selected_symbols):
        """Render performance metrics table"""
        st.header("Performance Metrics")

        metrics_data = []

        for symbol in selected_symbols:
            symbol_data = df[df['symbol'] == symbol].sort_values('date')

            if len(symbol_data) < 2:
                continue


            returns = symbol_data['close'].pct_change().dropna()
            total_return = (symbol_data['close'].iloc[-1] / symbol_data['close'].iloc[0] - 1) * 100

            metrics = {
                'Symbol': symbol,
                'Current Price': f"${symbol_data['close'].iloc[-1]:.2f}",
                'Total Return': f"{total_return:.2f}%",
                'Avg Daily Return': f"{returns.mean()*100:.3f}",
                'Volatility': f"{returns.std()*100:.2f}%",
                'Sharpe Ratio': f"{(returns.mean() / returns.std() * np.sqrt(252)):.2f}" if returns.std() > 0 else "N/A",
                'RSI': f"{symbol_data['rsi'].iloc[-1]:.1f}",
                "Volume": f"{symbol_data['volume'].iloc[-1]:,.0f}"
            }

            metrics_data.append(metrics)

        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, width='stretch', hide_index=True)


    def run(self):
        """Run the Dashboard"""
        self.render_header()

        # Get user inputs from sidebar
        sidebar_data = self.render_sidebar()
        
        if not sidebar_data or sidebar_data[0] is None:
            return
            
        selected_symbols, start_date, end_date, chart_type, show_ma = sidebar_data

        # Convert dates to string
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Load data
        with st.spinner('Loading data...'):
            df = self.load_data(selected_symbols, start_date_str, end_date_str)
            df_latest = self.get_latest_metrics(selected_symbols)

        if df.empty:
            st.error("No data available for selected parameters")
            return
        
        # Render metrics
        self.render_metrics(df_latest)

        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Price Charts", "Technical Indicators", "Analysis"])


        with tab1:
            self.render_price_chart(df, selected_symbols, chart_type, show_ma)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                self.render_rsi_chart(df, selected_symbols)
            with col2:

                fig = px.bar(
                    df[df['symbol'] == selected_symbols[0]],
                    x='date',
                    y='volume',
                    title=f"Volume - {selected_symbols[0]}",
                    template='plotly_white'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')

        with tab3:
            col1, col2 = st.columns([1, 1])
            with col1:
                self.render_correlation_matrix(df, selected_symbols)
            with col2:
                self.render_performance_table(df, selected_symbols)


        if st.session_state.get('refresh_rate', 0) > 0:
            st.empty()
            st.caption(f"Auto-refreshing every {st.session_state.refresh_rate} seconds")
            st.rerun()
    
if __name__=="__main__":
    dashboard = FinancialDashboard()
    dashboard.run()