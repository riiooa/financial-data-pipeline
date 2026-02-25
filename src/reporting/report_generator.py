"""
Automated PDF Report Generator for Financial Analysis
"""
import hashlib
# Fix for ReportLab md5 error in some environments
try:
    hashlib.md5(usedforsecurity=False)
except TypeError:
    # If the error occurs, we redefine md5 to ignore the 'usedforsecurity' argument
    real_md5 = hashlib.md5
    def md5_fix(data=b'', usedforsecurity=True):
        return real_md5(data)
    hashlib.md5 = md5_fix

import os
import logging
import json
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import smtplib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialReportGenerator:
    """Generates professional PDF reports for financial analysis"""

    def __init__(self, output_dir="reports"):
        """Initialize report generator"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        
        # Define Colors
        self.primary_color = colors.HexColor('#1E88E5')
        self.secondary_color = colors.HexColor('#A23B72')
        self.dark_gray = colors.HexColor('#757575')
        
        self.setup_custom_styles()

    def setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=self.primary_color,
            spaceAfter=30,
            alignment=1
        ))
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=self.primary_color,
            spaceBefore=20,
            spaceAfter=10
        ))
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.secondary_color,
            spaceBefore=15,
            spaceAfter=5
        ))
        self.styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=self.dark_gray,
            alignment=1
        ))

    def create_table_from_df(self, df: pd.DataFrame, max_rows: int = 20) -> Table:
        if len(df) > max_rows:
            df = df.head(max_rows)
        
        data = [df.columns.tolist()] + df.values.tolist()
        t = Table(data, hAlign='LEFT')
        
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.primary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ])
        
        # Zebra Striping
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.add('BACKGROUND', (0, i), (-1, i), colors.whitesmoke)
        
        t.setStyle(style)
        return t

    def create_metric_table(self, metrics: dict) -> Table:
        data = [['Metric', 'Value']]
        for key, value in metrics.items():
            data.append([str(key), str(value)])

        t = Table(data, colWidths=[2.5*inch, 1.5*inch], hAlign='LEFT')
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.secondary_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ])
        t.setStyle(style)
        return t

    def create_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> Image:
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0, square=True)
        plt.title('Returns Correlation Matrix', fontsize=12, fontweight='bold')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        img = Image(buf)
        img.drawHeight = 3*inch
        img.drawWidth = 4*inch
        return img

    def _create_market_intelligence_section(self, symbols_metrics):
        """ Membuat tabel ringkasan kondisi pasar """
        from reportlab.platypus import Table, TableStyle
        from reportlab.lib import colors

        # Header Tabel
        data = [['Symbol', 'Price', 'Signal', 'YTD Return', 'Vol (20d)']]
        
        for symbol, m in symbols_metrics.items():
            if 'error' in m:
                continue
            
            price = f"${m.get('last_close', 0):.2f}"
            signal = m.get('market_position', 'NEUTRAL')
            ytd = f"{m.get('ytd_return_pct', 0):.2f}%"
            vol = f"{m.get('volatility_20d', 0):.2f}%"
            
            data.append([symbol, price, signal, ytd, vol])

        # Styling Tabel
        table = Table(data, colWidths=[80, 80, 100, 100, 80])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table

    def _create_market_summary_table(self, analysis_data):
    
        # Header Tabel
        data = [['Symbol', 'Last Close', 'Market Signal', 'YTD Return', 'Volatility']]
        
        symbols_metrics = analysis_data.get('symbols_metrics', {})
        
        for symbol, metrics in symbols_metrics.items():
            if 'error' in metrics: continue
            
            signal = metrics.get('market_position', 'NEUTRAL')
            ytd = f"{metrics.get('ytd_return_pct', 0):.2f}%"
            vol = f"{metrics.get('volatility_20d', 0):.2f}%"
            price = f"${metrics.get('last_close', 0):.2f}"
            
            data.append([symbol, price, signal, ytd, vol])

        # Styling agar terlihat profesional
        table = Table(data, colWidths=[70, 90, 110, 90, 90])
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a237e')), # Biru gelap
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ])
        
        # Pewarnaan Sinyal Otomatis
        for i, row in enumerate(data[1:], 1):
            if row[2] == 'OVERBOUGHT':
                style.add('TEXTCOLOR', (2, i), (2, i), colors.red)
            elif row[2] == 'OVERSOLD':
                style.add('TEXTCOLOR', (2, i), (2, i), colors.green)
                
        table.setStyle(style)
        return table

    def generate_pdf_report(self, analysis_report: dict, output_filename: str = None) -> str:
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"financial_report_{timestamp}.pdf"

        output_path = self.output_dir / output_filename
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []

        # 1. Header
        story.append(Paragraph("Financial Analysis Report", self.styles['CustomTitle']))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", self.styles['CustomNormal']))
        story.append(Spacer(1, 0.1*inch))

        # 2. Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        summary = f"""This report analyzes {len(analysis_report.get('symbol_metrics', {}))} stocks. 
                     Data range: {analysis_report.get('data_range', {}).get('start', 'N/A')} to 
                     {analysis_report.get('data_range', {}).get('end', 'N/A')}."""
        story.append(Paragraph(summary, self.styles['CustomNormal']))

        # 3. MARKET INTELLIGENCE (BAGIAN BARU)
        story.append(Paragraph("Market Intelligence & Signals", self.styles['CustomHeading2']))
        story.append(self._create_market_summary_table(analysis_report))
        story.append(Spacer(1, 0.2*inch))

        # 4. Portfolio Table
        story.append(Paragraph("Portfolio Performance Metrics", self.styles['CustomHeading2']))
        p_metrics = analysis_report.get('portfolio_metrics', {})
        summary_data = {
            'Sharpe Ratio': f"{p_metrics.get('portfolio_sharpe', 0):.2f}",
            'Volatility (Annual)': f"{p_metrics.get('portfolio_volatility', 0):.2f}%",
            'Avg Symbol Correlation': f"{p_metrics.get('avg_correlation', 0):.3f}"
        }
        story.append(self.create_metric_table(summary_data))
        story.append(Spacer(1, 0.2*inch))

        # 5. Heatmap
        if 'correlation_matrix' in analysis_report:
            story.append(Paragraph("Correlation Analysis", self.styles['CustomHeading1']))
            corr_df = pd.DataFrame(analysis_report['correlation_matrix'])
            story.append(self.create_correlation_heatmap(corr_df))

        # 6. Footer
        story.append(Paragraph("Generated by Financial Data Pipeline System", self.styles['Footer']))

        doc.build(story)
        return str(output_path)

    def send_email_report(self, pdf_path: str, recipient: str = "user@example.com", subject: str = "Daily Financial Analysis Report"):
        """
        Send report via email or fallback to simulation if credentials are missing.
        """
        email_user = os.getenv('EMAIL_USER')
        email_password = os.getenv('EMAIL_PASSWORD')

        # Cek apakah kredensial tersedia dan bukan nilai default
        is_missing = not email_user or not email_password
        is_default = email_user == "username@gmail.com"

        if is_missing or is_default:
            logger.info("="*60)
            logger.info("EMAIL SIMULATION MODE ACTIVE")
            logger.info("="*60)
            logger.info(f"Recipient : {recipient}")
            logger.info(f"Subject   : {subject}")
            logger.info(f"Attachment: {os.path.basename(pdf_path)}")
            logger.info(f"Location  : {pdf_path}")
            logger.info("-"*60)
            logger.info("Status    : SUCCESS (Simulation only, no email sent)")
            logger.info("Reason    : Credentials missing or still using default values in .env")
            logger.info("="*60)
            return

        # Jika kredensial valid, jalankan pengiriman asli
        try:
            msg = MIMEMultipart()
            msg['From'] = email_user
            msg['To'] = recipient
            msg['Subject'] = subject

            body = "Please find the attached daily financial analysis report."
            msg.attach(MIMEText(body, 'plain'))

            # Handle attachment
            with open(pdf_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition', 
                    f'attachment; filename={os.path.basename(pdf_path)}'
                )
                msg.attach(part)

            # SMTP Server Execution
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(email_user, email_password)
                server.send_message(msg)
            
            logger.info(f"Real email sent successfully to {recipient}!")
            
        except Exception as e:
            logger.error(f"Failed to send real email: {e}")
            logger.warning("Check your App Password or SMTP settings.")

    def generate_daily_report(self, analysis_report: dict):
        pdf_path = self.generate_pdf_report(analysis_report)
        self.send_email_report(pdf_path)
        return pdf_path

if __name__ == "__main__":
    mock_report = {
        'symbol_metrics': {
            'AAPL': {'last_close': 185.20, 'market_position': 'OVERBOUGHT', 'ytd_return_pct': 12.5, 'volatility_20d': 15.2},
            'TSLA': {'last_close': 210.15, 'market_position': 'OVERSOLD', 'ytd_return_pct': -5.2, 'volatility_20d': 35.8},
            'GOOGL': {'last_close': 145.10, 'market_position': 'NEUTRAL', 'ytd_return_pct': 8.1, 'volatility_20d': 18.4}
        },
        'data_range': {'start': '2026-01-01', 'end': '2026-02-24'},
        'portfolio_metrics': {'portfolio_sharpe': 1.5, 'portfolio_volatility': 12.5, 'avg_correlation': 0.6},
        'correlation_matrix': {'AAPL': {'AAPL': 1.0, 'MSFT': 0.5}, 'MSFT': {'AAPL': 0.5, 'MSFT': 1.0}}
    }
    reporter = FinancialReportGenerator(output_dir="reports")
    reporter.generate_daily_report(mock_report)