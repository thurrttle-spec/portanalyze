"""
PDF Report Generator with Embedded Charts

Generates comprehensive PDF reports including all charts and analysis data.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import tempfile

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas


class PDFReportGenerator:
    """Generate comprehensive PDF reports with charts and analysis."""
    
    def __init__(self, report_title: str = "Portfolio Analysis Report", author: str = "Dashboard User"):
        """
        Initialize PDF report generator.
        
        Args:
            report_title: Title of the report
            author: Report author name
        """
        self.report_title = report_title
        self.author = author
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.HexColor('#1f77b4'),
            borderPadding=5,
            backColor=colors.HexColor('#ecf0f1')
        ))
        
        # Subsection header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica-Bold'
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=6
        ))
        
        # Info box
        self.styles.add(ParagraphStyle(
            name='InfoBox',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#31708f'),
            backColor=colors.HexColor('#d9edf7'),
            borderWidth=1,
            borderColor=colors.HexColor('#bce8f1'),
            borderPadding=10,
            spaceAfter=12
        ))
        
        # Warning box
        self.styles.add(ParagraphStyle(
            name='WarningBox',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#8a6d3b'),
            backColor=colors.HexColor('#fcf8e3'),
            borderWidth=1,
            borderColor=colors.HexColor('#faebcc'),
            borderPadding=10,
            spaceAfter=12
        ))
        
        # Success box
        self.styles.add(ParagraphStyle(
            name='SuccessBox',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#3c763d'),
            backColor=colors.HexColor('#dff0d8'),
            borderWidth=1,
            borderColor=colors.HexColor('#d6e9c6'),
            borderPadding=10,
            spaceAfter=12
        ))
    
    def _create_header_footer(self, canvas_obj, doc):
        """Add header and footer to each page."""
        canvas_obj.saveState()
        
        # Header
        canvas_obj.setFont('Helvetica-Bold', 10)
        canvas_obj.setFillColor(colors.HexColor('#1f77b4'))
        canvas_obj.drawString(inch, doc.height + doc.topMargin + 0.3*inch, self.report_title)
        
        # Footer
        canvas_obj.setFont('Helvetica', 8)
        canvas_obj.setFillColor(colors.gray)
        canvas_obj.drawRightString(
            doc.width + inch,
            0.5*inch,
            f"Page {canvas_obj.getPageNumber()}"
        )
        canvas_obj.drawString(
            inch,
            0.5*inch,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        canvas_obj.restoreState()
    
    def _create_table(self, data: List[List[str]], col_widths: Optional[List[float]] = None,
                      header_row: bool = True) -> Table:
        """
        Create a styled table.
        
        Args:
            data: Table data (list of rows)
            col_widths: Column widths in inches
            header_row: Whether first row is header
            
        Returns:
            Styled Table object
        """
        table = Table(data, colWidths=col_widths)
        
        # Style
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db') if header_row else colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke if header_row else colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f2f2')])
        ])
        
        table.setStyle(style)
        return table
    
    def generate_report(
        self,
        output_path: str,
        capm_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        crisis_data: Dict[str, Any],
        chart_paths: Dict[str, str],
        data_info: Dict[str, Any]
    ):
        """
        Generate comprehensive PDF report.
        
        Args:
            output_path: Path to save PDF
            capm_data: CAPM analysis data
            portfolio_data: Portfolio optimization data
            crisis_data: Crisis simulation data
            chart_paths: Dictionary of chart names to file paths
            data_info: General data information
        """
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=inch,
            bottomMargin=0.75*inch
        )
        
        # Story (content)
        story = []
        
        # Title page
        story.extend(self._create_title_page(data_info))
        story.append(PageBreak())
        
        # Table of contents
        story.extend(self._create_toc())
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(capm_data, portfolio_data, crisis_data, data_info))
        story.append(PageBreak())
        
        # Data exploration
        if 'data_exploration' in chart_paths:
            story.extend(self._create_data_exploration_section(chart_paths, data_info))
            story.append(PageBreak())
        
        # CAPM analysis
        story.extend(self._create_capm_section(capm_data, chart_paths))
        story.append(PageBreak())
        
        # Portfolio optimization
        story.extend(self._create_portfolio_section(portfolio_data, chart_paths))
        story.append(PageBreak())
        
        # Crisis analysis
        story.extend(self._create_crisis_section(crisis_data))
        story.append(PageBreak())
        
        # Disclaimer
        story.extend(self._create_disclaimer())
        
        # Build PDF
        doc.build(story, onFirstPage=self._create_header_footer, onLaterPages=self._create_header_footer)
    
    def _create_title_page(self, data_info: Dict[str, Any]) -> List:
        """Create title page."""
        elements = []
        
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(self.report_title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 0.3*inch))
        
        elements.append(Paragraph(
            f"<b>Author:</b> {self.author}",
            self.styles['Metric']
        ))
        elements.append(Paragraph(
            f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
            self.styles['Metric']
        ))
        elements.append(Spacer(1, 0.5*inch))
        
        # Summary box
        summary_text = f"""
        <b>Analysis Summary:</b><br/>
        • Total Stocks Analyzed: {data_info.get('n_stocks', 'N/A')}<br/>
        • Data Points per Stock: {data_info.get('n_observations', 'N/A')}<br/>
        • Risk-Free Rate: {data_info.get('risk_free_rate', 0)*100:.2f}%<br/>
        • Analysis Date: {datetime.now().strftime('%Y-%m-%d')}<br/>
        """
        elements.append(Paragraph(summary_text, self.styles['InfoBox']))
        
        elements.append(Spacer(1, inch))
        elements.append(Paragraph(
            "<b>⚠️ IMPORTANT DISCLAIMER</b><br/>"
            "This report is for educational and informational purposes only. "
            "It does not constitute financial advice, investment recommendations, "
            "or a solicitation to buy or sell securities.",
            self.styles['WarningBox']
        ))
        
        return elements
    
    def _create_toc(self) -> List:
        """Create table of contents."""
        elements = []
        
        elements.append(Paragraph("Table of Contents", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        toc_items = [
            "1. Executive Summary",
            "2. Data Exploration",
            "3. CAPM Analysis",
            "4. Portfolio Optimization",
            "5. Crisis Scenario Analysis",
            "6. Disclaimer"
        ]
        
        for item in toc_items:
            elements.append(Paragraph(f"• {item}", self.styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
        
        return elements
    
    def _create_executive_summary(
        self,
        capm_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        crisis_data: Dict[str, Any],
        data_info: Dict[str, Any]
    ) -> List:
        """Create executive summary section."""
        elements = []
        
        elements.append(Paragraph("1. Executive Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Key highlights
        elements.append(Paragraph("Key Highlights", self.styles['SubsectionHeader']))
        
        highlights = f"""
        • <b>Total Stocks Analyzed:</b> {capm_data.get('total_stocks', 'N/A')}<br/>
        • <b>Undervalued Stocks (BUY):</b> {capm_data.get('n_undervalued', 'N/A')}<br/>
        • <b>Fairly Valued (HOLD):</b> {capm_data.get('n_hold', 'N/A')}<br/>
        • <b>Overvalued Stocks (SELL):</b> {capm_data.get('n_overvalued', 'N/A')}<br/>
        """
        elements.append(Paragraph(highlights, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Portfolio performance
        elements.append(Paragraph("Portfolio Performance", self.styles['SubsectionHeader']))
        
        performance = f"""
        • <b>Expected Annual Return:</b> {portfolio_data.get('expected_return', 0)*100:.2f}%<br/>
        • <b>Annual Volatility:</b> {portfolio_data.get('volatility', 0)*100:.2f}%<br/>
        • <b>Sharpe Ratio:</b> {portfolio_data.get('sharpe_ratio', 0):.2f}<br/>
        • <b>Number of Holdings:</b> {portfolio_data.get('n_holdings', 'N/A')}<br/>
        """
        elements.append(Paragraph(performance, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Crisis resilience
        elements.append(Paragraph("Crisis Resilience", self.styles['SubsectionHeader']))
        
        crisis_text = f"""
        • <b>Scenario Tested:</b> {crisis_data.get('scenario_name', 'N/A')}<br/>
        • <b>Investment Recommendation:</b> {crisis_data.get('recommendation', 'N/A')}<br/>
        • <b>Rationale:</b> {crisis_data.get('rationale', 'N/A')}<br/>
        """
        elements.append(Paragraph(crisis_text, self.styles['Normal']))
        
        # Overall assessment
        if crisis_data.get('recommendation') == 'Worthy':
            assessment = "The portfolio demonstrates strong fundamentals and reasonable resilience under stress scenarios. Investment appears favorable based on current analysis."
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(assessment, self.styles['SuccessBox']))
        elif crisis_data.get('recommendation') == 'Proceed with Caution':
            assessment = "The portfolio shows mixed signals. While fundamentals are sound, stress scenarios indicate elevated risk. Careful consideration recommended."
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(assessment, self.styles['WarningBox']))
        
        return elements
    
    def _create_data_exploration_section(self, chart_paths: Dict[str, str], data_info: Dict[str, Any]) -> List:
        """Create data exploration section with charts."""
        elements = []
        
        elements.append(Paragraph("2. Data Exploration", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        elements.append(Paragraph(
            "This section provides an overview of the uploaded data, including price trends, "
            "returns distribution, correlation patterns, and volatility analysis.",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 0.2*inch))
        
        # Add charts if available
        chart_names = [
            ('price_trends', 'Stock Price Trends'),
            ('returns_dist', 'Returns Distribution'),
            ('correlation_heatmap', 'Correlation Heatmap'),
            ('volatility', 'Volatility Analysis')
        ]
        
        for chart_key, chart_title in chart_names:
            if chart_key in chart_paths and Path(chart_paths[chart_key]).exists():
                elements.append(Paragraph(chart_title, self.styles['SubsectionHeader']))
                try:
                    img = Image(chart_paths[chart_key], width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    elements.append(Paragraph(f"[Chart not available: {str(e)}]", self.styles['Normal']))
        
        return elements
    
    def _create_capm_section(self, capm_data: Dict[str, Any], chart_paths: Dict[str, str]) -> List:
        """Create CAPM analysis section."""
        elements = []
        
        elements.append(Paragraph("3. CAPM Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Overview
        overview_text = """
        The Capital Asset Pricing Model (CAPM) evaluates individual stock performance 
        against market expectations. Stocks with positive alpha are undervalued (BUY), 
        negative alpha indicates overvaluation (SELL).
        """
        elements.append(Paragraph(overview_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Methodology
        elements.append(Paragraph("Methodology", self.styles['SubsectionHeader']))
        
        methodology = f"""
        <b>CAPM Formula:</b> E(Ri) = Rf + βi × (E(Rm) - Rf)<br/>
        • <b>Risk-Free Rate (Rf):</b> {capm_data.get('risk_free_rate', 0)*100:.2f}%<br/>
        • <b>Market Return (E(Rm)):</b> {capm_data.get('market_return', 0)*100:.2f}%<br/>
        • <b>Market Risk Premium:</b> {capm_data.get('market_risk_premium', 0)*100:.2f}%<br/>
        """
        elements.append(Paragraph(methodology, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Summary statistics
        elements.append(Paragraph("Summary Statistics", self.styles['SubsectionHeader']))
        
        summary = f"""
        • <b>Total Stocks Analyzed:</b> {capm_data.get('total_stocks', 'N/A')}<br/>
        • <b>Mean Beta:</b> {capm_data.get('mean_beta', 0):.3f}<br/>
        • <b>Mean Alpha:</b> {capm_data.get('mean_alpha', 0)*100:.2f}%<br/>
        • <b>Mean R²:</b> {capm_data.get('mean_r_squared', 0):.3f}<br/>
        """
        elements.append(Paragraph(summary, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Stock classifications
        elements.append(Paragraph("Stock Classifications", self.styles['SubsectionHeader']))
        
        class_data = [
            ['Classification', 'Count', 'Percentage'],
            ['Undervalued (BUY)', str(capm_data.get('n_undervalued', 0)), 
             f"{capm_data.get('n_undervalued', 0)/capm_data.get('total_stocks', 1)*100:.1f}%"],
            ['Fairly Valued (HOLD)', str(capm_data.get('n_hold', 0)),
             f"{capm_data.get('n_hold', 0)/capm_data.get('total_stocks', 1)*100:.1f}%"],
            ['Overvalued (SELL)', str(capm_data.get('n_overvalued', 0)),
             f"{capm_data.get('n_overvalued', 0)/capm_data.get('total_stocks', 1)*100:.1f}%"]
        ]
        
        table = self._create_table(class_data, col_widths=[3*inch, 1.5*inch, 1.5*inch])
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Top undervalued stocks
        if capm_data.get('undervalued_stocks'):
            elements.append(Paragraph("Top 10 Undervalued Stocks (BUY)", self.styles['SubsectionHeader']))
            
            stock_data = [['Ticker', 'Beta', 'Exp. Return', 'Alpha', 'R²']]
            for stock in capm_data['undervalued_stocks'][:10]:
                stock_data.append([
                    stock['ticker'],
                    f"{stock['beta']:.2f}",
                    f"{stock['expected_return']*100:.2f}%",
                    f"{stock['alpha']*100:.2f}%",
                    f"{stock['r_squared']:.3f}"
                ])
            
            table = self._create_table(stock_data, col_widths=[1.2*inch, 1*inch, 1.3*inch, 1*inch, 1*inch])
            elements.append(table)
            elements.append(Spacer(1, 0.3*inch))
        
        # SML Chart
        if 'sml_chart' in chart_paths and Path(chart_paths['sml_chart']).exists():
            elements.append(PageBreak())
            elements.append(Paragraph("Security Market Line (SML)", self.styles['SubsectionHeader']))
            elements.append(Paragraph(
                "The SML chart visualizes stock positions relative to expected returns. "
                "Points above the line (green) are undervalued, below (red) are overvalued.",
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 0.2*inch))
            try:
                img = Image(chart_paths['sml_chart'], width=6.5*inch, height=4.5*inch)
                elements.append(img)
            except Exception as e:
                elements.append(Paragraph(f"[SML chart not available: {str(e)}]", self.styles['Normal']))
        
        return elements
    
    def _create_portfolio_section(self, portfolio_data: Dict[str, Any], chart_paths: Dict[str, str]) -> List:
        """Create portfolio optimization section."""
        elements = []
        
        elements.append(Paragraph("4. Portfolio Optimization", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Overview
        overview = """
        The Black-Litterman model combines market equilibrium with investor views to 
        construct an optimal portfolio. This approach balances diversification with 
        active insights about expected returns.
        """
        elements.append(Paragraph(overview, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Portfolio metrics
        elements.append(Paragraph("Portfolio Metrics", self.styles['SubsectionHeader']))
        
        metrics = f"""
        • <b>Expected Annual Return:</b> {portfolio_data.get('expected_return', 0)*100:.2f}%<br/>
        • <b>Annual Volatility:</b> {portfolio_data.get('volatility', 0)*100:.2f}%<br/>
        • <b>Sharpe Ratio:</b> {portfolio_data.get('sharpe_ratio', 0):.2f}<br/>
        • <b>Number of Holdings:</b> {portfolio_data.get('n_holdings', 'N/A')}<br/>
        """
        elements.append(Paragraph(metrics, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Portfolio allocation table
        if portfolio_data.get('holdings'):
            elements.append(Paragraph("Portfolio Allocation", self.styles['SubsectionHeader']))
            
            holdings_data = [['Ticker', 'Weight', 'Expected Return']]
            for holding in portfolio_data['holdings'][:15]:  # Top 15
                holdings_data.append([
                    holding['ticker'],
                    f"{holding['weight']*100:.2f}%",
                    f"{holding.get('expected_return', 0)*100:.2f}%"
                ])
            
            table = self._create_table(holdings_data, col_widths=[2*inch, 1.5*inch, 2*inch])
            elements.append(table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Portfolio charts
        if 'portfolio_pie' in chart_paths and Path(chart_paths['portfolio_pie']).exists():
            elements.append(PageBreak())
            elements.append(Paragraph("Portfolio Weight Distribution", self.styles['SubsectionHeader']))
            elements.append(Spacer(1, 0.2*inch))
            try:
                img = Image(chart_paths['portfolio_pie'], width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
            except Exception as e:
                elements.append(Paragraph(f"[Pie chart not available: {str(e)}]", self.styles['Normal']))
        
        if 'portfolio_bar' in chart_paths and Path(chart_paths['portfolio_bar']).exists():
            try:
                img = Image(chart_paths['portfolio_bar'], width=6*inch, height=4*inch)
                elements.append(img)
            except Exception as e:
                elements.append(Paragraph(f"[Bar chart not available: {str(e)}]", self.styles['Normal']))
        
        return elements
    
    def _create_crisis_section(self, crisis_data: Dict[str, Any]) -> List:
        """Create crisis scenario analysis section."""
        elements = []
        
        elements.append(Paragraph("5. Crisis Scenario Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Overview
        overview = """
        Monte Carlo simulation evaluates portfolio performance under geopolitical crisis 
        conditions. This stress test helps assess risk tolerance and downside potential.
        """
        elements.append(Paragraph(overview, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Scenario details
        elements.append(Paragraph(f"Scenario: {crisis_data.get('scenario_name', 'N/A')}", 
                                 self.styles['SubsectionHeader']))
        
        # Recommendation
        recommendation = crisis_data.get('recommendation', 'N/A')
        rationale = crisis_data.get('rationale', 'No rationale provided.')
        
        rec_text = f"""
        <b>Investment Recommendation:</b> {recommendation}<br/><br/>
        <b>Rationale:</b> {rationale}
        """
        
        if recommendation == 'Worthy':
            elements.append(Paragraph(rec_text, self.styles['SuccessBox']))
        elif recommendation == 'Proceed with Caution':
            elements.append(Paragraph(rec_text, self.styles['WarningBox']))
        else:
            elements.append(Paragraph(rec_text, self.styles['Normal']))
        
        return elements
    
    def _create_disclaimer(self) -> List:
        """Create disclaimer section."""
        elements = []
        
        elements.append(Paragraph("6. Disclaimer", self.styles['SectionHeader']))
        elements.append(Spacer(1, 0.2*inch))
        
        disclaimer_text = """
        <b>IMPORTANT LEGAL NOTICE</b><br/><br/>
        
        This report is provided for educational and informational purposes only. 
        It does not constitute financial advice, investment recommendations, or 
        a solicitation to buy or sell any securities.<br/><br/>
        
        <b>No Warranty:</b> The information is provided "as is" without any warranties, 
        express or implied. We make no guarantees about the accuracy, completeness, 
        or timeliness of the information.<br/><br/>
        
        <b>Investment Risk:</b> All investments involve risk, including potential loss 
        of principal. Past performance does not guarantee future results. Market conditions 
        can change rapidly and unpredictably.<br/><br/>
        
        <b>Professional Advice:</b> Before making any investment decisions, consult with 
        a qualified financial advisor who can assess your personal financial situation, 
        risk tolerance, and investment objectives.<br/><br/>
        
        <b>Model Limitations:</b> The models and analyses in this report rely on historical 
        data and mathematical assumptions. They may not accurately predict future performance 
        or account for all risk factors.<br/><br/>
        
        <b>No Liability:</b> We accept no liability for any losses or damages arising 
        from the use of this report or reliance on its contents.<br/><br/>
        
        By using this report, you acknowledge that you have read and understood this disclaimer.
        """
        
        elements.append(Paragraph(disclaimer_text, self.styles['Normal']))
        
        return elements
