"""
Report generator for comprehensive financial analysis reports.

Generates reports in multiple formats (Markdown, HTML, text) combining
CAPM analysis, portfolio optimization, and crisis scenarios.
"""

from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import base64

from .templates import (
    generate_executive_summary,
    generate_capm_section,
    generate_portfolio_section,
    generate_crisis_section
)


class ReportGenerator:
    """Generate comprehensive financial analysis reports.
    
    Combines results from CAPM, Black-Litterman, and crisis analysis
    into professional reports in multiple formats.
    
    Attributes:
        report_title: Title of the report
        author: Report author name
        include_images: Whether to embed images in HTML reports
    """
    
    def __init__(
        self,
        report_title: str = "Financial Analysis Report",
        author: str = "Portfolio Analysis System",
        include_images: bool = True
    ):
        """Initialize report generator.
        
        Args:
            report_title: Title for the report
            author: Author name for the report
            include_images: Whether to include images in HTML output
        """
        self.report_title = report_title
        self.author = author
        self.include_images = include_images
    
    def generate_markdown_report(
        self,
        capm_results: Optional[Dict] = None,
        portfolio_results: Optional[Dict] = None,
        crisis_results: Optional[Dict] = None,
        image_paths: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate comprehensive report in Markdown format.
        
        Args:
            capm_results: CAPM analysis results dictionary
            portfolio_results: Portfolio optimization results dictionary
            crisis_results: Crisis simulation results dictionary
            image_paths: Dictionary mapping section names to image file paths
            
        Returns:
            Complete Markdown report as string
        """
        report = []
        
        # Header
        report.append(f"# {self.report_title}\n\n")
        report.append(f"**Author**: {self.author}\n")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.append("---\n\n")
        
        # Table of contents
        report.append("## Table of Contents\n\n")
        report.append("1. [Executive Summary](#executive-summary)\n")
        if capm_results:
            report.append("2. [CAPM Analysis](#capm-analysis)\n")
        if portfolio_results:
            report.append("3. [Portfolio Optimization](#portfolio-optimization)\n")
        if crisis_results:
            report.append("4. [Crisis Scenario Analysis](#crisis-scenario-analysis)\n")
        report.append("5. [Disclaimer](#disclaimer)\n\n")
        report.append("---\n\n")
        
        # Executive summary
        capm_summary = self._extract_capm_summary(capm_results) if capm_results else None
        portfolio_summary = self._extract_portfolio_summary(portfolio_results) if portfolio_results else None
        crisis_summary = self._extract_crisis_summary(crisis_results) if crisis_results else None
        
        report.append(generate_executive_summary(capm_summary, portfolio_summary, crisis_summary))
        report.append("\n---\n\n")
        
        # CAPM section
        if capm_results:
            report.append(generate_capm_section(capm_results))
            
            # Add CAPM visualizations
            if image_paths and 'capm' in image_paths:
                report.append(f"![CAPM Visualization]({image_paths['capm']})\n\n")
            
            report.append("---\n\n")
        
        # Portfolio section
        if portfolio_results:
            report.append(generate_portfolio_section(portfolio_results))
            
            # Add portfolio visualizations
            if image_paths and 'portfolio' in image_paths:
                report.append(f"![Portfolio Visualization]({image_paths['portfolio']})\n\n")
            
            report.append("---\n\n")
        
        # Crisis section
        if crisis_results:
            report.append(generate_crisis_section(crisis_results))
            
            # Add crisis visualizations
            if image_paths and 'crisis' in image_paths:
                report.append(f"![Crisis Visualization]({image_paths['crisis']})\n\n")
            
            report.append("---\n\n")
        
        # Disclaimer
        report.append(self._generate_disclaimer())
        
        return ''.join(report)
    
    def generate_html_report(
        self,
        capm_results: Optional[Dict] = None,
        portfolio_results: Optional[Dict] = None,
        crisis_results: Optional[Dict] = None,
        image_paths: Optional[Dict[str, str]] = None
    ) -> str:
        """Generate comprehensive report in HTML format.
        
        Args:
            capm_results: CAPM analysis results dictionary
            portfolio_results: Portfolio optimization results dictionary
            crisis_results: Crisis simulation results dictionary
            image_paths: Dictionary mapping section names to image file paths
            
        Returns:
            Complete HTML report as string
        """
        # Convert markdown to HTML sections
        markdown_content = self.generate_markdown_report(
            capm_results,
            portfolio_results,
            crisis_results,
            image_paths
        )
        
        # Create HTML with styling
        html = []
        html.append("<!DOCTYPE html>\n")
        html.append("<html lang='en'>\n")
        html.append("<head>\n")
        html.append("    <meta charset='UTF-8'>\n")
        html.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
        html.append(f"    <title>{self.report_title}</title>\n")
        html.append("    <style>\n")
        html.append(self._get_html_style())
        html.append("    </style>\n")
        html.append("</head>\n")
        html.append("<body>\n")
        html.append("    <div class='container'>\n")
        
        # Convert markdown to basic HTML
        html_content = self._markdown_to_html(markdown_content)
        html.append(html_content)
        
        html.append("    </div>\n")
        html.append("</body>\n")
        html.append("</html>\n")
        
        return ''.join(html)
    
    def save_report(
        self,
        output_path: str,
        capm_results: Optional[Dict] = None,
        portfolio_results: Optional[Dict] = None,
        crisis_results: Optional[Dict] = None,
        image_paths: Optional[Dict[str, str]] = None,
        format: str = 'markdown'
    ) -> Path:
        """Generate and save report to file.
        
        Args:
            output_path: Path where report should be saved
            capm_results: CAPM analysis results dictionary
            portfolio_results: Portfolio optimization results dictionary
            crisis_results: Crisis simulation results dictionary
            image_paths: Dictionary mapping section names to image file paths
            format: Output format ('markdown' or 'html')
            
        Returns:
            Path object pointing to saved report
        """
        output_path = Path(output_path)
        
        if format.lower() == 'markdown':
            content = self.generate_markdown_report(
                capm_results,
                portfolio_results,
                crisis_results,
                image_paths
            )
            if not output_path.suffix:
                output_path = output_path.with_suffix('.md')
        
        elif format.lower() == 'html':
            content = self.generate_html_report(
                capm_results,
                portfolio_results,
                crisis_results,
                image_paths
            )
            if not output_path.suffix:
                output_path = output_path.with_suffix('.html')
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'markdown' or 'html'.")
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        output_path.write_text(content, encoding='utf-8')
        
        return output_path
    
    def _extract_capm_summary(self, capm_results: Dict) -> Dict:
        """Extract summary metrics from CAPM results."""
        summary = {
            'total_stocks': capm_results.get('total_stocks', 0),
            'n_undervalued': capm_results.get('n_undervalued', 0),
            'n_hold': capm_results.get('n_hold', 0),
            'n_overvalued': capm_results.get('n_overvalued', 0)
        }
        
        if 'undervalued_stocks' in capm_results and capm_results['undervalued_stocks']:
            top_stock = capm_results['undervalued_stocks'][0]
            summary['top_pick'] = {
                'ticker': top_stock['ticker'],
                'alpha': top_stock['alpha']
            }
        
        return summary
    
    def _extract_portfolio_summary(self, portfolio_results: Dict) -> Dict:
        """Extract summary metrics from portfolio results."""
        summary = {
            'expected_return': portfolio_results.get('expected_return', 0),
            'volatility': portfolio_results.get('volatility', 0),
            'sharpe_ratio': portfolio_results.get('sharpe_ratio', 0),
            'n_holdings': portfolio_results.get('n_holdings', 0)
        }
        
        if 'holdings' in portfolio_results:
            summary['top_holdings'] = portfolio_results['holdings'][:3]
        
        return summary
    
    def _extract_crisis_summary(self, crisis_results: Dict) -> Dict:
        """Extract summary metrics from crisis results."""
        return {
            'scenario_name': crisis_results.get('scenario_name', 'N/A'),
            'mean_return': crisis_results.get('mean_return', 0),
            'var_95': crisis_results.get('var_95', 0),
            'worst_drawdown': crisis_results.get('worst_drawdown', 0),
            'recommendation': crisis_results.get('recommendation', 'N/A')
        }
    
    def _generate_disclaimer(self) -> str:
        """Generate disclaimer section."""
        disclaimer = []
        disclaimer.append("# Disclaimer\n\n")
        disclaimer.append("**Important Notice**\n\n")
        disclaimer.append("This report is generated by an automated financial analysis system ")
        disclaimer.append("and is provided for informational purposes only. It should not be ")
        disclaimer.append("considered as financial advice or a recommendation to buy, sell, ")
        disclaimer.append("or hold any securities.\n\n")
        
        disclaimer.append("**Key Points:**\n\n")
        disclaimer.append("- Past performance does not guarantee future results\n")
        disclaimer.append("- All investments carry risk, including potential loss of principal\n")
        disclaimer.append("- Models and simulations are based on historical data and assumptions\n")
        disclaimer.append("- Actual market conditions may differ significantly from projections\n")
        disclaimer.append("- Consult a qualified financial advisor before making investment decisions\n\n")
        
        disclaimer.append("**Model Limitations:**\n\n")
        disclaimer.append("- CAPM assumes markets are efficient and investors are rational\n")
        disclaimer.append("- Black-Litterman model depends on accuracy of investor views\n")
        disclaimer.append("- Monte Carlo simulations assume normal return distributions\n")
        disclaimer.append("- Crisis scenarios may not capture all possible market events\n\n")
        
        disclaimer.append(f"**Report generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return ''.join(disclaimer)
    
    def _get_html_style(self) -> str:
        """Get CSS styling for HTML reports."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }
        h3 {
            color: #7f8c8d;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        img {
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .disclaimer {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-top: 30px;
        }
        """
    
    def _markdown_to_html(self, markdown: str) -> str:
        """Convert basic Markdown to HTML.
        
        Simple converter for headings, lists, tables, and paragraphs.
        For full Markdown support, consider using a library like markdown or mistune.
        """
        lines = markdown.split('\n')
        html = []
        in_list = False
        in_table = False
        
        for line in lines:
            # Headings
            if line.startswith('# '):
                html.append(f"<h1>{line[2:]}</h1>\n")
            elif line.startswith('## '):
                html.append(f"<h2>{line[3:]}</h2>\n")
            elif line.startswith('### '):
                html.append(f"<h3>{line[4:]}</h3>\n")
            
            # Lists
            elif line.startswith('- '):
                if not in_list:
                    html.append("<ul>\n")
                    in_list = True
                html.append(f"<li>{line[2:]}</li>\n")
            else:
                if in_list:
                    html.append("</ul>\n")
                    in_list = False
                
                # Tables
                if '|' in line:
                    if not in_table:
                        html.append("<table>\n")
                        in_table = True
                    
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    
                    # Skip separator lines
                    if all(set(c) <= {'-', ' '} for c in cells):
                        continue
                    
                    # Determine if header row (simple heuristic)
                    if '**' in line or cells[0].isupper():
                        html.append("<tr>")
                        for cell in cells:
                            html.append(f"<th>{cell}</th>")
                        html.append("</tr>\n")
                    else:
                        html.append("<tr>")
                        for cell in cells:
                            html.append(f"<td>{cell}</td>")
                        html.append("</tr>\n")
                else:
                    if in_table:
                        html.append("</table>\n")
                        in_table = False
                    
                    # Horizontal rule
                    if line.strip() == '---':
                        html.append("<hr>\n")
                    # Paragraphs
                    elif line.strip():
                        html.append(f"<p>{line}</p>\n")
                    else:
                        html.append("<br>\n")
        
        # Close open tags
        if in_list:
            html.append("</ul>\n")
        if in_table:
            html.append("</table>\n")
        
        return ''.join(html)
