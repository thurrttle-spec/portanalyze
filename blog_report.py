"""
Comprehensive Blog-Style Report Generator

Creates detailed, narrative-driven financial analysis reports in HTML format.
"""

from typing import Dict, Any
from pathlib import Path
from datetime import datetime


def generate_comprehensive_report(
    output_path: str,
    results: Dict[str, Any],
    report_title: str = "Portfolio Analysis Report",
    author: str = "Financial Analyst"
) -> None:
    """Generate comprehensive blog-style HTML report.
    
    Args:
        output_path: Path to save the HTML report
        results: Complete analysis results from dashboard
        report_title: Title of the report
        author: Author name
    """
    
    # Extract data from results
    data_info = results.get('data_info', {})
    capm = results.get('capm', {})
    portfolio = results.get('portfolio', {})
    crisis = results.get('crisis', {})
    stock_data = results.get('stock_data', {})
    market_data = results.get('market_data')
    capm_results = capm.get('results')
    
    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
{_get_modern_css()}
    </style>
</head>
<body>
    <div class="container">
        <!-- Hero Section -->
        <div class="hero">
            <div class="hero-content">
                <h1 class="hero-title">{report_title}</h1>
                <p class="hero-subtitle">A Comprehensive Analysis of Energy Sector Investment Opportunities</p>
                <div class="hero-meta">
                    <span class="author">📝 By {author}</span>
                    <span class="date">📅 {datetime.now().strftime('%B %d, %Y')}</span>
                    <span class="time">🕒 {datetime.now().strftime('%I:%M %p')}</span>
                </div>
            </div>
        </div>

        <!-- Table of Contents -->
        <div class="toc-card">
            <h2 class="toc-title">📑 Table of Contents</h2>
            <ul class="toc-list">
                <li><a href="#executive-summary">Executive Summary</a></li>
                <li><a href="#market-overview">Market Overview & Data</a></li>
                <li><a href="#capm-analysis">Stock Valuation Analysis (CAPM)</a></li>
                <li><a href="#portfolio-optimization">Portfolio Construction</a></li>
                <li><a href="#crisis-resilience">Crisis Resilience Testing</a></li>
                <li><a href="#recommendations">Investment Recommendations</a></li>
                <li><a href="#methodology">Methodology</a></li>
                <li><a href="#disclaimer">Disclaimer</a></li>
            </ul>
        </div>

{_generate_executive_summary(data_info, capm, portfolio, crisis)}

{_generate_market_overview(data_info, market_data, stock_data)}

{_generate_capm_section(capm, capm_results, stock_data)}

{_generate_portfolio_section(portfolio, capm)}

{_generate_crisis_section(crisis, portfolio)}

{_generate_recommendations(capm, portfolio, crisis, data_info)}

{_generate_methodology()}

{_generate_disclaimer()}

    </div>
    
    <!-- Back to top button -->
    <a href="#" class="back-to-top">↑ Back to Top</a>
    
</body>
</html>
"""
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding='utf-8')


def _get_modern_css() -> str:
    """Get modern CSS styling for the report."""
    return """
        /* Global Styles */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.8;
            color: #1a1a1a;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            border-radius: 16px;
            overflow: hidden;
        }
        
        /* Hero Section */
        .hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 80px 60px;
            text-align: center;
        }
        
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .hero-subtitle {
            font-size: 1.3rem;
            margin-bottom: 30px;
            opacity: 0.95;
            font-weight: 300;
        }
        
        .hero-meta {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            font-size: 0.95rem;
            opacity: 0.9;
        }
        
        .hero-meta span {
            background: rgba(255,255,255,0.2);
            padding: 8px 16px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }
        
        /* Table of Contents */
        .toc-card {
            background: #f8f9fa;
            padding: 40px 60px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .toc-title {
            font-size: 1.8rem;
            margin-bottom: 20px;
            color: #495057;
        }
        
        .toc-list {
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .toc-list li a {
            display: block;
            padding: 15px 20px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
            font-weight: 500;
        }
        
        .toc-list li a:hover {
            background: #667eea;
            color: white;
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        /* Content Sections */
        .section {
            padding: 60px 60px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .section:last-child {
            border-bottom: none;
        }
        
        .section-title {
            font-size: 2.2rem;
            color: #212529;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
            display: inline-block;
        }
        
        .section-subtitle {
            font-size: 1.6rem;
            color: #495057;
            margin: 40px 0 20px 0;
            font-weight: 600;
        }
        
        .section-text {
            font-size: 1.1rem;
            line-height: 1.9;
            color: #495057;
            margin-bottom: 25px;
        }
        
        /* Cards and Highlights */
        .highlight-box {
            background: linear-gradient(135deg, #667eea15, #764ba215);
            border-left: 5px solid #667eea;
            padding: 25px 30px;
            margin: 30px 0;
            border-radius: 8px;
        }
        
        .highlight-box strong {
            color: #667eea;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            border: 1px solid #e9ecef;
            transition: all 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .stat-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #212529;
            margin-bottom: 5px;
        }
        
        .stat-value.positive {
            color: #28a745;
        }
        
        .stat-value.negative {
            color: #dc3545;
        }
        
        .stat-value.neutral {
            color: #667eea;
        }
        
        .stat-description {
            font-size: 0.9rem;
            color: #6c757d;
        }
