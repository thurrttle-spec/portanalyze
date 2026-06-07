"""
Report templates for different analysis sections.

Provides formatting functions for CAPM, Black-Litterman,
crisis analysis, and executive summaries.
"""

from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


def generate_executive_summary(
    capm_summary: Dict,
    portfolio_summary: Dict,
    crisis_summary: Optional[Dict] = None
) -> str:
    """Generate executive summary section.
    
    Args:
        capm_summary: Summary of CAPM analysis results
        portfolio_summary: Summary of portfolio optimization
        crisis_summary: Summary of crisis analysis (optional)
        
    Returns:
        Formatted executive summary as string
    """
    summary = []
    summary.append("# Executive Summary\n")
    summary.append(f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    summary.append("---\n\n")
    
    # Key Highlights
    summary.append("## Key Highlights\n\n")
    
    # CAPM highlights
    if capm_summary:
        summary.append("### Stock Analysis (CAPM)\n")
        summary.append(f"- **Total stocks analyzed**: {capm_summary.get('total_stocks', 0)}\n")
        summary.append(f"- **Undervalued (BUY)**: {capm_summary.get('n_undervalued', 0)} stocks\n")
        summary.append(f"- **Fairly valued (HOLD)**: {capm_summary.get('n_hold', 0)} stocks\n")
        summary.append(f"- **Overvalued (SELL)**: {capm_summary.get('n_overvalued', 0)} stocks\n")
        
        if 'top_pick' in capm_summary:
            summary.append(f"- **Top pick**: {capm_summary['top_pick']['ticker']} "
                         f"(alpha: {capm_summary['top_pick']['alpha']*100:.2f}%)\n")
        summary.append("\n")
    
    # Portfolio highlights
    if portfolio_summary:
        summary.append("### Portfolio Optimization\n")
        summary.append(f"- **Expected return**: {portfolio_summary.get('expected_return', 0)*100:.2f}% annually\n")
        summary.append(f"- **Volatility**: {portfolio_summary.get('volatility', 0)*100:.2f}% annually\n")
        summary.append(f"- **Sharpe ratio**: {portfolio_summary.get('sharpe_ratio', 0):.2f}\n")
        summary.append(f"- **Number of holdings**: {portfolio_summary.get('n_holdings', 0)}\n")
        
        if 'top_holdings' in portfolio_summary:
            summary.append(f"- **Top holdings**:\n")
            for holding in portfolio_summary['top_holdings'][:3]:
                summary.append(f"  - {holding['ticker']}: {holding['weight']*100:.2f}%\n")
        summary.append("\n")
    
    # Crisis highlights
    if crisis_summary:
        summary.append("### Crisis Resilience\n")
        summary.append(f"- **Scenario**: {crisis_summary.get('scenario_name', 'N/A')}\n")
        summary.append(f"- **Expected return (crisis)**: {crisis_summary.get('mean_return', 0)*100:.2f}%\n")
        summary.append(f"- **Value at Risk (95%)**: {crisis_summary.get('var_95', 0)*100:.2f}%\n")
        summary.append(f"- **Max drawdown**: {crisis_summary.get('worst_drawdown', 0)*100:.2f}%\n")
        summary.append(f"- **Investment recommendation**: **{crisis_summary.get('recommendation', 'N/A')}**\n")
        summary.append("\n")
    
    # Overall recommendation
    summary.append("## Overall Assessment\n\n")
    
    if portfolio_summary and portfolio_summary.get('sharpe_ratio', 0) > 2.0:
        summary.append("✅ **Excellent risk-adjusted returns**: ")
        summary.append(f"Sharpe ratio of {portfolio_summary.get('sharpe_ratio', 0):.2f} ")
        summary.append("indicates exceptional performance.\n\n")
    
    if crisis_summary and crisis_summary.get('recommendation') == 'Worthy':
        summary.append("✅ **Crisis-resilient portfolio**: ")
        summary.append("Portfolio demonstrates strong resilience under stress scenarios.\n\n")
    
    return ''.join(summary)


def generate_capm_section(capm_results: Dict) -> str:
    """Generate CAPM analysis section.
    
    Args:
        capm_results: Dictionary with CAPM analysis results
        
    Returns:
        Formatted CAPM section as string
    """
    section = []
    section.append("# CAPM Analysis\n\n")
    section.append("## Overview\n\n")
    section.append("Capital Asset Pricing Model (CAPM) analysis identifies stocks that are ")
    section.append("undervalued or overvalued relative to their systematic risk (beta).\n\n")
    
    # Methodology
    section.append("### Methodology\n\n")
    section.append("- **CAPM Formula**: E(Ri) = Rf + βi × (E(Rm) - Rf)\n")
    section.append(f"- **Risk-free rate**: {capm_results.get('risk_free_rate', 0)*100:.2f}%\n")
    section.append(f"- **Market return**: {capm_results.get('market_return', 0)*100:.2f}%\n")
    section.append(f"- **Market risk premium**: {capm_results.get('market_risk_premium', 0)*100:.2f}%\n\n")
    
    # Summary statistics
    section.append("## Summary Statistics\n\n")
    section.append(f"- **Total stocks analyzed**: {capm_results.get('total_stocks', 0)}\n")
    section.append(f"- **Mean beta**: {capm_results.get('mean_beta', 0):.3f}\n")
    section.append(f"- **Mean alpha**: {capm_results.get('mean_alpha', 0)*100:.2f}%\n")
    section.append(f"- **Mean R-squared**: {capm_results.get('mean_r_squared', 0):.3f}\n\n")
    
    # Classifications
    section.append("## Stock Classifications\n\n")
    section.append(f"- **Undervalued (BUY)**: {capm_results.get('n_undervalued', 0)} stocks ")
    section.append("(alpha > 2%)\n")
    section.append(f"- **Fairly valued (HOLD)**: {capm_results.get('n_hold', 0)} stocks ")
    section.append("(-2% ≤ alpha ≤ 2%)\n")
    section.append(f"- **Overvalued (SELL)**: {capm_results.get('n_overvalued', 0)} stocks ")
    section.append("(alpha < -2%)\n\n")
    
    # Top undervalued stocks
    if 'undervalued_stocks' in capm_results and capm_results['undervalued_stocks']:
        section.append("### Top 10 Undervalued Stocks (BUY Recommendations)\n\n")
        section.append("| Ticker | Beta | Expected Return | Actual Return | Alpha | R² |\n")
        section.append("|--------|------|-----------------|---------------|-------|----|\n")
        
        for stock in capm_results['undervalued_stocks'][:10]:
            section.append(f"| {stock['ticker']} | ")
            section.append(f"{stock['beta']:.3f} | ")
            section.append(f"{stock['expected_return']*100:.2f}% | ")
            section.append(f"{stock['actual_return']*100:.2f}% | ")
            section.append(f"{stock['alpha']*100:.2f}% | ")
            section.append(f"{stock['r_squared']:.3f} |\n")
        section.append("\n")
    
    # Top overvalued stocks
    if 'overvalued_stocks' in capm_results and capm_results['overvalued_stocks']:
        section.append("### Top 10 Overvalued Stocks (SELL Recommendations)\n\n")
        section.append("| Ticker | Beta | Expected Return | Actual Return | Alpha | R² |\n")
        section.append("|--------|------|-----------------|---------------|-------|----|")
        section.append("\n")
        
        for stock in capm_results['overvalued_stocks'][:10]:
            section.append(f"| {stock['ticker']} | ")
            section.append(f"{stock['beta']:.3f} | ")
            section.append(f"{stock['expected_return']*100:.2f}% | ")
            section.append(f"{stock['actual_return']*100:.2f}% | ")
            section.append(f"{stock['alpha']*100:.2f}% | ")
            section.append(f"{stock['r_squared']:.3f} |\n")
        section.append("\n")
    
    # Interpretation
    section.append("## Interpretation\n\n")
    section.append("- **Positive alpha**: Stock is undervalued (actual return exceeds expected return)\n")
    section.append("- **Negative alpha**: Stock is overvalued (actual return below expected return)\n")
    section.append("- **Beta > 1**: Stock is more volatile than the market\n")
    section.append("- **Beta < 1**: Stock is less volatile than the market\n")
    section.append("- **Higher R²**: Beta explains more of the stock's variance\n\n")
    
    return ''.join(section)


def generate_portfolio_section(portfolio_results: Dict) -> str:
    """Generate Black-Litterman portfolio optimization section.
    
    Args:
        portfolio_results: Dictionary with portfolio optimization results
        
    Returns:
        Formatted portfolio section as string
    """
    section = []
    section.append("# Portfolio Optimization\n\n")
    section.append("## Overview\n\n")
    section.append("Black-Litterman model combines market equilibrium with investor views ")
    section.append("to construct an optimal portfolio that maximizes risk-adjusted returns.\n\n")
    
    # Methodology
    section.append("### Methodology\n\n")
    section.append("- **Model**: Black-Litterman with investor views\n")
    section.append(f"- **Risk aversion (λ)**: {portfolio_results.get('risk_aversion', 2.5):.2f}\n")
    section.append(f"- **Prior uncertainty (τ)**: {portfolio_results.get('tau', 0.025):.3f}\n")
    section.append(f"- **Number of views**: {portfolio_results.get('n_views', 0)}\n")
    section.append("- **Optimization objective**: Maximize Sharpe ratio\n")
    section.append("- **Constraints**: Long-only, weights sum to 1\n\n")
    
    # Portfolio metrics
    section.append("## Portfolio Metrics\n\n")
    section.append(f"- **Expected return**: {portfolio_results.get('expected_return', 0)*100:.2f}% annually\n")
    section.append(f"- **Volatility**: {portfolio_results.get('volatility', 0)*100:.2f}% annually\n")
    section.append(f"- **Sharpe ratio**: {portfolio_results.get('sharpe_ratio', 0):.2f}\n")
    section.append(f"- **Number of holdings**: {portfolio_results.get('n_holdings', 0)}\n\n")
    
    # Comparison with equilibrium
    if 'equilibrium_sharpe' in portfolio_results:
        eq_sharpe = portfolio_results['equilibrium_sharpe']
        opt_sharpe = portfolio_results['sharpe_ratio']
        improvement = ((opt_sharpe / eq_sharpe) - 1) * 100 if eq_sharpe > 0 else 0
        
        section.append("### Performance vs Equilibrium\n\n")
        section.append(f"- **Equilibrium Sharpe ratio**: {eq_sharpe:.2f}\n")
        section.append(f"- **Optimized Sharpe ratio**: {opt_sharpe:.2f}\n")
        section.append(f"- **Improvement**: {improvement:.1f}%\n\n")
    
    # Portfolio allocation
    if 'holdings' in portfolio_results:
        section.append("## Portfolio Allocation\n\n")
        section.append("| Ticker | Weight | Expected Return | Contribution to Risk |\n")
        section.append("|--------|--------|-----------------|----------------------|\n")
        
        for holding in portfolio_results['holdings']:
            section.append(f"| {holding['ticker']} | ")
            section.append(f"{holding['weight']*100:.2f}% | ")
            section.append(f"{holding.get('expected_return', 0)*100:.2f}% | ")
            section.append(f"{holding.get('risk_contribution', 0)*100:.2f}% |\n")
        section.append("\n")
        
        # Top holdings summary
        top_5_weight = sum(h['weight'] for h in portfolio_results['holdings'][:5])
        section.append(f"**Top 5 holdings**: {top_5_weight*100:.1f}% of portfolio\n\n")
    
    # Investor views
    if 'views' in portfolio_results and portfolio_results['views']:
        section.append("## Investor Views\n\n")
        section.append("The following subjective views were incorporated:\n\n")
        
        for i, view in enumerate(portfolio_results['views'], 1):
            section.append(f"{i}. **{view['description']}**\n")
            section.append(f"   - Expected return: {view['return']*100:.2f}%\n")
            section.append(f"   - Confidence: {view['confidence']:.1%}\n")
        section.append("\n")
    
    # Interpretation
    section.append("## Interpretation\n\n")
    section.append("- **Higher Sharpe ratio**: Better risk-adjusted returns\n")
    section.append("- **Diversification**: Multiple holdings reduce idiosyncratic risk\n")
    section.append("- **View integration**: Combines market equilibrium with expert insights\n\n")
    
    return ''.join(section)


def generate_crisis_section(crisis_results: Dict) -> str:
    """Generate crisis scenario analysis section.
    
    Args:
        crisis_results: Dictionary with crisis simulation results
        
    Returns:
        Formatted crisis section as string
    """
    section = []
    section.append("# Crisis Scenario Analysis\n\n")
    section.append("## Overview\n\n")
    section.append("Monte Carlo simulation assesses portfolio performance under ")
    section.append("geopolitical crisis conditions with increased volatility and negative return shocks.\n\n")
    
    # Scenario details
    section.append("### Scenario Details\n\n")
    section.append(f"- **Scenario**: {crisis_results.get('scenario_name', 'N/A')}\n")
    section.append(f"- **Duration**: {crisis_results.get('duration_days', 0)} days\n")
    section.append(f"- **Simulations**: {crisis_results.get('n_simulations', 0):,} Monte Carlo runs\n")
    section.append(f"- **Volatility multiplier**: {crisis_results.get('volatility_multiplier', 0):.1f}x\n")
    section.append(f"- **Return impact**: {crisis_results.get('mean_shift', 0)*100:.1f}%\n\n")
    
    # Risk metrics
    section.append("## Risk Metrics\n\n")
    section.append("### Return Statistics\n\n")
    section.append(f"- **Mean return**: {crisis_results.get('mean_return', 0)*100:.2f}%\n")
    section.append(f"- **Median return**: {crisis_results.get('median_return', 0)*100:.2f}%\n")
    section.append(f"- **Standard deviation**: {crisis_results.get('std_return', 0)*100:.2f}%\n")
    section.append(f"- **Min return**: {crisis_results.get('min_return', 0)*100:.2f}%\n")
    section.append(f"- **Max return**: {crisis_results.get('max_return', 0)*100:.2f}%\n\n")
    
    section.append("### Value at Risk (VaR)\n\n")
    section.append(f"- **VaR 95%**: {crisis_results.get('var_95', 0)*100:.2f}% ")
    section.append("(5% chance of losing more than this)\n")
    section.append(f"- **VaR 99%**: {crisis_results.get('var_99', 0)*100:.2f}% ")
    section.append("(1% chance of losing more than this)\n")
    section.append(f"- **Dollar VaR 95%**: ${abs(crisis_results.get('dollar_var_95', 0)):,.0f} ")
    section.append("(on $1M portfolio)\n\n")
    
    section.append("### Conditional VaR (CVaR / Expected Shortfall)\n\n")
    section.append(f"- **CVaR 95%**: {crisis_results.get('cvar_95', 0)*100:.2f}% ")
    section.append("(average loss in worst 5% of cases)\n")
    section.append(f"- **CVaR 99%**: {crisis_results.get('cvar_99', 0)*100:.2f}% ")
    section.append("(average loss in worst 1% of cases)\n\n")
    
    section.append("### Drawdown Analysis\n\n")
    section.append(f"- **Mean max drawdown**: {crisis_results.get('mean_max_drawdown', 0)*100:.2f}%\n")
    section.append(f"- **Median max drawdown**: {crisis_results.get('median_max_drawdown', 0)*100:.2f}%\n")
    section.append(f"- **Worst drawdown**: {crisis_results.get('worst_drawdown', 0)*100:.2f}%\n")
    section.append(f"- **Average recovery time**: {crisis_results.get('avg_recovery_days', 0):.0f} days\n\n")
    
    # Probability outcomes
    section.append("## Outcome Probabilities\n\n")
    section.append(f"- **Probability of profit**: {crisis_results.get('prob_profit', 0)*100:.1f}%\n")
    section.append(f"- **Probability of loss**: {crisis_results.get('prob_loss', 0)*100:.1f}%\n")
    section.append(f"- **Probability of severe loss** (>20%): ")
    section.append(f"{crisis_results.get('prob_severe_loss', 0)*100:.1f}%\n\n")
    
    # Investment recommendation
    section.append("## Investment Recommendation\n\n")
    recommendation = crisis_results.get('recommendation', 'N/A')
    rationale = crisis_results.get('rationale', 'No rationale provided.')
    
    if recommendation == 'Worthy':
        emoji = "✅"
    elif recommendation == 'Proceed with Caution':
        emoji = "⚠️"
    else:
        emoji = "❌"
    
    section.append(f"### {emoji} {recommendation}\n\n")
    section.append(f"{rationale}\n\n")
    
    # Interpretation
    section.append("## Interpretation\n\n")
    section.append("- **VaR**: Maximum loss at given confidence level\n")
    section.append("- **CVaR**: Average loss when VaR threshold is exceeded (tail risk)\n")
    section.append("- **Drawdown**: Peak-to-trough decline from highest portfolio value\n")
    section.append("- **Recovery time**: Days needed to return to initial value\n\n")
    
    return ''.join(section)
