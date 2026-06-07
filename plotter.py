"""
Visualization tools for crisis scenario analysis.

Provides plotting functions for Monte Carlo simulation results,
return distributions, risk metrics, and crisis comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import seaborn as sns


# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_return_distribution(
    crisis_results: Dict,
    baseline_return: Optional[float] = None,
    baseline_volatility: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot distribution of returns from Monte Carlo simulation.
    
    Args:
        crisis_results: Results dictionary from CrisisSimulator.simulate_crisis()
        baseline_return: Optional baseline expected return for comparison
        baseline_volatility: Optional baseline volatility for comparison
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    final_returns = crisis_results['final_returns_distribution']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Histogram with KDE
    ax1.hist(final_returns * 100, bins=50, alpha=0.7, color='steelblue',
             edgecolor='black', density=True, label='Simulated Returns')
    
    # Add KDE
    from scipy import stats
    kde = stats.gaussian_kde(final_returns * 100)
    x_range = np.linspace(final_returns.min() * 100, final_returns.max() * 100, 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Mark key percentiles
    var_95 = crisis_results['var_95'] * 100
    mean_return = crisis_results['mean_return'] * 100
    median_return = crisis_results['median_return'] * 100
    
    ax1.axvline(var_95, color='red', linestyle='--', linewidth=2, 
                label=f'VaR 95%: {var_95:.1f}%')
    ax1.axvline(mean_return, color='green', linestyle='--', linewidth=2,
                label=f'Mean: {mean_return:.2f}%')
    ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Return Distribution - {crisis_results["scenario_name"]}\n'
                  f'({crisis_results["n_simulations"]:,} simulations, '
                  f'{crisis_results["duration_days"]} days)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Cumulative distribution (CDF)
    sorted_returns = np.sort(final_returns) * 100
    cumulative_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    ax2.plot(sorted_returns, cumulative_prob * 100, 'b-', linewidth=2)
    
    # Mark key points
    ax2.axvline(var_95, color='red', linestyle='--', linewidth=2,
                label=f'VaR 95%: {var_95:.1f}%')
    ax2.axhline(5, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Return (%)')
    ax2.set_ylabel('Cumulative Probability (%)')
    ax2.set_title('Cumulative Distribution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        f'Mean: {mean_return:.2f}%\n'
        f'Median: {median_return:.2f}%\n'
        f'Std Dev: {crisis_results["std_return"]*100:.2f}%\n'
        f'Min: {crisis_results["min_return"]*100:.1f}%\n'
        f'Max: {crisis_results["max_return"]*100:.1f}%\n'
        f'VaR 95%: {var_95:.2f}%\n'
        f'CVaR 95%: {crisis_results["cvar_95"]*100:.2f}%'
    )
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_drawdown_distribution(
    crisis_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot distribution of maximum drawdowns from simulation.
    
    Args:
        crisis_results: Results dictionary from CrisisSimulator.simulate_crisis()
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    max_drawdowns = crisis_results['max_drawdowns_distribution']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Histogram
    ax1.hist(max_drawdowns * 100, bins=50, alpha=0.7, color='coral',
             edgecolor='black', density=False)
    
    mean_dd = crisis_results['mean_max_drawdown'] * 100
    median_dd = crisis_results['median_max_drawdown'] * 100
    worst_dd = crisis_results['worst_drawdown'] * 100
    
    ax1.axvline(mean_dd, color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {mean_dd:.1f}%')
    ax1.axvline(median_dd, color='green', linestyle='--', linewidth=2,
                label=f'Median: {median_dd:.1f}%')
    ax1.axvline(worst_dd, color='red', linestyle='--', linewidth=2,
                label=f'Worst: {worst_dd:.1f}%')
    
    ax1.set_xlabel('Maximum Drawdown (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Drawdown Distribution - {crisis_results["scenario_name"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Box plot with violin
    ax2 = plt.subplot(1, 2, 2)
    parts = ax2.violinplot([max_drawdowns * 100], positions=[0], widths=0.7,
                           showmeans=True, showmedians=True)
    
    # Customize violin plot colors
    for pc in parts['bodies']:
        pc.set_facecolor('coral')
        pc.set_alpha(0.7)
    
    ax2.set_ylabel('Maximum Drawdown (%)')
    ax2.set_title('Drawdown Statistics')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics annotation
    stats_text = (
        f'Mean: {mean_dd:.2f}%\n'
        f'Median: {median_dd:.2f}%\n'
        f'Worst: {worst_dd:.2f}%\n'
        f'Best: {max_drawdowns.max()*100:.2f}%\n'
        f'Std Dev: {max_drawdowns.std()*100:.2f}%'
    )
    ax2.text(0.5, 0.02, stats_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_risk_metrics_comparison(
    baseline_metrics: Dict,
    crisis_results: Dict,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare risk metrics between baseline and crisis scenarios.
    
    Args:
        baseline_metrics: Dictionary with baseline metrics
                         (expected_return, volatility, sharpe_ratio)
        crisis_results: Results dictionary from CrisisSimulator.simulate_crisis()
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metric 1: Returns comparison
    ax1 = axes[0, 0]
    returns = [
        baseline_metrics.get('expected_return', 0) * 100,
        crisis_results['mean_return'] * 100
    ]
    colors = ['green' if r > 0 else 'red' for r in returns]
    bars1 = ax1.bar(['Baseline', 'Crisis'], returns, color=colors, alpha=0.7,
                    edgecolor='black')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Expected Return (%)')
    ax1.set_title('Expected Returns')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom' if height > 0 else 'top')
    
    # Metric 2: Volatility comparison
    ax2 = axes[0, 1]
    volatilities = [
        baseline_metrics.get('volatility', 0) * 100,
        crisis_results['std_return'] * 100
    ]
    bars2 = ax2.bar(['Baseline', 'Crisis'], volatilities,
                    color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Volatility (%)')
    ax2.set_title('Return Volatility')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    # Metric 3: VaR and CVaR
    ax3 = axes[1, 0]
    var_cvar = [
        crisis_results['var_95'] * 100,
        crisis_results['cvar_95'] * 100
    ]
    bars3 = ax3.bar(['VaR 95%', 'CVaR 95%'], var_cvar,
                    color=['orange', 'red'], alpha=0.7, edgecolor='black')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Loss (%)')
    ax3.set_title('Value at Risk Metrics (Crisis)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='top')
    
    # Metric 4: Probability outcomes
    ax4 = axes[1, 1]
    prob_data = [
        crisis_results['prob_profit'] * 100,
        crisis_results['prob_loss'] * 100,
        crisis_results['prob_severe_loss'] * 100
    ]
    colors4 = ['green', 'orange', 'red']
    bars4 = ax4.bar(['Profit\n(>0%)', 'Loss\n(<0%)', 'Severe Loss\n(<-20%)'],
                    prob_data, color=colors4, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Probability (%)')
    ax4.set_title('Outcome Probabilities (Crisis)')
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Overall title
    fig.suptitle(f'Risk Metrics Comparison - {crisis_results["scenario_name"]}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_scenario_comparison(
    comparison_results: Dict,
    metrics: List[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Compare multiple crisis scenarios.
    
    Args:
        comparison_results: Dictionary from CrisisSimulator.compare_scenarios()
        metrics: List of metric names to compare (default: key metrics)
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    if metrics is None:
        metrics = ['mean_return', 'var_95', 'cvar_95', 'worst_drawdown', 'prob_profit']
    
    scenario_names = list(comparison_results.keys())
    n_scenarios = len(scenario_names)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    metric_configs = {
        'mean_return': ('Mean Return (%)', 'green', 100),
        'var_95': ('VaR 95% (%)', 'orange', 100),
        'cvar_95': ('CVaR 95% (%)', 'red', 100),
        'worst_drawdown': ('Worst Drawdown (%)', 'darkred', 100),
        'prob_profit': ('Probability of Profit (%)', 'blue', 100),
        'avg_recovery_days': ('Avg Recovery (days)', 'purple', 1)
    }
    
    for idx, metric in enumerate(metrics):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        values = [comparison_results[name][metric] * metric_configs[metric][2] 
                  for name in scenario_names]
        
        bars = ax.bar(range(n_scenarios), values,
                     color=metric_configs[metric][1], alpha=0.7,
                     edgecolor='black')
        
        ax.set_xticks(range(n_scenarios))
        ax.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax.set_ylabel(metric_configs[metric][0])
        ax.set_title(metric_configs[metric][0])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)
    
    # Remove extra subplots
    for idx in range(len(metrics), len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle('Crisis Scenario Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_crisis_forecast(
    time_days: np.ndarray,
    baseline_forecast: np.ndarray,
    crisis_forecast: np.ndarray,
    baseline_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    crisis_ci: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    crisis_start: int = 0,
    crisis_end: Optional[int] = None,
    scenario_name: str = "Crisis Scenario",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot baseline vs crisis forecast with confidence intervals.
    
    Args:
        time_days: Array of time points (days)
        baseline_forecast: Baseline forecasted returns
        crisis_forecast: Crisis forecasted returns
        baseline_ci: Tuple of (lower, upper) confidence intervals for baseline
        crisis_ci: Tuple of (lower, upper) confidence intervals for crisis
        crisis_start: Day when crisis begins
        crisis_end: Day when crisis ends (None = end of forecast)
        scenario_name: Name of the crisis scenario
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if crisis_end is None:
        crisis_end = len(time_days)
    
    # Plot baseline forecast
    ax.plot(time_days, baseline_forecast * 100, 'b-', linewidth=2,
            label='Baseline Forecast', alpha=0.8)
    
    # Plot baseline confidence interval
    if baseline_ci is not None:
        ax.fill_between(time_days, baseline_ci[0] * 100, baseline_ci[1] * 100,
                        color='blue', alpha=0.2, label='Baseline 95% CI')
    
    # Plot crisis forecast
    ax.plot(time_days, crisis_forecast * 100, 'r-', linewidth=2,
            label='Crisis Forecast', alpha=0.8)
    
    # Plot crisis confidence interval
    if crisis_ci is not None:
        ax.fill_between(time_days, crisis_ci[0] * 100, crisis_ci[1] * 100,
                        color='red', alpha=0.2, label='Crisis 95% CI')
    
    # Highlight crisis period
    ax.axvspan(crisis_start, crisis_end, color='red', alpha=0.1,
               label='Crisis Period')
    
    # Zero line
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Days')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(f'Baseline vs Crisis Forecast - {scenario_name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    final_baseline = baseline_forecast[-1] * 100
    final_crisis = crisis_forecast[-1] * 100
    impact = final_crisis - final_baseline
    
    stats_text = (
        f'Final Baseline: {final_baseline:.2f}%\n'
        f'Final Crisis: {final_crisis:.2f}%\n'
        f'Impact: {impact:.2f}%'
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_crisis_report_summary(
    crisis_results: Dict,
    recommendation: str,
    rationale: str,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create a comprehensive summary figure for crisis analysis.
    
    Args:
        crisis_results: Results dictionary from CrisisSimulator.simulate_crisis()
        recommendation: Investment recommendation string
        rationale: Rationale for the recommendation
        save_path: Path to save the figure (optional)
        
    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle(f'Crisis Analysis Report - {crisis_results["scenario_name"]}',
                 fontsize=16, fontweight='bold')
    
    # 1. Return distribution (top left, wide)
    ax1 = fig.add_subplot(gs[0, :2])
    final_returns = crisis_results['final_returns_distribution']
    ax1.hist(final_returns * 100, bins=50, alpha=0.7, color='steelblue',
             edgecolor='black', density=True)
    
    from scipy import stats
    kde = stats.gaussian_kde(final_returns * 100)
    x_range = np.linspace(final_returns.min() * 100, final_returns.max() * 100, 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2)
    ax1.axvline(crisis_results['var_95'] * 100, color='red', linestyle='--', 
                linewidth=2, label='VaR 95%')
    ax1.axvline(crisis_results['mean_return'] * 100, color='green',
                linestyle='--', linewidth=2, label='Mean')
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Return Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Key metrics (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    metrics_text = (
        'Key Risk Metrics\n'
        '═══════════════\n\n'
        f'Mean Return: {crisis_results["mean_return"]*100:.2f}%\n'
        f'Volatility: {crisis_results["std_return"]*100:.2f}%\n\n'
        f'VaR 95%: {crisis_results["var_95"]*100:.2f}%\n'
        f'CVaR 95%: {crisis_results["cvar_95"]*100:.2f}%\n\n'
        f'Max DD: {crisis_results["worst_drawdown"]*100:.2f}%\n'
        f'Avg DD: {crisis_results["mean_max_drawdown"]*100:.2f}%\n\n'
        f'Prob Profit: {crisis_results["prob_profit"]*100:.1f}%\n'
        f'Prob Loss: {crisis_results["prob_loss"]*100:.1f}%\n\n'
        f'Recovery: {crisis_results["avg_recovery_days"]:.0f} days'
    )
    ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 3. Drawdown distribution (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    max_drawdowns = crisis_results['max_drawdowns_distribution']
    ax3.hist(max_drawdowns * 100, bins=30, alpha=0.7, color='coral',
             edgecolor='black')
    ax3.axvline(crisis_results['mean_max_drawdown'] * 100, color='blue',
                linestyle='--', linewidth=2, label='Mean')
    ax3.set_xlabel('Max Drawdown (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Drawdown Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Probability bars (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    probs = [crisis_results['prob_profit'] * 100,
             crisis_results['prob_loss'] * 100,
             crisis_results['prob_severe_loss'] * 100]
    colors = ['green', 'orange', 'red']
    bars = ax4.bar(['Profit', 'Loss', 'Severe\nLoss'], probs,
                   color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Probability (%)')
    ax4.set_title('Outcome Probabilities')
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 5. Risk metrics bars (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    risk_values = [abs(crisis_results['var_95']) * 100,
                   abs(crisis_results['cvar_95']) * 100]
    bars5 = ax5.bar(['VaR\n95%', 'CVaR\n95%'], risk_values,
                    color=['orange', 'red'], alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Loss (%)')
    ax5.set_title('Value at Risk')
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar in bars5:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
    
    # 6. Recommendation (bottom, full width)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Color code recommendation
    rec_colors = {
        'Worthy': 'lightgreen',
        'Proceed with Caution': 'yellow',
        'Not Recommended': 'lightcoral'
    }
    rec_color = rec_colors.get(recommendation, 'lightgray')
    
    recommendation_text = (
        f'INVESTMENT RECOMMENDATION\n'
        f'═══════════════════════════════════\n\n'
        f'Status: {recommendation.upper()}\n\n'
        f'Rationale:\n{rationale}\n\n'
        f'Simulation Details:\n'
        f'• Scenarios: {crisis_results["n_simulations"]:,} Monte Carlo simulations\n'
        f'• Duration: {crisis_results["duration_days"]} days\n'
        f'• Scenario: {crisis_results["scenario_name"]}'
    )
    
    ax6.text(0.5, 0.5, recommendation_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=rec_color, alpha=0.7,
                     edgecolor='black', linewidth=2))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
