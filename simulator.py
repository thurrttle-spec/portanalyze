"""
Monte Carlo crisis simulator for portfolio stress testing.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .scenario import CrisisScenario
from .metrics import calculate_var, calculate_cvar, calculate_max_drawdown


class CrisisSimulator:
    """Monte Carlo simulator for crisis scenarios.
    
    Simulates portfolio returns under crisis conditions using
    modified return distributions and Monte Carlo sampling.
    
    Attributes:
        n_simulations: Number of Monte Carlo runs (default 10,000)
        random_seed: Random seed for reproducibility
    """
    
    def __init__(self, n_simulations: int = 10000, random_seed: Optional[int] = None):
        """Initialize crisis simulator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility (None = not set)
        """
        if not isinstance(n_simulations, int):
            raise TypeError(f"n_simulations must be int, got {type(n_simulations)}")
        
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be positive, got {n_simulations}")
        
        if n_simulations < 1000:
            import warnings
            warnings.warn(
                f"n_simulations={n_simulations} may be too low for reliable estimates. "
                f"Consider using at least 1000 simulations."
            )
        
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_crisis(
        self,
        baseline_return: float,
        baseline_volatility: float,
        scenario: CrisisScenario,
        portfolio_value: float = 1000000.0
    ) -> Dict:
        """Simulate portfolio performance under crisis scenario.
        
        Args:
            baseline_return: Expected daily return under normal conditions
            baseline_volatility: Daily volatility under normal conditions
            scenario: Crisis scenario to simulate
            portfolio_value: Initial portfolio value (for dollar VaR)
            
        Returns:
            Dictionary with simulation results and risk metrics
        """
        # Validate inputs
        if not isinstance(baseline_return, (int, float)):
            raise TypeError(
                f"baseline_return must be numeric, got {type(baseline_return)}"
            )
        
        if not isinstance(baseline_volatility, (int, float)):
            raise TypeError(
                f"baseline_volatility must be numeric, got {type(baseline_volatility)}"
            )
        
        if baseline_volatility <= 0:
            raise ValueError(
                f"baseline_volatility must be positive, got {baseline_volatility}"
            )
        
        if not isinstance(scenario, CrisisScenario):
            raise TypeError(
                f"scenario must be CrisisScenario, got {type(scenario)}"
            )
        
        if portfolio_value <= 0:
            raise ValueError(
                f"portfolio_value must be positive, got {portfolio_value}"
            )
        
        # Apply crisis adjustments
        crisis_return = baseline_return + scenario.mean_shift / scenario.duration_days
        crisis_volatility = baseline_volatility * scenario.volatility_multiplier
        
        # Generate random returns for each simulation
        # Using normal distribution (can be extended to t-distribution for fat tails)
        simulated_returns = np.random.normal(
            loc=crisis_return,
            scale=crisis_volatility,
            size=(self.n_simulations, scenario.duration_days)
        )
        
        # Calculate cumulative returns for each simulation
        cumulative_returns = np.cumprod(1 + simulated_returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        # Calculate risk metrics
        var_95 = calculate_var(final_returns, confidence=0.95)
        var_99 = calculate_var(final_returns, confidence=0.99)
        cvar_95 = calculate_cvar(final_returns, confidence=0.95)
        cvar_99 = calculate_cvar(final_returns, confidence=0.99)
        
        # Calculate maximum drawdown for each simulation
        max_drawdowns = []
        for i in range(self.n_simulations):
            cum_return_path = cumulative_returns[i, :]
            portfolio_path = portfolio_value * (1 + cum_return_path)
            max_dd = calculate_max_drawdown(portfolio_path)
            max_drawdowns.append(max_dd)
        
        max_drawdowns = np.array(max_drawdowns)
        
        # Calculate probability of profit/loss
        prob_profit = np.mean(final_returns > 0)
        prob_loss = np.mean(final_returns < 0)
        prob_severe_loss = np.mean(final_returns < -0.20)  # >20% loss
        
        # Estimate recovery time (simplified)
        # Average number of days to recover to initial value
        recovery_times = []
        for i in range(self.n_simulations):
            cum_return_path = cumulative_returns[i, :]
            below_zero = cum_return_path < 0
            if below_zero.any():
                # Find last day below zero
                recovery_day = np.where(below_zero)[0][-1] + 1
                recovery_times.append(min(recovery_day, scenario.duration_days))
            else:
                recovery_times.append(0)
        
        avg_recovery_time = np.mean(recovery_times)
        
        return {
            # Scenario info
            'scenario_name': scenario.name,
            'duration_days': scenario.duration_days,
            'n_simulations': self.n_simulations,
            
            # Return statistics
            'mean_return': float(np.mean(final_returns)),
            'median_return': float(np.median(final_returns)),
            'std_return': float(np.std(final_returns)),
            'min_return': float(np.min(final_returns)),
            'max_return': float(np.max(final_returns)),
            
            # Risk metrics
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99),
            
            # Dollar VaR (for portfolio_value)
            'dollar_var_95': float(var_95 * portfolio_value),
            'dollar_var_99': float(var_99 * portfolio_value),
            
            # Drawdown statistics
            'mean_max_drawdown': float(np.mean(max_drawdowns)),
            'median_max_drawdown': float(np.median(max_drawdowns)),
            'worst_drawdown': float(np.min(max_drawdowns)),
            
            # Probabilities
            'prob_profit': float(prob_profit),
            'prob_loss': float(prob_loss),
            'prob_severe_loss': float(prob_severe_loss),
            
            # Recovery
            'avg_recovery_days': float(avg_recovery_time),
            
            # Full distributions (for analysis)
            'final_returns_distribution': final_returns,
            'max_drawdowns_distribution': max_drawdowns
        }
    
    def compare_scenarios(
        self,
        baseline_return: float,
        baseline_volatility: float,
        scenarios: list,
        portfolio_value: float = 1000000.0
    ) -> Dict:
        """Compare multiple crisis scenarios.
        
        Args:
            baseline_return: Expected daily return under normal conditions
            baseline_volatility: Daily volatility under normal conditions
            scenarios: List of CrisisScenario objects to compare
            portfolio_value: Initial portfolio value
            
        Returns:
            Dictionary with comparison results for all scenarios
        """
        results = {}
        
        for scenario in scenarios:
            scenario_results = self.simulate_crisis(
                baseline_return,
                baseline_volatility,
                scenario,
                portfolio_value
            )
            
            results[scenario.name] = scenario_results
        
        return results
    
    def assess_investment_worthiness(
        self,
        crisis_results: Dict,
        thresholds: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """Assess whether investment is worthy under crisis conditions.
        
        Args:
            crisis_results: Results from simulate_crisis()
            thresholds: Custom thresholds for assessment (optional)
                       Keys: 'max_var_95', 'max_cvar_95', 'max_drawdown',
                             'min_prob_profit'
            
        Returns:
            Tuple of (recommendation, rationale)
            recommendation: "Worthy", "Proceed with Caution", "Not Recommended"
        """
        # Default thresholds
        if thresholds is None:
            thresholds = {
                'max_var_95': -0.30,        # VaR shouldn't exceed -30%
                'max_cvar_95': -0.40,       # CVaR shouldn't exceed -40%
                'max_drawdown': -0.50,      # Max DD shouldn't exceed -50%
                'min_prob_profit': 0.30     # At least 30% chance of profit
            }
        
        # Extract metrics
        var_95 = crisis_results['var_95']
        cvar_95 = crisis_results['cvar_95']
        worst_dd = crisis_results['worst_drawdown']
        prob_profit = crisis_results['prob_profit']
        mean_return = crisis_results['mean_return']
        
        # Score the investment (0-100)
        score = 100
        reasons = []
        
        # Check VaR
        if var_95 < thresholds['max_var_95']:
            penalty = 30
            score -= penalty
            reasons.append(
                f"High Value at Risk ({var_95*100:.1f}% exceeds {thresholds['max_var_95']*100:.1f}% threshold)"
            )
        
        # Check CVaR
        if cvar_95 < thresholds['max_cvar_95']:
            penalty = 25
            score -= penalty
            reasons.append(
                f"High Conditional VaR ({cvar_95*100:.1f}% exceeds {thresholds['max_cvar_95']*100:.1f}% threshold)"
            )
        
        # Check max drawdown
        if worst_dd < thresholds['max_drawdown']:
            penalty = 25
            score -= penalty
            reasons.append(
                f"Severe potential drawdown ({worst_dd*100:.1f}% exceeds {thresholds['max_drawdown']*100:.1f}% threshold)"
            )
        
        # Check probability of profit
        if prob_profit < thresholds['min_prob_profit']:
            penalty = 20
            score -= penalty
            reasons.append(
                f"Low probability of profit ({prob_profit*100:.0f}% below {thresholds['min_prob_profit']*100:.0f}% threshold)"
            )
        
        # Positive factors
        if mean_return > 0:
            score += 10
            reasons.append(f"Expected return remains positive ({mean_return*100:.2f}%)")
        
        if prob_profit > 0.50:
            score += 10
            reasons.append(f"Favorable probability of profit ({prob_profit*100:.0f}%)")
        
        # Make recommendation based on score
        if score >= 70:
            recommendation = "Worthy"
            rationale = (
                f"Investment shows resilience under crisis scenario (score: {score}/100). "
                + " ".join(reasons[:2]) if reasons else "Risk metrics within acceptable thresholds."
            )
        elif score >= 40:
            recommendation = "Proceed with Caution"
            rationale = (
                f"Investment has moderate risk under crisis (score: {score}/100). "
                f"Consider risk management strategies. "
                + " ".join(reasons[:3])
            )
        else:
            recommendation = "Not Recommended"
            rationale = (
                f"Investment shows high vulnerability to crisis (score: {score}/100). "
                + " ".join(reasons[:3])
            )
        
        return recommendation, rationale
