"""
Risk metrics calculations for crisis analysis.
"""

import numpy as np
from typing import Union


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """Calculate Value at Risk (VaR).
    
    VaR is the maximum loss not exceeded with a given confidence level.
    For example, 95% VaR of -10% means there's a 95% chance the loss
    won't exceed 10%.
    
    Args:
        returns: Array of returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        VaR as a negative number (loss)
        
    Example:
        >>> returns = np.random.normal(-0.01, 0.02, 10000)
        >>> var_95 = calculate_var(returns, 0.95)
        >>> # var_95 might be -0.04 (4% loss at 95% confidence)
    """
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"returns must be numpy array, got {type(returns)}")
    
    if len(returns) == 0:
        raise ValueError("returns array is empty")
    
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    
    # VaR is the (1-confidence) quantile
    var = np.percentile(returns, (1 - confidence) * 100)
    
    return float(var)


def calculate_cvar(
    returns: np.ndarray,
    confidence: float = 0.95
) -> float:
    """Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.
    
    CVaR is the expected loss given that we're in the worst (1-confidence)% of outcomes.
    It's the average of all losses beyond VaR.
    
    Args:
        returns: Array of returns
        confidence: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        CVaR as a negative number (expected loss in tail)
        
    Example:
        >>> returns = np.random.normal(-0.01, 0.02, 10000)
        >>> cvar_95 = calculate_cvar(returns, 0.95)
        >>> # cvar_95 might be -0.06 (6% average loss in worst 5% of cases)
    """
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"returns must be numpy array, got {type(returns)}")
    
    if len(returns) == 0:
        raise ValueError("returns array is empty")
    
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
    
    # First calculate VaR
    var = calculate_var(returns, confidence)
    
    # CVaR is the mean of all returns <= VaR
    tail_returns = returns[returns <= var]
    
    if len(tail_returns) == 0:
        # Fallback to VaR if no tail returns (shouldn't happen with proper data)
        return var
    
    cvar = np.mean(tail_returns)
    
    return float(cvar)


def calculate_max_drawdown(
    portfolio_values: np.ndarray
) -> float:
    """Calculate maximum drawdown from portfolio value series.
    
    Drawdown is the peak-to-trough decline. Maximum drawdown is the
    largest such decline over the entire period.
    
    Args:
        portfolio_values: Time series of portfolio values
        
    Returns:
        Maximum drawdown as a negative percentage
        
    Example:
        >>> values = np.array([100, 110, 105, 95, 100, 120])
        >>> max_dd = calculate_max_drawdown(values)
        >>> # max_dd ≈ -0.136 (13.6% drawdown from 110 to 95)
    """
    if not isinstance(portfolio_values, np.ndarray):
        raise TypeError(
            f"portfolio_values must be numpy array, got {type(portfolio_values)}"
        )
    
    if len(portfolio_values) == 0:
        raise ValueError("portfolio_values array is empty")
    
    if len(portfolio_values) == 1:
        return 0.0  # No drawdown with single value
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Calculate drawdown at each point
    drawdowns = (portfolio_values - running_max) / running_max
    
    # Maximum drawdown is the most negative value
    max_dd = np.min(drawdowns)
    
    return float(max_dd)


def calculate_recovery_time(
    portfolio_values: np.ndarray,
    drawdown_threshold: float = -0.10
) -> int:
    """Calculate time to recover from drawdown.
    
    Args:
        portfolio_values: Time series of portfolio values
        drawdown_threshold: Drawdown threshold to measure recovery from
        
    Returns:
        Number of periods to recover (0 if no drawdown beyond threshold)
    """
    if not isinstance(portfolio_values, np.ndarray):
        raise TypeError(
            f"portfolio_values must be numpy array, got {type(portfolio_values)}"
        )
    
    if len(portfolio_values) < 2:
        return 0
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    
    # Calculate drawdown at each point
    drawdowns = (portfolio_values - running_max) / running_max
    
    # Find periods where drawdown exceeds threshold
    below_threshold = drawdowns < drawdown_threshold
    
    if not below_threshold.any():
        return 0  # Never exceeded threshold
    
    # Find first period below threshold
    first_below = np.where(below_threshold)[0][0]
    
    # Find when it recovers (drawdown >= 0)
    recovered = drawdowns[first_below:] >= 0
    
    if not recovered.any():
        # Never recovered
        return len(portfolio_values) - first_below
    
    # First period after first_below where drawdown >= 0
    recovery_period = np.where(recovered)[0][0]
    
    return int(recovery_period)


def calculate_downside_deviation(
    returns: np.ndarray,
    target_return: float = 0.0
) -> float:
    """Calculate downside deviation (semi-deviation).
    
    Measures volatility of returns below a target return.
    Useful for risk-averse investors who care more about downside.
    
    Args:
        returns: Array of returns
        target_return: Target return threshold (default 0)
        
    Returns:
        Downside deviation
    """
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"returns must be numpy array, got {type(returns)}")
    
    if len(returns) == 0:
        raise ValueError("returns array is empty")
    
    # Only consider returns below target
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return 0.0  # No downside
    
    # Calculate semi-deviation
    downside_dev = np.sqrt(np.mean((downside_returns - target_return) ** 2))
    
    return float(downside_dev)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """Calculate Sortino ratio.
    
    Similar to Sharpe ratio but uses downside deviation instead of
    total volatility. Better for asymmetric return distributions.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        target_return: Target return for downside calculation
        
    Returns:
        Sortino ratio
    """
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"returns must be numpy array, got {type(returns)}")
    
    if len(returns) == 0:
        raise ValueError("returns array is empty")
    
    mean_return = np.mean(returns)
    downside_dev = calculate_downside_deviation(returns, target_return)
    
    if downside_dev == 0:
        return float('inf') if mean_return > risk_free_rate else 0.0
    
    sortino = (mean_return - risk_free_rate) / downside_dev
    
    return float(sortino)


def calculate_calmar_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Calmar ratio.
    
    Ratio of annualized return to maximum drawdown.
    Measures return per unit of drawdown risk.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year for annualization
        
    Returns:
        Calmar ratio
    """
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"returns must be numpy array, got {type(returns)}")
    
    if len(returns) == 0:
        raise ValueError("returns array is empty")
    
    # Annualize return
    mean_return = np.mean(returns) * periods_per_year
    
    # Calculate max drawdown from cumulative returns
    cumulative_returns = np.cumprod(1 + returns)
    max_dd = calculate_max_drawdown(cumulative_returns)
    
    if max_dd == 0:
        return float('inf') if mean_return > risk_free_rate else 0.0
    
    calmar = (mean_return - risk_free_rate) / abs(max_dd)
    
    return float(calmar)
