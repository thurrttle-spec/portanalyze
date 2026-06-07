"""
CAPM calculation functions for financial analysis.

This module provides core calculation functions for Capital Asset Pricing Model (CAPM)
analysis including returns, beta, expected returns, and alpha calculations.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series.
    
    Formula: r_t = ln(P_t / P_{t-1})
    
    Args:
        prices: Price series indexed by date
        
    Returns:
        Log returns series (first value will be NaN)
        
    Raises:
        ValueError: If prices contain non-positive values, NaN, or inf
    """
    # Validate inputs
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be pd.Series, got {type(prices)}")
    
    if len(prices) == 0:
        raise ValueError("prices series is empty")
    
    if (prices <= 0).any():
        raise ValueError("prices must be positive, found non-positive values")
    
    if prices.isna().any():
        raise ValueError("prices contains NaN values")
    
    if np.isinf(prices).any():
        raise ValueError("prices contains infinite values")
    
    # Calculate log returns
    returns = np.log(prices / prices.shift(1))
    
    # Validate returns
    if np.isinf(returns.dropna()).any():
        raise ValueError("Calculated returns contain infinite values")
    
    return returns


def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualize returns from periodic returns.
    
    Args:
        returns: Series of periodic returns (daily, monthly, etc.)
        periods_per_year: Number of periods in a year (252 for daily, 12 for monthly)
        
    Returns:
        Annualized return as decimal
        
    Raises:
        ValueError: If returns contain inf/nan or periods_per_year is invalid
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(f"returns must be pd.Series, got {type(returns)}")
    
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        raise ValueError("returns series has no valid values")
    
    if np.isinf(returns_clean).any():
        raise ValueError("returns contains infinite values")
    
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    
    # Calculate annualized return
    mean_return = returns_clean.mean()
    annualized = mean_return * periods_per_year
    
    return float(annualized)


def calculate_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    min_observations: int = 30
) -> Tuple[float, float]:
    """Calculate stock beta using covariance method.
    
    Formula: β = Cov(R_stock, R_market) / Var(R_market)
    
    Also calculates R-squared to measure goodness of fit.
    
    Args:
        stock_returns: Stock return series
        market_returns: Market return series
        min_observations: Minimum number of observations required
        
    Returns:
        Tuple of (beta, r_squared)
        
    Raises:
        ValueError: If insufficient data, misaligned data, or invalid values
    """
    # Validate inputs
    if not isinstance(stock_returns, pd.Series):
        raise TypeError(f"stock_returns must be pd.Series, got {type(stock_returns)}")
    
    if not isinstance(market_returns, pd.Series):
        raise TypeError(f"market_returns must be pd.Series, got {type(market_returns)}")
    
    # Align the series by index and drop NaN values
    combined = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    if len(combined) < min_observations:
        raise ValueError(
            f"Insufficient observations: {len(combined)} < {min_observations}"
        )
    
    stock_clean = combined['stock']
    market_clean = combined['market']
    
    # Check for infinite values
    if np.isinf(stock_clean).any() or np.isinf(market_clean).any():
        raise ValueError("Returns contain infinite values")
    
    # Calculate covariance and variance
    covariance = np.cov(stock_clean, market_clean)[0, 1]
    market_variance = np.var(market_clean, ddof=1)
    
    if market_variance <= 0:
        raise ValueError(
            f"Market variance must be positive, got {market_variance}"
        )
    
    # Calculate beta
    beta = covariance / market_variance
    
    # Calculate R-squared
    # R² = (correlation)²
    correlation = np.corrcoef(stock_clean, market_clean)[0, 1]
    r_squared = correlation ** 2
    
    return float(beta), float(r_squared)


def calculate_expected_return(
    risk_free_rate: float,
    beta: float,
    market_return: float
) -> float:
    """Calculate expected return using CAPM formula.
    
    Formula: E(R_i) = R_f + β_i * (E(R_m) - R_f)
    
    Args:
        risk_free_rate: Risk-free rate (annual, as decimal)
        beta: Stock beta coefficient
        market_return: Expected market return (annual, as decimal)
        
    Returns:
        Expected return as decimal
        
    Raises:
        ValueError: If inputs are invalid or out of realistic range
    """
    # Validate inputs
    if not isinstance(risk_free_rate, (int, float)):
        raise TypeError(
            f"risk_free_rate must be numeric, got {type(risk_free_rate)}"
        )
    
    if not isinstance(beta, (int, float)):
        raise TypeError(f"beta must be numeric, got {type(beta)}")
    
    if not isinstance(market_return, (int, float)):
        raise TypeError(f"market_return must be numeric, got {type(market_return)}")
    
    # Validate ranges
    if not 0 <= risk_free_rate <= 0.20:
        raise ValueError(
            f"risk_free_rate must be between 0 and 0.20, got {risk_free_rate}"
        )
    
    if np.isnan(beta) or np.isinf(beta):
        raise ValueError(f"beta must be finite number, got {beta}")
    
    if not -0.5 <= market_return <= 0.5:
        raise ValueError(
            f"market_return must be between -0.5 and 0.5, got {market_return}"
        )
    
    # Calculate market risk premium
    market_risk_premium = market_return - risk_free_rate
    
    # Apply CAPM formula
    expected_return = risk_free_rate + beta * market_risk_premium
    
    return float(expected_return)


def calculate_alpha(actual_return: float, expected_return: float) -> float:
    """Calculate Jensen's alpha.
    
    Formula: α = R_actual - R_expected
    
    Positive alpha indicates stock outperformed (undervalued),
    negative alpha indicates stock underperformed (overvalued).
    
    Args:
        actual_return: Actual historical return
        expected_return: Expected return from CAPM
        
    Returns:
        Alpha as decimal
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(actual_return, (int, float)):
        raise TypeError(f"actual_return must be numeric, got {type(actual_return)}")
    
    if not isinstance(expected_return, (int, float)):
        raise TypeError(
            f"expected_return must be numeric, got {type(expected_return)}"
        )
    
    if np.isnan(actual_return) or np.isinf(actual_return):
        raise ValueError(f"actual_return must be finite, got {actual_return}")
    
    if np.isnan(expected_return) or np.isinf(expected_return):
        raise ValueError(f"expected_return must be finite, got {expected_return}")
    
    # Calculate alpha
    alpha = actual_return - expected_return
    
    return float(alpha)


def classify_stock(
    alpha: float,
    threshold: float = 0.02
) -> str:
    """Classify stock as undervalued, overvalued, or fairly valued.
    
    Args:
        alpha: Jensen's alpha
        threshold: Classification threshold (default 2%)
        
    Returns:
        Classification string: "undervalued", "overvalued", or "fairly_valued"
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be numeric, got {type(alpha)}")
    
    if not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be numeric, got {type(threshold)}")
    
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    
    if np.isnan(alpha) or np.isinf(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    
    # Classify based on threshold
    if alpha > threshold:
        return "undervalued"
    elif alpha < -threshold:
        return "overvalued"
    else:
        return "fairly_valued"


def get_investment_recommendation(classification: str) -> str:
    """Get investment recommendation based on classification.
    
    Args:
        classification: Stock classification (undervalued/overvalued/fairly_valued)
        
    Returns:
        Investment recommendation: "BUY", "SELL", or "HOLD"
        
    Raises:
        ValueError: If classification is invalid
    """
    classification_map = {
        "undervalued": "BUY",
        "overvalued": "SELL",
        "fairly_valued": "HOLD"
    }
    
    if classification not in classification_map:
        raise ValueError(
            f"Invalid classification: {classification}. "
            f"Must be one of {list(classification_map.keys())}"
        )
    
    return classification_map[classification]
