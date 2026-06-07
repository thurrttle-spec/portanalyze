"""
Portfolio optimization functions for Black-Litterman model.

This module provides portfolio optimization using mean-variance optimization
with Sharpe ratio maximization.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy.optimize import minimize
import warnings


def optimize_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    constraints: Optional[Dict] = None
) -> np.ndarray:
    """Optimize portfolio weights using mean-variance optimization.
    
    Maximizes Sharpe ratio: (E[R_p] - R_f) / σ_p
    Subject to:
        - Weights sum to 1
        - Long-only: all weights >= 0
        - Optional: max position size constraint
    
    Args:
        expected_returns: Expected returns for each asset (N x 1)
        cov_matrix: Covariance matrix (N x N)
        risk_free_rate: Risk-free rate (annual)
        constraints: Optional dictionary with additional constraints:
                    - 'max_position': Maximum weight for any single asset
                    - 'min_position': Minimum weight for any asset (default 0)
    
    Returns:
        Optimal portfolio weights (N x 1 array)
    
    Raises:
        ValueError: If optimization fails or inputs are invalid
    
    Example:
        >>> returns = np.array([0.10, 0.08, 0.12])
        >>> cov = np.array([[0.04, 0.01, 0.01],
        ...                 [0.01, 0.03, 0.01],
        ...                 [0.01, 0.01, 0.02]])
        >>> weights = optimize_portfolio(returns, cov, risk_free_rate=0.03)
        >>> np.sum(weights)  # Should be 1.0
        1.0
    """
    # Validate inputs
    if not isinstance(expected_returns, np.ndarray):
        raise TypeError(
            f"expected_returns must be numpy array, got {type(expected_returns)}"
        )
    
    if expected_returns.ndim != 1:
        raise ValueError(
            f"expected_returns must be 1-dimensional, got shape {expected_returns.shape}"
        )
    
    n_assets = len(expected_returns)
    
    if not isinstance(cov_matrix, np.ndarray):
        raise TypeError(f"cov_matrix must be numpy array, got {type(cov_matrix)}")
    
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"cov_matrix shape {cov_matrix.shape} incompatible with "
            f"expected_returns length {n_assets}"
        )
    
    if not isinstance(risk_free_rate, (int, float)):
        raise TypeError(f"risk_free_rate must be numeric, got {type(risk_free_rate)}")
    
    if not 0 <= risk_free_rate <= 0.20:
        raise ValueError(
            f"risk_free_rate must be between 0 and 0.20, got {risk_free_rate}"
        )
    
    # Parse constraints
    constraints = constraints or {}
    max_position = constraints.get('max_position', 1.0)
    min_position = constraints.get('min_position', 0.0)
    
    if not 0 < max_position <= 1.0:
        raise ValueError(f"max_position must be between 0 and 1, got {max_position}")
    
    if not 0 <= min_position < max_position:
        raise ValueError(
            f"min_position must be between 0 and max_position, "
            f"got min={min_position}, max={max_position}"
        )
    
    # Define objective function: negative Sharpe ratio
    def negative_sharpe(weights):
        """Calculate negative Sharpe ratio for minimization."""
        portfolio_return = weights @ expected_returns
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        # Handle zero volatility
        if portfolio_vol < 1e-10:
            return 1e10  # Large penalty
        
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
        return -sharpe  # Negative because we minimize
    
    # Define constraints for optimizer
    opt_constraints = [
        # Weights sum to 1
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    ]
    
    # Define bounds (long-only with position limits)
    bounds = tuple((min_position, max_position) for _ in range(n_assets))
    
    # Initial guess: equal weights
    initial_weights = np.ones(n_assets) / n_assets
    
    # Adjust if equal weights violate constraints
    if initial_weights[0] > max_position:
        initial_weights = np.full(n_assets, max_position)
        initial_weights = initial_weights / np.sum(initial_weights)
    
    # Run optimization
    try:
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=opt_constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            warnings.warn(
                f"Optimization did not converge: {result.message}. "
                f"Using equal weights as fallback."
            )
            # Fallback to equal weights
            weights = np.ones(n_assets) / n_assets
        else:
            weights = result.x
            
    except Exception as e:
        warnings.warn(
            f"Optimization failed: {str(e)}. Using equal weights as fallback."
        )
        weights = np.ones(n_assets) / n_assets
    
    # Normalize to ensure sum = 1 (handle numerical errors)
    weights = weights / np.sum(weights)
    
    # Validate output
    if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
        raise RuntimeError(
            f"Optimized weights sum to {np.sum(weights)}, not 1.0"
        )
    
    if np.any(weights < -1e-6):  # Allow small numerical errors
        raise RuntimeError(
            f"Optimized weights contain negative values: {weights[weights < 0]}"
        )
    
    # Clip small negative values to zero
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)  # Renormalize
    
    return weights


def calculate_portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> Dict[str, float]:
    """Calculate portfolio performance metrics.
    
    Args:
        weights: Portfolio weights (N x 1)
        expected_returns: Expected returns for each asset (N x 1)
        cov_matrix: Covariance matrix (N x N)
        risk_free_rate: Risk-free rate
    
    Returns:
        Dictionary with:
            - 'return': Expected portfolio return
            - 'volatility': Portfolio volatility (std dev)
            - 'sharpe_ratio': Sharpe ratio
            - 'var_95': Value at Risk at 95% confidence (parametric)
            - 'diversification_ratio': Diversification ratio
    
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if not isinstance(weights, np.ndarray):
        raise TypeError(f"weights must be numpy array, got {type(weights)}")
    
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1-dimensional, got shape {weights.shape}")
    
    n_assets = len(weights)
    
    if not isinstance(expected_returns, np.ndarray):
        raise TypeError(
            f"expected_returns must be numpy array, got {type(expected_returns)}"
        )
    
    if len(expected_returns) != n_assets:
        raise ValueError(
            f"expected_returns length {len(expected_returns)} != weights length {n_assets}"
        )
    
    if not isinstance(cov_matrix, np.ndarray):
        raise TypeError(f"cov_matrix must be numpy array, got {type(cov_matrix)}")
    
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(
            f"cov_matrix shape {cov_matrix.shape} incompatible with weights length {n_assets}"
        )
    
    # Calculate portfolio return
    portfolio_return = float(weights @ expected_returns)
    
    # Calculate portfolio volatility
    portfolio_variance = weights @ cov_matrix @ weights
    portfolio_volatility = float(np.sqrt(portfolio_variance))
    
    # Calculate Sharpe ratio
    if portfolio_volatility > 1e-10:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    else:
        sharpe_ratio = 0.0
    
    # Calculate Value at Risk (parametric, 95% confidence)
    # VaR = -mean + z * std, where z = 1.645 for 95% confidence
    var_95 = -(portfolio_return - 1.645 * portfolio_volatility)
    
    # Calculate diversification ratio
    # DR = (sum of weighted volatilities) / portfolio volatility
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vols = weights @ individual_vols
    
    if portfolio_volatility > 1e-10:
        diversification_ratio = weighted_vols / portfolio_volatility
    else:
        diversification_ratio = 1.0
    
    return {
        'return': float(portfolio_return),
        'volatility': float(portfolio_volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'var_95': float(var_95),
        'diversification_ratio': float(diversification_ratio)
    }


def generate_efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate efficient frontier curve.
    
    Args:
        expected_returns: Expected returns for each asset (N x 1)
        cov_matrix: Covariance matrix (N x N)
        risk_free_rate: Risk-free rate
        n_points: Number of points on the frontier
    
    Returns:
        Tuple of:
            - volatilities: Portfolio volatilities
            - returns: Portfolio returns
            - sharpe_ratios: Sharpe ratios for each point
    
    Raises:
        ValueError: If inputs are invalid
    """
    n_assets = len(expected_returns)
    
    # Generate target returns from min to max
    min_return = np.min(expected_returns)
    max_return = np.max(expected_returns)
    target_returns = np.linspace(min_return, max_return, n_points)
    
    volatilities = []
    returns_list = []
    sharpe_ratios = []
    
    for target_return in target_returns:
        # Define objective: minimize volatility
        def portfolio_volatility(weights):
            return np.sqrt(weights @ cov_matrix @ weights)
        
        # Constraints: weights sum to 1, target return achieved
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: w @ expected_returns - target_return}
        ]
        
        # Bounds: long-only
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                portfolio_volatility,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = result.x
                vol = portfolio_volatility(weights)
                ret = weights @ expected_returns
                
                volatilities.append(vol)
                returns_list.append(ret)
                
                if vol > 1e-10:
                    sharpe = (ret - risk_free_rate) / vol
                else:
                    sharpe = 0.0
                sharpe_ratios.append(sharpe)
        except Exception:
            # Skip points that fail optimization
            continue
    
    return (
        np.array(volatilities),
        np.array(returns_list),
        np.array(sharpe_ratios)
    )
