"""
CAPM analysis results data structures.

This module provides data classes for storing and managing CAPM analysis results.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd


@dataclass
class CAPMResults:
    """Results from CAPM analysis on multiple stocks.
    
    Attributes:
        tickers: List of stock ticker symbols
        betas: Beta coefficients for each stock
        expected_returns: Expected returns from CAPM formula
        actual_returns: Actual historical returns
        alphas: Jensen's alpha for each stock
        r_squared: R-squared values (goodness of fit)
        classifications: Stock classifications (undervalued/overvalued/fairly_valued)
        recommendations: Investment recommendations (BUY/SELL/HOLD)
        sml_line: Tuple of (beta_range, expected_return_range) for SML plotting
        risk_free_rate: Risk-free rate used in analysis
        market_return: Market return used in analysis
    """
    tickers: List[str]
    betas: np.ndarray
    expected_returns: np.ndarray
    actual_returns: np.ndarray
    alphas: np.ndarray
    r_squared: np.ndarray
    classifications: List[str]
    recommendations: List[str]
    sml_line: Tuple[np.ndarray, np.ndarray]
    risk_free_rate: float
    market_return: float
    
    def __post_init__(self):
        """Validate results after initialization."""
        validation = self.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid CAPMResults:\n{validation}")
    
    def validate(self) -> 'ValidationResult':
        """Validate CAPM results.
        
        Returns:
            ValidationResult with any errors or warnings
        """
        from src.data.models import ValidationResult
        
        result = ValidationResult(is_valid=True)
        
        # Check lengths match
        n = len(self.tickers)
        
        if len(self.betas) != n:
            result.add_error(f"betas length {len(self.betas)} != tickers length {n}")
        
        if len(self.expected_returns) != n:
            result.add_error(
                f"expected_returns length {len(self.expected_returns)} != tickers length {n}"
            )
        
        if len(self.actual_returns) != n:
            result.add_error(
                f"actual_returns length {len(self.actual_returns)} != tickers length {n}"
            )
        
        if len(self.alphas) != n:
            result.add_error(f"alphas length {len(self.alphas)} != tickers length {n}")
        
        if len(self.r_squared) != n:
            result.add_error(
                f"r_squared length {len(self.r_squared)} != tickers length {n}"
            )
        
        if len(self.classifications) != n:
            result.add_error(
                f"classifications length {len(self.classifications)} != tickers length {n}"
            )
        
        if len(self.recommendations) != n:
            result.add_error(
                f"recommendations length {len(self.recommendations)} != tickers length {n}"
            )
        
        # Validate value ranges
        if not result.is_valid:
            return result  # Don't continue if lengths don't match
        
        # Check R-squared values
        if not np.all((self.r_squared >= 0) & (self.r_squared <= 1)):
            result.add_error("r_squared values must be between 0 and 1")
        
        # Check for NaN or inf values
        if np.isnan(self.betas).any() or np.isinf(self.betas).any():
            result.add_error("betas contains NaN or infinite values")
        
        if np.isnan(self.expected_returns).any() or np.isinf(self.expected_returns).any():
            result.add_error("expected_returns contains NaN or infinite values")
        
        if np.isnan(self.actual_returns).any() or np.isinf(self.actual_returns).any():
            result.add_error("actual_returns contains NaN or infinite values")
        
        if np.isnan(self.alphas).any() or np.isinf(self.alphas).any():
            result.add_error("alphas contains NaN or infinite values")
        
        # Check classifications are valid
        valid_classifications = {"undervalued", "overvalued", "fairly_valued"}
        for i, cls in enumerate(self.classifications):
            if cls not in valid_classifications:
                result.add_error(
                    f"Invalid classification '{cls}' for {self.tickers[i]}"
                )
        
        # Check recommendations are valid
        valid_recommendations = {"BUY", "SELL", "HOLD"}
        for i, rec in enumerate(self.recommendations):
            if rec not in valid_recommendations:
                result.add_error(
                    f"Invalid recommendation '{rec}' for {self.tickers[i]}"
                )
        
        # Validate SML line
        if not isinstance(self.sml_line, tuple) or len(self.sml_line) != 2:
            result.add_error("sml_line must be tuple of (beta_range, return_range)")
        else:
            beta_range, return_range = self.sml_line
            if len(beta_range) != len(return_range):
                result.add_error("sml_line beta_range and return_range must have same length")
        
        # Validate rates
        if not 0 <= self.risk_free_rate <= 0.20:
            result.add_error(
                f"risk_free_rate must be between 0 and 0.20, got {self.risk_free_rate}"
            )
        
        if not -0.5 <= self.market_return <= 0.5:
            result.add_warning(
                f"market_return {self.market_return:.2%} seems unusual"
            )
        
        return result
    
    def get_undervalued_stocks(self, threshold: float = 0.02) -> List[str]:
        """Get list of undervalued stocks (alpha > threshold).
        
        Args:
            threshold: Alpha threshold (default 2%)
            
        Returns:
            List of ticker symbols for undervalued stocks
        """
        return [
            ticker for ticker, alpha in zip(self.tickers, self.alphas)
            if alpha > threshold
        ]
    
    def get_overvalued_stocks(self, threshold: float = 0.02) -> List[str]:
        """Get list of overvalued stocks (alpha < -threshold).
        
        Args:
            threshold: Alpha threshold (default 2%)
            
        Returns:
            List of ticker symbols for overvalued stocks
        """
        return [
            ticker for ticker, alpha in zip(self.tickers, self.alphas)
            if alpha < -threshold
        ]
    
    def get_fairly_valued_stocks(self, threshold: float = 0.02) -> List[str]:
        """Get list of fairly valued stocks (|alpha| <= threshold).
        
        Args:
            threshold: Alpha threshold (default 2%)
            
        Returns:
            List of ticker symbols for fairly valued stocks
        """
        return [
            ticker for ticker, alpha in zip(self.tickers, self.alphas)
            if abs(alpha) <= threshold
        ]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame.
        
        Returns:
            DataFrame with all CAPM metrics
        """
        return pd.DataFrame({
            'Ticker': self.tickers,
            'Beta': self.betas,
            'Expected_Return': self.expected_returns,
            'Actual_Return': self.actual_returns,
            'Alpha': self.alphas,
            'R_Squared': self.r_squared,
            'Classification': self.classifications,
            'Recommendation': self.recommendations
        })
    
    def get_stock_details(self, ticker: str) -> Dict:
        """Get detailed metrics for a specific stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock metrics
            
        Raises:
            ValueError: If ticker not found
        """
        if ticker not in self.tickers:
            raise ValueError(f"Ticker '{ticker}' not found in results")
        
        idx = self.tickers.index(ticker)
        
        return {
            'ticker': ticker,
            'beta': float(self.betas[idx]),
            'expected_return': float(self.expected_returns[idx]),
            'actual_return': float(self.actual_returns[idx]),
            'alpha': float(self.alphas[idx]),
            'r_squared': float(self.r_squared[idx]),
            'classification': self.classifications[idx],
            'recommendation': self.recommendations[idx]
        }
    
    def summary_statistics(self) -> Dict:
        """Calculate summary statistics across all stocks.
        
        Returns:
            Dictionary with summary metrics
        """
        return {
            'num_stocks': len(self.tickers),
            'mean_beta': float(np.mean(self.betas)),
            'median_beta': float(np.median(self.betas)),
            'mean_alpha': float(np.mean(self.alphas)),
            'median_alpha': float(np.median(self.alphas)),
            'mean_r_squared': float(np.mean(self.r_squared)),
            'num_undervalued': len(self.get_undervalued_stocks()),
            'num_overvalued': len(self.get_overvalued_stocks()),
            'num_fairly_valued': len(self.get_fairly_valued_stocks()),
            'num_buy': sum(1 for r in self.recommendations if r == 'BUY'),
            'num_sell': sum(1 for r in self.recommendations if r == 'SELL'),
            'num_hold': sum(1 for r in self.recommendations if r == 'HOLD'),
        }


@dataclass
class ValidationResult:
    """Results from validation operations.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages
        warnings: List of warning messages
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
    
    def __str__(self) -> str:
        """String representation."""
        if self.is_valid:
            result = "Validation passed"
            if self.warnings:
                result += f" with {len(self.warnings)} warning(s)"
        else:
            result = f"Validation failed with {len(self.errors)} error(s)"
        
        if self.errors:
            result += "\nErrors:\n" + "\n".join(f"  - {e}" for e in self.errors)
        if self.warnings:
            result += "\nWarnings:\n" + "\n".join(f"  - {w}" for w in self.warnings)
        
        return result
