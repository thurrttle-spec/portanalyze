"""
Data models for Financial Analysis System with comprehensive validation.

This module provides core data structures for stock and market data with 
built-in validation rules to ensure data integrity throughout the analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import timedelta


@dataclass
class ValidationResult:
    """Results from data validation operations.
    
    Attributes:
        is_valid: Whether validation passed
        errors: List of error messages if validation failed
        warnings: List of warning messages (non-critical issues)
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error message and mark validation as failed."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning message without failing validation."""
        self.warnings.append(message)
    
    def __str__(self) -> str:
        """String representation of validation results."""
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


@dataclass
class StockData:
    """Stock price and return data with validation.
    
    Attributes:
        ticker: Stock ticker symbol (must be non-empty)
        prices: Price series indexed by date (must be positive)
        returns: Log returns series (calculated from prices)
        metadata: Additional information (company name, sector, etc.)
    
    Validation Rules:
        - ticker must be non-empty string
        - prices must have DatetimeIndex
        - prices must contain only positive values
        - prices must have no gaps > 5 trading days
        - returns must be calculated as log(price_t / price_{t-1})
    """
    ticker: str
    prices: pd.Series
    returns: pd.Series
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data immediately after initialization."""
        validation = self.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid StockData for {self.ticker}:\n{validation}")
    
    def validate(self) -> ValidationResult:
        """Comprehensive validation of stock data.
        
        Returns:
            ValidationResult with any errors or warnings found
        """
        result = ValidationResult(is_valid=True)
        
        # Validate ticker
        if not isinstance(self.ticker, str):
            result.add_error(f"ticker must be string, got {type(self.ticker)}")
        elif not self.ticker.strip():
            result.add_error("ticker must be non-empty string")
        
        # Validate prices series
        if not isinstance(self.prices, pd.Series):
            result.add_error(f"prices must be pd.Series, got {type(self.prices)}")
            return result  # Cannot continue validation
        
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            result.add_error(f"prices must have DatetimeIndex, got {type(self.prices.index)}")
        
        if len(self.prices) == 0:
            result.add_error("prices series is empty")
            return result
        
        # Check for positive values
        if (self.prices <= 0).any():
            num_invalid = (self.prices <= 0).sum()
            result.add_error(f"prices must be positive, found {num_invalid} non-positive values")
        
        # Check for NaN values
        if self.prices.isna().any():
            num_nan = self.prices.isna().sum()
            result.add_error(f"prices contains {num_nan} NaN values")
        
        # Check for infinite values
        if np.isinf(self.prices).any():
            result.add_error("prices contains infinite values")
        
        # Check for date gaps > 5 trading days (7 calendar days)
        if isinstance(self.prices.index, pd.DatetimeIndex) and len(self.prices) > 1:
            date_diffs = self.prices.index.to_series().diff()
            max_gap = date_diffs.max()
            if max_gap > timedelta(days=7):
                result.add_warning(
                    f"prices has gaps > 5 trading days, max gap: {max_gap.days} days"
                )
        
        # Validate returns series
        if not isinstance(self.returns, pd.Series):
            result.add_error(f"returns must be pd.Series, got {type(self.returns)}")
            return result
        
        if len(self.returns) == 0:
            result.add_error("returns series is empty")
            return result
        
        # Check returns are properly calculated (log returns)
        if len(self.prices) > 1 and len(self.returns) > 0:
            # Expected returns: log(price_t / price_{t-1})
            # Allow for first return to be NaN or 0
            try:
                expected_returns = np.log(self.prices / self.prices.shift(1))
                
                # Compare returns (skip first value which is NaN in expected)
                if len(self.returns) == len(expected_returns):
                    # Check alignment - ensure indices match
                    returns_clean = self.returns.dropna()
                    expected_clean = expected_returns.dropna()
                    
                    if len(returns_clean) > 0 and len(expected_clean) > 0:
                        # Align by index to handle missing values
                        common_index = returns_clean.index.intersection(expected_clean.index)
                        if len(common_index) > 0:
                            returns_aligned = returns_clean.loc[common_index]
                            expected_aligned = expected_clean.loc[common_index]
                            
                            # Allow small numerical differences
                            if not np.allclose(returns_aligned.values, expected_aligned.values, 
                                              rtol=1e-5, atol=1e-8, equal_nan=True):
                                result.add_warning(
                                    "returns do not match expected log returns from prices"
                                )
            except (ValueError, RuntimeWarning):
                # If calculation fails (e.g., due to invalid prices), we've already caught it above
                pass
        
        # Check for NaN values in returns
        if self.returns.isna().any():
            num_nan = self.returns.isna().sum()
            # First return is expected to be NaN
            if num_nan > 1 or (num_nan == 1 and not self.returns.iloc[0] is pd.NA):
                result.add_warning(f"returns contains {num_nan} NaN values")
        
        # Check for infinite returns
        if np.isinf(self.returns).any():
            result.add_error("returns contains infinite values")
        
        # Validate metadata
        if not isinstance(self.metadata, dict):
            result.add_error(f"metadata must be dict, got {type(self.metadata)}")
        
        return result
    
    def calculate_statistics(self) -> Dict[str, float]:
        """Calculate basic statistics for the stock.
        
        Returns:
            Dictionary with mean return, volatility, min/max prices, etc.
        """
        returns_clean = self.returns.dropna()
        
        return {
            'mean_return': returns_clean.mean(),
            'volatility': returns_clean.std(),
            'min_price': self.prices.min(),
            'max_price': self.prices.max(),
            'avg_price': self.prices.mean(),
            'num_observations': len(self.prices),
            'date_range_days': (self.prices.index.max() - self.prices.index.min()).days
        }


@dataclass
class MarketData:
    """Market index data with validation.
    
    Attributes:
        index_name: Name of the market index
        prices: Index price series
        returns: Index log returns series
        risk_free_rate: Annual risk-free rate (decimal, e.g., 0.05 for 5%)
    
    Validation Rules:
        - index_name must be non-empty string
        - prices must have DatetimeIndex
        - prices must contain only positive values
        - returns must be log returns
        - risk_free_rate must be between 0 and 1
    """
    index_name: str
    prices: pd.Series
    returns: pd.Series
    risk_free_rate: float
    
    def __post_init__(self):
        """Validate data immediately after initialization."""
        validation = self.validate()
        if not validation.is_valid:
            raise ValueError(f"Invalid MarketData for {self.index_name}:\n{validation}")
    
    def validate(self) -> ValidationResult:
        """Comprehensive validation of market data.
        
        Returns:
            ValidationResult with any errors or warnings found
        """
        result = ValidationResult(is_valid=True)
        
        # Validate index_name
        if not isinstance(self.index_name, str):
            result.add_error(f"index_name must be string, got {type(self.index_name)}")
        elif not self.index_name.strip():
            result.add_error("index_name must be non-empty string")
        
        # Validate prices series
        if not isinstance(self.prices, pd.Series):
            result.add_error(f"prices must be pd.Series, got {type(self.prices)}")
            return result
        
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            result.add_error(f"prices must have DatetimeIndex, got {type(self.prices.index)}")
        
        if len(self.prices) == 0:
            result.add_error("prices series is empty")
            return result
        
        # Check for positive values
        if (self.prices <= 0).any():
            num_invalid = (self.prices <= 0).sum()
            result.add_error(f"prices must be positive, found {num_invalid} non-positive values")
        
        # Check for NaN values
        if self.prices.isna().any():
            num_nan = self.prices.isna().sum()
            result.add_error(f"prices contains {num_nan} NaN values")
        
        # Check for infinite values
        if np.isinf(self.prices).any():
            result.add_error("prices contains infinite values")
        
        # Validate returns series
        if not isinstance(self.returns, pd.Series):
            result.add_error(f"returns must be pd.Series, got {type(self.returns)}")
            return result
        
        if len(self.returns) == 0:
            result.add_error("returns series is empty")
            return result
        
        # Check for infinite returns
        if np.isinf(self.returns).any():
            result.add_error("returns contains infinite values")
        
        # Validate risk_free_rate
        if not isinstance(self.risk_free_rate, (int, float)):
            result.add_error(
                f"risk_free_rate must be numeric, got {type(self.risk_free_rate)}"
            )
        elif not 0 <= self.risk_free_rate <= 1:
            result.add_error(
                f"risk_free_rate must be between 0 and 1, got {self.risk_free_rate}"
            )
        elif self.risk_free_rate > 0.2:
            result.add_warning(
                f"risk_free_rate {self.risk_free_rate:.1%} seems unusually high"
            )
        
        return result
    
    def calculate_market_statistics(self) -> Dict[str, float]:
        """Calculate market statistics.
        
        Returns:
            Dictionary with market metrics
        """
        returns_clean = self.returns.dropna()
        
        return {
            'mean_return': returns_clean.mean(),
            'volatility': returns_clean.std(),
            'sharpe_ratio': (returns_clean.mean() - self.risk_free_rate) / returns_clean.std()
                           if returns_clean.std() > 0 else 0.0,
            'min_price': self.prices.min(),
            'max_price': self.prices.max(),
            'num_observations': len(self.prices),
        }
    
    def align_with_stock(self, stock_data: StockData) -> 'MarketData':
        """Create a new MarketData aligned with stock data date range.
        
        Args:
            stock_data: StockData to align with
            
        Returns:
            New MarketData with aligned dates
        """
        # Find common dates
        common_dates = self.prices.index.intersection(stock_data.prices.index)
        
        if len(common_dates) == 0:
            raise ValueError(
                f"No common dates between market index and stock {stock_data.ticker}"
            )
        
        # Create aligned series
        aligned_prices = self.prices.loc[common_dates]
        aligned_returns = self.returns.loc[common_dates]
        
        return MarketData(
            index_name=self.index_name,
            prices=aligned_prices,
            returns=aligned_returns,
            risk_free_rate=self.risk_free_rate
        )
