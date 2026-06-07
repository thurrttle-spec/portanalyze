"""
Investor views system for Black-Litterman model.

This module provides the InvestorViews class for managing and constructing
investor views on asset returns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class InvestorViews:
    """Manager for investor views in Black-Litterman model.
    
    Views can be:
    - Absolute: "Asset i will return x%"
    - Relative: "Asset i will outperform asset j by x%"
    
    Attributes:
        n_assets: Number of assets in the portfolio
        view_matrix: P matrix (K views x N assets)
        view_returns: Q vector (K x 1) of expected returns for each view
        view_uncertainty: Ω matrix (K x K) of view confidence
        view_descriptions: List of textual descriptions for each view
    """
    n_assets: int
    view_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    view_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    view_uncertainty: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    view_descriptions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize empty view structures."""
        if self.n_assets <= 0:
            raise ValueError(f"n_assets must be positive, got {self.n_assets}")
        
        # Initialize empty views if not provided
        if self.view_matrix.size == 0:
            self.view_matrix = np.empty((0, self.n_assets))
        if self.view_returns.size == 0:
            self.view_returns = np.array([])
        if self.view_uncertainty.size == 0:
            self.view_uncertainty = np.empty((0, 0))
    
    @property
    def n_views(self) -> int:
        """Number of views currently defined."""
        return len(self.view_returns)
    
    def add_absolute_view(
        self,
        asset_index: int,
        expected_return: float,
        confidence: float
    ) -> None:
        """Add an absolute view on a single asset.
        
        Absolute view: "Asset i will return expected_return"
        
        Args:
            asset_index: Index of the asset (0-based)
            expected_return: Expected return for the asset (as decimal)
            confidence: Confidence in the view (0-1, higher = more confident)
                       Used to calculate view uncertainty: Ω_k = (1-confidence) * var
        
        Raises:
            ValueError: If inputs are invalid
        
        Example:
            >>> views = InvestorViews(n_assets=3)
            >>> views.add_absolute_view(0, 0.10, confidence=0.8)
            >>> # "Asset 0 will return 10%" with 80% confidence
        """
        # Validate inputs
        if not isinstance(asset_index, int):
            raise TypeError(f"asset_index must be int, got {type(asset_index)}")
        
        if not 0 <= asset_index < self.n_assets:
            raise ValueError(
                f"asset_index must be between 0 and {self.n_assets-1}, got {asset_index}"
            )
        
        if not isinstance(expected_return, (int, float)):
            raise TypeError(
                f"expected_return must be numeric, got {type(expected_return)}"
            )
        
        if not -0.5 <= expected_return <= 0.5:
            raise ValueError(
                f"expected_return must be between -0.5 and 0.5, got {expected_return}"
            )
        
        if not isinstance(confidence, (int, float)):
            raise TypeError(f"confidence must be numeric, got {type(confidence)}")
        
        if not 0 < confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0 and 1.0, got {confidence}"
            )
        
        # Create view vector: [0, 0, ..., 1, ..., 0] with 1 at asset_index
        view_vector = np.zeros(self.n_assets)
        view_vector[asset_index] = 1.0
        
        # Calculate view uncertainty from confidence
        # Higher confidence -> lower uncertainty
        # Use a simple scaling: uncertainty = (1 - confidence) * 0.01
        uncertainty = (1 - confidence) * 0.01
        
        # Add view
        self._add_view(
            view_vector,
            expected_return,
            uncertainty,
            f"Asset {asset_index} will return {expected_return*100:.2f}% "
            f"(confidence: {confidence*100:.0f}%)"
        )
    
    def add_relative_view(
        self,
        asset1_index: int,
        asset2_index: int,
        outperformance: float,
        confidence: float
    ) -> None:
        """Add a relative view between two assets.
        
        Relative view: "Asset i will outperform asset j by outperformance"
        
        Args:
            asset1_index: Index of first asset (0-based)
            asset2_index: Index of second asset (0-based)
            outperformance: Expected outperformance (as decimal)
                          Positive: asset1 outperforms asset2
                          Negative: asset2 outperforms asset1
            confidence: Confidence in the view (0-1, higher = more confident)
        
        Raises:
            ValueError: If inputs are invalid
        
        Example:
            >>> views = InvestorViews(n_assets=3)
            >>> views.add_relative_view(0, 1, 0.03, confidence=0.7)
            >>> # "Asset 0 will outperform asset 1 by 3%" with 70% confidence
        """
        # Validate inputs
        if not isinstance(asset1_index, int):
            raise TypeError(f"asset1_index must be int, got {type(asset1_index)}")
        
        if not isinstance(asset2_index, int):
            raise TypeError(f"asset2_index must be int, got {type(asset2_index)}")
        
        if not 0 <= asset1_index < self.n_assets:
            raise ValueError(
                f"asset1_index must be between 0 and {self.n_assets-1}, got {asset1_index}"
            )
        
        if not 0 <= asset2_index < self.n_assets:
            raise ValueError(
                f"asset2_index must be between 0 and {self.n_assets-1}, got {asset2_index}"
            )
        
        if asset1_index == asset2_index:
            raise ValueError(
                f"asset1_index and asset2_index must be different, both are {asset1_index}"
            )
        
        if not isinstance(outperformance, (int, float)):
            raise TypeError(
                f"outperformance must be numeric, got {type(outperformance)}"
            )
        
        if not -0.5 <= outperformance <= 0.5:
            raise ValueError(
                f"outperformance must be between -0.5 and 0.5, got {outperformance}"
            )
        
        if not isinstance(confidence, (int, float)):
            raise TypeError(f"confidence must be numeric, got {type(confidence)}")
        
        if not 0 < confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0 and 1.0, got {confidence}"
            )
        
        # Create view vector: [0, ..., 1, ..., -1, ..., 0]
        # with 1 at asset1_index and -1 at asset2_index
        view_vector = np.zeros(self.n_assets)
        view_vector[asset1_index] = 1.0
        view_vector[asset2_index] = -1.0
        
        # Calculate view uncertainty
        uncertainty = (1 - confidence) * 0.01
        
        # Add view
        direction = "outperform" if outperformance > 0 else "underperform"
        self._add_view(
            view_vector,
            outperformance,
            uncertainty,
            f"Asset {asset1_index} will {direction} asset {asset2_index} "
            f"by {abs(outperformance)*100:.2f}% (confidence: {confidence*100:.0f}%)"
        )
    
    def add_custom_view(
        self,
        view_vector: np.ndarray,
        expected_return: float,
        uncertainty: float,
        description: str = ""
    ) -> None:
        """Add a custom view with arbitrary portfolio weights.
        
        Args:
            view_vector: Portfolio weights for the view (length n_assets)
            expected_return: Expected return for this view portfolio
            uncertainty: Variance of the view (higher = less confident)
            description: Textual description of the view
        
        Raises:
            ValueError: If inputs are invalid
        
        Example:
            >>> views = InvestorViews(n_assets=3)
            >>> # "50% asset 0 + 50% asset 1 will return 8%"
            >>> views.add_custom_view(
            ...     np.array([0.5, 0.5, 0.0]),
            ...     0.08,
            ...     0.001,
            ...     "Equal-weight portfolio of assets 0 and 1 will return 8%"
            ... )
        """
        if not isinstance(view_vector, np.ndarray):
            raise TypeError(f"view_vector must be numpy array, got {type(view_vector)}")
        
        if view_vector.shape != (self.n_assets,):
            raise ValueError(
                f"view_vector must have shape ({self.n_assets},), got {view_vector.shape}"
            )
        
        if np.all(view_vector == 0):
            raise ValueError("view_vector cannot be all zeros")
        
        self._add_view(view_vector, expected_return, uncertainty, description)
    
    def _add_view(
        self,
        view_vector: np.ndarray,
        expected_return: float,
        uncertainty: float,
        description: str
    ) -> None:
        """Internal method to add a view to the system.
        
        Args:
            view_vector: View portfolio weights
            expected_return: Expected return for the view
            uncertainty: View uncertainty (variance)
            description: Text description
        """
        # Add view vector as new row in P matrix
        if self.view_matrix.size == 0:
            self.view_matrix = view_vector.reshape(1, -1)
        else:
            self.view_matrix = np.vstack([self.view_matrix, view_vector])
        
        # Add expected return to Q vector
        self.view_returns = np.append(self.view_returns, expected_return)
        
        # Add uncertainty to diagonal of Ω matrix
        if self.view_uncertainty.size == 0:
            self.view_uncertainty = np.array([[uncertainty]])
        else:
            # Expand uncertainty matrix
            n = len(self.view_uncertainty)
            new_uncertainty = np.zeros((n + 1, n + 1))
            new_uncertainty[:n, :n] = self.view_uncertainty
            new_uncertainty[n, n] = uncertainty
            self.view_uncertainty = new_uncertainty
        
        # Add description
        self.view_descriptions.append(description)
    
    def remove_view(self, index: int) -> None:
        """Remove a view by index.
        
        Args:
            index: Index of the view to remove (0-based)
        
        Raises:
            ValueError: If index is invalid
        """
        if not isinstance(index, int):
            raise TypeError(f"index must be int, got {type(index)}")
        
        if not 0 <= index < self.n_views:
            raise ValueError(
                f"index must be between 0 and {self.n_views-1}, got {index}"
            )
        
        # Remove from all structures
        self.view_matrix = np.delete(self.view_matrix, index, axis=0)
        self.view_returns = np.delete(self.view_returns, index)
        
        # Remove row and column from uncertainty matrix
        mask = np.ones(self.n_views, dtype=bool)
        mask[index] = False
        self.view_uncertainty = self.view_uncertainty[mask][:, mask]
        
        # Remove description
        del self.view_descriptions[index]
    
    def clear_views(self) -> None:
        """Remove all views."""
        self.view_matrix = np.empty((0, self.n_assets))
        self.view_returns = np.array([])
        self.view_uncertainty = np.empty((0, 0))
        self.view_descriptions = []
    
    def get_view_summary(self) -> List[Dict[str, any]]:
        """Get summary of all views.
        
        Returns:
            List of dictionaries with view information
        """
        summaries = []
        for i in range(self.n_views):
            summaries.append({
                'index': i,
                'description': self.view_descriptions[i],
                'expected_return': float(self.view_returns[i]),
                'uncertainty': float(self.view_uncertainty[i, i]),
                'confidence': 1.0 - (self.view_uncertainty[i, i] / 0.01),  # Reverse calculation
                'weights': self.view_matrix[i].copy()
            })
        return summaries
    
    def validate(self) -> None:
        """Validate consistency of view structures.
        
        Raises:
            ValueError: If views are inconsistent
        """
        if self.n_views == 0:
            return  # Empty views are valid
        
        # Check dimensions
        if self.view_matrix.shape != (self.n_views, self.n_assets):
            raise ValueError(
                f"view_matrix has shape {self.view_matrix.shape}, "
                f"expected ({self.n_views}, {self.n_assets})"
            )
        
        if len(self.view_returns) != self.n_views:
            raise ValueError(
                f"view_returns has length {len(self.view_returns)}, "
                f"expected {self.n_views}"
            )
        
        if self.view_uncertainty.shape != (self.n_views, self.n_views):
            raise ValueError(
                f"view_uncertainty has shape {self.view_uncertainty.shape}, "
                f"expected ({self.n_views}, {self.n_views})"
            )
        
        if len(self.view_descriptions) != self.n_views:
            raise ValueError(
                f"view_descriptions has length {len(self.view_descriptions)}, "
                f"expected {self.n_views}"
            )
        
        # Check uncertainty matrix is positive definite
        eigenvalues = np.linalg.eigvalsh(self.view_uncertainty)
        if np.any(eigenvalues <= 0):
            raise ValueError("view_uncertainty matrix is not positive definite")
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"InvestorViews(n_assets={self.n_assets}, n_views={self.n_views})\n"
            + "\n".join(f"  {i+1}. {desc}" for i, desc in enumerate(self.view_descriptions))
        )
