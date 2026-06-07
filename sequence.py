"""
Sequence preparation functions for LSTM training.

This module provides utilities for preparing time-series data into sequences
suitable for LSTM neural network training.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def prepare_sequences(
    returns: np.ndarray,
    lookback: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training from time-series returns.
    
    For each time step t, creates:
        X[t] = [returns[t-lookback], ..., returns[t-1]]
        y[t] = returns[t]
    
    This ensures no data leakage - X[i] strictly precedes y[i].
    
    Args:
        returns: Time-series of returns (T x 1 or T,)
        lookback: Number of past time steps to use as input
        
    Returns:
        Tuple of (X, y) where:
            - X has shape (T-lookback, lookback, 1)
            - y has shape (T-lookback,)
    
    Raises:
        ValueError: If returns is too short or contains invalid values
    
    Example:
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
        >>> X, y = prepare_sequences(returns, lookback=2)
        >>> X.shape
        (3, 2, 1)
        >>> y.shape
        (3,)
        >>> X[0]  # First sequence: returns[0:2]
        array([[0.01],
               [0.02]])
        >>> y[0]  # First target: returns[2]
        -0.01
    """
    # Validate inputs
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"returns must be numpy array, got {type(returns)}")
    
    # Flatten if needed
    if returns.ndim > 1:
        returns = returns.flatten()
    
    if returns.ndim != 1:
        raise ValueError(f"returns must be 1-dimensional, got shape {returns.shape}")
    
    T = len(returns)
    
    if T == 0:
        raise ValueError("returns array is empty")
    
    if not isinstance(lookback, int):
        raise TypeError(f"lookback must be int, got {type(lookback)}")
    
    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")
    
    if lookback >= T:
        raise ValueError(
            f"lookback ({lookback}) must be less than returns length ({T})"
        )
    
    # Check for NaN or inf
    if np.any(np.isnan(returns)):
        raise ValueError("returns contains NaN values")
    
    if np.any(np.isinf(returns)):
        raise ValueError("returns contains infinite values")
    
    # Create sequences
    X = []
    y = []
    
    for i in range(lookback, T):
        # Extract lookback window: returns[i-lookback:i]
        sequence = returns[i - lookback:i]
        X.append(sequence)
        
        # Target is next value: returns[i]
        y.append(returns[i])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for LSTM input: (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Validate output
    expected_samples = T - lookback
    assert X.shape[0] == expected_samples, f"X samples {X.shape[0]} != expected {expected_samples}"
    assert X.shape[1] == lookback, f"X timesteps {X.shape[1]} != lookback {lookback}"
    assert X.shape[2] == 1, f"X features {X.shape[2]} != 1"
    assert y.shape[0] == expected_samples, f"y samples {y.shape[0]} != expected {expected_samples}"
    
    return X, y


def create_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into training and test sets.
    
    Uses time-based split (not random) to preserve temporal order.
    First train_ratio of data for training, rest for testing.
    
    Args:
        X: Input sequences (n_samples, lookback, n_features)
        y: Target values (n_samples,)
        train_ratio: Fraction of data for training (default 0.8)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    
    Raises:
        ValueError: If inputs are invalid
    
    Example:
        >>> X = np.random.randn(100, 60, 1)
        >>> y = np.random.randn(100)
        >>> X_train, X_test, y_train, y_test = create_train_test_split(X, y, 0.8)
        >>> X_train.shape
        (80, 60, 1)
        >>> X_test.shape
        (20, 60, 1)
    """
    # Validate inputs
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be numpy array, got {type(X)}")
    
    if not isinstance(y, np.ndarray):
        raise TypeError(f"y must be numpy array, got {type(y)}")
    
    if X.ndim != 3:
        raise ValueError(f"X must be 3-dimensional, got shape {X.shape}")
    
    if y.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
    
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")
    
    if not isinstance(train_ratio, (int, float)):
        raise TypeError(f"train_ratio must be numeric, got {type(train_ratio)}")
    
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    n_samples = len(X)
    
    if n_samples == 0:
        raise ValueError("X and y are empty")
    
    # Calculate split point
    split_idx = int(n_samples * train_ratio)
    
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError(
            f"Invalid split_idx {split_idx} for {n_samples} samples with ratio {train_ratio}"
        )
    
    # Time-based split (preserve order)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    # Validate split
    assert len(X_train) + len(X_test) == n_samples
    assert len(y_train) + len(y_test) == n_samples
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    return X_train, X_test, y_train, y_test


def validate_no_data_leakage(
    X: np.ndarray,
    y: np.ndarray,
    original_returns: np.ndarray,
    lookback: int
) -> bool:
    """Validate that sequences have no data leakage.
    
    Checks that X[i] contains data strictly before y[i] in the original series.
    
    Args:
        X: Input sequences (n_samples, lookback, 1)
        y: Target values (n_samples,)
        original_returns: Original return series
        lookback: Lookback window size
        
    Returns:
        True if no data leakage detected
    
    Raises:
        ValueError: If data leakage is detected
    """
    n_samples = len(X)
    
    for i in range(n_samples):
        # Check that X[i] comes from returns[i:i+lookback]
        # and y[i] comes from returns[i+lookback]
        sequence = X[i, :, 0]
        target = y[i]
        
        # Get corresponding slice from original
        expected_sequence = original_returns[i:i+lookback]
        expected_target = original_returns[i+lookback]
        
        # Verify no leakage
        if not np.allclose(sequence, expected_sequence, rtol=1e-10):
            raise ValueError(f"Data leakage detected in sample {i}: sequence mismatch")
        
        if not np.isclose(target, expected_target, rtol=1e-10):
            raise ValueError(f"Data leakage detected in sample {i}: target mismatch")
    
    return True


def normalize_sequences(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Normalize sequences using training set statistics.
    
    Uses mean and std from training set to normalize both train and test.
    This prevents data leakage from test set statistics.
    
    Args:
        X_train: Training input sequences
        X_test: Test input sequences
        y_train: Training targets
        y_test: Test targets
        
    Returns:
        Tuple of (X_train_norm, X_test_norm, y_train_norm, y_test_norm, mean, std)
    
    Note:
        Returns are typically already centered around 0, so normalization
        may not be necessary. This function is provided for completeness.
    """
    # Calculate statistics from training data only
    mean = np.mean(X_train)
    std = np.std(X_train)
    
    if std < 1e-10:
        # Handle constant series
        std = 1.0
    
    # Normalize using training statistics
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    y_train_norm = (y_train - mean) / std
    y_test_norm = (y_test - mean) / std
    
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, mean, std


def denormalize_predictions(
    predictions: np.ndarray,
    mean: float,
    std: float
) -> np.ndarray:
    """Denormalize predictions back to original scale.
    
    Args:
        predictions: Normalized predictions
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized predictions
    """
    return predictions * std + mean
