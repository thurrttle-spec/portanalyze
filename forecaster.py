"""
LSTM Forecaster for time-series prediction.

This module provides the LSTMForecaster class for training and forecasting
with LSTM neural networks.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    callbacks = None
    warnings.warn(
        "TensorFlow not available. LSTM forecasting will not work. "
        "Install with: pip install tensorflow"
    )

from .sequence import prepare_sequences, create_train_test_split


class LSTMForecaster:
    """LSTM-based time-series forecaster.
    
    Attributes:
        lookback: Number of past timesteps to use as input
        hidden_units: Number of LSTM units in first layer
        dropout_rate: Dropout rate for regularization
        model: Compiled Keras model (None until built)
        history: Training history (None until trained)
        mean: Training data mean (for normalization)
        std: Training data std (for normalization)
    """
    
    def __init__(
        self,
        lookback: int = 60,
        hidden_units: int = 128,
        dropout_rate: float = 0.2
    ):
        """Initialize LSTM forecaster.
        
        Args:
            lookback: Number of past timesteps (default 60 days)
            hidden_units: LSTM units in first layer (default 128)
            dropout_rate: Dropout for regularization (default 0.2)
            
        Raises:
            ImportError: If TensorFlow is not available
            ValueError: If parameters are invalid
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for LSTM forecasting. "
                "Install with: pip install tensorflow"
            )
        
        if not isinstance(lookback, int) or lookback <= 0:
            raise ValueError(f"lookback must be positive int, got {lookback}")
        
        if not isinstance(hidden_units, int) or hidden_units <= 0:
            raise ValueError(f"hidden_units must be positive int, got {hidden_units}")
        
        if not isinstance(dropout_rate, (int, float)):
            raise TypeError(f"dropout_rate must be numeric, got {type(dropout_rate)}")
        
        if not 0 <= dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        self.lookback = lookback
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.history = None
        self.mean = 0.0
        self.std = 1.0
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM neural network architecture.
        
        Architecture:
            - LSTM layer 1: hidden_units neurons, return sequences
            - Dropout: dropout_rate
            - LSTM layer 2: hidden_units // 2 neurons
            - Dropout: dropout_rate
            - Dense output: 1 neuron (predicted return)
        
        Args:
            input_shape: (lookback, n_features) typically (60, 1)
            
        Returns:
            Compiled Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required but not installed")
        
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(
                self.hidden_units,
                return_sequences=True,
                input_shape=input_shape,
                name='lstm_1'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_1'),
            
            # Second LSTM layer
            layers.LSTM(
                self.hidden_units // 2,
                return_sequences=False,
                name='lstm_2'
            ),
            layers.Dropout(self.dropout_rate, name='dropout_2'),
            
            # Output layer
            layers.Dense(1, name='output')
        ])
        
        # Compile with Adam optimizer and MSE loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(
        self,
        returns: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: int = 0
    ) -> Dict:
        """Train LSTM model on historical returns.
        
        Args:
            returns: Time-series of returns (1D array)
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data for validation
            early_stopping_patience: Epochs to wait before early stopping
            verbose: Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch)
            
        Returns:
            Dictionary with training history and metrics
            
        Raises:
            ValueError: If returns are invalid or insufficient
        """
        # Prepare sequences
        X, y = prepare_sequences(returns, self.lookback)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = create_train_test_split(
            X, y, train_ratio=0.8
        )
        
        if len(X_train) < 10:
            raise ValueError(
                f"Insufficient training data: {len(X_train)} samples. "
                f"Need at least 10 samples."
            )
        
        # Build model if not already built
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape)
        
        # Setup callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=verbose
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        # Train model
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callback_list,
                verbose=verbose
            )
            
            self.history = history.history
            
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}") from e
        
        # Evaluate on test set
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        train_metrics = {
            'final_train_loss': float(self.history['loss'][-1]),
            'final_val_loss': float(self.history['val_loss'][-1]),
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'epochs_trained': len(self.history['loss']),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        return train_metrics
    
    def predict(
        self,
        last_sequence: np.ndarray,
        steps_ahead: int = 1
    ) -> np.ndarray:
        """Generate multi-step ahead forecasts.
        
        Args:
            last_sequence: Last lookback values (lookback,) or (lookback, 1)
            steps_ahead: Number of steps to forecast
            
        Returns:
            Forecasted returns (steps_ahead,)
            
        Raises:
            ValueError: If model not trained or inputs invalid
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if not isinstance(last_sequence, np.ndarray):
            raise TypeError(f"last_sequence must be numpy array, got {type(last_sequence)}")
        
        # Reshape if needed
        if last_sequence.ndim == 1:
            last_sequence = last_sequence.reshape(-1, 1)
        
        if last_sequence.shape[0] != self.lookback:
            raise ValueError(
                f"last_sequence must have {self.lookback} timesteps, "
                f"got {last_sequence.shape[0]}"
            )
        
        if not isinstance(steps_ahead, int) or steps_ahead <= 0:
            raise ValueError(f"steps_ahead must be positive int, got {steps_ahead}")
        
        # Rolling forecast
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps_ahead):
            # Reshape for prediction: (1, lookback, 1)
            X_input = current_sequence.reshape(1, self.lookback, 1)
            
            # Predict next value
            pred = self.model.predict(X_input, verbose=0)[0, 0]
            forecasts.append(pred)
            
            # Update sequence: remove first value, append prediction
            current_sequence = np.vstack([current_sequence[1:], [[pred]]])
        
        return np.array(forecasts)
    
    def evaluate(
        self,
        returns: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Evaluate model performance on historical data.
        
        Args:
            returns: Time-series of returns
            metrics: List of metrics to calculate ['mae', 'rmse', 'mape']
            
        Returns:
            Dictionary with metric values
        """
        if metrics is None:
            metrics = ['mae', 'rmse']
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare sequences
        X, y = prepare_sequences(returns, self.lookback)
        
        # Make predictions
        y_pred = self.model.predict(X, verbose=0).flatten()
        
        # Calculate metrics
        results = {}
        
        if 'mae' in metrics:
            results['mae'] = float(np.mean(np.abs(y - y_pred)))
        
        if 'rmse' in metrics:
            results['rmse'] = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        
        if 'mape' in metrics:
            # Avoid division by zero
            mask = y != 0
            if np.any(mask):
                mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100
                results['mape'] = float(mape)
            else:
                results['mape'] = float('inf')
        
        if 'r2' in metrics:
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            results['r2'] = float(r2)
        
        return results
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file.
        
        Args:
            filepath: Path to save model (e.g., 'models/lstm_model.h5')
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        self.model.save(filepath)
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file.
        
        Args:
            filepath: Path to saved model
        """
        self.model = keras.models.load_model(filepath)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet"
        
        # Capture summary as string
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return '\n'.join(summary_lines)
