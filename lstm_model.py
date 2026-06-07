"""
LSTM-based forecasting model for stock prices and returns.

Implements Long Short-Term Memory neural network for time series prediction.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class LSTMForecaster:
    """
    LSTM neural network for time series forecasting.
    
    Uses sliding window approach to predict future values based on
    historical patterns.
    """
    
    def __init__(self, lookback=60, epochs=50, batch_size=32):
        """
        Initialize LSTM forecaster.
        
        Args:
            lookback: Number of historical time steps to use for prediction
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def _create_model(self, input_shape):
        """Create LSTM model architecture (optimized for speed)."""
        model = Sequential([
            LSTM(units=32, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=32),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        # Use Adam with higher learning rate for faster convergence
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def _create_sequences(self, data):
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def fit(self, data):
        """
        Train the LSTM model.
        
        Args:
            data: Pandas Series or numpy array of historical values
            
        Returns:
            Training history
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(-1, 1)
            
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Create and train model
        self.model = self._create_model((X.shape[1], 1))
        
        # Early stopping with reduced patience for faster training
        early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, min_delta=0.0001)
        
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[early_stop],
            validation_split=0.1  # Use validation for better early stopping
        )
        
        return self.history
    
    def predict(self, data, steps):
        """
        Forecast future values.
        
        Args:
            data: Historical data to base predictions on
            steps: Number of steps ahead to forecast
            
        Returns:
            Array of predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values.reshape(-1, 1)
        elif isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(-1, 1)
            
        # Scale data
        scaled_data = self.scaler.transform(data)
        
        # Get last lookback values
        current_batch = scaled_data[-self.lookback:].reshape(1, self.lookback, 1)
        
        # Generate predictions
        predictions = []
        for i in range(steps):
            # Predict next value
            pred = self.model.predict(current_batch, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update batch with prediction
            current_batch = np.append(current_batch[:, 1:, :], [[[ pred]]], axis=1)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def evaluate(self, actual, predicted):
        """
        Calculate performance metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2
        }
