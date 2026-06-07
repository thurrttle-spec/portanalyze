"""
ARIMA-based forecasting model for stock prices and returns.

Implements AutoRegressive Integrated Moving Average for time series prediction.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """
    ARIMA model for time series forecasting.
    
    Automatically selects optimal (p, d, q) parameters or uses
    user-specified parameters.
    """
    
    def __init__(self, order=None, auto_order=True):
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: Tuple (p, d, q) for ARIMA model. If None, will auto-select
            auto_order: If True, automatically determine best order
        """
        self.order = order
        self.auto_order = auto_order
        self.model = None
        self.fitted_model = None
        self.best_order = None
        
    def _check_stationarity(self, data):
        """
        Check if data is stationary using ADF test.
        
        Returns:
            Tuple (is_stationary, d_value)
        """
        result = adfuller(data.dropna())
        p_value = result[1]
        
        # If p-value < 0.05, data is stationary
        if p_value < 0.05:
            return True, 0
        
        # Try first difference
        diff_data = data.diff().dropna()
        result = adfuller(diff_data)
        if result[1] < 0.05:
            return True, 1
        
        # Try second difference
        diff2_data = diff_data.diff().dropna()
        result = adfuller(diff2_data)
        if result[1] < 0.05:
            return True, 2
        
        return False, 2
    
    def _auto_arima(self, data, max_p=5, max_q=5):
        """
        Automatically select best ARIMA order using AIC criterion.
        
        Args:
            data: Time series data
            max_p: Maximum p value to test
            max_q: Maximum q value to test
            
        Returns:
            Best order tuple (p, d, q)
        """
        # Determine d
        _, d = self._check_stationarity(data)
        
        best_aic = np.inf
        best_order = None
        
        # Grid search for best p and q
        for p in range(0, max_p + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except:
                    continue
        
        if best_order is None:
            # Fallback to ARIMA(1,1,1)
            best_order = (1, 1, 1)
        
        return best_order
    
    def fit(self, data):
        """
        Fit the ARIMA model.
        
        Args:
            data: Pandas Series or numpy array of historical values
            
        Returns:
            Fitted model
        """
        # Convert to pandas Series if needed
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        # Remove any NaN values
        data = data.dropna()
        
        # Determine order
        if self.auto_order or self.order is None:
            self.best_order = self._auto_arima(data)
        else:
            self.best_order = self.order
        
        # Fit model
        self.model = ARIMA(data, order=self.best_order)
        self.fitted_model = self.model.fit()
        
        return self.fitted_model
    
    def predict(self, steps):
        """
        Forecast future values.
        
        Args:
            steps: Number of steps ahead to forecast
            
        Returns:
            Tuple (predictions, confidence_intervals)
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate forecast
        forecast = self.fitted_model.forecast(steps=steps)
        
        # Get confidence intervals
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        conf_int = forecast_result.conf_int()
        
        return forecast.values, conf_int.values
    
    def predict_in_sample(self):
        """
        Get in-sample predictions for validation.
        
        Returns:
            Array of fitted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self.fitted_model.fittedvalues.values
    
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
        actual = actual[-min_len:]
        predicted = predicted[-min_len:]
        
        # Calculate metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.all(actual != 0) else np.inf
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'AIC': self.fitted_model.aic if self.fitted_model else None,
            'BIC': self.fitted_model.bic if self.fitted_model else None
        }
    
    def get_model_summary(self):
        """Get model summary statistics."""
        if self.fitted_model is None:
            return None
        
        return {
            'order': self.best_order,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'params': self.fitted_model.params.to_dict(),
            'summary': str(self.fitted_model.summary())
        }
    
    def plot_diagnostics(self):
        """
        Generate diagnostic plots for model validation.
        
        Returns:
            Matplotlib figure with diagnostic plots
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before plotting")
        
        return self.fitted_model.plot_diagnostics(figsize=(12, 8))
