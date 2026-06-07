"""
GARCH-based volatility forecasting model.

Implements Generalized AutoRegressive Conditional Heteroskedasticity
for volatility prediction.
"""

import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


class GARCHForecaster:
    """
    GARCH model for volatility forecasting.
    
    Models time-varying volatility (conditional heteroskedasticity)
    in financial returns.
    """
    
    def __init__(self, p=1, q=1, mean='Constant', vol='GARCH', dist='normal'):
        """
        Initialize GARCH forecaster.
        
        Args:
            p: Order of GARCH terms (default: 1)
            q: Order of ARCH terms (default: 1)
            mean: Mean model specification ('Constant', 'Zero', 'AR')
            vol: Volatility model ('GARCH', 'EGARCH', 'GJR-GARCH')
            dist: Error distribution ('normal', 't', 'skewt')
        """
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.model = None
        self.fitted_model = None
        
    def fit(self, returns):
        """
        Fit the GARCH model.
        
        Args:
            returns: Pandas Series or numpy array of returns (not prices!)
            
        Returns:
            Fitted model
        """
        # Convert to pandas Series if needed
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        # Remove any NaN or infinite values
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Scale returns to percentage for better convergence
        returns_pct = returns * 100
        
        # Create GARCH model
        self.model = arch_model(
            returns_pct,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist
        )
        
        # Fit model
        self.fitted_model = self.model.fit(disp='off')
        
        return self.fitted_model
    
    def predict(self, horizon):
        """
        Forecast future volatility.
        
        Args:
            horizon: Number of periods ahead to forecast
            
        Returns:
            Dictionary with forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate forecast
        forecast = self.fitted_model.forecast(horizon=horizon)
        
        # Extract forecasted variance and convert back from percentage scale
        variance_forecast = forecast.variance.values[-1, :] / 10000  # Convert back from %²
        
        # Convert variance to volatility (standard deviation)
        volatility_forecast = np.sqrt(variance_forecast)
        
        # Get mean forecast
        mean_forecast = forecast.mean.values[-1, :]
        
        return {
            'mean': mean_forecast / 100,  # Convert back from percentage
            'variance': variance_forecast,
            'volatility': volatility_forecast,
            'annualized_volatility': volatility_forecast * np.sqrt(252)
        }
    
    def conditional_volatility(self):
        """
        Get conditional volatility from fitted model.
        
        Returns:
            Array of conditional volatility values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting conditional volatility")
        
        # Get conditional volatility from fitted model
        cond_vol = self.fitted_model.conditional_volatility / 100  # Convert from percentage
        
        return cond_vol.values
    
    def standardized_residuals(self):
        """
        Get standardized residuals for diagnostic checking.
        
        Returns:
            Array of standardized residuals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting residuals")
        
        return self.fitted_model.std_resid.values
    
    def evaluate(self, actual_volatility, predicted_volatility):
        """
        Calculate performance metrics.
        
        Args:
            actual_volatility: Actual volatility values
            predicted_volatility: Predicted volatility values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure same length
        min_len = min(len(actual_volatility), len(predicted_volatility))
        actual = actual_volatility[-min_len:]
        predicted = predicted_volatility[-min_len:]
        
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
            'R²': r2
        }
    
    def get_model_summary(self):
        """Get model summary statistics."""
        if self.fitted_model is None:
            return None
        
        return {
            'model_type': f'{self.vol}({self.p},{self.q})',
            'mean_model': self.mean,
            'distribution': self.dist,
            'log_likelihood': self.fitted_model.loglikelihood,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'params': self.fitted_model.params.to_dict(),
            'summary': str(self.fitted_model.summary())
        }
    
    def forecast_returns_with_volatility(self, mean_return, horizon):
        """
        Forecast returns incorporating volatility forecast.
        
        Args:
            mean_return: Expected mean return
            horizon: Forecast horizon
            
        Returns:
            Dictionary with return forecasts and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        # Get volatility forecast
        vol_forecast = self.predict(horizon)
        volatility = vol_forecast['volatility']
        
        # Generate return scenarios
        # Mean forecast
        mean_forecast = np.full(horizon, mean_return)
        
        # Confidence intervals (assuming normal distribution)
        lower_95 = mean_forecast - 1.96 * volatility
        upper_95 = mean_forecast + 1.96 * volatility
        lower_68 = mean_forecast - volatility
        upper_68 = mean_forecast + volatility
        
        return {
            'mean_return': mean_forecast,
            'volatility': volatility,
            'lower_95': lower_95,
            'upper_95': upper_95,
            'lower_68': lower_68,
            'upper_68': upper_68
        }
    
    def plot_diagnostics(self):
        """
        Generate diagnostic plots for model validation.
        
        Returns:
            Matplotlib figure with diagnostic plots
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before plotting")
        
        return self.fitted_model.plot(annualize='D')
