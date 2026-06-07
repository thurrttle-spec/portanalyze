"""
Data validation module for forecasting models.

Checks if data meets requirements and assumptions for LSTM, ARIMA, and GARCH models.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf
from scipy import stats
from typing import Dict, Tuple, List

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ForecastingDataValidator:
    """
    Validates time series data for forecasting model suitability.
    
    Checks assumptions and requirements for:
    - LSTM: Data scale, length, patterns
    - ARIMA: Stationarity, autocorrelation
    - GARCH: Returns distribution, volatility clustering
    """
    
    def __init__(self, data: pd.Series, data_type: str = 'price'):
        """
        Initialize validator.
        
        Args:
            data: Time series data (prices or returns)
            data_type: 'price' or 'return'
        """
        self.data = data.dropna()
        self.data_type = data_type
        
        # Calculate returns if data is prices
        if data_type == 'price':
            self.returns = self.data.pct_change().dropna()
        else:
            self.returns = self.data
            
        self.validation_results = {}
        
    def validate_all(self) -> Dict:
        """
        Run all validation checks.
        
        Returns:
            Dictionary with validation results for all models
        """
        results = {
            'data_quality': self._check_data_quality(),
            'lstm': self._validate_lstm(),
            'arima': self._validate_arima(),
            'garch': self._validate_garch(),
            'general': self._general_checks()
        }
        
        self.validation_results = results
        return results
    
    def _check_data_quality(self) -> Dict:
        """Check basic data quality issues."""
        issues = []
        warnings = []
        
        # Check for missing values
        n_missing = self.data.isna().sum()
        if n_missing > 0:
            issues.append(f"{n_missing} missing values detected")
        
        # Check for zeros in price data
        if self.data_type == 'price':
            n_zeros = (self.data == 0).sum()
            if n_zeros > 0:
                issues.append(f"{n_zeros} zero values in price data")
        
        # Check for outliers (> 3 std deviations)
        z_scores = np.abs(stats.zscore(self.returns))
        n_outliers = (z_scores > 3).sum()
        if n_outliers > len(self.returns) * 0.05:  # > 5% outliers
            warnings.append(f"{n_outliers} potential outliers detected (Z-score > 3)")
        
        # Check for constant values
        if self.data.std() == 0:
            issues.append("Data is constant (no variation)")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'n_missing': int(n_missing),
            'n_outliers': int(n_outliers),
            'n_observations': len(self.data)
        }
    
    def _validate_lstm(self) -> Dict:
        """Validate data for LSTM model."""
        issues = []
        warnings = []
        recommendations = []
        
        # Minimum data length check
        min_length = 100  # Reasonable minimum for LSTM
        if len(self.data) < min_length:
            issues.append(f"Insufficient data: {len(self.data)} observations (minimum: {min_length})")
        elif len(self.data) < 200:
            warnings.append(f"Limited data: {len(self.data)} observations (recommended: 200+)")
        
        # Check for trends/patterns
        # Autocorrelation check
        acf_vals = acf(self.data.values, nlags=min(40, len(self.data)//4), fft=False)
        significant_lags = np.sum(np.abs(acf_vals[1:]) > 1.96/np.sqrt(len(self.data)))
        
        if significant_lags < 3:
            warnings.append("Low autocorrelation detected - data may be too random for LSTM")
        
        # Check volatility
        volatility = self.returns.std()
        if volatility < 0.001:
            warnings.append("Very low volatility - model may have difficulty learning patterns")
        elif volatility > 0.10:
            warnings.append("Very high volatility - consider preprocessing or shorter lookback")
        
        # Recommendations
        if len(self.data) >= 200:
            recommendations.append("✓ Sufficient data length for LSTM")
        
        if significant_lags >= 5:
            recommendations.append("✓ Good autocorrelation structure for pattern learning")
        
        # Suggest lookback window
        suggested_lookback = min(60, len(self.data) // 5)
        recommendations.append(f"Suggested lookback window: {suggested_lookback} days")
        
        return {
            'suitable': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'data_length': len(self.data),
            'min_required': min_length,
            'autocorrelation_lags': int(significant_lags),
            'volatility': float(volatility)
        }
    
    def _validate_arima(self) -> Dict:
        """Validate data for ARIMA model."""
        issues = []
        warnings = []
        recommendations = []
        
        # Minimum data length
        min_length = 50
        if len(self.data) < min_length:
            issues.append(f"Insufficient data: {len(self.data)} observations (minimum: {min_length})")
        
        # Stationarity test (ADF test)
        try:
            adf_result = adfuller(self.data.dropna())
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
            is_stationary = adf_pvalue < 0.05
            
            if not is_stationary:
                warnings.append(f"Data is non-stationary (p-value: {adf_pvalue:.4f})")
                recommendations.append("ARIMA will automatically difference the data")
            else:
                recommendations.append("✓ Data is stationary (differencing may not be needed)")
            
            # Test first difference if non-stationary
            if not is_stationary:
                diff_data = self.data.diff().dropna()
                adf_diff = adfuller(diff_data)
                if adf_diff[1] < 0.05:
                    recommendations.append("✓ First differencing makes data stationary")
                    d_order = 1
                else:
                    recommendations.append("Second differencing may be needed")
                    d_order = 2
            else:
                d_order = 0
                
        except Exception as e:
            warnings.append(f"Could not perform stationarity test: {str(e)}")
            adf_pvalue = None
            is_stationary = None
            d_order = 1
        
        # Check autocorrelation
        acf_vals = acf(self.data.values, nlags=min(40, len(self.data)//4), fft=False)
        significant_lags = np.sum(np.abs(acf_vals[1:]) > 1.96/np.sqrt(len(self.data)))
        
        if significant_lags == 0:
            warnings.append("No significant autocorrelation - ARIMA may not be suitable")
        else:
            recommendations.append(f"✓ Significant autocorrelation detected at {significant_lags} lags")
        
        # Suggest ARIMA order
        suggested_p = min(5, significant_lags)
        suggested_q = min(5, significant_lags // 2)
        recommendations.append(f"Suggested starting order: ARIMA({suggested_p},{d_order},{suggested_q})")
        
        return {
            'suitable': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'data_length': len(self.data),
            'min_required': min_length,
            'is_stationary': is_stationary,
            'adf_pvalue': float(adf_pvalue) if adf_pvalue is not None else None,
            'suggested_d': d_order,
            'autocorrelation_lags': int(significant_lags)
        }
    
    def _validate_garch(self) -> Dict:
        """Validate data for GARCH model."""
        issues = []
        warnings = []
        recommendations = []
        
        # GARCH requires returns, not prices
        if self.data_type == 'price':
            recommendations.append("Note: GARCH models returns, not prices (will be converted)")
        
        # Minimum data length
        min_length = 100
        if len(self.returns) < min_length:
            issues.append(f"Insufficient returns data: {len(self.returns)} observations (minimum: {min_length})")
        
        # Check for volatility clustering (ARCH effects)
        # Test on squared returns
        returns_squared = self.returns ** 2
        acf_squared = acf(returns_squared.values, nlags=min(20, len(returns_squared)//4), fft=False)
        significant_lags_squared = np.sum(np.abs(acf_squared[1:]) > 1.96/np.sqrt(len(returns_squared)))
        
        if significant_lags_squared < 3:
            warnings.append("Weak volatility clustering - GARCH may not be necessary")
            recommendations.append("Consider simple volatility estimation instead of GARCH")
        else:
            recommendations.append(f"✓ Strong volatility clustering detected ({significant_lags_squared} significant lags)")
        
        # Check for heavy tails (excess kurtosis)
        kurtosis = stats.kurtosis(self.returns.dropna())
        if kurtosis > 3:
            recommendations.append(f"✓ Heavy tails detected (kurtosis: {kurtosis:.2f}) - GARCH is appropriate")
            recommendations.append("Consider using Student-t distribution instead of normal")
        elif kurtosis < 0:
            warnings.append(f"Light tails (kurtosis: {kurtosis:.2f}) - returns may be too uniform")
        
        # Check for asymmetry (leverage effect)
        skewness = stats.skew(self.returns.dropna())
        if abs(skewness) > 0.5:
            recommendations.append(f"Asymmetric returns (skewness: {skewness:.2f}) - consider EGARCH or GJR-GARCH")
        
        # Check mean return
        mean_return = self.returns.mean()
        if abs(mean_return) < self.returns.std() * 0.1:
            recommendations.append("Low mean return relative to volatility - focus on volatility forecasting")
        
        # Suggested GARCH order
        suggested_p = min(2, max(1, significant_lags_squared // 3))
        suggested_q = min(2, max(1, significant_lags_squared // 4))
        recommendations.append(f"Suggested starting order: GARCH({suggested_p},{suggested_q})")
        
        return {
            'suitable': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations,
            'returns_length': len(self.returns),
            'min_required': min_length,
            'volatility_clustering_lags': int(significant_lags_squared),
            'kurtosis': float(kurtosis),
            'skewness': float(skewness),
            'mean_return': float(mean_return),
            'volatility': float(self.returns.std())
        }
    
    def _general_checks(self) -> Dict:
        """General checks applicable to all models."""
        # Data frequency check
        if isinstance(self.data.index, pd.DatetimeIndex):
            freq = pd.infer_freq(self.data.index)
            time_gaps = self.data.index.to_series().diff()
            max_gap = time_gaps.max()
            
            has_gaps = len(time_gaps.unique()) > 2  # More than just first NaT and one frequency
        else:
            freq = "Unknown"
            has_gaps = False
            max_gap = None
        
        # Trend detection
        # Simple linear regression
        X = np.arange(len(self.data)).reshape(-1, 1)
        y = self.data.values
        
        # Calculate trend
        if SKLEARN_AVAILABLE:
            lr = LinearRegression()
            lr.fit(X, y)
            trend_slope = lr.coef_[0]
            trend_strength = lr.score(X, y)  # R²
        else:
            # Fallback: simple correlation-based trend
            trend_slope = np.corrcoef(X.flatten(), y)[0, 1] * (y.std() / X.std())
            trend_strength = np.corrcoef(X.flatten(), y)[0, 1] ** 2
        
        has_trend = abs(trend_slope) > 0 and trend_strength > 0.1
        
        return {
            'data_frequency': freq,
            'has_gaps': has_gaps,
            'max_gap': str(max_gap) if max_gap is not None else None,
            'has_trend': has_trend,
            'trend_slope': float(trend_slope),
            'trend_strength': float(trend_strength),
            'data_range': {
                'min': float(self.data.min()),
                'max': float(self.data.max()),
                'mean': float(self.data.mean()),
                'std': float(self.data.std())
            }
        }
    
    def get_summary_report(self) -> str:
        """
        Generate human-readable summary report.
        
        Returns:
            Formatted string report
        """
        if not self.validation_results:
            self.validate_all()
        
        report = []
        report.append("=" * 60)
        report.append("FORECASTING DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Data quality
        dq = self.validation_results['data_quality']
        report.append("DATA QUALITY:")
        report.append(f"  Observations: {dq['n_observations']}")
        report.append(f"  Missing values: {dq['n_missing']}")
        report.append(f"  Outliers: {dq['n_outliers']}")
        report.append(f"  Status: {'✓ PASSED' if dq['passed'] else '✗ FAILED'}")
        if dq['issues']:
            report.append("  Issues:")
            for issue in dq['issues']:
                report.append(f"    - {issue}")
        report.append("")
        
        # LSTM
        lstm = self.validation_results['lstm']
        report.append("LSTM MODEL:")
        report.append(f"  Suitable: {'✓ YES' if lstm['suitable'] else '✗ NO'}")
        report.append(f"  Data length: {lstm['data_length']} (min: {lstm['min_required']})")
        report.append(f"  Autocorrelation: {lstm['autocorrelation_lags']} significant lags")
        if lstm['issues']:
            report.append("  Issues:")
            for issue in lstm['issues']:
                report.append(f"    - {issue}")
        if lstm['recommendations']:
            report.append("  Recommendations:")
            for rec in lstm['recommendations']:
                report.append(f"    {rec}")
        report.append("")
        
        # ARIMA
        arima = self.validation_results['arima']
        report.append("ARIMA MODEL:")
        report.append(f"  Suitable: {'✓ YES' if arima['suitable'] else '✗ NO'}")
        report.append(f"  Data length: {arima['data_length']} (min: {arima['min_required']})")
        report.append(f"  Stationary: {'✓ YES' if arima['is_stationary'] else '✗ NO'}")
        report.append(f"  Autocorrelation: {arima['autocorrelation_lags']} significant lags")
        if arima['issues']:
            report.append("  Issues:")
            for issue in arima['issues']:
                report.append(f"    - {issue}")
        if arima['recommendations']:
            report.append("  Recommendations:")
            for rec in arima['recommendations']:
                report.append(f"    {rec}")
        report.append("")
        
        # GARCH
        garch = self.validation_results['garch']
        report.append("GARCH MODEL:")
        report.append(f"  Suitable: {'✓ YES' if garch['suitable'] else '✗ NO'}")
        report.append(f"  Returns length: {garch['returns_length']} (min: {garch['min_required']})")
        report.append(f"  Volatility clustering: {garch['volatility_clustering_lags']} significant lags")
        report.append(f"  Kurtosis: {garch['kurtosis']:.2f}")
        if garch['issues']:
            report.append("  Issues:")
            for issue in garch['issues']:
                report.append(f"    - {issue}")
        if garch['recommendations']:
            report.append("  Recommendations:")
            for rec in garch['recommendations']:
                report.append(f"    {rec}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def get_model_suitability_scores(self) -> Dict[str, float]:
        """
        Calculate suitability scores (0-100) for each model.
        
        Returns:
            Dictionary with scores for each model
        """
        if not self.validation_results:
            self.validate_all()
        
        scores = {}
        
        # LSTM score
        lstm = self.validation_results['lstm']
        lstm_score = 100
        if not lstm['suitable']:
            lstm_score = 0
        else:
            # Deduct points for warnings
            lstm_score -= len(lstm['warnings']) * 15
            # Bonus for good autocorrelation
            if lstm['autocorrelation_lags'] >= 5:
                lstm_score = min(100, lstm_score + 10)
            # Bonus for sufficient data
            if lstm['data_length'] >= 200:
                lstm_score = min(100, lstm_score + 10)
        
        scores['LSTM'] = max(0, lstm_score)
        
        # ARIMA score
        arima = self.validation_results['arima']
        arima_score = 100
        if not arima['suitable']:
            arima_score = 0
        else:
            # Deduct points for warnings
            arima_score -= len(arima['warnings']) * 15
            # Bonus for stationarity or easy differencing
            if arima['is_stationary'] or arima.get('suggested_d', 1) <= 1:
                arima_score = min(100, arima_score + 10)
            # Bonus for good autocorrelation
            if arima['autocorrelation_lags'] >= 3:
                arima_score = min(100, arima_score + 10)
        
        scores['ARIMA'] = max(0, arima_score)
        
        # GARCH score
        garch = self.validation_results['garch']
        garch_score = 100
        if not garch['suitable']:
            garch_score = 0
        else:
            # Deduct points for warnings
            garch_score -= len(garch['warnings']) * 15
            # Bonus for strong volatility clustering
            if garch['volatility_clustering_lags'] >= 5:
                garch_score = min(100, garch_score + 10)
            # Bonus for heavy tails
            if garch['kurtosis'] > 3:
                garch_score = min(100, garch_score + 10)
        
        scores['GARCH'] = max(0, garch_score)
        
        return scores
