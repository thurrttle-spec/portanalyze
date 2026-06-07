"""
CAPM Analyzer for stock analysis and valuation.

This module provides the CAPMAnalyzer class for performing comprehensive
Capital Asset Pricing Model analysis on stocks.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.data.models import StockData, MarketData
from .calculator import (
    calculate_beta,
    calculate_expected_return,
    calculate_alpha,
    annualize_return,
    classify_stock,
    get_investment_recommendation
)
from .results import CAPMResults


class CAPMAnalyzer:
    """Analyzer for CAPM-based stock valuation.
    
    Attributes:
        risk_free_rate: Annual risk-free rate (as decimal)
        threshold: Alpha threshold for classification (default 0.02)
        min_observations: Minimum observations required for beta calculation
    """
    
    def __init__(
        self,
        risk_free_rate: float,
        threshold: float = 0.02,
        min_observations: int = 30
    ):
        """Initialize CAPM analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (e.g., 0.0493 for 4.93%)
            threshold: Alpha threshold for stock classification (default 2%)
            min_observations: Minimum data points for beta calculation (default 30)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(risk_free_rate, (int, float)):
            raise TypeError(
                f"risk_free_rate must be numeric, got {type(risk_free_rate)}"
            )
        
        if not 0 <= risk_free_rate <= 0.20:
            raise ValueError(
                f"risk_free_rate must be between 0 and 0.20, got {risk_free_rate}"
            )
        
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be numeric, got {type(threshold)}")
        
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        
        if not isinstance(min_observations, int):
            raise TypeError(
                f"min_observations must be int, got {type(min_observations)}"
            )
        
        if min_observations < 2:
            raise ValueError(
                f"min_observations must be >= 2, got {min_observations}"
            )
        
        self.risk_free_rate = risk_free_rate
        self.threshold = threshold
        self.min_observations = min_observations
    
    def analyze_stock(
        self,
        stock_data: StockData,
        market_data: MarketData
    ) -> Dict:
        """Analyze a single stock using CAPM.
        
        Args:
            stock_data: Stock price and return data
            market_data: Market index data
            
        Returns:
            Dictionary with CAPM metrics for the stock
            
        Raises:
            ValueError: If data is invalid or insufficient
        """
        # Validate inputs
        if not isinstance(stock_data, StockData):
            raise TypeError(f"stock_data must be StockData, got {type(stock_data)}")
        
        if not isinstance(market_data, MarketData):
            raise TypeError(f"market_data must be MarketData, got {type(market_data)}")
        
        # Align data by date
        common_dates = stock_data.returns.index.intersection(market_data.returns.index)
        
        if len(common_dates) < self.min_observations:
            raise ValueError(
                f"Insufficient common dates for {stock_data.ticker}: "
                f"{len(common_dates)} < {self.min_observations}"
            )
        
        stock_returns = stock_data.returns.loc[common_dates]
        market_returns = market_data.returns.loc[common_dates]
        
        # Calculate beta and R-squared
        beta, r_squared = calculate_beta(
            stock_returns,
            market_returns,
            self.min_observations
        )
        
        # Calculate market return (annualized)
        market_return = annualize_return(market_returns)
        
        # Calculate expected return from CAPM
        expected_return = calculate_expected_return(
            self.risk_free_rate,
            beta,
            market_return
        )
        
        # Calculate actual return (annualized)
        actual_return = annualize_return(stock_returns)
        
        # Calculate alpha
        alpha = calculate_alpha(actual_return, expected_return)
        
        # Classify stock
        classification = classify_stock(alpha, self.threshold)
        
        # Get recommendation
        recommendation = get_investment_recommendation(classification)
        
        return {
            'ticker': stock_data.ticker,
            'beta': beta,
            'expected_return': expected_return,
            'actual_return': actual_return,
            'alpha': alpha,
            'r_squared': r_squared,
            'classification': classification,
            'recommendation': recommendation,
            'num_observations': len(common_dates)
        }
    
    def analyze_stocks(
        self,
        stock_data_dict: Dict[str, StockData],
        market_data: MarketData
    ) -> CAPMResults:
        """Perform CAPM analysis on multiple stocks.
        
        Args:
            stock_data_dict: Dictionary mapping tickers to StockData
            market_data: Market index data
            
        Returns:
            CAPMResults with analysis for all stocks
            
        Raises:
            ValueError: If no valid stocks to analyze
        """
        if not isinstance(stock_data_dict, dict):
            raise TypeError(
                f"stock_data_dict must be dict, got {type(stock_data_dict)}"
            )
        
        if len(stock_data_dict) == 0:
            raise ValueError("stock_data_dict is empty")
        
        if not isinstance(market_data, MarketData):
            raise TypeError(f"market_data must be MarketData, got {type(market_data)}")
        
        # Analyze each stock
        results_list = []
        failed_stocks = []
        
        for ticker, stock_data in stock_data_dict.items():
            try:
                result = self.analyze_stock(stock_data, market_data)
                results_list.append(result)
            except (ValueError, TypeError) as e:
                failed_stocks.append((ticker, str(e)))
                continue
        
        if len(results_list) == 0:
            raise ValueError(
                f"No valid stocks to analyze. Failed stocks: {failed_stocks}"
            )
        
        # Extract results into arrays
        tickers = [r['ticker'] for r in results_list]
        betas = np.array([r['beta'] for r in results_list])
        expected_returns = np.array([r['expected_return'] for r in results_list])
        actual_returns = np.array([r['actual_return'] for r in results_list])
        alphas = np.array([r['alpha'] for r in results_list])
        r_squared = np.array([r['r_squared'] for r in results_list])
        classifications = [r['classification'] for r in results_list]
        recommendations = [r['recommendation'] for r in results_list]
        
        # Calculate market return for SML
        market_return = annualize_return(market_data.returns)
        
        # Generate SML line
        sml_line = self._generate_sml_line(market_return)
        
        # Create CAPMResults
        capm_results = CAPMResults(
            tickers=tickers,
            betas=betas,
            expected_returns=expected_returns,
            actual_returns=actual_returns,
            alphas=alphas,
            r_squared=r_squared,
            classifications=classifications,
            recommendations=recommendations,
            sml_line=sml_line,
            risk_free_rate=self.risk_free_rate,
            market_return=market_return
        )
        
        return capm_results
    
    def _generate_sml_line(self, market_return: float) -> tuple:
        """Generate Security Market Line for plotting.
        
        Args:
            market_return: Market return (annualized)
            
        Returns:
            Tuple of (beta_range, expected_return_range)
        """
        # Generate beta range from -0.5 to 2.5
        beta_range = np.linspace(-0.5, 2.5, 100)
        
        # Calculate expected returns for each beta
        expected_return_range = np.array([
            calculate_expected_return(self.risk_free_rate, beta, market_return)
            for beta in beta_range
        ])
        
        return (beta_range, expected_return_range)
    
    def identify_investment_opportunities(
        self,
        results: CAPMResults,
        min_r_squared: float = 0.3
    ) -> Dict[str, List[str]]:
        """Identify investment opportunities from CAPM results.
        
        Args:
            results: CAPM analysis results
            min_r_squared: Minimum R-squared for high-confidence recommendations
            
        Returns:
            Dictionary with categorized stock lists
        """
        if not isinstance(results, CAPMResults):
            raise TypeError(f"results must be CAPMResults, got {type(results)}")
        
        if not 0 <= min_r_squared <= 1:
            raise ValueError(
                f"min_r_squared must be between 0 and 1, got {min_r_squared}"
            )
        
        # Categorize stocks
        high_confidence_buy = []
        high_confidence_sell = []
        low_confidence_buy = []
        low_confidence_sell = []
        hold = []
        
        for i, ticker in enumerate(results.tickers):
            recommendation = results.recommendations[i]
            r_sq = results.r_squared[i]
            
            if recommendation == 'BUY':
                if r_sq >= min_r_squared:
                    high_confidence_buy.append(ticker)
                else:
                    low_confidence_buy.append(ticker)
            elif recommendation == 'SELL':
                if r_sq >= min_r_squared:
                    high_confidence_sell.append(ticker)
                else:
                    low_confidence_sell.append(ticker)
            else:  # HOLD
                hold.append(ticker)
        
        return {
            'high_confidence_buy': high_confidence_buy,
            'low_confidence_buy': low_confidence_buy,
            'high_confidence_sell': high_confidence_sell,
            'low_confidence_sell': low_confidence_sell,
            'hold': hold
        }
