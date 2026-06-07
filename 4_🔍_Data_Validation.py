"""
Data Validation Page

Validates time series data for forecasting model suitability.
Checks assumptions and requirements for LSTM, ARIMA, and GARCH models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

# Import validation module
from src.forecasting import ForecastingDataValidator

# Page configuration
st.set_page_config(
    page_title="Data Validation",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .validation-passed {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .validation-warning {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .validation-failed {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 Data Validation for Forecasting")
st.markdown("### Check Model Suitability Before Forecasting")
st.markdown("---")

# Check if portfolio optimization was run
if 'portfolio_results' not in st.session_state or not st.session_state.portfolio_run:
    st.warning("⚠️ Please run Portfolio Optimization first!")
    if st.button("← Go to Portfolio Optimization", use_container_width=True):
        st.switch_page("pages/2_💼_Portfolio_Optimization.py")
    st.stop()

# Introduction
with st.expander("📖 About Data Validation", expanded=False):
    st.markdown("""
    ### Why Validate Data Before Forecasting?
    
    Each forecasting model has **specific requirements and assumptions**:
    
    **LSTM (Neural Network)**:
    - Needs sufficient data (100+ observations)
    - Works best with patterns and autocorrelation
    - Requires stable data without too many outliers
    
    **ARIMA (Statistical)**:
    - Requires stationary data (or can difference it)
    - Needs significant autocorrelation structure
    - Best for linear time series patterns
    
    **GARCH (Volatility)**:
    - Models volatility, not price levels
    - Requires volatility clustering (ARCH effects)
    - Best for financial returns with changing variance
    
    **This page helps you**:
    - ✅ Check if your data meets model requirements
    - 📊 Visualize data characteristics
    - 🎯 Choose the most suitable model
    - ⚠️ Identify potential issues before forecasting
    """)

st.markdown("---")

# Portfolio Overview
st.header("📊 Portfolio Stock Validation")

# Get portfolio stocks from Portfolio Optimization
portfolio_data = st.session_state.portfolio_results
portfolio_tickers = portfolio_data['tickers']
portfolio_weights = portfolio_data['weights']
stock_data = portfolio_data['stock_data']

st.info(f"""
**Portfolio Overview**:
- Total stocks in portfolio: **{len(portfolio_tickers)}**
- From Portfolio Optimization results
- Validating all stocks simultaneously for forecasting
""")

# Display portfolio composition
col1, col2 = st.columns(2)

with col1:
    st.subheader("Portfolio Stocks")
    portfolio_table = pd.DataFrame({
        'Ticker': portfolio_tickers,
        'Weight (%)': [w*100 for w in portfolio_weights]
    })
    st.dataframe(portfolio_table, use_container_width=True, height=min(400, len(portfolio_tickers)*35+38))

with col2:
    data_type = st.radio(
        "Data Type for Validation",
        options=["Prices", "Returns"],
        help="Prices for LSTM/ARIMA, Returns for GARCH"
    )
    
    st.metric("Stocks to Validate", len(portfolio_tickers))
    st.caption("All portfolio stocks will be validated")

# Calculate basic stats for all stocks
data_overview = []
for ticker in portfolio_tickers:
    stock = stock_data[ticker]
    prices = stock.prices
    data_overview.append({
        'Ticker': ticker,
        'Start': prices.index[0].strftime('%Y-%m-%d'),
        'End': prices.index[-1].strftime('%Y-%m-%d'),
        'Observations': len(prices)
    })

st.subheader("Data Overview")
overview_df = pd.DataFrame(data_overview)
st.dataframe(overview_df, use_container_width=True)

st.markdown("---")

# Run Validation Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Run Validation for All Stocks", type="primary", use_container_width=True):
        with st.spinner(f"Validating {len(portfolio_tickers)} stocks for all forecasting models..."):
            try:
                all_validation_results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, ticker in enumerate(portfolio_tickers):
                    status_text.info(f"Validating {ticker} ({idx+1}/{len(portfolio_tickers)})...")
                    
                    # Get stock data
                    stock = stock_data[ticker]
                    prices = stock.prices
                    returns = prices.pct_change().dropna()
                    
                    # Select data based on user choice
                    data_to_validate = prices if data_type == "Prices" else returns
                    data_type_for_validator = 'price' if data_type == "Prices" else 'return'
                    
                    # Create validator
                    validator = ForecastingDataValidator(
                        data=data_to_validate,
                        data_type=data_type_for_validator
                    )
                    
                    # Run all validations
                    validation_results = validator.validate_all()
                    suitability_scores = validator.get_model_suitability_scores()
                    
                    # Store results for this stock
                    all_validation_results[ticker] = {
                        'validation_results': validation_results,
                        'suitability_scores': suitability_scores,
                        'prices': prices,
                        'returns': returns
                    }
                    
                    # Update progress
                    progress_bar.progress((idx + 1) / len(portfolio_tickers))
                
                status_text.empty()
                progress_bar.empty()
                
                # Store in session state
                st.session_state.all_validation_results = all_validation_results
                st.session_state.validated_data_type = data_type
                st.session_state.validation_run = True
                
                st.success(f"✅ Validation completed successfully for all {len(portfolio_tickers)} stocks!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during validation: {str(e)}")
                st.exception(e)
                st.stop()

    # Display results if validation was run
    if st.session_state.get('validation_run', False) and st.session_state.get('all_validation_results'):
        all_results = st.session_state.all_validation_results
        
        st.markdown("---")
        
        # Summary Table for All Stocks
        st.header("📊 Validation Summary - All Stocks")
        
        # Create summary dataframe
        summary_data = []
        for ticker in portfolio_tickers:
            if ticker in all_results:
                scores = all_results[ticker]['suitability_scores']
                results = all_results[ticker]['validation_results']
                
                summary_data.append({
                    'Ticker': ticker,
                    'LSTM Score': scores['LSTM'],
                    'ARIMA Score': scores['ARIMA'],
                    'GARCH Score': scores['GARCH'],
                    'Avg Score': np.mean(list(scores.values())),
                    'Best Model': max(scores, key=scores.get),
                    'Data Quality': '✅' if results['data_quality']['passed'] else '❌',
                    'Observations': results['data_quality']['n_observations']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Color coding function
        def score_color(val):
            if isinstance(val, (int, float)):
                if val >= 70:
                    return 'background-color: #d4edda'
                elif val >= 50:
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            return ''
        
        st.dataframe(
            summary_df.style.applymap(score_color, subset=['LSTM Score', 'ARIMA Score', 'GARCH Score', 'Avg Score'])
            .format({
                'LSTM Score': '{:.0f}',
                'ARIMA Score': '{:.0f}',
                'GARCH Score': '{:.0f}',
                'Avg Score': '{:.0f}'
            }),
            use_container_width=True,
            height=min(600, len(summary_df)*35+38)
        )
        
        st.info("""
        **Score Interpretation**:
        - 🟢 **70-100**: Excellent - Highly recommended
        - 🟡 **50-69**: Good - Usable with caution  
        - 🔴 **0-49**: Poor - Not recommended
        """)
        
        # Overall portfolio statistics
        st.subheader("Portfolio-Wide Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            excellent_stocks = len(summary_df[summary_df['Avg Score'] >= 70])
            st.metric("Excellent Stocks", f"{excellent_stocks}/{len(summary_df)}")
            st.caption("Avg Score ≥ 70")
        
        with col2:
            good_stocks = len(summary_df[summary_df['Avg Score'] >= 50])
            st.metric("Suitable for Forecasting", f"{good_stocks}/{len(summary_df)}")
            st.caption("Avg Score ≥ 50")
        
        with col3:
            avg_lstm = summary_df['LSTM Score'].mean()
            st.metric("Avg LSTM Score", f"{avg_lstm:.0f}")
            st.caption("Portfolio average")
        
        with col4:
            avg_arima = summary_df['ARIMA Score'].mean()
            st.metric("Avg ARIMA Score", f"{avg_arima:.0f}")
            st.caption("Portfolio average")
        
        st.markdown("---")
        
        # Individual Stock Details (Expandable)
        st.header("📋 Individual Stock Details")
        
        # Stock selector for detailed view
        selected_ticker = st.selectbox(
            "Select stock for detailed analysis:",
            options=portfolio_tickers,
            help="View detailed validation results for a specific stock"
        )
        
        if selected_ticker in all_results:
            stock_result = all_results[selected_ticker]
            results = stock_result['validation_results']
            scores = stock_result['suitability_scores']
            prices = stock_result['prices']
            returns = stock_result['returns']
            
            # Show weight in portfolio
            ticker_index = portfolio_tickers.index(selected_ticker)
            weight = portfolio_weights[ticker_index]
            
            st.info(f"""
            **{selected_ticker}** - Portfolio weight: **{weight*100:.2f}%**
            - Period: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}
            - Observations: {len(prices)}
            """)
            
            # Model Suitability Scores
            st.subheader("🎯 Model Suitability Scores")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score = scores['LSTM']
                color = "normal" if score >= 70 else "inverse"
                st.metric(
                    "LSTM Score",
                    f"{score:.0f}/100",
                    delta="Neural Network",
                    delta_color=color
                )
            
            with col2:
                score = scores['ARIMA']
                color = "normal" if score >= 70 else "inverse"
                st.metric(
                    "ARIMA Score",
                    f"{score:.0f}/100",
                    delta="Statistical",
                    delta_color=color
                )
            
            with col3:
                score = scores['GARCH']
                color = "normal" if score >= 70 else "inverse"
                st.metric(
                    "GARCH Score",
                    f"{score:.0f}/100",
                    delta="Volatility",
                    delta_color=color
                )

            # Score interpretation
            st.progress(max(scores.values()) / 100)
            
            best_model = max(scores, key=scores.get)
            best_score = scores[best_model]
            
            if best_score >= 80:
                st.success(f"✅ **Recommended Model**: {best_model} (Score: {best_score:.0f}/100) - Excellent fit!")
            elif best_score >= 60:
                st.info(f"ℹ️ **Recommended Model**: {best_model} (Score: {best_score:.0f}/100) - Good fit with minor cautions")
            else:
                st.warning(f"⚠️ **Best Available**: {best_model} (Score: {best_score:.0f}/100) - Proceed with caution, consider data improvements")
            
            st.markdown("---")
        
        # Data Quality Check
        st.header("✅ Data Quality Assessment")
        
        dq = results['data_quality']
        
        if dq['passed']:
            st.markdown("""
            <div class="validation-passed">
                <h3 style="color: #28a745; margin-top: 0;">✅ Data Quality: PASSED</h3>
                <p style="margin-bottom: 0;">Your data has no critical quality issues.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="validation-failed">
                <h3 style="color: #dc3545; margin-top: 0;">❌ Data Quality: FAILED</h3>
                <p style="margin-bottom: 0;">Critical data quality issues detected!</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Observations", dq['n_observations'])
        with col2:
            st.metric("Missing Values", dq['n_missing'])
        with col3:
            st.metric("Outliers", dq['n_outliers'])
        
        if dq['issues']:
            st.error("**Critical Issues**:")
            for issue in dq['issues']:
                st.write(f"- {issue}")
        
        if dq['warnings']:
            st.warning("**Warnings**:")
            for warning in dq['warnings']:
                st.write(f"- {warning}")

        st.markdown("---")
        
        # Model-Specific Validation Results
        st.header("📋 Model-Specific Validation Results")
        
        tab1, tab2, tab3 = st.tabs(["🤖 LSTM", "📈 ARIMA", "📊 GARCH"])
        
        with tab1:
            lstm = results['lstm']
            
            st.subheader("LSTM (Long Short-Term Memory)")
            
            # Status
            if lstm['suitable']:
                st.success("✅ **Data is suitable for LSTM**")
            else:
                st.error("❌ **Data is NOT suitable for LSTM**")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Length", lstm['data_length'])
                st.caption(f"Min required: {lstm['min_required']}")
            with col2:
                st.metric("Autocorrelation Lags", lstm['autocorrelation_lags'])
                st.caption("Higher = better for pattern learning")
            with col3:
                st.metric("Volatility", f"{lstm['volatility']*100:.2f}%")
                st.caption("Daily volatility")
            
            # Issues
            if lstm['issues']:
                st.error("**Issues**:")
                for issue in lstm['issues']:
                    st.write(f"- {issue}")
            
            # Warnings
            if lstm['warnings']:
                st.warning("**Warnings**:")
                for warning in lstm['warnings']:
                    st.write(f"- {warning}")
            
            # Recommendations
            if lstm['recommendations']:
                st.info("**Recommendations**:")
                for rec in lstm['recommendations']:
                    st.write(f"{rec}")

        with tab2:
            arima = results['arima']
            
            st.subheader("ARIMA (AutoRegressive Integrated Moving Average)")
            
            # Status
            if arima['suitable']:
                st.success("✅ **Data is suitable for ARIMA**")
            else:
                st.error("❌ **Data is NOT suitable for ARIMA**")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Length", arima['data_length'])
                st.caption(f"Min required: {arima['min_required']}")
            with col2:
                stationary_text = "Yes ✓" if arima['is_stationary'] else "No ✗"
                st.metric("Stationary", stationary_text)
                st.caption(f"p-value: {arima['adf_pvalue']:.4f}" if arima['adf_pvalue'] else "N/A")
            with col3:
                st.metric("Suggested d", arima['suggested_d'])
                st.caption("Differencing order")
            
            st.metric("Autocorrelation Lags", arima['autocorrelation_lags'])
            st.caption("Significant autocorrelation structure")
            
            # Issues
            if arima['issues']:
                st.error("**Issues**:")
                for issue in arima['issues']:
                    st.write(f"- {issue}")
            
            # Warnings
            if arima['warnings']:
                st.warning("**Warnings**:")
                for warning in arima['warnings']:
                    st.write(f"- {warning}")
            
            # Recommendations
            if arima['recommendations']:
                st.info("**Recommendations**:")
                for rec in arima['recommendations']:
                    st.write(f"{rec}")

        with tab3:
            garch = results['garch']
            
            st.subheader("GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)")
            
            # Status
            if garch['suitable']:
                st.success("✅ **Data is suitable for GARCH**")
            else:
                st.error("❌ **Data is NOT suitable for GARCH**")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Returns Length", garch['returns_length'])
                st.caption(f"Min required: {garch['min_required']}")
            with col2:
                st.metric("Vol. Clustering", garch['volatility_clustering_lags'])
                st.caption("ARCH effects")
            with col3:
                st.metric("Kurtosis", f"{garch['kurtosis']:.2f}")
                st.caption("Heavy tails if > 3")
            with col4:
                st.metric("Skewness", f"{garch['skewness']:.2f}")
                st.caption("Asymmetry")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Return", f"{garch['mean_return']*100:.4f}%")
            with col2:
                st.metric("Volatility", f"{garch['volatility']*100:.2f}%")
            
            # Issues
            if garch['issues']:
                st.error("**Issues**:")
                for issue in garch['issues']:
                    st.write(f"- {issue}")
            
            # Warnings
            if garch['warnings']:
                st.warning("**Warnings**:")
                for warning in garch['warnings']:
                    st.write(f"- {warning}")
            
            # Recommendations
            if garch['recommendations']:
                st.info("**Recommendations**:")
                for rec in garch['recommendations']:
                    st.write(f"{rec}")

        st.markdown("---")
        
        # Visual Diagnostics
        st.header("📊 Visual Diagnostics")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Time Series Plot", 
            "📊 Distribution", 
            "🔗 Autocorrelation (ACF)", 
            "🔗 Partial Autocorrelation (PACF)"
        ])
        
        with tab1:
            st.subheader("Time Series Visualization")
            
            # Plot price and returns
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Price Over Time', 'Returns Over Time'),
                vertical_spacing=0.12
            )
            
            # Price plot
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices.values,
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            # Returns plot
            fig.add_trace(
                go.Scatter(
                    x=returns.index,
                    y=returns.values,
                    mode='lines',
                    name='Returns',
                    line=dict(color='orange', width=1)
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Return", row=2, col=1)
            
            fig.update_layout(
                height=600,
                showlegend=True,
                title_text=f"{selected_ticker} - Time Series"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Interpretation**:
            - **Top**: Price movements over time
            - **Bottom**: Daily returns showing volatility
            - Look for trends, seasonality, and structural breaks
            """)

        with tab2:
            st.subheader("Distribution Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram of returns
                fig_hist = go.Figure()
                
                fig_hist.add_trace(go.Histogram(
                    x=returns.values,
                    nbinsx=50,
                    name='Returns',
                    marker_color='steelblue',
                    opacity=0.7
                ))
                
                # Add normal distribution overlay
                mu, std = returns.mean(), returns.std()
                x_range = np.linspace(returns.min(), returns.max(), 100)
                y_normal = stats.norm.pdf(x_range, mu, std) * len(returns) * (returns.max() - returns.min()) / 50
                
                fig_hist.add_trace(go.Scatter(
                    x=x_range,
                    y=y_normal,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_hist.update_layout(
                    title="Returns Distribution",
                    xaxis_title="Return",
                    yaxis_title="Frequency",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Q-Q plot
                fig_qq = go.Figure()
                
                # Calculate theoretical quantiles
                sorted_returns = np.sort(returns.dropna())
                n = len(sorted_returns)
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
                
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_returns,
                    mode='markers',
                    name='Data',
                    marker=dict(color='steelblue', size=4)
                ))
                
                # Add reference line
                fig_qq.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=theoretical_quantiles * std + mu,
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_qq.update_layout(
                    title="Q-Q Plot (Normality Check)",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_qq, use_container_width=True)

            # Distribution statistics
            st.markdown("**Distribution Statistics**:")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{returns.mean()*100:.4f}%")
            with col2:
                st.metric("Std Dev", f"{returns.std()*100:.2f}%")
            with col3:
                st.metric("Skewness", f"{stats.skew(returns.dropna()):.2f}")
            with col4:
                st.metric("Kurtosis", f"{stats.kurtosis(returns.dropna()):.2f}")
            
            st.info("""
            **Interpretation**:
            - **Histogram**: Shows return distribution vs. normal distribution (red dashed line)
            - **Q-Q Plot**: If points follow red line, data is normally distributed
            - **Skewness**: 0 = symmetric, positive = right tail, negative = left tail
            - **Kurtosis**: > 3 = heavy tails (common in financial data), < 3 = light tails
            """)

        with tab3:
            st.subheader("Autocorrelation Function (ACF)")
            
            # Calculate ACF for prices or returns based on selection
            data_for_acf = prices if data_type == "Prices" else returns
            from statsmodels.tsa.stattools import acf
            
            nlags = min(40, len(data_for_acf) // 4)
            acf_values = acf(data_for_acf.dropna(), nlags=nlags, fft=False)
            
            # Confidence intervals
            confidence_interval = 1.96 / np.sqrt(len(data_for_acf))
            
            # Plot ACF
            fig_acf = go.Figure()
            
            # Add bars
            fig_acf.add_trace(go.Bar(
                x=list(range(len(acf_values))),
                y=acf_values,
                name='ACF',
                marker_color='steelblue'
            ))
            
            # Add confidence interval lines
            fig_acf.add_hline(
                y=confidence_interval, 
                line_dash="dash", 
                line_color="red",
                annotation_text="95% CI"
            )
            fig_acf.add_hline(
                y=-confidence_interval, 
                line_dash="dash", 
                line_color="red"
            )
            
            fig_acf.update_layout(
                title=f"Autocorrelation Function - {data_type}",
                xaxis_title="Lag",
                yaxis_title="ACF",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_acf, use_container_width=True)
            
            # Count significant lags
            significant_lags = np.sum(np.abs(acf_values[1:]) > confidence_interval)
            
            st.info(f"""
            **Interpretation**:
            - **Significant lags**: {significant_lags} (bars exceeding red dashed lines)
            - **High ACF**: Data has strong autocorrelation (good for LSTM/ARIMA)
            - **Low ACF**: Data is more random (may be difficult to forecast)
            - **LSTM/ARIMA**: Need significant autocorrelation structure
            """)

        with tab4:
            st.subheader("Partial Autocorrelation Function (PACF)")
            
            # Calculate PACF
            from statsmodels.tsa.stattools import pacf
            
            pacf_values = pacf(data_for_acf.dropna(), nlags=nlags)
            
            # Plot PACF
            fig_pacf = go.Figure()
            
            # Add bars
            fig_pacf.add_trace(go.Bar(
                x=list(range(len(pacf_values))),
                y=pacf_values,
                name='PACF',
                marker_color='coral'
            ))
            
            # Add confidence interval lines
            fig_pacf.add_hline(
                y=confidence_interval, 
                line_dash="dash", 
                line_color="red",
                annotation_text="95% CI"
            )
            fig_pacf.add_hline(
                y=-confidence_interval, 
                line_dash="dash", 
                line_color="red"
            )
            
            fig_pacf.update_layout(
                title=f"Partial Autocorrelation Function - {data_type}",
                xaxis_title="Lag",
                yaxis_title="PACF",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_pacf, use_container_width=True)
            
            # Count significant lags
            significant_pacf_lags = np.sum(np.abs(pacf_values[1:]) > confidence_interval)
            
            st.info(f"""
            **Interpretation**:
            - **Significant lags**: {significant_pacf_lags} (bars exceeding red dashed lines)
            - **PACF**: Shows direct correlation at each lag (removing indirect effects)
            - **ARIMA**: PACF helps determine the AR order (p parameter)
            - **Cutoff pattern**: Suggests AR model order
            """)

        st.markdown("---")
        
        # Volatility Clustering Check (for GARCH)
        st.header("📉 Volatility Clustering (GARCH Check)")
        
        # Calculate squared returns and their ACF
        returns_squared = returns ** 2
        acf_squared = acf(returns_squared.dropna(), nlags=min(20, len(returns_squared) // 4), fft=False)
        
        fig_vol = go.Figure()
        
        # Plot squared returns
        fig_vol.add_trace(go.Scatter(
            x=returns.index,
            y=returns_squared.values,
            mode='lines',
            name='Squared Returns',
            line=dict(color='purple', width=1),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.2)'
        ))
        
        fig_vol.update_layout(
            title="Squared Returns (Volatility Proxy)",
            xaxis_title="Date",
            yaxis_title="Squared Return",
            height=400
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
        
        # ACF of squared returns
        fig_acf_sq = go.Figure()
        
        fig_acf_sq.add_trace(go.Bar(
            x=list(range(len(acf_squared))),
            y=acf_squared,
            name='ACF Squared Returns',
            marker_color='purple'
        ))
        
        confidence_interval_sq = 1.96 / np.sqrt(len(returns_squared))
        fig_acf_sq.add_hline(y=confidence_interval_sq, line_dash="dash", line_color="red")
        fig_acf_sq.add_hline(y=-confidence_interval_sq, line_dash="dash", line_color="red")
        
        fig_acf_sq.update_layout(
            title="ACF of Squared Returns (Volatility Clustering)",
            xaxis_title="Lag",
            yaxis_title="ACF",
            height=400
        )
        
        st.plotly_chart(fig_acf_sq, use_container_width=True)
        
        vol_clustering_lags = np.sum(np.abs(acf_squared[1:]) > confidence_interval_sq)
        
        if vol_clustering_lags >= 3:
            st.success(f"✅ **Strong volatility clustering detected** ({vol_clustering_lags} significant lags) - GARCH is appropriate")
        elif vol_clustering_lags > 0:
            st.info(f"ℹ️ **Moderate volatility clustering** ({vol_clustering_lags} significant lags) - GARCH may be useful")
        else:
            st.warning("⚠️ **Weak volatility clustering** - GARCH may not be necessary")

        st.markdown("---")
        
        # Export and Next Steps
        st.header("📥 Export & Next Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Text report
            validator = ForecastingDataValidator(
                data=prices if data_type == "Prices" else returns,
                data_type='price' if data_type == "Prices" else 'return'
            )
            validator.validation_results = results
            
            report_text = validator.get_summary_report()
            
            st.download_button(
                label="📄 Download Report (TXT)",
                data=report_text,
                file_name=f"validation_report_{selected_ticker}_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # JSON results
            import json
            json_data = json.dumps(results, indent=2, default=str)
            
            st.download_button(
                label="📊 Download Results (JSON)",
                data=json_data,
                file_name=f"validation_results_{selected_ticker}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # CSV summary
            summary_data = {
                'Metric': ['Model', 'Suitability Score', 'Data Length', 'Suitable'],
                'LSTM': ['LSTM', scores['LSTM'], lstm['data_length'], lstm['suitable']],
                'ARIMA': ['ARIMA', scores['ARIMA'], arima['data_length'], arima['suitable']],
                'GARCH': ['GARCH', scores['GARCH'], garch['returns_length'], garch['suitable']]
            }
            
            summary_df = pd.DataFrame(summary_data).T
            summary_df.columns = summary_df.iloc[0]
            summary_df = summary_df.drop('Metric')
            
            csv_buffer = BytesIO()
            summary_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📈 Download Summary (CSV)",
                data=csv_data,
                file_name=f"validation_summary_{selected_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")
    
    # Decision Guide (portfolio-level, outside individual stock block)
    st.header("🎯 What's Next?")
    
    # Get portfolio-level summary_df from session state
    # Recreate it if needed
    all_results = st.session_state.get('all_validation_results', {})
    summary_data = []
    for ticker in portfolio_tickers:
        if ticker in all_results:
            scores = all_results[ticker]['suitability_scores']
            results = all_results[ticker]['validation_results']
            
            summary_data.append({
                'Ticker': ticker,
                'LSTM Score': scores['LSTM'],
                'ARIMA Score': scores['ARIMA'],
                'GARCH Score': scores['GARCH'],
                'Avg Score': np.mean(list(scores.values())),
                'Best Model': max(scores, key=scores.get),
                'Data Quality': '✅' if results['data_quality']['passed'] else '❌',
                'Observations': results['data_quality']['n_observations']
            })
    
    portfolio_summary_df = pd.DataFrame(summary_data)
    
    # Portfolio-level decision
    suitable_stocks = portfolio_summary_df[portfolio_summary_df['Avg Score'] >= 50]
    excellent_stocks = portfolio_summary_df[portfolio_summary_df['Avg Score'] >= 70]
    
    if len(excellent_stocks) > 0:
        st.success(f"""
        ✅ **Portfolio is ready for forecasting!**
        
        - **{len(excellent_stocks)} stocks** with excellent scores (≥70)
        - **{len(suitable_stocks)} stocks** suitable for forecasting (≥50)
        - **{len(portfolio_summary_df) - len(suitable_stocks)} stocks** with lower confidence
        
        You can proceed to forecast all suitable stocks simultaneously.
        """)
    elif len(suitable_stocks) > 0:
        st.warning(f"""
        ⚠️ **Proceed with caution**
        
        - **{len(suitable_stocks)} stocks** are suitable for forecasting (score ≥50)
        - **{len(portfolio_summary_df) - len(suitable_stocks)} stocks** have lower confidence
        
        Review warnings and recommendations before proceeding.
        """)
    else:
        st.error(f"""
        ❌ **Data quality issues detected**
        
        No stocks meet the minimum threshold (score ≥50) for reliable forecasting.
        
        Consider:
        - Collecting more data
        - Addressing data quality issues
        - Reviewing individual stock details above
        """)
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Crisis Scenarios", use_container_width=True):
            st.switch_page("pages/3_⚠️_Crisis_Scenarios.py")
    
    with col2:
        # Enable forecasting button if at least one stock is suitable
        can_proceed = len(suitable_stocks) > 0
        
        if can_proceed:
            if st.button("Next: Forecast All Stocks →", type="primary", use_container_width=True):
                st.switch_page("pages/5_📈_Forecasting.py")
        else:
            st.button(
                "Next: Forecast All Stocks →", 
                type="primary", 
                use_container_width=True, 
                disabled=True,
                help="Improve data quality before proceeding to forecasting"
            )
