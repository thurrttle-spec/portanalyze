"""
Forecasting Page

Implements LSTM, ARIMA, and GARCH forecasting models for multiple stocks simultaneously.
Optimized with parallel processing for faster execution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
warnings.filterwarnings('ignore')

# Import forecasting modules
from src.forecasting import LSTMForecaster, ARIMAForecaster, GARCHForecaster

# Page configuration
st.set_page_config(
    page_title="Forecasting",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .forecast-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Portfolio Forecasting")
st.markdown("### LSTM, ARIMA & GARCH Models for All Stocks")
st.markdown("---")

# Check if validation was run
if 'all_validation_results' not in st.session_state or not st.session_state.get('validation_run', False):
    st.warning("⚠️ Please run Data Validation first!")
    if st.button("← Go to Data Validation", use_container_width=True):
        st.switch_page("pages/4_🔍_Data_Validation.py")
    st.stop()

# Get validated data
all_validation_results = st.session_state.all_validation_results
validated_data_type = st.session_state.validated_data_type

# Get portfolio data
portfolio_results = st.session_state.portfolio_results
portfolio_tickers = portfolio_results['tickers']
portfolio_weights = portfolio_results['weights']
stock_data = portfolio_results['stock_data']

# Filter stocks with suitable scores (≥ 50)
suitable_stocks = {}
for ticker in portfolio_tickers:
    if ticker in all_validation_results:
        scores = all_validation_results[ticker]['suitability_scores']
        avg_score = np.mean(list(scores.values()))
        if avg_score >= 50:
            suitable_stocks[ticker] = {
                'scores': scores,
                'avg_score': avg_score,
                'weight': portfolio_weights[portfolio_tickers.index(ticker)]
            }

if len(suitable_stocks) == 0:
    st.error("❌ No stocks meet the minimum suitability threshold (score ≥ 50) for forecasting!")
    if st.button("← Back to Data Validation", use_container_width=True):
        st.switch_page("pages/4_🔍_Data_Validation.py")
    st.stop()

# Introduction
with st.expander("📖 About Forecasting Models", expanded=False):
    st.markdown("""
    ### Three Complementary Models
    
    **LSTM**: Complex patterns, long-term dependencies
    **ARIMA**: Linear trends, statistical approach  
    **GARCH**: Volatility and risk forecasting
    
    All suitable portfolio stocks will be forecasted simultaneously!
    """)

st.markdown("---")

# Display validation summary
st.header("📊 Portfolio Forecast Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Portfolio Stocks", len(portfolio_tickers))
with col2:
    st.metric("Suitable for Forecasting", len(suitable_stocks))
    st.caption("Score ≥ 50")
with col3:
    avg_score = np.mean([s['avg_score'] for s in suitable_stocks.values()])
    st.metric("Avg Suitability Score", f"{avg_score:.0f}/100")
with col4:
    st.metric("Data Type", validated_data_type)

# Suitable stocks table
st.subheader("Stocks Ready for Forecasting")
suitable_df = pd.DataFrame([
    {
        'Ticker': ticker,
        'Weight (%)': data['weight']*100,
        'LSTM': data['scores']['LSTM'],
        'ARIMA': data['scores']['ARIMA'],
        'GARCH': data['scores']['GARCH'],
        'Avg Score': data['avg_score']
    }
    for ticker, data in suitable_stocks.items()
])
st.dataframe(suitable_df.style.format({
    'Weight (%)': '{:.2f}', 'LSTM': '{:.0f}', 'ARIMA': '{:.0f}', 'GARCH': '{:.0f}', 'Avg Score': '{:.0f}'
}).background_gradient(subset=['Avg Score'], cmap='Greens'), use_container_width=True)

st.markdown("---")

# Forecasting Configuration
st.header("⚙️ Forecasting Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Selection")
    enable_lstm = st.checkbox("🤖 LSTM Neural Network", value=True, help="Deep learning for complex patterns")
    enable_arima = st.checkbox("📈 ARIMA Statistical", value=True, help="Classical time series model")
    enable_garch = st.checkbox("📊 GARCH Volatility", value=True, help="Volatility forecasting")
    
    if not (enable_lstm or enable_arima or enable_garch):
        st.warning("⚠️ Please select at least one model!")

with col2:
    st.subheader("Forecast Parameters")
    forecast_days = st.slider("Forecast Horizon (Days)", 5, 60, 30, 5)
    train_split = st.slider("Training Data Split (%)", 70, 95, 80, 5)
    
st.markdown("---")

# Advanced Configuration
with st.expander("🔧 Advanced Model Parameters"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("LSTM")
        lstm_lookback = st.number_input("Lookback", 10, 120, 60, 5)
        lstm_epochs = st.number_input("Epochs", 10, 100, 30, 10)
    with col2:
        st.subheader("ARIMA")
        arima_auto = st.checkbox("Auto-select Order", value=True)
    with col3:
        st.subheader("GARCH")
        garch_p = st.number_input("GARCH p", 1, 3, 1)
        garch_q = st.number_input("ARCH q", 1, 3, 1)

st.markdown("---")

# Initialize session state
if 'all_forecast_results' not in st.session_state:
    st.session_state.all_forecast_results = None
if 'forecast_run' not in st.session_state:
    st.session_state.forecast_run = False

# Helper function for parallel forecasting
def forecast_single_stock(ticker, stock_info, stock_data, train_split, forecast_days,
                         enable_lstm, enable_arima, enable_garch,
                         lstm_lookback, lstm_epochs, arima_auto, garch_p, garch_q):
    """Forecast a single stock (for parallel processing)."""
    try:
        # Get data
        stock = stock_data[ticker]
        prices = stock.prices
        returns = prices.pct_change().dropna()
        
        # Split data
        split_idx = int(len(prices) * train_split / 100)
        train_prices = prices[:split_idx]
        test_prices = prices[split_idx:]
        train_returns = returns[:split_idx-1]
        
        forecast_result = {
            'ticker': ticker,
            'weight': stock_info['weight'],
            'scores': stock_info['scores'],
            'current_price': prices.iloc[-1],
            'models': {}
        }
        
        # LSTM
        if enable_lstm and stock_info['scores']['LSTM'] >= 50:
            try:
                lstm = LSTMForecaster(lookback=lstm_lookback, epochs=lstm_epochs, batch_size=32)
                lstm.fit(train_prices)
                future_lstm = lstm.predict(prices, forecast_days)
                test_lstm = lstm.predict(train_prices, len(test_prices))
                metrics_lstm = lstm.evaluate(test_prices.values, test_lstm)
                forecast_result['models']['LSTM'] = {
                    'future': future_lstm,
                    'metrics': metrics_lstm
                }
            except Exception as e:
                pass
        
        # ARIMA
        if enable_arima and stock_info['scores']['ARIMA'] >= 50:
            try:
                arima = ARIMAForecaster(auto_order=arima_auto)
                arima.fit(train_prices)
                test_arima, _ = arima.predict(len(test_prices))
                arima_full = ARIMAForecaster(order=arima.best_order, auto_order=False)
                arima_full.fit(prices)
                future_arima, conf_int = arima_full.predict(forecast_days)
                metrics_arima = arima.evaluate(test_prices.values, test_arima)
                forecast_result['models']['ARIMA'] = {
                    'future': future_arima,
                    'conf_int': conf_int,
                    'metrics': metrics_arima
                }
            except Exception as e:
                pass
        
        # GARCH
        if enable_garch and stock_info['scores']['GARCH'] >= 50:
            try:
                garch = GARCHForecaster(p=garch_p, q=garch_q)
                garch.fit(returns)
                future_garch = garch.predict(forecast_days)
                forecast_result['models']['GARCH'] = {
                    'volatility': future_garch['volatility'],
                    'annualized_vol': future_garch['annualized_volatility']
                }
            except Exception as e:
                pass
        
        return ticker, forecast_result
    except Exception as e:
        return ticker, None

# Run Forecasting Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Add performance mode toggle
    use_parallel = st.checkbox("⚡ Fast Mode (Parallel Processing)", value=True,
                               help="Process multiple stocks simultaneously for faster results")
    
    if st.button("🚀 Generate Forecasts for All Stocks", type="primary", use_container_width=True):
        if not (enable_lstm or enable_arima or enable_garch):
            st.error("❌ Please select at least one model!")
            st.stop()
        
        with st.spinner(f"Training models and generating forecasts for {len(suitable_stocks)} stocks..."):
            try:
                all_forecasts = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if use_parallel and len(suitable_stocks) > 1:
                    # PARALLEL PROCESSING MODE (FASTER)
                    status_text.info(f"⚡ Fast mode: Processing {len(suitable_stocks)} stocks in parallel...")
                    
                    # Use ThreadPoolExecutor for parallel processing
                    max_workers = min(4, len(suitable_stocks))  # Limit to 4 parallel workers
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all forecasting tasks
                        future_to_ticker = {
                            executor.submit(
                                forecast_single_stock,
                                ticker, stock_info, stock_data, train_split, forecast_days,
                                enable_lstm, enable_arima, enable_garch,
                                lstm_lookback, lstm_epochs, arima_auto, garch_p, garch_q
                            ): ticker
                            for ticker, stock_info in suitable_stocks.items()
                        }
                        
                        # Collect results as they complete
                        completed = 0
                        for future in as_completed(future_to_ticker):
                            ticker = future_to_ticker[future]
                            try:
                                result_ticker, forecast_result = future.result()
                                if forecast_result:
                                    all_forecasts[result_ticker] = forecast_result
                                completed += 1
                                progress_bar.progress(completed / len(suitable_stocks))
                                status_text.info(f"⚡ Completed {completed}/{len(suitable_stocks)} stocks...")
                            except Exception as e:
                                st.warning(f"⚠️ Error forecasting {ticker}: {str(e)}")
                                completed += 1
                                progress_bar.progress(completed / len(suitable_stocks))
                else:
                    # SEQUENTIAL PROCESSING MODE (ORIGINAL)
                    status_text.info("Processing stocks sequentially...")
                    
                    for stock_idx, (ticker, stock_info) in enumerate(suitable_stocks.items()):
                        status_text.info(f"Forecasting {ticker} ({stock_idx+1}/{len(suitable_stocks)})...")
                        
                        result_ticker, forecast_result = forecast_single_stock(
                            ticker, stock_info, stock_data, train_split, forecast_days,
                            enable_lstm, enable_arima, enable_garch,
                            lstm_lookback, lstm_epochs, arima_auto, garch_p, garch_q
                        )
                        
                        if forecast_result:
                            all_forecasts[result_ticker] = forecast_result
                        
                        progress_bar.progress((stock_idx + 1) / len(suitable_stocks))
                
                status_text.empty()
                progress_bar.empty()
                
                # Generate future dates (use any stock's data for date reference)
                sample_ticker = list(suitable_stocks.keys())[0]
                sample_prices = stock_data[sample_ticker].prices
                last_date = sample_prices.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
                
                # Store results
                st.session_state.all_forecast_results = {
                    'forecasts': all_forecasts,
                    'future_dates': future_dates,
                    'forecast_days': forecast_days,
                    'suitable_stocks': suitable_stocks
                }
                st.session_state.forecast_run = True
                
                st.success(f"✅ Forecasts generated for {len(all_forecasts)} stocks!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during forecasting: {str(e)}")
                st.exception(e)
                st.stop()

# Display results if available
if st.session_state.forecast_run and st.session_state.all_forecast_results:
    results = st.session_state.all_forecast_results
    all_forecasts = results['forecasts']
    future_dates = results['future_dates']
    forecast_days = results['forecast_days']
    
    st.markdown("---")
    
    # Portfolio-Level Summary
    st.header("📊 Portfolio Forecast Summary")
    
    # Calculate portfolio-level metrics
    portfolio_current = 0
    portfolio_forecast_lstm = 0
    portfolio_forecast_arima = 0
    weights_sum = 0
    
    forecast_summary = []
    for ticker, forecast in all_forecasts.items():
        weight = forecast['weight']
        current = forecast['current_price']
        
        row = {
            'Ticker': ticker,
            'Weight (%)': weight * 100,
            'Current Price': current,
            'LSTM Forecast': None,
            'LSTM Change (%)': None,
            'ARIMA Forecast': None,
            'ARIMA Change (%)': None,
            'Consensus': None,
            'Consensus Change (%)': None
        }
        
        forecasts_list = []
        
        if 'LSTM' in forecast['models']:
            lstm_final = forecast['models']['LSTM']['future'][-1]
            lstm_change = ((lstm_final - current) / current) * 100
            row['LSTM Forecast'] = lstm_final
            row['LSTM Change (%)'] = lstm_change
            forecasts_list.append(lstm_final)
            portfolio_forecast_lstm += lstm_final * weight
        
        if 'ARIMA' in forecast['models']:
            arima_final = forecast['models']['ARIMA']['future'][-1]
            arima_change = ((arima_final - current) / current) * 100
            row['ARIMA Forecast'] = arima_final
            row['ARIMA Change (%)'] = arima_change
            forecasts_list.append(arima_final)
            portfolio_forecast_arima += arima_final * weight
        
        if forecasts_list:
            consensus = np.mean(forecasts_list)
            consensus_change = ((consensus - current) / current) * 100
            row['Consensus'] = consensus
            row['Consensus Change (%)'] = consensus_change
        
        forecast_summary.append(row)
        portfolio_current += current * weight
        weights_sum += weight
    
    forecast_summary_df = pd.DataFrame(forecast_summary)
    
    st.dataframe(
        forecast_summary_df.style.format({
            'Weight (%)': '{:.2f}',
            'Current Price': '${:.2f}',
            'LSTM Forecast': '${:.2f}',
            'LSTM Change (%)': '{:+.2f}%',
            'ARIMA Forecast': '${:.2f}',
            'ARIMA Change (%)': '{:+.2f}%',
            'Consensus': '${:.2f}',
            'Consensus Change (%)': '{:+.2f}%'
        }, na_rep='-'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Portfolio-Level Forecast
    st.header("💼 Portfolio-Level Forecast")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks Forecasted", len(all_forecasts))
    
    with col2:
        avg_consensus_change = forecast_summary_df['Consensus Change (%)'].mean()
        st.metric("Avg Expected Return", f"{avg_consensus_change:+.2f}%", delta=f"{forecast_days} days")
    
    with col3:
        positive_stocks = len(forecast_summary_df[forecast_summary_df['Consensus Change (%)'] > 0])
        st.metric("Bullish Stocks", f"{positive_stocks}/{len(all_forecasts)}")
    
    with col4:
        negative_stocks = len(forecast_summary_df[forecast_summary_df['Consensus Change (%)'] < 0])
        st.metric("Bearish Stocks", f"{negative_stocks}/{len(all_forecasts)}")
    
    # Calculate volatility risk from GARCH if available
    high_volatility_count = 0
    avg_volatility = 0
    volatility_stocks = []
    
    for ticker, forecast in all_forecasts.items():
        if 'GARCH' in forecast['models']:
            ann_vol = forecast['models']['GARCH']['annualized_vol']
            # Handle both scalar and array values
            if isinstance(ann_vol, (list, np.ndarray)):
                ann_vol_value = float(np.mean(ann_vol))
            else:
                ann_vol_value = float(ann_vol)
            
            avg_volatility += ann_vol_value
            if ann_vol_value > 0.40:  # 40% annualized volatility threshold
                high_volatility_count += 1
                volatility_stocks.append(f"{ticker} ({ann_vol_value*100:.1f}%)")
    
    if high_volatility_count > 0:
        avg_volatility = avg_volatility / len([f for f in all_forecasts.values() if 'GARCH' in f['models']])
    
    # Risk assessment based on forecasts and volatility
    risk_level = "LOW"
    risk_score = 0
    risk_factors = []
    
    # Factor 1: Negative outlook
    if avg_consensus_change < -5:
        risk_score += 30
        risk_factors.append(f"Strong negative forecast ({avg_consensus_change:+.2f}%)")
    elif avg_consensus_change < -2:
        risk_score += 15
        risk_factors.append(f"Negative forecast ({avg_consensus_change:+.2f}%)")
    
    # Factor 2: High proportion of bearish stocks
    bearish_ratio = negative_stocks / len(all_forecasts)
    if bearish_ratio > 0.6:
        risk_score += 25
        risk_factors.append(f"{int(bearish_ratio*100)}% of stocks bearish")
    elif bearish_ratio > 0.4:
        risk_score += 10
        risk_factors.append(f"{int(bearish_ratio*100)}% of stocks bearish")
    
    # Factor 3: High volatility
    if high_volatility_count > len(all_forecasts) * 0.5:
        risk_score += 30
        risk_factors.append(f"{high_volatility_count} stocks with high volatility (>40%)")
    elif high_volatility_count > 0:
        risk_score += 15
        risk_factors.append(f"{high_volatility_count} stocks with elevated volatility")
    
    # Factor 4: Large expected losses
    max_loss = forecast_summary_df['Consensus Change (%)'].min()
    if max_loss < -15:
        risk_score += 20
        risk_factors.append(f"Largest expected loss: {max_loss:.1f}%")
    elif max_loss < -10:
        risk_score += 10
        risk_factors.append(f"Significant expected loss: {max_loss:.1f}%")
    
    # Determine risk level
    if risk_score >= 50:
        risk_level = "HIGH"
    elif risk_score >= 25:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # Portfolio recommendation with risk assessment
    st.markdown("---")
    st.subheader("📊 Forecast-Based Recommendation")
    
    if avg_consensus_change > 5 and risk_level == "LOW":
        st.success(f"""
        ✅ **STRONG BUY - Portfolio Level**
        
        Average expected return: **{avg_consensus_change:+.2f}%** across all stocks
        - {positive_stocks} stocks trending upward
        - Strong positive momentum
        - Risk Level: **{risk_level}** ✅
        """)
    elif avg_consensus_change > 2 and risk_level in ["LOW", "MEDIUM"]:
        st.info(f"""
        📈 **BUY - Portfolio Level**
        
        Average expected return: **{avg_consensus_change:+.2f}%** across all stocks
        - Positive outlook for portfolio
        - Risk Level: **{risk_level}** {"✅" if risk_level == "LOW" else "⚠️"}
        """)
    elif avg_consensus_change > -2:
        st.warning(f"""
        ⚖️ **HOLD - Portfolio Level**
        
        Average expected return: **{avg_consensus_change:+.2f}%** across all stocks
        - Mixed signals, maintain current positions
        - Risk Level: **{risk_level}** {"⚠️" if risk_level == "MEDIUM" else "🔴" if risk_level == "HIGH" else "✅"}
        """)
    else:
        st.error(f"""
        ⚠️ **CAUTION - Portfolio Level**
        
        Average expected return: **{avg_consensus_change:+.2f}%** across all stocks
        - Negative outlook detected
        - Risk Level: **{risk_level}** 🔴
        """)
    
    # Risk Management Alert
    if risk_level in ["MEDIUM", "HIGH"] or avg_consensus_change < -2:
        st.markdown("---")
        st.warning("### ⚠️ Risk Management Recommended")
        
        if risk_level == "HIGH":
            st.error(f"""
            🔴 **HIGH RISK DETECTED** (Risk Score: {risk_score}/100)
            
            **Risk Factors Identified**:
            """)
            for factor in risk_factors:
                st.markdown(f"- {factor}")
            
            st.markdown("""
            **IMMEDIATE ACTIONS REQUIRED**:
            1. 🛡️ Review comprehensive risk management strategies
            2. 🚨 Set stop-loss orders immediately
            3. 📊 Consider position size reduction
            4. ⚖️ Evaluate portfolio rebalancing
            5. 🔍 Monitor high-volatility stocks closely
            """)
            
            if high_volatility_count > 0:
                st.markdown("**High Volatility Stocks**:")
                for stock in volatility_stocks:
                    st.markdown(f"- {stock}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🛡️ Go to Risk Management Dashboard", type="primary", use_container_width=True):
                    st.switch_page("pages/6_🛡️_Risk_Management.py")
            with col2:
                if st.button("⚠️ Review Crisis Scenarios", use_container_width=True):
                    st.switch_page("pages/3_⚠️_Crisis_Scenarios.py")
        
        elif risk_level == "MEDIUM":
            st.warning(f"""
            🟡 **MODERATE RISK DETECTED** (Risk Score: {risk_score}/100)
            
            **Risk Factors Identified**:
            """)
            for factor in risk_factors:
                st.markdown(f"- {factor}")
            
            st.markdown("""
            **RECOMMENDED ACTIONS**:
            1. 🛡️ Review risk management strategies
            2. 📊 Monitor positions closely
            3. 🚨 Consider setting stop-loss orders
            4. 📈 Track forecast accuracy over time
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🛡️ Go to Risk Management Dashboard", use_container_width=True):
                    st.switch_page("pages/6_🛡️_Risk_Management.py")
            with col2:
                st.info("💡 Proactive risk management can help protect your portfolio")
    
    st.markdown("---")
    
    # Individual Stock Forecasts
    st.header("📈 Individual Stock Forecasts")
    
    # Stock selector
    selected_stock = st.selectbox(
        "Select stock for detailed forecast:",
        options=list(all_forecasts.keys()),
        help="View detailed forecast for a specific stock"
    )
    
    if selected_stock in all_forecasts:
        forecast = all_forecasts[selected_stock]
        stock_prices = stock_data[selected_stock].prices
        
        st.subheader(f"{selected_stock} Forecast Details")
        
        # Price forecast chart
        fig_price = go.Figure()
        
        # Historical
        fig_price.add_trace(go.Scatter(
            x=stock_prices.index,
            y=stock_prices.values,
            mode='lines',
            name='Historical',
            line=dict(color='gray', width=2)
        ))
        
        # LSTM
        if 'LSTM' in forecast['models']:
            fig_price.add_trace(go.Scatter(
                x=future_dates,
                y=forecast['models']['LSTM']['future'],
                mode='lines',
                name='LSTM',
                line=dict(color='blue', width=2, dash='dash')
            ))
        
        # ARIMA with confidence interval
        if 'ARIMA' in forecast['models']:
            arima_pred = forecast['models']['ARIMA']['future']
            conf_int = forecast['models']['ARIMA']['conf_int']
            
            fig_price.add_trace(go.Scatter(
                x=future_dates,
                y=conf_int[:, 1],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig_price.add_trace(go.Scatter(
                x=future_dates,
                y=conf_int[:, 0],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(255, 0, 0, 0.2)',
                fill='tonexty',
                name='ARIMA 95% CI',
                hoverinfo='skip'
            ))
            fig_price.add_trace(go.Scatter(
                x=future_dates,
                y=arima_pred,
                mode='lines',
                name='ARIMA',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig_price.update_layout(
            title=f"{selected_stock} - Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${forecast['current_price']:.2f}")
            st.caption(f"Weight: {forecast['weight']*100:.2f}%")
        
        with col2:
            if 'LSTM' in forecast['models']:
                lstm_final = forecast['models']['LSTM']['future'][-1]
                lstm_change = ((lstm_final - forecast['current_price']) / forecast['current_price']) * 100
                st.metric("LSTM Forecast", f"${lstm_final:.2f}", f"{lstm_change:+.2f}%")
        
        with col3:
            if 'ARIMA' in forecast['models']:
                arima_final = forecast['models']['ARIMA']['future'][-1]
                arima_change = ((arima_final - forecast['current_price']) / forecast['current_price']) * 100
                st.metric("ARIMA Forecast", f"${arima_final:.2f}", f"{arima_change:+.2f}%")
        
        # Volatility forecast (if GARCH available)
        if 'GARCH' in forecast['models']:
            st.subheader("Volatility Forecast (GARCH)")
            
            fig_vol = go.Figure()
            
            vol_forecast = forecast['models']['GARCH']['volatility']
            fig_vol.add_trace(go.Scatter(
                x=future_dates,
                y=vol_forecast * 100,
                mode='lines',
                name='Volatility',
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ))
            
            fig_vol.update_layout(
                title=f"{selected_stock} - Volatility Forecast",
                xaxis_title="Date",
                yaxis_title="Daily Volatility (%)",
                height=350
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
            
            avg_vol = vol_forecast.mean() * 100
            ann_vol = forecast['models']['GARCH']['annualized_vol'].mean() * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg Daily Volatility", f"{avg_vol:.2f}%")
            with col2:
                st.metric("Annualized Volatility", f"{ann_vol:.2f}%")
    
    st.markdown("---")
    
    # Export Section
    st.header("📥 Export All Forecasts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV - Summary
        csv_buffer = io.StringIO()
        forecast_summary_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📄 Download Summary (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"portfolio_forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Detailed forecasts for all stocks
        detailed_data = []
        for date_idx, date in enumerate(future_dates):
            for ticker, forecast in all_forecasts.items():
                row = {'Date': date, 'Ticker': ticker}
                if 'LSTM' in forecast['models']:
                    row['LSTM'] = forecast['models']['LSTM']['future'][date_idx]
                if 'ARIMA' in forecast['models']:
                    row['ARIMA'] = forecast['models']['ARIMA']['future'][date_idx]
                if 'GARCH' in forecast['models']:
                    row['Volatility'] = forecast['models']['GARCH']['volatility'][date_idx]
                detailed_data.append(row)
        
        detailed_df = pd.DataFrame(detailed_data)
        csv_detailed = io.StringIO()
        detailed_df.to_csv(csv_detailed, index=False)
        
        st.download_button(
            label="📊 Download Detailed (CSV)",
            data=csv_detailed.getvalue(),
            file_name=f"detailed_forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Summary report
        summary_buffer = io.StringIO()
        summary_buffer.write(f"=== PORTFOLIO FORECAST SUMMARY ===\n\n")
        summary_buffer.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        summary_buffer.write(f"Forecast Horizon: {forecast_days} days\n")
        summary_buffer.write(f"Stocks Forecasted: {len(all_forecasts)}\n\n")
        summary_buffer.write(f"Portfolio Average Expected Return: {avg_consensus_change:+.2f}%\n")
        summary_buffer.write(f"Bullish Stocks: {positive_stocks}\n")
        summary_buffer.write(f"Bearish Stocks: {negative_stocks}\n\n")
        summary_buffer.write("=== INDIVIDUAL STOCK FORECASTS ===\n\n")
        
        for _, row in forecast_summary_df.iterrows():
            summary_buffer.write(f"{row['Ticker']}:\n")
            summary_buffer.write(f"  Current: ${row['Current Price']:.2f}\n")
            if pd.notna(row['Consensus']):
                summary_buffer.write(f"  Forecast: ${row['Consensus']:.2f} ({row['Consensus Change (%)']:+.2f}%)\n")
            summary_buffer.write("\n")
        
        st.download_button(
            label="📝 Download Report (TXT)",
            data=summary_buffer.getvalue(),
            file_name=f"forecast_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Investment Disclaimer
    st.header("⚠️ Important Considerations")
    
    st.warning("""
    **Investment Disclaimer**:
    
    - 📊 Models are based on historical patterns
    - 📉 Past performance ≠ Future results
    - 🎲 Market conditions can change unpredictably
    - 💼 Consult financial advisors before decisions
    - 🛡️ Never invest more than you can afford to lose
    - 📈 Diversification is essential
    
    These forecasts are for **educational purposes only**.
    """)
    
    st.markdown("---")
    
    # Navigation
    st.header("🎯 Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Data Validation", use_container_width=True):
            st.switch_page("pages/4_🔍_Data_Validation.py")
    
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            st.switch_page("Home.py")
    
    st.success("""
    ✅ **Portfolio Forecasting Complete!**
    
    You've completed the full workflow for all portfolio stocks:
    1. ✅ CAPM Analysis
    2. ✅ Portfolio Optimization
    3. ✅ Crisis Scenarios
    4. ✅ Data Validation
    5. ✅ Forecasting
    
    Use these insights for informed investment decisions!
    """)

else:
    # Instructions
    st.info("""
    👋 **Welcome to Portfolio Forecasting!**
    
    This module forecasts **all suitable stocks** in your portfolio simultaneously:
    - **LSTM**: Neural network for complex patterns
    - **ARIMA**: Statistical model for trends
    - **GARCH**: Volatility and risk forecasting
    
    ### Process:
    
    1. ✅ Complete Data Validation first
    2. ⚙️ Select models and configure parameters
    3. 🚀 Generate forecasts for all stocks
    4. 📊 View portfolio-level and individual forecasts
    5. 📥 Export results
    
    All stocks with suitability score ≥ 50 will be forecasted!
    """)
    
    if st.button("← Go to Data Validation", type="primary", use_container_width=True):
        st.switch_page("pages/4_🔍_Data_Validation.py")
