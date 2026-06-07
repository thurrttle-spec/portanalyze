"""
CAPM Analysis Page

Performs Capital Asset Pricing Model analysis to identify undervalued and overvalued stocks.
Includes Security Market Line visualization, detailed statistics, and export functionality.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import io
from datetime import datetime

# Import analysis modules
from src.data.loader import DataLoader
from src.capm.analyzer import CAPMAnalyzer
from src.capm import plot_sml

# Page configuration
st.set_page_config(
    page_title="CAPM Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stock-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 CAPM Analysis")
st.markdown("### Capital Asset Pricing Model - Stock Valuation")
st.markdown("---")

# Check if data is configured
if 'data_configured' not in st.session_state or not st.session_state.data_configured:
    st.warning("⚠️ Please configure your data on the Home page first!")
    if st.button("← Go to Home", use_container_width=True):
        st.switch_page("Home.py")
    st.stop()

# Initialize session state for CAPM results
if 'capm_results' not in st.session_state:
    st.session_state.capm_results = None
if 'capm_run' not in st.session_state:
    st.session_state.capm_run = False

# Run CAPM Analysis Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Run CAPM Analysis", type="primary", use_container_width=True):
        with st.spinner("Running CAPM analysis..."):
            try:
                # Load data
                loader = DataLoader()
                
                # Get sheet name (default or from session state)
                sheet_name = st.session_state.get('data_sheet', 'Data Gabungan Harian')
                
                stock_data, market_data = loader.load_financial_dataset(
                    file_path=st.session_state.data_file,
                    sheet_name=sheet_name
                )
                
                # Run CAPM analysis
                analyzer = CAPMAnalyzer(risk_free_rate=market_data.risk_free_rate)
                capm_results = analyzer.analyze_stocks(stock_data, market_data)
                
                # Store results
                st.session_state.capm_results = {
                    'results': capm_results,
                    'stock_data': stock_data,
                    'market_data': market_data,
                    'risk_free_rate': market_data.risk_free_rate
                }
                st.session_state.capm_run = True
                st.success("✅ CAPM Analysis completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
                st.stop()

# Display results if available
if st.session_state.capm_run and st.session_state.capm_results:
    results = st.session_state.capm_results
    capm_res = results['results']
    stock_data = results['stock_data']
    market_data = results['market_data']
    risk_free_rate = results['risk_free_rate']
    
    st.markdown("---")
    
    # Summary metrics
    st.header("📈 Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Stocks", len(capm_res.tickers))
    
    with col2:
        undervalued = capm_res.get_undervalued_stocks(threshold=st.session_state.alpha_threshold)
        st.metric("Undervalued", len(undervalued), delta="BUY", delta_color="normal")
    
    with col3:
        overvalued = capm_res.get_overvalued_stocks(threshold=st.session_state.alpha_threshold)
        st.metric("Overvalued", len(overvalued), delta="SELL", delta_color="inverse")
    
    with col4:
        hold_stocks = [t for t in capm_res.tickers 
                      if -st.session_state.alpha_threshold <= capm_res.alphas[capm_res.tickers.index(t)] <= st.session_state.alpha_threshold]
        st.metric("Hold", len(hold_stocks), delta="HOLD", delta_color="off")
    
    with col5:
        market_return = market_data.returns.mean() * 252
        st.metric("Market Return", f"{market_return*100:.2f}%")
    
    st.markdown("---")
    
    # Security Market Line
    st.header("📉 Security Market Line (SML)")
    
    # The plot_sml function uses matplotlib, but we need plotly for interactivity
    # Let's create a custom plotly SML instead
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Plot SML line
    beta_range = np.linspace(-0.5, 2.5, 100)
    market_risk_premium = market_data.returns.mean() * 252 - risk_free_rate
    sml_returns = risk_free_rate + beta_range * market_risk_premium
    
    fig.add_trace(go.Scatter(
        x=beta_range,
        y=sml_returns * 100,
        mode='lines',
        name='Security Market Line',
        line=dict(color='blue', width=2)
    ))
    
    # Classify stocks
    undervalued = []
    overvalued = []
    fairly_valued = []
    
    for i, ticker in enumerate(capm_res.tickers):
        alpha = capm_res.alphas[i]
        if alpha > st.session_state.alpha_threshold:
            undervalued.append(i)
        elif alpha < -st.session_state.alpha_threshold:
            overvalued.append(i)
        else:
            fairly_valued.append(i)
    
    # Plot undervalued stocks (green)
    if undervalued:
        fig.add_trace(go.Scatter(
            x=[capm_res.betas[i] for i in undervalued],
            y=[stock_data[capm_res.tickers[i]].returns.mean() * 252 * 100 for i in undervalued],
            mode='markers+text',
            name='Undervalued (BUY)',
            text=[capm_res.tickers[i] for i in undervalued],
            textposition='top center',
            marker=dict(color='green', size=12, line=dict(color='darkgreen', width=2)),
            hovertemplate='<b>%{text}</b><br>Beta: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>'
        ))
    
    # Plot overvalued stocks (red)
    if overvalued:
        fig.add_trace(go.Scatter(
            x=[capm_res.betas[i] for i in overvalued],
            y=[stock_data[capm_res.tickers[i]].returns.mean() * 252 * 100 for i in overvalued],
            mode='markers+text',
            name='Overvalued (SELL)',
            text=[capm_res.tickers[i] for i in overvalued],
            textposition='bottom center',
            marker=dict(color='red', size=12, line=dict(color='darkred', width=2)),
            hovertemplate='<b>%{text}</b><br>Beta: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>'
        ))
    
    # Plot fairly valued stocks (gray)
    if fairly_valued:
        fig.add_trace(go.Scatter(
            x=[capm_res.betas[i] for i in fairly_valued],
            y=[stock_data[capm_res.tickers[i]].returns.mean() * 252 * 100 for i in fairly_valued],
            mode='markers+text',
            name='Fairly Valued (HOLD)',
            text=[capm_res.tickers[i] for i in fairly_valued],
            textposition='top center',
            marker=dict(color='gray', size=12, line=dict(color='black', width=2)),
            hovertemplate='<b>%{text}</b><br>Beta: %{x:.3f}<br>Return: %{y:.2f}%<extra></extra>'
        ))
    
    # Add risk-free rate point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[risk_free_rate * 100],
        mode='markers',
        name=f'Risk-Free Rate ({risk_free_rate*100:.2f}%)',
        marker=dict(color='blue', size=15, symbol='diamond', line=dict(color='darkblue', width=2))
    ))
    
    # Add market portfolio point
    fig.add_trace(go.Scatter(
        x=[1],
        y=[market_data.returns.mean() * 252 * 100],
        mode='markers',
        name=f'Market Portfolio ({market_data.returns.mean()*252*100:.2f}%)',
        marker=dict(color='orange', size=15, symbol='square', line=dict(color='darkorange', width=2))
    ))
    
    fig.update_layout(
        title="Security Market Line (SML) - CAPM Analysis",
        xaxis_title="Beta (Systematic Risk)",
        yaxis_title="Expected Return (%)",
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, xanchor='left', yanchor='top')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Interpretation:**
    - 🟢 **Green dots (Undervalued)**: Stocks above the SML line → Higher returns than expected → **BUY**
    - 🔴 **Red dots (Overvalued)**: Stocks below the SML line → Lower returns than expected → **SELL**
    - ⚪ **Gray dots (Fairly Valued)**: Stocks near the SML line → Returns as expected → **HOLD**
    """)
    
    st.markdown("---")
    
    # Additional CAPM Visualizations
    st.header("📊 CAPM Metrics Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Beta Distribution", "📈 Alpha Distribution", "🎯 Expected vs Actual Returns", "📉 Beta vs Returns"])
    
    # Initialize figures (will be used for export too)
    fig_beta = None
    fig_alpha = None
    fig_returns = None
    fig_beta_return = None
    
    with tab1:
        st.subheader("Beta Distribution by Stock")
        
        # Sort stocks by beta
        sorted_data = sorted(zip(capm_res.tickers, capm_res.betas), key=lambda x: x[1], reverse=True)
        tickers_sorted = [x[0] for x in sorted_data]
        betas_sorted = [x[1] for x in sorted_data]
        
        fig_beta = go.Figure()
        
        # Color based on beta value
        colors = ['red' if b > 1.5 else 'orange' if b > 1.0 else 'green' for b in betas_sorted]
        
        fig_beta.add_trace(go.Bar(
            x=tickers_sorted,
            y=betas_sorted,
            marker_color=colors,
            text=[f'{b:.3f}' for b in betas_sorted],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Beta: %{y:.3f}<extra></extra>'
        ))
        
        # Add reference lines
        fig_beta.add_hline(y=1.0, line_dash="dash", line_color="blue", 
                          annotation_text="Market Beta = 1.0", annotation_position="right")
        fig_beta.add_hline(y=0, line_dash="dot", line_color="gray")
        
        fig_beta.update_layout(
            title="Beta Distribution (Systematic Risk)",
            xaxis_title="Stock Ticker",
            yaxis_title="Beta",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_beta, use_container_width=True)
        
        st.info("""
        **Beta Interpretation:**
        - 🔴 **Beta > 1.5**: Very high volatility - moves 50%+ more than market
        - 🟠 **Beta 1.0-1.5**: High volatility - moves more than market
        - 🟢 **Beta < 1.0**: Low volatility - moves less than market
        - **Beta = 1.0**: Moves exactly with market
        """)
    
    with tab2:
        st.subheader("Alpha Distribution by Stock")
        
        # Sort stocks by alpha
        sorted_data = sorted(zip(capm_res.tickers, capm_res.alphas), key=lambda x: x[1], reverse=True)
        tickers_sorted = [x[0] for x in sorted_data]
        alphas_sorted = [x[1] * 100 for x in sorted_data]  # Convert to percentage
        
        fig_alpha = go.Figure()
        
        # Color based on alpha value (undervalued/overvalued)
        colors = ['green' if a > st.session_state.alpha_threshold*100 else 
                 'red' if a < -st.session_state.alpha_threshold*100 else 
                 'gray' for a in alphas_sorted]
        
        fig_alpha.add_trace(go.Bar(
            x=tickers_sorted,
            y=alphas_sorted,
            marker_color=colors,
            text=[f'{a:.2f}%' for a in alphas_sorted],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Alpha: %{y:.2f}%<extra></extra>'
        ))
        
        # Add threshold lines
        fig_alpha.add_hline(y=st.session_state.alpha_threshold*100, line_dash="dash", 
                           line_color="green", annotation_text="Buy Threshold", 
                           annotation_position="right")
        fig_alpha.add_hline(y=-st.session_state.alpha_threshold*100, line_dash="dash", 
                           line_color="red", annotation_text="Sell Threshold", 
                           annotation_position="right")
        fig_alpha.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        fig_alpha.update_layout(
            title="Alpha Distribution (Excess Returns)",
            xaxis_title="Stock Ticker",
            yaxis_title="Alpha (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_alpha, use_container_width=True)
        
        st.info("""
        **Alpha Interpretation:**
        - 🟢 **Positive Alpha (Green)**: Undervalued - Returns above expected → **BUY**
        - 🔴 **Negative Alpha (Red)**: Overvalued - Returns below expected → **SELL**
        - ⚪ **Near Zero (Gray)**: Fairly valued - Returns as expected → **HOLD**
        """)
    
    with tab3:
        st.subheader("Expected vs Actual Returns")
        
        # Create scatter plot
        expected_returns = [capm_res.expected_returns[i] * 100 for i in range(len(capm_res.tickers))]
        actual_returns = [stock_data[ticker].returns.mean() * 252 * 100 for ticker in capm_res.tickers]
        
        fig_returns = go.Figure()
        
        # Color by classification
        colors_map = {'undervalued': 'green', 'overvalued': 'red', 'fairly_valued': 'gray'}
        
        for i, ticker in enumerate(capm_res.tickers):
            alpha = capm_res.alphas[i]
            if alpha > st.session_state.alpha_threshold:
                classification = 'undervalued'
                symbol = 'circle'
            elif alpha < -st.session_state.alpha_threshold:
                classification = 'overvalued'
                symbol = 'circle'
            else:
                classification = 'fairly_valued'
                symbol = 'circle'
            
            fig_returns.add_trace(go.Scatter(
                x=[expected_returns[i]],
                y=[actual_returns[i]],
                mode='markers+text',
                name=ticker,
                text=[ticker],
                textposition='top center',
                marker=dict(
                    color=colors_map[classification],
                    size=15,
                    line=dict(color='black', width=1),
                    symbol=symbol
                ),
                showlegend=False,
                hovertemplate=f'<b>{ticker}</b><br>Expected: %{{x:.2f}}%<br>Actual: %{{y:.2f}}%<br>Alpha: {capm_res.alphas[i]*100:.2f}%<extra></extra>'
            ))
        
        # Add 45-degree line (perfect prediction)
        min_val = min(min(expected_returns), min(actual_returns))
        max_val = max(max(expected_returns), max(actual_returns))
        fig_returns.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='blue', dash='dash', width=2),
            showlegend=True
        ))
        
        fig_returns.update_layout(
            title="Expected Return (CAPM) vs Actual Return",
            xaxis_title="Expected Return (%)",
            yaxis_title="Actual Return (%)",
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        st.info("""
        **Chart Interpretation:**
        - **Blue dashed line**: Perfect prediction (Expected = Actual)
        - **Above line**: Stock outperformed expectations (Positive Alpha) → Undervalued
        - **Below line**: Stock underperformed expectations (Negative Alpha) → Overvalued
        - **On line**: Stock performed as expected (Zero Alpha) → Fairly Valued
        """)
    
    with tab4:
        st.subheader("Beta vs Returns Relationship")
        
        # Create scatter plot with R² as color
        fig_beta_return = go.Figure()
        
        fig_beta_return.add_trace(go.Scatter(
            x=capm_res.betas,
            y=[stock_data[ticker].returns.mean() * 252 * 100 for ticker in capm_res.tickers],
            mode='markers+text',
            text=capm_res.tickers,
            textposition='top center',
            marker=dict(
                size=15,
                color=capm_res.r_squared,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="R²<br>(Fit Quality)"),
                line=dict(color='black', width=1)
            ),
            hovertemplate='<b>%{text}</b><br>Beta: %{x:.3f}<br>Return: %{y:.2f}%<br>R²: %{marker.color:.3f}<extra></extra>'
        ))
        
        fig_beta_return.update_layout(
            title="Beta vs Actual Returns (colored by R²)",
            xaxis_title="Beta (Systematic Risk)",
            yaxis_title="Actual Return (%)",
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_beta_return, use_container_width=True)
        
        st.info("""
        **R² (Color) Interpretation:**
        - **Dark colors (R² > 0.7)**: Strong relationship - Beta is very reliable
        - **Medium colors (0.4 < R² < 0.7)**: Moderate relationship - Beta is somewhat reliable
        - **Light colors (R² < 0.4)**: Weak relationship - Beta is less reliable
        
        **General Pattern:**
        Higher beta should generally correlate with higher returns (risk-return tradeoff)
        """)
    
    st.markdown("---")
    
    # Detailed Results Table
    st.header("📊 Detailed CAPM Results")
    
    # Create comprehensive results dataframe
    results_data = []
    for i, ticker in enumerate(capm_res.tickers):
        # Get actual return
        actual_return = stock_data[ticker].returns.mean() * 252
        
        # Determine recommendation
        alpha = capm_res.alphas[i]
        if alpha > st.session_state.alpha_threshold:
            recommendation = "BUY"
            valuation = "Undervalued"
        elif alpha < -st.session_state.alpha_threshold:
            recommendation = "SELL"
            valuation = "Overvalued"
        else:
            recommendation = "HOLD"
            valuation = "Fairly Valued"
        
        results_data.append({
            'Ticker': ticker,
            'Beta': capm_res.betas[i],
            'Alpha': alpha,
            'Expected Return': capm_res.expected_returns[i],
            'Actual Return': actual_return,
            'R²': capm_res.r_squared[i],
            'Valuation': valuation,
            'Recommendation': recommendation
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Display table with formatting
    st.dataframe(
        results_df.style.format({
            'Beta': '{:.4f}',
            'Alpha': '{:.4f}',
            'Expected Return': '{:.2%}',
            'Actual Return': '{:.2%}',
            'R²': '{:.4f}'
        }).background_gradient(subset=['Alpha'], cmap='RdYlGn', vmin=-0.1, vmax=0.1),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # Statistical Descriptives
    st.header("📈 Statistical Descriptives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Beta Statistics")
        beta_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
            'Value': [
                np.mean(capm_res.betas),
                np.median(capm_res.betas),
                np.std(capm_res.betas),
                np.min(capm_res.betas),
                np.max(capm_res.betas),
                np.percentile(capm_res.betas, 25),
                np.percentile(capm_res.betas, 75)
            ]
        })
        st.dataframe(
            beta_stats.style.format({'Value': '{:.4f}'}),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("""
        **Beta Interpretation:**
        - Beta < 1: Less volatile than market
        - Beta = 1: Moves with market
        - Beta > 1: More volatile than market
        """)
    
    with col2:
        st.subheader("Alpha Statistics")
        alpha_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
            'Value': [
                np.mean(capm_res.alphas),
                np.median(capm_res.alphas),
                np.std(capm_res.alphas),
                np.min(capm_res.alphas),
                np.max(capm_res.alphas),
                np.percentile(capm_res.alphas, 25),
                np.percentile(capm_res.alphas, 75)
            ]
        })
        st.dataframe(
            alpha_stats.style.format({'Value': '{:.4f}'}),
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("""
        **Alpha Interpretation:**
        - Alpha > 0: Outperforming (Undervalued)
        - Alpha = 0: Expected performance
        - Alpha < 0: Underperforming (Overvalued)
        """)
    
    st.markdown("---")
    
    # R-squared distribution
    st.subheader("📊 Model Fit Quality (R²)")
    
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Bar(
        x=capm_res.tickers,
        y=capm_res.r_squared,
        marker_color=['green' if r > 0.5 else 'orange' if r > 0.3 else 'red' for r in capm_res.r_squared],
        text=[f'{r:.3f}' for r in capm_res.r_squared],
        textposition='outside'
    ))
    
    fig_r2.update_layout(
        title="R² Values by Stock (Model Fit Quality)",
        xaxis_title="Stock Ticker",
        yaxis_title="R² Value",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig_r2, use_container_width=True)
    
    st.info("""
    **R² Interpretation:**
    - 🟢 **R² > 0.5**: Good model fit - Beta is reliable
    - 🟠 **0.3 < R² < 0.5**: Moderate fit - Beta has some uncertainty
    - 🔴 **R² < 0.3**: Poor fit - Beta is less reliable
    """)
    
    st.markdown("---")
    
    # Export Section
    st.header("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"capm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Export with multiple sheets AND charts
        excel_buffer = io.BytesIO()
        
        with st.spinner("Creating Excel file with charts..."):
            try:
                # Save Plotly charts as images first
                import plotly.io as pio
                from PIL import Image
                
                # Configure kaleido for image export
                chart_images = {}
                temp_files = []  # Keep track of temp files
                
                # Export SML chart
                try:
                    sml_img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
                    chart_images['sml'] = sml_img_bytes
                except Exception as e:
                    st.warning(f"Could not export SML chart: {e}")
                
                # Export Beta Distribution
                try:
                    beta_img_bytes = fig_beta.to_image(format="png", width=1200, height=600, scale=2)
                    chart_images['beta'] = beta_img_bytes
                except Exception as e:
                    st.warning(f"Could not export Beta chart: {e}")
                
                # Export Alpha Distribution
                try:
                    alpha_img_bytes = fig_alpha.to_image(format="png", width=1200, height=600, scale=2)
                    chart_images['alpha'] = alpha_img_bytes
                except Exception as e:
                    st.warning(f"Could not export Alpha chart: {e}")
                
                # Export Expected vs Actual
                try:
                    returns_img_bytes = fig_returns.to_image(format="png", width=1200, height=800, scale=2)
                    chart_images['returns'] = returns_img_bytes
                except Exception as e:
                    st.warning(f"Could not export Returns chart: {e}")
                
                # Export Beta vs Returns
                try:
                    beta_return_img_bytes = fig_beta_return.to_image(format="png", width=1200, height=800, scale=2)
                    chart_images['beta_return'] = beta_return_img_bytes
                except Exception as e:
                    st.warning(f"Could not export Beta vs Returns chart: {e}")
                
                # Export R² Distribution
                try:
                    r2_img_bytes = fig_r2.to_image(format="png", width=1200, height=600, scale=2)
                    chart_images['r2'] = r2_img_bytes
                except Exception as e:
                    st.warning(f"Could not export R² chart: {e}")
                
            except Exception as e:
                st.warning(f"⚠️ Chart export may not be available: {e}")
                chart_images = {}
        
        try:
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Sheet 1: Main results
                results_df.to_excel(writer, sheet_name='CAPM Results', index=False)
                
                # Sheet 2: Beta statistics
                beta_stats.to_excel(writer, sheet_name='Beta Statistics', index=False)
                
                # Sheet 3: Alpha statistics
                alpha_stats.to_excel(writer, sheet_name='Alpha Statistics', index=False)
                
                # Sheet 4: Summary
                summary_df = pd.DataFrame({
                    'Metric': [
                        'Total Stocks',
                        'Undervalued (BUY)',
                        'Fairly Valued (HOLD)',
                        'Overvalued (SELL)',
                        'Risk-Free Rate',
                        'Market Return',
                        'Alpha Threshold'
                    ],
                    'Value': [
                        len(capm_res.tickers),
                        len(undervalued),
                        len(hold_stocks),
                        len(overvalued),
                        f"{risk_free_rate*100:.2f}%",
                        f"{market_return*100:.2f}%",
                        f"{st.session_state.alpha_threshold*100:.2f}%"
                    ]
                })
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 5-10: Charts (if available)
                if chart_images:
                    try:
                        from openpyxl.drawing.image import Image as XLImage
                        import tempfile
                        import os
                        
                        # Add charts to separate sheets
                        chart_sheets = {
                            'sml': 'Chart - SML',
                            'beta': 'Chart - Beta Distribution',
                            'alpha': 'Chart - Alpha Distribution',
                            'returns': 'Chart - Expected vs Actual',
                            'beta_return': 'Chart - Beta vs Returns',
                            'r2': 'Chart - R² Distribution'
                        }
                        
                        for chart_key, sheet_name in chart_sheets.items():
                            if chart_key in chart_images:
                                # Create empty sheet
                                pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                                
                                # Get worksheet
                                ws = writer.sheets[sheet_name]
                                
                                # Save image to temporary file
                                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                                tmp.write(chart_images[chart_key])
                                tmp.close()
                                temp_files.append(tmp.name)
                                
                                # Add image to worksheet
                                img = XLImage(tmp.name)
                                ws.add_image(img, 'A1')
                        
                        st.success("✅ Charts successfully embedded in Excel!")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not embed charts in Excel: {e}")
            
            # Clean up temp files after Excel is saved
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"❌ Error creating Excel file: {e}")
            # Still try to provide basic Excel without charts
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='CAPM Results', index=False)
                beta_stats.to_excel(writer, sheet_name='Beta Statistics', index=False)
                alpha_stats.to_excel(writer, sheet_name='Alpha Statistics', index=False)
        
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📊 Download Excel (with Charts)",
            data=excel_data,
            file_name=f"capm_analysis_with_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # Statistics summary
        stats_buffer = io.StringIO()
        stats_buffer.write("=== CAPM ANALYSIS SUMMARY ===\n\n")
        stats_buffer.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        stats_buffer.write(f"Total Stocks: {len(capm_res.tickers)}\n")
        stats_buffer.write(f"Undervalued (BUY): {len(undervalued)}\n")
        stats_buffer.write(f"Fairly Valued (HOLD): {len(hold_stocks)}\n")
        stats_buffer.write(f"Overvalued (SELL): {len(overvalued)}\n\n")
        stats_buffer.write(f"Risk-Free Rate: {risk_free_rate*100:.2f}%\n")
        stats_buffer.write(f"Market Return: {market_return*100:.2f}%\n")
        stats_buffer.write(f"Alpha Threshold: {st.session_state.alpha_threshold*100:.2f}%\n\n")
        stats_buffer.write("=== BETA STATISTICS ===\n")
        stats_buffer.write(f"Mean: {np.mean(capm_res.betas):.4f}\n")
        stats_buffer.write(f"Median: {np.median(capm_res.betas):.4f}\n")
        stats_buffer.write(f"Std Dev: {np.std(capm_res.betas):.4f}\n\n")
        stats_buffer.write("=== ALPHA STATISTICS ===\n")
        stats_buffer.write(f"Mean: {np.mean(capm_res.alphas):.4f}\n")
        stats_buffer.write(f"Median: {np.median(capm_res.alphas):.4f}\n")
        stats_buffer.write(f"Std Dev: {np.std(capm_res.alphas):.4f}\n")
        
        stats_data = stats_buffer.getvalue()
        
        st.download_button(
            label="📝 Download Summary",
            data=stats_data,
            file_name=f"capm_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.success("""
    ✅ **Export files include:**
    
    **CSV**: Simple results table
    
    **Excel**: 
    - Sheet 1: CAPM Results (detailed table)
    - Sheet 2: Beta Statistics
    - Sheet 3: Alpha Statistics  
    - Sheet 4: Summary
    - Sheet 5: Chart - Security Market Line
    - Sheet 6: Chart - Beta Distribution
    - Sheet 7: Chart - Alpha Distribution
    - Sheet 8: Chart - Expected vs Actual Returns
    - Sheet 9: Chart - Beta vs Returns
    - Sheet 10: Chart - R² Distribution
    
    **TXT**: Summary statistics in plain text
    """)
    
    st.markdown("---")
    
    # Next steps
    st.header("🎯 Next Steps")
    
    st.info(f"""
    ✅ **CAPM Analysis Complete!**
    
    **Key Findings:**
    - Found **{len(undervalued)} undervalued stocks** (BUY candidates)
    - Found **{len(overvalued)} overvalued stocks** (SELL candidates)
    - Found **{len(hold_stocks)} fairly valued stocks** (HOLD)
    
    **Recommended Next Step:**
    Navigate to **💼 Portfolio Optimization** to build an optimal portfolio using the {len(undervalued)} undervalued stocks.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Home", use_container_width=True):
            st.switch_page("Home.py")
    with col2:
        if st.button("Next: Portfolio →", type="primary", use_container_width=True):
            st.switch_page("pages/2_💼_Portfolio_Optimization.py")

else:
    # Instructions when no results
    st.info("""
    👋 **Welcome to CAPM Analysis!**
    
    This module performs Capital Asset Pricing Model analysis to identify:
    - **Undervalued stocks** (Alpha > threshold) → BUY recommendations
    - **Overvalued stocks** (Alpha < -threshold) → SELL recommendations
    - **Fairly valued stocks** (Alpha near 0) → HOLD recommendations
    
    **What you'll get:**
    - Security Market Line visualization
    - Detailed CAPM metrics (Beta, Alpha, R²)
    - Statistical descriptives
    - Export to CSV/Excel
    
    Click **"Run CAPM Analysis"** above to start!
    """)
    
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("Home.py")

# Footer
st.markdown("---")
st.caption("📊 CAPM Analysis | Capital Asset Pricing Model")
