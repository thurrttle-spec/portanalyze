"""
Portfolio Optimization Page

Builds optimal portfolio using Black-Litterman model with investor views from CAPM analysis.
Includes portfolio metrics, allocation charts, and export functionality.
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
from src.black_litterman import (
    calculate_equilibrium_returns,
    calculate_posterior_returns,
    InvestorViews,
    optimize_portfolio,
    calculate_portfolio_metrics
)

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="💼",
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
</style>
""", unsafe_allow_html=True)

# Title
st.title("💼 Portfolio Optimization")
st.markdown("### Black-Litterman Model with CAPM Views")
st.markdown("---")

# Check if data is configured
if 'data_configured' not in st.session_state or not st.session_state.data_configured:
    st.warning("⚠️ Please configure your data on the Home page first!")
    if st.button("← Go to Home", use_container_width=True):
        st.switch_page("Home.py")
    st.stop()

# Check if CAPM analysis was run
if 'capm_results' not in st.session_state or not st.session_state.capm_run:
    st.warning("⚠️ Please run CAPM Analysis first to identify undervalued stocks!")
    if st.button("← Go to CAPM Analysis", use_container_width=True):
        st.switch_page("pages/1_📊_CAPM_Analysis.py")
    st.stop()

# Portfolio Configuration Section
st.header("⚙️ Portfolio Configuration")

# Get undervalued stocks count
capm_data = st.session_state.capm_results
capm_res = capm_data['results']
undervalued = capm_res.get_undervalued_stocks(threshold=st.session_state.alpha_threshold)
n_undervalued = len(undervalued)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Undervalued Stocks Available", n_undervalued)
    st.caption(f"Alpha threshold: {st.session_state.alpha_threshold*100:.1f}%")

with col2:
    # Portfolio size selector
    if n_undervalued > 0:
        # Ensure min_value is always less than max_value
        min_portfolio = min(3, n_undervalued)
        max_portfolio = max(min_portfolio + 1, min(30, n_undervalued))
        default_portfolio = min(10, n_undervalued)
        
        portfolio_size = st.slider(
            "Number of Stocks in Portfolio",
            min_value=min_portfolio,
            max_value=max_portfolio,
            value=default_portfolio,
            step=1,
            help=f"Select how many stocks to include in the optimized portfolio (max {n_undervalued} undervalued stocks available)"
        )
    else:
        st.error("No undervalued stocks available!")
        st.stop()

with col3:
    risk_aversion = st.slider(
        "Risk Aversion (λ)",
        min_value=1.0,
        max_value=5.0,
        value=st.session_state.get('risk_aversion', 2.5),
        step=0.1,
        help="Higher values = more conservative (lower risk)"
    )

st.info(f"""
📊 **Configuration Summary**:
- Portfolio will include **{portfolio_size} stocks** from **{n_undervalued} undervalued** stocks
- Risk aversion parameter: **{risk_aversion:.1f}** ({"Conservative" if risk_aversion > 3 else "Moderate" if risk_aversion > 2 else "Aggressive"})
- Optimization method: **Black-Litterman** with CAPM views
""")

# Data Preview Section
st.header("👀 Preview Your Data")

col1, col2 = st.columns([2, 1])
with col1:
    st.write("Review the uploaded data that will be used for portfolio optimization.")
with col2:
    preview_button = st.button("📊 Preview Uploaded Data", type="secondary", use_container_width=True)

if preview_button:
    with st.spinner("Loading your uploaded data..."):
        try:
            # Use data from CAPM results (which already loaded your uploaded data)
            if 'capm_results' in st.session_state and st.session_state.capm_results:
                capm_data = st.session_state.capm_results
                stock_data = capm_data['stock_data']
                market_data = capm_data['market_data']
                
                n_stocks = len(stock_data)
                n_observations = len(next(iter(stock_data.values())).prices)
                
                st.success("✅ Your uploaded data loaded successfully!")
                
                # Store in session for display
                st.session_state.preview_portfolio_data = {
                    'stock_data': stock_data,
                    'market_data': market_data,
                    'n_stocks': n_stocks,
                    'n_observations': n_observations
                }
            else:
                st.error("❌ Please run CAPM Analysis first to load your data!")
            
        except Exception as e:
            st.error(f"❌ Error loading your data: {str(e)}")
            st.exception(e)

# Display data exploration if previewed
if 'preview_portfolio_data' in st.session_state:
    st.markdown("---")
    st.subheader("📊 Your Uploaded Data Overview")
    
    preview_data = st.session_state.preview_portfolio_data
    stock_data_dict = preview_data['stock_data']
    market_data_obj = preview_data['market_data']
    n_stocks = preview_data['n_stocks']
    n_observations = preview_data['n_observations']
    
    # Summary Statistics
    with st.expander("📈 Data Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", n_stocks)
        
        with col2:
            st.metric("Data Points", n_observations)
        
        with col3:
            first_stock = next(iter(stock_data_dict.values()))
            date_range = f"{first_stock.prices.index[0].strftime('%Y-%m-%d')} to {first_stock.prices.index[-1].strftime('%Y-%m-%d')}"
            st.metric("Date Range", "")
            st.caption(date_range)
        
        with col4:
            market_return = market_data_obj.returns.mean() * 252
            st.metric("Market Return (Annual)", f"{market_return*100:.2f}%")
    
    # Stock List
    with st.expander("📋 Stock Tickers in Your Data"):
        tickers_list = list(stock_data_dict.keys())
        
        # Display in columns
        n_cols = 5
        cols = st.columns(n_cols)
        for i, ticker in enumerate(tickers_list):
            with cols[i % n_cols]:
                st.write(f"• {ticker}")
    
    # Price Trends (first 10 stocks)
    with st.expander("📉 Price Trends (Sample)"):
        fig = go.Figure()
        
        stock_list = list(stock_data_dict.keys())[:10]
        for ticker in stock_list:
            stock = stock_data_dict[ticker]
            fig.add_trace(go.Scatter(
                x=stock.prices.index,
                y=stock.prices.values,
                mode='lines',
                name=ticker,
                hovertemplate=f'{ticker}<br>Price: %{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Stock Prices Over Time (First 10 Stocks)",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if n_stocks > 10:
            st.caption(f"📝 Showing first 10 of {n_stocks} stocks from your uploaded data")
    
    # Volatility Analysis
    with st.expander("⚡ Volatility Analysis"):
        volatilities = {}
        for ticker, stock in stock_data_dict.items():
            vol = stock.returns.std() * np.sqrt(252)
            volatilities[ticker] = vol
        
        sorted_vols = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)
        
        # Show top 15 most volatile
        top_15 = sorted_vols[:15]
        tickers = [x[0] for x in top_15]
        vols = [x[1] * 100 for x in top_15]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tickers,
            y=vols,
            marker_color=['indianred' if v > np.mean(vols) else 'steelblue' for v in vols],
            text=[f'{v:.1f}%' for v in vols],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Top 15 Most Volatile Stocks (Annualized)",
            xaxis_title="Stock Ticker",
            yaxis_title="Volatility (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Highest Volatility", f"{sorted_vols[0][0]}: {sorted_vols[0][1]*100:.1f}%")
        col2.metric("Lowest Volatility", f"{sorted_vols[-1][0]}: {sorted_vols[-1][1]*100:.1f}%")
        col3.metric("Average Volatility", f"{np.mean(list(volatilities.values()))*100:.1f}%")
    
    st.info("👆 This is your uploaded data that will be used for portfolio optimization!")

st.markdown("---")

st.markdown("---")

# Initialize session state for portfolio results
if 'portfolio_results' not in st.session_state:
    st.session_state.portfolio_results = None
if 'portfolio_run' not in st.session_state:
    st.session_state.portfolio_run = False

# Run Portfolio Optimization Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True):
        with st.spinner("Optimizing portfolio using Black-Litterman model..."):
            try:
                # Get CAPM results
                capm_data = st.session_state.capm_results
                capm_res = capm_data['results']
                stock_data = capm_data['stock_data']
                market_data = capm_data['market_data']
                risk_free_rate = capm_data['risk_free_rate']
                
                # Get undervalued stocks
                undervalued = capm_res.get_undervalued_stocks(threshold=st.session_state.alpha_threshold)
                
                if len(undervalued) == 0:
                    st.error("❌ No undervalued stocks found! Try lowering the alpha threshold.")
                    st.stop()
                
                # Build portfolio from undervalued stocks
                n_portfolio = min(portfolio_size, len(undervalued))
                portfolio_tickers = undervalued[:n_portfolio]
                
                st.info(f"ℹ️ Building portfolio with {n_portfolio} undervalued stocks (from {len(undervalued)} available)")
                
                portfolio_stocks = {ticker: stock_data[ticker] for ticker in portfolio_tickers}
                
                # Calculate returns
                returns_data = {}
                for ticker, stock in portfolio_stocks.items():
                    returns = stock.prices.pct_change().dropna()
                    returns_data[ticker] = returns
                
                returns_df = pd.DataFrame(returns_data)
                
                # Calculate covariance matrix (annualized)
                cov_matrix = returns_df.cov().values * 252
                
                # Equal market cap weights (as proxy)
                market_weights = np.ones(len(portfolio_tickers)) / len(portfolio_tickers)
                
                # Calculate equilibrium returns
                risk_aversion_param = risk_aversion
                equilibrium_returns = calculate_equilibrium_returns(
                    market_weights,
                    cov_matrix,
                    risk_aversion=risk_aversion_param,
                    risk_free_rate=risk_free_rate
                )
                
                # Add investor views (top stocks by alpha)
                views = InvestorViews(n_assets=len(portfolio_tickers))
                
                # Sort by alpha and take top 3
                alpha_data = [(i, capm_res.alphas[capm_res.tickers.index(t)]) 
                             for i, t in enumerate(portfolio_tickers)]
                top_alphas = sorted(alpha_data, key=lambda x: x[1], reverse=True)[:min(3, len(portfolio_tickers))]
                
                # Store view details for display
                view_details = []
                
                for idx, alpha in top_alphas:
                    ticker = portfolio_tickers[idx]
                    ticker_idx_capm = capm_res.tickers.index(ticker)
                    expected_return = capm_res.expected_returns[ticker_idx_capm]
                    expected_return = np.clip(expected_return, -0.40, 0.40)
                    r_squared = capm_res.r_squared[ticker_idx_capm]
                    confidence = min(0.9, 0.5 + r_squared * 0.5)
                    
                    views.add_absolute_view(idx, expected_return, confidence)
                    
                    # Store for display
                    view_details.append({
                        'ticker': ticker,
                        'alpha': alpha,
                        'expected_return': expected_return,
                        'r_squared': r_squared,
                        'confidence': confidence
                    })
                
                # Calculate posterior returns
                tau = 0.025
                posterior_returns, posterior_cov = calculate_posterior_returns(
                    equilibrium_returns,
                    cov_matrix,
                    tau,
                    views.view_matrix,
                    views.view_returns,
                    views.view_uncertainty
                )
                
                # Optimize portfolio
                optimal_weights = optimize_portfolio(
                    posterior_returns,
                    posterior_cov,
                    risk_free_rate
                )
                
                # Calculate metrics
                portfolio_metrics = calculate_portfolio_metrics(
                    optimal_weights,
                    posterior_returns,
                    posterior_cov,
                    risk_free_rate
                )
                
                # Store results
                st.session_state.portfolio_results = {
                    'tickers': portfolio_tickers,
                    'weights': optimal_weights,
                    'returns': posterior_returns,
                    'metrics': portfolio_metrics,
                    'equilibrium_returns': equilibrium_returns,
                    'market_weights': market_weights,
                    'n_views': len(top_alphas),
                    'view_details': view_details,  # Add view details
                    'stock_data': stock_data,
                    'returns_df': returns_df,
                    'cov_matrix': cov_matrix,
                    'risk_aversion': risk_aversion,  # Store the risk aversion used
                    'portfolio_size': n_portfolio  # Store the portfolio size used
                }
                st.session_state.portfolio_run = True
                st.success("✅ Portfolio optimized successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during optimization: {str(e)}")
                st.exception(e)
                st.stop()

# Display results if available
if st.session_state.portfolio_run and st.session_state.portfolio_results:
    results = st.session_state.portfolio_results
    
    st.markdown("---")
    
    # Portfolio Metrics Summary
    st.header("📊 Portfolio Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Return",
            f"{results['metrics']['return']*100:.2f}%",
            delta="Annual"
        )
    
    with col2:
        st.metric(
            "Volatility (Risk)",
            f"{results['metrics']['volatility']*100:.2f}%",
            delta="Annual"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{results['metrics']['sharpe_ratio']:.3f}",
            delta="Risk-Adjusted"
        )
    
    with col4:
        st.metric(
            "Number of Holdings",
            len(results['tickers'])
        )
    
    st.markdown("---")
    
    # Black-Litterman Investor Views
    st.header("🎯 Black-Litterman Investor Views")
    
    st.info("""
    **What are Investor Views?**
    
    In Black-Litterman model, investor views represent your beliefs about expected returns of specific assets.
    These views are combined with market equilibrium to produce optimal portfolio weights.
    """)
    
    # Display view determination methodology
    with st.expander("📖 How Were Views Determined?", expanded=True):
        st.markdown("""
        ### View Selection Methodology
        
        The investor views in this optimization were automatically determined from **CAPM analysis results**:
        
        1. **Stock Selection**: Top 3 stocks ranked by **Alpha** (highest to lowest)
           - Higher alpha = more undervalued = stronger view
        
        2. **Expected Return**: From **CAPM expected return** formula
           - Formula: `Risk-Free Rate + Beta × Market Risk Premium`
           - Capped at ±40% for realistic bounds
        
        3. **Confidence Level**: Based on **R² (model fit quality)**
           - Formula: `Confidence = 0.5 + (R² × 0.5)`
           - Higher R² = more reliable Beta = higher confidence
           - Range: 0.5 to 0.9 (50% to 90% confidence)
        
        ### Why This Approach?
        
        - ✅ **Objective**: Based on statistical analysis (CAPM), not subjective opinions
        - ✅ **Data-Driven**: Uses historical return patterns
        - ✅ **Confidence-Weighted**: More weight to stocks with better model fit
        - ✅ **Conservative**: Limited to top 3 views to avoid over-specification
        """)
    
    # Display the actual views used
    st.subheader("📊 Views Used in This Optimization")
    
    if 'view_details' in results:
        views_df = pd.DataFrame(results['view_details'])
        
        # Create display dataframe
        views_display = pd.DataFrame({
            'Rank': range(1, len(views_df) + 1),
            'Ticker': views_df['ticker'],
            'Alpha (%)': views_df['alpha'] * 100,
            'Expected Return (%)': views_df['expected_return'] * 100,
            'R² (Fit Quality)': views_df['r_squared'],
            'Confidence Level': views_df['confidence']
        })
        
        st.dataframe(
            views_display.style.format({
                'Alpha (%)': '{:.3f}%',
                'Expected Return (%)': '{:.2f}%',
                'R² (Fit Quality)': '{:.3f}',
                'Confidence Level': '{:.1%}'
            }).background_gradient(subset=['Alpha (%)'], cmap='Greens')
            .background_gradient(subset=['Confidence Level'], cmap='Blues'),
            use_container_width=True
        )
        
        # Interpretation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Views", len(views_df))
            st.caption("Limited to top 3 for stability")
        
        with col2:
            avg_confidence = views_df['confidence'].mean()
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
            st.caption("Based on R² quality")
        
        with col3:
            avg_return = views_df['expected_return'].mean() * 100
            st.metric("Avg Expected Return", f"{avg_return:.2f}%")
            st.caption("From CAPM formula")
        
        st.markdown("---")
        
        # Visual comparison
        st.subheader("📈 View Impact Visualization")
        
        # Create comparison chart
        fig_views = go.Figure()
        
        # Equilibrium returns (market view)
        fig_views.add_trace(go.Bar(
            x=results['tickers'],
            y=results['equilibrium_returns'] * 100,
            name='Equilibrium Returns (Market)',
            marker_color='lightblue',
            opacity=0.6
        ))
        
        # Posterior returns (after incorporating views)
        fig_views.add_trace(go.Bar(
            x=results['tickers'],
            y=results['returns'] * 100,
            name='Posterior Returns (With Views)',
            marker_color='darkblue'
        ))
        
        # Highlight stocks with views
        for view in results['view_details']:
            fig_views.add_annotation(
                x=view['ticker'],
                y=view['expected_return'] * 100,
                text='VIEW',
                showarrow=True,
                arrowhead=2,
                arrowcolor='red',
                ax=0,
                ay=-40
            )
        
        fig_views.update_layout(
            title="Returns: Market Equilibrium vs. With Investor Views",
            xaxis_title="Stock Ticker",
            yaxis_title="Expected Return (%)",
            height=500,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_views, use_container_width=True)
        
        st.info("""
        **Chart Interpretation:**
        - **Light Blue**: Market equilibrium returns (before views)
        - **Dark Blue**: Posterior returns (after incorporating investor views)
        - **Red Arrows**: Stocks where views were applied
        
        Black-Litterman blends market equilibrium with investor views, weighted by confidence.
        """)
    
    st.markdown("---")
    
    # Portfolio Allocation
    st.header("📈 Portfolio Allocation")
    
    # Create allocation dataframe
    allocation_df = pd.DataFrame({
        'Ticker': results['tickers'],
        'Weight (%)': results['weights'] * 100,
        'Expected Return (%)': results['returns'] * 100,
        'Dollar Amount ($)': results['weights'] * 1000000  # Assuming $1M portfolio
    })
    
    # Sort by weight
    allocation_df = allocation_df.sort_values('Weight (%)', ascending=False)
    
    # Display table
    st.dataframe(
        allocation_df.style.format({
            'Weight (%)': '{:.2f}%',
            'Expected Return (%)': '{:.2f}%',
            'Dollar Amount ($)': '${:,.2f}'
        }).background_gradient(subset=['Weight (%)'], cmap='Blues'),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # Visualizations
    st.header("📊 Portfolio Visualization")
    
    tab1, tab2, tab3 = st.tabs(["🥧 Pie Chart", "📊 Bar Chart", "🎯 Risk-Return Plot"])
    
    with tab1:
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=results['tickers'],
            values=results['weights'],
            hole=0.4,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(line=dict(color='white', width=2))
        )])
        
        fig_pie.update_layout(
            title="Portfolio Allocation (Donut Chart)",
            height=600,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        # Bar chart
        fig_bar = go.Figure()
        
        fig_bar.add_trace(go.Bar(
            x=allocation_df['Ticker'],
            y=allocation_df['Weight (%)'],
            text=allocation_df['Weight (%)'].apply(lambda x: f'{x:.2f}%'),
            textposition='outside',
            marker_color='steelblue',
            hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>'
        ))
        
        fig_bar.update_layout(
            title="Portfolio Weights by Stock",
            xaxis_title="Stock Ticker",
            yaxis_title="Weight (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        # Risk-Return scatter
        fig_risk = go.Figure()
        
        # Calculate individual stock metrics
        for i, ticker in enumerate(results['tickers']):
            stock_returns = results['returns_df'][ticker]
            stock_return = stock_returns.mean() * 252
            stock_vol = stock_returns.std() * np.sqrt(252)
            
            fig_risk.add_trace(go.Scatter(
                x=[stock_vol * 100],
                y=[stock_return * 100],
                mode='markers+text',
                name=ticker,
                text=[ticker],
                textposition='top center',
                marker=dict(
                    size=results['weights'][i] * 1000,  # Size by weight
                    color='steelblue',
                    line=dict(color='darkblue', width=1)
                ),
                showlegend=False,
                hovertemplate=f'<b>{ticker}</b><br>Return: %{{y:.2f}}%<br>Volatility: %{{x:.2f}}%<br>Weight: {results["weights"][i]*100:.2f}%<extra></extra>'
            ))
        
        # Add portfolio point
        fig_risk.add_trace(go.Scatter(
            x=[results['metrics']['volatility'] * 100],
            y=[results['metrics']['return'] * 100],
            mode='markers+text',
            name='Portfolio',
            text=['PORTFOLIO'],
            textposition='top center',
            marker=dict(
                size=30,
                color='red',
                symbol='diamond',
                line=dict(color='darkred', width=2)
            ),
            showlegend=True,
            hovertemplate=f'<b>Portfolio</b><br>Return: {results["metrics"]["return"]*100:.2f}%<br>Volatility: {results["metrics"]["volatility"]*100:.2f}%<extra></extra>'
        ))
        
        fig_risk.update_layout(
            title="Risk-Return Profile (Bubble size = Weight)",
            xaxis_title="Volatility / Risk (%)",
            yaxis_title="Expected Return (%)",
            height=600,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.info("""
        **Chart Interpretation:**
        - **Bubble size**: Proportional to stock weight in portfolio
        - **Red diamond**: Overall portfolio risk-return
        - **Diversification benefit**: Portfolio typically has lower risk than weighted average of stocks
        """)
    
    st.markdown("---")
    
    # Portfolio Insights
    st.header("💡 Portfolio Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Concentration Analysis")
        
        # Top holdings
        top_3_weight = allocation_df.head(3)['Weight (%)'].sum()
        top_5_weight = allocation_df.head(5)['Weight (%)'].sum()
        
        st.metric("Top 3 Holdings", f"{top_3_weight:.1f}%")
        st.metric("Top 5 Holdings", f"{top_5_weight:.1f}%")
        
        if top_3_weight > 60:
            st.warning("⚠️ **High concentration**: Top 3 stocks represent > 60% of portfolio")
        elif top_5_weight > 75:
            st.info("ℹ️ **Moderate concentration**: Top 5 stocks represent > 75% of portfolio")
        else:
            st.success("✅ **Well diversified**: Portfolio weight is well distributed")
    
    with col2:
        st.subheader("Comparison with Equal Weight")
        
        # Equal weight portfolio
        equal_weights = np.ones(len(results['tickers'])) / len(results['tickers'])
        equal_return = np.dot(equal_weights, results['returns'])
        equal_vol = np.sqrt(np.dot(equal_weights, np.dot(results['cov_matrix'], equal_weights)))
        equal_sharpe = (equal_return - st.session_state.get('risk_free_rate', 0.0493)) / equal_vol
        
        st.metric("Optimized Sharpe", f"{results['metrics']['sharpe_ratio']:.3f}")
        st.metric("Equal-Weight Sharpe", f"{equal_sharpe:.3f}")
        
        improvement = ((results['metrics']['sharpe_ratio'] - equal_sharpe) / equal_sharpe) * 100
        
        if improvement > 0:
            st.success(f"✅ **{improvement:.1f}% improvement** over equal-weight portfolio")
        else:
            st.info("ℹ️ Equal-weight portfolio performs similarly")
    
    st.markdown("---")
    
    # Export Section
    st.header("📥 Export Portfolio")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv_buffer = io.StringIO()
        allocation_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"portfolio_allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel Export with charts
        excel_buffer = io.BytesIO()
        
        with st.spinner("Creating Excel with charts..."):
            # Export charts
            chart_images = {}
            temp_files = []
            
            try:
                chart_images['pie'] = fig_pie.to_image(format="png", width=1200, height=800, scale=2)
                chart_images['bar'] = fig_bar.to_image(format="png", width=1200, height=600, scale=2)
                chart_images['risk'] = fig_risk.to_image(format="png", width=1200, height=800, scale=2)
            except Exception as e:
                st.warning(f"Could not export charts: {e}")
                chart_images = {}
        
        try:
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Sheet 1: Portfolio allocation
                allocation_df.to_excel(writer, sheet_name='Portfolio Allocation', index=False)
                
                # Sheet 2: Metrics
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Expected Return (Annual)',
                        'Volatility (Annual)',
                        'Sharpe Ratio',
                        'Number of Holdings',
                        'Risk Aversion Parameter',
                        'Number of Views',
                        'Top 3 Concentration',
                        'Top 5 Concentration'
                    ],
                    'Value': [
                        f"{results['metrics']['return']*100:.2f}%",
                        f"{results['metrics']['volatility']*100:.2f}%",
                        f"{results['metrics']['sharpe_ratio']:.3f}",
                        len(results['tickers']),
                        results.get('risk_aversion', 'N/A'),
                        results['n_views'],
                        f"{top_3_weight:.2f}%",
                        f"{top_5_weight:.2f}%"
                    ]
                })
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Add charts
                if chart_images:
                    from openpyxl.drawing.image import Image as XLImage
                    import tempfile
                    import os
                    
                    for chart_key, sheet_name in [('pie', 'Chart - Pie'), ('bar', 'Chart - Bar'), ('risk', 'Chart - Risk-Return')]:
                        if chart_key in chart_images:
                            pd.DataFrame().to_excel(writer, sheet_name=sheet_name, index=False)
                            ws = writer.sheets[sheet_name]
                            
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                            tmp.write(chart_images[chart_key])
                            tmp.close()
                            temp_files.append(tmp.name)
                            
                            img = XLImage(tmp.name)
                            ws.add_image(img, 'A1')
            
            # Clean up
            for tmp_path in temp_files:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
        except Exception as e:
            st.warning(f"Chart embedding failed: {e}")
        
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name=f"portfolio_with_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # Summary text
        summary_buffer = io.StringIO()
        summary_buffer.write("=== PORTFOLIO OPTIMIZATION SUMMARY ===\n\n")
        summary_buffer.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        summary_buffer.write(f"Expected Return: {results['metrics']['return']*100:.2f}%\n")
        summary_buffer.write(f"Volatility: {results['metrics']['volatility']*100:.2f}%\n")
        summary_buffer.write(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.3f}\n\n")
        summary_buffer.write(f"Number of Holdings: {len(results['tickers'])}\n\n")
        summary_buffer.write("=== ALLOCATION ===\n")
        for _, row in allocation_df.iterrows():
            summary_buffer.write(f"{row['Ticker']}: {row['Weight (%)']:.2f}%\n")
        
        summary_data = summary_buffer.getvalue()
        
        st.download_button(
            label="📝 Download Summary",
            data=summary_data,
            file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.success("""
    ✅ **Export includes:**
    - CSV: Simple allocation table
    - Excel: Allocation + Metrics + 3 Charts
    - TXT: Quick summary
    """)
    
    st.markdown("---")
    
    # Next steps
    st.header("🎯 Next Steps")
    
    st.info(f"""
    ✅ **Portfolio Optimized!**
    
    **Key Results:**
    - Built portfolio with **{len(results['tickers'])} stocks**
    - Expected Return: **{results['metrics']['return']*100:.2f}%**
    - Sharpe Ratio: **{results['metrics']['sharpe_ratio']:.3f}**
    
    **Recommended Next Step:**
    Navigate to **⚠️ Crisis Scenarios** to test how this portfolio performs under stress conditions.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to CAPM", use_container_width=True):
            st.switch_page("pages/1_📊_CAPM_Analysis.py")
    with col2:
        if st.button("Next: Crisis Scenarios →", type="primary", use_container_width=True, disabled=True):
            st.info("Coming soon!")

else:
    # Instructions
    st.info("""
    👋 **Welcome to Portfolio Optimization!**
    
    This module uses the **Black-Litterman model** to build an optimal portfolio:
    - Uses undervalued stocks from CAPM analysis
    - Incorporates investor views based on alpha
    - Optimizes for maximum Sharpe ratio
    - Provides detailed allocation and metrics
    
    **What you'll get:**
    - Optimal portfolio weights
    - Expected return and risk metrics
    - Allocation visualizations (pie, bar, risk-return)
    - Concentration analysis
    - Export to CSV/Excel with charts
    
    Click **"Optimize Portfolio"** above to start!
    """)
    
    if st.button("← Go to CAPM Analysis", use_container_width=True):
        st.switch_page("pages/1_📊_CAPM_Analysis.py")

# Footer
st.markdown("---")
st.caption("💼 Portfolio Optimization | Black-Litterman Model")
