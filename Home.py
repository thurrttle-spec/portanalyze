"""
Portfolio Analysis Dashboard - Home Page

Multi-page dashboard for comprehensive financial analysis.
Navigate through different analysis modules using the sidebar.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

def combine_uploaded_files(stock_file, market_file, risk_free_rate=None, risk_free_file=None):
    """
    Combine separate uploaded files into a single dataset format.
    
    Args:
        stock_file: Uploaded stock prices file (CSV/Excel)
        market_file: Uploaded market index file (CSV/Excel)
        risk_free_rate: Risk-free rate as decimal (constant value) or None
        risk_free_file: Uploaded BI Rate file (CSV/Excel) or None
        
    Returns:
        Path to temporary combined Excel file
    """
    # Read stock data
    if stock_file.name.endswith('.csv'):
        stock_df = pd.read_csv(stock_file)
    else:
        stock_df = pd.read_excel(stock_file)
    
    # Read market data
    if market_file.name.endswith('.csv'):
        market_df = pd.read_csv(market_file)
    else:
        market_df = pd.read_excel(market_file)
    
    # Ensure Date columns are datetime
    stock_df.iloc[:, 0] = pd.to_datetime(stock_df.iloc[:, 0])
    market_df.iloc[:, 0] = pd.to_datetime(market_df.iloc[:, 0])
    
    # Merge on date
    date_col_stock = stock_df.columns[0]
    date_col_market = market_df.columns[0]
    
    # Rename date columns to match
    stock_df = stock_df.rename(columns={date_col_stock: 'Date'})
    market_df = market_df.rename(columns={date_col_market: 'Date'})
    
    # Merge stock and market dataframes
    combined_df = pd.merge(stock_df, market_df, on='Date', how='inner')
    
    # Handle BI Rate data
    if risk_free_file is not None:
        # Read BI Rate file
        if risk_free_file.name.endswith('.csv'):
            bi_rate_df = pd.read_csv(risk_free_file)
        else:
            bi_rate_df = pd.read_excel(risk_free_file)
        
        # Ensure Date column is datetime
        bi_rate_df.iloc[:, 0] = pd.to_datetime(bi_rate_df.iloc[:, 0])
        
        # Rename columns
        date_col_bi = bi_rate_df.columns[0]
        rate_col_bi = bi_rate_df.columns[1]
        bi_rate_df = bi_rate_df.rename(columns={date_col_bi: 'Date', rate_col_bi: 'BI_Rate'})
        
        # Check if BI_Rate is in percentage (values > 1) and convert to decimal
        if bi_rate_df['BI_Rate'].mean() > 1:
            bi_rate_df['BI_Rate'] = bi_rate_df['BI_Rate'] / 100
        
        # Merge with combined data
        combined_df = pd.merge(combined_df, bi_rate_df[['Date', 'BI_Rate']], on='Date', how='inner')
    else:
        # Use constant risk-free rate
        combined_df['BI_Rate'] = risk_free_rate
    
    # Create temporary Excel file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    
    with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
        combined_df.to_excel(writer, sheet_name='Data Gabungan Harian', index=False)
    
    return temp_file.name

# Page configuration
st.set_page_config(
    page_title="Portfolio Analysis - Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏠 Stock Portfolio Analysis</h1>', unsafe_allow_html=True)
st.markdown("### Multi-Step Financial Analysis Dashboard")
st.markdown("---")

# Welcome message
st.info("""
👋 **Welcome to the Portfolio Analysis Dashboard!**

This dashboard provides comprehensive financial analysis for energy sector stocks through 4 specialized modules.
Upload your data below, then navigate through each analysis step using the sidebar menu.
""")

st.markdown("---")

# Step-by-step guide
st.header("📋 Analysis Workflow")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="step-card">
        <h3>📊 Step 1: CAPM Analysis</h3>
        <p>Identify undervalued and overvalued stocks using the Capital Asset Pricing Model</p>
        <ul>
            <li>Security Market Line visualization</li>
            <li>Beta, Alpha, R² for each stock</li>
            <li>Statistical descriptives</li>
            <li>Export results to CSV/Excel</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-card">
        <h3>⚠️ Step 3: Crisis Scenarios</h3>
        <p>Evaluate portfolio resilience under geopolitical crises</p>
        <ul>
            <li>Monte Carlo simulation</li>
            <li>Value at Risk (VaR)</li>
            <li>Investment recommendations</li>
            <li>Export crisis analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
        <h3>💼 Step 2: Portfolio Optimization</h3>
        <p>Build an optimal portfolio using Black-Litterman model</p>
        <ul>
            <li>Optimal asset weights</li>
            <li>Expected return & volatility</li>
            <li>Sharpe ratio</li>
            <li>Export portfolio details</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-card">
        <h3>📈 Step 4: Forecasting</h3>
        <p>Predict future prices and returns using ML models</p>
        <ul>
            <li>LSTM, ARIMA, GARCH models</li>
            <li>Price & volatility forecasts</li>
            <li>Performance metrics</li>
            <li>Export forecasts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Data upload section
st.header("📁 Data Configuration")

st.write("Configure your data source and analysis parameters below. These settings will be used across all analysis modules.")

# Initialize session state for global settings
if 'data_file' not in st.session_state:
    st.session_state.data_file = None
if 'risk_free_rate' not in st.session_state:
    st.session_state.risk_free_rate = 0.0493
if 'risk_aversion' not in st.session_state:
    st.session_state.risk_aversion = 2.5
if 'alpha_threshold' not in st.session_state:
    st.session_state.alpha_threshold = 0.02
if 'portfolio_size' not in st.session_state:
    st.session_state.portfolio_size = 10
if 'data_configured' not in st.session_state:
    st.session_state.data_configured = False

# Data source selection
col1, col2 = st.columns([2, 1])

with col1:
    upload_mode = st.radio(
        "Choose data input method:",
        ["Use default data", "Upload separate files", "Upload combined file"],
        help="Select how you want to provide the data"
    )

with col2:
    st.markdown("### 📊 Data Format")
    st.caption("""
    **Required columns:**
    - Date
    - Stock prices (multiple columns)
    - Market index
    - BI Rate (optional)
    """)

# Initialize variables
temp_file_path = None

if upload_mode == "Use default data":
    data_file = "data_gabungan_skripsi.xlsx"
    if Path(data_file).exists():
        st.success("✅ Using default data: data_gabungan_skripsi.xlsx")
        st.session_state.data_file = data_file
        st.session_state.data_sheet = 'Data Gabungan Harian'
    else:
        st.error("❌ Default data file not found!")
        st.stop()

elif upload_mode == "Upload separate files":
    st.info("📤 Upload 3 separate files that will be automatically combined")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        stock_file = st.file_uploader(
            "1️⃣ Stock Prices",
            type=['csv', 'xlsx', 'xls'],
            help="Upload file with columns: Date, Ticker1, Ticker2, ...",
            key="stock_upload"
        )
        if stock_file:
            st.success("✅ Stock file uploaded")
    
    with col2:
        market_file = st.file_uploader(
            "2️⃣ Market Index",
            type=['csv', 'xlsx', 'xls'],
            help="Upload file with columns: Date, Index_Price",
            key="market_upload"
        )
        if market_file:
            st.success("✅ Market file uploaded")
    
    with col3:
        st.write("3️⃣ BI Rate")
        risk_free_option = st.radio(
            "Choose BI Rate method:",
            ["Upload daily BI Rate", "Use constant rate"],
            help="Upload file with daily rates or enter a constant",
            key="rf_option"
        )
        
        if risk_free_option == "Upload daily BI Rate":
            bi_rate_file = st.file_uploader(
                "Upload BI Rate Data",
                type=['csv', 'xlsx', 'xls'],
                help="File with columns: Date, BI_Rate",
                key="bi_rate_upload"
            )
            if bi_rate_file:
                st.success("✅ BI Rate file uploaded")
        else:
            bi_rate_constant = st.number_input(
                "BI Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=4.93,
                step=0.01,
                help="Enter constant BI Rate"
            )
            bi_rate_file = None
            st.session_state.bi_rate_constant = bi_rate_constant / 100
    
    # Validation
    if risk_free_option == "Upload daily BI Rate":
        files_ready = stock_file and market_file and bi_rate_file
    else:
        files_ready = stock_file and market_file
    
    if files_ready:
        st.success("✅ All files ready! Click 'Confirm Configuration' below to combine them.")
        # Store files in session state for later processing
        st.session_state.stock_file = stock_file
        st.session_state.market_file = market_file
        st.session_state.bi_rate_file = bi_rate_file if risk_free_option == "Upload daily BI Rate" else None
        st.session_state.upload_mode = upload_mode
    else:
        missing = []
        if not stock_file:
            missing.append("Stock Prices")
        if not market_file:
            missing.append("Market Index")
        if risk_free_option == "Upload daily BI Rate" and not bi_rate_file:
            missing.append("BI Rate Data")
        st.warning(f"⚠️ Missing: {', '.join(missing)}")

else:  # Upload combined file
    st.info("📤 Upload 1 combined Excel file with all data")
    uploaded_file = st.file_uploader(
        "Upload Combined Data (Excel)",
        type=['xlsx', 'xls'],
        help="Excel file with stock prices, market index, and BI rate"
    )
    if uploaded_file:
        st.session_state.data_file = uploaded_file
        st.session_state.data_sheet = 'Data Gabungan Harian'
        st.success("✅ File uploaded successfully")
    else:
        st.warning("⚠️ Please upload a data file to continue")

# Analysis parameters
if st.session_state.data_file:
    st.markdown("---")
    st.subheader("⚙️ Global Analysis Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.risk_aversion = st.slider(
            "Risk Aversion (λ)",
            min_value=1.0,
            max_value=5.0,
            value=2.5,
            step=0.1,
            help="Higher values indicate more risk aversion"
        )
        
        st.session_state.alpha_threshold = st.slider(
            "Alpha Threshold (%)",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Threshold for undervalued/overvalued classification"
        ) / 100
    
    with col2:
        st.session_state.portfolio_size = st.slider(
            "Portfolio Size (stocks)",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            help="Number of stocks in optimized portfolio"
        )
        
        n_simulations = st.selectbox(
            "Monte Carlo Simulations",
            options=[1000, 5000, 10000],
            index=1,
            help="More simulations = more accurate"
        )
        st.session_state.n_simulations = n_simulations
    
    with col3:
        # Show current settings
        st.markdown("**Current Settings:**")
        st.write(f"• Risk Aversion: {st.session_state.risk_aversion:.1f}")
        st.write(f"• Alpha Threshold: {st.session_state.alpha_threshold*100:.1f}%")
        st.write(f"• Portfolio Size: {st.session_state.portfolio_size}")
        st.write(f"• Simulations: {n_simulations:,}")
    
    st.markdown("---")
    
    # Preview Data Button
    st.subheader("👀 Preview Your Data")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("Before running analysis, you can preview and explore your uploaded data.")
    with col2:
        preview_button = st.button("📊 Preview Data", type="secondary", use_container_width=True)
    
    if preview_button:
        with st.spinner("Loading data for preview..."):
            try:
                from src.data.loader import DataLoader
                import plotly.graph_objects as go
                import numpy as np
                
                loader = DataLoader()
                
                # Get sheet name
                sheet_name = st.session_state.get('data_sheet', 'Data Gabungan Harian')
                
                # Load data
                stock_data, market_data = loader.load_financial_dataset(
                    file_path=st.session_state.data_file,
                    sheet_name=sheet_name
                )
                
                n_stocks = len(stock_data)
                n_observations = len(next(iter(stock_data.values())).prices)
                
                st.success("✅ Data loaded successfully for preview!")
                
                # Store in session for later use
                st.session_state.preview_data = {
                    'stock_data': stock_data,
                    'market_data': market_data,
                    'n_stocks': n_stocks,
                    'n_observations': n_observations
                }
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.exception(e)
                st.stop()
    
    # Display data exploration if previewed
    if 'preview_data' in st.session_state:
        st.markdown("---")
        st.header("📊 Data Exploration")
        
        preview_data = st.session_state.preview_data
        stock_data_dict = preview_data['stock_data']
        market_data_obj = preview_data['market_data']
        n_stocks = preview_data['n_stocks']
        n_observations = preview_data['n_observations']
        
        # Summary Statistics
        with st.expander("📈 Summary Statistics", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Number of Stocks", n_stocks)
            
            with col2:
                st.metric("Observations", n_observations)
            
            with col3:
                first_stock = next(iter(stock_data_dict.values()))
                date_range = f"{first_stock.prices.index[0].strftime('%Y-%m-%d')} to {first_stock.prices.index[-1].strftime('%Y-%m-%d')}"
                st.metric("Date Range", "")
                st.caption(date_range)
            
            with col4:
                market_return = market_data_obj.returns.mean() * 252
                st.metric("Market Return (Annual)", f"{market_return*100:.2f}%")
        
        # Price Trends
        with st.expander("📉 Stock Price Trends"):
            import plotly.graph_objects as go
            
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
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if n_stocks > 10:
                st.caption(f"📝 Showing first 10 of {n_stocks} stocks")
        
        # Returns Distribution
        with st.expander("📊 Returns Distribution"):
            all_returns = []
            for ticker, stock in stock_data_dict.items():
                all_returns.extend(stock.returns.dropna().values)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=all_returns,
                nbinsx=50,
                name='Returns',
                marker_color='steelblue'
            ))
            
            fig.update_layout(
                title="Distribution of Daily Returns (All Stocks)",
                xaxis_title="Daily Return",
                yaxis_title="Frequency",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            all_returns_array = np.array(all_returns)
            col1.metric("Mean Return", f"{np.mean(all_returns_array)*100:.3f}%")
            col2.metric("Std Dev", f"{np.std(all_returns_array)*100:.3f}%")
            col3.metric("Min Return", f"{np.min(all_returns_array)*100:.2f}%")
            col4.metric("Max Return", f"{np.max(all_returns_array)*100:.2f}%")
        
        # Correlation Heatmap
        with st.expander("🔥 Correlation Heatmap"):
            returns_data = {}
            for ticker, stock in stock_data_dict.items():
                returns_data[ticker] = stock.returns
            
            returns_df = pd.DataFrame(returns_data).dropna()
            corr_matrix = returns_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 8},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Stock Returns Correlation Matrix",
                height=600,
                xaxis={'side': 'bottom'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            st.info(f"💡 **Average Correlation**: {avg_corr:.3f} - {'High' if avg_corr > 0.7 else 'Moderate' if avg_corr > 0.4 else 'Low'} correlation among stocks")
        
        # Volatility Analysis
        with st.expander("⚡ Volatility Analysis"):
            volatilities = {}
            for ticker, stock in stock_data_dict.items():
                vol = stock.returns.std() * np.sqrt(252)
                volatilities[ticker] = vol
            
            sorted_vols = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)
            
            tickers = [x[0] for x in sorted_vols]
            vols = [x[1] * 100 for x in sorted_vols]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=tickers,
                y=vols,
                marker_color=['indianred' if v > np.mean(vols) else 'steelblue' for v in vols],
                text=[f'{v:.1f}%' for v in vols],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Annualized Volatility by Stock",
                xaxis_title="Stock Ticker",
                yaxis_title="Volatility (%)",
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Highest Volatility", f"{sorted_vols[0][0]}: {sorted_vols[0][1]*100:.1f}%")
            col2.metric("Lowest Volatility", f"{sorted_vols[-1][0]}: {sorted_vols[-1][1]*100:.1f}%")
            col3.metric("Average Volatility", f"{np.mean(list(volatilities.values()))*100:.1f}%")
        
        # Data Quality Check
        with st.expander("✅ Data Quality Check"):
            quality_data = []
            for ticker, stock in stock_data_dict.items():
                total_points = len(stock.prices)
                missing_prices = stock.prices.isna().sum()
                missing_returns = stock.returns.isna().sum()
                completeness = ((total_points - missing_prices) / total_points) * 100
                
                quality_data.append({
                    'Ticker': ticker,
                    'Total Points': total_points,
                    'Missing Prices': missing_prices,
                    'Missing Returns': missing_returns,
                    'Completeness': f"{completeness:.1f}%"
                })
            
            quality_df = pd.DataFrame(quality_data)
            st.dataframe(quality_df, use_container_width=True, height=300)
            
            total_completeness = quality_df['Completeness'].apply(lambda x: float(x.strip('%'))).mean()
            
            if total_completeness >= 95:
                st.success(f"✅ Data quality excellent: {total_completeness:.1f}% complete")
            elif total_completeness >= 85:
                st.info(f"ℹ️ Data quality good: {total_completeness:.1f}% complete")
            else:
                st.warning(f"⚠️ Data quality issue: {total_completeness:.1f}% complete")
        
        st.info("👆 Review the data exploration above to understand your dataset before running analysis!")
    
    st.markdown("---")
    
    # Configuration complete
    if st.button("✅ Confirm Configuration", type="primary", use_container_width=True):
        # Handle separate file upload mode - combine files first
        if 'upload_mode' in st.session_state and st.session_state.upload_mode == "Upload separate files":
            with st.spinner("Combining uploaded files..."):
                try:
                    # Get files from session state
                    stock_file = st.session_state.stock_file
                    market_file = st.session_state.market_file
                    bi_rate_file = st.session_state.get('bi_rate_file', None)
                    bi_rate_constant = st.session_state.get('bi_rate_constant', None)
                    
                    # Combine files
                    combined_path = combine_uploaded_files(
                        stock_file, 
                        market_file, 
                        risk_free_rate=bi_rate_constant,
                        risk_free_file=bi_rate_file
                    )
                    
                    # Store combined file path
                    st.session_state.data_file = combined_path
                    st.session_state.data_sheet = 'Data Gabungan Harian'
                    
                    st.success("✅ Files combined successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error combining files: {str(e)}")
                    st.stop()
        
        # Mark configuration as complete
        st.session_state.data_configured = True
        st.balloons()
        st.success("""
        ✅ **Configuration Complete!**
        
        Your settings have been saved. Now navigate to the analysis pages using the sidebar:
        
        1. **📊 CAPM Analysis** - Start here to identify undervalued stocks
        2. **💼 Portfolio Optimization** - Build optimal portfolio from undervalued stocks  
        3. **⚠️ Crisis Scenarios** - Test portfolio under crisis conditions
        4. **📈 Forecasting** - Predict future prices and returns
        
        👉 Click on any page in the sidebar to begin!
        """)

# Navigation guide
if st.session_state.data_configured:
    st.markdown("---")
    st.header("🧭 Quick Navigation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("### 📊 CAPM")
        st.caption("Identify undervalued stocks")
        if st.button("Go to CAPM →", key="nav_capm", use_container_width=True):
            st.switch_page("pages/1_📊_CAPM_Analysis.py")
    
    with col2:
        st.markdown("### 💼 Portfolio")
        st.caption("Optimize your holdings")
        if st.button("Go to Portfolio →", key="nav_portfolio", use_container_width=True, disabled=True):
            st.info("Coming soon!")
    
    with col3:
        st.markdown("### ⚠️ Crisis")
        st.caption("Test resilience")
        if st.button("Go to Crisis →", key="nav_crisis", use_container_width=True, disabled=True):
            st.info("Coming soon!")
    
    with col4:
        st.markdown("### 📈 Forecast")
        st.caption("Predict the future")
        if st.button("Go to Forecast →", key="nav_forecast", use_container_width=True, disabled=True):
            st.info("Coming soon!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Financial Analysis System v2.0 | Multi-Page Dashboard</p>
    <p>⚠️ Disclaimer: This is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
