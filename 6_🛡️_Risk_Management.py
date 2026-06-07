"""
Risk Management Dashboard

Comprehensive risk management strategies integrating portfolio optimization,
crisis scenarios, and forecasting results for ongoing risk monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Risk Management",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .risk-high {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .risk-medium {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .risk-low {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .action-item {
        background-color: #e7f3ff;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        border-left: 3px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🛡️ Risk Management Dashboard")
st.markdown("### Comprehensive Risk Monitoring & Action Plan")
st.markdown("---")

# Check prerequisites
if 'portfolio_results' not in st.session_state or not st.session_state.portfolio_run:
    st.warning("⚠️ Please run Portfolio Optimization first!")
    if st.button("← Go to Portfolio Optimization", use_container_width=True):
        st.switch_page("pages/2_💼_Portfolio_Optimization.py")
    st.stop()

# Get portfolio data
portfolio_results = st.session_state.portfolio_results
portfolio_tickers = portfolio_results['tickers']
portfolio_weights = portfolio_results['weights']
stock_data = portfolio_results['stock_data']

# Check if crisis analysis was run
crisis_run = st.session_state.get('crisis_run', False)
forecast_run = st.session_state.get('forecast_run', False)

# Introduction
with st.expander("📖 About Risk Management Dashboard", expanded=False):
    st.markdown("""
    ### Integrated Risk Management
    
    This dashboard combines insights from:
    - **Portfolio Optimization**: Current allocation and weights
    - **Crisis Scenarios**: Stress test results and VaR metrics
    - **Forecasting**: Future price predictions and volatility
    - **Data Validation**: Model suitability and data quality
    
    **Key Features**:
    1. 🎯 **Risk Score**: Overall portfolio risk assessment
    2. 📊 **Position Limits**: Recommended position sizing
    3. 🚨 **Stop-Loss Levels**: Calculated exit points
    4. 📈 **Rebalancing Alerts**: When to adjust portfolio
    5. 🔔 **Risk Triggers**: Automated warning thresholds
    6. 📋 **Action Plan**: Prioritized risk mitigation steps
    """)

st.markdown("---")

# Portfolio Overview
st.header("📊 Portfolio Risk Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Stocks", len(portfolio_tickers))
    
with col2:
    # Calculate concentration (Herfindahl Index)
    hhi = sum([w**2 for w in portfolio_weights]) * 10000
    concentration = "High" if hhi > 2500 else "Medium" if hhi > 1500 else "Low"
    st.metric("Concentration", concentration)
    st.caption(f"HHI: {hhi:.0f}")

with col3:
    # Top 3 concentration
    top3_weight = sum(sorted(portfolio_weights, reverse=True)[:3]) * 100
    st.metric("Top 3 Holdings", f"{top3_weight:.1f}%")
    
with col4:
    # Max single position
    max_position = max(portfolio_weights) * 100
    st.metric("Largest Position", f"{max_position:.1f}%")

# Risk Score Calculation
st.markdown("---")
st.header("🎯 Overall Risk Score")

risk_factors = {}
risk_score = 100  # Start with perfect score

# Factor 1: Concentration Risk
if hhi > 2500:
    risk_factors['Concentration'] = {'score': -20, 'level': 'High', 'message': 'Portfolio is highly concentrated'}
elif hhi > 1500:
    risk_factors['Concentration'] = {'score': -10, 'level': 'Medium', 'message': 'Moderate concentration'}
else:
    risk_factors['Concentration'] = {'score': 0, 'level': 'Low', 'message': 'Well diversified'}

# Factor 2: Position Size Risk
if max_position > 25:
    risk_factors['Position Size'] = {'score': -15, 'level': 'High', 'message': f'Largest position ({max_position:.1f}%) exceeds 25%'}
elif max_position > 15:
    risk_factors['Position Size'] = {'score': -8, 'level': 'Medium', 'message': f'Largest position ({max_position:.1f}%) is moderate'}
else:
    risk_factors['Position Size'] = {'score': 0, 'level': 'Low', 'message': 'Position sizes are appropriate'}

# Factor 3: Crisis Scenario Risk
if crisis_run and 'crisis_results' in st.session_state:
    crisis_results = st.session_state.crisis_results
    var_95 = crisis_results['var_95']
    
    if var_95 < -0.25:
        risk_factors['Crisis Resilience'] = {'score': -25, 'level': 'High', 'message': f'VaR 95% is {var_95*100:.1f}% (very high risk)'}
    elif var_95 < -0.15:
        risk_factors['Crisis Resilience'] = {'score': -15, 'level': 'Medium', 'message': f'VaR 95% is {var_95*100:.1f}% (moderate risk)'}
    else:
        risk_factors['Crisis Resilience'] = {'score': 0, 'level': 'Low', 'message': f'VaR 95% is {var_95*100:.1f}% (acceptable)'}
else:
    risk_factors['Crisis Resilience'] = {'score': -10, 'level': 'Unknown', 'message': 'Crisis analysis not run yet'}

# Factor 4: Forecast Volatility Risk
if forecast_run and 'all_forecast_results' in st.session_state:
    forecast_results = st.session_state.all_forecast_results
    forecasts = forecast_results['forecasts']
    
    # Check GARCH volatility if available
    high_vol_count = 0
    for ticker, forecast in forecasts.items():
        if 'GARCH' in forecast['models']:
            ann_vol = forecast['models']['GARCH']['annualized_vol']
            # Handle both scalar and array values
            if isinstance(ann_vol, (list, np.ndarray)):
                ann_vol_value = float(np.mean(ann_vol))
            else:
                ann_vol_value = float(ann_vol)
            
            if ann_vol_value > 0.40:  # 40% annualized volatility
                high_vol_count += 1
    
    if high_vol_count > len(forecasts) * 0.5:
        risk_factors['Forecast Volatility'] = {'score': -15, 'level': 'High', 'message': f'{high_vol_count} stocks show high volatility'}
    elif high_vol_count > 0:
        risk_factors['Forecast Volatility'] = {'score': -8, 'level': 'Medium', 'message': f'{high_vol_count} stocks show elevated volatility'}
    else:
        risk_factors['Forecast Volatility'] = {'score': 0, 'level': 'Low', 'message': 'Volatility levels are acceptable'}
else:
    risk_factors['Forecast Volatility'] = {'score': -5, 'level': 'Unknown', 'message': 'Forecasting not run yet'}

# Calculate total risk score
for factor, data in risk_factors.items():
    risk_score += data['score']

# Display risk score
col1, col2 = st.columns([1, 2])

with col1:
    # Risk score gauge
    if risk_score >= 80:
        color = "green"
        risk_level = "LOW RISK"
    elif risk_score >= 60:
        color = "orange"
        risk_level = "MEDIUM RISK"
    else:
        color = "red"
        risk_level = "HIGH RISK"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background-color: #f0f2f6; border-radius: 1rem;'>
        <h1 style='color: {color}; font-size: 4rem; margin: 0;'>{risk_score}</h1>
        <h3 style='color: {color}; margin: 0.5rem 0;'>{risk_level}</h3>
        <p style='color: #666; margin: 0;'>out of 100</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("Risk Factor Breakdown")
    
    for factor, data in risk_factors.items():
        level_color = "🔴" if data['level'] == 'High' else "🟡" if data['level'] == 'Medium' else "🟢"
        st.markdown(f"""
        <div class='action-item'>
            <strong>{level_color} {factor}</strong>: {data['level']}<br>
            <small>{data['message']}</small>
        </div>
        """, unsafe_allow_html=True)

# Position-Level Risk Management
st.markdown("---")
st.header("📊 Position-Level Risk Management")

# Calculate recommended position limits
position_data = []
for i, ticker in enumerate(portfolio_tickers):
    current_weight = portfolio_weights[i] * 100
    
    # Recommended max based on portfolio size
    if len(portfolio_tickers) < 5:
        recommended_max = 25
    elif len(portfolio_tickers) < 10:
        recommended_max = 15
    else:
        recommended_max = 10
    
    # Status
    if current_weight > recommended_max * 1.2:
        status = "🔴 Reduce"
        action = f"Reduce by {current_weight - recommended_max:.1f}%"
    elif current_weight > recommended_max:
        status = "🟡 Monitor"
        action = "Watch closely"
    else:
        status = "🟢 OK"
        action = "No action needed"
    
    position_data.append({
        'Ticker': ticker,
        'Current (%)': current_weight,
        'Recommended Max (%)': recommended_max,
        'Status': status,
        'Action': action
    })

position_df = pd.DataFrame(position_data)
st.dataframe(position_df, use_container_width=True, height=min(400, len(position_df)*35+38))

# Stop-Loss Recommendations
st.markdown("---")
st.header("🚨 Stop-Loss Recommendations")

st.info("""
**Stop-Loss Strategy**: Protect your capital by setting automatic exit points.

**Three-Tier Approach**:
1. **Alert Level**: Monitor closely, consider reducing position
2. **Action Level**: Reduce position by 50%
3. **Hard Stop**: Exit position completely
""")

# Calculate stop-loss levels
if crisis_run and 'crisis_results' in st.session_state:
    crisis_results = st.session_state.crisis_results
    var_95 = crisis_results['var_95']
    cvar_95 = crisis_results['cvar_95']
    
    # Stop-loss levels based on VaR
    alert_level = var_95 * 0.6  # 60% of VaR
    action_level = var_95 * 0.8  # 80% of VaR
    hard_stop = var_95  # At VaR level
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='risk-low'>
            <h4>🟡 Alert Level</h4>
            <h2>{alert_level*100:.1f}%</h2>
            <p>Monitor closely<br>Consider reducing position</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='risk-medium'>
            <h4>🟠 Action Level</h4>
            <h2>{action_level*100:.1f}%</h2>
            <p>Reduce position by 50%<br>Reassess portfolio</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='risk-high'>
            <h4>🔴 Hard Stop</h4>
            <h2>{hard_stop*100:.1f}%</h2>
            <p>Exit position completely<br>Preserve capital</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Individual stock stop-losses
    st.subheader("Individual Stock Stop-Loss Levels")
    
    stop_loss_data = []
    for i, ticker in enumerate(portfolio_tickers):
        stock = stock_data[ticker]
        current_price = stock.prices.iloc[-1]
        
        # Calculate stop-loss prices
        alert_price = current_price * (1 + alert_level)
        action_price = current_price * (1 + action_level)
        hard_stop_price = current_price * (1 + hard_stop)
        
        stop_loss_data.append({
            'Ticker': ticker,
            'Current Price': f"${current_price:.2f}",
            'Alert ($)': f"${alert_price:.2f}",
            'Action ($)': f"${action_price:.2f}",
            'Hard Stop ($)': f"${hard_stop_price:.2f}",
            'Weight (%)': portfolio_weights[i] * 100
        })
    
    stop_loss_df = pd.DataFrame(stop_loss_data)
    st.dataframe(stop_loss_df, use_container_width=True, height=min(400, len(stop_loss_df)*35+38))
    
else:
    st.warning("⚠️ Run Crisis Scenarios analysis to get stop-loss recommendations")
    if st.button("→ Go to Crisis Scenarios", use_container_width=True):
        st.switch_page("pages/3_⚠️_Crisis_Scenarios.py")

# Rebalancing Triggers
st.markdown("---")
st.header("⚖️ Rebalancing Triggers")

st.markdown("""
**When to Rebalance**:

Rebalancing maintains your target allocation and manages risk. Consider rebalancing when:
""")

rebalance_triggers = []

# Check drift from target
for i, ticker in enumerate(portfolio_tickers):
    current_weight = portfolio_weights[i] * 100
    target_weight = 100 / len(portfolio_tickers)  # Equal weight as baseline
    drift = abs(current_weight - target_weight)
    
    if drift > 5:
        rebalance_triggers.append({
            'Trigger': f'{ticker} Weight Drift',
            'Current': f'{current_weight:.1f}%',
            'Target': f'{target_weight:.1f}%',
            'Drift': f'{drift:.1f}%',
            'Action': 'Rebalance to target' if drift > 10 else 'Monitor'
        })

# Time-based trigger
last_rebalance = st.session_state.get('last_rebalance_date', None)
if last_rebalance:
    days_since = (datetime.now() - last_rebalance).days
    if days_since > 90:
        rebalance_triggers.append({
            'Trigger': 'Time-Based',
            'Current': f'{days_since} days',
            'Target': '90 days',
            'Drift': f'+{days_since-90} days',
            'Action': 'Quarterly rebalance due'
        })

if rebalance_triggers:
    trigger_df = pd.DataFrame(rebalance_triggers)
    st.dataframe(trigger_df, use_container_width=True)
else:
    st.success("✅ No rebalancing triggers detected. Portfolio is well-balanced.")

# Action Plan
st.markdown("---")
st.header("📋 Risk Management Action Plan")

st.markdown("### Immediate Actions (Next 48 Hours)")

immediate_actions = []

# High concentration
if hhi > 2500:
    immediate_actions.append("🔴 **URGENT**: Reduce concentration - Portfolio HHI is too high")

# Large positions
oversized = [t for i, t in enumerate(portfolio_tickers) if portfolio_weights[i] * 100 > 20]
if oversized:
    immediate_actions.append(f"🔴 **URGENT**: Reduce oversized positions: {', '.join(oversized)}")

# Crisis risk
if crisis_run and 'crisis_results' in st.session_state:
    crisis_results = st.session_state.crisis_results
    var_95 = crisis_results['var_95']
    if var_95 < -0.25:
        immediate_actions.append("🔴 **URGENT**: High crisis risk detected - Review portfolio allocation")

# Set stop-losses
if not st.session_state.get('stop_losses_set', False):
    immediate_actions.append("🟡 Set stop-loss orders for all positions")

if immediate_actions:
    for action in immediate_actions:
        st.markdown(f"- {action}")
else:
    st.success("✅ No immediate actions required")

st.markdown("### Short-Term Actions (This Week)")

short_term_actions = [
    "📊 Review individual stock performance",
    "📈 Check if forecasts align with expectations",
    "🔍 Monitor news and market conditions",
    "📝 Document any portfolio changes"
]

for action in short_term_actions:
    st.markdown(f"- {action}")

st.markdown("### Medium-Term Actions (This Month)")

medium_term_actions = [
    "⚖️ Quarterly rebalancing review",
    "📊 Update crisis scenario analysis",
    "🔄 Refresh forecasting models",
    "📈 Review and adjust risk tolerance if needed",
    "💼 Consider tax implications of rebalancing"
]

for action in medium_term_actions:
    st.markdown(f"- {action}")

# Risk Management Tools
st.markdown("---")
st.header("🔧 Risk Management Tools")

tab1, tab2, tab3 = st.tabs(["📊 Position Sizing", "🛡️ Hedging Strategies", "📈 Monitoring Checklist"])

with tab1:
    st.markdown("""
    ### Position Sizing Calculator
    
    **The 2% Rule**: Never risk more than 2% of portfolio on a single trade.
    
    **Formula**:
    ```
    Position Size = (Portfolio Value × Risk %) / (Entry Price - Stop Loss Price)
    ```
    
    **Example**:
    - Portfolio: $100,000
    - Risk per trade: 2% = $2,000
    - Entry price: $50
    - Stop loss: $45
    - Risk per share: $5
    
    **Position Size** = $2,000 / $5 = **400 shares** (or $20,000 position = 20% of portfolio)
    """)
    
    # Interactive calculator
    st.subheader("Calculate Your Position Size")
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", 10000, 10000000, 100000, 10000)
        risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
    
    with col2:
        entry_price = st.number_input("Entry Price ($)", 1.0, 10000.0, 50.0, 1.0)
        stop_loss_price = st.number_input("Stop Loss Price ($)", 1.0, 10000.0, 45.0, 1.0)
    
    if entry_price > stop_loss_price:
        risk_per_share = entry_price - stop_loss_price
        risk_amount = portfolio_value * (risk_percent / 100)
        position_shares = int(risk_amount / risk_per_share)
        position_value = position_shares * entry_price
        position_percent = (position_value / portfolio_value) * 100
        
        st.success(f"""
        **Recommended Position**:
        - Shares: **{position_shares:,}**
        - Value: **${position_value:,.2f}**
        - Portfolio %: **{position_percent:.1f}%**
        - Max Risk: **${risk_amount:,.2f}** ({risk_percent}%)
        """)
    else:
        st.error("Entry price must be higher than stop loss price")

with tab2:
    st.markdown("""
    ### Hedging Strategies
    
    **1. Protective Puts** 🛡️
    - Buy put options on individual stocks or index
    - Provides downside protection (insurance)
    - Cost: Premium paid for puts
    - Best for: High-conviction positions you want to protect
    
    **2. Collar Strategy** 🔒
    - Buy protective put + Sell covered call
    - Zero or low-cost protection
    - Trade-off: Limited upside for downside protection
    - Best for: Moderate market outlook
    
    **3. Portfolio Insurance** 📊
    - Buy index puts (e.g., SPY, QQQ)
    - Cheaper than individual stock puts
    - Protects overall portfolio
    - Best for: Market-wide concerns
    
    **4. Inverse ETFs** ⬇️
    - Short exposure to market/sector
    - No options knowledge required
    - Daily rebalancing can cause tracking error
    - Best for: Short-term hedges
    
    **5. Cash Allocation** 💵
    - Simplest hedge: Hold cash
    - No cost, no complexity
    - Opportunity cost of missing gains
    - Best for: High uncertainty periods
    
    ### When to Hedge
    - Portfolio value > $100,000
    - High market volatility (VIX > 25)
    - Concentrated positions (>20% in one stock)
    - Before major events (earnings, elections)
    - When you can't afford large losses
    """)

with tab3:
    st.markdown("""
    ### Daily Monitoring Checklist
    
    **Every Day** (5 minutes):
    - [ ] Check portfolio value
    - [ ] Review any stop-loss triggers
    - [ ] Scan major market news
    - [ ] Check for earnings announcements
    
    **Weekly** (30 minutes):
    - [ ] Review individual stock performance
    - [ ] Check position weights vs targets
    - [ ] Update watchlist
    - [ ] Review sector allocation
    - [ ] Check correlation changes
    
    **Monthly** (2 hours):
    - [ ] Full portfolio review
    - [ ] Rebalancing assessment
    - [ ] Update forecasts
    - [ ] Review risk metrics
    - [ ] Tax loss harvesting opportunities
    - [ ] Document changes and rationale
    
    **Quarterly** (4 hours):
    - [ ] Comprehensive risk analysis
    - [ ] Crisis scenario testing
    - [ ] Strategy review and adjustment
    - [ ] Performance attribution
    - [ ] Goal progress check
    - [ ] Consider professional review
    
    ### Red Flags - Act Immediately
    - 🚨 Any position down >15% from entry
    - 🚨 Portfolio down >10% in a week
    - 🚨 Major news affecting holdings
    - 🚨 Correlation breakdown (diversification failing)
    - 🚨 Volatility spike (VIX >30)
    """)

# Export Action Plan
st.markdown("---")
st.header("📥 Export Risk Management Plan")

col1, col2 = st.columns(2)

with col1:
    # Generate text report
    report_lines = [
        "RISK MANAGEMENT ACTION PLAN",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Portfolio: {len(portfolio_tickers)} stocks",
        "",
        f"OVERALL RISK SCORE: {risk_score}/100 ({risk_level})",
        "",
        "RISK FACTORS:",
    ]
    
    for factor, data in risk_factors.items():
        report_lines.append(f"- {factor}: {data['level']} - {data['message']}")
    
    report_lines.extend([
        "",
        "IMMEDIATE ACTIONS:",
    ])
    
    if immediate_actions:
        for action in immediate_actions:
            report_lines.append(f"- {action}")
    else:
        report_lines.append("- No immediate actions required")
    
    if crisis_run and 'crisis_results' in st.session_state:
        crisis_results = st.session_state.crisis_results
        var_95 = crisis_results['var_95']
        alert_level = var_95 * 0.6
        action_level = var_95 * 0.8
        hard_stop = var_95
        
        report_lines.extend([
            "",
            "STOP-LOSS LEVELS:",
            f"- Alert: {alert_level*100:.1f}%",
            f"- Action: {action_level*100:.1f}%",
            f"- Hard Stop: {hard_stop*100:.1f}%",
        ])
    
    report_text = "\n".join(report_lines)
    
    st.download_button(
        label="📄 Download Action Plan (TXT)",
        data=report_text,
        file_name=f"risk_management_plan_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True
    )

with col2:
    # Export position data
    if not position_df.empty:
        csv_buffer = position_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Position Analysis (CSV)",
            data=csv_buffer,
            file_name=f"position_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.info("""
💡 **Remember**: Risk management is an ongoing process, not a one-time task.
Review this dashboard regularly and adjust your strategy as market conditions change.
""")

# Made with Bob
