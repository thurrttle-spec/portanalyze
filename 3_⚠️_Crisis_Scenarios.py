"""
Crisis Scenarios Page

Tests portfolio resilience under stress conditions using Monte Carlo simulation.
Includes VaR, CVaR, maximum drawdown, and investment recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import io
from datetime import datetime

# Import crisis analysis modules
from src.crisis import (
    CrisisScenario,
    CrisisSimulator,
    calculate_var,
    calculate_cvar
)

# Page configuration
st.set_page_config(
    page_title="Crisis Scenarios",
    page_icon="⚠️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
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
st.title("⚠️ Crisis Scenarios Analysis")
st.markdown("### Portfolio Stress Testing with Monte Carlo Simulation")
st.markdown("---")

# Check if portfolio optimization was run
if 'portfolio_results' not in st.session_state or not st.session_state.portfolio_run:
    st.warning("⚠️ Please run Portfolio Optimization first to test crisis scenarios!")
    if st.button("← Go to Portfolio Optimization", use_container_width=True):
        st.switch_page("pages/2_💼_Portfolio_Optimization.py")
    st.stop()

# Introduction
with st.expander("📖 What is Crisis Scenario Analysis?", expanded=False):
    st.markdown("""
    ### Crisis Scenario Analysis
    
    **Purpose**: Test how your optimized portfolio performs under extreme market conditions.
    
    **Methodology**:
    - **Monte Carlo Simulation**: Runs thousands of simulations (10,000 by default)
    - **Crisis Scenarios**: Applies stress conditions (increased volatility, negative returns)
    - **Risk Metrics**: Calculates VaR, CVaR, Maximum Drawdown
    - **Investment Recommendation**: Assesses portfolio resilience
    
    **Key Risk Metrics**:
    
    1. **Value at Risk (VaR)**: Maximum expected loss at a confidence level
       - Example: 95% VaR of -15% means there's 95% chance loss won't exceed 15%
    
    2. **Conditional VaR (CVaR)**: Average loss in worst-case scenarios
       - Also called "Expected Shortfall"
       - More conservative than VaR
    
    3. **Maximum Drawdown**: Largest peak-to-trough decline
       - Shows worst-case portfolio drop
    
    4. **Recovery Time**: Days needed to recover from losses
    
    **Crisis Scenarios Available**:
    - **US-Iran Tension**: Geopolitical stress in Middle East
    - **Oil Supply Disruption**: Major oil price shock
    - **Moderate Recession**: Economic slowdown
    """)

st.markdown("---")

# Scenario Selection
st.header("⚙️ Configure Crisis Scenario")

col1, col2, col3 = st.columns(3)

with col1:
    # Scenario selector
    scenario_options = {
        "US-Iran Geopolitical Tension": "us_iran",
        "Oil Supply Disruption": "oil_shock",
        "Moderate Economic Recession": "recession"
    }
    
    selected_scenario_name = st.selectbox(
        "Select Crisis Scenario",
        options=list(scenario_options.keys()),
        help="Choose the crisis scenario to test your portfolio against"
    )
    
    scenario_type = scenario_options[selected_scenario_name]

with col2:
    # Number of simulations
    n_simulations = st.slider(
        "Number of Simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        help="More simulations = more accurate but slower"
    )

with col3:
    # Portfolio value
    portfolio_value = st.number_input(
        "Portfolio Value ($)",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=10000,
        help="Initial investment amount for dollar VaR calculation"
    )

# Display scenario details
st.subheader("📋 Scenario Details")

if scenario_type == "us_iran":
    scenario = CrisisScenario.us_iran_tension()
elif scenario_type == "oil_shock":
    scenario = CrisisScenario.oil_price_shock()
else:
    scenario = CrisisScenario.moderate_recession()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Duration", f"{scenario.duration_days} days")

with col2:
    st.metric("Volatility Multiplier", f"{scenario.volatility_multiplier}x")

with col3:
    st.metric("Mean Return Shift", f"{scenario.mean_shift*100:+.1f}%")

with col4:
    st.metric("Affected Sectors", len(scenario.affected_sectors))

st.info(f"""
**Scenario Description**: {scenario.description}

**Affected Sectors**: {', '.join(scenario.affected_sectors)}
""")

st.markdown("---")

# Initialize session state for crisis results
if 'crisis_results' not in st.session_state:
    st.session_state.crisis_results = None
if 'crisis_run' not in st.session_state:
    st.session_state.crisis_run = False

# Run Crisis Simulation Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 Run Crisis Simulation", type="primary", use_container_width=True):
        with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
            try:
                # Get portfolio results
                portfolio_data = st.session_state.portfolio_results
                
                # Calculate portfolio daily return and volatility
                returns_df = portfolio_data['returns_df']
                weights = portfolio_data['weights']
                
                # Portfolio daily returns
                portfolio_returns = (returns_df * weights).sum(axis=1)
                
                # Calculate baseline metrics (annualized)
                baseline_return_annual = portfolio_returns.mean() * 252
                baseline_volatility_annual = portfolio_returns.std() * np.sqrt(252)
                
                # Convert to daily
                baseline_return_daily = portfolio_returns.mean()
                baseline_volatility_daily = portfolio_returns.std()
                
                st.info(f"""
                **Portfolio Baseline Metrics**:
                - Daily Return: {baseline_return_daily*100:.4f}%
                - Daily Volatility: {baseline_volatility_daily*100:.4f}%
                - Annual Return: {baseline_return_annual*100:.2f}%
                - Annual Volatility: {baseline_volatility_annual*100:.2f}%
                """)
                
                # Run simulation
                simulator = CrisisSimulator(n_simulations=n_simulations, random_seed=42)
                
                crisis_results = simulator.simulate_crisis(
                    baseline_return=baseline_return_daily,
                    baseline_volatility=baseline_volatility_daily,
                    scenario=scenario,
                    portfolio_value=portfolio_value
                )
                
                # Assess investment worthiness
                recommendation, rationale = simulator.assess_investment_worthiness(crisis_results)
                
                crisis_results['recommendation'] = recommendation
                crisis_results['rationale'] = rationale
                
                # Store results
                st.session_state.crisis_results = crisis_results
                st.session_state.crisis_scenario = scenario
                st.session_state.crisis_portfolio_value = portfolio_value
                st.session_state.crisis_run = True
                
                st.success("✅ Crisis simulation completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error during simulation: {str(e)}")
                st.exception(e)
                st.stop()

# Display results if available
if st.session_state.crisis_run and st.session_state.crisis_results:
    results = st.session_state.crisis_results
    scenario = st.session_state.crisis_scenario
    portfolio_value = st.session_state.crisis_portfolio_value
    
    st.markdown("---")
    
    # Investment Recommendation (Prominent Display)
    st.header("🎯 Investment Recommendation")
    
    recommendation = results['recommendation']
    rationale = results['rationale']
    
    if recommendation == "Worthy":
        st.markdown(f"""
        <div class="success-box">
            <h3 style="color: #28a745; margin-top: 0;">✅ {recommendation}</h3>
            <p style="margin-bottom: 0;">{rationale}</p>
        </div>
        """, unsafe_allow_html=True)
    elif recommendation == "Proceed with Caution":
        st.markdown(f"""
        <div class="warning-box">
            <h3 style="color: #ffc107; margin-top: 0;">⚠️ {recommendation}</h3>
            <p style="margin-bottom: 0;">{rationale}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background-color: #f8d7da; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #dc3545;">
            <h3 style="color: #dc3545; margin-top: 0;">❌ {recommendation}</h3>
            <p style="margin-bottom: 0;">{rationale}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Risk Metrics Summary
    st.header("📊 Risk Metrics Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "VaR 95%",
            f"{results['var_95']*100:.2f}%",
            delta=f"${results['dollar_var_95']:,.0f}",
            delta_color="inverse"
        )
        st.caption("Maximum loss at 95% confidence")
    
    with col2:
        st.metric(
            "CVaR 95%",
            f"{results['cvar_95']*100:.2f}%",
            delta=f"${results['cvar_95']*portfolio_value:,.0f}",
            delta_color="inverse"
        )
        st.caption("Expected loss in worst 5%")
    
    with col3:
        st.metric(
            "Worst Drawdown",
            f"{results['worst_drawdown']*100:.2f}%",
            delta="Peak to Trough",
            delta_color="inverse"
        )
        st.caption("Maximum peak-to-trough decline")
    
    with col4:
        st.metric(
            "Recovery Time",
            f"{results['avg_recovery_days']:.0f} days",
            delta=f"{results['avg_recovery_days']/252:.1f} years",
            delta_color="off"
        )
        st.caption("Average time to recover losses")
    
    st.markdown("---")
    
    # Detailed Statistics
    st.header("📈 Return Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Mean Return", f"{results['mean_return']*100:.2f}%")
    
    with col2:
        st.metric("Median Return", f"{results['median_return']*100:.2f}%")
    
    with col3:
        st.metric("Std Deviation", f"{results['std_return']*100:.2f}%")
    
    with col4:
        st.metric("Min Return", f"{results['min_return']*100:.2f}%")
    
    with col5:
        st.metric("Max Return", f"{results['max_return']*100:.2f}%")
    
    # Probability metrics
    st.subheader("🎲 Probability Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        prob_profit = results['prob_profit']
        st.metric("Probability of Profit", f"{prob_profit*100:.1f}%")
        st.progress(prob_profit)
    
    with col2:
        prob_loss = results['prob_loss']
        st.metric("Probability of Loss", f"{prob_loss*100:.1f}%")
        st.progress(prob_loss)
    
    with col3:
        prob_severe = results['prob_severe_loss']
        st.metric("Probability of Severe Loss (>20%)", f"{prob_severe*100:.1f}%")
        st.progress(prob_severe)
    
    st.markdown("---")
    
    # Visualizations
    st.header("📊 Crisis Analysis Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Return Distribution", "📉 Drawdown Distribution", "📈 Risk Comparison", "🎯 Probability Ranges"])
    
    with tab1:
        # Return distribution histogram
        fig_returns = go.Figure()
        
        final_returns = results['final_returns_distribution']
        
        fig_returns.add_trace(go.Histogram(
            x=final_returns * 100,
            nbinsx=50,
            name='Simulated Returns',
            marker_color='steelblue',
            opacity=0.7
        ))
        
        # Add VaR lines
        fig_returns.add_vline(
            x=results['var_95'] * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"VaR 95%: {results['var_95']*100:.2f}%",
            annotation_position="top"
        )
        
        fig_returns.add_vline(
            x=results['var_99'] * 100,
            line_dash="dash",
            line_color="darkred",
            annotation_text=f"VaR 99%: {results['var_99']*100:.2f}%",
            annotation_position="bottom"
        )
        
        # Add mean line
        fig_returns.add_vline(
            x=results['mean_return'] * 100,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Mean: {results['mean_return']*100:.2f}%",
            annotation_position="top"
        )
        
        fig_returns.update_layout(
            title=f"Distribution of Portfolio Returns<br><sub>{results['n_simulations']:,} Simulations over {scenario.duration_days} Days</sub>",
            xaxis_title="Portfolio Return (%)",
            yaxis_title="Frequency",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        st.info("""
        **Chart Interpretation**:
        - **Blue histogram**: Distribution of simulated portfolio returns
        - **Red dashed lines**: Value at Risk (VaR) at 95% and 99% confidence
        - **Green dotted line**: Mean (expected) return
        - The fatter the left tail, the higher the downside risk
        """)
    
    with tab2:
        # Drawdown distribution
        fig_dd = go.Figure()
        
        max_drawdowns = results['max_drawdowns_distribution']
        
        fig_dd.add_trace(go.Histogram(
            x=max_drawdowns * 100,
            nbinsx=50,
            name='Max Drawdowns',
            marker_color='coral',
            opacity=0.7
        ))
        
        # Add mean and median lines
        fig_dd.add_vline(
            x=results['mean_max_drawdown'] * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {results['mean_max_drawdown']*100:.2f}%",
            annotation_position="top"
        )
        
        fig_dd.add_vline(
            x=results['worst_drawdown'] * 100,
            line_dash="dot",
            line_color="darkred",
            annotation_text=f"Worst: {results['worst_drawdown']*100:.2f}%",
            annotation_position="bottom"
        )
        
        fig_dd.update_layout(
            title=f"Distribution of Maximum Drawdowns<br><sub>{results['n_simulations']:,} Simulations</sub>",
            xaxis_title="Maximum Drawdown (%)",
            yaxis_title="Frequency",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
        
        st.info("""
        **Chart Interpretation**:
        - **Coral histogram**: Distribution of maximum drawdowns across simulations
        - **Red dashed line**: Average maximum drawdown
        - **Dark red dotted line**: Worst-case drawdown observed
        - Shows how much your portfolio could decline from peak to trough
        """)
    
    with tab3:
        # Risk metrics comparison
        metrics_df = pd.DataFrame({
            'Metric': ['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%', 'Mean Drawdown', 'Worst Drawdown'],
            'Value': [
                results['var_95'],
                results['var_99'],
                results['cvar_95'],
                results['cvar_99'],
                results['mean_max_drawdown'],
                results['worst_drawdown']
            ]
        })
        
        fig_metrics = go.Figure()
        
        colors = ['#ff6b6b' if v < -0.20 else '#ffd93d' if v < -0.10 else '#6bcf7f' 
                 for v in metrics_df['Value']]
        
        fig_metrics.add_trace(go.Bar(
            x=metrics_df['Metric'],
            y=metrics_df['Value'] * 100,
            marker_color=colors,
            text=[f"{v*100:.2f}%" for v in metrics_df['Value']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
        ))
        
        fig_metrics.update_layout(
            title="Risk Metrics Comparison",
            xaxis_title="Risk Metric",
            yaxis_title="Loss (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_metrics, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(
                metrics_df.assign(
                    **{'Value (%)': metrics_df['Value'] * 100,
                       'Dollar Impact ($)': metrics_df['Value'] * portfolio_value}
                )[['Metric', 'Value (%)', 'Dollar Impact ($)']].style.format({
                    'Value (%)': '{:.2f}%',
                    'Dollar Impact ($)': '${:,.0f}'
                }).background_gradient(subset=['Value (%)'], cmap='RdYlGn_r'),
                use_container_width=True
            )
        
        with col2:
            st.info("""
            **Color Coding**:
            - 🟢 Green: Loss < 10% (manageable)
            - 🟡 Yellow: Loss 10-20% (moderate risk)
            - 🔴 Red: Loss > 20% (high risk)
            """)
    
    with tab4:
        # Probability ranges visualization
        return_ranges = [
            ('Severe Loss\n(< -20%)', (final_returns < -0.20).sum() / len(final_returns)),
            ('Moderate Loss\n(-20% to -10%)', ((final_returns >= -0.20) & (final_returns < -0.10)).sum() / len(final_returns)),
            ('Small Loss\n(-10% to 0%)', ((final_returns >= -0.10) & (final_returns < 0)).sum() / len(final_returns)),
            ('Small Gain\n(0% to 10%)', ((final_returns >= 0) & (final_returns < 0.10)).sum() / len(final_returns)),
            ('Moderate Gain\n(10% to 20%)', ((final_returns >= 0.10) & (final_returns < 0.20)).sum() / len(final_returns)),
            ('Large Gain\n(> 20%)', (final_returns >= 0.20).sum() / len(final_returns))
        ]
        
        ranges_df = pd.DataFrame(return_ranges, columns=['Range', 'Probability'])
        
        fig_ranges = go.Figure()
        
        colors_ranges = ['#d32f2f', '#f57c00', '#fbc02d', '#9ccc65', '#66bb6a', '#43a047']
        
        fig_ranges.add_trace(go.Bar(
            x=ranges_df['Range'],
            y=ranges_df['Probability'] * 100,
            marker_color=colors_ranges,
            text=[f"{p*100:.1f}%" for p in ranges_df['Probability']],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
        ))
        
        fig_ranges.update_layout(
            title="Probability Distribution by Return Range",
            xaxis_title="Return Range",
            yaxis_title="Probability (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_ranges, use_container_width=True)
        
        st.dataframe(
            ranges_df.assign(**{'Probability (%)': ranges_df['Probability'] * 100})[['Range', 'Probability (%)']].style.format({
                'Probability (%)': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Export Section
    st.header("📥 Export Crisis Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        export_df = pd.DataFrame({
            'Metric': [
                'Scenario', 'Duration (days)', 'Simulations', 'Portfolio Value',
                'Mean Return (%)', 'Median Return (%)', 'Std Return (%)',
                'Min Return (%)', 'Max Return (%)',
                'VaR 95% (%)', 'VaR 99% (%)',
                'CVaR 95% (%)', 'CVaR 99% (%)',
                'Dollar VaR 95%', 'Dollar VaR 99%',
                'Mean Max Drawdown (%)', 'Worst Drawdown (%)',
                'Avg Recovery Days', 'Prob Profit (%)', 'Prob Loss (%)', 'Prob Severe Loss (%)',
                'Recommendation', 'Rationale'
            ],
            'Value': [
                scenario.name, scenario.duration_days, results['n_simulations'], f"${portfolio_value:,}",
                f"{results['mean_return']*100:.2f}", f"{results['median_return']*100:.2f}", f"{results['std_return']*100:.2f}",
                f"{results['min_return']*100:.2f}", f"{results['max_return']*100:.2f}",
                f"{results['var_95']*100:.2f}", f"{results['var_99']*100:.2f}",
                f"{results['cvar_95']*100:.2f}", f"{results['cvar_99']*100:.2f}",
                f"${results['dollar_var_95']:,.0f}", f"${results['dollar_var_99']:,.0f}",
                f"{results['mean_max_drawdown']*100:.2f}", f"{results['worst_drawdown']*100:.2f}",
                f"{results['avg_recovery_days']:.0f}",
                f"{results['prob_profit']*100:.1f}", f"{results['prob_loss']*100:.1f}", f"{results['prob_severe_loss']*100:.1f}",
                results['recommendation'], results['rationale']
            ]
        })
        
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📄 Download CSV",
            data=csv_data,
            file_name=f"crisis_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Summary text
        summary_buffer = io.StringIO()
        summary_buffer.write("=== CRISIS SCENARIO ANALYSIS ===\n\n")
        summary_buffer.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        summary_buffer.write(f"Scenario: {scenario.name}\n")
        summary_buffer.write(f"Duration: {scenario.duration_days} days\n")
        summary_buffer.write(f"Simulations: {results['n_simulations']:,}\n")
        summary_buffer.write(f"Portfolio Value: ${portfolio_value:,}\n\n")
        summary_buffer.write("=== INVESTMENT RECOMMENDATION ===\n")
        summary_buffer.write(f"{results['recommendation']}\n")
        summary_buffer.write(f"{results['rationale']}\n\n")
        summary_buffer.write("=== KEY RISK METRICS ===\n")
        summary_buffer.write(f"VaR 95%: {results['var_95']*100:.2f}% (${results['dollar_var_95']:,.0f})\n")
        summary_buffer.write(f"CVaR 95%: {results['cvar_95']*100:.2f}%\n")
        summary_buffer.write(f"Worst Drawdown: {results['worst_drawdown']*100:.2f}%\n")
        summary_buffer.write(f"Recovery Time: {results['avg_recovery_days']:.0f} days\n\n")
        summary_buffer.write("=== RETURN STATISTICS ===\n")
        summary_buffer.write(f"Mean Return: {results['mean_return']*100:.2f}%\n")
        summary_buffer.write(f"Median Return: {results['median_return']*100:.2f}%\n")
        summary_buffer.write(f"Std Deviation: {results['std_return']*100:.2f}%\n\n")
        summary_buffer.write("=== PROBABILITIES ===\n")
        summary_buffer.write(f"Profit: {results['prob_profit']*100:.1f}%\n")
        summary_buffer.write(f"Loss: {results['prob_loss']*100:.1f}%\n")
        summary_buffer.write(f"Severe Loss (>20%): {results['prob_severe_loss']*100:.1f}%\n")
        
        summary_data = summary_buffer.getvalue()
        
        st.download_button(
            label="📝 Download Summary",
            data=summary_data,
            file_name=f"crisis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        st.info("""
        **Export Options**:
        - CSV: Detailed metrics table
        - TXT: Quick summary report
        
        Excel with charts coming soon!
        """)
    
    st.markdown("---")
    
    # Risk Management Strategies
    st.header("🛡️ Risk Management Strategies")
    
    # Calculate specific thresholds for recommendations
    var_95 = results['var_95']
    cvar_95 = results['cvar_95']
    worst_dd = results['worst_drawdown']
    prob_profit = results['prob_profit']
    recommendation = results['recommendation']
    
    # Get portfolio data for specific recommendations
    portfolio_data = st.session_state.portfolio_results
    tickers = portfolio_data['tickers']
    weights = portfolio_data['weights']
    
    # Identify top concentrated holdings
    allocation_df = pd.DataFrame({
        'Ticker': tickers,
        'Weight (%)': weights * 100
    }).sort_values('Weight (%)', ascending=False)
    
    top_3_weight = allocation_df.head(3)['Weight (%)'].sum()
    top_holding = allocation_df.iloc[0]
    
    if recommendation == "Worthy":
        st.success("✅ **Portfolio passed stress test!**")
        
        st.markdown("""
        Your portfolio shows **good resilience** under crisis conditions. However, maintaining this resilience requires ongoing risk management.
        """)
        
        # Specific actions for Worthy portfolios
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Recommended Actions")
            
            st.markdown(f"""
            **1. Set Stop-Loss Orders**
            - Primary level: **{var_95*1.1:.1f}%** (10% buffer above VaR)
            - Secondary level: **{var_95:.1f}%** (at VaR 95%)
            - Hard stop: **{cvar_95:.1f}%** (at CVaR 95%)
            
            **2. Position Monitoring**
            - Review portfolio **monthly**
            - Watch for VaR deterioration > 3 percentage points
            - Re-run crisis analysis if major market events occur
            
            **3. Concentration Management**
            - Top 3 holdings: **{top_3_weight:.1f}%**
            - {f"✅ Well diversified" if top_3_weight < 50 else f"⚠️ Consider rebalancing - top holding ({top_holding['Ticker']}) at {top_holding['Weight (%)']:.1f}%"}
            
            **4. Regular Rebalancing**
            - Quarterly rebalancing recommended
            - Trim positions that exceed 20% weight
            - Add to underweight positions
            """)
        
        with col2:
            st.subheader("⚡ Quick Wins")
            
            st.markdown(f"""
            **Immediate Actions** (This Week):
            - [ ] Set stop-loss orders at **{var_95*1.1:.1f}%**
            - [ ] Set up price alerts for portfolio holdings
            - [ ] Document current allocation for comparison
            
            **Short-term** (This Month):
            - [ ] Review and adjust if any holding > 25%
            - [ ] Set calendar reminder for monthly review
            - [ ] Prepare exit strategy document
            
            **Medium-term** (This Quarter):
            - [ ] Re-run crisis analysis monthly
            - [ ] Track actual returns vs. expected
            - [ ] Evaluate adding defensive positions
            """)
            
            st.info("""
            **💡 Pro Tip**: Even "Worthy" portfolios can face unexpected events. 
            Maintain vigilance and stick to your risk management plan.
            """)
    
    elif recommendation == "Proceed with Caution":
        st.warning("⚠️ **Portfolio has moderate vulnerabilities**")
        
        st.markdown("""
        Your portfolio shows **moderate risk** under crisis conditions. Immediate action is recommended to improve resilience.
        """)
        
        # Calculate specific adjustments
        position_reduction = min(30, max(15, abs(var_95 * 100) - 15))  # 15-30% reduction
        target_holdings = min(len(tickers) + 3, 15)  # Add 3 more stocks
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Critical Actions Required")
            
            st.markdown(f"""
            **1. Position Size Reduction** 🔴 PRIORITY
            - Reduce overall exposure by **{position_reduction:.0f}%**
            - Method: Trim top 3 holdings by 20-30% each
            - Top holding **{top_holding['Ticker']}** ({top_holding['Weight (%)']:.1f}%) → Target: < 15%
            
            **2. Increase Diversification** 🔴 PRIORITY
            - Current holdings: **{len(tickers)} stocks**
            - Target: **{target_holdings} stocks** (add {target_holdings - len(tickers)} more)
            - Focus on low-correlation sectors
            
            **3. Add Defensive Positions**
            - Allocate **15-20%** to defensive sectors:
              - Utilities (stable dividends)
              - Consumer Staples (recession-resistant)
              - Healthcare (inelastic demand)
            
            **4. Tighter Stop-Loss Orders**
            - Primary stop: **{var_95*0.8:.1f}%** (20% tighter than VaR)
            - Hard stop: **{var_95:.1f}%** (at VaR 95%)
            - No exceptions on hard stop
            
            **5. Hedging Strategies**
            - Consider protective puts on largest holdings
            - Collar strategy: Buy puts + Sell calls
            - Allocate 2-5% of portfolio to hedging costs
            """)
        
        with col2:
            st.subheader("📋 Action Plan Timeline")
            
            st.markdown(f"""
            **URGENT** (Next 48 Hours):
            - [ ] Set stop-loss orders at **{var_95*0.8:.1f}%**
            - [ ] Identify 3-5 defensive stocks to add
            - [ ] Calculate rebalancing trades
            
            **This Week**:
            - [ ] Execute position size reduction ({position_reduction:.0f}%)
            - [ ] Add {target_holdings - len(tickers)} defensive stocks
            - [ ] Review correlation matrix
            - [ ] Set up daily portfolio monitoring
            
            **This Month**:
            - [ ] Implement hedging strategy (if appropriate)
            - [ ] Re-run optimization with higher risk aversion
            - [ ] Re-test with crisis scenarios
            - [ ] Verify improvement in VaR/CVaR
            
            **Ongoing**:
            - [ ] Weekly portfolio review
            - [ ] Monthly crisis re-testing
            - [ ] Adjust strategy based on results
            """)
            
            st.warning(f"""
            **⚠️ Warning Thresholds**:
            
            If any of these occur, **reduce exposure immediately**:
            - Any single position > 25%
            - VaR worsens to < **{var_95*1.2:.1f}%**
            - Market volatility doubles
            - Major geopolitical event in your sectors
            """)
    
    else:  # Not Recommended
        st.error("❌ **Portfolio shows high vulnerability**")
        
        st.markdown("""
        Your portfolio shows **high vulnerability** to crisis conditions. **Immediate and significant action is required** to avoid potential large losses.
        """)
        
        # Calculate drastic adjustments
        exposure_reduction = 50  # 50% reduction
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 URGENT Actions Required")
            
            st.markdown(f"""
            **⛔ STOP - Do Not Proceed with Current Portfolio**
            
            Your portfolio is **not suitable** for investment under crisis conditions.
            
            ---
            
            **Option A: Redesign Portfolio** (RECOMMENDED)
            
            **1. Return to Portfolio Optimization**
            - Increase portfolio size: **{len(tickers)} → {min(len(tickers)*2, 20)} stocks**
            - Increase risk aversion: **{st.session_state.portfolio_results.get('risk_aversion', 2.5):.1f} → {min(st.session_state.portfolio_results.get('risk_aversion', 2.5) + 1.5, 5.0):.1f}**
            - Goal: Get VaR to < **-20%**
            
            **2. Return to CAPM Analysis**
            - Lower alpha threshold: Select more stable stocks
            - Focus on high R² stocks (> 0.6)
            - Avoid highly volatile stocks
            
            **3. Diversification Strategy**
            - Add **10-15 stocks** minimum
            - Include 3-5 defensive sectors
            - Max single position: **10%**
            - Top 5 holdings: < **40%** total
            
            **4. Risk Budget Allocation**
            - Defensive (40%): Utilities, Staples, Healthcare
            - Core (40%): Quality Energy stocks (high R²)
            - Satellite (20%): Higher growth potential
            
            ---
            
            **Option B: Reduce Exposure Drastically**
            
            If you must proceed with current portfolio:
            
            **1. Cut Position Size by 50%**
            - Only invest **${portfolio_value/2:,.0f}** instead of ${portfolio_value:,.0f}
            - Keep rest in cash or bonds
            - This limits maximum loss
            
            **2. Ultra-Tight Stop-Loss**
            - Hard stop at **{max(var_95, -0.15):.1f}%** (15% max loss)
            - No emotional exceptions
            - Automate the stop order
            
            **3. Aggressive Hedging**
            - Buy protective puts on all holdings
            - Cost: 5-10% of portfolio value
            - Accept the insurance cost
            
            **4. Plan for Exit**
            - Set specific conditions for full exit
            - Have cash ready for opportunities
            - Don't average down on losses
            """)
        
        with col2:
            st.subheader("❌ Do NOT Proceed Until...")
            
            st.markdown(f"""
            **Minimum Requirements Before Investing**:
            
            - [ ] VaR 95% improves to > **-20%**
            - [ ] CVaR 95% improves to > **-30%**
            - [ ] Worst Drawdown < **-50%**
            - [ ] Probability of Profit > **40%**
            - [ ] Portfolio size ≥ **12 stocks**
            - [ ] Top 3 concentration < **45%**
            - [ ] Crisis recommendation = "Caution" or better
            
            ---
            
            **Why This Portfolio is Too Risky**:
            
            📊 **Current Risk Metrics**:
            - VaR 95%: **{var_95*100:.1f}%**
            - CVaR 95%: **{cvar_95*100:.1f}%**
            - Worst Drawdown: **{worst_dd*100:.1f}%**
            - Prob of Profit: **{prob_profit*100:.1f}%**
            
            ⚠️ **What This Means**:
            - **{(1-prob_profit)*100:.0f}% chance** you will lose money
            - **{results['prob_severe_loss']*100:.0f}% chance** of losing > 20%
            - Potential loss: **${abs(var_95)*portfolio_value:,.0f}** (VaR)
            - Worst case: **${abs(worst_dd)*portfolio_value:,.0f}** (Drawdown)
            
            💰 **On ${portfolio_value:,.0f} Investment**:
            - Expected loss in crisis: **${abs(results['mean_return'])*portfolio_value:,.0f}**
            - 95% VaR loss: **${abs(var_95)*portfolio_value:,.0f}**
            - Worst case loss: **${abs(worst_dd)*portfolio_value:,.0f}**
            
            **Can you afford to lose this much?**
            
            If answer is NO → **Do not invest** with this portfolio.
            """)
            
            st.error("""
            **🚨 STRONG RECOMMENDATION**:
            
            1. **Do NOT proceed** with this portfolio
            2. **Redesign** using safer parameters
            3. **Re-test** until you get "Caution" or "Worthy"
            4. **Only then** consider investing
            
            This is not a suggestion - it's a strong warning based on your risk metrics.
            """)
    
    st.markdown("---")
    
    # Additional Risk Management Resources
    st.header("📚 Risk Management Resources")
    
    tab1, tab2, tab3 = st.tabs(["📖 Best Practices", "🔧 Tools & Techniques", "📞 When to Seek Help"])
    
    with tab1:
        st.markdown("""
        ### Portfolio Risk Management Best Practices
        
        **1. The 2% Rule**
        - Never risk more than 2% of portfolio on any single position
        - For ${:,.0f} portfolio: Max loss per stock = ${:,.0f}
        - Use position sizing to enforce this
        
        **2. Diversification Guidelines**
        - Minimum 10 stocks for retail portfolios
        - Maximum 20% in any single sector
        - Include 3-5 different sectors
        - Consider international diversification
        
        **3. Rebalancing Discipline**
        - Rebalance quarterly or when drift > 5%
        - Trim winners that exceed target allocation
        - Add to losers that remain fundamentally strong
        - Avoid emotional rebalancing during volatility
        
        **4. Stop-Loss Discipline**
        - Set stops at time of purchase
        - Use mental stops if automatic execution is costly
        - Stick to stops - no exceptions
        - Re-evaluate after stop-out, don't re-enter immediately
        
        **5. Position Sizing Formula**
        ```
        Position Size = (Portfolio Value × Risk per Trade) / (Entry Price - Stop Price)
        
        Example for 2% risk:
        Position = ($100,000 × 0.02) / ($50 - $45) = $2,000 / $5 = 400 shares
        ```
        
        **6. Crisis Monitoring**
        - Run crisis analysis monthly
        - Watch for VaR deterioration
        - Increase monitoring frequency during high volatility
        - Have pre-planned response to VaR threshold breaches
        """.format(portfolio_value, portfolio_value * 0.02))
    
    with tab2:
        st.markdown(f"""
        ### Risk Management Tools & Techniques
        
        **1. Stop-Loss Strategies**
        
        **Fixed Percentage Stop**:
        - Set stop at VaR level: **{var_95*1.1:.1f}%** (with buffer)
        - Pros: Simple, objective
        - Cons: May be triggered by normal volatility
        
        **Trailing Stop**:
        - Trail stop by 15-20% below peak
        - Locks in profits while giving room for growth
        - Adjust upward only, never downward
        
        **Time-Based Stop**:
        - Exit if position hasn't moved favorably in X days
        - Useful for preventing dead money
        - Combine with price stops
        
        ---
        
        **2. Hedging Techniques**
        
        **Protective Puts**:
        - Buy put options on individual holdings
        - Strike price at your stop-loss level
        - Cost: 2-5% of position value (insurance premium)
        - Best for: Large concentrated positions
        
        **Collar Strategy**:
        - Buy protective put (downside protection)
        - Sell call option (finance the put)
        - Net cost: Low or zero
        - Trade-off: Cap upside for downside protection
        
        **Portfolio Insurance**:
        - Buy index puts (e.g., on market index)
        - Cheaper than individual stock puts
        - Protects against systematic risk only
        
        ---
        
        **3. Position Sizing Techniques**
        
        **Equal Weight**:
        - Each stock = 1/N of portfolio
        - Simple and prevents over-concentration
        - May not be optimal for returns
        
        **Risk Parity**:
        - Size positions by volatility
        - Low volatility stocks = larger positions
        - High volatility stocks = smaller positions
        
        **Kelly Criterion**:
        - Optimal position size based on win probability
        - Formula: f = (p × b - q) / b
        - Use fractional Kelly (e.g., 25%) for safety
        
        ---
        
        **4. Monitoring Dashboard**
        
        Create a spreadsheet to track:
        - [ ] Current VaR (re-calculate monthly)
        - [ ] Position sizes (% of portfolio)
        - [ ] Stop-loss levels for each holding
        - [ ] Unrealized gains/losses
        - [ ] Sector exposures
        - [ ] Correlation matrix (quarterly)
        - [ ] Maximum drawdown (current)
        - [ ] Recovery time (if in drawdown)
        
        ---
        
        **5. Stress Testing Schedule**
        
        **Monthly** (Quick check):
        - Re-run crisis scenarios
        - Compare VaR to last month
        - Check for new concentrated positions
        
        **Quarterly** (Deep dive):
        - Full portfolio re-optimization
        - Review all three crisis scenarios
        - Correlation matrix update
        - Rebalancing decisions
        
        **Event-Driven** (When these occur):
        - Major geopolitical events
        - Market volatility spike (VIX > 30)
        - Individual stock news (earnings, scandals)
        - Macro changes (Fed rate changes, recession indicators)
        """)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚨 Seek Professional Help If:
            
            **Portfolio Thresholds**:
            - Portfolio value > $250,000
            - Single investment > $100,000
            - Retirement funds at stake
            
            **Risk Concerns**:
            - VaR consistently < -25%
            - Can't improve portfolio despite adjustments
            - Don't understand risk metrics
            - Making emotional decisions
            
            **Life Changes**:
            - Approaching retirement (< 10 years)
            - Need portfolio income
            - Major life events (marriage, inheritance)
            - Risk tolerance uncertainty
            
            **Complex Needs**:
            - Tax optimization
            - Estate planning
            - Multi-asset allocation
            - International exposure
            
            ---
            
            ### ⚠️ Red Flags - Get Help NOW:
            
            🚨 **Stop if you're**:
            - Losing sleep over investments
            - Checking prices constantly
            - Making impulsive trades
            - Ignoring your stop-losses
            - Portfolio affects your mood
            - Considering debt to invest
            
            *These are signs of emotional/gambling behavior.*
            """)
        
        with col2:
            st.markdown("""
            ### 👔 Finding a Financial Advisor
            
            **Look for**:
            - ✅ CFP (Certified Financial Planner)
            - ✅ Fiduciary duty
            - ✅ Fee-only (no commissions)
            - ✅ Experience with your portfolio size
            
            **Key Questions**:
            1. Are you a fiduciary?
            2. How are you compensated?
            3. What's your investment philosophy?
            4. How do you manage risk?
            5. Review frequency?
            
            ---
            
            ### 📚 Self-Help Resources
            
            **Books**:
            - "A Random Walk Down Wall Street" - Malkiel
            - "The Intelligent Investor" - Graham
            
            **Online**:
            - Coursera: "Financial Markets" (Yale)
            - Khan Academy: Finance section
            - Portfolio Visualizer (portfoliovisualizer.com)
            
            **Tools**:
            - Morningstar Portfolio Manager
            - Personal Capital (free tracking)
            
            ---
            
            ### 💡 General Guidance
            
            **DIY Appropriate For**:
            - Portfolio < $100,000
            - Long time horizon (> 10 years)
            - Stable emotions
            - Willing to learn
            
            **Advisor Recommended For**:
            - Portfolio > $250,000
            - Near retirement
            - Complex tax situations
            - Peace of mind needed
            """)
            
        st.info("""
        **Remember**: This tool provides analysis, not financial advice. 
        Use it as input for your own decisions or discussions with your advisor.
        """)
    
    st.markdown("---")
    
    # Next steps navigation
    st.header("🎯 Next Steps")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Portfolio Optimization", use_container_width=True):
            st.switch_page("pages/2_💼_Portfolio_Optimization.py")
    with col2:
        st.button("Next: Forecasting →", type="primary", use_container_width=True, disabled=True)
        st.caption("Coming soon!")

else:
    # Instructions
    st.info("""
    👋 **Welcome to Crisis Scenarios Analysis!**
    
    This module tests your portfolio under extreme market conditions:
    - Monte Carlo simulation with 10,000+ scenarios
    - Multiple crisis scenarios (US-Iran tension, oil shocks, recessions)
    - Comprehensive risk metrics (VaR, CVaR, Max Drawdown)
    - Investment worthiness assessment
    
    **How to use**:
    1. Select a crisis scenario from the dropdown
    2. Configure number of simulations (more = more accurate)
    3. Enter your portfolio value for dollar VaR calculation
    4. Click "Run Crisis Simulation"
    5. Review risk metrics and investment recommendation
    6. Export results for your records
    
    **Click the button above to start testing!** 🚀
    """)
