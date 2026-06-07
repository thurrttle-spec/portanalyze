"""
Academic theory and equations for forecasting models.

Provides theoretical background, mathematical formulations,
and academic references for LSTM, ARIMA, and GARCH models.
"""


class ModelTheory:
    """Provides academic theory and equations for forecasting models."""
    
    @staticmethod
    def lstm_theory():
        """Return LSTM theory and equations."""
        return {
            'name': 'LSTM - Long Short-Term Memory',
            'category': 'Deep Learning / Recurrent Neural Networks',
            'description': """
            LSTM is a type of Recurrent Neural Network (RNN) specifically designed to 
            learn long-term dependencies in sequential data. It addresses the vanishing 
            gradient problem in traditional RNNs through its gating mechanisms.
            """,
            'equations': {
                'forget_gate': r'f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)',
                'input_gate': r'i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)',
                'candidate': r'\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)',
                'cell_state': r'C_t = f_t * C_{t-1} + i_t * \tilde{C}_t',
                'output_gate': r'o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)',
                'hidden_state': r'h_t = o_t * \tanh(C_t)'
            },
            'components': {
                'Forget Gate (f_t)': 'Decides what information to discard from cell state',
                'Input Gate (i_t)': 'Decides what new information to store in cell state',
                'Cell State (C_t)': 'Carries information across time steps',
                'Output Gate (o_t)': 'Decides what information to output',
                'Hidden State (h_t)': 'Output passed to next time step'
            },
            'advantages': [
                'Captures long-term dependencies in time series',
                'Handles non-linear patterns effectively',
                'No assumptions about data distribution',
                'Can model complex relationships'
            ],
            'limitations': [
                'Requires large amounts of training data',
                'Computationally expensive',
                'Black box model (less interpretable)',
                'Prone to overfitting with small datasets'
            ],
            'references': [
                'Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.',
                'Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669.',
                'Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. Neural computation, 12(10), 2451-2471.'
            ],
            'use_case': 'Best for: Complex non-linear patterns, long-term dependencies, sufficient historical data'
        }
    
    @staticmethod
    def arima_theory():
        """Return ARIMA theory and equations."""
        return {
            'name': 'ARIMA - AutoRegressive Integrated Moving Average',
            'category': 'Classical Time Series Econometrics',
            'description': """
            ARIMA combines three components: AutoRegression (AR), Integration (I), and 
            Moving Average (MA). It is one of the most widely used methods for time series 
            forecasting in finance and economics.
            """,
            'model': 'ARIMA(p, d, q)',
            'parameters': {
                'p': 'Order of autoregression (AR) - number of lag observations',
                'd': 'Degree of differencing (I) - number of times data is differenced',
                'q': 'Order of moving average (MA) - size of moving average window'
            },
            'equations': {
                'general': r'(1 - \sum_{i=1}^{p} \phi_i L^i)(1-L)^d X_t = (1 + \sum_{i=1}^{q} \theta_i L^i) \epsilon_t',
                'ar_component': r'X_t = c + \sum_{i=1}^{p} \phi_i X_{t-i} + \epsilon_t',
                'ma_component': r'X_t = \mu + \epsilon_t + \sum_{i=1}^{q} \theta_i \epsilon_{t-i}',
                'integrated': r'Y_t = (1-L)^d X_t',
                'full_arima': r'Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \epsilon_t + \sum_{i=1}^{q} \theta_i \epsilon_{t-i}'
            },
            'components': {
                'φ_i': 'Autoregressive coefficients',
                'θ_i': 'Moving average coefficients',
                'L': 'Lag operator: L(X_t) = X_{t-1}',
                'ε_t': 'White noise error term',
                'c': 'Constant term',
                'd': 'Order of differencing'
            },
            'assumptions': [
                'Stationarity (after differencing)',
                'Linear relationships',
                'Constant variance (homoscedasticity)',
                'Normally distributed errors',
                'No autocorrelation in residuals'
            ],
            'advantages': [
                'Well-established theoretical foundation',
                'Interpretable parameters',
                'Works well with limited data',
                'Fast computation',
                'Confidence intervals available'
            ],
            'limitations': [
                'Assumes linear relationships',
                'Requires stationary data',
                'May miss non-linear patterns',
                'Sensitive to outliers'
            ],
            'references': [
                'Box, G. E. P., & Jenkins, G. M. (1976). Time series analysis: Forecasting and control. Holden-Day.',
                'Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts.',
                'Ariyo, A. A., Adewumi, A. O., & Ayo, C. K. (2014). Stock price prediction using the ARIMA model. UKSim-AMSS 16th International Conference on Computer Modelling and Simulation.'
            ],
            'use_case': 'Best for: Linear trends, seasonal patterns, economic/financial time series'
        }
    
    @staticmethod
    def garch_theory():
        """Return GARCH theory and equations."""
        return {
            'name': 'GARCH - Generalized AutoRegressive Conditional Heteroskedasticity',
            'category': 'Volatility Modeling / Risk Management',
            'description': """
            GARCH models the time-varying volatility (heteroskedasticity) in financial returns.
            It is particularly useful for modeling and forecasting volatility clustering, where
            high volatility periods tend to cluster together.
            """,
            'model': 'GARCH(p, q)',
            'parameters': {
                'p': 'Order of GARCH terms (lagged variance terms)',
                'q': 'Order of ARCH terms (lagged squared residual terms)'
            },
            'equations': {
                'return': r'r_t = \mu + \epsilon_t',
                'error': r'\epsilon_t = \sigma_t z_t, \quad z_t \sim N(0,1)',
                'variance': r'\sigma_t^2 = \omega + \sum_{i=1}^{q} \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j \sigma_{t-j}^2',
                'garch11': r'\sigma_t^2 = \omega + \alpha_1 \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2',
                'constraints': r'\omega > 0, \quad \alpha_i \geq 0, \quad \beta_j \geq 0, \quad \sum_{i=1}^{max(p,q)} (\alpha_i + \beta_i) < 1'
            },
            'components': {
                'σ²_t': 'Conditional variance (volatility squared) at time t',
                'ω': 'Constant term (long-run variance)',
                'α_i': 'ARCH coefficients (impact of past shocks)',
                'β_j': 'GARCH coefficients (persistence of volatility)',
                'ε_t': 'Error term (innovations)',
                'z_t': 'Standardized residuals'
            },
            'features': [
                'Volatility clustering: Large changes followed by large changes',
                'Fat tails: Higher probability of extreme events than normal distribution',
                'Mean reversion: Volatility returns to long-run average',
                'Asymmetric responses (with extensions like EGARCH, GJR-GARCH)'
            ],
            'advantages': [
                'Captures volatility clustering effectively',
                'Flexible model for financial returns',
                'Provides volatility forecasts',
                'Well-suited for risk management',
                'Widely used in finance industry'
            ],
            'limitations': [
                'Models volatility, not price levels',
                'Symmetric response to positive/negative shocks (in basic GARCH)',
                'Assumes specific error distribution',
                'Parameter estimation can be unstable'
            ],
            'references': [
                'Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of econometrics, 31(3), 307-327.',
                'Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 987-1007.',
                'Hansen, P. R., & Lunde, A. (2005). A forecast comparison of volatility models: does anything beat a GARCH(1,1)?. Journal of applied econometrics, 20(7), 873-889.'
            ],
            'use_case': 'Best for: Volatility forecasting, risk management, option pricing'
        }
    
    @staticmethod
    def get_theory(model_name):
        """Get theory for specified model."""
        theories = {
            'LSTM': ModelTheory.lstm_theory(),
            'ARIMA': ModelTheory.arima_theory(),
            'GARCH': ModelTheory.garch_theory()
        }
        return theories.get(model_name, {})
    
    @staticmethod
    def comparison_table():
        """Return comparison table of all models."""
        return {
            'headers': ['Aspect', 'LSTM', 'ARIMA', 'GARCH'],
            'rows': [
                ['Type', 'Machine Learning', 'Statistical', 'Econometric'],
                ['Best For', 'Prices/Complex Patterns', 'Returns/Linear Trends', 'Volatility'],
                ['Data Required', 'Large (500+ points)', 'Medium (100+ points)', 'Large (252+ points)'],
                ['Interpretability', 'Low (Black Box)', 'High', 'Medium'],
                ['Computation', 'Slow', 'Fast', 'Medium'],
                ['Non-linearity', 'Excellent', 'Poor', 'Medium'],
                ['Volatility Modeling', 'Implicit', 'No', 'Explicit'],
                ['Confidence Intervals', 'Difficult', 'Available', 'Available'],
                ['Assumptions', 'None', 'Stationarity', 'Specific Distribution']
            ]
        }
