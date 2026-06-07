# 🚀 Portfolio Analysis Dashboard - Deployment Package

## 📦 What's Included

This deployment package contains everything you need to run the Portfolio Analysis Dashboard application.

### Directory Structure
```
deployment_package/
├── Home.py                          # Main dashboard entry point
├── requirements.txt                 # Python dependencies
├── runtime.txt                      # Python version specification
├── data_gabungan_skripsi.xlsx      # Default dataset
├── sample_bi_rate.csv              # Sample BI Rate data
├── README.md                        # This file
├── DEPLOYMENT_INSTRUCTIONS.md       # Detailed deployment guide
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
├── pages/                          # Dashboard pages
│   ├── 1_📊_CAPM_Analysis.py
│   ├── 2_💼_Portfolio_Optimization.py
│   ├── 3_⚠️_Crisis_Scenarios.py
│   ├── 4_🔍_Data_Validation.py
│   ├── 5_📈_Forecasting.py
│   └── 6_🛡️_Risk_Management.py
├── src/                            # Source code modules
│   ├── capm/                       # CAPM analysis
│   ├── black_litterman/            # Portfolio optimization
│   ├── crisis/                     # Crisis scenarios
│   ├── forecasting/                # Forecasting models
│   ├── data/                       # Data loading
│   ├── lstm/                       # LSTM models
│   └── reporting/                  # Report generation
├── data/                           # Additional data files
└── output/                         # Generated outputs

```

## 🎯 Quick Start

### Option 1: Local Deployment (Recommended for Testing)

1. **Install Python 3.9+**
   - Download from https://www.python.org/downloads/

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Dashboard**
   ```bash
   streamlit run Home.py
   ```

5. **Access the Dashboard**
   - Open browser to: http://localhost:8501

### Option 2: Streamlit Cloud (Free Hosting)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial deployment"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `Home.py`
   - Click "Deploy"

3. **Your app will be live at:**
   `https://your-app-name.streamlit.app`

### Option 3: Docker Deployment

1. **Create Dockerfile** (see DEPLOYMENT_INSTRUCTIONS.md)

2. **Build Image**
   ```bash
   docker build -t portfolio-analysis .
   ```

3. **Run Container**
   ```bash
   docker run -p 8501:8501 portfolio-analysis
   ```

## 📋 System Requirements

- **Python**: 3.9 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 500MB free space
- **OS**: Windows, Linux, or macOS

## 🔧 Configuration

### Data Files
- Default data: `data_gabungan_skripsi.xlsx`
- You can upload your own data through the dashboard
- Required format: Excel with Date, Stock Prices, Market Index, BI Rate

### Analysis Parameters
Configure in the Home page:
- Risk Aversion (λ): 1.0 - 5.0
- Alpha Threshold: 0% - 5%
- Portfolio Size: 3 - 30 stocks
- Monte Carlo Simulations: 1,000 - 50,000

## 📊 Features

### 1. CAPM Analysis
- Identify undervalued/overvalued stocks
- Security Market Line visualization
- Beta, Alpha, R² calculations
- Export results to CSV/Excel

### 2. Portfolio Optimization
- Black-Litterman model
- Optimal asset allocation
- Expected return & volatility
- Sharpe ratio calculation

### 3. Crisis Scenarios
- Monte Carlo simulation
- Value at Risk (VaR)
- Multiple crisis scenarios
- Investment recommendations

### 4. Forecasting
- LSTM, ARIMA, GARCH models
- Price & volatility predictions
- Performance metrics
- Multi-stock forecasting

### 5. Data Validation
- Data quality checks
- Missing value detection
- Statistical summaries
- Correlation analysis

### 6. Risk Management
- Portfolio risk metrics
- Diversification analysis
- Risk-adjusted returns
- Stress testing

## 🆘 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**2. Port Already in Use**
```bash
# Use different port
streamlit run Home.py --server.port 8502
```

**3. Memory Issues**
- Reduce Monte Carlo simulations
- Use smaller dataset
- Close other applications

**4. Data Loading Errors**
- Check Excel file format
- Ensure Date column is first
- Verify column names

## 📚 Documentation

- **DEPLOYMENT_INSTRUCTIONS.md** - Detailed deployment guide
- **User Guide** - Available in the dashboard Help section
- **API Documentation** - In `src/` module docstrings

## 🔐 Security Notes

- This is for educational/research purposes
- Not financial advice
- Keep sensitive data secure
- Use environment variables for credentials

## 📞 Support

For issues or questions:
1. Check DEPLOYMENT_INSTRUCTIONS.md
2. Review error messages in terminal
3. Check Streamlit documentation: https://docs.streamlit.io

## 📄 License

This project is for academic/educational use.

## 🎓 Citation

If you use this system in your research, please cite appropriately.

---

**Version**: 2.0  
**Last Updated**: June 2026  
**Python Version**: 3.9+  
**Streamlit Version**: Latest

---

## 🚀 Next Steps

1. ✅ Install dependencies
2. ✅ Run the dashboard
3. ✅ Upload your data
4. ✅ Configure parameters
5. ✅ Run analysis
6. ✅ Export results

**Ready to deploy? Follow the Quick Start guide above!**