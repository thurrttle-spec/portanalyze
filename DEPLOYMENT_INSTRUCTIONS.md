# 📖 Detailed Deployment Instructions

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Heroku Deployment](#heroku-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Python 3.9 or higher** - [Download](https://www.python.org/downloads/)
- **Git** (for cloud deployment) - [Download](https://git-scm.com/downloads)
- **Text Editor** - VS Code, Notepad++, etc.

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 500MB free space
- **Internet**: Required for package installation and cloud deployment

---

## Local Deployment

### Step 1: Extract Files
Extract the `deployment_package` folder to your desired location, e.g.:
- Windows: `C:\Users\YourName\portfolio-analysis`
- Mac/Linux: `~/portfolio-analysis`

### Step 2: Open Terminal/Command Prompt
Navigate to the deployment folder:
```bash
cd path/to/deployment_package
```

### Step 3: Create Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 4: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- Streamlit (dashboard framework)
- Pandas, NumPy (data processing)
- TensorFlow (machine learning)
- Matplotlib, Plotly (visualizations)
- And all other required packages

**Installation time**: 5-10 minutes depending on internet speed.

### Step 5: Verify Installation
```bash
python -c "import streamlit; import pandas; import tensorflow; print('All packages installed successfully!')"
```

### Step 6: Run the Dashboard
```bash
streamlit run Home.py
```

Expected output:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

### Step 7: Access Dashboard
Open your web browser and go to: **http://localhost:8501**

### Step 8: Test the Application
1. The Home page should load
2. Click "Use default data"
3. Click "Confirm Configuration"
4. Navigate to "📊 CAPM Analysis" in sidebar
5. Click "Run CAPM Analysis"

If analysis completes successfully, your deployment is working! ✅

---

## Streamlit Cloud Deployment

### Why Streamlit Cloud?
- ✅ **Free hosting**
- ✅ **Automatic updates** from GitHub
- ✅ **Custom URL**: `your-app.streamlit.app`
- ✅ **No server management**

### Step 1: Create GitHub Repository

1. **Create account** at https://github.com (if you don't have one)

2. **Create new repository**:
   - Click "+" → "New repository"
   - Name: `portfolio-analysis-dashboard`
   - Description: "Financial Portfolio Analysis Dashboard"
   - Public or Private (your choice)
   - Don't initialize with README
   - Click "Create repository"

### Step 2: Push Code to GitHub

**Windows (Command Prompt):**
```bash
cd deployment_package
git init
git add .
git commit -m "Initial deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/portfolio-analysis-dashboard.git
git push -u origin main
```

**Mac/Linux (Terminal):**
```bash
cd deployment_package
git init
git add .
git commit -m "Initial deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/portfolio-analysis-dashboard.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 3: Deploy on Streamlit Cloud

1. **Go to** https://streamlit.io/cloud

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure deployment**:
   - Repository: `YOUR_USERNAME/portfolio-analysis-dashboard`
   - Branch: `main`
   - Main file path: `Home.py`
   - App URL: Choose a custom name (e.g., `my-portfolio-analysis`)

5. **Click "Deploy"**

6. **Wait for deployment** (2-5 minutes)

7. **Your app is live!** 🎉
   - URL: `https://your-app-name.streamlit.app`

### Step 4: Update Your App

When you make changes:
```bash
git add .
git commit -m "Update description"
git push
```

Streamlit Cloud will automatically redeploy! 🚀

---

## Docker Deployment

### Why Docker?
- ✅ **Consistent environment** across all systems
- ✅ **Easy to share** with team members
- ✅ **Isolated dependencies**
- ✅ **Production-ready**

### Step 1: Install Docker
Download from: https://www.docker.com/products/docker-desktop

### Step 2: Create Dockerfile

Create a file named `Dockerfile` (no extension) in `deployment_package`:

```dockerfile
# Use official Python runtime
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 3: Create .dockerignore

Create `.dockerignore` file:
```
venv/
__pycache__/
*.pyc
.git/
.gitignore
*.md
output/*.png
output/*.csv
.kiro/
```

### Step 4: Build Docker Image

```bash
cd deployment_package
docker build -t portfolio-analysis:latest .
```

Build time: 5-10 minutes (first time only)

### Step 5: Run Docker Container

```bash
docker run -p 8501:8501 portfolio-analysis:latest
```

Access at: http://localhost:8501

### Step 6: Run with Volume (Persistent Data)

```bash
docker run -p 8501:8501 -v ${PWD}/output:/app/output portfolio-analysis:latest
```

This saves outputs to your local `output` folder.

### Step 7: Stop Container

Press `Ctrl+C` or:
```bash
docker ps  # Find container ID
docker stop <container-id>
```

---

## Heroku Deployment

### Why Heroku?
- ✅ **Free tier available**
- ✅ **Easy deployment**
- ✅ **Custom domain support**
- ✅ **Automatic SSL**

### Step 1: Install Heroku CLI
Download from: https://devcenter.heroku.com/articles/heroku-cli

### Step 2: Create Heroku Account
Sign up at: https://signup.heroku.com/

### Step 3: Create Procfile

Create `Procfile` (no extension) in `deployment_package`:
```
web: sh setup.sh && streamlit run Home.py
```

### Step 4: Create setup.sh

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### Step 5: Deploy to Heroku

```bash
cd deployment_package
heroku login
heroku create your-app-name
git init
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### Step 6: Open Your App

```bash
heroku open
```

Your app is live at: `https://your-app-name.herokuapp.com`

---

## Troubleshooting

### Issue 1: "Module not found" Error

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue 2: Port Already in Use

**Solution:**
```bash
# Use different port
streamlit run Home.py --server.port 8502
```

### Issue 3: TensorFlow Installation Fails

**Windows Solution:**
```bash
pip install tensorflow-cpu
```

**Mac M1/M2 Solution:**
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

### Issue 4: Streamlit Cloud Build Fails

**Check:**
1. `requirements.txt` is in root directory
2. `Home.py` path is correct
3. Python version in `runtime.txt` is 3.9 or higher

**Fix runtime.txt:**
```
python-3.9.18
```

### Issue 5: Data File Not Found

**Solution:**
Ensure `data_gabungan_skripsi.xlsx` is in the same directory as `Home.py`

### Issue 6: Memory Error on Streamlit Cloud

**Solution:**
Reduce Monte Carlo simulations in the dashboard settings (use 1,000 instead of 10,000)

### Issue 7: Slow Performance

**Solutions:**
1. Use smaller dataset
2. Reduce number of stocks
3. Lower simulation count
4. Close other browser tabs

### Issue 8: Docker Build Fails

**Solution:**
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t portfolio-analysis:latest .
```

### Issue 9: Git Push Fails

**Solution:**
```bash
# Check remote
git remote -v

# Reset remote
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

### Issue 10: Dashboard Shows Blank Page

**Solutions:**
1. Clear browser cache (Ctrl+Shift+Delete)
2. Try incognito/private mode
3. Check browser console for errors (F12)
4. Restart Streamlit server

---

## Performance Optimization

### For Large Datasets
1. Use data sampling for initial analysis
2. Enable caching in Streamlit
3. Reduce visualization complexity
4. Use batch processing

### For Faster Deployment
1. Use Docker for consistent builds
2. Cache pip packages
3. Minimize dependencies
4. Use CDN for static assets

---

## Security Best Practices

1. **Don't commit sensitive data** to GitHub
2. **Use environment variables** for credentials
3. **Enable authentication** on Streamlit Cloud (Settings → Sharing)
4. **Keep dependencies updated**: `pip install --upgrade -r requirements.txt`
5. **Use HTTPS** in production

---

## Maintenance

### Regular Updates
```bash
# Update packages
pip install --upgrade -r requirements.txt

# Update Streamlit
pip install --upgrade streamlit

# Test after updates
streamlit run Home.py
```

### Backup Data
Regularly backup:
- `data/` folder
- `output/` folder
- Custom configurations

---

## Getting Help

### Resources
- **Streamlit Docs**: https://docs.streamlit.io
- **Python Docs**: https://docs.python.org
- **Docker Docs**: https://docs.docker.com
- **Heroku Docs**: https://devcenter.heroku.com

### Common Commands Reference

**Streamlit:**
```bash
streamlit run Home.py                    # Run app
streamlit run Home.py --server.port 8502 # Custom port
streamlit cache clear                    # Clear cache
streamlit --version                      # Check version
```

**Docker:**
```bash
docker build -t app .                    # Build image
docker run -p 8501:8501 app             # Run container
docker ps                                # List containers
docker stop <id>                         # Stop container
docker logs <id>                         # View logs
```

**Git:**
```bash
git status                               # Check status
git add .                                # Stage changes
git commit -m "message"                  # Commit
git push                                 # Push to remote
git pull                                 # Pull updates
```

---

## Success Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Dashboard runs locally
- [ ] All pages load correctly
- [ ] Analysis completes successfully
- [ ] Outputs generated correctly
- [ ] (Optional) Deployed to cloud
- [ ] (Optional) Custom domain configured
- [ ] Documentation reviewed

---

**Congratulations! Your Portfolio Analysis Dashboard is ready to use! 🎉**

For questions or issues, refer to the troubleshooting section above.

---

*Last Updated: June 2026*
*Version: 2.0*