@echo off
echo ========================================
echo Portfolio Analysis Dashboard
echo ========================================
echo.
echo Starting dashboard...
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
call venv\Scripts\activate

REM Check if requirements are installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    echo This may take a few minutes...
    pip install -r requirements.txt
    echo.
)

REM Run Streamlit
echo.
echo ========================================
echo Dashboard is starting...
echo Open your browser to: http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo ========================================
echo.

streamlit run Home.py

pause

@REM Made with Bob
