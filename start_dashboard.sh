#!/bin/bash

echo "========================================"
echo "Portfolio Analysis Dashboard"
echo "========================================"
echo ""
echo "Starting dashboard..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo ""
fi

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    echo "This may take a few minutes..."
    pip install -r requirements.txt
    echo ""
fi

# Run Streamlit
echo ""
echo "========================================"
echo "Dashboard is starting..."
echo "Open your browser to: http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo "========================================"
echo ""

streamlit run Home.py

# Made with Bob
