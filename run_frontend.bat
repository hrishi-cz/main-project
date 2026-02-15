@echo off
REM APEX Streamlit Frontend Startup Script

echo.
echo ============================================================
echo APEX AutoVision Framework - Streamlit Frontend
echo ============================================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Streamlit not found.
    echo Please install dependencies: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Check if API server is running
echo Checking if API server is running on http://localhost:8000...
python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)" >nul 2>&1

if errorlevel 1 (
    echo [WARNING] API Server not running on localhost:8000
    echo You may want to start it in another terminal with: python run_api.py
    echo.
)

echo Starting Streamlit Frontend...
echo API will connect to http://localhost:8000
echo.
echo Streamlit app should open in your browser automatically
echo If not, navigate to: http://localhost:8501
echo.

streamlit run frontend/app_enhanced.py --logger.level=warning
