@echo off
REM ============================================================================
REM APEX AutoML - Complete System Startup Script
REM ============================================================================
REM Starts both API server and Streamlit frontend with proper environment setup
REM ============================================================================

echo.
echo ============================================================================
echo  APEX AutoML - System Startup
echo ============================================================================
echo.
echo [1/3] Activating Python Virtual Environment...
call .venv\Scripts\activate.bat

if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Make sure .venv exists. Run setup.bat first if needed.
    pause
    exit /b 1
)

echo.
echo [2/3] Starting API Server (http://localhost:8000)...
echo        Documentation: http://localhost:8000/docs
echo        Available endpoints:
echo          - GET  /                 - Health check
echo          - POST /api/ingest       - Phase 1: Data Ingestion
echo          - POST /api/detect-schema - Phase 2: Schema Detection
echo          - POST /api/preprocess   - Phase 3: Preprocessing
echo          - POST /api/select-model - Phase 4: Model Selection
echo          - POST /api/train-pipeline - Phases 5-7: Training
echo.

REM Start API server in a new terminal window
start "APEX API Server" cmd /k "python run_api.py"

REM Wait for API to start
echo Waiting for API server to initialize...
timeout /t 3 /nobreak

echo.
echo [3/3] Starting Streamlit Frontend (http://localhost:8501)...
echo        Multi-phase workflow dashboard with:
echo          - Phase 1: Data Ingestion & Caching
echo          - Phase 2: Schema Detection
echo          - Phase 3: Preprocessing
echo          - Phase 4: Model Selection with Rationale
echo          - Phase 5: GPU Training Progress
echo          - Phase 6: Monitoring & Drift Detection
echo.

REM Start Streamlit in a new terminal window
start "APEX Frontend Dashboard" cmd /k "streamlit run frontend\app_enhanced.py"

REM Keep main window open
echo.
echo ============================================================================
echo  ✅ System Started Successfully!
echo ============================================================================
echo.
echo Open your browser to: http://localhost:8501
echo API Documentation: http://localhost:8000/docs
echo.
echo To stop the system:
echo   - Close the API Server window (Ctrl+C)
echo   - Close the Frontend window (Ctrl+C)
echo.
echo ============================================================================
pause
