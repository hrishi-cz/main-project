@echo off
REM APEX Project Startup Script

echo.
echo ============================================================
echo APEX AutoVision Framework - Startup
echo ============================================================
echo.

REM Color codes for output
setlocal enabledelayedexpansion

REM Start API Server
echo Starting API Server on http://localhost:8000
echo API Documentation available at http://localhost:8000/docs
echo.
echo Press CTRL+C to stop the server
echo.

python api/main_enhanced.py
