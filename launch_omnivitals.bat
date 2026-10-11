@echo off
title OmniVitals - Contactless Heart Rate Monitoring
echo ============================================================
echo   OMNIVITALS: CONTACTLESS HEART RATE MONITORING SYSTEM
echo ============================================================
echo.
cd /d "%~dp0PulseVision"
echo Starting OmniVitals Web Server...
echo Initializing computer vision models and camera...
echo Your web browser will open automatically once the dashboard is ready!
echo.
python omnivitals_server.py
if %errorlevel% neq 0 (
    echo.
    echo An error occurred launching OmniVitals.
    pause
)
