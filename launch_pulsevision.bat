@echo off
<<<<<<< HEAD
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
=======
title PulseVision - Contactless Heart Rate Monitoring
echo ============================================================
echo   PULSEVISION: CONTACTLESS HEART RATE MONITORING SYSTEM
echo ============================================================
echo.
echo Launching Streamlit Dashboard...
echo Open your browser at http://localhost:8501
echo.
cd /d "%~dp0"
python -m streamlit run app.py
if %errorlevel% neq 0 (
    echo.
    echo An error occurred launching PulseVision.
>>>>>>> 4a30745bd7822e3c11a563c5c9a937cf5dffcbb8
    pause
)
