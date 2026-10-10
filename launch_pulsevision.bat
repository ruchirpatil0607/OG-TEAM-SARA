@echo off
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
    pause
)
