@echo off
setlocal enabledelayedexpansion

:: Navigate to the directory where this script is located
cd /d "%~dp0"

echo ============================================================
echo   REMOTE-SENSING VISION AI - Gradio Web Launcher
echo ============================================================
echo.

:: Check if the virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    echo.
    echo Please create the virtual environment and install dependencies first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment (.venv)...
call .venv\Scripts\activate.bat

echo [INFO] Starting Gradio web interface (app.py)...
echo [INFO] Note: First-time model loading takes ~90-100s into GPU VRAM.
echo.

python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application terminated with exit code %ERRORLEVEL%.
    pause
    exit /b %ERRORLEVEL%
)

endlocal
