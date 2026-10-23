@echo off
cd /d "%~dp0"

set VENV_DIR=%~dp0.venv
set VENV_PY=%VENV_DIR%\Scripts\python.exe
set APP_PATH=%~dp0main_modular_single_file.py

if not exist "%VENV_PY%" (
    echo Creating virtual environment...
    py -3 -m venv "%VENV_DIR%"
)

echo Installing required packages...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install numpy sounddevice soundfile openai faster-whisper requests pillow mss websocket-client playwright yt-dlp

echo Starting app...
"%VENV_PY%" "%APP_PATH%"
pause