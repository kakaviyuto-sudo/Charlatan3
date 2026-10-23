@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set VENV_DIR=%~dp0.venv
set VENV_PY=%VENV_DIR%\Scripts\python.exe
set APP_PATH=%~dp0app\main.py

where py >nul 2>nul
if errorlevel 1 (
    echo Python launcher 'py' was not found on PATH.
    echo Install Python 3.11 from python.org and ensure 'py' is available.
    pause
    exit /b 1
)

py -3.11 -V >nul 2>nul
if errorlevel 1 (
    echo Python 3.11 is required for this project.
    echo Install Python 3.11 and retry.
    pause
    exit /b 1
)

where cl >nul 2>nul
if errorlevel 1 (
    echo MSVC compiler not detected.
    echo Install Visual Studio 2022 Build Tools with:
    echo   - MSVC v143 C++ x64/x86 build tools
    echo   - Windows 10/11 SDK
    echo   - Desktop development with C++
    pause
    exit /b 1
)

if exist "%VENV_DIR%" (
    "%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Existing virtual environment is not using Python 3.11. Recreating...
        rmdir /s /q "%VENV_DIR%"
    )
)

if not exist "%VENV_PY%" (
    echo Creating virtual environment with Python 3.11...
    py -3.11 -m venv "%VENV_DIR%"
)

if not exist "%VENV_PY%" (
    echo Failed to create the virtual environment.
    pause
    exit /b 1
)

echo Upgrading pip, setuptools, and wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel

echo Installing base runtime dependencies...
"%VENV_PY%" -m pip install --prefer-binary numpy requests pillow websocket-client

echo Installing media and optional runtime dependencies...
"%VENV_PY%" -m pip install --prefer-binary sounddevice soundfile openai mss playwright yt-dlp

echo Installing speech stack...
"%VENV_PY%" -m pip install --prefer-binary --only-binary=:all: "av==11.0.0"
if errorlevel 1 (
    echo No prebuilt av wheel found for this environment.
    echo Falling back to source build using the installed MSVC toolchain...
    "%VENV_PY%" -m pip install --no-cache-dir "av==11.0.0"
)

"%VENV_PY%" -m pip install --prefer-binary faster-whisper

echo Verifying environment...
"%VENV_PY%" -c "import sys, numpy, av; print(sys.version); print('numpy: OK'); print('av: OK')"

echo Starting app...
"%VENV_PY%" "%APP_PATH%"
if errorlevel 1 (
    echo App failed to start.
    pause
    exit /b 1
)

pause
