@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: ZEMO — LAN / Wi-Fi Server Startup Script (Windows)
:: Starts the unified Zemo Frontend + FastAPI Backend server bound to 0.0.0.0
:: Optional: pass --dual-port to also start a separate frontend server on 3000
:: ============================================================================

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "BACKEND_DIR=%ROOT_DIR%\zemo_backend"

if not exist "%BACKEND_DIR%\app\main.py" (
    echo [ERROR] Could not locate Zemo backend at "%BACKEND_DIR%\app\main.py".
    exit /b 1
)

:: 1. Locate Python executable (prefer local virtualenv if present)
set "PYTHON_CMD="
if exist "%ROOT_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD="%ROOT_DIR%\.venv\Scripts\python.exe""
) else if exist "%BACKEND_DIR%\.venv\Scripts\python.exe" (
    set "PYTHON_CMD="%BACKEND_DIR%\.venv\Scripts\python.exe""
) else (
    where py >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=py"
    ) else (
        where python >nul 2>nul
        if !errorlevel! equ 0 (
            set "PYTHON_CMD=python"
        )
    )
)

if "%PYTHON_CMD%"=="" (
    echo [ERROR] Python 3.11+ was not found on your PATH.
    echo Please install Python 3.11+ and run: pip install -r zemo_backend\requirements.txt
    exit /b 1
)

:: 2. Configure host and ports (respects environment overrides)
if "%ZEMO_HOST%"=="" set "ZEMO_HOST=0.0.0.0"
if "%ZEMO_PORT%"=="" set "ZEMO_PORT=8000"
if "%ZEMO_FRONTEND_PORT%"=="" set "ZEMO_FRONTEND_PORT=3000"

set "DUAL_PORT=0"
if /i "%~1"=="--dual-port" set "DUAL_PORT=1"

:: 3. Verify backend dependencies are installed
pushd "%BACKEND_DIR%"
%PYTHON_CMD% -c "import fastapi, uvicorn, pydantic_settings" >nul 2>nul
if !errorlevel! neq 0 (
    echo [ERROR] Required Python packages are missing.
    echo Run the following command to install dependencies:
    echo     %PYTHON_CMD% -m pip install -r "%BACKEND_DIR%\requirements.txt"
    popd
    exit /b 1
)

:: 4. Detect LAN IPv4 address dynamically
for /f "usebackq delims=" %%I in (`%PYTHON_CMD% -c "from app.core.config import get_primary_lan_ip; print(get_primary_lan_ip())"`) do (
    set "LAN_IP=%%I"
)
if "%LAN_IP%"=="" set "LAN_IP=127.0.0.1"

:: 5. Check if ZEMO_PORT is already in use to avoid duplicate server processes
%PYTHON_CMD% -c "import socket, sys; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.5); res = s.connect_ex(('127.0.0.1', int('%ZEMO_PORT%'))); s.close(); sys.exit(1 if res == 0 else 0)"
if !errorlevel! neq 0 (
    echo ============================================================================
    echo  [NOTICE] Port %ZEMO_PORT% is already in use on this computer.
    echo  Avoiding duplicate server startup.
    echo ============================================================================
    echo  Host Computer Local URL : http://localhost:%ZEMO_PORT%
    echo  LAN Client URL          : http://%LAN_IP%:%ZEMO_PORT%
    echo  Backend API URL         : http://%LAN_IP%:%ZEMO_PORT%/api/v1
    echo  API Documentation       : http://%LAN_IP%:%ZEMO_PORT%/docs
    echo ============================================================================
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%ZEMO_PORT% .*LISTENING"') do (
        echo  Active Process PID on port %ZEMO_PORT%: %%P
    )
    popd
    exit /b 0
)

echo ============================================================================
echo  ZEMO - AI DATA PRIVACY, SECURITY ^& TRAINING SAFETY PLATFORM (LAN MODE)
echo ============================================================================
echo  Host Bind Interface     : %ZEMO_HOST% (All Network Interfaces)
echo  Host LAN IPv4 Address   : %LAN_IP%
if "%DUAL_PORT%"=="1" (
    echo  Architecture Mode       : Dual-Port ^(Frontend: %ZEMO_FRONTEND_PORT%, Backend: %ZEMO_PORT%^)
    echo  Host Local Frontend URL : http://localhost:%ZEMO_FRONTEND_PORT%
    echo  LAN Client Frontend URL : http://%LAN_IP%:%ZEMO_FRONTEND_PORT%
    echo  LAN Backend API URL     : http://%LAN_IP%:%ZEMO_PORT%/api/v1
) else (
    echo  Architecture Mode       : Unified Single-Port ^(Frontend + Backend on %ZEMO_PORT%^)
    echo  Host Local URL          : http://localhost:%ZEMO_PORT%
    echo  LAN Client URL ^(Open^)   : http://%LAN_IP%:%ZEMO_PORT%
    echo  LAN Backend API URL     : http://%LAN_IP%:%ZEMO_PORT%/api/v1
)
echo  LAN Health Check        : http://%LAN_IP%:%ZEMO_PORT%/health
echo  LAN Swagger Docs        : http://%LAN_IP%:%ZEMO_PORT%/docs
echo ============================================================================
echo  NOTE: Ensure Windows Defender Firewall allows inbound TCP port %ZEMO_PORT%.
echo  Press Ctrl+C to stop the server.
echo ============================================================================
echo.

:: Optional separate frontend static server when --dual-port is requested
if "%DUAL_PORT%"=="1" (
    start "Zemo Frontend Dev Server (%ZEMO_FRONTEND_PORT%)" /D "%ROOT_DIR%" %PYTHON_CMD% -m http.server %ZEMO_FRONTEND_PORT% --bind %ZEMO_HOST%
)

:: 6. Start FastAPI / Uvicorn bound to 0.0.0.0
%PYTHON_CMD% -m uvicorn app.main:app --host %ZEMO_HOST% --port %ZEMO_PORT%
set "EXIT_CODE=!errorlevel!"
if !EXIT_CODE! neq 0 (
    echo.
    echo [ERROR] Zemo server exited with error code !EXIT_CODE!.
)
popd
exit /b !EXIT_CODE!
