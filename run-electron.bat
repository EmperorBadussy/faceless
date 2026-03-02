@echo off
title PHANTOM-FACE
echo.
echo  ╔═══════════════════════════════════════╗
echo  ║         PHANTOM-FACE v1.0.0           ║
echo  ║     Real-time Face Swap Engine        ║
echo  ╚═══════════════════════════════════════╝
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)

:: Check Node
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found in PATH
    pause
    exit /b 1
)

:: Install UI deps if needed
if not exist "ui\node_modules" (
    echo [SETUP] Installing UI dependencies...
    cd ui
    call npm install
    cd ..
    echo.
)

:: Launch Electron (which spawns Python internally)
echo [START] Launching PHANTOM-FACE...
cd ui
call npm run dev
