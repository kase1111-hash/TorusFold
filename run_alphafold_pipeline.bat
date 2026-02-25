@echo off
REM ============================================================
REM  TorusFold AlphaFold Pipeline Launcher
REM  
REM  Edit DATA_DIR below to point to your AlphaFold data,
REM  then double-click to run.
REM ============================================================

setlocal enabledelayedexpansion

REM ── EDIT THESE ──
set DATA_DIR=G:\bps-proteome-atlas-main\alphafold_cache
set WORKERS=4
set OUTPUT=results

echo.
echo  ============================================
echo   TorusFold AlphaFold Pipeline
echo  ============================================
echo.
echo  Data:    %DATA_DIR%
echo  Workers: %WORKERS%
echo  Output:  %OUTPUT%
echo.

python --version 2>nul
if %errorlevel% neq 0 (
    echo  ERROR: Python not found in PATH
    pause
    exit /b 1
)

echo  Installing/verifying dependencies...
pip install numpy scipy gemmi >nul 2>&1
echo  Done.
echo.

if not exist "%DATA_DIR%" (
    echo  ERROR: Data directory not found: %DATA_DIR%
    echo  Edit DATA_DIR in this bat file.
    pause
    exit /b 1
)

echo.
echo  Select run mode:
echo    1. Quick test - 100 structures, no loops
echo    2. BPS/L only - all structures, no loops
echo    3. Full analysis - all structures plus loops
echo    4. Resume interrupted run
echo.
set /p MODE="  Enter 1-4: "

if "%MODE%"=="1" python alphafold_pipeline.py "%DATA_DIR%" -o %OUTPUT% -j %WORKERS% --max-structures 100 --no-loops
if "%MODE%"=="2" python alphafold_pipeline.py "%DATA_DIR%" -o %OUTPUT% -j %WORKERS% --no-loops --checkpoint %OUTPUT%\checkpoint
if "%MODE%"=="3" python alphafold_pipeline.py "%DATA_DIR%" -o %OUTPUT% -j %WORKERS% --checkpoint %OUTPUT%\checkpoint
if "%MODE%"=="4" python alphafold_pipeline.py "%DATA_DIR%" -o %OUTPUT% -j %WORKERS% --checkpoint %OUTPUT%\checkpoint

echo.
if %errorlevel% neq 0 (
    echo  Pipeline failed. Common fixes:
    echo    pip install numpy scipy gemmi
    echo    pip install biopython
)
echo.
pause
