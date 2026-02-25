@echo off
REM ============================================================
REM  2RHE FOLD-CLASS LABEL DIAGNOSTIC
REM  Traces the "all-alpha" mislabel back to its source.
REM
REM  Usage:
REM    diagnose_2rhe.bat                   (searches current dir)
REM    diagnose_2rhe.bat C:\path\to\project (searches given root)
REM    diagnose_2rhe.bat --skip-download   (offline mode)
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo  ============================================
echo   2RHE Fold-Class Label Diagnostic
echo  ============================================
echo.

REM -- Find Python --
set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if not defined PYTHON (
    where python3 >nul 2>&1 && set PYTHON=python3
)
if not defined PYTHON (
    where py >nul 2>&1 && set PYTHON=py
)
if not defined PYTHON (
    echo  ERROR: Python not found on PATH.
    echo  Install Python 3.8+ or add it to PATH.
    pause
    exit /b 1
)

echo  Using: %PYTHON%

REM -- Locate the .py script (same dir as this .bat) --
set SCRIPT_DIR=%~dp0
set DIAG_SCRIPT=%SCRIPT_DIR%diagnose_2rhe.py

if not exist "%DIAG_SCRIPT%" (
    echo  ERROR: diagnose_2rhe.py not found next to this .bat file.
    echo  Expected: %DIAG_SCRIPT%
    pause
    exit /b 1
)

REM -- Build arguments --
set ROOT_ARG=.
set EXTRA_ARGS=

:parse_args
if "%~1"=="" goto run
if "%~1"=="--skip-download" (
    set EXTRA_ARGS=!EXTRA_ARGS! --skip-download
    shift
    goto parse_args
)
REM Treat first non-flag arg as root
if not "%~1"=="" (
    set ROOT_ARG=%~1
    shift
    goto parse_args
)

:run
echo  Project root: %ROOT_ARG%
echo  Script:       %DIAG_SCRIPT%
echo.

%PYTHON% "%DIAG_SCRIPT%" --root "%ROOT_ARG%" %EXTRA_ARGS%

echo.
echo  Done. Review output above.
echo.
pause
