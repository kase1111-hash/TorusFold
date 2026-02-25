@echo off
REM ============================================================
REM  LOOP PATH TAXONOMY v2 — Scaled Analysis
REM  ~500 structures, recursive subclustering, length-stratified
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo  ============================================
echo   Loop Path Taxonomy v2 — Scaled Analysis
echo  ============================================
echo.

set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if not defined PYTHON (where python3 >nul 2>&1 && set PYTHON=python3)
if not defined PYTHON (where py >nul 2>&1 && set PYTHON=py)
if not defined PYTHON (
    echo  ERROR: Python not found.
    pause & exit /b 1
)

echo  Using: %PYTHON%
echo.
echo  Installing dependencies...
%PYTHON% -m pip install numpy scipy biopython matplotlib scikit-learn --quiet
echo.

set SCRIPT_DIR=%~dp0
set SCRIPT=%SCRIPT_DIR%loop_taxonomy_v2.py

if not exist "%SCRIPT%" (
    echo  ERROR: loop_taxonomy_v2.py not found.
    pause & exit /b 1
)

echo  ============================================================
echo   Starting scaled loop path taxonomy...
echo   Downloads ~500 PDB files (~250 MB).
echo   Estimated runtime: 15-45 minutes.
echo  ============================================================
echo.

%PYTHON% "%SCRIPT%" --cache-dir "%SCRIPT_DIR%pdb_cache" --output-dir "%SCRIPT_DIR%loop_taxonomy_v2_output" %*

echo.
echo  ============================================================
echo   DONE
echo   Report: loop_taxonomy_v2_output\loop_taxonomy_v2_report.md
echo   Plots:  loop_taxonomy_v2_output\
echo  ============================================================
echo.
pause
