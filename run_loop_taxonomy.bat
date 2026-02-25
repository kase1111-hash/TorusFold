@echo off
REM ============================================================
REM  LOOP PATH TAXONOMY - The Gate Experiment
REM  Do loop paths cluster into canonical families on T²?
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo  ============================================
echo   Loop Path Taxonomy on the Ramachandran Torus
echo   The experiment that gates everything.
echo  ============================================
echo.

set PYTHON=
where python >nul 2>&1 && set PYTHON=python
if not defined PYTHON (
    where python3 >nul 2>&1 && set PYTHON=python3
)
if not defined PYTHON (
    where py >nul 2>&1 && set PYTHON=py
)
if not defined PYTHON (
    echo  ERROR: Python not found.
    pause
    exit /b 1
)

echo  Using: %PYTHON%
echo.

echo  Installing dependencies...
%PYTHON% -m pip install numpy scipy biopython matplotlib scikit-learn --quiet
echo.

set SCRIPT_DIR=%~dp0
set SCRIPT=%SCRIPT_DIR%loop_taxonomy.py

if not exist "%SCRIPT%" (
    echo  ERROR: loop_taxonomy.py not found next to this .bat
    pause
    exit /b 1
)

echo  ============================================================
echo   Starting loop path taxonomy analysis...
echo   Downloads ~100 PDB files (~50 MB).
echo   Estimated runtime: 5-15 minutes.
echo  ============================================================
echo.

%PYTHON% "%SCRIPT%" --cache-dir "%SCRIPT_DIR%pdb_cache" --output-dir "%SCRIPT_DIR%loop_taxonomy_output" %*

echo.
echo  ============================================================
echo   DONE
echo   Report: loop_taxonomy_output\loop_taxonomy_report.md
echo   Plots:  loop_taxonomy_output\
echo  ============================================================
echo.
pause
