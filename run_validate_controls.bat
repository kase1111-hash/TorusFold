@echo off
setlocal enabledelayedexpansion
echo ======================================================================
echo  BPS VALIDATION CONTROLS
echo  Peer-review response tests for the BPS proteome atlas paper.
echo  Runs 5 independent validation tests and produces validation_report.md
echo ======================================================================
echo.

:: ---- Check Python ----
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo         Install Python 3.8+ from https://www.python.org/downloads/
    echo         and make sure "Add Python to PATH" is checked.
    goto :fail
)

:: Show Python version
for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found %PYVER%
echo.

:: ---- Install / upgrade dependencies ----
echo Installing required Python packages...
echo.
pip install --upgrade numpy scipy biopython matplotlib 2>nul
if errorlevel 1 (
    echo.
    echo [WARNING] pip install had issues. Trying with --user flag...
    pip install --user --upgrade numpy scipy biopython matplotlib 2>nul
)
echo.

:: ---- Verify critical imports ----
echo Checking imports...
python -c "import numpy; import scipy; print('  numpy', numpy.__version__); print('  scipy', scipy.__version__)" 2>nul
if errorlevel 1 (
    echo [ERROR] numpy or scipy not importable. Install them manually:
    echo         pip install numpy scipy
    goto :fail
)
echo  Imports OK.
echo.

:: ---- Check database exists ----
set DB_FOUND=0
if exist "alphafold_bps_results\bps_database.db" (
    set DB_FOUND=1
    set DB_PATH=alphafold_bps_results\bps_database.db
)
if exist "bps_output\bps_results.db" (
    set DB_FOUND=1
    set DB_PATH=bps_output\bps_results.db
)

if %DB_FOUND%==0 (
    echo [ERROR] BPS database not found.
    echo         Expected one of:
    echo           alphafold_bps_results\bps_database.db
    echo           bps_output\bps_results.db
    echo.
    echo         You need to run the pipeline first:
    echo           1. python bps_download.py --mode all   ^(download CIF files^)
    echo           2. python bps_process.py  --mode all   ^(compute BPS, create DB^)
    echo.
    set /p RUNDL="Run the full download+process pipeline now? (y/N): "
    if /i "!RUNDL!"=="y" (
        echo.
        echo Starting download...
        python bps_download.py --mode all
        echo.
        echo Starting processing...
        python bps_process.py --mode all
        echo.
        echo Checking for database again...
        if exist "alphafold_bps_results\bps_database.db" (
            set DB_PATH=alphafold_bps_results\bps_database.db
        ) else (
            echo [ERROR] Database still not found after pipeline run.
            goto :fail
        )
    ) else (
        goto :fail
    )
)
echo Database: %DB_PATH%

:: ---- Check CIF cache exists ----
if not exist "alphafold_cache" (
    echo [ERROR] alphafold_cache/ directory not found.
    echo         CIF files are required for Tests 1, 2, 4, and 5.
    echo         Run:  python bps_download.py --mode all
    goto :fail
)

:: Count organism directories in cache
set ORG_COUNT=0
for /d %%d in (alphafold_cache\*) do set /a ORG_COUNT+=1
echo CIF cache: %ORG_COUNT% organism directories in alphafold_cache\
if %ORG_COUNT%==0 (
    echo [WARNING] Cache is empty. Tests requiring CIF parsing will be skipped.
    echo          Run:  python bps_download.py --mode all
)
echo.

:: ---- Run validation ----
echo ======================================================================
echo  Starting validation tests...
echo  This may take 30-60 minutes depending on sample size and cache.
echo ======================================================================
echo.
python bps_validate_controls.py
if errorlevel 1 (
    echo.
    echo [ERROR] Validation script exited with an error.
    goto :fail
)

echo.
echo ======================================================================
echo  DONE
echo  Output: validation_report.md
echo ======================================================================
echo.
if exist "validation_report.md" (
    echo Report preview (first 40 lines^):
    echo ----------------------------------------------------------------------
    powershell -command "Get-Content validation_report.md -Head 40"
    echo ----------------------------------------------------------------------
)
echo.
pause
goto :eof

:fail
echo.
echo Exiting due to errors.
pause
exit /b 1
