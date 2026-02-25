@echo off
setlocal enabledelayedexpansion
echo ======================================================================
echo  PDB EXPERIMENTAL VALIDATION
echo  Validates BPS/L against real crystal structures from RCSB PDB.
echo  Downloads ~200 high-resolution X-ray structures and computes BPS/L.
echo ======================================================================
echo.

:: ---- Check Python ----
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH.
    echo         Install Python 3.8+ from https://www.python.org/downloads/
    goto :fail
)

for /f "tokens=*" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found %PYVER%
echo.

:: ---- Install / upgrade dependencies ----
echo Installing required Python packages...
pip install --upgrade numpy scipy biopython 2>nul
if errorlevel 1 (
    pip install --user --upgrade numpy scipy biopython 2>nul
)
echo.

:: ---- Check imports ----
echo Checking imports...
python -c "import numpy; import scipy; from Bio.PDB import MMCIFParser; print('  All imports OK')" 2>nul
if errorlevel 1 (
    echo [WARNING] Some imports failed. Trying without BioPython...
    python -c "import numpy; import scipy; print('  numpy + scipy OK')" 2>nul
    if errorlevel 1 (
        echo [ERROR] numpy or scipy not importable.
        goto :fail
    )
    echo   BioPython not available - will use manual CIF parser.
)
echo.

:: ---- Check AlphaFold cache (needed for phi sign determination) ----
if not exist "alphafold_cache" (
    echo [ERROR] alphafold_cache/ not found.
    echo         The phi sign calibration requires at least one AlphaFold CIF.
    echo         Run:  python bps_download.py --mode all
    goto :fail
)
echo AlphaFold cache found (needed for phi sign calibration).
echo.

:: ---- Run validation ----
echo ======================================================================
echo  Starting PDB validation...
echo  This downloads ~200 CIF files from RCSB PDB (~100-200 MB).
echo  Estimated runtime: 10-30 minutes depending on network speed.
echo ======================================================================
echo.
python bps_pdb_validate.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Validation script exited with an error.
    goto :fail
)

echo.
echo ======================================================================
echo  DONE
echo  Output: pdb_validation_report.md
echo ======================================================================
echo.
if exist "pdb_validation_report.md" (
    echo Report preview (first 40 lines^):
    echo ----------------------------------------------------------------------
    powershell -command "Get-Content pdb_validation_report.md -Head 40"
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
