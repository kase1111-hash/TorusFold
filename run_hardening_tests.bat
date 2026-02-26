@echo off
REM  TorusFold: Pre-Submission Hardening Tests
REM  Three quick tests to close remaining attack vectors

echo.
echo  PRE-SUBMISSION HARDENING TESTS
echo  A: Local-window null (positional gradient defense)
echo  B: PDB torus Seg/Real (AlphaFold compression defense)
echo  C: Steric rejection diagnostics (polymer null defense)
echo.

SET DATA_DIR=alphafold_cache
SET PDB_DIR=pdb_cache
SET OUTPUT_DIR=results

if exist "%PDB_DIR%" (
    echo  PDB directory found. Running all 3 tests.
    python hardening_tests.py --data "%DATA_DIR%" --pdb-dir "%PDB_DIR%" --output "%OUTPUT_DIR%" --sample 200
) else if exist "%DATA_DIR%" (
    echo  No PDB directory. Running Tests A and C.
    python hardening_tests.py --data "%DATA_DIR%" --output "%OUTPUT_DIR%" --sample 200
) else (
    echo  No data directories. Running Test C only.
    python hardening_tests.py --output "%OUTPUT_DIR%"
)

echo.
if errorlevel 1 (
    echo  ERROR: Tests failed.
) else (
    echo  COMPLETE. Report: %OUTPUT_DIR%\hardening_tests_report.md
)
echo.
pause
