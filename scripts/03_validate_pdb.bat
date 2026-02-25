@echo off
REM ============================================================
REM TorusFold Step 3 - PDB Experimental Validation
REM ============================================================
REM Computes BPS/L across hundreds of PDB structures.
REM Reports CV, fold-class breakdown, three-level decomposition.
REM Produces: output\pdb_validation_report.md
REM Runtime: ~2-5 minutes
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 3] Running PDB experimental validation...
echo.

python -m bps.validate_pdb

echo.
pause
