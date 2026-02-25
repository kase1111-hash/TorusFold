@echo off
REM ============================================================
REM TorusFold Step 2 - Validate Dihedral Extraction
REM ============================================================
REM Tests single-chain extraction on 1UBQ (or synthetic PDB).
REM Validates basin fractions and BPS/L computation.
REM Requires: Network access or data\pdb_cache\1UBQ.pdb
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 2] Validating dihedral extraction...
echo.

python -m bps.extract

echo.
pause
