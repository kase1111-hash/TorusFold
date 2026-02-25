@echo off
REM ============================================================
REM TorusFold Step 1 - Build and Validate Superpotential W
REM ============================================================
REM Builds W(phi,psi) from empirical KDE on cached PDB structures.
REM Validates landscape depths and basin assignments.
REM Requires: PDB files in data\pdb_cache\
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 1] Building superpotential W(phi, psi)...
echo.

python -m bps.superpotential

echo.
pause
