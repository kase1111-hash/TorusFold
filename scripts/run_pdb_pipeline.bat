@echo off
REM ============================================================
REM TorusFold - Run Full PDB Pipeline (Steps 1-7)
REM ============================================================
REM Runs all PDB-based analyses in sequence:
REM   1. Build superpotential W
REM   2. Validate dihedral extraction
REM   3. PDB experimental validation
REM   4. Loop path taxonomy
REM   5. Loop family classifier
REM   6. Forward kinematics validation
REM   7. End-to-end reconstruction test
REM
REM Requires: PDB files in data\pdb_cache\
REM Runtime: ~15-20 minutes total
REM ============================================================
cd /d "%~dp0.."
echo.
echo ===== TorusFold PDB Pipeline =====
echo.

echo [1/7] Building superpotential W...
python -m bps.superpotential
if errorlevel 1 (echo FAILED at step 1 & pause & exit /b 1)
echo.

echo [2/7] Validating dihedral extraction...
python -m bps.extract
if errorlevel 1 (echo FAILED at step 2 & pause & exit /b 1)
echo.

echo [3/7] PDB experimental validation...
python -m bps.validate_pdb
if errorlevel 1 (echo FAILED at step 3 & pause & exit /b 1)
echo.

echo [4/7] Loop path taxonomy...
python -m loops.taxonomy
if errorlevel 1 (echo FAILED at step 4 & pause & exit /b 1)
echo.

echo [5/7] Loop family classifier...
python -m loops.classifier
if errorlevel 1 (echo FAILED at step 5 & pause & exit /b 1)
echo.

echo [6/7] Forward kinematics validation...
python -m loops.forward_kinematics
if errorlevel 1 (echo FAILED at step 6 & pause & exit /b 1)
echo.

echo [7/7] End-to-end reconstruction test...
python -m loops.reconstruct_test
if errorlevel 1 (echo FAILED at step 7 & pause & exit /b 1)
echo.

echo ===== PDB Pipeline Complete =====
echo.
echo Reports generated in output\
echo.
pause
