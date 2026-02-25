@echo off
REM ============================================================
REM TorusFold Step 6 - Forward Kinematics Validation
REM ============================================================
REM Validates NeRF backbone reconstruction algorithm.
REM Reconstructs backbone from experimental dihedrals, computes
REM RMSD against crystal structure. Target: RMSD < 0.5 A.
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 6] Validating forward kinematics...
echo.

python -m loops.forward_kinematics

echo.
pause
