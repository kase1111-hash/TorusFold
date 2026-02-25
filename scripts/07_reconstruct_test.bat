@echo off
REM ============================================================
REM TorusFold Step 7 - End-to-End Reconstruction Test
REM ============================================================
REM Tests canonical loop path replacement: how much RMSD do you
REM lose by using family centroids instead of actual conformations?
REM Produces: output\reconstruction_test_report.md
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 7] Running end-to-end reconstruction test...
echo.

python -m loops.reconstruct_test

echo.
pause
