@echo off
REM ============================================================
REM TorusFold Step 5 - Loop Family Classifier
REM ============================================================
REM Trains Random Forest to predict loop family from AA sequence.
REM Target: >60%% accuracy. Current result: 80.0%%.
REM Produces: output\loop_classifier_report.md
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 5] Running loop family classifier...
echo.

python -m loops.classifier

echo.
pause
