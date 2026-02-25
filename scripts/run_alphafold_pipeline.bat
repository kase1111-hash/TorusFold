@echo off
REM ============================================================
REM TorusFold - Run Full AlphaFold Pipeline (Steps 8-10)
REM ============================================================
REM Runs the AlphaFold proteome analysis pipeline:
REM   8. Download proteome tarballs (23 organisms, ~50 GB)
REM   9. Process all CIF files (BPS/L + fold classification)
REM  10. Compile results into paper-ready tables
REM
REM Requires: Network access for download step
REM Runtime: ~2-3 hours total (download + processing)
REM ============================================================
cd /d "%~dp0.."
echo.
echo ===== TorusFold AlphaFold Pipeline =====
echo.

echo [8/10] Downloading AlphaFold proteomes...
python -m bps.bps_download
if errorlevel 1 (echo FAILED at step 8 & pause & exit /b 1)
echo.

echo [9/10] Processing AlphaFold proteomes...
python -m bps.bps_process
if errorlevel 1 (echo FAILED at step 9 & pause & exit /b 1)
echo.

echo [10/10] Compiling results...
python -m bps.compile_results
if errorlevel 1 (echo FAILED at step 10 & pause & exit /b 1)
echo.

echo ===== AlphaFold Pipeline Complete =====
echo.
echo Results:
echo   results\per_protein_bpsl.csv
echo   results\per_organism.csv
echo   output\alphafold_compilation_report.md
echo.
pause
