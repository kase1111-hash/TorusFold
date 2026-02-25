@echo off
echo ═══════════════════════════════════════════════════════
echo   TorusFold: W Independence Analysis
echo ═══════════════════════════════════════════════════════
echo.

cd /d "%~dp0"

REM Adjust these paths to match your setup
set DATA_DIR=G:\bps-proteome-atlas-main\alphafold_cache
set PDB_DIR=G:\bps-proteome-atlas-main\data\pdb_test_set
set OUTPUT=G:\bps-proteome-atlas-main\results

REM Sample sizes (reduce for faster testing, increase for publication)
set TEST_SAMPLE=500
set W_SAMPLE=300

python w_independence.py ^
    --data "%DATA_DIR%" ^
    --pdb-dir "%PDB_DIR%" ^
    --sample %TEST_SAMPLE% ^
    --w-sample %W_SAMPLE% ^
    --output "%OUTPUT%"

echo.
echo ═══════════════════════════════════════════════════════
echo   Results: %OUTPUT%\w_independence_report.md
echo ═══════════════════════════════════════════════════════
pause
