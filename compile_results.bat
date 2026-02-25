@echo off
echo Compiling BPS results...
python compile_results.py --db alphafold_bps_results\bps_database.db --out bps_results_for_claude.txt
echo.
echo Output: bps_results_for_claude.txt
echo Copy the contents and paste to Claude.
pause
