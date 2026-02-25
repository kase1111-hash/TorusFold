@echo off
echo ======================================================================
echo  BPS PROCESSOR — Compute from cached CIFs
echo  Processes whatever is in alphafold_cache/ into the database.
echo  Can run while downloader is still fetching — just re-run to catch up.
echo ======================================================================
echo.
python bps_process.py --mode all --reset
echo.
echo Processing complete. Run compile_results.py to generate report.
pause
