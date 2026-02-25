@echo off
echo ======================================================================
echo  BPS PROCESSOR — Catch-up run
echo  Processes any new CIFs that appeared since last run.
echo  Does NOT reset the database.
echo ======================================================================
echo.
python bps_process.py --mode all
echo.
python bps_process.py --mode summary
pause
