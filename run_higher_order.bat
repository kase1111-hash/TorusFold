@echo off
cd /d G:\bps-proteome-atlas-main
echo  Running higher-order null model analysis (200 structures)...
echo.
python higher_order_null.py --sample 200 --trials 10
echo.
pause
