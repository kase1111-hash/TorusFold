@echo off
cd /d G:\bps-proteome-atlas-main
pip install scikit-learn pandas >nul 2>&1
echo  Running loop clustering (10k sample)...
echo.
python cluster_loops.py --sample 10000
echo.
pause
