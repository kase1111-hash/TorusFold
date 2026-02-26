@echo off
echo ======================================================================
echo  BPS DOWNLOADER — Fetch AlphaFold CIF files
echo  Run this in one terminal while processing in another.
echo ======================================================================
echo.
python bps_download.py --mode all
echo.
echo ----------------------------------------------------------------------
echo  Downloading Plaxco 23 validation set...
echo ----------------------------------------------------------------------
echo.
python bps_download.py --mode plaxco23
echo.
echo Download complete. Now run: python bps_process.py --mode all
pause
