@echo off
REM ============================================================
REM TorusFold Step 8 - Download AlphaFold Proteomes
REM ============================================================
REM Downloads proteome tarballs from AlphaFold EBI database.
REM 23 organisms covering all kingdoms of life.
REM Disk space: ~50 GB total (~2 GB bacteria, ~20 GB human)
REM
REM To download a single organism instead:
REM   python -m bps.bps_download --organism ecoli
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 8] Downloading AlphaFold proteomes...
echo WARNING: This will download ~50 GB of data.
echo.

python -m bps.bps_download

echo.
pause
