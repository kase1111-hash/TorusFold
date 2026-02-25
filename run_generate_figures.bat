@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  TorusFold: Publication Figure Generator
REM  Generates all manuscript figures from the proteome atlas data
REM ═══════════════════════════════════════════════════════════════════════

echo.
echo ========================================================================
echo   TorusFold: Publication Figure Generator
echo ========================================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip install numpy gemmi matplotlib --quiet --upgrade

echo.
echo [2/3] Creating output directories...
if not exist "results" mkdir results
if not exist "results\figures" mkdir results\figures

echo.
echo [3/3] Generating figures...
echo   - Fig 1: Conceptual overview (Ramachandran, W traces, decomposition)
echo   - Fig 2: Five-level null model hierarchy with error bars
echo   - Fig 3: Cross-organism conservation + fold-class boxplots
echo   - Fig 4: Within-fold-class CV validation
echo   - Bonus: W landscape heatmap
echo.
echo   Figs 1-2 use 200 proteins/organism (fast)
echo   Figs 3-4 process the full dataset (slower)
echo.

cd /d G:\bps-proteome-atlas-main
python generate_figures.py --data "alphafold_cache" --sample 200 --output "results"

echo.
if exist "results\figures\fig1_conceptual.png" (
    echo   SUCCESS: Figures saved to results\figures\
    echo.
    echo   Files:
    dir /b results\figures\*.png 2>nul
    echo.
    start "" "results\figures"
) else (
    echo   WARNING: Figures not found. Check console output for errors.
)

echo.
pause
