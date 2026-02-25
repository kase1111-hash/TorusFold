@echo off
REM ═══════════════════════════════════════════════════════════════════════
REM  TorusFold: 5x Subsample Cluster Stability Test
REM  Tests whether loop taxonomy families are stable across random subsamples
REM ═══════════════════════════════════════════════════════════════════════

echo.
echo ========================================================================
echo   TorusFold: 5x Subsample Cluster Stability Test
echo ========================================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
pip install numpy gemmi hdbscan scikit-learn --quiet --upgrade

echo.
echo [2/3] Creating output directory...
if not exist "results" mkdir results

echo.
echo [3/3] Running cluster stability analysis...
echo   - Extracts loop segments from all proteins
echo   - Builds 8-dim feature vector per loop
echo   - Clusters full dataset with HDBSCAN (reference)
echo   - Runs 5 independent 80%% subsamples
echo   - Computes pairwise ARI and Jaccard similarity
echo   - This will take several hours on the full dataset
echo.

cd /d G:\bps-proteome-atlas-main
python subsample_cluster_stability.py --data "alphafold_cache" --sample 0 --output "results"

echo.
if exist "results\subsample_cluster_stability_report.md" (
    echo   SUCCESS: Report written to results\subsample_cluster_stability_report.md
    start "" "results\subsample_cluster_stability_report.md"
) else (
    echo   WARNING: Report not found. Check console output for errors.
)

echo.
pause
