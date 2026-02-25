@echo off
REM ============================================================
REM TorusFold Step 4 - Loop Path Taxonomy
REM ============================================================
REM Extracts loops between alpha/beta basins, clusters with DBSCAN.
REM Length-stratified: short (<=7), medium (8-10), long (11-15).
REM Produces: output\loop_taxonomy_report.md
REM           output\plots\torus_*.png
REM Runtime: ~3-5 minutes
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 4] Running loop path taxonomy...
echo.

python -m loops.taxonomy

echo.
pause
