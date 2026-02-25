@echo off
echo ═══════════════════════════════════════════════════════
echo   TorusFold: Within-Fold-Class Cross-Organism CV
echo ═══════════════════════════════════════════════════════
echo.

cd /d G:\bps-proteome-atlas-main

python within_foldclass_cv.py ^
    --data "alphafold_cache" ^
    --sample 0 ^
    --output "results"

echo.
echo ═══════════════════════════════════════════════════════
echo   Results: results\within_foldclass_cv_report.md
echo ═══════════════════════════════════════════════════════
pause
