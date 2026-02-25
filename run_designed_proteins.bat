@echo off
echo ═══════════════════════════════════════════════════════
echo   TorusFold: De Novo Designed Protein Analysis
echo ═══════════════════════════════════════════════════════
echo.

cd /d G:\bps-proteome-atlas-main

python designed_proteins.py --download --analyze ^
    --data-dir "data\designed_proteins" ^
    --output "results"

echo.
echo ═══════════════════════════════════════════════════════
echo   Results: results\designed_proteins_report.md
echo ═══════════════════════════════════════════════════════
pause
