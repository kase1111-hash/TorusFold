@echo off
REM ══════════════════════════════════════════════════════════════
REM  TorusFold: Length-Binned Torus Seg/Real
REM  Run AFTER build_superpotential.py. ~20-40 min with 500 proteins.
REM ══════════════════════════════════════════════════════════════

cd /d G:\bps-proteome-atlas-main

echo ══════════════════════════════════════════════════════════════
echo  Length-Binned Torus Seg/Real
echo  Started: %date% %time%
echo ══════════════════════════════════════════════════════════════

python length_binned_torus.py --data "alphafold_cache" --sample 500 --n-trials 10 --output "results"

echo.
echo ══════════════════════════════════════════════════════════════
echo  Finished: %date% %time%
echo  Report: results\length_binned_torus_report.md
echo ══════════════════════════════════════════════════════════════
pause
