@echo off
REM ══════════════════════════════════════════════════════════════
REM  TorusFold: Torus-Distance Control Test (Tier 0 Falsification)
REM  Run AFTER build_superpotential.py, BEFORE submission.
REM  ~15-30 min with 200 proteins, 10 trials.
REM ══════════════════════════════════════════════════════════════

cd /d G:\bps-proteome-atlas-main

echo ══════════════════════════════════════════════════════════════
echo  Torus-Distance Control Test
echo  Started: %date% %time%
echo ══════════════════════════════════════════════════════════════

python torus_distance_control.py --data "alphafold_cache" --sample 200 --n-trials 10 --output "results"

echo.
echo ══════════════════════════════════════════════════════════════
echo  Finished: %date% %time%
echo  Report: results\torus_distance_control_report.md
echo ══════════════════════════════════════════════════════════════
pause
