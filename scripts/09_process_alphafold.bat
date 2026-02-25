@echo off
REM ============================================================
REM TorusFold Step 9 - Process AlphaFold Proteomes
REM ============================================================
REM Computes BPS/L for all downloaded AlphaFold structures.
REM Fast CIF parser, pLDDT >= 85 quality filter.
REM Produces: results\per_protein_bpsl.csv
REM           results\per_organism.csv
REM Runtime: ~1 hour for all 23 organisms (~52k structures)
REM
REM To process a single organism:
REM   python -m bps.bps_process --organism ecoli
REM To test on PDB cache only:
REM   python -m bps.bps_process --pdb-only
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 9] Processing AlphaFold proteomes...
echo.

python -m bps.bps_process

echo.
pause
