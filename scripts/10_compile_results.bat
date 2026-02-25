@echo off
REM ============================================================
REM TorusFold Step 10 - Compile Results
REM ============================================================
REM Generates publication-ready tables from CSV results.
REM Per-organism summary, fold-class breakdown, cross-organism CV.
REM Produces: output\alphafold_compilation_report.md
REM Requires: results\per_protein_bpsl.csv (from Step 9)
REM ============================================================
cd /d "%~dp0.."
echo.
echo [Step 10] Compiling results...
echo.

python -m bps.compile_results

echo.
pause
