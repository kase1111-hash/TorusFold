@echo off
REM ══════════════════════════════════════════════════════════════
REM  TorusFold: Full Validation Pipeline
REM  Runs all analysis scripts in dependency order.
REM  
REM  Step 1: Build superpotential W (required by all others)
REM  Step 2: Within-fold-class CV
REM  Step 3: W independence
REM  Step 4: Cluster stability (longest — run overnight)
REM  Step 5: Generate publication figures
REM ══════════════════════════════════════════════════════════════

cd /d G:\bps-proteome-atlas-main

echo ══════════════════════════════════════════════════════════════
echo  TorusFold Validation Pipeline
echo  Started: %date% %time%
echo ══════════════════════════════════════════════════════════════

REM ── Step 1: Build Superpotential W ──────────────────────────
echo.
echo [1/5] Building superpotential W...
python build_superpotential.py --data "alphafold_cache" --output "results"
if %ERRORLEVEL% neq 0 (
    echo FATAL: build_superpotential.py failed. Cannot continue.
    pause
    exit /b 1
)

REM ── Step 2: Within-Fold-Class CV ────────────────────────────
echo.
echo [2/5] Within-fold-class CV analysis...
python within_foldclass_cv.py --data "alphafold_cache" --sample 0 --output "results"
if %ERRORLEVEL% neq 0 (
    echo WARNING: within_foldclass_cv.py failed. Continuing...
)

REM ── Step 3: W Independence ──────────────────────────────────
echo.
echo [3/5] W independence analysis...
python w_independence.py --data "alphafold_cache" --output "results"
if %ERRORLEVEL% neq 0 (
    echo WARNING: w_independence.py failed. Continuing...
)

REM ── Step 4: Cluster Stability ───────────────────────────────
echo.
echo [4/5] Cluster stability analysis (this one takes a while)...
python subsample_cluster_stability.py --data "alphafold_cache" --sample 0 --output "results"
if %ERRORLEVEL% neq 0 (
    echo WARNING: subsample_cluster_stability.py failed. Continuing...
)

REM ── Step 5: Generate Figures ────────────────────────────────
echo.
echo [5/5] Generating publication figures...
python generate_figures.py --data "alphafold_cache" --sample 200 --output "results"
if %ERRORLEVEL% neq 0 (
    echo WARNING: generate_figures.py failed.
)

echo.
echo ══════════════════════════════════════════════════════════════
echo  Pipeline complete: %date% %time%
echo  Check results/ for reports and results/figures/ for plots.
echo ══════════════════════════════════════════════════════════════
pause
