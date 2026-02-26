@echo off
REM  TorusFold: Polymer Null Experiment

echo.
echo  POLYMER NULL EXPERIMENT
echo  Does the 11pct geometric effect arise from
echo  fold-specific structure or peptide stereochemistry?
echo.

SET DATA_DIR=alphafold_cache
SET OUTPUT_DIR=results
SET N_CHAINS=500
SET CHAIN_LENGTH=200
SET N_TRIALS=10
SET N_REAL=200

python -c "import gemmi" 2>nul
if errorlevel 1 (
    echo  WARNING: gemmi not installed. Models 3+4 will be skipped.
    echo  Install with: pip install gemmi
    echo.
)

if exist "%DATA_DIR%" (
    echo  Data directory found. Running all 4 models.
    echo.
    python polymer_null_experiment.py --data "%DATA_DIR%" --output "%OUTPUT_DIR%" --n-chains %N_CHAINS% --chain-length %CHAIN_LENGTH% --n-trials %N_TRIALS% --n-real %N_REAL%
) else (
    echo  No data directory. Running Models 1+2 only.
    echo.
    python polymer_null_experiment.py --output "%OUTPUT_DIR%" --n-chains %N_CHAINS% --chain-length %CHAIN_LENGTH% --n-trials %N_TRIALS%
)

echo.
if errorlevel 1 (
    echo  ERROR: Experiment failed.
) else (
    echo  COMPLETE. Report: %OUTPUT_DIR%\polymer_null_report.md
)
echo.
pause
