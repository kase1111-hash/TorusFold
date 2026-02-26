@echo off
setlocal

REM ============================================================
REM  AUDIT FIX TEST RUN - Pre-Crunch Validation
REM  Run from TorusFold-main repo root
REM  ~2 minutes total
REM ============================================================

echo ======================================================================
echo  BPS AUDIT FIX - TEST RUN
echo  Quick validation before full crunch
echo ======================================================================
echo.

set "PY=python"
python --version >nul 2>&1
if %errorlevel% neq 0 (
    set "PY=py"
    py --version >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python not found.
        pause
        exit /b 1
    )
)

if not exist alphafold_pipeline.py (
    echo ERROR: Run this from the TorusFold-main repo root.
    pause
    exit /b 1
)

if not exist results\per_protein_bpsl.csv (
    echo ERROR: results\per_protein_bpsl.csv not found. Run pipeline first.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo  TEST 1/6: Outlier Rejection Impact
echo  THE CRITICAL ONE - does the [0.02, 0.50] filter hide problems?
echo ======================================================================
echo.
%PY% audit_tests\test1_outlier_rejection.py

echo.
echo ======================================================================
echo  TEST 2/6: Within Fold-Class CV
echo  Paper claims under 2%% CV within each fold class
echo ======================================================================
echo.
%PY% audit_tests\test2_foldclass_cv.py

echo.
echo ======================================================================
echo  TEST 3/6: pLDDT Sensitivity
echo  BPS/L at pLDDT thresholds 70/80/85/90
echo ======================================================================
echo.
%PY% audit_tests\test3_plddt_sensitivity.py

echo.
echo ======================================================================
echo  TEST 4/6: Basin Definition Consistency
echo  Scanning scripts for conflicting basin boundaries
echo ======================================================================
echo.
%PY% audit_tests\test4_basin_scan.py

echo.
echo ======================================================================
echo  TEST 5/6: Superpotential Formula Audit
echo  Confirming all scripts use shared -sqrt(P) module
echo ======================================================================
echo.
%PY% audit_tests\test5_formula_audit.py

echo.
echo ======================================================================
echo  TEST 6/6: Hardcoded Figure Data Scan
echo  Looking for hardcoded arrays near plot code
echo ======================================================================
echo.
%PY% audit_tests\test6_hardcoded_scan.py

echo.
echo ======================================================================
echo  TEST RUN COMPLETE
echo ======================================================================
echo.
echo  Results in: %CD%\audit_test_results\
echo.
echo  DECISION TREE:
echo    Test 1 excludes 0 proteins   - Filter harmless, remove anyway
echo    Test 1 CV under 5%%          - Core claim holds
echo    Test 1 CV 5-10%%             - Report both filtered/unfiltered
echo    Test 1 CV over 10%%          - Investigate excluded proteins
echo.
echo    Test 2 weighted CV under 5%% - Good for 1-organism data
echo    Test 2 note: under 2%% requires full 34-organism dataset
echo.
echo    Test 5 shows 0 own-W         - Refactor complete
echo    Test 5 shows any own-W       - Fix those files
echo.

pause
