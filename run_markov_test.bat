@echo off
echo ======================================================================
echo  MARKOV PSEUDO-PROTEOME TEST
echo  The critical control: is BPS/L conservation biology or statistics?
echo  100 pseudo-proteomes x 34 organisms = 3,400 synthetic proteomes
echo  Estimated time: 1-3 hours
echo ======================================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found.
    pause
    exit /b 1
)

if not exist alphafold_bps_results\bps_database.db (
    echo ERROR: Database not found. Run bps_process.py --mode all first.
    pause
    exit /b 1
)

echo Starting full test at %date% %time%
echo.

python markov_pseudoproteome_test.py --n-pseudo 100

echo.
echo ======================================================================
echo  TEST COMPLETE
echo ======================================================================
echo.
echo  Results in: markov_test_results\
echo    markov_pseudoproteome_results.csv  - per-organism data
echo    markov_test_summary.txt            - human-readable summary
echo    markov_test.log                    - full log
echo.
echo  KEY QUESTION ANSWERED:
echo    If gap is ~40-80%% across all organisms:
echo      -> Biology. Evolution minimizes backbone topological energy.
echo    If gap is ~0%% :
echo      -> Statistics. BPS/L is just a property of the torus.
echo.
echo  Finished at %date% %time%
echo.
pause
