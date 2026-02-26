@echo off
echo === per_protein_bpsl.csv ===
echo --- Headers ---
for /f "delims=" %%a in ('type results\per_protein_bpsl.csv ^| findstr /n "^" ^| findstr "^1:"') do echo %%a
echo --- First 5 rows ---
for /f "skip=1 delims=" %%a in ('type results\per_protein_bpsl.csv ^| findstr /n "^" ^| findstr "^[2-6]:"') do echo %%a
echo.
echo === per_organism_bpsl.csv ===
type results\per_organism_bpsl.csv
pause
