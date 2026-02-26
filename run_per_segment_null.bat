@echo off
REM  Per-Segment Permutation Null
REM  Decomposes Layer 1 into within-segment ordering vs inter-element heterogeneity

echo.
echo  PER-SEGMENT PERMUTATION NULL
echo  Is Layer 1 consecutive-pair smoothness or inter-element heterogeneity?
echo.

python per_segment_null.py --data alphafold_cache --output results --sample 200

echo.
if errorlevel 1 (
    echo  ERROR: Test failed.
) else (
    echo  COMPLETE. Report: results\per_segment_null_report.md
)
echo.
pause
