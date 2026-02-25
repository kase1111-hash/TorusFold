@echo off
cd /d G:\bps-proteome-atlas-main
del results\checkpoint\checkpoint.jsonl 2>nul
echo  Running full pipeline with loops...
python alphafold_pipeline.py alphafold_cache -o results -j 4 --checkpoint results\checkpoint
pause
