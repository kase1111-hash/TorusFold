# TorusFold Usage Guide

Complete reference for all 23 Python scripts and their Windows batch files.

---

## Prerequisites

**Python 3.10+** required. Install dependencies:

```
pip install numpy scipy biopython matplotlib scikit-learn gemmi
```

Optional (for cluster stability test):
```
pip install hdbscan pandas
```

**All root-level scripts run directly:** `python script_name.py [args]`
**Package scripts use module syntax:** `python -m bps.scriptname`

---

## Pipeline Overview

The scripts fall into six groups:

| Group | Scripts | Purpose |
|-------|---------|---------|
| **Foundation** | build_superpotential, bps_download, bps_process | Build W, download data, compute BPS/L |
| **Validation** | bps_validate_controls, bps_pdb_validate, bps_statistics, bps_plaxco | 6 controls, PDB comparison, statistics, folding rates |
| **Analysis** | compile_results, bps_figures, generate_figures | Reports and publication figures |
| **Loop taxonomy** | loop_taxonomy, loop_taxonomy_v2, cluster_loops, alphafold_pipeline | Loop extraction, clustering, classification |
| **Advanced** | higher_order_null, within_foldclass_cv, w_independence, torus_distance_control, subsample_cluster_stability, designed_proteins, designed_seg_real | Null models, robustness, designed proteins |
| **Diagnostics** | diagnose_2rhe, diagnose_foldclass | Bug tracing |

### Execution Order

```
1. build_superpotential.py     Build W landscape
2. bps_download.py             Download AlphaFold proteomes
3. bps_process.py              Compute BPS/L for all organisms
4. compile_results.py          Generate summary report
5. bps_validate_controls.py    Run 6 validation controls
6. bps_pdb_validate.py         PDB experimental validation
7. bps_statistics.py           Advanced statistical analyses
8. bps_plaxco.py               Folding rate correlation
9. bps_figures.py              Publication figures (database)
10. generate_figures.py         Publication figures (raw CIF)
11. loop_taxonomy.py            Loop clustering v1 (~119 structures)
12. loop_taxonomy_v2.py         Loop clustering v2 (~500 structures)
13. cluster_loops.py            DBSCAN on AlphaFold-scale loops
14. alphafold_pipeline.py       Full AlphaFold pipeline (BPS + loops)
15. higher_order_null.py        4-level null model hierarchy
16. within_foldclass_cv.py      Within-fold-class cross-organism CV
17. w_independence.py           W train/test split robustness
18. torus_distance_control.py   Tier-0 falsification (W artifact?)
19. subsample_cluster_stability.py  5x subsample cluster stability
20. designed_proteins.py        De novo designed protein analysis
21. designed_seg_real.py        Designed protein decomposition
22. diagnose_2rhe.py            2RHE mislabel diagnostic
23. diagnose_foldclass.py       Fold classification bug diagnostic
```

---

## Script Reference

---

### 1. build_superpotential.py

**Purpose:** Constructs the superpotential W = -ln P(phi,psi) from all AlphaFold
structures. This is the foundational data product that all other scripts depend on.

**Run:** `python build_superpotential.py [args]`
**Batch file:** Part of `run_full_pipeline.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `alphafold_cache` | AlphaFold data directory |
| `--output` | `results` | Output directory |
| `--grid-size` | `360` | Histogram resolution (360 = 1 deg/bin) |
| `--sample` | `0` | Max proteins per organism (0 = all) |
| `--plddt-min` | `70.0` | Minimum pLDDT for residue inclusion |
| `--epsilon` | `1e-7` | Floor value before log transform |

**Requires:** `alphafold_cache/*/AF-*.cif`
**Produces:** `results/superpotential_W.npz`
**Dependencies:** `gemmi`, numpy

---

### 2. bps_download.py

**Purpose:** Downloads proteome CIF files from AlphaFold. Network-bound only —
no computation, no database. Run this first (or in background) while processing
already-cached data.

**Run:** `python bps_download.py --mode all`
**Batch file:** `run_download.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `status` | `all`, `organism`, `status`, `retry`, `watch` |
| `--organism` | `ecoli` | Single organism to download |
| `--max` | None | Max proteins per organism |
| `--poll` | `60` | Poll interval for watch mode (seconds) |

**Requires:** Network access to UniProt + AlphaFold EBI
**Produces:** `alphafold_cache/<organism>/AF-*.cif`, `alphafold_cache/<organism>/_manifest.json`

---

### 3. bps_process.py

**Purpose:** Compute BPS/L from cached AlphaFold CIF files. Writes everything to
SQLite database. Determines phi sign convention automatically (hard fails if
it cannot verify).

**Run:** `python bps_process.py --mode all`
**Batch file:** `run_process.bat` (full reset) or `run_process_catchup.bat` (incremental)

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `status` | `all`, `organism`, `status`, `summary`, `export`, `validate`, `phi-check`, `watch` |
| `--organism` | `ecoli` | Organism to process |
| `--reset` | off | Delete database and recompute |
| `--poll` | `30` | Poll interval for watch mode |

**Requires:** `alphafold_cache/*/AF-*.cif`
**Produces:** `alphafold_bps_results/bps_database.db`, `alphafold_bps_results/pipeline.log`
**Dependencies:** numpy, scipy

---

### 4. compile_results.py

**Purpose:** Reads bps_database.db and produces a single text report optimized for
analysis. Includes pLDDT-controlled comparisons to separate real biology from
AlphaFold model quality artifacts.

**Run:** `python compile_results.py --db alphafold_bps_results/bps_database.db`
**Batch file:** `compile_results.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./alphafold_bps_results/bps_database.db` | Database path |
| `--out` | `./bps_results_for_claude.txt` | Output file |
| `--mode` | one-shot | `watch` for continuous updates |
| `--poll` | `60` | Poll interval for watch mode |

**Requires:** `alphafold_bps_results/bps_database.db`
**Produces:** `bps_results_for_claude.txt`

---

### 5. bps_validate_controls.py

**Purpose:** Six independent validation tests to answer peer review concerns.
Tests: shuffled control, W-construction robustness, fold-class conditioning,
Gly/Pro composition, basin transition matrix prediction, Markov null model.

**Run:** `python bps_validate_controls.py`
**Batch file:** `run_validate_controls.bat`

No command-line arguments. Reads database and CIF cache automatically.

**Requires:** `bps_database.db`, `alphafold_cache/*/AF-*.cif`
**Produces:** `validation_report.md`
**Runtime:** 30-60 minutes

---

### 6. bps_pdb_validate.py

**Purpose:** Downloads high-resolution X-ray crystal structures from RCSB PDB,
computes BPS/L, and compares against AlphaFold results. Addresses the reviewer
concern: "Is BPS/L a property of real proteins, or AlphaFold's learned prior?"

**Run:** `python bps_pdb_validate.py`
**Batch file:** `run_pdb_validate.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--max` | None | Max chains to process |
| `--skip-download` | off | Use cached CIFs only |

**Requires:** Network access to RCSB/PISCES, `bps_process.determine_phi_sign()`
**Produces:** `pdb_validation_report.md`, `pdb_validation_cache/*.cif`
**Dependencies:** numpy, scipy, BioPython

---

### 7. bps_statistics.py

**Purpose:** Advanced statistical analyses: partial correlations, multiple regression,
PCA, nested ANOVA, bootstrap CIs, pLDDT-matched subsampling, pathogen vs host
comparison, fold-class analysis, size convergence.

**Run:** `python bps_statistics.py --analysis all`

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./alphafold_bps_results/bps_database.db` | Database path |
| `--out` | `./alphafold_bps_results/bps_statistics.txt` | Output path |
| `--plddt-threshold` | `0` | Min pLDDT (0=all, 85=high, 90=very high) |
| `--analysis` | `all` | `partial`, `regression`, `pca`, `anova`, `bootstrap`, `plddt-match`, `pathogen`, `fold-class`, `size-convergence` |

**Requires:** `alphafold_bps_results/bps_database.db`
**Produces:** `alphafold_bps_results/bps_statistics.txt`

---

### 8. bps_plaxco.py

**Purpose:** Standalone folding rate analysis. Downloads 23 Plaxco proteins, computes
BPS + contact order, correlates with experimental ln(kf), performs leave-one-out
cross-validation.

**Run:** `python bps_plaxco.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./alphafold_bps_results/bps_database.db` | Database path |
| `--out` | `./alphafold_bps_results/bps_plaxco_validation.txt` | Report path |
| `--download` | off | Download Plaxco proteins and exit |
| `--recompute` | off | Recompute BPS even if DB has values |

**Requires:** Network for download, `bps_process` module
**Produces:** `alphafold_bps_results/bps_plaxco_validation.txt`

---

### 9. bps_figures.py

**Purpose:** Publication-quality figures from the SQLite database: BPS distributions,
organism barplots, BPS vs pLDDT, BPS vs length, correlation matrix, fold-class
heatmap, Plaxco scatter.

**Run:** `python bps_figures.py --figure all`

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `./alphafold_bps_results/bps_database.db` | Database path |
| `--plddt-threshold` | `0` | pLDDT filter |
| `--format` | `png` | `png`, `pdf`, `svg` |
| `--figure` | `all` | `distribution`, `barplot`, `bps-plddt`, `bps-length`, `correlation`, `fold-class`, `plaxco` |

**Requires:** `alphafold_bps_results/bps_database.db`
**Produces:** `alphafold_bps_results/figures/*.{png,pdf,svg}` (7 figures)

---

### 10. generate_figures.py

**Purpose:** Publication figures directly from raw AlphaFold CIF data (not database).
Fig 1: conceptual overview, Fig 2: 5-level null hierarchy, Fig 3: cross-organism
conservation, Fig 4: within-fold-class CV, Bonus: W landscape heatmap.

**Run:** `python generate_figures.py --figures all`
**Batch file:** `run_generate_figures.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `alphafold_cache` | AlphaFold data directory |
| `--sample` | `200` | Max proteins per organism |
| `--output` | `results` | Output directory |
| `--w-path` | auto-detected | Path to `superpotential_W.npz` |
| `--figures` | all | Which figures: `1`, `2`, `3`, `4`, `W`, or `all` |

**Requires:** `alphafold_cache/`, `results/superpotential_W.npz`
**Produces:** `results/figures/fig{1,2,3,4}_*.{png,pdf}`, `results/figures/fig_ramachandran_W.*`
**Dependencies:** gemmi, numpy, matplotlib

---

### 11. loop_taxonomy.py

**Purpose:** The "gate experiment" — do loop paths between secondary structure basins
cluster into canonical families on T^2? Uses ~119 curated high-resolution PDB chains.
DBSCAN clustering with torus-aware distance.

**Run:** `python loop_taxonomy.py`
**Batch file:** `run_loop_taxonomy.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-dir` | `./pdb_cache` | PDB cache directory |
| `--output-dir` | `./loop_taxonomy_output` | Output directory |
| `--skip-download` | off | Use cached files only |
| `--max-structs` | None | Limit structures for testing |
| `--no-plots` | off | Skip generating plots |

**Requires:** Network for PDB downloads (or cached PDB files)
**Produces:** `loop_taxonomy_output/loop_taxonomy_report.md`, `loop_taxonomy_output/*.png`
**Runtime:** 5-15 minutes
**Dependencies:** numpy, scipy, BioPython, scikit-learn, matplotlib

---

### 12. loop_taxonomy_v2.py

**Purpose:** Scaled version of loop taxonomy. Fetches 500-800 chains via RCSB Search
API. Recursive DBSCAN on high-variance clusters. Length-stratified analysis
(short/medium/long). Tight vs catch-all family classification.

**Run:** `python loop_taxonomy_v2.py`
**Batch file:** `run_loop_taxonomy_v2.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--max-structs` | None | Limit structures |
| `--skip-download` | off | Use cached only |
| `--cache-dir` | `./pdb_cache` | PDB cache |
| `--output-dir` | `./loop_taxonomy_v2_output` | Output |
| `--no-plots` | off | Skip plots |

**Requires:** Network or PDB cache
**Produces:** `loop_taxonomy_v2_output/loop_taxonomy_v2_report.md`, plots, CSV, JSON
**Runtime:** 15-45 minutes

---

### 13. cluster_loops.py

**Purpose:** DBSCAN clustering on short loops from the AlphaFold pipeline output.
Stratifies by length, identifies tight families, reports coverage. Uses
RandomForest classifier to test sequence-to-family prediction.

**Run:** `python cluster_loops.py --sample 10000`
**Batch file:** `run_clustering.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--sample` | `10000` | Sample size for short loops |
| `--full` | off | Use all short loops (slow) |
| `--input` | `results/loops.csv` | Input loops CSV |
| `--output` | `results` | Output directory |

**Requires:** `results/loops.csv` (from alphafold_pipeline.py)
**Produces:** `results/loop_families.csv`, `results/loop_taxonomy_report.md`
**Dependencies:** pandas, scikit-learn

---

### 14. alphafold_pipeline.py

**Purpose:** Full end-to-end AlphaFold pipeline: per-protein BPS/L with realistic
superpotential, fold-class breakdown, cross-organism conservation, and loop path
taxonomy at scale. Multiprocessing support.

**Run:** `python alphafold_pipeline.py /path/to/alphafold_data`
**Batch file:** `run_alphafold_pipeline.bat` (interactive menu) or `run_with_loops.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `DATA_DIR` (positional) | required | AlphaFold data directory |
| `--output` / `-o` | `results/` | Output directory |
| `--sample` | None | Max proteins per organism |
| `--min-plddt` | `85.0` | pLDDT quality threshold |
| `--max-workers` / `-j` | `4` | Multiprocessing workers |
| `--no-plots` | off | Skip figure generation |

**Requires:** `alphafold_data/organism_*/AF-*.cif`
**Produces:** `results/proteins.csv`, `results/loops.csv`, `results/organisms.csv`,
`results/superpotential_W.npz`, `results/report.md`
**Dependencies:** gemmi (preferred) or BioPython fallback, numpy, scipy, multiprocessing

---

### 15. higher_order_null.py

**Purpose:** Tests whether the Markov/Real gap reflects intra-basin conformational
coherence or segment-length structure. Implements FOUR levels:
Real < Segment < Markov (1st) < Shuffled.

**Run:** `python higher_order_null.py --sample 200`
**Batch file:** `run_higher_order.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `alphafold_cache` | AlphaFold data directory |
| `--w-cache` | `results/superpotential_W.npz` | Cached W path |
| `--sample` | `200` | Structures to sample |
| `--trials` | `10` | Null model trials per structure |
| `--output` | `results` | Output directory |

**Requires:** `alphafold_cache/`, `results/superpotential_W.npz`
**Produces:** `results/higher_order_null_report.md`
**Dependencies:** gemmi, numpy, scipy

---

### 16. within_foldclass_cv.py

**Purpose:** Tests whether cross-organism BPS/L conservation is genuine or just
compositional (averaging different fold compositions). For each fold class:
computes per-organism mean BPS/L, cross-organism CV, compares to overall CV.

**Run:** `python within_foldclass_cv.py`
**Batch file:** `run_within_foldclass_cv.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `alphafold_cache` | AlphaFold data directory |
| `--sample` | `0` | Max per organism (0 = all) |
| `--output` | `results` | Output directory |
| `--w-path` | None | Explicit W path |

**Requires:** `alphafold_cache/`, `results/superpotential_W.npz`
**Produces:** `results/within_foldclass_cv_report.md`

---

### 17. w_independence.py

**Purpose:** Tests robustness of BPS/L to W construction.
Test 1: W from 10% of AlphaFold, applied to 90%.
Test 2: W from organism A, applied to organism B.
Test 3: W from PDB structures, applied to AlphaFold.
Test 4: W at different bin resolutions (180, 360, 720).

**Run:** `python w_independence.py --sample 500`
**Batch file:** `run_w_independence.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `alphafold_cache` | AlphaFold data directory |
| `--pdb-dir` | `data/pdb_test_set` | PDB structures (Test 3) |
| `--sample` | `500` | Test set size |
| `--w-sample` | `300` | Structures for W building |
| `--output` | `results` | Output directory |
| `--w-path` | None | Explicit W path |

**Requires:** `alphafold_cache/`, optionally `data/pdb_test_set/` or `data/pdb_cache/`
**Produces:** `results/w_independence_report.md`

---

### 18. torus_distance_control.py

**Purpose:** Tier-0 falsification test. Tests whether Seg/Real suppression is real
protein geometry or a W-curvature artifact. Compares: Torus L1, Torus L2,
flat-basin W, original W. Also runs epsilon sensitivity sweep.

**Run:** `python torus_distance_control.py --sample 200`
**Batch file:** `run_torus_distance_control.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `alphafold_cache` | AlphaFold data directory |
| `--output` | `results` | Output directory |
| `--w-path` | None | Explicit W path |
| `--sample` | `200` | Proteins to evaluate |
| `--n-trials` | `10` | Null model trials per protein |

**Requires:** `alphafold_cache/`, `results/superpotential_W.npz`
**Produces:** `results/torus_distance_control_report.md`
**Runtime:** 15-30 minutes

---

### 19. subsample_cluster_stability.py

**Purpose:** Tests whether loop taxonomy families are stable across random subsamples.
5x 80% subsamples with HDBSCAN clustering, pairwise Adjusted Rand Index.

**Run:** `python subsample_cluster_stability.py --data alphafold_cache`
**Batch file:** `run_cluster_stability.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `alphafold_cache` | AlphaFold data directory |
| `--sample` | `0` | Max per organism (0 = all) |
| `--n-subsamples` | `5` | Number of subsamples |
| `--subsample-frac` | `0.8` | Fraction per subsample |
| `--min-cluster-size` | `50` | HDBSCAN min_cluster_size |
| `--min-samples` | `10` | HDBSCAN min_samples |
| `--output` | `results` | Output directory |
| `--w-path` | None | Explicit W path |

**Requires:** `alphafold_cache/`, `results/superpotential_W.npz`
**Produces:** `results/subsample_cluster_stability_report.md`
**Dependencies:** gemmi, hdbscan, scikit-learn

---

### 20. designed_proteins.py

**Purpose:** Downloads and analyzes computationally designed (de novo) proteins from
RCSB PDB. Tests the hypothesis: designed proteins show the same intra-basin
coherence as evolved proteins (physics, not biology).

**Run:** `python designed_proteins.py --download --analyze`
**Batch file:** `run_designed_proteins.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--download` | off | Fetch CIF files from RCSB |
| `--analyze` | off | Compute BPS/L on downloaded files |
| `--data-dir` | `data/designed_proteins` | CIF file directory |
| `--output` | `results` | Output directory |

**Requires:** Network for download, `results/superpotential_W.npz`
**Produces:** `results/designed_proteins_report.md`
**Dependencies:** gemmi

---

### 21. designed_seg_real.py

**Purpose:** Computes full decomposition (Real, Segment, M1, Shuffled) for de novo
designed proteins. Tests whether evolution optimizes intra-basin coherence
beyond what computational design achieves.

**Run:** `python designed_seg_real.py`
**Batch file:** `run_designed_seg_real.bat`

No command-line arguments (hardcoded paths).

**Requires:** `data/designed_proteins/*.cif`, `results/superpotential_W.npz`
**Produces:** `results/designed_seg_real_report.md`
**Dependencies:** gemmi

---

### 22. diagnose_2rhe.py

**Purpose:** Traces the "all-alpha" misclassification of 2RHE (an all-beta
immunoglobulin VL domain) back to its source. Checks SCOP files, PISCES lists,
hardcoded dicts, CSV/JSON data files, downloads 2RHE and computes SS from dihedrals.

**Run:** `python diagnose_2rhe.py --root .`
**Batch file:** `diagnose_2rhe.bat`

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | `.` | Project root to search |
| `--skip-download` | off | Skip PDB download |

**Requires:** Network for 2RHE download (or `--skip-download`)
**Produces:** Diagnostic report to stdout (exit code 0 or 1)

---

### 23. diagnose_foldclass.py

**Purpose:** Three-part diagnostic for the fold classification bug.
Part 1: CSV summary of per-protein BPS/L fractions.
Part 2: Raw dihedral angles from first CIF file.
Part 3: Gemmi built-in phi/psi as ground truth comparison.

**Run:** `python diagnose_foldclass.py`
**Batch file:** `diagnose_foldclass.bat`

No command-line arguments.

**Requires:** `results/per_protein_bpsl.csv`, a `.cif` file in `alphafold_cache/` or `data/`
**Produces:** Diagnostic output to stdout
**Dependencies:** gemmi

---

## Windows Batch Files

22 batch files in the project root. Double-click or run from Command Prompt.

### Setup and Download

| Batch file | Runs | Description |
|------------|------|-------------|
| `run_download.bat` | `bps_download.py --mode all` | Download all AlphaFold proteomes |

### Core Pipeline

| Batch file | Runs | Description |
|------------|------|-------------|
| `run_process.bat` | `bps_process.py --mode all --reset` | Process CIFs, fresh database |
| `run_process_catchup.bat` | `bps_process.py --mode all` then `--mode summary` | Incremental processing (no reset) |
| `compile_results.bat` | `compile_results.py` | Generate text report from database |

### Validation

| Batch file | Runs | Description |
|------------|------|-------------|
| `run_validate_controls.bat` | `bps_validate_controls.py` | 6 validation tests (30-60 min) |
| `run_pdb_validate.bat` | `bps_pdb_validate.py` | PDB experimental validation |

### Loop Taxonomy

| Batch file | Runs | Description |
|------------|------|-------------|
| `run_loop_taxonomy.bat` | `loop_taxonomy.py` | v1: ~119 PDB structures (5-15 min) |
| `run_loop_taxonomy_v2.bat` | `loop_taxonomy_v2.py` | v2: ~500 structures (15-45 min) |
| `run_clustering.bat` | `cluster_loops.py --sample 10000` | DBSCAN on 10k loop sample |

### AlphaFold Pipeline

| Batch file | Runs | Description |
|------------|------|-------------|
| `run_alphafold_pipeline.bat` | `alphafold_pipeline.py` | Interactive menu (4 modes) |
| `run_with_loops.bat` | `alphafold_pipeline.py` with `--checkpoint` | Full pipeline with loops |

### Advanced Analysis

| Batch file | Runs | Description |
|------------|------|-------------|
| `run_higher_order.bat` | `higher_order_null.py --sample 200` | 4-level null model |
| `run_within_foldclass_cv.bat` | `within_foldclass_cv.py` | Within-fold cross-organism CV |
| `run_w_independence.bat` | `w_independence.py` | W train/test split (editable paths) |
| `run_torus_distance_control.bat` | `torus_distance_control.py --sample 200` | Tier-0 falsification (15-30 min) |
| `run_cluster_stability.bat` | `subsample_cluster_stability.py` | 5x subsample stability |
| `run_designed_proteins.bat` | `designed_proteins.py --download --analyze` | De novo protein analysis |
| `run_designed_seg_real.bat` | `designed_seg_real.py` | Designed protein decomposition |
| `run_generate_figures.bat` | `generate_figures.py --sample 200` | Publication figures |

### Orchestration

| Batch file | Runs | Description |
|------------|------|-------------|
| `run_full_pipeline.bat` | build_superpotential, within_foldclass_cv, w_independence, subsample_cluster_stability, generate_figures | Full validation suite in sequence |

### Diagnostics

| Batch file | Runs | Description |
|------------|------|-------------|
| `diagnose_2rhe.bat` | `diagnose_2rhe.py` | Trace 2RHE mislabel |
| `diagnose_foldclass.bat` | `diagnose_foldclass.py` | Fold classification bug check |

---

## Package Scripts (bps/ and loops/)

These are the refactored module versions inside the `bps/` and `loops/` packages.
Run with `python -m` from the project root.

| Command | Purpose |
|---------|---------|
| `python -m bps.superpotential` | Build and validate W (self-test) |
| `python -m bps.extract` | Test dihedral extraction (1UBQ) |
| `python -m bps.validate_pdb` | PDB validation (235 structures) |
| `python -m bps.bps_download` | Download AlphaFold proteomes (23 organisms) |
| `python -m bps.bps_process` | Process CIF/PDB files for BPS/L |
| `python -m bps.compile_results` | Compile results to markdown |
| `python -m loops.taxonomy` | Loop taxonomy (cached PDB) |
| `python -m loops.classifier` | Loop family classifier (Random Forest) |
| `python -m loops.forward_kinematics` | Forward kinematics validation |
| `python -m loops.reconstruct_test` | End-to-end reconstruction test |

Package batch files are in `scripts/` (numbered 01-10, plus `run_pdb_pipeline.bat`
and `run_alphafold_pipeline.bat`).

---

## Data Flow

```
                    AlphaFold EBI
                         |
                   bps_download.py
                         |
                         v
              alphafold_cache/*/AF-*.cif
                    |          |
     build_superpotential.py   |
            |                  |
            v                  v
   superpotential_W.npz    bps_process.py
            |                  |
            |                  v
            |          bps_database.db
            |           /    |     \
            |          v     v      v
            |   compile   bps_    bps_
            |   results  stats   figures
            |          \   |    /
            |           v  v   v
            |         Reports + Figures
            |
            +-----> alphafold_pipeline.py
            |              |
            |              v
            |      proteins.csv + loops.csv
            |              |
            |              v
            |       cluster_loops.py
            |
            +-----> higher_order_null.py
            +-----> within_foldclass_cv.py
            +-----> w_independence.py
            +-----> torus_distance_control.py
            +-----> subsample_cluster_stability.py
            +-----> designed_proteins.py
            +-----> designed_seg_real.py
            +-----> generate_figures.py


              RCSB PDB
                 |
                 v
        loop_taxonomy.py (v1/v2)
        bps_pdb_validate.py
        bps_plaxco.py
```

---

## Output Files Summary

| File | Generated by |
|------|-------------|
| `results/superpotential_W.npz` | build_superpotential.py |
| `alphafold_bps_results/bps_database.db` | bps_process.py |
| `alphafold_bps_results/pipeline.log` | bps_process.py |
| `bps_results_for_claude.txt` | compile_results.py |
| `validation_report.md` | bps_validate_controls.py |
| `pdb_validation_report.md` | bps_pdb_validate.py |
| `alphafold_bps_results/bps_statistics.txt` | bps_statistics.py |
| `alphafold_bps_results/bps_plaxco_validation.txt` | bps_plaxco.py |
| `alphafold_bps_results/figures/*.png` | bps_figures.py |
| `results/figures/fig*.png` | generate_figures.py |
| `results/proteins.csv` | alphafold_pipeline.py |
| `results/loops.csv` | alphafold_pipeline.py |
| `results/organisms.csv` | alphafold_pipeline.py |
| `results/report.md` | alphafold_pipeline.py |
| `results/loop_families.csv` | cluster_loops.py |
| `results/loop_taxonomy_report.md` | cluster_loops.py |
| `loop_taxonomy_output/*.md` | loop_taxonomy.py |
| `loop_taxonomy_v2_output/*.md` | loop_taxonomy_v2.py |
| `results/higher_order_null_report.md` | higher_order_null.py |
| `results/within_foldclass_cv_report.md` | within_foldclass_cv.py |
| `results/w_independence_report.md` | w_independence.py |
| `results/torus_distance_control_report.md` | torus_distance_control.py |
| `results/subsample_cluster_stability_report.md` | subsample_cluster_stability.py |
| `results/designed_proteins_report.md` | designed_proteins.py |
| `results/designed_seg_real_report.md` | designed_seg_real.py |

---

## Troubleshooting

**"No module named 'gemmi'"** — Install with `pip install gemmi`. Required for
the root-level scripts that parse AlphaFold CIF files directly.

**"No module named 'bps'"** — You're running a package script wrong. Use
`python -m bps.scriptname` from the project root, not `python bps/scriptname.py`.

**"Chain 'A' not found"** — Some PDB files use non-standard chain IDs. The
pipeline skips these automatically.

**99% classified as "other"** — Fold classification bug. Run `diagnose_foldclass.py`
to check SS fractions. If mean alpha and beta are near zero, angles are being
passed in radians instead of degrees to `assign_basin()`.

**"bps_database.db not found"** — Run `python bps_process.py --mode all` first
to create the database from cached CIF files.

**"superpotential_W.npz not found"** — Run `python build_superpotential.py` first.
Most analysis scripts auto-detect or build W if the `.npz` is missing, but some
require it explicitly.

**Database locked errors** — SQLite WAL mode is used. If multiple scripts try to
write simultaneously, wait for one to finish. `bps_download.py` (network only) and
`bps_process.py` (compute only) can run in parallel safely.

**RCSB API timeout** — Loop taxonomy and PDB validation scripts fall back to
curated lists and GitHub mirrors. Fully offline operation works with cached files.
