# TorusFold Usage Guide

A complete guide to running every script in the TorusFold pipeline.
Includes Windows batch files (`.bat`) for one-click execution.

---

## Prerequisites

**Python 3.10+** required. Install dependencies:

```
pip install numpy scipy biopython matplotlib scikit-learn
```

GPU is **not** needed for any analysis script. Only the future `torusfold/`
prediction module (not yet implemented) will require PyTorch + GPU.

**All scripts must be run from the project root directory** (`TorusFold/`).
They use `python -m` module syntax for correct import resolution.

---

## Quick Start

Run the core pipeline in order:

```
python -m bps.superpotential          # 1. Build & validate W landscape
python -m bps.extract                 # 2. Test dihedral extraction
python -m bps.validate_pdb            # 3. PDB experimental validation
python -m loops.taxonomy              # 4. Loop path taxonomy
python -m loops.classifier            # 5. Loop family classifier
python -m loops.forward_kinematics    # 6. Forward kinematics validation
python -m loops.reconstruct_test      # 7. End-to-end reconstruction test
```

For AlphaFold proteome analysis (requires ~50 GB disk for all 23 organisms):

```
python -m bps.bps_download            # Download AlphaFold proteomes
python -m bps.bps_process             # Compute BPS/L across all organisms
python -m bps.compile_results         # Generate paper-ready tables
```

Windows users: see the `.bat` files in the `scripts/` directory (described below).

---

## Script Reference

### 1. `bps/superpotential.py` — Build the Superpotential W

**What it does:** Constructs W(phi,psi) = -ln(p + epsilon) on the Ramachandran
torus from empirical backbone dihedral statistics. This is the foundational
landscape that all other analyses depend on.

**Run:**
```
python -m bps.superpotential
```

**No arguments.** Runs a self-test that:
- Builds W from KDE on cached PDB structures
- Validates landscape depths (alpha is global minimum, beta is secondary)
- Tests W lookup at key Ramachandran points
- Computes BPS/L on test structures (1UBQ, 1AKI, etc.)
- Verifies basin assignment logic

**Requires:** PDB files in `data/pdb_cache/`

**Produces:** Nothing on disk (prints validation results to stdout).

**Key exports (for other scripts):**
- `build_superpotential()` -> (W_grid, phi_grid, psi_grid)
- `lookup_W_batch()` -> vectorized W interpolation
- `assign_basin(phi_deg, psi_deg)` -> basin classification
- `compute_bps_per_residue(W_values)` -> BPS/L

---

### 2. `bps/extract.py` — Dihedral Extraction from PDB

**What it does:** Extracts backbone (phi, psi) dihedral angles from PDB files
using BioPython. Tests single-chain isolation (the multi-chain bug guard).

**Run:**
```
python -m bps.extract
```

**No arguments.** Runs Priority 2 validation:
- Downloads 1UBQ (or generates synthetic PDB if offline)
- Extracts dihedrals for chain A only
- Validates basin fractions (~40% alpha, ~20% beta for ubiquitin)
- Computes end-to-end BPS/L
- Tests that requesting a missing chain raises ValueError

**Requires:** Network access (to download 1UBQ) or existing `data/pdb_cache/1UBQ.pdb`

**Produces:** Downloaded PDB file saved to `data/pdb_cache/`

**Key exports:**
- `download_pdb(pdb_id)` -> downloads from RCSB with GitHub mirror fallback
- `extract_dihedrals_pdb(pdb_path, chain_id)` -> list of residue dicts
- `compute_basin_fractions(residues)` -> dict of basin -> fraction

---

### 3. `bps/validate_pdb.py` — PDB Experimental Validation

**What it does:** The main validation pipeline. Processes hundreds of high-
resolution PDB structures, computes BPS/L for each, reports per-protein CV,
fold-class breakdown, and the three-level backbone decomposition
(Real < Markov < Shuffled).

**Run:**
```
python -m bps.validate_pdb
```

**No arguments.** Processing tiers:
1. **Tier 0:** Process already-cached PDB files (instant)
2. **Tier 1:** Query RCSB Search API for additional structures (if < 100 cached)
3. **Tier 2:** Download from curated list with GitHub fallback (if < 50 total)

Quality filters: excludes NMR structures and chains < 50 residues.

**Requires:** PDB files in `data/pdb_cache/` or network access

**Produces:**
- `output/pdb_validation_report.md` — full validation report with:
  - Filtered/unfiltered statistics
  - Three-level decomposition (realistic W + smooth W)
  - Fold-class breakdown with Cohen's d separations
  - Per-structure results table

**Runtime:** ~2-5 minutes (depends on cache size and network).

---

### 4. `loops/taxonomy.py` — Loop Path Taxonomy

**What it does:** Extracts loop regions between alpha-helix and beta-sheet
basins, computes torus-aware pairwise distances, clusters loops by geometric
similarity using DBSCAN. Length-stratified: short (<=7 residues), medium (8-10),
long (11-15). Recursive subclustering on high-variance clusters.

**Run:**
```
python -m loops.taxonomy
```

**No arguments.** Key parameters (hardcoded):
- `MIN_FLANK = 3` — minimum consecutive residues to count as SS element
- `MAX_LOOP_LEN = 20` — maximum loop length to extract
- `TIGHT_CV_THRESH = 30.0` — CV% threshold for "tight" vs "catch-all" families
- `EPS_CANDIDATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]`

**Requires:** PDB files in `data/pdb_cache/`

**Produces:**
- `output/loop_taxonomy_report.md` — taxonomy report with family tables
- `output/plots/torus_Short_le7.png` — short loop families on T^2
- `output/plots/torus_Medium_8_10.png` — medium loop families
- `output/plots/torus_Long_11_15.png` — long loop families

**Runtime:** ~3-5 minutes.

---

### 5. `loops/classifier.py` — Loop Family Classifier

**What it does:** Trains a Random Forest to predict loop family from amino acid
sequence composition. 26-dimensional feature vector: 20 AA composition + loop
length + direction + Gly count + Pro count + N/C terminal AA identity.
Evaluated with stratified cross-validation.

**Run:**
```
python -m loops.classifier
```

**No arguments.**

**Requires:** PDB files in `data/pdb_cache/` (re-extracts loops internally)

**Produces:**
- `output/loop_classifier_report.md` — accuracy, per-family precision/recall,
  feature importances, confusion matrix

**Target:** >60% accuracy. Current result: **80.0%**.

---

### 6. `loops/forward_kinematics.py` — Forward Kinematics Validation

**What it does:** Validates the NeRF backbone reconstruction algorithm.
Extracts experimental (phi, psi, omega) from PDB crystal structures,
reconstructs backbone coordinates, computes RMSD against the experimental
structure after Kabsch alignment.

**Run:**
```
python -m loops.forward_kinematics
```

**No arguments.**

**Requires:** PDB files in `data/pdb_cache/`

**Produces:** Stdout only (prints RMSD for each test structure).

**Target:** RMSD < 0.5 A from experimental dihedrals.

---

### 7. `loops/reconstruct_test.py` — End-to-End Reconstruction Test

**What it does:** Tests what happens when you replace actual loop paths with
the nearest canonical family centroid and SS element angles with basin centers.
Measures the RMSD penalty of using canonical paths vs experimental dihedrals.

**Run:**
```
python -m loops.reconstruct_test
```

**No arguments.**

**Requires:** PDB files in `data/pdb_cache/`

**Produces:**
- `output/reconstruction_test_report.md` — per-protein RMSD comparison

**Key finding:** ~15.8 A mean RMSD increase when using canonical centroids alone,
confirming that intra-basin conformational coherence is critical.

---

### 8. `bps/bps_download.py` — AlphaFold Proteome Downloader

**What it does:** Downloads proteome-level predicted structure tarballs from the
AlphaFold EBI database. 23 organisms covering all major kingdoms of life.

**Run:**
```
python -m bps.bps_download                     # all 23 organisms
python -m bps.bps_download --organism ecoli     # just E. coli
python -m bps.bps_download --organism human     # just human
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--organism NAME` | all | Short name: ecoli, human, yeast, drosophila, etc. |
| `--cache-dir PATH` | `data/alphafold_cache` | Where to save extracted CIF files |
| `--version N` | 4 | AlphaFold model version |

**Available organisms (23):**

| Short name | Species |
|------------|---------|
| ecoli | Escherichia coli K-12 |
| bsubtilis | Bacillus subtilis 168 |
| mtb | Mycobacterium tuberculosis H37Rv |
| caulobacter | Caulobacter vibrioides |
| pseudomonas | Pseudomonas aeruginosa PAO1 |
| mjannaschii | Methanocaldococcus jannaschii |
| hvolcanii | Haloferax volcanii |
| yeast | Saccharomyces cerevisiae S288C |
| spombe | Schizosaccharomyces pombe |
| arabidopsis | Arabidopsis thaliana |
| rice | Oryza sativa japonica |
| celegans | Caenorhabditis elegans |
| drosophila | Drosophila melanogaster |
| honeybee | Apis mellifera |
| zebrafish | Danio rerio |
| chicken | Gallus gallus |
| mouse | Mus musculus |
| rat | Rattus norvegicus |
| human | Homo sapiens |
| plasmodium | Plasmodium falciparum 3D7 |
| leishmania | Leishmania infantum |
| trypanosoma | Trypanosoma brucei brucei TREU927 |
| thermus | Thermus thermophilus HB27 |

**Requires:** Network access to `ftp.ebi.ac.uk`

**Produces:** `data/alphafold_cache/<organism_id>/` directories with CIF files.

**Disk space:** ~2 GB per small proteome (bacteria), ~20 GB for human. ~50 GB total.

---

### 9. `bps/bps_process.py` — AlphaFold BPS/L Processing

**What it does:** The main proteome processing pipeline. Fast custom CIF parser
(~40x faster than BioPython) extracts backbone N/CA/C atoms, computes (phi, psi)
from atomic coordinates, then BPS/L, SS fractions, and fold class for every
protein. Outputs CSV files for downstream analysis.

**Run:**
```
python -m bps.bps_process                        # all organisms
python -m bps.bps_process --organism ecoli        # one organism
python -m bps.bps_process --pdb-only              # PDB cache only (for testing)
python -m bps.bps_process --plddt 90              # stricter quality filter
python -m bps.bps_process --max-per-organism 100  # subsample for quick test
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--organism NAME` | all | Short name (e.g., ecoli) |
| `--pdb-only` | off | Process PDB cache instead of AlphaFold |
| `--plddt N` | 85.0 | pLDDT quality threshold (AlphaFold only) |
| `--max-per-organism N` | 0 (all) | Cap structures per organism |
| `--results-dir PATH` | `results/` | Output directory |

**Requires:**
- `data/alphafold_cache/` with CIF files (from `bps_download.py`)
- OR `data/pdb_cache/` with PDB files (with `--pdb-only`)

**Produces:**
- `results/per_protein_bpsl.csv` — one row per protein with BPS/L, SS fractions, fold class
- `results/per_organism.csv` — per-organism mean/std/CV + cross-organism summary

**CSV columns (per_protein_bpsl.csv):**
`name, organism, species, length, n_valid, bps_l, frac_alpha, frac_beta,
frac_ppII, frac_alphaL, frac_other, fold_class, mean_plddt`

**Built-in diagnostic:** Prints SS fraction check and warns if >90% classify as "other"
(catches the radians/degrees bug).

**Runtime:** ~1 hour for all 23 organisms (~52k structures). ~30 seconds for `--pdb-only`.

---

### 10. `bps/compile_results.py` — Results Compiler

**What it does:** Reads the CSV output from `bps_process.py` and generates
publication-ready markdown tables: per-organism summary (paper Table 1),
fold-class breakdown (paper Table 2), cross-organism CV, and SS diagnostic.

**Run:**
```
python -m bps.compile_results
python -m bps.compile_results --output my_report.md
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--results-dir PATH` | `results/` | Input CSV directory |
| `--output PATH` | `output/alphafold_compilation_report.md` | Output report |

**Requires:** `results/per_protein_bpsl.csv` (from `bps_process.py`)

**Produces:** `output/alphafold_compilation_report.md`

---

## Windows Batch Files

All batch files are in the `scripts/` directory. Double-click to run, or
execute from a Command Prompt opened in the TorusFold root directory.

| Batch file | What it runs |
|------------|-------------|
| `scripts\setup.bat` | Install Python dependencies |
| `scripts\01_superpotential.bat` | Build and validate W landscape |
| `scripts\02_extract.bat` | Test dihedral extraction |
| `scripts\03_validate_pdb.bat` | PDB experimental validation |
| `scripts\04_loop_taxonomy.bat` | Loop path taxonomy (DBSCAN) |
| `scripts\05_loop_classifier.bat` | Loop family classifier |
| `scripts\06_forward_kinematics.bat` | Forward kinematics validation |
| `scripts\07_reconstruct_test.bat` | End-to-end reconstruction test |
| `scripts\08_download_alphafold.bat` | Download AlphaFold proteomes |
| `scripts\09_process_alphafold.bat` | Process AlphaFold data |
| `scripts\10_compile_results.bat` | Compile results into report |
| `scripts\run_pdb_pipeline.bat` | Run steps 1-7 in sequence |
| `scripts\run_alphafold_pipeline.bat` | Run steps 8-10 in sequence |

---

## Pipeline Data Flow

```
data/pdb_cache/*.pdb
       |
       v
  [superpotential.py] --> W(phi,psi) grid (in memory)
       |
       +---> [extract.py] ---------> dihedral validation
       |
       +---> [validate_pdb.py] ----> output/pdb_validation_report.md
       |
       +---> [taxonomy.py] --------> output/loop_taxonomy_report.md
       |                              output/plots/torus_*.png
       |
       +---> [classifier.py] ------> output/loop_classifier_report.md
       |
       +---> [forward_kinematics.py] -> stdout (RMSD results)
       |
       +---> [reconstruct_test.py] -> output/reconstruction_test_report.md


data/alphafold_cache/*/*.cif
       |
  [bps_download.py]  (populates the cache)
       |
       v
  [bps_process.py] -----> results/per_protein_bpsl.csv
       |                   results/per_organism.csv
       v
  [compile_results.py] -> output/alphafold_compilation_report.md
```

---

## Output Files Summary

| File | Generated by | Contents |
|------|-------------|----------|
| `output/pdb_validation_report.md` | `validate_pdb.py` | Full PDB validation: CV, fold-class, three-level decomposition |
| `output/loop_taxonomy_report.md` | `taxonomy.py` | Loop families: tight/catch-all, per-length strata |
| `output/loop_classifier_report.md` | `classifier.py` | Random forest accuracy, feature importances |
| `output/reconstruction_test_report.md` | `reconstruct_test.py` | RMSD: canonical loops vs experimental |
| `output/alphafold_compilation_report.md` | `compile_results.py` | Cross-organism summary, fold-class table |
| `output/plots/torus_*.png` | `taxonomy.py` | Loop paths plotted on Ramachandran torus |
| `results/per_protein_bpsl.csv` | `bps_process.py` | Per-protein BPS/L + SS fractions |
| `results/per_organism.csv` | `bps_process.py` | Per-organism means + cross-organism CV |

---

## Troubleshooting

**"No module named 'bps'"** — You must run from the project root (`TorusFold/`)
using `python -m bps.scriptname`, not `python bps/scriptname.py`.

**"Chain 'A' not found"** — Some PDB files use non-standard chain IDs. The
pipeline skips these automatically. Not an error.

**99% classified as "other"** — The fold classification bug. Check SS fractions:
```
python -c "
import csv
alphas, betas = [], []
with open('results/per_protein_bpsl.csv') as f:
    for row in csv.DictReader(f):
        alphas.append(float(row['frac_alpha']))
        betas.append(float(row['frac_beta']))
print(f'Mean alpha: {sum(alphas)/len(alphas):.3f}')
print(f'Mean beta:  {sum(betas)/len(betas):.3f}')
print(f'Alpha>35%:  {sum(1 for a in alphas if a>=0.35)}')
print(f'Beta>25%:   {sum(1 for b in betas if b>=0.25)}')
"
```
If mean alpha and beta are near zero, angles are being passed in radians
instead of degrees to `assign_basin()`. The current pipeline handles this
correctly.

**RCSB API timeout** — The pipeline falls back to the curated list and then
to GitHub mirrors. Fully offline operation works with cached PDB files.

**Memory issues with large proteomes** — Process one organism at a time:
```
python -m bps.bps_process --organism human --max-per-organism 5000
```
