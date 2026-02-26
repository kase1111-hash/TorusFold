# CLAUDE.md — TorusFold

## What This Project Is

TorusFold is a research program that treats protein backbones as one-dimensional paths on the Ramachandran torus T² = S¹ × S¹ and derives structural invariants, loop path taxonomies, and (eventually) structure prediction from this framework.

The core discovery: a scalar superpotential W(φ,ψ) = −ln p(φ,ψ) defined on T² yields a universal invariant BPS/L ≈ 0.202 ± 0.004 across all domains of life, validated on 66,549 AlphaFold structures and 162 experimental PDB crystal structures. Loop paths between secondary structure basins cluster into discrete canonical families for short loops (≤7 residues), enabling classification-based backbone prediction.

## Repository Structure

```
TorusFold/
├── CLAUDE.md                          # This file
├── README.md                          # Public-facing project description
│
├── bps/                               # BPS/L proteome atlas pipeline
│   ├── bps_download.py                # AlphaFold CIF file downloader
│   ├── bps_process.py                 # Dihedral extraction + BPS computation
│   ├── bps_validate_controls.py       # 6 validation controls (shuffle, Markov, etc.)
│   ├── bps_pdb_validate.py            # PDB experimental validation (162 structures)
│   ├── compile_results.py             # Report generator from SQLite
│   └── superpotential.py              # W(φ,ψ) construction and lookup
│
├── loops/                             # Loop path taxonomy
│   ├── loop_taxonomy_v2.py            # Main loop extraction + clustering pipeline
│   ├── loop_classifier.py             # Sequence → path family classifier (TODO)
│   └── forward_kinematics.py          # (φ,ψ) → Cartesian reconstruction (TODO)
│
├── torusfold/                         # Structure prediction module (TODO)
│   ├── w_landscape.py                 # Differentiable W embedding (PyTorch)
│   ├── dihedral_head.py               # φ,ψ prediction head
│   ├── bps_loss.py                    # BPS/L-aware loss functions
│   └── coherence_loss.py              # Intra-basin coherence loss
│
├── data/                              # Generated data (gitignored except schema)
│   ├── alphafold_cache/               # Cached CIF files
│   ├── pdb_cache/                     # Cached PDB files
│   └── results.db                     # SQLite results database
│
├── papers/                            # Manuscripts
│   ├── bps_paper_v03.md               # Main BPS/L paper draft
│   └── loop_taxonomy_paper.md         # Loop path families paper (TODO)
│
├── docs/                              # Architecture specs
│   └── alphafold_integration.md       # BPS/L-informed prediction architecture
│
└── tests/                             # Validation and regression tests
    ├── test_superpotential.py
    ├── test_dihedral_extraction.py
    └── test_bps_computation.py
```

## Key Concepts

### The Superpotential W(φ,ψ)

A fixed scalar field on the Ramachandran torus derived from the empirical probability density of backbone dihedral angles across all known protein structures. Primary superpotential: W = −√P(φ,ψ) (used in the main BPS pipeline, `bps_process.py` and `bps_validate_controls.py`). Alternative forms including W = −ln(P + ε) are tested for transform invariance (see W-robustness validation). The `bps/superpotential.py` module and several analysis scripts use −ln(P + ε) with ε = 1e-7. The absolute value of BPS/L depends on the W construction (gauge choice); what matters is the three-level decomposition: Real < Markov < Shuffled.

W is **never trainable**. It is the landscape. Proteins navigate it.

Critical points of W:
- **Minima (valleys):** α-helix basin at (−63°,−43°), β-sheet basin at (−120°,130°), polyproline II at (−75°,150°), left-handed helix at (57°,47°)
- **Saddle points (passes):** Mountain passes between basins — these are where loops cross
- **Maxima (peaks):** Steric clash zones

### BPS/L (Backbone Potential Superpotential per residue)

```
BPS = Σ_{i=1}^{L-1} |W(φ_{i+1}, ψ_{i+1}) − W(φ_i, ψ_i)|
BPS/L = BPS / L
```

Total variation of W along the sequential backbone path, normalized by chain length. Measures "roughness" — how much the backbone fluctuates through the W landscape per residue.

**Universal value:** 0.202 ± 0.004 (CV = 2.1%) across 22 organisms, all kingdoms of life.
**PDB experimental:** 0.207 ± 0.051 on 162 crystal structures (Δ = 2.4%).

**Key independence results (at pLDDT ≥ 85):**
- vs contact order: r = −0.05
- vs helix content: r = 0.03
- vs sheet content: r = −0.11

### Loop Path Families

Transitions between α and β Ramachandran basins follow canonical paths on T².

**Established results (1,207 loops from 470 PDB structures):**
- Short loops (≤7 residues): 30 tight families, 100% coverage, |ΔW| CV = 6.6–9.3%
- Medium loops (8–10): 1–2 families per direction, CV ~24%
- Long loops (11+): no tight clustering, need 3D context

**Interpretation:** Short loops are geometrically constrained by saddle-point topology. Long loops escape that constraint.

### Three-Level Backbone Decomposition

| Level | BPS/L | What it captures |
|-------|-------|------------------|
| Shuffled (destroy all order) | 0.55 | Basin occupancy only |
| Markov (preserve transitions, randomize intra-basin) | 0.30 | Transition architecture |
| Real proteins | 0.20 | Full conformational coherence |

The 0.10 gap between Markov and real = 97% suppression of intra-basin roughness = the quantitative signature of secondary structure.

## Development Guidelines

### Python Environment

```bash
# Core dependencies
pip install numpy scipy biopython matplotlib scikit-learn

# For PyTorch-based prediction modules (torusfold/)
pip install torch

# For testing
pip install pytest
```

Python 3.10+ required. No GPU needed for BPS/loop analysis. GPU needed only for torusfold/ training.

### Critical Implementation Rules

**Dihedral angles:**
- Always in radians internally. Convert to degrees only for display/basin assignment.
- BioPython returns (φ,ψ) in radians. Our CIF parser requires sign correction (PHI_SIGN = −1 for raw extraction; determined empirically per run).
- Circular arithmetic everywhere: use `atan2(sin(a−b), cos(a−b))` for angular differences, never raw subtraction.
- Terminal residues have φ=None or ψ=None. Skip them in BPS computation.

**Superpotential W:**
- W grid is 360×360 over [−π, π)² with periodic boundary conditions.
- Lookup uses bilinear interpolation with periodic wrapping.
- W is normalized so min = 0. Absolute values are arbitrary; only differences matter.
- The density floor ε = 1e-7 prevents log(0) in unpopulated regions.
- **Never modify W during analysis.** It is a fixed reference frame.

**Basin assignment (wide definition — used in all published analyses):**
- α-helix: φ ∈ (−160°, 0°), ψ ∈ (−120°, 30°)
- β-sheet: φ ∈ (−170°, −70°), ψ > 90° OR ψ < −120° — wraps around ±180°
- Everything else: "other"
- The β-sheet basin MUST merge the two ψ ranges (primary + wrapped). Failing to handle the ±180° wrap is a known bug source.
- **Note:** `bps/superpotential.py` uses a narrower five-basin classification (α, β, ppII, αL, other) for detailed Ramachandran analysis. The wide two-basin definition above is used for all BPS/L and fold-class analyses. Results under the narrow definition (φ ∈ (−100°, −30°), ψ ∈ (−67°, −7°) for α) should be tested as a sensitivity control.

**Chain extraction:**
- ALWAYS extract a single chain (usually chain A). Multi-chain extraction was a confirmed bug in earlier versions (1AON/GroEL produced BPS 10× too high because BioPython walked all 14 subunits).
- Verify extracted residue count against expected chain length. Flag mismatches > 30%.

**Fold-class labels:**
- Do NOT trust hardcoded fold-class labels without cross-validation against computed SS from dihedrals. A confirmed bug assigned 2RHE (immunoglobulin VL, all-β) as "all-alpha" due to a hardcoded mislabel. Always validate labels against computed helix/sheet fractions.

### Data Sources

**AlphaFold Database:**
- Proteome tarballs from https://alphafold.ebi.ac.uk/
- mmCIF format, parsed with custom fast parser (bps_process.py)
- Per-residue pLDDT in B-factor field
- Quality threshold: pLDDT ≥ 85 for "well-modeled"

**RCSB PDB:**
- Individual structures via https://files.rcsb.org/download/{PDB_ID}.pdb
- Parsed with BioPython PDBParser + PPBuilder
- Quality: resolution < 2.0Å, X-ray diffraction, R-free < 0.25
- Search API: https://search.rcsb.org/rcsbsearch/v2/query

**PISCES:**
- Non-redundant chain lists from https://dunbrack.fccc.edu/pisces/
- Server is unreliable (404s common). Always have a curated fallback list.

### Testing

```bash
# Run all tests
pytest tests/ -v

# Quick sanity check: compute BPS/L for a single protein
python -c "
from bps.superpotential import build_superpotential, lookup_W
from bps.bps_process import extract_dihedrals, compute_bps
W, phi_grid, psi_grid = build_superpotential()
# ... extract dihedrals from a CIF file ...
# BPS/L should be in [0.10, 0.35] for any well-folded protein
"
```

**Regression checks after any code change:**
1. BPS/L for E. coli proteome (pLDDT ≥ 85) should be 0.199 ± 0.041
2. Shuffled BPS/L should be ~0.55 (2.7× real)
3. Transition matrix prediction should give 0.2005 (within 1.3% of observed)
4. PDB experimental mean should be ~0.207

### Known Issues and Gotchas

1. **BioPython PPBuilder** walks all chains by default. Always filter to target chain before building peptides.

2. **Sign convention:** Our CIF parser extracts raw dihedral angles that may need negation. The sign is determined empirically by checking what fraction of residues land in the α-helix region. PHI_SIGN is determined once per run and applied to both φ and ψ.

3. **AlphaFold version effects:** v4 models have systematically higher pLDDT than v6. This creates a false kingdom-level signal if not controlled. Always filter by pLDDT, never compare raw BPS/L across AF versions.

4. **DBSCAN eps sensitivity:** Loop clustering results depend on the eps parameter. The v2 script tries multiple eps values and picks the first that gives ≥2 clusters with <50% noise. For publication, report results at multiple eps values.

5. **β-sheet basin wrap:** The β-sheet region straddles ψ = ±180°. Any distance calculation, interpolation, or binning that doesn't handle this wrap will silently misclassify ~15% of sheet residues as "other." This was a confirmed bug that made our SS classifier report 2RHE as 0.9% sheet when DSSP says ~50%.

6. **Circular interpolation for path resampling:** When resampling loop paths to a common length, interpolate sin/cos components separately, then recover the angle via atan2. Linear interpolation on raw angles fails at the ±180° boundary.

## Current Status and Priorities

### Completed
- [x] Superpotential W construction and validation
- [x] BPS/L computation across 66,549 AlphaFold structures (25 organisms)
- [x] PDB experimental validation (162 structures, Δ = 2.4%)
- [x] All 6 validation controls (shuffle, W-robustness, fold-class, amino acid, transition matrix, Markov)
- [x] Three-level backbone decomposition
- [x] pLDDT confound identification and resolution
- [x] Kingdom-level Simpson's paradox demonstration
- [x] Loop path taxonomy v1 (119 structures, 216 loops — proof of concept)
- [x] Loop path taxonomy v2 (470 structures, 1207 loops — length-stratified)
- [x] 2RHE fold-class mislabel diagnosed and traced to hardcoded list
- [x] BPS paper draft v0.3
- [x] Medium article (public communication)
- [x] AlphaFold integration architecture spec

### Active Priorities
1. **Fix 2RHE and audit full hardcoded fold-class list** in bps_pdb_validate.py — one-line fix for 2RHE, but grep for other duplicates and cross-validate all labels against computed SS
2. **Widen β-sheet ψ window** in assign_ss() — current thresholds miss real sheet residues at basin edges, producing "coil-dominant" classifications for known β-rich proteins
3. **Build loop path classifier** — random forest on amino acid composition of short (≤7) loops → family label. Target: >60% accuracy on held-out loops. If this works, the sequence-to-path mapping exists.
4. **Extract per-family centroid paths** — the canonical path for each of the 30 short-loop families, stored as resampled (φ,ψ) trajectories for use as prediction templates

### Next Phase
5. Forward kinematics module: (φ,ψ) trajectory → Cartesian backbone coordinates
6. End-to-end reconstruction test: SS prediction + intra-basin positions + loop family selection → backbone → RMSD vs experimental structure
7. Differentiable W module in PyTorch for integration with OpenFold
8. Loop taxonomy paper: "Discrete Loop Path Families on the Ramachandran Torus and Their Sequence Determinants"
9. BPS paper submission (target: PNAS or Physical Review E)

### Backlog
- Expand PDB validation to 500+ experimental structures with fresh PISCES pull
- Compute BPS/L on all CASP14/15 predictions (Phase 0 of AlphaFold integration)
- Second topological invariant: test whether winding number Q_ψ provides additional constraint on loop paths
- Higher-order (2nd, 3rd) transition matrices conditioned on amino acid identity
- Medium-loop (8–10 residue) path prediction: single dominant mode + variance model

## File-by-File Reference

### bps/superpotential.py
Builds W(φ,ψ) from a 10-component von Mises mixture fit to PDB backbone statistics. Outputs a 360×360 grid with bilinear interpolation lookup. Also provides gradient ∇W for transition loss computation.

Key functions:
- `build_superpotential(grid_size=360)` → (W_grid, phi_grid, psi_grid)
- `lookup_W(W_grid, phi_grid, psi_grid, phi, psi)` → float
- `lookup_W_batch(W_grid, phi_grid, psi_grid, phi_array, psi_array)` → ndarray

### bps/bps_process.py
Main proteome processing pipeline. Parses AlphaFold mmCIF files, extracts (φ,ψ,pLDDT) per residue, computes BPS and auxiliary metrics (SS composition, Rg, CO), writes to SQLite.

Critical: contains the custom CIF parser that's ~40× faster than BioPython for bulk processing. The parser extracts only ATOM records for N, CA, C backbone atoms.

### bps/bps_pdb_validate.py
PDB experimental validation. Downloads ~200 high-resolution structures from RCSB, computes BPS/L, compares to AlphaFold. Contains the hardcoded fold-class lists (KNOWN BUG: 2RHE mislabeled, needs audit of full list).

### loops/loop_taxonomy_v2.py
Loop extraction, torus distance computation, recursive DBSCAN clustering, length-stratified analysis. Self-contained script — downloads PDB files, processes, clusters, generates report and plots.

Key parameters:
- `min_flank=3`: minimum consecutive residues in a basin to count as SS element
- `max_loop_len=20`: maximum loop length to extract
- DBSCAN eps search: [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
- Tight/catch-all threshold: CV < 30% on both |ΔW| and torus path length

### docs/alphafold_integration.md
Architecture specification for BPS/L-informed structure prediction. Describes dihedral prediction head, frozen W embedding, three loss functions (L_bps, L_coherence, L_transition), phased training strategy, and evaluation plan.

## Paper Status

### BPS Paper (v0.3, in preparation)
"Proteins as One-Dimensional Paths on the Ramachandran Torus: A Superpotential Framework for Backbone Structural Invariants"

Target journals: PNAS (biophysics), Physical Review E (mathematical physics), or Proteins (structural biology). The PDB experimental validation closes the biggest reviewer objection (AlphaFold artifact). Remaining work: fix fold-class labels, finalize figures, add PDB validation as Section 5b.7.

### Loop Taxonomy Paper (planned)
"Discrete Loop Path Families on the Ramachandran Torus and Their Sequence Determinants"

Core result: short loops (≤7 residues) cluster into ~30 canonical families with sub-10% variance. Needs the sequence classifier results to demonstrate the sequence-to-path mapping.

## Conventions

- All angles in radians internally, degrees for display and basin assignment
- All PDB IDs uppercase (e.g., "1UBQ" not "1ubq")
- Chain IDs case-sensitive (chain "A" ≠ chain "a")
- Basin names lowercase: "alpha", "beta", "ppII", "alphaL", "other"
- Pair keys directional: "alpha->beta" (not "alpha-beta" or "alpha<->beta")
- BPS/L reported to 3 decimal places (0.202, not 0.20)
- CV reported as percentage (2.1%, not 0.021)
- All statistical tests two-sided unless stated otherwise
- p-values reported with appropriate precision (p = 0.34, not p = 0.3)
