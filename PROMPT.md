# TorusFold Bootstrap — Clean Repo

Read CLAUDE.md first. It contains the full mathematical framework, all known pitfalls from prior implementations, numerical targets for validation, and the priority stack.

## Starting state

Clean repo with only: CLAUDE.md, README.md, this file, .gitignore. No existing code. Build everything from scratch.

## Priority 1: The superpotential module

Create `bps/superpotential.py` — the foundation everything else depends on.

```python
def build_superpotential(grid_size=360) -> Tuple[ndarray, ndarray, ndarray]:
    """Build W(φ,ψ) = -ln(p + ε) on T².
    10-component von Mises mixture, Gaussian smoothed σ=1.5, periodic BCs.
    Returns (W_grid, phi_grid, psi_grid). W normalized so min=0."""

def lookup_W(W_grid, phi_grid, psi_grid, phi, psi) -> float:
    """Bilinear interpolation with periodic wrapping. Angles in radians."""

def lookup_W_batch(W_grid, phi_grid, psi_grid, phi_array, psi_array) -> ndarray:
    """Vectorized version."""

def assign_basin(phi_deg, psi_deg) -> str:
    """Assign residue to Ramachandran basin. Angles in DEGREES.
    Returns: 'alpha', 'beta', 'ppII', 'alphaL', or 'other'.
    
    CRITICAL: β-sheet wraps around ψ=±180°. Must be:
        (-170 < φ < -70) AND (ψ > 90 OR ψ < -120)
    as a SINGLE condition. Do NOT use two separate ranges with a gap."""

def compute_bps(W_values: ndarray) -> float:
    """BPS = Σ|ΔW_i|. Input: array of W values along the chain."""

def compute_bps_per_residue(W_values: ndarray) -> float:
    """BPS/L = BPS / len(W_values)."""
```

Validation: build W, look up the α-helix center (-63°,-43°) and β-sheet center (-120°,130°). The α center should have a lower W value than the saddle point between them. Write a quick self-test that prints these values.

## Priority 2: Dihedral extraction from PDB files

Create `bps/extract.py` — parses PDB files and returns (φ,ψ) per residue.

```python
def extract_dihedrals_pdb(pdb_path: str, chain_id: str) -> List[Dict]:
    """Extract backbone dihedrals from a PDB file using BioPython.
    
    CRITICAL: Only extract the specified chain. BioPython's PPBuilder
    walks ALL chains by default. Filter to target chain BEFORE building
    peptides. Multi-chain extraction was a confirmed bug that produced
    10x inflated BPS for oligomers.
    
    Returns list of dicts: {resnum, resname, phi, psi} with angles in radians.
    phi and psi are None for terminal residues."""

def download_pdb(pdb_id: str, cache_dir: str) -> Optional[str]:
    """Download PDB file from RCSB. Returns path or None."""
```

Validation: download 1UBQ (ubiquitin, 76 residues, chain A), extract dihedrals, assign basins. Should get ~40% alpha, ~20% beta, ~40% other/coil. Print a summary.

## Priority 3: BPS/L computation and PDB validation

Create `bps/validate_pdb.py` — downloads ~200 high-resolution PDB structures, computes BPS/L, reports statistics.

Use the RCSB Search API to get high-resolution chains (resolution < 1.8Å, X-ray, protein). Fall back to a curated list of ~100 well-known structures if the API fails. Do NOT hardcode fold-class labels — compute SS composition from (φ,ψ) directly using `assign_basin()`.

Key output:
- Mean BPS/L across all PDB structures
- Standard deviation and CV
- Histogram of BPS/L values
- Comparison to the AlphaFold target: 0.202 ± 0.004

**Target result:** PDB mean BPS/L should be ~0.207 ± 0.05. If it's wildly different, something is wrong with the superpotential or dihedral extraction.

Write results to `output/pdb_validation_report.md`.

## Priority 4: Loop path taxonomy

Create `loops/taxonomy.py` — the loop extraction and clustering pipeline.

This is a self-contained pipeline:
1. Download ~500 PDB structures (reuse cache from Priority 3)
2. Extract (φ,ψ) trajectories
3. Assign basins per residue
4. Find transitions between α and β basins (loops)
5. Extract loop paths on T²
6. Cluster by geometric similarity using DBSCAN with torus-aware distance
7. Recursive subclustering on high-variance clusters (CV > 30%)
8. Length-stratified analysis: separate clustering for short (≤7), medium (8-10), long (11-15)

Key implementation details from CLAUDE.md:
- Torus distance: resample paths to common length using circular interpolation (interpolate sin/cos, recover angle via atan2)
- DBSCAN: try eps in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50], pick first giving ≥2 clusters with <50% noise
- Classify clusters as "tight" (CV < 30% on both |ΔW| and torus path length) or "catch-all"
- min_flank=3 consecutive residues in a basin to count as SS element
- max_loop_len=20

**Target results:**
- ~1000+ loops from 400+ structures
- Short loops: ~30 tight families, 100% coverage, |ΔW| CV < 10%
- Medium loops: fewer tight families, CV ~24%
- Long loops: no tight clustering

Write results to `output/loop_taxonomy_report.md` with per-family statistics.
Generate torus plots (W landscape background + colored loop paths per family).

## Priority 5: Loop path classifier

Create `loops/classifier.py` — predicts loop family from amino acid sequence.

**Input:** Loop data from Priority 4 (short loops only, ≤7 residues, tight families).

**Features per loop:**
- 20-dim amino acid composition vector (fraction of each AA in the loop)
- Loop length
- Flanking basin pair (alpha->beta = 0, beta->alpha = 1)
- Glycine count, proline count (separate features — these dominate Ramachandran)
- N-terminal flanking residue identity (one-hot or index)
- C-terminal flanking residue identity

**Model:** Random forest. No deep learning.

**Evaluation:** 5-fold stratified cross-validation.

**Report:**
- Overall accuracy (target: >60%)
- Per-family precision and recall
- Top 10 feature importances
- Confusion matrix
- Verdict: does sequence determine path family?

Write results to `output/loop_classifier_report.md`.

## Priority 6: Forward kinematics

Create `loops/forward_kinematics.py` — reconstruct Cartesian backbone from (φ,ψ).

Standard peptide geometry:
- N-Cα: 1.458 Å, Cα-C: 1.525 Å, C-N: 1.329 Å
- N-Cα-C: 111.2°, Cα-C-N: 116.2°, C-N-Cα: 121.7°
- ω = 180° (trans) unless specified

```python
def reconstruct_backbone(phi: List[float], psi: List[float],
                         omega: Optional[List[float]] = None) -> ndarray:
    """Returns (L, 3, 3) array: [residue, atom(N/Cα/C), xyz]."""
```

**Validation:** Extract experimental (φ,ψ) from 1UBQ crystal structure, reconstruct, compute backbone RMSD vs crystal coordinates. Should be < 0.5 Å. If higher, the geometry constants or rotation math have an error.

## Priority 7: End-to-end reconstruction test

Create `loops/reconstruct_test.py`:

For 10 test proteins:
1. Extract experimental (φ,ψ)
2. Identify loop regions
3. Replace loop (φ,ψ) with nearest canonical family centroid
4. Keep SS element (φ,ψ) at basin centers
5. Reconstruct via forward kinematics
6. Compute RMSD vs experimental backbone

This answers: how much accuracy do you lose by using canonical loop paths instead of actual conformations?

## Coding standards

- Python 3.10+, type hints on all public functions
- All angles in radians internally, degrees only for display and basin assignment
- Use `math.atan2(sin(a-b), cos(a-b))` for angular differences, NEVER raw subtraction
- Circular interpolation: interpolate sin/cos components, recover via atan2
- β-sheet basin: MUST handle ψ ±180° wrap as single condition
- Chain extraction: ALWAYS single chain, verify residue count
- Print progress for operations >10 seconds
- Each script runnable standalone: `python bps/validate_pdb.py`
- All reports as markdown in `output/`
- `pip install numpy scipy biopython matplotlib scikit-learn` — nothing else

## Execution order

Build and validate each priority before moving to the next. After each one, show me: what was built, what the validation numbers are, and whether they match the targets in CLAUDE.md. Stop and ask if any validation number is off by more than 20% from the target — that means there's a bug.
