This document records the canonical choices for all parameters and conventions used across the TorusFold codebase. All scripts should follow these conventions.

## Superpotential W

| Parameter | Value | Notes |
|-----------|-------|-------|
| Primary W formula | W = −√P(φ,ψ) | Used in bps_process.py, null model tests, paper results |
| Alternative W formula | W = −ln(P + ε) | Used in bps/superpotential.py, generate_figures.py, loop taxonomy scripts |
| Density floor ε | 1e-7 | Prevents log(0) in unpopulated regions (−ln P pipeline only) |
| Grid size | 360×360 | 1 degree per bin |
| Normalization | min(W) = 0 | Absolute values arbitrary; only differences matter |
| Smoothing (−√P pipeline) | σ = 1.5 bins | Gaussian smoothing with torus wrapping. Used in bps_process.py and all null model tests. |
| Smoothing (−ln P pipeline) | σ = 0 (KDE bandwidth only) | bps/superpotential.py uses Gaussian KDE with Scott's rule; no additional smoothing. |

The four-level null model hierarchy (Real ≈ Markov < IID < Permuted) is transform-invariant: it holds under both −√P and −ln(P + ε).

## Von Mises Mixture Components

The −√P superpotential is built from a 10-component bivariate von Mises mixture (Table 1 of paper):

| # | Weight | μ_φ (°) | μ_ψ (°) | κ_φ | κ_ψ | ρ | Assignment |
|---|--------|---------|---------|-----|-----|---|------------|
| 1 | 0.35 | −63 | −43 | 12.0 | 10.0 | 2.0 | α-helix core |
| 2 | 0.05 | −60 | −27 | 8.0 | 6.0 | 1.0 | α-helix shoulder |
| 3 | 0.25 | −120 | 135 | 4.0 | 3.5 | −1.5 | β-sheet core |
| 4 | 0.05 | −140 | 155 | 5.0 | 4.0 | −1.0 | β-sheet shoulder |
| 5 | 0.12 | −75 | 150 | 8.0 | 5.0 | 0.5 | PPII |
| 6 | 0.05 | −95 | 150 | 3.0 | 4.0 | 0.0 | PPII shoulder |
| 7 | 0.03 | 57 | 40 | 6.0 | 6.0 | 1.5 | left-handed α |
| 8 | 0.03 | 60 | −130 | 5.0 | 4.0 | 0.0 | γ-turn |
| 9 | 0.01 | 75 | −65 | 5.0 | 5.0 | 0.0 | δ |
| 10 | 0.06 | 0 | 0 | 0.01 | 0.01 | 0.0 | uniform background |

These are hardcoded identically in bps_process.py, markov_pseudoproteome_test.py, permutation_null_test.py, and markov_transition_test.py.

## Basin Definitions

**Wide definition** (used in all BPS/L and fold-class analyses):

```
if -160 < phi_d < 0 and -120 < psi_d < 30:
    ss = 'alpha'
elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
    ss = 'beta'
else:
    ss = 'other'
```

**Narrow definition** (used in bps/superpotential.py for detailed Ramachandran analysis):

| Basin | φ range | ψ range |
|-------|---------|---------|
| α-helix | (−100°, −30°) | (−67°, −7°) |
| β-sheet | (−170°, −70°) | (90°, 180°) ∪ (−180°, −120°) |
| ppII | (−100°, −50°) | (120°, 180°) |
| αL | (30°, 90°) | (10°, 70°) |
| other | everything else | |

Results under the narrow definition should be tested as a sensitivity control.

## BPS/L Normalization

```
BPS = Σ_{i=1}^{L-1} |W(φ_{i+1}, ψ_{i+1}) − W(φ_i, ψ_i)|
BPS/L = BPS / L
```

Divides by L (the number of residues with valid dihedrals). This is the convention used by the primary pipeline (bps_process.py) and all null model tests.

**Known inconsistency:** Several analysis scripts use `np.mean(np.abs(np.diff(w)))` which divides by L−1 (number of differences) instead of L. For typical chain lengths (L > 50) the difference is < 2%, but this should be standardized.

- Scripts using /L: bps_process.py, bps_pdb_validate.py, bps_validate_controls.py, markov_pseudoproteome_test.py, permutation_null_test.py, markov_transition_test.py
- Scripts using /(L-1): generate_figures.py, within_foldclass_cv.py, designed_proteins.py, designed_seg_real.py, w_independence.py, higher_order_null.py, alphafold_pipeline.py

## W Formula and BPS/L Scale

**CRITICAL:** The absolute value of BPS/L depends entirely on which W formula is used:

| W Formula | Typical BPS/L | W Range | Pipeline |
|-----------|---------------|---------|----------|
| W = −√P (von Mises, σ=1.5) | ~0.175 (all data), ~0.202 (pLDDT ≥ 85) | [−1.29, −0.001] | bps_process.py, all null model tests |
| W = −ln(P + ε) (histogram) | ~2.2 | [5.48, 16.12] | generate_figures.py, within_foldclass_cv.py, bps/superpotential.py |

The −√P pipeline produces the headline results. The −ln(P + ε) pipeline produces BPS/L ≈ 2.2 — this is NOT a bug but a different gauge.

## Four-Level Null Model Hierarchy

**CRITICAL: These are the definitive results from the null model test suite (Feb 2026).**

| Level | BPS/L | What's preserved | Test script | N proteins |
|-------|-------|-----------------|-------------|------------|
| Real proteins | 0.175 | Everything | (database) | 124,475 |
| Markov transition (K=50) | 0.171 | Nearest-neighbor transitions | markov_transition_test.py | 150 (quick) |
| IID from P(φ,ψ) | 0.346 | Single-residue distribution (theoretical) | markov_pseudoproteome_test.py | 3,400 synth proteomes |
| Permutation shuffle | 0.516 | Exact angles, random order | permutation_null_test.py | 14,699 |

**Key findings:**
- Real ≈ Markov (gap = +1.2%): BPS/L is emergent from nearest-neighbor backbone transition dynamics. It does NOT require evolutionary selection beyond local correlations.
- Markov << IID (gap = +102%): Local correlations between consecutive residues dramatically reduce BPS/L vs independent sampling.
- IID << Permutation (gap = +49%): Real proteins concentrate angles more tightly in basins than the von Mises model predicts.
- DB cross-check r = 1.000 for all organisms in permutation test, confirming W functions are identical.

**The old "three-level decomposition" (Real < Markov < Shuffled) from 83 PDB structures is SUPERSEDED by these results.** The old Markov null had a sampling bug that produced BPS/L ≈ 0 (chains trapped in single bins). The corrected non-parametric Markov model shows Real ≈ Markov.

**Interpretation:** BPS/L ≈ 0.20 is a physical constant of the peptide bond's local transition geometry, not an evolutionary optimization. The universality across organisms reflects the shared backbone chemistry, not convergent evolution.

## Quality Thresholds

| Parameter | Default | Notes |
|-----------|---------|-------|
| pLDDT threshold | 70 | For AlphaFold structures. Use 85 for "high-quality" subset. |
| Min chain length (pipeline) | 30 | bps_process.py excludes L < 30 |
| Min chain length (null tests) | 10 | Permutation and Markov tests accept L ≥ 10 |
| Min chain length (IID test) | 5 | IID null accepts L ≥ 5 |
| Resolution (PDB) | < 2.0 Å | X-ray diffraction only for validation |
| R-free (PDB) | < 0.25 | |

**Important:** Always report results at multiple pLDDT thresholds (70, 80, 85, 90) to demonstrate that conservation is not an artifact of aggressive filtering.

## Sign Convention

- φ and ψ are negated from raw dihedral output (positive = standard Ramachandran convention)
- Both φ and ψ are negated together (determined empirically per run in bps_process.py)
- The null model tests (permutation_null_test.py, markov_transition_test.py) independently determine phi sign from cached CIF files using the same alpha-helix-occupancy test
- BioPython returns angles in standard Ramachandran convention (no negation needed)

## Random Seeds

- Seed: 42 (used throughout for reproducibility)
- Sanity check seed: 99999 (used in markov_pseudoproteome_test.py for pre-run validation)
- **Limitation:** Only one random realization of null models is tested per seed. For publication, results should be verified with multiple seeds (41, 42, 43 minimum).

## Naming Conventions

- All angles in radians internally, degrees for display and basin assignment
- All PDB IDs uppercase (e.g., "1UBQ" not "1ubq")
- Chain IDs case-sensitive (chain "A" ≠ chain "a")
- Basin names lowercase: "alpha", "beta", "ppII", "alphaL", "other"
- Pair keys directional: "alpha->beta"
- BPS/L reported to 3 decimal places (0.175) for all-data, (0.202) for pLDDT ≥ 85
- CV reported as percentage (7.8% all-data, 1.9% pLDDT ≥ 85)

## Interpolator Convention

The RegularGridInterpolator for W takes arguments as `np.column_stack([psi, phi])` — **psi first, phi second**. This convention is used consistently in:
- bps_process.py line 512: `W_interp(np.column_stack([psis, phis]))`
- All null model tests: `W_interp(np.column_stack([psis, phis]))`

The W grid is built with `meshgrid(angles, angles, indexing="ij")` so W_grid[phi_idx, psi_idx]. The interpolator transposes this: `RegularGridInterpolator((psi_grid, phi_grid), W.T)`.

## File Locations

| File | Purpose |
|------|---------|
| alphafold_bps_results/bps_database.db | Main SQLite database with all protein results |
| alphafold_cache/{organism}/AF-{uid}-F1-model_v*.cif | Cached AlphaFold CIF files |
| markov_test_results/ | Output directory for all null model tests |
| markov_test_results/markov_pseudoproteome_results.csv | IID null results |
| markov_test_results/permutation_test_results.csv | Permutation null results |
| markov_test_results/markov_transition_v2_results.csv | Markov transition results |
