# Scientific Impartiality Audit — TorusFold Python Scripts

**Date:** 2026-02-26
**Scope:** All 38 Python scripts in the repository
**Methodology:** Line-by-line code review checking for results-steering, confirmation bias, cherry-picking, circular reasoning, strawman nulls, and other practices that compromise scientific impartiality.

---

## CRITICAL ISSUES (7 findings that directly affect headline claims)

### 1. Figure 1C uses hardcoded values, not computed data
**File:** `generate_figures.py:528-529`
```python
labels = ['Real', 'Seg', 'M2', 'M1', 'Shuf']
values = [1.28, 1.80, 1.80, 1.80, 2.06]
```
A publication figure embeds pre-chosen numbers rather than computing from data. Three null models are given the identical value 1.80. There is no "M2" null model implemented anywhere in the codebase. The annotation "1.30x" at line 547 is inconsistent with the hardcoded values (1.80/1.28 = 1.41, not 1.30). **This is the single most concerning finding.**

### 2. W = -sqrt(p) in code vs W = -ln(p) in documentation
**Files:** `bps_process.py:293`, `bps_validate_controls.py:335`
```python
W = -np.sqrt(p)   # actual code
```
CLAUDE.md and the paper methodology describe `W = -ln(p + epsilon)`. The production pipeline uses `W = -sqrt(p)`, which is a fundamentally different mathematical object with compressed dynamic range. The W-robustness validation test also uses `-sqrt(p)`, meaning the "robustness test" doesn't even test the documented formula.

### 3. Silent outlier rejection biases PDB validation
**Files:** `bps_pdb_validate.py:747-749`, `bps_process.py:717`
```python
if not (0.02 < bps < 0.50):
    n_skipped_parse += 1
    continue
```
Proteins with BPS/L > 0.50 are silently excluded and counted as "parse/quality failures." Any protein that genuinely violates the universality claim is removed before statistics are computed.

### 4. Alpha-helix basin is 5.7x wider than documented
**Files:** `alphafold_pipeline.py:362-368`, `generate_figures.py:161-165`, `higher_order_null.py:113-119`, and 5+ other files
```python
# Documented in CLAUDE.md: alpha: phi in (-100, -30), psi in (-67, -7)
# Actual code in most scripts:
if -160 < phi < 0 and -120 < psi < 30:
    return 'a'
```
The alpha basin used in most analysis scripts covers ~5.7x the area of the documented definition. This inflates alpha-helix classification, changes fold-class assignments, and systematically affects the three-level decomposition and Markov null model. Only `diagnose_2rhe.py` and `bps_process.py` use the narrow (documented) definition.

### 5. DBSCAN eps search guarantees finding clusters
**Files:** `loop_taxonomy_v2.py:443`, `loop_taxonomy.py:489`, `cluster_loops.py:78-99`, `loops/taxonomy.py:267`
```python
for eps in eps_values:
    ...
    if n_clusters >= 2 and n_noise < len(loops) * 0.5:
        best_labels = labels
        break
```
The clustering pipeline searches eps values until it finds one producing at least 2 clusters with <50% noise. This is an optimization for finding clusters, not a test of whether clusters exist. **No file includes a null-model control** (e.g., applying the same search to shuffled data to measure the false-positive clustering rate).

### 6. Recursive subclustering inflates family count
**Files:** `loop_taxonomy_v2.py:475-517`, `loops/taxonomy.py:391-441`

Catch-all clusters (high variance) are recursively split up to 3 levels deep. Even noise points rejected by DBSCAN are recycled through re-clustering. This procedure will find apparently-tight sub-clusters in any dataset if you split enough times, as small subsets trivially have low CV. No multiple-testing correction is applied.

### 7. Classifier validates its own clustering labels
**File:** `loops/classifier.py:370-390`

The random forest trains on labels produced by the eps-searching DBSCAN, then good accuracy is interpreted as evidence that "the sequence-to-path mapping exists." This is circular: the clustering creates the labels, and the classifier validates them. Additionally, only tight families are used (noise and catch-all are excluded), inflating apparent accuracy.

---

## SIGNIFICANT CONCERNS (6 findings that weaken key claims)

### 8. Smoothing parameter optimized on headline metric
**File:** `bps/superpotential.py:9-12, 40`

`_SMOOTH_SIGMA = 0` was chosen because it maximizes the Markov/Real ratio — which *is* the three-level decomposition signal being reported. This is circular parameter tuning.

### 9. Fold-class verdict checks only one class
**File:** `bps_validate_controls.py:733-743`

Test 3 (fold-class independence) only checks "Mixed alpha/beta" CV for the pass/fail verdict, ignoring Alpha-rich and Beta-rich classes.

### 10. Within-fold-class CV test uses 1.5x threshold
**File:** `within_foldclass_cv.py:478-493`

Declares conservation "genuine" if within-class CV is no more than 1.5x the overall CV. The script pre-emptively dismisses high CVs in small fold classes as "driven by small sample sizes, not biological variation."

### 11. Strawman steric null model in polymer experiment
**File:** `polymer_null_experiment.py:342-409`

The "make-or-break" experiment uses a minimal steric model: backbone-only (no side chains), 6-residue lookback, 2.0 A hard-sphere clash. Real steric interactions involve side chains, electrostatics, and longer-range contacts. Designed to be easy to beat.

### 12. Hardening tests are defensive, not investigative
**File:** `hardening_tests.py`

Title says "close the remaining attack vectors." Verdict thresholds consistently favor the desired conclusion (2 of 3 branches are favorable in each test). The PDB experimental test (the strongest one) is optional.

### 13. KS test uses synthetic normal, not real AlphaFold data
**File:** `bps_pdb_validate.py:815-820`

Compares PDB BPS/L against a synthetic normal distribution, not actual AlphaFold values. A normal approximation is easier to match against than the real (potentially skewed) distribution.

---

## MINOR CONCERNS (11 findings — individually justifiable but collectively biasing)

| # | Finding | File(s) |
|---|---------|---------|
| 14 | pLDDT >= 85 threshold reduces reported CV | `bps/bps_process.py:45` |
| 15 | Minimum chain length of 50 excludes short proteins | Multiple files |
| 16 | NMR exclusion in filtered PDB stats (shown first) | `bps/validate_pdb.py:733` |
| 17 | Resolution 1.8A instead of documented 2.0A | `bps/validate_pdb.py:106` |
| 18 | Sign convention inconsistency (negate vs not) | `alphafold_pipeline.py` vs `bps_process.py` |
| 19 | BPS/L normalization: L-1 vs L | `alphafold_pipeline.py` vs `bps/superpotential.py` |
| 20 | Epsilon: CLAUDE.md says 1e-8; code uses 1e-7 or max(p)*1e-6 | Multiple files |
| 21 | Boxplots suppress outliers (showfliers=False) | `generate_figures.py:783` |
| 22 | Designed proteins reference values ~6x off from CLAUDE.md | `designed_proteins.py:420-424` |
| 23 | Fixed random seed (42) — only one null realization tested | Multiple files |
| 24 | Forced pLDDT >= 85 override for fold-class analysis | `bps_statistics.py:787` |

---

## CLEAN FILES (no concerns)

- `bps/extract.py` — Properly guards against multi-chain bug
- `bps/compile_results.py` — No filtering, reports all data
- `bps/bps_download.py` — Broad organism selection
- `bps_download.py` (root) — Pure data acquisition
- `bps_plaxco.py` — Honest three-tier decision framework with failure path
- `torus_distance_control.py` — Well-designed falsification test
- `subsample_cluster_stability.py` — Uses ARI, has honest "POOR" verdict
- `loops/forward_kinematics.py` — Utility module
- `bps/__init__.py`, `loops/__init__.py` — Empty

---

## SYSTEMIC PATTERNS

### A. Multiple incompatible pipeline implementations
At least 3 different W formulas (-ln(p), -sqrt(p), histogram-based), 2 sign conventions, 2 normalization schemes (L vs L-1), 3+ basin definitions, and 3+ epsilon values. Results from different scripts are not directly comparable.

### B. Cumulative filtering in one direction
Each individual filter (pLDDT >= 85, length >= 50, NMR exclusion, resolution < 1.8A, BPS/L in [0.02, 0.50]) is individually justifiable. But they all reduce variance and make BPS/L look more universal. A robust analysis would show results at multiple filter levels.

### C. No null-model validation of clustering
The loop taxonomy claims (~30 canonical families) rest entirely on DBSCAN with searched parameters and recursive subclustering, with zero null-model controls.

---

## RECOMMENDATIONS

1. **Fix Figure 1C** — Compute all bar chart values from data. Remove the nonexistent M2 model or implement it. Fix the inconsistent 1.30x annotation.
2. **Reconcile W formula** — Either change the paper to describe -sqrt(p) or change the code to use -ln(p + eps). Document the choice and its impact.
3. **Add null-model controls for clustering** — Apply the same eps-search + recursive subclustering pipeline to shuffled loop paths. Report how many "tight families" emerge by chance.
4. **Standardize basin definitions** — Pick one alpha-helix basin definition, document it, and use it everywhere. Quantify the impact of the choice.
5. **Remove or report BPS/L range filter** — The [0.02, 0.50] filter should either be removed or its excluded proteins reported as a separate category (not lumped with parse failures).
6. **Show results at multiple filter levels** — Present BPS/L statistics at pLDDT >= 70, 80, 85, and 90 to demonstrate that universality is not an artifact of aggressive filtering.
7. **Benchmark clustering tightness threshold** — Compute expected CV for random subsets of loops to establish whether 30% is a meaningful threshold.
8. **Use formal hypothesis tests** — Replace arbitrary ratio thresholds in verdict logic with permutation tests or bootstrap hypothesis tests with p-values.

---

## VERDICT SUMMARY BY FILE

| File | Verdict |
|------|---------|
| `generate_figures.py` | **CRITICAL** |
| `bps_process.py` | **CRITICAL** |
| `bps_validate_controls.py` | **CRITICAL** |
| `bps_pdb_validate.py` | **CRITICAL** |
| `alphafold_pipeline.py` | **CRITICAL** |
| `loop_taxonomy_v2.py` | **CRITICAL** |
| `loop_taxonomy.py` | **CRITICAL** |
| `cluster_loops.py` | **CRITICAL** |
| `loops/taxonomy.py` | **CRITICAL** |
| `loops/classifier.py` | **CRITICAL** |
| `bps/superpotential.py` | SIGNIFICANT |
| `within_foldclass_cv.py` | SIGNIFICANT |
| `polymer_null_experiment.py` | SIGNIFICANT |
| `hardening_tests.py` | SIGNIFICANT |
| `higher_order_null.py` | MINOR |
| `per_segment_null.py` | MINOR |
| `bps_statistics.py` | MINOR |
| `w_independence.py` | MINOR |
| `bps/bps_process.py` | MINOR |
| `bps/validate_pdb.py` | MINOR |
| `bps_figures.py` | MINOR |
| `designed_proteins.py` | SIGNIFICANT |
| `designed_seg_real.py` | MINOR |
| `compile_results.py` | MINOR |
| `build_superpotential.py` | MINOR |
| `diagnose_foldclass.py` | MINOR |
| `diagnose_2rhe.py` | MINOR |
| `length_binned_torus.py` | MINOR |
| `bps/extract.py` | CLEAN |
| `bps/compile_results.py` | CLEAN |
| `bps/bps_download.py` | CLEAN |
| `bps_download.py` | CLEAN |
| `bps_plaxco.py` | CLEAN |
| `torus_distance_control.py` | CLEAN |
| `subsample_cluster_stability.py` | CLEAN |
| `loops/forward_kinematics.py` | CLEAN |
| `loops/reconstruct_test.py` | MODERATE |
