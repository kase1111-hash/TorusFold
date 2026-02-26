# CONVENTIONS.md — TorusFold Codebase Conventions

This document records the canonical choices for all parameters and conventions
used across the TorusFold codebase. All scripts should follow these conventions.

## Superpotential W

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Primary W formula** | W = −√P(φ,ψ) | Used in `bps_process.py`, `bps_validate_controls.py` |
| **Alternative W formula** | W = −ln(P + ε) | Used in `bps/superpotential.py`, `generate_figures.py`, loop taxonomy scripts |
| **Density floor ε** | 1e-7 | Prevents log(0) in unpopulated regions |
| **Grid size** | 360×360 | 1 degree per bin |
| **Normalization** | min(W) = 0 | Absolute values arbitrary; only differences matter |
| **Smoothing** | σ = 0 (KDE bandwidth only) | `bps/superpotential.py` uses Gaussian KDE with Scott's rule; no additional smoothing. `bps_process.py` uses σ=1.5 bins on von Mises mixture. |

The three-level decomposition (Real < Markov < Shuffled) is transform-invariant:
it holds under both −√P and −ln(P + ε).

## Basin Definitions

**Wide definition (used in all BPS/L and fold-class analyses):**

```python
if -160 < phi_d < 0 and -120 < psi_d < 30:
    ss = 'alpha'
elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
    ss = 'beta'
else:
    ss = 'other'
```

**Narrow definition (used in `bps/superpotential.py` for detailed Ramachandran analysis):**

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
BPS/L = BPS / (L − 1)
```

Divides by L−1 (the number of sequential differences), not by L (the number of residues).

## Quality Thresholds

| Parameter | Default | Notes |
|-----------|---------|-------|
| pLDDT threshold | 70 | For AlphaFold structures. Use 85 for "high-quality" subset. |
| Min chain length | 50 | BPS/L undefined for very short chains |
| Resolution (PDB) | < 2.0 Å | X-ray diffraction only for validation |
| R-free (PDB) | < 0.25 | |

**Important:** Always report results at multiple pLDDT thresholds (70, 80, 85, 90) to demonstrate that conservation is not an artifact of aggressive filtering.

## Sign Convention

- φ and ψ are negated from gemmi raw output (positive = standard Ramachandran convention)
- Both φ and ψ are negated together (determined empirically per run in `bps_process.py`)
- BioPython returns angles in standard Ramachandran convention (no negation needed)

## Random Seeds

- Seed: 42 (used throughout for reproducibility)
- **Limitation:** Only one random realization of null models is tested. For publication, results should be verified with multiple seeds.

## Naming Conventions

- All angles in radians internally, degrees for display and basin assignment
- All PDB IDs uppercase (e.g., "1UBQ" not "1ubq")
- Chain IDs case-sensitive (chain "A" ≠ chain "a")
- Basin names lowercase: "alpha", "beta", "ppII", "alphaL", "other"
- Pair keys directional: "alpha->beta"
- BPS/L reported to 3 decimal places (0.202)
- CV reported as percentage (2.1%)
