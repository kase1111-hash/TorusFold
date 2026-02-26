# TorusFold: Three Layers of Sequential Organization in Protein Backbones

A proteome-scale analysis of backbone torsion-angle organization on the Ramachandran torus, decomposed into three multiplicatively independent layers.

## What This Is

Protein backbones trace paths through (φ, ψ) torsion space — the Ramachandran torus T². We define a family of superpotentials W(φ, ψ) = f(P(φ, ψ)) on this torus and measure how smoothly real proteins traverse it compared to null models. This reveals three separable layers of sequential organization:

**Layer 1 — Geometric structural coherence (~11%).** Real proteins are smoother on T² than expected from within-basin randomization. This decomposes into inter-element structural fingerprinting (~6%, each helix/strand has its own torsion centroid) and within-element sequential ordering (~4%, consecutive-pair smoothness within elements). Both are absent in random-coil polymers and present in experimental PDB structures.

**Layer 2 — Density-peak alignment (~17%).** The superpotential amplifies the geometric signal because real proteins concentrate near Ramachandran density peaks. Transform-invariant across seven functional forms.

**Layer 3 — Transition architecture (~14%).** Basin-to-basin transitions follow non-random patterns beyond first-order Markov statistics. Metric-independent.

These layers are multiplicatively independent (interaction term I = 1.004) and together produce ~1.6× roughness suppression relative to random sequences. The per-residue BPS/L constant of ~0.200 is conserved across 29 organisms spanning ~4 billion years of divergence (CV = 1.8%).

## Key Results

| Finding | Value |
|---|---|
| Cross-organism BPS/L conservation | CV = 1.8% across 29 organisms |
| Torus Seg/Real (AlphaFold) | 1.106× [1.059, 1.159] |
| Torus Seg/Real (PDB experimental) | 1.069× [1.047, 1.092] |
| Steric-coupled polymer null | 0.997× (no smoothing) |
| Neighbor-conditional model | 1.087× (84% of effect) |
| Transform invariance (7 forms) | r > 0.96, ratios within 5% |
| W independence (9 constructions) | CV = 3% |
| Natural vs designed proteins | Δ = 0.01 |
| Layer independence | I = 1.004 across 5 length bins |
| Markov model prediction | 0.200 predicted vs 0.200 ± 0.005 observed |

## Installation

```bash
pip install gemmi numpy --break-system-packages
```

Optional (for figures): `pip install matplotlib scipy hdbscan`

## Data Setup

Download AlphaFold proteome predictions and organize as:

```
alphafold_cache/
  human/
    AF-A0A0A0MRZ7-F1-model_v4.cif
    ...
  ecoli/
    ...
```

PDB structures for validation go in `pdb_cache/`.

## Pipeline

Run in order (step 1 must be first, steps 2–10 can run in any order after):

```bash
# 1. Build superpotential (MUST BE FIRST, ~1-2 hours)
python build_superpotential.py --data alphafold_cache --output results

# 2. Torus distance control (~15-30 min)
python torus_distance_control.py --data alphafold_cache --sample 200 --output results

# 3. Length-binned analysis (~20-40 min)
python length_binned_torus.py --data alphafold_cache --sample 500 --output results

# 4. Polymer null experiment (~30 min)
python polymer_null_experiment.py --data alphafold_cache --output results

# 5. Per-segment permutation null (~10 min)
python per_segment_null.py --data alphafold_cache --output results --sample 200

# 6. Pre-submission hardening tests (~15 min)
python hardening_tests.py --data alphafold_cache --pdb-dir pdb_cache --output results

# 7. Within-fold-class CV (~2-4 hours, full dataset)
python within_foldclass_cv.py --data alphafold_cache --sample 0 --output results

# 8. W independence (~1-2 hours)
python w_independence.py --data alphafold_cache --output results

# 9. Cluster stability (~6-12 hours, run overnight)
python subsample_cluster_stability.py --data alphafold_cache --sample 0 --output results

# 10. Figures (after 2-9 complete)
python generate_figures.py --data alphafold_cache --sample 200 --output results
```

Windows users: batch files (`run_*.bat`) are provided for each step.

All scripts use deterministic random seeds for reproducibility.

## Script Descriptions

| Script | What it does |
|---|---|
| `build_superpotential.py` | Builds W(φ,ψ) from all structures. Outputs `superpotential_W.npz`. |
| `torus_distance_control.py` | Tier 0 falsification: four distance metrics on identical null models. Separates geometric signal from W-curvature artifacts. |
| `length_binned_torus.py` | Bins proteins by length (50–1090 residues), confirms effect is length-independent. |
| `polymer_null_experiment.py` | **The key experiment.** Four polymer models (IID, steric-coupled, neighbor-conditional, real). Proves Layer 1 requires secondary structure. |
| `per_segment_null.py` | Decomposes Layer 1 into within-element ordering vs inter-element heterogeneity. |
| `hardening_tests.py` | Three pre-submission controls: local-window null, PDB torus Seg/Real, steric rejection diagnostics. |
| `within_foldclass_cv.py` | Cross-organism and within-fold-class coefficient of variation. Full dataset analysis. |
| `w_independence.py` | Nine independent W constructions (train/test splits, grid resolutions). |
| `subsample_cluster_stability.py` | Loop taxonomy: HDBSCAN clustering with 5× subsample stability. |
| `generate_figures.py` | Publication figures. |

## Null Model Hierarchy

From least to most constrained:

1. **Shuffled** — random permutation of full (φ,ψ) sequence
2. **First-order Markov (M1)** — preserves single-step basin transition probabilities
3. **Second-order Markov (M2)** — preserves bigram transitions
4. **Segment-preserving (full pool)** — preserves SS sequence, draws from protein's own basin pool
5. **Per-segment permutation** — permutes within each contiguous SS segment only
6. **Real backbone** — observed structure

## Polymer Null Models

From no physics to full structure:

| Model | Physics | Seg/Real |
|---|---|---|
| IID Ramachandran | None | 0.998× |
| Steric-coupled | Bond geometry + 2.0Å clash rejection | 0.997× |
| Neighbor-conditional | Empirical P(φᵢ₊₁,ψᵢ₊₁ \| φᵢ,ψᵢ) | 1.087× |
| Real proteins | Full folded structure | 1.104× |

The steric model builds 3D backbones atom-by-atom with realistic bond lengths, angles, and trans ω. Its Seg/Real of 0.997× proves that peptide bond stereochemistry alone produces zero sequential smoothing. The effect requires secondary structure.

## Outputs

All results go to `results/`. Key files:

- `superpotential_W.npz` — W grid, histogram, metadata
- `polymer_null_report.md` — polymer null experiment results
- `per_segment_null_report.md` — Layer 1 sub-decomposition
- `hardening_tests_report.md` — PDB torus validation, rejection diagnostics
- `*_report.md` — markdown reports from each analysis script

## Citation

Paper in preparation. If you use this code, please cite:

> TorusFold: Three Layers of Sequential Organization in Protein Backbones. Kase Branham. 2026.

## License

MIT
