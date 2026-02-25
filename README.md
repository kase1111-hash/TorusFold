# TorusFold

**BPS/L-Informed Protein Structure Prediction**

TorusFold is a proposed architecture that enhances protein structure prediction by embedding explicit backbone physics into the AlphaFold pipeline. It uses the Ramachandran superpotential W(φ,ψ) — the known energy landscape of the peptide backbone — as a frozen, non-trainable prior, and introduces BPS/L (Backbone Phi-Psi Roughness per unit Length) as a universal constraint on predicted structures.

> **Status:** Design document / White paper — not yet implemented.

---

## Motivation

AlphaFold2 and its successors learn an implicit mapping from sequence and evolutionary signal to 3D coordinates, but none of their information channels explicitly encode the one-dimensional physics of the backbone — the sequential path through dihedral angle space.

This gap produces characteristic failure modes:

- **Over-regularized disorder** — disordered regions predicted as spurious helices
- **Under-sampled loops** — poor conformations at secondary structure junctions
- **Weak de novo prediction** — low confidence when MSA signal is absent
- **No physics-based validation** — no internal metric for whether a backbone is biologically realistic at the local level

BPS/L addresses these failures. It is a universal, sequence-independent, experimentally validated invariant (≈ 0.20 across all kingdoms of life, 2.1% coefficient of variation) that quantifies backbone roughness. It works even without evolutionary signal.

## Architecture

TorusFold augments the standard Evoformer → Structure Module → Coordinate Output pipeline with three new components:

```
                        ┌─────────────────────┐
                        │   MSA Features       │
                        │   Pair Features      │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │     EVOFORMER        │
                        │   (unchanged)        │
                        └──────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
           ┌──────────────┐ ┌───────────┐ ┌──────────────────┐
           │  Structure   │ │  Dihedral │ │  W-Landscape     │
           │  Module      │ │  Head     │ │  Embedding       │
           │  (IPA → xyz) │ │  (→ φ,ψ) │ │  (fixed, frozen) │
           └──────┬───────┘ └─────┬─────┘ └────────┬─────────┘
                  │               │                 │
                  ▼               ▼                 ▼
           ┌──────────────────────────────────────────────┐
           │              LOSS COMPUTATION                 │
           │                                               │
           │  L_total = L_fape + L_aux                     │
           │          + λ_bps · L_bps                      │
           │          + λ_coh · L_coherence                 │
           │          + λ_trans · L_transition               │
           └───────────────────────────────────────────────┘
```

### Component A: Dihedral Prediction Head

Predicts (φ,ψ) dihedral angles directly from Evoformer single representations, parallel to the structure module's coordinate predictions. Uses `atan2` parameterization to handle circular topology without discontinuity at ±180°. A differentiable forward kinematics layer cross-checks dihedral predictions against 3D coordinate predictions, providing a novel confidence signal.

### Component B: W-Landscape Embedding

Embeds the fixed Ramachandran superpotential W(φ,ψ) = −ln p(φ,ψ) as a non-trainable buffer. W is precomputed from PDB backbone statistics and stored as a bicubic interpolation table. The `grid_sample` operation makes the lookup differentiable — gradients flow through W to adjust predicted angles, but the landscape itself remains frozen.

### Component C: BPS/L-Aware Loss Functions

Three loss terms target different levels of backbone organization:

| Loss | What it enforces | Key target |
|------|-----------------|------------|
| **L_bps** (Global BPS/L) | Overall backbone roughness matches biological range | Mixed α/β: 0.202 ± 0.004 |
| **L_coherence** (Intra-basin) | Consecutive residues in the same SS element have near-identical (φ,ψ) | 97% suppression of intra-basin fluctuations |
| **L_transition** (Transition density) | SS transitions produce correct energy gap magnitudes | α→β: \|ΔW\| ≈ 0.45, α→coil: ≈ 0.30 |

## Training Strategy

Losses are introduced in phases to avoid conflicting gradients:

| Phase | Epochs | Active losses |
|-------|--------|--------------|
| 1 | 1–50 | Standard AlphaFold losses only |
| 2 | 50–150 | + Global BPS/L constraint (λ_bps ramps 0 → 1.0) |
| 3 | 150–300 | + Coherence and transition losses |
| 4 | Fine-tuning | Reduce λ_bps, increase λ_coh; focus on difficult cases |

## Expected Impact

| Failure mode | Expected impact | Mechanism |
|-------------|----------------|-----------|
| Loop/coil prediction | **High** | Transition loss constrains saddle-point crossings of W |
| Orphan & de novo proteins | **High** | BPS/L is sequence-independent; works without MSA |
| Over-regularized disorder | **Medium** | BPS/L corrects spurious secondary structure assignment |
| Multi-domain linkers | **Medium** | Transition loss constrains linker Ramachandran traversal |
| Standard globular proteins | **Low** | Marginal dihedral improvement; AlphaFold already near-optimal |

## Evaluation

### New Metrics

| Metric | Target |
|--------|--------|
| BPS/L accuracy (predicted vs. ground truth) | Δ < 0.01 |
| Intra-basin MAD within SS elements | < 5° (helix), < 8° (sheet) |
| Transition fidelity (correct \|ΔW\| fraction) | > 0.85 |
| Dihedral-coordinate consistency | < 3° RMSD |

### Ablation Design

Four model variants: Baseline → + Dihedral Head → + W Embedding → + Full BPS Losses, evaluated on CASP15 targets, de novo proteins, loop accuracy subsets, and disorder benchmarks (DisProt).

### Success Criteria

1. Match or exceed baseline on standard CASP metrics (GDT-TS, lDDT, RMSD)
2. Statistically significant improvement on at least one failure mode
3. Predicted BPS/L distributions match experimental PDB values
4. Each component contributes in ablation

## Implementation Roadmap

| Phase | Duration | Milestone |
|-------|----------|-----------|
| 0 — Standalone validation | 2–4 weeks | BPS/L analysis of CASP14/15 predictions; differentiable W pipeline |
| 1 — Dihedral head prototype | 4–6 weeks | Dihedral head on frozen Evoformer (OpenFold); forward kinematics |
| 2 — Loss integration | 6–10 weeks | BPS losses in OpenFold training loop; hyperparameter sweep |
| 3 — Full-scale training | 8–12 weeks | Full PDB training; CASP15 & de novo evaluation; ablation |
| 4 — Publication & release | 4 weeks | Results paper; model weights & W module release |

**Total: 6–8 months from start to submission.**

## Broader Applications

Beyond accuracy improvements, TorusFold introduces an interpretable, physics-grounded internal metric for backbone quality:

- **Structure validation** — flag predictions with anomalous BPS/L, even when pLDDT is high
- **Protein design** — enforce BPS/L as a foldability constraint in computational design
- **Evolutionary analysis** — identify misfolded, disordered, or unusual proteins via BPS/L outliers
- **Training data curation** — score AlphaFold Database entries by BPS/L compliance

## References

- Branham, K. (2026). *Proteins as One-Dimensional Paths on the Ramachandran Torus.* In preparation.
- See [`spec.md`](spec.md) for the full technical specification.

## Author

**Kase Branham** · True North Research · February 2026

## License

See repository for license information.
