# BPS/L-Informed Protein Structure Prediction: Architecture Specification

**Kase Branham · True North Research · February 2026**

**Status:** Design document / White paper

---

## 1. The Gap in Current Structure Prediction

AlphaFold2 and its successors solve the structure prediction problem by learning an implicit mapping from sequence + evolutionary signal to 3D coordinates. The architecture works through four information channels: evolutionary couplings extracted from multiple sequence alignments (MSA), pairwise attention over sequence positions, geometric constraints enforced through invariant point attention (IPA), and learned internal potentials that emerge during training.

What none of these channels encode explicitly is the one-dimensional physics of the backbone itself — the sequential path through dihedral angle space.

This matters because AlphaFold's failure modes are concentrated exactly where backbone physics is most constrained. Disordered regions are over-regularized into spurious helices. Loop conformations are under-sampled. De novo proteins with no evolutionary signal (sparse or absent MSAs) produce low-confidence predictions. And the model has no internal metric for whether a predicted backbone is *biologically realistic* at the local level — only whether it satisfies learned geometric priors derived from training data.

BPS/L provides that metric. It is a universal, sequence-independent, experimentally validated invariant of backbone architecture. Incorporating it into structure prediction means giving the network explicit access to the one-dimensional physics that AlphaFold currently learns only implicitly — and imperfectly.

---

## 2. What BPS/L Provides That AlphaFold Lacks

### 2.1 A fixed energy landscape as prior knowledge

The superpotential W(φ,ψ) = −ln p(φ,ψ) on the Ramachandran torus is a *known* quantity. It does not need to be learned. It is determined by backbone stereochemistry and has been measured to high precision from hundreds of thousands of experimental structures. Every residue in every protein navigates this same landscape.

AlphaFold has no explicit representation of this landscape. Its Evoformer and structure module implicitly learn something like it from training data, but that learned representation is entangled with sequence features, MSA statistics, and 3D coordinate losses. It cannot be inspected, validated, or constrained independently.

Providing W as a fixed, non-trainable input gives the network direct access to the physics of the peptide bond. This is analogous to how physics-informed neural networks in fluid dynamics embed the Navier-Stokes equations rather than learning them from data.

### 2.2 A universal regularizer independent of evolutionary signal

BPS/L ≈ 0.20 holds across all kingdoms of life with 2.1% coefficient of variation, validated in both AlphaFold predictions (66,549 proteins, 22 organisms) and experimental crystal structures (162 high-resolution PDB entries, Δ = 2.4%). It is independent of contact order (r = −0.05), helix content (r = 0.03), and sheet content (r = −0.11).

This means BPS/L can constrain predictions even when the evolutionary signal is weak or absent. For orphan proteins, de novo designs, and intrinsically disordered regions transitioning to ordered states, MSA-based features provide little guidance. BPS/L provides a backbone-level constraint that does not require homologs.

### 2.3 Quantified intra-basin coherence

The Markov null model demonstrates that real proteins suppress intra-basin conformational fluctuations by 97% compared to random within-basin sampling. This is the quantitative signature of secondary structure: consecutive residues in a helix occupy not merely the α-basin of the Ramachandran torus, but a thin filament within it, centered at (φ ≈ −63°, ψ ≈ −43°) with minimal variance.

No current architecture explicitly enforces this coherence. AlphaFold learns it statistically, but the learning is indirect — mediated through 3D coordinate losses and recycling iterations. A direct intra-basin coherence term would enforce the defining physical characteristic of secondary structure at the dihedral level.

---

## 3. Architecture Design

### 3.1 Overview

The proposed architecture augments the standard Evoformer → Structure Module → Coordinate Output pipeline with three new components:

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

### 3.2 Component A: Dihedral Prediction Head

**Purpose:** Predict (φ,ψ) directly from Evoformer representations, parallel to the structure module's coordinate predictions.

**Architecture:**
```
Evoformer single representation s_i  (per-residue, dim d_s)
    → Linear(d_s, 256) → ReLU → Linear(256, 128) → ReLU
    → Two output branches:
        → Linear(128, 2) → atan2 normalization → φ_i  (circular output)
        → Linear(128, 2) → atan2 normalization → ψ_i  (circular output)
```

The atan2 parameterization (predicting sin and cos components, then computing the angle) handles the circular topology of dihedral angles correctly. This avoids the discontinuity at ±180° that would plague a naive linear output.

**Key design choice:** The dihedral head operates on the single representation (per-residue features) rather than the pair representation. Dihedral angles are local quantities — they depend on four consecutive backbone atoms. The single representation already encodes sufficient local context after Evoformer processing.

**Consistency enforcement:** The predicted (φ,ψ) and the predicted 3D coordinates must be geometrically consistent. A differentiable forward kinematics layer converts (φ,ψ) predictions to Cartesian coordinates, and a consistency loss penalizes divergence between coordinate-derived and directly-predicted dihedrals. During inference, this cross-check flags regions where the two prediction pathways disagree — a novel confidence signal.

### 3.3 Component B: W-Landscape Embedding

**Purpose:** Provide the network with the fixed Ramachandran energy landscape as non-trainable prior knowledge.

**Implementation:**
```python
class WLandscapeEmbedding(nn.Module):
    """Fixed superpotential W(φ,ψ) on the Ramachandran torus.
    
    W is precomputed from PDB backbone statistics and stored
    as a bicubic interpolation table. It is NOT trainable.
    The gradient ∇W is also precomputed for the transition loss.
    """
    def __init__(self, grid_resolution=360):
        super().__init__()
        # Load precomputed W and ∇W (from PDB statistics)
        W_grid = load_ramachandran_pmf(grid_resolution)     # [360, 360]
        dW_dphi, dW_dpsi = np.gradient(W_grid)              # gradients
        
        # Register as non-trainable buffers
        self.register_buffer('W', torch.tensor(W_grid, dtype=torch.float32))
        self.register_buffer('dW_dphi', torch.tensor(dW_dphi, dtype=torch.float32))
        self.register_buffer('dW_dpsi', torch.tensor(dW_dpsi, dtype=torch.float32))
    
    def forward(self, phi, psi):
        """Look up W values and gradients for predicted dihedrals.
        
        Args:
            phi: [batch, L] predicted phi angles in radians
            psi: [batch, L] predicted psi angles in radians
        Returns:
            W_values:  [batch, L] superpotential at each residue
            grad_W:    [batch, L, 2] gradient of W at each residue
        """
        # Convert angles to grid indices (bilinear interpolation)
        phi_idx = (phi + π) / (2π) * self.W.shape[0]
        psi_idx = (psi + π) / (2π) * self.W.shape[1]
        
        # Differentiable grid sample (enables backprop through W)
        W_values = grid_sample_2d(self.W, phi_idx, psi_idx)
        grad_phi = grid_sample_2d(self.dW_dphi, phi_idx, psi_idx)
        grad_psi = grid_sample_2d(self.dW_dpsi, phi_idx, psi_idx)
        
        return W_values, torch.stack([grad_phi, grad_psi], dim=-1)
    
    def bps_per_residue(self, W_values):
        """Compute |ΔW| between consecutive residues."""
        return torch.abs(W_values[:, 1:] - W_values[:, :-1])
```

**Critical property:** The grid_sample operation is differentiable with respect to φ and ψ. This means the BPS/L loss backpropagates gradients through the W landscape to the dihedral prediction head, directly adjusting predicted angles to satisfy backbone roughness constraints. The landscape itself remains frozen — only the predicted positions on it are optimized.

### 3.4 Component C: BPS/L-Aware Loss Functions

Three loss terms, each targeting a different level of the backbone organization hierarchy identified in the paper.

#### Loss 1: Global BPS/L constraint (λ_bps)

```python
def bps_loss(W_values, fold_class_target=0.202, fold_class_std=0.004):
    """Penalize deviation from expected BPS/L for the protein's fold class.
    
    For mixed α/β: target = 0.202 ± 0.004
    For alpha-rich: target = 0.169 ± 0.051
    For beta-rich:  target = 0.092 ± 0.044
    """
    L = W_values.shape[1]
    delta_W = torch.abs(W_values[:, 1:] - W_values[:, :-1])
    bps_l = delta_W.sum(dim=1) / L                          # [batch]
    
    # Gaussian penalty centered on fold-class target
    return ((bps_l - fold_class_target) / fold_class_std).pow(2).mean()
```

**When this fires:** This loss activates when the overall backbone roughness deviates from the biological range. A prediction that is too smooth (BPS/L ≪ 0.20) likely has over-regularized loops or missing secondary structure transitions. A prediction that is too rough (BPS/L ≫ 0.20) likely has noisy dihedral predictions or spurious basin crossings.

**Fold-class conditioning:** The target BPS/L depends on the predicted fold class. During early training, use the universal target (0.202). As the model learns to classify folds, condition on the predicted class. This avoids penalizing legitimately low-BPS/L beta-rich proteins.

#### Loss 2: Intra-basin coherence (λ_coh)

```python
def coherence_loss(phi, psi, ss_assignment):
    """Penalize intra-basin variance within predicted secondary structure elements.
    
    Enforces the 97% suppression of intra-basin roughness observed in
    real proteins. Consecutive residues assigned to the same SS type
    should have nearly identical (φ,ψ).
    """
    loss = 0.0
    
    # Basin centers (radians)
    basin_centers = {
        'H': (torch.tensor(-1.10), torch.tensor(-0.75)),   # α-helix
        'E': (torch.tensor(-2.09), torch.tensor(2.27)),     # β-sheet
    }
    
    for ss_type, (phi_c, psi_c) in basin_centers.items():
        mask = (ss_assignment == ss_type)                    # [batch, L]
        if mask.sum() == 0:
            continue
        
        # Circular distance from basin center
        d_phi = circular_distance(phi[mask], phi_c)
        d_psi = circular_distance(psi[mask], psi_c)
        
        # Penalize spread within basin (target: near-zero variance)
        loss += (d_phi.pow(2) + d_psi.pow(2)).mean()
    
    return loss
```

**What this encodes:** The Markov null model showed that random within-basin placement yields BPS/L = 0.30, while real proteins achieve 0.20 through 97% suppression of intra-basin fluctuations. This loss directly enforces that suppression. It does not constrain *which* basin a residue occupies — only that consecutive residues in the same secondary structure element occupy *nearly the same point* within their basin.

**Interaction with existing losses:** This complements AlphaFold's FAPE loss (which enforces 3D distances) and the auxiliary distogram loss (which enforces inter-residue distance distributions). The coherence loss operates in a completely different space — dihedral angles — and constrains a property (intra-basin variance) that 3D losses address only indirectly.

#### Loss 3: Transition density (λ_trans)

```python
def transition_loss(W_values, delta_W, ss_transitions):
    """Ensure transitions between secondary structure elements produce
    appropriate |ΔW| magnitudes — neither too sharp nor too gradual.
    
    Uses the empirical transition matrix to set expected |ΔW| for each
    basin pair (e.g., α→β should produce |ΔW| ≈ 0.45, α→coil ≈ 0.30).
    """
    # Empirical inter-basin energy gaps (from transition matrix analysis)
    expected_gaps = {
        ('H', 'E'): 0.45,    # helix → sheet: largest gap
        ('H', 'C'): 0.30,    # helix → coil
        ('E', 'C'): 0.15,    # sheet → coil: smallest (basins nearly adjacent)
        ('H', 'L'): 0.52,    # helix → left-handed helix: full torus traversal
    }
    
    loss = 0.0
    for (ss_from, ss_to), expected_dW in expected_gaps.items():
        # Find residue pairs where SS type changes from ss_from to ss_to
        mask = find_transitions(ss_transitions, ss_from, ss_to)
        if mask.sum() == 0:
            continue
        
        observed_dW = delta_W[mask]
        loss += (observed_dW - expected_dW).pow(2).mean()
    
    return loss
```

**What this encodes:** The first-principles derivation showed that BPS/L = 0.2005 is determined by the weighted sum of inter-basin energy differences. This loss ensures that individual transitions produce the correct energy gap magnitudes, not just the correct average. A helix-to-sheet transition should cross the full saddle point of W; a sheet-to-coil transition should be a gentle step within the extended region of the torus.

---

## 4. Training Strategy

### 4.1 Phased introduction

The BPS/L losses should not be active from the start of training. The network needs to first learn basic protein geometry before backbone roughness constraints become meaningful.

```
Phase 1 (epochs 1–50):     Standard AlphaFold losses only
                            L = L_fape + L_aux + L_distogram
                            Dihedral head trains but BPS losses inactive

Phase 2 (epochs 50–150):   Introduce BPS/L global constraint
                            L = L_fape + L_aux + λ_bps · L_bps
                            λ_bps ramps linearly from 0 to 1.0

Phase 3 (epochs 150–300):  Add coherence and transition losses
                            L = L_fape + L_aux + λ_bps · L_bps 
                            + λ_coh · L_coherence + λ_trans · L_transition
                            Full loss landscape active

Phase 4 (fine-tuning):     Reduce λ_bps, increase λ_coh
                            Shift emphasis from global to local constraints
                            Target: difficult cases (sparse MSA, de novo)
```

**Rationale:** Global BPS/L is a soft constraint — it guides overall backbone character without dictating local geometry. Coherence and transition losses are harder constraints that require the model to have already learned approximate secondary structure. Introducing them too early would create conflicting gradients with FAPE.

### 4.2 Loss weighting

| Loss term | λ range | Rationale |
|-----------|---------|-----------|
| L_fape | 1.0 (fixed) | Primary structural loss, unchanged |
| L_bps | 0.01 – 0.1 | Soft global regularizer; should not dominate |
| L_coherence | 0.05 – 0.5 | Stronger; directly enforces secondary structure quality |
| L_transition | 0.01 – 0.05 | Weakest; ensures transition physics without over-constraining loop geometry |

These weights should be tuned by monitoring BPS/L statistics on the validation set. The diagnostic: if validation BPS/L matches the universal constant (0.202 ± 0.004) but FAPE is degraded, reduce λ. If FAPE is fine but BPS/L is off, increase λ.

### 4.3 Fold-class conditioning schedule

Early training uses the universal BPS/L target (0.202). Once the model develops fold-class discrimination (measurable by secondary structure prediction accuracy > 80%), switch to fold-class-conditioned targets:

```python
def get_bps_target(predicted_ss_fractions):
    """Adaptive BPS/L target based on predicted fold class."""
    helix_frac, sheet_frac = predicted_ss_fractions
    
    if sheet_frac > 0.40:                              # beta-rich
        return 0.092, 0.044
    elif helix_frac > 0.40 and sheet_frac < 0.10:     # alpha-rich
        return 0.169, 0.051
    else:                                                # mixed α/β
        return 0.202, 0.004
```

---

## 5. Predicted Impact by Failure Mode

### 5.1 Loop and coil prediction (HIGH impact)

AlphaFold's weakest predictions cluster in loop regions connecting secondary structure elements. These are precisely the regions where the backbone crosses saddle points of W — the mountain passes of the Ramachandran landscape.

The transition loss (L_transition) directly constrains these crossings. By specifying the expected |ΔW| magnitude for each basin-pair transition, the model receives explicit guidance on how a helix-to-sheet loop should navigate the energy landscape. Current AlphaFold receives no such guidance — loop geometry is constrained only by FAPE distances to nearby atoms.

**Expected improvement:** Loop RMSD reduction, particularly for loops connecting different secondary structure types (α→β junctions, which cross the largest W barriers).

### 5.2 Orphan and de novo proteins (HIGH impact)

For proteins with no detectable homologs (no MSA) or designed proteins never seen in evolution, AlphaFold's primary information channel (evolutionary couplings) is absent. The model falls back on learned structural priors, which produce generic, low-confidence predictions.

BPS/L is sequence-independent. It constrains backbone architecture regardless of evolutionary history. For a de novo protein with a designed helix-sheet-helix motif, the BPS/L constraint tells the model: "this backbone should have roughness ~0.20, with coherent intra-basin dihedral angles and appropriate transition magnitudes at the element boundaries." No homologs required.

**Expected improvement:** Higher-confidence predictions for designed proteins and orphan sequences, particularly in secondary structure element placement and relative orientation.

### 5.3 Over-regularized disorder (MEDIUM impact)

AlphaFold tends to assign helical structure to regions that are intrinsically disordered in vivo. This is partly because its training data is biased toward well-folded structures, and partly because the FAPE loss penalizes disorder (disordered regions have high FAPE because they lack a single target conformation).

BPS/L provides a complementary signal. Genuinely disordered regions have low BPS/L (< 0.10) because they don't cross Ramachandran basin boundaries in a structured way. If the model predicts high BPS/L for a region that the MSA signal suggests is disordered, the BPS loss creates a corrective gradient toward lower roughness — preventing spurious secondary structure assignment.

**Expected improvement:** Better discrimination between ordered and disordered regions, fewer spurious helices in predicted IDRs.

### 5.4 Multi-domain proteins (MEDIUM impact)

Current AlphaFold struggles with inter-domain geometry — relative orientations of domains are often wrong even when individual domains are well-predicted. BPS/L doesn't directly constrain inter-domain geometry (it's a local, sequential metric), but it constrains the *linker regions* between domains.

Inter-domain linkers are loops that connect structured domains. The transition loss constrains these linkers to have appropriate Ramachandran traversal patterns. Combined with the dihedral prediction head's consistency check against 3D coordinates, this provides additional information for linker conformation prediction.

**Expected improvement:** Modest improvement in inter-domain linker geometry, particularly for proteins with structured (non-disordered) linkers.

### 5.5 Standard globular proteins (LOW impact)

For well-studied globular proteins with deep MSAs, AlphaFold already achieves near-experimental accuracy. BPS/L losses will act as mild regularizers that marginally improve backbone dihedral angle accuracy without significantly affecting overall structure quality.

**Expected improvement:** Marginal improvement in backbone dihedral MAE (mean absolute error); negligible change in GDT or RMSD.

---

## 6. The Differentiable W Pipeline

The critical engineering requirement is that the entire BPS/L computation must be differentiable. Gradients must flow from the BPS losses back through the W landscape lookup to the dihedral prediction head.

```
Dihedral Head output: φ_i, ψ_i  (differentiable)
         │
         ▼
W Lookup: W(φ_i, ψ_i)  (differentiable via grid_sample)
         │
         ▼
|ΔW|:   |W(φ_{i+1}, ψ_{i+1}) - W(φ_i, ψ_i)|  (differentiable except at 0)
         │
         ▼
BPS/L:  (1/L) Σ |ΔW_i|  (differentiable)
         │
         ▼
Loss:   (BPS/L - target)²  (differentiable)
```

The only non-differentiable point is |x| at x = 0. In practice this is handled by the standard smooth approximation |x| ≈ √(x² + ε) with ε = 10⁻⁶, or by using the Huber loss formulation.

The grid_sample operation (bilinear interpolation on the W grid) is natively differentiable in PyTorch. The W grid itself has zero gradient (it's a frozen buffer), but the *index* into the grid — which depends on φ and ψ — carries gradients. This means the loss "tells" the dihedral head: "move φ and ψ in the direction that brings BPS/L closer to the target."

---

## 7. Evaluation Metrics

### 7.1 New metrics enabled by BPS/L integration

| Metric | Definition | Target |
|--------|-----------|--------|
| BPS/L accuracy | Mean BPS/L of predicted structures vs. ground truth | Δ < 0.01 |
| Intra-basin MAD | Mean absolute dihedral deviation within SS elements | < 5° (helix), < 8° (sheet) |
| Transition fidelity | Fraction of SS transitions with correct |ΔW| (within 20%) | > 0.85 |
| Dihedral-coordinate consistency | RMSD between coordinate-derived and head-predicted dihedrals | < 3° |

### 7.2 Ablation experiments

The contribution of each component should be measured by training four model variants:

```
Baseline:     Standard AlphaFold3 (no BPS components)
+ Dihedral:   Add dihedral head only (no BPS losses)
+ W-embed:    Add dihedral head + W landscape embedding
+ Full BPS:   Add all three loss terms

Measure on:
  - CASP15 targets (standard benchmark)
  - De novo protein test set (no MSA available)
  - Loop accuracy subset (loops > 6 residues)
  - Disorder prediction accuracy (DisProt benchmark)
```

### 7.3 Success criteria

The architecture is validated if:

1. Full BPS model matches or exceeds baseline on standard CASP metrics (GDT-TS, lDDT, RMSD)
2. Full BPS model shows statistically significant improvement on at least one failure mode (loop prediction, orphan proteins, or disorder discrimination)
3. Predicted structures have BPS/L distributions matching experimental PDB values (mean 0.207, per-protein CV < 25%)
4. Ablation shows each component contributes — removing any one degrades at least one metric

---

## 8. Implementation Roadmap

### Phase 0: Standalone validation (2–4 weeks)

- Compute BPS/L for all CASP14/15 predictions (AlphaFold, RoseTTAFold, ESMFold) and compare to experimental structures
- Quantify: do current models already produce correct BPS/L? If so, the explicit loss term adds less value and the architecture benefit shifts to the dihedral head and coherence loss
- Build the differentiable W pipeline as a standalone PyTorch module with unit tests

### Phase 1: Dihedral head prototype (4–6 weeks)

- Train a dihedral prediction head on top of frozen Evoformer features (OpenFold checkpoint)
- Evaluate dihedral MAE against experimental structures
- Implement forward kinematics layer for consistency checking
- Establish baseline BPS/L statistics from the dihedral head alone

### Phase 2: Loss integration (6–10 weeks)

- Implement L_bps, L_coherence, L_transition as differentiable PyTorch loss functions
- Integrate into OpenFold training loop with phased scheduling
- Run initial training on a subset (CATH S40 representatives, ~12,000 structures)
- Hyperparameter sweep on λ values

### Phase 3: Full-scale training and evaluation (8–12 weeks)

- Train on full PDB training set with all BPS losses active
- Evaluate on CASP15 targets, de novo proteins, and disorder benchmarks
- Run ablation experiments
- Compare to baseline AlphaFold3 and ESMFold

### Phase 4: Publication and release (4 weeks)

- Write results paper
- Release BPS-augmented model weights and differentiable W module
- Submit to CASP16 if timeline permits

**Total estimated timeline: 6–8 months from start to submission.**

---

## 9. What Could Go Wrong

### 9.1 BPS/L is already implicitly learned

If current AlphaFold predictions already satisfy BPS/L constraints (plausible — AlphaFold was trained on real proteins), the explicit loss terms add minimal value. The dihedral head and W embedding might still help, but the headline "BPS/L improves structure prediction" would not hold.

**Mitigation:** Phase 0 tests this directly before committing to full training. If CASP predictions already have correct BPS/L, pivot the architecture toward the interpretability and confidence benefits rather than accuracy improvement.

### 9.2 Loss conflicts degrade FAPE

The BPS losses operate in dihedral space while FAPE operates in Cartesian space. In principle, there exist dihedral configurations that minimize BPS/L loss but produce poor 3D geometry (e.g., accumulation of small dihedral errors propagating into large coordinate errors over long chains).

**Mitigation:** The consistency loss between the dihedral head and the coordinate predictions explicitly couples these two spaces. Additionally, the phased training strategy ensures FAPE convergence before BPS losses activate.

### 9.3 Fold-class conditioning creates circular dependencies

If the BPS/L target depends on fold-class prediction, and fold-class prediction depends on the model's structural output, there's a circularity. Early in training, wrong fold-class predictions could set wrong BPS/L targets, creating incorrect gradients.

**Mitigation:** Use the universal target (0.202) for the first 150 epochs. Switch to conditioned targets only after fold-class accuracy exceeds 80% on the validation set. The universal target is correct for ~73% of proteins (mixed α/β) and approximately correct for another ~21% (alpha-rich), so the early-training gradients point in the right direction for >94% of cases.

### 9.4 The constant isn't universal enough

The 2.1% CV is measured across organism-level means. Per-protein variance is much larger (SD ≈ 0.04, or ~20% of the mean). If the "universal constant" is actually a broad distribution that the mean happens to compress, the loss term may be too permissive (wide tolerance) or too restrictive (narrow tolerance penalizing real biological variation).

**Mitigation:** Use fold-class-conditioned targets with appropriately wide standard deviations. The per-protein SD within mixed α/β is 0.036 — use this as σ in the Gaussian penalty, not the cross-organism SD of 0.004. The loss should penalize extreme outliers (BPS/L < 0.10 or > 0.35 for mixed folds), not enforce a precise value.

---

## 10. Why This Matters Beyond AlphaFold

Even if the accuracy improvements are modest, the architecture introduces something valuable: **an interpretable, physics-grounded internal metric for backbone quality.**

Current AlphaFold produces a pLDDT confidence score, but pLDDT is a learned quantity with no direct physical interpretation. BPS/L is a measured physical invariant. A predicted structure with BPS/L far from the expected range is provably non-biological in its backbone organization — regardless of what pLDDT says.

This enables:

**Structure validation.** Flag predictions where BPS/L deviates from the expected range, even if pLDDT is high. These are likely cases where the model has produced geometrically plausible but physically unrealistic backbones.

**Protein design.** Constrain computational protein design to produce sequences whose predicted backbones satisfy BPS/L. This is a necessary condition for foldability that current design pipelines don't enforce.

**Evolutionary analysis.** BPS/L is conserved across all life. Proteins that violate the constraint may be misfolded, disordered, or under unusual evolutionary pressure. This is a filter for comparative genomics.

**Training data curation.** Score the AlphaFold Database entries by BPS/L compliance. Structures with anomalous BPS/L (controlling for fold class) are candidates for re-prediction or experimental validation.

The long-term value of embedding the Ramachandran superpotential into structure prediction networks may exceed the near-term accuracy gains. It makes the backbone physics of protein structure explicit, inspectable, and enforceable — rather than leaving it buried in 100 million learned parameters.

---

*This document is a technical specification, not a claim of completed work. Implementation requires access to GPU training infrastructure and the OpenFold codebase. The BPS/L measurements and validation controls referenced here are from "Proteins as One-Dimensional Paths on the Ramachandran Torus" (Branham, 2026, in preparation).*
