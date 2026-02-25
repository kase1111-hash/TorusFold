# TorusFold

**Protein backbone structure from one-dimensional paths on the Ramachandran torus.**

TorusFold treats protein backbones as discrete paths on the flat torus T² = S¹ × S¹ (the Ramachandran configuration space), evaluates them against a fixed scalar superpotential W(φ,ψ) derived from empirical backbone statistics, and extracts structural invariants, loop path families, and prediction constraints.

---

## Key Results

### A Universal Backbone Invariant

The total variation of W along the sequential backbone path, normalized by chain length (BPS/L), converges to a universal constant:

**BPS/L = 0.202 ± 0.004** (CV = 2.1%)

across 22 organisms spanning all kingdoms of life (66,549 AlphaFold structures), confirmed on 162 experimental PDB crystal structures (BPS/L = 0.207, Δ = 2.4%).

BPS/L is independent of contact order (r = −0.05), helix content (r = 0.03), and sheet content (r = −0.11), establishing it as a novel structural descriptor orthogonal to all previously reported backbone metrics.

### Canonical Loop Path Families

Loop transitions between α-helix and β-sheet basins cluster into discrete families on T²:

| Loop length | Families | Coverage | |ΔW| CV |
|-------------|----------|----------|---------|
| Short (≤7 residues) | 30 | 100% | 6.6–9.3% |
| Medium (8–10) | 1–3 | 52–100% | ~24% |
| Long (11+) | 0 tight | — | >37% |

Short loops are geometrically constrained by saddle-point topology. The loop prediction problem for this class is classification, not generation.

### Three-Level Backbone Decomposition

| Model | BPS/L | What's preserved |
|-------|-------|------------------|
| Shuffled | 0.55 | Basin occupancy only |
| Markov | 0.30 | Transition frequencies |
| Real | 0.20 | Full conformational coherence |

Real proteins suppress intra-basin roughness by 97% compared to random within-basin sampling. This suppression is the quantitative signature of secondary structure.

---

## Installation

```bash
git clone https://github.com/yourusername/TorusFold.git
cd TorusFold
pip install numpy scipy biopython matplotlib scikit-learn
```

Python 3.10+ required. No GPU needed for analysis pipelines.

## Quick Start

### Compute BPS/L for PDB structures

```bash
# Run PDB experimental validation (~200 structures)
python bps/bps_pdb_validate.py

# Output: pdb_validation_report.md
```

### Run loop path taxonomy

```bash
# Full analysis (~500 structures, 15-45 min)
python loops/loop_taxonomy_v2.py

# Quick test (100 structures, ~5 min)
python loops/loop_taxonomy_v2.py --max-structs 100

# Output: loop_taxonomy_v2_output/loop_taxonomy_v2_report.md
#         loop_taxonomy_v2_output/*.png (torus plots)
```

### Compute BPS/L for AlphaFold proteomes

```bash
# Download and process a single organism
python bps/bps_download.py --organism ecoli
python bps/bps_process.py --organism ecoli

# Generate cross-proteome report
python bps/compile_results.py
```

---

## How It Works

### The Superpotential

Every residue in a protein backbone has two dihedral angles (φ, ψ) that live on the flat torus T². We define a scalar field on this torus:

```
W(φ, ψ) = −ln p(φ, ψ)
```

where p is the empirical probability density of backbone dihedrals across all known structures. W is the potential of mean force — valleys are populated regions (α-helix, β-sheet), peaks are forbidden regions (steric clashes), saddle points are the passes between basins.

### The Observable

A protein backbone traces a sequential path through this landscape. BPS/L measures the roughness of that path:

```
BPS/L = (1/L) Σ |W(φ_{i+1}, ψ_{i+1}) − W(φ_i, ψ_i)|
```

This is the total variation of W along the chain, normalized by length. It counts every transition through the landscape regardless of direction.

### The Result

BPS/L ≈ 0.20 is determined by:
1. **Fixed landscape topology** — the energy gaps between Ramachandran basins are universal
2. **Conserved transition frequencies** — secondary structure element lengths are conserved across all life
3. **Intra-basin conformational coherence** — consecutive residues in a helix/sheet lock to a narrow attractor, not the full basin

---

## Repository Structure

```
TorusFold/
├── CLAUDE.md              # Development context for Claude Code
├── bps/                   # BPS/L proteome atlas pipeline
├── loops/                 # Loop path taxonomy and clustering
├── torusfold/             # Structure prediction module (in development)
├── docs/                  # Architecture specs and papers
├── data/                  # Generated data (gitignored)
└── tests/                 # Validation and regression tests
```

## Papers

- **BPS Paper:** "Proteins as One-Dimensional Paths on the Ramachandran Torus: A Superpotential Framework for Backbone Structural Invariants" (Branham, 2026, in preparation)
- **Loop Taxonomy:** "Discrete Loop Path Families on the Ramachandran Torus and Their Sequence Determinants" (planned)
- **Architecture Spec:** TorusFold integration with structure prediction networks (see `docs/alphafold_integration.md`)

## Citation

If you use TorusFold or the BPS/L framework in your work, please cite:

```
Branham, K. (2026). Proteins as One-Dimensional Paths on the Ramachandran Torus:
A Superpotential Framework for Backbone Structural Invariants. In preparation.
```

## License

MIT

## Author

**Kase Branham** · Independent Research · Portland, Oregon
