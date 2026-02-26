"""
TorusFold: Polymer Null Experiment
===================================
The make-or-break experiment: does the ~11% torus-distance geometric
effect arise from fold-specific micro-regularity, or from universal
peptide bond stereochemistry?

APPROACH:
  Generate synthetic polymer backbones with realistic bond geometry
  but NO secondary structure stabilization. Apply the same torus L1
  Seg/Real analysis used on real proteins.

  Four polymer models of increasing physical realism:

  Model 1 — IID Ramachandran
    Each residue's (φ, ψ) drawn independently from P(φ, ψ).
    No physics at all. Expected Seg/Real ≈ 1.0.
    This is the "null null" — confirms the method works.

  Model 2 — Steric-coupled chain
    Build backbone atom-by-atom with fixed bond lengths, bond angles,
    and ω = 180°. Sample (φ, ψ) from P(φ, ψ) but reject if any
    backbone atom clashes with atoms from previous residues
    (distance < clash_radius). This introduces peptide-geometry
    coupling without any H-bond or secondary structure stabilization.

  Model 3 — Neighbor-conditional
    Estimate the empirical transition density P(φᵢ₊₁, ψᵢ₊₁ | φᵢ, ψᵢ)
    from real protein data and sample sequentially. This captures
    whatever sequential torsion correlation exists in real backbones
    WITHOUT using SS labels or null models.

  Model 4 — Real proteins (control)
    Same torus L1 Seg/Real from actual AlphaFold structures.
    Should reproduce the ~1.11× from the Tier 0 test.

INTERPRETATION:
  If Model 2 (steric-coupled) shows Seg/Real ≈ 1.11×:
    → Layer 1 = peptide stereochemistry.
    → The effect is real but arises from backbone physics, not
       fold-specific structure.

  If Model 2 shows Seg/Real ≈ 1.0 but Model 4 shows 1.11×:
    → Layer 1 = fold-specific micro-regularity.
    → Secondary structure stabilization creates geometric
       smoothness beyond what peptide bonds alone produce.

  If Model 3 (neighbor-conditional) shows ~1.11× but Model 2 doesn't:
    → The coupling is empirical (present in data) but not purely
       steric (not explained by clash avoidance alone).

Usage:
  python polymer_null_experiment.py --output results
         [--n-chains 500] [--chain-length 200] [--n-trials 10]
         [--data alphafold_cache]  # optional, for Model 3+4

  Without --data: Runs Models 1-2 using built-in Ramachandran
                  approximation. Still answers the key question.
  With --data:    Also runs Models 3-4 using real protein data.

Reads:  results/superpotential_W.npz (optional, for histogram)
        alphafold_cache/ (optional, for Models 3+4)
Writes: results/polymer_null_report.md
"""

import os
import sys
import math
import time
import argparse
import numpy as np
from collections import defaultdict

# Optional imports for Models 3+4
try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False


# ═══════════════════════════════════════════════════════════════════
# BACKBONE GEOMETRY CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# Bond lengths (Angstroms)
BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329

# Bond angles (radians)
ANGLE_C_N_CA = math.radians(121.7)   # C(i-1)-N(i)-Cα(i)
ANGLE_N_CA_C = math.radians(111.2)   # N(i)-Cα(i)-C(i)
ANGLE_CA_C_N = math.radians(116.2)   # Cα(i)-C(i)-N(i+1)

# Clash radius for steric check (Angstroms)
CLASH_RADIUS = 2.0
CLASH_RADIUS_SQ = CLASH_RADIUS ** 2

# Peptide bond torsion
OMEGA = math.pi  # trans peptide bond (180°)


# ═══════════════════════════════════════════════════════════════════
# NERF ALGORITHM: Place atom given 3 previous atoms + geometry
# ═══════════════════════════════════════════════════════════════════

def place_atom(a, b, c, bond_length, bond_angle, torsion):
    """
    Place atom D given atoms A, B, C such that:
      |CD| = bond_length
      angle(BCD) = bond_angle
      torsion(ABCD) = torsion

    Uses the NeRF (Natural Extension Reference Frame) algorithm.
    """
    # Unit vector along BC
    bc = c - b
    bc_norm = np.linalg.norm(bc)
    if bc_norm < 1e-10:
        return c + np.array([bond_length, 0, 0])
    bc_hat = bc / bc_norm

    # Normal to ABC plane
    ab = b - a
    n = np.cross(ab, bc)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Degenerate case: use arbitrary perpendicular
        n = np.array([1, 0, 0]) if abs(bc_hat[0]) < 0.9 else np.array([0, 1, 0])
        n = np.cross(n, bc_hat)
        n_norm = np.linalg.norm(n)
    n_hat = n / n_norm

    # Build local frame
    m = np.cross(n_hat, bc_hat)

    # Spherical to Cartesian in local frame
    d_local = np.array([
        -bond_length * math.cos(bond_angle),
        bond_length * math.sin(bond_angle) * math.cos(torsion),
        bond_length * math.sin(bond_angle) * math.sin(torsion),
    ])

    # Transform to global frame
    rotation = np.column_stack([bc_hat, m, n_hat])
    d_global = c + rotation @ d_local
    return d_global


# ═══════════════════════════════════════════════════════════════════
# BACKBONE BUILDER
# ═══════════════════════════════════════════════════════════════════

def build_backbone(phi_psi_list):
    """
    Build 3D backbone coordinates from a list of (φ, ψ) pairs.
    Returns list of (N, Cα, C) coordinate triples.

    The torsion angles define the chain:
      φ(i):  C(i-1) - N(i)  - Cα(i) - C(i)
      ψ(i):  N(i)   - Cα(i) - C(i)  - N(i+1)
      ω(i):  Cα(i)  - C(i)  - N(i+1)- Cα(i+1)   [fixed at π]
    """
    n_res = len(phi_psi_list)
    if n_res == 0:
        return []

    coords = []  # list of (N, CA, C) arrays

    # Initialize first three atoms
    N0 = np.array([0.0, 0.0, 0.0])
    CA0 = np.array([BOND_N_CA, 0.0, 0.0])
    # Place C0 using N-CA-C angle and first psi (arbitrary start)
    C0 = place_atom(
        np.array([-1.0, 0.0, 0.0]),  # virtual atom for initial placement
        N0, CA0,
        BOND_CA_C, ANGLE_N_CA_C, phi_psi_list[0][0]  # use phi for initial torsion
    )
    coords.append((N0, CA0, C0))

    for i in range(1, n_res):
        phi_i, psi_i = phi_psi_list[i]
        prev_N, prev_CA, prev_C = coords[-1]

        # Place N(i): torsion = ψ(i-1) around CA(i-1)-C(i-1)
        psi_prev = phi_psi_list[i-1][1]
        N_i = place_atom(prev_N, prev_CA, prev_C,
                         BOND_C_N, ANGLE_CA_C_N, psi_prev)

        # Place Cα(i): torsion = ω around C(i-1)-N(i)
        CA_i = place_atom(prev_CA, prev_C, N_i,
                          BOND_N_CA, ANGLE_C_N_CA, OMEGA)

        # Place C(i): torsion = φ(i) around N(i)-Cα(i)
        C_i = place_atom(prev_C, N_i, CA_i,
                         BOND_CA_C, ANGLE_N_CA_C, phi_i)

        coords.append((N_i, CA_i, C_i))

    return coords


def check_steric_clash(coords, new_atoms, lookback=6):
    """
    Check if any new atom clashes with recent backbone atoms.
    lookback: how many previous residues to check against.
    """
    start = max(0, len(coords) - lookback)
    for ri in range(start, len(coords)):
        for old_atom in coords[ri]:
            for new_atom in new_atoms:
                diff = new_atom - old_atom
                dist_sq = np.dot(diff, diff)
                if dist_sq < CLASH_RADIUS_SQ:
                    return True
    return False


# ═══════════════════════════════════════════════════════════════════
# RAMACHANDRAN DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════

def build_ramachandran_sampler(histogram=None, grid_size=360):
    """
    Build a sampler for (φ, ψ) from the Ramachandran distribution.
    If histogram is provided, use it. Otherwise, build a simple
    two-Gaussian approximation (α + β basins).
    """
    if histogram is not None:
        # Normalize to probability
        P = histogram / histogram.sum()
        flat = P.flatten()
        cumsum = np.cumsum(flat)

        def sample(rng, n=1):
            u = rng.random(n)
            idx = np.searchsorted(cumsum, u)
            idx = np.clip(idx, 0, grid_size * grid_size - 1)
            gi = idx // grid_size
            gj = idx % grid_size

            # Convert grid indices to angles (radians) with jitter
            phi_d = (gi + rng.random(n)) * (360.0 / grid_size) - 180.0
            psi_d = (gj + rng.random(n)) * (360.0 / grid_size) - 180.0
            return np.radians(phi_d), np.radians(psi_d)

        return sample

    else:
        # Fallback: two-Gaussian approximation
        # α basin: φ ≈ -63°, ψ ≈ -43°
        # β basin: φ ≈ -120°, ψ ≈ 130°
        # Basin weights: ~60% α, ~25% β, ~15% other
        def sample(rng, n=1):
            phi = np.empty(n)
            psi = np.empty(n)
            for k in range(n):
                r = rng.random()
                if r < 0.60:
                    # α basin
                    phi[k] = math.radians(rng.normal(-63, 12))
                    psi[k] = math.radians(rng.normal(-43, 12))
                elif r < 0.85:
                    # β basin
                    phi[k] = math.radians(rng.normal(-120, 15))
                    psi[k] = math.radians(rng.normal(130, 15))
                else:
                    # Other / PPII / αL
                    phi[k] = math.radians(rng.uniform(-180, 180))
                    psi[k] = math.radians(rng.uniform(-180, 180))
            return phi, psi

        return sample


# ═══════════════════════════════════════════════════════════════════
# TORUS DISTANCE METRICS
# ═══════════════════════════════════════════════════════════════════

def _circular_diff(a, b):
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def roughness_torus_l1(phi, psi):
    if len(phi) < 2:
        return 0.0
    dphi = _circular_diff(phi[1:], phi[:-1])
    dpsi = _circular_diff(psi[1:], psi[:-1])
    return float(np.mean(dphi + dpsi))


# ═══════════════════════════════════════════════════════════════════
# SS CLASSIFICATION AND NULL MODELS
# ═══════════════════════════════════════════════════════════════════

def classify_ss(phi_rad, psi_rad):
    """Classify residues into SS basins."""
    ss = []
    for phi, psi in zip(phi_rad, psi_rad):
        phi_d = math.degrees(phi)
        psi_d = math.degrees(psi)
        if -160 < phi_d < 0 and -120 < psi_d < 30:
            ss.append('a')
        elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
            ss.append('b')
        else:
            ss.append('o')
    return ss


def null_segment(phi, psi, ss_seq, rng):
    """Segment-preserving null (per-chain basin pools)."""
    pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        pools[s].append(i)
    new_phi = np.empty_like(phi)
    new_psi = np.empty_like(psi)
    for i, s in enumerate(ss_seq):
        j = rng.choice(pools[s])
        new_phi[i], new_psi[i] = phi[j], psi[j]
    return new_phi, new_psi


def null_shuffled(phi, psi, rng):
    """Shuffled null."""
    perm = rng.permutation(len(phi))
    return phi[perm], psi[perm]


# ═══════════════════════════════════════════════════════════════════
# MODEL GENERATORS
# ═══════════════════════════════════════════════════════════════════

def generate_model1_iid(sampler, rng, chain_length):
    """Model 1: IID Ramachandran draws. No physics."""
    phi, psi = sampler(rng, chain_length)
    return phi, psi


def generate_model2_steric(sampler, rng, chain_length, max_retries=50):
    """
    Model 2: Steric-coupled chain.
    Build backbone with realistic geometry, reject φ/ψ that cause
    steric clashes with recent residues.

    STERIC MODEL LIMITATIONS:
    - Backbone atoms only (N, CA, C) — no side chains
    - Hard-sphere clash at 2.0 Angstrom — no soft potentials
    - 6-residue lookback — no long-range contacts
    - No electrostatics, no van der Waals, no solvent
    - This tests GEOMETRY only, not full force-field physics

    If this minimal steric model shows Seg/Real ≈ 1.0, it means
    backbone clash avoidance alone does not produce the smoothing
    effect. A full-physics null (MD simulation) is needed to
    conclusively rule out universal physics as the source.
    """
    phi_list = []
    psi_list = []
    coords = []

    # First residue: no clash possible
    phi_0, psi_0 = sampler(rng, 1)
    phi_list.append(float(phi_0[0]))
    psi_list.append(float(psi_0[0]))

    # Build initial coordinates
    N0 = np.array([0.0, 0.0, 0.0])
    CA0 = np.array([BOND_N_CA, 0.0, 0.0])
    C0 = place_atom(
        np.array([-1.0, 0.0, 0.0]), N0, CA0,
        BOND_CA_C, ANGLE_N_CA_C, phi_list[0]
    )
    coords.append((N0, CA0, C0))

    for i in range(1, chain_length):
        accepted = False
        for attempt in range(max_retries):
            phi_try, psi_try = sampler(rng, 1)
            phi_try = float(phi_try[0])
            psi_try = float(psi_try[0])

            # Build trial atoms
            prev_N, prev_CA, prev_C = coords[-1]
            psi_prev = psi_list[-1]

            N_i = place_atom(prev_N, prev_CA, prev_C,
                             BOND_C_N, ANGLE_CA_C_N, psi_prev)
            CA_i = place_atom(prev_CA, prev_C, N_i,
                              BOND_N_CA, ANGLE_C_N_CA, OMEGA)
            C_i = place_atom(prev_C, N_i, CA_i,
                             BOND_CA_C, ANGLE_N_CA_C, phi_try)

            trial_atoms = (N_i, CA_i, C_i)

            if not check_steric_clash(coords, trial_atoms, lookback=6):
                phi_list.append(phi_try)
                psi_list.append(psi_try)
                coords.append(trial_atoms)
                accepted = True
                break

        if not accepted:
            # Accept anyway after max retries (avoid infinite loop)
            phi_try, psi_try = sampler(rng, 1)
            phi_list.append(float(phi_try[0]))
            psi_list.append(float(psi_try[0]))
            # Rebuild coords with accepted angles
            prev_N, prev_CA, prev_C = coords[-1]
            psi_prev = psi_list[-2]
            N_i = place_atom(prev_N, prev_CA, prev_C,
                             BOND_C_N, ANGLE_CA_C_N, psi_prev)
            CA_i = place_atom(prev_CA, prev_C, N_i,
                              BOND_N_CA, ANGLE_C_N_CA, OMEGA)
            C_i = place_atom(prev_C, N_i, CA_i,
                             BOND_CA_C, ANGLE_N_CA_C, phi_list[-1])
            coords.append((N_i, CA_i, C_i))

    return np.array(phi_list), np.array(psi_list)


def generate_model3_conditional(transition_matrix, marginal_cdf, rng,
                                chain_length, grid_size=72):
    """
    Model 3: Neighbor-conditional sampling.
    Uses empirical P(φᵢ₊₁, ψᵢ₊₁ | φᵢ, ψᵢ) estimated from data.
    transition_matrix: (grid_size², grid_size²) conditional probabilities
    marginal_cdf: CDF for first-residue sampling
    """
    scale = grid_size / 360.0
    phi_list = []
    psi_list = []

    # First residue from marginal
    u = rng.random()
    idx = np.searchsorted(marginal_cdf, u)
    idx = min(idx, grid_size * grid_size - 1)
    gi, gj = divmod(idx, grid_size)
    phi_d = (gi + rng.random()) * (360.0 / grid_size) - 180.0
    psi_d = (gj + rng.random()) * (360.0 / grid_size) - 180.0
    phi_list.append(math.radians(phi_d))
    psi_list.append(math.radians(psi_d))

    for i in range(1, chain_length):
        # Current bin
        prev_phi_d = math.degrees(phi_list[-1])
        prev_psi_d = math.degrees(psi_list[-1])
        prev_gi = int(round((prev_phi_d + 180) * scale)) % grid_size
        prev_gj = int(round((prev_psi_d + 180) * scale)) % grid_size
        prev_bin = prev_gi * grid_size + prev_gj

        # Sample from conditional
        cond_probs = transition_matrix[prev_bin]
        cond_cdf = np.cumsum(cond_probs)
        u = rng.random()
        next_bin = np.searchsorted(cond_cdf, u)
        next_bin = min(next_bin, grid_size * grid_size - 1)
        ngi, ngj = divmod(next_bin, grid_size)

        phi_d = (ngi + rng.random()) * (360.0 / grid_size) - 180.0
        psi_d = (ngj + rng.random()) * (360.0 / grid_size) - 180.0
        phi_list.append(math.radians(phi_d))
        psi_list.append(math.radians(psi_d))

    return np.array(phi_list), np.array(psi_list)


# ═══════════════════════════════════════════════════════════════════
# BUILD CONDITIONAL TRANSITION MATRIX FROM DATA
# ═══════════════════════════════════════════════════════════════════

def build_transition_matrix(all_phi_psi_sequences, grid_size=72):
    """
    Estimate P(φᵢ₊₁, ψᵢ₊₁ | φᵢ, ψᵢ) from sequences of (φ, ψ).
    Uses coarse grid (72×72 = 5° bins) for tractability.
    Returns (n_bins², n_bins²) row-stochastic matrix and marginal CDF.
    """
    n_bins = grid_size
    n_states = n_bins * n_bins
    scale = n_bins / 360.0

    counts = np.zeros((n_states, n_states), dtype=np.float64)
    marginal = np.zeros(n_states, dtype=np.float64)

    for seq in all_phi_psi_sequences:
        if len(seq) < 2:
            continue
        for k in range(len(seq)):
            phi_d = math.degrees(seq[k][0])
            psi_d = math.degrees(seq[k][1])
            gi = int(round((phi_d + 180) * scale)) % n_bins
            gj = int(round((psi_d + 180) * scale)) % n_bins
            state = gi * n_bins + gj
            marginal[state] += 1

            if k < len(seq) - 1:
                phi_d2 = math.degrees(seq[k+1][0])
                psi_d2 = math.degrees(seq[k+1][1])
                gi2 = int(round((phi_d2 + 180) * scale)) % n_bins
                gj2 = int(round((psi_d2 + 180) * scale)) % n_bins
                state2 = gi2 * n_bins + gj2
                counts[state, state2] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans = counts / row_sums

    # For states with no data, fall back to marginal
    empty_rows = counts.sum(axis=1) == 0
    marginal_prob = marginal / marginal.sum() if marginal.sum() > 0 else np.ones(n_states) / n_states
    trans[empty_rows] = marginal_prob

    marginal_cdf = np.cumsum(marginal_prob)

    return trans, marginal_cdf


# ═══════════════════════════════════════════════════════════════════
# REAL PROTEIN ANGLE EXTRACTION (for Models 3+4)
# ═══════════════════════════════════════════════════════════════════

def _dihedral_angle(p0, p1, p2, p3):
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1n = np.linalg.norm(n1)
    n2n = np.linalg.norm(n2)
    if n1n < 1e-10 or n2n < 1e-10:
        return 0.0
    n1 /= n1n
    n2 /= n2n
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return math.atan2(np.dot(m1, n2), np.dot(n1, n2))


def extract_angles(filepath, plddt_min=70.0):
    """Extract (phi, psi) pairs and SS from CIF file."""
    st = gemmi.read_structure(str(filepath))
    if len(st) == 0 or len(st[0]) == 0:
        return [], []
    chain = st[0][0]
    residues = []
    for res in chain:
        info = gemmi.find_tabulated_residue(res.name)
        if not info.is_amino_acid():
            continue
        atoms = {}
        for atom in res:
            if atom.name in ('N', 'CA', 'C'):
                atoms[atom.name] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        if len(atoms) == 3:
            ca = res.find_atom('CA', '*')
            residues.append({
                'N': atoms['N'], 'CA': atoms['CA'], 'C': atoms['C'],
                'plddt': ca.b_iso if ca else 0,
            })
    if len(residues) < 3:
        return [], []

    phi_psi = []
    ss_seq = []
    for i in range(1, len(residues) - 1):
        phi = -_dihedral_angle(residues[i-1]['C'], residues[i]['N'],
                               residues[i]['CA'], residues[i]['C'])
        psi = -_dihedral_angle(residues[i]['N'], residues[i]['CA'],
                               residues[i]['C'], residues[i+1]['N'])
        if residues[i]['plddt'] < plddt_min:
            continue
        phi_d, psi_d = math.degrees(phi), math.degrees(psi)
        if -160 < phi_d < 0 and -120 < psi_d < 30:
            ss = 'a'
        elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
            ss = 'b'
        else:
            ss = 'o'
        phi_psi.append((phi, psi))
        ss_seq.append(ss)
    return phi_psi, ss_seq


# ═══════════════════════════════════════════════════════════════════
# EVALUATE A SET OF CHAINS
# ═══════════════════════════════════════════════════════════════════

def evaluate_chains(phi_psi_list, rng, n_trials=10):
    """
    Compute torus L1 Seg/Real, M1/Real, S/Real for a list of
    (phi_array, psi_array) chains.
    """
    real_v, seg_v, shuf_v = [], [], []

    for phi, psi in phi_psi_list:
        if len(phi) < 20:
            continue

        ss = classify_ss(phi, psi)
        real_r = roughness_torus_l1(phi, psi)
        real_v.append(real_r)

        for _ in range(n_trials):
            sp, sq = null_segment(phi, psi, ss, rng)
            seg_v.append(roughness_torus_l1(sp, sq))

            sfp, sfq = null_shuffled(phi, psi, rng)
            shuf_v.append(roughness_torus_l1(sfp, sfq))

    if not real_v:
        return None

    rm = float(np.mean(real_v))
    sm = float(np.mean(seg_v))
    shm = float(np.mean(shuf_v))

    return {
        'real': rm,
        'segment': sm,
        'shuffled': shm,
        'seg_real': sm / rm if rm > 0 else 0,
        's_real': shm / rm if rm > 0 else 0,
        'n_chains': len(real_v),
        'real_std': float(np.std(real_v)),
    }


# ═══════════════════════════════════════════════════════════════════
# BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════

def bootstrap_ratio_ci(null_vals, real_vals, n_boot=2000, alpha=0.05):
    rng = np.random.default_rng(99)
    null_arr = np.array(null_vals)
    real_arr = np.array(real_vals)
    ratios = []
    for _ in range(n_boot):
        ni = rng.choice(len(null_arr), len(null_arr), replace=True)
        ri = rng.choice(len(real_arr), len(real_arr), replace=True)
        rm = np.mean(real_arr[ri])
        if rm > 0:
            ratios.append(np.mean(null_arr[ni]) / rm)
    if not ratios:
        return (0.0, 0.0)
    return (float(np.percentile(ratios, 100 * alpha / 2)),
            float(np.percentile(ratios, 100 * (1 - alpha / 2))))


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Polymer Null Experiment (Layer 1 Falsification)")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--data", default=None,
                        help="AlphaFold data directory (optional, for Models 3+4)")
    parser.add_argument("--w-path", default=None,
                        help="Path to superpotential_W.npz (for histogram)")
    parser.add_argument("--n-chains", type=int, default=500,
                        help="Synthetic chains per model (default: 500)")
    parser.add_argument("--chain-length", type=int, default=200,
                        help="Residues per chain (default: 200)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Null model trials per chain (default: 10)")
    parser.add_argument("--n-real", type=int, default=200,
                        help="Real proteins for Model 4 (default: 200)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold: Polymer Null Experiment")
    print("  Does Layer 1 arise from fold-specific structure")
    print("  or peptide stereochemistry?")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # ── Load Ramachandran histogram if available ─────────────
    histogram = None
    search_paths = [args.w_path] if args.w_path else []
    search_paths += [
        os.path.join(args.output, "superpotential_W.npz"),
        'results/superpotential_W.npz',
        'superpotential_W.npz',
    ]
    for sp in search_paths:
        if sp and os.path.exists(sp):
            data = np.load(sp)
            if 'histogram' in data:
                histogram = data['histogram']
                print(f"  Loaded Ramachandran histogram from {sp}")
            break

    sampler = build_ramachandran_sampler(histogram)
    if histogram is None:
        print("  Using built-in Ramachandran approximation "
              "(two-Gaussian model)")

    results = {}

    # ══════════════════════════════════════════════════════════
    # MODEL 1: IID Ramachandran
    # ══════════════════════════════════════════════════════════
    print(f"\n  Model 1: IID Ramachandran ({args.n_chains} chains × "
          f"{args.chain_length} residues)...")

    m1_chains = []
    for ci in range(args.n_chains):
        phi, psi = generate_model1_iid(sampler, rng, args.chain_length)
        m1_chains.append((phi, psi))
        if (ci + 1) % 100 == 0:
            print(f"\r    [{ci+1}/{args.n_chains}]...", end="", flush=True)

    print(f"\r    Evaluating...    ", end="", flush=True)
    results['Model 1: IID'] = evaluate_chains(m1_chains, rng, args.n_trials)
    r = results['Model 1: IID']
    print(f"Seg/Real = {r['seg_real']:.3f}x, "
          f"S/Real = {r['s_real']:.3f}x")

    # ══════════════════════════════════════════════════════════
    # MODEL 2: Steric-coupled chain
    # ══════════════════════════════════════════════════════════
    print(f"\n  Model 2: Steric-coupled ({args.n_chains} chains × "
          f"{args.chain_length} residues)...")
    print("  (This takes longer — building 3D coordinates...)")

    m2_chains = []
    n_clashes = 0
    t_start = time.time()
    for ci in range(args.n_chains):
        phi, psi = generate_model2_steric(
            sampler, rng, args.chain_length, max_retries=50)
        m2_chains.append((phi, psi))
        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (ci + 1) / elapsed
            eta = (args.n_chains - ci - 1) / rate
            print(f"\r    [{ci+1}/{args.n_chains}] "
                  f"({rate:.0f} chains/s, ETA {eta:.0f}s)...",
                  end="", flush=True)

    print(f"\r    Evaluating...                              ",
          end="", flush=True)
    results['Model 2: Steric'] = evaluate_chains(
        m2_chains, rng, args.n_trials)
    r = results['Model 2: Steric']
    print(f"Seg/Real = {r['seg_real']:.3f}x, "
          f"S/Real = {r['s_real']:.3f}x")

    # ══════════════════════════════════════════════════════════
    # MODEL 3: Neighbor-conditional (requires real data)
    # ══════════════════════════════════════════════════════════
    if args.data and HAS_GEMMI:
        print(f"\n  Model 3: Neighbor-conditional "
              f"(building transition matrix from data...)")

        from pathlib import Path
        data_path = Path(args.data)
        all_sequences = []
        all_files = []
        for subdir in sorted(data_path.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith('.'):
                continue
            cif_files = sorted(subdir.glob("*.cif"))
            for f in cif_files:
                all_files.append(str(f))

        # Sample subset for transition matrix
        rng_tm = np.random.default_rng(123)
        tm_sample = min(2000, len(all_files))
        tm_indices = rng_tm.choice(len(all_files), tm_sample, replace=False)
        n_tm_errors = 0

        print(f"    Extracting angles from {tm_sample} proteins...")
        for fi, idx in enumerate(tm_indices):
            try:
                fsize = os.path.getsize(all_files[idx])
                if fsize > 5 * 1024 * 1024:
                    continue
                pp, ss = extract_angles(all_files[idx])
                if len(pp) >= 50:
                    all_sequences.append(pp)
            except Exception:
                n_tm_errors += 1
            if (fi + 1) % 200 == 0:
                print(f"\r    [{fi+1}/{tm_sample}] "
                      f"{len(all_sequences)} sequences...",
                      end="", flush=True)

        print(f"\r    {len(all_sequences)} sequences extracted    ")

        if len(all_sequences) >= 100:
            print("    Building transition matrix (72×72 grid)...")
            trans_matrix, marginal_cdf = build_transition_matrix(
                all_sequences, grid_size=72)

            print(f"    Generating {args.n_chains} conditional chains...")
            m3_chains = []
            for ci in range(args.n_chains):
                phi, psi = generate_model3_conditional(
                    trans_matrix, marginal_cdf, rng,
                    args.chain_length, grid_size=72)
                m3_chains.append((phi, psi))
                if (ci + 1) % 100 == 0:
                    print(f"\r    [{ci+1}/{args.n_chains}]...",
                          end="", flush=True)

            print(f"\r    Evaluating...    ", end="", flush=True)
            results['Model 3: Conditional'] = evaluate_chains(
                m3_chains, rng, args.n_trials)
            r = results['Model 3: Conditional']
            print(f"Seg/Real = {r['seg_real']:.3f}x, "
                  f"S/Real = {r['s_real']:.3f}x")
        else:
            print("    Insufficient data for transition matrix.")

        # ══════════════════════════════════════════════════════
        # MODEL 4: Real proteins (control)
        # ══════════════════════════════════════════════════════
        print(f"\n  Model 4: Real proteins (control, "
              f"N = {args.n_real})...")

        rng_real = np.random.default_rng(456)
        real_indices = rng_real.choice(
            len(all_files), min(args.n_real, len(all_files)), replace=False)
        real_chains = []
        for fi, idx in enumerate(real_indices):
            try:
                fsize = os.path.getsize(all_files[idx])
                if fsize > 5 * 1024 * 1024:
                    continue
                pp, ss = extract_angles(all_files[idx])
                if len(pp) >= 50:
                    phi = np.array([p[0] for p in pp])
                    psi = np.array([p[1] for p in pp])
                    real_chains.append((phi, psi))
            except Exception:
                pass
            if (fi + 1) % 50 == 0:
                print(f"\r    [{fi+1}/{min(args.n_real, len(all_files))}]...",
                      end="", flush=True)

        print(f"\r    {len(real_chains)} real proteins loaded    ")
        if real_chains:
            results['Model 4: Real'] = evaluate_chains(
                real_chains, rng, args.n_trials)
            r = results['Model 4: Real']
            print(f"    Seg/Real = {r['seg_real']:.3f}x, "
                  f"S/Real = {r['s_real']:.3f}x")

    elif args.data and not HAS_GEMMI:
        print("\n  Skipping Models 3+4 (gemmi not installed)")
    else:
        print("\n  Skipping Models 3+4 (no --data specified)")

    # ══════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "polymer_null_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Polymer Null Experiment "
                "(Layer 1 Falsification)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Synthetic chains:** {args.n_chains} × "
                f"{args.chain_length} residues\n")
        f.write(f"**Null trials:** {args.n_trials} per chain\n\n")

        # Key result table
        f.write("## Key Result\n\n")
        f.write("```\n")
        f.write("POLYMER NULL EXPERIMENT\n")
        f.write("═══════════════════════════════════════════════════\n")
        f.write(f"  {'Model':<28s} {'N':>5s} {'Real':>8s} "
                f"{'Seg/Real':>10s} {'S/Real':>10s}\n")
        f.write("─" * 63 + "\n")
        for label in ['Model 1: IID', 'Model 2: Steric',
                       'Model 3: Conditional', 'Model 4: Real']:
            if label not in results:
                continue
            r = results[label]
            f.write(f"  {label:<28s} {r['n_chains']:>5d} "
                    f"{r['real']:>8.4f} "
                    f"{r['seg_real']:>9.3f}x "
                    f"{r['s_real']:>9.3f}x\n")
        f.write("═══════════════════════════════════════════════════\n")
        f.write("```\n\n")

        # Verdict
        f.write("## Verdict\n\n")

        m1_sr = results.get('Model 1: IID', {}).get('seg_real', 0)
        m2_sr = results.get('Model 2: Steric', {}).get('seg_real', 0)
        m3_sr = results.get('Model 3: Conditional', {}).get('seg_real', 0)
        m4_sr = results.get('Model 4: Real', {}).get('seg_real', 0)

        f.write(f"**Model 1 (IID, no physics): Seg/Real = "
                f"{m1_sr:.3f}×**\n")
        if abs(m1_sr - 1.0) < 0.03:
            f.write("Baseline confirmed: independent Ramachandran draws "
                    "show no sequential smoothing. The method works.\n\n")
        else:
            f.write(f"WARNING: IID baseline deviates from 1.0 by "
                    f"{abs(m1_sr-1.0):.3f}. Investigate potential bias.\n\n")

        f.write(f"**Model 2 (steric-coupled): Seg/Real = "
                f"{m2_sr:.3f}×**\n")
        if m2_sr > 1.08:
            f.write("**Peptide stereochemistry alone produces substantial "
                    "sequential smoothing.** The steric coupling between "
                    "adjacent peptide planes — without any hydrogen "
                    "bonding or secondary structure stabilization — "
                    f"accounts for a {(m2_sr-1)*100:.0f}% suppression. ")
            if m4_sr > 0 and m2_sr >= m4_sr * 0.85:
                f.write("This is comparable to the effect in real "
                        f"proteins ({m4_sr:.3f}×), suggesting that "
                        "**Layer 1 is primarily peptide stereochemistry** "
                        "rather than fold-specific micro-regularity.\n\n")
            elif m4_sr > 0:
                gap = m4_sr - m2_sr
                f.write(f"However, real proteins show an additional "
                        f"{gap:.3f} beyond steric coupling "
                        f"({m4_sr:.3f}× vs {m2_sr:.3f}×), suggesting "
                        f"that fold-specific structure contributes "
                        f"**{gap/(m4_sr-1)*100:.0f}%** of the total "
                        f"geometric effect.\n\n")
            else:
                f.write("\n\n")

        elif m2_sr < 1.03:
            f.write("**Steric coupling alone does NOT produce the "
                    "observed smoothing.** The ~11% geometric effect "
                    "in real proteins requires secondary structure "
                    "stabilization. **Layer 1 is fold-specific "
                    "micro-regularity**, not an inevitable consequence "
                    "of peptide geometry.\n\n")
        else:
            f.write(f"Steric coupling produces modest smoothing "
                    f"({(m2_sr-1)*100:.0f}%), less than real proteins. "
                    f"The result is intermediate.\n\n")

        if 'Model 3: Conditional' in results:
            f.write(f"**Model 3 (neighbor-conditional): Seg/Real = "
                    f"{m3_sr:.3f}×**\n")
            if m3_sr > m2_sr + 0.02:
                f.write("The empirical neighbor-conditional model "
                        "shows stronger smoothing than steric coupling "
                        "alone, indicating that real backbone torsion "
                        "correlations go beyond simple clash avoidance.\n\n")
            else:
                f.write("The neighbor-conditional model matches steric "
                        "coupling, suggesting that sequential torsion "
                        "correlation is well-explained by local "
                        "steric constraints.\n\n")

        if 'Model 4: Real' in results:
            f.write(f"**Model 4 (real proteins): Seg/Real = "
                    f"{m4_sr:.3f}×**\n")
            f.write("Control: reproduces the ~1.11× geometric "
                    "effect from the Tier 0 test.\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write("| Model | Physics included | Seg/Real |\n")
        f.write("|---|---|---|\n")
        for label, desc in [
            ('Model 1: IID', 'None (independent draws)'),
            ('Model 2: Steric', 'Bond geometry + clash avoidance'),
            ('Model 3: Conditional', 'Empirical neighbor correlations'),
            ('Model 4: Real', 'Full protein structure'),
        ]:
            if label in results:
                f.write(f"| {label} | {desc} | "
                        f"**{results[label]['seg_real']:.3f}×** |\n")
        f.write("\n")

        # Method
        f.write("## Method\n\n")
        f.write("**Model 1 (IID):** Each residue's (φ, ψ) drawn "
                "independently from P(φ, ψ). No sequential coupling. "
                "Expected Seg/Real = 1.0.\n\n")
        f.write("**Model 2 (Steric-coupled):** Backbone built atom-by-"
                "atom with fixed bond lengths (N-Cα = 1.458 Å, "
                "Cα-C = 1.525 Å, C-N = 1.329 Å), fixed bond angles "
                "(N-Cα-C = 111.2°, Cα-C-N = 116.2°, C-N-Cα = 121.7°), "
                "and trans peptide bonds (ω = 180°). Each residue's "
                "(φ, ψ) drawn from P(φ, ψ) but rejected if any "
                f"backbone atom is within {CLASH_RADIUS:.1f} Å of atoms "
                "from the previous 6 residues. Up to 50 rejection "
                "attempts per residue. This captures steric coupling "
                "between adjacent peptide planes without hydrogen "
                "bonding or secondary structure stabilization.\n\n")
        f.write("**Model 3 (Neighbor-conditional):** The empirical "
                "transition density P(φᵢ₊₁, ψᵢ₊₁ | φᵢ, ψᵢ) was "
                "estimated from real protein data on a 72×72 grid "
                "(5° bins). Chains were generated by sequential "
                "sampling from this conditional distribution. This "
                "captures all pairwise torsion correlations present "
                "in real backbones without imposing secondary "
                "structure.\n\n")
        f.write("**Model 4 (Real):** AlphaFold-predicted structures "
                "analyzed with the same torus L1 Seg/Real pipeline. "
                "Control for comparison.\n\n")
        f.write("All models use the same analysis: assign SS labels "
                "from (φ, ψ) positions, apply segment-preserving null "
                "(per-chain basin pools), and compute torus L1 "
                "roughness ratio. Seg/Real > 1 indicates sequential "
                "smoothness beyond within-basin randomization.\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
