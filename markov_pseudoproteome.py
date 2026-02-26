#!/usr/bin/env python3
"""Markov pseudo-proteome generator.

Generates 1,000 Markov pseudo-proteomes by sampling (φ,ψ) sequences from the
empirical transition matrix of the PDB dataset.  Each pseudo-proteome matches
the real proteome's length distribution.  Computes BPS/L for each pseudo-protein
and compares the distribution (mean, CV) to the real data.

Key question: is the CV of pseudo-proteome BPS/L as tight as real (~1.9%),
or substantially wider?

Usage:
    python markov_pseudoproteome.py                  # default: data/*.pdb
    python markov_pseudoproteome.py --data-dir data  # explicit path
    python markov_pseudoproteome.py --n-pseudo 5000  # more pseudo-proteomes
"""

import sys
import math
import time
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import gemmi

sys.path.insert(0, str(Path(__file__).parent))
from bps.superpotential import build_superpotential, lookup_W_batch


# ============================================================
# CONFIGURATION
# ============================================================

RNG_SEED = 42
N_PSEUDO = 1000          # pseudo-proteomes per real proteome
MIN_CHAIN_LEN = 30       # skip very short chains
BASIN_NAMES = ['alpha', 'beta', 'other']


# ============================================================
# DIHEDRAL EXTRACTION (gemmi-based, PDB format)
# ============================================================

def _dihedral_angle(p1, p2, p3, p4):
    """Signed dihedral angle in radians via atan2."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-12:
        return 0.0
    m1 = np.cross(n1, b2 / norm_b2)
    return math.atan2(np.dot(m1, n2), np.dot(n1, n2))


def extract_phi_psi(filepath):
    """Extract (phi, psi) in radians from a PDB file (chain A, first model).

    Returns list of (phi, psi) tuples, or None on failure.
    """
    try:
        st = gemmi.read_structure(str(filepath))
    except Exception:
        return None
    if len(st) == 0 or len(st[0]) == 0:
        return None

    chain = st[0][0]  # first chain of first model
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
            residues.append(atoms)

    if len(residues) < MIN_CHAIN_LEN:
        return None

    phi_psi = []
    for i in range(1, len(residues) - 1):
        phi = -_dihedral_angle(
            residues[i-1]['C'], residues[i]['N'],
            residues[i]['CA'], residues[i]['C'])
        psi = -_dihedral_angle(
            residues[i]['N'], residues[i]['CA'],
            residues[i]['C'], residues[i+1]['N'])
        phi_psi.append((phi, psi))

    if len(phi_psi) < MIN_CHAIN_LEN:
        return None
    return phi_psi


# ============================================================
# BASIN CLASSIFIER (wide definition, consistent with CLAUDE.md)
# ============================================================

def classify_basin(phi_deg, psi_deg):
    """Classify (phi, psi) in degrees into a Ramachandran basin.
    Returns: 0=alpha, 1=beta, 2=other.
    """
    if -160 < phi_deg < 0 and -120 < psi_deg < 30:
        return 0
    if -170 < phi_deg < -70 and (psi_deg > 90 or psi_deg < -120):
        return 1
    return 2


# ============================================================
# CORE: PARSE ALL PROTEINS
# ============================================================

def parse_all_proteins(data_dir, W_grid, phi_grid, psi_grid):
    """Parse all PDB files, compute BPS/L, collect basin statistics.

    Returns:
        proteins: list of dicts with keys {name, L, bps_l, phi_psi}
        transition_counts: (3, 3) int array
        basin_angles: dict basin_id -> np.ndarray of shape (N, 2)
        basin_occupancy: (3,) int array
    """
    pdb_files = sorted(Path(data_dir).glob("*.pdb"))
    logging.info(f"Found {len(pdb_files)} PDB files in {data_dir}")

    n_basins = 3
    transition_counts = np.zeros((n_basins, n_basins), dtype=np.int64)
    basin_occupancy = np.zeros(n_basins, dtype=np.int64)
    basin_angles_list = {b: [] for b in range(n_basins)}
    proteins = []
    n_fail = 0

    for fi, fpath in enumerate(pdb_files):
        phi_psi = extract_phi_psi(fpath)
        if phi_psi is None:
            n_fail += 1
            continue

        L = len(phi_psi)
        phis = np.array([p[0] for p in phi_psi])
        psis = np.array([p[1] for p in phi_psi])

        # BPS/L
        W_vals = lookup_W_batch(W_grid, phi_grid, psi_grid, phis, psis)
        bps_l = float(np.sum(np.abs(np.diff(W_vals)))) / L

        # Basin classification
        phi_deg = np.degrees(phis)
        psi_deg = np.degrees(psis)
        basins = np.array([classify_basin(phi_deg[j], psi_deg[j])
                           for j in range(L)])

        # Accumulate transitions
        for j in range(L - 1):
            transition_counts[basins[j], basins[j + 1]] += 1

        # Accumulate basin occupancy and angle pools
        for j in range(L):
            b = basins[j]
            basin_occupancy[b] += 1
            basin_angles_list[b].append((phis[j], psis[j]))

        proteins.append({
            'name': fpath.stem,
            'L': L,
            'bps_l': bps_l,
        })

        if (fi + 1) % 100 == 0:
            logging.info(f"  Parsed {fi + 1}/{len(pdb_files)} "
                         f"({len(proteins)} OK, {n_fail} failed)")

    logging.info(f"  Parsing complete: {len(proteins)} proteins, {n_fail} failed")

    # Convert angle lists to arrays
    basin_angles = {}
    for b in range(n_basins):
        if basin_angles_list[b]:
            basin_angles[b] = np.array(basin_angles_list[b])
        else:
            # Placeholder for empty basins
            placeholders = {0: (-1.1, -0.44), 1: (-2.1, 2.36), 2: (0.0, 0.0)}
            basin_angles[b] = np.array([placeholders[b]])
            logging.warning(f"  Basin {BASIN_NAMES[b]} empty — using placeholder")

    return proteins, transition_counts, basin_angles, basin_occupancy


# ============================================================
# CORE: MARKOV PSEUDO-PROTEOME GENERATION
# ============================================================

def generate_markov_pseudoproteomes(
    proteins, transition_counts, basin_angles, basin_occupancy,
    W_grid, phi_grid, psi_grid, n_pseudo=N_PSEUDO, rng_seed=RNG_SEED,
):
    """Generate Markov pseudo-proteomes matching real length distribution.

    For each pseudo-proteome (= one copy of the real proteome):
      - For each real protein of length L, generate a synthetic protein of
        length L by:
        1. Sample starting basin from stationary distribution
        2. Walk the Markov chain for L steps
        3. At each step, sample (phi, psi) from that basin's empirical pool
        4. Compute BPS/L on the resulting (phi, psi) sequence

    Returns:
        all_synth_bps: list of N_PSEUDO floats — mean BPS/L per pseudo-proteome
        per_protein_synth: (n_pseudo, n_proteins) array of per-protein BPS/L
    """
    rng = np.random.default_rng(rng_seed)
    n_basins = 3

    # Row-normalize transition matrix
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums > 0, row_sums, 1)
    T_matrix = transition_counts / row_sums_safe

    # Stationary distribution from basin occupancy
    total_residues = basin_occupancy.sum()
    stationary = basin_occupancy / total_residues

    # Cumulative distributions for fast sampling
    T_cumsum = np.cumsum(T_matrix, axis=1)
    stat_cumsum = np.cumsum(stationary)

    logging.info(f"  Transition matrix (row-normalized):")
    for i in range(n_basins):
        row_str = "  ".join(f"{T_matrix[i, j]:.4f}" for j in range(n_basins))
        logging.info(f"    {BASIN_NAMES[i]:>6s} → [{row_str}]")
    logging.info(f"  Stationary: {', '.join(f'{BASIN_NAMES[b]}={stationary[b]:.3f}' for b in range(n_basins))}")

    real_lengths = [p['L'] for p in proteins]
    n_proteins = len(real_lengths)

    all_synth_bps = []        # mean BPS/L per pseudo-proteome
    per_protein_synth = np.empty((n_pseudo, n_proteins))

    t0 = time.time()

    for pi in range(n_pseudo):
        pseudo_bps_vals = []

        for si, L in enumerate(real_lengths):
            # 1. Sample starting basin
            start_basin = min(int(np.searchsorted(stat_cumsum, rng.random())),
                              n_basins - 1)

            # 2. Generate basin sequence via Markov chain
            basins = np.empty(L, dtype=np.int32)
            basins[0] = start_basin
            uniforms = rng.random(L - 1)
            for k in range(1, L):
                prev = basins[k - 1]
                basins[k] = min(int(np.searchsorted(T_cumsum[prev], uniforms[k - 1])),
                                n_basins - 1)

            # 3. Sample (phi, psi) from each basin's empirical pool
            phis = np.empty(L)
            psis = np.empty(L)
            for b in range(n_basins):
                mask = basins == b
                count = int(mask.sum())
                if count == 0:
                    continue
                pool = basin_angles[b]
                indices = rng.integers(0, len(pool), size=count)
                phis[mask] = pool[indices, 0]
                psis[mask] = pool[indices, 1]

            # 4. Compute BPS/L
            W_vals = lookup_W_batch(W_grid, phi_grid, psi_grid, phis, psis)
            dW = np.abs(np.diff(W_vals))
            bps_l = float(np.sum(dW)) / L
            pseudo_bps_vals.append(bps_l)
            per_protein_synth[pi, si] = bps_l

        # Mean BPS/L for this pseudo-proteome
        all_synth_bps.append(float(np.mean(pseudo_bps_vals)))

        if (pi + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (pi + 1) / elapsed
            eta = (n_pseudo - pi - 1) / rate
            logging.info(f"  Pseudo-proteome {pi + 1}/{n_pseudo} "
                         f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed = time.time() - t0
    logging.info(f"  Generation complete: {n_pseudo} pseudo-proteomes in {elapsed:.1f}s")

    return all_synth_bps, per_protein_synth


# ============================================================
# REPORT
# ============================================================

def report(proteins, all_synth_bps, per_protein_synth):
    """Print summary statistics comparing real vs Markov BPS/L."""
    real_bps = np.array([p['bps_l'] for p in proteins])
    synth_bps = np.array(all_synth_bps)

    # --- Real proteome stats ---
    mean_real = float(np.mean(real_bps))
    std_real = float(np.std(real_bps, ddof=1))
    cv_real = std_real / mean_real * 100

    # --- Per-protein Markov stats (pool all synthetic proteins) ---
    synth_all_proteins = per_protein_synth.ravel()
    mean_synth_protein = float(np.mean(synth_all_proteins))
    std_synth_protein = float(np.std(synth_all_proteins, ddof=1))
    cv_synth_protein = std_synth_protein / mean_synth_protein * 100

    # --- Pseudo-proteome mean BPS/L stats ---
    mean_synth_proteome = float(np.mean(synth_bps))
    std_synth_proteome = float(np.std(synth_bps, ddof=1))
    cv_synth_proteome = std_synth_proteome / mean_synth_proteome * 100

    # --- Per-protein real CV vs per-protein synth CV ---
    # For each pseudo-proteome, compute per-protein CV
    per_pseudo_cvs = []
    for pi in range(per_protein_synth.shape[0]):
        row = per_protein_synth[pi]
        m = np.mean(row)
        if m > 0:
            per_pseudo_cvs.append(float(np.std(row, ddof=1) / m * 100))
    mean_per_pseudo_cv = float(np.mean(per_pseudo_cvs)) if per_pseudo_cvs else 0.0

    print()
    print("=" * 70)
    print("MARKOV PSEUDO-PROTEOME ANALYSIS")
    print("=" * 70)
    print()
    print(f"  Real proteins:          {len(proteins)}")
    print(f"  Pseudo-proteomes:       {per_protein_synth.shape[0]}")
    print(f"  Synthetic proteins/pp:  {len(proteins)}")
    print()
    print("-" * 70)
    print("  LEVEL 1: Per-protein BPS/L distribution")
    print("-" * 70)
    print(f"  Real:     mean = {mean_real:.4f},  std = {std_real:.4f},  "
          f"CV = {cv_real:.1f}%")
    print(f"  Markov:   mean = {mean_synth_protein:.4f},  std = {std_synth_protein:.4f},  "
          f"CV = {cv_synth_protein:.1f}%")
    print(f"  Δ mean:   {abs(mean_synth_protein - mean_real):.4f} "
          f"({abs(mean_synth_protein - mean_real)/mean_real*100:.1f}%)")
    print(f"  CV ratio: Markov/Real = {cv_synth_protein/cv_real:.2f}×")
    print()
    print("-" * 70)
    print("  LEVEL 2: Pseudo-proteome mean BPS/L (each = mean over all proteins)")
    print("-" * 70)
    print(f"  Real proteome mean:     {mean_real:.4f}")
    print(f"  Pseudo-proteome means:  {mean_synth_proteome:.4f} ± {std_synth_proteome:.4f}  "
          f"CV = {cv_synth_proteome:.1f}%")
    print()
    print("-" * 70)
    print("  LEVEL 3: Intra-pseudo-proteome CV (tightness)")
    print("-" * 70)
    print(f"  Real proteome CV:          {cv_real:.1f}%")
    print(f"  Mean Markov proteome CV:   {mean_per_pseudo_cv:.1f}%")
    print(f"  CV ratio: Markov/Real =    {mean_per_pseudo_cv/cv_real:.2f}×")
    print()

    # KS test
    try:
        from scipy.stats import ks_2samp
        # Compare real per-protein distribution vs pooled synthetic per-protein
        ks_stat, ks_p = ks_2samp(real_bps, synth_all_proteins)
        print(f"  KS test (real vs Markov per-protein): D = {ks_stat:.4f}, p = {ks_p:.2e}")
    except ImportError:
        pass

    # Verdict
    mean_gap_pct = abs(mean_synth_protein - mean_real) / mean_real * 100
    cv_ratio = cv_synth_protein / cv_real if cv_real > 0 else 0

    print()
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if mean_gap_pct > 20:
        print(f"  Mean BPS/L gap: {mean_gap_pct:.1f}%  (Markov {mean_synth_protein:.3f} "
              f"vs Real {mean_real:.3f})")
        print("  → Markov chain OVER-PREDICTS BPS/L by a large margin.")
        print("    Real proteins suppress intra-basin roughness via secondary")
        print("    structure coherence that first-order transitions cannot capture.")
    elif mean_gap_pct > 5:
        print(f"  Mean BPS/L gap: {mean_gap_pct:.1f}%  (Markov ≈ Real)")
        print("  → Markov chain roughly reproduces the mean.")
    else:
        print(f"  Mean BPS/L gap: {mean_gap_pct:.1f}%  (Markov ≈ Real)")
        print("  → Markov chain accurately reproduces the mean.")
    print()
    if cv_ratio > 1.5:
        print(f"  CV ratio (Markov/Real): {cv_ratio:.2f}×")
        print("  → Markov CV is substantially WIDER than real CV.")
        print("    The tightness of real BPS/L is not explained by transitions alone.")
    elif cv_ratio > 1.2:
        print(f"  CV ratio (Markov/Real): {cv_ratio:.2f}×")
        print("  → Markov CV is moderately wider than real CV.")
    elif cv_ratio < 0.8:
        print(f"  CV ratio (Markov/Real): {cv_ratio:.2f}×")
        print("  → Markov CV is TIGHTER than real CV.")
        print("    Markov sampling averages out the diversity seen in real proteins.")
    else:
        print(f"  CV ratio (Markov/Real): {cv_ratio:.2f}×")
        print("  → Markov CV approximately matches real CV.")
    print("=" * 70)
    print()

    # Length-stratified breakdown
    lengths = np.array([p['L'] for p in proteins])
    bins = [(30, 100, "short (30–100)"),
            (100, 200, "medium (100–200)"),
            (200, 400, "long (200–400)"),
            (400, 10000, "very long (400+)")]

    print("-" * 70)
    print("  LENGTH-STRATIFIED BREAKDOWN")
    print("-" * 70)
    print(f"  {'Bin':<22s}  {'N':>5s}  {'Real mean':>10s}  {'Real CV':>8s}  "
          f"{'Markov CV':>10s}  {'Ratio':>6s}")
    for lo, hi, label in bins:
        mask = (lengths >= lo) & (lengths < hi)
        n = int(mask.sum())
        if n < 5:
            continue
        r = real_bps[mask]
        s = per_protein_synth[:, mask].ravel()
        r_mean = np.mean(r)
        r_cv = np.std(r, ddof=1) / r_mean * 100 if r_mean > 0 else 0
        s_mean = np.mean(s)
        s_cv = np.std(s, ddof=1) / s_mean * 100 if s_mean > 0 else 0
        ratio = s_cv / r_cv if r_cv > 0 else 0
        print(f"  {label:<22s}  {n:>5d}  {r_mean:>10.4f}  {r_cv:>7.1f}%  "
              f"{s_cv:>9.1f}%  {ratio:>5.2f}×")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Markov pseudo-proteome generator")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing PDB files (default: data)")
    parser.add_argument("--n-pseudo", type=int, default=N_PSEUDO,
                        help=f"Number of pseudo-proteomes (default: {N_PSEUDO})")
    parser.add_argument("--seed", type=int, default=RNG_SEED,
                        help=f"RNG seed (default: {RNG_SEED})")
    args = parser.parse_args()

    logging.info("=" * 70)
    logging.info("MARKOV PSEUDO-PROTEOME GENERATOR")
    logging.info("=" * 70)

    # Build superpotential
    logging.info("Building superpotential W(φ,ψ)...")
    W_grid, phi_grid, psi_grid = build_superpotential()
    logging.info(f"  W grid: {W_grid.shape}, range [{W_grid.min():.4f}, {W_grid.max():.4f}]")

    # Parse all proteins
    logging.info("Parsing PDB files...")
    proteins, transition_counts, basin_angles, basin_occupancy = \
        parse_all_proteins(args.data_dir, W_grid, phi_grid, psi_grid)

    if len(proteins) < 10:
        logging.error("Too few proteins parsed. Aborting.")
        sys.exit(1)

    # Real proteome stats
    real_bps = np.array([p['bps_l'] for p in proteins])
    logging.info(f"  Real proteome: N={len(proteins)}, "
                 f"mean BPS/L={np.mean(real_bps):.3f}, "
                 f"std={np.std(real_bps, ddof=1):.3f}, "
                 f"CV={np.std(real_bps, ddof=1)/np.mean(real_bps)*100:.1f}%")

    # Basin stats
    total_res = basin_occupancy.sum()
    for b in range(3):
        logging.info(f"  {BASIN_NAMES[b]:>6s}: {basin_occupancy[b]:>8,d} "
                     f"({basin_occupancy[b]/total_res*100:.1f}%), "
                     f"{len(basin_angles[b])} angle samples")

    # Generate pseudo-proteomes
    logging.info(f"Generating {args.n_pseudo} Markov pseudo-proteomes...")
    all_synth_bps, per_protein_synth = generate_markov_pseudoproteomes(
        proteins, transition_counts, basin_angles, basin_occupancy,
        W_grid, phi_grid, psi_grid,
        n_pseudo=args.n_pseudo,
        rng_seed=args.seed,
    )

    # Report
    report(proteins, all_synth_bps, per_protein_synth)


if __name__ == "__main__":
    main()
