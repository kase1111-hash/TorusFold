#!/usr/bin/env python3
"""
MARKOV TRANSITION NULL TEST v2
===============================
The decisive experiment: does a first-order Markov model trained on
real backbone transitions reproduce BPS/L ≈ 0.20?

METHOD (non-parametric, no binning):
  1. Collect all consecutive (phi_i, psi_i) → (phi_{i+1}, psi_{i+1}) pairs
     from real proteins in an organism
  2. Build a KD-tree on the "source" angles (phi_i, psi_i)
  3. To generate a Markov chain:
     a. Start from a randomly chosen real residue
     b. At each step, find the K nearest neighbors to the current angle
        in the source tree
     c. Randomly pick one of those neighbors
     d. The SUCCESSOR of that neighbor in the real protein becomes the
        next angle in the synthetic chain
  4. This preserves the local transition structure exactly — no binning,
     no discretization, no trapping

The key parameter K controls the "blur" of the transition kernel:
  K=1:   exact replay of a real trajectory fragment (too faithful)
  K=50:  moderate smoothing (preserves local correlations)
  K=200: heavy smoothing (approaches IID from local neighborhoods)

TORUS DISTANCE: angles wrap around, so we use the standard torus
metric: d² = min(|Δφ|, 2π-|Δφ|)² + min(|Δψ|, 2π-|Δψ|)²

Usage:
  python markov_transition_test.py --quick
  python markov_transition_test.py
  python markov_transition_test.py --k-neighbors 50
"""

import os
import sys
import math
import time
import csv
import sqlite3
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("alphafold_bps_results") / "bps_database.db"
CACHE_DIR = Path("alphafold_cache")
OUTPUT_DIR = Path("markov_test_results")
K_NEIGHBORS_DEFAULT = 50   # nearest neighbors for transition kernel
N_SYNTH_DEFAULT = 50       # synthetic chains per protein
MAX_PROTEINS_DEFAULT = 200  # proteins per organism (for Markov chains)
MAX_TRAIN_DEFAULT = 500     # proteins for building transition data

VON_MISES_COMPONENTS = [
    (0.35, -63,  -43,  12.0, 10.0,  2.0),
    (0.05, -60,  -27,   8.0,  6.0,  1.0),
    (0.25, -120,  135,  4.0,  3.5, -1.5),
    (0.05, -140,  155,  5.0,  4.0, -1.0),
    (0.12, -75,   150,  8.0,  5.0,  0.5),
    (0.05, -95,   150,  3.0,  4.0,  0.0),
    (0.03,  57,    40,  6.0,  6.0,  1.5),
    (0.03,  60,  -130,  5.0,  4.0,  0.0),
    (0.01,  75,   -65,  5.0,  5.0,  0.0),
    (0.06,   0,     0,  0.01, 0.01, 0.0),
]

PHI_SIGN = None


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "markov_transition_test.log",
                                mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ============================================================
# SUPERPOTENTIAL
# ============================================================

def build_W_interp(grid_size=360, sigma=1.5):
    angles = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    PHI, PSI = np.meshgrid(angles, angles, indexing="ij")
    p = np.zeros_like(PHI)
    for w, mu_phi_deg, mu_psi_deg, kp, ks, rho in VON_MISES_COMPONENTS:
        mu_phi, mu_psi = np.radians(mu_phi_deg), np.radians(mu_psi_deg)
        exponent = (kp * np.cos(PHI - mu_phi) + ks * np.cos(PSI - mu_psi) +
                    rho * np.sin(PHI - mu_phi) * np.sin(PSI - mu_psi))
        p += w * np.exp(exponent)
    dphi = 2 * np.pi / grid_size
    p /= (p.sum() * dphi * dphi)
    p = np.maximum(p, 1e-6 * p.max())
    W = -np.sqrt(p)
    W = gaussian_filter(W, sigma=sigma, mode="wrap")
    return RegularGridInterpolator(
        (angles, angles), W.T,
        method='linear', bounds_error=False, fill_value=None
    )


# ============================================================
# CIF PARSER (matching bps_process.py)
# ============================================================

def _parse_atoms_from_cif(filepath):
    residues = {}
    with open(filepath, 'r') as f:
        col_map = {}
        in_header = False
        col_idx = 0
        for line in f:
            ls = line.strip()
            if ls.startswith('_atom_site.'):
                col_map[ls.split('.')[1].strip()] = col_idx
                col_idx += 1
                in_header = True
                continue
            if in_header and not ls.startswith('_atom_site.'):
                in_header = False
            if not col_map:
                continue
            if not (ls.startswith('ATOM') or ls.startswith('HETATM')):
                if col_map and ls in ('', '#'):
                    break
                continue
            parts = ls.split()
            try:
                if parts[col_map.get('group_PDB', 0)] != 'ATOM':
                    continue
                atom_name = parts[col_map.get('label_atom_id', 3)]
                if atom_name not in ('N', 'CA', 'C'):
                    continue
                alt = parts[col_map.get('label_alt_id', 4)] if 'label_alt_id' in col_map else '.'
                if alt not in ('.', 'A', '?'):
                    continue
                resnum = int(parts[col_map.get('label_seq_id', 8)])
                x = float(parts[col_map.get('Cartn_x', 10)])
                y = float(parts[col_map.get('Cartn_y', 11)])
                z = float(parts[col_map.get('Cartn_z', 12)])
                if resnum not in residues:
                    residues[resnum] = {}
                residues[resnum][atom_name] = np.array([x, y, z])
            except (ValueError, IndexError, KeyError):
                continue
    result = []
    for rn in sorted(residues.keys()):
        r = residues[rn]
        if 'N' in r and 'CA' in r and 'C' in r:
            result.append(r)
    return result


def _calc_dihedral(p0, p1, p2, p3):
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    nb2 = np.linalg.norm(b2)
    if nb2 < 1e-10:
        return None
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    nn1, nn2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if nn1 < 1e-10 or nn2 < 1e-10:
        return None
    n1 /= nn1
    n2 /= nn2
    m1 = np.cross(n1, b2 / nb2)
    return math.atan2(np.dot(m1, n2), np.dot(n1, n2))


def extract_phi_psi(cif_path, phi_sign):
    residues = _parse_atoms_from_cif(cif_path)
    n = len(residues)
    if n < 10:
        return []
    phi_psi = []
    for i in range(n):
        raw_phi = _calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                                 residues[i]['CA'], residues[i]['C']) if i > 0 else None
        raw_psi = _calc_dihedral(residues[i]['N'], residues[i]['CA'],
                                 residues[i]['C'], residues[i+1]['N']) if i < n-1 else None
        phi = phi_sign * raw_phi if raw_phi is not None else None
        psi = phi_sign * raw_psi if raw_psi is not None else None
        if phi is not None and psi is not None:
            phi_psi.append((phi, psi))
    return phi_psi


# ============================================================
# PHI SIGN
# ============================================================

def determine_phi_sign(n_test=10):
    global PHI_SIGN
    cifs = []
    for org_dir in sorted(CACHE_DIR.iterdir()):
        if not org_dir.is_dir():
            continue
        for cif in org_dir.glob("AF-*-F1-model_v*.cif"):
            if cif.stat().st_size > 1000:
                cifs.append(cif)
                break
        if len(cifs) >= n_test:
            break
    score_pos, score_neg, tested = 0, 0, 0
    for cif_path in cifs:
        residues = _parse_atoms_from_cif(cif_path)
        n = len(residues)
        if n < 30:
            continue
        phis_pos, phis_neg = [], []
        for i in range(1, n):
            raw = _calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                                 residues[i]['CA'], residues[i]['C'])
            if raw is not None:
                phis_pos.append(math.degrees(+1 * raw))
                phis_neg.append(math.degrees(-1 * raw))
        if not phis_pos:
            continue
        arr_pos, arr_neg = np.array(phis_pos), np.array(phis_neg)
        n_alpha_pos = np.sum((arr_pos > -100) & (arr_pos < -20))
        n_alpha_neg = np.sum((arr_neg > -100) & (arr_neg < -20))
        if n_alpha_pos > n_alpha_neg:
            score_pos += 1
        elif n_alpha_neg > n_alpha_pos:
            score_neg += 1
        tested += 1
    if tested == 0:
        logging.error("Cannot determine phi sign")
        sys.exit(1)
    PHI_SIGN = +1 if score_pos >= score_neg else -1
    logging.info(f"  Phi sign: {PHI_SIGN:+d} (tested {tested} files)")


# ============================================================
# NON-PARAMETRIC MARKOV TRANSITION MODEL
# ============================================================

def build_transition_data(all_sequences):
    """Collect all consecutive angle pairs from real proteins.

    Returns:
        sources: (N, 2) array of (phi_i, psi_i) for each transition
        targets: (N, 2) array of (phi_{i+1}, psi_{i+1}) — the successor
        source_tree: cKDTree on the "unwrapped" source coordinates

    For KD-tree compatibility on the torus, we embed angles as
    (cos(phi), sin(phi), cos(psi), sin(psi)) in R^4.
    Euclidean distance in this embedding approximates torus distance.
    """
    src_list = []
    tgt_list = []

    for seq in all_sequences:
        for k in range(len(seq) - 1):
            src_list.append(seq[k])
            tgt_list.append(seq[k + 1])

    sources = np.array(src_list)  # (N, 2) in radians
    targets = np.array(tgt_list)  # (N, 2) in radians

    # Embed on torus for KD-tree
    embedded = np.column_stack([
        np.cos(sources[:, 0]), np.sin(sources[:, 0]),
        np.cos(sources[:, 1]), np.sin(sources[:, 1]),
    ])
    tree = cKDTree(embedded)

    return sources, targets, tree


def embed_angle(phi, psi):
    """Embed a single (phi, psi) into R^4 for tree query."""
    return np.array([np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)])


def sample_markov_chain(length, sources, targets, tree, K, rng):
    """Generate a synthetic chain using non-parametric Markov transitions.

    At each step:
      1. Embed current (phi, psi) in R^4
      2. Find K nearest neighbors in the source tree
      3. Pick one at random
      4. Use its TARGET (successor in the real protein) as the next angle
    """
    # Start from a random real residue
    start_idx = rng.integers(len(sources))
    phis = np.empty(length)
    psis = np.empty(length)
    phis[0] = sources[start_idx, 0]
    psis[0] = sources[start_idx, 1]

    for k in range(1, length):
        # Query K nearest neighbors to current angle
        q = embed_angle(phis[k-1], psis[k-1])
        _, indices = tree.query(q, k=K)
        if K == 1:
            indices = [indices]
        # Pick one neighbor at random
        chosen = rng.choice(indices)
        # Use the TARGET of that neighbor
        phis[k] = targets[chosen, 0]
        psis[k] = targets[chosen, 1]

    return phis, psis


# ============================================================
# BPS/L
# ============================================================

def compute_bps_l(phis, psis, W_interp):
    W_vals = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_vals))
    return float(np.sum(dW)) / len(phis)


# ============================================================
# DATABASE + FILES
# ============================================================

def get_organisms(conn, filter_organisms=None):
    cur = conn.execute(
        "SELECT organism, COUNT(*) FROM proteins WHERE bps_norm IS NOT NULL "
        "GROUP BY organism ORDER BY COUNT(*)"
    )
    orgs = [(row[0], row[1]) for row in cur.fetchall()]
    if filter_organisms:
        orgs = [(o, n) for o, n in orgs if o in filter_organisms]
    return orgs


def get_protein_ids(conn, organism, max_proteins):
    cur = conn.execute(
        "SELECT uniprot_id, bps_norm FROM proteins WHERE organism = ? "
        "AND bps_norm IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        (organism, max_proteins)
    )
    return [(row[0], row[1]) for row in cur.fetchall()]


def find_cif_file(uniprot_id, organism):
    org_dir = CACHE_DIR / organism
    if org_dir.exists():
        for cif in org_dir.glob(f"AF-{uniprot_id}-F1-model_v*.cif"):
            return cif
    for subdir in CACHE_DIR.iterdir():
        if subdir.is_dir():
            for cif in subdir.glob(f"AF-{uniprot_id}-F1-model_v*.cif"):
                return cif
    return None


# ============================================================
# PER-ORGANISM TEST
# ============================================================

def run_organism_test(organism, conn, W_interp, K, n_synth,
                      max_proteins, max_train, rng):
    t0 = time.time()

    # Get training proteins (for building transition data)
    train_ids = get_protein_ids(conn, organism, max_train)
    # Get test proteins (for computing real BPS/L and generating Markov)
    test_ids = get_protein_ids(conn, organism, max_proteins)

    if len(train_ids) < 10:
        logging.warning(f"  {organism}: only {len(train_ids)} proteins, skipping")
        return None

    # Phase 1: Extract sequences for training
    train_sequences = []
    for uniprot_id, _ in train_ids:
        cif_path = find_cif_file(uniprot_id, organism)
        if cif_path is None:
            continue
        phi_psi = extract_phi_psi(cif_path, PHI_SIGN)
        if len(phi_psi) >= 10:
            train_sequences.append(phi_psi)

    if len(train_sequences) < 10:
        logging.warning(f"  {organism}: insufficient training data")
        return None

    # Phase 2: Build non-parametric transition model
    sources, targets, tree = build_transition_data(train_sequences)
    n_transitions = len(sources)

    # Phase 3: Extract test proteins and compute real + Markov BPS/L
    real_bps_values = []
    markov_bps_values = []
    db_bps_values = []
    n_parsed = 0

    for uniprot_id, db_bps_norm in test_ids:
        cif_path = find_cif_file(uniprot_id, organism)
        if cif_path is None:
            continue
        phi_psi = extract_phi_psi(cif_path, PHI_SIGN)
        if len(phi_psi) < 10:
            continue

        phis = np.array([pp[0] for pp in phi_psi])
        psis = np.array([pp[1] for pp in phi_psi])
        L = len(phis)

        # Real BPS/L
        real_bps = compute_bps_l(phis, psis, W_interp)
        real_bps_values.append(real_bps)
        db_bps_values.append(db_bps_norm)

        # Markov BPS/L (averaged over n_synth chains)
        protein_markov = []
        for _ in range(n_synth):
            m_phis, m_psis = sample_markov_chain(
                L, sources, targets, tree, K, rng
            )
            m_bps = compute_bps_l(m_phis, m_psis, W_interp)
            protein_markov.append(m_bps)
        markov_bps_values.append(float(np.mean(protein_markov)))
        n_parsed += 1

    if n_parsed < 5:
        logging.warning(f"  {organism}: only {n_parsed} test proteins")
        return None

    real_arr = np.array(real_bps_values)
    markov_arr = np.array(markov_bps_values)
    db_arr = np.array(db_bps_values)

    real_mean = float(np.mean(real_arr))
    markov_mean = float(np.mean(markov_arr))
    db_mean = float(np.mean(db_arr))
    gap_pct = 100 * (markov_mean - real_mean) / real_mean if real_mean > 0 else 0
    db_corr = float(np.corrcoef(real_arr, db_arr)[0, 1]) if len(real_arr) > 2 else 0

    elapsed = time.time() - t0

    logging.info(
        f"  {organism:20s}  N={n_parsed:>5}  "
        f"Real={real_mean:.4f}  Markov={markov_mean:.4f}  "
        f"Gap={gap_pct:+.1f}%  "
        f"DB={db_mean:.4f}  r={db_corr:.3f}  "
        f"Trans={n_transitions:,}  K={K}  "
        f"({elapsed:.0f}s)"
    )

    return {
        'organism': organism,
        'n_proteins': n_parsed,
        'n_train': len(train_sequences),
        'n_transitions': n_transitions,
        'real_mean': real_mean,
        'markov_mean': markov_mean,
        'gap_pct': gap_pct,
        'db_mean': db_mean,
        'db_corr': db_corr,
        'K': K,
        'n_synth': n_synth,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Markov transition null v2")
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS_DEFAULT,
                        help="Nearest neighbors for transition kernel (default: 50)")
    parser.add_argument("--n-synth", type=int, default=N_SYNTH_DEFAULT)
    parser.add_argument("--max-proteins", type=int, default=MAX_PROTEINS_DEFAULT)
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN_DEFAULT)
    parser.add_argument("--organisms", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 3 organisms, 30 proteins, 10 synth")
    args = parser.parse_args()

    setup_logging()

    if args.quick:
        args.n_synth = 10
        args.max_proteins = 30
        args.max_train = 100
        if args.organisms is None:
            args.organisms = ["ecoli", "human", "yeast"]

    logging.info("=" * 70)
    logging.info("MARKOV TRANSITION NULL TEST v2 (non-parametric)")
    logging.info("Does nearest-neighbor correlation explain BPS/L?")
    logging.info("=" * 70)
    logging.info(f"  K nearest neighbors: {args.k_neighbors}")
    logging.info(f"  Synthetic chains per protein: {args.n_synth}")
    logging.info(f"  Max test proteins: {args.max_proteins}")
    logging.info(f"  Max training proteins: {args.max_train}")
    logging.info(f"  Seed: {args.seed}")
    logging.info("")

    if not DB_PATH.exists():
        logging.error(f"Database not found: {DB_PATH}")
        sys.exit(1)
    if not CACHE_DIR.exists():
        logging.error(f"Cache not found: {CACHE_DIR}")
        sys.exit(1)

    logging.info("Building superpotential...")
    W_interp = build_W_interp()

    logging.info("Determining phi sign...")
    determine_phi_sign()

    rng = np.random.default_rng(args.seed)

    conn = sqlite3.connect(str(DB_PATH))
    organisms = get_organisms(conn, args.organisms)
    logging.info(f"  Organisms: {len(organisms)}")
    logging.info("")
    logging.info(f"  {'Organism':20s}  {'N':>5}  {'Real':>7}  {'Markov':>7}  "
                 f"{'Gap':>6}  {'DB':>7}  {'r':>5}  {'Transitions':>12}")
    logging.info(f"  {'-'*20}  {'-'*5}  {'-'*7}  {'-'*7}  "
                 f"{'-'*6}  {'-'*7}  {'-'*5}  {'-'*12}")

    results = []
    for organism, n_proteins in organisms:
        result = run_organism_test(
            organism, conn, W_interp, args.k_neighbors, args.n_synth,
            args.max_proteins, args.max_train, rng
        )
        if result:
            results.append(result)

    conn.close()

    if not results:
        logging.error("No results.")
        sys.exit(1)

    # ============================================================
    # SUMMARY
    # ============================================================

    real_means = np.array([r['real_mean'] for r in results])
    markov_means = np.array([r['markov_mean'] for r in results])
    gaps = np.array([r['gap_pct'] for r in results])

    mean_gap = float(np.mean(gaps))

    logging.info("")
    logging.info("=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)
    logging.info("")
    logging.info(f"  Organisms tested: {len(results)}")
    logging.info(f"  Total proteins: {sum(r['n_proteins'] for r in results):,}")
    logging.info(f"  K neighbors: {args.k_neighbors}")
    logging.info("")
    logging.info(f"  REAL:")
    logging.info(f"    Mean: {np.mean(real_means):.4f}")
    logging.info(f"    CV:   {100*np.std(real_means)/np.mean(real_means):.1f}%")
    logging.info("")
    logging.info(f"  MARKOV (nearest-neighbor transitions, K={args.k_neighbors}):")
    logging.info(f"    Mean: {np.mean(markov_means):.4f}")
    logging.info(f"    CV:   {100*np.std(markov_means)/np.mean(markov_means):.1f}%")
    logging.info("")
    logging.info(f"  GAP: {mean_gap:+.1f}% [{np.min(gaps):+.1f}%, {np.max(gaps):+.1f}%]")
    logging.info("")
    logging.info("  FOUR-LEVEL HIERARCHY:")
    logging.info(f"    Real:        {np.mean(real_means):.4f}")
    logging.info(f"    Markov:      {np.mean(markov_means):.4f}")
    logging.info(f"    IID:         0.3464")
    logging.info(f"    Permutation: 0.5156")
    logging.info("")

    # VERDICT
    logging.info("  VERDICT:")
    if mean_gap < 5:
        logging.info(f"  Markov gap = {mean_gap:+.1f}%: Real ≈ Markov")
        logging.info(f"  → BPS/L is emergent from nearest-neighbor dynamics.")
        logging.info(f"  → The constant does NOT require evolutionary selection")
        logging.info(f"    beyond local backbone correlations.")
    elif mean_gap < 30:
        logging.info(f"  Markov gap = {mean_gap:+.1f}%: Real < Markov (moderate)")
        logging.info(f"  → Nearest-neighbor correlations explain most smoothness.")
        logging.info(f"  → Some longer-range organization is present.")
    else:
        logging.info(f"  Markov gap = {mean_gap:+.1f}%: Real << Markov")
        logging.info(f"  → Local transitions alone cannot explain BPS/L.")
        logging.info(f"  → Evolution minimizes beyond nearest-neighbor physics.")

    # SAVE
    csv_path = OUTPUT_DIR / "markov_transition_v2_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'organism', 'n_proteins', 'n_train', 'n_transitions',
            'real_mean', 'markov_mean', 'gap_pct', 'db_mean', 'db_corr',
            'K', 'n_synth'
        ])
        writer.writeheader()
        writer.writerows(results)

    summary_path = OUTPUT_DIR / "markov_transition_v2_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MARKOV TRANSITION NULL TEST v2 - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"K neighbors: {args.k_neighbors}\n")
        f.write(f"Organisms: {len(results)}\n\n")
        f.write(f"Real mean:   {np.mean(real_means):.4f}\n")
        f.write(f"Markov mean: {np.mean(markov_means):.4f}\n")
        f.write(f"Gap: {mean_gap:+.1f}%\n\n")
        f.write(f"{'Organism':20s} {'N':>5} {'Real':>8} {'Markov':>8} "
                f"{'Gap':>7} {'r':>6}\n")
        f.write("-" * 60 + "\n")
        for r in sorted(results, key=lambda x: x['gap_pct']):
            f.write(f"{r['organism']:20s} {r['n_proteins']:>5} "
                    f"{r['real_mean']:>8.4f} {r['markov_mean']:>8.4f} "
                    f"{r['gap_pct']:>+6.1f}% {r['db_corr']:>5.3f}\n")

    logging.info(f"  Results: {csv_path}")
    logging.info(f"  Summary: {summary_path}")
    logging.info("Done.")


if __name__ == "__main__":
    main()
