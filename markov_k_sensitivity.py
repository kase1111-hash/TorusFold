#!/usr/bin/env python3
"""
MARKOV K-SENSITIVITY SWEEP
============================
Tests how sensitive the Markov transition null result is to the choice of
K (nearest neighbors in the KD-tree transition kernel).

Runs the v2 non-parametric Markov test across K = 5, 10, 20, 50, 100, 200
on a representative subset of organisms (or all 34).

KEY QUESTION:
  If the +0.5% gap at K=50 holds across a wide range of K,
  the result is robust. If it balloons at K=10 or K=200,
  the specific K choice matters and must be reported.

EXPECTED BEHAVIOR:
  K=1:   Near-replay of real trajectories → gap ≈ 0% (trivial)
  K=5:   Very local → slight overfitting, gap near 0%
  K=10:  Still local → gap should be small
  K=50:  Our default → gap = +0.5% (known)
  K=100: More smoothing → gap may increase slightly
  K=200: Heavy smoothing → approaching IID within neighborhoods

Usage:
  python markov_k_sensitivity.py --quick       # 3 organisms, fast
  python markov_k_sensitivity.py               # 6 representative organisms
  python markov_k_sensitivity.py --all         # all 34 organisms (slow)

Output:
  markov_test_results/k_sensitivity_results.csv
  markov_test_results/k_sensitivity_summary.txt
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

K_VALUES = [5, 10, 20, 50, 100, 200]
N_SYNTH_DEFAULT = 30       # chains per protein (lower for speed)
MAX_PROTEINS_DEFAULT = 100  # test proteins per organism
MAX_TRAIN_DEFAULT = 500     # training proteins per organism

# Representative organisms spanning diversity
REPRESENTATIVE_ORGANISMS = [
    "ecoli", "human", "yeast", "arabidopsis", "fly", "tuberculosis"
]

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
            logging.FileHandler(OUTPUT_DIR / "k_sensitivity.log",
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
# CIF PARSER
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
    """Build KD-tree on source angles; return sources, targets, tree."""
    src_list = []
    tgt_list = []
    for seq in all_sequences:
        for k in range(len(seq) - 1):
            src_list.append(seq[k])
            tgt_list.append(seq[k + 1])
    sources = np.array(src_list)
    targets = np.array(tgt_list)
    embedded = np.column_stack([
        np.cos(sources[:, 0]), np.sin(sources[:, 0]),
        np.cos(sources[:, 1]), np.sin(sources[:, 1]),
    ])
    tree = cKDTree(embedded)
    return sources, targets, tree


def embed_angle(phi, psi):
    return np.array([np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)])


def sample_markov_chain(length, sources, targets, tree, K, rng):
    """Generate synthetic chain via non-parametric Markov transitions."""
    start_idx = rng.integers(len(sources))
    phis = np.empty(length)
    psis = np.empty(length)
    phis[0] = sources[start_idx, 0]
    psis[0] = sources[start_idx, 1]

    for k in range(1, length):
        q = embed_angle(phis[k-1], psis[k-1])
        _, indices = tree.query(q, k=K)
        if K == 1:
            indices = [indices]
        chosen = rng.choice(indices)
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
# PRELOAD DATA (parse CIFs once, reuse across K values)
# ============================================================

def preload_organism_data(organism, conn, max_proteins, max_train):
    """Parse all CIF files once. Returns train_sequences, test_data."""
    train_ids = get_protein_ids(conn, organism, max_train)
    test_ids = get_protein_ids(conn, organism, max_proteins)

    if len(train_ids) < 10:
        return None, None

    # Parse training sequences
    train_sequences = []
    for uniprot_id, _ in train_ids:
        cif_path = find_cif_file(uniprot_id, organism)
        if cif_path is None:
            continue
        phi_psi = extract_phi_psi(cif_path, PHI_SIGN)
        if len(phi_psi) >= 10:
            train_sequences.append(phi_psi)

    if len(train_sequences) < 10:
        return None, None

    # Parse test proteins
    test_data = []  # list of (phis, psis, db_bps_norm)
    for uniprot_id, db_bps_norm in test_ids:
        cif_path = find_cif_file(uniprot_id, organism)
        if cif_path is None:
            continue
        phi_psi = extract_phi_psi(cif_path, PHI_SIGN)
        if len(phi_psi) < 10:
            continue
        phis = np.array([pp[0] for pp in phi_psi])
        psis = np.array([pp[1] for pp in phi_psi])
        test_data.append((phis, psis, db_bps_norm))

    if len(test_data) < 5:
        return None, None

    return train_sequences, test_data


def run_k_sweep(organism, train_sequences, test_data, W_interp,
                k_values, n_synth, rng):
    """Run Markov test at multiple K values. Returns list of result dicts."""
    # Build transition data once (KD-tree is the same for all K)
    sources, targets, tree = build_transition_data(train_sequences)
    n_transitions = len(sources)

    # Compute real BPS/L once
    real_bps_values = []
    for phis, psis, _ in test_data:
        real_bps_values.append(compute_bps_l(phis, psis, W_interp))
    real_arr = np.array(real_bps_values)
    real_mean = float(np.mean(real_arr))
    real_sd = float(np.std(real_arr))

    results = []
    for K in k_values:
        t0 = time.time()

        # Clamp K to available transitions
        K_eff = min(K, len(sources))

        markov_bps_values = []
        for phis, psis, _ in test_data:
            L = len(phis)
            protein_markov = []
            for _ in range(n_synth):
                m_phis, m_psis = sample_markov_chain(
                    L, sources, targets, tree, K_eff, rng
                )
                m_bps = compute_bps_l(m_phis, m_psis, W_interp)
                protein_markov.append(m_bps)
            markov_bps_values.append(float(np.mean(protein_markov)))

        markov_arr = np.array(markov_bps_values)
        markov_mean = float(np.mean(markov_arr))
        markov_sd = float(np.std(markov_arr))
        gap_pct = 100 * (markov_mean - real_mean) / real_mean if real_mean > 0 else 0

        elapsed = time.time() - t0

        logging.info(
            f"    K={K:>4}  Markov={markov_mean:.4f}  "
            f"Gap={gap_pct:+.1f}%  SD={markov_sd:.4f}  ({elapsed:.0f}s)"
        )

        results.append({
            'organism': organism,
            'K': K,
            'n_proteins': len(test_data),
            'n_transitions': n_transitions,
            'real_mean': real_mean,
            'real_sd': real_sd,
            'markov_mean': markov_mean,
            'markov_sd': markov_sd,
            'gap_pct': gap_pct,
            'n_synth': n_synth,
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Markov K-sensitivity sweep")
    parser.add_argument("--k-values", nargs="*", type=int, default=K_VALUES,
                        help=f"K values to test (default: {K_VALUES})")
    parser.add_argument("--n-synth", type=int, default=N_SYNTH_DEFAULT)
    parser.add_argument("--max-proteins", type=int, default=MAX_PROTEINS_DEFAULT)
    parser.add_argument("--max-train", type=int, default=MAX_TRAIN_DEFAULT)
    parser.add_argument("--organisms", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 3 organisms, 30 proteins, 10 synth")
    parser.add_argument("--all", action="store_true",
                        help="All 34 organisms (slow)")
    args = parser.parse_args()

    setup_logging()

    if args.quick:
        args.n_synth = 10
        args.max_proteins = 30
        args.max_train = 100
        if args.organisms is None:
            args.organisms = ["ecoli", "human", "yeast"]
    elif not args.all and args.organisms is None:
        args.organisms = REPRESENTATIVE_ORGANISMS

    logging.info("=" * 70)
    logging.info("MARKOV K-SENSITIVITY SWEEP")
    logging.info("How robust is the +0.5% gap to choice of K?")
    logging.info("=" * 70)
    logging.info(f"  K values: {args.k_values}")
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

    all_results = []

    for organism, n_total in organisms:
        logging.info(f"  {organism} ({n_total} total in DB)")
        logging.info(f"    Parsing CIF files...")

        train_seq, test_data = preload_organism_data(
            organism, conn, args.max_proteins, args.max_train
        )
        if train_seq is None:
            logging.warning(f"    Skipping {organism}: insufficient data")
            continue

        real_mean = np.mean([compute_bps_l(p[0], p[1], W_interp) for p in test_data])
        logging.info(f"    N_train={len(train_seq)}  N_test={len(test_data)}  "
                     f"Real={real_mean:.4f}")

        results = run_k_sweep(
            organism, train_seq, test_data, W_interp,
            args.k_values, args.n_synth, rng
        )
        all_results.extend(results)
        logging.info("")

    conn.close()

    if not all_results:
        logging.error("No results.")
        sys.exit(1)

    # ============================================================
    # SUMMARY
    # ============================================================

    logging.info("=" * 70)
    logging.info("K-SENSITIVITY SUMMARY")
    logging.info("=" * 70)
    logging.info("")

    # Aggregate by K
    organisms_tested = sorted(set(r['organism'] for r in all_results))
    logging.info(f"  Organisms: {len(organisms_tested)}")
    logging.info(f"  K values:  {sorted(set(r['K'] for r in all_results))}")
    logging.info("")

    logging.info(f"  {'K':>5}  {'Mean Gap':>10}  {'SD Gap':>8}  "
                 f"{'Min Gap':>8}  {'Max Gap':>8}  {'Markov Mean':>12}")
    logging.info(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")

    for K in sorted(set(r['K'] for r in all_results)):
        k_results = [r for r in all_results if r['K'] == K]
        gaps = np.array([r['gap_pct'] for r in k_results])
        markov_means = np.array([r['markov_mean'] for r in k_results])
        logging.info(
            f"  {K:>5}  {np.mean(gaps):>+9.1f}%  {np.std(gaps):>7.1f}%  "
            f"{np.min(gaps):>+7.1f}%  {np.max(gaps):>+7.1f}%  "
            f"{np.mean(markov_means):>11.4f}"
        )

    logging.info("")

    # Per-organism breakdown
    logging.info("  Per-organism gaps by K:")
    logging.info("")
    header = f"  {'Organism':20s}  {'Real':>7}"
    for K in sorted(set(r['K'] for r in all_results)):
        header += f"  K={K:>3}"
    logging.info(header)
    logging.info(f"  {'-'*20}  {'-'*7}" + "  " + "  ".join(
        ["-" * 7 for _ in set(r['K'] for r in all_results)]))

    for org in organisms_tested:
        org_results = {r['K']: r for r in all_results if r['organism'] == org}
        if not org_results:
            continue
        real_mean = list(org_results.values())[0]['real_mean']
        line = f"  {org:20s}  {real_mean:>7.4f}"
        for K in sorted(set(r['K'] for r in all_results)):
            if K in org_results:
                line += f"  {org_results[K]['gap_pct']:>+6.1f}%"
            else:
                line += "       -"
        logging.info(line)

    logging.info("")

    # VERDICT
    all_gaps_by_k = {}
    for K in sorted(set(r['K'] for r in all_results)):
        gaps = [r['gap_pct'] for r in all_results if r['K'] == K]
        all_gaps_by_k[K] = np.mean(gaps)

    gap_range = max(all_gaps_by_k.values()) - min(all_gaps_by_k.values())

    logging.info("  VERDICT:")
    if gap_range < 10:
        logging.info(f"  Gap varies by only {gap_range:.1f} percentage points across K values.")
        logging.info(f"  The Markov null result is ROBUST to K choice.")
        logging.info(f"  K=50 is not a cherrypicked value.")
    elif gap_range < 25:
        logging.info(f"  Gap varies by {gap_range:.1f} percentage points across K values.")
        logging.info(f"  The result shows MODERATE sensitivity to K.")
        logging.info(f"  Report the K-dependence explicitly in the paper.")
    else:
        logging.info(f"  Gap varies by {gap_range:.1f} percentage points across K values.")
        logging.info(f"  The result is HIGHLY SENSITIVE to K choice.")
        logging.info(f"  The +0.5% gap at K=50 may not be generalizable.")

    logging.info("")

    # ============================================================
    # SAVE
    # ============================================================

    csv_path = OUTPUT_DIR / "k_sensitivity_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'organism', 'K', 'n_proteins', 'n_transitions',
            'real_mean', 'real_sd', 'markov_mean', 'markov_sd',
            'gap_pct', 'n_synth'
        ])
        writer.writeheader()
        writer.writerows(all_results)

    summary_path = OUTPUT_DIR / "k_sensitivity_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MARKOV K-SENSITIVITY SWEEP - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Organisms: {len(organisms_tested)}\n")
        f.write(f"K values: {sorted(set(r['K'] for r in all_results))}\n")
        f.write(f"Synth per protein: {args.n_synth}\n\n")

        f.write(f"{'K':>5}  {'Mean Gap':>10}  {'SD Gap':>8}  "
                f"{'Markov Mean':>12}\n")
        f.write("-" * 45 + "\n")
        for K in sorted(set(r['K'] for r in all_results)):
            k_results = [r for r in all_results if r['K'] == K]
            gaps = np.array([r['gap_pct'] for r in k_results])
            markov_means = np.array([r['markov_mean'] for r in k_results])
            f.write(f"{K:>5}  {np.mean(gaps):>+9.1f}%  {np.std(gaps):>7.1f}%  "
                    f"{np.mean(markov_means):>11.4f}\n")

        f.write(f"\nGap range across K: {gap_range:.1f} pp\n")

        f.write(f"\nPer-organism detail:\n")
        f.write(f"{'Organism':20s}  {'Real':>7}")
        for K in sorted(set(r['K'] for r in all_results)):
            f.write(f"  K={K:>3}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        for org in organisms_tested:
            org_results = {r['K']: r for r in all_results if r['organism'] == org}
            if not org_results:
                continue
            real_mean = list(org_results.values())[0]['real_mean']
            f.write(f"{org:20s}  {real_mean:>7.4f}")
            for K in sorted(set(r['K'] for r in all_results)):
                if K in org_results:
                    f.write(f"  {org_results[K]['gap_pct']:>+6.1f}%")
                else:
                    f.write("       -")
            f.write("\n")

    logging.info(f"  Results: {csv_path}")
    logging.info(f"  Summary: {summary_path}")
    logging.info("")
    logging.info("Done.")


if __name__ == "__main__":
    main()
