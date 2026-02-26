#!/usr/bin/env python3
"""
PERMUTATION NULL TEST (v1)
===========================
The airtight control: shuffle each protein's actual (phi, psi) sequence
and recompute BPS/L. No model assumptions whatsoever.

This tests whether sequential ORDER matters, using each protein as its
own control. The shuffled version has the exact same set of angles —
same empirical Ramachandran distribution — but random order.

EXPECTED:
  - If the IID null result (+100% gap) is real:
      shuffled BPS/L ~ 0.35 (matching the von Mises IID result)
  - If the IID gap was a model artifact:
      shuffled BPS/L ~ 0.17 (close to real)

This script:
  1. Reads CIF files from alphafold_cache/
  2. Extracts (phi, psi) sequences
  3. Computes BPS/L on real sequence
  4. Shuffles the (phi, psi) PAIRS and recomputes BPS/L
  5. Repeats shuffle N times per protein
  6. Reports real vs shuffled gap

Usage:
  python permutation_null_test.py
  python permutation_null_test.py --organisms ecoli human yeast
  python permutation_null_test.py --n-shuffles 50 --max-proteins 200
  python permutation_null_test.py --quick   # 3 organisms, 50 proteins, 10 shuffles
"""

import os
import sys
import math
import time
import sqlite3
import argparse
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("alphafold_bps_results") / "bps_database.db"
CACHE_DIR = Path("alphafold_cache")
OUTPUT_DIR = Path("markov_test_results")
N_SHUFFLES_DEFAULT = 100
MAX_PROTEINS_DEFAULT = 500

# Von Mises components (same as markov test — only used to build W)
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

# This gets set by determine_phi_sign() before processing
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
            logging.FileHandler(OUTPUT_DIR / "permutation_test.log", mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ============================================================
# SUPERPOTENTIAL (self-contained, same as markov test v4)
# ============================================================

def build_W_interp(grid_size=360, sigma=1.5):
    """Build W = -sqrt(P) interpolator. Returns W_interp that takes (psi_rad, phi_rad)."""
    angles = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    PHI, PSI = np.meshgrid(angles, angles, indexing="ij")

    p = np.zeros_like(PHI)
    for w, mu_phi_deg, mu_psi_deg, kp, ks, rho in VON_MISES_COMPONENTS:
        mu_phi = np.radians(mu_phi_deg)
        mu_psi = np.radians(mu_psi_deg)
        exponent = (kp * np.cos(PHI - mu_phi) +
                    ks * np.cos(PSI - mu_psi) +
                    rho * np.sin(PHI - mu_phi) * np.sin(PSI - mu_psi))
        p += w * np.exp(exponent)

    dphi = 2 * np.pi / grid_size
    p /= (p.sum() * dphi * dphi)
    p = np.maximum(p, 1e-6 * p.max())

    W = -np.sqrt(p)
    W = gaussian_filter(W, sigma=sigma, mode="wrap")

    W_interp = RegularGridInterpolator(
        (angles, angles), W.T,
        method='linear', bounds_error=False, fill_value=None
    )
    return W_interp


# ============================================================
# CIF PARSER (copied from bps_process.py for self-containment)
# ============================================================

def _parse_atoms_from_cif(filepath):
    """Parse backbone atoms from mmCIF file. Returns dict of residues."""
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
                chain = parts[col_map.get('label_asym_id', 6)]
                seq_id = int(parts[col_map.get('label_seq_id', 8)])
                x = float(parts[col_map.get('Cartn_x', 10)])
                y = float(parts[col_map.get('Cartn_y', 11)])
                z = float(parts[col_map.get('Cartn_z', 12)])
                key = (chain, seq_id)
                if key not in residues:
                    residues[key] = {}
                residues[key][atom_name] = np.array([x, y, z])
            except (IndexError, ValueError):
                continue
    return residues


def _calc_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle in radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return None
    n1 /= n1_norm
    n2 /= n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return math.atan2(y, x)


def extract_phi_psi(cif_path, phi_sign):
    """Extract (phi, psi) sequence from CIF file. Returns list of (phi, psi) in radians."""
    residues = _parse_atoms_from_cif(cif_path)
    if not residues:
        return []

    # Sort by (chain, seq_id)
    sorted_keys = sorted(residues.keys())

    # Group by chain
    chains = {}
    for key in sorted_keys:
        chain_id = key[0]
        if chain_id not in chains:
            chains[chain_id] = []
        atoms = residues[key]
        if 'N' in atoms and 'CA' in atoms and 'C' in atoms:
            chains[chain_id].append((key[1], atoms))

    # Use longest chain
    if not chains:
        return []
    longest_chain = max(chains.values(), key=len)

    phi_psi = []
    for i in range(len(longest_chain)):
        phi = None
        psi = None

        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            prev = longest_chain[i-1][1]
            curr = longest_chain[i][1]
            if 'C' in prev and 'N' in curr and 'CA' in curr and 'C' in curr:
                raw = _calc_dihedral(prev['C'], curr['N'], curr['CA'], curr['C'])
                if raw is not None:
                    phi = phi_sign * raw

        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        if i < len(longest_chain) - 1:
            curr = longest_chain[i][1]
            nxt = longest_chain[i+1][1]
            if 'N' in curr and 'CA' in curr and 'C' in curr and 'N' in nxt:
                raw = _calc_dihedral(curr['N'], curr['CA'], curr['C'], nxt['N'])
                if raw is not None:
                    psi = phi_sign * raw

        if phi is not None and psi is not None:
            phi_psi.append((phi, psi))

    return phi_psi


# ============================================================
# PHI SIGN DETERMINATION (same logic as bps_process.py)
# ============================================================

def determine_phi_sign(cif_paths, n_test=10):
    """Test both +1 and -1 phi sign on a sample of CIF files."""
    global PHI_SIGN

    tested = 0
    score_pos = 0
    score_neg = 0

    for cif_path in cif_paths[:n_test]:
        residues = _parse_atoms_from_cif(cif_path)
        sorted_keys = sorted(residues.keys())
        chains = {}
        for key in sorted_keys:
            chain_id = key[0]
            if chain_id not in chains:
                chains[chain_id] = []
            atoms = residues[key]
            if 'N' in atoms and 'CA' in atoms and 'C' in atoms:
                chains[chain_id].append((key[1], atoms))

        if not chains:
            continue
        longest = max(chains.values(), key=len)

        phis_pos = []
        phis_neg = []
        for i in range(1, len(longest)):
            prev = longest[i-1][1]
            curr = longest[i][1]
            if 'C' in prev and 'N' in curr and 'CA' in curr and 'C' in curr:
                raw = _calc_dihedral(prev['C'], curr['N'], curr['CA'], curr['C'])
                if raw is not None:
                    phis_pos.append(math.degrees(+1 * raw))
                    phis_neg.append(math.degrees(-1 * raw))

        if not phis_pos:
            continue

        arr_pos = np.array(phis_pos)
        arr_neg = np.array(phis_neg)

        # Alpha region: phi in [-100, -20]
        n_alpha_pos = np.sum((arr_pos > -100) & (arr_pos < -20))
        n_alpha_neg = np.sum((arr_neg > -100) & (arr_neg < -20))

        if n_alpha_pos > n_alpha_neg:
            score_pos += 1
        elif n_alpha_neg > n_alpha_pos:
            score_neg += 1
        tested += 1

    if tested == 0:
        logging.error("Could not determine phi sign — no valid CIF files")
        sys.exit(1)

    if score_pos >= score_neg:
        PHI_SIGN = +1
    else:
        PHI_SIGN = -1

    logging.info(f"  Phi sign: {PHI_SIGN:+d} (tested {tested} files, +1 won {score_pos}/{tested})")
    return PHI_SIGN


# ============================================================
# BPS/L COMPUTATION
# ============================================================

def compute_bps_l(phis, psis, W_interp):
    """Compute BPS/L from arrays of phi, psi in radians."""
    W_vals = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_vals))
    L = len(phis)
    return float(np.sum(dW)) / L


# ============================================================
# DATABASE + FILE DISCOVERY
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
    """Get uniprot IDs for an organism that have computed BPS values."""
    cur = conn.execute(
        "SELECT uniprot_id, bps_norm FROM proteins WHERE organism = ? AND bps_norm IS NOT NULL "
        "ORDER BY RANDOM() LIMIT ?",
        (organism, max_proteins)
    )
    return [(row[0], row[1]) for row in cur.fetchall()]


def find_cif_file(uniprot_id, organism=None):
    """Find CIF file for a uniprot ID in the cache directory.
    Layout: alphafold_cache/{organism}/AF-{uid}-F1-model_v*.cif
    """
    # If organism is known, search there first
    if organism:
        org_dir = CACHE_DIR / organism
        if org_dir.exists():
            for cif in org_dir.glob(f"AF-{uniprot_id}-F1-model_v*.cif"):
                return cif

    # Search all organism subdirectories
    for subdir in CACHE_DIR.iterdir():
        if subdir.is_dir():
            for cif in subdir.glob(f"AF-{uniprot_id}-F1-model_v*.cif"):
                return cif

    # Also check top-level (just in case)
    for cif in CACHE_DIR.glob(f"AF-{uniprot_id}-F1-model_v*.cif"):
        return cif

    return None


# ============================================================
# PER-ORGANISM TEST
# ============================================================

def run_organism_test(organism, conn, W_interp, n_shuffles, max_proteins, rng):
    t0 = time.time()

    proteins = get_protein_ids(conn, organism, max_proteins)
    if len(proteins) < 5:
        logging.warning(f"  {organism}: only {len(proteins)} proteins, skipping")
        return None

    real_bps_values = []
    shuffled_means = []  # mean of shuffled BPS/L per protein
    db_bps_values = []
    n_parsed = 0
    n_skipped = 0

    for uniprot_id, db_bps_norm in proteins:
        cif_path = find_cif_file(uniprot_id, organism)
        if cif_path is None:
            n_skipped += 1
            continue

        phi_psi = extract_phi_psi(cif_path, PHI_SIGN)
        if len(phi_psi) < 10:
            n_skipped += 1
            continue

        phis = np.array([pp[0] for pp in phi_psi])
        psis = np.array([pp[1] for pp in phi_psi])

        # Real BPS/L (from actual angle sequence)
        real_bps = compute_bps_l(phis, psis, W_interp)
        real_bps_values.append(real_bps)
        db_bps_values.append(db_bps_norm)

        # Shuffled BPS/L: randomly permute the (phi, psi) PAIRS
        protein_shuffled = []
        for _ in range(n_shuffles):
            idx = rng.permutation(len(phis))
            shuf_bps = compute_bps_l(phis[idx], psis[idx], W_interp)
            protein_shuffled.append(shuf_bps)
        shuffled_means.append(float(np.mean(protein_shuffled)))
        n_parsed += 1

    if n_parsed < 5:
        logging.warning(f"  {organism}: only {n_parsed} parsed, skipping")
        return None

    real_arr = np.array(real_bps_values)
    shuf_arr = np.array(shuffled_means)
    db_arr = np.array(db_bps_values)

    real_mean = float(np.mean(real_arr))
    shuf_mean = float(np.mean(shuf_arr))
    db_mean = float(np.mean(db_arr))

    gap_pct = 100 * (shuf_mean - real_mean) / real_mean if real_mean > 0 else 0

    # Cross-check: our real BPS/L vs database BPS/L
    db_corr = float(np.corrcoef(real_arr, db_arr)[0, 1]) if len(real_arr) > 2 else 0
    db_diff = abs(real_mean - db_mean)

    elapsed = time.time() - t0

    logging.info(
        f"  {organism:20s}  N={n_parsed:>5}  "
        f"Real={real_mean:.4f}  Shuffled={shuf_mean:.4f}  "
        f"Gap={gap_pct:+.1f}%  "
        f"DB={db_mean:.4f}  r={db_corr:.3f}  "
        f"({elapsed:.0f}s)"
    )

    return {
        'organism': organism,
        'n_proteins': n_parsed,
        'n_skipped': n_skipped,
        'real_mean': real_mean,
        'shuf_mean': shuf_mean,
        'gap_pct': gap_pct,
        'db_mean': db_mean,
        'db_corr': db_corr,
        'db_diff': db_diff,
        'real_cv': 100 * float(np.std(real_arr)) / real_mean if real_mean > 0 else 0,
        'shuf_cv': 100 * float(np.std(shuf_arr)) / shuf_mean if shuf_mean > 0 else 0,
        'n_shuffles': n_shuffles,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Permutation null test")
    parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES_DEFAULT)
    parser.add_argument("--max-proteins", type=int, default=MAX_PROTEINS_DEFAULT)
    parser.add_argument("--organisms", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 3 organisms, 50 proteins, 10 shuffles")
    args = parser.parse_args()

    setup_logging()

    if args.quick:
        args.n_shuffles = 10
        args.max_proteins = 50
        if args.organisms is None:
            args.organisms = ["ecoli", "human", "yeast"]

    logging.info("=" * 70)
    logging.info("PERMUTATION NULL TEST v1")
    logging.info("The airtight control: shuffle actual angles, recompute BPS/L")
    logging.info("=" * 70)
    logging.info(f"  Shuffles per protein: {args.n_shuffles}")
    logging.info(f"  Max proteins per organism: {args.max_proteins}")
    logging.info(f"  Database: {DB_PATH}")
    logging.info(f"  Cache: {CACHE_DIR}")
    logging.info(f"  Seed: {args.seed}")
    logging.info("")

    if not DB_PATH.exists():
        logging.error(f"Database not found: {DB_PATH}")
        sys.exit(1)
    if not CACHE_DIR.exists():
        logging.error(f"Cache directory not found: {CACHE_DIR}")
        sys.exit(1)

    # Build superpotential
    logging.info("Building superpotential...")
    W_interp = build_W_interp()
    logging.info("  Done.")

    # Find CIF files for phi sign determination
    logging.info("Determining phi sign convention...")
    sample_cifs = []
    for p in CACHE_DIR.rglob("*.cif"):
        sample_cifs.append(p)
        if len(sample_cifs) >= 20:
            break
    if not sample_cifs:
        logging.error("No CIF files found in cache!")
        sys.exit(1)
    determine_phi_sign(sample_cifs)

    rng = np.random.default_rng(args.seed)

    # Database
    conn = sqlite3.connect(str(DB_PATH))
    organisms = get_organisms(conn, args.organisms)
    logging.info(f"  Organisms: {len(organisms)}")
    logging.info("")
    logging.info(f"  {'Organism':20s}  {'N':>5}  {'Real':>7}  {'Shuf':>7}  "
                 f"{'Gap':>6}  {'DB':>7}  {'r':>5}")
    logging.info(f"  {'-'*20}  {'-'*5}  {'-'*7}  {'-'*7}  "
                 f"{'-'*6}  {'-'*7}  {'-'*5}")

    results = []
    for organism, n_proteins in organisms:
        result = run_organism_test(
            organism, conn, W_interp, args.n_shuffles, args.max_proteins, rng
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
    shuf_means = np.array([r['shuf_mean'] for r in results])
    db_means = np.array([r['db_mean'] for r in results])
    gaps = np.array([r['gap_pct'] for r in results])
    db_corrs = np.array([r['db_corr'] for r in results])

    mean_gap = float(np.mean(gaps))

    logging.info("")
    logging.info("=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)
    logging.info("")
    logging.info(f"  Organisms tested: {len(results)}")
    logging.info(f"  Total proteins: {sum(r['n_proteins'] for r in results):,}")
    logging.info(f"  Total skipped (no CIF): {sum(r['n_skipped'] for r in results):,}")
    logging.info("")
    logging.info(f"  REAL (sequential order):")
    logging.info(f"    Mean of organism means: {np.mean(real_means):.4f}")
    logging.info(f"    Range: [{np.min(real_means):.4f}, {np.max(real_means):.4f}]")
    logging.info("")
    logging.info(f"  SHUFFLED (random order, same angles):")
    logging.info(f"    Mean of organism means: {np.mean(shuf_means):.4f}")
    logging.info(f"    Range: [{np.min(shuf_means):.4f}, {np.max(shuf_means):.4f}]")
    logging.info("")
    logging.info(f"  GAP (Shuffled - Real) / Real:")
    logging.info(f"    Mean gap:  {mean_gap:+.1f}%")
    logging.info(f"    Gap range: [{np.min(gaps):+.1f}%, {np.max(gaps):+.1f}%]")
    logging.info(f"    Gap SD:    {np.std(gaps):.1f}%")
    logging.info("")
    logging.info(f"  DATABASE CROSS-CHECK:")
    logging.info(f"    Our real mean: {np.mean(real_means):.4f}")
    logging.info(f"    DB real mean:  {np.mean(db_means):.4f}")
    logging.info(f"    Diff: {abs(np.mean(real_means) - np.mean(db_means)):.4f}")
    logging.info(f"    Mean r(ours, DB): {np.mean(db_corrs):.3f}")
    logging.info("")

    # INTERPRETATION
    logging.info("  INTERPRETATION:")
    logging.info("")

    if mean_gap > 50:
        logging.info(f"  SHUFFLED GAP = {mean_gap:+.0f}%")
        logging.info(f"  This CONFIRMS the IID null result.")
        logging.info(f"  Sequential order reduces BPS/L by ~{abs(mean_gap):.0f}% vs random order.")
        logging.info(f"  The gap is NOT a model artifact — it persists with the protein's")
        logging.info(f"  own angles. Biology actively organizes backbone trajectories.")
    elif mean_gap > 10:
        logging.info(f"  SHUFFLED GAP = {mean_gap:+.0f}%")
        logging.info(f"  Moderate gap — sequential order matters but less than IID null suggested.")
        logging.info(f"  The von Mises model may overestimate the gap slightly.")
    else:
        logging.info(f"  SHUFFLED GAP = {mean_gap:+.1f}%")
        logging.info(f"  WARNING: Small gap — the IID null result may have been a model artifact.")
        logging.info(f"  The von Mises distribution may not match real Ramachandran coverage.")

    logging.info("")

    if np.mean(db_corrs) > 0.95:
        logging.info(f"  DB CROSS-CHECK: r={np.mean(db_corrs):.3f} — our W matches the pipeline's W.")
    elif np.mean(db_corrs) > 0.8:
        logging.info(f"  DB CROSS-CHECK: r={np.mean(db_corrs):.3f} — mostly matches, minor differences.")
    else:
        logging.info(f"  DB CROSS-CHECK: r={np.mean(db_corrs):.3f} — WARNING: W functions may differ!")

    # ============================================================
    # SAVE
    # ============================================================

    import csv
    csv_path = OUTPUT_DIR / "permutation_test_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'organism', 'n_proteins', 'n_skipped', 'real_mean', 'shuf_mean',
            'gap_pct', 'db_mean', 'db_corr', 'db_diff', 'real_cv', 'shuf_cv',
            'n_shuffles'
        ])
        writer.writeheader()
        writer.writerows(results)

    summary_path = OUTPUT_DIR / "permutation_test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("PERMUTATION NULL TEST v1 - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Organisms: {len(results)}\n")
        f.write(f"Shuffles per protein: {args.n_shuffles}\n")
        f.write(f"Total proteins: {sum(r['n_proteins'] for r in results):,}\n\n")
        f.write(f"Real mean of means:     {np.mean(real_means):.4f}\n")
        f.write(f"Shuffled mean of means: {np.mean(shuf_means):.4f}\n")
        f.write(f"Mean gap: {mean_gap:+.1f}%\n")
        f.write(f"Gap range: [{np.min(gaps):+.1f}%, {np.max(gaps):+.1f}%]\n\n")
        f.write(f"DB cross-check: mean r = {np.mean(db_corrs):.3f}\n\n")
        f.write(f"{'Organism':20s} {'N':>5} {'Real':>8} {'Shuf':>8} {'Gap':>7} {'DB':>8} {'r':>6}\n")
        f.write("-" * 68 + "\n")
        for r in sorted(results, key=lambda x: x['gap_pct'], reverse=True):
            f.write(f"{r['organism']:20s} {r['n_proteins']:>5} "
                    f"{r['real_mean']:>8.4f} {r['shuf_mean']:>8.4f} "
                    f"{r['gap_pct']:>+6.1f}% {r['db_mean']:>8.4f} {r['db_corr']:>5.3f}\n")

    logging.info(f"  Results: {csv_path}")
    logging.info(f"  Summary: {summary_path}")
    logging.info("")
    logging.info("Done.")


if __name__ == "__main__":
    main()
