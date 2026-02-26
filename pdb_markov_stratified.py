#!/usr/bin/env python3
"""
STRATIFIED MARKOV NULL ON PDB STRUCTURES
==========================================
The pooled PDB Markov test showed +8.6% per-chain gap because a single
transition kernel averaged over all fold classes. This test stratifies
by secondary structure class (all-alpha, all-beta, alpha/beta, other)
and builds separate transition models per class — mirroring the
per-organism stratification that gave +0.5% in AlphaFold.

Uses the PDB cache from pdb_replication.py or pdb_markov_test.py.

Usage:
  python pdb_markov_stratified.py --quick     # 500 chains, 10 synth
  python pdb_markov_stratified.py             # all cached, 30 synth
"""

import os
import sys
import math
import time
import csv
import gzip
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

# ============================================================
# CONFIG
# ============================================================

PDB_CACHE = Path("pdb_cache")
OUTPUT_DIR = Path("pdb_markov_stratified_results")
K_NEIGHBORS_DEFAULT = 50
N_SYNTH_DEFAULT = 30
MIN_LENGTH = 30

# Alpha helix basin: phi in [-160, 0], psi in [-80, 50]
# Beta sheet basin:  phi in [-180, -40], psi in [80, 180] union [-180, -140]
ALPHA_PHI = (-160, 0)
ALPHA_PSI = (-80, 50)
BETA_PHI = (-180, -40)
# Beta psi wraps around ±180; we check |psi| > 80

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

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD = "https://files.rcsb.org/download"


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
            logging.FileHandler(OUTPUT_DIR / "pdb_markov_stratified.log",
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

def _parse_atoms_from_pdb_cif(filepath, target_chain='A'):
    opener = gzip.open if str(filepath).endswith('.gz') else open
    residues = {}
    with opener(filepath, 'rt', errors='replace') as f:
        col_map = {}
        in_header = False
        col_idx = 0
        first_model = None
        model_num_col = None

        for line in f:
            ls = line.strip()
            if ls.startswith('_atom_site.'):
                col_map[ls.split('.')[1].strip()] = col_idx
                col_idx += 1
                in_header = True
                continue
            if in_header and not ls.startswith('_atom_site.'):
                in_header = False
                model_num_col = col_map.get('pdbx_PDB_model_num')
            if not col_map:
                continue
            if not (ls.startswith('ATOM') or ls.startswith('HETATM')):
                if col_map and ls in ('', '#', 'loop_'):
                    if residues:
                        break
                continue
            parts = ls.split()
            try:
                if parts[col_map.get('group_PDB', 0)] != 'ATOM':
                    continue
                if model_num_col is not None:
                    model = parts[model_num_col]
                    if first_model is None:
                        first_model = model
                    elif model != first_model:
                        continue
                chain = None
                if 'auth_asym_id' in col_map:
                    chain = parts[col_map['auth_asym_id']]
                elif 'label_asym_id' in col_map:
                    chain = parts[col_map['label_asym_id']]
                if chain != target_chain:
                    continue
                atom_name = parts[col_map.get('label_atom_id', 3)]
                if atom_name not in ('N', 'CA', 'C'):
                    continue
                alt = '.'
                if 'label_alt_id' in col_map:
                    alt = parts[col_map['label_alt_id']]
                if alt not in ('.', 'A', '?', ' '):
                    continue
                if 'label_seq_id' in col_map:
                    seq_id_str = parts[col_map['label_seq_id']]
                    if seq_id_str == '.':
                        continue
                    resnum = int(seq_id_str)
                elif 'auth_seq_id' in col_map:
                    resnum = int(parts[col_map['auth_seq_id']])
                else:
                    continue
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


def extract_phi_psi(residues):
    n = len(residues)
    if n < MIN_LENGTH:
        return []
    raw_angles = []
    for i in range(n):
        raw_phi = _calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                                 residues[i]['CA'], residues[i]['C']) if i > 0 else None
        raw_psi = _calc_dihedral(residues[i]['N'], residues[i]['CA'],
                                 residues[i]['C'], residues[i+1]['N']) if i < n-1 else None
        if raw_phi is not None and raw_psi is not None:
            raw_angles.append((raw_phi, raw_psi))
    if len(raw_angles) < MIN_LENGTH:
        return []
    phis_raw = np.array([a[0] for a in raw_angles])
    n_alpha_pos = np.sum((np.degrees(phis_raw) > -160) &
                         (np.degrees(phis_raw) < 0))
    n_alpha_neg = np.sum((np.degrees(-phis_raw) > -160) &
                         (np.degrees(-phis_raw) < 0))
    phi_sign = +1 if n_alpha_pos >= n_alpha_neg else -1
    return [(phi_sign * phi, phi_sign * psi) for phi, psi in raw_angles]


# ============================================================
# FOLD CLASS ASSIGNMENT (from dihedral angles)
# ============================================================

def classify_fold_class(phis, psis):
    """
    Classify a chain into fold class based on Ramachandran basin occupancy.
    Returns: 'all_alpha', 'all_beta', 'alpha_beta', 'other'
    """
    n = len(phis)
    n_alpha = 0
    n_beta = 0

    for phi, psi in zip(phis, psis):
        phi_deg = np.degrees(phi)
        psi_deg = np.degrees(psi)

        # Alpha basin
        if ALPHA_PHI[0] <= phi_deg <= ALPHA_PHI[1] and \
           ALPHA_PSI[0] <= psi_deg <= ALPHA_PSI[1]:
            n_alpha += 1

        # Beta basin (psi wraps around ±180)
        if BETA_PHI[0] <= phi_deg <= BETA_PHI[1] and \
           (psi_deg > 80 or psi_deg < -140):
            n_beta += 1

    frac_alpha = n_alpha / n
    frac_beta = n_beta / n

    # Thresholds (similar to SCOP/CATH class definitions)
    if frac_alpha >= 0.40 and frac_beta < 0.10:
        return 'all_alpha'
    elif frac_beta >= 0.30 and frac_alpha < 0.15:
        return 'all_beta'
    elif frac_alpha >= 0.15 and frac_beta >= 0.15:
        return 'alpha_beta'
    else:
        return 'other'


# ============================================================
# BPS/L
# ============================================================

def compute_bps_l(phis, psis, W_interp):
    W_vals = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_vals))
    return float(np.sum(dW)) / len(phis)


# ============================================================
# MARKOV TRANSITION MODEL
# ============================================================

def build_transition_data(all_sequences):
    src_list, tgt_list = [], []
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


def sample_markov_chain(length, sources, targets, tree, K, rng):
    start_idx = rng.integers(len(sources))
    phis = np.empty(length)
    psis = np.empty(length)
    phis[0] = sources[start_idx, 0]
    psis[0] = sources[start_idx, 1]
    K_eff = min(K, len(sources))
    for k in range(1, length):
        q = np.array([np.cos(phis[k-1]), np.sin(phis[k-1]),
                       np.cos(psis[k-1]), np.sin(psis[k-1])])
        _, indices = tree.query(q, k=K_eff)
        if K_eff == 1:
            indices = [indices]
        chosen = rng.choice(indices)
        phis[k] = targets[chosen, 0]
        psis[k] = targets[chosen, 1]
    return phis, psis


# ============================================================
# PDB DOWNLOAD
# ============================================================

def download_pdb_structures(max_structures=5000, resolution_max=2.0):
    import json
    import urllib.request

    PDB_CACHE.mkdir(parents=True, exist_ok=True)
    existing = list(PDB_CACHE.glob("*.cif.gz"))
    if len(existing) >= 50:
        logging.info(f"  PDB cache has {len(existing)} files, skipping download")
        return

    logging.info(f"  Querying RCSB for X-ray structures ≤{resolution_max}Å...")
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "exptl.method",
                                "operator": "exact_match",
                                "value": "X-RAY DIFFRACTION"}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                                "operator": "less_or_equal",
                                "value": resolution_max}},
                {"type": "terminal", "service": "text",
                 "parameters": {"attribute": "entity_poly.rcsb_entity_polymer_type",
                                "operator": "exact_match",
                                "value": "Protein"}}
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 10000},
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }

    try:
        req = urllib.request.Request(
            RCSB_SEARCH,
            data=json.dumps(query).encode('utf-8'),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        pdb_ids = [hit['identifier'] for hit in data.get('result_set', [])]
        logging.info(f"  Found {len(pdb_ids)} PDB entries")
    except Exception as e:
        logging.error(f"  RCSB query failed: {e}")
        return

    rng = np.random.default_rng(42)
    if len(pdb_ids) > max_structures:
        pdb_ids = list(rng.choice(pdb_ids, size=max_structures, replace=False))

    n_ok, n_fail = 0, 0
    for i, pdb_id in enumerate(pdb_ids):
        if i > 0 and i % 200 == 0:
            logging.info(f"  Downloaded {i}/{len(pdb_ids)} ({n_ok} OK)")
        outpath = PDB_CACHE / f"{pdb_id.lower()}.cif.gz"
        if outpath.exists():
            n_ok += 1
            continue
        url = f"{RCSB_DOWNLOAD}/{pdb_id.lower()}.cif.gz"
        try:
            import urllib.request as ur
            ur.urlretrieve(url, outpath)
            n_ok += 1
        except Exception:
            n_fail += 1

    logging.info(f"  Download complete: {n_ok} OK, {n_fail} failed")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stratified Markov null on PDB structures")
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS_DEFAULT)
    parser.add_argument("--n-synth", type=int, default=N_SYNTH_DEFAULT)
    parser.add_argument("--max-chains", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 500 chains, 10 synth")
    args = parser.parse_args()

    setup_logging()

    if args.quick:
        args.n_synth = 10
        args.max_chains = 500

    logging.info("=" * 70)
    logging.info("STRATIFIED MARKOV NULL ON PDB STRUCTURES")
    logging.info("Per fold-class transition models")
    logging.info("=" * 70)
    logging.info(f"  K neighbors: {args.k_neighbors}")
    logging.info(f"  Synthetic chains per protein: {args.n_synth}")
    logging.info(f"  Max chains: {args.max_chains or 'all'}")
    logging.info("")

    # Auto-download if needed
    if not PDB_CACHE.exists() or len(list(PDB_CACHE.glob("*.cif*"))) < 50:
        logging.info("PDB cache empty. Attempting download...")
        try:
            download_pdb_structures(
                max_structures=args.max_chains or 5000)
        except Exception as e:
            logging.warning(f"Download failed ({e})")

    if not PDB_CACHE.exists():
        logging.error(f"PDB cache not found: {PDB_CACHE}")
        sys.exit(1)

    W_interp = build_W_interp()
    rng = np.random.default_rng(args.seed)

    cif_files = sorted(PDB_CACHE.glob("*.cif.gz"))
    if not cif_files:
        cif_files = sorted(PDB_CACHE.glob("*.cif"))
    logging.info(f"  Found {len(cif_files)} cached PDB files")

    if args.max_chains and len(cif_files) > args.max_chains:
        indices = rng.choice(len(cif_files), size=args.max_chains, replace=False)
        cif_files = [cif_files[i] for i in sorted(indices)]

    # ============================================================
    # Phase 1: Parse all structures, classify fold class
    # ============================================================
    logging.info("")
    logging.info("Phase 1: Parsing and classifying PDB structures...")
    t0 = time.time()

    # Store per class: list of (pdb_id, phis, psis, phi_psi_pairs)
    class_data = defaultdict(list)
    n_failed = 0

    for i, cif_path in enumerate(cif_files):
        if i > 0 and i % 500 == 0:
            counts = {k: len(v) for k, v in class_data.items()}
            logging.info(f"  Parsed {i}/{len(cif_files)} | "
                         f"Classes: {dict(counts)} | Failed: {n_failed}")

        pdb_id = cif_path.stem.replace('.cif', '').upper()

        phi_psi = []
        for chain in ['A', 'B', 'C', 'D']:
            residues = _parse_atoms_from_pdb_cif(cif_path, target_chain=chain)
            if len(residues) >= MIN_LENGTH:
                phi_psi = extract_phi_psi(residues)
                if len(phi_psi) >= MIN_LENGTH:
                    break

        if len(phi_psi) < MIN_LENGTH:
            n_failed += 1
            continue

        phis = np.array([pp[0] for pp in phi_psi])
        psis = np.array([pp[1] for pp in phi_psi])

        fold_class = classify_fold_class(phis, psis)
        class_data[fold_class].append((pdb_id, phis, psis, phi_psi))

    parse_time = time.time() - t0
    total_chains = sum(len(v) for v in class_data.values())
    logging.info(f"  Parsed {total_chains} chains in {parse_time:.0f}s "
                 f"({n_failed} failed)")
    for fc in sorted(class_data.keys()):
        logging.info(f"    {fc:12s}: {len(class_data[fc]):>5} chains")

    # ============================================================
    # Phase 2: Per-class Markov test
    # ============================================================
    logging.info("")
    logging.info("Phase 2: Per-class Markov tests...")

    all_results = []
    class_summaries = {}

    for fc in sorted(class_data.keys()):
        chains = class_data[fc]
        if len(chains) < 20:
            logging.info(f"  {fc}: only {len(chains)} chains, skipping")
            continue

        logging.info(f"")
        logging.info(f"  --- {fc} ({len(chains)} chains) ---")

        # Build transition model from this class only
        t0 = time.time()
        class_sequences = [c[3] for c in chains]  # phi_psi pairs
        sources, targets, tree = build_transition_data(class_sequences)
        logging.info(f"    Transitions: {len(sources):,} "
                     f"(KD-tree in {time.time()-t0:.1f}s)")

        # Compute Real and Markov BPS/L
        t0 = time.time()
        class_results = []
        for j, (pdb_id, phis, psis, _) in enumerate(chains):
            if j > 0 and j % 500 == 0:
                logging.info(f"    Progress: {j}/{len(chains)}")

            L = len(phis)
            real_bps = compute_bps_l(phis, psis, W_interp)

            markov_values = []
            for _ in range(args.n_synth):
                m_phis, m_psis = sample_markov_chain(
                    L, sources, targets, tree, args.k_neighbors, rng
                )
                m_bps = compute_bps_l(m_phis, m_psis, W_interp)
                markov_values.append(m_bps)

            markov_mean = float(np.mean(markov_values))
            gap = 100 * (markov_mean - real_bps) / real_bps if real_bps > 0 else 0

            class_results.append({
                'pdb_id': pdb_id,
                'fold_class': fc,
                'n_residues': L,
                'real_bps': real_bps,
                'markov_bps': markov_mean,
                'gap_pct': gap,
            })

        compute_time = time.time() - t0

        # Class summary
        real_arr = np.array([r['real_bps'] for r in class_results])
        markov_arr = np.array([r['markov_bps'] for r in class_results])
        gaps = np.array([r['gap_pct'] for r in class_results])
        corr = float(np.corrcoef(real_arr, markov_arr)[0, 1]) \
            if len(real_arr) > 2 else 0

        real_mean = float(np.mean(real_arr))
        markov_mean = float(np.mean(markov_arr))
        mean_gap = float(np.mean(gaps))
        agg_gap = 100 * (markov_mean - real_mean) / real_mean \
            if real_mean > 0 else 0

        class_summaries[fc] = {
            'n': len(class_results),
            'transitions': len(sources),
            'real_mean': real_mean,
            'markov_mean': markov_mean,
            'agg_gap': agg_gap,
            'per_chain_gap': mean_gap,
            'per_chain_gap_sd': float(np.std(gaps)),
            'corr': corr,
        }

        logging.info(f"    Real:   {real_mean:.4f} ± {np.std(real_arr):.4f}")
        logging.info(f"    Markov: {markov_mean:.4f} ± {np.std(markov_arr):.4f}")
        logging.info(f"    Aggregate gap: {agg_gap:+.1f}%")
        logging.info(f"    Per-chain gap: {mean_gap:+.1f}% ± {np.std(gaps):.1f}%")
        logging.info(f"    Per-chain r:   {corr:.3f}")
        logging.info(f"    ({compute_time:.0f}s)")

        all_results.extend(class_results)

    # ============================================================
    # OVERALL SUMMARY
    # ============================================================

    logging.info("")
    logging.info("=" * 70)
    logging.info("STRATIFIED RESULTS SUMMARY")
    logging.info("=" * 70)
    logging.info("")

    logging.info(f"  {'Class':12s} {'N':>6} {'Real':>8} {'Markov':>8} "
                 f"{'AggGap':>8} {'ChainGap':>9} {'r':>6}")
    logging.info(f"  {'-'*12:12s} {'-'*6:>6} {'-'*8:>8} {'-'*8:>8} "
                 f"{'-'*8:>8} {'-'*9:>9} {'-'*6:>6}")

    weighted_agg_gaps = []
    weighted_chain_gaps = []
    weights = []

    for fc in sorted(class_summaries.keys()):
        s = class_summaries[fc]
        logging.info(f"  {fc:12s} {s['n']:>6} {s['real_mean']:>8.4f} "
                     f"{s['markov_mean']:>8.4f} {s['agg_gap']:>+7.1f}% "
                     f"{s['per_chain_gap']:>+8.1f}% {s['corr']:>6.3f}")
        weighted_agg_gaps.append(s['agg_gap'])
        weighted_chain_gaps.append(s['per_chain_gap'])
        weights.append(s['n'])

    if weights:
        w = np.array(weights, dtype=float)
        overall_agg = np.average(weighted_agg_gaps, weights=w)
        overall_chain = np.average(weighted_chain_gaps, weights=w)

        total_real = np.mean([r['real_bps'] for r in all_results])
        total_markov = np.mean([r['markov_bps'] for r in all_results])
        total_gaps = np.array([r['gap_pct'] for r in all_results])
        total_corr = float(np.corrcoef(
            [r['real_bps'] for r in all_results],
            [r['markov_bps'] for r in all_results]
        )[0, 1]) if len(all_results) > 2 else 0

        logging.info(f"  {'-'*12:12s} {'-'*6:>6} {'-'*8:>8} {'-'*8:>8} "
                     f"{'-'*8:>8} {'-'*9:>9} {'-'*6:>6}")
        logging.info(f"  {'OVERALL':12s} {sum(weights):>6.0f} "
                     f"{total_real:>8.4f} {total_markov:>8.4f} "
                     f"{overall_agg:>+7.1f}% {overall_chain:>+8.1f}% "
                     f"{total_corr:>6.3f}")

    logging.info("")

    # ============================================================
    # COMPARISON
    # ============================================================

    logging.info("=" * 70)
    logging.info("COMPARISON: POOLED vs STRATIFIED vs ALPHAFOLD")
    logging.info("=" * 70)
    logging.info("")
    logging.info(f"  {'Method':30s} {'AggGap':>10} {'ChainGap':>10}")
    logging.info(f"  {'-'*30:30s} {'-'*10:>10} {'-'*10:>10}")
    logging.info(f"  {'AlphaFold (per-organism)':30s} {'n/a':>10} "
                 f"{'+0.5%':>10}")
    logging.info(f"  {'PDB pooled':30s} {'+1.7%':>10} {'+8.6%':>10}")
    if weights:
        logging.info(f"  {'PDB stratified (this run)':30s} "
                     f"{overall_agg:>+9.1f}% {overall_chain:>+9.1f}%")
    logging.info("")

    if weights and abs(overall_chain) < 5:
        logging.info("  *** STRATIFICATION CLOSES THE GAP ***")
        logging.info("  Per-chain gap < 5% with fold-class stratification.")
        logging.info("  The mechanism is fully validated across:")
        logging.info("    - AlphaFold predictions (34 proteomes, per-organism)")
        logging.info("    - Experimental PDB structures (per fold-class)")
        logging.info("  BPS/L ≈ 0.20 is set by local backbone transitions.")
    elif weights and abs(overall_chain) < 10:
        logging.info("  Stratification reduced the gap from pooled result.")
        logging.info("  Finer stratification may further improve per-chain fit.")
    logging.info("")

    # ============================================================
    # SAVE
    # ============================================================

    csv_path = OUTPUT_DIR / "pdb_markov_stratified.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pdb_id', 'fold_class', 'n_residues',
            'real_bps', 'markov_bps', 'gap_pct'
        ])
        writer.writeheader()
        writer.writerows(all_results)

    summary_path = OUTPUT_DIR / "pdb_markov_stratified_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("STRATIFIED MARKOV NULL ON PDB STRUCTURES\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total chains: {len(all_results)}\n")
        f.write(f"K neighbors: {args.k_neighbors}\n")
        f.write(f"Synth per chain: {args.n_synth}\n\n")

        f.write(f"{'Class':12s} {'N':>6} {'Real':>8} {'Markov':>8} "
                f"{'AggGap':>8} {'ChainGap':>9} {'r':>6}\n")
        f.write("-" * 60 + "\n")
        for fc in sorted(class_summaries.keys()):
            s = class_summaries[fc]
            f.write(f"{fc:12s} {s['n']:>6} {s['real_mean']:>8.4f} "
                    f"{s['markov_mean']:>8.4f} {s['agg_gap']:>+7.1f}% "
                    f"{s['per_chain_gap']:>+8.1f}% {s['corr']:>6.3f}\n")
        if weights:
            f.write("-" * 60 + "\n")
            f.write(f"{'OVERALL':12s} {sum(weights):>6.0f} "
                    f"{total_real:>8.4f} {total_markov:>8.4f} "
                    f"{overall_agg:>+7.1f}% {overall_chain:>+8.1f}% "
                    f"{total_corr:>6.3f}\n")

        f.write("\nComparison:\n")
        f.write(f"  AlphaFold per-organism:   Chain gap = +0.5%\n")
        f.write(f"  PDB pooled:              Chain gap = +8.6%\n")
        if weights:
            f.write(f"  PDB stratified:          Chain gap = "
                    f"{overall_chain:+.1f}%\n")

    logging.info(f"  Results: {csv_path}")
    logging.info(f"  Summary: {summary_path}")
    logging.info("")
    logging.info("Done.")


if __name__ == "__main__":
    main()
