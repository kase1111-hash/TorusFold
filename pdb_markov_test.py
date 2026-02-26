#!/usr/bin/env python3
"""
MARKOV NULL ON PDB EXPERIMENTAL STRUCTURES
=============================================
The capstone test: does the Markov transition null reproduce BPS/L
on experimental X-ray structures, not just AlphaFold predictions?

If PDB Real ≈ PDB Markov (±few %), the mechanism is validated
across both prediction models AND experimental structures.

Uses the PDB cache from pdb_replication.py (pdb_cache/*.cif.gz).

Usage:
  python pdb_markov_test.py --quick       # 200 chains, K=50
  python pdb_markov_test.py               # all cached chains, K=50
  python pdb_markov_test.py --k-neighbors 20  # different K
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

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

# ============================================================
# CONFIG
# ============================================================

PDB_CACHE = Path("pdb_cache")
OUTPUT_DIR = Path("pdb_markov_results")
K_NEIGHBORS_DEFAULT = 50
N_SYNTH_DEFAULT = 30
MIN_LENGTH = 30
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD = "https://files.rcsb.org/download"

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
            logging.FileHandler(OUTPUT_DIR / "pdb_markov_test.log",
                                mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ============================================================
# SUPERPOTENTIAL (identical to all pipelines)
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
# CIF PARSER (PDB format, handles gzip)
# ============================================================

def _parse_atoms_from_pdb_cif(filepath, target_chain='A'):
    """Parse mmCIF from PDB. Takes first model, specified chain."""
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
    """Extract (phi, psi) with auto sign detection."""
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
# BPS/L
# ============================================================

def compute_bps_l(phis, psis, W_interp):
    W_vals = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_vals))
    return float(np.sum(dW)) / len(phis)


# ============================================================
# MARKOV TRANSITION MODEL (identical to v2)
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


def embed_angle(phi, psi):
    return np.array([np.cos(phi), np.sin(phi), np.cos(psi), np.sin(psi)])


def sample_markov_chain(length, sources, targets, tree, K, rng):
    start_idx = rng.integers(len(sources))
    phis = np.empty(length)
    psis = np.empty(length)
    phis[0] = sources[start_idx, 0]
    psis[0] = sources[start_idx, 1]
    K_eff = min(K, len(sources))
    for k in range(1, length):
        q = embed_angle(phis[k-1], psis[k-1])
        _, indices = tree.query(q, k=K_eff)
        if K_eff == 1:
            indices = [indices]
        chosen = rng.choice(indices)
        phis[k] = targets[chosen, 0]
        psis[k] = targets[chosen, 1]
    return phis, psis


# ============================================================
# PDB DOWNLOAD (integrated from pdb_replication.py)
# ============================================================

def download_pdb_structures(max_structures=5000, resolution_max=2.0):
    """Download PDB structures if cache is empty."""
    import json
    import urllib.request
    import urllib.error

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

    # Random sample
    rng = np.random.default_rng(42)
    if len(pdb_ids) > max_structures:
        pdb_ids = list(rng.choice(pdb_ids, size=max_structures, replace=False))

    # Download
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
            urllib.request.urlretrieve(url, outpath)
            n_ok += 1
        except Exception:
            n_fail += 1
            continue

    logging.info(f"  Download complete: {n_ok} OK, {n_fail} failed")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Markov null on PDB structures")
    parser.add_argument("--k-neighbors", type=int, default=K_NEIGHBORS_DEFAULT)
    parser.add_argument("--n-synth", type=int, default=N_SYNTH_DEFAULT)
    parser.add_argument("--max-chains", type=int, default=None,
                        help="Max chains to process (default: all cached)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 200 chains, 10 synth")
    args = parser.parse_args()

    setup_logging()

    if args.quick:
        args.n_synth = 10
        args.max_chains = 200

    logging.info("=" * 70)
    logging.info("MARKOV NULL ON PDB EXPERIMENTAL STRUCTURES")
    logging.info("Does Real ≈ Markov hold on X-ray crystal structures?")
    logging.info("=" * 70)
    logging.info(f"  K neighbors: {args.k_neighbors}")
    logging.info(f"  Synthetic chains per protein: {args.n_synth}")
    logging.info(f"  Max chains: {args.max_chains or 'all'}")
    logging.info(f"  Seed: {args.seed}")
    logging.info("")

    if not PDB_CACHE.exists() or len(list(PDB_CACHE.glob("*.cif*"))) < 50:
        logging.info("PDB cache empty or missing. Attempting download...")
        try:
            download_pdb_structures(
                max_structures=args.max_chains or 5000,
                resolution_max=2.0
            )
        except Exception as e:
            logging.warning(f"Download failed ({e}). Checking for existing cache...")

    if not PDB_CACHE.exists():
        logging.error(f"PDB cache not found: {PDB_CACHE}")
        logging.error("Run pdb_replication.py first to download structures.")
        sys.exit(1)

    # Build superpotential
    logging.info("Building superpotential...")
    W_interp = build_W_interp()

    rng = np.random.default_rng(args.seed)

    # Discover cached CIF files
    cif_files = sorted(PDB_CACHE.glob("*.cif.gz"))
    if not cif_files:
        cif_files = sorted(PDB_CACHE.glob("*.cif"))
    logging.info(f"  Found {len(cif_files)} cached PDB files")

    if args.max_chains and len(cif_files) > args.max_chains:
        indices = rng.choice(len(cif_files), size=args.max_chains, replace=False)
        cif_files = [cif_files[i] for i in sorted(indices)]
        logging.info(f"  Randomly selected {len(cif_files)} files")

    # Phase 1: Parse all structures and extract (phi, psi) sequences
    logging.info("")
    logging.info("Phase 1: Parsing PDB structures...")
    t0 = time.time()

    all_sequences = []    # for building transition model
    test_data = []        # (pdb_id, phis, psis) for testing
    n_failed = 0

    for i, cif_path in enumerate(cif_files):
        if i > 0 and i % 500 == 0:
            logging.info(f"  Parsed {i}/{len(cif_files)} "
                         f"({len(test_data)} OK, {n_failed} failed)")

        pdb_id = cif_path.stem.replace('.cif', '').upper()

        # Try chain A first, then B, C, D
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

        all_sequences.append(phi_psi)
        test_data.append((pdb_id, phis, psis))

    parse_time = time.time() - t0
    logging.info(f"  Parsed {len(test_data)} chains in {parse_time:.0f}s "
                 f"({n_failed} failed)")

    if len(test_data) < 20:
        logging.error("Too few chains for meaningful analysis.")
        sys.exit(1)

    # Phase 2: Build transition model from ALL PDB sequences
    logging.info("")
    logging.info("Phase 2: Building KD-tree transition model...")
    t0 = time.time()
    sources, targets, tree = build_transition_data(all_sequences)
    logging.info(f"  Transitions: {len(sources):,}")
    logging.info(f"  KD-tree built in {time.time()-t0:.1f}s")

    # Phase 3: Compute Real and Markov BPS/L for each chain
    logging.info("")
    logging.info("Phase 3: Computing Real and Markov BPS/L...")
    t0 = time.time()

    results = []
    for i, (pdb_id, phis, psis) in enumerate(test_data):
        if i > 0 and i % 500 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(test_data) - i) / rate if rate > 0 else 0
            logging.info(f"  Progress: {i}/{len(test_data)} "
                         f"({rate:.1f}/s, ETA {eta:.0f}s)")

        L = len(phis)
        real_bps = compute_bps_l(phis, psis, W_interp)

        # Markov chains
        markov_values = []
        for _ in range(args.n_synth):
            m_phis, m_psis = sample_markov_chain(
                L, sources, targets, tree, args.k_neighbors, rng
            )
            m_bps = compute_bps_l(m_phis, m_psis, W_interp)
            markov_values.append(m_bps)

        markov_mean = float(np.mean(markov_values))

        results.append({
            'pdb_id': pdb_id,
            'n_residues': L,
            'real_bps': real_bps,
            'markov_bps': markov_mean,
            'gap_pct': 100 * (markov_mean - real_bps) / real_bps if real_bps > 0 else 0,
        })

    compute_time = time.time() - t0
    logging.info(f"  Computed {len(results)} chains in {compute_time:.0f}s")

    # ============================================================
    # ANALYSIS
    # ============================================================

    real_arr = np.array([r['real_bps'] for r in results])
    markov_arr = np.array([r['markov_bps'] for r in results])
    gaps = np.array([r['gap_pct'] for r in results])

    real_mean = float(np.mean(real_arr))
    markov_mean = float(np.mean(markov_arr))
    mean_gap = float(np.mean(gaps))

    # Per-chain correlation
    corr = float(np.corrcoef(real_arr, markov_arr)[0, 1]) if len(real_arr) > 2 else 0

    logging.info("")
    logging.info("=" * 70)
    logging.info("RESULTS")
    logging.info("=" * 70)
    logging.info("")
    logging.info(f"  Chains: {len(results)}")
    logging.info(f"  K neighbors: {args.k_neighbors}")
    logging.info(f"  Synth per chain: {args.n_synth}")
    logging.info(f"  Total transitions in model: {len(sources):,}")
    logging.info("")
    logging.info(f"  PDB Real BPS/L:")
    logging.info(f"    Mean:   {real_mean:.4f}")
    logging.info(f"    SD:     {np.std(real_arr):.4f}")
    logging.info(f"    Range:  [{np.min(real_arr):.4f}, {np.max(real_arr):.4f}]")
    logging.info("")
    logging.info(f"  PDB Markov BPS/L (K={args.k_neighbors}):")
    logging.info(f"    Mean:   {markov_mean:.4f}")
    logging.info(f"    SD:     {np.std(markov_arr):.4f}")
    logging.info(f"    Range:  [{np.min(markov_arr):.4f}, {np.max(markov_arr):.4f}]")
    logging.info("")
    logging.info(f"  Gap (Markov - Real) / Real:")
    logging.info(f"    Mean gap:  {mean_gap:+.1f}%")
    logging.info(f"    Gap SD:    {np.std(gaps):.1f}%")
    logging.info(f"    Gap range: [{np.min(gaps):+.1f}%, {np.max(gaps):+.1f}%]")
    logging.info("")
    logging.info(f"  Per-chain correlation (Real vs Markov): r = {corr:.3f}")
    logging.info("")

    # Comparison to AlphaFold Markov result
    logging.info("=" * 70)
    logging.info("COMPARISON TO ALPHAFOLD MARKOV RESULT")
    logging.info("=" * 70)
    logging.info("")
    logging.info(f"  AlphaFold: Real=0.175, Markov=0.176, Gap=+0.5%")
    logging.info(f"  PDB:       Real={real_mean:.3f}, Markov={markov_mean:.3f}, "
                 f"Gap={mean_gap:+.1f}%")
    logging.info("")

    if abs(mean_gap) < 5:
        logging.info("  *** PDB Real ≈ PDB Markov ***")
        logging.info("  The Markov transition null reproduces BPS/L on")
        logging.info("  experimental X-ray structures.")
        logging.info("  The mechanism is VALIDATED across:")
        logging.info("    - AlphaFold predictions (34 proteomes)")
        logging.info("    - Experimental X-ray structures (PDB)")
        logging.info("  BPS/L is an emergent constant of local backbone")
        logging.info("  transition geometry, independent of structure")
        logging.info("  determination method.")
    elif abs(mean_gap) < 15:
        logging.info("  PDB shows moderate Real-Markov gap.")
        logging.info("  Local transitions explain most but not all of the signal.")
    else:
        logging.info("  PDB shows large Real-Markov gap.")
        logging.info("  The Markov null does not fully explain PDB BPS/L.")

    logging.info("")

    # ============================================================
    # SAVE
    # ============================================================

    csv_path = OUTPUT_DIR / "pdb_markov_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pdb_id', 'n_residues', 'real_bps', 'markov_bps', 'gap_pct'
        ])
        writer.writeheader()
        writer.writerows(results)

    summary_path = OUTPUT_DIR / "pdb_markov_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("MARKOV NULL ON PDB STRUCTURES - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Chains: {len(results)}\n")
        f.write(f"K neighbors: {args.k_neighbors}\n")
        f.write(f"Synth per chain: {args.n_synth}\n")
        f.write(f"Transitions in model: {len(sources):,}\n\n")
        f.write(f"PDB Real:   {real_mean:.4f} +/- {np.std(real_arr):.4f}\n")
        f.write(f"PDB Markov: {markov_mean:.4f} +/- {np.std(markov_arr):.4f}\n")
        f.write(f"Gap: {mean_gap:+.1f}%\n\n")
        f.write(f"Per-chain r: {corr:.3f}\n\n")
        f.write("Cross-method comparison:\n")
        f.write(f"  AlphaFold: Real=0.175, Markov=0.176, Gap=+0.5%\n")
        f.write(f"  PDB:       Real={real_mean:.3f}, Markov={markov_mean:.3f}, "
                f"Gap={mean_gap:+.1f}%\n")

    logging.info(f"  Results: {csv_path}")
    logging.info(f"  Summary: {summary_path}")
    logging.info("")
    logging.info("Done.")


if __name__ == "__main__":
    main()
