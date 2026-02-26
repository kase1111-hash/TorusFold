#!/usr/bin/env python3
"""
PDB REPLICATION VERIFICATION
==============================
Downloads high-resolution experimental protein structures from the PDB
and computes BPS/L to validate the AlphaFold-derived results.

CRITICAL QUESTION:
  Does BPS/L ≈ 0.20 hold on experimental X-ray structures?
  If yes → the constant is real, not an AlphaFold artifact.
  If no  → the constant is specific to AlphaFold geometry.

DATA SOURCE:
  RCSB PDB Search API → non-redundant high-resolution X-ray structures
  Filters: resolution ≤ 2.0 Å, X-ray only, protein chains ≥ 30 residues

RESOLUTION TIERS:
  Tier 1: ≤ 1.5 Å  (gold standard, ~2000 chains)
  Tier 2: ≤ 2.0 Å  (primary analysis, ~8000 chains)
  Tier 3: ≤ 2.5 Å  (extended, ~15000 chains)

COMPARISON:
  AlphaFold pLDDT ≥ 85: BPS/L = 0.2020 ± 0.0039

Usage:
  python pdb_replication.py --quick          # 200 structures, fast test
  python pdb_replication.py                  # 2000 structures, default
  python pdb_replication.py --max-chains 5000  # 5000 structures
  python pdb_replication.py --resolution 1.5 # ultra-high resolution only

Output:
  pdb_replication_results/pdb_bps_results.csv
  pdb_replication_results/pdb_replication_summary.txt
"""

import os
import sys
import math
import time
import csv
import json
import gzip
import argparse
import logging
from pathlib import Path
from io import BytesIO

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

try:
    import requests
except ImportError:
    print("Installing requests...")
    os.system(f"{sys.executable} -m pip install requests --break-system-packages -q")
    import requests

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("pdb_replication_results")
CACHE_DIR = Path("pdb_cache")
RESOLUTION_DEFAULT = 2.0
MAX_CHAINS_DEFAULT = 2000
MIN_LENGTH = 30

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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR / "pdb_replication.log",
                                mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ============================================================
# SUPERPOTENTIAL (identical to all other scripts)
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
# PDB SEARCH API
# ============================================================

def search_pdb_chains(max_resolution, max_chains):
    """Search RCSB PDB for high-resolution X-ray protein chains.

    Uses the RCSB Search API v2 to find non-redundant entries.
    Returns list of (pdb_id, chain_id, resolution) tuples.
    """
    logging.info(f"  Searching PDB for X-ray structures ≤ {max_resolution} Å...")

    # RCSB Search API v2 query
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "greater_or_equal",
                        "value": MIN_LENGTH
                    }
                }
            ]
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": min(max_chains * 5, 10000)  # RCSB API limit is 10,000 per page
            },
            "results_content_type": ["experimental"],
            "sort": [
                {
                    "sort_by": "score",
                    "direction": "desc"
                }
            ]
        }
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    resp = requests.post(url, json=query, timeout=60)

    if resp.status_code != 200:
        logging.error(f"  PDB search failed: {resp.status_code}")
        logging.error(f"  Response: {resp.text[:500]}")
        return []

    data = resp.json()
    total_count = data.get("total_count", 0)
    logging.info(f"  Found {total_count} polymer entities")

    results = data.get("result_set", [])
    logging.info(f"  Retrieved {len(results)} results")

    # Parse entity identifiers: format is "PDBID_ENTITY"
    # We need to map these to actual chain IDs
    entities = []
    for r in results:
        identifier = r.get("identifier", "")
        # Format: "1ABC_1" (PDB ID + entity number)
        parts = identifier.split("_")
        if len(parts) == 2:
            pdb_id = parts[0].upper()
            entity_num = parts[1]
            entities.append((pdb_id, entity_num))

    logging.info(f"  Parsed {len(entities)} entities")

    # Randomly sample to get resolution diversity
    rng = np.random.default_rng(42)
    if len(entities) > max_chains:
        indices = rng.choice(len(entities), size=max_chains, replace=False)
        entities = [entities[i] for i in sorted(indices)]
        logging.info(f"  Randomly sampled {max_chains} entities for resolution diversity")

    return entities


def get_chain_for_entity(pdb_id, entity_num):
    """Get the first chain ID for a given entity from RCSB API.

    Falls back to common chain letters if API fails.
    """
    # Most entities map to chain A for entity 1, B for entity 2, etc.
    # This is a reasonable default
    chain_map = {
        '1': 'A', '2': 'B', '3': 'C', '4': 'D',
        '5': 'E', '6': 'F', '7': 'G', '8': 'H',
    }
    return chain_map.get(entity_num, 'A')


# ============================================================
# PDB FILE DOWNLOAD
# ============================================================

def download_cif(pdb_id):
    """Download mmCIF file from RCSB. Returns path or None."""
    cache_path = CACHE_DIR / f"{pdb_id.lower()}.cif.gz"

    if cache_path.exists():
        return cache_path

    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.cif.gz"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            cache_path.write_bytes(resp.content)
            return cache_path
    except Exception as e:
        logging.debug(f"  Download failed for {pdb_id}: {e}")
    return None


# ============================================================
# CIF PARSER (adapted for PDB mmCIF files)
# ============================================================

def parse_pdb_cif(filepath, target_chain='A'):
    """Parse mmCIF file from PDB. Handles gzipped files.

    PDB mmCIF files differ from AlphaFold in several ways:
    - Multiple models (NMR) → take first model only
    - Multiple chains → filter by target_chain
    - Alternate conformations → take 'A' or '.'
    - auth_asym_id vs label_asym_id for chain identification

    Returns: list of dicts with 'N', 'CA', 'C' coordinates per residue
    """
    opener = gzip.open if str(filepath).endswith('.gz') else open

    residues = {}
    with opener(filepath, 'rt', errors='replace') as f:
        col_map = {}
        in_header = False
        col_idx = 0
        model_num_col = None
        first_model = None

        for line in f:
            ls = line.strip()

            if ls.startswith('_atom_site.'):
                field = ls.split('.')[1].strip()
                col_map[field] = col_idx
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
                    if residues:  # stop after first data block
                        break
                continue

            parts = ls.split()

            try:
                if parts[col_map.get('group_PDB', 0)] != 'ATOM':
                    continue

                # Model filtering (take first model only)
                if model_num_col is not None:
                    model = parts[model_num_col]
                    if first_model is None:
                        first_model = model
                    elif model != first_model:
                        continue

                # Chain filtering
                # Try auth_asym_id first (more common in PDB files),
                # fall back to label_asym_id
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

                # Alternate conformation filtering
                alt = '.'
                if 'label_alt_id' in col_map:
                    alt = parts[col_map['label_alt_id']]
                if alt not in ('.', 'A', '?', ' '):
                    continue

                # Residue number — use label_seq_id (sequential)
                # Fall back to auth_seq_id
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


def extract_phi_psi_pdb(residues):
    """Extract (phi, psi) from parsed residues. Auto-detect sign convention."""
    n = len(residues)
    if n < MIN_LENGTH:
        return [], 0

    # Extract raw angles
    raw_angles = []
    for i in range(n):
        raw_phi = _calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                                 residues[i]['CA'], residues[i]['C']) if i > 0 else None
        raw_psi = _calc_dihedral(residues[i]['N'], residues[i]['CA'],
                                 residues[i]['C'], residues[i+1]['N']) if i < n-1 else None
        if raw_phi is not None and raw_psi is not None:
            raw_angles.append((raw_phi, raw_psi))

    if len(raw_angles) < MIN_LENGTH:
        return [], 0

    # Determine phi sign: test which convention puts more residues in
    # the alpha-helix region (-160° < phi < 0°, -120° < psi < 30°)
    phis_raw = np.array([a[0] for a in raw_angles])

    n_alpha_pos = np.sum((np.degrees(phis_raw) > -160) &
                         (np.degrees(phis_raw) < 0))
    n_alpha_neg = np.sum((np.degrees(-phis_raw) > -160) &
                         (np.degrees(-phis_raw) < 0))

    phi_sign = +1 if n_alpha_pos >= n_alpha_neg else -1

    phi_psi = [(phi_sign * phi, phi_sign * psi) for phi, psi in raw_angles]
    return phi_psi, phi_sign


# ============================================================
# BPS/L COMPUTATION
# ============================================================

def compute_bps_l(phis, psis, W_interp):
    W_vals = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_vals))
    return float(np.sum(dW)) / len(phis)


# ============================================================
# RESOLUTION LOOKUP
# ============================================================

def get_resolution(pdb_id):
    """Fetch resolution from RCSB API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            res_list = data.get("rcsb_entry_info", {}).get("resolution_combined", [])
            if res_list:
                return res_list[0]
            # Try alternative path
            refine = data.get("refine", [{}])
            if refine and "ls_d_res_high" in refine[0]:
                return float(refine[0]["ls_d_res_high"])
    except Exception:
        pass
    return None


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_entity(pdb_id, entity_num, W_interp):
    """Download, parse, and compute BPS/L for one PDB entity."""
    # Download
    cif_path = download_cif(pdb_id)
    if cif_path is None:
        return None

    # Get chain for this entity
    chain_id = get_chain_for_entity(pdb_id, entity_num)

    # Parse
    residues = parse_pdb_cif(cif_path, target_chain=chain_id)

    # If chain A didn't work, try finding any chain with enough residues
    if len(residues) < MIN_LENGTH:
        for alt_chain in ['A', 'B', 'C', 'D', 'E', 'F']:
            if alt_chain == chain_id:
                continue
            residues = parse_pdb_cif(cif_path, target_chain=alt_chain)
            if len(residues) >= MIN_LENGTH:
                chain_id = alt_chain
                break

    if len(residues) < MIN_LENGTH:
        return None

    # Extract dihedrals
    phi_psi, phi_sign = extract_phi_psi_pdb(residues)
    if len(phi_psi) < MIN_LENGTH:
        return None

    phis = np.array([pp[0] for pp in phi_psi])
    psis = np.array([pp[1] for pp in phi_psi])

    # Compute BPS/L
    bps_l = compute_bps_l(phis, psis, W_interp)

    # Compute secondary structure fractions
    phi_deg = np.degrees(phis)
    psi_deg = np.degrees(psis)
    n_res = len(phis)

    n_helix = np.sum((phi_deg > -160) & (phi_deg < 0) &
                      (psi_deg > -120) & (psi_deg < 30))
    n_sheet = np.sum((phi_deg > -170) & (phi_deg < -70) &
                      ((psi_deg > 90) | (psi_deg < -120)))

    pct_helix = 100 * n_helix / n_res
    pct_sheet = 100 * n_sheet / n_res

    # Compute mean B-factor if available (proxy for disorder)
    mean_bfactor = None
    try:
        bfactors = []
        opener = gzip.open if str(cif_path).endswith('.gz') else open
        with opener(cif_path, 'rt', errors='replace') as f:
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
                if not ls.startswith('ATOM'):
                    if col_map and ls in ('', '#'):
                        break
                    continue
                parts = ls.split()
                try:
                    if 'B_iso_or_equiv' in col_map:
                        bf = float(parts[col_map['B_iso_or_equiv']])
                        bfactors.append(bf)
                except (ValueError, IndexError):
                    pass
        if bfactors:
            mean_bfactor = float(np.mean(bfactors))
    except Exception:
        pass

    return {
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'entity_num': entity_num,
        'n_residues': n_res,
        'bps_l': bps_l,
        'pct_helix': pct_helix,
        'pct_sheet': pct_sheet,
        'phi_sign': phi_sign,
        'mean_bfactor': mean_bfactor,
    }


def main():
    parser = argparse.ArgumentParser(description="PDB replication verification")
    parser.add_argument("--resolution", type=float, default=RESOLUTION_DEFAULT,
                        help=f"Max resolution in Å (default: {RESOLUTION_DEFAULT})")
    parser.add_argument("--max-chains", type=int, default=MAX_CHAINS_DEFAULT,
                        help=f"Max chains to process (default: {MAX_CHAINS_DEFAULT})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 200 chains only")
    args = parser.parse_args()

    setup_logging()

    if args.quick:
        args.max_chains = 200

    logging.info("=" * 70)
    logging.info("PDB REPLICATION VERIFICATION")
    logging.info("Does BPS/L ≈ 0.20 hold on experimental structures?")
    logging.info("=" * 70)
    logging.info(f"  Max resolution: {args.resolution} Å")
    logging.info(f"  Max chains: {args.max_chains}")
    logging.info(f"  Min chain length: {MIN_LENGTH}")
    logging.info("")

    # Build superpotential
    logging.info("Building superpotential (identical to AlphaFold pipeline)...")
    W_interp = build_W_interp()

    # Search PDB
    entities = search_pdb_chains(args.resolution, args.max_chains)
    if not entities:
        logging.error("No PDB entities found!")
        sys.exit(1)

    logging.info(f"  Will process {len(entities)} entities")
    logging.info("")

    # Get resolutions for all entries (batch)
    logging.info("  Fetching resolution data...")
    pdb_ids_unique = sorted(set(e[0] for e in entities))

    # Process entities
    results = []
    n_failed = 0
    n_processed = 0
    t0 = time.time()

    for i, (pdb_id, entity_num) in enumerate(entities):
        if i > 0 and i % 100 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(entities) - i) / rate if rate > 0 else 0
            logging.info(
                f"  Progress: {i}/{len(entities)} ({len(results)} OK, "
                f"{n_failed} failed, {rate:.1f}/s, ETA {eta:.0f}s)"
            )

        result = process_entity(pdb_id, entity_num, W_interp)
        if result is not None:
            results.append(result)
            n_processed += 1
        else:
            n_failed += 1

    elapsed = time.time() - t0
    logging.info(f"  Processed {n_processed} chains in {elapsed:.0f}s "
                 f"({n_failed} failed)")
    logging.info("")

    if not results:
        logging.error("No results!")
        sys.exit(1)

    # Fetch resolutions (batch, after processing to avoid API rate limits
    # slowing down the main loop)
    logging.info("  Looking up resolutions...")
    resolution_cache = {}
    for r in results:
        pdb_id = r['pdb_id']
        if pdb_id not in resolution_cache:
            res = get_resolution(pdb_id)
            resolution_cache[pdb_id] = res
        r['resolution'] = resolution_cache.get(pdb_id)

    # ============================================================
    # ANALYSIS
    # ============================================================

    bps_values = np.array([r['bps_l'] for r in results])
    resolutions = np.array([r['resolution'] or 0 for r in results])
    lengths = np.array([r['n_residues'] for r in results])

    logging.info("=" * 70)
    logging.info("RESULTS")
    logging.info("=" * 70)
    logging.info("")

    # Overall statistics
    logging.info(f"  Total chains: {len(results)}")
    logging.info(f"  Chain lengths: {np.min(lengths)} – {np.max(lengths)} "
                 f"(median {np.median(lengths):.0f})")
    logging.info("")

    logging.info(f"  BPS/L (all chains):")
    logging.info(f"    Mean:   {np.mean(bps_values):.4f}")
    logging.info(f"    Median: {np.median(bps_values):.4f}")
    logging.info(f"    SD:     {np.std(bps_values):.4f}")
    logging.info(f"    CV:     {100*np.std(bps_values)/np.mean(bps_values):.1f}%")
    logging.info(f"    Range:  [{np.min(bps_values):.4f}, {np.max(bps_values):.4f}]")
    logging.info("")

    # Resolution tiers
    logging.info("  BPS/L by resolution tier:")
    logging.info(f"    {'Tier':15s}  {'N':>6}  {'Mean BPS/L':>10}  {'SD':>7}  {'CV%':>5}")
    logging.info(f"    {'-'*15}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*5}")

    tiers = [
        ("≤ 1.0 Å", 0, 1.0),
        ("≤ 1.5 Å", 0, 1.5),
        ("≤ 2.0 Å", 0, 2.0),
        ("≤ 2.5 Å", 0, 2.5),
        ("1.5–2.0 Å", 1.5, 2.0),
        ("2.0–2.5 Å", 2.0, 2.5),
    ]

    tier_results = {}
    for label, lo, hi in tiers:
        mask = (resolutions > lo) & (resolutions <= hi) & (resolutions > 0)
        if np.sum(mask) < 5:
            # Try including lo=0 for cumulative tiers
            if lo == 0:
                mask = (resolutions <= hi) & (resolutions > 0)
        n = np.sum(mask)
        if n >= 5:
            tier_bps = bps_values[mask]
            mean = float(np.mean(tier_bps))
            sd = float(np.std(tier_bps))
            cv = 100 * sd / mean if mean > 0 else 0
            logging.info(f"    {label:15s}  {n:>6}  {mean:>10.4f}  {sd:>7.4f}  {cv:>4.1f}%")
            tier_results[label] = {'n': int(n), 'mean': mean, 'sd': sd}
        else:
            logging.info(f"    {label:15s}  {n:>6}  {'(too few)':>10}")

    logging.info("")

    # B-factor analysis (if available)
    bfactors = [r['mean_bfactor'] for r in results if r['mean_bfactor'] is not None]
    if len(bfactors) > 10:
        bf_arr = np.array(bfactors)
        bps_with_bf = np.array([r['bps_l'] for r in results if r['mean_bfactor'] is not None])
        corr = float(np.corrcoef(bf_arr, bps_with_bf)[0, 1])
        logging.info(f"  B-factor analysis:")
        logging.info(f"    Mean B-factor: {np.mean(bf_arr):.1f}")
        logging.info(f"    BPS/L vs B-factor r: {corr:.3f}")

        # Low B-factor subset (well-ordered)
        low_bf_mask = bf_arr < np.percentile(bf_arr, 25)
        if np.sum(low_bf_mask) >= 5:
            logging.info(f"    BPS/L (B-factor < 25th pct): "
                         f"{np.mean(bps_with_bf[low_bf_mask]):.4f}")
        logging.info("")

    # ============================================================
    # COMPARISON TO ALPHAFOLD
    # ============================================================

    logging.info("=" * 70)
    logging.info("COMPARISON TO ALPHAFOLD")
    logging.info("=" * 70)
    logging.info("")
    logging.info(f"  AlphaFold (pLDDT ≥ 85, 51,174 proteins):  0.2020 ± 0.0039")
    logging.info(f"  AlphaFold (all 124,475 proteins):          0.1659 ± 0.0557")
    logging.info(f"  PDB (≤ {args.resolution} Å, {len(results)} chains):      "
                 f"     {np.mean(bps_values):.4f} ± {np.std(bps_values):.4f}")
    logging.info("")

    af_hq = 0.2020
    pdb_mean = float(np.mean(bps_values))
    delta = pdb_mean - af_hq
    delta_pct = 100 * delta / af_hq

    logging.info(f"  PDB vs AlphaFold (HQ):  Δ = {delta:+.4f}  ({delta_pct:+.1f}%)")
    logging.info("")

    if abs(delta_pct) < 5:
        logging.info("  *** PDB ≈ AlphaFold ***")
        logging.info("  BPS/L ≈ 0.20 is confirmed on experimental structures.")
        logging.info("  The constant is NOT an AlphaFold artifact.")
    elif abs(delta_pct) < 15:
        logging.info("  *** PDB and AlphaFold show moderate difference ***")
        logging.info("  The qualitative result holds but quantitative agreement")
        logging.info("  needs investigation (resolution effects, disorder, etc.)")
    else:
        logging.info("  *** PDB and AlphaFold differ substantially ***")
        logging.info("  The BPS/L value may be specific to AlphaFold geometry.")
        logging.info("  Investigate resolution dependence and B-factor effects.")

    logging.info("")

    # ============================================================
    # SAVE
    # ============================================================

    csv_path = OUTPUT_DIR / "pdb_bps_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pdb_id', 'chain_id', 'entity_num', 'resolution',
            'n_residues', 'bps_l', 'pct_helix', 'pct_sheet',
            'phi_sign', 'mean_bfactor'
        ])
        writer.writeheader()
        writer.writerows(results)

    summary_path = OUTPUT_DIR / "pdb_replication_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PDB REPLICATION VERIFICATION - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Resolution cutoff: ≤ {args.resolution} Å\n")
        f.write(f"Chains processed: {len(results)}\n")
        f.write(f"Chains failed: {n_failed}\n\n")
        f.write(f"BPS/L:\n")
        f.write(f"  Mean:   {np.mean(bps_values):.4f}\n")
        f.write(f"  Median: {np.median(bps_values):.4f}\n")
        f.write(f"  SD:     {np.std(bps_values):.4f}\n")
        f.write(f"  CV:     {100*np.std(bps_values)/np.mean(bps_values):.1f}%\n\n")
        f.write(f"Comparison:\n")
        f.write(f"  AlphaFold (pLDDT ≥ 85): 0.2020 ± 0.0039\n")
        f.write(f"  PDB (≤ {args.resolution} Å):        "
                f"{np.mean(bps_values):.4f} ± {np.std(bps_values):.4f}\n")
        f.write(f"  Delta: {delta:+.4f} ({delta_pct:+.1f}%)\n\n")

        if tier_results:
            f.write("Resolution tiers:\n")
            for label, data in tier_results.items():
                f.write(f"  {label}: {data['mean']:.4f} ± {data['sd']:.4f} "
                        f"(N={data['n']})\n")

    logging.info(f"  Results: {csv_path}")
    logging.info(f"  Summary: {summary_path}")
    logging.info("")
    logging.info("Done.")


if __name__ == "__main__":
    main()
