#!/usr/bin/env python3
"""BPS Validation Controls — peer review response tests.

Reads from the existing BPS database and CIF cache to run 6 independent
validation tests.  Produces validation_report.md.

Tests (in priority order):
  1. Shuffled control    — is BPS/L a sequential property or a statistical artifact?
  2. W-construction robustness — is BPS/L universal across superpotential variants?
  3. Fold-class conditioning  — does BPS/L ≈ 0.20 hold within dominant fold classes?
  4. Gly/Pro composition      — do extreme Gly/Pro proteins deviate from 0.20?
  5. Basin transition matrix   — can BPS/L ≈ 0.20 be derived from first principles?
  6. Markov null model simulation — does a first-order Markov chain reproduce the
     tightness (CV ≈ 2%) as well as the mean?

Usage:
    python bps_validate_controls.py
"""

import sqlite3
import numpy as np
from pathlib import Path
import sys
import time
import math
import logging
from datetime import datetime
from collections import defaultdict

# Import from existing pipeline
sys.path.insert(0, str(Path(__file__).parent))
from bps_process import (
    build_superpotential,
    find_cached_cifs,
    _parse_atoms_from_cif,
    _calc_dihedral,
    determine_phi_sign,
    CACHE_DIR,
    DB_PATH,
    PROTEOMES,
)
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wilcoxon, ks_2samp

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_N = 2000        # proteins to sample for CIF-based tests
HQ_THRESHOLD = 85      # pLDDT cutoff for high-quality proteins
SHUFFLE_REPS = 10      # shuffle repetitions per protein
RNG_SEED = 42          # reproducible sampling

# Von Mises mixture components (duplicated from bps_process.build_superpotential
# so we can build modified variants for robustness testing)
VM_COMPONENTS = [
    {'weight': 0.35, 'mu_phi': -63,  'mu_psi': -43,  'kappa_phi': 12.0, 'kappa_psi': 10.0, 'rho':  2.0},
    {'weight': 0.05, 'mu_phi': -60,  'mu_psi': -27,  'kappa_phi':  8.0, 'kappa_psi':  6.0, 'rho':  1.0},
    {'weight': 0.25, 'mu_phi': -120, 'mu_psi': 135,  'kappa_phi':  4.0, 'kappa_psi':  3.5, 'rho': -1.5},
    {'weight': 0.05, 'mu_phi': -140, 'mu_psi': 155,  'kappa_phi':  5.0, 'kappa_psi':  4.0, 'rho': -1.0},
    {'weight': 0.12, 'mu_phi': -75,  'mu_psi': 150,  'kappa_phi':  8.0, 'kappa_psi':  5.0, 'rho':  0.5},
    {'weight': 0.05, 'mu_phi': -95,  'mu_psi': 150,  'kappa_phi':  3.0, 'kappa_psi':  4.0, 'rho':  0.0},
    {'weight': 0.03, 'mu_phi':  57,  'mu_psi':  40,  'kappa_phi':  6.0, 'kappa_psi':  6.0, 'rho':  1.5},
    {'weight': 0.03, 'mu_phi':  60,  'mu_psi': -130, 'kappa_phi':  5.0, 'kappa_psi':  4.0, 'rho':  0.0},
    {'weight': 0.01, 'mu_phi':  75,  'mu_psi': -65,  'kappa_phi':  5.0, 'kappa_psi':  5.0, 'rho':  0.0},
    {'weight': 0.06, 'mu_phi':   0,  'mu_psi':   0,  'kappa_phi':  0.01,'kappa_psi':  0.01,'rho':  0.0},
]

BASIN_NAMES = ['alpha', 'beta', 'other']


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ============================================================
# DATABASE HELPERS
# ============================================================

def find_database():
    """Locate the BPS database file."""
    candidates = [
        DB_PATH,
        Path("bps_output/bps_results.db"),
        Path("alphafold_bps_results/bps_database.db"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def get_hq_proteins(conn):
    """Get all high-quality proteins (pLDDT >= threshold) from the DB."""
    rows = conn.execute(
        "SELECT uniprot_id, organism, L, bps_norm, plddt_mean, "
        "       pct_helix, pct_sheet, pct_coil "
        "FROM proteins WHERE plddt_mean >= ?",
        (HQ_THRESHOLD,)
    ).fetchall()
    return [{'uniprot_id': r[0], 'organism': r[1], 'L': r[2],
             'bps_norm': r[3], 'plddt_mean': r[4],
             'pct_helix': r[5], 'pct_sheet': r[6], 'pct_coil': r[7]}
            for r in rows]


def get_db_stats(conn):
    """Get basic database statistics."""
    total = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
    hq = conn.execute(
        "SELECT COUNT(*) FROM proteins WHERE plddt_mean >= ?",
        (HQ_THRESHOLD,)
    ).fetchone()[0]
    organisms = conn.execute(
        "SELECT COUNT(DISTINCT organism) FROM proteins"
    ).fetchone()[0]
    return {'total_proteins': total, 'hq_proteins': hq, 'n_organisms': organisms}


# ============================================================
# SAMPLING
# ============================================================

def sample_proteins(conn, n=SAMPLE_N):
    """Sample n HQ proteins uniformly across organisms (stratified)."""
    rng = np.random.default_rng(RNG_SEED)

    orgs = conn.execute(
        "SELECT DISTINCT organism FROM proteins WHERE plddt_mean >= ?",
        (HQ_THRESHOLD,)
    ).fetchall()
    orgs = [r[0] for r in orgs]

    if not orgs:
        logging.error("No high-quality proteins found in database.")
        return []

    per_org = max(1, n // len(orgs))
    sampled = []

    for org in sorted(orgs):
        rows = conn.execute(
            "SELECT uniprot_id, organism, bps_norm FROM proteins "
            "WHERE organism = ? AND plddt_mean >= ? ORDER BY uniprot_id",
            (org, HQ_THRESHOLD)
        ).fetchall()
        if not rows:
            continue
        k = min(per_org, len(rows))
        indices = rng.choice(len(rows), size=k, replace=False)
        for idx in indices:
            r = rows[idx]
            sampled.append({
                'uniprot_id': r[0], 'organism': r[1], 'db_bps_norm': r[2],
            })

    # Trim to target size if oversampled
    if len(sampled) > n:
        indices = rng.choice(len(sampled), size=n, replace=False)
        sampled = [sampled[i] for i in sorted(indices)]

    logging.info(f"Sampled {len(sampled)} proteins across {len(orgs)} organisms")
    return sampled


# ============================================================
# CIF PARSING HELPERS
# ============================================================

def find_cif_for_protein(uniprot_id, organism):
    """Find CIF file for a specific protein."""
    org_dir = CACHE_DIR / organism
    if not org_dir.exists():
        return None
    for cif in org_dir.glob(f"AF-{uniprot_id}-F1-model_v*.cif"):
        if cif.stat().st_size > 100:
            return cif
    return None


def parse_cif_full(filepath):
    """Parse CIF file, extracting backbone atoms and residue names.

    Returns (residues, residue_names) where:
      residues: list of dicts with 'N', 'CA', 'C', 'bfactor' keys
      residue_names: list of 3-letter residue codes (parallel to residues)
    Returns (None, None) on failure.
    """
    atom_data = {}
    resnames = {}
    try:
        with open(str(filepath), 'r') as f:
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
                    bf = float(parts[col_map.get('B_iso_or_equiv', 14)])
                    comp_id = parts[col_map.get('label_comp_id', 5)]
                    if resnum not in atom_data:
                        atom_data[resnum] = {'bfactor': bf}
                        resnames[resnum] = comp_id
                    atom_data[resnum][atom_name] = np.array([x, y, z])
                except (ValueError, IndexError, KeyError):
                    continue
    except Exception:
        return None, None

    sorted_keys = sorted(atom_data.keys())
    res_list = []
    name_list = []
    for rn in sorted_keys:
        r = atom_data[rn]
        if 'N' in r and 'CA' in r and 'C' in r:
            res_list.append(r)
            name_list.append(resnames.get(rn, 'UNK'))

    if len(res_list) < 30:
        return None, None
    return res_list, name_list


def compute_phi_psi(residues, phi_sign):
    """Compute (phi, psi) in radians for each residue.
    Applies PHI_SIGN correction to both phi and psi."""
    n = len(residues)
    phi_psi = []
    for i in range(n):
        raw_phi = (_calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                                  residues[i]['CA'], residues[i]['C'])
                   if i > 0 else None)
        raw_psi = (_calc_dihedral(residues[i]['N'], residues[i]['CA'],
                                  residues[i]['C'], residues[i+1]['N'])
                   if i < n - 1 else None)
        phi = phi_sign * raw_phi if raw_phi is not None else None
        psi = phi_sign * raw_psi if raw_psi is not None else None
        phi_psi.append((phi, psi))
    return phi_psi


def compute_bps_norm(phi_psi, W_interp):
    """Compute BPS/L from a (phi, psi) list. Returns float or None."""
    valid = [(p, s) for p, s in phi_psi if p is not None and s is not None]
    if len(valid) < 3:
        return None
    phis = np.array([v[0] for v in valid])
    psis = np.array([v[1] for v in valid])
    W_chain = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_chain))
    return float(np.sum(dW)) / len(valid)


def compute_bps_norm_from_valid(valid_pairs, W_interp):
    """Compute BPS/L from pre-filtered valid (phi, psi) pairs."""
    if len(valid_pairs) < 3:
        return None
    phis = np.array([v[0] for v in valid_pairs])
    psis = np.array([v[1] for v in valid_pairs])
    W_chain = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_chain))
    return float(np.sum(dW)) / len(valid_pairs)


# ============================================================
# SUPERPOTENTIAL VARIANT BUILDER (Test 2)
# ============================================================

def build_superpotential_variant(kappa_scale=1.0, grid_n=360, epsilon=None):
    """Build W with modified parameters for robustness testing.

    Args:
        kappa_scale: multiply all kappa values by this factor
        grid_n: grid resolution (default 360)
        epsilon: absolute floor for density p (None = use default relative floor)
    """
    N = grid_n
    phi_grid = np.linspace(-np.pi, np.pi, N, endpoint=False)
    psi_grid = np.linspace(-np.pi, np.pi, N, endpoint=False)
    PHI, PSI = np.meshgrid(phi_grid, psi_grid)
    p = np.zeros_like(PHI)

    for c in VM_COMPONENTS:
        mu_p = np.radians(c['mu_phi'])
        mu_s = np.radians(c['mu_psi'])
        dp = PHI - mu_p
        ds = PSI - mu_s
        kp = c['kappa_phi'] * kappa_scale
        ks = c['kappa_psi'] * kappa_scale
        p += c['weight'] * np.exp(kp * np.cos(dp) + ks * np.cos(ds)
                                  + c['rho'] * np.sin(dp) * np.sin(ds))

    p /= np.sum(p) * (phi_grid[1] - phi_grid[0]) * (psi_grid[1] - psi_grid[0])

    if epsilon is not None:
        p = np.maximum(p, epsilon)
    else:
        p = np.maximum(p, np.max(p) * 1e-6)

    # W = -sqrt(P): matches the primary pipeline (bps_process.py).
    # Alternative: W = -ln(P + eps) used in bps/superpotential.py.
    # Three-level decomposition is transform-invariant across W choices.
    W = -np.sqrt(p)
    W = gaussian_filter(W, sigma=1.5)
    return RegularGridInterpolator(
        (psi_grid, phi_grid), W,
        method='linear', bounds_error=False, fill_value=None
    )


# ============================================================
# RAMACHANDRAN BASIN CLASSIFIER (Test 5)
# ============================================================

def classify_basin(phi_deg, psi_deg):
    """Classify (phi, psi) in degrees into a Ramachandran basin.
    Returns index: 0=alpha, 1=beta, 2=other.

    Uses the wide basin definition consistent with all other analysis
    scripts (generate_figures.py, alphafold_pipeline.py, etc.).
    """
    # Alpha-helix (wide definition)
    if -160 < phi_deg < 0 and -120 < psi_deg < 30:
        return 0
    # Beta-sheet (handles +/-180 wrap)
    if -170 < phi_deg < -70 and (psi_deg > 90 or psi_deg < -120):
        return 1
    # Other
    return 2


# ============================================================
# PREPARE SAMPLE DATA (parse CIFs once, reuse for all tests)
# ============================================================

def prepare_sample_data(sample, phi_sign):
    """Parse CIF files for all sampled proteins. Populates phi_psi and
    residue_types fields in-place. Returns count of successfully parsed proteins."""
    n_success = 0
    n_total = len(sample)
    t0 = time.time()

    for i, entry in enumerate(sample):
        cif_path = find_cif_for_protein(entry['uniprot_id'], entry['organism'])
        if cif_path is None:
            entry['phi_psi'] = None
            entry['residue_types'] = None
            continue

        residues, resnames = parse_cif_full(cif_path)
        if residues is None:
            entry['phi_psi'] = None
            entry['residue_types'] = None
            continue

        phi_psi = compute_phi_psi(residues, phi_sign)
        entry['phi_psi'] = phi_psi
        entry['residue_types'] = resnames
        n_success += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            logging.info(f"  Parsed {i+1}/{n_total} CIFs "
                         f"({n_success} OK, {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    logging.info(f"  CIF parsing complete: {n_success}/{n_total} successful")
    return n_success


# ============================================================
# TEST 1: SHUFFLED CONTROL
# ============================================================

def test_shuffled_control(sample, W_interp):
    """Test 1: Compare real BPS/L to shuffled-sequence BPS/L.

    If shuffled ≈ real, BPS/L is a trivial basin-statistics artifact.
    If shuffled ≠ real, BPS/L encodes genuine sequential structure.
    """
    logging.info("=" * 60)
    logging.info("TEST 1: SHUFFLED CONTROL")
    logging.info("=" * 60)

    rng = np.random.default_rng(RNG_SEED)
    real_vals = []
    shuffled_vals = []
    per_organism_real = defaultdict(list)
    per_organism_shuffled = defaultdict(list)
    n_processed = 0

    for entry in sample:
        if entry.get('phi_psi') is None:
            continue

        # Filter to valid pairs only
        valid = [(p, s) for p, s in entry['phi_psi']
                 if p is not None and s is not None]
        if len(valid) < 10:
            continue

        # Real BPS/L
        real_bps = compute_bps_norm_from_valid(valid, W_interp)
        if real_bps is None:
            continue

        # Shuffled BPS/L: shuffle 10 times, take mean
        shuf_scores = []
        for _ in range(SHUFFLE_REPS):
            shuffled = list(valid)
            rng.shuffle(shuffled)
            s_bps = compute_bps_norm_from_valid(shuffled, W_interp)
            if s_bps is not None:
                shuf_scores.append(s_bps)

        if not shuf_scores:
            continue

        mean_shuf = float(np.mean(shuf_scores))
        real_vals.append(real_bps)
        shuffled_vals.append(mean_shuf)
        per_organism_real[entry['organism']].append(real_bps)
        per_organism_shuffled[entry['organism']].append(mean_shuf)
        n_processed += 1

        if n_processed % 200 == 0:
            logging.info(f"  Processed {n_processed} proteins...")

    if n_processed < 10:
        logging.warning("  Too few proteins for shuffled control test.")
        return {'status': 'SKIPPED', 'reason': 'Insufficient data'}

    real_arr = np.array(real_vals)
    shuf_arr = np.array(shuffled_vals)

    mean_real = float(np.mean(real_arr))
    mean_shuf = float(np.mean(shuf_arr))
    std_real = float(np.std(real_arr, ddof=1))
    std_shuf = float(np.std(shuf_arr, ddof=1))

    # Cohen's d (pooled standard deviation)
    pooled_std = math.sqrt((std_real**2 + std_shuf**2) / 2)
    cohens_d = (mean_real - mean_shuf) / pooled_std if pooled_std > 0 else 0.0

    # Wilcoxon signed-rank test (paired: real vs shuffled for each protein)
    try:
        stat, p_value = wilcoxon(real_arr, shuf_arr)
    except Exception:
        stat, p_value = None, None

    # Per-organism breakdown
    org_breakdown = {}
    for org in sorted(per_organism_real.keys()):
        org_real = np.array(per_organism_real[org])
        org_shuf = np.array(per_organism_shuffled[org])
        org_breakdown[org] = {
            'n': len(org_real),
            'mean_real': float(np.mean(org_real)),
            'mean_shuffled': float(np.mean(org_shuf)),
            'delta': float(np.mean(org_real) - np.mean(org_shuf)),
        }

    # Determine pass/fail
    # Pass if effect size |d| > 0.5 and p < 0.001
    if p_value is not None and p_value < 0.001 and abs(cohens_d) > 0.5:
        verdict = 'PASS'
    elif p_value is not None and p_value < 0.05 and abs(cohens_d) > 0.2:
        verdict = 'MARGINAL'
    else:
        verdict = 'FAIL'

    logging.info(f"  Result: mean_real={mean_real:.4f}, mean_shuffled={mean_shuf:.4f}, "
                 f"Cohen's d={cohens_d:.3f}, p={p_value}")
    logging.info(f"  Verdict: {verdict}")

    return {
        'status': verdict,
        'n_proteins': n_processed,
        'mean_real': mean_real,
        'std_real': std_real,
        'mean_shuffled': mean_shuf,
        'std_shuffled': std_shuf,
        'cohens_d': cohens_d,
        'wilcoxon_stat': stat,
        'p_value': p_value,
        'per_organism': org_breakdown,
        'real_values': real_vals,
        'shuffled_values': shuffled_vals,
    }


# ============================================================
# TEST 2: W-CONSTRUCTION ROBUSTNESS
# ============================================================

def test_w_robustness(sample, W_baseline):
    """Test 2: Sensitivity of BPS/L to superpotential construction choices."""
    logging.info("=" * 60)
    logging.info("TEST 2: W-CONSTRUCTION ROBUSTNESS")
    logging.info("=" * 60)

    # Get valid phi_psi data from sample
    valid_entries = [e for e in sample if e.get('phi_psi') is not None]
    if len(valid_entries) < 50:
        logging.warning("  Too few proteins for W-robustness test.")
        return {'status': 'SKIPPED', 'reason': 'Insufficient data'}

    # Define W variants
    variants = [
        {'name': 'kappa x 0.50',  'kappa_scale': 0.50, 'grid_n': 360, 'epsilon': None},
        {'name': 'kappa x 0.75',  'kappa_scale': 0.75, 'grid_n': 360, 'epsilon': None},
        {'name': 'kappa x 1.00 (baseline)', 'kappa_scale': 1.00, 'grid_n': 360, 'epsilon': None},
        {'name': 'kappa x 1.25',  'kappa_scale': 1.25, 'grid_n': 360, 'epsilon': None},
        {'name': 'kappa x 1.50',  'kappa_scale': 1.50, 'grid_n': 360, 'epsilon': None},
        {'name': 'Grid 180x180',  'kappa_scale': 1.00, 'grid_n': 180, 'epsilon': None},
        {'name': 'Grid 720x720',  'kappa_scale': 1.00, 'grid_n': 720, 'epsilon': None},
        {'name': 'eps = 1e-8',    'kappa_scale': 1.00, 'grid_n': 360, 'epsilon': 1e-8},
        {'name': 'eps = 1e-6',    'kappa_scale': 1.00, 'grid_n': 360, 'epsilon': 1e-6},
        {'name': 'eps = 1e-4',    'kappa_scale': 1.00, 'grid_n': 360, 'epsilon': 1e-4},
        {'name': 'eps = 1e-2',    'kappa_scale': 1.00, 'grid_n': 360, 'epsilon': 1e-2},
    ]

    results_table = []

    for vi, var in enumerate(variants):
        logging.info(f"  Building W variant {vi+1}/{len(variants)}: {var['name']}")
        t0 = time.time()

        # Build the W variant (use baseline for kappa=1, grid=360, eps=None)
        is_baseline = (var['kappa_scale'] == 1.0 and var['grid_n'] == 360
                       and var['epsilon'] is None)
        if is_baseline:
            W_var = W_baseline
        else:
            W_var = build_superpotential_variant(
                kappa_scale=var['kappa_scale'],
                grid_n=var['grid_n'],
                epsilon=var['epsilon'],
            )

        # Compute BPS/L for all valid proteins with this W variant
        per_org_bps = defaultdict(list)
        all_bps = []

        for entry in valid_entries:
            valid_pairs = [(p, s) for p, s in entry['phi_psi']
                          if p is not None and s is not None]
            if len(valid_pairs) < 3:
                continue
            bps = compute_bps_norm_from_valid(valid_pairs, W_var)
            if bps is not None:
                all_bps.append(bps)
                per_org_bps[entry['organism']].append(bps)

        if not all_bps:
            results_table.append({
                'name': var['name'], 'mean_bps': None, 'cv_pct': None, 'delta': None,
            })
            continue

        mean_bps = float(np.mean(all_bps))

        # Cross-organism CV: std of per-organism means / grand mean
        org_means = [float(np.mean(v)) for v in per_org_bps.values() if len(v) >= 5]
        if len(org_means) >= 2:
            cv_pct = float(np.std(org_means, ddof=1) / np.mean(org_means) * 100)
        else:
            cv_pct = None

        elapsed = time.time() - t0
        logging.info(f"    Mean BPS/L = {mean_bps:.4f}, CV = {cv_pct:.1f}% ({elapsed:.1f}s)")

        results_table.append({
            'name': var['name'],
            'mean_bps': mean_bps,
            'cv_pct': cv_pct,
            'delta': None,  # filled in below relative to baseline
        })

    # Compute deltas relative to baseline
    baseline_mean = None
    for row in results_table:
        if 'baseline' in row['name']:
            baseline_mean = row['mean_bps']
            break

    if baseline_mean is not None:
        for row in results_table:
            if row['mean_bps'] is not None:
                row['delta'] = row['mean_bps'] - baseline_mean

    # Verdict: pass if all CVs ≤ 4%
    all_cvs = [r['cv_pct'] for r in results_table if r['cv_pct'] is not None]
    max_cv = max(all_cvs) if all_cvs else None
    if max_cv is not None and max_cv <= 4.0:
        verdict = 'PASS'
    elif max_cv is not None and max_cv <= 6.0:
        verdict = 'MARGINAL'
    else:
        verdict = 'FAIL'

    logging.info(f"  Max CV across variants: {max_cv:.1f}%  Verdict: {verdict}")

    return {
        'status': verdict,
        'table': results_table,
        'max_cv': max_cv,
        'n_proteins': len(valid_entries),
    }


# ============================================================
# TEST 3: FOLD-CLASS-CONDITIONED UNIVERSALITY
# ============================================================

def classify_fold_class(pct_helix, pct_sheet):
    """Classify protein into fold class based on SS percentages."""
    if pct_helix is None or pct_sheet is None:
        return None
    if pct_helix > 15 and pct_sheet > 15:
        return 'Mixed alpha/beta'
    if pct_helix > 40 and pct_sheet <= 15:
        return 'Alpha-rich'
    if pct_sheet > 25 and pct_helix <= 15:
        return 'Beta-rich'
    return 'Other'


def test_fold_class_conditioning(conn):
    """Test 3: BPS/L universality within fold classes.

    Tests whether BPS/L ≈ 0.20 holds within the dominant fold class (mixed α/β)
    across organisms, or only after mixture-averaging.
    """
    logging.info("=" * 60)
    logging.info("TEST 3: FOLD-CLASS-CONDITIONED UNIVERSALITY")
    logging.info("=" * 60)

    hq = get_hq_proteins(conn)
    if len(hq) < 100:
        logging.warning("  Too few HQ proteins for fold-class test.")
        return {'status': 'SKIPPED', 'reason': 'Insufficient data'}

    # Classify each protein
    fold_data = defaultdict(lambda: defaultdict(list))  # fold_class -> organism -> [bps_norm]
    fold_counts = defaultdict(int)

    for p in hq:
        fc = classify_fold_class(p['pct_helix'], p['pct_sheet'])
        if fc is None or p['bps_norm'] is None:
            continue
        fold_data[fc][p['organism']].append(p['bps_norm'])
        fold_counts[fc] += 1

    results_table = []
    for fc in ['Mixed alpha/beta', 'Alpha-rich', 'Beta-rich', 'Other']:
        if fc not in fold_data:
            results_table.append({
                'fold_class': fc, 'n_proteins': 0, 'n_organisms': 0,
                'mean_bps': None, 'cv_pct': None,
            })
            continue

        org_data = fold_data[fc]
        n_proteins = fold_counts[fc]
        n_organisms = len(org_data)

        # Per-organism mean BPS/L
        org_means = []
        for org, vals in org_data.items():
            if len(vals) >= 5:  # need enough proteins per organism for a stable mean
                org_means.append(float(np.mean(vals)))

        if len(org_means) >= 2:
            grand_mean = float(np.mean(org_means))
            cv_pct = float(np.std(org_means, ddof=1) / grand_mean * 100)
        elif len(org_means) == 1:
            grand_mean = org_means[0]
            cv_pct = None
        else:
            grand_mean = None
            cv_pct = None

        results_table.append({
            'fold_class': fc,
            'n_proteins': n_proteins,
            'n_organisms': n_organisms,
            'mean_bps': grand_mean,
            'cv_pct': cv_pct,
        })

        if grand_mean is not None:
            logging.info(f"  {fc}: N={n_proteins}, orgs={n_organisms}, "
                         f"mean={grand_mean:.4f}, CV={cv_pct:.1f}%" if cv_pct else
                         f"  {fc}: N={n_proteins}, orgs={n_organisms}, "
                         f"mean={grand_mean:.4f}")

    # Verdict: check ALL fold classes with sufficient data, not just one
    cv_results = [r for r in results_table
                  if r['cv_pct'] is not None and r['n_proteins'] >= 20]
    if cv_results:
        max_cv = max(r['cv_pct'] for r in cv_results)
        mean_cv = np.mean([r['cv_pct'] for r in cv_results])
        if max_cv <= 5.0:
            verdict = 'PASS — geometric universality (all classes CV <= 5%)'
        elif mean_cv <= 5.0:
            verdict = f'MARGINAL — mean CV = {mean_cv:.1f}%, max CV = {max_cv:.1f}%'
        else:
            verdict = f'FAIL — mean CV = {mean_cv:.1f}% across fold classes'
    else:
        verdict = 'INCONCLUSIVE'

    logging.info(f"  Verdict: {verdict}")

    return {
        'status': verdict,
        'table': results_table,
    }


# ============================================================
# TEST 4: GLY/PRO COMPOSITION EFFECTS
# ============================================================

# Three-letter to one-letter mapping for Gly and Pro
RESIDUE_MAP_GP = {
    'GLY': 'G', 'PRO': 'P',
}


def test_gly_pro(sample, W_interp):
    """Test 4: Do proteins with extreme Gly/Pro content deviate from BPS/L ≈ 0.20?"""
    logging.info("=" * 60)
    logging.info("TEST 4: GLY/PRO COMPOSITION EFFECTS")
    logging.info("=" * 60)

    entries_with_comp = []
    for entry in sample:
        if entry.get('phi_psi') is None or entry.get('residue_types') is None:
            continue
        restypes = entry['residue_types']
        n_res = len(restypes)
        if n_res < 10:
            continue

        n_gly = sum(1 for r in restypes if r == 'GLY')
        n_pro = sum(1 for r in restypes if r == 'PRO')
        pct_gly = n_gly / n_res * 100
        pct_pro = n_pro / n_res * 100

        # Compute BPS/L
        bps = compute_bps_norm(entry['phi_psi'], W_interp)
        if bps is None:
            continue

        entries_with_comp.append({
            'uniprot_id': entry['uniprot_id'],
            'organism': entry['organism'],
            'pct_gly': pct_gly,
            'pct_pro': pct_pro,
            'bps_norm': bps,
            'L': n_res,
        })

    if len(entries_with_comp) < 40:
        logging.warning("  Too few proteins with composition data.")
        return {'status': 'SKIPPED', 'reason': 'Insufficient data'}

    logging.info(f"  {len(entries_with_comp)} proteins with composition data")

    # Bin by %Gly quartiles
    gly_vals = np.array([e['pct_gly'] for e in entries_with_comp])
    pro_vals = np.array([e['pct_pro'] for e in entries_with_comp])
    bps_vals = np.array([e['bps_norm'] for e in entries_with_comp])

    gly_quartiles = np.percentile(gly_vals, [25, 50, 75])
    pro_quartiles = np.percentile(pro_vals, [25, 50, 75])

    def quartile_label(val, q):
        if val <= q[0]:
            return 'Q1 (lowest)'
        elif val <= q[1]:
            return 'Q2'
        elif val <= q[2]:
            return 'Q3'
        else:
            return 'Q4 (highest)'

    # Gly bins
    gly_bins = defaultdict(list)
    for e in entries_with_comp:
        label = quartile_label(e['pct_gly'], gly_quartiles)
        gly_bins[label].append(e['bps_norm'])

    gly_table = []
    for label in ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']:
        vals = gly_bins.get(label, [])
        if vals:
            gly_table.append({
                'bin': label,
                'n': len(vals),
                'mean_bps': float(np.mean(vals)),
                'std_bps': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
            })

    # Pro bins
    pro_bins = defaultdict(list)
    for e in entries_with_comp:
        label = quartile_label(e['pct_pro'], pro_quartiles)
        pro_bins[label].append(e['bps_norm'])

    pro_table = []
    for label in ['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)']:
        vals = pro_bins.get(label, [])
        if vals:
            pro_table.append({
                'bin': label,
                'n': len(vals),
                'mean_bps': float(np.mean(vals)),
                'std_bps': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0,
            })

    # Report quartile boundaries
    gly_boundaries = {
        'q25': float(gly_quartiles[0]),
        'q50': float(gly_quartiles[1]),
        'q75': float(gly_quartiles[2]),
    }
    pro_boundaries = {
        'q25': float(pro_quartiles[0]),
        'q50': float(pro_quartiles[1]),
        'q75': float(pro_quartiles[2]),
    }

    # Check if extreme bins deviate significantly
    gly_bps_vals = [r['mean_bps'] for r in gly_table if r['mean_bps'] is not None]
    pro_bps_vals = [r['mean_bps'] for r in pro_table if r['mean_bps'] is not None]
    max_gly_spread = (max(gly_bps_vals) - min(gly_bps_vals)) if len(gly_bps_vals) >= 2 else 0
    max_pro_spread = (max(pro_bps_vals) - min(pro_bps_vals)) if len(pro_bps_vals) >= 2 else 0

    # Verdict: if Q1-Q4 spread < 0.02 for both, composition doesn't matter much
    if max_gly_spread < 0.02 and max_pro_spread < 0.02:
        verdict = 'PASS — negligible composition effect'
    elif max_gly_spread < 0.04 and max_pro_spread < 0.04:
        verdict = 'MARGINAL'
    else:
        verdict = 'NOTABLE — composition affects BPS/L'

    for row in gly_table:
        logging.info(f"  Gly {row['bin']}: N={row['n']}, mean BPS/L={row['mean_bps']:.4f}")
    for row in pro_table:
        logging.info(f"  Pro {row['bin']}: N={row['n']}, mean BPS/L={row['mean_bps']:.4f}")
    logging.info(f"  Verdict: {verdict}")

    return {
        'status': verdict,
        'n_proteins': len(entries_with_comp),
        'gly_table': gly_table,
        'pro_table': pro_table,
        'gly_boundaries': gly_boundaries,
        'pro_boundaries': pro_boundaries,
        'gly_spread': max_gly_spread,
        'pro_spread': max_pro_spread,
    }


# ============================================================
# TEST 5: BASIN TRANSITION MATRIX
# ============================================================

def test_transition_matrix(sample, W_interp):
    """Test 5: Derive BPS/L from Ramachandran basin transition probabilities.

    If predicted ≈ observed within ~5%, we've derived the constant from
    first principles.
    """
    logging.info("=" * 60)
    logging.info("TEST 5: BASIN TRANSITION MATRIX")
    logging.info("=" * 60)

    # Accumulate basin classifications and W values across all proteins
    n_basins = len(BASIN_NAMES)  # 3: alpha, beta, other
    transition_counts = np.zeros((n_basins, n_basins), dtype=np.int64)
    basin_W_sums = np.zeros(n_basins)
    basin_W_counts = np.zeros(n_basins, dtype=np.int64)
    n_processed = 0

    for entry in sample:
        if entry.get('phi_psi') is None:
            continue

        valid = [(p, s) for p, s in entry['phi_psi']
                 if p is not None and s is not None]
        if len(valid) < 10:
            continue

        phis = np.array([v[0] for v in valid])
        psis = np.array([v[1] for v in valid])

        # Compute W value at each residue
        W_vals = W_interp(np.column_stack([psis, phis]))

        # Classify each residue into a basin
        phi_deg = np.degrees(phis)
        psi_deg = np.degrees(psis)
        basins = np.array([classify_basin(phi_deg[j], psi_deg[j])
                          for j in range(len(valid))])

        # Accumulate W values per basin
        for b in range(n_basins):
            mask = basins == b
            basin_W_sums[b] += np.sum(W_vals[mask])
            basin_W_counts[b] += int(np.sum(mask))

        # Accumulate transitions
        for j in range(len(basins) - 1):
            transition_counts[basins[j], basins[j+1]] += 1

        n_processed += 1

    if n_processed < 50:
        logging.warning("  Too few proteins for transition matrix test.")
        return {'status': 'SKIPPED', 'reason': 'Insufficient data'}

    # Mean W per basin
    W_bar = np.zeros(n_basins)
    for b in range(n_basins):
        if basin_W_counts[b] > 0:
            W_bar[b] = basin_W_sums[b] / basin_W_counts[b]

    # Transition matrix (probabilities)
    total_transitions = transition_counts.sum()
    if total_transitions == 0:
        return {'status': 'SKIPPED', 'reason': 'No transitions found'}

    T_prob = transition_counts / total_transitions  # fraction of ALL transitions

    # Predicted BPS/L = Σ_ij T[i→j] × |W̄_i - W̄_j|
    predicted_bps = 0.0
    for i in range(n_basins):
        for j in range(n_basins):
            predicted_bps += T_prob[i, j] * abs(W_bar[i] - W_bar[j])

    # Observed BPS/L (from the sample)
    observed_vals = []
    for entry in sample:
        if entry.get('phi_psi') is None:
            continue
        bps = compute_bps_norm(entry['phi_psi'], W_interp)
        if bps is not None:
            observed_vals.append(bps)

    observed_mean = float(np.mean(observed_vals)) if observed_vals else None

    # Percentage error
    if observed_mean and observed_mean > 0:
        pct_error = abs(predicted_bps - observed_mean) / observed_mean * 100
    else:
        pct_error = None

    # Basin occupancy
    basin_occupancy = {}
    total_residues = int(np.sum(basin_W_counts))
    for b in range(n_basins):
        basin_occupancy[BASIN_NAMES[b]] = {
            'count': int(basin_W_counts[b]),
            'pct': float(basin_W_counts[b] / total_residues * 100) if total_residues > 0 else 0,
            'mean_W': float(W_bar[b]),
        }

    # Format transition matrix as percentages
    T_pct = (transition_counts / total_transitions * 100) if total_transitions > 0 else transition_counts

    # Verdict
    if pct_error is not None and pct_error <= 5:
        verdict = 'PASS — first-principles derivation successful'
    elif pct_error is not None and pct_error <= 15:
        verdict = 'MARGINAL — approximate derivation'
    elif pct_error is not None:
        verdict = f'FAIL — {pct_error:.1f}% error exceeds tolerance'
    else:
        verdict = 'INCONCLUSIVE'

    logging.info(f"  Predicted BPS/L = {predicted_bps:.4f}")
    logging.info(f"  Observed BPS/L  = {observed_mean:.4f}" if observed_mean else
                 "  Observed BPS/L  = N/A")
    logging.info(f"  Error: {pct_error:.1f}%" if pct_error else "  Error: N/A")
    logging.info(f"  Verdict: {verdict}")

    return {
        'status': verdict,
        'n_proteins': n_processed,
        'predicted_bps': predicted_bps,
        'observed_bps': observed_mean,
        'pct_error': pct_error,
        'transition_matrix': T_pct.tolist(),
        'transition_counts': transition_counts.tolist(),
        'total_transitions': int(total_transitions),
        'basin_occupancy': basin_occupancy,
        'W_bar': {BASIN_NAMES[b]: float(W_bar[b]) for b in range(n_basins)},
    }


# ============================================================
# TEST 6: MARKOV NULL MODEL SIMULATION
# ============================================================

def test_markov_simulation(sample, W_interp):
    """Test 6: Markov null model simulation.

    Tests whether a first-order Markov chain over Ramachandran basins
    reproduces not only the mean BPS/L (~0.20) but also the tightness
    (CV ~2%) observed across real proteins.

    Steps:
      1. Build basin transition matrix from sample
      2. Collect per-basin empirical (phi, psi) distributions
      3. Simulate synthetic proteins matching real length distribution
      4. Compare real vs synthetic BPS/L distributions
    """
    logging.info("=" * 60)
    logging.info("TEST 6: MARKOV NULL MODEL SIMULATION")
    logging.info("=" * 60)

    rng = np.random.default_rng(RNG_SEED + 6)  # distinct seed for this test

    # ------------------------------------------------------------------
    # Step 0: Gather valid (phi, psi) sequences and metadata from sample
    # ------------------------------------------------------------------
    real_bps_vals = []
    real_organisms = []
    real_lengths = []
    n_basins = len(BASIN_NAMES)  # 3: alpha, beta, other
    transition_counts = np.zeros((n_basins, n_basins), dtype=np.int64)
    basin_occupancy = np.zeros(n_basins, dtype=np.int64)
    basin_angles = {b: [] for b in range(n_basins)}  # basin -> list of (phi, psi)
    n_processed = 0

    for entry in sample:
        if entry.get('phi_psi') is None:
            continue

        valid = [(p, s) for p, s in entry['phi_psi']
                 if p is not None and s is not None]
        if len(valid) < 10:
            continue

        # Compute real BPS/L
        bps = compute_bps_norm_from_valid(valid, W_interp)
        if bps is None:
            continue

        real_bps_vals.append(bps)
        real_organisms.append(entry['organism'])
        real_lengths.append(len(valid))

        # Classify residues into basins and collect angles
        for k, (phi, psi) in enumerate(valid):
            phi_deg = math.degrees(phi)
            psi_deg = math.degrees(psi)
            b = classify_basin(phi_deg, psi_deg)
            basin_occupancy[b] += 1
            basin_angles[b].append((phi, psi))
            if k > 0:
                prev_phi_deg = math.degrees(valid[k-1][0])
                prev_psi_deg = math.degrees(valid[k-1][1])
                b_prev = classify_basin(prev_phi_deg, prev_psi_deg)
                transition_counts[b_prev, b] += 1

        n_processed += 1

    if n_processed < 50:
        logging.warning("  Too few proteins for Markov simulation test.")
        return {'status': 'SKIPPED', 'reason': 'Insufficient data'}

    logging.info(f"  {n_processed} proteins with valid phi/psi data")

    # ------------------------------------------------------------------
    # Step 1: Build transition matrix (row-normalized)
    # ------------------------------------------------------------------
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    # Avoid division by zero for empty basins
    row_sums_safe = np.where(row_sums > 0, row_sums, 1)
    T_matrix = transition_counts / row_sums_safe

    # Stationary distribution = basin occupancy frequencies
    total_residues = int(basin_occupancy.sum())
    stationary = basin_occupancy / total_residues

    logging.info(f"  Total residues classified: {total_residues:,}")
    for b in range(n_basins):
        logging.info(f"    {BASIN_NAMES[b]}: {basin_occupancy[b]:,} "
                     f"({stationary[b]*100:.1f}%), "
                     f"{len(basin_angles[b])} angle samples")

    # ------------------------------------------------------------------
    # Step 2: Verify per-basin angle distributions are non-empty
    # ------------------------------------------------------------------
    placeholder_angles = {0: (-1.1, -0.44), 1: (-2.1, 2.36), 2: (0.0, 0.0)}
    for b in range(n_basins):
        if len(basin_angles[b]) == 0:
            basin_angles[b] = [placeholder_angles.get(b, (0.0, 0.0))]
            logging.warning(f"    Basin {BASIN_NAMES[b]} had no samples, "
                            f"using placeholder angle")

    # Convert to arrays for fast random indexing
    basin_angle_arrays = {}
    for b in range(n_basins):
        basin_angle_arrays[b] = np.array(basin_angles[b])

    # ------------------------------------------------------------------
    # Step 3: Simulate synthetic proteins
    # ------------------------------------------------------------------
    n_synth = len(real_lengths)
    logging.info(f"  Simulating {n_synth} synthetic proteins...")
    t0 = time.time()

    synth_bps_vals = []
    # Track which real protein each synthetic one corresponds to (for per-organism grouping)
    synth_organisms = list(real_organisms)  # same organism labels, length-matched

    # Precompute cumulative transition probabilities for fast sampling
    T_cumsum = np.cumsum(T_matrix, axis=1)
    stat_cumsum = np.cumsum(stationary)

    for si, L in enumerate(real_lengths):
        # 1. Sample starting basin from stationary distribution
        u = rng.random()
        start_basin = int(np.searchsorted(stat_cumsum, u))
        start_basin = min(start_basin, 4)

        # 2. Generate basin sequence via Markov chain
        basins = np.empty(L, dtype=np.int32)
        basins[0] = start_basin
        uniforms = rng.random(L - 1)
        for k in range(1, L):
            prev = basins[k - 1]
            basins[k] = min(int(np.searchsorted(T_cumsum[prev], uniforms[k - 1])), 4)

        # 3. Sample (phi, psi) from each basin's empirical distribution
        phis = np.empty(L)
        psis = np.empty(L)
        for b in range(5):
            mask = basins == b
            count = int(mask.sum())
            if count == 0:
                continue
            angles = basin_angle_arrays[b]
            indices = rng.integers(0, len(angles), size=count)
            phis[mask] = angles[indices, 0]
            psis[mask] = angles[indices, 1]

        # 4. Compute BPS/L on synthetic sequence
        W_chain = W_interp(np.column_stack([psis, phis]))
        dW = np.abs(np.diff(W_chain))
        synth_bps = float(np.sum(dW)) / L
        synth_bps_vals.append(synth_bps)

        if (si + 1) % 500 == 0:
            elapsed = time.time() - t0
            logging.info(f"    Simulated {si+1}/{n_synth} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    logging.info(f"  Simulation complete ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # Step 4: Compare real vs synthetic
    # ------------------------------------------------------------------
    real_arr = np.array(real_bps_vals)
    synth_arr = np.array(synth_bps_vals)

    mean_real = float(np.mean(real_arr))
    std_real = float(np.std(real_arr, ddof=1))
    cv_real = std_real / mean_real * 100 if mean_real > 0 else 0.0

    mean_synth = float(np.mean(synth_arr))
    std_synth = float(np.std(synth_arr, ddof=1))
    cv_synth = std_synth / mean_synth * 100 if mean_synth > 0 else 0.0

    # KS test
    ks_stat, ks_pvalue = ks_2samp(real_arr, synth_arr)

    logging.info(f"  Real:      mean={mean_real:.4f}, std={std_real:.4f}, CV={cv_real:.1f}%")
    logging.info(f"  Synthetic: mean={mean_synth:.4f}, std={std_synth:.4f}, CV={cv_synth:.1f}%")
    logging.info(f"  KS test:   D={ks_stat:.4f}, p={ks_pvalue:.2e}")

    # Per-organism comparison
    org_real = defaultdict(list)
    org_synth = defaultdict(list)
    for i in range(len(real_bps_vals)):
        org = real_organisms[i]
        org_real[org].append(real_bps_vals[i])
    for i in range(len(synth_bps_vals)):
        org = synth_organisms[i]
        org_synth[org].append(synth_bps_vals[i])

    org_breakdown = {}
    real_org_means = []
    synth_org_means = []
    for org in sorted(org_real.keys()):
        r_vals = org_real[org]
        s_vals = org_synth.get(org, [])
        r_mean = float(np.mean(r_vals))
        s_mean = float(np.mean(s_vals)) if s_vals else None
        real_org_means.append(r_mean)
        if s_mean is not None:
            synth_org_means.append(s_mean)
        org_breakdown[org] = {
            'n': len(r_vals),
            'mean_real': r_mean,
            'mean_synth': s_mean,
        }

    # Cross-organism CVs
    if len(real_org_means) >= 2:
        cross_cv_real = float(np.std(real_org_means, ddof=1) /
                              np.mean(real_org_means) * 100)
    else:
        cross_cv_real = None

    if len(synth_org_means) >= 2:
        cross_cv_synth = float(np.std(synth_org_means, ddof=1) /
                               np.mean(synth_org_means) * 100)
    else:
        cross_cv_synth = None

    logging.info(f"  Cross-organism CV: real={cross_cv_real:.1f}%, "
                 f"synth={cross_cv_synth:.1f}%"
                 if cross_cv_real is not None and cross_cv_synth is not None
                 else "  Cross-organism CV: insufficient data")

    # ------------------------------------------------------------------
    # Step 5: Verdict
    # ------------------------------------------------------------------
    mean_match = abs(mean_real - mean_synth) / mean_real * 100 < 5 if mean_real > 0 else False
    cv_match = abs(cv_real - cv_synth) < 2.0  # within 2 percentage points

    if mean_match and cv_match:
        verdict = 'PASS — Markov model explains mean AND tightness'
    elif mean_match and not cv_match:
        verdict = 'PARTIAL — Markov explains mean but NOT tightness'
    else:
        verdict = 'FAIL — Markov model insufficient'

    logging.info(f"  Verdict: {verdict}")

    return {
        'status': verdict,
        'n_proteins': n_processed,
        'n_synthetic': n_synth,
        'mean_real': mean_real,
        'std_real': std_real,
        'cv_real': cv_real,
        'mean_synth': mean_synth,
        'std_synth': std_synth,
        'cv_synth': cv_synth,
        'ks_stat': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'cross_cv_real': cross_cv_real,
        'cross_cv_synth': cross_cv_synth,
        'per_organism': org_breakdown,
        'transition_matrix': T_matrix.tolist(),
        'stationary': stationary.tolist(),
    }


# ============================================================
# REPORT WRITER
# ============================================================

def write_report(results, db_stats, outpath="validation_report.md"):
    """Generate markdown validation report."""
    lines = []

    def add(text=""):
        lines.append(text)

    add("# BPS Validation Controls — Report")
    add()
    add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"**Database:** {db_stats.get('total_proteins', '?')} total proteins, "
        f"{db_stats.get('hq_proteins', '?')} high-quality (pLDDT >= {HQ_THRESHOLD}), "
        f"{db_stats.get('n_organisms', '?')} organisms")
    add(f"**Sample size:** {SAMPLE_N} target, "
        f"{results.get('n_parsed', '?')} successfully parsed from CIF")
    add(f"**Shuffle repetitions:** {SHUFFLE_REPS}")
    add()

    # ---- Executive Summary ----
    add("## Executive Summary")
    add()
    add("| # | Test | Verdict |")
    add("|---|------|---------|")

    test_names = [
        ('shuffled', '1. Shuffled Control (Existential)'),
        ('robustness', '2. W-Construction Robustness'),
        ('fold_class', '3. Fold-Class Conditioning'),
        ('gly_pro', '4. Gly/Pro Composition'),
        ('transition', '5. Basin Transition Matrix'),
        ('markov_sim', '6. Markov Null Model Simulation'),
    ]
    for key, name in test_names:
        r = results.get(key, {})
        status = r.get('status', 'NOT RUN')
        add(f"| {name} | **{status}** |")
    add()

    # ---- Test 1 ----
    add("---")
    add()
    add("## Test 1: Shuffled Control (Existential)")
    add()
    r1 = results.get('shuffled', {})
    if r1.get('status') == 'SKIPPED':
        add(f"*Skipped: {r1.get('reason', 'unknown')}*")
    else:
        add("**Purpose:** Determine whether BPS/L ≈ 0.20 is a property of sequential "
            "backbone organization or merely a statistical artifact of basin occupancy "
            "frequencies.")
        add()
        add("**Method:** For each sampled protein, shuffle the (φ,ψ) pairs randomly "
            f"({SHUFFLE_REPS}× per protein), preserving basin occupancy but destroying "
            "sequential order. Compare real vs shuffled BPS/L.")
        add()
        add(f"| Metric | Value |")
        add(f"|--------|-------|")
        add(f"| Proteins tested | {r1.get('n_proteins', '?')} |")
        add(f"| Mean real BPS/L | {r1.get('mean_real', 0):.4f} ± {r1.get('std_real', 0):.4f} |")
        add(f"| Mean shuffled BPS/L | {r1.get('mean_shuffled', 0):.4f} ± {r1.get('std_shuffled', 0):.4f} |")
        add(f"| Cohen's d | {r1.get('cohens_d', 0):.3f} |")
        pv = r1.get('p_value')
        pv_str = f"{pv:.2e}" if pv is not None else "N/A"
        add(f"| Wilcoxon p-value | {pv_str} |")
        add(f"| **Verdict** | **{r1.get('status', '?')}** |")
        add()

        # Per-organism breakdown
        org_data = r1.get('per_organism', {})
        if org_data:
            add("### Per-Organism Breakdown")
            add()
            add("| Organism | N | Mean Real | Mean Shuffled | Δ |")
            add("|----------|---|-----------|---------------|---|")
            for org in sorted(org_data.keys()):
                od = org_data[org]
                add(f"| {org} | {od['n']} | {od['mean_real']:.4f} | "
                    f"{od['mean_shuffled']:.4f} | {od['delta']:+.4f} |")
            add()

        add("### Interpretation")
        add()
        add("- **If shuffled ≈ real:** BPS/L is trivial — it reflects only basin "
            "occupancy statistics, not sequential backbone structure. The paper's "
            "central claim would need fundamental rethinking.")
        add("- **If shuffled ≠ real:** BPS/L encodes genuine sequential structure "
            "in the protein backbone. The paper's central claim is validated.")
    add()

    # ---- Test 2 ----
    add("---")
    add()
    add("## Test 2: W-Construction Robustness")
    add()
    r2 = results.get('robustness', {})
    if r2.get('status') == 'SKIPPED':
        add(f"*Skipped: {r2.get('reason', 'unknown')}*")
    else:
        add("**Purpose:** Show that BPS/L universality is robust to superpotential "
            "construction choices (kernel bandwidth, grid resolution, density floor).")
        add()
        add(f"**Proteins tested:** {r2.get('n_proteins', '?')}")
        add()
        table = r2.get('table', [])
        if table:
            add("| W variant | Mean BPS/L | CV (%) | Δ from baseline |")
            add("|-----------|-----------|--------|-----------------|")
            for row in table:
                mean_s = f"{row['mean_bps']:.4f}" if row['mean_bps'] is not None else "N/A"
                cv_s = f"{row['cv_pct']:.1f}" if row['cv_pct'] is not None else "N/A"
                delta_s = f"{row['delta']:+.4f}" if row['delta'] is not None else "—"
                add(f"| {row['name']} | {mean_s} | {cv_s} | {delta_s} |")
            add()

        max_cv = r2.get('max_cv')
        add(f"**Max CV across all variants:** {max_cv:.1f}%" if max_cv is not None
            else "**Max CV:** N/A")
        add()
        add("**Key result:** If CV stays ≤ 3–4% across all variants (even when the "
            "absolute mean shifts), the universality of BPS/L is robust to W construction.")
        add()
        add(f"**Verdict:** **{r2.get('status', '?')}**")
    add()

    # ---- Test 3 ----
    add("---")
    add()
    add("## Test 3: Fold-Class-Conditioned Universality")
    add()
    r3 = results.get('fold_class', {})
    if r3.get('status') == 'SKIPPED':
        add(f"*Skipped: {r3.get('reason', 'unknown')}*")
    else:
        add("**Purpose:** Determine whether BPS/L ≈ 0.20 holds *within* the dominant "
            "fold class (mixed α/β) across organisms, or only after mixture-averaging.")
        add()
        add("**Fold class definitions:**")
        add("- Mixed α/β: pct_helix > 15 AND pct_sheet > 15")
        add("- Alpha-rich: pct_helix > 40 AND pct_sheet ≤ 15")
        add("- Beta-rich: pct_sheet > 25 AND pct_helix ≤ 15")
        add("- Other: everything else")
        add()

        table = r3.get('table', [])
        if table:
            add("| Fold class | N proteins | N organisms | Mean BPS/L | Cross-org CV (%) |")
            add("|------------|-----------|-------------|-----------|-----------------|")
            for row in table:
                mean_s = f"{row['mean_bps']:.4f}" if row['mean_bps'] is not None else "N/A"
                cv_s = f"{row['cv_pct']:.1f}" if row['cv_pct'] is not None else "N/A"
                add(f"| {row['fold_class']} | {row['n_proteins']} | "
                    f"{row['n_organisms']} | {mean_s} | {cv_s} |")
            add()

        add("**Key result:** If Mixed α/β CV ≤ 3% across organisms → universality is "
            "*geometric* (holds within fold class). If CV blows up → universality is "
            "*ecological* (fold class mixture effect).")
        add()
        add(f"**Verdict:** **{r3.get('status', '?')}**")
    add()

    # ---- Test 4 ----
    add("---")
    add()
    add("## Test 4: Gly/Pro Composition Effects")
    add()
    r4 = results.get('gly_pro', {})
    if r4.get('status') == 'SKIPPED':
        add(f"*Skipped: {r4.get('reason', 'unknown')}*")
    else:
        add("**Purpose:** Test whether proteins with extreme glycine or proline content "
            "deviate from BPS/L ≈ 0.20.")
        add()
        add(f"**Proteins tested:** {r4.get('n_proteins', '?')}")
        add()

        gb = r4.get('gly_boundaries', {})
        pb = r4.get('pro_boundaries', {})
        add(f"**Glycine quartile boundaries:** {gb.get('q25', '?'):.1f}%, "
            f"{gb.get('q50', '?'):.1f}%, {gb.get('q75', '?'):.1f}%")
        add(f"**Proline quartile boundaries:** {pb.get('q25', '?'):.1f}%, "
            f"{pb.get('q50', '?'):.1f}%, {pb.get('q75', '?'):.1f}%")
        add()

        gly_table = r4.get('gly_table', [])
        if gly_table:
            add("### %Glycine Bins")
            add()
            add("| Quartile | N | Mean BPS/L | Std |")
            add("|----------|---|-----------|-----|")
            for row in gly_table:
                add(f"| {row['bin']} | {row['n']} | {row['mean_bps']:.4f} | "
                    f"{row['std_bps']:.4f} |")
            add()

        pro_table = r4.get('pro_table', [])
        if pro_table:
            add("### %Proline Bins")
            add()
            add("| Quartile | N | Mean BPS/L | Std |")
            add("|----------|---|-----------|-----|")
            for row in pro_table:
                add(f"| {row['bin']} | {row['n']} | {row['mean_bps']:.4f} | "
                    f"{row['std_bps']:.4f} |")
            add()

        add(f"**Glycine Q1–Q4 spread:** {r4.get('gly_spread', 0):.4f}")
        add(f"**Proline Q1–Q4 spread:** {r4.get('pro_spread', 0):.4f}")
        add()
        add(f"**Verdict:** **{r4.get('status', '?')}**")
    add()

    # ---- Test 5 ----
    add("---")
    add()
    add("## Test 5: Basin Transition Matrix (Theoretical Derivation)")
    add()
    r5 = results.get('transition', {})
    if r5.get('status') == 'SKIPPED':
        add(f"*Skipped: {r5.get('reason', 'unknown')}*")
    else:
        add("**Purpose:** Test whether BPS/L ≈ 0.20 can be derived from first principles "
            "using Ramachandran basin transition probabilities.")
        add()
        add(f"**Proteins analyzed:** {r5.get('n_proteins', '?')}")
        add()

        add("### Ramachandran Basin Definitions")
        add()
        add("| Basin | φ range | ψ range |")
        add("|-------|---------|---------|")
        add("| α-helix | [-100, -30] | [-67, -7] |")
        add("| β-sheet | [-170, -70] | >90 or <-120 |")
        add("| ppII | [-100, -40] | [110, 180] |")
        add("| αL | [30, 90] | [0, 60] |")
        add("| Other | — | — |")
        add()

        # Basin occupancy
        occ = r5.get('basin_occupancy', {})
        if occ:
            add("### Basin Occupancy")
            add()
            add("| Basin | Count | % | Mean W |")
            add("|-------|-------|---|--------|")
            for bname in BASIN_NAMES:
                if bname in occ:
                    bd = occ[bname]
                    add(f"| {bname} | {bd['count']:,} | {bd['pct']:.1f}% | "
                        f"{bd['mean_W']:.4f} |")
            add()

        # Transition matrix
        T = r5.get('transition_matrix')
        if T is not None:
            add("### Transition Matrix (% of all transitions)")
            add()
            header = "| From \\ To | " + " | ".join(BASIN_NAMES) + " |"
            sep = "|-----------|" + "|".join(["------"] * 5) + "|"
            add(header)
            add(sep)
            for i, bname in enumerate(BASIN_NAMES):
                row_vals = " | ".join(f"{T[i][j]:.2f}" for j in range(5))
                add(f"| {bname} | {row_vals} |")
            add()

        # Prediction
        add("### Prediction vs Observation")
        add()
        add(f"| Metric | Value |")
        add(f"|--------|-------|")
        pred = r5.get('predicted_bps')
        obs = r5.get('observed_bps')
        err = r5.get('pct_error')
        add(f"| Predicted BPS/L (Σ T[i→j] × |W̄_i - W̄_j|) | "
            f"{pred:.4f}" if pred is not None else
            f"| Predicted BPS/L | N/A")
        add(f"| Observed BPS/L (sample mean) | "
            f"{obs:.4f}" if obs is not None else
            f"| Observed BPS/L | N/A")
        add(f"| Percentage error | "
            f"{err:.1f}%" if err is not None else
            f"| Percentage error | N/A")
        add()
        add("**Key result:** If predicted ≈ observed within ~5%, the BPS/L constant "
            "can be derived from basin transition probabilities — a first-principles "
            "derivation of the topological invariant.")
        add()
        add(f"**Verdict:** **{r5.get('status', '?')}**")
    add()

    # ---- Test 6 ----
    add("---")
    add()
    add("## Test 6: Markov Null Model Simulation")
    add()
    r6 = results.get('markov_sim', {})
    if r6.get('status') == 'SKIPPED':
        add(f"*Skipped: {r6.get('reason', 'unknown')}*")
    else:
        add("**Purpose:** Test whether a first-order Markov chain over Ramachandran basins "
            "reproduces not only the mean BPS/L but also the tightness (CV ≈ 2%) "
            "observed across real proteins.")
        add()
        add("**Method:** Build a basin transition matrix and per-basin empirical angle "
            "distributions from the sample. For each real protein length, generate a "
            "synthetic protein by Markov-sampling basin sequences then drawing random "
            "(φ,ψ) pairs from each basin's distribution. Compute BPS/L on each synthetic "
            "sequence and compare distributions.")
        add()
        add(f"**Proteins:** {r6.get('n_proteins', '?')} real, "
            f"{r6.get('n_synthetic', '?')} synthetic")
        add()

        add("| Metric | Real | Markov Synthetic |")
        add("|--------|------|------------------|")
        add(f"| Mean BPS/L | {r6.get('mean_real', 0):.4f} | {r6.get('mean_synth', 0):.4f} |")
        add(f"| Std BPS/L | {r6.get('std_real', 0):.4f} | {r6.get('std_synth', 0):.4f} |")
        add(f"| CV (%) | {r6.get('cv_real', 0):.1f} | {r6.get('cv_synth', 0):.1f} |")
        cross_r = r6.get('cross_cv_real')
        cross_s = r6.get('cross_cv_synth')
        cross_r_s = f"{cross_r:.1f}" if cross_r is not None else "N/A"
        cross_s_s = f"{cross_s:.1f}" if cross_s is not None else "N/A"
        add(f"| Cross-organism CV (%) | {cross_r_s} | {cross_s_s} |")
        add()

        ks_p = r6.get('ks_pvalue')
        ks_d = r6.get('ks_stat')
        ks_p_str = f"{ks_p:.2e}" if ks_p is not None else "N/A"
        ks_d_str = f"{ks_d:.4f}" if ks_d is not None else "N/A"
        add(f"**KS test:** D = {ks_d_str}, p-value = {ks_p_str}")
        add()

        # Per-organism breakdown
        org_data = r6.get('per_organism', {})
        if org_data:
            add("### Per-Organism Breakdown")
            add()
            add("| Organism | N | Mean Real | Mean Synthetic | Δ |")
            add("|----------|---|-----------|----------------|---|")
            for org in sorted(org_data.keys()):
                od = org_data[org]
                s_mean = od.get('mean_synth')
                if s_mean is not None:
                    delta = od['mean_real'] - s_mean
                    add(f"| {org} | {od['n']} | {od['mean_real']:.4f} | "
                        f"{s_mean:.4f} | {delta:+.4f} |")
                else:
                    add(f"| {org} | {od['n']} | {od['mean_real']:.4f} | N/A | — |")
            add()

        add("### Interpretation")
        add()
        add("- **If synthetic mean ≈ real mean AND synthetic CV ≈ real CV →** "
            "First-order Markov explains everything. BPS/L universality is a "
            "consequence of conserved secondary-structure transition frequencies.")
        add("- **If synthetic mean ≈ real mean BUT synthetic CV > real CV →** "
            "Markov gets the mean but not the tightness. Higher-order correlations "
            "(helix run lengths, sheet run lengths, specific loop structures) constrain "
            "the variance beyond what first-order statistics predict.")
        add("- **If synthetic mean ≠ real mean →** First-order Markov is insufficient. "
            "Higher-order sequential structure contributes to the mean itself.")
        add()
        add(f"**Verdict:** **{r6.get('status', '?')}**")
    add()

    # ---- Footer ----
    add("---")
    add()
    add("*Report generated by `bps_validate_controls.py`*")

    report_text = "\n".join(lines) + "\n"
    outpath = Path(outpath)
    outpath.write_text(report_text, encoding='utf-8')
    logging.info(f"Report written to {outpath}")
    return report_text


# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()

    logging.info("=" * 60)
    logging.info("BPS VALIDATION CONTROLS")
    logging.info("=" * 60)

    # Find database
    db_path = find_database()
    if db_path is None:
        logging.error("Database not found. Checked paths:")
        logging.error(f"  - {DB_PATH}")
        logging.error("  - bps_output/bps_results.db")
        logging.error("  - alphafold_bps_results/bps_database.db")
        logging.error("Run bps_process.py first to create the database.")
        sys.exit(1)

    logging.info(f"Database: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Database statistics
    db_stats = get_db_stats(conn)
    logging.info(f"  Total proteins: {db_stats['total_proteins']}")
    logging.info(f"  HQ proteins (pLDDT >= {HQ_THRESHOLD}): {db_stats['hq_proteins']}")
    logging.info(f"  Organisms: {db_stats['n_organisms']}")

    if db_stats['hq_proteins'] < 50:
        logging.error("Not enough high-quality proteins in database for validation.")
        conn.close()
        sys.exit(1)

    # Determine phi sign convention
    logging.info("")
    logging.info("Determining phi sign convention...")
    phi_sign = determine_phi_sign()
    logging.info(f"PHI_SIGN = {phi_sign}")

    # Build baseline superpotential
    logging.info("")
    logging.info("Building baseline superpotential...")
    W_baseline = build_superpotential()
    logging.info("  Done.")

    # Sample proteins for CIF-based tests
    logging.info("")
    logging.info("Sampling proteins for CIF-based tests...")
    sample = sample_proteins(conn, SAMPLE_N)
    if not sample:
        logging.error("No proteins could be sampled.")
        conn.close()
        sys.exit(1)

    # Parse CIF files (once, reused by Tests 1, 2, 4, 5)
    logging.info("")
    logging.info("Parsing CIF files for sampled proteins...")
    n_parsed = prepare_sample_data(sample, phi_sign)
    logging.info(f"  Successfully parsed: {n_parsed}/{len(sample)}")

    # Run tests
    results = {}
    results['n_parsed'] = n_parsed

    logging.info("")
    results['shuffled'] = test_shuffled_control(sample, W_baseline)

    logging.info("")
    results['robustness'] = test_w_robustness(sample, W_baseline)

    logging.info("")
    results['fold_class'] = test_fold_class_conditioning(conn)

    logging.info("")
    results['gly_pro'] = test_gly_pro(sample, W_baseline)

    logging.info("")
    results['transition'] = test_transition_matrix(sample, W_baseline)

    logging.info("")
    results['markov_sim'] = test_markov_simulation(sample, W_baseline)

    # Write report
    logging.info("")
    logging.info("Writing validation report...")
    write_report(results, db_stats, outpath="validation_report.md")

    conn.close()

    logging.info("")
    logging.info("=" * 60)
    logging.info("VALIDATION COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
