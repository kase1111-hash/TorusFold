#!/usr/bin/env python3
"""
MARKOV PSEUDO-PROTEOME TEST (v4 - self-contained)
===================================================
The critical control: does the tight BPS/L conservation reflect biology
or just the statistics of sampling from the Ramachandran distribution?

NULL MODEL: For each protein, sample (phi, psi) IID from P(phi, psi).
This destroys ALL sequential correlations while preserving the
single-residue Ramachandran distribution.

EXPECTED RESULTS:
  - IID null BPS/L ~ 0.35-0.40 (higher than real ~0.17-0.21)
  - Gap ~ +70-80% (null is higher than real)
  - Null CV across organisms ~ tight (any large IID sample converges)

This version builds its own W grid from the von Mises mixture parameters
to avoid any dependency on bps.superpotential return format.

Usage:
  python markov_pseudoproteome_test.py
  python markov_pseudoproteome_test.py --n-pseudo 50
  python markov_pseudoproteome_test.py --organisms ecoli human yeast
  python markov_pseudoproteome_test.py --quick   # 10 pseudo, 3 organisms
"""

import os
import sys
import csv
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
OUTPUT_DIR = Path("markov_test_results")
N_PSEUDO_DEFAULT = 100
MAX_PROTEINS_PER_PSEUDO = 500

# Von Mises mixture components (Table 1 of paper)
# (weight, mu_phi_deg, mu_psi_deg, kappa_phi, kappa_psi, rho)
VON_MISES_COMPONENTS = [
    (0.35, -63,  -43,  12.0, 10.0,  2.0),   # alpha-helix core
    (0.05, -60,  -27,   8.0,  6.0,  1.0),   # alpha-helix shoulder
    (0.25, -120,  135,  4.0,  3.5, -1.5),   # beta-sheet core
    (0.05, -140,  155,  5.0,  4.0, -1.0),   # beta-sheet shoulder
    (0.12, -75,   150,  8.0,  5.0,  0.5),   # PPII
    (0.05, -95,   150,  3.0,  4.0,  0.0),   # PPII shoulder
    (0.03,  57,    40,  6.0,  6.0,  1.5),   # left-handed alpha
    (0.03,  60,  -130,  5.0,  4.0,  0.0),   # gamma-turn
    (0.01,  75,   -65,  5.0,  5.0,  0.0),   # delta
    (0.06,   0,     0,  0.01, 0.01, 0.0),   # uniform background
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
            logging.FileHandler(OUTPUT_DIR / "markov_test.log", mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ============================================================
# BUILD SUPERPOTENTIAL (self-contained)
# ============================================================

def build_W_grid_and_interpolator(grid_size=360, sigma=1.5):
    """Build W = -sqrt(P) grid and interpolator from von Mises mixture.

    Grid covers [-pi, pi) in both phi and psi (= [-180, 180) degrees).
    Interpolator expects np.column_stack([psi_rad, phi_rad]) in RADIANS.

    Returns: (W_grid, W_interp, phi_grid_rad, psi_grid_rad)
        W_grid: ndarray (360, 360), indexed as W_grid[phi_idx, psi_idx]
        W_interp: RegularGridInterpolator, takes (psi_rad, phi_rad)
        phi_grid_rad, psi_grid_rad: 1D arrays of grid points in radians
    """
    angles = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    PHI, PSI = np.meshgrid(angles, angles, indexing="ij")  # PHI[i,j], PSI[i,j]

    # Compute density
    p = np.zeros_like(PHI)
    for w, mu_phi_deg, mu_psi_deg, kp, ks, rho in VON_MISES_COMPONENTS:
        mu_phi = np.radians(mu_phi_deg)
        mu_psi = np.radians(mu_psi_deg)
        exponent = (kp * np.cos(PHI - mu_phi) +
                    ks * np.cos(PSI - mu_psi) +
                    rho * np.sin(PHI - mu_phi) * np.sin(PSI - mu_psi))
        p += w * np.exp(exponent)

    # Normalize
    dphi = 2 * np.pi / grid_size
    p /= (p.sum() * dphi * dphi)

    # Floor at 1e-6 * max
    p = np.maximum(p, 1e-6 * p.max())

    # W = -sqrt(p), then Gaussian smooth with torus wrapping
    W = -np.sqrt(p)
    W = gaussian_filter(W, sigma=sigma, mode="wrap")

    # Build interpolator: axes are (psi, phi) to match bps_process.py convention
    # W_grid is indexed [phi, psi], so transpose for RegularGridInterpolator
    # which expects grid values as grid[psi_idx, phi_idx] when axes=(psi, phi)
    W_interp = RegularGridInterpolator(
        (angles, angles),  # (psi_grid, phi_grid) - both same grid
        W.T,  # transpose so first axis = psi, second = phi
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    return W, W_interp, angles, angles


# ============================================================
# IID SAMPLING FROM P(phi, psi)
# ============================================================

def build_sampling_table(W_grid):
    """Build inverse-CDF table from W grid for fast IID sampling.

    Since W = -sqrt(P) and was smoothed, we recover P ~ W^2.
    Flatten the grid into a 1D CDF.

    The grid is in [-180, 180) degree space (or equivalently [-pi, pi) radians).
    Grid indices map to angles via: angle_deg = index * (360/N) - 180

    Returns: (cdf, n_grid)
    """
    n_grid = W_grid.shape[0]  # 360

    # P proportional to W^2
    P = W_grid ** 2
    P_flat = P.ravel()
    total = P_flat.sum()
    if total <= 0:
        logging.error("P sums to zero — W grid is broken.")
        sys.exit(1)
    P_flat /= total

    cdf = np.cumsum(P_flat)
    cdf[-1] = 1.0

    # Diagnostics
    logging.info(f"  Sampling table: {n_grid}x{n_grid} = {n_grid*n_grid} bins")
    logging.info(f"  W range: [{W_grid.min():.4f}, {W_grid.max():.4f}]")
    logging.info(f"  P max: {P_flat.max()*100:.3f}%")
    logging.info(f"  Non-zero bins: {np.sum(P_flat > 0)} / {len(P_flat)}")

    top_idx = np.argsort(P_flat)[-5:][::-1]
    for idx in top_idx:
        phi_i = idx // n_grid
        psi_i = idx % n_grid
        phi_deg = phi_i * (360.0 / n_grid) - 180.0
        psi_deg = psi_i * (360.0 / n_grid) - 180.0
        logging.info(f"    Top bin: phi={phi_deg:.0f}, psi={psi_deg:.0f}, P={P_flat[idx]*100:.3f}%")

    return cdf, n_grid


def sample_iid_chain(length, cdf, n_grid, rng):
    """Sample (phi, psi) IID from P(phi, psi) using inverse-CDF.

    Returns phi_rad, psi_rad arrays in [-pi, pi) RADIANS.
    (Radians because the interpolator uses radians.)
    """
    u = rng.random(length)
    indices = np.searchsorted(cdf, u)
    indices = np.clip(indices, 0, n_grid * n_grid - 1)

    # Unflatten: flat_index = phi_idx * n_grid + psi_idx
    phi_idx = indices // n_grid
    psi_idx = indices % n_grid

    # Convert grid index to radians with jitter
    # Grid spans [-pi, pi) with spacing 2*pi/n_grid
    step = 2 * np.pi / n_grid
    phi_rad = -np.pi + phi_idx * step + rng.random(length) * step
    psi_rad = -np.pi + psi_idx * step + rng.random(length) * step

    return phi_rad, psi_rad


def compute_bps_l(phi_rad, psi_rad, W_interp):
    """Compute BPS/L from (phi, psi) in radians.

    W_interp expects np.column_stack([psi, phi]) in radians.
    BPS/L = sum(|dW|) / L
    """
    W_vals = W_interp(np.column_stack([psi_rad, phi_rad]))
    dW = np.abs(np.diff(W_vals))
    L = len(phi_rad)
    return float(np.sum(dW)) / L


# ============================================================
# SANITY CHECK
# ============================================================

def run_sanity_check(W_interp, cdf, n_grid, rng):
    """Validate the full pipeline before burning compute."""
    logging.info("")
    logging.info("  ---- SANITY CHECK ----")

    phi_rad, psi_rad = sample_iid_chain(200, cdf, n_grid, rng)
    phi_deg = np.degrees(phi_rad)
    psi_deg = np.degrees(psi_rad)

    logging.info(f"  Chain length: 200")
    logging.info(f"  phi: min={phi_deg.min():.1f}  max={phi_deg.max():.1f}  "
                 f"std={phi_deg.std():.1f}  mean={phi_deg.mean():.1f}")
    logging.info(f"  psi: min={psi_deg.min():.1f}  max={psi_deg.max():.1f}  "
                 f"std={psi_deg.std():.1f}  mean={psi_deg.mean():.1f}")

    logging.info(f"  First 10 (phi, psi) in degrees:")
    for i in range(10):
        logging.info(f"    [{i:2d}] phi={phi_deg[i]:7.1f}  psi={psi_deg[i]:7.1f}")

    # W values
    W_vals = W_interp(np.column_stack([psi_rad, phi_rad]))
    dW = np.abs(np.diff(W_vals))

    logging.info(f"  W: min={W_vals.min():.4f}  max={W_vals.max():.4f}  "
                 f"std={W_vals.std():.4f}  mean={W_vals.mean():.4f}")
    logging.info(f"  |dW|: min={dW.min():.6f}  max={dW.max():.4f}  mean={dW.mean():.4f}")

    logging.info(f"  First 10 W values:")
    for i in range(10):
        logging.info(f"    [{i:2d}] W={W_vals[i]:.6f}")
    logging.info(f"  First 10 |dW|:")
    for i in range(min(10, len(dW))):
        logging.info(f"    [{i:2d}] |dW|={dW[i]:.6f}")

    bps_l = float(np.sum(dW)) / 200
    logging.info(f"  BPS/L = {bps_l:.4f}")
    logging.info(f"  (Expected ~0.35-0.45 for IID; real proteins ~0.17-0.21)")

    # Validate
    errors = []
    if bps_l < 0.05:
        errors.append(f"BPS/L={bps_l:.4f} near zero — sampling or lookup broken")
    if bps_l > 2.0:
        errors.append(f"BPS/L={bps_l:.4f} implausibly high")
    if phi_deg.std() < 5:
        errors.append(f"phi std={phi_deg.std():.1f} — too clustered")
    if psi_deg.std() < 5:
        errors.append(f"psi std={psi_deg.std():.1f} — too clustered")
    if W_vals.std() < 0.01:
        errors.append(f"W std={W_vals.std():.6f} — barely varies")
    if dW.mean() < 0.001:
        errors.append(f"|dW| mean={dW.mean():.6f} — no transitions")

    # Short chain test
    phi_s, psi_s = sample_iid_chain(10, cdf, n_grid, rng)
    bps_s = compute_bps_l(phi_s, psi_s, W_interp)
    logging.info(f"  Short chain (L=10) BPS/L = {bps_s:.4f}")

    if errors:
        for e in errors:
            logging.error(f"  FAIL: {e}")
        logging.info("  ---- SANITY CHECK FAILED ----")
        return False

    logging.info("  PASS: IID sampling produces valid BPS/L")
    logging.info("  ---- END SANITY CHECK ----")
    logging.info("")
    return True


# ============================================================
# DATABASE ACCESS
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


def get_protein_lengths(conn, organism):
    """Get chain lengths. Auto-detect column name."""
    cur = conn.execute("PRAGMA table_info(proteins)")
    columns = [row[1] for row in cur.fetchall()]

    if 'L' in columns:
        col = 'L'
    elif 'n_residues' in columns:
        col = 'n_residues'
    else:
        logging.error(f"No length column. Columns: {columns}")
        sys.exit(1)

    cur = conn.execute(
        f"SELECT {col} FROM proteins WHERE organism = ? AND bps_norm IS NOT NULL",
        (organism,)
    )
    lengths = np.array([row[0] for row in cur.fetchall()])
    if np.any(lengths < 1):
        lengths = lengths[lengths >= 1]
    return lengths


def get_real_bps_l(conn, organism):
    cur = conn.execute(
        "SELECT bps_norm FROM proteins WHERE organism = ? AND bps_norm IS NOT NULL",
        (organism,)
    )
    return np.array([row[0] for row in cur.fetchall()])


# ============================================================
# PER-ORGANISM TEST
# ============================================================

def run_organism_test(organism, conn, W_interp, cdf, n_grid, n_pseudo, rng):
    t0 = time.time()

    real_bps = get_real_bps_l(conn, organism)
    lengths = get_protein_lengths(conn, organism)

    if len(real_bps) < 10:
        logging.warning(f"  {organism}: only {len(real_bps)} proteins, skipping")
        return None

    real_mean = float(np.mean(real_bps))
    real_std = float(np.std(real_bps))
    real_cv = 100 * real_std / real_mean if real_mean > 0 else 0

    pseudo_means = []

    for _ in range(n_pseudo):
        if len(lengths) > MAX_PROTEINS_PER_PSEUDO:
            sample_lengths = rng.choice(lengths, size=MAX_PROTEINS_PER_PSEUDO, replace=False)
        else:
            sample_lengths = lengths

        bps_values = []
        for L in sample_lengths:
            L = int(L)
            if L < 5:
                continue
            phi_rad, psi_rad = sample_iid_chain(L, cdf, n_grid, rng)
            bps_l = compute_bps_l(phi_rad, psi_rad, W_interp)
            bps_values.append(bps_l)

        if bps_values:
            pseudo_means.append(float(np.mean(bps_values)))

    if not pseudo_means:
        logging.warning(f"  {organism}: no pseudo-proteome means")
        return None

    pseudo_arr = np.array(pseudo_means)
    null_mean = float(np.mean(pseudo_arr))
    null_std = float(np.std(pseudo_arr))
    null_cv = 100 * null_std / null_mean if null_mean > 0 else 0

    gap_pct = 100 * (null_mean - real_mean) / real_mean if real_mean > 0 else 0

    elapsed = time.time() - t0

    logging.info(
        f"  {organism:20s}  N={len(real_bps):>6}  "
        f"Real={real_mean:.4f}  Null={null_mean:.4f}  "
        f"Gap={gap_pct:+.1f}%  "
        f"RealCV={real_cv:.1f}%  NullCV={null_cv:.1f}%  "
        f"({elapsed:.0f}s)"
    )

    return {
        'organism': organism,
        'n_proteins': int(len(real_bps)),
        'real_mean': real_mean,
        'real_std': real_std,
        'real_cv': real_cv,
        'null_mean': null_mean,
        'null_std': null_std,
        'null_cv': null_cv,
        'gap_pct': gap_pct,
        'n_pseudo': n_pseudo,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Markov pseudo-proteome test v4")
    parser.add_argument("--n-pseudo", type=int, default=N_PSEUDO_DEFAULT)
    parser.add_argument("--organisms", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick: 10 pseudo-proteomes, 3 organisms")
    args = parser.parse_args()

    setup_logging()

    if args.quick:
        args.n_pseudo = 10
        if args.organisms is None:
            args.organisms = ["ecoli", "human", "yeast"]

    logging.info("=" * 70)
    logging.info("MARKOV PSEUDO-PROTEOME TEST v4 (self-contained IID null)")
    logging.info("=" * 70)
    logging.info(f"  Pseudo-proteomes per organism: {args.n_pseudo}")
    logging.info(f"  Database: {DB_PATH}")
    logging.info(f"  Output: {OUTPUT_DIR}")
    logging.info(f"  Seed: {args.seed}")
    logging.info("")

    if not DB_PATH.exists():
        logging.error(f"Database not found: {DB_PATH}")
        sys.exit(1)

    # Build superpotential from scratch (no external dependency)
    logging.info("Building superpotential from von Mises mixture...")
    W_grid, W_interp, phi_grid, psi_grid = build_W_grid_and_interpolator()
    logging.info(f"  W grid shape: {W_grid.shape}")
    logging.info(f"  W range: [{W_grid.min():.4f}, {W_grid.max():.4f}]")

    # Verify interpolator matches grid at a known point
    test_phi = phi_grid[100]  # some arbitrary grid point
    test_psi = psi_grid[200]
    W_direct = W_grid[100, 200]
    W_lookup = float(W_interp(np.array([[test_psi, test_phi]]))[0])
    logging.info(f"  Interpolator check: grid={W_direct:.6f}, interp={W_lookup:.6f}, "
                 f"diff={abs(W_direct - W_lookup):.2e}")
    if abs(W_direct - W_lookup) > 0.01:
        logging.error("  Interpolator does NOT match grid! Axis order may be wrong.")
        sys.exit(1)
    logging.info("  Interpolator matches grid. OK.")

    # Build sampling table
    cdf, n_grid = build_sampling_table(W_grid)

    # Sanity check
    check_rng = np.random.default_rng(99999)
    if not run_sanity_check(W_interp, cdf, n_grid, check_rng):
        logging.error("SANITY CHECK FAILED — aborting.")
        sys.exit(1)

    # Main RNG
    rng = np.random.default_rng(args.seed)

    # Database
    conn = sqlite3.connect(str(DB_PATH))
    organisms = get_organisms(conn, args.organisms)
    logging.info(f"  Organisms: {len(organisms)}")
    logging.info("")
    logging.info(f"  {'Organism':20s}  {'N':>6}  {'Real':>7}  {'Null':>7}  "
                 f"{'Gap':>6}  {'RealCV':>7}  {'NullCV':>7}")
    logging.info(f"  {'-'*20}  {'-'*6}  {'-'*7}  {'-'*7}  "
                 f"{'-'*6}  {'-'*7}  {'-'*7}")

    results = []
    for organism, n_proteins in organisms:
        result = run_organism_test(
            organism, conn, W_interp, cdf, n_grid, args.n_pseudo, rng
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
    null_means = np.array([r['null_mean'] for r in results])
    gaps = np.array([r['gap_pct'] for r in results])

    real_cv_of_means = 100 * np.std(real_means) / np.mean(real_means)
    null_cv_of_means = 100 * np.std(null_means) / np.mean(null_means)
    mean_gap = float(np.mean(gaps))

    logging.info("")
    logging.info("=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)
    logging.info("")
    logging.info(f"  Organisms tested: {len(results)}")
    logging.info(f"  Total real proteins: {sum(r['n_proteins'] for r in results):,}")
    logging.info("")
    logging.info(f"  REAL proteomes:")
    logging.info(f"    Mean of organism means: {np.mean(real_means):.4f}")
    logging.info(f"    SD of organism means:   {np.std(real_means):.4f}")
    logging.info(f"    CV of organism means:   {real_cv_of_means:.1f}%")
    logging.info(f"    Range: [{np.min(real_means):.4f}, {np.max(real_means):.4f}]")
    logging.info("")
    logging.info(f"  NULL (IID from P) proteomes:")
    logging.info(f"    Mean of organism means: {np.mean(null_means):.4f}")
    logging.info(f"    SD of organism means:   {np.std(null_means):.4f}")
    logging.info(f"    CV of organism means:   {null_cv_of_means:.1f}%")
    logging.info(f"    Range: [{np.min(null_means):.4f}, {np.max(null_means):.4f}]")
    logging.info("")
    logging.info(f"  GAP (Null - Real) / Real:")
    logging.info(f"    Mean gap:  {mean_gap:+.1f}%")
    logging.info(f"    Gap range: [{np.min(gaps):+.1f}%, {np.max(gaps):+.1f}%]")
    logging.info(f"    Gap SD:    {np.std(gaps):.1f}%")
    logging.info("")

    logging.info("  CONCLUSIONS:")
    logging.info("")

    if mean_gap > 10:
        logging.info(f"  1. MEAN GAP = {mean_gap:+.0f}%: IID null produces HIGHER BPS/L.")
        logging.info(f"     Real proteins are {abs(mean_gap):.0f}% smoother than random from P.")
        logging.info(f"     -> BIOLOGY: evolution minimizes topological energy beyond")
        logging.info(f"        what single-residue preferences dictate.")
    elif mean_gap > -10:
        logging.info(f"  1. MEAN GAP = {mean_gap:+.1f}%: Real and null are SIMILAR.")
        logging.info(f"     -> BPS/L may be primarily a statistical property of P.")
    else:
        logging.info(f"  1. MEAN GAP = {mean_gap:+.1f}%: Real is HIGHER than null.")
        logging.info(f"     -> Unexpected. Check for bugs.")

    logging.info("")
    if null_cv_of_means < real_cv_of_means:
        logging.info(f"  2. CV: Null ({null_cv_of_means:.1f}%) TIGHTER than Real ({real_cv_of_means:.1f}%)")
        logging.info(f"     -> Cross-organism tightness is partly STATISTICAL.")
    else:
        logging.info(f"  2. CV: Null ({null_cv_of_means:.1f}%) WIDER than Real ({real_cv_of_means:.1f}%)")
        logging.info(f"     -> Both value AND tightness reflect biological constraint.")

    logging.info("")
    logging.info(f"  3. CONSISTENCY: Gap = {mean_gap:+.1f}% +/- {np.std(gaps):.1f}% across {len(results)} organisms.")
    if np.std(gaps) < 10:
        logging.info(f"     -> Smoothness advantage is UNIVERSAL across life.")
    else:
        logging.info(f"     -> Some organism-level variation.")

    # ============================================================
    # SAVE
    # ============================================================

    csv_path = OUTPUT_DIR / "markov_pseudoproteome_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'organism', 'n_proteins', 'real_mean', 'real_std', 'real_cv',
            'null_mean', 'null_std', 'null_cv', 'gap_pct', 'n_pseudo'
        ])
        writer.writeheader()
        writer.writerows(results)

    summary_path = OUTPUT_DIR / "markov_test_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("MARKOV PSEUDO-PROTEOME TEST v4 - SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Organisms: {len(results)}\n")
        f.write(f"Pseudo-proteomes per organism: {args.n_pseudo}\n")
        f.write(f"Total real proteins: {sum(r['n_proteins'] for r in results):,}\n\n")
        f.write(f"Real mean of means:  {np.mean(real_means):.4f} (CV={real_cv_of_means:.1f}%)\n")
        f.write(f"Null mean of means:  {np.mean(null_means):.4f} (CV={null_cv_of_means:.1f}%)\n")
        f.write(f"Mean gap: {mean_gap:+.1f}%\n")
        f.write(f"Gap range: [{np.min(gaps):+.1f}%, {np.max(gaps):+.1f}%]\n\n")
        f.write(f"{'Organism':20s} {'N':>6} {'Real':>8} {'Null':>8} {'Gap':>7} {'RealCV':>7} {'NullCV':>7}\n")
        f.write("-" * 72 + "\n")
        for r in sorted(results, key=lambda x: x['gap_pct'], reverse=True):
            f.write(f"{r['organism']:20s} {r['n_proteins']:>6} "
                    f"{r['real_mean']:>8.4f} {r['null_mean']:>8.4f} "
                    f"{r['gap_pct']:>+6.1f}% {r['real_cv']:>6.1f}% {r['null_cv']:>6.1f}%\n")

    logging.info("")
    logging.info(f"  Results: {csv_path}")
    logging.info(f"  Summary: {summary_path}")
    logging.info("")
    logging.info("Done.")


if __name__ == "__main__":
    main()
