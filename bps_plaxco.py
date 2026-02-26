#!/usr/bin/env python3
"""
BPS PLAXCO VALIDATION — Standalone folding rate analysis
=========================================================
Downloads the 23 Plaxco proteins, computes BPS + CO, correlates with
experimental ln(kf), performs LOO cross-validation, and generates
publication-ready output.

Usage:
  python bps_plaxco.py                       # full analysis
  python bps_plaxco.py --download            # download proteins only
  python bps_plaxco.py --db path/to/db       # use existing DB for BPS values
"""

import argparse
import sqlite3
import math
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress

# ============================================================
# CONFIG
# ============================================================

CACHE_DIR = Path("./alphafold_cache")
DB_PATH = Path("./alphafold_bps_results/bps_database.db")
OUTPUT_DIR = Path("./alphafold_bps_results")
VALIDATION_DIR = CACHE_DIR / "_validation"

# Plaxco 23 validation set: (uniprot_id, pdb_id, name, ln_kf, CO, exp_L)
PLAXCO_PROTEINS = [
    ("P00648", "1SRL", "Sarcin",              -3.20, 0.080, 110),
    ("P56849", "1APS", "AcP",                 -1.88, 0.193, 98),
    ("P0A6X3", "1HKS", "Hpr",                 -2.42, 0.120, 85),
    ("P00698", "1HEL", "HEWL",                -1.18, 0.169, 129),
    ("P02489", "3HMZ", "alpha-A crystallin",   0.80, 0.116, 173),
    ("P00636", "2RN2", "Barnase",             -3.31, 0.131, 110),
    ("P00720", "2LZM", "T4 Lysozyme",         0.20, 0.130, 164),
    ("P23928", "1MJC", "alphaB crystallin",   -0.50, 0.126, 175),
    ("P0AEH5", "2CI2", "CI2",                 -4.90, 0.070, 64),
    ("P06654", "1UBQ", "Ubiquitin",           -3.48, 0.118, 76),
    ("P01112", "5P21", "p21 Ras",             -0.80, 0.119, 166),
    ("P62937", "2CYP", "CypA",               -2.80, 0.119, 165),
    ("P23229", "1TIT", "Titin I27",           -1.30, 0.151, 89),
    ("P0ABP8", "1DPS", "Dps",                  0.00, 0.080, 167),
    ("P04637", "2OCJ", "p53",                 -0.50, 0.110, 94),
    ("P05067", "1AAP", "APPI",                -5.40, 0.081, 56),
    ("P00974", "5PTI", "BPTI",               -5.80, 0.082, 58),
    ("P01133", "1EGF", "EGF",                 -5.00, 0.084, 53),
    ("P62993", "1SHG", "SH3",                 -1.20, 0.108, 57),
    ("P00178", "1YCC", "Cyt c",               -3.40, 0.088, 108),
    ("P0CY58", "1AON", "GroEL",                1.80, 0.057, 524),
    ("P00925", "3ENL", "Enolase",              0.80, 0.075, 432),
    ("P00517", "1CDK", "PKA",                  0.50, 0.088, 350),
]


# ============================================================
# DOWNLOAD PLAXCO PROTEINS
# ============================================================

def download_plaxco_proteins():
    """Download AlphaFold CIF files for all 23 Plaxco proteins."""
    import urllib.request
    import urllib.error

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = 0

    for uid, pdb, name, _, _, _ in PLAXCO_PROTEINS:
        # Check if already cached anywhere
        found = False
        for search_dir in [CACHE_DIR / "plaxco23", VALIDATION_DIR,
                           CACHE_DIR / "_selftest",
                           CACHE_DIR / "ecoli", CACHE_DIR / "human",
                           CACHE_DIR / "yeast", CACHE_DIR]:
            for v in (4, 3, 2, 1):
                p = search_dir / f"AF-{uid}-F1-model_v{v}.cif"
                if p.exists() and p.stat().st_size > 100:
                    found = True
                    break
            if found:
                break

        if found:
            skipped += 1
            continue

        # Download
        success = False
        for v in (4, 3, 2):
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{v}.cif"
            dest = VALIDATION_DIR / f"AF-{uid}-F1-model_v{v}.cif"
            try:
                urllib.request.urlretrieve(url, str(dest))
                if dest.stat().st_size > 100:
                    print(f"  Downloaded {uid} ({name}) v{v}")
                    downloaded += 1
                    success = True
                    break
                dest.unlink(missing_ok=True)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    continue
                dest.unlink(missing_ok=True)
            except Exception:
                dest.unlink(missing_ok=True)

        if not success:
            print(f"  FAILED {uid} ({name})")
            failed += 1

    print(f"\nPlaxco download: {downloaded} new, {skipped} cached, {failed} failed")
    return failed == 0


# ============================================================
# FIND CIF FOR A PROTEIN
# ============================================================

def find_cif(uid):
    """Find cached CIF for a UniProt ID, searching multiple locations."""
    for search_dir in [CACHE_DIR / "plaxco23", VALIDATION_DIR,
                       CACHE_DIR / "_selftest",
                       CACHE_DIR / "ecoli", CACHE_DIR / "human",
                       CACHE_DIR / "yeast", CACHE_DIR]:
        for v in (4, 3, 2, 1):
            p = search_dir / f"AF-{uid}-F1-model_v{v}.cif"
            if p.exists() and p.stat().st_size > 100:
                return p
    return None


# ============================================================
# COMPUTE BPS FOR PLAXCO SET
# ============================================================

def compute_plaxco_bps():
    """Compute BPS for all 23 Plaxco proteins. Returns list of result dicts."""
    # Import processing functions from bps_process
    from bps_process import (build_superpotential, process_one,
                             determine_phi_sign, PHI_SIGN)

    # Ensure phi sign is set
    if PHI_SIGN is None:
        determine_phi_sign()

    W = build_superpotential()
    results = []

    for uid, pdb, name, ln_kf, co_exp, exp_L in PLAXCO_PROTEINS:
        cif = find_cif(uid)
        if cif is None:
            print(f"  {uid}/{pdb} ({name}): SKIP — not cached")
            continue

        result, err = process_one(cif, W)
        if result is None:
            print(f"  {uid}/{pdb} ({name}): FAIL — {err}")
            continue

        result['uniprot_id'] = uid
        result['pdb_id'] = pdb
        result['name'] = name
        result['ln_kf'] = ln_kf
        result['co_exp'] = co_exp
        result['exp_L'] = exp_L
        results.append(result)

    return results


def load_plaxco_from_db(conn):
    """Load Plaxco protein results from existing database."""
    results = []
    for uid, pdb, name, ln_kf, co_exp, exp_L in PLAXCO_PROTEINS:
        row = conn.execute("""
            SELECT L, bps_energy, bps_norm, contact_order, plddt_mean,
                   pct_helix, pct_sheet, n_transitions
            FROM proteins WHERE uniprot_id = ?
        """, (uid,)).fetchone()
        if row:
            results.append({
                'uniprot_id': uid, 'pdb_id': pdb, 'name': name,
                'ln_kf': ln_kf, 'co_exp': co_exp, 'exp_L': exp_L,
                'L': row[0], 'bps_energy': row[1], 'bps_norm': row[2],
                'contact_order': row[3], 'plddt_mean': row[4],
                'pct_helix': row[5], 'pct_sheet': row[6],
                'n_transitions': row[7],
            })
    return results


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def run_correlations(results, out):
    """Single-variable and combined correlations with ln(kf)."""
    out.append(f"\n{'='*72}")
    out.append(f"  PLAXCO VALIDATION: FOLDING RATE CORRELATIONS")
    out.append(f"{'='*72}\n")
    out.append(f"  N = {len(results)} proteins\n")

    ln_kf = np.array([r['ln_kf'] for r in results])
    co_exp = np.array([r['co_exp'] for r in results])
    bps = np.array([r['bps_energy'] for r in results])
    bps_norm = np.array([r['bps_norm'] for r in results])
    L = np.array([r['L'] for r in results])
    co_comp = np.array([r.get('contact_order', 0) or 0 for r in results])

    # Single-variable correlations
    out.append("  Single-variable correlations with ln(kf):")
    h = f"    {'Predictor':<20} {'r_pearson':>10} {'p':>12} {'r_spearman':>12}"
    out.append(h)
    out.append("    " + "-" * (len(h) - 4))

    predictors = [
        ("CO (experimental)", co_exp),
        ("CO (computed)", co_comp),
        ("BPS (raw)", bps),
        ("BPS/res", bps_norm),
        ("L", L),
    ]

    for name, vals in predictors:
        if np.std(vals) < 1e-10:
            continue
        rp, pp = pearsonr(vals, ln_kf)
        rs, _ = spearmanr(vals, ln_kf)
        out.append(f"    {name:<20} {rp:>10.4f} {pp:>12.2e} {rs:>12.4f}")

    # Combined models
    out.append(f"\n  Combined regression models:")
    n = len(results)

    models = [
        ("CO only", np.column_stack([co_exp, np.ones(n)])),
        ("BPS only", np.column_stack([bps, np.ones(n)])),
        ("BPS/res only", np.column_stack([bps_norm, np.ones(n)])),
        ("CO + BPS", np.column_stack([co_exp, bps, np.ones(n)])),
        ("CO + BPS/res", np.column_stack([co_exp, bps_norm, np.ones(n)])),
        ("CO + BPS + L", np.column_stack([co_exp, bps, L, np.ones(n)])),
    ]

    h2 = f"    {'Model':<20} {'R-squared':>10} {'Adj R2':>10} {'RMSE':>8}"
    out.append(h2)
    out.append("    " + "-" * (len(h2) - 4))

    for name, X in models:
        beta, _, _, _ = np.linalg.lstsq(X, ln_kf, rcond=None)
        pred = X @ beta
        ss_res = np.sum((ln_kf - pred)**2)
        ss_tot = np.sum((ln_kf - ln_kf.mean())**2)
        r2 = 1 - ss_res / ss_tot
        p = X.shape[1] - 1  # number of predictors (excluding intercept)
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        rmse = np.sqrt(ss_res / n)
        out.append(f"    {name:<20} {r2:>10.4f} {r2_adj:>10.4f} {rmse:>8.3f}")

    out.append("")


# ============================================================
# LOO CROSS-VALIDATION
# ============================================================

def run_loo_cv(results, out):
    """Leave-one-out cross-validation for CO, BPS, and combined."""
    out.append(f"\n{'='*72}")
    out.append(f"  LEAVE-ONE-OUT CROSS-VALIDATION")
    out.append(f"{'='*72}\n")

    n = len(results)
    ln_kf = np.array([r['ln_kf'] for r in results])
    co = np.array([r['co_exp'] for r in results])
    bps = np.array([r['bps_energy'] for r in results])
    bps_norm = np.array([r['bps_norm'] for r in results])

    models = {
        "CO only": lambda: np.column_stack([co, np.ones(n)]),
        "BPS only": lambda: np.column_stack([bps, np.ones(n)]),
        "BPS/res only": lambda: np.column_stack([bps_norm, np.ones(n)]),
        "CO + BPS": lambda: np.column_stack([co, bps, np.ones(n)]),
        "CO + BPS/res": lambda: np.column_stack([co, bps_norm, np.ones(n)]),
    }

    h = f"    {'Model':<20} {'MAE':>8} {'RMSE':>8} {'R2_cv':>8}"
    out.append(h)
    out.append("    " + "-" * (len(h) - 4))

    for name, X_fn in models.items():
        X = X_fn()
        errors = []
        preds = np.zeros(n)
        for i in range(n):
            # Train on all except i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_train = X[mask]
            y_train = ln_kf[mask]
            beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
            pred_i = X[i] @ beta
            preds[i] = pred_i
            errors.append(abs(ln_kf[i] - pred_i))

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        ss_res = np.sum((ln_kf - preds)**2)
        ss_tot = np.sum((ln_kf - ln_kf.mean())**2)
        r2_cv = 1 - ss_res / ss_tot
        out.append(f"    {name:<20} {mae:>8.3f} {rmse:>8.3f} {r2_cv:>8.4f}")

    out.append("")


# ============================================================
# PER-PROTEIN TABLE
# ============================================================

def write_protein_table(results, out):
    """Publication-ready table of all Plaxco proteins."""
    out.append(f"\n{'='*72}")
    out.append(f"  PLAXCO PROTEIN TABLE")
    out.append(f"{'='*72}\n")

    # Sort by ln(kf)
    results_sorted = sorted(results, key=lambda r: r['ln_kf'])

    h = (f"  {'PDB':>4} {'Name':<18} {'L':>4} {'ln(kf)':>7} {'CO_exp':>7} "
         f"{'CO_comp':>7} {'BPS/res':>8} {'pLDDT':>6} {'%a':>5} {'%b':>5}")
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))

    for r in results_sorted:
        co_c = f"{r['contact_order']:.4f}" if r.get('contact_order') else "—"
        plddt = f"{r['plddt_mean']:.0f}" if r.get('plddt_mean') else "—"
        helix = f"{r['pct_helix']:.0f}%" if r.get('pct_helix') is not None else "—"
        sheet = f"{r['pct_sheet']:.0f}%" if r.get('pct_sheet') is not None else "—"
        out.append(f"  {r['pdb_id']:>4} {r['name']:<18} {r['L']:>4} {r['ln_kf']:>7.2f} "
                   f"{r['co_exp']:>7.3f} {co_c:>7} {r['bps_norm']:>8.3f} "
                   f"{plddt:>6} {helix:>5} {sheet:>5}")

    out.append("")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BPS Plaxco Validation")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--out", type=str,
                        default=str(OUTPUT_DIR / "bps_plaxco_validation.txt"))
    parser.add_argument("--download", action="store_true",
                        help="Download Plaxco proteins and exit")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute BPS even if DB has values")
    args = parser.parse_args()

    # Download mode
    if args.download:
        download_plaxco_proteins()
        return

    # Try loading from DB first
    results = []
    db_path = Path(args.db)
    if db_path.exists() and not args.recompute:
        conn = sqlite3.connect(str(db_path))
        try:
            results = load_plaxco_from_db(conn)
            if results:
                print(f"Loaded {len(results)}/{len(PLAXCO_PROTEINS)} "
                      f"Plaxco proteins from database")
        finally:
            conn.close()

    # If not enough from DB, compute fresh
    if len(results) < 10:
        print("Computing BPS for Plaxco proteins...")
        download_plaxco_proteins()
        results = compute_plaxco_bps()

    if len(results) < 5:
        print(f"ERROR: Only {len(results)} proteins available. Need at least 5.")
        return

    # Run analyses
    out = []
    out.append("=" * 72)
    out.append("  BPS PLAXCO FOLDING RATE VALIDATION")
    out.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out.append(f"  Proteins: {len(results)}/{len(PLAXCO_PROTEINS)}")
    out.append("=" * 72)

    write_protein_table(results, out)
    run_correlations(results, out)
    run_loo_cv(results, out)

    # Decision point from research plan
    out.append(f"\n{'='*72}")
    out.append(f"  DECISION POINT")
    out.append(f"{'='*72}\n")

    ln_kf = np.array([r['ln_kf'] for r in results])
    bps = np.array([r['bps_energy'] for r in results])
    r_bps, _ = pearsonr(bps, ln_kf)

    if abs(r_bps) > 0.5:
        out.append(f"  r(BPS, ln(kf)) = {r_bps:.4f} > 0.5")
        out.append(f"  -> PUBLISHABLE: BPS is a folding rate predictor")
        out.append(f"  -> Next: Scale to larger folding rate datasets (PFDB)")
    elif abs(r_bps) > 0.3:
        out.append(f"  r(BPS, ln(kf)) = {r_bps:.4f} (moderate)")
        out.append(f"  -> PROMISING: BPS has some predictive power")
        out.append(f"  -> Next: Check if combined CO+BPS model improves significantly")
    else:
        out.append(f"  r(BPS, ln(kf)) = {r_bps:.4f} < 0.3")
        out.append(f"  -> BPS does not predict folding kinetics")
        out.append(f"  -> Reframe as structural descriptor, not kinetic predictor")

    out.append("")

    text = "\n".join(out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"\nWrote {text.count(chr(10))} lines -> {out_path}")


if __name__ == "__main__":
    main()
