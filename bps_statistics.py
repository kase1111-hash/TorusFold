#!/usr/bin/env python3
"""
BPS STATISTICS — Advanced statistical analyses for BPS cross-proteome data
==========================================================================
Partial correlations, multiple regression, PCA, nested ANOVA,
bootstrap CIs, and pLDDT-matched subsampling.

All data comes from bps_database.db (produced by bps_process.py).

Usage:
  python bps_statistics.py                          # run all analyses
  python bps_statistics.py --db path/to/db
  python bps_statistics.py --analysis partial       # just partial correlations
  python bps_statistics.py --plddt-threshold 85     # high-quality subset
"""

import argparse
import sqlite3
import math
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import pearsonr, f_oneway, ttest_ind

# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("./alphafold_bps_results/bps_database.db")
OUTPUT_DIR = Path("./alphafold_bps_results")

BACTERIA = {"ecoli", "bacillus", "tuberculosis", "staph", "pseudomonas",
            "salmonella", "campylobacter", "helicobacter"}
METAZOA = {"human", "mouse", "rat", "fly", "worm", "zebrafish", "chicken",
           "pig", "cow", "dog", "cat", "gorilla", "chimp", "mosquito", "honeybee"}
PLANTS = {"arabidopsis", "rice", "soybean", "maize"}
FUNGI = {"yeast", "fission_yeast", "candida"}
PROTISTS = {"malaria", "leishmania", "trypanosoma", "dictyostelium"}
EUKARYOTES = METAZOA | PLANTS | FUNGI | PROTISTS

PATHOGENS = {"tuberculosis", "staph", "salmonella", "pseudomonas",
             "helicobacter", "malaria", "campylobacter", "leishmania",
             "trypanosoma", "candida"}
FREE_LIVING = {"ecoli", "bacillus", "yeast", "fission_yeast", "dictyostelium"}

KINGDOM_MAP = {}
for _org in BACTERIA: KINGDOM_MAP[_org] = "Bacteria"
for _org in METAZOA: KINGDOM_MAP[_org] = "Metazoa"
for _org in PLANTS: KINGDOM_MAP[_org] = "Plants"
for _org in FUNGI: KINGDOM_MAP[_org] = "Fungi"
for _org in PROTISTS: KINGDOM_MAP[_org] = "Protists"

VARIABLES = ["bps_norm", "L", "plddt_mean", "pct_helix", "pct_sheet",
             "rg", "contact_order", "n_transitions"]
VAR_LABELS = ["BPS/res", "L", "pLDDT", "%helix", "%sheet", "Rg", "CO", "trans"]


# ============================================================
# DATA LOADING
# ============================================================

def load_complete_data(conn, plddt_threshold=0, min_L=30):
    """Load proteins with all 8 variables populated, applying quality filters."""
    rows = conn.execute(f"""
        SELECT organism, {', '.join(VARIABLES)}
        FROM proteins
        WHERE plddt_mean IS NOT NULL
          AND contact_order IS NOT NULL
          AND rg IS NOT NULL
          AND pct_helix IS NOT NULL
          AND L >= ?
          AND plddt_mean >= ?
    """, (min_L, plddt_threshold)).fetchall()

    orgs = [r[0] for r in rows]
    data = np.array([r[1:] for r in rows])
    return orgs, data


def load_organism_means(conn, plddt_threshold=0, min_L=30, min_N=100):
    """Load per-organism mean BPS/res for organisms with sufficient data."""
    rows = conn.execute(f"""
        SELECT organism, COUNT(*) as n, AVG(bps_norm),
               AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm)
        FROM proteins
        WHERE plddt_mean >= ? AND L >= ?
        GROUP BY organism HAVING n >= ?
        ORDER BY AVG(bps_norm) DESC
    """, (plddt_threshold, min_L, min_N)).fetchall()
    return rows


# ============================================================
# 1. PARTIAL CORRELATIONS (controlling for pLDDT)
# ============================================================

def partial_corr(x, y, z):
    """Partial correlation r(x, y | z). z can be 1D or 2D."""
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    # Residualize x and y on z via OLS
    Z = np.column_stack([z, np.ones(len(z))])
    def resid(v):
        beta, _, _, _ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    rx, ry = resid(x), resid(y)
    r, p = pearsonr(rx, ry)
    return r, p


def run_partial_correlations(conn, out, plddt_threshold=0):
    """Partial correlation matrix controlling for pLDDT."""
    out.append(f"\n{'='*72}")
    out.append(f"  PARTIAL CORRELATIONS (controlling for pLDDT, threshold >= {plddt_threshold})")
    out.append(f"{'='*72}\n")

    orgs, data = load_complete_data(conn, plddt_threshold)
    if len(data) < 100:
        out.append(f"  [Insufficient data: {len(data)} proteins]")
        return
    out.append(f"  N = {len(data):,} proteins with all 8 variables\n")

    # Raw correlations
    out.append(f"  RAW Pearson correlations:")
    h = f"  {'':>10}" + "".join(f"{l:>10}" for l in VAR_LABELS)
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))
    for i in range(len(VARIABLES)):
        row = f"  {VAR_LABELS[i]:>10}"
        for j in range(len(VARIABLES)):
            if i == j:
                row += f"{'—':>10}"
            else:
                r, _ = pearsonr(data[:, i], data[:, j])
                row += f"{r:>10.4f}"
        out.append(row)

    # Partial correlations controlling for pLDDT (column 2)
    plddt_col = 2  # index of plddt_mean in VARIABLES
    out.append(f"\n  PARTIAL correlations r(X, Y | pLDDT):")
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))
    for i in range(len(VARIABLES)):
        row = f"  {VAR_LABELS[i]:>10}"
        for j in range(len(VARIABLES)):
            if i == j:
                row += f"{'—':>10}"
            elif i == plddt_col or j == plddt_col:
                row += f"{'':>10}"
            else:
                r, _ = partial_corr(data[:, i], data[:, j], data[:, plddt_col])
                row += f"{r:>10.4f}"
        out.append(row)

    out.append("")


# ============================================================
# 2. MULTIPLE REGRESSION (BPS ~ all predictors)
# ============================================================

def run_multiple_regression(conn, out, plddt_threshold=0):
    """BPS/res ~ pLDDT + %helix + %sheet + L + CO + Rg + transitions."""
    out.append(f"\n{'='*72}")
    out.append(f"  MULTIPLE REGRESSION: BPS/res ~ all predictors")
    out.append(f"{'='*72}\n")

    orgs, data = load_complete_data(conn, plddt_threshold)
    if len(data) < 100:
        out.append(f"  [Insufficient data: {len(data)} proteins]")
        return

    y = data[:, 0]  # bps_norm
    X_raw = data[:, 1:]  # all other variables
    predictor_names = VAR_LABELS[1:]

    # Standardize for comparable coefficients
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_z = (X_raw - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std()
    y_z = (y - y_mean) / y_std

    X = np.column_stack([X_z, np.ones(len(y))])
    beta, residuals, _, _ = np.linalg.lstsq(X, y_z, rcond=None)

    pred = X @ beta
    ss_res = np.sum((y_z - pred)**2)
    ss_tot = np.sum((y_z - y_z.mean())**2)
    r2 = 1 - ss_res / ss_tot
    n, p = len(y), len(predictor_names)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    out.append(f"  N = {n:,}, predictors = {p}")
    out.append(f"  R-squared = {r2:.4f}")
    out.append(f"  Adjusted R-squared = {r2_adj:.4f}")
    out.append(f"  Residual variance = {1 - r2:.4f} (= UNIQUE BPS signal)\n")

    h = f"  {'Predictor':<12} {'beta_std':>10} {'direction':>10}"
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))
    for i, name in enumerate(predictor_names):
        direction = "+" if beta[i] > 0 else "-"
        out.append(f"  {name:<12} {beta[i]:>10.4f} {direction:>10}")

    # Incremental R-squared: add predictors one at a time
    out.append(f"\n  INCREMENTAL R-squared (each predictor added alone):")
    for i, name in enumerate(predictor_names):
        Xi = np.column_stack([X_z[:, i], np.ones(len(y))])
        beta_i, _, _, _ = np.linalg.lstsq(Xi, y_z, rcond=None)
        pred_i = Xi @ beta_i
        r2_i = 1 - np.sum((y_z - pred_i)**2) / ss_tot
        out.append(f"    {name:<12} R2 = {r2_i:.4f}")

    out.append("")


# ============================================================
# 3. PCA / VARIANCE DECOMPOSITION
# ============================================================

def run_pca(conn, out, plddt_threshold=0):
    """PCA on all 8 variables. Where does BPS sit?"""
    out.append(f"\n{'='*72}")
    out.append(f"  PCA: 8-variable decomposition")
    out.append(f"{'='*72}\n")

    orgs, data = load_complete_data(conn, plddt_threshold)
    if len(data) < 100:
        out.append(f"  [Insufficient data: {len(data)} proteins]")
        return

    # Standardize
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    stds[stds < 1e-10] = 1.0
    Z = (data - means) / stds

    # Covariance and eigen decomposition
    cov = np.cov(Z, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    total_var = eigenvalues.sum()
    cum_var = np.cumsum(eigenvalues) / total_var

    out.append(f"  N = {len(data):,} proteins\n")
    out.append(f"  {'PC':>4} {'Eigenvalue':>12} {'% Variance':>12} {'Cumulative':>12}")
    out.append("  " + "-" * 44)
    for i in range(len(eigenvalues)):
        out.append(f"  {i+1:>4} {eigenvalues[i]:>12.4f} "
                   f"{eigenvalues[i]/total_var*100:>11.1f}% {cum_var[i]*100:>11.1f}%")

    # Loadings for first 3 PCs
    out.append(f"\n  LOADINGS (first 3 PCs):")
    h = f"  {'Variable':<12}" + "".join(f"{'PC'+str(i+1):>10}" for i in range(3))
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))
    for j, label in enumerate(VAR_LABELS):
        row = f"  {label:<12}"
        for i in range(3):
            row += f"{eigenvectors[j, i]:>10.4f}"
        out.append(row)

    # Interpretation
    bps_idx = 0
    bps_loadings = eigenvectors[bps_idx, :3]
    dominant_pc = np.argmax(np.abs(bps_loadings)) + 1
    out.append(f"\n  BPS/res loads most strongly on PC{dominant_pc} "
               f"(loading = {bps_loadings[dominant_pc-1]:.4f})")

    plddt_idx = 2
    plddt_loadings = eigenvectors[plddt_idx, :3]
    plddt_pc = np.argmax(np.abs(plddt_loadings)) + 1

    if dominant_pc == plddt_pc:
        out.append(f"  WARNING: BPS and pLDDT share the same dominant PC (PC{dominant_pc})")
        out.append(f"  -> BPS may largely be a pLDDT proxy in this dataset")
    else:
        out.append(f"  pLDDT loads most strongly on PC{plddt_pc}")
        out.append(f"  -> BPS and pLDDT define different axes — BPS captures unique information")

    out.append("")


# ============================================================
# 4. NESTED ANOVA (kingdom / organism / protein)
# ============================================================

def run_nested_anova(conn, out, plddt_threshold=0):
    """Variance decomposition: kingdom vs organism vs protein level."""
    out.append(f"\n{'='*72}")
    out.append(f"  NESTED ANOVA: kingdom / organism / protein")
    out.append(f"{'='*72}\n")

    rows = conn.execute("""
        SELECT organism, bps_norm FROM proteins
        WHERE plddt_mean IS NOT NULL AND plddt_mean >= ?
          AND L >= 30 AND bps_norm IS NOT NULL
    """, (plddt_threshold,)).fetchall()

    if len(rows) < 100:
        out.append(f"  [Insufficient data: {len(rows)} proteins]")
        return

    # Group by organism
    org_data = {}
    for org, bps in rows:
        org_data.setdefault(org, []).append(bps)

    # Filter to organisms with enough data
    org_data = {k: v for k, v in org_data.items() if len(v) >= 100}

    if len(org_data) < 3:
        out.append(f"  [Too few organisms with >= 100 proteins: {len(org_data)}]")
        return

    all_bps = np.concatenate(list(org_data.values()))
    grand_mean = all_bps.mean()
    n_total = len(all_bps)

    # Total SS
    ss_total = np.sum((all_bps - grand_mean)**2)

    # Between-organism SS
    ss_organism = 0
    for org, vals in org_data.items():
        arr = np.array(vals)
        ss_organism += len(arr) * (arr.mean() - grand_mean)**2

    # Between-kingdom SS
    kingdom_data = {}
    for org, vals in org_data.items():
        k = KINGDOM_MAP.get(org, "Unknown")
        kingdom_data.setdefault(k, []).extend(vals)

    kingdom_means = {k: np.mean(v) for k, v in kingdom_data.items()}
    ss_kingdom = 0
    for k, vals in kingdom_data.items():
        ss_kingdom += len(vals) * (kingdom_means[k] - grand_mean)**2

    # Within-organism SS
    ss_within = ss_total - ss_organism

    out.append(f"  N = {n_total:,} proteins, {len(org_data)} organisms, "
               f"{len(kingdom_data)} kingdoms")
    out.append(f"  pLDDT threshold >= {plddt_threshold}\n")

    h = f"  {'Source':<20} {'SS':>12} {'% of Total':>12}"
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))
    out.append(f"  {'Kingdom':<20} {ss_kingdom:>12.2f} {ss_kingdom/ss_total*100:>11.1f}%")
    out.append(f"  {'Organism (in king.)':<20} {ss_organism - ss_kingdom:>12.2f} "
               f"{(ss_organism - ss_kingdom)/ss_total*100:>11.1f}%")
    out.append(f"  {'Within organism':<20} {ss_within:>12.2f} {ss_within/ss_total*100:>11.1f}%")
    out.append(f"  {'Total':<20} {ss_total:>12.2f} {'100.0%':>12}")

    # One-way ANOVA across kingdoms
    kingdom_arrays = [np.array(v) for v in kingdom_data.values() if len(v) >= 10]
    if len(kingdom_arrays) >= 2:
        f_stat, p_val = f_oneway(*kingdom_arrays)
        out.append(f"\n  One-way ANOVA across kingdoms: F = {f_stat:.2f}, p = {p_val:.2e}")

    # Per-kingdom means
    out.append(f"\n  Kingdom means:")
    for k in sorted(kingdom_data.keys()):
        vals = kingdom_data[k]
        out.append(f"    {k:<12} {np.mean(vals):.4f} +/- {np.std(vals):.4f} (N={len(vals):,})")

    out.append("")


# ============================================================
# 5. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def run_bootstrap_ci(conn, out, plddt_threshold=0, n_boot=10000):
    """Per-organism BPS/res mean with 95% CI via bootstrap."""
    out.append(f"\n{'='*72}")
    out.append(f"  BOOTSTRAP 95% CIs FOR ORGANISM MEANS (n_boot={n_boot:,})")
    out.append(f"{'='*72}\n")

    org_rows = load_organism_means(conn, plddt_threshold, min_N=100)
    if not org_rows:
        out.append("  [No organisms with >= 100 proteins]")
        return

    h = f"  {'Organism':<18} {'N':>7} {'Mean':>9} {'95% CI':>20} {'Width':>8} {'Kingdom':>10}"
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))

    rng = np.random.default_rng(42)
    all_means = []

    for org, n, mean_bps, var_bps in org_rows:
        # Fetch individual values for bootstrap
        vals = [r[0] for r in conn.execute(
            "SELECT bps_norm FROM proteins WHERE organism=? AND plddt_mean >= ? AND L >= 30",
            (org, plddt_threshold)).fetchall()]
        arr = np.array(vals)

        # Bootstrap
        boot_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                               for _ in range(n_boot)])
        ci_lo = np.percentile(boot_means, 2.5)
        ci_hi = np.percentile(boot_means, 97.5)
        width = ci_hi - ci_lo
        k = KINGDOM_MAP.get(org, "?")

        out.append(f"  {org:<18} {len(arr):>7} {arr.mean():>9.4f} "
                   f"[{ci_lo:.4f}, {ci_hi:.4f}] {width:>8.4f} {k:>10}")
        all_means.append((org, arr.mean(), ci_lo, ci_hi))

    # Check overlap
    out.append(f"\n  CI overlap analysis:")
    n_overlap = 0
    n_pairs = 0
    for i in range(len(all_means)):
        for j in range(i + 1, len(all_means)):
            n_pairs += 1
            _, _, lo_i, hi_i = all_means[i]
            _, _, lo_j, hi_j = all_means[j]
            if lo_i <= hi_j and lo_j <= hi_i:
                n_overlap += 1
    if n_pairs > 0:
        out.append(f"    {n_overlap}/{n_pairs} organism pairs have overlapping 95% CIs "
                   f"({n_overlap/n_pairs*100:.0f}%)")

    out.append("")


# ============================================================
# 6. pLDDT-MATCHED SUBSAMPLING
# ============================================================

def run_plddt_matching(conn, out, reference_org="ecoli", n_boot=1000):
    """Subsample each organism to match the pLDDT distribution of a reference."""
    out.append(f"\n{'='*72}")
    out.append(f"  pLDDT-MATCHED SUBSAMPLING (reference: {reference_org})")
    out.append(f"{'='*72}\n")

    # Get reference pLDDT distribution
    ref_plddts = [r[0] for r in conn.execute(
        "SELECT plddt_mean FROM proteins WHERE organism=? AND plddt_mean IS NOT NULL AND L >= 30",
        (reference_org,)).fetchall()]
    if len(ref_plddts) < 100:
        out.append(f"  [Reference {reference_org} has < 100 proteins with pLDDT]")
        return

    ref_arr = np.array(ref_plddts)
    # Bin reference into pLDDT tiers
    bins = [0, 50, 70, 85, 90, 95, 101]
    ref_hist, _ = np.histogram(ref_arr, bins=bins)
    ref_fracs = ref_hist / ref_hist.sum()

    out.append(f"  Reference pLDDT distribution ({reference_org}, N={len(ref_arr):,}):")
    for i in range(len(bins) - 1):
        out.append(f"    [{bins[i]}, {bins[i+1]}): {ref_fracs[i]*100:.1f}%")

    # For each organism, subsample to match
    orgs = [r[0] for r in conn.execute(
        "SELECT organism FROM proteins WHERE plddt_mean IS NOT NULL "
        "GROUP BY organism HAVING COUNT(*) >= 200 ORDER BY organism"
    ).fetchall()]

    rng = np.random.default_rng(42)

    h = f"  {'Organism':<18} {'N_total':>8} {'N_matched':>10} {'BPS_raw':>9} {'BPS_matched':>12} {'Delta':>8}"
    out.append(f"\n{h}")
    out.append("  " + "-" * (len(h) - 2))

    for org in orgs:
        rows = conn.execute(
            "SELECT bps_norm, plddt_mean FROM proteins "
            "WHERE organism=? AND plddt_mean IS NOT NULL AND L >= 30",
            (org,)).fetchall()
        bps = np.array([r[0] for r in rows])
        plddt = np.array([r[1] for r in rows])
        bps_raw = bps.mean()

        # Bin this organism
        org_bins = np.digitize(plddt, bins) - 1  # 0-indexed bin
        org_bins = np.clip(org_bins, 0, len(bins) - 2)

        # Target count per bin: match reference fractions
        min_matched = min(200, len(bps))
        target_per_bin = (ref_fracs * min_matched).astype(int)
        target_per_bin = np.maximum(target_per_bin, 0)

        # Check feasibility
        available = np.array([np.sum(org_bins == b) for b in range(len(bins) - 1)])
        feasible = all(available[b] >= target_per_bin[b] for b in range(len(bins) - 1)
                       if target_per_bin[b] > 0)
        if not feasible:
            # Reduce targets proportionally
            ratios = np.where(target_per_bin > 0,
                              np.minimum(available / np.maximum(target_per_bin, 1), 1.0), 0)
            scale = ratios[ratios > 0].min() if np.any(ratios > 0) else 0
            target_per_bin = (target_per_bin * scale).astype(int)

        total_matched = target_per_bin.sum()
        if total_matched < 50:
            out.append(f"  {org:<18} {len(bps):>8} {'< 50':>10} {bps_raw:>9.3f} {'—':>12} {'—':>8}")
            continue

        # Bootstrap matched means
        boot_means = []
        for _ in range(n_boot):
            sampled = []
            for b in range(len(bins) - 1):
                idx = np.where(org_bins == b)[0]
                if target_per_bin[b] > 0 and len(idx) > 0:
                    chosen = rng.choice(idx, size=min(target_per_bin[b], len(idx)), replace=True)
                    sampled.extend(bps[chosen].tolist())
            if sampled:
                boot_means.append(np.mean(sampled))

        if boot_means:
            matched_mean = np.mean(boot_means)
            delta = matched_mean - bps_raw
            out.append(f"  {org:<18} {len(bps):>8} {total_matched:>10} {bps_raw:>9.3f} "
                       f"{matched_mean:>12.3f} {delta:>+8.3f}")

    out.append("")


# ============================================================
# 7. PATHOGEN vs FREE-LIVING COMPARISON
# ============================================================

def run_pathogen_comparison(conn, out, plddt_threshold=0):
    """Compare BPS/res between pathogens and free-living organisms."""
    out.append(f"\n{'='*72}")
    out.append(f"  PATHOGEN vs FREE-LIVING COMPARISON (pLDDT >= {plddt_threshold})")
    out.append(f"{'='*72}\n")

    path_bps = []
    free_bps = []
    path_org_means = []
    free_org_means = []

    h = f"  {'Organism':<18} {'Type':<12} {'N':>7} {'BPS/res':>9} {'pLDDT':>7}"
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))

    for org_set, label in [(PATHOGENS, "pathogen"), (FREE_LIVING, "free-living")]:
        for org in sorted(org_set):
            row = conn.execute("""
                SELECT COUNT(*), AVG(bps_norm), AVG(plddt_mean)
                FROM proteins WHERE organism=? AND plddt_mean >= ? AND L >= 30
                HAVING COUNT(*) >= 10
            """, (org, plddt_threshold)).fetchone()
            if row and row[0]:
                out.append(f"  {org:<18} {label:<12} {row[0]:>7} {row[1]:>9.4f} "
                           f"{row[2]:>7.1f}")
                vals = [r[0] for r in conn.execute(
                    "SELECT bps_norm FROM proteins WHERE organism=? AND plddt_mean >= ? AND L >= 30",
                    (org, plddt_threshold)).fetchall()]
                if label == "pathogen":
                    path_bps.extend(vals)
                    path_org_means.append(np.mean(vals))
                else:
                    free_bps.extend(vals)
                    free_org_means.append(np.mean(vals))

    if path_bps and free_bps:
        out.append(f"\n  Pathogen:     {np.mean(path_bps):.3f} +/- {np.std(path_bps):.3f} "
                   f"(N={len(path_bps):,})")
        out.append(f"  Free-living:  {np.mean(free_bps):.3f} +/- {np.std(free_bps):.3f} "
                   f"(N={len(free_bps):,})")

        if len(path_org_means) >= 2 and len(free_org_means) >= 2:
            t, p = ttest_ind(path_org_means, free_org_means)
            out.append(f"  t-test (organism means): t={t:.3f}, p={p:.2e}")

    out.append("")


# ============================================================
# 8. FOLD-CLASS ANALYSIS
# ============================================================

def run_fold_class_analysis(conn, out, plddt_threshold=85):
    """BPS/res by fold class, controlling for size and pLDDT."""
    out.append(f"\n{'='*72}")
    out.append(f"  FOLD-CLASS ANALYSIS (pLDDT >= {plddt_threshold})")
    out.append(f"{'='*72}\n")

    # Uncontrolled
    out.append("  A. All proteins:")
    classes = [
        ("Alpha-rich", "pct_helix > 40 AND pct_sheet < 15"),
        ("Beta-rich", "pct_sheet > 25 AND pct_helix < 15"),
        ("Mixed a/b", "pct_helix > 15 AND pct_sheet > 15"),
        ("Coil-dom.", "pct_coil > 80"),
    ]
    h = f"    {'Class':<14} {'N':>8} {'BPS/res':>9} {'SD':>7} {'<L>':>6} {'<pLDDT>':>8}"
    out.append(h)
    out.append("    " + "-" * (len(h) - 4))
    for label, where in classes:
        r = conn.execute(f"""
            SELECT COUNT(*), AVG(bps_norm),
                   AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm),
                   AVG(L), AVG(plddt_mean)
            FROM proteins
            WHERE pct_helix IS NOT NULL AND plddt_mean >= ? AND L >= 30 AND {where}
        """, (plddt_threshold,)).fetchone()
        if r[0] and r[0] > 10:
            sd = math.sqrt(max(r[2] or 0, 0))
            out.append(f"    {label:<14} {r[0]:>8,} {r[1]:>9.4f} {sd:>7.4f} "
                       f"{r[3]:>6.0f} {r[4]:>8.1f}")

    # Size-controlled (200-400 residues)
    out.append(f"\n  B. Size-controlled (L = 200-400, pLDDT >= {plddt_threshold}):")
    out.append(h)
    out.append("    " + "-" * (len(h) - 4))
    for label, where in classes:
        r = conn.execute(f"""
            SELECT COUNT(*), AVG(bps_norm),
                   AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm),
                   AVG(L), AVG(plddt_mean)
            FROM proteins
            WHERE pct_helix IS NOT NULL AND plddt_mean >= ? AND L >= 200 AND L <= 400
              AND {where}
        """, (plddt_threshold,)).fetchone()
        if r[0] and r[0] > 10:
            sd = math.sqrt(max(r[2] or 0, 0))
            out.append(f"    {label:<14} {r[0]:>8,} {r[1]:>9.4f} {sd:>7.4f} "
                       f"{r[3]:>6.0f} {r[4]:>8.1f}")

    # 2D heatmap data (text version): 5x5 grid of helix% x sheet%
    out.append(f"\n  C. 2D fold-class heatmap (mean BPS/res in each bin):")
    helix_edges = [0, 10, 20, 35, 50, 100]
    sheet_edges = [0, 5, 15, 25, 40, 100]
    label = 'helix\\sheet'
    h2 = f"    {label:>12}"
    for i in range(len(sheet_edges) - 1):
        h2 += f" {sheet_edges[i]}-{sheet_edges[i+1]}%".rjust(10)
    out.append(h2)
    out.append("    " + "-" * (len(h2) - 4))

    for hi in range(len(helix_edges) - 1):
        row = f"    {helix_edges[hi]:>3}-{helix_edges[hi+1]:<3}%   "
        for si in range(len(sheet_edges) - 1):
            r = conn.execute("""
                SELECT COUNT(*), AVG(bps_norm) FROM proteins
                WHERE pct_helix >= ? AND pct_helix < ?
                  AND pct_sheet >= ? AND pct_sheet < ?
                  AND plddt_mean >= ? AND L >= 30
            """, (helix_edges[hi], helix_edges[hi+1],
                  sheet_edges[si], sheet_edges[si+1],
                  plddt_threshold)).fetchone()
            if r[0] and r[0] >= 10:
                row += f"{r[1]:>10.4f}"
            else:
                n = r[0] or 0
                row += f"{'('+str(n)+')':>10}"
        out.append(row)

    out.append("")


# ============================================================
# 9. SIZE-DEPENDENT CONVERGENCE
# ============================================================

def run_size_convergence(conn, out, plddt_threshold=0):
    """BPS/res vs L in bins — does it converge for large proteins?"""
    out.append(f"\n{'='*72}")
    out.append(f"  SIZE-DEPENDENT CONVERGENCE (pLDDT >= {plddt_threshold})")
    out.append(f"{'='*72}\n")

    orgs = [r[0] for r in conn.execute("""
        SELECT organism FROM proteins
        WHERE plddt_mean >= ? AND L >= 30
        GROUP BY organism HAVING COUNT(*) >= 200
        ORDER BY organism
    """, (plddt_threshold,)).fetchall()]

    bin_edges = list(range(50, 2050, 50))  # 50, 100, 150, ..., 2000

    # Header: show select bins
    show_bins = [50, 100, 200, 300, 500, 800, 1000, 1500]
    h = f"  {'Organism':<16}" + "".join(f"  L={b:>4}".rjust(10) for b in show_bins)
    out.append(h)
    out.append("  " + "-" * (len(h) - 2))

    for org in orgs:
        row = f"  {org:<16}"
        for target in show_bins:
            lo = target - 25 if target <= 100 else target - 50
            hi = target + 25 if target <= 100 else target + 50
            r = conn.execute("""
                SELECT COUNT(*), AVG(bps_norm) FROM proteins
                WHERE organism=? AND L >= ? AND L < ? AND plddt_mean >= ?
            """, (org, lo, hi, plddt_threshold)).fetchone()
            if r[0] and r[0] >= 5:
                row += f"{r[1]:>10.4f}"
            else:
                row += f"{'—':>10}"
        out.append(row)

    # Global convergence
    out.append(f"\n  Global (all organisms pooled):")
    row = f"  {'ALL':<16}"
    for target in show_bins:
        lo = target - 25 if target <= 100 else target - 50
        hi = target + 25 if target <= 100 else target + 50
        r = conn.execute("""
            SELECT COUNT(*), AVG(bps_norm) FROM proteins
            WHERE L >= ? AND L < ? AND plddt_mean >= ?
        """, (lo, hi, plddt_threshold)).fetchone()
        if r[0] and r[0] >= 10:
            row += f"{r[1]:>10.4f}"
        else:
            row += f"{'—':>10}"
    out.append(row)

    out.append("")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BPS Statistics — advanced analyses")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR / "bps_statistics.txt"))
    parser.add_argument("--plddt-threshold", type=float, default=0,
                        help="Minimum pLDDT for quality filtering (0=all, 85=high, 90=very high)")
    parser.add_argument("--analysis", type=str, default="all",
                        choices=["all", "partial", "regression", "pca", "anova",
                                 "bootstrap", "plddt-match", "pathogen", "fold-class",
                                 "size-convergence"])
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        print("Run bps_process.py first.")
        return

    conn = sqlite3.connect(str(db_path))

    try:
        total = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
        if total == 0:
            print("ERROR: Database is empty.")
            return

        out = []
        out.append("=" * 72)
        out.append("  BPS STATISTICAL ANALYSIS")
        out.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        out.append(f"  Database: {db_path} ({db_path.stat().st_size/1024/1024:.1f} MB)")
        out.append(f"  Proteins: {total:,}")
        out.append(f"  pLDDT threshold: {args.plddt_threshold}")
        out.append("=" * 72)

        analyses = {
            "partial": run_partial_correlations,
            "regression": run_multiple_regression,
            "pca": run_pca,
            "anova": run_nested_anova,
            "bootstrap": run_bootstrap_ci,
            "plddt-match": run_plddt_matching,
            "pathogen": run_pathogen_comparison,
            "fold-class": run_fold_class_analysis,
            "size-convergence": run_size_convergence,
        }

        if args.analysis == "all":
            run_list = list(analyses.values())
        else:
            run_list = [analyses[args.analysis]]

        for fn in run_list:
            try:
                if fn == run_fold_class_analysis:
                    # Use user-specified threshold; do not force pLDDT >= 85
                    fn(conn, out, args.plddt_threshold)
                else:
                    fn(conn, out, args.plddt_threshold)
            except Exception as e:
                out.append(f"\n  [Error in {fn.__name__}: {e}]\n")
    finally:
        conn.close()

    text = "\n".join(out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"\nWrote {text.count(chr(10))} lines ({len(text):,} chars) -> {out_path}")


if __name__ == "__main__":
    main()
