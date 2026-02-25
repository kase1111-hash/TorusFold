#!/usr/bin/env python3
"""
BPS FIGURES — Publication-quality figures for BPS cross-proteome analysis
=========================================================================
Generates matplotlib figures from bps_database.db.

Usage:
  python bps_figures.py                          # all figures
  python bps_figures.py --figure distribution    # single figure
  python bps_figures.py --plddt-threshold 85     # high-quality subset
  python bps_figures.py --format pdf             # output format
"""

import argparse
import sqlite3
import math
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("./alphafold_bps_results/bps_database.db")
FIG_DIR = Path("./alphafold_bps_results/figures")

BACTERIA = {"ecoli", "bacillus", "tuberculosis", "staph", "pseudomonas",
            "salmonella", "campylobacter", "helicobacter"}
METAZOA = {"human", "mouse", "rat", "fly", "worm", "zebrafish", "chicken",
           "pig", "cow", "dog", "cat", "gorilla", "chimp", "mosquito", "honeybee"}
PLANTS = {"arabidopsis", "rice", "soybean", "maize"}
FUNGI = {"yeast", "fission_yeast", "candida"}
PROTISTS = {"malaria", "leishmania", "trypanosoma", "dictyostelium"}

KINGDOM_MAP = {}
for _org in BACTERIA: KINGDOM_MAP[_org] = "Bacteria"
for _org in METAZOA: KINGDOM_MAP[_org] = "Metazoa"
for _org in PLANTS: KINGDOM_MAP[_org] = "Plants"
for _org in FUNGI: KINGDOM_MAP[_org] = "Fungi"
for _org in PROTISTS: KINGDOM_MAP[_org] = "Protists"

KINGDOM_COLORS = {
    "Bacteria": "#e74c3c",
    "Metazoa": "#3498db",
    "Fungi": "#2ecc71",
    "Plants": "#f39c12",
    "Protists": "#9b59b6",
}


# ============================================================
# 1. BPS/res DISTRIBUTION OVERLAY
# ============================================================

def fig_distribution(conn, plddt_thr, fmt):
    """BPS/res distribution overlays — all organisms on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    orgs = [r[0] for r in conn.execute("""
        SELECT organism FROM proteins WHERE plddt_mean >= ?
        GROUP BY organism HAVING COUNT(*) >= 100 ORDER BY organism
    """, (plddt_thr,)).fetchall()]

    bins = np.linspace(0.05, 0.35, 60)
    for org in orgs:
        vals = [r[0] for r in conn.execute(
            "SELECT bps_norm FROM proteins WHERE organism=? AND plddt_mean >= ?",
            (org, plddt_thr)).fetchall()]
        k = KINGDOM_MAP.get(org, "?")
        color = KINGDOM_COLORS.get(k, "#999")
        ax.hist(vals, bins=bins, alpha=0.3, color=color, density=True, label=None)

    # Kingdom legend
    handles = [Patch(facecolor=c, alpha=0.5, label=k)
               for k, c in KINGDOM_COLORS.items()
               if any(KINGDOM_MAP.get(o) == k for o in orgs)]
    ax.legend(handles=handles, loc='upper right')

    ax.set_xlabel("BPS/res", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"BPS/res Distributions Across Proteomes (pLDDT >= {plddt_thr})")
    ax.axvline(0.18, color='black', linestyle='--', alpha=0.5, label='~0.18')

    path = FIG_DIR / f"bps_distributions.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# 2. CROSS-PROTEOME BARPLOT
# ============================================================

def fig_barplot(conn, plddt_thr, fmt):
    """BPS/res per organism with error bars, colored by kingdom."""
    rows = conn.execute("""
        SELECT organism, COUNT(*), AVG(bps_norm),
               AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm)
        FROM proteins WHERE plddt_mean >= ?
        GROUP BY organism HAVING COUNT(*) >= 100
        ORDER BY AVG(bps_norm) DESC
    """, (plddt_thr,)).fetchall()

    if not rows:
        return

    orgs = [r[0] for r in rows]
    means = [r[2] for r in rows]
    sds = [math.sqrt(max(r[3] or 0, 0)) for r in rows]
    colors = [KINGDOM_COLORS.get(KINGDOM_MAP.get(o, "?"), "#999") for o in orgs]

    fig, ax = plt.subplots(figsize=(max(12, len(orgs) * 0.5), 6))
    x = np.arange(len(orgs))
    bars = ax.bar(x, means, yerr=sds, color=colors, alpha=0.8,
                  capsize=3, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(orgs, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("BPS/res", fontsize=12)
    ax.set_title(f"BPS/res Across Proteomes (pLDDT >= {plddt_thr})")
    ax.axhline(np.mean(means), color='black', linestyle='--', alpha=0.5)

    handles = [Patch(facecolor=c, label=k) for k, c in KINGDOM_COLORS.items()
               if any(KINGDOM_MAP.get(o) == k for o in orgs)]
    ax.legend(handles=handles, loc='upper right')

    path = FIG_DIR / f"bps_barplot.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# 3. BPS vs pLDDT SCATTER
# ============================================================

def fig_bps_plddt(conn, plddt_thr, fmt):
    """BPS/res vs pLDDT scatter, colored by kingdom."""
    fig, ax = plt.subplots(figsize=(8, 6))

    orgs = [r[0] for r in conn.execute("""
        SELECT organism FROM proteins WHERE plddt_mean IS NOT NULL
        GROUP BY organism HAVING COUNT(*) >= 100 ORDER BY organism
    """).fetchall()]

    for org in orgs:
        rows = conn.execute("""
            SELECT bps_norm, plddt_mean FROM proteins
            WHERE organism=? AND plddt_mean IS NOT NULL
        """, (org,)).fetchall()
        bps = [r[0] for r in rows]
        plddt = [r[1] for r in rows]
        k = KINGDOM_MAP.get(org, "?")
        color = KINGDOM_COLORS.get(k, "#999")
        ax.scatter(plddt, bps, alpha=0.05, s=2, color=color, rasterized=True)

    ax.set_xlabel("pLDDT (AlphaFold confidence)", fontsize=12)
    ax.set_ylabel("BPS/res", fontsize=12)
    ax.set_title("BPS/res vs AlphaFold Confidence")

    handles = [Patch(facecolor=c, alpha=0.7, label=k)
               for k, c in KINGDOM_COLORS.items()
               if any(KINGDOM_MAP.get(o) == k for o in orgs)]
    ax.legend(handles=handles, loc='upper left')

    path = FIG_DIR / f"bps_vs_plddt.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# 4. BPS vs LENGTH CURVES
# ============================================================

def fig_bps_vs_length(conn, plddt_thr, fmt):
    """BPS/res vs L in bins, all organisms overlaid."""
    fig, ax = plt.subplots(figsize=(10, 6))

    orgs = [r[0] for r in conn.execute("""
        SELECT organism FROM proteins WHERE plddt_mean >= ?
        GROUP BY organism HAVING COUNT(*) >= 200 ORDER BY organism
    """, (plddt_thr,)).fetchall()]

    bin_centers = list(range(75, 1525, 50))

    for org in orgs:
        xs, ys = [], []
        for center in bin_centers:
            lo, hi = center - 25, center + 25
            r = conn.execute("""
                SELECT COUNT(*), AVG(bps_norm) FROM proteins
                WHERE organism=? AND L >= ? AND L < ? AND plddt_mean >= ?
            """, (org, lo, hi, plddt_thr)).fetchone()
            if r[0] and r[0] >= 5:
                xs.append(center)
                ys.append(r[1])

        k = KINGDOM_MAP.get(org, "?")
        color = KINGDOM_COLORS.get(k, "#999")
        ax.plot(xs, ys, '-o', color=color, alpha=0.5, markersize=3, linewidth=1)

    ax.set_xlabel("Protein Length (residues)", fontsize=12)
    ax.set_ylabel("BPS/res", fontsize=12)
    ax.set_title(f"BPS/res vs Protein Length (pLDDT >= {plddt_thr})")

    handles = [Patch(facecolor=c, alpha=0.7, label=k)
               for k, c in KINGDOM_COLORS.items()
               if any(KINGDOM_MAP.get(o) == k for o in orgs)]
    ax.legend(handles=handles)

    path = FIG_DIR / f"bps_vs_length.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# 5. CORRELATION MATRIX HEATMAP
# ============================================================

def fig_correlation_matrix(conn, plddt_thr, fmt):
    """Heatmap of correlations between all 8 variables."""
    rows = conn.execute("""
        SELECT bps_norm, L, plddt_mean, pct_helix, pct_sheet,
               rg, contact_order, n_transitions
        FROM proteins
        WHERE plddt_mean IS NOT NULL AND contact_order IS NOT NULL
          AND rg IS NOT NULL AND plddt_mean >= ?
    """, (plddt_thr,)).fetchall()

    if len(rows) < 100:
        print("  Skipping correlation matrix: insufficient data")
        return

    data = np.array(rows)
    names = ["BPS/res", "L", "pLDDT", "%helix", "%sheet", "Rg", "CO", "trans"]

    n_vars = len(names)
    corr = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            from scipy.stats import pearsonr
            corr[i, j], _ = pearsonr(data[:, i], data[:, j])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n_vars))
    ax.set_yticks(range(n_vars))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)

    # Add correlation values
    for i in range(n_vars):
        for j in range(n_vars):
            color = 'white' if abs(corr[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{corr[i,j]:.2f}", ha='center', va='center',
                    color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Cross-Variable Correlation Matrix (pLDDT >= {plddt_thr})")

    path = FIG_DIR / f"correlation_matrix.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# 6. FOLD-CLASS HEATMAP
# ============================================================

def fig_fold_class_heatmap(conn, plddt_thr, fmt):
    """%helix vs %sheet colored by mean BPS/res."""
    helix_edges = np.arange(0, 105, 5)
    sheet_edges = np.arange(0, 65, 5)

    grid = np.full((len(helix_edges) - 1, len(sheet_edges) - 1), np.nan)

    for hi in range(len(helix_edges) - 1):
        for si in range(len(sheet_edges) - 1):
            r = conn.execute("""
                SELECT COUNT(*), AVG(bps_norm) FROM proteins
                WHERE pct_helix >= ? AND pct_helix < ?
                  AND pct_sheet >= ? AND pct_sheet < ?
                  AND plddt_mean >= ? AND L >= 30
            """, (helix_edges[hi], helix_edges[hi+1],
                  sheet_edges[si], sheet_edges[si+1],
                  plddt_thr)).fetchone()
            if r[0] and r[0] >= 5:
                grid[hi, si] = r[1]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid, origin='lower', aspect='auto',
                   extent=[sheet_edges[0], sheet_edges[-1],
                           helix_edges[0], helix_edges[-1]],
                   cmap='viridis', vmin=0.08, vmax=0.25)

    ax.set_xlabel("% Sheet", fontsize=12)
    ax.set_ylabel("% Helix", fontsize=12)
    ax.set_title(f"BPS/res by Fold Class (pLDDT >= {plddt_thr})")
    fig.colorbar(im, ax=ax, label="Mean BPS/res")

    path = FIG_DIR / f"fold_class_heatmap.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# 7. PLAXCO SCATTER (if data available)
# ============================================================

def fig_plaxco_scatter(conn, plddt_thr, fmt):
    """BPS vs ln(kf) for Plaxco proteins."""
    from bps_plaxco import PLAXCO_PROTEINS, load_plaxco_from_db

    results = load_plaxco_from_db(conn)
    if len(results) < 5:
        print("  Skipping Plaxco scatter: fewer than 5 proteins in DB")
        return

    ln_kf = np.array([r['ln_kf'] for r in results])
    bps = np.array([r['bps_energy'] for r in results])
    co = np.array([r['co_exp'] for r in results])
    names = [r['name'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # BPS vs ln(kf)
    ax = axes[0]
    ax.scatter(bps, ln_kf, s=50, color='#e74c3c', zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (bps[i], ln_kf[i]), fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')
    sl = np.polyfit(bps, ln_kf, 1)
    x_fit = np.linspace(bps.min(), bps.max(), 100)
    ax.plot(x_fit, np.polyval(sl, x_fit), '--', color='gray')
    from scipy.stats import pearsonr
    r, p = pearsonr(bps, ln_kf)
    ax.set_xlabel("BPS Energy", fontsize=12)
    ax.set_ylabel("ln(kf)", fontsize=12)
    ax.set_title(f"BPS vs Folding Rate (r={r:.3f}, p={p:.2e})")

    # CO vs ln(kf)
    ax = axes[1]
    ax.scatter(co, ln_kf, s=50, color='#3498db', zorder=3)
    for i, name in enumerate(names):
        ax.annotate(name, (co[i], ln_kf[i]), fontsize=7, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')
    sl = np.polyfit(co, ln_kf, 1)
    x_fit = np.linspace(co.min(), co.max(), 100)
    ax.plot(x_fit, np.polyval(sl, x_fit), '--', color='gray')
    r, p = pearsonr(co, ln_kf)
    ax.set_xlabel("Contact Order (experimental)", fontsize=12)
    ax.set_ylabel("ln(kf)", fontsize=12)
    ax.set_title(f"CO vs Folding Rate (r={r:.3f}, p={p:.2e})")

    fig.tight_layout()
    path = FIG_DIR / f"plaxco_scatter.{fmt}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not HAS_MPL:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        return

    parser = argparse.ArgumentParser(description="BPS Figures — publication plots")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--plddt-threshold", type=float, default=0)
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"])
    parser.add_argument("--figure", type=str, default="all",
                        choices=["all", "distribution", "barplot", "bps-plddt",
                                 "bps-length", "correlation", "fold-class", "plaxco"])
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    plddt = args.plddt_threshold
    fmt = args.format

    figures = {
        "distribution": fig_distribution,
        "barplot": fig_barplot,
        "bps-plddt": fig_bps_plddt,
        "bps-length": fig_bps_vs_length,
        "correlation": fig_correlation_matrix,
        "fold-class": fig_fold_class_heatmap,
        "plaxco": fig_plaxco_scatter,
    }

    try:
        if args.figure == "all":
            for name, fn in figures.items():
                try:
                    fn(conn, plddt, fmt)
                except Exception as e:
                    print(f"  Error in {name}: {e}")
        else:
            figures[args.figure](conn, plddt, fmt)
    finally:
        conn.close()

    print(f"\nFigures saved to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
