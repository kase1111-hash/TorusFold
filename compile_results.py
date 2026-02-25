#!/usr/bin/env python3
"""
BPS RESULTS COMPILER v3
========================
Reads bps_database.db (from bps_process.py), produces one text file
optimized for pasting to Claude for analysis.

Key addition: pLDDT-controlled comparisons to separate real biology
from AlphaFold model quality artifacts.

Usage:
  python compile_results.py
  python compile_results.py --db path/to/bps_database.db
  python compile_results.py --out my_results.txt
  python compile_results.py --mode watch          # continuously update report
  python compile_results.py --mode watch --poll 45

Concurrent operation:
  Safe to run alongside bps_download.py and bps_process.py.
  Uses SQLite WAL mode + busy_timeout for concurrent reads.
  Report file is written atomically (temp + rename).
"""

import sys
import time
import logging
import argparse
import sqlite3
import math
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import pearsonr, linregress, ttest_ind, mannwhitneyu, wasserstein_distance

# ============================================================
# CONFIG
# ============================================================

DB_PATH = Path("./alphafold_bps_results/bps_database.db")
OUTPUT_FILE = Path("./bps_results_for_claude.txt")

# ============================================================
# TAXONOMY
# ============================================================

BACTERIA = {"ecoli", "bacillus", "tuberculosis", "staph", "pseudomonas",
            "salmonella", "campylobacter", "helicobacter"}

METAZOA = {"human", "mouse", "rat", "fly", "worm", "zebrafish", "chicken",
           "pig", "cow", "dog", "cat", "gorilla", "chimp", "mosquito", "honeybee"}

PLANTS = {"arabidopsis", "rice", "soybean", "maize"}

FUNGI = {"yeast", "fission_yeast", "candida"}

PROTISTS = {"malaria", "leishmania", "trypanosoma", "dictyostelium"}

EUKARYOTES = METAZOA | PLANTS | FUNGI | PROTISTS

def kingdom(org):
    if org in BACTERIA: return "bact"
    if org in METAZOA: return "metaz"
    if org in PLANTS: return "plant"
    if org in FUNGI: return "fungi"
    if org in PROTISTS: return "prot"
    return "?"

# ============================================================
# HELPERS
# ============================================================

def S(title):
    return f"\n{'='*72}\n  {title}\n{'='*72}\n"

def safe_pearsonr(x, y):
    """Pearson r that returns (None, None) if inputs are degenerate."""
    if len(x) < 5:
        return None, None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None, None
    return pearsonr(x, y)


# ============================================================
# SECTION 1: DATABASE OVERVIEW
# ============================================================

def sec_overview(conn, out):
    out.append(S("1. DATABASE OVERVIEW"))

    total = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
    n_fail = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    n_orgs = conn.execute("SELECT COUNT(DISTINCT organism) FROM proteins").fetchone()[0]

    out.append(f"Proteins:  {total:,}")
    out.append(f"Failures:  {n_fail:,}")
    out.append(f"Organisms: {n_orgs}")

    rows = conn.execute(
        "SELECT error_type, COUNT(*) FROM failures GROUP BY error_type ORDER BY COUNT(*) DESC"
    ).fetchall()
    if rows:
        out.append(f"\nFailure types:")
        for etype, count in rows:
            out.append(f"  {etype:<10} {count:>7,}")

    row = conn.execute("""
        SELECT AVG(bps_norm), AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm),
               MIN(bps_norm), MAX(bps_norm),
               AVG(plddt_mean), AVG(pct_helix), AVG(pct_sheet), AVG(contact_order)
        FROM proteins
    """).fetchone()
    if row[0]:
        sd = math.sqrt(max(row[1] or 0, 0))
        out.append(f"\nGlobal BPS/res: {row[0]:.4f} +/- {sd:.4f}  (range {row[2]:.4f} - {row[3]:.4f})")
        if row[4]: out.append(f"Global pLDDT:   {row[4]:.1f}")
        if row[5] is not None: out.append(f"Global SS:      {row[5]:.1f}% helix, {row[6]:.1f}% sheet")
        if row[7]: out.append(f"Global CO:      {row[7]:.4f}")

    out.append("")


# ============================================================
# SECTION 2: ORGANISM TABLE
# ============================================================

def sec_organism_table(conn, out):
    out.append(S("2. ORGANISM FINGERPRINTS"))

    rows = conn.execute("""
        SELECT organism, COUNT(*) as n,
               AVG(bps_norm), AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm),
               AVG(L), MIN(L), MAX(L),
               AVG(n_transitions),
               AVG(plddt_mean), AVG(plddt_below50),
               AVG(pct_helix), AVG(pct_sheet), AVG(pct_coil),
               AVG(rg), AVG(contact_order)
        FROM proteins GROUP BY organism HAVING n >= 10
        ORDER BY AVG(bps_norm) DESC
    """).fetchall()

    h = (f"{'Organism':<18} {'Kind':<6} {'N':>6} {'BPS/res':>8} {'SD':>7} {'<L>':>5} "
         f"{'trans':>6} {'pLDDT':>6} {'%<50':>6} {'%a':>6} {'%b':>6} {'%coil':>6} "
         f"{'Rg':>6} {'CO':>7}")
    out.append(h)
    out.append("-" * len(h))

    for org, n, mb, vb, ml, minl, maxl, mt, mp, mb50, mh, ms, mc, mrg, mco in rows:
        sd = math.sqrt(max(vb or 0, 0))
        k = kingdom(org)
        p_s = f"{mp:>6.1f}" if mp else f"{'--':>6}"
        b50_s = f"{mb50*100:>5.1f}%" if mb50 is not None else f"{'--':>6}"
        h_s = f"{mh:>5.1f}%" if mh is not None else f"{'--':>6}"
        s_s = f"{ms:>5.1f}%" if ms is not None else f"{'--':>6}"
        c_s = f"{mc:>5.1f}%" if mc is not None else f"{'--':>6}"
        rg_s = f"{mrg:>6.1f}" if mrg else f"{'--':>6}"
        co_s = f"{mco:>7.4f}" if mco else f"{'--':>7}"
        out.append(f"{org:<18} {k:<6} {n:>6} {mb:>8.4f} {sd:>7.4f} {ml:>5.0f} "
                   f"{mt:>6.1f} {p_s} {b50_s} {h_s} {s_s} {c_s} {rg_s} {co_s}")
    out.append("")


# ============================================================
# SECTION 3: SIZE-CLASS GRADIENT
# ============================================================

def sec_size_gradient(conn, out):
    out.append(S("3. SIZE-CLASS BPS GRADIENT"))

    orgs = [r[0] for r in conn.execute(
        "SELECT DISTINCT organism FROM proteins ORDER BY organism").fetchall()]

    bins = [("S<100", 0, 100), ("M100-300", 100, 300),
            ("L300-1K", 300, 1000), ("XL>1K", 1000, 999999)]

    h = f"{'Organism':<18}"
    for label, _, _ in bins:
        h += f"  {label:>16}"
    out.append(h)
    out.append("-" * len(h))

    for org in orgs:
        row = f"{org:<18}"
        for _, lo, hi in bins:
            r = conn.execute("""
                SELECT COUNT(*), AVG(bps_norm)
                FROM proteins WHERE organism=? AND L>=? AND L<?
            """, (org, lo, hi)).fetchone()
            if r[0] and r[0] >= 5:
                row += f"  {r[1]:>8.4f} n={r[0]:>4}"
            else:
                row += f"  {'--':>16}"
        out.append(row)
    out.append("")


# ============================================================
# SECTION 4: BPS vs LENGTH REGRESSION
# ============================================================

def sec_bps_vs_length(conn, out):
    out.append(S("4. BPS vs LENGTH (linear regression per organism)"))

    orgs = [r[0] for r in conn.execute(
        "SELECT organism FROM proteins GROUP BY organism HAVING COUNT(*)>=50 ORDER BY organism"
    ).fetchall()]

    h = f"{'Organism':<18} {'N':>6} {'slope':>10} {'intercept':>10} {'r':>8} {'r2':>8}"
    out.append(h)
    out.append("-" * len(h))

    for org in orgs:
        data = conn.execute(
            "SELECT L, bps_energy FROM proteins WHERE organism=?", (org,)).fetchall()
        L = np.array([d[0] for d in data])
        BPS = np.array([d[1] for d in data])
        if np.std(L) < 1e-10 or np.std(BPS) < 1e-10:
            out.append(f"{org:<18} {len(L):>6} {'—':>10} {'—':>10} {'—':>8} {'—':>8}")
            continue
        sl = linregress(L, BPS)
        out.append(f"{org:<18} {len(L):>6} {sl.slope:>10.5f} {sl.intercept:>10.3f} "
                   f"{sl.rvalue:>8.4f} {sl.rvalue**2:>8.4f}")
    out.append("")


# ============================================================
# SECTION 5: KINGDOM COMPARISON
# ============================================================

def sec_kingdom(conn, out):
    out.append(S("5. KINGDOM COMPARISON"))

    groups = [("Bacteria", BACTERIA), ("Metazoa", METAZOA), ("Fungi", FUNGI),
              ("Plants", PLANTS), ("Protists", PROTISTS)]

    all_means = {}

    for gname, gset in groups:
        out.append(f"\n{gname}:")
        h = f"  {'Organism':<18} {'N':>6} {'BPS/res':>8} {'pLDDT':>6} {'%a':>6} {'%b':>6} {'CO':>7}"
        out.append(h)
        for org in sorted(gset):
            r = conn.execute("""
                SELECT COUNT(*), AVG(bps_norm), AVG(plddt_mean),
                       AVG(pct_helix), AVG(pct_sheet), AVG(contact_order)
                FROM proteins WHERE organism=? HAVING COUNT(*)>=10
            """, (org,)).fetchone()
            if r and r[0]:
                all_means[org] = r[1:]
                p_s = f"{r[2]:.1f}" if r[2] else "--"
                h_s = f"{r[3]:.1f}%" if r[3] is not None else "--"
                s_s = f"{r[4]:.1f}%" if r[4] is not None else "--"
                co_s = f"{r[5]:.4f}" if r[5] else "--"
                out.append(f"  {org:<18} {r[0]:>6} {r[1]:>8.4f} {p_s:>6} "
                           f"{h_s:>6} {s_s:>6} {co_s:>7}")

    bact_bps = [all_means[o][0] for o in BACTERIA if o in all_means]
    euk_bps = [all_means[o][0] for o in EUKARYOTES if o in all_means]

    if bact_bps and euk_bps:
        out.append(f"\nBacteria vs Eukaryote (organism-level means):")
        out.append(f"  Bacteria:  {np.mean(bact_bps):.4f} +/- {np.std(bact_bps):.4f} (n={len(bact_bps)})")
        out.append(f"  Eukaryote: {np.mean(euk_bps):.4f} +/- {np.std(euk_bps):.4f} (n={len(euk_bps)})")
        if len(bact_bps) >= 2 and len(euk_bps) >= 2:
            t, p_t = ttest_ind(bact_bps, euk_bps)
            u, p_u = mannwhitneyu(bact_bps, euk_bps, alternative='two-sided')
            sep = (np.mean(bact_bps) - np.mean(euk_bps)) / np.mean(bact_bps) * 100
            out.append(f"  Separation: {sep:.1f}%")
            out.append(f"  t-test: t={t:.3f}, p={p_t:.2e}")
            out.append(f"  Mann-Whitney: U={u:.0f}, p={p_u:.2e}")
    out.append("")


# ============================================================
# SECTION 6: pLDDT vs BPS — THE UNIVERSALITY TEST
# ============================================================

def sec_plddt_vs_bps(conn, out):
    out.append(S("6. pLDDT vs BPS (universality test)"))
    out.append("Does BPS/res depend on AlphaFold confidence?\n")

    orgs = [r[0] for r in conn.execute(
        "SELECT organism FROM proteins WHERE plddt_mean IS NOT NULL "
        "GROUP BY organism HAVING COUNT(*)>=50 ORDER BY organism"
    ).fetchall()]

    h = f"{'Organism':<18} {'N':>6} {'pLDDT':>6} {'%<50':>6} {'r(BPS,pLDDT)':>13} {'p':>12}"
    out.append(h)
    out.append("-" * len(h))

    global_bps, global_plddt = [], []
    for org in orgs:
        rows = conn.execute("""
            SELECT bps_norm, plddt_mean, plddt_below50
            FROM proteins WHERE organism=? AND plddt_mean IS NOT NULL
        """, (org,)).fetchall()
        bps = np.array([r[0] for r in rows])
        plddt = np.array([r[1] for r in rows])
        below50 = np.mean([r[2] for r in rows if r[2] is not None])
        r_val, p_val = safe_pearsonr(bps, plddt)
        r_s = f"{r_val:>13.4f}" if r_val is not None else f"{'--':>13}"
        p_s = f"{p_val:>12.2e}" if p_val is not None else f"{'--':>12}"
        out.append(f"{org:<18} {len(rows):>6} {np.mean(plddt):>6.1f} "
                   f"{below50*100:>5.1f}% {r_s} {p_s}")
        global_bps.extend(bps.tolist())
        global_plddt.extend(plddt.tolist())

    if len(global_bps) > 100:
        r_g, p_g = pearsonr(global_bps, global_plddt)
        out.append(f"\nGLOBAL: r(BPS/res, pLDDT) = {r_g:.4f}  (p={p_g:.2e}, N={len(global_bps):,})")
        strength = 'WEAK/NO' if abs(r_g) < 0.15 else 'MODERATE' if abs(r_g) < 0.4 else 'STRONG'
        out.append(f"  -> {strength} dependence on AlphaFold confidence")

    out.append(f"\nBPS/res by pLDDT tier (all organisms pooled):")
    tiers = [(0, 50, "< 50"), (50, 70, "50-70"), (70, 90, "70-90"), (90, 101, "> 90")]
    h2 = f"  {'pLDDT tier':<12} {'N':>8} {'BPS/res':>9} {'SD':>7}"
    out.append(h2)
    for lo, hi, label in tiers:
        r = conn.execute("""
            SELECT COUNT(*), AVG(bps_norm),
                   AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm)
            FROM proteins WHERE plddt_mean >= ? AND plddt_mean < ?
        """, (lo, hi)).fetchone()
        if r[0] and r[0] > 0:
            sd = math.sqrt(max(r[2] or 0, 0))
            out.append(f"  {label:<12} {r[0]:>8,} {r[1]:>9.4f} {sd:>7.4f}")
    out.append("")


# ============================================================
# SECTION 7: SECONDARY STRUCTURE vs BPS
# ============================================================

def sec_ss_vs_bps(conn, out):
    out.append(S("7. SECONDARY STRUCTURE vs BPS"))
    out.append("Does fold class affect BPS/res?\n")

    orgs = [r[0] for r in conn.execute(
        "SELECT organism FROM proteins WHERE pct_helix IS NOT NULL "
        "GROUP BY organism HAVING COUNT(*)>=50 ORDER BY organism"
    ).fetchall()]

    h = f"{'Organism':<18} {'N':>6} {'%a':>6} {'%b':>6} {'%coil':>6} {'r(BPS,a)':>10} {'r(BPS,b)':>10}"
    out.append(h)
    out.append("-" * len(h))

    for org in orgs:
        rows = conn.execute("""
            SELECT bps_norm, pct_helix, pct_sheet, pct_coil
            FROM proteins WHERE organism=? AND pct_helix IS NOT NULL
        """, (org,)).fetchall()
        bps = np.array([r[0] for r in rows])
        helix = np.array([r[1] for r in rows])
        sheet = np.array([r[2] for r in rows])
        r_h, _ = safe_pearsonr(bps, helix) or (None, None)
        r_s, _ = safe_pearsonr(bps, sheet) or (None, None)
        rh_s = f"{r_h:>10.4f}" if r_h is not None else f"{'--':>10}"
        rs_s = f"{r_s:>10.4f}" if r_s is not None else f"{'--':>10}"
        out.append(f"{org:<18} {len(rows):>6} {np.mean(helix):>5.1f}% {np.mean(sheet):>5.1f}% "
                   f"{np.mean([r[3] for r in rows]):>5.1f}% {rh_s} {rs_s}")

    out.append(f"\nBPS/res by dominant fold class (all organisms):")
    classes = [
        ("Alpha-rich (>40% helix)", "pct_helix > ? AND pct_sheet < ?", (40, 15)),
        ("Beta-rich (>25% sheet)", "pct_sheet > ? AND pct_helix < ?", (25, 15)),
        ("Mixed (both >15%)", "pct_helix > ? AND pct_sheet > ?", (15, 15)),
        ("Coil-dominant (>80% coil)", "pct_coil > ?", (80,)),
    ]
    h2 = f"  {'Class':<30} {'N':>8} {'BPS/res':>9} {'SD':>7}"
    out.append(h2)
    for label, where, params in classes:
        r = conn.execute(f"""
            SELECT COUNT(*), AVG(bps_norm),
                   AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm)
            FROM proteins WHERE pct_helix IS NOT NULL AND {where}
        """, params).fetchone()
        if r[0] and r[0] > 10:
            sd = math.sqrt(max(r[2] or 0, 0))
            out.append(f"  {label:<30} {r[0]:>8,} {r[1]:>9.4f} {sd:>7.4f}")
    out.append("")


# ============================================================
# SECTION 8: CONTACT ORDER vs BPS
# ============================================================

def sec_co_vs_bps(conn, out):
    out.append(S("8. CONTACT ORDER vs BPS (independence test)"))

    orgs = [r[0] for r in conn.execute(
        "SELECT organism FROM proteins WHERE contact_order IS NOT NULL "
        "GROUP BY organism HAVING COUNT(*)>=50 ORDER BY organism"
    ).fetchall()]

    h = f"{'Organism':<18} {'N':>6} {'<CO>':>8} {'r(BPS,CO)':>11} {'p':>12} {'r(CO,L)':>9}"
    out.append(h)
    out.append("-" * len(h))

    global_bps, global_co = [], []
    for org in orgs:
        rows = conn.execute("""
            SELECT bps_norm, contact_order, L
            FROM proteins WHERE organism=? AND contact_order IS NOT NULL
        """, (org,)).fetchall()
        bps = np.array([r[0] for r in rows])
        co = np.array([r[1] for r in rows])
        L = np.array([r[2] for r in rows])
        r_bc, p_bc = safe_pearsonr(bps, co) or (None, None)
        r_cl, _ = safe_pearsonr(co, L) or (None, None)
        rbc_s = f"{r_bc:>11.4f}" if r_bc is not None else f"{'--':>11}"
        pbc_s = f"{p_bc:>12.2e}" if p_bc is not None else f"{'--':>12}"
        rcl_s = f"{r_cl:>9.4f}" if r_cl is not None else f"{'--':>9}"
        out.append(f"{org:<18} {len(rows):>6} {np.mean(co):>8.4f} {rbc_s} {pbc_s} {rcl_s}")
        global_bps.extend(bps.tolist())
        global_co.extend(co.tolist())

    if len(global_bps) > 100:
        r_g, p_g = pearsonr(global_bps, global_co)
        out.append(f"\nGLOBAL: r(BPS/res, CO) = {r_g:.4f}  (p={p_g:.2e}, N={len(global_bps):,})")
        rel = 'INDEPENDENT' if abs(r_g) < 0.15 else 'WEAKLY RELATED' if abs(r_g) < 0.4 else 'CORRELATED'
        out.append(f"  -> {rel}")
    out.append("")


# ============================================================
# SECTION 9: Rg vs BPS
# ============================================================

def sec_rg_vs_bps(conn, out):
    out.append(S("9. RADIUS OF GYRATION vs BPS"))

    orgs = [r[0] for r in conn.execute(
        "SELECT organism FROM proteins WHERE rg IS NOT NULL "
        "GROUP BY organism HAVING COUNT(*)>=50 ORDER BY organism"
    ).fetchall()]

    h = f"{'Organism':<18} {'N':>6} {'<Rg>':>7} {'r(BPS,Rg)':>11} {'r(Rg/sqrtL,BPS)':>16}"
    out.append(h)
    out.append("-" * len(h))

    for org in orgs:
        rows = conn.execute("""
            SELECT bps_norm, rg, L FROM proteins WHERE organism=? AND rg IS NOT NULL
        """, (org,)).fetchall()
        bps = np.array([r[0] for r in rows])
        rg = np.array([r[1] for r in rows])
        L = np.array([r[2] for r in rows])
        r_br, _ = safe_pearsonr(bps, rg) or (None, None)
        rg_norm = rg / np.sqrt(L)
        r_bn, _ = safe_pearsonr(bps, rg_norm) or (None, None)
        rbr_s = f"{r_br:>11.4f}" if r_br is not None else f"{'--':>11}"
        rbn_s = f"{r_bn:>16.4f}" if r_bn is not None else f"{'--':>16}"
        out.append(f"{org:<18} {len(rows):>6} {np.mean(rg):>7.1f} {rbr_s} {rbn_s}")
    out.append("")


# ============================================================
# SECTION 10: DISTRIBUTION DISTANCES
# ============================================================

def sec_distances(conn, out):
    out.append(S("10. DISTRIBUTION DISTANCES (Wasserstein)"))

    orgs = [r[0] for r in conn.execute(
        "SELECT organism FROM proteins GROUP BY organism HAVING COUNT(*)>=50 ORDER BY organism"
    ).fetchall()]

    dists = {}
    for org in orgs:
        rows = conn.execute("SELECT bps_norm FROM proteins WHERE organism=?", (org,)).fetchall()
        dists[org] = [r[0] for r in rows]

    pairs = []
    for i, o1 in enumerate(orgs):
        for j, o2 in enumerate(orgs):
            if i < j:
                d = wasserstein_distance(dists[o1], dists[o2])
                pairs.append((o1, o2, d))
    pairs.sort(key=lambda x: x[2])

    h = f"{'Organism 1':<18} {'Organism 2':<18} {'Distance':>10}"
    for label, subset in [("CLOSEST 15", pairs[:15]), ("MOST DISTANT 15", pairs[-15:])]:
        out.append(f"\n{label}:")
        out.append(h)
        out.append("-" * len(h))
        for o1, o2, d in subset:
            out.append(f"{o1:<18} {o2:<18} {d:>10.6f}")
    out.append("")


# ============================================================
# SECTION 11: FAILURE ANALYSIS
# ============================================================

def sec_failures(conn, out):
    out.append(S("11. FAILURE BREAKDOWN BY ORGANISM"))

    rows = conn.execute("""
        SELECT t.organism, COALESCE(p.n, 0), COALESCE(f.n, 0),
               COALESCE(f.n_404, 0), COALESCE(f.n_parse, 0)
        FROM (SELECT DISTINCT organism FROM proteins
              UNION SELECT DISTINCT organism FROM failures) t
        LEFT JOIN (SELECT organism, COUNT(*) as n FROM proteins GROUP BY organism) p
            ON t.organism = p.organism
        LEFT JOIN (SELECT organism, COUNT(*) as n,
                   SUM(CASE WHEN error_type='404' THEN 1 ELSE 0 END) as n_404,
                   SUM(CASE WHEN error_type='parse' THEN 1 ELSE 0 END) as n_parse
                   FROM failures GROUP BY organism) f
            ON t.organism = f.organism
        ORDER BY t.organism
    """).fetchall()

    h = f"{'Organism':<18} {'OK':>7} {'Fail':>7} {'%Fail':>7} {'404':>7} {'Parse':>7}"
    out.append(h)
    out.append("-" * len(h))

    for org, n_ok, n_fail, n_404, n_parse in rows:
        total = n_ok + n_fail
        pct_fail = n_fail / total * 100 if total > 0 else 0
        out.append(f"{org:<18} {n_ok:>7} {n_fail:>7} {pct_fail:>6.1f}% "
                   f"{n_404:>7} {n_parse:>7}")

    flagged = [(org, n_ok, n_fail) for org, n_ok, n_fail, _, n_parse in rows
               if (n_ok + n_fail) > 0 and n_fail / (n_ok + n_fail) > 0.10]
    if flagged:
        out.append(f"\nWARNING - High failure rate (>10%):")
        for org, n_ok, n_fail in flagged:
            out.append(f"  {org}: {n_fail/(n_ok+n_fail)*100:.0f}%")
    out.append("")


# ============================================================
# SECTION 12: BPS HISTOGRAMS
# ============================================================

def sec_histograms(conn, out):
    out.append(S("12. BPS/res DISTRIBUTIONS (12-bin histograms)"))

    orgs = [r[0] for r in conn.execute(
        "SELECT organism FROM proteins GROUP BY organism HAVING COUNT(*)>=50 ORDER BY organism"
    ).fetchall()]

    edges = np.linspace(0.05, 0.35, 13)
    labels = [f"{edges[i]:.2f}" for i in range(len(edges)-1)]

    h = f"{'Organism':<16} " + " ".join(f"{l:>6}" for l in labels)
    out.append(h)
    out.append("-" * len(h))

    for org in orgs:
        vals = [r[0] for r in conn.execute(
            "SELECT bps_norm FROM proteins WHERE organism=?", (org,)).fetchall()]
        counts, _ = np.histogram(vals, bins=edges)
        pcts = counts / len(vals) * 100
        out.append(f"{org:<16} " + " ".join(f"{p:>5.1f}%" for p in pcts))
    out.append("")


# ============================================================
# SECTION 13: CROSS-VARIABLE CORRELATION MATRIX
# ============================================================

def sec_correlation_matrix(conn, out):
    out.append(S("13. CROSS-VARIABLE CORRELATION MATRIX (global)"))
    out.append("Pearson r across all proteins with complete data.\n")

    rows = conn.execute("""
        SELECT bps_norm, L, plddt_mean, pct_helix, pct_sheet, rg, contact_order, n_transitions
        FROM proteins
        WHERE plddt_mean IS NOT NULL AND contact_order IS NOT NULL AND rg IS NOT NULL
    """).fetchall()

    if len(rows) < 100:
        out.append("  [Insufficient data with all fields populated]")
        return

    names = ["BPS/res", "L", "pLDDT", "%helix", "%sheet", "Rg", "CO", "trans"]
    data = np.array(rows)

    h = f"{'':>10}" + "".join(f"{n:>10}" for n in names)
    out.append(h)
    out.append("-" * len(h))

    for i, name_i in enumerate(names):
        row = f"{name_i:>10}"
        for j in range(len(names)):
            if i == j:
                row += f"{'1.000':>10}"
            elif j > i:
                r, _ = pearsonr(data[:, i], data[:, j])
                row += f"{r:>10.4f}"
            else:
                row += f"{'':>10}"
        out.append(row)

    out.append(f"\n  N = {len(rows):,} proteins with all 8 variables populated")
    out.append("")


# ============================================================
# SECTION 14: pLDDT-CONTROLLED COMPARISON (the key test)
# ============================================================

def sec_plddt_controlled(conn, out):
    out.append(S("14. pLDDT-CONTROLLED COMPARISON"))
    out.append("Repeat all key comparisons using ONLY well-modeled proteins (pLDDT >= 85).")
    out.append("This separates real biology from AlphaFold confidence artifacts.\n")

    PLDDT_CUTOFF = 85

    total_hq = conn.execute(
        "SELECT COUNT(*) FROM proteins WHERE plddt_mean >= ?", (PLDDT_CUTOFF,)
    ).fetchone()[0]
    total_all = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
    out.append(f"High-quality proteins (pLDDT >= {PLDDT_CUTOFF}): {total_hq:,} / {total_all:,} "
               f"({total_hq/total_all*100:.1f}%)\n")

    # 14a: Organism table (high-quality only)
    out.append(f"--- 14a. Organism fingerprints (pLDDT >= {PLDDT_CUTOFF}) ---\n")

    rows = conn.execute(f"""
        SELECT organism, COUNT(*) as n,
               AVG(bps_norm), AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm),
               AVG(L), AVG(plddt_mean),
               AVG(pct_helix), AVG(pct_sheet), AVG(contact_order)
        FROM proteins
        WHERE plddt_mean >= {PLDDT_CUTOFF}
        GROUP BY organism HAVING n >= 10
        ORDER BY AVG(bps_norm) DESC
    """).fetchall()

    h = f"{'Organism':<18} {'Kind':<6} {'N':>6} {'N_hq':>6} {'BPS/res':>8} {'SD':>7} {'<L>':>5} {'pLDDT':>6} {'%a':>6} {'%b':>6} {'CO':>7}"
    out.append(h)
    out.append("-" * len(h))

    hq_means = {}
    for org, n, mb, vb, ml, mp, mh, ms, mco in rows:
        sd = math.sqrt(max(vb or 0, 0))
        # Get total N for this organism
        n_total = conn.execute("SELECT COUNT(*) FROM proteins WHERE organism=?", (org,)).fetchone()[0]
        hq_means[org] = mb
        h_s = f"{mh:>5.1f}%" if mh is not None else f"{'--':>6}"
        s_s = f"{ms:>5.1f}%" if ms is not None else f"{'--':>6}"
        co_s = f"{mco:>7.4f}" if mco else f"{'--':>7}"
        out.append(f"{org:<18} {kingdom(org):<6} {n_total:>6} {n:>6} {mb:>8.4f} {sd:>7.4f} "
                   f"{ml:>5.0f} {mp:>6.1f} {h_s} {s_s} {co_s}")

    # 14b: Kingdom comparison (high-quality only)
    out.append(f"\n--- 14b. Kingdom comparison (pLDDT >= {PLDDT_CUTOFF}) ---\n")

    bact_hq = [hq_means[o] for o in BACTERIA if o in hq_means]
    euk_hq = [hq_means[o] for o in EUKARYOTES if o in hq_means]

    if bact_hq and euk_hq:
        out.append(f"Bacteria:  {np.mean(bact_hq):.4f} +/- {np.std(bact_hq):.4f} (n={len(bact_hq)})")
        out.append(f"Eukaryote: {np.mean(euk_hq):.4f} +/- {np.std(euk_hq):.4f} (n={len(euk_hq)})")
        if len(bact_hq) >= 2 and len(euk_hq) >= 2:
            t, p_t = ttest_ind(bact_hq, euk_hq)
            sep = (np.mean(bact_hq) - np.mean(euk_hq)) / np.mean(bact_hq) * 100
            out.append(f"Separation: {sep:.1f}%")
            out.append(f"t-test: t={t:.3f}, p={p_t:.2e}")

    # Compare with ALL-data separation
    bact_all = [r[0] for r in conn.execute("""
        SELECT AVG(bps_norm) FROM proteins WHERE organism IN
        ('ecoli','bacillus','tuberculosis','staph','pseudomonas','salmonella','campylobacter','helicobacter')
        GROUP BY organism HAVING COUNT(*)>=10
    """).fetchall()]
    euk_all = [r[0] for r in conn.execute("""
        SELECT AVG(bps_norm) FROM proteins WHERE organism NOT IN
        ('ecoli','bacillus','tuberculosis','staph','pseudomonas','salmonella','campylobacter','helicobacter')
        GROUP BY organism HAVING COUNT(*)>=10
    """).fetchall()]

    if bact_all and euk_all and bact_hq and euk_hq:
        sep_all = (np.mean(bact_all) - np.mean(euk_all)) / np.mean(bact_all) * 100
        sep_hq = (np.mean(bact_hq) - np.mean(euk_hq)) / np.mean(bact_hq) * 100
        out.append(f"\nSeparation change: {sep_all:.1f}% (all data) -> {sep_hq:.1f}% (pLDDT>={PLDDT_CUTOFF})")
        if abs(sep_hq) < abs(sep_all) * 0.5:
            out.append(f"  -> COLLAPSED: most of the split was pLDDT-driven")
        elif abs(sep_hq) > abs(sep_all) * 0.7:
            out.append(f"  -> PERSISTS: real biological difference")
        else:
            out.append(f"  -> REDUCED: partially pLDDT-driven")

    # 14c: BPS/res by pLDDT tier within bacteria vs eukaryotes
    out.append(f"\n--- 14c. BPS/res by pLDDT tier, bacteria vs eukaryotes ---\n")

    tiers = [(70, 80), (80, 90), (90, 95), (95, 101)]
    h = f"  {'pLDDT tier':<12} {'Bact N':>8} {'Bact BPS':>10} {'Euk N':>8} {'Euk BPS':>10} {'Delta':>8}"
    out.append(h)

    bact_list = "','".join(BACTERIA)
    for lo, hi in tiers:
        rb = conn.execute(f"""
            SELECT COUNT(*), AVG(bps_norm) FROM proteins
            WHERE organism IN ('{bact_list}') AND plddt_mean >= ? AND plddt_mean < ?
        """, (lo, hi)).fetchone()
        re = conn.execute(f"""
            SELECT COUNT(*), AVG(bps_norm) FROM proteins
            WHERE organism NOT IN ('{bact_list}') AND plddt_mean >= ? AND plddt_mean < ?
        """, (lo, hi)).fetchone()

        if rb[0] and rb[0] > 10 and re[0] and re[0] > 10:
            delta = rb[1] - re[1]
            out.append(f"  {lo}-{hi:<9} {rb[0]:>8,} {rb[1]:>10.4f} {re[0]:>8,} {re[1]:>10.4f} {delta:>+8.4f}")

    out.append(f"\n  If Delta is consistent across tiers -> real biology")
    out.append(f"  If Delta shrinks at high pLDDT -> confounded by model quality")

    # 14d: Correlation matrix (HQ only)
    out.append(f"\n--- 14d. Key correlations (pLDDT >= {PLDDT_CUTOFF}) ---\n")

    hq_rows = conn.execute(f"""
        SELECT bps_norm, contact_order, pct_helix, pct_sheet, rg
        FROM proteins
        WHERE plddt_mean >= {PLDDT_CUTOFF} AND contact_order IS NOT NULL AND rg IS NOT NULL
    """).fetchall()

    if len(hq_rows) > 100:
        data = np.array(hq_rows)
        pairs = [("BPS/res", "CO", 0, 1), ("BPS/res", "%helix", 0, 2),
                 ("BPS/res", "%sheet", 0, 3), ("BPS/res", "Rg", 0, 4)]
        h2 = f"  {'Variable 1':<12} {'Variable 2':<12} {'r (all)':>10} {'r (HQ)':>10} {'N_HQ':>8}"
        out.append(h2)
        # Get all-data correlations for comparison
        all_rows = conn.execute("""
            SELECT bps_norm, contact_order, pct_helix, pct_sheet, rg
            FROM proteins WHERE contact_order IS NOT NULL AND rg IS NOT NULL
        """).fetchall()
        all_data = np.array(all_rows) if len(all_rows) > 100 else None

        for name1, name2, i, j in pairs:
            r_hq, _ = pearsonr(data[:, i], data[:, j])
            r_all_s = "--"
            if all_data is not None:
                r_all, _ = pearsonr(all_data[:, i], all_data[:, j])
                r_all_s = f"{r_all:>10.4f}"
            out.append(f"  {name1:<12} {name2:<12} {r_all_s} {r_hq:>10.4f} {len(data):>8,}")

    out.append("")


# ============================================================
# SECTION 15: AF VERSION DISTRIBUTION
# ============================================================

def sec_af_versions(conn, out):
    out.append(S("15. ALPHAFOLD MODEL VERSIONS"))

    # Check if af_version column exists
    cols = [r[1] for r in conn.execute("PRAGMA table_info(proteins)").fetchall()]
    if 'af_version' not in cols:
        out.append("  [af_version column not present — run with updated processor]")
        out.append("")
        return

    rows = conn.execute("""
        SELECT organism, af_version, COUNT(*), AVG(bps_norm), AVG(plddt_mean)
        FROM proteins
        WHERE af_version IS NOT NULL
        GROUP BY organism, af_version
        ORDER BY organism, af_version
    """).fetchall()

    if not rows:
        out.append("  [No version data recorded]")
        out.append("")
        return

    h = f"{'Organism':<18} {'Version':>8} {'N':>7} {'BPS/res':>9} {'pLDDT':>7}"
    out.append(h)
    out.append("-" * len(h))
    for org, ver, n, mb, mp in rows:
        p_s = f"{mp:.1f}" if mp else "--"
        out.append(f"{org:<18} {'v'+str(ver) if ver else '?':>8} {n:>7} {mb:>9.4f} {p_s:>7}")

    # Global version summary
    global_rows = conn.execute("""
        SELECT af_version, COUNT(*), AVG(bps_norm), AVG(plddt_mean)
        FROM proteins WHERE af_version IS NOT NULL
        GROUP BY af_version ORDER BY af_version
    """).fetchall()
    if global_rows:
        out.append(f"\nGlobal version summary:")
        for ver, n, mb, mp in global_rows:
            p_s = f"{mp:.1f}" if mp else "--"
            out.append(f"  v{ver}: N={n:,}, BPS/res={mb:.4f}, pLDDT={p_s}")

    # Check for mixed versions within organisms
    mixed = conn.execute("""
        SELECT organism, COUNT(DISTINCT af_version) as n_ver
        FROM proteins WHERE af_version IS NOT NULL
        GROUP BY organism HAVING n_ver > 1
    """).fetchall()
    if mixed:
        out.append(f"\nWARNING - Mixed AF versions within organism:")
        for org, nv in mixed:
            out.append(f"  {org}: {nv} different versions")
    out.append("")


# ============================================================
# SECTION 16: DATA INTEGRITY AUDIT
# ============================================================

def sec_integrity(conn, out):
    out.append(S("16. DATA INTEGRITY AUDIT"))

    checks = []

    # Check 1: NULL values in key fields
    for col in ['bps_norm', 'plddt_mean', 'pct_helix', 'rg', 'contact_order']:
        n_null = conn.execute(f"SELECT COUNT(*) FROM proteins WHERE {col} IS NULL").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
        pct = n_null / total * 100 if total > 0 else 0
        status = "OK" if pct < 1 else "WARN" if pct < 5 else "BAD"
        checks.append((f"NULL {col}", n_null, f"{pct:.1f}%", status))

    # Check 2: BPS/res outliers
    n_extreme = conn.execute(
        "SELECT COUNT(*) FROM proteins WHERE bps_norm < 0.03 OR bps_norm > 0.40"
    ).fetchone()[0]
    checks.append(("BPS/res extreme (<0.03 or >0.40)", n_extreme, f"{n_extreme}", "INFO"))

    # Check 3: pLDDT out of range
    n_bad_plddt = conn.execute(
        "SELECT COUNT(*) FROM proteins WHERE plddt_mean < 0 OR plddt_mean > 100"
    ).fetchone()[0]
    checks.append(("pLDDT out of [0,100]", n_bad_plddt, str(n_bad_plddt),
                    "OK" if n_bad_plddt == 0 else "BAD"))

    # Check 4: SS doesn't sum to ~100%
    n_bad_ss = conn.execute("""
        SELECT COUNT(*) FROM proteins
        WHERE pct_helix IS NOT NULL
        AND ABS(pct_helix + pct_sheet + pct_coil - 100.0) > 1.0
    """).fetchone()[0]
    checks.append(("SS% doesn't sum to 100", n_bad_ss, str(n_bad_ss),
                    "OK" if n_bad_ss == 0 else "BAD"))

    # Check 5: Duplicate UniProt IDs across organisms
    n_dupes = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT uniprot_id FROM proteins GROUP BY uniprot_id HAVING COUNT(DISTINCT organism) > 1
        )
    """).fetchone()[0]
    checks.append(("UniProt IDs in multiple organisms", n_dupes, str(n_dupes), "INFO"))

    # Check 6: Very small organisms (unreliable stats)
    small_orgs = conn.execute("""
        SELECT organism, COUNT(*) as n FROM proteins GROUP BY organism HAVING n < 100
    """).fetchall()
    for org, n in small_orgs:
        checks.append((f"{org}: only {n} proteins", n, str(n), "WARN"))

    # Check 7: Negative Rg or CO
    n_neg_rg = conn.execute("SELECT COUNT(*) FROM proteins WHERE rg <= 0").fetchone()[0]
    n_neg_co = conn.execute("SELECT COUNT(*) FROM proteins WHERE contact_order <= 0").fetchone()[0]
    checks.append(("Negative Rg", n_neg_rg, str(n_neg_rg), "OK" if n_neg_rg == 0 else "BAD"))
    checks.append(("Negative CO", n_neg_co, str(n_neg_co), "OK" if n_neg_co == 0 else "BAD"))

    h = f"{'Check':<42} {'Count':>8} {'Pct/Val':>8} {'Status':>8}"
    out.append(h)
    out.append("-" * len(h))
    for name, count, pct, status in checks:
        out.append(f"{name:<42} {count:>8} {pct:>8} {status:>8}")

    n_bad = sum(1 for _, _, _, s in checks if s == "BAD")
    n_warn = sum(1 for _, _, _, s in checks if s == "WARN")
    out.append(f"\nSummary: {n_bad} BAD, {n_warn} WARN, {len(checks)-n_bad-n_warn} OK")
    out.append("")


# ============================================================
# SECTION 17: QUICK ANSWERS
# ============================================================

def sec_quick_answers(conn, out):
    out.append(S("17. QUICK ANSWERS TO KEY QUESTIONS"))

    # Q1: Universal constant?
    rows = conn.execute("""
        SELECT organism, AVG(bps_norm) FROM proteins
        GROUP BY organism HAVING COUNT(*)>=50
    """).fetchall()
    if rows:
        vals = [r[1] for r in rows]
        out.append(f"Q: Is BPS/res a universal constant?")
        out.append(f"  Across {len(vals)} organisms: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        out.append(f"  Range: {min(vals):.4f} - {max(vals):.4f}")
        cv = np.std(vals) / np.mean(vals) * 100
        out.append(f"  CV = {cv:.1f}% -> {'YES, approximately' if cv < 15 else 'NO, significant variation'}")

    # Q2: pLDDT-controlled universality
    hq_rows = conn.execute("""
        SELECT organism, AVG(bps_norm) FROM proteins
        WHERE plddt_mean >= 85
        GROUP BY organism HAVING COUNT(*)>=50
    """).fetchall()
    if hq_rows:
        hq_vals = [r[1] for r in hq_rows]
        out.append(f"\nQ: Is BPS/res universal at pLDDT >= 85?")
        out.append(f"  Across {len(hq_vals)} organisms: {np.mean(hq_vals):.4f} +/- {np.std(hq_vals):.4f}")
        cv_hq = np.std(hq_vals) / np.mean(hq_vals) * 100
        out.append(f"  CV = {cv_hq:.1f}% -> {'TIGHTER' if cv_hq < cv else 'UNCHANGED'} vs all-data CV={cv:.1f}%")

    # Q3: Does pLDDT confound everything?
    n_plddt = conn.execute(
        "SELECT COUNT(*) FROM proteins WHERE plddt_mean IS NOT NULL"
    ).fetchone()[0]
    if n_plddt > 100:
        rows = conn.execute(
            "SELECT bps_norm, plddt_mean FROM proteins WHERE plddt_mean IS NOT NULL"
        ).fetchall()
        r_all, _ = pearsonr([r[0] for r in rows], [r[1] for r in rows])
        hq_rows2 = conn.execute(
            "SELECT bps_norm, plddt_mean FROM proteins WHERE plddt_mean >= 85"
        ).fetchall()
        if len(hq_rows2) > 100:
            r_hq, _ = pearsonr([r[0] for r in hq_rows2], [r[1] for r in hq_rows2])
            out.append(f"\nQ: Does pLDDT confound BPS?")
            out.append(f"  r(BPS, pLDDT) all data:    {r_all:.4f}")
            out.append(f"  r(BPS, pLDDT) pLDDT>=85:   {r_hq:.4f}")
            out.append(f"  -> {'Confound persists' if abs(r_hq) > 0.3 else 'Confound reduced'} when restricting to HQ proteins")

    # Q4: BPS vs CO
    n_co = conn.execute(
        "SELECT COUNT(*) FROM proteins WHERE contact_order IS NOT NULL AND plddt_mean >= 85"
    ).fetchone()[0]
    if n_co > 100:
        rows = conn.execute(
            "SELECT bps_norm, contact_order FROM proteins WHERE contact_order IS NOT NULL AND plddt_mean >= 85"
        ).fetchall()
        r_val, _ = pearsonr([r[0] for r in rows], [r[1] for r in rows])
        out.append(f"\nQ: Is BPS independent of CO (HQ proteins only)?")
        out.append(f"  r(BPS/res, CO) pLDDT>=85 = {r_val:.4f} across {n_co:,} proteins")
        out.append(f"  -> {'Independent' if abs(r_val) < 0.15 else 'Partially overlapping'}")

    # Q5: Bacteria vs eukaryotes (controlled)
    bact_hq = [r[0] for r in conn.execute("""
        SELECT AVG(bps_norm) FROM proteins
        WHERE organism IN ('ecoli','bacillus','tuberculosis','staph',
                           'pseudomonas','salmonella','campylobacter','helicobacter')
        AND plddt_mean >= 85
        GROUP BY organism HAVING COUNT(*)>=10
    """).fetchall()]
    euk_hq = [r[0] for r in conn.execute("""
        SELECT AVG(bps_norm) FROM proteins
        WHERE organism NOT IN ('ecoli','bacillus','tuberculosis','staph',
                               'pseudomonas','salmonella','campylobacter','helicobacter')
        AND plddt_mean >= 85
        GROUP BY organism HAVING COUNT(*)>=10
    """).fetchall()]
    if bact_hq and euk_hq:
        out.append(f"\nQ: Do bacteria and eukaryotes differ (pLDDT>=85)?")
        out.append(f"  Bacteria: {np.mean(bact_hq):.4f} +/- {np.std(bact_hq):.4f}")
        out.append(f"  Eukaryotes: {np.mean(euk_hq):.4f} +/- {np.std(euk_hq):.4f}")
        if len(bact_hq) >= 2 and len(euk_hq) >= 2:
            t, p = ttest_ind(bact_hq, euk_hq)
            out.append(f"  t-test p = {p:.2e} -> {'Significant' if p < 0.05 else 'Not significant'}")

    # Q6: Phi sign check
    row = conn.execute("""
        SELECT AVG(pct_helix), AVG(pct_sheet), AVG(pct_coil), AVG(n_transitions)
        FROM proteins WHERE pct_helix IS NOT NULL
    """).fetchone()
    if row and row[0] is not None:
        out.append(f"\nQ: Is the phi/psi sign fix working?")
        out.append(f"  Global SS: {row[0]:.1f}% helix, {row[1]:.1f}% sheet, {row[2]:.1f}% coil")
        out.append(f"  Avg transitions: {row[3]:.1f}")
        if row[0] > 20 and row[3] > 10:
            out.append(f"  -> YES: helix detection working, transitions are real")
        elif row[0] < 5:
            out.append(f"  -> NO: helix still near zero -- dihedral sign bug may persist")

    out.append("")


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(conn, db_path):
    """Generate the full report from the database. Returns text string or None."""
    total = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
    if total == 0:
        return None

    out = []
    out.append("=" * 72)
    out.append("  BPS CROSS-PROTEOME ANALYSIS")
    out.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out.append(f"  Database: {db_path} ({db_path.stat().st_size/1024/1024:.1f} MB)")
    out.append(f"  Proteins: {total:,}")
    out.append("=" * 72)

    sections = [
        sec_overview,
        sec_organism_table,
        sec_size_gradient,
        sec_bps_vs_length,
        sec_kingdom,
        sec_plddt_vs_bps,
        sec_ss_vs_bps,
        sec_co_vs_bps,
        sec_rg_vs_bps,
        sec_distances,
        sec_failures,
        sec_histograms,
        sec_correlation_matrix,
        sec_plddt_controlled,
        sec_af_versions,
        sec_integrity,
        sec_quick_answers,
    ]

    for sec_fn in sections:
        try:
            sec_fn(conn, out)
        except Exception as e:
            out.append(f"\n  [Error in {sec_fn.__name__}: {e}]\n")

    return "\n".join(out)


def write_report_atomic(text, out_path):
    """Write report atomically: temp file + rename. Safe for concurrent readers."""
    tmp_path = out_path.parent / f".{out_path.name}.tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
    tmp_path.replace(out_path)


# ============================================================
# WATCH MODE — continuous report updates
# ============================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def watch_loop(db_path, out_path, poll_interval=60):
    """Watch database for new data and regenerate report.
    Safe to run alongside bps_download.py and bps_process.py."""

    setup_logging()
    logging.info(f"WATCH MODE: checking every {poll_interval}s")
    logging.info(f"  Database: {db_path}")
    logging.info(f"  Output:   {out_path}")

    last_count = 0
    last_orgs = 0

    while True:
        if not db_path.exists():
            logging.info(f"Database not found yet. Sleeping {poll_interval}s...")
            try:
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                logging.info("\nWatch stopped.")
                return
            continue

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=10000")
        try:
            count = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
            n_orgs = conn.execute(
                "SELECT COUNT(DISTINCT organism) FROM proteins"
            ).fetchone()[0]
        except sqlite3.OperationalError as e:
            logging.warning(f"DB busy: {e}. Will retry...")
            conn.close()
            time.sleep(5)
            continue

        if count == last_count and n_orgs == last_orgs:
            conn.close()
            logging.info(f"No new data ({count:,} proteins, {n_orgs} orgs). "
                         f"Sleeping {poll_interval}s...")
            try:
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                logging.info("\nWatch stopped.")
                return
            continue

        logging.info(f"New data: {count:,} proteins, {n_orgs} organisms "
                     f"(was {last_count:,}, {last_orgs})")
        try:
            text = generate_report(conn, db_path)
            if text:
                write_report_atomic(text, out_path)
                n_lines = text.count('\n')
                est_tokens = len(text) // 4
                logging.info(f"Report updated: {n_lines} lines, ~{est_tokens:,} tokens "
                             f"-> {out_path}")
                last_count = count
                last_orgs = n_orgs
        except Exception as e:
            logging.error(f"Report generation error: {e}")
        finally:
            conn.close()

        try:
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            logging.info("\nWatch stopped.")
            return


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compile BPS results for Claude")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    parser.add_argument("--out", type=str, default=str(OUTPUT_FILE))
    parser.add_argument("--mode", choices=["once", "watch"], default="once",
                        help="'once' generates report and exits; "
                             "'watch' continuously updates as new data arrives")
    parser.add_argument("--poll", type=int, default=60,
                        help="Poll interval in seconds for watch mode (default: 60)")
    args = parser.parse_args()

    db_path = Path(args.db)
    out_path = Path(args.out)

    if args.mode == "watch":
        watch_loop(db_path, out_path, args.poll)
        return

    # ── One-shot mode (original behavior) ──
    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}")
        print(f"Run bps_process.py first.")
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA busy_timeout=10000")

    try:
        text = generate_report(conn, db_path)
    finally:
        conn.close()

    if text is None:
        print("ERROR: Database is empty. Run bps_process.py first.")
        return

    write_report_atomic(text, out_path)

    n_chars = len(text)
    n_lines = text.count('\n')
    est_tokens = n_chars // 4
    print(f"\nWrote {n_lines} lines ({n_chars:,} chars, ~{est_tokens:,} tokens) -> {out_path}")
    if est_tokens > 30000:
        print(f"WARNING: Large output ({est_tokens:,} tokens). Consider summarizing before pasting.")


if __name__ == "__main__":
    main()
