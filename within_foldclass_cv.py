"""
TorusFold: Within-Fold-Class Cross-Organism CV
===============================================
Tests whether cross-organism BPS/L conservation is trivial (just
averaging over different fold compositions) or genuine (holds within
each fold class independently).

For each fold class (all-α, all-β, α+β, α/β):
  - Compute per-organism mean BPS/L
  - Compute cross-organism CV
  - Compare to overall cross-organism CV

If within-class CVs are similar to overall CV, conservation is real.
If within-class CVs are much larger, overall conservation is just
compositional averaging.

Also computes:
  - Fold-class composition per organism (% α, β, α+β, α/β)
  - Statistical test of bacteria vs eukaryote offset within each class
  - Effect sizes (Cohen's d) for fold-class separation within organisms

Usage:
  python within_foldclass_cv.py [--data DIR] [--sample N] [--w-path FILE]

Reads: alphafold_cache/, results/superpotential_W.npz (or --w-path)
Writes: results/within_foldclass_cv_report.md
"""

import os
import sys
import math
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi required. pip install gemmi")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def _dihedral_angle(p0, p1, p2, p3):
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1n = np.linalg.norm(n1)
    n2n = np.linalg.norm(n2)
    if n1n < 1e-10 or n2n < 1e-10:
        return 0.0
    n1 /= n1n
    n2 /= n2n
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return math.atan2(np.dot(m1, n2), np.dot(n1, n2))


def extract_angles(filepath, plddt_min=70.0):
    """Extract (phi, psi) pairs and SS assignments."""
    st = gemmi.read_structure(str(filepath))
    if len(st) == 0 or len(st[0]) == 0:
        return [], []
    chain = st[0][0]
    residues = []
    for res in chain:
        info = gemmi.find_tabulated_residue(res.name)
        if not info.is_amino_acid():
            continue
        atoms = {}
        for atom in res:
            if atom.name in ('N', 'CA', 'C'):
                atoms[atom.name] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        if len(atoms) == 3:
            ca = res.find_atom('CA', '*')
            residues.append({
                'N': atoms['N'], 'CA': atoms['CA'], 'C': atoms['C'],
                'plddt': ca.b_iso if ca else 0,
            })
    if len(residues) < 3:
        return [], []

    phi_psi = []
    ss_seq = []
    for i in range(1, len(residues) - 1):
        phi = -_dihedral_angle(residues[i-1]['C'], residues[i]['N'],
                               residues[i]['CA'], residues[i]['C'])
        psi = -_dihedral_angle(residues[i]['N'], residues[i]['CA'],
                               residues[i]['C'], residues[i+1]['N'])
        if residues[i]['plddt'] < plddt_min:
            continue
        phi_d, psi_d = math.degrees(phi), math.degrees(psi)
        if -160 < phi_d < 0 and -120 < psi_d < 30:
            ss = 'a'
        elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
            ss = 'b'
        else:
            ss = 'o'
        phi_psi.append((phi, psi))
        ss_seq.append(ss)
    return phi_psi, ss_seq


def classify_fold(ss_seq, alpha_thresh=0.15, beta_thresh=0.15):
    """Classify protein fold class from SS sequence."""
    n = len(ss_seq)
    if n == 0:
        return 'other'
    na = ss_seq.count('a') / n
    nb = ss_seq.count('b') / n
    has_alpha = na >= alpha_thresh
    has_beta = nb >= beta_thresh
    if has_alpha and not has_beta:
        return 'all-alpha'
    elif has_beta and not has_alpha:
        return 'all-beta'
    elif has_alpha and has_beta:
        # Distinguish α+β from α/β by transition density
        transitions = sum(1 for i in range(len(ss_seq)-1)
                         if ss_seq[i] != ss_seq[i+1]
                         and ss_seq[i] in ('a','b')
                         and ss_seq[i+1] in ('a','b'))
        n_ab = na * n + nb * n
        trans_rate = transitions / max(n_ab, 1)
        if trans_rate > 0.05:
            return 'alpha/beta'
        else:
            return 'alpha+beta'
    else:
        return 'other'


def compute_bps_l(phi_arr, psi_arr, W_grid, grid_size):
    """BPS/L from arrays of phi, psi in radians."""
    if len(phi_arr) < 2:
        return 0.0
    scale = grid_size / 360.0
    phi_d = np.degrees(phi_arr)
    psi_d = np.degrees(psi_arr)
    gi = (np.round((phi_d + 180) * scale).astype(int)) % grid_size
    gj = (np.round((psi_d + 180) * scale).astype(int)) % grid_size
    w = W_grid[gi, gj]
    return float(np.mean(np.abs(np.diff(w))))


def cohens_d(group1, group2):
    """Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    sp = math.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    if sp < 1e-10:
        return 0.0
    return (m1 - m2) / sp


# ═══════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def discover_files(data_dir):
    """Return {organism: [filepaths]}."""
    data_path = Path(data_dir)
    organisms = {}
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        files = sorted(subdir.glob("*.cif"))
        if files:
            organisms[subdir.name] = [str(f) for f in files]
    return organisms


# ═══════════════════════════════════════════════════════════════════
# ORGANISM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

BACTERIA = {'ecoli', 'bacillus', 'tuberculosis', 'salmonella', 'pseudomonas',
            'staph', 'helicobacter', 'campylobacter'}
EUKARYOTES = {'human', 'mouse', 'rat', 'chicken', 'chimp', 'gorilla', 'dog',
              'cat', 'cow', 'pig', 'yeast', 'fission_yeast', 'candida',
              'fly', 'worm', 'mosquito', 'honeybee', 'malaria', 'trypanosoma',
              'leishmania', 'dictyostelium'}


def domain_of(org):
    if org in BACTERIA:
        return 'bacteria'
    elif org in EUKARYOTES:
        return 'eukaryote'
    return 'other'


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Within-Fold-Class CV Analysis")
    parser.add_argument("--data", default="alphafold_cache",
                        help="AlphaFold data directory")
    parser.add_argument("--sample", type=int, default=0,
                        help="Max proteins per organism (0=all)")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--w-path", default=None,
                        help="Explicit path to superpotential_W.npz "
                             "(auto-detected if not specified)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold: Within-Fold-Class Cross-Organism CV")
    print("=" * 60)

    # Load W (search multiple locations)
    w_path = args.w_path or os.path.join(args.output, "superpotential_W.npz")
    W_grid = None

    if os.path.exists(w_path):
        data = np.load(w_path)
        W_grid = data['grid']
        print(f"  Loaded W: {W_grid.shape[0]}x{W_grid.shape[0]} from {w_path}")
    else:
        # Search common locations
        search_paths = [
            os.path.join(args.output, "superpotential_W.npz"),
            os.path.join(args.data, '..', 'results', 'superpotential_W.npz'),
            os.path.join(args.data, '..', 'superpotential_W.npz'),
            'superpotential_W.npz',
        ]
        for sp in search_paths:
            sp = os.path.normpath(sp)
            if os.path.exists(sp):
                print(f"  Found W at {sp}")
                data = np.load(sp)
                W_grid = data['grid']
                print(f"  Loaded W: {W_grid.shape[0]}x{W_grid.shape[0]}")
                break

    if W_grid is None:
        print(f"ERROR: superpotential_W.npz not found.")
        print(f"  Searched: {w_path}")
        print(f"  Use --w-path to specify the location, e.g.:")
        print(f"    python within_foldclass_cv.py --w-path path/to/superpotential_W.npz")
        sys.exit(1)

    grid_size = W_grid.shape[0]

    # Discover files
    organisms = discover_files(args.data)
    total = sum(len(f) for f in organisms.values())
    print(f"  Data: {total} files across {len(organisms)} organisms")

    # Process all proteins
    # Structure: {organism: {fold_class: [bpsl_values]}}
    org_fold_bpsl = defaultdict(lambda: defaultdict(list))

    n_processed = 0
    n_skipped = 0
    MAX_FILE_SIZE = 5 * 1024 * 1024  # skip files > 5MB (huge proteins hang gemmi)

    for org_name, files in sorted(organisms.items()):
        rng = np.random.default_rng(hash(org_name) % (2**31))
        if args.sample > 0 and len(files) > args.sample:
            idx = rng.choice(len(files), args.sample, replace=False)
            files = [files[i] for i in idx]

        print(f"  Processing {org_name} ({len(files)} files)...", end=" ", flush=True)
        org_count = 0
        org_errors = 0

        for fi, filepath in enumerate(files):
            try:
                # Skip very large files that can hang the parser
                fsize = os.path.getsize(filepath)
                if fsize > MAX_FILE_SIZE:
                    n_skipped += 1
                    continue

                pp, ss = extract_angles(filepath)
                if len(pp) < 50:
                    n_skipped += 1
                    continue

                fold = classify_fold(ss)
                if fold == 'other':
                    n_skipped += 1
                    continue

                phi = np.array([p[0] for p in pp])
                psi = np.array([p[1] for p in pp])
                bpsl = compute_bps_l(phi, psi, W_grid, grid_size)

                org_fold_bpsl[org_name][fold].append(bpsl)
                n_processed += 1
                org_count += 1

            except Exception as e:
                n_skipped += 1
                org_errors += 1
                # Log first few errors per organism for debugging
                if org_errors <= 3:
                    fname = os.path.basename(filepath)
                    print(f"\n    WARNING: {fname}: {type(e).__name__}: {e}",
                          end="", flush=True)

            # Progress for large organisms
            if (fi + 1) % 2000 == 0:
                print(f"\n    [{fi+1}/{len(files)}] {org_count} classified...",
                      end="", flush=True)

        error_note = f" ({org_errors} errors)" if org_errors > 0 else ""
        print(f"{org_count} classified{error_note}")

    print(f"\n  Total: {n_processed} proteins classified, {n_skipped} skipped")

    # ══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════

    fold_classes = ['all-alpha', 'alpha+beta', 'all-beta', 'alpha/beta']

    # 1. Overall cross-organism CV (all folds combined)
    org_overall = {}
    for org in org_fold_bpsl:
        all_vals = []
        for fc in fold_classes:
            all_vals.extend(org_fold_bpsl[org][fc])
        if len(all_vals) >= 10:
            org_overall[org] = np.mean(all_vals)

    overall_means = np.array(list(org_overall.values()))
    overall_cv = float(np.std(overall_means) / np.mean(overall_means) * 100)

    # 2. Within-fold-class cross-organism CV
    fold_stats = {}
    for fc in fold_classes:
        org_means = {}
        for org in org_fold_bpsl:
            vals = org_fold_bpsl[org][fc]
            if len(vals) >= 5:  # need minimum sample
                org_means[org] = float(np.mean(vals))

        if len(org_means) < 3:
            continue

        means_arr = np.array(list(org_means.values()))
        cv = float(np.std(means_arr) / np.mean(means_arr) * 100)
        grand_mean = float(np.mean(means_arr))
        grand_std = float(np.std(means_arr))

        # Bacteria vs eukaryote within this fold class
        bact_means = [v for o, v in org_means.items() if domain_of(o) == 'bacteria']
        euk_means = [v for o, v in org_means.items() if domain_of(o) == 'eukaryote']

        bact_euk = {}
        if len(bact_means) >= 2 and len(euk_means) >= 2:
            bact_euk = {
                'bact_mean': float(np.mean(bact_means)),
                'euk_mean': float(np.mean(euk_means)),
                'bact_n': len(bact_means),
                'euk_n': len(euk_means),
                'd': cohens_d(bact_means, euk_means),
                'delta': float(np.mean(bact_means) - np.mean(euk_means)),
            }

        fold_stats[fc] = {
            'n_organisms': len(org_means),
            'n_proteins': sum(len(org_fold_bpsl[o][fc]) for o in org_means),
            'grand_mean': grand_mean,
            'grand_std': grand_std,
            'cv': cv,
            'org_means': org_means,
            'bact_euk': bact_euk,
        }

    # 3. Fold-class composition per organism
    org_composition = {}
    for org in org_fold_bpsl:
        total_org = sum(len(org_fold_bpsl[org][fc]) for fc in fold_classes)
        if total_org < 10:
            continue
        comp = {}
        for fc in fold_classes:
            comp[fc] = len(org_fold_bpsl[org][fc]) / total_org * 100
        comp['total'] = total_org
        org_composition[org] = comp

    # 4. Within-organism fold-class separation
    org_separation = {}
    for org in org_fold_bpsl:
        alpha_vals = org_fold_bpsl[org].get('all-alpha', [])
        ab_vals = org_fold_bpsl[org].get('alpha/beta', [])
        if len(alpha_vals) >= 10 and len(ab_vals) >= 10:
            d = cohens_d(ab_vals, alpha_vals)  # positive if α/β > all-α
            org_separation[org] = {
                'd': d,
                'alpha_mean': float(np.mean(alpha_vals)),
                'ab_mean': float(np.mean(ab_vals)),
                'alpha_n': len(alpha_vals),
                'ab_n': len(ab_vals),
            }

    # ══════════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════════
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "within_foldclass_cv_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Within-Fold-Class Cross-Organism CV Analysis\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Proteins analyzed:** {n_processed}\n")
        f.write(f"**Organisms:** {len(org_overall)}\n\n")

        # Key result table
        f.write("## Key Result: Cross-Organism CV by Fold Class\n\n")
        f.write("| Fold Class | N proteins | N organisms | Mean BPS/L | "
                "Cross-org CV | Bact mean | Euk mean | Cohen's d |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")

        # Overall first
        f.write(f"| **Overall** | {n_processed} | {len(org_overall)} | "
                f"{float(np.mean(overall_means)):.3f} | "
                f"**{overall_cv:.1f}%** | — | — | — |\n")

        for fc in fold_classes:
            if fc not in fold_stats:
                continue
            s = fold_stats[fc]
            be = s['bact_euk']
            bm = f"{be['bact_mean']:.3f}" if be else "—"
            em = f"{be['euk_mean']:.3f}" if be else "—"
            cd = f"{be['d']:.2f}" if be else "—"
            f.write(f"| {fc} | {s['n_proteins']} | {s['n_organisms']} | "
                    f"{s['grand_mean']:.3f} | **{s['cv']:.1f}%** | "
                    f"{bm} | {em} | {cd} |\n")

        f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")

        within_cvs = [fold_stats[fc]['cv'] for fc in fold_classes if fc in fold_stats]
        within_ns = [fold_stats[fc]['n_proteins'] for fc in fold_classes if fc in fold_stats]
        if within_cvs:
            avg_within_unweighted = float(np.mean(within_cvs))
            # Protein-count-weighted average (proper weighting for unequal class sizes)
            total_n = sum(within_ns)
            avg_within_weighted = float(sum(cv * n for cv, n in
                                           zip(within_cvs, within_ns)) / total_n) \
                if total_n > 0 else avg_within_unweighted

            f.write(f"- Overall cross-organism CV: **{overall_cv:.1f}%**\n")
            f.write(f"- Unweighted average within-class CV: "
                    f"**{avg_within_unweighted:.1f}%**\n")
            f.write(f"- Protein-count-weighted average within-class CV: "
                    f"**{avg_within_weighted:.1f}%**\n\n")

            # Per-class breakdown
            for fc in fold_classes:
                if fc in fold_stats:
                    s = fold_stats[fc]
                    f.write(f"  - {fc}: CV = {s['cv']:.1f}%, "
                            f"N = {s['n_proteins']} proteins, "
                            f"{s['n_organisms']} organisms\n")
            f.write("\n")

            # Report raw numbers — let the reader judge
            ratio = avg_within_weighted / overall_cv if overall_cv > 0 else 0
            f.write(f"**Within-class / overall CV ratio: "
                    f"{ratio:.2f}x** "
                    f"(weighted within-class: {avg_within_weighted:.1f}%, "
                    f"overall: {overall_cv:.1f}%)\n\n")

            if ratio <= 1.2:
                f.write("Within-class CVs are close to the overall CV, "
                        "consistent with genuine conservation within "
                        "individual fold classes.\n\n")
            elif ratio <= 2.0:
                f.write("Within-class CVs are moderately higher than "
                        "the overall CV. Some of the overall conservation "
                        "may reflect consistent fold-class composition.\n\n")
            else:
                f.write("Within-class CVs substantially exceed the overall "
                        "CV, indicating that conservation is partly driven "
                        "by compositional averaging.\n\n")

            # Note on unweighted vs weighted
            if abs(avg_within_unweighted - avg_within_weighted) > 0.5:
                f.write(f"*Note: The unweighted average "
                        f"({avg_within_unweighted:.1f}%) differs from the "
                        f"weighted average ({avg_within_weighted:.1f}%) "
                        "because small fold classes have fewer per-organism "
                        "samples, inflating their CVs.*\n\n")

        # Fold-class composition by organism
        f.write("## Fold-Class Composition by Organism\n\n")
        f.write("| Organism | Domain | N | all-α % | α+β % | all-β % | α/β % |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for org in sorted(org_composition, key=lambda x: org_composition[x].get('total', 0),
                          reverse=True):
            c = org_composition[org]
            dom = domain_of(org)
            f.write(f"| {org} | {dom} | {c['total']} | "
                    f"{c.get('all-alpha', 0):.1f} | "
                    f"{c.get('alpha+beta', 0):.1f} | "
                    f"{c.get('all-beta', 0):.1f} | "
                    f"{c.get('alpha/beta', 0):.1f} |\n")
        f.write("\n")

        # Per-fold-class organism means
        for fc in fold_classes:
            if fc not in fold_stats:
                continue
            s = fold_stats[fc]
            f.write(f"## {fc}: Per-Organism Means\n\n")
            f.write(f"Cross-organism CV = {s['cv']:.1f}%\n\n")
            f.write("| Organism | Domain | N | Mean BPS/L |\n")
            f.write("|---|---|---|---|\n")
            for org in sorted(s['org_means'], key=lambda x: s['org_means'][x]):
                n_org = len(org_fold_bpsl[org][fc])
                dom = domain_of(org)
                f.write(f"| {org} | {dom} | {n_org} | "
                        f"{s['org_means'][org]:.3f} |\n")
            f.write("\n")

            if s['bact_euk']:
                be = s['bact_euk']
                f.write(f"Bacteria ({be['bact_n']} orgs): {be['bact_mean']:.3f}, "
                        f"Eukaryotes ({be['euk_n']} orgs): {be['euk_mean']:.3f}, "
                        f"Δ = {be['delta']:.3f}, Cohen's d = {be['d']:.2f}\n\n")

        # Within-organism fold-class separation
        f.write("## Within-Organism Fold-Class Separation (all-α vs α/β)\n\n")
        f.write("| Organism | all-α mean | N | α/β mean | N | Cohen's d |\n")
        f.write("|---|---|---|---|---|---|\n")
        for org in sorted(org_separation, key=lambda x: org_separation[x]['d'],
                          reverse=True):
            s = org_separation[org]
            f.write(f"| {org} | {s['alpha_mean']:.3f} | {s['alpha_n']} | "
                    f"{s['ab_mean']:.3f} | {s['ab_n']} | {s['d']:.2f} |\n")
        f.write("\n")

        if org_separation:
            ds = [s['d'] for s in org_separation.values()]
            f.write(f"Mean Cohen's d across organisms: {np.mean(ds):.2f} "
                    f"± {np.std(ds):.2f}\n")
            f.write(f"Range: {min(ds):.2f} to {max(ds):.2f}\n\n")

        # Summary
        f.write("## Conclusion\n\n")
        f.write("```\n")
        f.write("WITHIN-FOLD-CLASS CV SUMMARY\n")
        f.write("═══════════════════════════════════════════\n")
        f.write(f"  Overall cross-organism CV:     {overall_cv:.1f}%\n")
        for fc in fold_classes:
            if fc in fold_stats:
                s = fold_stats[fc]
                f.write(f"  {fc:15s} cross-org CV:  {s['cv']:.1f}%  "
                        f"(N={s['n_proteins']})\n")
        f.write("═══════════════════════════════════════════\n")
        if within_cvs:
            total_n = sum(within_ns)
            avg_weighted = float(sum(cv * n for cv, n in
                                     zip(within_cvs, within_ns)) / total_n) \
                if total_n > 0 else float(np.mean(within_cvs))
            f.write(f"  Unweighted avg within-class:   "
                    f"{float(np.mean(within_cvs)):.1f}%\n")
            f.write(f"  Weighted avg within-class:     "
                    f"{avg_weighted:.1f}%\n")
        f.write("```\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


def plddt_sensitivity(data_dir, W_grid, grid_size, sample_per_org=0):
    """Run BPS/L analysis at multiple pLDDT thresholds.

    Reports N proteins, mean BPS/L, CV, and Seg/Real ratio at each
    threshold to demonstrate that conservation is not an artifact of
    aggressive quality filtering.

    Returns list of dicts with threshold, n, mean, cv.
    """
    from generate_figures import (extract_angles as _extract_angles,
                                  compute_bps_l, null_segment_preserving,
                                  discover_files, classify_fold)
    import warnings
    warnings.filterwarnings('ignore')

    organisms = discover_files(data_dir)
    thresholds = [70, 80, 85, 90]
    results = []

    for plddt_thresh in thresholds:
        all_bpsl = []
        rng = np.random.default_rng(42)
        MAX_FILE_SIZE = 5 * 1024 * 1024

        for org_name, files in sorted(organisms.items()):
            sel = files
            if sample_per_org > 0 and len(files) > sample_per_org:
                idx = rng.choice(len(files), sample_per_org, replace=False)
                sel = [files[i] for i in idx]

            for filepath in sel:
                try:
                    if os.path.getsize(filepath) > MAX_FILE_SIZE:
                        continue
                    pp, ss = _extract_angles(filepath, plddt_min=float(plddt_thresh))
                    if len(pp) < 50:
                        continue
                    phi = np.array([p[0] for p in pp])
                    psi = np.array([p[1] for p in pp])
                    bpsl, _ = compute_bps_l(phi, psi, W_grid, grid_size)
                    all_bpsl.append(bpsl)
                except Exception:
                    continue

        if all_bpsl:
            arr = np.array(all_bpsl)
            mean_bps = float(np.mean(arr))
            std_bps = float(np.std(arr, ddof=1))
            cv = std_bps / mean_bps * 100 if mean_bps > 0 else 0
            results.append({
                'threshold': plddt_thresh,
                'n_proteins': len(arr),
                'mean_bps': mean_bps,
                'std_bps': std_bps,
                'cv_pct': cv,
            })
            print(f"  pLDDT >= {plddt_thresh}: N={len(arr)}, "
                  f"BPS/L={mean_bps:.3f} ± {std_bps:.3f}, CV={cv:.1f}%")

    return results


if __name__ == "__main__":
    main()
