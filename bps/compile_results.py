"""Compile BPS/L results into publication-ready tables and reports.

Reads from results/per_protein_bpsl.csv and results/per_organism.csv.
Generates:
  - Organism summary table (for paper Table 1)
  - Fold-class breakdown (for paper Table 2)
  - Cross-organism CV (headline number)
  - Three-level decomposition (requires re-running with --decomposition flag)

Usage:
    python -m bps.compile_results
    python -m bps.compile_results --decomposition  # include Markov/Shuffled
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")


def load_per_protein(path: str) -> List[Dict]:
    """Load per-protein CSV results."""
    results = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            row['bps_l'] = float(row['bps_l'])
            row['frac_alpha'] = float(row['frac_alpha'])
            row['frac_beta'] = float(row['frac_beta'])
            row['frac_other'] = float(row['frac_other'])
            row['length'] = int(row['length'])
            row['n_valid'] = int(row['n_valid'])
            row['mean_plddt'] = float(row.get('mean_plddt', 0))
            results.append(row)
    return results


def organism_table(results: List[Dict]) -> str:
    """Generate per-organism summary table (paper Table 1)."""
    by_org = {}
    for r in results:
        org = r['organism']
        if org not in by_org:
            by_org[org] = {'species': r['species'], 'vals': []}
        by_org[org]['vals'].append(r['bps_l'])

    lines = []
    lines.append("## Per-Organism BPS/L Summary")
    lines.append("")
    lines.append("| Organism | Species | N | Mean BPS/L | Std | CV% |")
    lines.append("|----------|---------|---|------------|-----|-----|")

    org_means = []
    for org_id in sorted(by_org.keys()):
        info = by_org[org_id]
        vals = np.array(info['vals'])
        m = np.mean(vals)
        s = np.std(vals)
        cv = s / m * 100 if m > 0 else 0
        org_means.append(m)
        lines.append(
            f"| {org_id} | {info['species']} | {len(vals)} "
            f"| {m:.3f} | {s:.3f} | {cv:.1f}% |"
        )

    if len(org_means) > 1:
        gm = np.mean(org_means)
        gs = np.std(org_means)
        gcv = gs / gm * 100 if gm > 0 else 0
        lines.append("")
        lines.append(f"**Cross-organism: mean = {gm:.3f}, "
                      f"std = {gs:.3f}, CV = {gcv:.1f}%** "
                      f"(N = {len(org_means)} organisms, "
                      f"{sum(len(by_org[o]['vals']) for o in by_org)} structures)")

    return "\n".join(lines)


def fold_class_table(results: List[Dict]) -> str:
    """Generate fold-class breakdown table (paper Table 2)."""
    by_fc = {}
    for r in results:
        fc = r['fold_class']
        if fc not in by_fc:
            by_fc[fc] = []
        by_fc[fc].append(r)

    lines = []
    lines.append("## Fold-Class Breakdown")
    lines.append("")
    lines.append("| Fold class | N | Mean BPS/L | Std | CV% | Mean alpha% | Mean beta% |")
    lines.append("|------------|---|------------|-----|-----|-------------|------------|")

    for fc in ['all-alpha', 'all-beta', 'alpha/beta', 'alpha+beta', 'other']:
        if fc not in by_fc or not by_fc[fc]:
            continue
        rs = by_fc[fc]
        vals = np.array([r['bps_l'] for r in rs])
        m = np.mean(vals)
        s = np.std(vals)
        cv = s / m * 100 if m > 0 else 0
        ma = np.mean([r['frac_alpha'] for r in rs]) * 100
        mb = np.mean([r['frac_beta'] for r in rs]) * 100
        lines.append(
            f"| {fc} | {len(rs)} | {m:.3f} | {s:.3f} | {cv:.1f}% "
            f"| {ma:.1f}% | {mb:.1f}% |"
        )

    # Diagnostic
    n_other = len(by_fc.get('other', []))
    n_total = len(results)
    if n_other > n_total * 0.90:
        lines.append("")
        lines.append(f"**WARNING:** {n_other}/{n_total} = {n_other/n_total*100:.1f}% "
                      f"classified as 'other'. Fold classification likely broken.")
        # Diagnostic info
        alphas = [r['frac_alpha'] for r in results]
        betas = [r['frac_beta'] for r in results]
        lines.append(f"  Mean frac_alpha: {np.mean(alphas):.3f}, "
                      f"Mean frac_beta: {np.mean(betas):.3f}")
        lines.append(f"  Alpha>35%: {sum(1 for a in alphas if a >= 0.35)}, "
                      f"Beta>25%: {sum(1 for b in betas if b >= 0.25)}")

    return "\n".join(lines)


def ss_diagnostic(results: List[Dict]) -> str:
    """SS fraction diagnostic — catches radians/degrees bugs."""
    alphas = [r['frac_alpha'] for r in results]
    betas = [r['frac_beta'] for r in results]
    others = [r['frac_other'] for r in results]

    lines = []
    lines.append("## SS Fraction Diagnostic")
    lines.append("")
    lines.append(f"Mean alpha: {np.mean(alphas):.3f}")
    lines.append(f"Mean beta:  {np.mean(betas):.3f}")
    lines.append(f"Mean other: {np.mean(others):.3f}")
    lines.append(f"Alpha>35%:  {sum(1 for a in alphas if a >= 0.35)}")
    lines.append(f"Beta>25%:   {sum(1 for b in betas if b >= 0.25)}")
    lines.append("")

    if np.mean(alphas) < 0.05 and np.mean(betas) < 0.05:
        lines.append("**DIAGNOSIS: SS assignment is broken.**")
        lines.append("Both alpha and beta fractions near zero.")
        lines.append("Likely cause: radians passed to assign_basin() instead of degrees,")
        lines.append("or a sign flip in dihedral computation.")
    elif np.mean(alphas) > 0.10 and np.mean(betas) > 0.05:
        n_classified = sum(1 for r in results if r['fold_class'] != 'other')
        lines.append(f"SS fractions look reasonable. "
                      f"{n_classified}/{len(results)} classified to named fold classes.")
        if n_classified < len(results) * 0.10:
            lines.append("**Fold-class thresholds may need adjusting.**")
    else:
        lines.append("SS fractions are unusual. Check dihedral extraction.")

    return "\n".join(lines)


def write_compilation_report(results: List[Dict], output_path: str) -> None:
    """Write full compilation report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vals = np.array([r['bps_l'] for r in results])

    lines = []
    lines.append("# AlphaFold BPS/L Compilation Report")
    lines.append("")
    lines.append(f"**Total structures:** {len(results)}")
    lines.append(f"**Mean BPS/L:** {np.mean(vals):.3f} +/- {np.std(vals):.3f}")
    lines.append(f"**CV:** {np.std(vals)/np.mean(vals)*100:.1f}%")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append(organism_table(results))
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append(fold_class_table(results))
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append(ss_diagnostic(results))

    with open(output_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compile BPS/L results")
    parser.add_argument("--results-dir", type=str, default=_RESULTS_DIR)
    parser.add_argument("--output", type=str,
                        default=os.path.join(_OUTPUT_DIR,
                                             "alphafold_compilation_report.md"))
    args = parser.parse_args()

    per_protein_path = os.path.join(args.results_dir, "per_protein_bpsl.csv")
    if not os.path.exists(per_protein_path):
        print(f"ERROR: {per_protein_path} not found.")
        print(f"Run 'python -m bps.bps_process' first to generate results.")
        sys.exit(1)

    print("Loading per-protein results...")
    results = load_per_protein(per_protein_path)
    print(f"  Loaded {len(results)} proteins")

    write_compilation_report(results, args.output)

    # Quick diagnostic print
    alphas = [r['frac_alpha'] for r in results]
    betas = [r['frac_beta'] for r in results]
    print(f"\nSS diagnostic:")
    print(f"  Mean alpha: {np.mean(alphas):.3f}")
    print(f"  Mean beta:  {np.mean(betas):.3f}")
    print(f"  Alpha>35%:  {sum(1 for a in alphas if a >= 0.35)}")
    print(f"  Beta>25%:   {sum(1 for b in betas if b >= 0.25)}")

    n_other = sum(1 for r in results if r['fold_class'] == 'other')
    print(f"  Fold 'other': {n_other}/{len(results)} "
          f"({n_other/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
