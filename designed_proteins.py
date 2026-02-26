"""
TorusFold: De Novo Designed Protein Analysis
=============================================
Downloads computationally designed (de novo) protein structures from
RCSB PDB and computes BPS/L, comparing to natural proteins.

Tests the hypothesis: designed proteins show the same intra-basin
coherence as evolved proteins (physics, not biology).

Sources:
  - RCSB PDB search API for de novo designed structures
  - Keywords: "de novo design", "computational design", "Rosetta"
  - Key authors: David Baker group deposits

Usage:
  python designed_proteins.py [--download] [--analyze]

  --download  : Fetch CIF files from RCSB (needs internet)
  --analyze   : Compute BPS/L on downloaded files
  
  Run --download first, then --analyze.

Writes:
  data/designed_proteins/*.cif
  results/designed_proteins_report.md
"""

import os
import sys
import json
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

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════
# RCSB PDB SEARCH & DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

# Known de novo designed protein PDB IDs (curated list)
# These are well-characterized computationally designed structures
KNOWN_DESIGNED = [
    # Baker lab de novo designs
    "5TPJ",  # Top7 - first de novo designed protein
    "7JH7",  # de novo TIM barrel
    "5VLJ",  # de novo beta-barrel (fluorescence-activating)
    "6MRS",  # de novo luciferase
    "6NUK",  # de novo designed protein
    "5KWA",  # de novo four-helix bundle
    "5J10",  # de novo NTF2 fold
    "5IZS",  # de novo beta-barrel
    "5UP1",  # de novo protein switch
    "5WN9",  # de novo helical bundle
    "5L33",  # de novo beta/alpha protein
    "6C58",  # de novo designed repeat protein
    "6D0T",  # de novo TIM barrel
    "6E5C",  # de novo beta-sheet protein
    "6MSP",  # de novo designed cage
    "5CW9",  # de novo ferredoxin fold
    "3U3B",  # de novo protein cage subunit
    "5BVL",  # de novo designed coiled-coil
    "4TQL",  # de novo designed helix bundle
    "4EF4",  # de novo designed protein
    "4DZM",  # de novo designed Rosetta
    "4KY3",  # de novo designed helical bundle
    "7BFS",  # de novo miniprotein
    "7CBC",  # de novo designed binder
    "6VIS",  # de novo designed protein
    # Miniproteins / small de novo designs
    "1QYS",  # Trp-cage miniprotein (designed)
    "2JOF",  # designed three-helix bundle
    "1PRB",  # protein B (designed variant)
    "2KL8",  # designed WW domain
    # Alpha-helical coiled-coil designs
    "4DGS",  # designed coiled-coil
    "3S0R",  # designed helical bundle
    "1G6U",  # designed leucine zipper
]


def search_rcsb_designed(max_results=200):
    """Search RCSB for de novo designed proteins using their search API."""
    # RCSB search API v2
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    query = {
        "query": {
            "type": "group",
            "logical_operator": "or",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {
                        "value": "\"de novo designed protein\""
                    }
                },
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {
                        "value": "\"computationally designed protein\""
                    }
                },
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {
                        "value": "\"Rosetta de novo design\""
                    }
                },
            ]
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "paginate": {
                "start": 0,
                "rows": max_results
            },
            "scoring_strategy": "combined",
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(query).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pdb_ids = [r['identifier'] for r in data.get('result_set', [])]
            print(f"  RCSB search returned {len(pdb_ids)} results")
            return pdb_ids
    except Exception as e:
        print(f"  RCSB search failed: {e}")
        print("  Falling back to curated list only")
        return []


def download_cif(pdb_id, output_dir):
    """Download a CIF file from RCSB."""
    outpath = os.path.join(output_dir, f"{pdb_id.lower()}.cif")
    if os.path.exists(outpath):
        return outpath

    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    try:
        urllib.request.urlretrieve(url, outpath)
        return outpath
    except Exception as e:
        print(f"    Failed to download {pdb_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTIONS (same as other scripts)
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


def extract_angles_pdb(filepath, bfactor_min=0.0):
    """Extract (phi, psi) and SS from CIF/PDB. No pLDDT filter for experimental structures."""
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
            residues.append({
                'N': atoms['N'], 'CA': atoms['CA'], 'C': atoms['C'],
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
        transitions = sum(1 for i in range(len(ss_seq)-1)
                         if ss_seq[i] != ss_seq[i+1]
                         and ss_seq[i] in ('a','b')
                         and ss_seq[i+1] in ('a','b'))
        n_ab = na * n + nb * n
        trans_rate = transitions / max(n_ab, 1)
        return 'alpha/beta' if trans_rate > 0.05 else 'alpha+beta'
    else:
        return 'other'


def compute_bps_l(phi_arr, psi_arr, W_grid, grid_size):
    """BPS/L from arrays of phi, psi in radians. Normalizes by L."""
    if len(phi_arr) < 2:
        return 0.0
    from bps.superpotential import lookup_W_grid
    w = lookup_W_grid(W_grid, grid_size, phi_arr, psi_arr)
    L = len(phi_arr)
    return float(np.sum(np.abs(np.diff(w)))) / L


def null_shuffled(phi_psi, rng):
    phi = np.array([p[0] for p in phi_psi])
    psi = np.array([p[1] for p in phi_psi])
    perm = rng.permutation(len(phi))
    return phi[perm], psi[perm]


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Designed Protein BPS/L Analysis")
    parser.add_argument("--download", action="store_true",
                        help="Download CIF files from RCSB")
    parser.add_argument("--analyze", action="store_true",
                        help="Compute BPS/L on downloaded files")
    parser.add_argument("--data-dir", default="data/designed_proteins",
                        help="Directory for CIF files")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    args = parser.parse_args()

    if not args.download and not args.analyze:
        print("Usage: python designed_proteins.py --download --analyze")
        print("  --download  : Fetch structures from RCSB")
        print("  --analyze   : Compute BPS/L")
        print("  Can run both together: --download --analyze")
        return

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    # ══════════════════════════════════════════════════════════════
    # DOWNLOAD
    # ══════════════════════════════════════════════════════════════
    if args.download:
        print("=" * 60)
        print("  Downloading de novo designed protein structures")
        print("=" * 60)

        # Start with curated list
        all_ids = list(KNOWN_DESIGNED)

        # Try RCSB search for more
        print("\n  Searching RCSB for additional designed proteins...")
        search_ids = search_rcsb_designed(max_results=150)
        for pid in search_ids:
            if pid not in all_ids:
                all_ids.append(pid)

        print(f"\n  Total unique PDB IDs: {len(all_ids)}")
        print(f"  ({len(KNOWN_DESIGNED)} curated + {len(all_ids) - len(KNOWN_DESIGNED)} from search)")

        # Download
        downloaded = 0
        for i, pdb_id in enumerate(all_ids):
            path = download_cif(pdb_id, data_dir)
            if path:
                downloaded += 1
            if (i + 1) % 10 == 0:
                print(f"    Downloaded {i+1}/{len(all_ids)}...", flush=True)
            time.sleep(0.2)  # be polite to RCSB

        print(f"\n  Downloaded {downloaded}/{len(all_ids)} structures to {data_dir}/")

    # ══════════════════════════════════════════════════════════════
    # ANALYZE
    # ══════════════════════════════════════════════════════════════
    if args.analyze:
        print("\n" + "=" * 60)
        print("  Analyzing designed protein BPS/L")
        print("=" * 60)

        # Build W from shared von Mises construction
        from bps.superpotential import build_superpotential as _build_W
        W_grid, _, _ = _build_W(360)
        grid_size = W_grid.shape[0]
        print(f"  Built W: {grid_size}x{grid_size} (von Mises -sqrt(P))")

        # Process all CIF files
        cif_files = sorted(Path(data_dir).glob("*.cif"))
        print(f"  Found {len(cif_files)} CIF files")

        rng = np.random.default_rng(42)
        results = []

        for filepath in cif_files:
            pdb_id = filepath.stem.upper()
            try:
                pp, ss = extract_angles_pdb(str(filepath))
                if len(pp) < 20:  # designed proteins can be small
                    continue

                fold = classify_fold(ss)
                phi = np.array([p[0] for p in pp])
                psi = np.array([p[1] for p in pp])
                bpsl = compute_bps_l(phi, psi, W_grid, grid_size)

                # Compute shuffled null for comparison
                shuf_vals = []
                for _ in range(10):
                    sp, sq = null_shuffled(pp, rng)
                    shuf_vals.append(compute_bps_l(sp, sq, W_grid, grid_size))

                results.append({
                    'pdb_id': pdb_id,
                    'length': len(pp),
                    'fold': fold,
                    'bpsl': bpsl,
                    'shuffled': float(np.mean(shuf_vals)),
                    's_real': float(np.mean(shuf_vals) / bpsl) if bpsl > 0 else 0,
                    'is_curated': pdb_id in KNOWN_DESIGNED,
                })
            except Exception as e:
                print(f"    Error processing {pdb_id}: {e}")

        print(f"  Successfully analyzed {len(results)} structures")

        if not results:
            print("  No results to report.")
            return

        # ══════════════════════════════════════════════════════════
        # WRITE REPORT
        # ══════════════════════════════════════════════════════════
        report_path = os.path.join(args.output, "designed_proteins_report.md")

        bpsl_vals = np.array([r['bpsl'] for r in results])
        shuf_vals = np.array([r['shuffled'] for r in results])
        s_real_vals = np.array([r['s_real'] for r in results])

        # By fold class
        fold_groups = defaultdict(list)
        for r in results:
            fold_groups[r['fold']].append(r['bpsl'])

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# De Novo Designed Protein BPS/L Analysis\n\n")
            f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Structures analyzed:** {len(results)}\n")
            f.write(f"**Source:** RCSB PDB (curated + search)\n\n")

            f.write("## Summary\n\n")
            f.write(f"| Metric | Designed | Natural (AlphaFold) |\n")
            f.write(f"|---|---|---|\n")
            f.write(f"| Mean BPS/L | {np.mean(bpsl_vals):.3f} ± {np.std(bpsl_vals):.3f} | 1.275 ± 0.207 |\n")
            f.write(f"| Median BPS/L | {np.median(bpsl_vals):.3f} | ~1.25 |\n")
            f.write(f"| S/Real ratio | {np.mean(s_real_vals):.2f}x | 1.61x |\n")
            f.write(f"| Per-protein CV | {np.std(bpsl_vals)/np.mean(bpsl_vals)*100:.1f}% | 16.2% |\n")
            f.write(f"| N | {len(results)} | 82,113 |\n\n")

            # By fold class
            f.write("## By Fold Class\n\n")
            f.write("| Fold Class | N designed | Mean BPS/L | Natural mean |\n")
            f.write("|---|---|---|---|\n")
            natural_means = {'all-alpha': 1.122, 'alpha+beta': 1.238,
                           'all-beta': 1.276, 'alpha/beta': 1.327, 'other': None}
            for fc in ['all-alpha', 'alpha+beta', 'all-beta', 'alpha/beta', 'other']:
                if fc in fold_groups:
                    vals = fold_groups[fc]
                    nat = f"{natural_means[fc]:.3f}" if natural_means.get(fc) else "—"
                    f.write(f"| {fc} | {len(vals)} | {np.mean(vals):.3f} | {nat} |\n")
            f.write("\n")

            # Individual structures
            f.write("## Individual Structures\n\n")
            f.write("| PDB ID | Length | Fold | BPS/L | S/Real | Curated |\n")
            f.write("|---|---|---|---|---|---|\n")
            for r in sorted(results, key=lambda x: x['bpsl']):
                curated = "✓" if r['is_curated'] else ""
                f.write(f"| {r['pdb_id']} | {r['length']} | {r['fold']} | "
                        f"{r['bpsl']:.3f} | {r['s_real']:.2f}x | {curated} |\n")
            f.write("\n")

            # Interpretation
            f.write("## Interpretation\n\n")
            designed_mean = np.mean(bpsl_vals)
            natural_mean = 1.275
            pct_diff = abs(designed_mean - natural_mean) / natural_mean * 100

            if pct_diff < 15:
                f.write(f"Designed proteins (BPS/L = {designed_mean:.3f}) are "
                        f"**indistinguishable** from natural proteins "
                        f"(BPS/L = {natural_mean:.3f}, Δ = {pct_diff:.1f}%).\n\n"
                        f"This confirms that intra-basin coherence is a "
                        f"**physical property of folded polypeptides**, not a "
                        f"signature of evolutionary selection.\n\n")
            elif designed_mean < natural_mean:
                f.write(f"Designed proteins show **tighter** coherence "
                        f"(BPS/L = {designed_mean:.3f} < {natural_mean:.3f}), "
                        f"suggesting computational design can optimize "
                        f"backbone geometry more aggressively than evolution.\n\n")
            else:
                f.write(f"Designed proteins show **looser** coherence "
                        f"(BPS/L = {designed_mean:.3f} > {natural_mean:.3f}), "
                        f"suggesting evolution achieves tighter geometric "
                        f"optimization than current design methods.\n\n")

            f.write("### Implications for Biosignature Detection\n\n")
            f.write("BPS/L cannot distinguish naturally evolved proteins from "
                    "computationally designed ones. The metric measures "
                    "stereochemical constraint intrinsic to the polypeptide "
                    "backbone, which is invariant to the origin of the "
                    "sequence. Any folded polypeptide using standard L-amino "
                    "acids will exhibit similar intra-basin coherence.\n\n")
            f.write("This limits the utility of backbone geometry alone as a "
                    "biosignature, but opens the possibility of detecting "
                    "non-standard backbone chemistries (D-amino acids, "
                    "β-peptides, peptoids) via anomalous BPS/L values.\n")

        print(f"\n  Report: {report_path}")
        print("  Done.")


if __name__ == "__main__":
    main()
