"""AlphaFold proteome BPS/L processing pipeline.

Fast mmCIF parser (~40x faster than BioPython for bulk processing).
Extracts only backbone N, CA, C atoms, computes (phi, psi) from atomic
coordinates, then BPS/L and secondary structure composition.

Outputs:
  results/per_protein_bpsl.csv  — per-protein BPS/L, SS fractions, fold class
  results/per_organism.csv      — per-organism mean/std/CV

Usage:
    python -m bps.bps_process                      # process all organisms
    python -m bps.bps_process --organism ecoli      # process one
    python -m bps.bps_process --pdb-only            # process PDB cache only
"""

import argparse
import csv
import gzip
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bps.superpotential import (
    assign_basin,
    build_superpotential,
    compute_bps_per_residue,
    lookup_W_batch,
)
from bps.bps_download import ORGANISMS

_AF_CACHE = os.path.join(_PROJECT_ROOT, "data", "alphafold_cache")
_PDB_CACHE = os.path.join(_PROJECT_ROOT, "data", "pdb_cache")
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")

# pLDDT threshold for "well-modeled" residues
PLDDT_THRESHOLD = 85.0

# Minimum chain length to include
MIN_LENGTH = 50


# ---------------------------------------------------------------------------
# Fast CIF parser — backbone atoms only
# ---------------------------------------------------------------------------

def _parse_cif_backbone(path: str) -> Optional[List[Dict]]:
    """Parse an mmCIF file, extracting backbone N/CA/C atoms for chain A.

    Returns a list of dicts, one per residue, each with keys:
        resnum (int), resname (str), N (ndarray), CA (ndarray), C (ndarray),
        plddt (float)

    Returns None if parsing fails or chain A not found.

    CRITICAL: Extracts ONLY chain A (auth_asym_id). Multi-chain extraction
    was a confirmed bug in earlier versions.
    """
    opener = gzip.open if path.endswith('.gz') else open
    try:
        with opener(path, 'rt') as f:
            lines = f.readlines()
    except (OSError, gzip.BadGzipFile):
        return None

    # Find _atom_site column indices
    col_names = []
    in_atom_site = False
    data_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('_atom_site.'):
            if not in_atom_site:
                in_atom_site = True
                col_names = []
            col_names.append(stripped.split('.')[1])
        elif in_atom_site and not stripped.startswith('_'):
            data_start = i
            break

    if not col_names:
        return None

    # Map column names to indices
    try:
        col_group = col_names.index('group_PDB')
        col_atom = col_names.index('label_atom_id')
        col_resname = col_names.index('label_comp_id')
        col_chain = col_names.index('auth_asym_id')
        col_resnum = col_names.index('auth_seq_id')
        col_x = col_names.index('Cartn_x')
        col_y = col_names.index('Cartn_y')
        col_z = col_names.index('Cartn_z')
        col_bfactor = col_names.index('B_iso_or_equiv')
    except ValueError:
        # Try alternative column names
        try:
            col_chain = col_names.index('label_asym_id')
        except ValueError:
            return None
        try:
            col_resnum = col_names.index('label_seq_id')
        except ValueError:
            return None

    # Parse ATOM records for chain A backbone
    residues = {}  # resnum -> {resname, N, CA, C, plddt}
    target_atoms = {'N', 'CA', 'C'}

    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('#') or line.startswith('loop_'):
            if line.startswith('#'):
                break
            continue

        parts = line.split()
        if len(parts) < len(col_names):
            continue

        if parts[col_group] != 'ATOM':
            continue

        chain = parts[col_chain]
        if chain != 'A':
            continue

        atom_name = parts[col_atom]
        if atom_name not in target_atoms:
            continue

        try:
            resnum = int(parts[col_resnum])
            x = float(parts[col_x])
            y = float(parts[col_y])
            z = float(parts[col_z])
            bfactor = float(parts[col_bfactor])
        except (ValueError, IndexError):
            continue

        resname = parts[col_resname]

        if resnum not in residues:
            residues[resnum] = {
                'resnum': resnum,
                'resname': resname,
                'N': None, 'CA': None, 'C': None,
                'plddt': bfactor,
            }

        residues[resnum][atom_name] = np.array([x, y, z])

    if not residues:
        return None

    # Sort by residue number and filter complete residues
    sorted_res = []
    for resnum in sorted(residues.keys()):
        r = residues[resnum]
        if r['N'] is not None and r['CA'] is not None and r['C'] is not None:
            sorted_res.append(r)

    return sorted_res if sorted_res else None


def _compute_dihedral(p1: NDArray, p2: NDArray, p3: NDArray, p4: NDArray) -> float:
    """Compute dihedral angle defined by four points.

    Returns angle in RADIANS in range (-pi, pi].

    Uses the atan2 formula for numerical stability:
        phi = atan2(dot(n1 x n2, b2/|b2|), dot(n1, n2))
    where b_i = p_{i+1} - p_i, n1 = b1 x b2, n2 = b2 x b3.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize b2 for the projection
    b2_norm = np.linalg.norm(b2)
    if b2_norm < 1e-10:
        return 0.0
    b2_hat = b2 / b2_norm

    # m1 = n1 x b2_hat (in the plane perpendicular to b2)
    m1 = np.cross(n1, b2_hat)

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return math.atan2(y, x)


def extract_dihedrals_cif(residues: List[Dict]) -> List[Dict]:
    """Compute (phi, psi) for each residue from backbone coordinates.

    phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)
    psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})

    Returns list of dicts with keys: resnum, resname, phi, psi, plddt.
    Angles are in RADIANS. phi is None for first residue; psi is None
    for last residue.
    """
    result = []
    n = len(residues)

    for i in range(n):
        r = residues[i]
        phi = None
        psi = None

        # phi = dihedral(C_{i-1}, N_i, CA_i, C_i)
        if i > 0:
            prev = residues[i - 1]
            phi = _compute_dihedral(prev['C'], r['N'], r['CA'], r['C'])

        # psi = dihedral(N_i, CA_i, C_i, N_{i+1})
        if i < n - 1:
            nxt = residues[i + 1]
            psi = _compute_dihedral(r['N'], r['CA'], r['C'], nxt['N'])

        result.append({
            'resnum': r['resnum'],
            'resname': r['resname'],
            'phi': phi,
            'psi': psi,
            'plddt': r['plddt'],
        })

    return result


def compute_protein_stats(
    name: str,
    dihedrals: List[Dict],
    W_grid: NDArray,
    phi_grid: NDArray,
    psi_grid: NDArray,
    plddt_threshold: float = 0.0,
) -> Optional[Dict]:
    """Compute BPS/L and SS composition for one protein.

    Parameters
    ----------
    name : str
        Protein identifier.
    dihedrals : list of dict
        From extract_dihedrals_cif(). Angles in RADIANS.
    W_grid, phi_grid, psi_grid : ndarray
        Superpotential from build_superpotential().
    plddt_threshold : float
        Skip residues below this pLDDT (0 = no filter).

    Returns
    -------
    dict or None
        Per-protein stats, or None if too few valid residues.
    """
    valid = []
    for r in dihedrals:
        if r['phi'] is None or r['psi'] is None:
            continue
        if plddt_threshold > 0 and r.get('plddt', 100) < plddt_threshold:
            continue
        valid.append(r)

    if len(valid) < 10:
        return None

    phi_arr = np.array([r['phi'] for r in valid])
    psi_arr = np.array([r['psi'] for r in valid])

    W_vals = lookup_W_batch(W_grid, phi_grid, psi_grid, phi_arr, psi_arr)
    bps_l = compute_bps_per_residue(W_vals)

    # Basin assignment — convert to DEGREES for assign_basin()
    # CRITICAL: assign_basin expects DEGREES, internal angles are RADIANS
    counts = {'alpha': 0, 'beta': 0, 'ppII': 0, 'alphaL': 0, 'other': 0}
    for r in valid:
        phi_deg = math.degrees(r['phi'])
        psi_deg = math.degrees(r['psi'])
        basin = assign_basin(phi_deg, psi_deg)
        counts[basin] += 1

    n_valid = len(valid)
    frac_alpha = counts['alpha'] / n_valid
    frac_beta = counts['beta'] / n_valid
    frac_ppII = counts['ppII'] / n_valid
    frac_alphaL = counts['alphaL'] / n_valid
    frac_other = counts['other'] / n_valid

    # Fold-class assignment from computed SS fractions
    fold_class = classify_fold(frac_alpha, frac_beta)

    mean_plddt = np.mean([r.get('plddt', 0) for r in valid])

    return {
        'name': name,
        'length': len(dihedrals),
        'n_valid': n_valid,
        'bps_l': bps_l,
        'frac_alpha': frac_alpha,
        'frac_beta': frac_beta,
        'frac_ppII': frac_ppII,
        'frac_alphaL': frac_alphaL,
        'frac_other': frac_other,
        'fold_class': fold_class,
        'mean_plddt': mean_plddt,
        'phi_arr': phi_arr,
        'psi_arr': psi_arr,
    }


def classify_fold(frac_alpha: float, frac_beta: float) -> str:
    """Classify fold from computed SS fractions.

    SCOP-like classification. NEVER hardcoded — always derived from
    actual dihedral-based basin assignment.
    """
    if frac_alpha >= 0.35 and frac_beta < 0.10:
        return "all-alpha"
    if frac_beta >= 0.25 and frac_alpha < 0.15:
        return "all-beta"
    if frac_alpha >= 0.20 and frac_beta >= 0.15:
        return "alpha/beta"
    if frac_alpha >= 0.10 and frac_beta >= 0.10:
        return "alpha+beta"
    return "other"


# ---------------------------------------------------------------------------
# Organism-level processing
# ---------------------------------------------------------------------------

def process_organism(
    organism_id: str,
    species_name: str,
    W_grid: NDArray,
    phi_grid: NDArray,
    psi_grid: NDArray,
    cache_dir: str = _AF_CACHE,
    plddt_threshold: float = PLDDT_THRESHOLD,
    max_structures: int = 0,
) -> List[Dict]:
    """Process all CIF files for one organism.

    Returns list of per-protein result dicts.
    """
    org_dir = os.path.join(cache_dir, organism_id)
    if not os.path.isdir(org_dir):
        print(f"  {organism_id}: cache directory not found, skipping")
        return []

    cif_files = sorted(f for f in os.listdir(org_dir)
                       if f.endswith('.cif') or f.endswith('.cif.gz'))
    if max_structures > 0:
        cif_files = cif_files[:max_structures]

    results = []
    n_fail = 0
    for i, fname in enumerate(cif_files):
        if (i + 1) % 500 == 0:
            print(f"  {organism_id}: processed {i+1}/{len(cif_files)}, "
                  f"{len(results)} valid")

        cif_path = os.path.join(org_dir, fname)
        protein_name = fname.replace('.cif.gz', '').replace('.cif', '')

        backbone = _parse_cif_backbone(cif_path)
        if backbone is None:
            n_fail += 1
            continue

        if len(backbone) < MIN_LENGTH:
            continue

        dihedrals = extract_dihedrals_cif(backbone)
        stats = compute_protein_stats(
            protein_name, dihedrals, W_grid, phi_grid, psi_grid,
            plddt_threshold=plddt_threshold,
        )
        if stats is not None:
            stats['organism'] = organism_id
            stats['species'] = species_name
            results.append(stats)

    print(f"  {organism_id}: {len(results)}/{len(cif_files)} structures "
          f"({n_fail} parse failures)")
    return results


def process_pdb_cache(
    W_grid: NDArray,
    phi_grid: NDArray,
    psi_grid: NDArray,
    cache_dir: str = _PDB_CACHE,
) -> List[Dict]:
    """Process all PDB files in the PDB cache (for testing/validation).

    Uses BioPython extraction (same as validate_pdb.py) to provide
    a ground-truth comparison for the CIF parser.
    """
    from bps.extract import extract_dihedrals_pdb

    pdb_files = sorted(f for f in os.listdir(cache_dir) if f.endswith('.pdb'))
    results = []

    for fname in pdb_files:
        pdb_id = fname.replace('.pdb', '').upper()
        pdb_path = os.path.join(cache_dir, fname)

        try:
            residues = extract_dihedrals_pdb(pdb_path, chain_id='A')
        except (ValueError, Exception):
            continue

        if len(residues) < MIN_LENGTH:
            continue

        stats = compute_protein_stats(
            pdb_id, residues, W_grid, phi_grid, psi_grid,
            plddt_threshold=0,
        )
        if stats is not None:
            stats['organism'] = 'PDB'
            stats['species'] = 'Experimental'
            results.append(stats)

    print(f"  PDB cache: {len(results)}/{len(pdb_files)} structures")
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_per_protein_csv(results: List[Dict], path: str) -> None:
    """Write per-protein BPS/L results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        'name', 'organism', 'species', 'length', 'n_valid',
        'bps_l', 'frac_alpha', 'frac_beta', 'frac_ppII', 'frac_alphaL',
        'frac_other', 'fold_class', 'mean_plddt',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"  Wrote {len(results)} rows to {path}")


def write_per_organism_csv(results: List[Dict], path: str) -> None:
    """Write per-organism summary to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Group by organism
    by_org = {}
    for r in results:
        org = r.get('organism', 'unknown')
        if org not in by_org:
            by_org[org] = {'species': r.get('species', ''), 'vals': []}
        by_org[org]['vals'].append(r['bps_l'])

    rows = []
    for org_id in sorted(by_org.keys()):
        vals = np.array(by_org[org_id]['vals'])
        rows.append({
            'organism': org_id,
            'species': by_org[org_id]['species'],
            'n_structures': len(vals),
            'mean_bpsl': f"{np.mean(vals):.3f}",
            'std_bpsl': f"{np.std(vals):.3f}",
            'cv_pct': f"{np.std(vals)/np.mean(vals)*100:.1f}" if np.mean(vals) > 0 else "0.0",
            'median_bpsl': f"{np.median(vals):.3f}",
        })

    # Cross-organism summary
    org_means = [float(row['mean_bpsl']) for row in rows]
    if len(org_means) > 1:
        grand_mean = np.mean(org_means)
        grand_std = np.std(org_means)
        cross_cv = grand_std / grand_mean * 100 if grand_mean > 0 else 0
        rows.append({
            'organism': 'CROSS_ORGANISM',
            'species': f'N={len(org_means)} organisms',
            'n_structures': sum(len(by_org[o]['vals']) for o in by_org),
            'mean_bpsl': f"{grand_mean:.3f}",
            'std_bpsl': f"{grand_std:.3f}",
            'cv_pct': f"{cross_cv:.1f}",
            'median_bpsl': f"{np.median(org_means):.3f}",
        })

    fieldnames = ['organism', 'species', 'n_structures', 'mean_bpsl',
                  'std_bpsl', 'cv_pct', 'median_bpsl']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Wrote {len(rows)} rows to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process AlphaFold proteomes: compute BPS/L")
    parser.add_argument("--organism", type=str, default=None,
                        help="Short name (e.g., 'ecoli'). Default: all.")
    parser.add_argument("--pdb-only", action="store_true",
                        help="Process PDB cache only (no AlphaFold).")
    parser.add_argument("--plddt", type=float, default=PLDDT_THRESHOLD,
                        help=f"pLDDT threshold (default {PLDDT_THRESHOLD}).")
    parser.add_argument("--max-per-organism", type=int, default=0,
                        help="Max structures per organism (0=all).")
    parser.add_argument("--results-dir", type=str, default=_RESULTS_DIR)
    args = parser.parse_args()

    print("AlphaFold Proteome BPS/L Processing Pipeline")
    print("=" * 55)
    print()

    print("Building superpotential W(phi, psi)...")
    W_grid, phi_grid, psi_grid = build_superpotential()
    print(f"  W range: [{W_grid.min():.3f}, {W_grid.max():.3f}]")
    print()

    all_results = []

    if args.pdb_only:
        print("Processing PDB cache...")
        results = process_pdb_cache(W_grid, phi_grid, psi_grid)
        all_results.extend(results)
    else:
        targets = ORGANISMS
        if args.organism:
            targets = [(n, oid, sp) for n, oid, sp in ORGANISMS
                        if n == args.organism]
            if not targets:
                print(f"Unknown organism '{args.organism}'")
                sys.exit(1)

        for name, org_id, species in targets:
            print(f"\n[{name}] {species}")
            results = process_organism(
                org_id, species, W_grid, phi_grid, psi_grid,
                plddt_threshold=args.plddt,
                max_structures=args.max_per_organism,
            )
            all_results.extend(results)

    if not all_results:
        print("\nNo results. Check that data exists in data/alphafold_cache/ "
              "or data/pdb_cache/.")
        sys.exit(1)

    # Write outputs
    print(f"\nTotal proteins processed: {len(all_results)}")
    print()

    per_protein_path = os.path.join(args.results_dir, "per_protein_bpsl.csv")
    per_organism_path = os.path.join(args.results_dir, "per_organism.csv")

    write_per_protein_csv(all_results, per_protein_path)
    write_per_organism_csv(all_results, per_organism_path)

    # Print summary
    vals = np.array([r['bps_l'] for r in all_results])
    print(f"\nOverall: mean={np.mean(vals):.3f}, "
          f"std={np.std(vals):.3f}, "
          f"CV={np.std(vals)/np.mean(vals)*100:.1f}%")

    # Fold class breakdown
    print("\nFold-class breakdown:")
    fold_counts = {}
    for r in all_results:
        fc = r['fold_class']
        if fc not in fold_counts:
            fold_counts[fc] = []
        fold_counts[fc].append(r['bps_l'])

    print(f"  {'Class':<14s} {'N':>6s} {'Mean':>8s} {'Std':>8s} {'CV%':>6s}")
    for fc in ['all-alpha', 'all-beta', 'alpha/beta', 'alpha+beta', 'other']:
        if fc in fold_counts and fold_counts[fc]:
            fv = np.array(fold_counts[fc])
            m = np.mean(fv)
            s = np.std(fv)
            cv = s / m * 100 if m > 0 else 0
            print(f"  {fc:<14s} {len(fv):6d} {m:8.3f} {s:8.3f} {cv:5.1f}%")

    # SS fraction sanity check — diagnostic for fold classification bugs
    alphas = [r['frac_alpha'] for r in all_results]
    betas = [r['frac_beta'] for r in all_results]
    print(f"\nSS fraction diagnostic:")
    print(f"  Mean alpha: {np.mean(alphas):.3f}")
    print(f"  Mean beta:  {np.mean(betas):.3f}")
    print(f"  Alpha>35%:  {sum(1 for a in alphas if a >= 0.35)}")
    print(f"  Beta>25%:   {sum(1 for b in betas if b >= 0.25)}")

    n_other = sum(1 for r in all_results if r['fold_class'] == 'other')
    if n_other > len(all_results) * 0.90:
        print(f"\n  WARNING: {n_other}/{len(all_results)} classified as 'other'!")
        print(f"  Check dihedral extraction — likely radians/degrees mismatch.")
        print(f"  assign_basin() expects DEGREES. If mean alpha ~0 and mean beta ~0,")
        print(f"  angles are being passed in radians instead of degrees.")

    print("\nDone.")


if __name__ == "__main__":
    main()
