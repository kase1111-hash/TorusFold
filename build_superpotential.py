"""
TorusFold: Build Superpotential W(φ, ψ)
========================================
Constructs the superpotential landscape W = -ln P(φ, ψ) from all
AlphaFold structures in the cache. This is the foundational data
product that all other analysis scripts depend on.

Procedure:
  1. Extract (φ, ψ) dihedral angles from all CIF files
  2. Bin into a 2D histogram on the Ramachandran plane
  3. Build W from the canonical shared module (bps/superpotential.py,
     von Mises mixture, W = -√P)
  4. Save W grid + raw histogram as superpotential_W.npz

The output file contains:
  - 'grid': the W(φ, ψ) array (grid_size × grid_size)
  - 'grid_size': integer resolution
  - 'n_angles': total angles used
  - 'n_proteins': proteins processed
  - 'n_organisms': organisms included

Usage:
  python build_superpotential.py --data alphafold_cache --output results
         [--grid-size 360] [--sample N] [--plddt-min 70]

Reads:  alphafold_cache/*/  (organism subdirectories with .cif files)
Writes: results/superpotential_W.npz
"""

import os
import sys
import math
import time
import argparse
import numpy as np
from pathlib import Path

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
    """Extract (phi, psi) pairs from a CIF file. Returns list of (phi, psi) in radians."""
    st = gemmi.read_structure(str(filepath))
    if len(st) == 0 or len(st[0]) == 0:
        return []
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
        return []

    phi_psi = []
    for i in range(1, len(residues) - 1):
        if residues[i]['plddt'] < plddt_min:
            continue
        phi = -_dihedral_angle(residues[i-1]['C'], residues[i]['N'],
                               residues[i]['CA'], residues[i]['C'])
        psi = -_dihedral_angle(residues[i]['N'], residues[i]['CA'],
                               residues[i]['C'], residues[i+1]['N'])
        phi_psi.append((phi, psi))
    return phi_psi


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
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build superpotential W(φ, ψ) from AlphaFold structures")
    parser.add_argument("--data", default="alphafold_cache",
                        help="AlphaFold data directory")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--grid-size", type=int, default=360,
                        help="Histogram resolution (default: 360)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Max proteins per organism (0=all)")
    parser.add_argument("--plddt-min", type=float, default=70.0,
                        help="Minimum pLDDT for residue inclusion (default: 70)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold: Build Superpotential W(φ, ψ)")
    print("=" * 60)

    # Discover files
    organisms = discover_files(args.data)
    total_files = sum(len(f) for f in organisms.values())
    print(f"  Data: {total_files} files across {len(organisms)} organisms")
    print(f"  Grid: {args.grid_size}×{args.grid_size}")
    print(f"  pLDDT min: {args.plddt_min}")

    # Build histogram
    grid_size = args.grid_size
    scale = grid_size / 360.0
    histogram = np.zeros((grid_size, grid_size), dtype=np.float64)

    n_angles_total = 0
    n_proteins = 0
    n_skipped = 0
    MAX_FILE_SIZE = 5 * 1024 * 1024

    t_start = time.time()

    for org_name, files in sorted(organisms.items()):
        rng = np.random.default_rng(hash(org_name) % (2**31))
        if args.sample > 0 and len(files) > args.sample:
            idx = rng.choice(len(files), args.sample, replace=False)
            files = [files[i] for i in idx]

        print(f"  {org_name} ({len(files)} files)...", end=" ", flush=True)
        org_angles = 0
        org_proteins = 0
        org_errors = 0

        for fi, filepath in enumerate(files):
            try:
                fsize = os.path.getsize(filepath)
                if fsize > MAX_FILE_SIZE:
                    n_skipped += 1
                    continue

                angles = extract_angles(filepath, plddt_min=args.plddt_min)
                if len(angles) < 10:
                    n_skipped += 1
                    continue

                for phi, psi in angles:
                    phi_d = math.degrees(phi)
                    psi_d = math.degrees(psi)
                    gi = int(round((phi_d + 180) * scale)) % grid_size
                    gj = int(round((psi_d + 180) * scale)) % grid_size
                    histogram[gi, gj] += 1

                org_angles += len(angles)
                org_proteins += 1

            except Exception as e:
                n_skipped += 1
                org_errors += 1
                if org_errors <= 3:
                    fname = os.path.basename(filepath)
                    print(f"\n    WARNING: {fname}: {type(e).__name__}: {e}",
                          end="", flush=True)

            # Progress for large organisms
            if (fi + 1) % 2000 == 0:
                print(f"\n    [{fi+1}/{len(files)}] "
                      f"{org_proteins} proteins, {org_angles} angles...",
                      end="", flush=True)

        n_angles_total += org_angles
        n_proteins += org_proteins
        error_note = f" ({org_errors} errors)" if org_errors > 0 else ""
        print(f"{org_proteins} proteins, {org_angles:,} angles{error_note}")

    elapsed = time.time() - t_start
    print(f"\n  Total: {n_proteins:,} proteins, {n_angles_total:,} angles, "
          f"{n_skipped} skipped")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Build W from canonical shared module (von Mises mixture, W = -sqrt(P))
    # rather than from the histogram.  The histogram is still saved for
    # downstream epsilon-sweep tests (torus_distance_control.py).
    total_counts = histogram.sum()
    if total_counts == 0:
        print("ERROR: No angles collected. Check data directory.")
        sys.exit(1)

    from bps.superpotential import build_superpotential as _build_shared
    W_grid, _phi_grid, _psi_grid = _build_shared(grid_size)

    # Save
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "superpotential_W.npz")
    np.savez(out_path,
             grid=W_grid,
             histogram=histogram,
             grid_size=grid_size,
             n_angles=n_angles_total,
             n_proteins=n_proteins,
             n_organisms=len(organisms),
             plddt_min=args.plddt_min)

    print(f"\n  Saved: {out_path}")
    print(f"  Grid shape: {W_grid.shape}")
    print(f"  W range: [{W_grid.min():.2f}, {W_grid.max():.2f}]")

    # Quick sanity stats
    # α basin: roughly φ ∈ [-100, -40], ψ ∈ [-60, 0]
    # β basin: roughly φ ∈ [-150, -80], ψ ∈ [100, 170]
    alpha_i = slice(int((80) * scale), int((140) * scale))    # φ+180
    alpha_j = slice(int((120) * scale), int((180) * scale))   # ψ+180
    beta_i = slice(int((30) * scale), int((100) * scale))
    beta_j = slice(int((280) * scale), int((350) * scale))

    w_alpha = float(np.mean(W_grid[alpha_i, alpha_j]))
    w_beta = float(np.mean(W_grid[beta_i, beta_j]))
    w_overall = float(np.mean(W_grid))

    print(f"\n  Basin depths (lower = more populated):")
    print(f"    α basin mean W:  {w_alpha:.2f}")
    print(f"    β basin mean W:  {w_beta:.2f}")
    print(f"    Overall mean W:  {w_overall:.2f}")
    print(f"    α depth:         {w_overall - w_alpha:.2f} below mean")
    print(f"    β depth:         {w_overall - w_beta:.2f} below mean")

    print("\n  Done. All downstream scripts can now run.")


if __name__ == "__main__":
    main()
