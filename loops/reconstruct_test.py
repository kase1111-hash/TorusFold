"""End-to-end reconstruction test (Priority 7).

For test proteins:
  1. Extract experimental (phi, psi)
  2. Identify loop regions (transitions between alpha and beta basins)
  3. Replace loop (phi, psi) with nearest canonical family centroid
  4. Keep SS element (phi, psi) at basin centers
  5. Reconstruct via forward kinematics
  6. Compute RMSD vs experimental backbone

This answers: how much accuracy do you lose by using canonical loop
paths instead of actual conformations?
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bps.superpotential import assign_basin, build_superpotential
from bps.extract import extract_dihedrals_pdb
from loops.taxonomy import (
    _assign_basins_for_chain,
    _circular_interp,
    extract_loops,
    compute_distance_matrix,
    cluster_loops,
    _compute_cluster_stats,
    resample_path,
    torus_path_distance,
    SHORT_MAX,
    MIN_FLANK,
)
from loops.forward_kinematics import (
    reconstruct_backbone,
    compute_backbone_rmsd,
    extract_all_dihedrals_pdb,
    OMEGA_TRANS,
)


# Basin centers in radians (for SS element idealization)
BASIN_CENTERS = {
    "alpha": (math.radians(-63), math.radians(-43)),
    "beta": (math.radians(-120), math.radians(130)),
    "ppII": (math.radians(-75), math.radians(150)),
    "alphaL": (math.radians(57), math.radians(47)),
}


def _build_centroid_library(
    cache_dir: str,
    W_grid: np.ndarray,
    phi_grid: np.ndarray,
    psi_grid: np.ndarray,
) -> List[Dict]:
    """Build canonical loop family centroids from all cached PDB structures.

    Returns list of family dicts with centroid_phi, centroid_psi (resampled
    arrays), direction, and member count.
    """
    from loops.taxonomy import _get_pdb_entries

    entries = _get_pdb_entries(cache_dir)
    all_loops: List[Dict] = []

    for pdb_id, chain_id in entries:
        pdb_path = os.path.join(cache_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            continue
        try:
            residues = extract_dihedrals_pdb(pdb_path, chain_id)
        except (ValueError, Exception):
            continue
        if len(residues) < 20:
            continue
        basins = _assign_basins_for_chain(residues)
        loops = extract_loops(residues, basins)
        for lp in loops:
            lp["pdb_id"] = pdb_id
        all_loops.extend(loops)

    # Cluster short loops
    short_loops = [lp for lp in all_loops if lp["loop_len"] <= SHORT_MAX]
    if len(short_loops) < 5:
        return []

    D = compute_distance_matrix(short_loops)
    labels, eps, n_clusters = cluster_loops(D)
    stats = _compute_cluster_stats(
        short_loops, labels, W_grid, phi_grid, psi_grid
    )

    families = []
    for st in stats:
        families.append({
            "cluster_id": st["cluster_id"],
            "centroid_phi": st["centroid_phi"],
            "centroid_psi": st["centroid_psi"],
            "n_members": st["n_members"],
            "tight": st["tight"],
            "direction_counts": st["direction_counts"],
        })

    return families


def _find_nearest_centroid(
    phi_path: np.ndarray,
    psi_path: np.ndarray,
    families: List[Dict],
    n_pts: int = 20,
) -> Optional[Dict]:
    """Find the nearest canonical family centroid to a given loop path."""
    if not families:
        return None

    phi_r, psi_r = resample_path(phi_path, psi_path, n_pts)
    best_dist = float("inf")
    best_family = None

    for fam in families:
        d = torus_path_distance(phi_r, psi_r,
                                fam["centroid_phi"], fam["centroid_psi"])
        if d < best_dist:
            best_dist = d
            best_family = fam

    return best_family


def reconstruct_with_canonical_loops(
    residues: List[Dict],
    basins: List[Optional[str]],
    loops: List[Dict],
    families: List[Dict],
    phi_exp: List[float],
    psi_exp: List[float],
    omega_exp: List[float],
) -> Tuple[np.ndarray, Dict]:
    """Reconstruct backbone using canonical loop paths + basin center SS elements.

    Strategy:
      - SS residues (alpha/beta): replace (phi, psi) with basin center values
      - Loop residues: replace with nearest canonical family centroid path
      - Other residues: keep experimental (phi, psi)
      - Always use experimental omega angles

    Returns (recon_coords, stats_dict).
    """
    L = len(residues)
    phi_recon = list(phi_exp)
    psi_recon = list(psi_exp)

    n_ss_replaced = 0
    n_loop_replaced = 0
    n_kept = 0

    # Mark which residues belong to loops
    loop_residue_indices: Dict[int, int] = {}  # residue_idx -> loop_idx
    for li, lp in enumerate(loops):
        # Find the loop residues in the full chain
        loop_start_resnum = lp["residues"][0]["resnum"]
        for j, lr in enumerate(lp["residues"]):
            for k, r in enumerate(residues):
                if r["resnum"] == lr["resnum"]:
                    loop_residue_indices[k] = li
                    break

    # Replace SS element angles with basin centers
    for i in range(L):
        if i in loop_residue_indices:
            continue  # handled below
        b = basins[i]
        if b in BASIN_CENTERS:
            phi_recon[i] = BASIN_CENTERS[b][0]
            psi_recon[i] = BASIN_CENTERS[b][1]
            n_ss_replaced += 1
        else:
            n_kept += 1

    # Replace loop residue angles with nearest centroid
    processed_loops = set()
    for residue_idx, loop_idx in sorted(loop_residue_indices.items()):
        if loop_idx in processed_loops:
            continue
        processed_loops.add(loop_idx)

        lp = loops[loop_idx]
        nearest = _find_nearest_centroid(lp["phi_path"], lp["psi_path"], families)

        if nearest is not None:
            # Resample centroid path to match loop length
            centroid_phi = _circular_interp(nearest["centroid_phi"],
                                            lp["loop_len"])
            centroid_psi = _circular_interp(nearest["centroid_psi"],
                                            lp["loop_len"])

            # Apply to the correct residue positions
            loop_positions = []
            for lr in lp["residues"]:
                for k, r in enumerate(residues):
                    if r["resnum"] == lr["resnum"]:
                        loop_positions.append(k)
                        break

            for j, pos in enumerate(loop_positions):
                if j < len(centroid_phi):
                    phi_recon[pos] = centroid_phi[j]
                    psi_recon[pos] = centroid_psi[j]
                    n_loop_replaced += 1
        else:
            n_kept += lp["loop_len"]

    # Reconstruct backbone
    recon_coords = reconstruct_backbone(phi_recon, psi_recon, omega=omega_exp)

    stats = {
        "n_ss_replaced": n_ss_replaced,
        "n_loop_replaced": n_loop_replaced,
        "n_kept": n_kept,
        "n_total": L,
        "n_loops": len(loops),
    }

    return recon_coords, stats


def main() -> None:
    print("Priority 7: End-to-End Reconstruction Test")
    print("=" * 55)
    print()

    cache_dir = os.path.join(_PROJECT_ROOT, "data", "pdb_cache")
    output_dir = os.path.join(_PROJECT_ROOT, "output")

    # Build superpotential
    print("Building superpotential W(phi, psi)...")
    W_grid, phi_grid, psi_grid = build_superpotential()
    print()

    # Build centroid library from all cached structures
    print("Building canonical loop family library...")
    families = _build_centroid_library(cache_dir, W_grid, phi_grid, psi_grid)
    print(f"  Found {len(families)} loop families")
    print()

    # Select test proteins
    test_ids = ["1UBQ", "1AKI", "1UBI", "1EJG", "1A8O", "1D3Z",
                "1DIX", "1GYA", "1K6P", "3O5R"]
    test_structures = []
    for pdb_id in test_ids:
        path = os.path.join(cache_dir, f"{pdb_id}.pdb")
        if os.path.exists(path):
            test_structures.append((pdb_id, "A", path))

    if len(test_structures) < 5:
        # Fall back to whatever is available
        for fname in sorted(os.listdir(cache_dir)):
            if fname.endswith(".pdb") and not fname.startswith("SYN"):
                pid = fname.replace(".pdb", "").upper()
                if pid not in [t[0] for t in test_structures]:
                    test_structures.append(
                        (pid, "A", os.path.join(cache_dir, fname)))
                if len(test_structures) >= 10:
                    break

    print(f"Testing on {len(test_structures)} structures:")
    print()
    print(f"{'PDB':>6s}  {'Len':>4s}  {'Loops':>5s}  {'RMSD_exp':>8s}  "
          f"{'RMSD_can':>8s}  {'Delta':>8s}  {'SS_rep':>6s}  "
          f"{'Loop_rep':>8s}  {'Kept':>4s}")
    print("-" * 80)

    results = []
    for pdb_id, chain_id, pdb_path in test_structures:
        try:
            # Extract experimental geometry
            phi_exp, psi_exp, omega_exp, exp_coords = extract_all_dihedrals_pdb(
                pdb_path, chain_id
            )

            # Extract dihedrals for basin assignment
            residues = extract_dihedrals_pdb(pdb_path, chain_id)
            L = len(residues)

            # Ensure lengths match
            L_coords = len(exp_coords)
            if L != L_coords:
                L = min(L, L_coords)
                residues = residues[:L]
                phi_exp = phi_exp[:L]
                psi_exp = psi_exp[:L]
                omega_exp = omega_exp[:L]
                exp_coords = exp_coords[:L]

            basins = _assign_basins_for_chain(residues)
            loops = extract_loops(residues, basins)

            # RMSD with experimental dihedrals only (ideal geometry baseline)
            recon_exp = reconstruct_backbone(phi_exp, psi_exp, omega=omega_exp)
            rmsd_exp = compute_backbone_rmsd(recon_exp[:L], exp_coords[:L])

            # RMSD with canonical loop replacement
            recon_can, stats = reconstruct_with_canonical_loops(
                residues, basins, loops, families,
                phi_exp, psi_exp, omega_exp
            )
            rmsd_can = compute_backbone_rmsd(recon_can[:L], exp_coords[:L])

            delta = rmsd_can - rmsd_exp

            print(f"{pdb_id:>6s}  {L:>4d}  {len(loops):>5d}  "
                  f"{rmsd_exp:>8.3f}  {rmsd_can:>8.3f}  "
                  f"{delta:>+8.3f}  {stats['n_ss_replaced']:>6d}  "
                  f"{stats['n_loop_replaced']:>8d}  {stats['n_kept']:>4d}")

            results.append({
                "pdb_id": pdb_id,
                "length": L,
                "n_loops": len(loops),
                "rmsd_exp": rmsd_exp,
                "rmsd_canonical": rmsd_can,
                "delta": delta,
                **stats,
            })

        except Exception as e:
            print(f"{pdb_id:>6s}  ERROR: {e}")

    print()
    if results:
        exp_rmsds = [r["rmsd_exp"] for r in results]
        can_rmsds = [r["rmsd_canonical"] for r in results]
        deltas = [r["delta"] for r in results]

        print("Summary:")
        print(f"  Mean RMSD (experimental dihedrals): {np.mean(exp_rmsds):.3f} A")
        print(f"  Mean RMSD (canonical loops):        {np.mean(can_rmsds):.3f} A")
        print(f"  Mean delta (canonical - exp):       {np.mean(deltas):+.3f} A")
        print()
        print("Interpretation:")
        mean_delta = np.mean(deltas)
        if abs(mean_delta) < 1.0:
            print("  Canonical loop paths add < 1 A RMSD on average.")
            print("  Loop conformations are well-captured by family centroids.")
        elif abs(mean_delta) < 3.0:
            print("  Canonical loop paths add 1-3 A RMSD.")
            print("  Moderate accuracy loss from using canonical paths.")
        else:
            print(f"  Canonical loop paths add {abs(mean_delta):.1f} A RMSD.")
            print("  Significant accuracy loss; more loop families may be needed.")

    # Write report
    report_path = os.path.join(output_dir, "reconstruction_test_report.md")
    _write_report(results, report_path)

    print()
    print("Priority 7 complete.")


def _write_report(results: List[Dict], output_path: str) -> None:
    """Write reconstruction test report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# End-to-End Reconstruction Test Report",
        "",
        f"**Structures tested:** {len(results)}",
        "",
        "## Method",
        "",
        "1. Extract experimental (phi, psi, omega) from crystal structure",
        "2. Identify loop regions (alpha->beta / beta->alpha transitions)",
        "3. Replace SS element angles with basin centers",
        "4. Replace loop angles with nearest canonical family centroid",
        "5. Reconstruct backbone via forward kinematics",
        "6. Compute RMSD vs experimental backbone",
        "",
        "## Results",
        "",
        "| PDB | Length | Loops | RMSD (exp) | RMSD (canonical) | Delta |",
        "|-----|--------|-------|------------|------------------|-------|",
    ]

    for r in results:
        lines.append(
            f"| {r['pdb_id']} | {r['length']} | {r['n_loops']} "
            f"| {r['rmsd_exp']:.3f} | {r['rmsd_canonical']:.3f} "
            f"| {r['delta']:+.3f} |"
        )

    if results:
        exp_rmsds = [r["rmsd_exp"] for r in results]
        can_rmsds = [r["rmsd_canonical"] for r in results]
        deltas = [r["delta"] for r in results]
        lines.extend([
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean RMSD (experimental) | {np.mean(exp_rmsds):.3f} A |",
            f"| Mean RMSD (canonical) | {np.mean(can_rmsds):.3f} A |",
            f"| Mean delta | {np.mean(deltas):+.3f} A |",
            f"| Max delta | {max(deltas):+.3f} A |",
            "",
            "## Interpretation",
            "",
            f"Using canonical loop path centroids instead of actual loop ",
            f"conformations adds an average of {abs(np.mean(deltas)):.1f} A RMSD.",
            "",
        ])

    lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report written: {output_path}")


if __name__ == "__main__":
    main()
