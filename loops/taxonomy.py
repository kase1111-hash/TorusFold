"""Loop path taxonomy on the Ramachandran torus.

Self-contained pipeline:
  1. Load PDB structures from cache (or download)
  2. Extract (phi, psi) trajectories per chain
  3. Assign Ramachandran basins per residue
  4. Find transitions between alpha and beta basins (loops)
  5. Extract loop paths on T^2
  6. Cluster by geometric similarity using DBSCAN with torus-aware distance
  7. Recursive subclustering on high-variance clusters (CV > 30%)
  8. Length-stratified analysis: short (<=7), medium (8-10), long (11-15)

Writes report to output/loop_taxonomy_report.md and generates torus plots.
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

# Add project root to path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bps.superpotential import (
    assign_basin,
    build_superpotential,
    lookup_W,
    lookup_W_batch,
)
from bps.extract import download_pdb, extract_dihedrals_pdb


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MIN_FLANK = 3       # min consecutive residues in a basin to count as SS element
MAX_LOOP_LEN = 20   # max loop length to extract
TIGHT_CV_THRESH = 30.0   # CV% threshold for "tight" vs "catch-all"

# DBSCAN eps values to try (ascending); pick first giving >=2 clusters, <50% noise
EPS_CANDIDATES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# Length strata
SHORT_MAX = 7
MEDIUM_MAX = 10
LONG_MAX = 15


# ---------------------------------------------------------------------------
# Loop extraction
# ---------------------------------------------------------------------------

def _assign_basins_for_chain(residues: List[Dict]) -> List[Optional[str]]:
    """Assign Ramachandran basin to each residue. Returns list of basin strings."""
    basins = []
    for r in residues:
        if r["phi"] is not None and r["psi"] is not None:
            phi_deg = math.degrees(r["phi"])
            psi_deg = math.degrees(r["psi"])
            basins.append(assign_basin(phi_deg, psi_deg))
        else:
            basins.append(None)
    return basins


def _find_ss_runs(basins: List[Optional[str]], target: str,
                  min_flank: int) -> List[Tuple[int, int]]:
    """Find runs of >= min_flank consecutive residues in the target basin.

    Returns list of (start_idx, end_idx) inclusive.
    """
    runs = []
    start = None
    count = 0
    for i, b in enumerate(basins):
        if b == target:
            if start is None:
                start = i
            count += 1
        else:
            if start is not None and count >= min_flank:
                runs.append((start, start + count - 1))
            start = None
            count = 0
    if start is not None and count >= min_flank:
        runs.append((start, start + count - 1))
    return runs


def extract_loops(
    residues: List[Dict],
    basins: List[Optional[str]],
    min_flank: int = MIN_FLANK,
    max_loop_len: int = MAX_LOOP_LEN,
) -> List[Dict]:
    """Extract loop regions between alpha and beta SS elements.

    A loop is the stretch of residues between the end of one SS element
    and the start of the next, where the two flanking elements are
    alpha and beta (in either direction).

    Returns list of loop dicts with keys:
        direction: 'alpha->beta' or 'beta->alpha'
        residues: list of residue dicts (the loop residues, excluding flanks)
        loop_len: number of residues in the loop
        phi_path: array of phi values (radians)
        psi_path: array of psi values (radians)
        flank_start_basin: basin of the preceding SS element
        flank_end_basin: basin of the following SS element
    """
    # Find all alpha and beta runs
    alpha_runs = _find_ss_runs(basins, "alpha", min_flank)
    beta_runs = _find_ss_runs(basins, "beta", min_flank)

    # Merge and sort all SS elements by start position
    ss_elements = []
    for s, e in alpha_runs:
        ss_elements.append((s, e, "alpha"))
    for s, e in beta_runs:
        ss_elements.append((s, e, "beta"))
    ss_elements.sort(key=lambda x: x[0])

    loops = []
    for k in range(len(ss_elements) - 1):
        s1_start, s1_end, basin1 = ss_elements[k]
        s2_start, s2_end, basin2 = ss_elements[k + 1]

        # Only alpha<->beta transitions
        if basin1 == basin2:
            continue

        # Loop is from s1_end+1 to s2_start-1
        loop_start = s1_end + 1
        loop_end = s2_start - 1

        if loop_start > loop_end:
            # No residues between SS elements (direct transition)
            continue

        loop_len = loop_end - loop_start + 1
        if loop_len > max_loop_len:
            continue

        # Extract loop path: phi/psi for each loop residue
        loop_residues = residues[loop_start:loop_end + 1]
        phi_vals = []
        psi_vals = []
        valid = True
        for r in loop_residues:
            if r["phi"] is None or r["psi"] is None:
                valid = False
                break
            phi_vals.append(r["phi"])
            psi_vals.append(r["psi"])

        if not valid or len(phi_vals) < 1:
            continue

        direction = f"{basin1}->{basin2}"
        loops.append({
            "direction": direction,
            "residues": loop_residues,
            "loop_len": loop_len,
            "phi_path": np.array(phi_vals),
            "psi_path": np.array(psi_vals),
            "flank_start_basin": basin1,
            "flank_end_basin": basin2,
        })

    return loops


# ---------------------------------------------------------------------------
# Torus distance between loop paths
# ---------------------------------------------------------------------------

def _circular_interp(angles: np.ndarray, n_pts: int) -> np.ndarray:
    """Resample an angular path to n_pts using circular interpolation.

    Interpolates sin/cos components separately, recovers angle via atan2.
    This correctly handles the ±pi boundary.
    """
    m = len(angles)
    if m == n_pts:
        return angles.copy()

    old_t = np.linspace(0, 1, m)
    new_t = np.linspace(0, 1, n_pts)

    sin_vals = np.interp(new_t, old_t, np.sin(angles))
    cos_vals = np.interp(new_t, old_t, np.cos(angles))
    return np.arctan2(sin_vals, cos_vals)


def resample_path(phi_path: np.ndarray, psi_path: np.ndarray,
                  n_pts: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a loop path on T^2 to a fixed number of points.

    Uses circular interpolation (sin/cos -> atan2) to handle wrapping.
    """
    phi_r = _circular_interp(phi_path, n_pts)
    psi_r = _circular_interp(psi_path, n_pts)
    return phi_r, psi_r


def torus_path_distance(phi1: np.ndarray, psi1: np.ndarray,
                        phi2: np.ndarray, psi2: np.ndarray) -> float:
    """Compute distance between two paths on T^2.

    Both paths must be resampled to the same length.
    Uses sum of circular distances at each point.
    """
    assert len(phi1) == len(phi2), "Paths must have same length"
    # Circular difference: atan2(sin(a-b), cos(a-b))
    d_phi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    d_psi = np.arctan2(np.sin(psi1 - psi2), np.cos(psi1 - psi2))
    return float(np.mean(np.sqrt(d_phi**2 + d_psi**2)))


def compute_distance_matrix(loops: List[Dict], n_pts: int = 20) -> np.ndarray:
    """Compute pairwise torus distance matrix for a list of loops.

    Each loop is resampled to n_pts before distance computation.
    """
    n = len(loops)
    # Pre-resample all paths
    resampled = []
    for loop in loops:
        phi_r, psi_r = resample_path(loop["phi_path"], loop["psi_path"], n_pts)
        resampled.append((phi_r, psi_r))

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = torus_path_distance(resampled[i][0], resampled[i][1],
                                    resampled[j][0], resampled[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# DBSCAN clustering with eps search
# ---------------------------------------------------------------------------

def cluster_loops(
    D: np.ndarray,
    eps_candidates: List[float] = EPS_CANDIDATES,
    min_samples: int = 3,
) -> Tuple[np.ndarray, float, int]:
    """Run DBSCAN with eps search on a precomputed distance matrix.

    Tries eps values in order; picks first giving >=2 clusters with <50% noise.

    Returns (labels, best_eps, n_clusters).
    labels: array of cluster labels (-1 = noise).
    """
    best_labels = None
    best_eps = -1.0
    best_nclusters = 0

    for eps in eps_candidates:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
        labels = db.fit_predict(D)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        noise_frac = n_noise / len(labels) if len(labels) > 0 else 1.0

        if n_clusters >= 2 and noise_frac < 0.50:
            return labels, eps, n_clusters

        # Track the best result seen so far
        if n_clusters > best_nclusters:
            best_labels = labels
            best_eps = eps
            best_nclusters = n_clusters

    # If no eps gave >=2 clusters with <50% noise, return best attempt
    if best_labels is not None:
        return best_labels, best_eps, best_nclusters

    # Last resort: all noise
    return np.full(len(D), -1, dtype=int), eps_candidates[-1], 0


def _compute_cluster_stats(
    loops: List[Dict],
    labels: np.ndarray,
    W_grid: np.ndarray,
    phi_grid: np.ndarray,
    psi_grid: np.ndarray,
) -> List[Dict]:
    """Compute per-cluster statistics: |delta_W|, torus path length, CV."""
    cluster_ids = sorted(set(labels))
    if -1 in cluster_ids:
        cluster_ids.remove(-1)

    stats = []
    for cid in cluster_ids:
        mask = labels == cid
        cluster_loops = [loops[i] for i in range(len(loops)) if mask[i]]
        n = len(cluster_loops)

        # |delta_W| for each loop
        delta_w_list = []
        path_len_list = []
        for lp in cluster_loops:
            w_vals = lookup_W_batch(W_grid, phi_grid, psi_grid,
                                    lp["phi_path"], lp["psi_path"])
            delta_w = abs(float(w_vals[-1] - w_vals[0]))
            delta_w_list.append(delta_w)

            # Torus path length: sum of point-to-point circular distances
            if len(lp["phi_path"]) > 1:
                d_phi = np.arctan2(np.sin(np.diff(lp["phi_path"])),
                                   np.cos(np.diff(lp["phi_path"])))
                d_psi = np.arctan2(np.sin(np.diff(lp["psi_path"])),
                                   np.cos(np.diff(lp["psi_path"])))
                path_len = float(np.sum(np.sqrt(d_phi**2 + d_psi**2)))
            else:
                path_len = 0.0
            path_len_list.append(path_len)

        delta_w_arr = np.array(delta_w_list)
        path_len_arr = np.array(path_len_list)

        dw_mean = float(np.mean(delta_w_arr))
        dw_std = float(np.std(delta_w_arr))
        dw_cv = (dw_std / dw_mean * 100) if dw_mean > 0 else 0.0

        pl_mean = float(np.mean(path_len_arr))
        pl_std = float(np.std(path_len_arr))
        pl_cv = (pl_std / pl_mean * 100) if pl_mean > 0 else 0.0

        tight = dw_cv < TIGHT_CV_THRESH and pl_cv < TIGHT_CV_THRESH

        # Compute centroid path
        n_pts = 20
        centroid_phi = np.zeros(n_pts)
        centroid_psi = np.zeros(n_pts)
        sin_phi_sum = np.zeros(n_pts)
        cos_phi_sum = np.zeros(n_pts)
        sin_psi_sum = np.zeros(n_pts)
        cos_psi_sum = np.zeros(n_pts)
        for lp in cluster_loops:
            phi_r, psi_r = resample_path(lp["phi_path"], lp["psi_path"], n_pts)
            sin_phi_sum += np.sin(phi_r)
            cos_phi_sum += np.cos(phi_r)
            sin_psi_sum += np.sin(psi_r)
            cos_psi_sum += np.cos(psi_r)
        centroid_phi = np.arctan2(sin_phi_sum / n, cos_phi_sum / n)
        centroid_psi = np.arctan2(sin_psi_sum / n, cos_psi_sum / n)

        # Mean loop length
        mean_len = np.mean([lp["loop_len"] for lp in cluster_loops])

        # Direction distribution
        dirs = [lp["direction"] for lp in cluster_loops]
        dir_counts = {}
        for d in dirs:
            dir_counts[d] = dir_counts.get(d, 0) + 1

        stats.append({
            "cluster_id": cid,
            "n_members": n,
            "delta_w_mean": dw_mean,
            "delta_w_std": dw_std,
            "delta_w_cv": dw_cv,
            "path_len_mean": pl_mean,
            "path_len_std": pl_std,
            "path_len_cv": pl_cv,
            "tight": tight,
            "mean_loop_len": float(mean_len),
            "direction_counts": dir_counts,
            "centroid_phi": centroid_phi,
            "centroid_psi": centroid_psi,
        })

    return stats


# ---------------------------------------------------------------------------
# Recursive subclustering
# ---------------------------------------------------------------------------

def recursive_subcluster(
    loops: List[Dict],
    D: np.ndarray,
    labels: np.ndarray,
    W_grid: np.ndarray,
    phi_grid: np.ndarray,
    psi_grid: np.ndarray,
    depth: int = 0,
    max_depth: int = 2,
) -> Tuple[np.ndarray, List[Dict]]:
    """Recursively subcluster high-variance (CV > 30%) clusters.

    Returns updated labels and full stats list.
    """
    stats = _compute_cluster_stats(loops, labels, W_grid, phi_grid, psi_grid)

    if depth >= max_depth:
        return labels, stats

    new_labels = labels.copy()
    next_id = max(labels) + 1 if len(labels) > 0 else 0
    updated = False

    for st in stats:
        if not st["tight"] and st["n_members"] >= 6:
            cid = st["cluster_id"]
            mask = labels == cid
            indices = np.where(mask)[0]
            sub_D = D[np.ix_(indices, indices)]
            sub_loops = [loops[i] for i in indices]

            # Try subclustering with tighter eps
            sub_eps = [e for e in EPS_CANDIDATES if e < 0.30]
            sub_labels, _, sub_n = cluster_loops(sub_D, sub_eps, min_samples=3)

            if sub_n >= 2:
                for i, idx in enumerate(indices):
                    if sub_labels[i] == -1:
                        new_labels[idx] = -1
                    else:
                        new_labels[idx] = next_id + sub_labels[i]
                next_id += sub_n
                updated = True

    if updated:
        return recursive_subcluster(
            loops, D, new_labels, W_grid, phi_grid, psi_grid,
            depth + 1, max_depth
        )

    return labels, stats


# ---------------------------------------------------------------------------
# Torus visualization
# ---------------------------------------------------------------------------

def plot_torus_with_loops(
    W_grid: np.ndarray,
    phi_grid: np.ndarray,
    psi_grid: np.ndarray,
    loops: List[Dict],
    labels: np.ndarray,
    title: str,
    output_path: str,
) -> None:
    """Plot W landscape with colored loop paths on T^2."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Background: W landscape
    phi_deg = np.degrees(phi_grid)
    psi_deg = np.degrees(psi_grid)
    PHI_DEG, PSI_DEG = np.meshgrid(phi_deg, psi_deg, indexing="ij")
    ax.contourf(PHI_DEG, PSI_DEG, W_grid, levels=30, cmap="YlOrRd_r", alpha=0.6)
    ax.contour(PHI_DEG, PSI_DEG, W_grid, levels=10, colors="gray",
               linewidths=0.3, alpha=0.5)

    # Loop paths colored by cluster
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(unique_labels), 1)))

    for lp, label in zip(loops, labels):
        if label == -1:
            color = (0.7, 0.7, 0.7, 0.3)
            lw = 0.5
        else:
            cidx = unique_labels.index(label) % len(colors)
            color = colors[cidx]
            lw = 1.5
        phi_d = np.degrees(lp["phi_path"])
        psi_d = np.degrees(lp["psi_path"])
        ax.plot(phi_d, psi_d, color=color, linewidth=lw, alpha=0.7)

    ax.set_xlabel("phi (degrees)")
    ax.set_ylabel("psi (degrees)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_title(title)
    ax.set_aspect("equal")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_taxonomy_report(
    all_loops: List[Dict],
    strata_results: Dict[str, Dict],
    n_structures: int,
    output_path: str,
) -> None:
    """Write loop taxonomy report as markdown."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Loop Path Taxonomy Report",
        "",
        f"**Structures analyzed:** {n_structures}",
        f"**Total loops extracted:** {len(all_loops)}",
        "",
    ]

    # Direction summary
    dir_counts: Dict[str, int] = {}
    for lp in all_loops:
        d = lp["direction"]
        dir_counts[d] = dir_counts.get(d, 0) + 1

    lines.extend([
        "## Loop Direction Summary",
        "",
        "| Direction | Count |",
        "|-----------|-------|",
    ])
    for d, c in sorted(dir_counts.items()):
        lines.append(f"| {d} | {c} |")
    lines.append("")

    # Length distribution
    lens = [lp["loop_len"] for lp in all_loops]
    lines.extend([
        "## Loop Length Distribution",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Min | {min(lens)} |",
        f"| Max | {max(lens)} |",
        f"| Mean | {np.mean(lens):.1f} |",
        f"| Median | {np.median(lens):.1f} |",
        "",
    ])

    # Per-stratum results
    for stratum_name, sdata in sorted(strata_results.items()):
        n_loops = sdata["n_loops"]
        stats = sdata["stats"]
        eps = sdata["eps"]
        n_clusters = sdata["n_clusters"]
        n_noise = sdata["n_noise"]
        noise_frac = n_noise / n_loops * 100 if n_loops > 0 else 0

        n_tight = sum(1 for s in stats if s["tight"])
        n_catchall = sum(1 for s in stats if not s["tight"])

        coverage = (n_loops - n_noise) / n_loops * 100 if n_loops > 0 else 0

        lines.extend([
            f"## {stratum_name}",
            "",
            f"**Loops:** {n_loops}",
            f"**DBSCAN eps:** {eps:.2f}",
            f"**Clusters:** {n_clusters} ({n_tight} tight, {n_catchall} catch-all)",
            f"**Noise:** {n_noise} ({noise_frac:.0f}%)",
            f"**Coverage:** {coverage:.0f}%",
            "",
        ])

        if stats:
            lines.extend([
                "| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |",
                "|--------|---|----------|---------|---------------|-------------|------|------------|",
            ])
            for s in sorted(stats, key=lambda x: -x["n_members"]):
                ftype = "tight" if s["tight"] else "catch-all"
                dirs_str = ", ".join(f'{k}:{v}' for k, v in
                                     sorted(s["direction_counts"].items()))
                lines.append(
                    f"| {s['cluster_id']} | {s['n_members']} "
                    f"| {s['delta_w_mean']:.3f} | {s['delta_w_cv']:.1f}% "
                    f"| {s['path_len_mean']:.3f} | {s['path_len_cv']:.1f}% "
                    f"| {ftype} | {dirs_str} |"
                )
            lines.append("")

        # Summary stats for |dW| across all loops in this stratum
        if stats:
            all_dw_cvs = [s["delta_w_cv"] for s in stats if s["tight"]]
            if all_dw_cvs:
                lines.append(f"**Tight family |dW| CV range:** "
                             f"{min(all_dw_cvs):.1f}% - {max(all_dw_cvs):.1f}%")
                lines.append("")

    # Comparison to targets
    lines.extend([
        "## Comparison to Targets (from CLAUDE.md)",
        "",
        "| Metric | Target | Observed |",
        "|--------|--------|----------|",
        f"| Total loops | 1000+ | {len(all_loops)} |",
    ])

    if "Short (<=7)" in strata_results:
        sd = strata_results["Short (<=7)"]
        n_tight = sum(1 for s in sd["stats"] if s["tight"])
        cov = (sd["n_loops"] - sd["n_noise"]) / sd["n_loops"] * 100 if sd["n_loops"] > 0 else 0
        tight_cvs = [s["delta_w_cv"] for s in sd["stats"] if s["tight"]]
        avg_cv = np.mean(tight_cvs) if tight_cvs else 0
        lines.extend([
            f"| Short tight families | ~30 | {n_tight} |",
            f"| Short coverage | 100% | {cov:.0f}% |",
            f"| Short |dW| CV | <10% | {avg_cv:.1f}% |",
        ])
    if "Medium (8-10)" in strata_results:
        md = strata_results["Medium (8-10)"]
        tight_cvs = [s["delta_w_cv"] for s in md["stats"] if s["tight"]]
        avg_cv = np.mean(tight_cvs) if tight_cvs else float("nan")
        lines.append(f"| Medium CV | ~24% | {avg_cv:.1f}% |")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report written: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _get_pdb_entries(cache_dir: str) -> List[Tuple[str, str]]:
    """Get list of (pdb_id, chain_id) from cached PDB files."""
    chain_overrides = {
        "1LCD": "A", "5UGO": "A", "1LMB": "3", "1CSE": "E", "5CHA": "E",
    }
    entries = []
    if not os.path.isdir(cache_dir):
        return entries
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".pdb") or fname.startswith("SYN"):
            continue
        pdb_id = fname.replace(".pdb", "").upper()
        chain_id = chain_overrides.get(pdb_id, "A")
        entries.append((pdb_id, chain_id))
    return entries


def main() -> None:
    print("Priority 4: Loop Path Taxonomy")
    print("=" * 55)
    print()

    # Build superpotential
    print("Building superpotential W(phi, psi)...")
    W_grid, phi_grid, psi_grid = build_superpotential()
    print()

    cache_dir = os.path.join(_PROJECT_ROOT, "data", "pdb_cache")
    output_dir = os.path.join(_PROJECT_ROOT, "output")
    report_path = os.path.join(output_dir, "loop_taxonomy_report.md")
    plot_dir = os.path.join(output_dir, "plots")

    # --- Step 1: Load PDB structures ---
    entries = _get_pdb_entries(cache_dir)
    print(f"Found {len(entries)} PDB structures in cache")

    # --- Step 2-4: Extract loops from all structures ---
    all_loops: List[Dict] = []
    n_structures_used = 0

    for i, (pdb_id, chain_id) in enumerate(entries):
        pdb_path = os.path.join(cache_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            continue

        try:
            residues = extract_dihedrals_pdb(pdb_path, chain_id)
        except (ValueError, Exception) as e:
            print(f"  {pdb_id}: extraction failed: {e}")
            continue

        if len(residues) < 20:
            continue

        basins = _assign_basins_for_chain(residues)
        loops = extract_loops(residues, basins)

        if loops:
            # Tag each loop with source PDB
            for lp in loops:
                lp["pdb_id"] = pdb_id
            all_loops.extend(loops)
            n_structures_used += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(entries)} structures, "
                  f"{len(all_loops)} loops so far")

    print(f"\nExtracted {len(all_loops)} loops from {n_structures_used} structures")
    print()

    if len(all_loops) < 5:
        print("WARNING: Too few loops for meaningful clustering.")
        print("Need more PDB structures. Add files to data/pdb_cache/")

        # Still write a minimal report
        write_taxonomy_report(all_loops, {}, n_structures_used, report_path)
        return

    # --- Step 5-8: Length-stratified clustering ---
    strata = {
        f"Short (<={SHORT_MAX})": [lp for lp in all_loops
                                    if lp["loop_len"] <= SHORT_MAX],
        f"Medium ({SHORT_MAX+1}-{MEDIUM_MAX})": [lp for lp in all_loops
                                                   if SHORT_MAX < lp["loop_len"] <= MEDIUM_MAX],
        f"Long ({MEDIUM_MAX+1}-{LONG_MAX})": [lp for lp in all_loops
                                                if MEDIUM_MAX < lp["loop_len"] <= LONG_MAX],
    }

    strata_results: Dict[str, Dict] = {}

    for stratum_name, stratum_loops in strata.items():
        n = len(stratum_loops)
        print(f"--- {stratum_name}: {n} loops ---")

        if n < 5:
            print("  Too few loops for clustering, skipping")
            strata_results[stratum_name] = {
                "n_loops": n,
                "stats": [],
                "eps": 0.0,
                "n_clusters": 0,
                "n_noise": n,
                "labels": np.full(n, -1, dtype=int) if n > 0 else np.array([], dtype=int),
            }
            continue

        # Compute distance matrix
        print("  Computing pairwise torus distances...")
        D = compute_distance_matrix(stratum_loops)

        # DBSCAN clustering with eps search
        print("  Running DBSCAN clustering...")
        labels, eps, n_clusters = cluster_loops(D)
        n_noise = int(np.sum(labels == -1))
        print(f"  eps={eps:.2f}, {n_clusters} clusters, "
              f"{n_noise} noise ({n_noise/n*100:.0f}%)")

        # Recursive subclustering on high-variance clusters
        if n_clusters >= 1:
            print("  Recursive subclustering on high-variance families...")
            labels, stats = recursive_subcluster(
                stratum_loops, D, labels, W_grid, phi_grid, psi_grid
            )
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(np.sum(labels == -1))
            print(f"  After subclustering: {n_clusters} clusters, "
                  f"{n_noise} noise")
        else:
            stats = _compute_cluster_stats(
                stratum_loops, labels, W_grid, phi_grid, psi_grid
            )

        n_tight = sum(1 for s in stats if s["tight"])
        n_catchall = sum(1 for s in stats if not s["tight"])
        print(f"  {n_tight} tight, {n_catchall} catch-all families")

        strata_results[stratum_name] = {
            "n_loops": n,
            "stats": stats,
            "eps": eps,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "labels": labels,
        }

        # Generate torus plot for this stratum
        plot_path = os.path.join(plot_dir,
                                 f"torus_{stratum_name.replace(' ', '_').replace('(', '').replace(')', '').replace('<=', 'le').replace('-', '_')}.png")
        plot_torus_with_loops(
            W_grid, phi_grid, psi_grid,
            stratum_loops, labels,
            f"Loop Paths: {stratum_name}",
            plot_path,
        )
        print()

    # --- Write report ---
    print("Writing report...")
    write_taxonomy_report(all_loops, strata_results, n_structures_used,
                          report_path)

    # --- Summary ---
    print()
    print("=" * 55)
    print("Priority 4 Summary:")
    print(f"  Structures: {n_structures_used}")
    print(f"  Total loops: {len(all_loops)}")
    for sname, sdata in sorted(strata_results.items()):
        n_tight = sum(1 for s in sdata["stats"] if s["tight"])
        print(f"  {sname}: {sdata['n_loops']} loops, "
              f"{sdata['n_clusters']} clusters ({n_tight} tight)")
    print()
    print("Priority 4 complete.")


if __name__ == "__main__":
    main()
