#!/usr/bin/env python3
"""
LOOP PATH TAXONOMY ON THE RAMACHANDRAN TORUS
=============================================
The gate experiment: do loop paths between secondary structure basins
cluster into a small number of canonical families on T²?

If YES → deterministic structure prediction from dihedral space is viable
If NO  → the torus framework hits its ceiling at per-residue descriptors

This script:
  1. Downloads ~200 high-resolution PDB structures (or uses cached)
  2. Extracts backbone (φ,ψ) trajectories via BioPython
  3. Assigns secondary structure basins per residue
  4. Identifies all SS transitions (loops connecting basins)
  5. Extracts the (φ,ψ) path for each loop on T²
  6. Clusters loop paths by basin-pair type and geometric similarity
  7. Reports: how many canonical path families per basin pair?

Requirements:
  pip install numpy scipy biopython matplotlib scikit-learn

Usage:
  python loop_taxonomy.py                    # download + analyze
  python loop_taxonomy.py --cache-dir ./pdb  # use cached PDB files
  python loop_taxonomy.py --skip-download    # cached only, no network
  python loop_taxonomy.py --max-structs 50   # quick test run
"""

import os, sys, math, argparse, logging, time, warnings
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# Curated high-resolution, non-redundant PDB chains
# (PISCES-style: resolution < 2.0Å, R-free < 0.25, identity < 25%)
# Covers all SCOP classes: all-α, all-β, α/β, α+β
CURATED_CHAINS = [
    # all-alpha
    ("1MBN","A"), ("1CYO","A"), ("1LMB","3"), ("1ENH","A"), ("1VII","A"),
    ("1R69","A"), ("256B","A"), ("2HMB","A"), ("1HRC","A"), ("1BCF","A"),
    ("1MBC","A"), ("1UTG","A"), ("2CCY","A"), ("1FLM","A"), ("1AKG","A"),
    ("1BGE","A"), ("1BDD","A"), ("1PRB","A"), ("1HMD","A"), ("1ECM","A"),
    # all-beta
    ("1TEN","A"), ("1CSP","A"), ("1SHG","A"), ("1SRL","A"), ("1FNF","A"),
    ("1TIT","A"), ("1WIT","A"), ("1PKS","A"), ("1TTG","A"), ("2PTL","A"),
    ("1PGA","A"), ("1IGD","A"), ("1CDB","A"), ("1NSO","A"), ("1K85","A"),
    ("1CDO","A"), ("2RHE","A"), ("1BRS","A"), ("1FKB","A"), ("1NYG","A"),
    # alpha/beta
    ("2CI2","I"), ("1UBQ","A"), ("3CHY","A"), ("2TRX","A"), ("1RX2","A"),
    ("5P21","A"), ("1AKE","A"), ("1PHT","A"), ("1TIM","A"), ("4ENL","A"),
    ("1AJ8","A"), ("1THI","A"), ("1LDN","A"), ("1SAU","A"), ("1PII","A"),
    ("1CSE","I"), ("1YAC","A"), ("2NAC","A"), ("1MJC","A"), ("2ACY","A"),
    # alpha+beta
    ("1BNI","A"), ("1BTA","A"), ("7RSA","A"), ("2LZM","A"), ("1ROP","A"),
    ("1HKS","A"), ("1APS","A"), ("1STN","A"), ("1SCA","A"), ("1CEI","A"),
    ("1ARR","A"), ("1QOP","A"), ("1L63","A"), ("1BAL","A"), ("1COA","A"),
    ("1BRS","D"), ("1MOL","A"), ("1PNJ","A"), ("1DIV","A"), ("1JON","A"),
    # additional mixed for diversity
    ("1HMK","A"), ("1DPS","A"), ("1PHP","A"), ("1OPA","A"), ("1ABE","A"),
    ("1E0G","A"), ("1E0L","A"), ("1O6X","A"), ("1SPR","A"), ("1IMQ","A"),
    ("1K9Q","A"), ("1YZB","A"), ("2PDD","A"), ("1RIS","A"), ("1LOP","A"),
    ("1IDY","A"), ("1NTI","A"), ("1U9C","A"), ("2KFE","A"), ("1RFA","A"),
    ("1WTG","A"), ("1M4W","A"), ("1W4E","A"), ("1J5A","A"), ("1O5U","A"),
    ("1AYE","A"), ("1RXY","A"), ("1ANF","A"), ("1MUL","A"), ("1CBI","A"),
    ("1IFC","A"), ("1IGS","A"), ("2VH7","A"), ("1TYI","A"), ("1FIN","A"),
    ("1B9C","A"), ("1HNG","A"), ("1O0U","A"), ("1OTR","A"), ("1HCD","A"),
    ("1NUL","A"), ("1UNO","A"), ("1E0M","A"), ("2VKN","A"), ("1SCA","A"),
]

# Ramachandran basin definitions (degrees)
BASINS = {
    "alpha":  {"phi": (-100, -30), "psi": (-67,  -7)},
    "beta":   {"phi": (-170, -70), "psi": ( 90, 180)},  # primary sheet
    "beta2":  {"phi": (-170, -70), "psi": (-180,-120)},  # extended sheet (wraps)
    "ppII":   {"phi": (-100, -50), "psi": (120, 180)},
    "alphaL": {"phi": ( 30,  90), "psi": ( 10,  70)},
}

# Canonical basin centers (radians) for distance calculations
BASIN_CENTERS_DEG = {
    "alpha":  (-63.0, -43.0),
    "beta":   (-120.0, 130.0),
    "ppII":   (-75.0, 150.0),
    "alphaL": (57.0, 47.0),
}
BASIN_CENTERS_RAD = {k: (math.radians(v[0]), math.radians(v[1]))
                     for k, v in BASIN_CENTERS_DEG.items()}


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Residue:
    index: int
    phi: Optional[float] = None   # radians
    psi: Optional[float] = None   # radians
    basin: str = "other"
    resname: str = "UNK"

@dataclass
class LoopPath:
    """A transition path between two SS basins on the Ramachandran torus."""
    pdb_id: str
    chain: str
    start_res: int
    end_res: int
    basin_from: str               # e.g., "alpha"
    basin_to: str                 # e.g., "beta"
    path_phi: List[float] = field(default_factory=list)   # radians
    path_psi: List[float] = field(default_factory=list)   # radians
    path_basins: List[str] = field(default_factory=list)
    residues: List[str] = field(default_factory=list)      # amino acid sequence
    loop_length: int = 0
    delta_W: float = 0.0
    path_length_torus: float = 0.0

    @property
    def pair_key(self):
        """Canonical basin pair (alphabetically ordered for symmetry)."""
        a, b = self.basin_from, self.basin_to
        return f"{a}->{b}"

    @property
    def pair_key_undirected(self):
        a, b = sorted([self.basin_from, self.basin_to])
        return f"{a}<->{b}"


# ═══════════════════════════════════════════════════════════════════
# SUPERPOTENTIAL — delegates to canonical shared module
# ═══════════════════════════════════════════════════════════════════

from bps.superpotential import (
    build_superpotential,
    lookup_W,
)


# ═══════════════════════════════════════════════════════════════════
# DIHEDRAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_dihedrals(pdb_path: str, chain_id: str) -> List[Residue]:
    """Extract backbone (φ,ψ) from a PDB file using BioPython."""
    from Bio.PDB import PDBParser, PPBuilder

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", pdb_path)
    except Exception as e:
        log.warning(f"  Parse error {pdb_path}: {e}")
        return []

    model = structure[0]

    # Find the target chain
    target_chain = None
    for chain in model:
        if chain.id == chain_id:
            target_chain = chain
            break

    if target_chain is None:
        # Try case-insensitive
        for chain in model:
            if chain.id.upper() == chain_id.upper():
                target_chain = chain
                break

    if target_chain is None:
        log.warning(f"  Chain {chain_id} not found in {pdb_path}")
        return []

    ppb = PPBuilder()
    residues = []

    for pp in ppb.build_peptides(target_chain):
        phi_psi = pp.get_phi_psi_list()
        pp_residues = list(pp)

        for i, (phi, psi) in enumerate(phi_psi):
            res = pp_residues[i]
            resname = res.get_resname()
            resnum = res.get_id()[1]

            r = Residue(
                index=resnum,
                phi=phi,   # already in radians from BioPython
                psi=psi,
                resname=resname,
            )
            residues.append(r)

    return residues


def assign_basins(residues: List[Residue]) -> List[Residue]:
    """Assign each residue to a Ramachandran basin."""
    for r in residues:
        if r.phi is None or r.psi is None:
            r.basin = "other"
            continue

        phi_d = math.degrees(r.phi)
        psi_d = math.degrees(r.psi)

        assigned = False
        for basin_name, bounds in BASINS.items():
            phi_lo, phi_hi = bounds["phi"]
            psi_lo, psi_hi = bounds["psi"]

            phi_in = phi_lo <= phi_d <= phi_hi
            psi_in = psi_lo <= psi_d <= psi_hi

            if phi_in and psi_in:
                # Merge beta2 into beta
                r.basin = "beta" if basin_name == "beta2" else basin_name
                assigned = True
                break

        if not assigned:
            r.basin = "other"

    return residues


# ═══════════════════════════════════════════════════════════════════
# LOOP EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_loops(residues: List[Residue], pdb_id: str, chain: str,
                  W_grid, phi_grid, psi_grid,
                  min_flank: int = 3, max_loop_len: int = 20) -> List[LoopPath]:
    """Extract loop paths: transitions between defined SS basins.

    A loop is defined as:
    - At least `min_flank` consecutive residues in basin A
    - 1-N residues in transitional basins (other, ppII, alphaL, or different SS)
    - At least `min_flank` consecutive residues in basin B (B ≠ A)

    We keep the flanking residues to anchor the path endpoints.
    """
    if len(residues) < 2 * min_flank + 1:
        return []

    # Identify runs of consistent basin assignment
    runs = []  # (basin, start_idx, end_idx) in residue list
    current_basin = residues[0].basin
    run_start = 0

    for i in range(1, len(residues)):
        if residues[i].basin != current_basin:
            runs.append((current_basin, run_start, i - 1))
            current_basin = residues[i].basin
            run_start = i
    runs.append((current_basin, run_start, len(residues) - 1))

    # Find transitions between major basins (alpha, beta)
    major_basins = {"alpha", "beta"}
    loops = []

    for ri in range(len(runs) - 1):
        basin_a, a_start, a_end = runs[ri]
        if basin_a not in major_basins:
            continue
        if (a_end - a_start + 1) < min_flank:
            continue

        # Look for next major basin (possibly skipping minor runs)
        for rj in range(ri + 1, min(ri + 6, len(runs))):
            basin_b, b_start, b_end = runs[rj]
            if basin_b not in major_basins:
                continue
            if basin_b == basin_a:
                break  # same basin, not a transition
            if (b_end - b_start + 1) < min_flank:
                continue

            # The loop spans from end of run A to start of run B
            loop_start = max(a_start, a_end - min_flank + 1)  # last few of basin A
            loop_end = min(b_end, b_start + min_flank - 1)    # first few of basin B

            loop_len = loop_end - loop_start + 1
            if loop_len < min_flank * 2 + 1:
                continue
            if loop_len > max_loop_len:
                continue

            # Extract the path
            path_residues = residues[loop_start:loop_end + 1]
            path_phi = [r.phi for r in path_residues if r.phi is not None]
            path_psi = [r.psi for r in path_residues if r.psi is not None]

            if len(path_phi) < min_flank * 2 + 1:
                continue

            # Compute path metrics
            W_values = [lookup_W(W_grid, phi_grid, psi_grid, p, s)
                        for p, s in zip(path_phi, path_psi)]
            delta_W = sum(abs(W_values[i+1] - W_values[i])
                          for i in range(len(W_values) - 1))

            # Torus path length (circular distance)
            path_length = 0.0
            for i in range(len(path_phi) - 1):
                dphi = math.atan2(math.sin(path_phi[i+1] - path_phi[i]),
                                  math.cos(path_phi[i+1] - path_phi[i]))
                dpsi = math.atan2(math.sin(path_psi[i+1] - path_psi[i]),
                                  math.cos(path_psi[i+1] - path_psi[i]))
                path_length += math.sqrt(dphi**2 + dpsi**2)

            loop = LoopPath(
                pdb_id=pdb_id,
                chain=chain,
                start_res=path_residues[0].index,
                end_res=path_residues[-1].index,
                basin_from=basin_a,
                basin_to=basin_b,
                path_phi=path_phi,
                path_psi=path_psi,
                path_basins=[r.basin for r in path_residues],
                residues=[r.resname for r in path_residues],
                loop_length=loop_len,
                delta_W=delta_W,
                path_length_torus=path_length,
            )
            loops.append(loop)
            break  # found the transition partner, move on

    return loops


# ═══════════════════════════════════════════════════════════════════
# TORUS DISTANCE METRIC
# ═══════════════════════════════════════════════════════════════════

def torus_path_distance(path_a: LoopPath, path_b: LoopPath) -> float:
    """Compute distance between two loop paths on T².

    Uses dynamic time warping (DTW) on the torus to handle
    different-length paths. Each point is (φ,ψ) on T².
    """
    # Resample both paths to same length for simpler comparison
    n_points = 20  # standardized path length

    def resample(phi_list, psi_list, n):
        if len(phi_list) < 2:
            return phi_list * n, psi_list * n
        t_orig = np.linspace(0, 1, len(phi_list))
        t_new = np.linspace(0, 1, n)
        # Circular interpolation
        phi_cos = np.interp(t_new, t_orig, np.cos(phi_list))
        phi_sin = np.interp(t_new, t_orig, np.sin(phi_list))
        psi_cos = np.interp(t_new, t_orig, np.cos(psi_list))
        psi_sin = np.interp(t_new, t_orig, np.sin(psi_list))
        phi_new = np.arctan2(phi_sin, phi_cos)
        psi_new = np.arctan2(psi_sin, psi_cos)
        return phi_new, psi_new

    phi_a, psi_a = resample(path_a.path_phi, path_a.path_psi, n_points)
    phi_b, psi_b = resample(path_b.path_phi, path_b.path_psi, n_points)

    # Point-wise circular distance on torus
    dist = 0.0
    for i in range(n_points):
        dphi = abs(math.atan2(math.sin(phi_a[i] - phi_b[i]),
                              math.cos(phi_a[i] - phi_b[i])))
        dpsi = abs(math.atan2(math.sin(psi_a[i] - psi_b[i]),
                              math.cos(psi_a[i] - psi_b[i])))
        dist += math.sqrt(dphi**2 + dpsi**2)

    return dist / n_points  # average per-point torus distance


# ═══════════════════════════════════════════════════════════════════
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def cluster_loops(loops: List[LoopPath], max_clusters: int = 10
                  ) -> Dict[str, List[List[LoopPath]]]:
    """Cluster loop paths by basin pair, then by geometric similarity.

    Returns dict: pair_key -> list of clusters (each cluster is a list of LoopPaths)
    """
    from sklearn.cluster import DBSCAN

    # Group by basin pair
    by_pair = defaultdict(list)
    for loop in loops:
        by_pair[loop.pair_key].append(loop)

    results = {}

    for pair_key, pair_loops in sorted(by_pair.items()):
        if len(pair_loops) < 5:
            results[pair_key] = [pair_loops]  # too few to cluster
            continue

        # Build distance matrix
        n = len(pair_loops)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = torus_path_distance(pair_loops[i], pair_loops[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # DBSCAN clustering — finds natural clusters without specifying k
        # eps determines cluster radius on torus; 0.3 rad ≈ 17° average deviation
        for eps in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            clustering = DBSCAN(eps=eps, min_samples=max(2, len(pair_loops) // 20),
                                metric='precomputed').fit(dist_matrix)
            labels = clustering.labels_
            n_clusters = len(set(labels) - {-1})
            n_noise = (labels == -1).sum()

            if n_clusters >= 2 and n_noise < len(pair_loops) * 0.4:
                break  # found reasonable clustering

        # Group by cluster label
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(pair_loops[i])

        # Sort: real clusters first (label >= 0), then noise (label == -1)
        cluster_list = []
        for label in sorted(clusters.keys()):
            if label >= 0:
                cluster_list.append(clusters[label])
        if -1 in clusters:
            cluster_list.append(clusters[-1])  # noise at end

        results[pair_key] = cluster_list

    return results


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def plot_loop_taxonomy(clusters_by_pair: Dict, W_grid, phi_grid, psi_grid,
                       output_dir: Path):
    """Generate torus plots showing loop path families for each basin pair."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        log.warning("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Color palette for clusters
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
              '#a65628', '#f781bf', '#999999', '#66c2a5', '#fc8d62']

    for pair_key, cluster_list in clusters_by_pair.items():
        n_clusters = len(cluster_list)
        if n_clusters == 0:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Background: W landscape (convert to degrees for display)
        phi_deg = np.degrees(phi_grid)
        psi_deg = np.degrees(psi_grid)
        PHI_D, PSI_D = np.meshgrid(phi_deg, psi_deg, indexing='ij')
        ax.contourf(PHI_D, PSI_D, W_grid, levels=20, cmap='Greys_r', alpha=0.3)
        ax.contour(PHI_D, PSI_D, W_grid, levels=10, colors='grey', alpha=0.2, linewidths=0.5)

        # Mark basin centers
        for bname, (bc_phi, bc_psi) in BASIN_CENTERS_DEG.items():
            ax.plot(bc_phi, bc_psi, 'k*', markersize=15, zorder=10)
            ax.annotate(bname, (bc_phi + 5, bc_psi + 5), fontsize=10,
                        fontweight='bold', zorder=10)

        # Plot each cluster
        total_loops = sum(len(c) for c in cluster_list)
        legend_handles = []

        for ci, cluster in enumerate(cluster_list):
            color = colors[ci % len(colors)]
            is_noise = (ci == len(cluster_list) - 1 and len(cluster_list) > 1)
            label = f"Noise ({len(cluster)})" if is_noise else f"Family {ci+1} ({len(cluster)})"
            alpha = 0.15 if is_noise else 0.5
            lw = 0.5 if is_noise else 1.5

            for loop in cluster:
                phi_d = [math.degrees(p) for p in loop.path_phi]
                psi_d = [math.degrees(p) for p in loop.path_psi]
                line, = ax.plot(phi_d, psi_d, color=color, alpha=alpha,
                                linewidth=lw, zorder=5)

            # Plot cluster centroid path (thicker)
            if len(cluster) >= 3 and not is_noise:
                n_pts = 20
                all_phi = []
                all_psi = []
                for loop in cluster:
                    t_orig = np.linspace(0, 1, len(loop.path_phi))
                    t_new = np.linspace(0, 1, n_pts)
                    phi_cos = np.interp(t_new, t_orig, np.cos(loop.path_phi))
                    phi_sin = np.interp(t_new, t_orig, np.sin(loop.path_phi))
                    psi_cos = np.interp(t_new, t_orig, np.cos(loop.path_psi))
                    psi_sin = np.interp(t_new, t_orig, np.sin(loop.path_psi))
                    all_phi.append(np.arctan2(phi_sin, phi_cos))
                    all_psi.append(np.arctan2(psi_sin, psi_cos))

                mean_phi = np.degrees(np.arctan2(
                    np.mean(np.sin(all_phi), axis=0),
                    np.mean(np.cos(all_phi), axis=0)))
                mean_psi = np.degrees(np.arctan2(
                    np.mean(np.sin(all_psi), axis=0),
                    np.mean(np.cos(all_psi), axis=0)))

                ax.plot(mean_phi, mean_psi, color=color, linewidth=3.5,
                        alpha=0.9, zorder=8, label=label)
            else:
                # Just use a proxy for legend
                ax.plot([], [], color=color, linewidth=2, label=label)

        ax.set_xlabel("φ (degrees)", fontsize=12)
        ax.set_ylabel("ψ (degrees)", fontsize=12)
        ax.set_title(f"Loop Path Families: {pair_key}\n"
                     f"({total_loops} loops, {n_clusters} families)",
                     fontsize=14)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.2)

        safe_name = pair_key.replace("->", "_to_").replace("<->", "_")
        fig.tight_layout()
        fig.savefig(output_dir / f"loop_paths_{safe_name}.png", dpi=150)
        plt.close(fig)
        log.info(f"  Saved plot: loop_paths_{safe_name}.png")

    # Summary figure: all pairs overlaid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    pair_keys = list(clusters_by_pair.keys())

    for idx, ax in enumerate(axes.flat):
        ax.contourf(PHI_D, PSI_D, W_grid, levels=20, cmap='Greys_r', alpha=0.3)
        for bname, (bc_phi, bc_psi) in BASIN_CENTERS_DEG.items():
            ax.plot(bc_phi, bc_psi, 'k*', markersize=12, zorder=10)
            ax.annotate(bname, (bc_phi + 5, bc_psi + 5), fontsize=9, fontweight='bold')

        if idx < len(pair_keys):
            pk = pair_keys[idx]
            cluster_list = clusters_by_pair[pk]
            for ci, cluster in enumerate(cluster_list):
                color = colors[ci % len(colors)]
                for loop in cluster:
                    phi_d = [math.degrees(p) for p in loop.path_phi]
                    psi_d = [math.degrees(p) for p in loop.path_psi]
                    ax.plot(phi_d, psi_d, color=color, alpha=0.4, linewidth=1)

            total = sum(len(c) for c in cluster_list)
            n_fam = len([c for c in cluster_list if len(c) >= 2])
            ax.set_title(f"{pk} ({total} loops, {n_fam} families)", fontsize=11)
        else:
            ax.set_title("(no data)", fontsize=11)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.set_xlabel("φ (°)")
        ax.set_ylabel("ψ (°)")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Loop Path Taxonomy on the Ramachandran Torus", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "loop_taxonomy_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info("  Saved: loop_taxonomy_summary.png")


# ═══════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════

def write_report(all_loops: List[LoopPath], clusters_by_pair: Dict,
                 output_path: Path, n_structures: int, n_parsed: int):
    """Write the taxonomy report."""
    lines = []

    def add(s=""):
        lines.append(s)

    add("# Loop Path Taxonomy on the Ramachandran Torus")
    add()
    add(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M')}")
    add(f"**Structures attempted:** {n_structures}")
    add(f"**Structures parsed:** {n_parsed}")
    add(f"**Total loops extracted:** {len(all_loops)}")
    add()
    add("## The Question")
    add()
    add("Do loop paths (transitions between secondary structure basins) cluster")
    add("into a small number of canonical families on the Ramachandran torus?")
    add()
    add("If YES: deterministic structure prediction from dihedral space is viable.")
    add("If NO: the torus framework hits its ceiling at per-residue descriptors.")
    add()

    # Basin pair summary
    add("## Results by Basin Pair")
    add()
    add("| Basin Pair | N loops | N families | Largest family | Coverage (%) | Mean loop length |")
    add("|------------|---------|-----------|----------------|-------------|-----------------|")

    for pair_key, cluster_list in sorted(clusters_by_pair.items()):
        n_loops = sum(len(c) for c in cluster_list)
        real_clusters = [c for c in cluster_list if len(c) >= 2]
        n_families = len(real_clusters)
        largest = max(len(c) for c in cluster_list) if cluster_list else 0
        coverage = (sum(len(c) for c in real_clusters) / n_loops * 100) if n_loops > 0 else 0
        mean_len = np.mean([l.loop_length for l in
                            [loop for c in cluster_list for loop in c]]) if n_loops > 0 else 0
        add(f"| {pair_key:16s} | {n_loops:7d} | {n_families:9d} | {largest:14d} | {coverage:11.1f} | {mean_len:15.1f} |")

    add()

    # Per-pair details
    add("## Detailed Family Analysis")
    add()

    for pair_key, cluster_list in sorted(clusters_by_pair.items()):
        n_loops = sum(len(c) for c in cluster_list)
        if n_loops == 0:
            continue

        add(f"### {pair_key} ({n_loops} loops)")
        add()

        for ci, cluster in enumerate(cluster_list):
            is_noise = (ci == len(cluster_list) - 1 and len(cluster_list) > 1 and len(cluster) < 3)
            label = "Noise / unclustered" if is_noise else f"Family {ci + 1}"

            lengths = [l.loop_length for l in cluster]
            dW_vals = [l.delta_W for l in cluster]
            torus_lens = [l.path_length_torus for l in cluster]

            add(f"**{label}** ({len(cluster)} loops)")
            add(f"- Loop length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} residues")
            add(f"- |ΔW| along path: {np.mean(dW_vals):.3f} ± {np.std(dW_vals):.3f}")
            add(f"- Torus path length: {np.mean(torus_lens):.3f} ± {np.std(torus_lens):.3f} rad")

            # Amino acid composition
            aa_counts = Counter()
            for loop in cluster:
                for res in loop.residues:
                    aa_counts[res] += 1
            total_aa = sum(aa_counts.values())
            top_aa = aa_counts.most_common(5)
            aa_str = ", ".join(f"{aa} ({100*c/total_aa:.0f}%)" for aa, c in top_aa)
            add(f"- Top residues: {aa_str}")
            add()

        add("---")
        add()

    # The verdict
    add("## Verdict")
    add()

    total_loops = len(all_loops)
    total_in_families = 0
    total_families = 0
    family_sizes = []

    for pair_key, cluster_list in clusters_by_pair.items():
        real_clusters = [c for c in cluster_list if len(c) >= 3]
        total_families += len(real_clusters)
        for c in real_clusters:
            total_in_families += len(c)
            family_sizes.append(len(c))

    coverage = (total_in_families / total_loops * 100) if total_loops > 0 else 0

    add(f"**Total loop paths analyzed:** {total_loops}")
    add(f"**Total canonical families found:** {total_families}")
    add(f"**Loops in canonical families:** {total_in_families} ({coverage:.1f}%)")
    add()

    if total_families > 0 and coverage > 50:
        add("### ✓ LOOPS CLUSTER")
        add()
        add(f"Loop paths cluster into {total_families} canonical families covering "
            f"{coverage:.0f}% of all observed loops. The median family contains "
            f"{int(np.median(family_sizes)) if family_sizes else 0} loops.")
        add()
        add("**Implication:** The loop conformation problem on the Ramachandran torus")
        add("is a classification problem (which canonical path?) rather than a ")
        add("generation problem (what arbitrary path?). Deterministic structure ")
        add("prediction from dihedral space may be viable for single-domain proteins.")
    elif total_families > 0:
        add("### ~ PARTIAL CLUSTERING")
        add()
        add(f"Found {total_families} families covering {coverage:.0f}% of loops.")
        add("Clustering exists but does not capture the majority of loop paths.")
        add("The loop conformation space may be more diverse than hypothesized,")
        add("or the clustering parameters need refinement.")
    else:
        add("### ✗ LOOPS DO NOT CLUSTER")
        add()
        add("No canonical loop path families were detected. Loop conformations")
        add("on the Ramachandran torus appear to be continuously distributed")
        add("rather than falling into discrete families. The torus framework's")
        add("ceiling is at per-residue descriptors (BPS/L), not path prediction.")

    add()
    add("---")
    add()
    add("*This analysis is part of the TorusFold research program.*")
    add("*See: 'Proteins as One-Dimensional Paths on the Ramachandran Torus' (Branham, 2026)*")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Report written to {output_path}")


# ═══════════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

def download_pdb(pdb_id: str, cache_dir: Path) -> Optional[Path]:
    """Download a PDB file from RCSB."""
    import urllib.request

    pdb_id = pdb_id.upper()
    filename = f"{pdb_id}.pdb"
    filepath = cache_dir / filename

    if filepath.exists() and filepath.stat().st_size > 100:
        return filepath

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TorusFold/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        filepath.write_bytes(data)
        return filepath
    except Exception as e:
        log.warning(f"  Download failed {pdb_id}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Loop Path Taxonomy on the Ramachandran Torus")
    parser.add_argument("--cache-dir", type=str, default="./pdb_cache",
                        help="Directory for cached PDB files")
    parser.add_argument("--output-dir", type=str, default="./loop_taxonomy_output",
                        help="Output directory for report and figures")
    parser.add_argument("--skip-download", action="store_true",
                        help="Only use cached PDB files")
    parser.add_argument("--max-structs", type=int, default=None,
                        help="Limit number of structures (for testing)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("LOOP PATH TAXONOMY ON THE RAMACHANDRAN TORUS")
    log.info("=" * 60)

    # Build superpotential
    log.info("Building superpotential W(φ,ψ)...")
    W_grid, phi_grid, psi_grid = build_superpotential()
    log.info("  Done.")

    # Get chain list
    chains = CURATED_CHAINS
    if args.max_structs:
        chains = chains[:args.max_structs]
    log.info(f"Processing {len(chains)} chains...")

    # Download
    if not args.skip_download:
        unique_pdbs = set(pdb_id for pdb_id, _ in chains)
        log.info(f"Downloading {len(unique_pdbs)} unique PDB entries...")
        n_ok = 0
        for pdb_id in sorted(unique_pdbs):
            result = download_pdb(pdb_id, cache_dir)
            if result:
                n_ok += 1
        log.info(f"  Downloads: {n_ok}/{len(unique_pdbs)} OK")

    # Process
    all_loops = []
    n_parsed = 0
    n_total_residues = 0

    for i, (pdb_id, chain_id) in enumerate(chains):
        pdb_path = cache_dir / f"{pdb_id.upper()}.pdb"
        if not pdb_path.exists():
            continue

        residues = extract_dihedrals(str(pdb_path), chain_id)
        if len(residues) < 20:
            continue

        residues = assign_basins(residues)
        n_parsed += 1
        n_total_residues += len(residues)

        loops = extract_loops(residues, pdb_id, chain_id, W_grid, phi_grid, psi_grid)
        all_loops.extend(loops)

        if (i + 1) % 25 == 0:
            log.info(f"  Processed {i+1}/{len(chains)} ({n_parsed} parsed, "
                     f"{len(all_loops)} loops so far)")

    log.info(f"Processing complete: {n_parsed} structures, {len(all_loops)} loops, "
             f"{n_total_residues} total residues")

    if len(all_loops) < 10:
        log.error("Too few loops extracted. Check PDB downloads and chain IDs.")
        sys.exit(1)

    # Cluster
    log.info("Clustering loop paths...")
    clusters_by_pair = cluster_loops(all_loops)
    log.info("  Done.")

    # Report summary to console
    log.info("")
    log.info("=" * 60)
    log.info("RESULTS")
    log.info("=" * 60)
    for pair_key, cluster_list in sorted(clusters_by_pair.items()):
        n_loops = sum(len(c) for c in cluster_list)
        real_clusters = [c for c in cluster_list if len(c) >= 3]
        n_fam = len(real_clusters)
        coverage = sum(len(c) for c in real_clusters) / n_loops * 100 if n_loops else 0
        log.info(f"  {pair_key:16s}: {n_loops:4d} loops → {n_fam} families "
                 f"({coverage:.0f}% coverage)")

    total_in_fam = sum(len(c) for pk in clusters_by_pair
                       for c in clusters_by_pair[pk] if len(c) >= 3)
    total = len(all_loops)
    log.info(f"  TOTAL: {total} loops, {total_in_fam} in families "
             f"({100*total_in_fam/total:.0f}% coverage)")
    log.info("")

    # Visualize
    if not args.no_plots:
        log.info("Generating plots...")
        plot_loop_taxonomy(clusters_by_pair, W_grid, phi_grid, psi_grid, output_dir)

    # Write report
    write_report(all_loops, clusters_by_pair, output_dir / "loop_taxonomy_report.md",
                 len(chains), n_parsed)

    log.info("")
    log.info("=" * 60)
    log.info("DONE")
    log.info(f"  Report: {output_dir / 'loop_taxonomy_report.md'}")
    log.info(f"  Plots:  {output_dir}/")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
