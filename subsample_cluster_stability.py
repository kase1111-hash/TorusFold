"""
TorusFold: 5× Subsample Cluster Stability Test
================================================
Tests whether the BPS loop taxonomy families are stable across
random subsamples of the dataset.

Procedure:
  1. Extract loop segments from all proteins (loops = non-helix,
     non-sheet runs in the backbone)
  2. Compute BPS feature vector per loop segment
  3. Take 5 independent 80% random subsamples
  4. Cluster each subsample with HDBSCAN
  5. Compute pairwise Adjusted Rand Index (ARI) on shared loops
  6. Report mean ARI ± std as stability metric

Stability thresholds:
  - ARI > 0.8: Excellent stability (taxonomy is robust)
  - ARI 0.6-0.8: Good stability (families mostly stable)
  - ARI < 0.6: Poor stability (taxonomy may be artifact)

Also reports:
  - Number of clusters found per subsample
  - Noise fraction per subsample
  - Jaccard similarity of cluster membership as secondary metric

Usage:
  python subsample_cluster_stability.py [--data DIR] [--sample N]
         [--n-subsamples 5] [--subsample-frac 0.8] [--w-path FILE]
         [--output DIR]

Reads: alphafold_cache/, results/superpotential_W.npz (or --w-path)
Writes: results/subsample_cluster_stability_report.md
"""

import os
import sys
import math
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi required. pip install gemmi")
    sys.exit(1)

try:
    import hdbscan
except ImportError:
    print("ERROR: hdbscan required. pip install hdbscan")
    sys.exit(1)

try:
    from sklearn.metrics import adjusted_rand_score
except ImportError:
    print("ERROR: scikit-learn required. pip install scikit-learn")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (shared with within_foldclass_cv.py)
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
            ss = 'alpha'
        elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
            ss = 'beta'
        else:
            ss = 'other'
        phi_psi.append((phi, psi))
        ss_seq.append(ss)
    return phi_psi, ss_seq


# ═══════════════════════════════════════════════════════════════════
# LOOP EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_loop_segments(phi_psi, ss_seq, W_grid, grid_size, min_loop_len=3):
    """
    Extract loop segments and compute BPS feature vector per loop.

    A loop is a contiguous run of 'o' (other) residues in the SS sequence.
    Feature vector per loop:
      - mean |dW/ds| along the loop (BPS density)
      - std |dW/ds| (BPS variability)
      - loop length (normalized)
      - mean phi, mean psi (circular means, as sin/cos)
      - W range across the loop (topological span)
    → 8-dimensional feature vector per loop
    """
    loops = []
    n = len(ss_seq)
    if n < min_loop_len:
        return loops

    scale = grid_size / 360.0

    # Precompute W values for all residues
    w_vals = np.zeros(n)
    for k in range(n):
        phi_d = math.degrees(phi_psi[k][0])
        psi_d = math.degrees(phi_psi[k][1])
        gi = int(round((phi_d + 180) * scale)) % grid_size
        gj = int(round((psi_d + 180) * scale)) % grid_size
        w_vals[k] = W_grid[gi, gj]

    # Find contiguous loop runs
    i = 0
    while i < n:
        if ss_seq[i] == 'other':
            j = i
            while j < n and ss_seq[j] == 'other':
                j += 1
            loop_len = j - i
            if loop_len >= min_loop_len:
                loop_phi = [phi_psi[k][0] for k in range(i, j)]
                loop_psi = [phi_psi[k][1] for k in range(i, j)]
                loop_w = w_vals[i:j]

                # BPS density: |dW/ds| along loop
                dw = np.abs(np.diff(loop_w))
                bps_mean = float(np.mean(dw)) if len(dw) > 0 else 0.0
                bps_std = float(np.std(dw)) if len(dw) > 0 else 0.0

                # Circular means via sin/cos
                sin_phi = float(np.mean(np.sin(loop_phi)))
                cos_phi = float(np.mean(np.cos(loop_phi)))
                sin_psi = float(np.mean(np.sin(loop_psi)))
                cos_psi = float(np.mean(np.cos(loop_psi)))

                # W range (topological span)
                w_range = float(np.max(loop_w) - np.min(loop_w))

                # Normalized length (cap at 50 residues for normalization)
                norm_len = min(loop_len, 50) / 50.0

                feature = np.array([
                    bps_mean, bps_std, norm_len,
                    sin_phi, cos_phi, sin_psi, cos_psi,
                    w_range,
                ])
                loops.append(feature)
            i = j
        else:
            i += 1
    return loops


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
# CLUSTERING
# ═══════════════════════════════════════════════════════════════════

def cluster_loops(features, min_cluster_size=50, min_samples=10):
    """Cluster loop feature vectors with HDBSCAN."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
    )
    labels = clusterer.fit_predict(features)
    return labels


def jaccard_similarity(labels_a, labels_b, n_pairs=500000):
    """
    Jaccard-style cluster agreement on shared points via random pair sampling.

    For randomly sampled pairs of points: counts pairs where both are in the
    same cluster in at least one partition, then measures what fraction agree
    in both. Excludes noise points (label == -1) from "same cluster" matches.

    Uses random pair sampling for O(n_pairs) instead of O(n²) brute force.
    """
    n = len(labels_a)
    if n < 2:
        return 0.0

    rng = np.random.default_rng(42)
    n_pairs = min(n_pairs, n * (n - 1) // 2)

    # Generate random pairs
    pi = rng.integers(0, n, size=n_pairs)
    pj = rng.integers(0, n - 1, size=n_pairs)
    pj[pj >= pi] += 1  # ensure i != j

    both_same = 0
    either_same = 0

    sa = (labels_a[pi] == labels_a[pj]) & (labels_a[pi] != -1)
    sb = (labels_b[pi] == labels_b[pj]) & (labels_b[pi] != -1)
    either_mask = sa | sb
    both_mask = sa & sb

    either_same = int(np.sum(either_mask))
    both_same = int(np.sum(both_mask))

    if either_same == 0:
        return 0.0
    return both_same / either_same


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="5× Subsample Cluster Stability Test")
    parser.add_argument("--data", default="alphafold_cache",
                        help="AlphaFold data directory")
    parser.add_argument("--sample", type=int, default=0,
                        help="Max proteins per organism (0=all)")
    parser.add_argument("--n-subsamples", type=int, default=5,
                        help="Number of subsamples (default: 5)")
    parser.add_argument("--subsample-frac", type=float, default=0.8,
                        help="Fraction of loops per subsample (default: 0.8)")
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="HDBSCAN min_cluster_size (default: 50)")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="HDBSCAN min_samples (default: 10)")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--w-path", default=None,
                        help="Explicit path to superpotential_W.npz "
                             "(auto-detected if not specified)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold: 5× Subsample Cluster Stability Test")
    print("=" * 60)

    # Load W (search multiple locations)
    w_path = args.w_path or os.path.join(args.output, "superpotential_W.npz")
    W_grid = None

    if os.path.exists(w_path):
        data = np.load(w_path)
        W_grid = data['grid']
        print(f"  Loaded W: {W_grid.shape[0]}x{W_grid.shape[0]} from {w_path}")
    else:
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
        print(f"    python subsample_cluster_stability.py --w-path path/to/superpotential_W.npz")
        sys.exit(1)

    grid_size = W_grid.shape[0]

    # Discover files
    organisms = discover_files(args.data)
    total = sum(len(f) for f in organisms.values())
    print(f"  Data: {total} files across {len(organisms)} organisms")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Extract all loop segments
    # ══════════════════════════════════════════════════════════════
    print("\n  Phase 1: Extracting loop segments...")

    all_loops = []       # list of 8-dim feature vectors
    n_processed = 0
    n_skipped = 0
    MAX_FILE_SIZE = 5 * 1024 * 1024

    for org_name, files in sorted(organisms.items()):
        rng = np.random.default_rng(hash(org_name) % (2**31))
        if args.sample > 0 and len(files) > args.sample:
            idx = rng.choice(len(files), args.sample, replace=False)
            files = [files[i] for i in idx]

        print(f"    {org_name} ({len(files)} files)...", end=" ", flush=True)
        org_loops = 0
        org_errors = 0

        for fi, filepath in enumerate(files):
            try:
                fsize = os.path.getsize(filepath)
                if fsize > MAX_FILE_SIZE:
                    n_skipped += 1
                    continue

                pp, ss = extract_angles(filepath)
                if len(pp) < 50:
                    n_skipped += 1
                    continue

                loops = extract_loop_segments(pp, ss, W_grid, grid_size)
                for feat in loops:
                    all_loops.append(feat)
                    org_loops += 1

                n_processed += 1

            except Exception as e:
                n_skipped += 1
                org_errors += 1
                if org_errors <= 3:
                    fname = os.path.basename(filepath)
                    print(f"\n      WARNING: {fname}: {type(e).__name__}: {e}",
                          end="", flush=True)

            # Progress for large organisms
            if (fi + 1) % 2000 == 0:
                print(f"\n      [{fi+1}/{len(files)}] {org_loops} loops...",
                      end="", flush=True)

        error_note = f" ({org_errors} errors)" if org_errors > 0 else ""
        print(f"{org_loops} loops{error_note}")

    n_loops = len(all_loops)
    print(f"\n  Total: {n_processed} proteins, {n_loops} loop segments, "
          f"{n_skipped} skipped")

    if n_loops < 200:
        print("ERROR: Too few loops for stability analysis")
        sys.exit(1)

    all_loops = np.array(all_loops)

    # Standardize features for clustering
    means = np.mean(all_loops, axis=0)
    stds = np.std(all_loops, axis=0)
    stds[stds < 1e-10] = 1.0
    features_std = (all_loops - means) / stds

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Full-dataset clustering (reference)
    # ══════════════════════════════════════════════════════════════
    print("\n  Phase 2: Reference clustering (full dataset)...")
    ref_labels = cluster_loops(features_std, args.min_cluster_size,
                               args.min_samples)
    n_ref_clusters = len(set(ref_labels) - {-1})
    ref_noise_frac = float(np.sum(ref_labels == -1)) / len(ref_labels) * 100
    print(f"    Reference: {n_ref_clusters} clusters, "
          f"{ref_noise_frac:.1f}% noise")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Subsample clustering
    # ══════════════════════════════════════════════════════════════
    print(f"\n  Phase 3: {args.n_subsamples}× subsample clustering "
          f"({args.subsample_frac:.0%} each)...")

    subsample_size = int(n_loops * args.subsample_frac)
    subsample_results = []

    for s in range(args.n_subsamples):
        rng = np.random.default_rng(s * 12345 + 42)
        idx = rng.choice(n_loops, subsample_size, replace=False)
        idx_sorted = np.sort(idx)

        sub_features = features_std[idx_sorted]
        sub_labels = cluster_loops(sub_features, args.min_cluster_size,
                                   args.min_samples)
        n_clusters = len(set(sub_labels) - {-1})
        noise_frac = float(np.sum(sub_labels == -1)) / len(sub_labels) * 100

        subsample_results.append({
            'indices': idx_sorted,
            'labels': sub_labels,
            'n_clusters': n_clusters,
            'noise_frac': noise_frac,
        })
        print(f"    Subsample {s+1}: {n_clusters} clusters, "
              f"{noise_frac:.1f}% noise")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Pairwise stability metrics
    # ══════════════════════════════════════════════════════════════
    print("\n  Phase 4: Computing pairwise stability...")

    ari_matrix = np.zeros((args.n_subsamples, args.n_subsamples))
    jaccard_matrix = np.zeros((args.n_subsamples, args.n_subsamples))

    pairs = list(combinations(range(args.n_subsamples), 2))
    for i, j in pairs:
        # Find shared indices
        idx_i = subsample_results[i]['indices']
        idx_j = subsample_results[j]['indices']
        shared = np.intersect1d(idx_i, idx_j)

        if len(shared) < 100:
            print(f"    WARNING: Only {len(shared)} shared loops between "
                  f"subsamples {i+1} and {j+1}")
            continue

        # Map shared indices to labels in each subsample
        map_i = {v: k for k, v in enumerate(idx_i)}
        map_j = {v: k for k, v in enumerate(idx_j)}

        labels_i = np.array([subsample_results[i]['labels'][map_i[s]]
                             for s in shared])
        labels_j = np.array([subsample_results[j]['labels'][map_j[s]]
                             for s in shared])

        ari = adjusted_rand_score(labels_i, labels_j)
        jac = jaccard_similarity(labels_i, labels_j)

        ari_matrix[i, j] = ari_matrix[j, i] = ari
        jaccard_matrix[i, j] = jaccard_matrix[j, i] = jac

        print(f"    Subsamples {i+1}×{j+1}: ARI={ari:.3f}, "
              f"Jaccard={jac:.3f} ({len(shared)} shared)")

    # Also compute ARI of each subsample vs. reference
    ari_vs_ref = []
    for s in range(args.n_subsamples):
        idx_s = subsample_results[s]['indices']
        ref_sub = ref_labels[idx_s]
        sub_lab = subsample_results[s]['labels']
        ari = adjusted_rand_score(ref_sub, sub_lab)
        ari_vs_ref.append(ari)
        print(f"    Subsample {s+1} vs reference: ARI={ari:.3f}")

    # ══════════════════════════════════════════════════════════════
    # SUMMARY STATISTICS
    # ══════════════════════════════════════════════════════════════

    ari_values = [ari_matrix[i, j] for i, j in pairs]
    jac_values = [jaccard_matrix[i, j] for i, j in pairs]

    mean_ari = float(np.mean(ari_values))
    std_ari = float(np.std(ari_values))
    mean_jac = float(np.mean(jac_values))
    std_jac = float(np.std(jac_values))
    mean_ari_ref = float(np.mean(ari_vs_ref))
    std_ari_ref = float(np.std(ari_vs_ref))

    cluster_counts = [r['n_clusters'] for r in subsample_results]
    noise_fracs = [r['noise_frac'] for r in subsample_results]

    # Stability verdict
    if mean_ari > 0.8:
        verdict = "EXCELLENT"
        detail = "Taxonomy families are highly stable across subsamples"
    elif mean_ari > 0.6:
        verdict = "GOOD"
        detail = "Taxonomy families are mostly stable with minor variation"
    elif mean_ari > 0.4:
        verdict = "MODERATE"
        detail = "Some instability in taxonomy; may need parameter tuning"
    else:
        verdict = "POOR"
        detail = "Taxonomy is unstable; clusters may be artifactual"

    # ══════════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════════
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output,
                               "subsample_cluster_stability_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 5× Subsample Cluster Stability Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Proteins processed:** {n_processed}\n")
        f.write(f"**Loop segments extracted:** {n_loops}\n")
        f.write(f"**Subsamples:** {args.n_subsamples} × "
                f"{args.subsample_frac:.0%} = "
                f"{subsample_size} loops each\n")
        f.write(f"**Clustering:** HDBSCAN (min_cluster_size="
                f"{args.min_cluster_size}, min_samples="
                f"{args.min_samples})\n")
        f.write(f"**Features:** 8-dim (BPS mean, BPS std, length, "
                f"sin/cos φ, sin/cos ψ, W range)\n\n")

        # Key result
        f.write("## Key Result\n\n")
        f.write("```\n")
        f.write("CLUSTER STABILITY SUMMARY\n")
        f.write("═══════════════════════════════════════════\n")
        f.write(f"  Pairwise ARI:          {mean_ari:.3f} ± "
                f"{std_ari:.3f}\n")
        f.write(f"  Pairwise Jaccard:      {mean_jac:.3f} ± "
                f"{std_jac:.3f}\n")
        f.write(f"  ARI vs reference:      {mean_ari_ref:.3f} ± "
                f"{std_ari_ref:.3f}\n")
        f.write(f"  Cluster count range:   {min(cluster_counts)}-"
                f"{max(cluster_counts)} "
                f"(ref: {n_ref_clusters})\n")
        f.write(f"  Noise fraction range:  {min(noise_fracs):.1f}%-"
                f"{max(noise_fracs):.1f}% "
                f"(ref: {ref_noise_frac:.1f}%)\n")
        f.write("═══════════════════════════════════════════\n")
        f.write(f"  Verdict: {verdict}\n")
        f.write(f"  {detail}\n")
        f.write("```\n\n")

        # Per-subsample details
        f.write("## Per-Subsample Results\n\n")
        f.write("| Subsample | N loops | Clusters | Noise % | "
                "ARI vs ref |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **Reference** | {n_loops} | {n_ref_clusters} | "
                f"{ref_noise_frac:.1f} | — |\n")
        for s in range(args.n_subsamples):
            r = subsample_results[s]
            f.write(f"| Subsample {s+1} | {subsample_size} | "
                    f"{r['n_clusters']} | {r['noise_frac']:.1f} | "
                    f"{ari_vs_ref[s]:.3f} |\n")
        f.write("\n")

        # Pairwise ARI matrix
        f.write("## Pairwise ARI Matrix\n\n")
        header = "| | " + " | ".join(f"S{s+1}" for s in
                                      range(args.n_subsamples)) + " |\n"
        f.write(header)
        f.write("|---" * (args.n_subsamples + 1) + "|\n")
        for i in range(args.n_subsamples):
            row = f"| **S{i+1}** |"
            for j in range(args.n_subsamples):
                if i == j:
                    row += " 1.000 |"
                else:
                    row += f" {ari_matrix[i,j]:.3f} |"
            f.write(row + "\n")
        f.write("\n")

        # Pairwise Jaccard matrix
        f.write("## Pairwise Jaccard Similarity Matrix\n\n")
        f.write(header)
        f.write("|---" * (args.n_subsamples + 1) + "|\n")
        for i in range(args.n_subsamples):
            row = f"| **S{i+1}** |"
            for j in range(args.n_subsamples):
                if i == j:
                    row += " 1.000 |"
                else:
                    row += f" {jaccard_matrix[i,j]:.3f} |"
            f.write(row + "\n")
        f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        f.write(f"The {args.n_subsamples} independent 80% subsamples "
                f"produced cluster solutions with a mean pairwise "
                f"Adjusted Rand Index of **{mean_ari:.3f} ± "
                f"{std_ari:.3f}**.\n\n")

        if mean_ari > 0.8:
            f.write("This indicates **excellent stability**: the loop "
                    "taxonomy families are robust to subsampling and "
                    "are not artifacts of the specific proteins in the "
                    "dataset. The same families emerge regardless of "
                    "which 80% of loops are included.\n\n")
        elif mean_ari > 0.6:
            f.write("This indicates **good stability**: the major loop "
                    "taxonomy families are consistent across subsamples, "
                    "though some boundary cases may shift between "
                    "families. The core structure of the taxonomy is "
                    "reproducible.\n\n")
        elif mean_ari > 0.4:
            f.write("This indicates **moderate stability**: while some "
                    "cluster structure is consistent, there is meaningful "
                    "variation across subsamples. Consider whether "
                    "clustering parameters need adjustment or whether "
                    "the taxonomy represents a continuum rather than "
                    "discrete families.\n\n")
        else:
            f.write("This indicates **poor stability**: the cluster "
                    "assignments change substantially across subsamples, "
                    "suggesting the taxonomy may not reflect genuine "
                    "discrete families in the data.\n\n")

        cv_clusters = (float(np.std(cluster_counts)) /
                       float(np.mean(cluster_counts)) * 100
                       if np.mean(cluster_counts) > 0 else 0)
        f.write(f"Cluster count varied from {min(cluster_counts)} to "
                f"{max(cluster_counts)} (CV = {cv_clusters:.1f}%), "
                f"compared to {n_ref_clusters} clusters in the full "
                f"reference dataset.\n\n")

        # Method note
        f.write("## Method\n\n")
        f.write("Loop segments were extracted as contiguous runs of "
                "non-helix, non-sheet residues (≥3 residues). Each loop "
                "was characterized by an 8-dimensional feature vector: "
                "mean and std of |dW/ds| (BPS density), normalized "
                "length, circular means of φ and ψ (as sin/cos pairs), "
                "and W range across the loop.\n\n")
        f.write("Features were standardized (zero mean, unit variance) "
                "before clustering with HDBSCAN. Stability was assessed "
                "by comparing cluster assignments across subsamples "
                "using the Adjusted Rand Index (ARI), which corrects "
                "for chance agreement and ranges from −1 to 1 "
                "(1 = perfect agreement, 0 = random). Jaccard similarity "
                "was computed via random pair sampling (500k pairs): "
                "the fraction of randomly sampled point-pairs that are "
                "co-clustered in both partitions, among those co-clustered "
                "in at least one. Noise points (HDBSCAN label = −1) "
                "are excluded from co-cluster counts.\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
