"""
TorusFold: Per-Segment Permutation Null
========================================
The definitive test: is the ~11% torus Seg/Real from consecutive-pair
micro-regularity WITHIN secondary structure elements, or from
inter-element heterogeneity (different helices having different mean
torsions)?

Three null models compared:

1. Standard segment null (full pool):
   For each position i with SS label s, draw from ALL positions
   in the protein with label s. Destroys both within-segment
   ordering AND inter-segment distinctions.

2. Per-segment permutation null (NEW):
   Identify contiguous SS segments. Within each segment, randomly
   permute the (φ,ψ) values. Preserves per-segment distribution
   exactly, destroys only within-segment ordering.

3. Shuffled null (control):
   Random permutation of entire sequence.

INTERPRETATION:
  Let R_full  = full-pool Seg/Real      (~1.11×)
  Let R_perm  = per-segment-perm Seg/Real
  Let R_shuf  = shuffled S/Real         (~1.55×)

  If R_perm ≈ R_full:
    → Signal is within-segment consecutive-pair smoothness.
    → Inter-element heterogeneity contributes nothing.
    → "Micro-regularity" interpretation is correct.

  If R_perm ≈ 1.0:
    → Signal is entirely inter-element heterogeneity.
    → No consecutive-pair smoothness within segments.
    → Layer 1 = "each SS element has a characteristic torsion fingerprint."

  If 1.0 < R_perm < R_full:
    → Both contribute. Fraction from within-segment ordering =
      (R_perm - 1) / (R_full - 1).

Usage:
  python per_segment_null.py --data alphafold_cache --output results [--sample 200]
"""

import os
import sys
import math
import time
import argparse
import numpy as np
from collections import defaultdict

try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False


# ═══════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _circular_diff(a, b):
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def roughness_torus_l1(phi, psi):
    if len(phi) < 2:
        return 0.0
    dphi = _circular_diff(phi[1:], phi[:-1])
    dpsi = _circular_diff(psi[1:], psi[:-1])
    return float(np.mean(dphi + dpsi))


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
        if plddt_min > 0 and residues[i]['plddt'] < plddt_min:
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
# SEGMENT IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════

def identify_segments(ss_seq):
    """
    Identify contiguous SS segments.
    Returns list of (start, end, ss_type) tuples.
    end is exclusive.
    """
    if not ss_seq:
        return []
    segments = []
    start = 0
    current = ss_seq[0]
    for i in range(1, len(ss_seq)):
        if ss_seq[i] != current:
            segments.append((start, i, current))
            start = i
            current = ss_seq[i]
    segments.append((start, len(ss_seq), current))
    return segments


# ═══════════════════════════════════════════════════════════════════
# NULL MODELS
# ═══════════════════════════════════════════════════════════════════

def null_full_pool(phi, psi, ss_seq, rng):
    """Standard segment-preserving null: draw from full basin pool."""
    pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        pools[s].append(i)
    new_phi = np.empty_like(phi)
    new_psi = np.empty_like(psi)
    for i, s in enumerate(ss_seq):
        j = rng.choice(pools[s])
        new_phi[i], new_psi[i] = phi[j], psi[j]
    return new_phi, new_psi


def null_per_segment_permutation(phi, psi, ss_seq, rng):
    """
    Per-segment permutation null:
    Within each contiguous SS segment, randomly permute the (φ,ψ)
    values. Preserves per-segment distribution exactly, destroys
    only within-segment ordering.

    Inter-segment boundaries are preserved. Each segment's angles
    stay within that segment.
    """
    new_phi = phi.copy()
    new_psi = psi.copy()
    segments = identify_segments(ss_seq)
    for start, end, ss_type in segments:
        length = end - start
        if length < 2:
            continue
        perm = rng.permutation(length)
        new_phi[start:end] = phi[start:end][perm]
        new_psi[start:end] = psi[start:end][perm]
    return new_phi, new_psi


def null_shuffled(phi, psi, rng):
    """Full shuffle of entire sequence."""
    perm = rng.permutation(len(phi))
    return phi[perm], psi[perm]


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_protein(phi, psi, ss_seq, rng, n_trials=10):
    """Compute all three null model roughnesses for one protein."""
    real_r = roughness_torus_l1(phi, psi)

    full_pool_vals = []
    per_seg_vals = []
    shuffled_vals = []

    for _ in range(n_trials):
        fp, fq = null_full_pool(phi, psi, ss_seq, rng)
        full_pool_vals.append(roughness_torus_l1(fp, fq))

        pp, pq = null_per_segment_permutation(phi, psi, ss_seq, rng)
        per_seg_vals.append(roughness_torus_l1(pp, pq))

        sp, sq = null_shuffled(phi, psi, rng)
        shuffled_vals.append(roughness_torus_l1(sp, sq))

    return {
        'real': real_r,
        'full_pool': full_pool_vals,
        'per_segment': per_seg_vals,
        'shuffled': shuffled_vals,
    }


def segment_stats(ss_seq):
    """Compute segment length statistics."""
    segments = identify_segments(ss_seq)
    lengths = {'a': [], 'b': [], 'o': []}
    for start, end, ss in segments:
        lengths[ss].append(end - start)
    return {
        'n_segments': len(segments),
        'n_alpha_segments': len(lengths['a']),
        'n_beta_segments': len(lengths['b']),
        'mean_alpha_length': np.mean(lengths['a']) if lengths['a'] else 0,
        'mean_beta_length': np.mean(lengths['b']) if lengths['b'] else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Per-Segment Permutation Null (Layer 1 Decomposition)")
    parser.add_argument("--data", required=True,
                        help="AlphaFold data directory")
    parser.add_argument("--output", default="results")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()

    if not HAS_GEMMI:
        print("ERROR: gemmi required. pip install gemmi")
        sys.exit(1)

    print("=" * 60)
    print("  Per-Segment Permutation Null")
    print("  Inter-element heterogeneity vs within-element ordering")
    print("=" * 60)

    from pathlib import Path
    all_files = []
    for subdir in sorted(Path(args.data).iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        for f in sorted(subdir.glob("*.cif")):
            all_files.append(str(f))

    rng = np.random.default_rng(42)
    indices = rng.choice(len(all_files), min(args.sample, len(all_files)),
                         replace=False)

    all_real = []
    all_full_pool = []
    all_per_seg = []
    all_shuffled = []
    all_seg_stats = []
    n_loaded = 0

    for fi, idx in enumerate(indices):
        try:
            fsize = os.path.getsize(all_files[idx])
            if fsize > 5 * 1024 * 1024:
                continue
            pp, ss = extract_angles(all_files[idx])
            if len(pp) < 50:
                continue
            phi = np.array([p[0] for p in pp])
            psi = np.array([p[1] for p in pp])
            n_loaded += 1

            result = analyze_protein(phi, psi, ss, rng, args.n_trials)
            all_real.append(result['real'])
            all_full_pool.extend(result['full_pool'])
            all_per_seg.extend(result['per_segment'])
            all_shuffled.extend(result['shuffled'])
            all_seg_stats.append(segment_stats(ss))

        except Exception:
            pass

        if (fi + 1) % 50 == 0:
            print(f"\r    [{fi+1}/{len(indices)}] {n_loaded} loaded...",
                  end="", flush=True)

    print(f"\r    {n_loaded} proteins analyzed                    ")

    rm = np.mean(all_real)
    fp_ratio = np.mean(all_full_pool) / rm
    ps_ratio = np.mean(all_per_seg) / rm
    sh_ratio = np.mean(all_shuffled) / rm

    # Bootstrap CIs
    def boot_ci(null_vals, real_arr, n_boot=2000):
        brng = np.random.default_rng(99)
        null_arr = np.array(null_vals)
        ratios = []
        for _ in range(n_boot):
            ni = brng.choice(len(null_arr), len(null_arr), replace=True)
            ri = brng.choice(len(real_arr), len(real_arr), replace=True)
            rm_ = np.mean(real_arr[ri])
            if rm_ > 0:
                ratios.append(np.mean(null_arr[ni]) / rm_)
        return np.percentile(ratios, 2.5), np.percentile(ratios, 97.5)

    real_arr = np.array(all_real)
    ci_fp = boot_ci(all_full_pool, real_arr)
    ci_ps = boot_ci(all_per_seg, real_arr)
    ci_sh = boot_ci(all_shuffled, real_arr)

    # Decomposition
    within_seg = ps_ratio - 1.0
    inter_seg = fp_ratio - ps_ratio
    total = fp_ratio - 1.0
    frac_within = within_seg / total * 100 if total > 0 else 0
    frac_inter = inter_seg / total * 100 if total > 0 else 0

    # Segment statistics
    mean_n_seg = np.mean([s['n_segments'] for s in all_seg_stats])
    mean_n_alpha = np.mean([s['n_alpha_segments'] for s in all_seg_stats])
    mean_alpha_len = np.mean([s['mean_alpha_length']
                              for s in all_seg_stats if s['mean_alpha_length'] > 0])

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  N = {n_loaded} proteins, {args.n_trials} trials each")
    print(f"  Mean segments per protein: {mean_n_seg:.1f}")
    print(f"  Mean alpha segments: {mean_n_alpha:.1f}, "
          f"mean length: {mean_alpha_len:.1f} residues")
    print(f"\n  {'Null model':<30s} {'Seg/Real':>10s} {'95% CI':>20s}")
    print("  " + "─" * 62)
    print(f"  {'Per-segment permutation':<30s} {ps_ratio:>9.4f}x "
          f"  [{ci_ps[0]:.3f}, {ci_ps[1]:.3f}]")
    print(f"  {'Full pool (standard)':<30s} {fp_ratio:>9.4f}x "
          f"  [{ci_fp[0]:.3f}, {ci_fp[1]:.3f}]")
    print(f"  {'Shuffled':<30s} {sh_ratio:>9.4f}x "
          f"  [{ci_sh[0]:.3f}, {ci_sh[1]:.3f}]")

    print(f"\n  DECOMPOSITION OF LAYER 1:")
    print(f"    Total effect (full pool):           "
          f"{total*100:.1f}% ({fp_ratio:.4f}x)")
    print(f"    Within-segment ordering:            "
          f"{within_seg*100:.1f}% ({ps_ratio:.4f}x)  "
          f"= {frac_within:.0f}% of total")
    print(f"    Inter-element heterogeneity:        "
          f"{inter_seg*100:.1f}% ({fp_ratio:.4f} - {ps_ratio:.4f}x)  "
          f"= {frac_inter:.0f}% of total")

    # Verdict
    print(f"\n  VERDICT:")
    if ci_ps[0] > 1.01:
        print(f"    Per-segment permutation CI [{ci_ps[0]:.3f}, {ci_ps[1]:.3f}] "
              f"EXCLUDES 1.0.")
        print(f"    Genuine within-segment consecutive-pair smoothness exists.")
        print(f"    {frac_within:.0f}% of Layer 1 is micro-regularity, "
              f"{frac_inter:.0f}% is inter-element heterogeneity.")
    elif ci_ps[1] < 1.01:
        print(f"    Per-segment permutation CI [{ci_ps[0]:.3f}, {ci_ps[1]:.3f}] "
              f"consistent with 1.0.")
        print(f"    Layer 1 is ENTIRELY inter-element heterogeneity.")
        print(f"    No consecutive-pair micro-regularity detected.")
    else:
        print(f"    Per-segment permutation CI [{ci_ps[0]:.3f}, {ci_ps[1]:.3f}] "
              f"straddles 1.0.")
        print(f"    Within-segment ordering contributes {frac_within:.0f}% "
              f"but is not individually significant.")
        print(f"    Inter-element heterogeneity dominates ({frac_inter:.0f}%).")

    # Write report
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "per_segment_null_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Per-Segment Permutation Null\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**N:** {n_loaded} proteins, {args.n_trials} trials\n\n")

        f.write("## Results\n\n")
        f.write("| Null model | Seg/Real | 95% CI |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Per-segment permutation | **{ps_ratio:.4f}×** | "
                f"[{ci_ps[0]:.3f}, {ci_ps[1]:.3f}] |\n")
        f.write(f"| Full pool (standard) | **{fp_ratio:.4f}×** | "
                f"[{ci_fp[0]:.3f}, {ci_fp[1]:.3f}] |\n")
        f.write(f"| Shuffled | **{sh_ratio:.4f}×** | "
                f"[{ci_sh[0]:.3f}, {ci_sh[1]:.3f}] |\n\n")

        f.write("## Decomposition\n\n")
        f.write(f"| Component | Magnitude | % of Layer 1 |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| Within-segment ordering | "
                f"{within_seg*100:.1f}% | {frac_within:.0f}% |\n")
        f.write(f"| Inter-element heterogeneity | "
                f"{inter_seg*100:.1f}% | {frac_inter:.0f}% |\n")
        f.write(f"| **Total (Layer 1)** | "
                f"**{total*100:.1f}%** | **100%** |\n\n")

        f.write("## Segment Statistics\n\n")
        f.write(f"- Mean segments per protein: {mean_n_seg:.1f}\n")
        f.write(f"- Mean α segments: {mean_n_alpha:.1f}, "
                f"mean length: {mean_alpha_len:.1f}\n\n")

        f.write("## Method\n\n")
        f.write("The **per-segment permutation null** identifies contiguous "
                "secondary structure segments and randomly permutes (φ,ψ) "
                "values within each segment. This preserves per-segment "
                "distributions exactly while destroying only within-segment "
                "ordering. Comparing this to the standard full-pool null "
                "(which mixes angles across all segments of the same SS type) "
                "decomposes Layer 1 into within-segment ordering and "
                "inter-element heterogeneity.\n\n")
        f.write("**Key distinction:** The per-segment null cannot "
                "create roughness from inter-element differences (different "
                "helices having different mean torsions), because each "
                "segment's angles stay within that segment. Any Seg/Real > 1 "
                "from this null reflects genuine within-element consecutive-"
                "pair smoothness that is destroyed by random permutation.\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
