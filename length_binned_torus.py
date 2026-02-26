"""
TorusFold: Length-Binned Torus Seg/Real Analysis
=================================================
Tests whether the ~11% geometric micro-regularity (torus L1 Seg/Real)
scales with protein length.

QUESTION:
  Is the 1.11× torus suppression:
    (a) A global structural constraint (constant across lengths)
    (b) A per-protein pool-size artifact (shrinks with length)
    (c) Stronger in larger folds (increases with length)

METHOD:
  Bin proteins by residue count (post-pLDDT filter):
    50–100, 100–200, 200–400, 400–800, 800+
  Compute torus L1 Seg/Real within each bin.
  Also reports Original W Seg/Real per bin for comparison.

  Additionally computes the explicit interaction term:
    I = (S/R) / [(Seg/R) × (S/M1)]
  which should be ~1.0 if layers are multiplicatively independent.

Usage:
  python length_binned_torus.py --data alphafold_cache --output results
         [--sample 500] [--n-trials 10] [--w-path FILE]

Reads:  alphafold_cache/, results/superpotential_W.npz
Writes: results/length_binned_torus_report.md
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


def _circular_diff(a, b):
    """Shortest angular difference on a circle, for arrays in radians."""
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def roughness_torus_l1(phi, psi):
    """Mean sequential L1 distance on the torus."""
    if len(phi) < 2:
        return 0.0
    dphi = _circular_diff(phi[1:], phi[:-1])
    dpsi = _circular_diff(psi[1:], psi[:-1])
    return float(np.mean(dphi + dpsi))


def roughness_W(phi, psi, W_grid, grid_size):
    """Standard W-based BPS/L."""
    if len(phi) < 2:
        return 0.0
    scale = grid_size / 360.0
    phi_d = np.degrees(phi)
    psi_d = np.degrees(psi)
    gi = (np.round((phi_d + 180) * scale).astype(int)) % grid_size
    gj = (np.round((psi_d + 180) * scale).astype(int)) % grid_size
    w = W_grid[gi, gj]
    return float(np.mean(np.abs(np.diff(w))))


def null_segment(phi_psi, ss_seq, rng):
    """Segment-preserving null (per-protein basin pools)."""
    phi = np.array([p[0] for p in phi_psi])
    psi = np.array([p[1] for p in phi_psi])
    pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        pools[s].append(i)
    new_phi, new_psi = np.empty_like(phi), np.empty_like(psi)
    for i, s in enumerate(ss_seq):
        j = rng.choice(pools[s])
        new_phi[i], new_psi[i] = phi[j], psi[j]
    return new_phi, new_psi


def null_markov1(phi_psi, ss_seq, rng):
    """First-order Markov null."""
    phi = np.array([p[0] for p in phi_psi])
    psi = np.array([p[1] for p in phi_psi])
    pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        pools[s].append(i)
    basins = sorted(set(ss_seq))
    bidx = {b: i for i, b in enumerate(basins)}
    nb = len(basins)
    trans = np.zeros((nb, nb))
    for i in range(len(ss_seq) - 1):
        trans[bidx[ss_seq[i]], bidx[ss_seq[i+1]]] += 1
    rs = trans.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    tp = trans / rs

    n = len(ss_seq)
    mss = [ss_seq[0]]
    for i in range(1, n):
        nxt = rng.choice(nb, p=tp[bidx[mss[-1]]])
        mss.append(basins[nxt])

    new_phi, new_psi = np.empty(n), np.empty(n)
    for i, s in enumerate(mss):
        j = rng.choice(pools[s])
        new_phi[i], new_psi[i] = phi[j], psi[j]
    return new_phi, new_psi


def null_shuffled(phi_psi, rng):
    """Shuffled null."""
    phi = np.array([p[0] for p in phi_psi])
    psi = np.array([p[1] for p in phi_psi])
    perm = rng.permutation(len(phi))
    return phi[perm], psi[perm]


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
# BOOTSTRAP CI
# ═══════════════════════════════════════════════════════════════════

def bootstrap_ratio_ci(null_vals, real_vals, n_boot=2000, alpha=0.05):
    """Bootstrap 95% CI for mean(null) / mean(real)."""
    rng = np.random.default_rng(99)
    null_arr = np.array(null_vals)
    real_arr = np.array(real_vals)
    ratios = []
    for _ in range(n_boot):
        ni = rng.choice(len(null_arr), len(null_arr), replace=True)
        ri = rng.choice(len(real_arr), len(real_arr), replace=True)
        rm = np.mean(real_arr[ri])
        if rm > 0:
            ratios.append(np.mean(null_arr[ni]) / rm)
    if not ratios:
        return (0.0, 0.0)
    lo = float(np.percentile(ratios, 100 * alpha / 2))
    hi = float(np.percentile(ratios, 100 * (1 - alpha / 2)))
    return (lo, hi)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

LENGTH_BINS = [
    (50, 100, "50–100"),
    (100, 200, "100–200"),
    (200, 400, "200–400"),
    (400, 800, "400–800"),
    (800, 99999, "800+"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Length-Binned Torus Seg/Real Analysis")
    parser.add_argument("--data", default="alphafold_cache",
                        help="AlphaFold data directory")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--w-path", default=None,
                        help="Explicit path to superpotential_W.npz")
    parser.add_argument("--sample", type=int, default=500,
                        help="Total proteins to evaluate (default: 500)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Null model trials per protein (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold: Length-Binned Torus Seg/Real")
    print("=" * 60)

    # ── Load W ───────────────────────────────────────────────
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
                break

    if W_grid is None:
        print("ERROR: superpotential_W.npz not found.")
        print("  Use --w-path to specify location.")
        sys.exit(1)

    grid_size = W_grid.shape[0]

    # ── Discover and sample files ────────────────────────────
    organisms = discover_files(args.data)
    total_files = sum(len(f) for f in organisms.values())
    print(f"  Data: {total_files} files across {len(organisms)} organisms")

    all_files = []
    for org, files in organisms.items():
        for f in files:
            all_files.append((f, org))

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_files))
    sample_size = min(args.sample, len(all_files))
    sampled_files = [all_files[indices[i]] for i in range(sample_size)]

    print(f"  Sample: {sample_size} proteins")

    # ── Extract angles and bin by length ─────────────────────
    print("\n  Extracting and binning...")

    # {bin_label: [{'phi_psi':..., 'ss':..., 'length':..., 'org':...}]}
    binned = {label: [] for _, _, label in LENGTH_BINS}
    n_errors = 0
    MAX_FILE_SIZE = 5 * 1024 * 1024

    for fi, (filepath, org) in enumerate(sampled_files):
        try:
            fsize = os.path.getsize(filepath)
            if fsize > MAX_FILE_SIZE:
                continue
            pp, ss = extract_angles(filepath)
            if len(pp) < 50:
                continue

            prot_len = len(pp)
            for lo, hi, label in LENGTH_BINS:
                if lo <= prot_len < hi:
                    binned[label].append({
                        'phi_psi': pp, 'ss': ss,
                        'length': prot_len, 'org': org,
                    })
                    break

        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"    WARNING: {os.path.basename(filepath)}: "
                      f"{type(e).__name__}: {e}")

        if (fi + 1) % 100 == 0:
            print(f"\r    [{fi+1}/{sample_size}]...", end="", flush=True)

    total_loaded = sum(len(v) for v in binned.values())
    print(f"\r    {total_loaded} proteins loaded into "
          f"{sum(1 for v in binned.values() if v)} bins"
          f"{f' ({n_errors} errors)' if n_errors else ''}    ")

    for _, _, label in LENGTH_BINS:
        print(f"    {label}: {len(binned[label])} proteins")

    # ── Compute per-bin metrics ──────────────────────────────
    print(f"\n  Computing metrics ({args.n_trials} null trials each)...")

    bin_results = {}

    for _, _, label in LENGTH_BINS:
        prots = binned[label]
        if len(prots) < 5:
            print(f"  {label}: too few proteins ({len(prots)}), skipping")
            continue

        print(f"\n  Bin: {label} ({len(prots)} proteins)...")

        # Accumulators
        torus_real, torus_seg, torus_m1, torus_shuf = [], [], [], []
        w_real, w_seg, w_m1, w_shuf = [], [], [], []
        lengths = []

        for pi, prot in enumerate(prots):
            pp = prot['phi_psi']
            ss = prot['ss']
            phi = np.array([p[0] for p in pp])
            psi = np.array([p[1] for p in pp])
            lengths.append(prot['length'])

            # Real
            torus_real.append(roughness_torus_l1(phi, psi))
            w_real.append(roughness_W(phi, psi, W_grid, grid_size))

            # Null trials
            for _ in range(args.n_trials):
                sp, sq = null_segment(pp, ss, rng)
                torus_seg.append(roughness_torus_l1(sp, sq))
                w_seg.append(roughness_W(sp, sq, W_grid, grid_size))

                mp, mq = null_markov1(pp, ss, rng)
                torus_m1.append(roughness_torus_l1(mp, mq))
                w_m1.append(roughness_W(mp, mq, W_grid, grid_size))

                sfp, sfq = null_shuffled(pp, rng)
                torus_shuf.append(roughness_torus_l1(sfp, sfq))
                w_shuf.append(roughness_W(sfp, sfq, W_grid, grid_size))

            if (pi + 1) % 20 == 0:
                print(f"\r    [{pi+1}/{len(prots)}]...", end="", flush=True)

        print(f"\r    {len(prots)} done    ")

        # Compute ratios
        def safe_ratio(a, b):
            ma, mb = np.mean(a), np.mean(b)
            return float(ma / mb) if mb > 0 else 0.0

        torus_sr = safe_ratio(torus_seg, torus_real)
        torus_m1r = safe_ratio(torus_m1, torus_real)
        torus_shr = safe_ratio(torus_shuf, torus_real)
        w_sr = safe_ratio(w_seg, w_real)
        w_m1r = safe_ratio(w_m1, w_real)
        w_shr = safe_ratio(w_shuf, w_real)

        # Interaction term: I = S/R / [(Seg/R) × (S/M1)]
        # where S/M1 ≈ M1/R × (S/R / M1/R)... simpler:
        # I = (S/R) / ((Seg/R) × (S/R) / (Seg/R × M1/Seg))
        # Actually: I = H/R / [(S/R) × (H/M)]
        # With our notation: S/Real = total, Seg/Real = intra-basin,
        # S/M1 = transition arch
        # I = (S/Real) / (Seg/Real × S/M1)
        s_m1_torus = safe_ratio(torus_shuf, torus_m1)
        interaction_torus = (torus_shr / (torus_sr * s_m1_torus)
                            if torus_sr > 0 and s_m1_torus > 0 else 0)

        s_m1_w = safe_ratio(w_shuf, w_m1)
        interaction_w = (w_shr / (w_sr * s_m1_w)
                        if w_sr > 0 and s_m1_w > 0 else 0)

        # Bootstrap CI for torus Seg/Real
        torus_ci = bootstrap_ratio_ci(torus_seg, torus_real)

        bin_results[label] = {
            'n': len(prots),
            'mean_length': float(np.mean(lengths)),
            'torus': {
                'real': float(np.mean(torus_real)),
                'seg_real': torus_sr,
                'm1_real': torus_m1r,
                's_real': torus_shr,
                's_m1': s_m1_torus,
                'interaction': interaction_torus,
                'ci': torus_ci,
            },
            'w': {
                'real': float(np.mean(w_real)),
                'seg_real': w_sr,
                'm1_real': w_m1r,
                's_real': w_shr,
                's_m1': s_m1_w,
                'interaction': interaction_w,
            },
        }

        print(f"    Torus L1: Seg/Real = {torus_sr:.3f}x "
              f"[{torus_ci[0]:.3f}, {torus_ci[1]:.3f}]  "
              f"S/Real = {torus_shr:.3f}x  I = {interaction_torus:.3f}")
        print(f"    W:        Seg/Real = {w_sr:.3f}x  "
              f"S/Real = {w_shr:.3f}x  I = {interaction_w:.3f}")

    # ══════════════════════════════════════════════════════════
    # TREND ANALYSIS
    # ══════════════════════════════════════════════════════════

    sorted_bins = [label for _, _, label in LENGTH_BINS if label in bin_results]
    if len(sorted_bins) >= 3:
        lengths_x = [bin_results[b]['mean_length'] for b in sorted_bins]
        torus_y = [bin_results[b]['torus']['seg_real'] for b in sorted_bins]
        w_y = [bin_results[b]['w']['seg_real'] for b in sorted_bins]

        # Simple linear regression on log(length) vs Seg/Real
        log_x = np.log10(lengths_x)
        torus_slope, torus_intercept = np.polyfit(log_x, torus_y, 1)
        w_slope, w_intercept = np.polyfit(log_x, w_y, 1)
    else:
        torus_slope = w_slope = 0
        torus_intercept = w_intercept = 0

    # ══════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output,
                               "length_binned_torus_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Length-Binned Torus Seg/Real Analysis\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Proteins:** {total_loaded}\n")
        f.write(f"**Null trials:** {args.n_trials} per protein\n\n")

        # ── Key Result Table ──
        f.write("## Key Result: Torus L1 Seg/Real by Protein Length\n\n")
        f.write("```\n")
        f.write("LENGTH-BINNED TORUS SEG/REAL\n")
        f.write("═══════════════════════════════════════════════════════\n")
        f.write(f"  {'Bin':<12s} {'N':>5s} {'Mean L':>7s} "
                f"{'Torus Seg/R':>12s} {'95% CI':>16s} "
                f"{'W Seg/R':>10s}\n")
        f.write("─" * 66 + "\n")
        for label in sorted_bins:
            r = bin_results[label]
            ci = r['torus']['ci']
            f.write(f"  {label:<12s} {r['n']:>5d} {r['mean_length']:>7.0f} "
                    f"{r['torus']['seg_real']:>11.3f}x "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] "
                    f"{r['w']['seg_real']:>9.3f}x\n")
        f.write("═══════════════════════════════════════════════════════\n")

        if len(sorted_bins) >= 3:
            f.write(f"  Torus slope vs log₁₀(length): "
                    f"{torus_slope:+.4f} per decade\n")
            f.write(f"  W slope vs log₁₀(length):     "
                    f"{w_slope:+.4f} per decade\n")
        f.write("```\n\n")

        # ── Verdict ──
        f.write("## Verdict\n\n")

        torus_ratios = [bin_results[b]['torus']['seg_real']
                        for b in sorted_bins]
        if torus_ratios:
            torus_cv = float(np.std(torus_ratios) /
                             np.mean(torus_ratios) * 100)

            if torus_cv < 5 and abs(torus_slope) < 0.02:
                f.write(f"**Constant across lengths** (CV = {torus_cv:.1f}%, "
                        f"slope = {torus_slope:+.4f}). "
                        f"The ~11% geometric micro-regularity is a "
                        f"length-independent structural constraint, not a "
                        f"pool-size artifact. Short and long proteins show "
                        f"the same torus-distance suppression.\n\n")
            elif torus_slope < -0.02:
                f.write(f"**Decreases with length** "
                        f"(slope = {torus_slope:+.4f}). "
                        f"The geometric suppression weakens in longer "
                        f"proteins, consistent with a per-protein pool-size "
                        f"effect: larger proteins have more diverse "
                        f"within-basin angle distributions, reducing the "
                        f"apparent micro-regularity.\n\n")
            elif torus_slope > 0.02:
                f.write(f"**Increases with length** "
                        f"(slope = {torus_slope:+.4f}). "
                        f"Longer proteins show stronger geometric "
                        f"suppression, suggesting that larger folds impose "
                        f"tighter sequential angle constraints.\n\n")
            else:
                f.write(f"**Approximately constant** "
                        f"(CV = {torus_cv:.1f}%, "
                        f"slope = {torus_slope:+.4f}). "
                        f"No strong length dependence detected.\n\n")

        # ── Interaction Term ──
        f.write("## Interaction Term\n\n")
        f.write("The interaction term I = (S/Real) / "
                "[(Seg/Real) × (S/M1)] measures departure from "
                "multiplicative independence of intra-basin coherence "
                "and transition architecture.\n\n")
        f.write("| Bin | Torus I | W I |\n")
        f.write("|---|---|---|\n")
        for label in sorted_bins:
            r = bin_results[label]
            f.write(f"| {label} | "
                    f"{r['torus']['interaction']:.3f} | "
                    f"{r['w']['interaction']:.3f} |\n")
        f.write("\n")

        interaction_vals = [bin_results[b]['torus']['interaction']
                           for b in sorted_bins
                           if bin_results[b]['torus']['interaction'] > 0]
        if interaction_vals:
            mean_i = float(np.mean(interaction_vals))
            f.write(f"Mean torus interaction: **{mean_i:.3f}** ")
            if 0.95 <= mean_i <= 1.05:
                f.write("(≈1.0 — layers are multiplicatively independent)\n\n")
            elif mean_i < 0.95:
                f.write(f"(<1.0 — layers partially overlap, "
                        f"total suppression is less than product)\n\n")
            else:
                f.write(f"(>1.0 — modest coupling between layers, "
                        f"total suppression exceeds product "
                        f"by {(mean_i-1)*100:.0f}%)\n\n")

        # ── Full Decomposition per Bin ──
        f.write("## Full Decomposition by Length Bin\n\n")
        f.write("| Bin | N | Metric | Real | Seg/R | M1/R | "
                "S/R | S/M1 | I |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for label in sorted_bins:
            r = bin_results[label]
            for metric, key in [("Torus L1", "torus"), ("Original W", "w")]:
                d = r[key]
                f.write(f"| {label} | {r['n']} | {metric} | "
                        f"{d['real']:.4f} | "
                        f"{d['seg_real']:.3f}× | "
                        f"{d['m1_real']:.3f}× | "
                        f"{d['s_real']:.3f}× | "
                        f"{d['s_m1']:.3f}× | "
                        f"{d['interaction']:.3f} |\n")
        f.write("\n")

        # ── Method ──
        f.write("## Method\n\n")
        f.write("Proteins were binned by residue count (post-pLDDT "
                "filtering). Within each bin, torus L1 roughness "
                "(|Δφ| + |Δψ|, circular) and standard W-based BPS/L "
                "were computed for real proteins and three null models "
                "(segment-preserving, Markov-1, shuffled).\n\n")
        f.write("The interaction term I = (S/R) / [(Seg/R) × (S/M1)] "
                "measures how well the total suppression factorizes "
                "into intra-basin (Seg/R) and transition (S/M1) "
                "components. I = 1.0 indicates perfect multiplicative "
                "independence; I > 1 indicates coupling.\n\n")
        f.write("Trend analysis uses linear regression of Seg/Real "
                "against log₁₀(protein length). A flat slope indicates "
                "the effect is length-independent.\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
