"""
TorusFold: Torus-Distance Control Test (Tier 0 Falsification)
==============================================================
The most important stress test for the BPS/L framework.

QUESTION:
  Is the 1.30× Seg/Real suppression a property of protein geometry,
  or an artifact of W curvature inside Ramachandran basins?

METHOD:
  Replace |ΔW| with pure angular distance metrics that contain
  NO density information:

  Metric 1: Torus L1    — |Δφ| + |Δψ|  (circular wrapping)
  Metric 2: Torus L2    — sqrt(Δφ² + Δψ²)  (circular wrapping)
  Metric 3: Flat-basin W — W = constant inside each basin,
                           different constants between basins.
                           Tests whether signal comes from
                           inter-basin jumps vs intra-basin detail.
  Metric 4: Original W  — standard |ΔW| for comparison baseline.

  Also runs ε sensitivity sweep (1e-7, 1e-5, 1e-4, 1e-3) on
  the original W metric to test regularization dependence.

  For each metric, computes:
    - Mean sequential "roughness" for Real proteins
    - Same for Segment null, Markov-1 null, Shuffled null
    - Seg/Real, M1/Real, S/Real ratios

INTERPRETATION:
  If Seg/Real ≈ 1.3× with torus distance (Metrics 1–2):
    → Signal is geometric. Paper is robust.
  If Seg/Real collapses to ~1.0–1.05:
    → Signal is W-curvature artifact. Paper needs major revision.
  If Flat-basin W (Metric 3) shows Seg/Real ≈ 1.0 but S/Real >> 1:
    → Intra-basin coherence depends on density detail,
       but transition architecture is real.

Usage:
  python torus_distance_control.py --data alphafold_cache --output results
         [--sample 200] [--w-path FILE] [--n-trials 10]

Reads:  alphafold_cache/, results/superpotential_W.npz (or --w-path)
Writes: results/torus_distance_control_report.md
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


# ═══════════════════════════════════════════════════════════════════
# DISTANCE METRICS
# ═══════════════════════════════════════════════════════════════════

def _circular_diff(a, b):
    """Shortest angular difference on a circle, for arrays in radians."""
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def roughness_torus_l1(phi, psi):
    """Mean sequential L1 distance on the torus: |Δφ| + |Δψ| (circular)."""
    if len(phi) < 2:
        return 0.0
    dphi = _circular_diff(phi[1:], phi[:-1])
    dpsi = _circular_diff(psi[1:], psi[:-1])
    return float(np.mean(dphi + dpsi))


def roughness_torus_l2(phi, psi):
    """Mean sequential L2 distance on the torus: sqrt(Δφ² + Δψ²) (circular)."""
    if len(phi) < 2:
        return 0.0
    dphi = _circular_diff(phi[1:], phi[:-1])
    dpsi = _circular_diff(psi[1:], psi[:-1])
    return float(np.mean(np.sqrt(dphi**2 + dpsi**2)))


def roughness_flat_basin_w(phi, psi, ss_seq):
    """
    Flat-basin W: assign W = 0 inside α, W = 1 inside β, W = 2 for other.
    Sequential roughness = mean |ΔW| along backbone.
    
    This isolates transition architecture from intra-basin detail.
    If Seg/Real ≈ 1.0 with this metric, intra-basin coherence requires
    density information. If S/Real >> 1, transition architecture is real.
    """
    if len(ss_seq) < 2:
        return 0.0
    basin_w = {'a': 0.0, 'b': 1.0, 'o': 2.0}
    w = np.array([basin_w[s] for s in ss_seq])
    return float(np.mean(np.abs(np.diff(w))))


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


def roughness_W_epsilon(phi, psi, histogram, grid_size, epsilon):
    """W-based BPS/L with custom epsilon: W = -ln(P + ε)."""
    if len(phi) < 2:
        return 0.0
    total = histogram.sum()
    P = histogram / total if total > 0 else histogram
    W = -np.log(P + epsilon)
    scale = grid_size / 360.0
    phi_d = np.degrees(phi)
    psi_d = np.degrees(psi)
    gi = (np.round((phi_d + 180) * scale).astype(int)) % grid_size
    gj = (np.round((psi_d + 180) * scale).astype(int)) % grid_size
    w = W[gi, gj]
    return float(np.mean(np.abs(np.diff(w))))


# ═══════════════════════════════════════════════════════════════════
# NULL MODELS
# ═══════════════════════════════════════════════════════════════════

def null_segment(phi_psi, ss_seq, rng):
    """Segment-preserving null (per-protein basin pools)."""
    phi = np.array([p[0] for p in phi_psi])
    psi = np.array([p[1] for p in phi_psi])
    pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        pools[s].append(i)
    new_phi, new_psi = np.empty_like(phi), np.empty_like(psi)
    new_ss = list(ss_seq)  # SS sequence unchanged
    for i, s in enumerate(ss_seq):
        j = rng.choice(pools[s])
        new_phi[i], new_psi[i] = phi[j], psi[j]
    return new_phi, new_psi, new_ss


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
    return new_phi, new_psi, mss


def null_shuffled(phi_psi, rng):
    """Shuffled null (destroys all sequential structure)."""
    phi = np.array([p[0] for p in phi_psi])
    psi = np.array([p[1] for p in phi_psi])
    perm = rng.permutation(len(phi))
    # SS sequence also shuffled (needed for flat-basin metric)
    return phi[perm], psi[perm], perm


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
        description="Torus-Distance Control Test (Tier 0 Falsification)")
    parser.add_argument("--data", default="alphafold_cache",
                        help="AlphaFold data directory")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--w-path", default=None,
                        help="Explicit path to superpotential_W.npz")
    parser.add_argument("--sample", type=int, default=200,
                        help="Proteins to evaluate (default: 200)")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Null model trials per protein (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold: Torus-Distance Control Test")
    print("  (Tier 0 Falsification — W-curvature artifact?)")
    print("=" * 60)

    # ── Load W and histogram ─────────────────────────────────
    w_path = args.w_path or os.path.join(args.output, "superpotential_W.npz")
    W_grid = None
    histogram = None

    if os.path.exists(w_path):
        data = np.load(w_path)
        W_grid = data['grid']
        histogram = data['histogram'] if 'histogram' in data else None
        print(f"  Loaded W: {W_grid.shape[0]}x{W_grid.shape[0]} from {w_path}")
        if histogram is not None:
            print(f"  Loaded histogram (for ε sweep)")
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
                histogram = data['histogram'] if 'histogram' in data else None
                print(f"  Loaded W: {W_grid.shape[0]}x{W_grid.shape[0]}")
                break

    if W_grid is None:
        print(f"ERROR: superpotential_W.npz not found.")
        print(f"  Use --w-path to specify the location.")
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

    print(f"  Sample: {sample_size} proteins, {args.n_trials} null trials each")

    # ── Extract angles for all sampled proteins ──────────────
    print("\n  Extracting angles...")
    proteins = []
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
            proteins.append({'phi_psi': pp, 'ss': ss, 'org': org})
        except Exception as e:
            n_errors += 1
            if n_errors <= 3:
                print(f"    WARNING: {os.path.basename(filepath)}: "
                      f"{type(e).__name__}: {e}")

        if (fi + 1) % 50 == 0:
            print(f"\r    [{fi+1}/{sample_size}] "
                  f"{len(proteins)} loaded...", end="", flush=True)

    print(f"\r    {len(proteins)} proteins loaded"
          f"{f' ({n_errors} errors)' if n_errors else ''}    ")

    if len(proteins) < 20:
        print("ERROR: Too few proteins for meaningful test")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════
    # TEST A: Four distance metrics
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST A: Distance Metric Comparison")
    print("=" * 60)

    metric_names = ['Torus L1', 'Torus L2', 'Flat-basin W', 'Original W']

    # Accumulators: {metric: {model: [values]}}
    results = {m: {'real': [], 'segment': [], 'markov1': [], 'shuffled': []}
               for m in metric_names}

    for pi, prot in enumerate(proteins):
        pp = prot['phi_psi']
        ss = prot['ss']
        phi = np.array([p[0] for p in pp])
        psi = np.array([p[1] for p in pp])

        # Real protein roughness under each metric
        results['Torus L1']['real'].append(roughness_torus_l1(phi, psi))
        results['Torus L2']['real'].append(roughness_torus_l2(phi, psi))
        results['Flat-basin W']['real'].append(roughness_flat_basin_w(phi, psi, ss))
        results['Original W']['real'].append(roughness_W(phi, psi, W_grid, grid_size))

        # Null models
        for _ in range(args.n_trials):
            # Segment null
            sp, sq, s_ss = null_segment(pp, ss, rng)
            results['Torus L1']['segment'].append(roughness_torus_l1(sp, sq))
            results['Torus L2']['segment'].append(roughness_torus_l2(sp, sq))
            results['Flat-basin W']['segment'].append(
                roughness_flat_basin_w(sp, sq, s_ss))
            results['Original W']['segment'].append(
                roughness_W(sp, sq, W_grid, grid_size))

            # Markov-1 null
            mp, mq, m_ss = null_markov1(pp, ss, rng)
            results['Torus L1']['markov1'].append(roughness_torus_l1(mp, mq))
            results['Torus L2']['markov1'].append(roughness_torus_l2(mp, mq))
            results['Flat-basin W']['markov1'].append(
                roughness_flat_basin_w(mp, mq, m_ss))
            results['Original W']['markov1'].append(
                roughness_W(mp, mq, W_grid, grid_size))

            # Shuffled null
            sfp, sfq, sf_perm = null_shuffled(pp, rng)
            sf_ss = [ss[k] for k in sf_perm]
            results['Torus L1']['shuffled'].append(roughness_torus_l1(sfp, sfq))
            results['Torus L2']['shuffled'].append(roughness_torus_l2(sfp, sfq))
            results['Flat-basin W']['shuffled'].append(
                roughness_flat_basin_w(sfp, sfq, sf_ss))
            results['Original W']['shuffled'].append(
                roughness_W(sfp, sfq, W_grid, grid_size))

        if (pi + 1) % 20 == 0:
            print(f"\r    [{pi+1}/{len(proteins)}] processed...",
                  end="", flush=True)

    print(f"\r    {len(proteins)} proteins × 4 metrics × "
          f"3 nulls × {args.n_trials} trials = done    ")

    # Compute ratios
    metric_ratios = {}
    for m in metric_names:
        r = results[m]
        real_mean = float(np.mean(r['real']))
        seg_mean = float(np.mean(r['segment']))
        m1_mean = float(np.mean(r['markov1']))
        shuf_mean = float(np.mean(r['shuffled']))

        metric_ratios[m] = {
            'real': real_mean,
            'segment': seg_mean,
            'markov1': m1_mean,
            'shuffled': shuf_mean,
            'seg_real': seg_mean / real_mean if real_mean > 0 else 0,
            'm1_real': m1_mean / real_mean if real_mean > 0 else 0,
            's_real': shuf_mean / real_mean if real_mean > 0 else 0,
            'n': len(r['real']),
            # Bootstrap CI for Seg/Real
            'seg_real_ci': _bootstrap_ratio_ci(r['segment'], r['real']),
        }

        print(f"\n  {m}:")
        print(f"    Real = {real_mean:.4f}")
        print(f"    Seg/Real = {metric_ratios[m]['seg_real']:.3f}x  "
              f"M1/Real = {metric_ratios[m]['m1_real']:.3f}x  "
              f"S/Real = {metric_ratios[m]['s_real']:.3f}x")

    # ══════════════════════════════════════════════════════════
    # TEST B: ε Sensitivity Sweep
    # ══════════════════════════════════════════════════════════
    epsilon_results = {}

    if histogram is not None:
        print("\n" + "=" * 60)
        print("  TEST B: ε Sensitivity Sweep")
        print("=" * 60)

        epsilons = [1e-7, 1e-5, 1e-4, 1e-3]

        for eps in epsilons:
            real_v, seg_v = [], []
            for prot in proteins:
                pp = prot['phi_psi']
                ss = prot['ss']
                phi = np.array([p[0] for p in pp])
                psi = np.array([p[1] for p in pp])

                real_v.append(roughness_W_epsilon(
                    phi, psi, histogram, grid_size, eps))

                for _ in range(args.n_trials):
                    sp, sq, s_ss = null_segment(pp, ss, rng)
                    seg_v.append(roughness_W_epsilon(
                        sp, sq, histogram, grid_size, eps))

            rm = float(np.mean(real_v))
            sm = float(np.mean(seg_v))
            ratio = sm / rm if rm > 0 else 0

            epsilon_results[eps] = {
                'real': rm, 'segment': sm, 'seg_real': ratio,
            }
            print(f"  ε = {eps:.0e}: Real = {rm:.4f}, "
                  f"Seg/Real = {ratio:.3f}x")
    else:
        print("\n  Skipping ε sweep (no histogram in W file — "
              "rebuild W with build_superpotential.py to enable)")

    # ══════════════════════════════════════════════════════════
    # TEST C: Helix-Only Seg/Real (all-α subpopulation)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST C: Helix-Only Subpopulation")
    print("=" * 60)

    helix_threshold = 0.70  # ≥70% helix → "helix-dominated"
    helix_proteins = []
    for prot in proteins:
        frac_a = prot['ss'].count('a') / len(prot['ss'])
        if frac_a >= helix_threshold:
            helix_proteins.append(prot)

    print(f"  {len(helix_proteins)} helix-dominated proteins "
          f"(≥{helix_threshold:.0%} α)")

    helix_ratios = {}
    if len(helix_proteins) >= 10:
        for m in ['Torus L1', 'Original W']:
            real_v, seg_v = [], []
            for prot in helix_proteins:
                pp = prot['phi_psi']
                ss = prot['ss']
                phi = np.array([p[0] for p in pp])
                psi = np.array([p[1] for p in pp])

                if m == 'Torus L1':
                    real_v.append(roughness_torus_l1(phi, psi))
                else:
                    real_v.append(roughness_W(phi, psi, W_grid, grid_size))

                for _ in range(args.n_trials):
                    sp, sq, s_ss = null_segment(pp, ss, rng)
                    if m == 'Torus L1':
                        seg_v.append(roughness_torus_l1(sp, sq))
                    else:
                        seg_v.append(roughness_W(sp, sq, W_grid, grid_size))

            rm = float(np.mean(real_v))
            sm = float(np.mean(seg_v))
            ratio = sm / rm if rm > 0 else 0
            helix_ratios[m] = {'real': rm, 'segment': sm, 'seg_real': ratio}
            print(f"  {m} (helix-only): Seg/Real = {ratio:.3f}x")
    else:
        print("  Too few helix-dominated proteins for subpopulation test.")

    # ══════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "torus_distance_control_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Torus-Distance Control Test "
                "(Tier 0 Falsification)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Proteins:** {len(proteins)}\n")
        f.write(f"**Null trials:** {args.n_trials} per protein\n\n")

        # ── Key Result ──
        f.write("## Key Result\n\n")
        f.write("```\n")
        f.write("DISTANCE METRIC COMPARISON\n")
        f.write("═══════════════════════════════════════════════════\n")
        f.write(f"  {'Metric':<16s} {'Real':>8s} {'Seg/Real':>10s} "
                f"{'M1/Real':>10s} {'S/Real':>10s}\n")
        f.write("─" * 56 + "\n")
        for m in metric_names:
            r = metric_ratios[m]
            f.write(f"  {m:<16s} {r['real']:>8.4f} "
                    f"{r['seg_real']:>9.3f}x "
                    f"{r['m1_real']:>9.3f}x "
                    f"{r['s_real']:>9.3f}x\n")
        f.write("═══════════════════════════════════════════════════\n")
        f.write("```\n\n")

        # ── Verdict ──
        f.write("## Verdict\n\n")

        tl1_sr = metric_ratios['Torus L1']['seg_real']
        tl2_sr = metric_ratios['Torus L2']['seg_real']
        ow_sr = metric_ratios['Original W']['seg_real']
        fb_sr = metric_ratios['Flat-basin W']['seg_real']

        # Primary test: do torus metrics show suppression?
        torus_avg = (tl1_sr + tl2_sr) / 2
        w_ratio = ow_sr

        if torus_avg > 1.15:
            f.write(f"**Signal is GEOMETRIC.** Torus-distance Seg/Real "
                    f"= {tl1_sr:.3f}× (L1), {tl2_sr:.3f}× (L2). "
                    f"Suppression persists without any density weighting. "
                    f"The intra-basin coherence measured by BPS/L reflects "
                    f"genuine geometric regularity in protein backbones, "
                    f"not W-curvature amplification.\n\n")
            if abs(torus_avg - w_ratio) < 0.05:
                f.write(f"The torus-distance ratio ({torus_avg:.3f}×) closely "
                        f"matches the W-based ratio ({w_ratio:.3f}×), "
                        f"indicating that W adds no artificial amplification. "
                        f"**The paper's central claim is robust.**\n\n")
            else:
                boost = w_ratio - torus_avg
                f.write(f"Note: W-based ratio ({w_ratio:.3f}×) exceeds "
                        f"torus-distance ratio ({torus_avg:.3f}×) by "
                        f"{boost:.3f}. This {boost/torus_avg*100:.0f}% boost "
                        f"is attributable to density weighting, but the "
                        f"underlying geometric effect is real.\n\n")

        elif torus_avg > 1.05:
            f.write(f"**Signal is PARTIALLY GEOMETRIC.** Torus-distance "
                    f"Seg/Real = {tl1_sr:.3f}× (L1), {tl2_sr:.3f}× (L2). "
                    f"Some suppression exists without density weighting, "
                    f"but W curvature contributes a {w_ratio - torus_avg:.3f} "
                    f"amplification. The paper should report both metrics "
                    f"and acknowledge the density contribution.\n\n")

        else:
            f.write(f"**Signal is a W-CURVATURE ARTIFACT.** Torus-distance "
                    f"Seg/Real = {tl1_sr:.3f}× (L1), {tl2_sr:.3f}× (L2) — "
                    f"effectively no suppression. The 1.30× ratio requires "
                    f"density weighting to appear. The paper's claim of "
                    f"geometric intra-basin coherence is not supported by "
                    f"density-independent metrics.\n\n")

        # Flat-basin analysis
        fb_s_real = metric_ratios['Flat-basin W']['s_real']
        if fb_sr < 1.05 and fb_s_real > 1.3:
            f.write(f"Flat-basin W shows Seg/Real = {fb_sr:.3f}× but "
                    f"S/Real = {fb_s_real:.3f}×. This confirms that "
                    f"transition architecture (inter-basin jumps) is real "
                    f"while intra-basin detail requires continuous W.\n\n")
        elif fb_sr > 1.15:
            f.write(f"Flat-basin W shows Seg/Real = {fb_sr:.3f}×, "
                    f"suggesting that even coarse basin assignment captures "
                    f"some of the coherence signal.\n\n")

        # ── ε Sweep ──
        if epsilon_results:
            f.write("## ε Sensitivity Sweep\n\n")
            f.write("| ε | Real | Seg/Real |\n")
            f.write("|---|---|---|\n")
            for eps in sorted(epsilon_results.keys()):
                r = epsilon_results[eps]
                f.write(f"| {eps:.0e} | {r['real']:.4f} | "
                        f"**{r['seg_real']:.3f}×** |\n")
            f.write("\n")

            ratios = [r['seg_real'] for r in epsilon_results.values()]
            eps_cv = float(np.std(ratios) / np.mean(ratios) * 100) \
                if np.mean(ratios) > 0 else 999
            if eps_cv < 5:
                f.write(f"Seg/Real is **stable** across 4 orders of "
                        f"magnitude of ε (CV = {eps_cv:.1f}%). "
                        f"The regularization floor does not drive the "
                        f"signal.\n\n")
            else:
                f.write(f"Seg/Real varies with ε (CV = {eps_cv:.1f}%). "
                        f"Density-edge effects may contribute to the "
                        f"measured suppression.\n\n")

        # ── Helix-Only ──
        if helix_ratios:
            f.write("## Helix-Only Subpopulation\n\n")
            f.write(f"Proteins with ≥{helix_threshold:.0%} α content "
                    f"(N = {len(helix_proteins)}):\n\n")
            f.write("| Metric | Seg/Real (all) | Seg/Real (helix-only) |\n")
            f.write("|---|---|---|\n")
            for m in helix_ratios:
                f.write(f"| {m} | {metric_ratios[m]['seg_real']:.3f}× | "
                        f"**{helix_ratios[m]['seg_real']:.3f}×** |\n")
            f.write("\n")
            f.write("If helix-only Seg/Real persists, the effect is not "
                    "merely 'helices exist' — it is intra-helix "
                    "micro-regularity beyond basin occupancy.\n\n")

        # ── Detailed tables ──
        f.write("## Detailed Results\n\n")
        f.write("| Metric | N | Real | Segment | M1 | Shuffled | "
                "Seg/Real | M1/Real | S/Real | 95% CI (Seg/Real) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for m in metric_names:
            r = metric_ratios[m]
            ci = r.get('seg_real_ci', (0, 0))
            f.write(f"| {m} | {r['n']} | {r['real']:.4f} | "
                    f"{r['segment']:.4f} | {r['markov1']:.4f} | "
                    f"{r['shuffled']:.4f} | "
                    f"**{r['seg_real']:.3f}×** | "
                    f"{r['m1_real']:.3f}× | "
                    f"{r['s_real']:.3f}× | "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] |\n")
        f.write("\n")

        # ── Method ──
        f.write("## Method\n\n")
        f.write("This test replaces the density-weighted superpotential "
                "W(φ, ψ) = −ln P(φ, ψ) with pure angular distance "
                "metrics that contain no information about the "
                "Ramachandran density distribution.\n\n")
        f.write("**Torus L1:** Mean sequential |Δφ| + |Δψ| along the "
                "backbone, where Δ is the shortest circular difference. "
                "This is the natural L1 metric on the 2-torus T².\n\n")
        f.write("**Torus L2:** Mean sequential √(Δφ² + Δψ²), the "
                "Euclidean metric on T².\n\n")
        f.write("**Flat-basin W:** Assigns W = 0 (α), W = 1 (β), "
                "W = 2 (other). Removes all intra-basin density detail "
                "while preserving inter-basin jump structure. Tests "
                "whether the signal comes from transition architecture "
                "alone.\n\n")
        f.write("**Original W:** Standard BPS/L for comparison.\n\n")
        f.write("All four metrics use identical null models "
                "(segment-preserving, Markov-1, shuffled) with "
                "per-protein basin pools.\n\n")
        f.write("The segment-preserving null draws replacement angles "
                "exclusively from the same protein's basin-classified "
                "residues, not from a global pool. This means the null "
                "preserves: (1) the exact secondary structure sequence, "
                "(2) the protein's own basin angle distribution, and "
                "(3) segment boundaries. It randomizes only the specific "
                "torsion angle assigned to each position within its "
                "basin class.\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


def _bootstrap_ratio_ci(null_vals, real_vals, n_boot=1000, alpha=0.05):
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


if __name__ == "__main__":
    main()
