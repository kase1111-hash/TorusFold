"""
TorusFold: Publication Figure Generator
=======================================
Generates all manuscript figures from the BPS proteome atlas data.

Figures produced:
  Fig 1: Conceptual overview (3-panel)
         (A) Ramachandran landscape: real vs segment-null clustering
         (B) W(t) traces along backbone for real, segment, shuffled
         (C) Roughness decomposition bar chart (collapse demonstration)

  Fig 2: Five-level null model hierarchy (bar chart with error bars)

  Fig 3: Cross-organism conservation and fold-class separation (2-panel)
         (A) Per-organism mean BPS/L with domain coloring
         (B) Fold-class BPS/L boxplots

  Fig 4: Within-fold-class cross-organism CV (validation result)

Usage:
  python generate_figures.py [--data DIR] [--sample N] [--output DIR]

Reads: alphafold_cache/, results/superpotential_W.npz
Writes: results/figures/fig1_conceptual.png
        results/figures/fig2_null_hierarchy.png
        results/figures/fig3_conservation_foldclass.png
        results/figures/fig4_within_foldclass_cv.png
        results/figures/fig_ramachandran_W.png  (bonus: W landscape)

Dependencies:
  pip install numpy gemmi matplotlib
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

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# PUBLICATION STYLE
# ═══════════════════════════════════════════════════════════════════

def set_pub_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.6,
    })


# Color palette: colorblind-friendly
COLORS = {
    'real': '#2166AC',       # strong blue
    'segment': '#F4A582',    # salmon
    'm1': '#B2182B',         # dark red
    'shuffled': '#878787',   # gray
    'bacteria': '#D6604D',   # red-ish
    'eukaryote': '#2166AC',  # blue
    'alpha': '#4393C3',      # light blue
    'alpha_beta': '#92C5DE',  # very light blue
    'beta': '#F4A582',       # light salmon
    'ab': '#D6604D',         # red
    'other': '#878787',      # gray
}


# ═══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (shared with pipeline)
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


def classify_fold(ss_seq, alpha_thresh=0.15, beta_thresh=0.15):
    """Classify protein fold class from SS sequence."""
    n = len(ss_seq)
    if n == 0:
        return 'other'
    na = ss_seq.count('alpha') / n
    nb = ss_seq.count('beta') / n
    has_alpha = na >= alpha_thresh
    has_beta = nb >= beta_thresh
    if has_alpha and not has_beta:
        return 'all-alpha'
    elif has_beta and not has_alpha:
        return 'all-beta'
    elif has_alpha and has_beta:
        transitions = sum(1 for i in range(len(ss_seq)-1)
                         if ss_seq[i] != ss_seq[i+1]
                         and ss_seq[i] in ('alpha', 'beta')
                         and ss_seq[i+1] in ('alpha', 'beta'))
        n_ab = na * n + nb * n
        trans_rate = transitions / max(n_ab, 1)
        if trans_rate > 0.05:
            return 'alpha/beta'
        else:
            return 'alpha+beta'
    else:
        return 'other'


def compute_bps_l(phi_arr, psi_arr, W_grid, grid_size):
    """BPS/L from arrays of phi, psi in radians. Normalizes by L."""
    if len(phi_arr) < 2:
        return 0.0, np.array([])
    from bps.superpotential import lookup_W_grid
    w = lookup_W_grid(W_grid, grid_size, phi_arr, psi_arr)
    dw = np.abs(np.diff(w))
    L = len(phi_arr)
    return float(np.sum(dw)) / L, w


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


# Organism domain classification
BACTERIA = {'ecoli', 'bacillus', 'tuberculosis', 'salmonella', 'pseudomonas',
            'staph', 'helicobacter', 'campylobacter'}
EUKARYOTES = {'human', 'mouse', 'rat', 'chicken', 'chimp', 'gorilla', 'dog',
              'cat', 'cow', 'pig', 'yeast', 'fission_yeast', 'candida',
              'fly', 'worm', 'mosquito', 'honeybee', 'malaria', 'trypanosoma',
              'leishmania', 'dictyostelium'}

def domain_of(org):
    if org in BACTERIA:
        return 'bacteria'
    elif org in EUKARYOTES:
        return 'eukaryote'
    return 'other'


# ═══════════════════════════════════════════════════════════════════
# W CONSTRUCTION (fallback if .npz not found)
# ═══════════════════════════════════════════════════════════════════

def build_W_from_data(data_dir=None, grid_size=360, max_proteins=None, save_path=None):
    """Build W = -sqrt(P) from the shared von Mises mixture.

    Parameters are accepted for backward compatibility but ignored —
    W is always the canonical von Mises construction.
    """
    from bps.superpotential import build_superpotential as _build
    W_grid, phi_grid, psi_grid = _build(grid_size)
    if save_path:
        np.savez(save_path, grid=W_grid)
    return W_grid


# ═══════════════════════════════════════════════════════════════════
# NULL MODEL GENERATORS
# ═══════════════════════════════════════════════════════════════════

def null_shuffled(phi_psi, rng):
    """Random permutation of (phi, psi) sequence."""
    pp = list(phi_psi)
    rng.shuffle(pp)
    return pp


def null_segment_preserving(phi_psi, ss_seq, rng):
    """Preserve segment boundaries, randomize within-basin positions."""
    # Build pools per basin
    pools = defaultdict(list)
    for i, ss in enumerate(ss_seq):
        pools[ss].append(phi_psi[i])

    # Convert to arrays for fast sampling
    pool_arrays = {}
    for ss, angles in pools.items():
        pool_arrays[ss] = angles

    # Replace each position with random draw from same basin
    result = []
    for i, ss in enumerate(ss_seq):
        pool = pool_arrays[ss]
        idx = rng.integers(0, len(pool))
        result.append(pool[idx])
    return result


def null_markov_1(phi_psi, ss_seq, rng):
    """First-order Markov null: generate basin sequence from transition
    matrix, fill with random angles from basin pool."""
    # Build transition matrix
    basins = ['a', 'b', 'o']
    trans = defaultdict(lambda: defaultdict(int))
    for i in range(len(ss_seq) - 1):
        trans[ss_seq[i]][ss_seq[i+1]] += 1

    # Normalize
    trans_prob = {}
    for s in basins:
        total = sum(trans[s].values())
        if total == 0:
            trans_prob[s] = {b: 1.0/3 for b in basins}
        else:
            trans_prob[s] = {b: trans[s][b] / total for b in basins}

    # Build pools
    pools = defaultdict(list)
    for i, ss in enumerate(ss_seq):
        pools[ss].append(phi_psi[i])

    # Generate new sequence
    new_ss = [ss_seq[0]]
    for i in range(1, len(ss_seq)):
        probs = [trans_prob[new_ss[-1]].get(b, 0) for b in basins]
        total = sum(probs)
        probs = [p / total for p in probs]
        new_ss.append(rng.choice(basins, p=probs))

    # Fill with random angles
    result = []
    for ss in new_ss:
        pool = pools[ss] if pools[ss] else phi_psi
        idx = rng.integers(0, len(pool))
        result.append(pool[idx])
    return result


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Conceptual Overview (3-panel)
# ═══════════════════════════════════════════════════════════════════

def generate_fig1(data_dir, W_grid, grid_size, sample_proteins, fig_dir):
    """
    Fig 1: Conceptual overview.
    (A) Ramachandran landscape with real (green) vs segment-null (orange)
        within the alpha basin
    (B) W(t) traces along backbone
    (C) Roughness decomposition collapse
    """
    print("  Generating Figure 1: Conceptual overview...")

    # Collect angles from sample proteins
    all_phi_real = []
    all_psi_real = []
    all_ss_real = []
    all_phi_psi_list = []
    all_ss_list = []
    w_traces = {'real': [], 'segment': [], 'shuffled': []}

    rng = np.random.default_rng(42)

    for filepath in sample_proteins[:50]:
        try:
            pp, ss = extract_angles(filepath)
            if len(pp) < 50:
                continue
            phi = np.array([p[0] for p in pp])
            psi = np.array([p[1] for p in pp])

            for i in range(len(pp)):
                all_phi_real.append(math.degrees(pp[i][0]))
                all_psi_real.append(math.degrees(pp[i][1]))
                all_ss_real.append(ss[i])

            all_phi_psi_list.append(pp)
            all_ss_list.append(ss)

            # Compute W traces for one representative protein
            if len(w_traces['real']) == 0 and len(pp) > 80:
                _, w_real = compute_bps_l(phi, psi, W_grid, grid_size)
                w_traces['real'] = w_real[:100]

                seg_pp = null_segment_preserving(pp, ss, rng)
                seg_phi = np.array([p[0] for p in seg_pp])
                seg_psi = np.array([p[1] for p in seg_pp])
                _, w_seg = compute_bps_l(seg_phi, seg_psi, W_grid, grid_size)
                w_traces['segment'] = w_seg[:100]

                shuf_pp = null_shuffled(pp, rng)
                shuf_phi = np.array([p[0] for p in shuf_pp])
                shuf_psi = np.array([p[1] for p in shuf_pp])
                _, w_shuf = compute_bps_l(shuf_phi, shuf_psi, W_grid, grid_size)
                w_traces['shuffled'] = w_shuf[:100]

        except Exception:
            continue

    if not all_phi_real:
        print("    WARNING: No data for Fig 1, skipping.")
        return

    # Generate segment-null angles for the same proteins
    all_phi_seg = []
    all_psi_seg = []
    for pp, ss in zip(all_phi_psi_list, all_ss_list):
        seg_pp = null_segment_preserving(pp, ss, rng)
        for i in range(len(seg_pp)):
            all_phi_seg.append(math.degrees(seg_pp[i][0]))
            all_psi_seg.append(math.degrees(seg_pp[i][1]))

    # ── Panel layout ──
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.2, 0.8],
                           wspace=0.35, left=0.07, right=0.97,
                           top=0.88, bottom=0.15)

    # ── Panel A: Ramachandran zoom into alpha basin ──
    ax_a = fig.add_subplot(gs[0])

    # Filter to alpha-basin residues only
    alpha_phi_real = [all_phi_real[i] for i in range(len(all_ss_real))
                      if all_ss_real[i] == 'alpha']
    alpha_psi_real = [all_psi_real[i] for i in range(len(all_ss_real))
                      if all_ss_real[i] == 'alpha']

    # For segment null, use same count
    n_alpha = len(alpha_phi_real)
    # Filter segment null points in alpha basin
    alpha_phi_seg = [all_phi_seg[i] for i in range(min(len(all_phi_seg), len(all_psi_seg)))
                     if -160 < all_phi_seg[i] < 0 and -120 < all_psi_seg[i] < 30]
    alpha_psi_seg = [all_psi_seg[i] for i in range(min(len(all_phi_seg), len(all_psi_seg)))
                     if -160 < all_phi_seg[i] < 0 and -120 < all_psi_seg[i] < 30]

    # Subsample for clarity
    n_plot = min(3000, len(alpha_phi_real), len(alpha_phi_seg))
    idx_r = rng.choice(len(alpha_phi_real), n_plot, replace=False)
    idx_s = rng.choice(len(alpha_phi_seg), n_plot, replace=False)

    ax_a.scatter([alpha_phi_seg[i] for i in idx_s],
                 [alpha_psi_seg[i] for i in idx_s],
                 s=1, alpha=0.3, c=COLORS['segment'], label='Segment null',
                 rasterized=True)
    ax_a.scatter([alpha_phi_real[i] for i in idx_r],
                 [alpha_psi_real[i] for i in idx_r],
                 s=1, alpha=0.4, c=COLORS['real'], label='Real',
                 rasterized=True)

    ax_a.set_xlim(-120, -20)
    ax_a.set_ylim(-80, 10)
    ax_a.set_xlabel(r'$\phi$ (degrees)')
    ax_a.set_ylabel(r'$\psi$ (degrees)')
    ax_a.set_title(r'A  $\alpha$-basin clustering', fontsize=10, loc='left',
                   fontweight='bold')
    ax_a.legend(loc='upper right', markerscale=5, framealpha=0.9)

    # ── Panel B: W(t) traces ──
    ax_b = fig.add_subplot(gs[1])

    if len(w_traces['real']) > 0:
        n_trace = min(80, len(w_traces['real']))
        x = np.arange(n_trace)
        ax_b.plot(x, w_traces['shuffled'][:n_trace], color=COLORS['shuffled'],
                  alpha=0.7, linewidth=0.8, label='Shuffled')
        ax_b.plot(x, w_traces['segment'][:n_trace], color=COLORS['segment'],
                  alpha=0.8, linewidth=0.8, label='Segment null')
        ax_b.plot(x, w_traces['real'][:n_trace], color=COLORS['real'],
                  linewidth=1.0, label='Real')
        ax_b.set_xlabel('Residue position')
        ax_b.set_ylabel('W(t)')
        ax_b.legend(loc='upper right', framealpha=0.9)

    ax_b.set_title('B  Backbone W(t) traces', fontsize=10, loc='left',
                    fontweight='bold')

    # ── Panel C: Roughness decomposition bar chart ──
    ax_c = fig.add_subplot(gs[2])

    # Compute decomposition values from actual data
    decomp_real = []
    decomp_seg = []
    decomp_m1 = []
    decomp_shuf = []
    decomp_rng = np.random.default_rng(42)

    for pp, ss in zip(all_phi_psi_list, all_ss_list):
        if len(pp) < 50:
            continue
        phi = np.array([p[0] for p in pp])
        psi = np.array([p[1] for p in pp])
        bps_r, _ = compute_bps_l(phi, psi, W_grid, grid_size)
        decomp_real.append(bps_r)

        seg_pp = null_segment_preserving(pp, ss, decomp_rng)
        seg_phi = np.array([p[0] for p in seg_pp])
        seg_psi = np.array([p[1] for p in seg_pp])
        bps_s, _ = compute_bps_l(seg_phi, seg_psi, W_grid, grid_size)
        decomp_seg.append(bps_s)

        m1_pp = null_markov_1(pp, ss, decomp_rng)
        m1_phi = np.array([p[0] for p in m1_pp])
        m1_psi = np.array([p[1] for p in m1_pp])
        bps_m, _ = compute_bps_l(m1_phi, m1_psi, W_grid, grid_size)
        decomp_m1.append(bps_m)

        shuf_pp = null_shuffled(list(pp), decomp_rng)
        shuf_phi = np.array([p[0] for p in shuf_pp])
        shuf_psi = np.array([p[1] for p in shuf_pp])
        bps_sh, _ = compute_bps_l(shuf_phi, shuf_psi, W_grid, grid_size)
        decomp_shuf.append(bps_sh)

    if not decomp_real:
        print("    WARNING: No data for Panel C decomposition.")
        plt.savefig(os.path.join(fig_dir, 'fig1_conceptual.png'))
        plt.close()
        return

    mean_real = float(np.mean(decomp_real))
    mean_seg = float(np.mean(decomp_seg))
    mean_m1 = float(np.mean(decomp_m1))
    mean_shuf = float(np.mean(decomp_shuf))

    labels = ['Real', 'Seg', 'M1', 'Shuf']
    values = [mean_real, mean_seg, mean_m1, mean_shuf]
    colors = [COLORS['real'], COLORS['segment'],
              COLORS['m1'], COLORS['shuffled']]

    bars = ax_c.bar(labels, values, color=colors, edgecolor='white',
                    linewidth=0.5, width=0.7)
    ax_c.set_ylabel('BPS/L')
    ax_c.set_ylim(0, max(values) * 1.25)
    ax_c.set_title('C  Decomposition', fontsize=10, loc='left',
                    fontweight='bold')

    # Add value labels on bars
    for bar, v in zip(bars, values):
        ax_c.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + 0.02,
                  f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    # Add computed ratio annotations
    seg_real_ratio = mean_seg / mean_real if mean_real > 0 else 0
    shuf_m1_ratio = mean_shuf / mean_m1 if mean_m1 > 0 else 0
    ax_c.annotate(f'{seg_real_ratio:.2f}x', xy=(0.5, max(mean_real, mean_seg) + 0.05),
                  fontsize=7, ha='center', color='#333333')
    ax_c.annotate(f'{shuf_m1_ratio:.2f}x', xy=(2.5, max(mean_m1, mean_shuf) + 0.05),
                  fontsize=7, ha='center', color='#333333')

    plt.savefig(os.path.join(fig_dir, 'fig1_conceptual.png'))
    plt.savefig(os.path.join(fig_dir, 'fig1_conceptual.pdf'))
    plt.close()
    print("    Saved fig1_conceptual.png/pdf")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Five-Level Null Model Hierarchy (detailed)
# ═══════════════════════════════════════════════════════════════════

def generate_fig2(data_dir, W_grid, grid_size, sample_proteins, fig_dir):
    """
    Fig 2: Five-level null model hierarchy with error bars.
    Computes actual BPS/L for each null level from sample proteins.
    """
    print("  Generating Figure 2: Null model hierarchy...")

    n_trials = 10
    results = {
        'real': [],
        'segment': [],
        'm1': [],
        'shuffled': [],
    }

    rng = np.random.default_rng(42)
    n_done = 0

    for filepath in sample_proteins[:200]:
        try:
            pp, ss = extract_angles(filepath)
            if len(pp) < 50:
                continue

            phi = np.array([p[0] for p in pp])
            psi = np.array([p[1] for p in pp])
            bps_real, _ = compute_bps_l(phi, psi, W_grid, grid_size)
            results['real'].append(bps_real)

            for trial in range(n_trials):
                t_rng = np.random.default_rng(trial * 1000 + n_done)

                # Segment-preserving null
                seg_pp = null_segment_preserving(pp, ss, t_rng)
                seg_phi = np.array([p[0] for p in seg_pp])
                seg_psi = np.array([p[1] for p in seg_pp])
                bps_seg, _ = compute_bps_l(seg_phi, seg_psi, W_grid, grid_size)
                results['segment'].append(bps_seg)

                # M1 null
                m1_pp = null_markov_1(pp, ss, t_rng)
                m1_phi = np.array([p[0] for p in m1_pp])
                m1_psi = np.array([p[1] for p in m1_pp])
                bps_m1, _ = compute_bps_l(m1_phi, m1_psi, W_grid, grid_size)
                results['m1'].append(bps_m1)

                # Shuffled null
                shuf_pp = null_shuffled(list(pp), t_rng)
                shuf_phi = np.array([p[0] for p in shuf_pp])
                shuf_psi = np.array([p[1] for p in shuf_pp])
                bps_shuf, _ = compute_bps_l(shuf_phi, shuf_psi, W_grid, grid_size)
                results['shuffled'].append(bps_shuf)

            n_done += 1
            if n_done % 50 == 0:
                print(f"    {n_done} proteins processed...")

        except Exception:
            continue

    if n_done < 10:
        print("    WARNING: Too few proteins for Fig 2, skipping.")
        return

    # Compute statistics
    means = {
        'Real': np.mean(results['real']),
        'Segment': np.mean(results['segment']),
        'M1': np.mean(results['m1']),
        'Shuffled': np.mean(results['shuffled']),
    }
    sems = {
        'Real': np.std(results['real']) / np.sqrt(len(results['real'])),
        'Segment': np.std(results['segment']) / np.sqrt(len(results['segment'])),
        'M1': np.std(results['m1']) / np.sqrt(len(results['m1'])),
        'Shuffled': np.std(results['shuffled']) / np.sqrt(len(results['shuffled'])),
    }

    # Plot
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    labels = list(means.keys())
    vals = [means[k] for k in labels]
    errs = [sems[k] * 1.96 for k in labels]  # 95% CI
    colors = [COLORS['real'], COLORS['segment'], COLORS['m1'],
              COLORS['shuffled']]

    bars = ax.bar(labels, vals, yerr=errs, color=colors,
                  edgecolor='white', linewidth=0.5, width=0.65,
                  capsize=3, error_kw={'linewidth': 0.8})

    ax.set_ylabel('Mean BPS/L')
    ax.set_title('Five-Level Null Model Hierarchy', fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)

    # Add value labels on bars
    for bar, v, e in zip(bars, vals, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.02,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    # Add ratio annotations
    seg_real = means['Segment'] / means['Real']
    shuf_m1 = means['Shuffled'] / means['M1']
    total = means['Shuffled'] / means['Real']

    # Intra-basin coherence bracket
    y_brace = max(vals) * 1.05
    ax.annotate('', xy=(0, y_brace), xytext=(1, y_brace),
                arrowprops=dict(arrowstyle='<->', color=COLORS['real'], lw=1.2))
    ax.text(0.5, y_brace + 0.03, f'Intra-basin\n{seg_real:.2f}x',
            ha='center', va='bottom', fontsize=7.5, color=COLORS['real'],
            fontweight='bold')

    # Transition architecture bracket
    y_brace2 = max(vals) * 1.12
    ax.annotate('', xy=(2, y_brace2), xytext=(3, y_brace2),
                arrowprops=dict(arrowstyle='<->', color=COLORS['shuffled'], lw=1.2))
    ax.text(2.5, y_brace2 + 0.03, f'Transition\n{shuf_m1:.2f}x',
            ha='center', va='bottom', fontsize=7.5, color=COLORS['shuffled'],
            fontweight='bold')

    # Collapse annotation
    ax.annotate('Collapse (M/S ratio)',
                xy=(1.5, means['Segment']),
                xytext=(1.5, means['Segment'] - 0.25),
                fontsize=7, ha='center', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))

    # N annotation
    ax.text(0.98, 0.02, f'N = {n_done} proteins, {n_trials} trials each',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig2_null_hierarchy.png'))
    plt.savefig(os.path.join(fig_dir, 'fig2_null_hierarchy.pdf'))
    plt.close()
    print(f"    Saved fig2_null_hierarchy.png/pdf  (N={n_done})")
    print(f"    Real={means['Real']:.3f}, Seg={means['Segment']:.3f}, "
          f"M1={means['M1']:.3f}, Shuf={means['Shuffled']:.3f}")
    print(f"    Seg/Real={seg_real:.3f}, Shuf/M1={shuf_m1:.3f}, "
          f"Total={total:.3f}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: Cross-Organism Conservation + Fold-Class Separation
# ═══════════════════════════════════════════════════════════════════

def generate_fig3(org_data, fold_data, fig_dir):
    """
    Fig 3: Two-panel figure.
    (A) Per-organism mean BPS/L (colored by domain)
    (B) Fold-class boxplots
    """
    print("  Generating Figure 3: Conservation + fold-class separation...")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 3.5),
                                      gridspec_kw={'width_ratios': [1.3, 1],
                                                   'wspace': 0.35})

    # ── Panel A: Per-organism BPS/L ──
    # Sort by mean BPS/L
    sorted_orgs = sorted(org_data.keys(), key=lambda x: org_data[x]['mean'])

    y_pos = np.arange(len(sorted_orgs))
    means = [org_data[o]['mean'] for o in sorted_orgs]
    sems = [org_data[o]['sem'] for o in sorted_orgs]
    colors_a = [COLORS['bacteria'] if domain_of(o) == 'bacteria'
                else COLORS['eukaryote'] for o in sorted_orgs]

    ax_a.barh(y_pos, means, xerr=sems, color=colors_a, edgecolor='white',
              linewidth=0.3, height=0.7, capsize=1.5,
              error_kw={'linewidth': 0.5})
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(sorted_orgs, fontsize=6)
    ax_a.set_xlabel('Mean BPS/L')
    ax_a.set_title('A  Cross-organism conservation', fontsize=10,
                    loc='left', fontweight='bold')

    # Grand mean line
    grand_mean = np.mean(means)
    grand_std = np.std(means)
    ax_a.axvline(grand_mean, color='black', linestyle='--', linewidth=0.8,
                 alpha=0.5)
    ax_a.axvspan(grand_mean - grand_std, grand_mean + grand_std,
                 alpha=0.08, color='gray')

    # CV annotation
    cv = grand_std / grand_mean * 100
    ax_a.text(0.97, 0.03, f'CV = {cv:.1f}%\nN = {len(sorted_orgs)} organisms',
              transform=ax_a.transAxes, fontsize=7, ha='right', va='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='gray', alpha=0.9))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS['eukaryote'], label='Eukaryote'),
                       Patch(facecolor=COLORS['bacteria'], label='Bacteria')]
    ax_a.legend(handles=legend_elements, loc='upper right', fontsize=7,
                framealpha=0.9)

    # ── Panel B: Fold-class boxplots ──
    fold_order = ['all-alpha', 'alpha+beta', 'all-beta', 'alpha/beta']
    fold_labels = [r'all-$\alpha$', r'$\alpha$+$\beta$',
                   r'all-$\beta$', r'$\alpha$/$\beta$']
    fold_colors = [COLORS['alpha'], COLORS['alpha_beta'],
                   COLORS['beta'], COLORS['ab']]

    box_data = []
    for fc in fold_order:
        if fc in fold_data and len(fold_data[fc]) > 0:
            # Subsample for boxplot clarity
            vals = fold_data[fc]
            if len(vals) > 5000:
                rng = np.random.default_rng(42)
                vals = rng.choice(vals, 5000, replace=False)
            box_data.append(vals)
        else:
            box_data.append([0])

    bp = ax_b.boxplot(box_data, labels=fold_labels, patch_artist=True,
                      widths=0.6, showfliers=True,
                      flierprops=dict(marker='.', markersize=1, alpha=0.3),
                      medianprops=dict(color='black', linewidth=1),
                      whiskerprops=dict(linewidth=0.6),
                      capprops=dict(linewidth=0.6))

    for patch, color in zip(bp['boxes'], fold_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('gray')

    ax_b.set_ylabel('BPS/L')
    ax_b.set_title('B  Fold-class separation', fontsize=10, loc='left',
                    fontweight='bold')

    # Add N labels
    for i, fc in enumerate(fold_order):
        n = len(fold_data.get(fc, []))
        ax_b.text(i + 1, ax_b.get_ylim()[0] + 0.02,
                  f'N={n}', ha='center', fontsize=6, color='gray')

    # Cohen's d annotation
    if 'all-alpha' in fold_data and 'alpha/beta' in fold_data:
        m1 = np.mean(fold_data['all-alpha'])
        m2 = np.mean(fold_data['alpha/beta'])
        s1 = np.var(fold_data['all-alpha'], ddof=1)
        s2 = np.var(fold_data['alpha/beta'], ddof=1)
        n1 = len(fold_data['all-alpha'])
        n2 = len(fold_data['alpha/beta'])
        sp = math.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
        d = (m2 - m1) / sp if sp > 0 else 0

        ax_b.annotate(f"Cohen's d = {d:.2f}",
                      xy=(2.5, max(m1, m2) + 0.15),
                      fontsize=7, ha='center',
                      bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='lightyellow',
                                edgecolor='gray', alpha=0.9))

    plt.savefig(os.path.join(fig_dir, 'fig3_conservation_foldclass.png'))
    plt.savefig(os.path.join(fig_dir, 'fig3_conservation_foldclass.pdf'))
    plt.close()
    print(f"    Saved fig3_conservation_foldclass.png/pdf")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: Within-Fold-Class Cross-Organism CV
# ═══════════════════════════════════════════════════════════════════

def generate_fig4(org_fold_data, fig_dir):
    """
    Fig 4: Within-fold-class cross-organism CV.
    Shows per-organism means within each fold class, colored by domain.
    """
    print("  Generating Figure 4: Within-fold-class CV...")

    fold_classes = ['all-alpha', 'alpha/beta', 'all-beta', 'alpha+beta']
    fold_labels = [r'all-$\alpha$', r'$\alpha$/$\beta$',
                   r'all-$\beta$', r'$\alpha$+$\beta$']

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.0),
                              sharey=False, gridspec_kw={'wspace': 0.35})

    for idx, (fc, label) in enumerate(zip(fold_classes, fold_labels)):
        ax = axes[idx]
        if fc not in org_fold_data:
            ax.set_title(f'{label}\n(no data)', fontsize=9)
            continue

        fc_data = org_fold_data[fc]
        # Sort by mean
        sorted_orgs = sorted(fc_data.keys(), key=lambda x: fc_data[x])

        y_pos = np.arange(len(sorted_orgs))
        vals = [fc_data[o] for o in sorted_orgs]
        colors = [COLORS['bacteria'] if domain_of(o) == 'bacteria'
                  else COLORS['eukaryote'] for o in sorted_orgs]

        ax.barh(y_pos, vals, color=colors, edgecolor='white',
                linewidth=0.2, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_orgs, fontsize=4.5)

        # Grand mean line
        gm = np.mean(vals)
        gs = np.std(vals)
        cv = gs / gm * 100 if gm > 0 else 0
        ax.axvline(gm, color='black', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.axvspan(gm - gs, gm + gs, alpha=0.08, color='gray')

        ax.set_title(f'{label}\nCV = {cv:.1f}%', fontsize=8, fontweight='bold')
        ax.set_xlabel('Mean BPS/L', fontsize=7)

    plt.savefig(os.path.join(fig_dir, 'fig4_within_foldclass_cv.png'))
    plt.savefig(os.path.join(fig_dir, 'fig4_within_foldclass_cv.pdf'))
    plt.close()
    print(f"    Saved fig4_within_foldclass_cv.png/pdf")


# ═══════════════════════════════════════════════════════════════════
# BONUS: W Landscape Visualization
# ═══════════════════════════════════════════════════════════════════

def generate_fig_W_landscape(W_grid, fig_dir):
    """Visualize the superpotential W(phi, psi) as a heatmap."""
    print("  Generating bonus: W landscape...")

    fig, ax = plt.subplots(figsize=(5, 4.2))

    extent = [-180, 180, -180, 180]
    # Transpose because array is [phi, psi] but imshow wants [row=psi, col=phi]
    im = ax.imshow(W_grid.T, origin='lower', extent=extent, aspect='equal',
                   cmap='viridis_r', interpolation='bilinear')

    ax.set_xlabel(r'$\phi$ (degrees)')
    ax.set_ylabel(r'$\psi$ (degrees)')
    ax.set_title(r'Empirical Superpotential $W(\phi, \psi) = -\ln(P + \epsilon)$',
                 fontsize=10)

    # Add basin labels
    ax.text(-63, -43, r'$\alpha$', color='white', fontsize=14,
            fontweight='bold', ha='center', va='center')
    ax.text(-120, 135, r'$\beta$', color='white', fontsize=14,
            fontweight='bold', ha='center', va='center')
    ax.text(57, 40, r'$\alpha_L$', color='white', fontsize=11,
            fontweight='bold', ha='center', va='center')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('W (lower = more probable)', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_ramachandran_W.png'))
    plt.savefig(os.path.join(fig_dir, 'fig_ramachandran_W.pdf'))
    plt.close()
    print("    Saved fig_ramachandran_W.png/pdf")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TorusFold Figure Generator")
    parser.add_argument("--data", default="alphafold_cache",
                        help="AlphaFold data directory")
    parser.add_argument("--sample", type=int, default=200,
                        help="Max proteins per organism for Fig 1-2 (default: 200)")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--w-path", default=None,
                        help="Explicit path to superpotential_W.npz "
                             "(auto-detected if not specified)")
    parser.add_argument("--figures", nargs='+', default=['all'],
                        help="Which figures to generate (1 2 3 4 W, or 'all')")
    args = parser.parse_args()

    set_pub_style()

    print("=" * 60)
    print("  TorusFold: Publication Figure Generator")
    print("=" * 60)

    # Create figure directory
    fig_dir = os.path.join(args.output, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Build W from the canonical von Mises mixture (no caching needed)
    W_grid = build_W_from_data(grid_size=360)
    grid_size = W_grid.shape[0]

    # Discover files
    organisms = discover_files(args.data)
    total = sum(len(f) for f in organisms.values())
    print(f"  Data: {total} files across {len(organisms)} organisms")

    figs_to_gen = args.figures
    if 'all' in figs_to_gen:
        figs_to_gen = ['W', '1', '2', '3', '4']

    # ══════════════════════════════════════════════════════════════
    # Gather sample proteins for Figs 1 and 2
    # ══════════════════════════════════════════════════════════════
    MAX_FILE_SIZE = 5 * 1024 * 1024
    sample_proteins = []

    if any(f in figs_to_gen for f in ['1', '2']):
        print("\n  Gathering sample proteins for Figs 1-2...")
        rng = np.random.default_rng(42)
        for org_name, files in sorted(organisms.items()):
            n_per_org = min(args.sample, len(files))
            if n_per_org < len(files):
                idx = rng.choice(len(files), n_per_org, replace=False)
                sel = [files[i] for i in idx]
            else:
                sel = files
            for f in sel:
                if os.path.getsize(f) <= MAX_FILE_SIZE:
                    sample_proteins.append(f)
        print(f"    {len(sample_proteins)} sample proteins selected")

    # ══════════════════════════════════════════════════════════════
    # Process full dataset for Figs 3 and 4
    # ══════════════════════════════════════════════════════════════
    org_data = {}        # {org: {mean, sem, n}}
    fold_data = defaultdict(list)  # {fold_class: [bpsl values]}
    org_fold_means = defaultdict(dict)  # {fold_class: {org: mean}}

    if any(f in figs_to_gen for f in ['3', '4']):
        print("\n  Processing full dataset for Figs 3-4...")
        org_fold_bpsl = defaultdict(lambda: defaultdict(list))
        n_processed = 0

        for org_name, files in sorted(organisms.items()):
            rng_org = np.random.default_rng(hash(org_name) % (2**31))
            # Use all files for figs 3-4
            print(f"    {org_name} ({len(files)} files)...",
                  end=" ", flush=True)
            org_bpsl = []
            org_count = 0

            for filepath in files:
                try:
                    fsize = os.path.getsize(filepath)
                    if fsize > MAX_FILE_SIZE:
                        continue

                    pp, ss = extract_angles(filepath)
                    if len(pp) < 50:
                        continue

                    fold = classify_fold(ss)
                    phi = np.array([p[0] for p in pp])
                    psi = np.array([p[1] for p in pp])
                    bpsl, _ = compute_bps_l(phi, psi, W_grid, grid_size)

                    org_bpsl.append(bpsl)
                    if fold != 'other':
                        fold_data[fold].append(bpsl)
                        org_fold_bpsl[org_name][fold].append(bpsl)
                    n_processed += 1
                    org_count += 1

                except Exception:
                    continue

            if org_bpsl:
                org_data[org_name] = {
                    'mean': float(np.mean(org_bpsl)),
                    'sem': float(np.std(org_bpsl) / np.sqrt(len(org_bpsl))),
                    'n': len(org_bpsl),
                }
            print(f"{org_count} proteins")

        # Build per-fold-class per-organism means for Fig 4
        fold_classes = ['all-alpha', 'alpha+beta', 'all-beta', 'alpha/beta']
        for fc in fold_classes:
            for org in org_fold_bpsl:
                vals = org_fold_bpsl[org][fc]
                if len(vals) >= 5:
                    org_fold_means[fc][org] = float(np.mean(vals))

        print(f"    Total: {n_processed} proteins processed")

    # ══════════════════════════════════════════════════════════════
    # Generate figures
    # ══════════════════════════════════════════════════════════════
    print("\n  Generating figures...")

    if 'W' in figs_to_gen:
        generate_fig_W_landscape(W_grid, fig_dir)

    if '1' in figs_to_gen:
        generate_fig1(args.data, W_grid, grid_size, sample_proteins, fig_dir)

    if '2' in figs_to_gen:
        generate_fig2(args.data, W_grid, grid_size, sample_proteins, fig_dir)

    if '3' in figs_to_gen and org_data:
        generate_fig3(org_data, fold_data, fig_dir)

    if '4' in figs_to_gen and org_fold_means:
        generate_fig4(org_fold_means, fig_dir)

    print(f"\n  All figures saved to: {fig_dir}")
    print("  Done.")


if __name__ == "__main__":
    main()
