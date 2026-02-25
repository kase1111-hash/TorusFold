"""
TorusFold: Higher-Order Null Model Analysis
============================================
Tests whether the Markov/Real gap reflects:
  (a) intra-basin conformational coherence, or
  (b) segment-length structure (dwell-time distribution)

Implements FOUR levels:
  1. Real          - observed backbone
  2. Segment       - preserves segment boundaries, randomizes within-segment
                     positions from global basin pool
  3. Markov (1st)  - preserves one-step transition probabilities,
                     randomizes everything else
  4. Shuffled      - random permutation of all (phi, psi)

If Real < Segment < Markov < Shuffled:
  → The Real/Segment gap = intra-basin coherence (tight clustering)
  → The Segment/Markov gap = segment-length structure (dwell times)
  → Both contribute independently. Paper claim is clean.

If Real ≈ Segment:
  → Intra-basin coherence is negligible
  → The original M/R gap was segment-length structure
  → Paper needs reframing

Usage:
  python higher_order_null.py [--sample N] [--trials T]

Reads: alphafold_cache/ (CIF files), results/superpotential_W.npz
Writes: results/higher_order_null_report.md
"""

import os
import sys
import math
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi required. pip install gemmi")
    sys.exit(1)

from scipy.ndimage import gaussian_filter


# ═══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS (copied from pipeline to keep self-contained)
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


def extract_dihedrals(filepath, plddt_min=70.0):
    """Extract (phi, psi, plddt, ss) from a CIF file."""
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
                'plddt': ca.b_iso if ca else 0, 'resname': res.name,
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
        plddt = residues[i]['plddt']

        if plddt < plddt_min:
            continue

        phi_d = math.degrees(phi)
        psi_d = math.degrees(psi)

        # SS assignment
        if -160 < phi_d < 0 and -120 < psi_d < 30:
            ss = 'a'
        elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
            ss = 'b'
        else:
            ss = 'o'

        phi_psi.append((phi, psi))
        ss_seq.append(ss)

    return phi_psi, ss_seq


class Superpotential:
    """Lookup W values from cached grid."""
    def __init__(self, grid):
        self.grid = grid

    def lookup(self, phi_rad, psi_rad):
        phi_deg = math.degrees(phi_rad) % 360
        psi_deg = math.degrees(psi_rad) % 360
        i = phi_deg if phi_deg < 180 else phi_deg - 360
        j = psi_deg if psi_deg < 180 else psi_deg - 360
        gi = int(i + 180) % 360
        gj = int(j + 180) % 360
        return float(self.grid[gi, gj])

    def lookup_array(self, phi_arr, psi_arr):
        phi_deg = np.degrees(phi_arr) % 360
        psi_deg = np.degrees(psi_arr) % 360
        phi_idx = np.where(phi_deg < 180, phi_deg, phi_deg - 360)
        psi_idx = np.where(psi_deg < 180, psi_deg, psi_deg - 360)
        gi = (phi_idx.astype(int) + 180) % 360
        gj = (psi_idx.astype(int) + 180) % 360
        return self.grid[gi, gj]

    @staticmethod
    def load(filepath):
        data = np.load(filepath)
        return Superpotential(data['grid'])


def compute_bps_l(phi_arr, psi_arr, W):
    """BPS/L from arrays of phi, psi."""
    if len(phi_arr) < 2:
        return 0.0
    w_vals = W.lookup_array(phi_arr, psi_arr)
    return float(np.mean(np.abs(np.diff(w_vals))))


# ═══════════════════════════════════════════════════════════════════
# SEGMENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_segments(ss_seq):
    """Extract contiguous SS segments from a basin sequence.

    Returns list of (start_idx, end_idx, basin_type).
    E.g., 'aaabbboaa' → [(0,2,'a'), (3,5,'b'), (6,6,'o'), (7,8,'a')]
    """
    if not ss_seq:
        return []

    segments = []
    start = 0
    for i in range(1, len(ss_seq)):
        if ss_seq[i] != ss_seq[start]:
            segments.append((start, i - 1, ss_seq[start]))
            start = i
    segments.append((start, len(ss_seq) - 1, ss_seq[start]))
    return segments


# ═══════════════════════════════════════════════════════════════════
# NULL MODELS
# ═══════════════════════════════════════════════════════════════════

def null_segment_preserving(phi_psi, ss_seq, rng):
    """Segment-preserving null: keep segment boundaries, randomize
    within-segment positions from the global basin pool.

    This preserves:
      - Segment count and lengths (dwell-time distribution)
      - Basin transition sequence
      - Basin occupancy fractions

    This destroys:
      - Intra-basin positional coherence (tight clustering within segments)
      - Sequential correlation within a segment
    """
    phi_arr = np.array([p[0] for p in phi_psi])
    psi_arr = np.array([p[1] for p in phi_psi])

    # Build global basin pools
    basin_pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        basin_pools[s].append(i)

    # For each residue, replace with random draw from same basin's global pool
    # BUT preserve segment boundaries (don't mix across segments)
    segments = extract_segments(ss_seq)

    new_phi = np.empty_like(phi_arr)
    new_psi = np.empty_like(psi_arr)

    for seg_start, seg_end, basin in segments:
        pool = basin_pools[basin]
        for i in range(seg_start, seg_end + 1):
            j = rng.choice(pool)
            new_phi[i] = phi_arr[j]
            new_psi[i] = psi_arr[j]

    return new_phi, new_psi


def null_markov_order1(phi_psi, ss_seq, rng):
    """First-order Markov null: preserve one-step basin transition
    probabilities, randomize within-basin positions from global pool.

    This preserves:
      - One-step transition frequencies
      - Basin occupancy fractions

    This destroys:
      - Segment lengths (dwell-time distribution)
      - Intra-basin coherence
    """
    phi_arr = np.array([p[0] for p in phi_psi])
    psi_arr = np.array([p[1] for p in phi_psi])

    # Build basin pools
    basin_pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        basin_pools[s].append(i)

    # Build transition matrix
    basins = sorted(set(ss_seq))
    basin_idx = {b: i for i, b in enumerate(basins)}
    n_basins = len(basins)
    trans = np.zeros((n_basins, n_basins))
    for i in range(len(ss_seq) - 1):
        trans[basin_idx[ss_seq[i]], basin_idx[ss_seq[i + 1]]] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans / row_sums

    # Generate Markov chain
    n = len(ss_seq)
    markov_ss = [ss_seq[0]]
    for i in range(1, n):
        curr_basin = basin_idx[markov_ss[-1]]
        next_basin = rng.choice(n_basins, p=trans_prob[curr_basin])
        markov_ss.append(basins[next_basin])

    # Fill with random positions from basin pools
    new_phi = np.empty(n)
    new_psi = np.empty(n)
    for i, s in enumerate(markov_ss):
        pool = basin_pools[s]
        j = rng.choice(pool)
        new_phi[i] = phi_arr[j]
        new_psi[i] = psi_arr[j]

    return new_phi, new_psi


def null_markov_order2(phi_psi, ss_seq, rng):
    """Second-order Markov null: preserve two-step basin transition
    probabilities (bigram context), randomize within-basin positions.

    This preserves:
      - Two-step transition frequencies (P(basin_i | basin_{i-1}, basin_{i-2}))
      - Partially preserves segment-length distribution
      - Basin occupancy fractions

    This destroys:
      - Higher-order correlations
      - Intra-basin coherence
    """
    phi_arr = np.array([p[0] for p in phi_psi])
    psi_arr = np.array([p[1] for p in phi_psi])

    basin_pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        basin_pools[s].append(i)

    # Build second-order transition counts
    basins = sorted(set(ss_seq))
    basin_idx = {b: i for i, b in enumerate(basins)}
    n_basins = len(basins)

    # trans2[prev2][prev1] → counts for next
    trans2 = np.zeros((n_basins, n_basins, n_basins))
    for i in range(2, len(ss_seq)):
        b0 = basin_idx[ss_seq[i - 2]]
        b1 = basin_idx[ss_seq[i - 1]]
        b2 = basin_idx[ss_seq[i]]
        trans2[b0, b1, b2] += 1

    # Normalize
    trans2_prob = np.zeros_like(trans2)
    for i in range(n_basins):
        for j in range(n_basins):
            total = trans2[i, j].sum()
            if total > 0:
                trans2_prob[i, j] = trans2[i, j] / total
            else:
                # Fall back to uniform
                trans2_prob[i, j] = 1.0 / n_basins

    # Generate second-order Markov chain
    n = len(ss_seq)
    if n < 2:
        return phi_arr.copy(), psi_arr.copy()

    markov_ss = [ss_seq[0], ss_seq[1]]
    for i in range(2, n):
        b0 = basin_idx[markov_ss[-2]]
        b1 = basin_idx[markov_ss[-1]]
        next_basin = rng.choice(n_basins, p=trans2_prob[b0, b1])
        markov_ss.append(basins[next_basin])

    new_phi = np.empty(n)
    new_psi = np.empty(n)
    for i, s in enumerate(markov_ss):
        pool = basin_pools[s]
        j = rng.choice(pool)
        new_phi[i] = phi_arr[j]
        new_psi[i] = psi_arr[j]

    return new_phi, new_psi


def null_shuffled(phi_psi, rng):
    """Shuffled null: random permutation of all (phi, psi)."""
    phi_arr = np.array([p[0] for p in phi_psi])
    psi_arr = np.array([p[1] for p in phi_psi])
    perm = rng.permutation(len(phi_arr))
    return phi_arr[perm], psi_arr[perm]


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_protein(filepath, W, n_trials=10, rng=None):
    """Compute five-level decomposition for one protein.

    Returns dict with BPS/L for each level, or None if protein is too short.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    phi_psi, ss_seq = extract_dihedrals(filepath)
    if len(phi_psi) < 50:
        return None

    phi_arr = np.array([p[0] for p in phi_psi])
    psi_arr = np.array([p[1] for p in phi_psi])

    # Real
    real_bpsl = compute_bps_l(phi_arr, psi_arr, W)

    # Compute segment stats for reporting
    segments = extract_segments(ss_seq)
    seg_lengths = [e - s + 1 for s, e, _ in segments]

    # Null models (averaged over trials)
    seg_vals = []
    m1_vals = []
    m2_vals = []
    shuf_vals = []

    for _ in range(n_trials):
        # Segment-preserving
        sp, sq = null_segment_preserving(phi_psi, ss_seq, rng)
        seg_vals.append(compute_bps_l(sp, sq, W))

        # First-order Markov
        mp, mq = null_markov_order1(phi_psi, ss_seq, rng)
        m1_vals.append(compute_bps_l(mp, mq, W))

        # Second-order Markov
        m2p, m2q = null_markov_order2(phi_psi, ss_seq, rng)
        m2_vals.append(compute_bps_l(m2p, m2q, W))

        # Shuffled
        sfp, sfq = null_shuffled(phi_psi, rng)
        shuf_vals.append(compute_bps_l(sfp, sfq, W))

    return {
        'real': real_bpsl,
        'segment': float(np.mean(seg_vals)),
        'markov1': float(np.mean(m1_vals)),
        'markov2': float(np.mean(m2_vals)),
        'shuffled': float(np.mean(shuf_vals)),
        'n_residues': len(phi_psi),
        'n_segments': len(segments),
        'mean_seg_len': float(np.mean(seg_lengths)),
        'ss_composition': {
            'a': ss_seq.count('a') / len(ss_seq),
            'b': ss_seq.count('b') / len(ss_seq),
            'o': ss_seq.count('o') / len(ss_seq),
        },
    }


def discover_cif_files(data_dir, max_files=None):
    """Find CIF files across organism subdirectories."""
    data_path = Path(data_dir)
    all_files = []
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        files = sorted(subdir.glob("*.cif"))
        for f in files:
            all_files.append((str(f), subdir.name))
    if max_files and len(all_files) > max_files:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_files), max_files, replace=False)
        all_files = [all_files[i] for i in sorted(indices)]
    return all_files


def main():
    parser = argparse.ArgumentParser(description="Higher-Order Null Model Analysis")
    parser.add_argument("--data", default="alphafold_cache",
                        help="Path to AlphaFold data directory")
    parser.add_argument("--w-cache", default="results/superpotential_W.npz",
                        help="Path to cached superpotential")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of structures to sample (default: 200)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Null model trials per structure (default: 10)")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold: Higher-Order Null Model Analysis")
    print("=" * 60)

    # Load W
    if not os.path.exists(args.w_cache):
        print(f"ERROR: Superpotential not found at {args.w_cache}")
        print("Run the main pipeline first to generate it.")
        sys.exit(1)

    W = Superpotential.load(args.w_cache)
    print(f"  Loaded W from {args.w_cache}")

    # Discover files
    files = discover_cif_files(args.data, args.sample)
    print(f"  Found {len(files)} structures to analyze")
    print(f"  Trials per structure: {args.trials}")
    print()

    # Analyze
    results = []
    rng = np.random.default_rng(42)
    start = time.time()

    for idx, (filepath, organism) in enumerate(files):
        try:
            r = analyze_protein(filepath, W, n_trials=args.trials, rng=rng)
            if r is not None:
                r['organism'] = organism
                r['filepath'] = os.path.basename(filepath)
                results.append(r)
        except Exception as e:
            pass

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (idx + 1) / elapsed
            eta = (len(files) - idx - 1) / rate
            print(f"  {idx+1}/{len(files)} ({rate:.1f}/sec, ETA {eta:.0f}s) "
                  f"— {len(results)} valid")

    elapsed = time.time() - start
    print(f"\n  Processed {len(results)} proteins in {elapsed:.0f}s")

    if len(results) < 10:
        print("  ERROR: Too few valid results. Check data path.")
        sys.exit(1)

    # ── Aggregate ──
    real_vals = np.array([r['real'] for r in results])
    seg_vals = np.array([r['segment'] for r in results])
    m1_vals = np.array([r['markov1'] for r in results])
    m2_vals = np.array([r['markov2'] for r in results])
    shuf_vals = np.array([r['shuffled'] for r in results])

    real_mean = float(np.mean(real_vals))
    seg_mean = float(np.mean(seg_vals))
    m1_mean = float(np.mean(m1_vals))
    m2_mean = float(np.mean(m2_vals))
    shuf_mean = float(np.mean(shuf_vals))

    print()
    print("  ════════════════════════════════════════")
    print("  FIVE-LEVEL DECOMPOSITION")
    print("  ════════════════════════════════════════")
    print(f"  Real:              {real_mean:.3f}")
    print(f"  Segment-preserving:{seg_mean:.3f}  (Seg/Real = {seg_mean/real_mean:.2f}x)")
    print(f"  Markov (2nd order):{m2_mean:.3f}  (M2/Real = {m2_mean/real_mean:.2f}x)")
    print(f"  Markov (1st order):{m1_mean:.3f}  (M1/Real = {m1_mean/real_mean:.2f}x)")
    print(f"  Shuffled:          {shuf_mean:.3f}  (S/Real  = {shuf_mean/real_mean:.2f}x)")
    print("  ════════════════════════════════════════")
    print()

    # Interpretation
    seg_real_gap = seg_mean / real_mean
    m2_seg_gap = m2_mean / seg_mean
    m1_m2_gap = m1_mean / m2_mean
    shuf_m1_gap = shuf_mean / m1_mean

    print("  GAP DECOMPOSITION:")
    print(f"  Real → Segment:   {seg_real_gap:.3f}x = intra-basin coherence")
    print(f"  Segment → M2:     {m2_seg_gap:.3f}x = segment-length structure (partial)")
    print(f"  M2 → M1:          {m1_m2_gap:.3f}x = higher-order transition memory")
    print(f"  M1 → Shuffled:    {shuf_m1_gap:.3f}x = transition architecture")
    print()

    if seg_real_gap > 1.05:
        print("  ★ Real < Segment: intra-basin coherence is REAL")
        print("    (consecutive residues cluster tighter than random basin draws)")
    elif seg_real_gap > 1.01:
        print("  ~ Real ≈ Segment: intra-basin coherence is WEAK")
        print("    (most of the M/R gap was segment-length structure)")
    else:
        print("  ✗ Real ≈ Segment: intra-basin coherence is NEGLIGIBLE")
        print("    (the original M/R gap was entirely segment-length structure)")

    # ── Per-organism breakdown ──
    org_results = defaultdict(list)
    for r in results:
        org_results[r['organism']].append(r)

    # ── Write report ──
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "higher_order_null_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Higher-Order Null Model Analysis\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Structures:** {len(results)}\n")
        f.write(f"**Trials per structure:** {args.trials}\n\n")

        f.write("## Five-Level Decomposition\n\n")
        f.write("| Level | Mean BPS/L | Ratio to Real | What it preserves |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Real | {real_mean:.3f} | 1.00x | Everything |\n")
        f.write(f"| Segment-preserving | {seg_mean:.3f} | {seg_mean/real_mean:.2f}x | "
                f"Segment boundaries + basin sequence |\n")
        f.write(f"| Markov (2nd order) | {m2_mean:.3f} | {m2_mean/real_mean:.2f}x | "
                f"Bigram transitions |\n")
        f.write(f"| Markov (1st order) | {m1_mean:.3f} | {m1_mean/real_mean:.2f}x | "
                f"Unigram transitions |\n")
        f.write(f"| Shuffled | {shuf_mean:.3f} | {shuf_mean/real_mean:.2f}x | "
                f"Basin occupancy only |\n")
        f.write("\n")

        f.write("## Gap Decomposition\n\n")
        f.write("| Gap | Ratio | Interpretation |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Real → Segment | {seg_real_gap:.3f}x | Intra-basin coherence |\n")
        f.write(f"| Segment → M2 | {m2_seg_gap:.3f}x | Segment-length structure |\n")
        f.write(f"| M2 → M1 | {m1_m2_gap:.3f}x | Higher-order transition memory |\n")
        f.write(f"| M1 → Shuffled | {shuf_m1_gap:.3f}x | Transition architecture |\n")
        f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        if seg_real_gap > 1.05:
            f.write("**Intra-basin coherence is a real, independent signal.** "
                    "Real proteins achieve lower roughness than segment-preserving "
                    "surrogates that maintain identical segment boundaries but randomize "
                    "within-segment positions. This confirms that the Markov/Real gap "
                    "reported in the main analysis is not solely a segment-length artifact.\n\n")
            f.write(f"The gap decomposes as: intra-basin coherence ({seg_real_gap:.2f}x) + "
                    f"segment structure ({m2_seg_gap:.2f}x) + transition memory "
                    f"({m1_m2_gap:.2f}x) + transition architecture ({shuf_m1_gap:.2f}x) = "
                    f"total {shuf_mean/real_mean:.2f}x.\n\n")
        else:
            f.write("**Intra-basin coherence is weak or negligible.** "
                    "The segment-preserving null achieves similar roughness to real proteins, "
                    "indicating that most of the original Markov/Real gap was due to "
                    "segment-length structure rather than tight within-basin clustering.\n\n")

        # Per-organism
        if len(org_results) > 1:
            f.write("## Per-Organism Results\n\n")
            f.write("| Organism | N | Real | Segment | M1 | Shuffled | Seg/Real |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for org in sorted(org_results, key=lambda x: len(org_results[x]), reverse=True):
                rs = org_results[org]
                if len(rs) < 3:
                    continue
                or_real = np.mean([r['real'] for r in rs])
                or_seg = np.mean([r['segment'] for r in rs])
                or_m1 = np.mean([r['markov1'] for r in rs])
                or_shuf = np.mean([r['shuffled'] for r in rs])
                f.write(f"| {org} | {len(rs)} | {or_real:.3f} | {or_seg:.3f} | "
                        f"{or_m1:.3f} | {or_shuf:.3f} | {or_seg/or_real:.2f}x |\n")
            f.write("\n")

        # Per-protein details (first 20)
        f.write("## Sample Per-Protein Results (first 20)\n\n")
        f.write("| Protein | Res | Real | Segment | M2 | M1 | Shuffled | Seg/Real |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results[:20]:
            f.write(f"| {r['filepath'][:30]} | {r['n_residues']} | "
                    f"{r['real']:.3f} | {r['segment']:.3f} | {r['markov2']:.3f} | "
                    f"{r['markov1']:.3f} | {r['shuffled']:.3f} | "
                    f"{r['segment']/r['real']:.2f}x |\n")
        f.write("\n")

        # Summary
        f.write("## Summary\n\n")
        f.write("```\n")
        f.write("HIGHER-ORDER NULL MODEL ANALYSIS\n")
        f.write("═══════════════════════════════════════════\n")
        f.write(f"Structures:    {len(results)}\n")
        f.write(f"Real:          {real_mean:.3f}\n")
        f.write(f"Segment:       {seg_mean:.3f}  ({seg_real_gap:.2f}x)\n")
        f.write(f"Markov-2:      {m2_mean:.3f}  ({m2_mean/real_mean:.2f}x)\n")
        f.write(f"Markov-1:      {m1_mean:.3f}  ({m1_mean/real_mean:.2f}x)\n")
        f.write(f"Shuffled:      {shuf_mean:.3f}  ({shuf_mean/real_mean:.2f}x)\n")
        f.write(f"Intra-basin coherence: {seg_real_gap:.2f}x\n")
        f.write("═══════════════════════════════════════════\n")
        f.write("```\n")

    print(f"  Report: {report_path}")
    print()
    print("  Done.")


if __name__ == "__main__":
    main()
