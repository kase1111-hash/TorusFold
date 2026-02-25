"""
TorusFold: Designed Protein Seg/Real Analysis
==============================================
Computes the full decomposition (Real, Segment, M1, Shuffled) for
de novo designed proteins already downloaded.

Key question: Is Seg/Real lower for designed proteins than natural?
If yes → evolution optimizes intra-basin coherence beyond what design achieves.

Usage:
  python designed_seg_real.py

Reads: data/designed_proteins/*.cif, results/superpotential_W.npz
Writes: results/designed_seg_real_report.md
"""

import os
import sys
import math
import time
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


def extract_angles_pdb(filepath):
    """Extract (phi, psi) and SS. No pLDDT filter for experimental structures."""
    st = gemmi.read_structure(str(filepath))
    if len(st) == 0 or len(st[0]) == 0:
        return [], [], None, 0
    
    # Get experimental method
    method = st.find_mmcif_category('_exptl.method') if hasattr(st, 'find_mmcif_category') else None
    exp_method = "unknown"
    try:
        block = gemmi.cif.read(str(filepath))[0]
        m = block.find_value('_exptl.method')
        if m:
            exp_method = m.strip().strip("'\"").upper()
    except Exception:
        pass

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
            residues.append(atoms)
    if len(residues) < 3:
        return [], [], exp_method, 0

    phi_psi = []
    ss_seq = []
    for i in range(1, len(residues) - 1):
        phi = -_dihedral_angle(residues[i-1]['C'], residues[i]['N'],
                               residues[i]['CA'], residues[i]['C'])
        psi = -_dihedral_angle(residues[i]['N'], residues[i]['CA'],
                               residues[i]['C'], residues[i+1]['N'])
        phi_d, psi_d = math.degrees(phi), math.degrees(psi)
        if -160 < phi_d < 0 and -120 < psi_d < 30:
            ss = 'a'
        elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
            ss = 'b'
        else:
            ss = 'o'
        phi_psi.append((phi, psi))
        ss_seq.append(ss)
    return phi_psi, ss_seq, exp_method, len(residues)


def compute_bps_l(phi_arr, psi_arr, W_grid, grid_size):
    if len(phi_arr) < 2:
        return 0.0
    scale = grid_size / 360.0
    phi_d = np.degrees(phi_arr)
    psi_d = np.degrees(psi_arr)
    gi = (np.round((phi_d + 180) * scale).astype(int)) % grid_size
    gj = (np.round((psi_d + 180) * scale).astype(int)) % grid_size
    w = W_grid[gi, gj]
    return float(np.mean(np.abs(np.diff(w))))


def null_segment(phi_psi, ss_seq, rng):
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
    phi = np.array([p[0] for p in phi_psi])
    psi = np.array([p[1] for p in phi_psi])
    perm = rng.permutation(len(phi))
    return phi[perm], psi[perm]


def evaluate_protein(phi_psi, ss_seq, W_grid, grid_size, rng, n_trials=10):
    phi = np.array([p[0] for p in phi_psi])
    psi = np.array([p[1] for p in phi_psi])
    real = compute_bps_l(phi, psi, W_grid, grid_size)

    seg_v, m1_v, shuf_v = [], [], []
    for _ in range(n_trials):
        sp, sq = null_segment(phi_psi, ss_seq, rng)
        seg_v.append(compute_bps_l(sp, sq, W_grid, grid_size))
        mp, mq = null_markov1(phi_psi, ss_seq, rng)
        m1_v.append(compute_bps_l(mp, mq, W_grid, grid_size))
        sfp, sfq = null_shuffled(phi_psi, rng)
        shuf_v.append(compute_bps_l(sfp, sfq, W_grid, grid_size))

    return {
        'real': real,
        'segment': float(np.mean(seg_v)),
        'markov1': float(np.mean(m1_v)),
        'shuffled': float(np.mean(shuf_v)),
        'seg_real': float(np.mean(seg_v) / real) if real > 0 else 0,
        'm1_real': float(np.mean(m1_v) / real) if real > 0 else 0,
        's_real': float(np.mean(shuf_v) / real) if real > 0 else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Designed Protein Seg/Real Decomposition")
    print("=" * 60)

    data_dir = "data/designed_proteins"
    output_dir = "results"

    # Load W
    w_path = os.path.join(output_dir, "superpotential_W.npz")
    if not os.path.exists(w_path):
        print(f"ERROR: W not found at {w_path}")
        sys.exit(1)
    data = np.load(w_path)
    W_grid = data['grid']
    grid_size = W_grid.shape[0]

    cif_files = sorted(Path(data_dir).glob("*.cif"))
    print(f"  Found {len(cif_files)} CIF files")

    rng = np.random.default_rng(42)
    MIN_LENGTH = 50

    all_results = []
    xray_results = []

    for filepath in cif_files:
        pdb_id = filepath.stem.upper()
        try:
            pp, ss, method, chain_len = extract_angles_pdb(str(filepath))
            if len(pp) < 20:
                continue

            is_xray = method and ('X-RAY' in method or 'ELECTRON' in method
                                  or 'CRYO' in method)
            is_long = chain_len >= MIN_LENGTH

            print(f"  {pdb_id:6s} len={chain_len:4d} method={method:30s}", end="", flush=True)

            d = evaluate_protein(pp, ss, W_grid, grid_size, rng, n_trials=10)
            d['pdb_id'] = pdb_id
            d['length'] = chain_len
            d['method'] = method
            d['is_xray'] = is_xray

            all_results.append(d)
            if is_xray and is_long:
                xray_results.append(d)

            print(f"  BPS/L={d['real']:.3f}  Seg/Real={d['seg_real']:.2f}x")

        except Exception as e:
            print(f"  {pdb_id}: ERROR {e}")

    print(f"\n  Total: {len(all_results)} proteins")
    print(f"  X-ray/cryo-EM, length≥{MIN_LENGTH}: {len(xray_results)} proteins")

    # ══════════════════════════════════════════════════════════════
    # REPORT
    # ══════════════════════════════════════════════════════════════
    report_path = os.path.join(output_dir, "designed_seg_real_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Designed Protein Seg/Real Decomposition\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")

        # Summary comparison
        f.write("## Key Comparison\n\n")
        f.write("| Dataset | N | BPS/L | Seg/Real | M1/Real | S/Real |\n")
        f.write("|---|---|---|---|---|---|\n")

        # Natural reference (from higher-order null + W independence)
        f.write("| Natural (AlphaFold) | 82,113 | 1.275 | 1.30x* | 1.30x* | 1.47x* |\n")

        if all_results:
            bpsl = np.mean([r['real'] for r in all_results])
            sr = np.mean([r['seg_real'] for r in all_results])
            mr = np.mean([r['m1_real'] for r in all_results])
            shr = np.mean([r['s_real'] for r in all_results])
            f.write(f"| Designed (all) | {len(all_results)} | "
                    f"{bpsl:.3f} | {sr:.2f}x | {mr:.2f}x | {shr:.2f}x |\n")

        if xray_results:
            bpsl = np.mean([r['real'] for r in xray_results])
            sr = np.mean([r['seg_real'] for r in xray_results])
            mr = np.mean([r['m1_real'] for r in xray_results])
            shr = np.mean([r['s_real'] for r in xray_results])
            f.write(f"| Designed (X-ray, ≥{MIN_LENGTH}res) | {len(xray_results)} | "
                    f"{bpsl:.3f} | {sr:.2f}x | {mr:.2f}x | {shr:.2f}x |\n")

        f.write("\n*Train/test canonical values from W independence analysis\n\n")

        # Method breakdown
        f.write("## By Experimental Method\n\n")
        method_groups = defaultdict(list)
        for r in all_results:
            m = r['method'] if r['method'] else 'unknown'
            if 'X-RAY' in m:
                method_groups['X-ray'].append(r)
            elif 'NMR' in m.upper():
                method_groups['NMR'].append(r)
            elif 'ELECTRON' in m or 'CRYO' in m:
                method_groups['Cryo-EM'].append(r)
            else:
                method_groups['Other'].append(r)

        f.write("| Method | N | Mean BPS/L | Seg/Real | S/Real |\n")
        f.write("|---|---|---|---|---|\n")
        for meth in ['X-ray', 'Cryo-EM', 'NMR', 'Other']:
            if meth in method_groups:
                rs = method_groups[meth]
                bpsl = np.mean([r['real'] for r in rs])
                sr = np.mean([r['seg_real'] for r in rs])
                shr = np.mean([r['s_real'] for r in rs])
                f.write(f"| {meth} | {len(rs)} | {bpsl:.3f} | "
                        f"{sr:.2f}x | {shr:.2f}x |\n")
        f.write("\n")

        # Length breakdown
        f.write("## By Chain Length\n\n")
        f.write("| Length range | N | Mean BPS/L | Seg/Real |\n")
        f.write("|---|---|---|---|\n")
        for lo, hi, label in [(20, 49, "20-49"), (50, 99, "50-99"),
                               (100, 199, "100-199"), (200, 9999, "200+")]:
            subset = [r for r in all_results if lo <= r['length'] <= hi]
            if subset:
                bpsl = np.mean([r['real'] for r in subset])
                sr = np.mean([r['seg_real'] for r in subset])
                f.write(f"| {label} | {len(subset)} | {bpsl:.3f} | {sr:.2f}x |\n")
        f.write("\n")

        # Individual (X-ray, ≥50 only)
        f.write("## Individual Structures (X-ray/cryo-EM, ≥50 residues)\n\n")
        f.write("| PDB ID | Length | BPS/L | Seg/Real | M1/Real | S/Real |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in sorted(xray_results, key=lambda x: x['real']):
            f.write(f"| {r['pdb_id']} | {r['length']} | {r['real']:.3f} | "
                    f"{r['seg_real']:.2f}x | {r['m1_real']:.2f}x | "
                    f"{r['s_real']:.2f}x |\n")
        f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        if xray_results:
            sr_designed = np.mean([r['seg_real'] for r in xray_results])
            sr_natural = 1.30  # canonical train/test value
            diff = sr_designed - sr_natural
            pct = abs(diff) / sr_natural * 100

            if abs(diff) < 0.05:
                f.write(f"Filtered designed proteins show Seg/Real = {sr_designed:.2f}x, "
                        f"**matching** natural proteins ({sr_natural:.2f}x, Δ={pct:.0f}%).\n\n"
                        f"Intra-basin coherence is identical regardless of origin — "
                        f"**pure stereochemical constraint**.\n")
            elif diff > 0:
                f.write(f"Filtered designed proteins show Seg/Real = {sr_designed:.2f}x, "
                        f"**lower** intra-basin coherence than natural ({sr_natural:.2f}x).\n\n"
                        f"Designed helices cluster less tightly inside basins. "
                        f"Evolution achieves tighter micro-geometric optimization "
                        f"than current computational design methods.\n")
            else:
                f.write(f"Filtered designed proteins show Seg/Real = {sr_designed:.2f}x, "
                        f"**tighter** coherence than natural ({sr_natural:.2f}x).\n\n"
                        f"Computational design achieves even tighter intra-basin "
                        f"clustering than evolution.\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
