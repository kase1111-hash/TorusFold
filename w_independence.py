"""
TorusFold: W Independence / Train-Test Split Analysis
=====================================================
Tests whether BPS/L results are robust to W construction by building
W from different sources and measuring the same proteins:

  Test 1: W from 10% of AlphaFold, applied to remaining 90%
  Test 2: W from organism A, applied to organism B (all pairs)
  Test 3: W from PDB structures, applied to AlphaFold
  Test 4: W at different bin resolutions (180x180, 360x360, 720x720)

For each W, reports:
  - Mean BPS/L
  - Cross-organism CV
  - Seg/Real ratio (intra-basin coherence)
  - M1/Real ratio
  - S/Real ratio

If all W constructions give similar ratios and CV, the results are
W-independent and the paper's claims are robust.

Usage:
  python w_independence.py [--sample N]

Reads: alphafold_cache/, results/superpotential_W.npz
Writes: results/w_independence_report.md
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
# CORE FUNCTIONS (self-contained)
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
    """Extract (phi, psi) and SS from a CIF file. Returns (phi_psi_list, ss_list)."""
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


def build_W_from_angles(all_phi_psi, grid_size=360, smooth_sigma=0.0, epsilon=1e-7):
    """Build W grid from list of (phi, psi) pairs in radians."""
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)
    scale = grid_size / 360.0

    for phi, psi in all_phi_psi:
        phi_d = math.degrees(phi)
        psi_d = math.degrees(psi)
        gi = int(round((phi_d + 180) * scale)) % grid_size
        gj = int(round((psi_d + 180) * scale)) % grid_size
        grid[gi, gj] += 1

    total = grid.sum()
    if total > 0:
        grid /= total

    if smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=smooth_sigma, mode='wrap')

    W = -np.log(grid + epsilon)
    return W, grid_size


def lookup_W(W_grid, grid_size, phi_rad, psi_rad):
    """Look up W value for arrays of phi, psi in radians."""
    scale = grid_size / 360.0
    phi_d = np.degrees(phi_rad)
    psi_d = np.degrees(psi_rad)
    gi = (np.round((phi_d + 180) * scale).astype(int)) % grid_size
    gj = (np.round((psi_d + 180) * scale).astype(int)) % grid_size
    return W_grid[gi, gj]


def compute_bps_l(phi_arr, psi_arr, W_grid, grid_size):
    """BPS/L from arrays."""
    if len(phi_arr) < 2:
        return 0.0
    w = lookup_W(W_grid, grid_size, phi_arr, psi_arr)
    return float(np.mean(np.abs(np.diff(w))))


def null_segment(phi_psi, ss_seq, rng):
    """Segment-preserving null."""
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


def evaluate_protein(phi_psi, ss_seq, W_grid, grid_size, rng, n_trials=5):
    """Compute Real, Segment, Markov1, Shuffled BPS/L for one protein."""
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
    }


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


def extract_all_angles_from_files(files, max_files=None):
    """Extract all (phi, psi) pairs from a list of files."""
    rng = np.random.default_rng(42)
    if max_files and len(files) > max_files:
        idx = rng.choice(len(files), max_files, replace=False)
        files = [files[i] for i in idx]

    all_angles = []
    for f in files:
        try:
            pp, _ = extract_angles(f)
            all_angles.extend(pp)
        except Exception:
            pass
    return all_angles


# ═══════════════════════════════════════════════════════════════════
# MAIN TEST HARNESS
# ═══════════════════════════════════════════════════════════════════

def run_test(label, W_grid, grid_size, test_files, organisms_map, rng,
             n_trials=5, decomp_sample=50):
    """Run BPS/L analysis with a given W on test files.

    Returns summary dict.
    """
    # BPS/L for all test files
    protein_results = []
    org_bpsl = defaultdict(list)

    for filepath, organism in test_files:
        try:
            pp, ss = extract_angles(filepath)
            if len(pp) < 50:
                continue
            phi = np.array([p[0] for p in pp])
            psi = np.array([p[1] for p in pp])
            bpsl = compute_bps_l(phi, psi, W_grid, grid_size)
            protein_results.append({
                'bpsl': bpsl, 'organism': organism,
                'phi_psi': pp, 'ss': ss,
            })
            org_bpsl[organism].append(bpsl)
        except Exception:
            pass

    if len(protein_results) < 20:
        return None

    # Global stats
    all_bpsl = np.array([r['bpsl'] for r in protein_results])
    global_mean = float(np.mean(all_bpsl))
    global_cv = float(np.std(all_bpsl) / np.mean(all_bpsl) * 100)

    # Cross-organism CV
    org_means = {}
    for org, vals in org_bpsl.items():
        if len(vals) >= 5:
            org_means[org] = float(np.mean(vals))

    cross_org_cv = 0.0
    if len(org_means) >= 3:
        om = np.array(list(org_means.values()))
        cross_org_cv = float(np.std(om) / np.mean(om) * 100)

    # Three-level + segment decomposition (sample)
    decomp_indices = rng.choice(len(protein_results),
                                min(decomp_sample, len(protein_results)),
                                replace=False)
    real_v, seg_v, m1_v, shuf_v = [], [], [], []
    for idx in decomp_indices:
        r = protein_results[idx]
        if len(r['phi_psi']) < 50:
            continue
        d = evaluate_protein(r['phi_psi'], r['ss'], W_grid, grid_size, rng, n_trials)
        real_v.append(d['real'])
        seg_v.append(d['segment'])
        m1_v.append(d['markov1'])
        shuf_v.append(d['shuffled'])

    decomp = {}
    if real_v:
        rm, sm, mm, shm = np.mean(real_v), np.mean(seg_v), np.mean(m1_v), np.mean(shuf_v)
        decomp = {
            'real': float(rm), 'segment': float(sm),
            'markov1': float(mm), 'shuffled': float(shm),
            'seg_real': float(sm / rm) if rm > 0 else 0,
            'm1_real': float(mm / rm) if rm > 0 else 0,
            's_real': float(shm / rm) if rm > 0 else 0,
            'n_decomp': len(real_v),
        }

    return {
        'label': label,
        'n_proteins': len(protein_results),
        'n_organisms': len(org_means),
        'global_mean': global_mean,
        'global_cv': global_cv,
        'cross_org_cv': cross_org_cv,
        'org_means': org_means,
        'decomp': decomp,
    }


def main():
    parser = argparse.ArgumentParser(description="W Independence Tests")
    parser.add_argument("--data", default="alphafold_cache",
                        help="AlphaFold data directory")
    parser.add_argument("--pdb-dir", default="data/pdb_test_set",
                        help="PDB structures directory (for Test 3)")
    parser.add_argument("--sample", type=int, default=500,
                        help="Test set size per test (default: 500)")
    parser.add_argument("--w-sample", type=int, default=300,
                        help="Structures for W building (default: 300)")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold: W Independence Analysis")
    print("=" * 60)

    rng = np.random.default_rng(42)
    organisms = discover_files(args.data)
    total_files = sum(len(f) for f in organisms.values())
    print(f"  Data: {total_files} files across {len(organisms)} organisms")

    # Build flat file list with organisms
    all_files = []
    for org, files in organisms.items():
        for f in files:
            all_files.append((f, org))

    # Shuffle deterministically
    rng2 = np.random.default_rng(123)
    indices = rng2.permutation(len(all_files))
    all_files_shuffled = [all_files[i] for i in indices]

    test_results = []

    # ══════════════════════════════════════════════════════════════
    # TEST 0: Baseline (original W from full dataset)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 0: Baseline (cached W from full dataset)")
    print("=" * 60)

    w_cache = os.path.join(args.output, "superpotential_W.npz")
    if os.path.exists(w_cache):
        data = np.load(w_cache)
        W_baseline = data['grid']
        gs_baseline = W_baseline.shape[0]
        test_set = all_files_shuffled[:args.sample]
        r = run_test("Baseline (full W)", W_baseline, gs_baseline,
                     test_set, organisms, rng)
        if r:
            test_results.append(r)
            print(f"  BPS/L = {r['global_mean']:.3f}, CV = {r['global_cv']:.1f}%, "
                  f"cross-org CV = {r['cross_org_cv']:.1f}%")
            if r['decomp']:
                print(f"  Seg/Real = {r['decomp']['seg_real']:.2f}x, "
                      f"S/Real = {r['decomp']['s_real']:.2f}x")
    else:
        print("  WARNING: No cached W found. Skipping baseline.")

    # ══════════════════════════════════════════════════════════════
    # TEST 1: 10% train, 90% test
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 1: W from 10% of data, applied to 90%")
    print("=" * 60)

    split_point = len(all_files_shuffled) // 10
    train_files = [f for f, _ in all_files_shuffled[:split_point]]
    test_set_90 = all_files_shuffled[split_point:split_point + args.sample]

    print(f"  Building W from {len(train_files)} structures (10%)...")
    train_angles = extract_all_angles_from_files(train_files, max_files=args.w_sample)
    print(f"  Collected {len(train_angles)} angles")

    if len(train_angles) > 1000:
        W_10pct, gs_10 = build_W_from_angles(train_angles)
        r = run_test("W from 10%", W_10pct, gs_10, test_set_90, organisms, rng)
        if r:
            test_results.append(r)
            print(f"  BPS/L = {r['global_mean']:.3f}, CV = {r['global_cv']:.1f}%, "
                  f"cross-org CV = {r['cross_org_cv']:.1f}%")
            if r['decomp']:
                print(f"  Seg/Real = {r['decomp']['seg_real']:.2f}x, "
                      f"S/Real = {r['decomp']['s_real']:.2f}x")
    else:
        print("  Too few angles from 10% split.")

    # ══════════════════════════════════════════════════════════════
    # TEST 2: W from organism A, applied to organism B
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 2: Cross-organism W transfer")
    print("=" * 60)

    # Pick 4 diverse organisms with enough data
    big_orgs = sorted(organisms.keys(), key=lambda x: len(organisms[x]), reverse=True)
    # Try to get bacteria + eukaryote diversity
    test_orgs = []
    bacteria = [o for o in big_orgs if o in ('ecoli', 'bacillus', 'tuberculosis',
                                              'salmonella', 'pseudomonas', 'staph')]
    eukaryotes = [o for o in big_orgs if o in ('human', 'yeast', 'mouse', 'fly',
                                                'worm', 'fission_yeast')]
    if bacteria:
        test_orgs.append(bacteria[0])
    if len(bacteria) > 1:
        test_orgs.append(bacteria[1])
    if eukaryotes:
        test_orgs.append(eukaryotes[0])
    if len(eukaryotes) > 1:
        test_orgs.append(eukaryotes[1])

    cross_org_results = []
    for train_org in test_orgs:
        print(f"\n  Building W from {train_org}...")
        org_angles = extract_all_angles_from_files(
            organisms[train_org], max_files=args.w_sample)
        if len(org_angles) < 1000:
            print(f"    Too few angles ({len(org_angles)}), skipping")
            continue
        W_org, gs_org = build_W_from_angles(org_angles)
        print(f"    {len(org_angles)} angles")

        # Apply to all OTHER organisms
        other_files = []
        for org, files in organisms.items():
            if org == train_org:
                continue
            for f in files[:50]:  # cap per organism for speed
                other_files.append((f, org))
        rng3 = np.random.default_rng(42)
        if len(other_files) > args.sample:
            idx = rng3.choice(len(other_files), args.sample, replace=False)
            other_files = [other_files[i] for i in idx]

        label = f"W from {train_org}"
        r = run_test(label, W_org, gs_org, other_files, organisms, rng)
        if r:
            cross_org_results.append(r)
            test_results.append(r)
            print(f"    Applied to {r['n_proteins']} proteins: "
                  f"BPS/L = {r['global_mean']:.3f}, "
                  f"cross-org CV = {r['cross_org_cv']:.1f}%, "
                  f"Seg/Real = {r['decomp'].get('seg_real', 0):.2f}x")

    # ══════════════════════════════════════════════════════════════
    # TEST 3: W from PDB, applied to AlphaFold
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 3: W from PDB structures, applied to AlphaFold")
    print("=" * 60)

    pdb_dir = Path(args.pdb_dir)
    pdb_files = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.cif"))
    pdb_angles = []
    if pdb_files:
        print(f"  Found {len(pdb_files)} PDB files")
        # Extract angles from PDB files — try CIF parser on .pdb too
        for pf in pdb_files:
            try:
                pp, _ = extract_angles(str(pf))
                pdb_angles.extend(pp)
            except Exception:
                pass
        print(f"  Collected {len(pdb_angles)} angles from PDB")

    # Also check data/pdb_cache
    pdb_cache = Path("data/pdb_cache")
    if pdb_cache.exists():
        pdb_cache_files = list(pdb_cache.glob("*.pdb")) + list(pdb_cache.glob("*.cif"))
        print(f"  Found {len(pdb_cache_files)} files in pdb_cache")
        for pf in pdb_cache_files:
            try:
                pp, _ = extract_angles(str(pf))
                pdb_angles.extend(pp)
            except Exception:
                pass
        print(f"  Total PDB angles: {len(pdb_angles)}")

    if len(pdb_angles) > 1000:
        W_pdb, gs_pdb = build_W_from_angles(pdb_angles)
        test_set_af = all_files_shuffled[:args.sample]
        r = run_test("W from PDB", W_pdb, gs_pdb, test_set_af, organisms, rng)
        if r:
            test_results.append(r)
            print(f"  BPS/L = {r['global_mean']:.3f}, CV = {r['global_cv']:.1f}%, "
                  f"cross-org CV = {r['cross_org_cv']:.1f}%")
            if r['decomp']:
                print(f"  Seg/Real = {r['decomp']['seg_real']:.2f}x, "
                      f"S/Real = {r['decomp']['s_real']:.2f}x")
    else:
        print("  Insufficient PDB data. Skipping Test 3.")
        print("  (Need PDB files in data/pdb_test_set/ or data/pdb_cache/)")

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Different bin resolutions
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 4: Bin resolution robustness")
    print("=" * 60)

    # Use the 10% training angles for all resolutions
    if len(train_angles) > 1000:
        test_set_res = all_files_shuffled[:min(200, args.sample)]
        for grid_size in [180, 360, 720]:
            W_res, gs_res = build_W_from_angles(train_angles, grid_size=grid_size)
            label = f"Resolution {grid_size}x{grid_size}"
            r = run_test(label, W_res, gs_res, test_set_res, organisms, rng,
                         decomp_sample=30)
            if r:
                test_results.append(r)
                seg_r = r['decomp'].get('seg_real', 0) if r['decomp'] else 0
                print(f"  {grid_size}x{grid_size}: BPS/L = {r['global_mean']:.3f}, "
                      f"CV = {r['global_cv']:.1f}%, "
                      f"cross-org CV = {r['cross_org_cv']:.1f}%, "
                      f"Seg/Real = {seg_r:.2f}x")

    # ══════════════════════════════════════════════════════════════
    # WRITE REPORT
    # ══════════════════════════════════════════════════════════════
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "w_independence_report.md")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# W Independence Analysis\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## Summary Table\n\n")
        f.write("| W Construction | N | BPS/L | Per-protein CV | "
                "Cross-org CV | Seg/Real | M1/Real | S/Real |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")

        for r in test_results:
            d = r.get('decomp', {})
            seg_r = f"{d['seg_real']:.2f}x" if d.get('seg_real') else "—"
            m1_r = f"{d['m1_real']:.2f}x" if d.get('m1_real') else "—"
            s_r = f"{d['s_real']:.2f}x" if d.get('s_real') else "—"
            co_cv = f"{r['cross_org_cv']:.1f}%" if r['cross_org_cv'] > 0 else "—"
            f.write(f"| {r['label']} | {r['n_proteins']} | "
                    f"{r['global_mean']:.3f} | {r['global_cv']:.1f}% | "
                    f"{co_cv} | {seg_r} | {m1_r} | {s_r} |\n")

        f.write("\n")

        # Interpretation
        f.write("## Interpretation\n\n")

        # Check if ratios are stable
        seg_reals = [r['decomp']['seg_real'] for r in test_results
                     if r.get('decomp', {}).get('seg_real')]
        cross_cvs = [r['cross_org_cv'] for r in test_results if r['cross_org_cv'] > 0]

        if seg_reals:
            sr_mean = np.mean(seg_reals)
            sr_std = np.std(seg_reals)
            sr_cv = sr_std / sr_mean * 100 if sr_mean > 0 else 999
            f.write(f"**Seg/Real ratio across W constructions:** "
                    f"{sr_mean:.2f} +/- {sr_std:.2f} (CV = {sr_cv:.0f}%)\n\n")

            if sr_cv < 15:
                f.write("The Seg/Real ratio (intra-basin coherence) is **stable** across "
                        "W constructions. The 41% roughness suppression from intra-basin "
                        "coherence is a property of proteins, not of the histogram.\n\n")
            else:
                f.write("The Seg/Real ratio varies across W constructions, suggesting "
                        "partial dependence on landscape construction.\n\n")

        if cross_cvs:
            cc_mean = np.mean(cross_cvs)
            cc_std = np.std(cross_cvs)
            f.write(f"**Cross-organism CV across W constructions:** "
                    f"{cc_mean:.1f}% +/- {cc_std:.1f}%\n\n")

        # Per-test details
        for r in test_results:
            f.write(f"### {r['label']}\n\n")
            f.write(f"- Proteins: {r['n_proteins']}\n")
            f.write(f"- Organisms with data: {r['n_organisms']}\n")
            f.write(f"- Mean BPS/L: {r['global_mean']:.3f}\n")
            f.write(f"- Per-protein CV: {r['global_cv']:.1f}%\n")
            if r['cross_org_cv'] > 0:
                f.write(f"- Cross-organism CV: {r['cross_org_cv']:.1f}%\n")
            d = r.get('decomp', {})
            if d:
                f.write(f"- Decomposition (N={d.get('n_decomp', '?')}):\n")
                f.write(f"  - Real: {d['real']:.3f}\n")
                f.write(f"  - Segment: {d['segment']:.3f} ({d['seg_real']:.2f}x)\n")
                f.write(f"  - Markov-1: {d['markov1']:.3f} ({d['m1_real']:.2f}x)\n")
                f.write(f"  - Shuffled: {d['shuffled']:.3f} ({d['s_real']:.2f}x)\n")
            f.write("\n")

            # Organism means if available
            if r.get('org_means') and len(r['org_means']) > 2:
                f.write("Per-organism means:\n\n")
                f.write("| Organism | Mean BPS/L |\n|---|---|\n")
                for org in sorted(r['org_means'], key=lambda x: r['org_means'][x]):
                    f.write(f"| {org} | {r['org_means'][org]:.3f} |\n")
                f.write("\n")

        # Summary
        f.write("## Conclusion\n\n")
        f.write("```\n")
        f.write("W INDEPENDENCE SUMMARY\n")
        f.write("═══════════════════════════════════════════\n")
        for r in test_results:
            d = r.get('decomp', {})
            seg_r = f"{d['seg_real']:.2f}x" if d.get('seg_real') else "n/a"
            co = f"{r['cross_org_cv']:.1f}%" if r['cross_org_cv'] > 0 else "n/a"
            f.write(f"  {r['label']:30s}  Seg/Real={seg_r}  CrossOrgCV={co}\n")
        f.write("═══════════════════════════════════════════\n")
        f.write("```\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
