"""
TorusFold: Pre-Submission Hardening Tests
==========================================
Three cheap experiments that close the remaining attack vectors:

Test A — Local-window null (Limitation 6 defense)
  Instead of sampling from the entire basin pool, sample from ±k
  positions within the same segment. If Seg/Real drops but stays >1
  with CI excluding unity, the positional-gradient critique is dead.

Test B — PDB torus Seg/Real (Limitation 8 defense)
  Compute torus L1 Seg/Real on PDB structures directly.
  If ~1.08–1.12×, AlphaFold compression is not driving the signal.

Test C — Steric rejection diagnostics (Polymer null defense)
  Re-run a small steric-coupled sample and track:
  - Rejection rate per residue position
  - Accepted φ/ψ distribution vs original P(φ,ψ)
  - Late-chain bias check

Usage:
  python hardening_tests.py --data alphafold_cache --output results
         [--pdb-dir pdb_structures]

  --data:    AlphaFold cache (for Test A)
  --pdb-dir: Directory of PDB .cif files (for Test B, optional)
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
# SHARED UTILITIES (from main pipeline)
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


def classify_ss(phi_rad, psi_rad):
    ss = []
    for phi, psi in zip(phi_rad, psi_rad):
        phi_d = math.degrees(phi)
        psi_d = math.degrees(psi)
        if -160 < phi_d < 0 and -120 < psi_d < 30:
            ss.append('a')
        elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
            ss.append('b')
        else:
            ss.append('o')
    return ss


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
            ss = 'a'
        elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
            ss = 'b'
        else:
            ss = 'o'
        phi_psi.append((phi, psi))
        ss_seq.append(ss)
    return phi_psi, ss_seq


def null_segment(phi, psi, ss_seq, rng):
    """Standard segment-preserving null."""
    pools = defaultdict(list)
    for i, s in enumerate(ss_seq):
        pools[s].append(i)
    new_phi = np.empty_like(phi)
    new_psi = np.empty_like(psi)
    for i, s in enumerate(ss_seq):
        j = rng.choice(pools[s])
        new_phi[i], new_psi[i] = phi[j], psi[j]
    return new_phi, new_psi


def null_shuffled(phi, psi, rng):
    perm = rng.permutation(len(phi))
    return phi[perm], psi[perm]


# ═══════════════════════════════════════════════════════════════════
# TEST A: LOCAL-WINDOW NULL
# ═══════════════════════════════════════════════════════════════════

def null_local_window(phi, psi, ss_seq, rng, k=5):
    """
    Local-window segment null: for each position i, sample from
    positions within ±k that share the same SS label. Falls back
    to full basin pool if the local window has no same-SS neighbors.
    """
    n = len(phi)
    new_phi = np.empty_like(phi)
    new_psi = np.empty_like(psi)

    for i in range(n):
        # Find same-SS positions within ±k
        candidates = []
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        for j in range(lo, hi):
            if ss_seq[j] == ss_seq[i]:
                candidates.append(j)

        if len(candidates) < 2:
            # Fallback: full basin pool
            candidates = [j for j in range(n) if ss_seq[j] == ss_seq[i]]

        j = rng.choice(candidates)
        new_phi[i], new_psi[i] = phi[j], psi[j]

    return new_phi, new_psi


def run_test_a(data_path, output_dir, n_sample=200, n_trials=10):
    """Test A: Compare standard vs local-window segment null."""
    print("\n" + "=" * 60)
    print("  TEST A: Local-Window Null")
    print("  Does positional averaging within segments inflate Seg/Real?")
    print("=" * 60)

    from pathlib import Path
    all_files = []
    for subdir in sorted(Path(data_path).iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue
        for f in sorted(subdir.glob("*.cif")):
            all_files.append(str(f))

    rng = np.random.default_rng(42)
    indices = rng.choice(len(all_files), min(n_sample, len(all_files)),
                         replace=False)

    real_vals = []
    seg_standard_vals = []
    seg_local3_vals = []
    seg_local5_vals = []
    seg_local10_vals = []
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

            real_r = roughness_torus_l1(phi, psi)
            real_vals.append(real_r)

            for _ in range(n_trials):
                sp, sq = null_segment(phi, psi, ss, rng)
                seg_standard_vals.append(roughness_torus_l1(sp, sq))

                sp3, sq3 = null_local_window(phi, psi, ss, rng, k=3)
                seg_local3_vals.append(roughness_torus_l1(sp3, sq3))

                sp5, sq5 = null_local_window(phi, psi, ss, rng, k=5)
                seg_local5_vals.append(roughness_torus_l1(sp5, sq5))

                sp10, sq10 = null_local_window(phi, psi, ss, rng, k=10)
                seg_local10_vals.append(roughness_torus_l1(sp10, sq10))

        except Exception:
            pass

        if (fi + 1) % 50 == 0:
            print(f"\r    [{fi+1}/{len(indices)}] {n_loaded} loaded...",
                  end="", flush=True)

    print(f"\r    {n_loaded} proteins loaded                    ")

    rm = np.mean(real_vals)
    results = {
        'Standard (full pool)': np.mean(seg_standard_vals) / rm,
        'Local k=3': np.mean(seg_local3_vals) / rm,
        'Local k=5': np.mean(seg_local5_vals) / rm,
        'Local k=10': np.mean(seg_local10_vals) / rm,
    }

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

    real_arr = np.array(real_vals)
    ci_standard = boot_ci(seg_standard_vals, real_arr)
    ci_local3 = boot_ci(seg_local3_vals, real_arr)
    ci_local5 = boot_ci(seg_local5_vals, real_arr)
    ci_local10 = boot_ci(seg_local10_vals, real_arr)

    print("\n  Results:")
    print(f"    Standard (full pool):  Seg/Real = "
          f"{results['Standard (full pool)']:.4f}x  "
          f"[{ci_standard[0]:.3f}, {ci_standard[1]:.3f}]")
    print(f"    Local k=3:             Seg/Real = "
          f"{results['Local k=3']:.4f}x  "
          f"[{ci_local3[0]:.3f}, {ci_local3[1]:.3f}]")
    print(f"    Local k=5:             Seg/Real = "
          f"{results['Local k=5']:.4f}x  "
          f"[{ci_local5[0]:.3f}, {ci_local5[1]:.3f}]")
    print(f"    Local k=10:            Seg/Real = "
          f"{results['Local k=10']:.4f}x  "
          f"[{ci_local10[0]:.3f}, {ci_local10[1]:.3f}]")

    drop = (results['Standard (full pool)'] - results['Local k=5'])
    print(f"\n    Drop from standard to k=5: {drop:.4f}")
    if drop < 0.01:
        print("    VERDICT: Positional gradient effect is negligible.")
    elif drop < 0.03:
        print("    VERDICT: Small positional effect exists but "
              "does not explain the bulk of the signal.")
    else:
        print("    WARNING: Substantial positional effect. "
              "Investigate helix gradients.")

    return {
        'n_proteins': n_loaded,
        'real_mean': float(rm),
        'results': results,
        'ci_standard': ci_standard,
        'ci_local3': ci_local3,
        'ci_local5': ci_local5,
        'ci_local10': ci_local10,
    }


# ═══════════════════════════════════════════════════════════════════
# TEST B: PDB TORUS SEG/REAL
# ═══════════════════════════════════════════════════════════════════

def run_test_b(pdb_dir, output_dir, n_trials=10):
    """Test B: Compute torus L1 Seg/Real on PDB structures."""
    print("\n" + "=" * 60)
    print("  TEST B: PDB Torus Seg/Real")
    print("  Is the ~1.11x effect present in experimental structures?")
    print("=" * 60)

    from pathlib import Path
    pdb_path = Path(pdb_dir)
    pdb_files = sorted(list(pdb_path.glob("*.cif")) +
                       list(pdb_path.glob("*.pdb")))

    if not pdb_files:
        print(f"  No structure files found in {pdb_dir}")
        return None

    rng = np.random.default_rng(42)
    real_vals = []
    seg_vals = []
    shuf_vals = []
    n_loaded = 0

    for fi, fpath in enumerate(pdb_files):
        try:
            fsize = os.path.getsize(str(fpath))
            if fsize > 10 * 1024 * 1024:
                continue
            # PDB structures: no pLDDT filter (set to 0)
            pp, ss = extract_angles(str(fpath), plddt_min=0.0)
            if len(pp) < 30:
                continue
            phi = np.array([p[0] for p in pp])
            psi = np.array([p[1] for p in pp])
            n_loaded += 1

            real_r = roughness_torus_l1(phi, psi)
            real_vals.append(real_r)

            for _ in range(n_trials):
                sp, sq = null_segment(phi, psi, ss, rng)
                seg_vals.append(roughness_torus_l1(sp, sq))

                sfp, sfq = null_shuffled(phi, psi, rng)
                shuf_vals.append(roughness_torus_l1(sfp, sfq))

        except Exception as e:
            pass

        if (fi + 1) % 50 == 0:
            print(f"\r    [{fi+1}/{len(pdb_files)}] {n_loaded} loaded...",
                  end="", flush=True)

    print(f"\r    {n_loaded} PDB structures loaded                ")

    if not real_vals:
        print("  No valid PDB structures found.")
        return None

    rm = np.mean(real_vals)
    seg_real = np.mean(seg_vals) / rm
    s_real = np.mean(shuf_vals) / rm

    # Bootstrap
    brng = np.random.default_rng(99)
    real_arr = np.array(real_vals)
    seg_arr = np.array(seg_vals)
    ratios = []
    for _ in range(2000):
        ni = brng.choice(len(seg_arr), len(seg_arr), replace=True)
        ri = brng.choice(len(real_arr), len(real_arr), replace=True)
        rm_ = np.mean(real_arr[ri])
        if rm_ > 0:
            ratios.append(np.mean(seg_arr[ni]) / rm_)
    ci = (np.percentile(ratios, 2.5), np.percentile(ratios, 97.5))

    print(f"\n  Results (N = {n_loaded}):")
    print(f"    Torus L1 Seg/Real = {seg_real:.3f}x  "
          f"[{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"    Torus L1 S/Real   = {s_real:.3f}x")
    print(f"    Mean real roughness: {rm:.4f}")

    if ci[0] > 1.0:
        print("    VERDICT: CI excludes 1.0. Geometric effect present "
              "in experimental structures.")
    else:
        print("    VERDICT: CI includes 1.0. Effect may be attenuated "
              "or absent in PDB data.")

    return {
        'n_structures': n_loaded,
        'seg_real': seg_real,
        's_real': s_real,
        'ci': ci,
        'real_mean': float(rm),
    }


# ═══════════════════════════════════════════════════════════════════
# TEST C: STERIC REJECTION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════

# Import backbone builder from polymer_null_experiment
BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
ANGLE_C_N_CA = math.radians(121.7)
ANGLE_N_CA_C = math.radians(111.2)
ANGLE_CA_C_N = math.radians(116.2)
CLASH_RADIUS_SQ = 2.0 ** 2
OMEGA = math.pi


def place_atom(a, b, c, bond_length, bond_angle, torsion):
    bc = c - b
    bc_norm = np.linalg.norm(bc)
    if bc_norm < 1e-10:
        return c + np.array([bond_length, 0, 0])
    bc_hat = bc / bc_norm
    ab = b - a
    n = np.cross(ab, bc)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        n = np.array([1, 0, 0]) if abs(bc_hat[0]) < 0.9 else np.array([0, 1, 0])
        n = np.cross(n, bc_hat)
        n_norm = np.linalg.norm(n)
    n_hat = n / n_norm
    m = np.cross(n_hat, bc_hat)
    d_local = np.array([
        -bond_length * math.cos(bond_angle),
        bond_length * math.sin(bond_angle) * math.cos(torsion),
        bond_length * math.sin(bond_angle) * math.sin(torsion),
    ])
    rotation = np.column_stack([bc_hat, m, n_hat])
    return c + rotation @ d_local


def check_steric_clash(coords, new_atoms, lookback=6):
    start = max(0, len(coords) - lookback)
    for ri in range(start, len(coords)):
        for old_atom in coords[ri]:
            for new_atom in new_atoms:
                diff = new_atom - old_atom
                if np.dot(diff, diff) < CLASH_RADIUS_SQ:
                    return True
    return False


def build_ramachandran_sampler(histogram=None, grid_size=360):
    if histogram is not None:
        P = histogram / histogram.sum()
        flat = P.flatten()
        cumsum = np.cumsum(flat)

        def sample(rng, n=1):
            u = rng.random(n)
            idx = np.searchsorted(cumsum, u)
            idx = np.clip(idx, 0, grid_size * grid_size - 1)
            gi = idx // grid_size
            gj = idx % grid_size
            phi_d = (gi + rng.random(n)) * (360.0 / grid_size) - 180.0
            psi_d = (gj + rng.random(n)) * (360.0 / grid_size) - 180.0
            return np.radians(phi_d), np.radians(psi_d)
        return sample
    else:
        def sample(rng, n=1):
            phi = np.empty(n)
            psi = np.empty(n)
            for k in range(n):
                r = rng.random()
                if r < 0.60:
                    phi[k] = math.radians(rng.normal(-63, 12))
                    psi[k] = math.radians(rng.normal(-43, 12))
                elif r < 0.85:
                    phi[k] = math.radians(rng.normal(-120, 15))
                    psi[k] = math.radians(rng.normal(130, 15))
                else:
                    phi[k] = math.radians(rng.uniform(-180, 180))
                    psi[k] = math.radians(rng.uniform(-180, 180))
            return phi, psi
        return sample


def run_test_c(output_dir, histogram=None, n_chains=100, chain_length=200):
    """Test C: Steric rejection diagnostics."""
    print("\n" + "=" * 60)
    print("  TEST C: Steric Rejection Diagnostics")
    print("  Does clash rejection distort P(phi,psi)?")
    print("=" * 60)

    rng = np.random.default_rng(42)
    sampler = build_ramachandran_sampler(histogram)

    # Track rejection stats
    rejections_by_position = np.zeros(chain_length)
    attempts_by_position = np.zeros(chain_length)
    accepted_phi = []
    accepted_psi = []
    proposed_phi = []
    proposed_psi = []
    max_retries = 50

    for ci in range(n_chains):
        phi_list = []
        psi_list = []
        coords = []

        # First residue
        phi_0, psi_0 = sampler(rng, 1)
        phi_list.append(float(phi_0[0]))
        psi_list.append(float(psi_0[0]))
        accepted_phi.append(float(phi_0[0]))
        accepted_psi.append(float(psi_0[0]))

        N0 = np.array([0.0, 0.0, 0.0])
        CA0 = np.array([BOND_N_CA, 0.0, 0.0])
        C0 = place_atom(np.array([-1.0, 0.0, 0.0]), N0, CA0,
                        BOND_CA_C, ANGLE_N_CA_C, phi_list[0])
        coords.append((N0, CA0, C0))

        for i in range(1, chain_length):
            n_attempts = 0
            accepted = False
            for attempt in range(max_retries):
                phi_try, psi_try = sampler(rng, 1)
                phi_try = float(phi_try[0])
                psi_try = float(psi_try[0])
                proposed_phi.append(phi_try)
                proposed_psi.append(psi_try)
                n_attempts += 1

                prev_N, prev_CA, prev_C = coords[-1]
                psi_prev = psi_list[-1]
                N_i = place_atom(prev_N, prev_CA, prev_C,
                                 BOND_C_N, ANGLE_CA_C_N, psi_prev)
                CA_i = place_atom(prev_CA, prev_C, N_i,
                                  BOND_N_CA, ANGLE_C_N_CA, OMEGA)
                C_i = place_atom(prev_C, N_i, CA_i,
                                 BOND_CA_C, ANGLE_N_CA_C, phi_try)

                if not check_steric_clash(coords, (N_i, CA_i, C_i), lookback=6):
                    phi_list.append(phi_try)
                    psi_list.append(psi_try)
                    coords.append((N_i, CA_i, C_i))
                    accepted_phi.append(phi_try)
                    accepted_psi.append(psi_try)
                    accepted = True
                    break

            rejections_by_position[i] += (n_attempts - 1)
            attempts_by_position[i] += n_attempts

            if not accepted:
                # Force accept
                phi_try, psi_try = sampler(rng, 1)
                phi_list.append(float(phi_try[0]))
                psi_list.append(float(psi_try[0]))
                prev_N, prev_CA, prev_C = coords[-1]
                psi_prev = psi_list[-2]
                N_i = place_atom(prev_N, prev_CA, prev_C,
                                 BOND_C_N, ANGLE_CA_C_N, psi_prev)
                CA_i = place_atom(prev_CA, prev_C, N_i,
                                  BOND_N_CA, ANGLE_C_N_CA, OMEGA)
                C_i = place_atom(prev_C, N_i, CA_i,
                                 BOND_CA_C, ANGLE_N_CA_C, phi_list[-1])
                coords.append((N_i, CA_i, C_i))
                accepted_phi.append(phi_list[-1])
                accepted_psi.append(psi_list[-1])

        if (ci + 1) % 20 == 0:
            print(f"\r    [{ci+1}/{n_chains}]...", end="", flush=True)

    print(f"\r    {n_chains} chains complete            ")

    # Compute rejection statistics
    mean_rej = rejections_by_position / n_chains
    mean_att = attempts_by_position / n_chains
    rej_rate = np.where(mean_att > 0, mean_rej / mean_att, 0)

    overall_rej_rate = np.sum(rejections_by_position) / np.sum(attempts_by_position)
    late_rej_rate = np.mean(rej_rate[150:])  # last 50 positions
    early_rej_rate = np.mean(rej_rate[1:50])  # first 50 (skip 0)

    print(f"\n  Rejection Statistics:")
    print(f"    Overall rejection rate: {overall_rej_rate:.4f} "
          f"({overall_rej_rate*100:.2f}%)")
    print(f"    Early chain (pos 1-49): {early_rej_rate:.4f} "
          f"({early_rej_rate*100:.2f}%)")
    print(f"    Late chain (pos 150+):  {late_rej_rate:.4f} "
          f"({late_rej_rate*100:.2f}%)")
    print(f"    Max rejection rate at any position: "
          f"{np.max(rej_rate):.4f}")

    # Compare distributions
    # Bin into 36x36 grid (10-degree bins)
    grid_size = 36
    scale = grid_size / (2 * math.pi)

    def bin_angles(phi_list, psi_list, grid_size):
        hist = np.zeros((grid_size, grid_size))
        for phi, psi in zip(phi_list, psi_list):
            gi = int((phi + math.pi) * scale) % grid_size
            gj = int((psi + math.pi) * scale) % grid_size
            hist[gi, gj] += 1
        return hist / hist.sum() if hist.sum() > 0 else hist

    proposed_hist = bin_angles(proposed_phi, proposed_psi, grid_size)
    accepted_hist = bin_angles(accepted_phi, accepted_psi, grid_size)

    # KL divergence (accepted vs proposed)
    mask = (proposed_hist > 0) & (accepted_hist > 0)
    kl_div = np.sum(accepted_hist[mask] *
                    np.log(accepted_hist[mask] / proposed_hist[mask]))

    # L1 distance
    l1_dist = np.sum(np.abs(accepted_hist - proposed_hist))

    # Correlation
    corr = np.corrcoef(proposed_hist.flatten(),
                       accepted_hist.flatten())[0, 1]

    print(f"\n  Distribution Comparison (accepted vs proposed):")
    print(f"    Correlation:     {corr:.6f}")
    print(f"    L1 distance:     {l1_dist:.6f}")
    print(f"    KL divergence:   {kl_div:.6f}")

    if overall_rej_rate < 0.05:
        print("\n    VERDICT: Rejection rate < 5%. Steric filter "
              "has negligible effect on distribution.")
    elif overall_rej_rate < 0.15:
        print("\n    VERDICT: Moderate rejection rate. Check "
              "distribution correlation carefully.")
    else:
        print("\n    WARNING: High rejection rate. Distribution "
              "may be meaningfully distorted.")

    return {
        'overall_rejection_rate': float(overall_rej_rate),
        'early_rejection_rate': float(early_rej_rate),
        'late_rejection_rate': float(late_rej_rate),
        'distribution_correlation': float(corr),
        'l1_distance': float(l1_dist),
        'kl_divergence': float(kl_div),
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pre-submission hardening tests")
    parser.add_argument("--data", default=None,
                        help="AlphaFold data directory (for Test A)")
    parser.add_argument("--pdb-dir", default=None,
                        help="PDB structures directory (for Test B)")
    parser.add_argument("--output", default="results")
    parser.add_argument("--w-path", default=None,
                        help="Path to superpotential_W.npz")
    parser.add_argument("--sample", type=int, default=200,
                        help="Proteins for Test A (default: 200)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load histogram for Test C
    histogram = None
    search_paths = [args.w_path] if args.w_path else []
    search_paths += [
        os.path.join(args.output, "superpotential_W.npz"),
        'results/superpotential_W.npz',
    ]
    for sp in search_paths:
        if sp and os.path.exists(sp):
            data = np.load(sp)
            if 'histogram' in data:
                histogram = data['histogram']
                print(f"  Loaded histogram from {sp}")
            break

    results_all = {}

    # Test A
    if args.data and HAS_GEMMI:
        results_all['test_a'] = run_test_a(
            args.data, args.output, n_sample=args.sample)
    elif args.data:
        print("\n  Skipping Test A (gemmi not installed)")
    else:
        print("\n  Skipping Test A (no --data)")

    # Test B
    if args.pdb_dir and HAS_GEMMI:
        results_all['test_b'] = run_test_b(args.pdb_dir, args.output)
    elif args.pdb_dir:
        print("\n  Skipping Test B (gemmi not installed)")
    else:
        # Try common PDB locations
        for pdir in ['pdb_structures', 'pdb', 'PDB']:
            if os.path.isdir(pdir):
                print(f"  Found PDB directory: {pdir}")
                results_all['test_b'] = run_test_b(pdir, args.output)
                break
        else:
            print("\n  Skipping Test B (no --pdb-dir and no "
                  "pdb_structures/ found)")

    # Test C (always runs)
    results_all['test_c'] = run_test_c(
        args.output, histogram, n_chains=100, chain_length=200)

    # Write report
    report_path = os.path.join(args.output, "hardening_tests_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Pre-Submission Hardening Tests\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n\n")

        if 'test_a' in results_all:
            ta = results_all['test_a']
            f.write("## Test A: Local-Window Null\n\n")
            f.write("Does positional averaging within segments "
                    "inflate Seg/Real?\n\n")
            f.write(f"N = {ta['n_proteins']} proteins\n\n")
            f.write("| Null variant | Seg/Real | 95% CI |\n")
            f.write("|---|---|---|\n")
            for label, key, ci_key in [
                ('Standard (full pool)', 'Standard (full pool)',
                 'ci_standard'),
                ('Local k=3', 'Local k=3', 'ci_local3'),
                ('Local k=5', 'Local k=5', 'ci_local5'),
                ('Local k=10', 'Local k=10', 'ci_local10'),
            ]:
                val = ta['results'][key]
                ci = ta[ci_key]
                f.write(f"| {label} | **{val:.4f}×** | "
                        f"[{ci[0]:.3f}, {ci[1]:.3f}] |\n")
            f.write("\n")

        if 'test_b' in results_all and results_all['test_b']:
            tb = results_all['test_b']
            f.write("## Test B: PDB Torus Seg/Real\n\n")
            f.write(f"N = {tb['n_structures']} experimental structures\n\n")
            f.write(f"- Torus L1 Seg/Real = **{tb['seg_real']:.3f}×** "
                    f"[{tb['ci'][0]:.3f}, {tb['ci'][1]:.3f}]\n")
            f.write(f"- Torus L1 S/Real = **{tb['s_real']:.3f}×**\n")
            f.write(f"- Mean real roughness: {tb['real_mean']:.4f}\n\n")

        tc = results_all['test_c']
        f.write("## Test C: Steric Rejection Diagnostics\n\n")
        f.write(f"100 chains × 200 residues\n\n")
        f.write(f"- Overall rejection rate: "
                f"**{tc['overall_rejection_rate']*100:.2f}%**\n")
        f.write(f"- Early chain (pos 1-49): "
                f"{tc['early_rejection_rate']*100:.2f}%\n")
        f.write(f"- Late chain (pos 150+): "
                f"{tc['late_rejection_rate']*100:.2f}%\n")
        f.write(f"- Accepted vs proposed distribution correlation: "
                f"**{tc['distribution_correlation']:.6f}**\n")
        f.write(f"- L1 distance: {tc['l1_distance']:.6f}\n")
        f.write(f"- KL divergence: {tc['kl_divergence']:.6f}\n\n")

    print(f"\n  Report: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
