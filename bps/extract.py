"""Dihedral extraction from PDB files and PDB download utility.

Parses PDB files using BioPython and returns per-residue (phi, psi) angles
in radians. CRITICAL: always extracts a single specified chain to avoid the
confirmed multi-chain bug (1AON/GroEL: 14 subunits -> 10x inflated BPS).
"""

import math
import os
import urllib.request
import urllib.error
from typing import Dict, List, Optional

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.Polypeptide import is_aa

from bps.superpotential import assign_basin

# Default cache directory for downloaded PDB files
_DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "pdb_cache")


def download_pdb(pdb_id: str, cache_dir: str = _DEFAULT_CACHE) -> Optional[str]:
    """Download a PDB file from RCSB. Returns local path or None on failure.

    Parameters
    ----------
    pdb_id : str
        4-character PDB ID (e.g. '1UBQ'). Will be uppercased.
    cache_dir : str
        Directory to cache downloaded files.

    Returns
    -------
    str or None
        Path to the downloaded PDB file, or None if download failed.
    """
    pdb_id = pdb_id.upper()
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{pdb_id}.pdb")

    if os.path.exists(local_path):
        return local_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  Failed to download {pdb_id}: {e}")
        # Clean up partial downloads
        if os.path.exists(local_path):
            os.remove(local_path)
        return None


def extract_dihedrals_pdb(pdb_path: str, chain_id: str = "A") -> List[Dict]:
    """Extract backbone dihedrals from a PDB file using BioPython.

    CRITICAL: Only extracts the specified chain. BioPython's PPBuilder walks
    ALL chains by default. We filter to the target chain BEFORE building
    peptides to avoid the confirmed multi-chain bug.

    Parameters
    ----------
    pdb_path : str
        Path to a PDB file.
    chain_id : str
        Chain identifier to extract (default 'A'). Case-sensitive.

    Returns
    -------
    list of dict
        Each dict has keys: 'resnum' (int), 'resname' (str), 'phi' (float or
        None), 'psi' (float or None). Angles are in radians. phi is None for
        the first residue; psi is None for the last residue.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    # Get the first model (NMR structures may have multiple)
    model = structure[0]

    # Find the target chain
    if chain_id not in model:
        available = [c.id for c in model.get_chains()]
        raise ValueError(
            f"Chain '{chain_id}' not found in {pdb_path}. "
            f"Available chains: {available}")

    chain = model[chain_id]

    # Build polypeptides from ONLY the target chain.
    # PPBuilder.build_peptides() accepts a Chain object to restrict scope.
    ppb = PPBuilder()
    polypeptides = ppb.build_peptides(chain)

    residues = []
    for pp in polypeptides:
        phi_psi_list = pp.get_phi_psi_list()
        for residue, (phi, psi) in zip(pp, phi_psi_list):
            # Skip non-standard amino acids
            if not is_aa(residue, standard=True):
                continue
            residues.append({
                "resnum": residue.get_id()[1],
                "resname": residue.get_resname(),
                "phi": float(phi) if phi is not None else None,
                "psi": float(psi) if psi is not None else None,
            })

    return residues


def compute_basin_fractions(residues: List[Dict]) -> Dict[str, float]:
    """Compute the fraction of residues in each Ramachandran basin.

    Parameters
    ----------
    residues : list of dict
        Output from extract_dihedrals_pdb().

    Returns
    -------
    dict
        Basin name -> fraction. Keys: 'alpha', 'beta', 'ppII', 'alphaL', 'other'.
    """
    counts: Dict[str, int] = {
        "alpha": 0, "beta": 0, "ppII": 0, "alphaL": 0, "other": 0
    }
    n_assigned = 0

    for r in residues:
        if r["phi"] is None or r["psi"] is None:
            continue
        phi_deg = math.degrees(r["phi"])
        psi_deg = math.degrees(r["psi"])
        basin = assign_basin(phi_deg, psi_deg)
        counts[basin] += 1
        n_assigned += 1

    if n_assigned == 0:
        return {k: 0.0 for k in counts}

    return {k: v / n_assigned for k, v in counts.items()}


def _generate_test_pdb(path: str) -> None:
    """Generate a synthetic 20-residue, 2-chain PDB for testing.

    Chain A: 12 residues (6 alpha-helix + 2 coil + 4 beta-sheet).
    Chain B: 8 residues (all alpha-helix, should NOT appear in chain-A extraction).

    Backbone atoms are placed with standard peptide geometry and the target
    (phi, psi) at each residue, using NeRF (Natural Extension Reference Frame)
    for coordinate generation.
    """
    # Standard peptide bond geometry (Angstroms, radians)
    BOND_N_CA = 1.458
    BOND_CA_C = 1.525
    BOND_C_N = 1.329
    ANGLE_N_CA_C = math.radians(111.2)
    ANGLE_CA_C_N = math.radians(116.2)
    ANGLE_C_N_CA = math.radians(121.7)
    OMEGA = math.pi  # trans peptide

    def _nerf(a: np.ndarray, b: np.ndarray, c: np.ndarray,
              bond_len: float, bond_angle: float, torsion: float) -> np.ndarray:
        """Place atom D given atoms A, B, C, bond length C-D, angle B-C-D,
        and torsion A-B-C-D."""
        bc = c - b
        bc = bc / np.linalg.norm(bc)
        ab = b - a
        n = np.cross(ab, bc)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            # Degenerate case: pick an arbitrary perpendicular
            if abs(bc[0]) < 0.9:
                n = np.cross(np.array([1, 0, 0]), bc)
            else:
                n = np.cross(np.array([0, 1, 0]), bc)
            n_norm = np.linalg.norm(n)
        n = n / n_norm
        m = np.cross(n, bc)

        d = c + bond_len * (
            -math.cos(bond_angle) * bc
            + math.sin(bond_angle) * math.cos(torsion) * m
            + math.sin(bond_angle) * math.sin(torsion) * n
        )
        return d

    def _build_chain_coords(phi_psi_list):
        """Build N, CA, C coordinates for a chain given (phi, psi) per residue."""
        # Seed with first three atoms along x-axis
        coords = []
        N0 = np.array([0.0, 0.0, 0.0])
        CA0 = np.array([BOND_N_CA, 0.0, 0.0])
        C0 = _nerf(
            np.array([-1.0, 0.0, 0.0]), N0, CA0,
            BOND_CA_C, ANGLE_N_CA_C, math.pi  # initial psi placeholder
        )
        coords.append((N0, CA0, C0))

        for i in range(1, len(phi_psi_list)):
            prev_N, prev_CA, prev_C = coords[-1]
            _, psi_prev = phi_psi_list[i - 1]
            phi_curr, _ = phi_psi_list[i]

            # Place next N using psi of previous residue
            if i == 1:
                # For the first transition, use omega
                N_next = _nerf(prev_N, prev_CA, prev_C,
                               BOND_C_N, ANGLE_CA_C_N, psi_prev)
            else:
                N_next = _nerf(prev_N, prev_CA, prev_C,
                               BOND_C_N, ANGLE_CA_C_N, psi_prev)

            # Place CA using phi of current residue
            CA_next = _nerf(prev_CA, prev_C, N_next,
                            BOND_N_CA, ANGLE_C_N_CA, OMEGA)

            # Place C using psi of current residue (will be refined by next iter)
            C_next = _nerf(prev_C, N_next, CA_next,
                           BOND_CA_C, ANGLE_N_CA_C, phi_curr)

            coords.append((N_next, CA_next, C_next))
        return coords

    # Define target (phi, psi) in radians for chain A (12 residues):
    # 6 alpha-helix (-63, -43), 2 coil (-80, 60), 4 beta-sheet (-120, 130)
    chain_a_phi_psi = [
        (math.radians(-63), math.radians(-43)),   # alpha
        (math.radians(-63), math.radians(-43)),
        (math.radians(-63), math.radians(-43)),
        (math.radians(-63), math.radians(-43)),
        (math.radians(-63), math.radians(-43)),
        (math.radians(-63), math.radians(-43)),
        (math.radians(-80), math.radians(60)),     # coil
        (math.radians(-80), math.radians(60)),
        (math.radians(-120), math.radians(130)),   # beta
        (math.radians(-120), math.radians(130)),
        (math.radians(-120), math.radians(130)),
        (math.radians(-120), math.radians(130)),
    ]

    # Chain B: 8 alpha-helix residues (multi-chain trap)
    chain_b_phi_psi = [
        (math.radians(-63), math.radians(-43)),
    ] * 8

    coords_a = _build_chain_coords(chain_a_phi_psi)
    coords_b = _build_chain_coords(chain_b_phi_psi)
    # Offset chain B by 50 A so it doesn't overlap
    coords_b = [(n + 50, ca + 50, c + 50) for n, ca, c in coords_b]

    resnames = ["ALA"] * 20
    lines = []
    atom_num = 1

    def _atom_line(serial, name, resname, chain, resseq, x, y, z):
        # Standard PDB ATOM record format
        return (f"ATOM  {serial:5d} {name:<4s} {resname:3s} {chain}"
                f"{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00           {name[0]:>2s}  ")

    for i, (N, CA, C) in enumerate(coords_a):
        resseq = i + 1
        lines.append(_atom_line(atom_num, "N", resnames[i], "A", resseq,
                                N[0], N[1], N[2]))
        atom_num += 1
        lines.append(_atom_line(atom_num, "CA", resnames[i], "A", resseq,
                                CA[0], CA[1], CA[2]))
        atom_num += 1
        lines.append(_atom_line(atom_num, "C", resnames[i], "A", resseq,
                                C[0], C[1], C[2]))
        atom_num += 1

    lines.append("TER")
    for i, (N, CA, C) in enumerate(coords_b):
        resseq = i + 1
        lines.append(_atom_line(atom_num, "N", resnames[i], "B", resseq,
                                N[0], N[1], N[2]))
        atom_num += 1
        lines.append(_atom_line(atom_num, "CA", resnames[i], "B", resseq,
                                CA[0], CA[1], CA[2]))
        atom_num += 1
        lines.append(_atom_line(atom_num, "C", resnames[i], "B", resseq,
                                C[0], C[1], C[2]))
        atom_num += 1
    lines.append("TER")
    lines.append("END")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    from bps.superpotential import (build_superpotential, lookup_W_batch,
                                    compute_bps_per_residue)

    print("Priority 2 validation: dihedral extraction from PDB")
    print("=" * 55)
    print()

    # Try to download 1UBQ; fall back to synthetic PDB if network unavailable
    pdb_path = download_pdb("1UBQ")
    using_synthetic = False

    if pdb_path is not None:
        print(f"  Using real 1UBQ: {pdb_path}")
        expected_len = 76
        # Expected basin fractions for ubiquitin
        expected_alpha = (0.20, 0.60)
        expected_beta = (0.05, 0.40)
    else:
        print("  Network unavailable -- using synthetic test PDB")
        syn_path = os.path.join(_DEFAULT_CACHE, "SYNTHETIC_TEST.pdb")
        _generate_test_pdb(syn_path)
        pdb_path = syn_path
        using_synthetic = True
        expected_len = 12
        # Synthetic: 6 alpha, 2 coil, 4 beta out of 12 residues
        # Terminal residues lack phi or psi, so basin fractions are
        # computed over ~10 interior residues.
        expected_alpha = (0.20, 0.70)
        expected_beta = (0.15, 0.55)

    print()

    # --- Test 1: Single-chain extraction ---
    residues = extract_dihedrals_pdb(pdb_path, chain_id="A")
    print(f"  Extracted {len(residues)} residues from chain A")

    pct_diff = abs(len(residues) - expected_len) / expected_len * 100
    if pct_diff > 30:
        print(f"  WARNING: residue count {len(residues)} differs from "
              f"expected {expected_len} by {pct_diff:.0f}%")
    else:
        print(f"  PASS: Residue count OK ({len(residues)} vs expected {expected_len})")
    print()

    # --- Test 2: Multi-chain guard (synthetic only) ---
    if using_synthetic:
        residues_b = extract_dihedrals_pdb(pdb_path, chain_id="B")
        print(f"  Chain B: {len(residues_b)} residues (should be 8, not 20)")
        assert len(residues_b) == 8, (
            f"Multi-chain bug! Chain B has {len(residues_b)} residues, expected 8")
        assert len(residues) == 12, (
            f"Multi-chain bug! Chain A has {len(residues)} residues, expected 12")
        print("  PASS: Chain isolation works (A=12, B=8, no leakage)")
        print()

    # --- Test 3: Show extracted angles ---
    print("  First 5 residues:")
    for r in residues[:5]:
        phi_s = f"{math.degrees(r['phi']):+7.1f} deg" if r["phi"] is not None else "    None"
        psi_s = f"{math.degrees(r['psi']):+7.1f} deg" if r["psi"] is not None else "    None"
        print(f"    {r['resnum']:4d} {r['resname']}  phi={phi_s}  psi={psi_s}")
    if len(residues) > 5:
        print("  ...")
        for r in residues[-3:]:
            phi_s = f"{math.degrees(r['phi']):+7.1f} deg" if r["phi"] is not None else "    None"
            psi_s = f"{math.degrees(r['psi']):+7.1f} deg" if r["psi"] is not None else "    None"
            print(f"    {r['resnum']:4d} {r['resname']}  phi={phi_s}  psi={psi_s}")
    print()

    # --- Test 4: Basin fractions ---
    fractions = compute_basin_fractions(residues)
    print("  Basin fractions:")
    for basin, frac in sorted(fractions.items(), key=lambda x: -x[1]):
        print(f"    {basin:8s}: {frac:5.1%}")

    alpha_frac = fractions["alpha"]
    beta_frac = fractions["beta"]
    other_frac = fractions["other"] + fractions["ppII"] + fractions["alphaL"]

    print()
    print(f"  Summary: alpha={alpha_frac:.1%}, beta={beta_frac:.1%}, "
          f"other/coil={other_frac:.1%}")
    if not using_synthetic:
        print(f"  Target:  alpha~40%, beta~20%, other~40%")
    print()

    ok = True
    if not (expected_alpha[0] <= alpha_frac <= expected_alpha[1]):
        print(f"  WARNING: alpha fraction {alpha_frac:.1%} outside "
              f"[{expected_alpha[0]:.0%}, {expected_alpha[1]:.0%}]")
        ok = False
    else:
        print(f"  PASS: alpha fraction {alpha_frac:.1%} in expected range")

    if not (expected_beta[0] <= beta_frac <= expected_beta[1]):
        print(f"  WARNING: beta fraction {beta_frac:.1%} outside "
              f"[{expected_beta[0]:.0%}, {expected_beta[1]:.0%}]")
        ok = False
    else:
        print(f"  PASS: beta fraction {beta_frac:.1%} in expected range")

    # --- Test 5: End-to-end BPS/L ---
    print()
    print("  Computing BPS/L...")
    W, phi_g, psi_g = build_superpotential()

    valid = [(r["phi"], r["psi"]) for r in residues
             if r["phi"] is not None and r["psi"] is not None]
    phi_arr = np.array([v[0] for v in valid])
    psi_arr = np.array([v[1] for v in valid])

    W_values = lookup_W_batch(W, phi_g, psi_g, phi_arr, psi_arr)
    bps_l = compute_bps_per_residue(W_values)
    print(f"  BPS/L = {bps_l:.3f}")
    print(f"  Target range: [0.10, 0.35]")

    # Synthetic chain has abrupt alpha->coil->beta transitions in only 12
    # residues, so BPS/L will be much higher than for real 50+ residue proteins.
    bps_lo, bps_hi = (0.10, 0.35) if not using_synthetic else (0.20, 0.80)
    if bps_lo <= bps_l <= bps_hi:
        print("  PASS: BPS/L in expected range")
    else:
        print(f"  WARNING: BPS/L = {bps_l:.3f} outside expected "
              f"[{bps_lo:.2f}, {bps_hi:.2f}]")
        ok = False

    # --- Test 6: Invalid chain raises ValueError ---
    print()
    try:
        extract_dihedrals_pdb(pdb_path, chain_id="Z")
        print("  FAIL: Should have raised ValueError for missing chain Z")
        ok = False
    except ValueError as e:
        print(f"  PASS: Missing chain raises ValueError: {e}")

    print()
    if ok:
        print("All Priority 2 validations passed.")
    else:
        print("Some validations had warnings -- review above.")
