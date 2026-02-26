#!/usr/bin/env python3
"""
TorusFold AlphaFold Pipeline
============================
Processes AlphaFold proteome .cif files to compute:
  1. Per-protein BPS/L with realistic superpotential
  2. Fold-class breakdown (all-α, all-β, α/β, α+β)
  3. Cross-organism conservation (the missing paper number)
  4. Loop path taxonomy at scale

Usage:
  python alphafold_pipeline.py /path/to/alphafold_data [options]

Expected directory structure:
  /path/to/alphafold_data/
    organism_1/
      AF-*.cif
    organism_2/
      AF-*.cif
    ...

Dependencies: numpy, scipy, gemmi (pip install numpy scipy gemmi)
              Falls back to BioPython if gemmi unavailable.

Output: results/ directory with CSV files and markdown report.
"""

import os
import sys
import time
import math
import json
import hashlib
import argparse
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter

# ─── Try gemmi first (fast), fall back to BioPython ──────────────

CIF_BACKEND = None
try:
    import gemmi
    CIF_BACKEND = "gemmi"
except ImportError:
    try:
        from Bio.PDB.MMCIFParser import MMCIFParser as _BioParser
        CIF_BACKEND = "biopython"
    except ImportError:
        pass

if CIF_BACKEND is None:
    print("ERROR: Need either gemmi or BioPython installed.")
    print("  pip install gemmi          (recommended, 10x faster)")
    print("  pip install biopython      (fallback)")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ResidueAngles:
    phi: float  # radians
    psi: float  # radians
    plddt: float  # 0-100
    resname: str
    resnum: int


@dataclass
class ProteinResult:
    filepath: str
    organism: str
    uniprot_id: str
    chain_length: int
    n_residues_used: int  # after pLDDT filter
    bps_l: float
    bps_total: float
    frac_alpha: float
    frac_beta: float
    frac_other: float
    fold_class: str
    mean_plddt: float
    basin_sequence: str  # compact: 'aaabbboobba...'
    n_loops: int
    error: str = ""


@dataclass
class LoopResult:
    protein_id: str
    organism: str
    start_res: int
    end_res: int
    length: int
    direction: str  # 'ab' or 'ba'
    delta_w: float
    torus_path_len: float
    residues: str  # amino acid sequence
    phi_psi: str  # json-encoded list of (phi,psi) pairs


# ═══════════════════════════════════════════════════════════════════
# DIHEDRAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _dihedral_angle(p0, p1, p2, p3):
    """Compute dihedral angle from four 3D points. Returns radians."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    n1 /= n1_norm
    n2 /= n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return math.atan2(y, x)


def extract_dihedrals_gemmi(filepath: str, plddt_min: float = 70.0) -> list[ResidueAngles]:
    """Extract backbone dihedrals from .cif using gemmi."""
    st = gemmi.read_structure(str(filepath))
    if len(st) == 0:
        return []

    model = st[0]
    if len(model) == 0:
        return []

    chain = model[0]  # first chain

    # Collect backbone atoms by residue
    residues = []
    for res in chain:
        # Check if amino acid using gemmi's residue table
        info = gemmi.find_tabulated_residue(res.name)
        if not info.is_amino_acid():
            continue
        atoms = {}
        for atom in res:
            if atom.name in ('N', 'CA', 'C'):
                atoms[atom.name] = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
        if len(atoms) == 3:
            # pLDDT is in B-factor of CA
            ca_atom = res.find_atom('CA', '*')
            plddt = ca_atom.b_iso if ca_atom else 0.0
            residues.append({
                'N': atoms['N'], 'CA': atoms['CA'], 'C': atoms['C'],
                'plddt': plddt, 'resname': res.name, 'resnum': res.seqid.num
            })

    if len(residues) < 3:
        return []

    # Compute phi/psi
    angles = []
    for i in range(1, len(residues) - 1):
        prev_c = residues[i - 1]['C']
        curr_n = residues[i]['N']
        curr_ca = residues[i]['CA']
        curr_c = residues[i]['C']
        next_n = residues[i + 1]['N']

        phi = -_dihedral_angle(prev_c, curr_n, curr_ca, curr_c)
        psi = -_dihedral_angle(curr_n, curr_ca, curr_c, next_n)
        plddt = residues[i]['plddt']

        if plddt >= plddt_min:
            angles.append(ResidueAngles(
                phi=phi, psi=psi, plddt=plddt,
                resname=residues[i]['resname'],
                resnum=residues[i]['resnum']
            ))

    return angles


def extract_dihedrals_biopython(filepath: str, plddt_min: float = 70.0) -> list[ResidueAngles]:
    """Extract backbone dihedrals from .cif using BioPython (slower fallback)."""
    parser = _BioParser(QUIET=True)
    structure = parser.get_structure("prot", str(filepath))
    model = structure[0]
    chains = list(model.get_chains())
    if not chains:
        return []
    chain = chains[0]

    residues_data = []
    for res in chain:
        if res.id[0] != ' ':  # skip hetero
            continue
        try:
            n = res['N'].get_vector().get_array()
            ca = res['CA'].get_vector().get_array()
            c = res['C'].get_vector().get_array()
            plddt = res['CA'].get_bfactor()
            residues_data.append({
                'N': np.array(n), 'CA': np.array(ca), 'C': np.array(c),
                'plddt': plddt, 'resname': res.get_resname(),
                'resnum': res.id[1]
            })
        except KeyError:
            continue

    if len(residues_data) < 3:
        return []

    angles = []
    for i in range(1, len(residues_data) - 1):
        prev_c = residues_data[i - 1]['C']
        curr_n = residues_data[i]['N']
        curr_ca = residues_data[i]['CA']
        curr_c = residues_data[i]['C']
        next_n = residues_data[i + 1]['N']

        phi = -_dihedral_angle(prev_c, curr_n, curr_ca, curr_c)
        psi = -_dihedral_angle(curr_n, curr_ca, curr_c, next_n)
        plddt = residues_data[i]['plddt']

        if plddt >= plddt_min:
            angles.append(ResidueAngles(
                phi=phi, psi=psi, plddt=plddt,
                resname=residues_data[i]['resname'],
                resnum=residues_data[i]['resnum']
            ))

    return angles


def extract_dihedrals(filepath: str, plddt_min: float = 70.0) -> list[ResidueAngles]:
    """Extract dihedrals using best available backend."""
    if CIF_BACKEND == "gemmi":
        return extract_dihedrals_gemmi(filepath, plddt_min)
    else:
        return extract_dihedrals_biopython(filepath, plddt_min)


# ═══════════════════════════════════════════════════════════════════
# SUPERPOTENTIAL W
# ═══════════════════════════════════════════════════════════════════

class Superpotential:
    """Empirical potential of mean force W(phi, psi) on 360x360 grid."""

    def __init__(self, grid: np.ndarray):
        """grid: 360x360 array, W values. Index [i][j] = W at phi=i-180, psi=j-180 degrees."""
        assert grid.shape == (360, 360), f"Expected (360,360), got {grid.shape}"
        self.grid = grid
        self.w_min = float(np.min(grid))
        self.w_max = float(np.max(grid))

    def __call__(self, phi_rad: float, psi_rad: float) -> float:
        """Look up W at (phi, psi) in radians. Nearest-neighbor grid lookup."""
        gi = round(math.degrees(phi_rad) + 180) % 360
        gj = round(math.degrees(psi_rad) + 180) % 360
        return float(self.grid[gi, gj])

    def lookup_array(self, phi_rad: np.ndarray, psi_rad: np.ndarray) -> np.ndarray:
        """Vectorized lookup for arrays of angles."""
        gi = np.round(np.degrees(phi_rad) + 180).astype(int) % 360
        gj = np.round(np.degrees(psi_rad) + 180).astype(int) % 360
        return self.grid[gi, gj]

    @staticmethod
    def from_von_mises(grid_size: int = 360) -> 'Superpotential':
        """Build from the shared von Mises -sqrt(P) construction."""
        from bps.superpotential import build_superpotential as _build
        W_grid, _, _ = _build(grid_size)
        return Superpotential(W_grid)

    @staticmethod
    def from_angles(phi_psi_pairs: np.ndarray, smooth_sigma: float = 1.5,
                    ) -> 'Superpotential':
        """Build W = -sqrt(P) from observed (phi, psi) pairs.

        Args:
            phi_psi_pairs: Nx2 array of (phi, psi) in radians
            smooth_sigma: Gaussian smoothing in grid points (default 1.5)
        """
        grid = np.zeros((360, 360), dtype=np.float64)

        # Bin observations
        phi_deg = np.degrees(phi_psi_pairs[:, 0])
        psi_deg = np.degrees(phi_psi_pairs[:, 1])
        gi = (np.round(phi_deg).astype(int) + 180) % 360
        gj = (np.round(psi_deg).astype(int) + 180) % 360

        for i, j in zip(gi, gj):
            grid[i, j] += 1

        # Normalize to density
        total = grid.sum()
        if total > 0:
            grid /= total

        grid = np.maximum(grid, np.max(grid) * 1e-6 if grid.max() > 0 else 1e-10)
        W = -np.sqrt(grid)

        # Gaussian smoothing (periodic BC)
        if smooth_sigma > 0:
            W = gaussian_filter(W, sigma=smooth_sigma, mode='wrap')

        return Superpotential(W)

    def save(self, filepath: str):
        """Save W grid to compressed numpy file."""
        np.savez_compressed(filepath, grid=self.grid)

    @staticmethod
    def load(filepath: str) -> 'Superpotential':
        """Load W grid from numpy file."""
        data = np.load(filepath)
        return Superpotential(data['grid'])

    def stats(self) -> dict:
        """Compute landscape statistics."""
        # Alpha basin center: phi=-63, psi=-43 -> grid[117][137]
        alpha_idx = (117, 137)
        # Beta basin center: phi=-120, psi=130 -> grid[60][310]
        beta_idx = (60, 310)
        # Approximate saddle (ridge between basins)
        # Search for minimum along phi=-90 line (grid index 90)
        saddle_line = self.grid[90, :]
        saddle_psi = np.argmin(saddle_line)
        saddle_val = saddle_line[saddle_psi]

        w_alpha = self.grid[alpha_idx]
        w_beta = self.grid[beta_idx]

        return {
            'W_alpha': float(w_alpha),
            'W_beta': float(w_beta),
            'W_saddle': float(saddle_val),
            'depth_alpha': float(saddle_val - w_alpha),
            'depth_beta': float(saddle_val - w_beta),
            'W_range': [float(self.w_min), float(self.w_max)],
        }


# ═══════════════════════════════════════════════════════════════════
# SECONDARY STRUCTURE ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════

def assign_ss(phi_rad: float, psi_rad: float) -> str:
    """Assign secondary structure from (phi, psi) in radians.

    Alpha: -160 < phi < 0, -120 < psi < 30 (degrees)
    Beta:  -170 < phi < -70, psi > 90 OR psi < -120 (degrees)
           Single condition handles the psi wrap at ±180°
    """
    phi = math.degrees(phi_rad)
    psi = math.degrees(psi_rad)

    # Alpha basin
    if -160 < phi < 0 and -120 < psi < 30:
        return 'a'

    # Beta basin (single condition, handles wrap)
    if -170 < phi < -70 and (psi > 90 or psi < -120):
        return 'b'

    return 'o'


def classify_fold(frac_alpha: float, frac_beta: float) -> str:
    """Classify fold from SS fractions.

    Uses SCOP-like thresholds:
      all-alpha:  alpha >= 35% and beta < 10%
      all-beta:   beta >= 25% and alpha < 15%
      alpha/beta:  alpha >= 20% and beta >= 15%
      alpha+beta:  alpha >= 10% and beta >= 10%
    """
    if frac_alpha >= 0.35 and frac_beta < 0.10:
        return 'all-alpha'
    if frac_beta >= 0.25 and frac_alpha < 0.15:
        return 'all-beta'
    if frac_alpha >= 0.20 and frac_beta >= 0.15:
        return 'alpha/beta'
    if frac_alpha >= 0.10 and frac_beta >= 0.10:
        return 'alpha+beta'
    return 'other'


# ═══════════════════════════════════════════════════════════════════
# BPS/L COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_bps_l(angles: list[ResidueAngles], W: Superpotential) -> tuple[float, float]:
    """Compute BPS/L for a protein.

    Returns (bps_l, bps_total).
    """
    if len(angles) < 2:
        return 0.0, 0.0

    phi = np.array([a.phi for a in angles])
    psi = np.array([a.psi for a in angles])
    w_vals = W.lookup_array(phi, psi)

    dw = np.abs(np.diff(w_vals))
    bps_total = float(np.sum(dw))
    L = len(angles)
    bps_l = bps_total / L

    return bps_l, bps_total


# ═══════════════════════════════════════════════════════════════════
# LOOP EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _angular_diff(a: float, b: float) -> float:
    """Signed angular difference on circle, using atan2."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def extract_loops(angles: list[ResidueAngles], W: Superpotential,
                  protein_id: str, organism: str) -> list[LoopResult]:
    """Extract α↔β loops from a protein's dihedral sequence."""
    if len(angles) < 5:
        return []

    # Assign SS
    ss = [assign_ss(a.phi, a.psi) for a in angles]

    # Find SS segments (contiguous runs of same type)
    segments = []
    start = 0
    for i in range(1, len(ss)):
        if ss[i] != ss[start]:
            if ss[start] in ('a', 'b') and (i - start) >= 2:
                segments.append((start, i - 1, ss[start]))
            start = i
    if ss[start] in ('a', 'b') and (len(ss) - start) >= 2:
        segments.append((start, len(ss) - 1, ss[start]))

    # Find loops between consecutive α/β segments
    loops = []
    for idx in range(len(segments) - 1):
        s1_start, s1_end, s1_type = segments[idx]
        s2_start, s2_end, s2_type = segments[idx + 1]

        # Must be different types (α→β or β→α)
        if s1_type == s2_type:
            continue

        # Loop is the residues between the two SS segments
        loop_start = s1_end + 1
        loop_end = s2_start - 1

        if loop_start > loop_end:
            # Adjacent SS elements with no intervening residues
            continue

        loop_len = loop_end - loop_start + 1
        if loop_len < 1 or loop_len > 20:
            continue

        # Direction
        direction = f"{s1_type}{s2_type}"  # 'ab' or 'ba'

        # Compute |ΔW| across loop
        w_start = W(angles[s1_end].phi, angles[s1_end].psi)
        w_end = W(angles[s2_start].phi, angles[s2_start].psi)
        delta_w = abs(w_end - w_start)

        # Torus path length
        torus_len = 0.0
        for i in range(loop_start, loop_end + 1):
            if i > loop_start:
                dphi = _angular_diff(angles[i].phi, angles[i - 1].phi)
                dpsi = _angular_diff(angles[i].psi, angles[i - 1].psi)
                torus_len += math.sqrt(dphi ** 2 + dpsi ** 2)

        # Residue names
        residues = ''.join(a.resname[0] if len(a.resname) == 3 else '?' for a in angles[loop_start:loop_end + 1])

        # Phi/psi for clustering
        phi_psi = [(a.phi, a.psi) for a in angles[loop_start:loop_end + 1]]

        loops.append(LoopResult(
            protein_id=protein_id,
            organism=organism,
            start_res=angles[loop_start].resnum,
            end_res=angles[loop_end].resnum,
            length=loop_len,
            direction=direction,
            delta_w=delta_w,
            torus_path_len=torus_len,
            residues=residues,
            phi_psi=json.dumps(phi_psi),
        ))

    return loops


# ═══════════════════════════════════════════════════════════════════
# MARKOV AND SHUFFLE NULL MODELS
# ═══════════════════════════════════════════════════════════════════

def markov_shuffle(angles: list[ResidueAngles], W: Superpotential,
                   n_trials: int = 10) -> tuple[float, float]:
    """Compute Markov and Shuffled BPS/L for three-level decomposition.

    Markov: preserve basin transition sequence, randomize intra-basin position
            from GLOBAL basin pool (all residues in that basin across this protein).
    Shuffled: random permutation of all (phi, psi) angles.

    Returns (markov_bps_l, shuffled_bps_l) averaged over n_trials.
    """
    if len(angles) < 3:
        return 0.0, 0.0

    phi = np.array([a.phi for a in angles])
    psi = np.array([a.psi for a in angles])
    ss = np.array([assign_ss(a.phi, a.psi) for a in angles])

    # Build basin pools
    basin_pools = defaultdict(list)
    for i, s in enumerate(ss):
        basin_pools[s].append(i)

    rng = np.random.default_rng(42)

    markov_bps = []
    shuffled_bps = []

    for _ in range(n_trials):
        # Markov: same basin sequence, random position within basin
        m_phi = np.empty_like(phi)
        m_psi = np.empty_like(psi)
        for i, s in enumerate(ss):
            pool = basin_pools[s]
            j = rng.choice(pool)
            m_phi[i] = phi[j]
            m_psi[i] = psi[j]
        m_w = W.lookup_array(m_phi, m_psi)
        m_dw = np.abs(np.diff(m_w))
        L = len(angles)
        markov_bps.append(float(np.sum(m_dw)) / L)

        # Shuffled: random permutation
        perm = rng.permutation(len(phi))
        s_w = W.lookup_array(phi[perm], psi[perm])
        s_dw = np.abs(np.diff(s_w))
        shuffled_bps.append(float(np.sum(s_dw)) / L)

    return float(np.mean(markov_bps)), float(np.mean(shuffled_bps))


# ═══════════════════════════════════════════════════════════════════
# SINGLE PROTEIN PROCESSING
# ═══════════════════════════════════════════════════════════════════

def process_protein(filepath: str, organism: str, W: Superpotential,
                    plddt_min: float = 70.0, min_length: int = 50,
                    compute_loops: bool = True) -> tuple[Optional[ProteinResult], list[LoopResult]]:
    """Process a single protein structure file.

    Returns (ProteinResult or None, list of LoopResults).
    """
    try:
        angles = extract_dihedrals(filepath, plddt_min)
    except Exception as e:
        return ProteinResult(
            filepath=filepath, organism=organism, uniprot_id="",
            chain_length=0, n_residues_used=0, bps_l=0, bps_total=0,
            frac_alpha=0, frac_beta=0, frac_other=0, fold_class="error",
            mean_plddt=0, basin_sequence="", n_loops=0,
            error=str(e)
        ), []

    if len(angles) < min_length:
        return None, []

    # Extract UniProt ID from filename (AF-XXXXX-F1-model_v4.cif)
    fname = Path(filepath).stem
    parts = fname.split('-')
    uniprot_id = parts[1] if len(parts) >= 2 else fname

    # BPS/L
    bps_l, bps_total = compute_bps_l(angles, W)

    # SS composition
    ss = [assign_ss(a.phi, a.psi) for a in angles]
    n = len(ss)
    frac_a = ss.count('a') / n
    frac_b = ss.count('b') / n
    frac_o = ss.count('o') / n

    fold = classify_fold(frac_a, frac_b)
    mean_plddt = np.mean([a.plddt for a in angles])
    basin_seq = ''.join(ss)

    # Loops
    loops = []
    if compute_loops:
        loops = extract_loops(angles, W, uniprot_id, organism)

    result = ProteinResult(
        filepath=filepath,
        organism=organism,
        uniprot_id=uniprot_id,
        chain_length=len(angles) + 2,  # +2 for terminal residues
        n_residues_used=len(angles),
        bps_l=bps_l,
        bps_total=bps_total,
        frac_alpha=frac_a,
        frac_beta=frac_b,
        frac_other=frac_o,
        fold_class=fold,
        mean_plddt=float(mean_plddt),
        basin_sequence=basin_seq,
        n_loops=len(loops),
    )

    return result, loops


# ═══════════════════════════════════════════════════════════════════
# WORKER (for multiprocessing)
# ═══════════════════════════════════════════════════════════════════

# Globals for worker processes
_W_GRID = None
_PLDDT_MIN = 70.0
_MIN_LENGTH = 50
_COMPUTE_LOOPS = True


def _init_worker(w_grid_shared, plddt_min, min_length, compute_loops):
    global _W_GRID, _PLDDT_MIN, _MIN_LENGTH, _COMPUTE_LOOPS
    _W_GRID = Superpotential(w_grid_shared)
    _PLDDT_MIN = plddt_min
    _MIN_LENGTH = min_length
    _COMPUTE_LOOPS = compute_loops


def _worker(args):
    filepath, organism = args
    return process_protein(filepath, organism, _W_GRID, _PLDDT_MIN, _MIN_LENGTH, _COMPUTE_LOOPS)


# ═══════════════════════════════════════════════════════════════════
# DISCOVERY
# ═══════════════════════════════════════════════════════════════════

def discover_files(data_dir: str) -> dict[str, list[str]]:
    """Walk data directory, return {organism: [filepaths]}."""
    data_path = Path(data_dir)
    organisms = {}

    # Check if the data dir itself contains .cif files (flat structure)
    cif_files = list(data_path.glob("*.cif")) + list(data_path.glob("*.cif.gz"))
    if cif_files and not any(p.is_dir() for p in data_path.iterdir() if not p.name.startswith('.')):
        organisms["unknown"] = [str(f) for f in sorted(cif_files)]
        return organisms

    # Walk subdirectories as organism folders
    for subdir in sorted(data_path.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith('.'):
            continue

        files = sorted(
            list(subdir.glob("*.cif")) +
            list(subdir.glob("*.cif.gz")) +
            list(subdir.glob("**/*.cif")) +
            list(subdir.glob("**/*.cif.gz"))
        )
        # Deduplicate
        files = sorted(set(str(f) for f in files))

        if files:
            organisms[subdir.name] = files

    return organisms


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: BUILD SUPERPOTENTIAL
# ═══════════════════════════════════════════════════════════════════

def build_superpotential(organisms: dict[str, list[str]],
                         max_structures: int = 2000,
                         plddt_min: float = 70.0,
                         cache_path: Optional[str] = None) -> Superpotential:
    """Build W from the canonical von Mises -sqrt(P) shared module.

    The organisms/max_structures/plddt_min parameters are accepted for
    backward compatibility but ignored — W is now a fixed parametric
    model, not data-dependent.
    """
    print("  Building superpotential from canonical von Mises -sqrt(P) mixture...")
    W = Superpotential.from_von_mises(360)

    stats = W.stats()
    print(f"  W range: [{stats['W_range'][0]:.2f}, {stats['W_range'][1]:.2f}]")
    print(f"  Alpha basin depth: {stats['depth_alpha']:.2f}")
    print(f"  Beta basin depth: {stats['depth_beta']:.2f}")

    if cache_path:
        W.save(cache_path)
        print(f"  Saved to {cache_path}")

    return W


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: PROCESS ALL STRUCTURES
# ═══════════════════════════════════════════════════════════════════

def process_all(organisms: dict[str, list[str]], W: Superpotential,
                n_workers: int = 1, plddt_min: float = 70.0,
                min_length: int = 50, compute_loops: bool = True,
                checkpoint_dir: Optional[str] = None) -> tuple[list[ProteinResult], list[LoopResult]]:
    """Process all structures, optionally in parallel."""

    # Build task list
    tasks = []
    for org, files in organisms.items():
        for f in files:
            tasks.append((f, org))

    total = len(tasks)
    print(f"  Processing {total} structures across {len(organisms)} organisms...")

    # Check for checkpoint
    completed_ids = set()
    all_results = []
    all_loops = []
    checkpoint_file = None

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.jsonl")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    completed_ids.add(data['filepath'])
                    all_results.append(ProteinResult(**{k: v for k, v in data.items() if k != 'type' and k in ProteinResult.__dataclass_fields__}))
            print(f"  Resumed from checkpoint: {len(completed_ids)} already processed")

    # Filter to remaining tasks
    remaining = [(f, org) for f, org in tasks if f not in completed_ids]
    print(f"  {len(remaining)} remaining to process")

    if not remaining:
        return all_results, all_loops

    start_time = time.time()
    done = len(completed_ids)
    batch_size = 100

    if n_workers > 1:
        # Parallel processing
        with mp.Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(W.grid, plddt_min, min_length, compute_loops)
        ) as pool:
            for result, loops in pool.imap_unordered(_worker, remaining, chunksize=10):
                done += 1
                if result is not None:
                    all_results.append(result)
                    all_loops.extend(loops)

                    # Checkpoint
                    if checkpoint_file and done % batch_size == 0:
                        with open(checkpoint_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps({
                                'type': 'protein',
                                'filepath': result.filepath,
                                'organism': result.organism,
                                'uniprot_id': result.uniprot_id,
                                'chain_length': result.chain_length,
                                'n_residues_used': result.n_residues_used,
                                'bps_l': result.bps_l,
                                'bps_total': result.bps_total,
                                'frac_alpha': result.frac_alpha,
                                'frac_beta': result.frac_beta,
                                'frac_other': result.frac_other,
                                'fold_class': result.fold_class,
                                'mean_plddt': result.mean_plddt,
                                'basin_sequence': '',  # skip for checkpoint size
                                'n_loops': result.n_loops,
                            }) + '\n')

                if done % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = (done - len(completed_ids)) / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    n_valid = sum(1 for r in all_results if r.error == "")
                    print(f"    {done}/{total} ({rate:.1f}/sec, ETA {eta/60:.0f}min) — "
                          f"{n_valid} valid, {len(all_loops)} loops")
    else:
        # Serial processing
        for filepath, organism in remaining:
            result, loops = process_protein(filepath, organism, W, plddt_min, min_length, compute_loops)
            done += 1
            if result is not None:
                all_results.append(result)
                all_loops.extend(loops)

            if done % 200 == 0:
                elapsed = time.time() - start_time
                rate = (done - len(completed_ids)) / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                n_valid = sum(1 for r in all_results if r.error == "")
                print(f"    {done}/{total} ({rate:.1f}/sec, ETA {eta/60:.0f}min) — "
                      f"{n_valid} valid, {len(all_loops)} loops")

    elapsed = time.time() - start_time
    print(f"  Done: {len(all_results)} results, {len(all_loops)} loops in {elapsed:.0f}s")

    return all_results, all_loops


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: ANALYSIS AND REPORTING
# ═══════════════════════════════════════════════════════════════════

def analyze_results(results: list[ProteinResult], loops: list[LoopResult],
                    output_dir: str, W: Superpotential):
    """Generate CSV files and summary report."""

    os.makedirs(output_dir, exist_ok=True)

    valid = [r for r in results if r.error == "" and r.bps_l > 0]
    print(f"\n  {len(valid)} valid proteins for analysis")

    # ── Per-protein CSV ──
    csv_path = os.path.join(output_dir, "per_protein_bpsl.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("uniprot_id,organism,chain_length,n_residues,bps_l,frac_alpha,frac_beta,frac_other,fold_class,mean_plddt\n")
        for r in valid:
            f.write(f"{r.uniprot_id},{r.organism},{r.chain_length},{r.n_residues_used},"
                    f"{r.bps_l:.4f},{r.frac_alpha:.3f},{r.frac_beta:.3f},{r.frac_other:.3f},"
                    f"{r.fold_class},{r.mean_plddt:.1f}\n")
    print(f"  Wrote {csv_path}")

    # ── Loops CSV ──
    if loops:
        loop_csv = os.path.join(output_dir, "loops.csv")
        with open(loop_csv, 'w', encoding='utf-8') as f:
            f.write("protein_id,organism,start,end,length,direction,delta_w,torus_path_len,residues\n")
            for l in loops:
                f.write(f"{l.protein_id},{l.organism},{l.start_res},{l.end_res},"
                        f"{l.length},{l.direction},{l.delta_w:.4f},{l.torus_path_len:.4f},"
                        f"{l.residues}\n")
        print(f"  Wrote {loop_csv}")

    # ── Per-organism aggregation ──
    org_data = defaultdict(list)
    for r in valid:
        org_data[r.organism].append(r.bps_l)

    org_means = {}
    for org, vals in org_data.items():
        if len(vals) >= 10:  # need minimum sample per organism
            org_means[org] = {
                'n': len(vals),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'cv': float(np.std(vals) / np.mean(vals) * 100) if np.mean(vals) > 0 else 0,
            }

    org_csv = os.path.join(output_dir, "per_organism_bpsl.csv")
    with open(org_csv, 'w', encoding='utf-8') as f:
        f.write("organism,n_proteins,mean_bpsl,std_bpsl,cv_pct\n")
        for org in sorted(org_means, key=lambda x: org_means[x]['n'], reverse=True):
            d = org_means[org]
            f.write(f"{org},{d['n']},{d['mean']:.4f},{d['std']:.4f},{d['cv']:.1f}\n")
    print(f"  Wrote {org_csv}")

    # ── CORE ANALYSIS ──
    bps_vals = np.array([r.bps_l for r in valid])
    global_mean = float(np.mean(bps_vals))
    global_std = float(np.std(bps_vals))
    global_cv = global_std / global_mean * 100 if global_mean > 0 else 0

    # Cross-organism CV (THE missing paper number)
    if len(org_means) >= 3:
        org_mean_vals = np.array([d['mean'] for d in org_means.values()])
        cross_org_mean = float(np.mean(org_mean_vals))
        cross_org_std = float(np.std(org_mean_vals))
        cross_org_cv = cross_org_std / cross_org_mean * 100 if cross_org_mean > 0 else 0
    else:
        cross_org_mean = cross_org_std = cross_org_cv = 0

    # Fold-class breakdown
    fold_stats = {}
    for fc in ['all-alpha', 'all-beta', 'alpha/beta', 'alpha+beta', 'other']:
        fc_vals = [r.bps_l for r in valid if r.fold_class == fc]
        if fc_vals:
            fold_stats[fc] = {
                'n': len(fc_vals),
                'mean': float(np.mean(fc_vals)),
                'std': float(np.std(fc_vals)),
                'cv': float(np.std(fc_vals) / np.mean(fc_vals) * 100),
            }

    # Three-level decomposition (sample)
    print("  Computing three-level decomposition (sampling 100 structures)...")
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(valid), min(100, len(valid)), replace=False)
    markov_vals = []
    shuffled_vals = []
    real_vals = []
    for idx in sample_indices:
        r = valid[idx]
        try:
            angles = extract_dihedrals(r.filepath, 70.0)
            if len(angles) >= 50:
                m, s = markov_shuffle(angles, W, n_trials=5)
                markov_vals.append(m)
                shuffled_vals.append(s)
                real_vals.append(r.bps_l)
        except Exception:
            pass

    three_level = {}
    if real_vals:
        three_level = {
            'real': float(np.mean(real_vals)),
            'markov': float(np.mean(markov_vals)),
            'shuffled': float(np.mean(shuffled_vals)),
            'mr_ratio': float(np.mean(markov_vals) / np.mean(real_vals)),
            'sr_ratio': float(np.mean(shuffled_vals) / np.mean(real_vals)),
            'n_sampled': len(real_vals),
        }

    # Loop summary
    loop_summary = {}
    if loops:
        short = [l for l in loops if l.length <= 7]
        medium = [l for l in loops if 8 <= l.length <= 10]
        long_ = [l for l in loops if 11 <= l.length <= 15]
        loop_summary = {
            'total': len(loops),
            'short': len(short),
            'medium': len(medium),
            'long': len(long_),
            'n_organisms': len(set(l.organism for l in loops)),
        }

    # ── Write Report ──
    report_path = os.path.join(output_dir, "alphafold_validation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TorusFold AlphaFold Validation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Backend:** {CIF_BACKEND}\n")
        f.write(f"**Structures processed:** {len(results)} total, {len(valid)} valid\n")
        f.write(f"**Organisms:** {len(org_data)} total, {len(org_means)} with ≥10 structures\n\n")

        # W stats
        ws = W.stats()
        f.write("## Superpotential W\n\n")
        f.write(f"- W range: [{ws['W_range'][0]:.2f}, {ws['W_range'][1]:.2f}]\n")
        f.write(f"- Alpha basin depth: {ws['depth_alpha']:.2f}\n")
        f.write(f"- Beta basin depth: {ws['depth_beta']:.2f}\n\n")

        # BPS/L global
        f.write("## 1. BPS/L Conservation\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Mean BPS/L | {global_mean:.3f} |\n")
        f.write(f"| Std | {global_std:.3f} |\n")
        f.write(f"| Per-protein CV | {global_cv:.1f}% |\n")
        f.write(f"| Median | {float(np.median(bps_vals)):.3f} |\n")
        f.write(f"| Range | [{float(np.min(bps_vals)):.3f}, {float(np.max(bps_vals)):.3f}] |\n")
        f.write(f"| N | {len(valid)} |\n\n")

        # ★ Cross-organism CV ★
        f.write("## 2. Cross-Organism Conservation (★ KEY RESULT ★)\n\n")
        if cross_org_cv > 0:
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Number of organisms (≥10 structures) | {len(org_means)} |\n")
            f.write(f"| Mean of organism means | {cross_org_mean:.3f} |\n")
            f.write(f"| Std of organism means | {cross_org_std:.3f} |\n")
            f.write(f"| **Cross-organism CV** | **{cross_org_cv:.1f}%** |\n\n")

            f.write("### Per-Organism Means\n\n")
            f.write("| Organism | N | Mean BPS/L | Std | CV% |\n|---|---|---|---|---|\n")
            for org in sorted(org_means, key=lambda x: org_means[x]['n'], reverse=True):
                d = org_means[org]
                f.write(f"| {org} | {d['n']} | {d['mean']:.3f} | {d['std']:.3f} | {d['cv']:.1f}% |\n")
            f.write("\n")
        else:
            f.write("Insufficient organisms with ≥10 structures for cross-organism analysis.\n\n")

        # Fold-class
        f.write("## 3. Fold-Class Breakdown\n\n")
        f.write("| Fold Class | N | Mean BPS/L | Std | CV% |\n|---|---|---|---|---|\n")
        for fc in ['all-alpha', 'all-beta', 'alpha/beta', 'alpha+beta', 'other']:
            if fc in fold_stats:
                d = fold_stats[fc]
                f.write(f"| {fc} | {d['n']} | {d['mean']:.3f} | {d['std']:.3f} | {d['cv']:.1f}% |\n")
        f.write("\n")

        # Three-level
        f.write("## 4. Three-Level Decomposition\n\n")
        if three_level:
            f.write(f"Computed on {three_level['n_sampled']} sampled structures.\n\n")
            f.write(f"| Level | BPS/L |\n|---|---|\n")
            f.write(f"| Real | {three_level['real']:.3f} |\n")
            f.write(f"| Markov | {three_level['markov']:.3f} |\n")
            f.write(f"| Shuffled | {three_level['shuffled']:.3f} |\n")
            f.write(f"| **M/R ratio** | **{three_level['mr_ratio']:.2f}×** |\n")
            f.write(f"| **S/R ratio** | **{three_level['sr_ratio']:.2f}×** |\n\n")
        else:
            f.write("Not computed.\n\n")

        # Loops
        f.write("## 5. Loop Taxonomy\n\n")
        if loop_summary:
            f.write(f"| Stratum | Loops |\n|---|---|\n")
            f.write(f"| Short (≤7) | {loop_summary['short']} |\n")
            f.write(f"| Medium (8-10) | {loop_summary['medium']} |\n")
            f.write(f"| Long (11-15) | {loop_summary['long']} |\n")
            f.write(f"| Total | {loop_summary['total']} |\n")
            f.write(f"| Organisms | {loop_summary['n_organisms']} |\n\n")
            f.write("Note: DBSCAN clustering on loops should be run separately\n")
            f.write("on the loops.csv output using the existing taxonomy pipeline.\n\n")
        else:
            f.write("No loops extracted.\n\n")

        # Summary box
        f.write("## Validation Summary\n\n")
        f.write("```\n")
        f.write("TORUSFOLD ALPHAFOLD VALIDATION\n")
        f.write("═══════════════════════════════════════════\n")
        f.write(f"Structures:          {len(valid)} valid\n")
        f.write(f"Organisms:           {len(org_means)} (≥10 structures)\n")
        f.write(f"BPS/L mean:          {global_mean:.3f} ± {global_std:.3f}\n")
        f.write(f"Per-protein CV:      {global_cv:.1f}%\n")
        if cross_org_cv > 0:
            f.write(f"Cross-organism CV:   {cross_org_cv:.1f}%  ← KEY NUMBER\n")
        if three_level:
            f.write(f"Three-level (M/R):   {three_level['mr_ratio']:.2f}×\n")
            f.write(f"Three-level (S/R):   {three_level['sr_ratio']:.2f}×\n")
        if loop_summary:
            f.write(f"Loops extracted:     {loop_summary['total']}\n")
        f.write("═══════════════════════════════════════════\n")
        f.write("```\n")

    print(f"  Wrote {report_path}")

    return {
        'global_mean': global_mean,
        'global_cv': global_cv,
        'cross_org_cv': cross_org_cv,
        'n_organisms': len(org_means),
        'three_level': three_level,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TorusFold AlphaFold Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run (serial, 1 worker)
  python alphafold_pipeline.py /data/alphafold/

  # Fast parallel run (8 workers, skip loops for speed)
  python alphafold_pipeline.py /data/alphafold/ -j 8 --no-loops

  # Resume interrupted run
  python alphafold_pipeline.py /data/alphafold/ -j 4 --checkpoint results/checkpoint/

  # Custom output directory
  python alphafold_pipeline.py /data/alphafold/ -o my_results/

  # Quick test on first 100 structures
  python alphafold_pipeline.py /data/alphafold/ --max-structures 100
        """
    )
    parser.add_argument("data_dir", help="Path to AlphaFold data (organism subdirectories)")
    parser.add_argument("-o", "--output", default="results", help="Output directory (default: results)")
    parser.add_argument("-j", "--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--plddt-min", type=float, default=70.0, help="Minimum pLDDT filter (default: 70)")
    parser.add_argument("--min-length", type=int, default=50, help="Minimum chain length (default: 50)")
    parser.add_argument("--no-loops", action="store_true", help="Skip loop extraction (faster)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory for resume")
    parser.add_argument("--max-structures", type=int, default=None, help="Limit total structures (for testing)")
    parser.add_argument("--w-cache", default=None, help="Path to cached superpotential .npz")
    parser.add_argument("--w-sample", type=int, default=2000, help="Structures to sample for W building (default: 2000)")

    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold AlphaFold Pipeline")
    print("=" * 60)
    print(f"  Data:       {args.data_dir}")
    print(f"  Output:     {args.output}")
    print(f"  Backend:    {CIF_BACKEND}")
    print(f"  Workers:    {args.workers}")
    print(f"  pLDDT min:  {args.plddt_min}")
    print(f"  Min length: {args.min_length}")
    print(f"  Loops:      {'yes' if not args.no_loops else 'no'}")
    print()

    # ── Phase 0: Discover ──
    print("Phase 0: Discovering files...")
    organisms = discover_files(args.data_dir)
    total_files = sum(len(f) for f in organisms.values())
    print(f"  Found {total_files} .cif files across {len(organisms)} organisms")

    if total_files == 0:
        print("ERROR: No .cif files found. Check the data directory path.")
        sys.exit(1)

    # Peek at first file for diagnostics
    first_org = list(organisms.keys())[0]
    first_file = organisms[first_org][0]
    fsize = os.path.getsize(first_file)
    print(f"  First file: {first_file}")
    print(f"  File size: {fsize:,} bytes")
    with open(first_file, 'r', errors='replace') as fh:
        head = fh.read(200)
    print(f"  First 200 chars: {repr(head[:200])}")

    # Apply max-structures limit
    if args.max_structures:
        limited = {}
        count = 0
        for org in sorted(organisms):
            remaining = args.max_structures - count
            if remaining <= 0:
                break
            files = organisms[org][:remaining]
            limited[org] = files
            count += len(files)
        organisms = limited
        total_files = sum(len(f) for f in organisms.values())
        print(f"  Limited to {total_files} structures (--max-structures {args.max_structures})")

    # Print organism summary
    print("\n  Top organisms by structure count:")
    for org in sorted(organisms, key=lambda x: len(organisms[x]), reverse=True)[:15]:
        print(f"    {org}: {len(organisms[org])} structures")
    if len(organisms) > 15:
        print(f"    ... and {len(organisms) - 15} more")
    print()

    # ── Phase 1: Build W ──
    print("Phase 1: Building superpotential W...")
    w_cache = args.w_cache or os.path.join(args.output, "superpotential_W.npz")
    os.makedirs(args.output, exist_ok=True)
    W = build_superpotential(organisms, args.w_sample, args.plddt_min, w_cache)
    print()

    # ── Phase 2: Process all ──
    print("Phase 2: Processing all structures...")
    results, loops = process_all(
        organisms, W,
        n_workers=args.workers,
        plddt_min=args.plddt_min,
        min_length=args.min_length,
        compute_loops=not args.no_loops,
        checkpoint_dir=args.checkpoint,
    )
    print()

    # ── Phase 3: Analysis ──
    print("Phase 3: Generating analysis and report...")
    summary = analyze_results(results, loops, args.output, W)
    print()

    # ── Done ──
    print("=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    if summary.get('cross_org_cv', 0) > 0:
        print(f"  ★ Cross-organism CV = {summary['cross_org_cv']:.1f}% ★")
    print(f"  Results in: {args.output}/")
    print()


if __name__ == "__main__":
    main()
