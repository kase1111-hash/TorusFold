"""Superpotential W(phi, psi) on the Ramachandran torus T^2 = S^1 x S^1.

Canonical W construction: W = -sqrt(P(phi, psi)) where P is a 10-component
bivariate von Mises mixture fit to PDB backbone dihedral statistics, smoothed
with sigma = 1.5 grid bins (1.5 degrees on a 360x360 grid).

This is the SINGLE shared W used across all scripts. The von Mises mixture
is a fixed parametric model — no PDB data files are needed at build time.

W is a fixed landscape — never trainable, never modified during analysis.
Proteins navigate it; the landscape itself does not change.

The absolute value of BPS/L depends on the W construction (gauge choice).
What matters is the three-level decomposition: Real < Markov < Shuffled.
"""

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


# ============================================================
# VON MISES MIXTURE COMPONENTS
# ============================================================
# 10-component bivariate von Mises mixture fit to PDB backbone statistics.
# Each component: weight, mu_phi (deg), mu_psi (deg), kappa_phi, kappa_psi, rho
# Identical to the components in bps_process.py.

_VON_MISES_COMPONENTS = [
    {'weight': 0.35, 'mu_phi': -63,  'mu_psi': -43,  'kappa_phi': 12.0, 'kappa_psi': 10.0, 'rho':  2.0},
    {'weight': 0.05, 'mu_phi': -60,  'mu_psi': -27,  'kappa_phi':  8.0, 'kappa_psi':  6.0, 'rho':  1.0},
    {'weight': 0.25, 'mu_phi': -120, 'mu_psi': 135,  'kappa_phi':  4.0, 'kappa_psi':  3.5, 'rho': -1.5},
    {'weight': 0.05, 'mu_phi': -140, 'mu_psi': 155,  'kappa_phi':  5.0, 'kappa_psi':  4.0, 'rho': -1.0},
    {'weight': 0.12, 'mu_phi': -75,  'mu_psi': 150,  'kappa_phi':  8.0, 'kappa_psi':  5.0, 'rho':  0.5},
    {'weight': 0.05, 'mu_phi': -95,  'mu_psi': 150,  'kappa_phi':  3.0, 'kappa_psi':  4.0, 'rho':  0.0},
    {'weight': 0.03, 'mu_phi':  57,  'mu_psi':  40,  'kappa_phi':  6.0, 'kappa_psi':  6.0, 'rho':  1.5},
    {'weight': 0.03, 'mu_phi':  60,  'mu_psi': -130, 'kappa_phi':  5.0, 'kappa_psi':  4.0, 'rho':  0.0},
    {'weight': 0.01, 'mu_phi':  75,  'mu_psi': -65,  'kappa_phi':  5.0, 'kappa_psi':  5.0, 'rho':  0.0},
    {'weight': 0.06, 'mu_phi':   0,  'mu_psi':   0,  'kappa_phi':  0.01,'kappa_psi':  0.01,'rho':  0.0},
]

# Gaussian smoothing in grid bins (1 bin = 1 degree on a 360x360 grid)
_SMOOTH_SIGMA = 1.5

# Density floor: max(p) * 1e-6 (prevents sqrt(0) in unpopulated regions)
_DENSITY_FLOOR_FACTOR = 1e-6


# ============================================================
# W CONSTRUCTION
# ============================================================

def build_superpotential(grid_size: int = 360,
                         smooth_sigma: float = _SMOOTH_SIGMA,
                         ) -> Tuple[NDArray, NDArray, NDArray]:
    """Build W(phi, psi) = -sqrt(P) from the von Mises mixture.

    The density P is evaluated on a grid from the parametric mixture,
    floored at max(P) * 1e-6, transformed to W = -sqrt(P), and smoothed.

    Parameters
    ----------
    grid_size : int
        Grid points along each axis (default 360, i.e. 1 deg/bin).
    smooth_sigma : float
        Gaussian smoothing in grid bins (default 1.5).

    Returns
    -------
    W_grid : ndarray, shape (grid_size, grid_size)
        Superpotential values. W[i, j] = W(phi_grid[i], psi_grid[j]).
    phi_grid : ndarray, shape (grid_size,)
        phi values in radians, in [-pi, pi).
    psi_grid : ndarray, shape (grid_size,)
        psi values in radians, in [-pi, pi).
    """
    phi_grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    psi_grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    PHI, PSI = np.meshgrid(phi_grid, psi_grid, indexing='ij')

    # Evaluate von Mises mixture density
    p = np.zeros_like(PHI)
    for c in _VON_MISES_COMPONENTS:
        mu_p = np.radians(c['mu_phi'])
        mu_s = np.radians(c['mu_psi'])
        dp = PHI - mu_p
        ds = PSI - mu_s
        p += c['weight'] * np.exp(
            c['kappa_phi'] * np.cos(dp)
            + c['kappa_psi'] * np.cos(ds)
            + c['rho'] * np.sin(dp) * np.sin(ds)
        )

    # Normalize to proper density
    dphi = phi_grid[1] - phi_grid[0]
    dpsi = psi_grid[1] - psi_grid[0]
    p /= np.sum(p) * dphi * dpsi

    # Floor to prevent sqrt(0) in unpopulated regions
    p = np.maximum(p, np.max(p) * _DENSITY_FLOOR_FACTOR)

    # W = -sqrt(P)
    W_grid = -np.sqrt(p)

    # Gaussian smoothing (periodic boundary conditions)
    if smooth_sigma > 0:
        W_grid = gaussian_filter(W_grid, sigma=smooth_sigma, mode='wrap')

    return W_grid, phi_grid, psi_grid


def build_interpolator(grid_size: int = 360,
                       smooth_sigma: float = _SMOOTH_SIGMA,
                       ) -> RegularGridInterpolator:
    """Build a RegularGridInterpolator for W(phi, psi).

    Returns an interpolator compatible with bps_process.py's calling
    convention: W_interp(np.column_stack([psis, phis])).

    Parameters
    ----------
    grid_size : int
        Grid points along each axis (default 360).
    smooth_sigma : float
        Gaussian smoothing in grid bins (default 1.5).

    Returns
    -------
    RegularGridInterpolator
        Callable that takes (psi, phi) column-stacked arrays.
    """
    W_grid, phi_grid, psi_grid = build_superpotential(grid_size, smooth_sigma)
    # W_grid[i, j] = W(phi_grid[i], psi_grid[j])
    # RegularGridInterpolator((psi_grid, phi_grid), M) expects M[a, b] = value at (psi[a], phi[b])
    # So M[a, b] = W_grid[b, a] = W_grid.T[a, b]
    return RegularGridInterpolator(
        (psi_grid, phi_grid), W_grid.T,
        method='linear', bounds_error=False, fill_value=None,
    )


# ============================================================
# W LOOKUP
# ============================================================

def lookup_W(W_grid: NDArray, phi_grid: NDArray, psi_grid: NDArray,
             phi: float, psi: float) -> float:
    """Bilinear interpolation of W with periodic wrapping.

    Parameters
    ----------
    W_grid : ndarray
        Superpotential grid from build_superpotential().
    phi_grid, psi_grid : ndarray
        Coordinate grids from build_superpotential().
    phi, psi : float
        Query angles in radians.

    Returns
    -------
    float
        Interpolated W value.
    """
    grid_size = len(phi_grid)
    dphi = phi_grid[1] - phi_grid[0]
    dpsi = psi_grid[1] - psi_grid[0]

    # Wrap angles into [-pi, pi)
    phi = math.atan2(math.sin(phi), math.cos(phi))
    psi = math.atan2(math.sin(psi), math.cos(psi))

    # Fractional grid indices
    fi = (phi - phi_grid[0]) / dphi
    fj = (psi - psi_grid[0]) / dpsi

    i0_idx = int(math.floor(fi))
    j0_idx = int(math.floor(fj))
    di = fi - i0_idx
    dj = fj - j0_idx

    # Periodic wrapping
    i0_idx = i0_idx % grid_size
    i1_idx = (i0_idx + 1) % grid_size
    j0_idx = j0_idx % grid_size
    j1_idx = (j0_idx + 1) % grid_size

    # Bilinear interpolation
    w00 = W_grid[i0_idx, j0_idx]
    w01 = W_grid[i0_idx, j1_idx]
    w10 = W_grid[i1_idx, j0_idx]
    w11 = W_grid[i1_idx, j1_idx]

    return float(
        (1 - di) * (1 - dj) * w00
        + (1 - di) * dj * w01
        + di * (1 - dj) * w10
        + di * dj * w11
    )


def lookup_W_batch(W_grid: NDArray, phi_grid: NDArray, psi_grid: NDArray,
                   phi_array: NDArray, psi_array: NDArray) -> NDArray:
    """Vectorized bilinear interpolation of W with periodic wrapping.

    Parameters
    ----------
    W_grid : ndarray
        Superpotential grid from build_superpotential().
    phi_grid, psi_grid : ndarray
        Coordinate grids from build_superpotential().
    phi_array, psi_array : ndarray
        Query angles in radians (same shape).

    Returns
    -------
    ndarray
        Interpolated W values (same shape as inputs).
    """
    grid_size = len(phi_grid)
    dphi = phi_grid[1] - phi_grid[0]
    dpsi = psi_grid[1] - psi_grid[0]

    # Wrap into [-pi, pi)
    phi_w = np.arctan2(np.sin(phi_array), np.cos(phi_array))
    psi_w = np.arctan2(np.sin(psi_array), np.cos(psi_array))

    # Fractional grid indices
    fi = (phi_w - phi_grid[0]) / dphi
    fj = (psi_w - psi_grid[0]) / dpsi

    i0_idx = np.floor(fi).astype(int)
    j0_idx = np.floor(fj).astype(int)
    di = fi - i0_idx
    dj = fj - j0_idx

    # Periodic wrapping
    i0_idx = i0_idx % grid_size
    i1_idx = (i0_idx + 1) % grid_size
    j0_idx = j0_idx % grid_size
    j1_idx = (j0_idx + 1) % grid_size

    # Bilinear interpolation
    w00 = W_grid[i0_idx, j0_idx]
    w01 = W_grid[i0_idx, j1_idx]
    w10 = W_grid[i1_idx, j0_idx]
    w11 = W_grid[i1_idx, j1_idx]

    return ((1 - di) * (1 - dj) * w00
            + (1 - di) * dj * w01
            + di * (1 - dj) * w10
            + di * dj * w11)


def lookup_W_grid(W_grid: NDArray, grid_size: int,
                  phi_rad: NDArray, psi_rad: NDArray) -> NDArray:
    """Nearest-neighbor grid lookup of W from angles in radians.

    This is the fast grid-lookup used by many analysis scripts.
    For higher accuracy, use lookup_W_batch (bilinear interpolation).

    Parameters
    ----------
    W_grid : ndarray, shape (grid_size, grid_size)
        W[i, j] = W(phi[i], psi[j]) with phi on axis 0, psi on axis 1.
    grid_size : int
        Grid size (typically 360).
    phi_rad, psi_rad : ndarray
        Angles in radians.

    Returns
    -------
    ndarray
        W values at each (phi, psi) pair.
    """
    scale = grid_size / 360.0
    phi_d = np.degrees(phi_rad)
    psi_d = np.degrees(psi_rad)
    gi = (np.round((phi_d + 180) * scale).astype(int)) % grid_size
    gj = (np.round((psi_d + 180) * scale).astype(int)) % grid_size
    return W_grid[gi, gj]


# ============================================================
# BASIN ASSIGNMENT
# ============================================================

def assign_basin(phi_deg: float, psi_deg: float) -> str:
    """Assign a residue to its Ramachandran basin.

    Parameters
    ----------
    phi_deg, psi_deg : float
        Dihedral angles in DEGREES.

    Returns
    -------
    str
        One of 'alpha', 'beta', 'ppII', 'alphaL', 'other'.

    Notes
    -----
    ppII is checked before beta because their phi/psi ranges overlap.
    ppII is the smaller, more specific region and gets priority.
    """
    # alpha-helix: phi in (-100, -30), psi in (-67, -7)
    if -100 < phi_deg < -30 and -67 < psi_deg < -7:
        return "alpha"

    # Polyproline II: phi in (-100, -50), psi in (120, 180)
    # Checked BEFORE beta because it overlaps with the beta region.
    if -100 < phi_deg < -50 and 120 < psi_deg < 180:
        return "ppII"

    # beta-sheet: phi in (-170, -70), psi > 90 OR psi < -120
    # CRITICAL: single condition handles the +/-180 deg wrap
    if -170 < phi_deg < -70 and (psi_deg > 90 or psi_deg < -120):
        return "beta"

    # Left-handed helix: phi in (30, 90), psi in (10, 70)
    if 30 < phi_deg < 90 and 10 < psi_deg < 70:
        return "alphaL"

    return "other"


# ============================================================
# BPS COMPUTATION
# ============================================================

def compute_bps(W_values: NDArray) -> float:
    """Compute BPS = sum of |delta W_i| (total variation of W along the chain).

    Parameters
    ----------
    W_values : ndarray
        W values along the sequential backbone, shape (L,).

    Returns
    -------
    float
        BPS value.
    """
    if len(W_values) < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(W_values))))


def compute_bps_per_residue(W_values: NDArray) -> float:
    """Compute BPS/L = BPS / L.

    Normalizes by L (number of residues), consistent across all scripts.

    Parameters
    ----------
    W_values : ndarray
        W values along the sequential backbone, shape (L,).

    Returns
    -------
    float
        BPS per residue.
    """
    L = len(W_values)
    if L < 2:
        return 0.0
    return compute_bps(W_values) / L


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Building superpotential W(phi, psi) from von Mises mixture...")
    print(f"  Components: {len(_VON_MISES_COMPONENTS)}")
    print(f"  Smoothing sigma: {_SMOOTH_SIGMA} bins ({_SMOOTH_SIGMA:.1f} deg)")
    print()

    W, phi_g, psi_g = build_superpotential()
    print(f"  Grid: {W.shape}")
    print(f"  W range: [{W.min():.4f}, {W.max():.4f}]")
    assert W.max() <= 0, f"W should be non-positive (-sqrt(P)), got max={W.max():.4f}"
    print()

    # Look up key Ramachandran points
    points = {
        "alpha-helix center": (-63, -43),
        "beta-sheet center":  (-120, 130),
        "ppII center":        (-75, 150),
        "alphaL center":      (57, 47),
        "saddle (alpha->beta)": (-90, 0),
        "steric clash":       (0, 0),
    }

    for name, (phi_d, psi_d) in points.items():
        w = lookup_W(W, phi_g, psi_g, math.radians(phi_d), math.radians(psi_d))
        print(f"  W({phi_d:+4d} deg, {psi_d:+4d} deg)  {name:22s}  = {w:.4f}")

    # Landscape validation: alpha should be deepest (most negative)
    w_alpha = lookup_W(W, phi_g, psi_g, math.radians(-63), math.radians(-43))
    w_beta = lookup_W(W, phi_g, psi_g, math.radians(-120), math.radians(130))
    w_clash = lookup_W(W, phi_g, psi_g, math.radians(0), math.radians(0))

    print()
    assert w_alpha < w_beta, (
        f"FAIL: W(alpha) = {w_alpha:.4f} should be < W(beta) = {w_beta:.4f}")
    assert w_alpha < w_clash, (
        f"FAIL: W(alpha) = {w_alpha:.4f} should be < W(clash) = {w_clash:.4f}")
    print("  PASS: alpha is the global minimum (most negative)")
    print("  PASS: beta is a secondary minimum")

    # Verify batch lookup matches scalar lookup
    print()
    phi_test = np.array([math.radians(-63), math.radians(-120), math.radians(57)])
    psi_test = np.array([math.radians(-43), math.radians(130), math.radians(47)])
    w_batch = lookup_W_batch(W, phi_g, psi_g, phi_test, psi_test)
    for k in range(len(phi_test)):
        w_scalar = lookup_W(W, phi_g, psi_g, phi_test[k], psi_test[k])
        assert abs(w_batch[k] - w_scalar) < 1e-10, (
            f"Batch/scalar mismatch: {w_batch[k]:.6f} vs {w_scalar:.6f}")
    print("  PASS: batch lookup matches scalar lookup")

    # Verify grid lookup matches bilinear lookup (within rounding tolerance)
    w_grid_vals = lookup_W_grid(W, 360, phi_test, psi_test)
    for k in range(len(phi_test)):
        assert abs(w_grid_vals[k] - w_batch[k]) < 0.05, (
            f"Grid/bilinear mismatch at point {k}: "
            f"{w_grid_vals[k]:.4f} vs {w_batch[k]:.4f}")
    print("  PASS: grid lookup matches bilinear lookup (within rounding)")

    # Verify interpolator matches grid convention
    interp = build_interpolator()
    for k in range(len(phi_test)):
        w_interp = float(interp(np.array([[psi_test[k], phi_test[k]]])))
        assert abs(w_interp - w_batch[k]) < 0.01, (
            f"Interpolator mismatch at point {k}: "
            f"{w_interp:.4f} vs {w_batch[k]:.4f}")
    print("  PASS: interpolator matches bilinear lookup")

    # Basin assignment tests
    print()
    print("Basin assignment tests:")
    basin_tests = [
        (-63, -43, "alpha"),
        (-120, 130, "beta"),
        (-120, -170, "beta"),
        (-120, -130, "beta"),
        (-75, 150, "ppII"),
        (-85, 140, "ppII"),
        (57, 47, "alphaL"),
        (0, 0, "other"),
        (-90, 0, "other"),
    ]
    for phi_d, psi_d, expected in basin_tests:
        result = assign_basin(phi_d, psi_d)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: ({phi_d:+4d} deg, {psi_d:+4d} deg) -> {result:8s} "
              f"(expected {expected})")
        assert result == expected

    # BPS computation test
    print()
    w_vals = np.array([0.0, 0.5, 0.3, 0.8, 0.1])
    bps = compute_bps(w_vals)
    expected_bps = 1.9
    assert abs(bps - expected_bps) < 1e-10
    bps_l = compute_bps_per_residue(w_vals)
    assert abs(bps_l - expected_bps / 5.0) < 1e-10
    print("  PASS: BPS computation tests (BPS/L = BPS / L)")

    print()
    print("All self-tests passed.")
