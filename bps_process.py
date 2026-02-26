#!/usr/bin/env python3
"""
BPS PROCESSOR v2 — Compute BPS from cached AlphaFold CIF files
================================================================
Reads CIF files from alphafold_cache/, computes BPS + pLDDT + SS + Rg + CO,
writes everything to SQLite database. CPU-bound only — no network calls.

CRITICAL: Before processing, this script DETERMINES the correct phi sign
convention by testing both +raw and -raw on actual cached CIF files.
It will HARD FAIL if it cannot verify the sign convention. No silent passes.

Usage:
  python bps_process.py --mode all           # process all cached organisms
  python bps_process.py --mode organism --organism human
  python bps_process.py --mode watch         # continuously poll for new CIFs
  python bps_process.py --mode watch --poll 15   # poll every 15 seconds
  python bps_process.py --mode status        # show processing progress
  python bps_process.py --mode summary       # cross-proteome comparison
  python bps_process.py --mode export        # dump CSVs from database
  python bps_process.py --mode validate      # Plaxco23 validation
  python bps_process.py --mode all --reset   # delete DB and recompute
  python bps_process.py --mode phi-check     # just test phi sign and exit

Concurrent operation:
  All three pipeline scripts can run simultaneously:
    Terminal 1: python bps_download.py --mode watch
    Terminal 2: python bps_process.py --mode watch
    Terminal 3: python compile_results.py --mode watch
  SQLite WAL mode + busy_timeout ensures safe concurrent access.
"""

import sys
import time
import math
import sqlite3
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import pearsonr, linregress
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path("./alphafold_bps_results")
CACHE_DIR  = Path("./alphafold_cache")
DB_PATH    = OUTPUT_DIR / "bps_database.db"
LOG_PATH   = OUTPUT_DIR / "pipeline.log"

BATCH_SIZE = 200  # larger batches since no network wait

# This gets set by determine_phi_sign() before any processing.
# +1 means phi = +raw_dihedral, -1 means phi = -raw_dihedral.
PHI_SIGN = None


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ============================================================
# DATABASE
# ============================================================

def init_db():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=10000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS proteins (
            uniprot_id TEXT NOT NULL,
            organism   TEXT NOT NULL,
            L          INTEGER,
            bps_energy REAL,
            bps_norm   REAL,
            n_transitions INTEGER,
            n_kinks    INTEGER,
            Q_phi      REAL,
            Q_psi      REAL,
            Q_mag      REAL,
            Q_norm     REAL,
            plddt_mean   REAL,
            plddt_median REAL,
            plddt_below50 REAL,
            pct_helix  REAL,
            pct_sheet  REAL,
            pct_coil   REAL,
            rg         REAL,
            contact_order REAL,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (uniprot_id, organism)
        );

        CREATE TABLE IF NOT EXISTS failures (
            uniprot_id TEXT NOT NULL,
            organism   TEXT NOT NULL,
            error_type TEXT NOT NULL,
            error_msg  TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (uniprot_id, organism)
        );

        CREATE TABLE IF NOT EXISTS proteomes (
            organism    TEXT PRIMARY KEY,
            name        TEXT,
            taxid       TEXT,
            n_ids       INTEGER DEFAULT 0,
            n_done      INTEGER DEFAULT 0,
            n_failed    INTEGER DEFAULT 0,
            bps_mean    REAL,
            bps_std     REAL,
            status      TEXT DEFAULT 'pending',
            started_at  TEXT,
            finished_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_proteins_org ON proteins(organism);
        CREATE INDEX IF NOT EXISTS idx_failures_org ON failures(organism);
    """)

    # Schema migration for older databases
    new_columns = [
        ("plddt_mean", "REAL"), ("plddt_median", "REAL"), ("plddt_below50", "REAL"),
        ("pct_helix", "REAL"), ("pct_sheet", "REAL"), ("pct_coil", "REAL"),
        ("rg", "REAL"), ("contact_order", "REAL"),
        ("af_version", "INTEGER"),
    ]
    for col, dtype in new_columns:
        try:
            conn.execute(f"ALTER TABLE proteins ADD COLUMN {col} {dtype}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    return conn


def db_get_done_ids(conn, organism):
    rows = conn.execute(
        "SELECT uniprot_id FROM proteins WHERE organism = ?", (organism,)
    ).fetchall()
    return {r[0] for r in rows}


def db_get_failure_ids(conn, organism):
    rows = conn.execute(
        "SELECT uniprot_id FROM failures WHERE organism = ?", (organism,)
    ).fetchall()
    return {r[0] for r in rows}


def db_counts(conn, organism):
    done = conn.execute(
        "SELECT COUNT(*) FROM proteins WHERE organism = ?", (organism,)).fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(*) FROM failures WHERE organism = ?", (organism,)).fetchone()[0]
    return done, failed


def db_get_results(conn, organism):
    rows = conn.execute("""
        SELECT uniprot_id, L, bps_energy, bps_norm, n_transitions,
               n_kinks, Q_phi, Q_psi, Q_mag, Q_norm,
               plddt_mean, plddt_median, plddt_below50,
               pct_helix, pct_sheet, pct_coil, rg, contact_order
        FROM proteins WHERE organism = ?
    """, (organism,)).fetchall()
    return [{'uniprot_id': r[0], 'L': r[1], 'bps_energy': r[2], 'bps_norm': r[3],
             'n_transitions': r[4], 'n_kinks': r[5], 'Q_phi': r[6], 'Q_psi': r[7],
             'Q_mag': r[8], 'Q_norm': r[9],
             'plddt_mean': r[10], 'plddt_median': r[11], 'plddt_below50': r[12],
             'pct_helix': r[13], 'pct_sheet': r[14], 'pct_coil': r[15],
             'rg': r[16], 'contact_order': r[17]} for r in rows]


# ============================================================
# PROTEOME REGISTRY
# ============================================================

PROTEOMES = {
    "ecoli":       {"taxid": "83333",  "name": "Escherichia coli K12",       "count": "~4.4K"},
    "yeast":       {"taxid": "559292", "name": "Saccharomyces cerevisiae",   "count": "~6.0K"},
    "fission_yeast":{"taxid":"284812", "name": "Schizosaccharomyces pombe",  "count": "~5.1K"},
    "worm":        {"taxid": "6239",   "name": "Caenorhabditis elegans",     "count": "~20K"},
    "fly":         {"taxid": "7227",   "name": "Drosophila melanogaster",    "count": "~13K"},
    "zebrafish":   {"taxid": "7955",   "name": "Danio rerio",               "count": "~26K"},
    "mouse":       {"taxid": "10090",  "name": "Mus musculus",              "count": "~22K"},
    "rat":         {"taxid": "10116",  "name": "Rattus norvegicus",          "count": "~21K"},
    "human":       {"taxid": "9606",   "name": "Homo sapiens",              "count": "~20K"},
    "arabidopsis": {"taxid": "3702",   "name": "Arabidopsis thaliana",       "count": "~27K"},
    "rice":        {"taxid": "39947",  "name": "Oryza sativa",              "count": "~43K"},
    "soybean":     {"taxid": "3847",   "name": "Glycine max",               "count": "~56K"},
    "maize":       {"taxid": "4577",   "name": "Zea mays",                  "count": "~40K"},
    "chicken":     {"taxid": "9031",   "name": "Gallus gallus",             "count": "~15K"},
    "pig":         {"taxid": "9823",   "name": "Sus scrofa",                "count": "~22K"},
    "cow":         {"taxid": "9913",   "name": "Bos taurus",                "count": "~24K"},
    "dog":         {"taxid": "9615",   "name": "Canis lupus familiaris",    "count": "~20K"},
    "cat":         {"taxid": "9685",   "name": "Felis catus",               "count": "~19K"},
    "gorilla":     {"taxid": "9595",   "name": "Gorilla gorilla",           "count": "~21K"},
    "chimp":       {"taxid": "9598",   "name": "Pan troglodytes",           "count": "~19K"},
    "mosquito":    {"taxid": "7159",   "name": "Aedes aegypti",             "count": "~16K"},
    "honeybee":    {"taxid": "7460",   "name": "Apis mellifera",            "count": "~10K"},
    "tuberculosis":{"taxid": "83332",  "name": "Mycobacterium tuberculosis","count": "~4.0K"},
    "staph":       {"taxid": "93061",  "name": "Staphylococcus aureus",     "count": "~2.9K"},
    "malaria":     {"taxid": "36329",  "name": "Plasmodium falciparum",     "count": "~5.4K"},
    "leishmania":  {"taxid": "5671",   "name": "Leishmania infantum",       "count": "~8.0K"},
    "trypanosoma": {"taxid": "185431", "name": "Trypanosoma cruzi",         "count": "~19K"},
    "bacillus":    {"taxid": "224308", "name": "Bacillus subtilis",         "count": "~4.3K"},
    "pseudomonas": {"taxid": "208964", "name": "Pseudomonas aeruginosa",   "count": "~5.6K"},
    "salmonella":  {"taxid": "99287",  "name": "Salmonella typhimurium",    "count": "~4.5K"},
    "campylobacter":{"taxid":"192222", "name": "Campylobacter jejuni",      "count": "~1.6K"},
    "helicobacter":{"taxid": "85962",  "name": "Helicobacter pylori",       "count": "~1.6K"},
    "dictyostelium":{"taxid":"44689",  "name": "Dictyostelium discoideum",  "count": "~12K"},
    "candida":     {"taxid": "237561", "name": "Candida albicans",          "count": "~6.0K"},
}

# Plaxco 23 validation set
PLAXCO_PROTEINS = [
    ("P00648", "1SRL", "Sarcin",              -3.20, 0.080, 110),
    ("P56849", "1APS", "AcP",                 -1.88, 0.193, 98),
    ("P0A6X3", "1HKS", "Hpr",                 -2.42, 0.120, 85),
    ("P00698", "1HEL", "HEWL",                -1.18, 0.169, 129),
    ("P02489", "3HMZ", "alpha-A crystallin",   0.80, 0.116, 173),
    ("P00636", "2RN2", "Barnase",             -3.31, 0.131, 110),
    ("P00720", "2LZM", "T4 Lysozyme",         0.20, 0.130, 164),
    ("P23928", "1MJC", "alphaB crystallin",   -0.50, 0.126, 175),
    ("P0AEH5", "2CI2", "CI2",                 -4.90, 0.070, 64),
    ("P06654", "1UBQ", "Ubiquitin",           -3.48, 0.118, 76),
    ("P01112", "5P21", "p21 Ras",             -0.80, 0.119, 166),
    ("P62937", "2CYP", "CypA",               -2.80, 0.119, 165),
    ("P23229", "1TIT", "Titin I27",           -1.30, 0.151, 89),
    ("P0ABP8", "1DPS", "Dps",                  0.00, 0.080, 167),
    ("P04637", "2OCJ", "p53",                 -0.50, 0.110, 94),
    ("P05067", "1AAP", "APPI",                -5.40, 0.081, 56),
    ("P00974", "5PTI", "BPTI",               -5.80, 0.082, 58),
    ("P01133", "1EGF", "EGF",                 -5.00, 0.084, 53),
    ("P62993", "1SHG", "SH3",                 -1.20, 0.108, 57),
    ("P00178", "1YCC", "Cyt c",               -3.40, 0.088, 108),
    ("P0CY58", "1AON", "GroEL",                1.80, 0.057, 524),
    ("P00925", "3ENL", "Enolase",              0.80, 0.075, 432),
    ("P00517", "1CDK", "PKA",                  0.50, 0.088, 350),
]


# ============================================================
# SUPERPOTENTIAL
# ============================================================

def build_superpotential():
    COMPONENTS = [
        {'weight': 0.35, 'mu_phi': -63, 'mu_psi': -43, 'kappa_phi': 12.0, 'kappa_psi': 10.0, 'rho': 2.0},
        {'weight': 0.05, 'mu_phi': -60, 'mu_psi': -27, 'kappa_phi': 8.0,  'kappa_psi': 6.0,  'rho': 1.0},
        {'weight': 0.25, 'mu_phi': -120,'mu_psi': 135, 'kappa_phi': 4.0,  'kappa_psi': 3.5,  'rho': -1.5},
        {'weight': 0.05, 'mu_phi': -140,'mu_psi': 155, 'kappa_phi': 5.0,  'kappa_psi': 4.0,  'rho': -1.0},
        {'weight': 0.12, 'mu_phi': -75, 'mu_psi': 150, 'kappa_phi': 8.0,  'kappa_psi': 5.0,  'rho': 0.5},
        {'weight': 0.05, 'mu_phi': -95, 'mu_psi': 150, 'kappa_phi': 3.0,  'kappa_psi': 4.0,  'rho': 0.0},
        {'weight': 0.03, 'mu_phi': 57,  'mu_psi': 40,  'kappa_phi': 6.0,  'kappa_psi': 6.0,  'rho': 1.5},
        {'weight': 0.03, 'mu_phi': 60,  'mu_psi': -130,'kappa_phi': 5.0,  'kappa_psi': 4.0,  'rho': 0.0},
        {'weight': 0.01, 'mu_phi': 75,  'mu_psi': -65, 'kappa_phi': 5.0,  'kappa_psi': 5.0,  'rho': 0.0},
        {'weight': 0.06, 'mu_phi': 0,   'mu_psi': 0,   'kappa_phi': 0.01, 'kappa_psi': 0.01, 'rho': 0.0},
    ]
    N = 360
    phi_grid = np.linspace(-np.pi, np.pi, N, endpoint=False)
    psi_grid = np.linspace(-np.pi, np.pi, N, endpoint=False)
    PHI, PSI = np.meshgrid(phi_grid, psi_grid)
    p = np.zeros_like(PHI)
    for c in COMPONENTS:
        mu_p, mu_s = np.radians(c['mu_phi']), np.radians(c['mu_psi'])
        dp, ds = PHI - mu_p, PSI - mu_s
        p += c['weight'] * np.exp(c['kappa_phi']*np.cos(dp) + c['kappa_psi']*np.cos(ds) + c['rho']*np.sin(dp)*np.sin(ds))
    p /= np.sum(p) * (phi_grid[1]-phi_grid[0]) * (psi_grid[1]-psi_grid[0])
    p = np.maximum(p, np.max(p) * 1e-6)
    # W = -sqrt(P): primary superpotential form (compressed dynamic range).
    # Alternative: W = -ln(P + eps) used in bps/superpotential.py and analysis scripts.
    # Three-level decomposition is transform-invariant across W choices.
    W = -np.sqrt(p)
    W = gaussian_filter(W, sigma=1.5)
    return RegularGridInterpolator((psi_grid, phi_grid), W, method='linear', bounds_error=False, fill_value=None)


# ============================================================
# CIF PARSER + DIHEDRALS
# ============================================================

def _parse_atoms_from_cif(filepath):
    residues = {}
    with open(filepath, 'r') as f:
        col_map = {}
        in_header = False
        col_idx = 0
        for line in f:
            ls = line.strip()
            if ls.startswith('_atom_site.'):
                col_map[ls.split('.')[1].strip()] = col_idx
                col_idx += 1
                in_header = True
                continue
            if in_header and not ls.startswith('_atom_site.'):
                in_header = False
            if not col_map:
                continue
            if not (ls.startswith('ATOM') or ls.startswith('HETATM')):
                if col_map and ls in ('', '#'):
                    break
                continue
            parts = ls.split()
            try:
                if parts[col_map.get('group_PDB', 0)] != 'ATOM':
                    continue
                atom_name = parts[col_map.get('label_atom_id', 3)]
                if atom_name not in ('N', 'CA', 'C'):
                    continue
                alt = parts[col_map.get('label_alt_id', 4)] if 'label_alt_id' in col_map else '.'
                if alt not in ('.', 'A', '?'):
                    continue
                resnum = int(parts[col_map.get('label_seq_id', 8)])
                x = float(parts[col_map.get('Cartn_x', 10)])
                y = float(parts[col_map.get('Cartn_y', 11)])
                z = float(parts[col_map.get('Cartn_z', 12)])
                bf = float(parts[col_map.get('B_iso_or_equiv', 14)])
                if resnum not in residues:
                    residues[resnum] = {'bfactor': bf}
                residues[resnum][atom_name] = np.array([x, y, z])
            except (ValueError, IndexError, KeyError):
                continue
    result = []
    for rn in sorted(residues.keys()):
        r = residues[rn]
        if 'N' in r and 'CA' in r and 'C' in r:
            result.append(r)
    return result


def _calc_dihedral(p0, p1, p2, p3):
    """Raw dihedral angle in radians. Sign convention depends on
    atom ordering — caller must apply PHI_SIGN correction."""
    b1, b2, b3 = p1-p0, p2-p1, p3-p2
    nb2 = np.linalg.norm(b2)
    if nb2 < 1e-10:
        return None
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    nn1, nn2 = np.linalg.norm(n1), np.linalg.norm(n2)
    if nn1 < 1e-10 or nn2 < 1e-10:
        return None
    n1 /= nn1; n2 /= nn2
    m1 = np.cross(n1, b2 / nb2)
    return math.atan2(np.dot(m1, n2), np.dot(n1, n2))


# ============================================================
# PHI SIGN DETERMINATION — the critical piece
# ============================================================

def _find_any_cached_cif():
    """Find ANY valid CIF file in the cache. Returns path or None."""
    if not CACHE_DIR.exists():
        return None
    # Try organism directories first
    for org_dir in sorted(CACHE_DIR.iterdir()):
        if not org_dir.is_dir():
            continue
        for cif in org_dir.glob("AF-*-F1-model_v*.cif"):
            if cif.stat().st_size > 1000:  # needs to be a real structure
                return cif
    # Try top-level cache
    for cif in CACHE_DIR.glob("AF-*-F1-model_v*.cif"):
        if cif.stat().st_size > 1000:
            return cif
    return None


def _find_multiple_cached_cifs(n=10):
    """Find up to n CIF files from different parts of the cache."""
    found = []
    if not CACHE_DIR.exists():
        return found
    for org_dir in sorted(CACHE_DIR.iterdir()):
        if not org_dir.is_dir():
            continue
        for cif in org_dir.glob("AF-*-F1-model_v*.cif"):
            if cif.stat().st_size > 1000:
                found.append(cif)
                if len(found) >= n:
                    return found
                break  # one per organism
    return found


def _test_phi_sign_on_file(cif_path, sign):
    """Parse a CIF, compute phi with given sign, return (n_alpha, n_flipped, n_total).
    Alpha region: phi in [-100, -20] (where helices live in IUPAC convention).
    Flipped region: phi in [+20, +100] (where helices would appear with wrong sign)."""
    residues = _parse_atoms_from_cif(cif_path)
    n = len(residues)
    if n < 30:
        return 0, 0, 0

    phis_deg = []
    for i in range(1, n):
        raw = _calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                             residues[i]['CA'], residues[i]['C'])
        if raw is not None:
            phis_deg.append(math.degrees(sign * raw))

    if not phis_deg:
        return 0, 0, 0

    arr = np.array(phis_deg)
    n_alpha = int(np.sum((-100 < arr) & (arr < -20)))
    n_flip  = int(np.sum((20 < arr) & (arr < 100)))
    return n_alpha, n_flip, len(arr)


def determine_phi_sign():
    """Determine the correct phi sign by testing both +1 and -1 on cached CIFs.
    Returns +1 or -1. HARD FAILS if no CIFs found or sign is ambiguous.

    Logic: Real proteins have ~30-40% of residues in the alpha-helix phi region
    [-100, -20]. If +1*raw puts them there, use +1. If -1*raw puts them there,
    use -1. Test on multiple proteins to be sure."""

    global PHI_SIGN

    logging.info("=" * 70)
    logging.info("PHI SIGN DETERMINATION")
    logging.info("=" * 70)

    cifs = _find_multiple_cached_cifs(10)
    if not cifs:
        # Last resort: try to find even one
        single = _find_any_cached_cif()
        if single:
            cifs = [single]

    if not cifs:
        logging.error("FATAL: No CIF files found in cache. Run bps_download.py first.")
        logging.error(f"  Cache dir: {CACHE_DIR.resolve()}")
        sys.exit(1)

    logging.info(f"  Testing {len(cifs)} cached CIF files...")

    total_alpha_pos = 0  # alpha-region hits with sign = +1
    total_alpha_neg = 0  # alpha-region hits with sign = -1
    total_flip_pos = 0   # flipped-region hits with sign = +1
    total_flip_neg = 0   # flipped-region hits with sign = -1
    total_residues = 0

    for cif_path in cifs:
        n_alpha_pos, n_flip_pos, n_total = _test_phi_sign_on_file(cif_path, +1)
        n_alpha_neg, n_flip_neg, _       = _test_phi_sign_on_file(cif_path, -1)

        total_alpha_pos += n_alpha_pos
        total_flip_pos += n_flip_pos
        total_alpha_neg += n_alpha_neg
        total_flip_neg += n_flip_neg
        total_residues += n_total

        name = cif_path.name[:30]
        logging.info(f"  {name:<32} +1: alpha={n_alpha_pos:>4} flip={n_flip_pos:>4}  "
                     f"-1: alpha={n_alpha_neg:>4} flip={n_flip_neg:>4}  (n={n_total})")

    logging.info(f"\n  TOTALS across {len(cifs)} proteins ({total_residues} residues):")
    logging.info(f"    sign=+1: alpha_region={total_alpha_pos:>6}, flipped_region={total_flip_pos:>6}")
    logging.info(f"    sign=-1: alpha_region={total_alpha_neg:>6}, flipped_region={total_flip_neg:>6}")

    # Decision: whichever sign puts MORE residues in alpha region [-100, -20]
    pct_pos = total_alpha_pos / total_residues * 100 if total_residues > 0 else 0
    pct_neg = total_alpha_neg / total_residues * 100 if total_residues > 0 else 0

    logging.info(f"    sign=+1: {pct_pos:.1f}% in alpha region")
    logging.info(f"    sign=-1: {pct_neg:.1f}% in alpha region")

    if pct_pos > pct_neg and pct_pos > 15:
        PHI_SIGN = +1
        logging.info(f"\n  RESULT: PHI_SIGN = +1 (raw dihedral gives correct IUPAC phi)")
        logging.info(f"  {pct_pos:.1f}% residues in alpha region — GOOD")
    elif pct_neg > pct_pos and pct_neg > 15:
        PHI_SIGN = -1
        logging.info(f"\n  RESULT: PHI_SIGN = -1 (raw dihedral needs negation)")
        logging.info(f"  {pct_neg:.1f}% residues in alpha region — GOOD")
    else:
        logging.error(f"\n  FATAL: Cannot determine phi sign!")
        logging.error(f"  Neither sign gives >15% in alpha region.")
        logging.error(f"  sign=+1: {pct_pos:.1f}%, sign=-1: {pct_neg:.1f}%")
        logging.error(f"  This suggests a parser bug, not just a sign issue.")
        sys.exit(1)

    # Sanity: the WRONG sign should have most residues in the flipped region
    if PHI_SIGN == +1:
        ratio = total_alpha_pos / max(total_flip_pos, 1)
    else:
        ratio = total_alpha_neg / max(total_flip_neg, 1)

    if ratio < 2.0:
        logging.warning(f"  WARNING: alpha/flip ratio = {ratio:.1f} (expected >3). "
                        f"Sign determination may be unreliable.")

    logging.info(f"  Alpha/flip ratio: {ratio:.1f}")
    logging.info("=" * 70)

    return PHI_SIGN


# ============================================================
# BPS COMPUTATION
# ============================================================

def compute_bps(phi_psi_list, W_interp):
    valid = [(phi, psi) for phi, psi in phi_psi_list if phi is not None and psi is not None]
    if len(valid) < 3:
        return None
    phis = np.array([v[0] for v in valid])
    psis = np.array([v[1] for v in valid])
    W_chain = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_chain))
    bps_energy = float(np.sum(dW))

    try:
        from scipy.signal import find_peaks
        if len(dW) > 8:
            dW_s = np.convolve(dW, np.ones(3)/3, mode='same')
            peaks, _ = find_peaks(dW_s, prominence=np.median(dW_s), distance=3)
            n_kinks = len(peaks)
        else:
            n_kinks = int(np.sum(dW > np.mean(dW) + 2*np.std(dW)))
    except ImportError:
        n_kinks = int(np.sum(dW > np.mean(dW) + 2*np.std(dW)))

    dphi = np.arctan2(np.sin(np.diff(phis)), np.cos(np.diff(phis)))
    dpsi = np.arctan2(np.sin(np.diff(psis)), np.cos(np.diff(psis)))
    Q_phi = float(np.sum(dphi) / (2*np.pi))
    Q_psi = float(np.sum(dpsi) / (2*np.pi))

    phi_d, psi_d = np.degrees(phis), np.degrees(psis)
    n_res = len(phis)
    ss = np.zeros(n_res, dtype=int)
    ss[(-160 < phi_d) & (phi_d < 0) & (-120 < psi_d) & (psi_d < 30)] = 1   # helix (wide)
    ss[(-170 < phi_d) & (phi_d < -70) & ((psi_d > 90) | (psi_d < -120))] = 2 # sheet
    ss_nc = ss[ss > 0]
    n_trans = int(np.sum(np.diff(ss_nc) != 0)) if len(ss_nc) > 1 else 0

    n_helix = int(np.sum(ss == 1))
    n_sheet = int(np.sum(ss == 2))
    n_coil = n_res - n_helix - n_sheet

    return {'bps_energy': bps_energy, 'n_kinks': n_kinks,
            'Q_phi': Q_phi, 'Q_psi': Q_psi,
            'n_transitions': n_trans, 'n_residues': n_res,
            'pct_helix': n_helix / n_res * 100,
            'pct_sheet': n_sheet / n_res * 100,
            'pct_coil': n_coil / n_res * 100}


# ============================================================
# Rg + CONTACT ORDER
# ============================================================

def compute_rg(ca_coords):
    if len(ca_coords) < 3:
        return None
    coords = np.array(ca_coords)
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center)**2, axis=1))))


def compute_contact_order(ca_coords, cutoff=8.0):
    n = len(ca_coords)
    if n < 5:
        return None
    coords = np.array(ca_coords)
    total_sep = 0.0
    n_contacts = 0
    for i in range(n):
        diffs = coords[i+4:] - coords[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        contacts = np.where(dists < cutoff)[0]
        n_contacts += len(contacts)
        total_sep += np.sum(contacts + 4)
    if n_contacts == 0:
        return None
    return float(total_sep / (n * n_contacts))


# ============================================================
# PROCESS ONE PROTEIN
# ============================================================

def verify_dihedrals_batch(cif_paths, W_interp, n_test=5):
    """Verify dihedral sign on multiple proteins before committing to an organism.
    Returns True if SS detection looks correct, False otherwise."""
    total_helix = 0
    total_sheet = 0
    total_residues = 0
    tested = 0

    for cif_path in cif_paths[:n_test * 3]:  # try more in case some fail
        if tested >= n_test:
            break
        try:
            residues = _parse_atoms_from_cif(cif_path)
        except Exception:
            continue
        if len(residues) < 30:
            continue
        n = len(residues)
        phi_psi = []
        for i in range(n):
            raw_phi = _calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                                     residues[i]['CA'], residues[i]['C']) if i > 0 else None
            raw_psi = _calc_dihedral(residues[i]['N'], residues[i]['CA'],
                                     residues[i]['C'], residues[i+1]['N']) if i < n-1 else None
            phi = PHI_SIGN * raw_phi if raw_phi is not None else None
            psi = PHI_SIGN * raw_psi if raw_psi is not None else None
            phi_psi.append((phi, psi))
        valid_pairs = [(math.degrees(p), math.degrees(s))
                       for p, s in phi_psi if p is not None and s is not None]
        if not valid_pairs:
            continue
        phi_arr = np.array([v[0] for v in valid_pairs])
        psi_arr = np.array([v[1] for v in valid_pairs])
        helix_mask = ((-160 < phi_arr) & (phi_arr < 0) &
                      (-120 < psi_arr) & (psi_arr < 30))
        sheet_mask = ((-170 < phi_arr) & (phi_arr < -70) &
                      ((psi_arr > 90) | (psi_arr < -120)))
        total_helix += int(np.sum(helix_mask))
        total_sheet += int(np.sum(sheet_mask))
        total_residues += len(phi_arr)
        tested += 1
        logging.info(f"    verify {cif_path.name[:30]}: "
                     f"helix={np.sum(helix_mask)/len(phi_arr)*100:.1f}% "
                     f"sheet={np.sum(sheet_mask)/len(phi_arr)*100:.1f}%")

    if total_residues == 0:
        logging.warning("  Dihedral verification: no testable proteins found")
        return True  # allow processing, will fail on individual proteins if broken

    pct_h = total_helix / total_residues * 100
    pct_s = total_sheet / total_residues * 100
    logging.info(f"  DIHEDRAL VERIFICATION ({tested} proteins, {total_residues} residues):")
    logging.info(f"    helix={pct_h:.1f}%, sheet={pct_s:.1f}%")

    if pct_h < 5.0 and pct_s < 5.0:
        logging.error(f"    *** SS DETECTION FAILING across {tested} proteins ***")
        logging.error(f"    Expected 25-40% helix. Got helix={pct_h:.1f}%, sheet={pct_s:.1f}%")
        raise RuntimeError("Dihedral sign verification failed — SS detection broken")

    logging.info(f"    -> OK")
    return True


def process_one(cif_path, W_interp):
    """Process a single CIF file. Returns (result_dict, error_type).
    Uses global PHI_SIGN to correct BOTH phi and psi dihedral angles.
    Our _calc_dihedral returns the negative of IUPAC convention for ALL
    dihedrals, so both phi and psi need the same sign correction."""
    if PHI_SIGN is None:
        raise RuntimeError("PHI_SIGN not set — call determine_phi_sign() first")

    try:
        residues = _parse_atoms_from_cif(cif_path)
    except Exception:
        return None, 'parse'

    if len(residues) < 30:
        return None, 'short'

    n = len(residues)
    phi_psi = []
    plddts = []
    ca_coords = []
    for i in range(n):
        raw_phi = _calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                                 residues[i]['CA'], residues[i]['C']) if i > 0 else None
        raw_psi = _calc_dihedral(residues[i]['N'], residues[i]['CA'],
                                 residues[i]['C'], residues[i+1]['N']) if i < n-1 else None
        # CRITICAL: apply sign correction to BOTH phi and psi.
        # _calc_dihedral returns -IUPAC for all dihedrals (consistent convention).
        phi = PHI_SIGN * raw_phi if raw_phi is not None else None
        psi = PHI_SIGN * raw_psi if raw_psi is not None else None
        phi_psi.append((phi, psi))
        plddts.append(residues[i].get('bfactor', 0.0))
        ca_coords.append(residues[i]['CA'])

    try:
        bps = compute_bps(phi_psi, W_interp)
    except Exception:
        return None, 'parse'

    if bps is None:
        return None, 'parse'

    Q_mag = math.sqrt(bps['Q_phi']**2 + bps['Q_psi']**2)
    L = bps['n_residues']
    bps_norm = bps['bps_energy'] / L

    if L < 30:
        return None, 'short'
    if not np.isfinite(bps['bps_energy']):
        return None, 'parse'
    # Flag extreme BPS/L values but do not exclude them.
    # Previous behavior silently excluded these as 'range' errors,
    # hiding potential counterexamples to the universality claim.
    is_outlier = not (0.02 < bps_norm < 0.50)

    plddt_arr = np.array(plddts)

    # Validate pLDDT range (AlphaFold outputs 0-100)
    plddt_mean = float(np.mean(plddt_arr))
    plddt_median = float(np.median(plddt_arr))
    if not (0 <= plddt_mean <= 100):
        plddt_mean = None
        plddt_median = None
    plddt_below50 = float(np.sum(plddt_arr < 50) / len(plddt_arr)) if plddt_mean else None

    # SS sanity: percentages must sum to ~100%
    ss_sum = bps['pct_helix'] + bps['pct_sheet'] + bps['pct_coil']
    if not (99.0 < ss_sum < 101.0):
        return None, 'parse'

    # Rg sanity
    rg_val = compute_rg(ca_coords)
    if rg_val is not None and (rg_val <= 0 or rg_val > 500):
        rg_val = None  # implausible, null it out

    # CO sanity
    co_val = compute_contact_order(ca_coords)
    if co_val is not None and (co_val <= 0 or co_val > 1.0):
        co_val = None

    return {
        'L': L,
        'bps_energy': bps['bps_energy'], 'bps_norm': bps_norm,
        'n_transitions': bps['n_transitions'], 'n_kinks': bps['n_kinks'],
        'Q_phi': bps['Q_phi'], 'Q_psi': bps['Q_psi'],
        'Q_mag': Q_mag, 'Q_norm': Q_mag / math.sqrt(L),
        'plddt_mean': plddt_mean,
        'plddt_median': plddt_median,
        'plddt_below50': plddt_below50,
        'pct_helix': bps['pct_helix'], 'pct_sheet': bps['pct_sheet'],
        'pct_coil': bps['pct_coil'],
        'rg': rg_val,
        'contact_order': co_val,
        'is_outlier': is_outlier,
    }, None


# ============================================================
# FIND CIF FILES IN CACHE
# ============================================================

def _parse_cif_filename(path):
    """Extract (uniprot_id, af_version) from an AlphaFold CIF filename.
    E.g. 'AF-P12345-F1-model_v4.cif' -> ('P12345', 4)"""
    name = path.stem  # e.g. 'AF-P12345-F1-model_v4'
    parts = name.split('-')
    uid = parts[1] if len(parts) >= 3 else None
    version = None
    for p in parts:
        if p.startswith('model_v'):
            try:
                version = int(p.replace('model_v', ''))
            except ValueError:
                pass
    return uid, version


def find_cached_cifs(organism, min_age=3):
    """Return list of (uniprot_id, cif_path, af_version) for all cached CIFs.
    min_age: skip files modified less than this many seconds ago
    (avoids reading partially-written files from concurrent downloader)."""
    org_dir = CACHE_DIR / organism
    if not org_dir.exists():
        return []
    now = time.time()
    results = []
    for cif in org_dir.glob("AF-*-F1-model_v*.cif"):
        st = cif.stat()
        if st.st_size < 100:
            continue
        if min_age > 0 and (now - st.st_mtime) < min_age:
            continue  # still being written by downloader
        uid, version = _parse_cif_filename(cif)
        if uid:
            results.append((uid, cif, version))
    return results


# ============================================================
# PROCESS ORGANISM
# ============================================================

def process_organism(organism, W_interp, conn):
    logging.info("=" * 70)
    logging.info(f"PROCESSING: {organism.upper()}")
    logging.info("=" * 70)

    cached = find_cached_cifs(organism)
    if not cached:
        logging.warning(f"  No CIF files found in {CACHE_DIR / organism}")
        return

    info = PROTEOMES.get(organism, {})
    conn.execute("INSERT OR IGNORE INTO proteomes (organism) VALUES (?)", (organism,))
    conn.execute("UPDATE proteomes SET name=?, taxid=?, status='running', started_at=? WHERE organism=?",
                 (info.get('name', ''), info.get('taxid', ''), datetime.now().isoformat(), organism))
    conn.commit()

    # Log version distribution
    versions = [v for _, _, v in cached if v is not None]
    if versions:
        from collections import Counter
        vc = Counter(versions)
        ver_str = ", ".join(f"v{k}: {v}" for k, v in sorted(vc.items()))
        logging.info(f"  AF versions: {ver_str}")

    # Skip already processed
    done_ids = db_get_done_ids(conn, organism)
    fail_ids = db_get_failure_ids(conn, organism)
    skip = done_ids | fail_ids

    remaining = [(uid, path, ver) for uid, path, ver in cached if uid not in skip]

    if not remaining:
        n_done, n_fail = db_counts(conn, organism)
        logging.info(f"  Complete: {n_done} processed, {n_fail} failures")
        results = db_get_results(conn, organism)
        if len(results) > 10:
            analyze_proteome(results, organism)
        return

    # Verify dihedrals on a batch of proteins (not just the first one)
    if len(done_ids) == 0:
        verify_dihedrals_batch([path for _, path, _ in remaining], W_interp)

    logging.info(f"  Cached: {len(cached)}, done: {len(done_ids)}, "
                 f"failed: {len(fail_ids)}, remaining: {len(remaining)}")

    t_start = time.time()
    new_ok = 0
    new_fail = 0

    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_start:batch_start + BATCH_SIZE]

        batch_results = []
        batch_failures = []

        for uid, cif_path, af_ver in batch:
            result, err = process_one(cif_path, W_interp)
            if result:
                result['uniprot_id'] = uid
                result['af_version'] = af_ver
                batch_results.append(result)
                new_ok += 1
            else:
                batch_failures.append((uid, err or 'parse'))
                new_fail += 1

        # Batch DB write
        if batch_results:
            conn.executemany("""
                INSERT OR REPLACE INTO proteins
                (uniprot_id, organism, L, bps_energy, bps_norm, n_transitions,
                 n_kinks, Q_phi, Q_psi, Q_mag, Q_norm,
                 plddt_mean, plddt_median, plddt_below50,
                 pct_helix, pct_sheet, pct_coil, rg, contact_order, af_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(r['uniprot_id'], organism, r['L'],
                   r['bps_energy'], r['bps_norm'],
                   r['n_transitions'], r['n_kinks'],
                   r['Q_phi'], r['Q_psi'], r['Q_mag'], r['Q_norm'],
                   r['plddt_mean'], r['plddt_median'], r['plddt_below50'],
                   r['pct_helix'], r['pct_sheet'], r['pct_coil'],
                   r['rg'], r['contact_order'], r.get('af_version'))
                  for r in batch_results])

        if batch_failures:
            conn.executemany("""
                INSERT OR REPLACE INTO failures
                (uniprot_id, organism, error_type, error_msg)
                VALUES (?, ?, ?, '')
            """, [(uid, organism, err) for uid, err in batch_failures])

        conn.commit()

        # Progress
        so_far = batch_start + len(batch)
        elapsed = time.time() - t_start
        rate = so_far / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - so_far) / rate / 60 if rate > 0 else 0
        logging.info(f"  [{so_far}/{len(remaining)}] {rate:.1f}/sec, "
                     f"ETA: {eta:.0f} min, +{new_ok} ok, +{new_fail} fail")

    elapsed = time.time() - t_start
    n_done, n_fail = db_counts(conn, organism)
    logging.info(f"\n  Finished: +{new_ok} new in {elapsed:.0f}s ({new_ok/max(elapsed,1):.1f}/sec)")
    logging.info(f"  Database: {n_done} proteins, {n_fail} failures")

    # Update proteome stats
    results = db_get_results(conn, organism)
    bps_vals = [r['bps_norm'] for r in results]
    conn.execute("""UPDATE proteomes SET n_ids=?, n_done=?, n_failed=?, status='done',
                    finished_at=?, bps_mean=?, bps_std=? WHERE organism=?""",
                 (len(cached), n_done, n_fail, datetime.now().isoformat(),
                  float(np.mean(bps_vals)) if bps_vals else None,
                  float(np.std(bps_vals)) if bps_vals else None,
                  organism))
    conn.commit()

    if len(results) > 10:
        analyze_proteome(results, organism)


# ============================================================
# ANALYSIS (per-organism)
# ============================================================

def analyze_proteome(results, organism):
    logging.info(f"\n{'='*70}")
    logging.info(f"ANALYSIS: {organism.upper()} (N={len(results)})")
    logging.info(f"{'='*70}")
    L = np.array([r['L'] for r in results])
    bps_norm = np.array([r['bps_norm'] for r in results])
    BPS = np.array([r['bps_energy'] for r in results])

    logging.info(f"  Length: mean={L.mean():.0f}, median={np.median(L):.0f}")
    logging.info(f"  BPS/res: mean={bps_norm.mean():.4f}, std={bps_norm.std():.4f}")

    plddt = np.array([r.get('plddt_mean', 0) or 0 for r in results])
    if np.any(plddt > 0):
        logging.info(f"  pLDDT:   mean={plddt[plddt>0].mean():.1f}")

    pct_h = np.array([r.get('pct_helix', 0) or 0 for r in results])
    pct_s = np.array([r.get('pct_sheet', 0) or 0 for r in results])
    pct_c = np.array([r.get('pct_coil', 0) or 0 for r in results])
    if np.any(pct_h > 0) or np.any(pct_s > 0):
        logging.info(f"  SS:      helix={pct_h.mean():.1f}%, sheet={pct_s.mean():.1f}%, coil={pct_c.mean():.1f}%")

    # KEY: helix sanity check
    if pct_h.mean() < 5.0:
        logging.warning(f"  *** WARNING: helix={pct_h.mean():.1f}% is too low! ***")
        logging.warning(f"  *** Expected 25-40%. Phi sign may still be wrong. ***")
    elif pct_h.mean() > 20.0:
        logging.info(f"  Helix detection: GOOD ({pct_h.mean():.1f}%)")

    for lo, hi, label in [(0,100,"Small"), (100,300,"Medium"), (300,1000,"Large"), (1000,99999,"XL")]:
        m = (L >= lo) & (L < hi)
        if np.sum(m) > 0:
            logging.info(f"    {label:<8} n={np.sum(m):>5}  BPS/res={bps_norm[m].mean():.4f}+/-{bps_norm[m].std():.4f}")

    r_bl, _ = pearsonr(L, BPS)
    logging.info(f"  r(BPS, L) = {r_bl:.3f}")

    if np.any(plddt > 0):
        m = plddt > 0
        if np.sum(m) > 10:
            r_bp, p_bp = pearsonr(bps_norm[m], plddt[m])
            logging.info(f"  r(BPS/res, pLDDT) = {r_bp:.4f} (p={p_bp:.2e})")

    co_vals = np.array([r.get('contact_order', 0) or 0 for r in results])
    if np.any(co_vals > 0):
        m = co_vals > 0
        if np.sum(m) > 10:
            r_bc, p_bc = pearsonr(bps_norm[m], co_vals[m])
            logging.info(f"  r(BPS/res, CO)    = {r_bc:.4f} (p={p_bc:.2e})")

    # Figure
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        axes[0,0].hist(bps_norm, bins=50, color='#2a9d8f', edgecolor='k', alpha=0.8)
        axes[0,0].axvline(np.median(bps_norm), color='red', ls='--')
        axes[0,0].set_title(f'{organism} BPS/res (n={len(results)})')
        axes[0,1].hist(np.array([r['Q_norm'] for r in results]), bins=50, color='#e76f51', edgecolor='k', alpha=0.8)
        axes[0,1].set_title('Winding |Q|/sqrt(L)')
        axes[0,2].scatter(L, BPS, s=3, alpha=0.3, c='#457b9d')
        axes[0,2].set_title(f'BPS vs L (r={r_bl:.3f})')
        axes[1,0].scatter(L, np.array([r['n_kinks'] for r in results]), s=3, alpha=0.3, c='#2a9d8f')
        axes[1,0].set_title('Kinks vs L')
        axes[1,1].scatter(bps_norm, np.array([r['Q_norm'] for r in results]), s=3, alpha=0.3, c='#e63946')
        axes[1,1].set_title('BPS/res vs Winding')
        axes[1,2].hist(np.array([r['n_transitions'] for r in results]), bins=50, color='#264653', edgecolor='k', alpha=0.8)
        axes[1,2].set_title('SS Transitions')
        if np.any(plddt > 0):
            axes[2,0].scatter(plddt[plddt>0], bps_norm[plddt>0], s=3, alpha=0.3, c='#e9c46a')
            axes[2,0].set_xlabel('pLDDT'); axes[2,0].set_ylabel('BPS/res')
            axes[2,0].set_title('BPS/res vs pLDDT')
        if np.any(pct_h > 0):
            axes[2,1].scatter(pct_h, pct_s, s=3, alpha=0.3, c='#264653')
            axes[2,1].set_xlabel('% Helix'); axes[2,1].set_ylabel('% Sheet')
            axes[2,1].set_title('SS Composition')
        if np.any(co_vals > 0):
            m = co_vals > 0
            axes[2,2].scatter(co_vals[m], bps_norm[m], s=3, alpha=0.3, c='#a8dadc')
            axes[2,2].set_xlabel('Contact Order'); axes[2,2].set_ylabel('BPS/res')
            axes[2,2].set_title('BPS/res vs CO')
        plt.tight_layout()
        figpath = OUTPUT_DIR / f"{organism}_proteome_analysis.png"
        plt.savefig(figpath, dpi=150, bbox_inches='tight'); plt.close()
        logging.info(f"  Figure: {figpath}")
    except ImportError:
        pass


# ============================================================
# VALIDATION (Plaxco 23)
# ============================================================

def run_validation(W_interp, conn):
    logging.info("=" * 70)
    logging.info("VALIDATION (AlphaFold vs PDB on Plaxco 23)")
    logging.info("=" * 70)

    results = []
    for uid, pdb, name, ln_kf, co, exp_L in PLAXCO_PROTEINS:
        cif_path = None
        for v in (4, 3, 2, 1):
            for search_dir in [CACHE_DIR / "_selftest", CACHE_DIR / "_validation",
                               CACHE_DIR / "ecoli", CACHE_DIR / "human",
                               CACHE_DIR / "yeast", CACHE_DIR]:
                p = search_dir / f"AF-{uid}-F1-model_v{v}.cif"
                if p.exists() and p.stat().st_size > 100:
                    cif_path = p
                    break
            if cif_path:
                break

        if cif_path is None:
            logging.info(f"  {uid}/{pdb} ({name}): SKIP — not cached")
            continue

        result, err = process_one(cif_path, W_interp)
        if result is None:
            logging.info(f"  {uid}/{pdb} ({name}): PARSE FAIL")
            continue

        result['pdb_id'] = pdb
        result['name'] = name
        result['ln_kf'] = ln_kf
        result['co'] = co
        results.append(result)
        logging.info(f"  {uid}/{pdb}: L={result['L']} BPS/res={result['bps_norm']:.4f} "
                     f"pLDDT={result['plddt_mean']:.1f} helix={result['pct_helix']:.1f}%")

    if len(results) < 5:
        logging.warning("Too few results for correlation analysis")
        return

    ln_kf = np.array([r['ln_kf'] for r in results])
    co_arr = np.array([r['co'] for r in results])
    BPS = np.array([r['bps_energy'] for r in results])
    n = len(results)
    logging.info(f"\nCORRELATION ANALYSIS (N={n})")
    for nm, vals in [("CO", co_arr), ("BPS", BPS)]:
        rp, pp = pearsonr(vals, ln_kf)
        logging.info(f"  {nm}: r={rp:.4f}, p={pp:.2e}")
    X = np.column_stack([co_arr, BPS, np.ones(n)])
    beta, _, _, _ = np.linalg.lstsq(X, ln_kf, rcond=None)
    pred = X @ beta
    r2 = 1 - np.sum((ln_kf - pred)**2) / np.sum((ln_kf - ln_kf.mean())**2)
    logging.info(f"  CO+BPS R2={r2:.4f}")


# ============================================================
# STATUS / SUMMARY / EXPORT
# ============================================================

def _parse_count(s):
    """Parse approximate counts like '~4.4K' -> 4400"""
    s = s.replace('~', '').strip()
    return int(float(s.replace('K', '')) * 1000) if 'K' in s else int(s)


def show_status(conn):
    print(f"\n{'='*100}")
    print(f"  BPS PIPELINE COMPLETENESS REPORT")
    print(f"{'='*100}")

    total_proteins = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
    total_failures = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    print(f"\n  Database: {total_proteins:,} proteins, {total_failures:,} failures", end="")
    if DB_PATH.exists():
        print(f"  ({DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print()

    # ── Per-organism completeness table ──
    header = (f"  {'Organism':<18} {'Advert':>7} {'IDs':>7} {'Cached':>7} "
              f"{'Done':>7} {'Fail':>7} {'%Cache':>7} {'%Done':>7} {'BPS/res':>8} {'Status':>10}")
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    grand_advert = 0
    grand_ids = 0
    grand_cached = 0
    grand_done = 0
    grand_fail = 0

    orgs_sorted = sorted(PROTEOMES.keys(),
                         key=lambda k: _parse_count(PROTEOMES[k]['count']))

    for org in orgs_sorted:
        info = PROTEOMES[org]
        n_advert = _parse_count(info['count'])

        # IDs fetched
        ids_file = CACHE_DIR / f"{org}_uniprot_ids.txt"
        n_ids = 0
        if ids_file.exists():
            with open(ids_file) as f:
                n_ids = sum(1 for line in f if line.strip())

        # CIFs cached
        org_dir = CACHE_DIR / org
        n_cached = 0
        if org_dir.exists():
            n_cached = len(list(org_dir.glob("AF-*-F1-model_v*.cif")))

        # Processed + failed
        n_done = conn.execute(
            "SELECT COUNT(*) FROM proteins WHERE organism=?", (org,)).fetchone()[0]
        n_fail = conn.execute(
            "SELECT COUNT(*) FROM failures WHERE organism=?", (org,)).fetchone()[0]

        # BPS/res mean
        row = conn.execute(
            "SELECT AVG(bps_norm) FROM proteins WHERE organism=?", (org,)).fetchone()
        bps_s = f"{row[0]:.4f}" if row[0] else "—"

        # Percentages (use IDs as denominator if available, else advertised)
        denom_cache = n_ids if n_ids > 0 else n_advert
        denom_done = n_cached if n_cached > 0 else (n_ids if n_ids > 0 else n_advert)
        pct_cache = n_cached / denom_cache * 100 if denom_cache > 0 else 0
        pct_done = (n_done + n_fail) / denom_done * 100 if denom_done > 0 else 0

        # Status
        if n_done == 0 and n_cached == 0 and n_ids == 0:
            status = "—"
        elif n_cached == 0:
            status = "no cache"
        elif pct_done >= 99.0:
            status = "DONE"
        elif n_done > 0:
            status = "partial"
        else:
            status = "cached"

        grand_advert += n_advert
        grand_ids += n_ids
        grand_cached += n_cached
        grand_done += n_done
        grand_fail += n_fail

        # Only show rows that have some activity
        if n_ids > 0 or n_cached > 0 or n_done > 0:
            print(f"  {org:<18} {n_advert:>7,} {n_ids:>7,} {n_cached:>7,} "
                  f"{n_done:>7,} {n_fail:>7,} {pct_cache:>6.1f}% {pct_done:>6.1f}% "
                  f"{bps_s:>8} {status:>10}")

    # Show organisms with no activity at all
    inactive = [org for org in orgs_sorted
                if not (CACHE_DIR / f"{org}_uniprot_ids.txt").exists()
                and not (CACHE_DIR / org).exists()
                and conn.execute("SELECT COUNT(*) FROM proteins WHERE organism=?",
                                 (org,)).fetchone()[0] == 0]
    if inactive:
        print(f"\n  Not started ({len(inactive)}): {', '.join(inactive)}")

    # ── Totals ──
    print(f"\n  {'TOTAL':<18} {grand_advert:>7,} {grand_ids:>7,} {grand_cached:>7,} "
          f"{grand_done:>7,} {grand_fail:>7,} "
          f"{grand_cached/grand_ids*100 if grand_ids else 0:>6.1f}% "
          f"{(grand_done+grand_fail)/grand_cached*100 if grand_cached else 0:>6.1f}%")

    # ── Disk usage ──
    if CACHE_DIR.exists():
        total_bytes = sum(f.stat().st_size for f in CACHE_DIR.rglob("*.cif"))
        print(f"\n  Cache: {total_bytes / 1024**3:.2f} GB in {CACHE_DIR.resolve()}")

    # ── Warnings ──
    global_helix = conn.execute(
        "SELECT AVG(pct_helix) FROM proteins WHERE pct_helix IS NOT NULL").fetchone()[0]
    if global_helix is not None and global_helix < 10:
        print(f"\n  *** WARNING: Global helix = {global_helix:.1f}%. Expected 25-40%. ***")
        print(f"  *** Phi sign is likely wrong. Reset and reprocess. ***")

    unprocessed_total = 0
    for org in orgs_sorted:
        org_dir = CACHE_DIR / org
        if org_dir.exists():
            n_cifs = len(list(org_dir.glob("AF-*-F1-model_v*.cif")))
            n_done = conn.execute(
                "SELECT COUNT(*) FROM proteins WHERE organism=?", (org,)).fetchone()[0]
            n_fail = conn.execute(
                "SELECT COUNT(*) FROM failures WHERE organism=?", (org,)).fetchone()[0]
            unprocessed_total += max(n_cifs - n_done - n_fail, 0)
    if unprocessed_total > 0:
        print(f"\n  {unprocessed_total:,} cached CIFs not yet processed"
              f" — run: python bps_process.py --mode all")

    print()


def show_summary(conn):
    rows = conn.execute("""
        SELECT organism, COUNT(*), AVG(bps_norm),
               AVG(bps_norm*bps_norm) - AVG(bps_norm)*AVG(bps_norm),
               AVG(plddt_mean), AVG(pct_helix), AVG(pct_sheet), AVG(pct_coil),
               AVG(contact_order)
        FROM proteins GROUP BY organism HAVING COUNT(*) >= 10
        ORDER BY AVG(bps_norm) DESC
    """).fetchall()

    print(f"\n{'='*100}")
    print(f"  CROSS-PROTEOME SUMMARY")
    print(f"{'='*100}")
    header = (f"  {'Organism':<20} {'N':>7} {'BPS/res':>9} {'SD':>7} "
              f"{'pLDDT':>7} {'%helix':>8} {'%sheet':>8} {'%coil':>8} {'CO':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for org, n, mb, vb, plddt, helix, sheet, coil, co in rows:
        sd = math.sqrt(max(vb or 0, 0))
        plddt_s = f"{plddt:.1f}" if plddt else "—"
        co_s = f"{co:.4f}" if co else "—"
        h_s = f"{helix:.1f}%" if helix is not None else "—"
        s_s = f"{sheet:.1f}%" if sheet is not None else "—"
        c_s = f"{coil:.1f}%" if coil is not None else "—"
        print(f"  {org:<20} {n:>7} {mb:>9.4f} {sd:>7.4f} "
              f"{plddt_s:>7} {h_s:>8} {s_s:>8} {c_s:>8} {co_s:>8}")
    print()


def export_csvs(conn):
    import csv
    organisms = [r[0] for r in conn.execute(
        "SELECT DISTINCT organism FROM proteins ORDER BY organism").fetchall()]
    for org in organisms:
        rows = conn.execute(
            "SELECT * FROM proteins WHERE organism = ?", (org,)).fetchall()
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM proteins LIMIT 0").description]
        csv_path = OUTPUT_DIR / f"{org}_bps_results.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        logging.info(f"  Exported {len(rows)} rows -> {csv_path}")


# ============================================================
# WATCH MODE — continuous processing loop
# ============================================================

def watch_loop(poll_interval=30):
    """Watch for new cached CIFs and process them continuously.
    Safe to run alongside bps_download.py and compile_results.py.

    Builds the superpotential and determines phi sign once, then loops:
    scan all organism directories for unprocessed CIFs, process them in
    batches, sleep, repeat. Ctrl+C stops cleanly."""

    conn = init_db()
    try:
        determine_phi_sign()
        logging.info(f"Using PHI_SIGN = {PHI_SIGN}")

        W = build_superpotential()
        logging.info(f"WATCH MODE: polling every {poll_interval}s for new CIFs")

        while True:
            any_work = False

            for org in sorted(PROTEOMES.keys()):
                cached = find_cached_cifs(org)
                if not cached:
                    continue

                done_ids = db_get_done_ids(conn, org)
                fail_ids = db_get_failure_ids(conn, org)
                remaining = [(uid, p) for uid, p in cached
                             if uid not in done_ids and uid not in fail_ids]

                if not remaining:
                    continue

                any_work = True
                info = PROTEOMES.get(org, {})
                logging.info(f"\n  {org}: {len(remaining)} new CIFs to process")
                try:
                    process_organism(org, W, conn)
                except Exception as e:
                    logging.error(f"ERROR on {org}: {e}")
                    continue

            if not any_work:
                logging.info(f"No new CIFs found. Sleeping {poll_interval}s...")
            else:
                logging.info(f"Batch complete. Sleeping {poll_interval}s...")

            try:
                time.sleep(poll_interval)
            except KeyboardInterrupt:
                logging.info("\nWatch stopped. Database is safe.")
                return

    except KeyboardInterrupt:
        logging.info("\nWatch stopped. Database is safe.")
    finally:
        conn.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BPS Processor v2 — compute from cached CIFs")
    parser.add_argument("--mode", choices=[
        "all", "organism", "status", "summary", "export", "validate", "phi-check", "watch"
    ], default="status")
    parser.add_argument("--organism", default="ecoli",
                        choices=list(PROTEOMES.keys()))
    parser.add_argument("--reset", action="store_true",
                        help="Delete database and recompute from cached CIFs")
    parser.add_argument("--poll", type=int, default=30,
                        help="Poll interval in seconds for watch mode (default: 30)")
    args = parser.parse_args()

    setup_logging()

    # ── Phi check mode: just test and exit ──
    if args.mode == "phi-check":
        determine_phi_sign()
        print(f"\nPHI_SIGN = {PHI_SIGN}")
        return

    # ── Watch mode: continuous processing loop ──
    if args.mode == "watch":
        watch_loop(args.poll)
        return

    # ── Status/summary/export don't need phi determination ──
    if args.mode in ("status", "summary", "export"):
        if args.reset and args.mode != "status":
            print("--reset only works with processing modes")
            return
        conn = init_db()
        try:
            if args.mode == "status":
                show_status(conn)
            elif args.mode == "summary":
                show_summary(conn)
            elif args.mode == "export":
                export_csvs(conn)
        finally:
            conn.close()
        return

    # ── Processing modes: MUST determine phi sign first ──
    if args.reset:
        if DB_PATH.exists():
            logging.info(f"RESET: Deleting {DB_PATH} ({DB_PATH.stat().st_size/1024/1024:.1f} MB)")
            DB_PATH.unlink()
        else:
            logging.info("No database to delete.")

    conn = init_db()

    try:
        # CRITICAL: determine phi sign before any computation
        determine_phi_sign()
        logging.info(f"Using PHI_SIGN = {PHI_SIGN}\n")
        if args.mode == "validate":
            W = build_superpotential()
            run_validation(W, conn)

        elif args.mode == "organism":
            W = build_superpotential()
            process_organism(args.organism, W, conn)

        elif args.mode == "all":
            W = build_superpotential()

            # ── Check for orphaned CIFs in flat cache ──
            orphans = list(CACHE_DIR.glob("AF-*-F1-model_v*.cif"))
            orphans = [p for p in orphans if p.stat().st_size > 100]
            if orphans:
                logging.warning(f"\n  ORPHAN CHECK: {len(orphans)} CIF files in flat {CACHE_DIR}/")
                logging.warning(f"  These are from the old pipeline and won't be processed")
                logging.warning(f"  unless they match a known UniProt ID list.")
                # Sample a few to show what they are
                sample = orphans[:5]
                for p in sample:
                    uid, ver = _parse_cif_filename(p)
                    logging.warning(f"    {p.name} -> {uid} (v{ver})")
                if len(orphans) > 5:
                    logging.warning(f"    ... and {len(orphans)-5} more")

            # Validate first (if proteins available)
            try:
                run_validation(W, conn)
            except Exception as e:
                logging.warning(f"Validation skipped: {e}")

            # Process all organisms that have cached CIFs
            orgs = sorted(PROTEOMES.keys(), key=lambda k: _parse_count(PROTEOMES[k]['count']))
            logging.info(f"\nPROCESSING ALL CACHED PROTEOMES")

            for i, org in enumerate(orgs):
                org_dir = CACHE_DIR / org
                if not org_dir.exists():
                    continue
                n_cifs = len(list(org_dir.glob("AF-*-F1-model_v*.cif")))
                if n_cifs == 0:
                    continue
                info = PROTEOMES[org]
                logging.info(f"\n[{i+1}/{len(orgs)}] {info['name']} ({n_cifs} CIFs cached)")
                try:
                    process_organism(org, W, conn)
                except KeyboardInterrupt:
                    logging.info("\nInterrupted. All committed data is safe.")
                    break
                except Exception as e:
                    logging.error(f"ERROR on {org}: {e}\n{traceback.format_exc()}")
                    continue

            show_summary(conn)

    except KeyboardInterrupt:
        logging.info("\nInterrupted. Database is safe.")
    except Exception as e:
        logging.error(f"FATAL: {e}\n{traceback.format_exc()}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
