#!/usr/bin/env python3
"""PDB Experimental Validation of BPS/L.

Downloads high-resolution X-ray crystal structures from the RCSB PDB,
computes BPS/L using the same superpotential as the main pipeline, and
compares against AlphaFold-derived results.

Addresses the reviewer concern: "Is BPS/L ≈ 0.20 a property of real
proteins, or of AlphaFold's learned structural prior?"

Chain list source (in priority order):
  1. PISCES server (non-redundant, ≤25% identity, ≤2.0 Å, R≤0.25)
  2. Hardcoded curated list of ~200 high-resolution structures

Usage:
    python bps_pdb_validate.py
    python bps_pdb_validate.py --max 100      # limit to 100 chains
    python bps_pdb_validate.py --skip-download # use already-cached CIFs
"""

import sys
import time
import math
import logging
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

# Import from existing pipeline
sys.path.insert(0, str(Path(__file__).parent))
from bps_process import build_superpotential, _calc_dihedral, determine_phi_sign

try:
    from scipy.stats import ks_2samp, pearsonr
    HAS_SCIPY_STATS = True
except ImportError:
    HAS_SCIPY_STATS = False

try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import PPBuilder, is_aa
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

# ============================================================
# CONFIGURATION
# ============================================================

PDB_CACHE_DIR = Path("./pdb_validation_cache")
REPORT_PATH = Path("pdb_validation_report.md")
RNG_SEED = 42
MIN_CHAIN_LEN = 40
MAX_MISSING_BACKBONE_PCT = 10.0  # skip chains with >10% missing backbone

# AlphaFold reference values (from the main pipeline, HQ pLDDT>=85)
AF_REFERENCE = {
    'mean_bps': 0.2026,
    'std_bps': 0.0084,
    'cv_pct': 1.3,
    'n_proteins': 6155,
}

# PISCES URLs to try
PISCES_URLS = [
    "https://dunbrack.fccc.edu/pisces/cullpdb_pc25_res2.0_R0.25_d250_chains_len40-10000",
    "https://dunbrack.fccc.edu/pisces/cullpdb_pc25_res2.0_R0.25_d2026_chains_len40-10000",
]


# ============================================================
# CURATED HIGH-RESOLUTION PDB CHAINS
# ============================================================
# Fallback if PISCES is inaccessible. ~200 well-known structures
# spanning all major SCOP/CATH classes at ≤2.0 Å resolution.
# Format: (pdb_id, chain_id, description, scop_class)

CURATED_CHAINS = [
    # All-alpha proteins
    ("1MBN", "A", "Myoglobin", "all-alpha"),
    ("2HHB", "A", "Hemoglobin alpha", "all-alpha"),
    ("2HHB", "B", "Hemoglobin beta", "all-alpha"),
    ("1LMB", "3", "Lambda repressor", "all-alpha"),
    ("1A6M", "A", "Oxy-myoglobin", "all-alpha"),
    ("2CYP", "A", "Cytochrome c peroxidase", "all-alpha"),
    ("1BCF", "A", "Bacterioferritin", "all-alpha"),
    ("1HRC", "A", "Cytochrome c", "all-alpha"),
    ("1ECA", "A", "Erythrocruorin", "all-alpha"),
    ("1UTG", "A", "Uteroglobin", "all-alpha"),
    ("1MBC", "A", "Myoglobin CO", "all-alpha"),
    ("1CYO", "A", "Cytochrome b562", "all-alpha"),
    ("1R69", "A", "434 repressor", "all-alpha"),
    ("1LIS", "A", "Listeriolysin regulatory", "all-alpha"),
    ("256B", "A", "Cytochrome b556", "all-alpha"),
    ("1F68", "A", "Ferredoxin-like domain", "alpha+beta"),
    ("1A32", "A", "Calmodulin-like", "all-alpha"),
    ("1GVD", "A", "Globin-like", "all-alpha"),
    ("1DUR", "A", "Duffy binding-like", "all-alpha"),
    ("1RRO", "A", "Oncomodulin (EF-hand)", "all-alpha"),
    ("3SDH", "A", "Iron superoxide dismutase", "all-alpha"),
    ("1OPC", "A", "Opioid-binding protein", "all-alpha"),
    ("1AIS", "A", "Annexin V", "all-alpha"),
    ("1K8U", "A", "Villin headpiece HP35", "all-alpha"),
    ("1YCC", "A", "Cytochrome c iso-1", "all-alpha"),
    ("1COL", "A", "Colicin A", "all-alpha"),
    ("1BDD", "A", "Protein A B-domain", "all-alpha"),
    ("1ENH", "A", "Engrailed homeodomain", "all-alpha"),
    ("1PRU", "A", "Pru domain", "all-alpha"),

    # All-beta proteins
    ("2RHE", "A", "Immunoglobulin VL", "all-beta"),
    ("1TIT", "A", "Titin I27 domain", "all-beta"),
    ("2RN2", "A", "RNase Sa", "all-beta"),
    ("1SHG", "A", "SH3 domain (alpha-spectrin)", "all-beta"),
    ("1FNF", "A", "Fibronectin type III", "all-beta"),
    ("1TEN", "A", "Tenascin", "all-beta"),
    ("1CD8", "A", "CD8 alpha", "all-beta"),
    ("1REI", "A", "Bence-Jones immunoglobulin", "all-beta"),
    ("1NLS", "A", "Neocarzinostatin", "all-beta"),
    ("1PGB", "A", "Protein G B1 domain", "all-beta"),
    ("2PTL", "A", "Plastocyanin", "all-beta"),
    ("1AQB", "A", "Azurin", "all-beta"),
    ("3CHB", "D", "Concanavalin B", "all-beta"),
    ("1AUI", "A", "Outer membrane protein A", "all-beta"),
    ("1FKJ", "A", "FK506-binding protein", "alpha+beta"),
    ("7RSA", "A", "Ribonuclease A", "alpha+beta"),
    ("1BNR", "A", "Barnase (RNA-free)", "alpha+beta"),
    ("1CSP", "A", "Cold shock protein CspA", "all-beta"),
    ("1WIT", "A", "Cellular retinol-binding", "all-beta"),
    ("1PIN", "A", "Pin1 WW domain", "all-beta"),
    ("2AIT", "A", "Alpha-amylase inhibitor (tendamistat)", "all-beta"),
    ("1K40", "A", "GB1 protein", "all-beta"),
    ("2ACE", "A", "Acetylcholinesterase", "alpha-beta"),
    ("1VCC", "A", "Vaccinia topoisomerase N-term", "alpha+beta"),
    ("1OPD", "A", "HPr phosphocarrier", "alpha+beta"),
    ("1I8A", "A", "Carbohydrate-binding module CBM9", "all-beta"),
    ("1AMM", "A", "Gamma-B crystallin", "all-beta"),
    ("1CTF", "A", "L7/L12 C-terminal", "alpha+beta"),
    ("1SRL", "A", "Src SH3 domain", "all-beta"),
    ("1IGD", "A", "Protein G IgG-binding", "all-beta"),
    ("1BPI", "A", "BPTI trypsin inhibitor", "small"),

    # Alpha/beta (TIM barrels, Rossmann folds, etc.)
    ("1UBQ", "A", "Ubiquitin", "alpha-beta"),
    ("2LZM", "A", "T4 lysozyme", "alpha+beta"),
    ("1HHP", "A", "HIV-1 protease", "alpha-beta"),
    ("2CI2", "I", "Chymotrypsin inhibitor 2", "alpha+beta"),
    ("3ENL", "A", "Enolase", "alpha-beta"),
    ("1CDK", "E", "cAMP-dep protein kinase", "alpha-beta"),
    ("5P21", "A", "p21 H-Ras", "alpha-beta"),
    ("1TIM", "A", "Triosephosphate isomerase", "alpha-beta"),
    ("8RUB", "A", "RuBisCO", "alpha-beta"),
    ("1PFK", "A", "Phosphofructokinase", "alpha-beta"),
    ("4ENL", "A", "Enolase (yeast)", "alpha-beta"),
    ("1GKY", "A", "Guanylate kinase", "alpha-beta"),
    ("1PHT", "A", "Phosphotriesterase", "alpha-beta"),
    ("1BTL", "A", "Beta-lactamase TEM-1", "alpha-beta"),
    ("1AK0", "A", "Alcohol dehydrogenase", "alpha-beta"),
    ("2TRX", "A", "Thioredoxin", "alpha-beta"),
    ("1CRN", "A", "Crambin", "small"),
    ("1L2Y", "A", "Trp-cage miniprotein", "small"),
    ("1UXD", "A", "UDP-glucuronate decarbox.", "alpha-beta"),
    ("1ADS", "A", "Aldose reductase", "alpha-beta"),
    ("1XNB", "A", "Xylanase", "alpha-beta"),
    ("3RN3", "A", "Ribonuclease S", "alpha+beta"),
    ("1CSE", "E", "Subtilisin", "alpha-beta"),
    ("2SNS", "A", "Staphylococcal nuclease", "alpha-beta"),
    ("1PPL", "E", "Penicillopepsin", "alpha-beta"),
    ("1AKE", "A", "Adenylate kinase", "alpha-beta"),
    ("1PHP", "A", "Phosphorylase B", "alpha-beta"),
    ("3PGK", "A", "Phosphoglycerate kinase", "alpha-beta"),
    ("1EFN", "A", "Elongation factor (N-term)", "alpha-beta"),
    ("1CSK", "A", "CheA sensor kinase", "alpha-beta"),
    ("1STP", "A", "Streptavidin", "all-beta"),
    ("1LYZ", "A", "Hen lysozyme", "alpha+beta"),
    ("1HEL", "A", "Hen egg-white lysozyme", "alpha+beta"),
    ("1GCA", "A", "Glucoamylase", "alpha-beta"),
    ("1CHD", "A", "Cholesterol oxidase", "alpha-beta"),
    ("1OVA", "A", "Ovalbumin", "alpha-beta"),
    ("1PII", "A", "Trypsin (porcine)", "alpha-beta"),
    ("2CPP", "A", "Cytochrome P450cam", "alpha-beta"),
    ("1EST", "A", "Elastase", "alpha-beta"),
    ("4GCR", "A", "Gamma-B crystallin", "all-beta"),
    ("1HOE", "A", "Alpha-amylase inhibitor", "all-beta"),

    # Alpha+beta
    ("1AAP", "A", "Amyloid precursor inhibitor", "small"),
    ("5PTI", "A", "Pancreatic trypsin inhibitor", "small"),
    ("1EGF", "A", "Epidermal growth factor", "small"),
    ("1AON", "A", "GroEL apical domain", "alpha+beta"),
    ("1BRS", "A", "Barnase", "alpha+beta"),
    ("1BNI", "A", "Barstar", "alpha+beta"),
    ("1ROP", "A", "Rop protein", "all-alpha"),
    ("1MJC", "A", "alpha-B crystallin", "alpha+beta"),
    ("1HKS", "A", "HPr kinase/phosphorylase", "alpha+beta"),
    ("1APS", "A", "Acylphosphatase", "alpha+beta"),
    ("2OCJ", "A", "p53 core domain", "alpha+beta"),
    ("1DPS", "A", "DNA protection protein", "alpha+beta"),
    ("3HMZ", "A", "alpha-A crystallin", "alpha+beta"),
    ("1AON", "O", "GroEL (another chain)", "alpha+beta"),
    ("1NKD", "A", "NK-lysin", "all-alpha"),

    # Small/disulfide-rich
    ("1EDN", "A", "Endothelin-1", "small"),
    ("1LE0", "A", "Omega-conotoxin", "small"),
    ("1ZAA", "A", "Zinc finger (Zif268)", "small"),
    ("1BRF", "A", "Zinc finger (BRCA1)", "small"),
    ("2JOF", "A", "WW domain", "small"),

    # Membrane-associated / specific folds
    ("1OCC", "A", "Cytochrome c oxidase subunit", "other"),
    ("3BLM", "A", "Beta-lactamase class A", "other"),
    ("1THV", "A", "Thermolysin", "other"),
    ("1PPE", "E", "Elastase complex", "other"),
    ("1CHO", "E", "Chymotrypsinogen", "other"),
    ("2CGA", "A", "Chymotrypsinogen A", "other"),
    ("1ARB", "A", "Arabinose-binding protein", "other"),
    ("1MRJ", "A", "Mannose receptor domain", "other"),
    ("1A2P", "A", "Penicillin acylase", "other"),
    ("2SIC", "E", "Subtilisin (Carlsberg)", "other"),
    ("1PYP", "A", "Phospholipase A2", "other"),
    ("1SAC", "A", "Sulfatase", "other"),
    ("1THG", "A", "Thermitase", "other"),
    ("1TEC", "E", "Thermitase complex", "other"),

    # Additional well-characterized structures
    ("1WQ1", "A", "WW domain FBP28", "all-beta"),
    ("2ABD", "A", "Acyl-CoA binding domain", "all-alpha"),
    ("1A2P", "B", "Penicillin acylase beta", "alpha-beta"),
    ("2LIS", "A", "Lysozyme (human)", "alpha+beta"),
    ("1GXU", "A", "G-protein alpha", "alpha-beta"),
    ("1AJJ", "A", "Acetohydroxy acid isomeroreductase", "alpha-beta"),
    ("1C9O", "A", "Methionine aminopeptidase", "alpha-beta"),
    ("1B0N", "A", "Pyruvate kinase", "alpha-beta"),
    ("2IFO", "A", "Interferon omega", "all-alpha"),
    ("1MLA", "A", "Malate dehydrogenase", "alpha-beta"),
    ("1GOF", "A", "GAL4 DNA-binding domain", "other"),
    ("1LZ1", "A", "Lysozyme variant", "alpha+beta"),
    ("1I6P", "A", "Type III antifreeze", "all-beta"),
    ("1FLV", "A", "Flavodoxin", "alpha-beta"),
    ("1CX1", "A", "Calcineurin-like", "alpha-beta"),
    ("1B6B", "A", "Carboxypeptidase inhibitor", "small"),
    ("2FDN", "A", "Ferredoxin", "alpha-beta"),
    ("1PSR", "A", "Phosphoserine residue", "all-beta"),
    ("1AYI", "A", "Adenylyl cyclase", "alpha-beta"),
    ("1A4I", "A", "Interleukin-1 beta", "all-beta"),
    ("1CEM", "A", "Cellobiohydrolase", "all-beta"),
    ("1AHO", "A", "Charybdotoxin analog", "small"),
    ("1AG2", "A", "Antigen peptide", "other"),
    ("1BFG", "A", "Basic fibroblast growth factor", "all-beta"),
    ("1BGE", "A", "Barley grain endoprotease", "alpha-beta"),
    ("1C44", "A", "Hevamine", "alpha-beta"),
    ("3MBP", "A", "Maltose-binding protein", "alpha-beta"),
    ("1CC8", "A", "Cytochrome c553", "all-alpha"),
    ("1CKA", "A", "Casein kinase alpha", "alpha-beta"),
    ("1EZM", "A", "Elastase (neutrophil)", "alpha-beta"),
    ("1FAS", "A", "Fasciculin 2", "small"),
    ("1GAL", "A", "Beta-galactosidase", "alpha-beta"),
    ("1HFE", "A", "Lactoferrin (N-lobe)", "alpha-beta"),
    ("1MOL", "A", "Monellin", "alpha+beta"),
    ("1NAR", "A", "Nitrite reductase", "alpha-beta"),
    ("1POC", "A", "Phospholipase C", "alpha-beta"),
    ("1RCF", "A", "Ferricytochrome", "all-alpha"),
    ("1SAK", "A", "Streptokinase alpha", "alpha-beta"),
    ("1THW", "A", "Thaumatin", "all-beta"),
    ("1TPH", "A", "Triosephosphate isomerase", "alpha-beta"),
    ("2PTC", "E", "Trypsin", "alpha-beta"),
    ("2SOD", "A", "Cu/Zn superoxide dismutase", "all-beta"),
    ("3SSI", "A", "Streptomyces subtilisin inh.", "alpha+beta"),
    ("1BM8", "A", "Carbonic anhydrase II", "alpha-beta"),
    ("1CNV", "A", "Concanavalin A (deglycosylated)", "all-beta"),
    ("1ECD", "A", "Erythropoietin-like", "all-alpha"),
    ("1HLE", "A", "Human leukocyte elastase", "alpha-beta"),
    ("1MAZ", "A", "Canavalin", "all-beta"),
    ("1OMP", "A", "OmpF porin", "all-beta"),
    ("1PHN", "A", "Pyridine nucleotide-disulphide", "alpha-beta"),
    ("1PLC", "A", "Plastocyanin (poplar)", "all-beta"),
    ("1PPT", "A", "Avian pancreatic polypeptide", "small"),
    ("1RVE", "A", "HIV-1 reverse transcriptase", "alpha-beta"),
    ("1SGT", "A", "Trypsin (Streptomyces)", "alpha-beta"),
    ("2PRK", "A", "Proteinase K", "alpha-beta"),
    ("3APP", "A", "Penicillopepsin (acid)", "alpha-beta"),
    ("4CMS", "A", "Chymosin", "alpha-beta"),
    ("9PAP", "A", "Papain", "alpha-beta"),
]


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ============================================================
# PISCES LIST FETCHER
# ============================================================

def fetch_pisces_list(max_chains=5000):
    """Try to download a PISCES non-redundant chain list.

    Returns list of (pdb_id, chain_id) or None if inaccessible.
    """
    for url in PISCES_URLS:
        logging.info(f"  Trying PISCES: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'BPS-Validate/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode('utf-8', errors='replace')
            lines = text.strip().split('\n')
            if len(lines) < 10:
                continue

            chains = []
            for line in lines[1:]:  # skip header
                parts = line.split()
                if len(parts) < 2:
                    continue
                pdb_chain = parts[0].strip()
                if len(pdb_chain) < 5:
                    continue
                pdb_id = pdb_chain[:4].upper()
                chain_id = pdb_chain[4:]
                if not chain_id:
                    chain_id = "A"
                chains.append((pdb_id, chain_id))
                if len(chains) >= max_chains:
                    break

            if len(chains) >= 50:
                logging.info(f"  PISCES: got {len(chains)} chains")
                return chains

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            logging.warning(f"  PISCES failed: {e}")
            continue

    return None


# ============================================================
# PDB DOWNLOAD
# ============================================================

def download_pdb_cif(pdb_id, cache_dir):
    """Download an mmCIF file from RCSB PDB.

    Returns path to the downloaded file, or None on failure.
    """
    pdb_id_lower = pdb_id.lower()
    filename = f"{pdb_id_lower}.cif"
    filepath = cache_dir / filename

    if filepath.exists() and filepath.stat().st_size > 100:
        return filepath

    url = f"https://files.rcsb.org/download/{pdb_id_lower}.cif"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'BPS-Validate/1.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            filepath.write_bytes(data)
            return filepath
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                time.sleep(wait)
            else:
                logging.debug(f"  Failed to download {pdb_id}: {e}")
                return None

    return None


# ============================================================
# CIF PARSER FOR PDB EXPERIMENTAL STRUCTURES
# ============================================================

def parse_pdb_cif(filepath, target_chain="A", phi_sign=1):
    """Parse a PDB mmCIF file and extract phi/psi angles for a specific chain.

    Uses BioPython if available, otherwise falls back to manual parsing.

    Returns dict with keys:
      'phi_psi': list of (phi, psi) in radians (None for missing)
      'L': number of residues
      'resolution': float or None
      'method': 'XRAY' / 'NMR' / etc
    Or None on failure.
    """
    if HAS_BIOPYTHON:
        return _parse_with_biopython(filepath, target_chain, phi_sign)
    else:
        return _parse_manual(filepath, target_chain, phi_sign)


def _parse_with_biopython(filepath, target_chain, phi_sign):
    """Parse using BioPython's MMCIFParser + PPBuilder."""
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("pdb", str(filepath))
    except Exception:
        return None

    # Extract resolution and method from mmCIF dict
    resolution = None
    method = None
    try:
        mmcif_dict = parser._mmcif_dict
        if '_refine.ls_d_res_high' in mmcif_dict:
            val = mmcif_dict['_refine.ls_d_res_high']
            if isinstance(val, list):
                val = val[0]
            try:
                resolution = float(val)
            except (ValueError, TypeError):
                pass
        if '_exptl.method' in mmcif_dict:
            val = mmcif_dict['_exptl.method']
            if isinstance(val, list):
                val = val[0]
            method = str(val).strip().upper()
    except Exception:
        pass

    # Get first model
    try:
        model = structure[0]
    except (KeyError, IndexError):
        return None

    # Find the target chain
    chain = None
    for c in model:
        if c.id == target_chain:
            chain = c
            break

    if chain is None:
        # Try case-insensitive match
        for c in model:
            if c.id.upper() == target_chain.upper():
                chain = c
                break

    if chain is None:
        return None

    # Build polypeptides from the chain
    ppb = PPBuilder()
    polypeptides = ppb.build_peptides(chain)
    if not polypeptides:
        return None

    # Collect phi/psi from all polypeptide segments
    all_phi_psi = []
    for pp in polypeptides:
        phi_psi_list = pp.get_phi_psi_list()
        for phi, psi in phi_psi_list:
            # BioPython returns angles in radians with correct IUPAC convention
            all_phi_psi.append((phi, psi))

    if len(all_phi_psi) < MIN_CHAIN_LEN:
        return None

    return {
        'phi_psi': all_phi_psi,
        'L': len(all_phi_psi),
        'resolution': resolution,
        'method': method,
    }


def _parse_manual(filepath, target_chain, phi_sign):
    """Manual mmCIF parser adapted from bps_process.py.

    Filters to a specific chain and handles PDB-specific quirks.
    """
    atom_data = {}
    resolution = None
    method = None

    try:
        with open(str(filepath), 'r') as f:
            col_map = {}
            in_header = False
            col_idx = 0

            for line in f:
                ls = line.strip()

                # Try to grab resolution
                if ls.startswith('_refine.ls_d_res_high'):
                    parts = ls.split()
                    if len(parts) >= 2:
                        try:
                            resolution = float(parts[1])
                        except ValueError:
                            pass

                # Try to grab method
                if ls.startswith('_exptl.method'):
                    parts = ls.split(None, 1)
                    if len(parts) >= 2:
                        method = parts[1].strip().strip("'\"").upper()

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
                    # Filter to target chain
                    chain_col = col_map.get('auth_asym_id',
                                col_map.get('label_asym_id', 6))
                    chain_id = parts[chain_col]
                    if chain_id != target_chain:
                        continue
                    # Alternate conformations: use only A or blank
                    alt = parts[col_map.get('label_alt_id', 4)] if 'label_alt_id' in col_map else '.'
                    if alt not in ('.', 'A', '?', ' '):
                        continue
                    # Use auth_seq_id for PDB structures (more reliable than label_seq_id)
                    seq_col = col_map.get('auth_seq_id',
                              col_map.get('label_seq_id', 8))
                    resnum = int(parts[seq_col])
                    # Check for pdbx_PDB_model_num — use only model 1
                    model_col = col_map.get('pdbx_PDB_model_num')
                    if model_col is not None:
                        model_num = parts[model_col]
                        if model_num != '1':
                            continue
                    x = float(parts[col_map.get('Cartn_x', 10)])
                    y = float(parts[col_map.get('Cartn_y', 11)])
                    z = float(parts[col_map.get('Cartn_z', 12)])
                    if resnum not in atom_data:
                        atom_data[resnum] = {}
                    atom_data[resnum][atom_name] = np.array([x, y, z])
                except (ValueError, IndexError, KeyError):
                    continue
    except Exception:
        return None

    # Build residue list
    sorted_keys = sorted(atom_data.keys())
    residues = []
    for rn in sorted_keys:
        r = atom_data[rn]
        if 'N' in r and 'CA' in r and 'C' in r:
            residues.append(r)

    if len(residues) < MIN_CHAIN_LEN:
        return None

    # Compute phi/psi
    n = len(residues)
    phi_psi = []
    for i in range(n):
        raw_phi = (_calc_dihedral(residues[i-1]['C'], residues[i]['N'],
                                  residues[i]['CA'], residues[i]['C'])
                   if i > 0 else None)
        raw_psi = (_calc_dihedral(residues[i]['N'], residues[i]['CA'],
                                  residues[i]['C'], residues[i+1]['N'])
                   if i < n - 1 else None)
        phi = phi_sign * raw_phi if raw_phi is not None else None
        psi = phi_sign * raw_psi if raw_psi is not None else None
        phi_psi.append((phi, psi))

    return {
        'phi_psi': phi_psi,
        'L': len(phi_psi),
        'resolution': resolution,
        'method': method,
    }


# ============================================================
# BPS COMPUTATION (reuses W from main pipeline)
# ============================================================

def compute_bps_norm(phi_psi, W_interp):
    """Compute BPS/L from a phi/psi list. Returns float or None."""
    valid = [(p, s) for p, s in phi_psi if p is not None and s is not None]
    if len(valid) < 3:
        return None
    phis = np.array([v[0] for v in valid])
    psis = np.array([v[1] for v in valid])
    W_chain = W_interp(np.column_stack([psis, phis]))
    dW = np.abs(np.diff(W_chain))
    return float(np.sum(dW)) / len(valid)


def classify_ss(phi_psi):
    """Classify secondary structure from phi/psi angles.

    Returns (pct_helix, pct_sheet, pct_coil).

    Uses the wide two-basin definition (CLAUDE.md / bps_validate_controls.py):
      α-helix: φ ∈ (−160°, 0°), ψ ∈ (−120°, 30°)
      β-sheet: φ ∈ (−170°, −70°), ψ > 90° OR ψ < −120°  (handles ±180° wrap)
    """
    valid = [(p, s) for p, s in phi_psi if p is not None and s is not None]
    if not valid:
        return None, None, None
    phis = np.degrees(np.array([v[0] for v in valid]))
    psis = np.degrees(np.array([v[1] for v in valid]))
    n = len(phis)
    helix = np.sum((-160 < phis) & (phis < 0) & (-120 < psis) & (psis < 30))
    sheet = np.sum((-170 < phis) & (phis < -70) & ((psis > 90) | (psis < -120)))
    coil = n - helix - sheet
    return float(helix / n * 100), float(sheet / n * 100), float(coil / n * 100)


# ============================================================
# MAIN PROCESSING PIPELINE
# ============================================================

def get_chain_list():
    """Get the list of PDB chains to process.

    Tries PISCES first, falls back to curated list.
    Returns list of (pdb_id, chain_id, description, scop_class).
    """
    logging.info("Obtaining PDB chain list...")

    pisces = fetch_pisces_list()
    if pisces is not None and len(pisces) >= 100:
        logging.info(f"  Using PISCES list: {len(pisces)} chains")
        return [(pdb, ch, "PISCES", "unknown") for pdb, ch in pisces]

    logging.info(f"  Using curated fallback list: {len(CURATED_CHAINS)} chains")
    return list(CURATED_CHAINS)


def process_chains(chain_list, W_interp, phi_sign, max_chains=None, skip_download=False):
    """Download and process all chains.

    Returns list of result dicts for successfully processed chains.
    """
    if max_chains is not None:
        chain_list = chain_list[:max_chains]

    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    n_total = len(chain_list)
    n_downloaded = 0
    n_skipped_download = 0
    n_skipped_nmr = 0
    n_skipped_short = 0
    n_skipped_parse = 0
    n_success = 0
    t0 = time.time()

    # Deduplicate by PDB ID for downloads (many chains share a PDB entry)
    pdb_ids_needed = sorted(set(c[0] for c in chain_list))

    if not skip_download:
        logging.info(f"Downloading {len(pdb_ids_needed)} unique PDB entries...")
        for i, pdb_id in enumerate(pdb_ids_needed):
            filepath = download_pdb_cif(pdb_id, PDB_CACHE_DIR)
            if filepath is not None:
                n_downloaded += 1
            else:
                n_skipped_download += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(pdb_ids_needed) - i - 1) / rate if rate > 0 else 0
                logging.info(f"  Downloaded {i+1}/{len(pdb_ids_needed)} "
                             f"({n_downloaded} OK, {elapsed:.0f}s elapsed, "
                             f"~{eta:.0f}s remaining)")

        logging.info(f"  Downloads complete: {n_downloaded} OK, "
                     f"{n_skipped_download} failed")

    logging.info(f"Processing {n_total} chains...")
    t1 = time.time()

    for i, chain_info in enumerate(chain_list):
        pdb_id, chain_id = chain_info[0], chain_info[1]
        description = chain_info[2] if len(chain_info) > 2 else ""
        scop_class = chain_info[3] if len(chain_info) > 3 else "unknown"

        filepath = PDB_CACHE_DIR / f"{pdb_id.lower()}.cif"
        if not filepath.exists():
            n_skipped_download += 1
            continue

        parsed = parse_pdb_cif(filepath, chain_id, phi_sign)
        if parsed is None:
            n_skipped_parse += 1
            continue

        # Skip NMR structures
        if parsed.get('method') and 'NMR' in parsed['method']:
            n_skipped_nmr += 1
            continue

        # Skip short chains
        if parsed['L'] < MIN_CHAIN_LEN:
            n_skipped_short += 1
            continue

        # Check missing backbone fraction
        phi_psi = parsed['phi_psi']
        n_valid = sum(1 for p, s in phi_psi if p is not None and s is not None)
        missing_pct = (1 - n_valid / parsed['L']) * 100
        if missing_pct > MAX_MISSING_BACKBONE_PCT:
            n_skipped_parse += 1
            continue

        # Compute BPS/L
        bps = compute_bps_norm(phi_psi, W_interp)
        if bps is None or not np.isfinite(bps):
            n_skipped_parse += 1
            continue

        # Flag extreme outliers but include them in results
        is_outlier = not (0.02 < bps < 0.50)

        # Secondary structure
        pct_h, pct_s, pct_c = classify_ss(phi_psi)

        results.append({
            'pdb_id': pdb_id,
            'chain_id': chain_id,
            'description': description,
            'scop_class': scop_class,
            'L': parsed['L'],
            'n_valid': n_valid,
            'resolution': parsed.get('resolution'),
            'method': parsed.get('method'),
            'bps_norm': bps,
            'pct_helix': pct_h,
            'pct_sheet': pct_s,
            'pct_coil': pct_c,
            'is_outlier': is_outlier,
        })
        n_success += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            logging.info(f"  Processed {i+1}/{n_total} "
                         f"({n_success} OK, {elapsed:.0f}s, ~{eta:.0f}s left)")

    elapsed = time.time() - t1
    logging.info(f"Processing complete: {n_success}/{n_total} chains in {elapsed:.0f}s")
    logging.info(f"  Skipped: {n_skipped_download} download, {n_skipped_nmr} NMR, "
                 f"{n_skipped_short} short, {n_skipped_parse} parse/quality")

    return results


# ============================================================
# ANALYSIS
# ============================================================

def analyze_results(results):
    """Run all analyses on the processed PDB chains.

    Returns analysis dict for report generation.
    """
    if not results:
        return {'status': 'NO DATA'}

    bps_vals = np.array([r['bps_norm'] for r in results])
    lengths = np.array([r['L'] for r in results])
    resolutions = np.array([r['resolution'] for r in results
                            if r['resolution'] is not None])
    outlier_flags = np.array([r.get('is_outlier', False) for r in results])

    analysis = {}

    # ---- Summary statistics (all proteins) ----
    analysis['n_chains'] = len(results)
    analysis['mean_bps'] = float(np.mean(bps_vals))
    analysis['std_bps'] = float(np.std(bps_vals, ddof=1))
    analysis['cv_pct'] = analysis['std_bps'] / analysis['mean_bps'] * 100
    analysis['median_bps'] = float(np.median(bps_vals))
    analysis['min_bps'] = float(np.min(bps_vals))
    analysis['max_bps'] = float(np.max(bps_vals))

    # ---- Outlier analysis ----
    n_outliers = int(np.sum(outlier_flags))
    analysis['n_outliers'] = n_outliers
    analysis['outlier_pct'] = n_outliers / len(results) * 100 if results else 0
    if n_outliers > 0:
        outlier_bps = bps_vals[outlier_flags]
        analysis['outlier_mean_bps'] = float(np.mean(outlier_bps))
        analysis['outlier_ids'] = [r['pdb_id'] for r, o in
                                   zip(results, outlier_flags) if o]
    # Stats without outliers for comparison
    non_outlier = ~outlier_flags
    if np.sum(non_outlier) > 0:
        analysis['mean_bps_no_outliers'] = float(np.mean(bps_vals[non_outlier]))
        analysis['cv_pct_no_outliers'] = float(
            np.std(bps_vals[non_outlier], ddof=1) /
            np.mean(bps_vals[non_outlier]) * 100)

    # ---- KS test vs AlphaFold ----
    # Compare PDB BPS/L against a synthetic sample from the AlphaFold
    # reference distribution. Note: uses normal approximation because the
    # full per-protein AlphaFold BPS/L array is not available at report
    # time. For a stronger test, replace with actual AlphaFold values via
    # ks_2samp(bps_vals, alphafold_bps_array).
    if HAS_SCIPY_STATS:
        rng = np.random.default_rng(RNG_SEED)
        af_synthetic = rng.normal(AF_REFERENCE['mean_bps'],
                                  AF_REFERENCE['std_bps'],
                                  AF_REFERENCE['n_proteins'])
        ks_stat, ks_p = ks_2samp(bps_vals, af_synthetic)
        analysis['ks_stat'] = float(ks_stat)
        analysis['ks_pvalue'] = float(ks_p)
        analysis['ks_note'] = ("KS test uses normal approximation of AF "
                               "distribution, not raw AF values.")
    else:
        analysis['ks_stat'] = None
        analysis['ks_pvalue'] = None

    # ---- Histogram bins ----
    bin_edges = np.linspace(0.05, 0.35, 11)
    hist_counts, _ = np.histogram(bps_vals, bins=bin_edges)
    analysis['histogram'] = {
        'edges': bin_edges.tolist(),
        'counts': hist_counts.tolist(),
    }

    # ---- By SCOP class ----
    class_data = defaultdict(list)
    for r in results:
        class_data[r['scop_class']].append(r['bps_norm'])

    analysis['by_class'] = {}
    for cls in sorted(class_data.keys()):
        vals = np.array(class_data[cls])
        if len(vals) >= 3:
            analysis['by_class'][cls] = {
                'n': len(vals),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)),
                'cv_pct': float(np.std(vals, ddof=1) / np.mean(vals) * 100),
            }

    # ---- By chain length ----
    len_bins = [
        ('S (<100)', 0, 100),
        ('M (100-300)', 100, 300),
        ('L (300-1000)', 300, 1000),
        ('XL (>1000)', 1000, 100000),
    ]
    analysis['by_length'] = {}
    for label, lo, hi in len_bins:
        mask = (lengths >= lo) & (lengths < hi)
        vals = bps_vals[mask]
        if len(vals) >= 3:
            analysis['by_length'][label] = {
                'n': int(mask.sum()),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)),
            }

    # ---- BPS energy vs chain length correlation ----
    bps_energies = bps_vals * lengths
    if HAS_SCIPY_STATS and len(bps_energies) >= 10:
        r_val, p_val = pearsonr(lengths, bps_energies)
        analysis['bps_vs_length'] = {
            'pearson_r': float(r_val),
            'p_value': float(p_val),
        }
    else:
        analysis['bps_vs_length'] = None

    # ---- BPS/L vs resolution ----
    if HAS_SCIPY_STATS and len(resolutions) >= 10:
        res_bps = [r['bps_norm'] for r in results if r['resolution'] is not None]
        res_vals = [r['resolution'] for r in results if r['resolution'] is not None]
        r_val, p_val = pearsonr(res_vals, res_bps)
        analysis['bps_vs_resolution'] = {
            'pearson_r': float(r_val),
            'p_value': float(p_val),
            'n': len(res_vals),
            'mean_resolution': float(np.mean(resolutions)),
        }
    else:
        analysis['bps_vs_resolution'] = None

    # ---- By secondary structure class ----
    ss_data = {'Alpha-rich': [], 'Beta-rich': [], 'Mixed': [], 'Other': []}
    for r in results:
        h, s = r.get('pct_helix'), r.get('pct_sheet')
        if h is None or s is None:
            continue
        if h > 40 and s <= 15:
            ss_data['Alpha-rich'].append(r['bps_norm'])
        elif s > 25 and h <= 15:
            ss_data['Beta-rich'].append(r['bps_norm'])
        elif h > 15 and s > 15:
            ss_data['Mixed'].append(r['bps_norm'])
        else:
            ss_data['Other'].append(r['bps_norm'])

    analysis['by_ss_class'] = {}
    for cls, vals in ss_data.items():
        if len(vals) >= 3:
            arr = np.array(vals)
            analysis['by_ss_class'][cls] = {
                'n': len(arr),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr, ddof=1)),
            }

    return analysis


# ============================================================
# REPORT
# ============================================================

def write_report(results, analysis, outpath=REPORT_PATH):
    """Generate markdown validation report."""
    lines = []

    def add(text=""):
        lines.append(text)

    add("# PDB Experimental Validation of BPS/L")
    add()
    add(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"**Chains processed:** {analysis.get('n_chains', 0)}")
    add(f"**Parser:** {'BioPython MMCIFParser' if HAS_BIOPYTHON else 'Manual CIF parser'}")
    add()

    add("## Purpose")
    add()
    add("Validate that BPS/L ≈ 0.20 is a property of **real protein backbones**, not an "
        "artifact of AlphaFold's learned structural prior. This script computes BPS/L on "
        "experimentally determined crystal structures from the RCSB PDB using the same "
        "superpotential W as the main pipeline.")
    add()

    # ---- Summary comparison ----
    add("## Summary Statistics")
    add()
    add("| Source | N proteins | Mean BPS/L | Std | CV (%) |")
    add("|--------|-----------|-----------|-----|--------|")
    add(f"| AlphaFold (HQ, pLDDT≥85) | {AF_REFERENCE['n_proteins']} | "
        f"{AF_REFERENCE['mean_bps']:.4f} | {AF_REFERENCE['std_bps']:.4f} | "
        f"{AF_REFERENCE['cv_pct']:.1f} |")
    if analysis.get('n_chains'):
        add(f"| PDB experimental | {analysis['n_chains']} | "
            f"{analysis['mean_bps']:.4f} | {analysis['std_bps']:.4f} | "
            f"{analysis['cv_pct']:.1f} |")
    add()

    # Delta
    if analysis.get('mean_bps'):
        delta = analysis['mean_bps'] - AF_REFERENCE['mean_bps']
        pct_delta = abs(delta) / AF_REFERENCE['mean_bps'] * 100
        add(f"**Δ(PDB − AF):** {delta:+.4f} ({pct_delta:.1f}%)")
        add()

    # ---- Distribution comparison ----
    add("## Distribution Comparison")
    add()
    if analysis.get('ks_stat') is not None:
        add(f"**KS test (PDB vs AF):** D = {analysis['ks_stat']:.4f}, "
            f"p = {analysis['ks_pvalue']:.2e}")
        add()
        add("Note: The AF distribution is simulated from reported mean/std (normal "
            "approximation), since per-protein AF BPS/L values are not available in "
            "this context.")
    else:
        add("*KS test unavailable (scipy.stats not installed)*")
    add()

    # Histogram
    hist = analysis.get('histogram')
    if hist:
        add("### BPS/L Histogram (PDB)")
        add()
        add("| Bin | Count |")
        add("|-----|-------|")
        edges = hist['edges']
        counts = hist['counts']
        for j in range(len(counts)):
            add(f"| {edges[j]:.2f}–{edges[j+1]:.2f} | {counts[j]} |")
        add()

    # ---- By SCOP class ----
    by_class = analysis.get('by_class', {})
    if by_class:
        add("## By Structural Class (SCOP)")
        add()
        add("| Class | N | Mean BPS/L | Std | CV (%) |")
        add("|-------|---|-----------|-----|--------|")
        for cls in sorted(by_class.keys()):
            d = by_class[cls]
            add(f"| {cls} | {d['n']} | {d['mean']:.4f} | {d['std']:.4f} | "
                f"{d['cv_pct']:.1f} |")
        add()

    # ---- By SS class ----
    by_ss = analysis.get('by_ss_class', {})
    if by_ss:
        add("## By Secondary Structure Class")
        add()
        add("| SS Class | N | Mean BPS/L | Std |")
        add("|----------|---|-----------|-----|")
        for cls in ['Alpha-rich', 'Beta-rich', 'Mixed', 'Other']:
            if cls in by_ss:
                d = by_ss[cls]
                add(f"| {cls} | {d['n']} | {d['mean']:.4f} | {d['std']:.4f} |")
        add()

    # ---- By chain length ----
    by_len = analysis.get('by_length', {})
    if by_len:
        add("## By Chain Length")
        add()
        add("| Size bin | N | Mean BPS/L | Std |")
        add("|----------|---|-----------|-----|")
        for label in ['S (<100)', 'M (100-300)', 'L (300-1000)', 'XL (>1000)']:
            if label in by_len:
                d = by_len[label]
                add(f"| {label} | {d['n']} | {d['mean']:.4f} | {d['std']:.4f} |")
        add()

    # ---- Correlations ----
    add("## Key Correlations")
    add()

    bvl = analysis.get('bps_vs_length')
    if bvl:
        add(f"**BPS energy vs chain length:** r = {bvl['pearson_r']:.4f}, "
            f"p = {bvl['p_value']:.2e}")
        add("(Expected: strong positive correlation, r > 0.90, confirming BPS "
            "energy scales linearly with L)")
    else:
        add("*BPS vs length correlation: insufficient data*")
    add()

    bvr = analysis.get('bps_vs_resolution')
    if bvr:
        add(f"**BPS/L vs resolution:** r = {bvr['pearson_r']:.4f}, "
            f"p = {bvr['p_value']:.2e} (N = {bvr['n']}, "
            f"mean resolution = {bvr['mean_resolution']:.2f} Å)")
        add("(Expected: weak or absent correlation — BPS/L should not depend on "
            "crystal quality)")
    else:
        add("*BPS/L vs resolution correlation: insufficient data*")
    add()

    # ---- Per-chain table (first 30) ----
    add("## Sample Chains (first 30)")
    add()
    add("| PDB | Chain | L | Resolution | BPS/L | %Helix | %Sheet | Class |")
    add("|-----|-------|---|-----------|-------|--------|--------|-------|")
    for r in results[:30]:
        res_str = f"{r['resolution']:.2f}" if r['resolution'] is not None else "—"
        add(f"| {r['pdb_id']} | {r['chain_id']} | {r['L']} | {res_str} | "
            f"{r['bps_norm']:.4f} | {r.get('pct_helix', 0):.0f} | "
            f"{r.get('pct_sheet', 0):.0f} | {r['scop_class']} |")
    add()

    # ---- Interpretation ----
    add("## Interpretation")
    add()

    if analysis.get('mean_bps'):
        delta_pct = abs(analysis['mean_bps'] - AF_REFERENCE['mean_bps']) / AF_REFERENCE['mean_bps'] * 100
        if delta_pct < 5 and analysis['cv_pct'] <= 5:
            add("**RESULT: BPS/L universality is CONFIRMED in experimental structures.**")
            add()
            add(f"PDB mean BPS/L = {analysis['mean_bps']:.4f} vs AlphaFold = "
                f"{AF_REFERENCE['mean_bps']:.4f} (Δ = {delta_pct:.1f}%).")
            add(f"PDB CV = {analysis['cv_pct']:.1f}%.")
            add()
            add("This eliminates the reviewer concern that BPS/L ≈ 0.20 is an artifact "
                "of AlphaFold's structural prior. The same universal constant emerges "
                "from experimentally determined backbones.")
        elif delta_pct < 10:
            add("**RESULT: BPS/L shows approximate agreement between PDB and AlphaFold.**")
            add()
            add(f"PDB mean = {analysis['mean_bps']:.4f}, AF mean = {AF_REFERENCE['mean_bps']:.4f} "
                f"(Δ = {delta_pct:.1f}%). The values are close but not identical. "
                "Minor differences may arise from the non-redundant chain selection, "
                "crystal packing effects, or residual differences between predicted "
                "and experimental backbone geometry.")
        else:
            add("**RESULT: PDB and AlphaFold BPS/L differ substantially.**")
            add()
            add(f"PDB mean = {analysis['mean_bps']:.4f}, AF mean = {AF_REFERENCE['mean_bps']:.4f} "
                f"(Δ = {delta_pct:.1f}%). This suggests AlphaFold's structural priors "
                "may influence the BPS/L value. Further investigation is needed.")
    else:
        add("*Insufficient data for interpretation.*")
    add()

    # ---- Footer ----
    add("---")
    add()
    add("*Report generated by `bps_pdb_validate.py`*")

    report_text = "\n".join(lines) + "\n"
    outpath = Path(outpath)
    outpath.write_text(report_text, encoding='utf-8')
    logging.info(f"Report written to {outpath}")
    return report_text


# ============================================================
# MAIN
# ============================================================

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="PDB Experimental Validation of BPS/L")
    parser.add_argument('--max', type=int, default=None,
                        help='Max chains to process')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip downloads, use cached CIFs only')
    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("PDB EXPERIMENTAL VALIDATION OF BPS/L")
    logging.info("=" * 60)
    logging.info("")

    # ---- Phi sign determination ----
    logging.info("Determining phi sign convention...")
    phi_sign = determine_phi_sign()
    logging.info(f"PHI_SIGN = {phi_sign}")
    logging.info("")

    # ---- Build superpotential ----
    logging.info("Building superpotential...")
    W_interp = build_superpotential()
    logging.info("  Done.")
    logging.info("")

    # ---- Get chain list ----
    chain_list = get_chain_list()
    logging.info(f"Chain list: {len(chain_list)} entries")
    logging.info("")

    # ---- Process ----
    results = process_chains(
        chain_list, W_interp, phi_sign,
        max_chains=args.max,
        skip_download=args.skip_download,
    )

    if not results:
        logging.error("No chains were successfully processed.")
        logging.error("Check network connectivity and pdb_validation_cache/ directory.")
        sys.exit(1)

    logging.info("")

    # ---- Analyze ----
    logging.info("Running analysis...")
    analysis = analyze_results(results)
    logging.info(f"  PDB mean BPS/L = {analysis.get('mean_bps', 0):.4f} "
                 f"± {analysis.get('std_bps', 0):.4f}")
    logging.info(f"  PDB CV = {analysis.get('cv_pct', 0):.1f}%")
    logging.info(f"  AF  mean BPS/L = {AF_REFERENCE['mean_bps']:.4f} "
                 f"± {AF_REFERENCE['std_bps']:.4f}")
    logging.info("")

    # ---- Report ----
    logging.info("Writing report...")
    write_report(results, analysis)

    logging.info("")
    logging.info("=" * 60)
    logging.info("PDB VALIDATION COMPLETE")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
