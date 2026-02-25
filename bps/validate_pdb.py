"""PDB experimental validation of BPS/L.

Downloads high-resolution PDB structures, computes BPS/L for each, and
reports statistics. Compares to the AlphaFold target: 0.202 +/- 0.004.

Target result: PDB mean BPS/L ~ 0.207 +/- 0.05.

Three data-source tiers:
  1. RCSB Search API (resolution < 1.8A, X-ray, protein)
  2. Curated fallback list of ~100 well-known high-resolution structures
  3. Synthetic protein-like chains (for offline/network-blocked environments)

Fold-class labels are NEVER hardcoded -- SS composition is computed from
(phi,psi) via assign_basin().
"""

import json
import math
import os
import sys
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

import numpy as np

from bps.superpotential import (
    assign_basin,
    build_superpotential,
    compute_bps,
    compute_bps_per_residue,
    lookup_W_batch,
)
from bps.extract import (
    compute_basin_fractions,
    download_pdb,
    extract_dihedrals_pdb,
)

# ---------------------------------------------------------------------------
# Curated fallback list: ~100 well-known high-resolution structures
# (PDB ID, chain, expected approx length, short description)
# ---------------------------------------------------------------------------
_CURATED_LIST = [
    # Small proteins
    ("1UBQ", "A", 76),   ("1L2Y", "A", 20),   ("1VII", "A", 36),
    ("2RHE", "A", 114),  ("1CRN", "A", 46),   ("1ENH", "A", 54),
    ("2CI2", "I", 65),   ("1PGA", "A", 56),    ("1GB1", "A", 56),
    ("2PTH", "A", 193),  ("1HHP", "A", 99),    ("3BLM", "A", 257),
    # All-alpha
    ("1MBN", "A", 153),  ("2HHB", "A", 141),  ("1HRC", "A", 105),
    ("1LMB", "3", 87),   ("1UTG", "A", 70),   ("1R69", "A", 63),
    ("2WRP", "R", 104),  ("1BGE", "A", 159),  ("256B", "A", 106),
    ("1FLV", "A", 170),  ("1MBC", "A", 153),  ("4HHB", "A", 141),
    # All-beta
    ("1REI", "A", 107),  ("2AIT", "A", 74),   ("1TEN", "A", 90),
    ("1TIT", "A", 89),   ("1FNF", "A", 94),   ("1TTF", "A", 117),
    ("7RSA", "A", 124),  ("1BRS", "A", 110),  ("2SNS", "A", 149),
    ("3FGF", "A", 155),  ("1CDT", "A", 120),  ("1CTF", "A", 68),
    # Alpha/beta
    ("1LYZ", "A", 129),  ("4LYZ", "A", 129),  ("1AKE", "A", 214),
    ("1TIM", "A", 247),  ("1PII", "A", 327),  ("1SN3", "A", 65),
    ("2ACT", "A", 218),  ("1XNB", "A", 185),  ("1THW", "A", 185),
    ("3CLA", "A", 213),  ("1CSE", "E", 274),  ("1RIS", "A", 97),
    # Classic test cases
    ("2LZM", "A", 164),  ("3LZM", "A", 164),  ("1HEL", "A", 129),
    ("1OVA", "A", 385),  ("1PKA", "A", 240),  ("9PAP", "A", 212),
    ("5CHA", "E", 245),  ("2GCH", "A", 245),  ("1TRY", "A", 223),
    ("3APP", "A", 323),  ("2CPP", "A", 405),  ("1PHH", "A", 394),
    # High-resolution
    ("1EJG", "A", 144),  ("1GCI", "A", 269),  ("1MOL", "A", 94),
    ("1AHO", "A", 64),   ("1C75", "A", 71),   ("1BPI", "A", 58),
    ("1IGD", "A", 61),   ("4PTI", "A", 58),   ("1PPT", "A", 36),
    ("1ROP", "A", 63),   ("1SAP", "A", 66),   ("3SDH", "A", 156),
    ("5RXN", "A", 54),   ("1FRD", "A", 98),   ("1RCB", "A", 129),
    # More mixed
    ("1CHD", "A", 198),  ("2TRX", "A", 108),  ("3PGK", "A", 415),
    ("1ADS", "A", 315),  ("1FCB", "A", 173),  ("2SIL", "A", 113),
    ("1PLQ", "A", 99),   ("1AAP", "A", 56),   ("1ARB", "A", 263),
    ("1GEN", "A", 227),  ("1FKJ", "A", 107),  ("1MRJ", "A", 101),
    ("1ECA", "A", 136),  ("1A2P", "A", 108),  ("5P21", "A", 166),
    ("2IG2", "A", 167),  ("1RNH", "A", 155),  ("1THX", "A", 108),
    ("2FXB", "A", 81),   ("1RBP", "A", 182),  ("1POA", "A", 83),
    ("3EBX", "A", 62),   ("1UXD", "A", 303),  ("1EZM", "A", 301),
    ("2AK3", "A", 226),  ("1PHT", "A", 338),
]


def query_rcsb_search(max_results: int = 200) -> List[Tuple[str, str]]:
    """Query RCSB Search API for high-resolution X-ray protein structures.

    Returns list of (pdb_id, chain_id) tuples.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less",
                        "value": 1.8,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.experimental_method",
                        "operator": "exact_match",
                        "value": "X-ray",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [
                {
                    "sort_by": "rcsb_entry_info.resolution_combined",
                    "direction": "asc",
                }
            ],
        },
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    data = json.dumps(query).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        entries = []
        for hit in result.get("result_set", []):
            pdb_id = hit["identifier"].upper()
            entries.append((pdb_id, "A"))  # default to chain A
        return entries
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  RCSB Search API failed: {e}")
        return []


def _generate_synthetic_proteins(
    n_proteins: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Generate synthetic protein-like (phi,psi) sequences for offline testing.

    Creates chains with coherent secondary structure segments (alpha-helix,
    beta-sheet, coil/loop) to mimic real protein backbone distributions.
    Returns a list of dicts with 'name', 'residues', 'length' keys.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Basin centers (degrees) and per-residue jitter (std in degrees).
    # Real proteins have very tight intra-basin distributions: the 0.10
    # gap between Markov and real BPS/L comes from 97% suppression of
    # intra-basin roughness. Use small jitter to model this.
    basin_params = {
        "alpha": (-63.0, -43.0, 3.0, 3.0),   # mu_phi, mu_psi, std_phi, std_psi
        "beta":  (-120.0, 130.0, 4.0, 4.0),
        "coil":  (-80.0, 60.0, 15.0, 15.0),
    }

    proteins = []
    for i in range(n_proteins):
        # Random chain length 80-350 (avoid very short chains)
        length = rng.integers(80, 351)

        # Generate SS segment layout with autocorrelated angles.
        # Within each SS segment, each residue's angles are a small
        # perturbation from the previous one (random walk within basin),
        # not independent draws. This models the conformational coherence
        # that gives real proteins BPS/L ~ 0.20.
        residues = []
        pos = 0
        prev_phi_deg = None
        prev_psi_deg = None

        while pos < length:
            # Pick SS type with realistic proportions
            r = rng.random()
            if r < 0.40:
                ss_type = "alpha"
                seg_len = rng.integers(6, 25)
            elif r < 0.70:
                ss_type = "beta"
                seg_len = rng.integers(4, 14)
            else:
                ss_type = "coil"
                seg_len = rng.integers(2, 6)

            seg_len = min(seg_len, length - pos)
            mu_phi, mu_psi, std_phi, std_psi = basin_params[ss_type]

            # Start of segment: draw from basin center
            seg_phi = mu_phi + rng.normal(0, std_phi)
            seg_psi = mu_psi + rng.normal(0, std_psi)

            for j in range(seg_len):
                if j > 0:
                    # Autocorrelated walk: small step from previous position
                    seg_phi += rng.normal(0, std_phi * 0.3)
                    seg_psi += rng.normal(0, std_psi * 0.3)

                residues.append({
                    "resnum": pos + 1,
                    "resname": "ALA",
                    "phi": math.radians(seg_phi) if pos > 0 else None,
                    "psi": math.radians(seg_psi) if pos < length - 1 else None,
                })
                prev_phi_deg = seg_phi
                prev_psi_deg = seg_psi
                pos += 1

        proteins.append({
            "name": f"SYN_{i+1:03d}",
            "residues": residues,
            "length": length,
        })

    return proteins


def _is_nmr_structure(pdb_path: str) -> bool:
    """Detect NMR structures by checking for multiple MODEL records."""
    model_count = 0
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("MODEL "):
                    model_count += 1
                    if model_count >= 2:
                        return True
    except OSError:
        pass
    return False


def process_single_protein(
    pdb_id: str,
    chain_id: str,
    W_grid: np.ndarray,
    phi_grid: np.ndarray,
    psi_grid: np.ndarray,
    cache_dir: str,
) -> Optional[Dict]:
    """Download, extract dihedrals, compute BPS/L for one PDB structure.

    Returns a result dict or None on failure. The result dict includes
    'is_nmr' flag for downstream quality filtering.
    """
    pdb_path = download_pdb(pdb_id, cache_dir=cache_dir)
    if pdb_path is None:
        return None

    is_nmr = _is_nmr_structure(pdb_path)

    try:
        residues = extract_dihedrals_pdb(pdb_path, chain_id=chain_id)
    except (ValueError, Exception) as e:
        print(f"  {pdb_id}: extraction failed: {e}")
        return None

    if len(residues) < 20:
        print(f"  {pdb_id}: too short ({len(residues)} residues), skipping")
        return None

    result = _compute_protein_stats(pdb_id, residues, W_grid, phi_grid, psi_grid)
    if result is not None:
        result["is_nmr"] = is_nmr
    return result


def _compute_protein_stats(
    name: str,
    residues: List[Dict],
    W_grid: np.ndarray,
    phi_grid: np.ndarray,
    psi_grid: np.ndarray,
) -> Optional[Dict]:
    """Compute BPS/L and SS composition for a list of residue dicts."""
    valid = [
        (r["phi"], r["psi"])
        for r in residues
        if r["phi"] is not None and r["psi"] is not None
    ]
    if len(valid) < 10:
        return None

    phi_arr = np.array([v[0] for v in valid])
    psi_arr = np.array([v[1] for v in valid])

    W_values = lookup_W_batch(W_grid, phi_grid, psi_grid, phi_arr, psi_arr)
    bps_val = compute_bps(W_values)
    bps_l = compute_bps_per_residue(W_values)
    fractions = compute_basin_fractions(residues)

    return {
        "name": name,
        "length": len(residues),
        "n_valid": len(valid),
        "bps": bps_val,
        "bps_l": bps_l,
        "alpha_frac": fractions["alpha"],
        "beta_frac": fractions["beta"],
        "ppII_frac": fractions["ppII"],
        "alphaL_frac": fractions["alphaL"],
        "other_frac": fractions["other"],
    }


def _stats_block(results: List[Dict]) -> Tuple[float, float, float, float]:
    """Compute (mean, std, cv, median) for BPS/L from a list of results."""
    vals = np.array([r["bps_l"] for r in results])
    m = float(np.mean(vals))
    s = float(np.std(vals))
    cv = s / m * 100 if m > 0 else 0.0
    med = float(np.median(vals))
    return m, s, cv, med


def write_report(
    results: List[Dict],
    output_path: str,
    data_source: str,
    filtered_results: Optional[List[Dict]] = None,
) -> None:
    """Write PDB validation report as markdown.

    If filtered_results is provided, includes a filtered statistics section
    alongside the unfiltered (all-structures) statistics.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mean_bps_l, std_bps_l, cv_bps_l, median_bps_l = _stats_block(results)
    bps_l_values = np.array([r["bps_l"] for r in results])

    lines = [
        "# PDB Validation Report: BPS/L",
        "",
        f"**Data source:** {data_source}",
        f"**Structures processed:** {len(results)}",
        "",
    ]

    # Filtered statistics (quality-controlled subset)
    if filtered_results and len(filtered_results) > 0:
        f_mean, f_std, f_cv, f_median = _stats_block(filtered_results)
        f_vals = np.array([r["bps_l"] for r in filtered_results])
        n_nmr = sum(1 for r in results if r.get("is_nmr", False))
        n_short = sum(1 for r in results if r["length"] < 50)
        lines.extend([
            "## Filtered Statistics (X-ray, length >= 50)",
            "",
            f"**Quality filters applied:** exclude NMR ({n_nmr} removed), "
            f"exclude chains < 50 residues ({n_short} removed)",
            f"**Structures after filtering:** {len(filtered_results)}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean BPS/L | {f_mean:.3f} |",
            f"| Std BPS/L | {f_std:.3f} |",
            f"| CV | {f_cv:.1f}% |",
            f"| Median BPS/L | {f_median:.3f} |",
            f"| Min BPS/L | {min(f_vals):.3f} |",
            f"| Max BPS/L | {max(f_vals):.3f} |",
            "",
            "## Comparison to AlphaFold Target (filtered)",
            "",
            f"| | Value |",
            f"|---|-------|",
            f"| AlphaFold target | 0.202 +/- 0.004 |",
            f"| PDB filtered | {f_mean:.3f} +/- {f_std:.3f} |",
            f"| Delta | {abs(f_mean - 0.202):.3f} ({abs(f_mean - 0.202) / 0.202 * 100:.1f}%) |",
            "",
        ])

    # Unfiltered statistics
    lines.extend([
        "## Unfiltered Statistics (all structures)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean BPS/L | {mean_bps_l:.3f} |",
        f"| Std BPS/L | {std_bps_l:.3f} |",
        f"| CV | {cv_bps_l:.1f}% |",
        f"| Median BPS/L | {median_bps_l:.3f} |",
        f"| Min BPS/L | {min(bps_l_values):.3f} |",
        f"| Max BPS/L | {max(bps_l_values):.3f} |",
        "",
        "## BPS/L Histogram (all structures)",
        "",
        "```",
    ])

    # ASCII histogram
    hist, bin_edges = np.histogram(bps_l_values, bins=20)
    max_count = max(hist) if max(hist) > 0 else 1
    for i in range(len(hist)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        bar_len = int(hist[i] / max_count * 40)
        bar = "#" * bar_len
        lines.append(f"  {lo:.3f}-{hi:.3f} | {bar} ({hist[i]})")
    lines.append("```")
    lines.append("")

    # SS composition summary
    alpha_vals = [r["alpha_frac"] for r in results]
    beta_vals = [r["beta_frac"] for r in results]
    lines.extend([
        "## Secondary Structure Composition (computed from phi/psi)",
        "",
        f"| SS Type | Mean Fraction |",
        f"|---------|---------------|",
        f"| alpha | {np.mean(alpha_vals):.1%} |",
        f"| beta | {np.mean(beta_vals):.1%} |",
        f"| other | {np.mean([r['other_frac'] for r in results]):.1%} |",
        "",
    ])

    # Per-structure table (first 30 + last 5 if many)
    lines.extend([
        "## Per-Structure Results",
        "",
        "| # | Name | Length | BPS/L | alpha | beta | other | NMR |",
        "|---|------|--------|-------|-------|------|-------|-----|",
    ])
    show = results[:30]
    if len(results) > 35:
        show = results[:30]
        lines.extend(
            f"| {i+1} | {r['name']} | {r['length']} | {r['bps_l']:.3f} "
            f"| {r['alpha_frac']:.1%} | {r['beta_frac']:.1%} "
            f"| {r['other_frac']:.1%} | {'Y' if r.get('is_nmr') else ''} |"
            for i, r in enumerate(show)
        )
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... |")
        for i, r in enumerate(results[-5:]):
            idx = len(results) - 5 + i
            lines.append(
                f"| {idx+1} | {r['name']} | {r['length']} | {r['bps_l']:.3f} "
                f"| {r['alpha_frac']:.1%} | {r['beta_frac']:.1%} "
                f"| {r['other_frac']:.1%} | {'Y' if r.get('is_nmr') else ''} |"
            )
    else:
        for i, r in enumerate(results):
            lines.append(
                f"| {i+1} | {r['name']} | {r['length']} | {r['bps_l']:.3f} "
                f"| {r['alpha_frac']:.1%} | {r['beta_frac']:.1%} "
                f"| {r['other_frac']:.1%} | {'Y' if r.get('is_nmr') else ''} |"
            )
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Report written to {output_path}")


def _scan_cached_pdbs(cache_dir: str) -> List[Tuple[str, str]]:
    """Scan the PDB cache for already-downloaded structures.

    Returns (pdb_id, chain_id) pairs. Uses chain A by default;
    overrides for known multi-chain files where chain A is not the
    primary protein chain.
    """
    chain_overrides = {
        "1LCD": "A",
        "5UGO": "A",
        "1LMB": "3",
        "1CSE": "E",
        "5CHA": "E",
    }
    entries = []
    if not os.path.isdir(cache_dir):
        return entries
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".pdb") or fname.startswith("SYN"):
            continue
        pdb_id = fname.replace(".pdb", "").upper()
        chain_id = chain_overrides.get(pdb_id, "A")
        entries.append((pdb_id, chain_id))
    return entries


def main() -> None:
    print("Priority 3: PDB Validation of BPS/L")
    print("=" * 55)
    print()

    # Build superpotential (shared across all structures)
    print("Building superpotential W(phi, psi)...")
    W_grid, phi_grid, psi_grid = build_superpotential()
    print()

    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "pdb_cache")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "output")
    report_path = os.path.join(output_dir, "pdb_validation_report.md")

    results: List[Dict] = []
    data_source = ""
    processed_ids: set = set()

    # --- Tier 0: Process already-cached PDB files ---
    cached_entries = _scan_cached_pdbs(cache_dir)
    if cached_entries:
        print(f"Tier 0: Processing {len(cached_entries)} cached PDB files...")
        for pdb_id, chain_id in cached_entries:
            result = process_single_protein(
                pdb_id, chain_id, W_grid, phi_grid, psi_grid, cache_dir
            )
            if result is not None:
                results.append(result)
                processed_ids.add(pdb_id)
        print(f"  Successfully processed {len(results)}/{len(cached_entries)}")
        data_source = f"Cached PDB files ({len(results)} structures)"
        print()

    # --- Tier 1: Try RCSB Search API for additional structures ---
    if len(results) < 100:
        print("Tier 1: Querying RCSB Search API...")
        api_entries = query_rcsb_search(max_results=200)
        if api_entries:
            new_entries = [(p, c) for p, c in api_entries if p not in processed_ids]
            print(f"  Got {len(api_entries)} entries, {len(new_entries)} new")
            n_new = 0
            for i, (pdb_id, chain_id) in enumerate(new_entries):
                if (i + 1) % 20 == 0:
                    print(f"  Processing {i+1}/{len(new_entries)}...")
                result = process_single_protein(
                    pdb_id, chain_id, W_grid, phi_grid, psi_grid, cache_dir
                )
                if result is not None:
                    results.append(result)
                    processed_ids.add(pdb_id)
                    n_new += 1
            print(f"  Added {n_new} from RCSB API")
            if n_new > 0:
                data_source = (f"Cached ({len(cached_entries)}) + "
                               f"RCSB API ({n_new})")
        else:
            print("  RCSB API unavailable")
        print()

    # --- Tier 2: Curated fallback list (download with GitHub fallback) ---
    if len(results) < 50:
        uncached = [(p, c, n) for p, c, n in _CURATED_LIST
                    if p not in processed_ids]
        if uncached:
            print(f"Tier 2: Trying {len(uncached)} structures from curated list "
                  f"(with GitHub mirror fallback)...")
            n_new = 0
            n_fail = 0
            consecutive_fails = 0
            for i, (pdb_id, chain_id, _) in enumerate(uncached):
                if consecutive_fails >= 10 and n_new == 0:
                    print(f"  {consecutive_fails} consecutive failures, "
                          f"stopping tier 2")
                    break
                result = process_single_protein(
                    pdb_id, chain_id, W_grid, phi_grid, psi_grid, cache_dir
                )
                if result is not None:
                    results.append(result)
                    processed_ids.add(pdb_id)
                    n_new += 1
                    consecutive_fails = 0
                else:
                    n_fail += 1
                    consecutive_fails += 1
            print(f"  Added {n_new} from curated list ({n_fail} failed)")
            if n_new > 0 and "Curated" not in data_source:
                data_source += f" + Curated ({n_new})"
            print()

    if not results:
        print("ERROR: No structures processed successfully!")
        print("  Ensure PDB files are in data/pdb_cache/ or network is available.")
        sys.exit(1)

    # --- Quality filtering: exclude NMR and short chains ---
    MIN_LENGTH = 50
    filtered = [r for r in results
                if not r.get("is_nmr", False) and r["length"] >= MIN_LENGTH]
    n_nmr = sum(1 for r in results if r.get("is_nmr", False))
    n_short = sum(1 for r in results if r["length"] < MIN_LENGTH)

    # --- Unfiltered report ---
    bps_l_values = np.array([r["bps_l"] for r in results])
    mean_bps_l = float(np.mean(bps_l_values))
    std_bps_l = float(np.std(bps_l_values))
    cv_bps_l = std_bps_l / mean_bps_l * 100 if mean_bps_l > 0 else 0.0

    print("Unfiltered results (all structures):")
    print(f"  Structures: {len(results)}")
    print(f"  Mean BPS/L: {mean_bps_l:.3f}")
    print(f"  Std BPS/L:  {std_bps_l:.3f}")
    print(f"  CV:         {cv_bps_l:.1f}%")
    print(f"  Median:     {float(np.median(bps_l_values)):.3f}")
    print(f"  Range:      [{min(bps_l_values):.3f}, {max(bps_l_values):.3f}]")
    print()

    # --- Filtered report ---
    print(f"Quality filters: exclude NMR ({n_nmr}), "
          f"exclude length < {MIN_LENGTH} ({n_short})")
    print()

    if filtered:
        f_vals = np.array([r["bps_l"] for r in filtered])
        f_mean = float(np.mean(f_vals))
        f_std = float(np.std(f_vals))
        f_cv = f_std / f_mean * 100 if f_mean > 0 else 0.0

        print("Filtered results (X-ray, length >= 50):")
        print(f"  Structures: {len(filtered)}")
        print(f"  Mean BPS/L: {f_mean:.3f}")
        print(f"  Std BPS/L:  {f_std:.3f}")
        print(f"  CV:         {f_cv:.1f}%")
        print(f"  Median:     {float(np.median(f_vals)):.3f}")
        print(f"  Range:      [{min(f_vals):.3f}, {max(f_vals):.3f}]")
        print()
        print(f"  AlphaFold target: 0.202 +/- 0.004")
        print(f"  PDB target:       0.207 +/- 0.05")
        delta_f = abs(f_mean - 0.207)
        print(f"  Delta from PDB target: {delta_f:.3f} "
              f"({delta_f / 0.207 * 100:.1f}%)")
        print()
    else:
        f_mean = mean_bps_l
        filtered = None

    write_report(results, report_path, data_source,
                 filtered_results=filtered if filtered else None)

    # Validation check on filtered mean
    check_val = f_mean if filtered else mean_bps_l
    label = "filtered" if filtered else "unfiltered"
    if 0.18 <= check_val <= 0.23:
        print(f"  PASS: {label.capitalize()} mean BPS/L = {check_val:.3f} "
              f"in target range [0.18, 0.23]")
    elif 0.10 <= check_val <= 0.35:
        print(f"  WARN: {label.capitalize()} mean BPS/L = {check_val:.3f} "
              f"in broad range [0.10, 0.35] "
              f"but outside strict target [0.18, 0.23]")
    else:
        print(f"  FAIL: {label.capitalize()} mean BPS/L = {check_val:.3f} "
              f"outside [0.10, 0.35]")
        print("  Bug in superpotential or dihedral extraction.")

    print()
    print("Priority 3 complete.")


if __name__ == "__main__":
    main()
