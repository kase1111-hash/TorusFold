#!/usr/bin/env python3
"""
BPS DOWNLOADER — Fetch AlphaFold CIF files to local cache
==========================================================
Downloads proteome CIF files from AlphaFold. Network-bound only.
No computation, no database. Just fills the cache directory.

Run this first (or in the background) while processing already-cached data.

Usage:
  python bps_download.py --mode all          # download all 34 proteomes
  python bps_download.py --mode organism --organism human
  python bps_download.py --mode plaxco23     # download Plaxco 23 validation set
  python bps_download.py --mode uniprot --ids P00648,P06654  # download by UniProt ID
  python bps_download.py --mode status       # show download progress
  python bps_download.py --mode retry        # retry failed downloads

Progress is tracked in alphafold_cache/<organism>/_manifest.json
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIG
# ============================================================

CACHE_DIR = Path("./alphafold_cache")
LOG_PATH = CACHE_DIR / "download.log"

DOWNLOAD_THREADS = 3
DOWNLOAD_DELAY = 0.15
BACKOFF_BASE = 2

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


# ============================================================
# LOGGING
# ============================================================

def setup_logging():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[
            logging.FileHandler(LOG_PATH, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )


# ============================================================
# MANIFEST — tracks download state per organism
# ============================================================

def load_manifest(organism):
    """Load or create download manifest for an organism"""
    org_dir = CACHE_DIR / organism
    mf_path = org_dir / "_manifest.json"
    mf_bak = org_dir / "_manifest.json.bak"
    default = {"organism": organism, "ids": [], "done": [], "failed_404": [],
               "failed_other": [], "started": None, "finished": None}
    for path in (mf_path, mf_bak):
        if path.exists():
            try:
                with open(path) as f:
                    mf = json.load(f)
                # Deduplicate lists on load
                for key in ("done", "failed_404", "failed_other"):
                    if key in mf:
                        mf[key] = list(dict.fromkeys(mf[key]))
                return mf
            except (json.JSONDecodeError, ValueError):
                logging.warning(f"  Corrupt manifest {path}, trying backup...")
                continue
    return default


def save_manifest(organism, manifest):
    """Atomic manifest write: write to temp file, then rename"""
    org_dir = CACHE_DIR / organism
    org_dir.mkdir(parents=True, exist_ok=True)
    mf_path = org_dir / "_manifest.json"
    mf_tmp = org_dir / "_manifest.json.tmp"
    mf_bak = org_dir / "_manifest.json.bak"
    with open(mf_tmp, 'w') as f:
        json.dump(manifest, f)
        f.flush()
        os.fsync(f.fileno())
    if mf_path.exists():
        mf_path.replace(mf_bak)
    mf_tmp.replace(mf_path)


# ============================================================
# UniProt ID fetcher
# ============================================================

def get_proteome_uniprot_ids(organism, max_proteins=None):
    import urllib.request, urllib.parse
    info = PROTEOMES[organism]
    taxid = info['taxid']
    ids_cache = CACHE_DIR / f"{organism}_uniprot_ids.txt"
    if ids_cache.exists():
        with open(ids_cache) as f:
            ids = [l.strip() for l in f if l.strip()]
        logging.info(f"  Loaded {len(ids)} UniProt IDs from cache")
        return ids[:max_proteins] if max_proteins else ids

    logging.info(f"  Fetching UniProt IDs for {info['name']}...")
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": f"organism_id:{taxid} AND reviewed:true", "format": "list", "size": "500"}
    all_ids = []
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    while url:
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
                batch = [x.strip() for x in resp.read().decode().strip().split('\n') if x.strip()]
                all_ids.extend(batch)
                link = resp.headers.get('Link', '')
                url = None
                if link:
                    for part in link.split(','):
                        if 'rel="next"' in part:
                            url = part.split(';')[0].strip('<> ')
                            break
                logging.info(f"    Retrieved {len(all_ids)} IDs...")
                if max_proteins and len(all_ids) >= max_proteins:
                    all_ids = all_ids[:max_proteins]; break
        except Exception as e:
            logging.warning(f"    Error: {e}"); break

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ids_cache, 'w') as f:
        f.writelines(uid + '\n' for uid in all_ids)
    logging.info(f"  Total: {len(all_ids)} UniProt IDs")
    return all_ids[:max_proteins] if max_proteins else all_ids


# ============================================================
# DOWNLOAD
# ============================================================

def download_one(uniprot_id, cache_dir):
    """Download a single CIF. Returns (path, version, error_type).
    path is None on failure. error_type is '404', 'timeout', 'other', or None."""
    import urllib.request
    import urllib.error

    # Check cache
    for v in (4, 3, 2, 1):
        p = cache_dir / f"AF-{uniprot_id}-F1-model_v{v}.cif"
        if p.exists() and p.stat().st_size > 100:
            return p, v, None

    time.sleep(DOWNLOAD_DELAY)

    n_404 = 0
    for v in (4, 3, 2):
        cif_path = cache_dir / f"AF-{uniprot_id}-F1-model_v{v}.cif"
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v{v}.cif"
        try:
            if _fetch_to_file(url, cif_path):
                return cif_path, v, None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                n_404 += 1
                continue
            elif e.code == 429:
                time.sleep(BACKOFF_BASE)
                try:
                    if _fetch_to_file(url, cif_path):
                        return cif_path, v, None
                except Exception:
                    pass
                continue  # try next version instead of giving up
            else:
                return None, None, 'other'
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError):
            return None, None, 'timeout'
        except Exception:
            return None, None, 'other'

    # All 404 — try API
    if n_404 == 3:
        try:
            api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
            with urllib.request.urlopen(api_url, timeout=15) as resp:
                data = json.loads(resp.read())
                if isinstance(data, list) and data:
                    data = data[0]
                cif_url = data.get("cifUrl", "")
                if cif_url:
                    fname = cif_url.split('/')[-1]
                    version = 4
                    for vv in (1, 2, 3, 4):
                        if f"_v{vv}" in fname:
                            version = vv
                    cif_path = cache_dir / fname
                    if not (cif_path.exists() and cif_path.stat().st_size > 100):
                        _fetch_to_file(cif_url, cif_path)
                    if cif_path.exists() and cif_path.stat().st_size > 100:
                        return cif_path, version, None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, None, '404'
            return None, None, 'timeout'
        except Exception:
            return None, None, 'timeout'

    return None, None, '404'


def is_cached(uniprot_id, cache_dir):
    """Check if CIF already exists in cache"""
    for v in (4, 3, 2, 1):
        p = cache_dir / f"AF-{uniprot_id}-F1-model_v{v}.cif"
        if p.exists() and p.stat().st_size > 100:
            return True
    return False


def _fetch_to_file(url, target_path):
    """Download to temp file, then atomic rename. Raises on HTTP errors.
    Returns True if the downloaded file is valid (>100 bytes)."""
    import urllib.request
    tmp_path = target_path.parent / f".{target_path.name}.tmp"
    try:
        urllib.request.urlretrieve(url, str(tmp_path))
        if tmp_path.exists() and tmp_path.stat().st_size > 100:
            tmp_path.replace(target_path)
            return True
        tmp_path.unlink(missing_ok=True)
        return False
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ============================================================
# DOWNLOAD ORGANISM
# ============================================================

def download_organism(organism, max_proteins=None):
    logging.info("=" * 70)
    logging.info(f"DOWNLOADING: {organism.upper()}")
    logging.info("=" * 70)

    org_cache = CACHE_DIR / organism
    org_cache.mkdir(parents=True, exist_ok=True)

    # Get IDs
    ids = get_proteome_uniprot_ids(organism, max_proteins)
    if not ids:
        logging.warning("  No IDs found.")
        return

    # Load manifest
    mf = load_manifest(organism)
    mf['ids'] = ids
    if not mf['started']:
        mf['started'] = datetime.now().isoformat()

    # Skip already done
    done_set = set(mf['done'])
    fail_set = set(mf['failed_404'])
    skip = done_set | fail_set
    remaining = [uid for uid in ids if uid not in skip]

    # Also skip if file exists in cache but not in manifest
    still_needed = []
    newly_cached = 0
    for uid in remaining:
        if is_cached(uid, org_cache):
            mf['done'].append(uid)
            newly_cached += 1
        else:
            still_needed.append(uid)

    if newly_cached > 0:
        save_manifest(organism, mf)
        logging.info(f"  Found {newly_cached} already cached but not in manifest")

    remaining = still_needed

    if not remaining:
        logging.info(f"  Complete: {len(mf['done'])} downloaded, "
                     f"{len(mf['failed_404'])} not in AlphaFold")
        mf['finished'] = datetime.now().isoformat()
        save_manifest(organism, mf)
        return

    logging.info(f"  IDs: {len(ids)}, cached: {len(mf['done'])}, "
                 f"404: {len(mf['failed_404'])}, remaining: {len(remaining)}")

    t_start = time.time()
    batch_size = 50
    new_ok = 0
    new_fail = 0

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]

        # Download batch with thread pool
        with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
            futs = {pool.submit(download_one, uid, org_cache): uid for uid in batch}
            for fut in as_completed(futs):
                uid = futs[fut]
                try:
                    path, version, err = fut.result()
                    if path:
                        mf['done'].append(uid)
                        new_ok += 1
                    elif err == '404':
                        mf['failed_404'].append(uid)
                        new_fail += 1
                    else:
                        mf['failed_other'].append(uid)
                        new_fail += 1
                except Exception:
                    mf['failed_other'].append(uid)
                    new_fail += 1

        # Save manifest every batch
        save_manifest(organism, mf)

        # Progress
        so_far = batch_start + len(batch)
        elapsed = time.time() - t_start
        rate = so_far / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - so_far) / rate / 60 if rate > 0 else 0
        logging.info(f"  [{so_far}/{len(remaining)}] {rate:.1f}/sec, "
                     f"ETA: {eta:.0f} min, +{new_ok} ok, +{new_fail} fail")

    elapsed = time.time() - t_start
    mf['finished'] = datetime.now().isoformat()
    save_manifest(organism, mf)
    logging.info(f"\n  Done: +{new_ok} new downloads in {elapsed:.0f}s")
    logging.info(f"  Total: {len(mf['done'])} cached, {len(mf['failed_404'])} 404, "
                 f"{len(mf['failed_other'])} other failures")


# ============================================================
# STATUS
# ============================================================

def show_status():
    print(f"\n{'='*80}")
    print(f"  DOWNLOAD STATUS")
    print(f"{'='*80}")

    header = f"  {'Organism':<20} {'IDs':>7} {'Cached':>7} {'404':>7} {'Other':>7} {'Remain':>7} {'Status':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_ids = 0
    total_cached = 0
    total_404 = 0

    def parse_count(s):
        s = s.replace('~', '').strip()
        return int(float(s.replace('K', '')) * 1000) if 'K' in s else int(s)

    for org in sorted(PROTEOMES.keys(), key=lambda k: parse_count(PROTEOMES[k]['count'])):
        mf = load_manifest(org)
        n_ids = len(mf['ids'])
        n_done = len(mf['done'])
        n_404 = len(mf['failed_404'])
        n_other = len(mf['failed_other'])
        n_remain = n_ids - n_done - n_404 - n_other if n_ids > 0 else 0

        # Also count actual CIF files
        org_dir = CACHE_DIR / org
        n_files = len(list(org_dir.glob("AF-*-F1-model_v*.cif"))) if org_dir.exists() else 0

        if n_ids == 0 and n_files == 0:
            status = "—"
        elif n_remain <= 0:
            status = "DONE"
        elif n_done > 0:
            status = f"{n_done/n_ids*100:.0f}%"
        else:
            status = "pending"

        # Use file count if manifest is empty but files exist
        display_done = max(n_done, n_files)

        total_ids += n_ids
        total_cached += display_done
        total_404 += n_404

        if n_ids > 0 or n_files > 0:
            print(f"  {org:<20} {n_ids:>7} {display_done:>7} {n_404:>7} "
                  f"{n_other:>7} {max(n_remain, 0):>7} {status:>10}")

    print(f"  {'-'*75}")
    print(f"  {'TOTAL':<20} {total_ids:>7} {total_cached:>7} {total_404:>7}")

    # Disk usage
    if CACHE_DIR.exists():
        total_bytes = sum(f.stat().st_size for f in CACHE_DIR.rglob("*.cif"))
        print(f"\n  Cache size: {total_bytes / 1024**3:.2f} GB")
        print(f"  Cache dir:  {CACHE_DIR.resolve()}")

    print()


# ============================================================
# RETRY
# ============================================================

def retry_failures():
    """Retry downloads that failed with non-404 errors"""
    for org in sorted(PROTEOMES.keys()):
        mf = load_manifest(org)
        retryable = mf.get('failed_other', [])
        if not retryable:
            continue

        logging.info(f"\nRetrying {len(retryable)} failures for {org}...")
        mf['failed_other'] = []
        org_cache = CACHE_DIR / org

        for uid in retryable:
            path, version, err = download_one(uid, org_cache)
            if path:
                mf['done'].append(uid)
                logging.info(f"  {uid}: OK (v{version})")
            elif err == '404':
                mf['failed_404'].append(uid)
            else:
                mf['failed_other'].append(uid)

        save_manifest(org, mf)
        logging.info(f"  {org}: {len(mf['done'])} done, "
                     f"{len(mf['failed_404'])} 404, {len(mf['failed_other'])} still failing")


# ============================================================
# DOWNLOAD BY UNIPROT ID — individual CIF files
# ============================================================

# Plaxco 23 validation set: proteins from diverse organisms, many outside
# the 34 proteomes.  (uniprot_id, pdb_id, name)
PLAXCO_23 = [
    ("P00648", "1SRL", "Sarcin"),
    ("P56849", "1APS", "AcP"),
    ("P0A6X3", "1HKS", "Hpr"),
    ("P00698", "1HEL", "HEWL"),
    ("P02489", "3HMZ", "alpha-A crystallin"),
    ("P00636", "2RN2", "Barnase"),
    ("P00720", "2LZM", "T4 Lysozyme"),
    ("P23928", "1MJC", "alphaB crystallin"),
    ("P0AEH5", "2CI2", "CI2"),
    ("P06654", "1UBQ", "Ubiquitin"),
    ("P01112", "5P21", "p21 Ras"),
    ("P62937", "2CYP", "CypA"),
    ("P23229", "1TIT", "Titin I27"),
    ("P0ABP8", "1DPS", "Dps"),
    ("P04637", "2OCJ", "p53"),
    ("P05067", "1AAP", "APPI"),
    ("P00974", "5PTI", "BPTI"),
    ("P01133", "1EGF", "EGF"),
    ("P62993", "1SHG", "SH3"),
    ("P00178", "1YCC", "Cyt c"),
    ("P0CY58", "1AON", "GroEL"),
    ("P00925", "3ENL", "Enolase"),
    ("P00517", "1CDK", "PKA"),
]


def download_uniprot_ids(uniprot_ids, target_dir, labels=None):
    """Download AlphaFold CIF files for a list of UniProt IDs.

    Args:
        uniprot_ids: list of UniProt accessions (e.g. ["P00648", "P06654"])
        target_dir: Path to download into (created if needed)
        labels: optional dict {uid: label} for display

    Returns:
        (n_ok, n_skip, n_fail)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if labels is None:
        labels = {uid: uid for uid in uniprot_ids}

    n_ok, n_skip, n_fail = 0, 0, 0
    for uid in uniprot_ids:
        label = labels.get(uid, uid)

        if is_cached(uid, target_dir):
            logging.info(f"  {uid} ({label}): cached")
            n_skip += 1
            continue

        path, version, err = download_one(uid, target_dir)
        if path:
            logging.info(f"  {uid} ({label}): OK (v{version})")
            n_ok += 1
        elif err == '404':
            logging.warning(f"  {uid} ({label}): NOT FOUND (404)")
            n_fail += 1
        else:
            logging.warning(f"  {uid} ({label}): FAILED ({err})")
            n_fail += 1

    logging.info(f"\n  Summary: {n_ok} downloaded, {n_skip} cached, {n_fail} failed")
    return n_ok, n_skip, n_fail


def download_plaxco23():
    """Download all 23 Plaxco validation proteins into alphafold_cache/plaxco23/."""
    target = CACHE_DIR / "plaxco23"
    logging.info("=" * 70)
    logging.info("DOWNLOADING PLAXCO 23 VALIDATION SET")
    logging.info(f"  Target: {target}")
    logging.info("=" * 70)

    ids = [uid for uid, _, _ in PLAXCO_23]
    labels = {uid: f"{pdb}/{name}" for uid, pdb, name in PLAXCO_23}
    return download_uniprot_ids(ids, target, labels)


# ============================================================
# WATCH MODE — continuous download loop
# ============================================================

def watch_loop(poll_interval, max_proteins=None):
    """Continuously download all organisms, sleeping between cycles.
    Safe to run alongside bps_process.py and compile_results.py."""

    def parse_count(s):
        s = s.replace('~', '').strip()
        return int(float(s.replace('K', '')) * 1000) if 'K' in s else int(s)

    orgs = sorted(PROTEOMES.keys(), key=lambda k: parse_count(PROTEOMES[k]['count']))
    logging.info(f"WATCH MODE: {len(orgs)} organisms, poll every {poll_interval}s")

    while True:
        did_work = False
        for i, org in enumerate(orgs):
            # Check if already complete before starting
            mf = load_manifest(org)
            n_ids = len(mf.get('ids', []))
            n_done = len(set(mf.get('done', [])))
            n_fail = len(set(mf.get('failed_404', [])))
            if n_ids > 0 and (n_done + n_fail) >= n_ids:
                continue  # already complete

            info = PROTEOMES[org]
            logging.info(f"\n[{i+1}/{len(orgs)}] {info['name']} ({info['count']})")
            try:
                download_organism(org, max_proteins)
                did_work = True
            except KeyboardInterrupt:
                logging.info("\nWatch stopped. Progress saved.")
                return
            except Exception as e:
                logging.error(f"ERROR on {org}: {e}")
                continue

        if not did_work:
            logging.info(f"All organisms complete or up to date.")

        logging.info(f"Sleeping {poll_interval}s... (Ctrl+C to stop)")
        try:
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            logging.info("\nWatch stopped.")
            return


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BPS Downloader — fetch AlphaFold CIFs")
    parser.add_argument("--mode", choices=["all", "organism", "status", "retry", "watch",
                                           "plaxco23", "uniprot"],
                        default="status")
    parser.add_argument("--organism", default="ecoli",
                        choices=list(PROTEOMES.keys()))
    parser.add_argument("--ids", default=None,
                        help="Comma-separated UniProt IDs (for --mode uniprot)")
    parser.add_argument("--target-dir", default=None,
                        help="Target directory for --mode uniprot (default: alphafold_cache/custom/)")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--poll", type=int, default=60,
                        help="Poll interval in seconds for watch mode (default: 60)")
    args = parser.parse_args()

    setup_logging()

    if args.mode == "status":
        show_status()

    elif args.mode == "organism":
        download_organism(args.organism, args.max)

    elif args.mode == "plaxco23":
        download_plaxco23()

    elif args.mode == "uniprot":
        if not args.ids:
            parser.error("--ids required for --mode uniprot")
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        target = Path(args.target_dir) if args.target_dir else CACHE_DIR / "custom"
        download_uniprot_ids(ids, target)

    elif args.mode == "retry":
        retry_failures()

    elif args.mode == "watch":
        watch_loop(args.poll, args.max)

    elif args.mode == "all":
        def parse_count(s):
            s = s.replace('~', '').strip()
            return int(float(s.replace('K', '')) * 1000) if 'K' in s else int(s)

        orgs = sorted(PROTEOMES.keys(), key=lambda k: parse_count(PROTEOMES[k]['count']))
        logging.info(f"\nDOWNLOADING ALL {len(orgs)} PROTEOMES")

        for i, org in enumerate(orgs):
            info = PROTEOMES[org]
            logging.info(f"\n[{i+1}/{len(orgs)}] {info['name']} ({info['count']})")
            try:
                download_organism(org, args.max)
            except KeyboardInterrupt:
                logging.info("\nInterrupted. Progress saved in manifests.")
                break
            except Exception as e:
                logging.error(f"ERROR on {org}: {e}")
                continue

        show_status()


if __name__ == "__main__":
    main()
