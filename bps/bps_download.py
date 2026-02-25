"""AlphaFold proteome tarball downloader.

Downloads proteome-level predicted structure files from the AlphaFold
EBI database. Each organism's proteome is a gzipped tarball of mmCIF
files. Files are extracted to data/alphafold_cache/<organism_id>/.

Usage:
    python -m bps.bps_download              # download all 23 organisms
    python -m bps.bps_download --organism ecoli  # download one
"""

import argparse
import os
import sys
import tarfile
import urllib.request
import urllib.error

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_CACHE = os.path.join(_PROJECT_ROOT, "data", "alphafold_cache")

# 23 organisms: covers bacteria, archaea, eukaryotes (plants, animals,
# fungi, protists, parasites) — all major kingdoms of life.
# Format: (short_name, organism_id used in AF URL, species_name)
ORGANISMS = [
    # Bacteria
    ("ecoli",       "UP000000625_83333",   "Escherichia coli K-12"),
    ("bsubtilis",   "UP000001570_224308",  "Bacillus subtilis 168"),
    ("mtb",         "UP000001584_83332",   "Mycobacterium tuberculosis H37Rv"),
    ("caulobacter", "UP000001816_190650",  "Caulobacter vibrioides"),
    ("pseudomonas", "UP000002438_208964",  "Pseudomonas aeruginosa PAO1"),
    # Archaea
    ("mjannaschii", "UP000000805_243232",  "Methanocaldococcus jannaschii"),
    ("hvolcanii",   "UP000008243_309800",  "Haloferax volcanii"),
    # Fungi
    ("yeast",       "UP000002311_559292",  "Saccharomyces cerevisiae S288C"),
    ("spombe",      "UP000002485_284812",  "Schizosaccharomyces pombe"),
    # Plants
    ("arabidopsis", "UP000006548_3702",    "Arabidopsis thaliana"),
    ("rice",        "UP000059680_39947",   "Oryza sativa japonica"),
    # Invertebrates
    ("celegans",    "UP000001940_6239",    "Caenorhabditis elegans"),
    ("drosophila",  "UP000000803_7227",    "Drosophila melanogaster"),
    ("honeybee",    "UP000005203_7460",    "Apis mellifera"),
    # Vertebrates
    ("zebrafish",   "UP000000437_7955",    "Danio rerio"),
    ("chicken",     "UP000000539_9031",    "Gallus gallus"),
    ("mouse",       "UP000000589_10090",   "Mus musculus"),
    ("rat",         "UP000002494_10116",   "Rattus norvegicus"),
    ("human",       "UP000005640_9606",    "Homo sapiens"),
    # Parasites / protists
    ("plasmodium",  "UP000001450_36329",   "Plasmodium falciparum 3D7"),
    ("leishmania",  "UP000000542_5671",    "Leishmania infantum"),
    ("trypanosoma", "UP000008524_185431",  "Trypanosoma brucei brucei TREU927"),
    # Thermophile
    ("thermus",     "UP000000815_262724",  "Thermus thermophilus HB27"),
]


def download_proteome(organism_id: str, cache_dir: str = _DEFAULT_CACHE,
                      version: int = 4) -> str:
    """Download and extract an AlphaFold proteome tarball.

    Parameters
    ----------
    organism_id : str
        AlphaFold proteome ID (e.g., 'UP000000625_83333').
    cache_dir : str
        Root cache directory.
    version : int
        AlphaFold model version (default 4).

    Returns
    -------
    str
        Path to the extracted directory containing CIF files.
    """
    out_dir = os.path.join(cache_dir, organism_id)
    if os.path.isdir(out_dir) and any(f.endswith('.cif') or f.endswith('.cif.gz')
                                       for f in os.listdir(out_dir)):
        return out_dir

    os.makedirs(out_dir, exist_ok=True)

    url = (f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/"
           f"{organism_id}_ALPHAFOLD_v{version}.tar")
    tar_path = os.path.join(cache_dir, f"{organism_id}_v{version}.tar")

    print(f"  Downloading {organism_id}...")
    try:
        urllib.request.urlretrieve(url, tar_path)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  Download failed: {e}")
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return out_dir

    print(f"  Extracting to {out_dir}...")
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=out_dir)
    except (tarfile.TarError, OSError) as e:
        print(f"  Extraction failed: {e}")
    finally:
        if os.path.exists(tar_path):
            os.remove(tar_path)

    n_cif = sum(1 for f in os.listdir(out_dir)
                if f.endswith('.cif') or f.endswith('.cif.gz'))
    print(f"  {organism_id}: {n_cif} CIF files")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Download AlphaFold proteomes")
    parser.add_argument("--organism", type=str, default=None,
                        help="Short name (e.g., 'ecoli'). Default: all.")
    parser.add_argument("--cache-dir", type=str, default=_DEFAULT_CACHE)
    parser.add_argument("--version", type=int, default=4)
    args = parser.parse_args()

    targets = ORGANISMS
    if args.organism:
        targets = [(n, oid, sp) for n, oid, sp in ORGANISMS
                    if n == args.organism]
        if not targets:
            print(f"Unknown organism '{args.organism}'. Available:")
            for n, _, sp in ORGANISMS:
                print(f"  {n:15s}  {sp}")
            sys.exit(1)

    print(f"Downloading {len(targets)} proteomes to {args.cache_dir}")
    print()

    for name, org_id, species in targets:
        print(f"[{name}] {species}")
        out = download_proteome(org_id, args.cache_dir, args.version)
        n = sum(1 for f in os.listdir(out)
                if f.endswith('.cif') or f.endswith('.cif.gz')) if os.path.isdir(out) else 0
        print(f"  -> {n} structures")
        print()


if __name__ == "__main__":
    main()
