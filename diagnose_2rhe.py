#!/usr/bin/env python3
"""
2RHE FOLD-CLASS LABEL DIAGNOSTIC
=================================
Traces the "all-alpha" classification of 2RHE back to its source.
Run from the project root (or pass --root <path>).

Checks, in order:
  1. SCOP/SCOPe classification files
  2. PISCES culling lists
  3. Hardcoded dicts/lists in pipeline .py files
  4. Any CSV/TSV/JSON data files referencing 2RHE
  5. Downloads 2RHE from PDB and computes actual SS from (phi,psi)

Exit codes:
  0  = diagnosis complete (see report)
  1  = no data files found — check --root
"""

import os, sys, re, glob, json, math, argparse
from pathlib import Path
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────
TARGET_PDB = "2RHE"
EXPECTED_CLASS = "all-beta"  # immunoglobulin VL domain — this is ground truth
REPORTED_CLASS = "all-alpha" # what our pipeline said — this is the bug

# ── Helpers ─────────────────────────────────────────────────────────
def section(title):
    w = 72
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")

def found(source, line_num, raw_line, parsed_label):
    print(f"  ** FOUND in {source} line {line_num} **")
    print(f"     Raw:    {raw_line.rstrip()[:120]}")
    print(f"     Parsed: {parsed_label}")
    return {"source": source, "line": line_num, "raw": raw_line.rstrip(), "label": parsed_label}

# ── 1. SCOP / SCOPe files ──────────────────────────────────────────
def check_scop_files(root):
    section("CHECK 1: SCOP / SCOPe classification files")
    hits = []
    patterns = ["**/scop*cla*", "**/scop*des*", "**/scop*.txt", "**/scope*.txt",
                "**/dir.cla.scope*", "**/dir.des.scope*"]
    scop_files = set()
    for p in patterns:
        scop_files.update(root.glob(p))
    
    if not scop_files:
        print("  No SCOP files found. Searched patterns:")
        for p in patterns:
            print(f"    {root / p}")
        print("  → If you downloaded scop-cla-latest.txt, make sure it's under --root")
        return hits

    # SCOP class letter mapping
    scop_letter = {"a": "all-alpha", "b": "all-beta", "c": "alpha/beta (a/b)",
                   "d": "alpha+beta (a+b)", "e": "multi-domain", "f": "membrane",
                   "g": "small"}

    for fpath in sorted(scop_files):
        print(f"\n  Scanning: {fpath.relative_to(root)}")
        try:
            with open(fpath, "r", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if TARGET_PDB.lower() in line.lower():
                        # Try to extract SCOP class
                        label = "?"
                        # Format: d2rhea_ 2rhe A: b.1.1.1
                        m = re.search(r'\b([a-g])\.\d+\.\d+\.\d+', line)
                        if m:
                            label = scop_letter.get(m.group(1), m.group(1))
                        # Alt format: cl=X,cf=X,sf=X,fa=X
                        m2 = re.search(r'cl=(\d+)', line)
                        if m2:
                            label = f"cl={m2.group(1)} (need des file to decode)"
                        hits.append(found(str(fpath.relative_to(root)), i, line, label))
        except Exception as e:
            print(f"  !! Error reading {fpath}: {e}")

    if not hits:
        print(f"\n  '{TARGET_PDB}' not found in any SCOP file.")
        print(f"  → This means SCOP lookup would have FAILED for this protein.")
        print(f"  → Check if your code has a silent fallback (try/except).")
    return hits


# ── 2. PISCES culling lists ────────────────────────────────────────
def check_pisces_files(root):
    section("CHECK 2: PISCES culling lists")
    hits = []
    patterns = ["**/cullpdb*", "**/pisces*", "**/*culled*", "**/*pdb_list*"]
    pisces_files = set()
    for p in patterns:
        pisces_files.update(root.glob(p))

    if not pisces_files:
        print("  No PISCES files found.")
        return hits

    for fpath in sorted(pisces_files):
        print(f"\n  Scanning: {fpath.relative_to(root)}")
        try:
            with open(fpath, "r", errors="replace") as f:
                header = None
                for i, line in enumerate(f, 1):
                    if i == 1 and ("PDB" in line.upper() or "IDS" in line.upper()):
                        header = line
                        continue
                    if TARGET_PDB.lower() in line.lower():
                        cols = line.split()
                        label = f"columns: {cols}"
                        if header:
                            print(f"  Header: {header.rstrip()[:120]}")
                        hits.append(found(str(fpath.relative_to(root)), i, line, label))
                        # Flag column-shift risk
                        if len(cols) >= 4:
                            print(f"  ⚠ Column count = {len(cols)}. "
                                  f"Verify which column your parser reads as 'class'.")
        except Exception as e:
            print(f"  !! Error reading {fpath}: {e}")

    if not hits:
        print(f"\n  '{TARGET_PDB}' not found in any PISCES file.")
    return hits


# ── 3. Hardcoded dicts in pipeline Python files ────────────────────
def check_python_sources(root):
    section("CHECK 3: Hardcoded labels in Python source files")
    hits = []
    py_files = list(root.rglob("*.py"))
    if not py_files:
        print("  No .py files found under --root")
        return hits

    # Patterns that would assign a fold class
    label_patterns = [
        re.compile(r'["\']2RHE["\'].*(?:alpha|beta|class|fold|scop)', re.I),
        re.compile(r'(?:alpha|beta|class|fold|scop).*["\']2RHE["\']', re.I),
        re.compile(r'2RHE', re.I),
    ]

    for fpath in sorted(py_files):
        try:
            text = fpath.read_text(errors="replace")
            if TARGET_PDB.lower() not in text.lower():
                continue
            lines = text.splitlines()
            for i, line in enumerate(lines, 1):
                if TARGET_PDB.lower() in line.lower():
                    # Check context: ±3 lines for class assignment
                    context_start = max(0, i - 4)
                    context_end = min(len(lines), i + 3)
                    context = lines[context_start:context_end]

                    label = "reference found"
                    for ctx_line in context:
                        if re.search(r'all.?alpha|"a"|class.*=.*["\']a["\']', ctx_line, re.I):
                            label = "⚠ LIKELY BUG: 'all-alpha' or class='a' in context"
                        elif re.search(r'all.?beta|"b"|class.*=.*["\']b["\']', ctx_line, re.I):
                            label = "correct: 'all-beta' or class='b' in context"

                    hits.append(found(str(fpath.relative_to(root)), i, line, label))
                    # Print context
                    print(f"     Context (lines {context_start+1}–{context_end}):")
                    for ci, cl in enumerate(context, context_start + 1):
                        marker = " >>>" if ci == i else "    "
                        print(f"     {marker} {ci}: {cl.rstrip()[:100]}")
        except Exception as e:
            print(f"  !! Error reading {fpath}: {e}")

    if not hits:
        print(f"  '{TARGET_PDB}' not found in any .py file.")
    return hits


# ── 4. CSV / TSV / JSON data files ─────────────────────────────────
def check_data_files(root):
    section("CHECK 4: CSV / TSV / JSON / SQLite data files")
    hits = []
    extensions = ["*.csv", "*.tsv", "*.json", "*.jsonl", "*.dat", "*.txt"]
    data_files = set()
    for ext in extensions:
        data_files.update(root.rglob(ext))
    # Exclude SCOP/PISCES (already checked)
    data_files = [f for f in data_files
                  if not any(k in f.name.lower() for k in ["scop", "pisces", "cullpdb"])]

    for fpath in sorted(data_files):
        try:
            # Skip large files (>50MB)
            if fpath.stat().st_size > 50_000_000:
                continue
            with open(fpath, "r", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if TARGET_PDB.lower() in line.lower():
                        hits.append(found(str(fpath.relative_to(root)), i, line, "data file reference"))
        except Exception as e:
            pass

    # Check SQLite databases
    sqlite_files = list(root.rglob("*.db")) + list(root.rglob("*.sqlite"))
    for fpath in sorted(sqlite_files):
        try:
            import sqlite3
            conn = sqlite3.connect(str(fpath))
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cur.fetchall()]
            for table in tables:
                try:
                    cur.execute(f"SELECT * FROM [{table}] WHERE CAST(rowid AS TEXT) LIKE '%' LIMIT 0")
                    col_names = [d[0] for d in cur.description]
                    for col in col_names:
                        try:
                            cur.execute(f"SELECT rowid, * FROM [{table}] WHERE [{col}] LIKE ?",
                                        (f"%{TARGET_PDB}%",))
                            for row in cur.fetchall():
                                line_str = str(row)[:200]
                                hits.append(found(f"{fpath.relative_to(root)} → {table}.{col}",
                                                  row[0], line_str, "SQLite match"))
                        except:
                            pass
                except:
                    pass
            conn.close()
        except Exception as e:
            pass

    if not hits:
        print(f"  '{TARGET_PDB}' not found in data files.")
    return hits


# ── 5. Fetch 2RHE and compute actual SS ────────────────────────────
def check_actual_structure():
    section("CHECK 5: Compute actual SS from PDB structure")
    try:
        import urllib.request
        url = f"https://files.rcsb.org/download/{TARGET_PDB}.pdb"
        print(f"  Downloading {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "BPS-Diagnostic/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            pdb_text = resp.read().decode("utf-8")
        print(f"  Downloaded {len(pdb_text)} bytes")
    except Exception as e:
        print(f"  !! Could not download: {e}")
        print(f"  → Place {TARGET_PDB}.pdb in the project directory and rerun,")
        print(f"    or manually check: https://www.rcsb.org/structure/{TARGET_PDB}")
        return None

    # Extract backbone atoms (chain A, first model)
    atoms = {}  # resnum -> {atom_name: (x,y,z)}
    in_model = False
    for line in pdb_text.splitlines():
        if line.startswith("MODEL"):
            in_model = True
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name not in ("N", "CA", "C"):
            continue
        chain = line[21]
        if chain != "A" and chain != " ":
            continue
        try:
            resnum = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        if resnum not in atoms:
            atoms[resnum] = {}
        atoms[resnum][atom_name] = (x, y, z)

    # Compute dihedrals
    import numpy as np

    def dihedral(p1, p2, p3, p4):
        b1 = np.array(p2) - np.array(p1)
        b2 = np.array(p3) - np.array(p2)
        b3 = np.array(p4) - np.array(p3)
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1 /= max(np.linalg.norm(n1), 1e-10)
        n2 /= max(np.linalg.norm(n2), 1e-10)
        m1 = np.cross(n1, b2 / max(np.linalg.norm(b2), 1e-10))
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        return math.degrees(math.atan2(y, x))

    residues = sorted(atoms.keys())
    phi_psi = []
    for idx, rn in enumerate(residues):
        if "N" not in atoms[rn] or "CA" not in atoms[rn] or "C" not in atoms[rn]:
            continue
        phi, psi = None, None
        # phi: C(i-1), N(i), CA(i), C(i)
        prev = residues[idx - 1] if idx > 0 else None
        if prev and prev in atoms and "C" in atoms[prev]:
            phi = dihedral(atoms[prev]["C"], atoms[rn]["N"], atoms[rn]["CA"], atoms[rn]["C"])
        # psi: N(i), CA(i), C(i), N(i+1)
        nxt = residues[idx + 1] if idx < len(residues) - 1 else None
        if nxt and nxt in atoms and "N" in atoms[nxt]:
            psi = dihedral(atoms[rn]["N"], atoms[rn]["CA"], atoms[rn]["C"], atoms[nxt]["N"])
        phi_psi.append((rn, phi, psi))

    # Classify
    n_helix = n_sheet = n_coil = 0
    for rn, phi, psi in phi_psi:
        if phi is None or psi is None:
            n_coil += 1
            continue
        if -100 < phi < -30 and -67 < psi < -7:
            n_helix += 1
        elif (-170 < phi < -70 and 90 < psi < 180) or (-170 < phi < -70 and -180 < psi < -120):
            n_sheet += 1
        else:
            n_coil += 1

    total = n_helix + n_sheet + n_coil
    if total == 0:
        print("  !! No residues extracted — PDB parsing issue")
        return None

    h_frac = n_helix / total
    s_frac = n_sheet / total
    c_frac = n_coil / total

    print(f"\n  {TARGET_PDB} chain A: {total} residues")
    print(f"  Helix:  {n_helix:3d}  ({h_frac*100:5.1f}%)")
    print(f"  Sheet:  {n_sheet:3d}  ({s_frac*100:5.1f}%)")
    print(f"  Coil:   {n_coil:3d}  ({c_frac*100:5.1f}%)")

    if s_frac > 0.40:
        computed = "all-beta"
    elif h_frac > 0.40:
        computed = "all-alpha"
    elif h_frac > 0.15 and s_frac > 0.15:
        computed = "mixed alpha/beta"
    else:
        computed = "other / coil-dominant"

    print(f"\n  Computed classification: {computed}")
    print(f"  Expected (SCOP):        {EXPECTED_CLASS}")
    print(f"  Pipeline reported:      {REPORTED_CLASS}")

    if computed == EXPECTED_CLASS:
        print(f"\n  ✓ Actual structure confirms {EXPECTED_CLASS}.")
        print(f"    The 'all-alpha' label did NOT come from dihedral-based SS assignment.")
        print(f"    → Bug is in the SCOP/PISCES parsing or a hardcoded mapping.")
    elif "alpha" in computed:
        print(f"\n  ✗ Actual structure computes as '{computed}' — unexpected!")
        print(f"    Possible causes:")
        print(f"    - Wrong chain extracted (not the VL domain)")
        print(f"    - Dihedral sign convention error")
        print(f"    - Different PDB entry than expected")
    else:
        print(f"\n  ~ Actual structure computes as '{computed}' (not exactly all-beta).")
        print(f"    This may reflect classification threshold differences.")

    return {"helix": h_frac, "sheet": s_frac, "coil": c_frac, "computed": computed}


# ── 6. Check for silent fallback patterns ──────────────────────────
def check_fallback_patterns(root):
    section("CHECK 6: Silent fallback patterns in Python code")
    py_files = list(root.rglob("*.py"))
    risky_patterns = [
        (r'except.*(?:KeyError|Exception).*:\s*\n\s*(?:pass|continue|class.*=)',
         "Silent exception swallowing near class assignment"),
        (r'\.get\s*\(\s*["\']?(?:scop|class|fold)',
         ".get() with possible default value for SCOP class"),
        (r'if.*not\s+in.*(?:scop|class).*:\s*\n\s*.*=.*["\'](?:a|alpha)',
         "Default to 'alpha' when SCOP lookup fails"),
        (r'(?:scop|fold)_class\s*=\s*["\'](?:a|alpha)["\']',
         "Hardcoded default class = alpha"),
    ]

    hits = []
    for fpath in sorted(py_files):
        try:
            text = fpath.read_text(errors="replace")
            for pattern, description in risky_patterns:
                for m in re.finditer(pattern, text, re.I | re.MULTILINE):
                    start = max(0, m.start() - 100)
                    context = text[start:m.end() + 100]
                    line_num = text[:m.start()].count("\n") + 1
                    print(f"\n  ⚠ {description}")
                    print(f"    File: {fpath.relative_to(root)}:{line_num}")
                    print(f"    Match: ...{context.strip()[:150]}...")
                    hits.append({"file": str(fpath), "line": line_num, "pattern": description})
        except:
            pass

    if not hits:
        print("  No risky fallback patterns detected.")
        print("  (This doesn't rule out the bug — just means it's not an obvious pattern.)")
    return hits


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Diagnose 2RHE fold-class mislabel")
    parser.add_argument("--root", type=str, default=".",
                        help="Project root directory to search (default: current dir)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip PDB download (if offline)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"2RHE FOLD-CLASS LABEL DIAGNOSTIC")
    print(f"Project root: {root}")
    print(f"Target: {TARGET_PDB} — expected {EXPECTED_CLASS}, pipeline says {REPORTED_CLASS}")

    all_hits = defaultdict(list)
    all_hits["scop"] = check_scop_files(root)
    all_hits["pisces"] = check_pisces_files(root)
    all_hits["python"] = check_python_sources(root)
    all_hits["data"] = check_data_files(root)
    all_hits["fallback"] = check_fallback_patterns(root)

    if not args.skip_download:
        all_hits["structure"] = check_actual_structure()
    else:
        section("CHECK 5: SKIPPED (--skip-download)")

    # ── Summary ─────────────────────────────────────────────────────
    section("DIAGNOSIS SUMMARY")

    total = sum(len(v) for v in all_hits.values() if isinstance(v, list))
    if total == 0:
        print("  No references to 2RHE found anywhere in the project.")
        print("  This means the label was likely assigned by:")
        print("    a) An external file that's not under --root")
        print("    b) A runtime API call (SCOP web service, etc.)")
        print("    c) A computed SS fallback using dihedral angles")
        print(f"\n  Rerun with --root pointing to the full project tree,")
        print(f"  or check whether 2RHE appears in your SQLite results database.")
    else:
        print(f"  Found {total} references to {TARGET_PDB} across all sources.\n")

        # Determine most likely cause
        if all_hits["python"]:
            bug_in_py = any("LIKELY BUG" in h.get("label", "") for h in all_hits["python"])
            if bug_in_py:
                print("  ★ MOST LIKELY CAUSE: Hardcoded mislabel in Python source.")
                print("    Fix the label and rerun the fold-class analysis.")
            else:
                print("  Python references found but no obvious mislabel in context.")
                print("  Check the context blocks above carefully.")

        if all_hits["scop"]:
            scop_labels = [h.get("label", "") for h in all_hits["scop"]]
            if any("all-beta" in l for l in scop_labels):
                print("  ★ SCOP correctly says all-beta for 2RHE.")
                print("    → Bug is in how your parser reads this file, not in SCOP itself.")
                print("    → Check for off-by-one errors, column misalignment, or regex bugs.")
            elif any("alpha" in l for l in scop_labels):
                print("  ✗ SCOP file itself contains an alpha label for 2RHE?!")
                print("    → This would be genuinely surprising. Double-check the raw line.")

        if all_hits["fallback"]:
            print(f"  ⚠ Found {len(all_hits['fallback'])} risky fallback patterns.")
            print("    If SCOP lookup failed silently, these could assign a default class.")

        if not all_hits["scop"] and not all_hits["pisces"]:
            print("  ★ No SCOP/PISCES entry for 2RHE found.")
            print("    → The label MUST have come from computed SS or a hardcoded list.")
            print("    → This is your most likely bug: a fallback classifier or manual entry.")

    print(f"\n{'='*72}")
    print(f"  BOTTOM LINE: This is almost certainly a pipeline data-mapping error.")
    print(f"  Fix it, note 'fold-class labels validated against computed SS' in")
    print(f"  methods, and move on. Do NOT post about a SCOP misclassification.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
