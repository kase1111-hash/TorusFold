"""Diagnose fold classification bug - run from repo root."""
import csv
import math
import os
import sys
import numpy as np

# ── Part 1: CSV summary ──
print("=" * 60)
print("PART 1: CSV fold-class summary")
print("=" * 60)

alphas, betas, others = [], [], []
with open('results/per_protein_bpsl.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        alphas.append(float(row['frac_alpha']))
        betas.append(float(row['frac_beta']))
        others.append(float(row['frac_other']))

n = len(alphas)
print(f"N = {n}")
print(f"Mean fractions:  alpha={sum(alphas)/n:.3f}  beta={sum(betas)/n:.3f}  other={sum(others)/n:.3f}")
print()

# ── Part 2: Load one CIF and look at raw angles ──
print("=" * 60)
print("PART 2: Raw dihedral angles from first CIF file")
print("=" * 60)

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi not installed")
    sys.exit(1)

def dihedral_angle(p0, p1, p2, p3):
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

# Find first CIF file
cif_path = None
for candidate in ['alphafold_cache', 'data', '.']:
    if os.path.isdir(candidate):
        for root, dirs, files in os.walk(candidate):
            for f in files:
                if f.endswith('.cif'):
                    cif_path = os.path.join(root, f)
                    break
            if cif_path:
                break
    if cif_path:
        break

if not cif_path:
    print("No .cif file found.")
    sys.exit(1)

print(f"Loading: {cif_path}")
st = gemmi.read_structure(cif_path)
model = st[0]
chain = model[0]

# Collect residues with our method
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
            'resname': res.name, 'resnum': res.seqid.num,
            'plddt': ca.b_iso if ca else 0
        })

print(f"Residues with backbone atoms: {len(residues)}")
print()

# Compute phi/psi with our method
print(f"{'Res#':>5} {'Name':>4} {'phi_deg':>8} {'psi_deg':>8} {'SS':>3}")
print("-" * 35)

ss_counts = {'a': 0, 'b': 0, 'o': 0}
all_phi = []
all_psi = []

for i in range(1, len(residues) - 1):
    phi = dihedral_angle(residues[i-1]['C'], residues[i]['N'], residues[i]['CA'], residues[i]['C'])
    psi = dihedral_angle(residues[i]['N'], residues[i]['CA'], residues[i]['C'], residues[i+1]['N'])
    phi_d = math.degrees(phi)
    psi_d = math.degrees(psi)
    all_phi.append(phi_d)
    all_psi.append(psi_d)

    if -160 < phi_d < 0 and -120 < psi_d < 30:
        ss = 'a'
    elif -170 < phi_d < -70 and (psi_d > 90 or psi_d < -120):
        ss = 'b'
    else:
        ss = 'o'
    ss_counts[ss] += 1

    if i <= 20:
        print(f"{residues[i]['resnum']:5d} {residues[i]['resname']:>4} {phi_d:8.1f} {psi_d:8.1f}  {ss}")

total = sum(ss_counts.values())
print(f"\nOur method: alpha={ss_counts['a']}/{total} ({100*ss_counts['a']/total:.1f}%)  "
      f"beta={ss_counts['b']}/{total} ({100*ss_counts['b']/total:.1f}%)  "
      f"other={ss_counts['o']}/{total} ({100*ss_counts['o']/total:.1f}%)")

phi_arr = np.array(all_phi)
psi_arr = np.array(all_psi)
print(f"\nPhi: min={phi_arr.min():.1f} max={phi_arr.max():.1f} mean={phi_arr.mean():.1f}")
print(f"Psi: min={psi_arr.min():.1f} max={psi_arr.max():.1f} mean={psi_arr.mean():.1f}")

# ── Part 3: Compare with gemmi's built-in phi/psi ──
print()
print("=" * 60)
print("PART 3: Gemmi built-in phi/psi (ground truth)")
print("=" * 60)

try:
    poly = chain.get_polymer()
    print(f"Polymer length: {len(poly)}")
    print(f"{'Res#':>5} {'Name':>4} {'phi_deg':>8} {'psi_deg':>8} {'SS':>3}")
    print("-" * 35)

    gemmi_ss = {'a': 0, 'b': 0, 'o': 0}
    for i in range(1, min(len(poly) - 1, 9999)):
        prev_res = poly[i - 1]
        curr_res = poly[i]
        next_res = poly[i + 1]
        phi_g = gemmi.calculate_phi(prev_res, curr_res)
        psi_g = gemmi.calculate_psi(curr_res, next_res)
        if math.isnan(phi_g) or math.isnan(psi_g):
            continue
        phi_gd = math.degrees(phi_g)
        psi_gd = math.degrees(psi_g)

        if -160 < phi_gd < 0 and -120 < psi_gd < 30:
            ss = 'a'
        elif -170 < phi_gd < -70 and (psi_gd > 90 or psi_gd < -120):
            ss = 'b'
        else:
            ss = 'o'
        gemmi_ss[ss] += 1

        if i <= 20:
            print(f"{curr_res.seqid.num:5d} {curr_res.name:>4} {phi_gd:8.1f} {psi_gd:8.1f}  {ss}")

    gtotal = sum(gemmi_ss.values())
    if gtotal > 0:
        print(f"\nGemmi method: alpha={gemmi_ss['a']}/{gtotal} ({100*gemmi_ss['a']/gtotal:.1f}%)  "
              f"beta={gemmi_ss['b']}/{gtotal} ({100*gemmi_ss['b']/gtotal:.1f}%)  "
              f"other={gemmi_ss['o']}/{gtotal} ({100*gemmi_ss['o']/gtotal:.1f}%)")
except Exception as e:
    print(f"Gemmi phi/psi failed: {e}")
    import traceback
    traceback.print_exc()
