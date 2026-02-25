"""Loop path classifier: predicts loop family from amino acid sequence.

Input: Loop data from taxonomy pipeline (short loops only, <=7 residues,
tight families).

Features per loop:
  - 20-dim amino acid composition vector (fraction of each AA)
  - Loop length
  - Flanking basin pair (alpha->beta = 0, beta->alpha = 1)
  - Glycine count, proline count
  - N-terminal flanking residue identity (index 0-19)
  - C-terminal flanking residue identity (index 0-19)

Model: Random forest (no deep learning).
Evaluation: 5-fold stratified cross-validation.

Writes results to output/loop_classifier_report.md.
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bps.superpotential import build_superpotential, assign_basin
from bps.extract import extract_dihedrals_pdb
from loops.taxonomy import (
    extract_loops,
    _assign_basins_for_chain,
    compute_distance_matrix,
    cluster_loops,
    _compute_cluster_stats,
    SHORT_MAX,
    MIN_FLANK,
)


# Standard amino acid ordering for composition vector
AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}


def _aa_composition(residues: List[Dict]) -> np.ndarray:
    """Compute 20-dimensional amino acid composition vector."""
    comp = np.zeros(20)
    for r in residues:
        name = r.get("resname", "UNK")
        if name in AA_INDEX:
            comp[AA_INDEX[name]] += 1
    total = comp.sum()
    if total > 0:
        comp /= total
    return comp


def _count_aa(residues: List[Dict], target: str) -> int:
    """Count occurrences of a specific amino acid."""
    return sum(1 for r in residues if r.get("resname", "") == target)


def _terminal_aa_index(residues: List[Dict], position: int) -> int:
    """Get amino acid index for terminal residue (0-19, or -1 for unknown)."""
    if not residues:
        return -1
    r = residues[position]
    name = r.get("resname", "UNK")
    return AA_INDEX.get(name, -1)


def build_features(loops: List[Dict]) -> np.ndarray:
    """Build feature matrix from a list of loop dicts.

    Features (25 total):
      [0:20]  - AA composition (20-dim)
      [20]    - Loop length
      [21]    - Direction (0 = alpha->beta, 1 = beta->alpha)
      [22]    - Glycine count
      [23]    - Proline count
      [24]    - N-terminal flanking residue AA index
      [25]    - C-terminal flanking residue AA index
    """
    n = len(loops)
    X = np.zeros((n, 26))

    for i, lp in enumerate(loops):
        res = lp["residues"]
        # AA composition
        X[i, :20] = _aa_composition(res)
        # Loop length
        X[i, 20] = lp["loop_len"]
        # Direction encoding
        X[i, 21] = 0.0 if lp["direction"] == "alpha->beta" else 1.0
        # Glycine and proline counts
        X[i, 22] = _count_aa(res, "GLY")
        X[i, 23] = _count_aa(res, "PRO")
        # Terminal residue identities
        X[i, 24] = _terminal_aa_index(res, 0) if res else -1
        X[i, 25] = _terminal_aa_index(res, -1) if res else -1

    return X


FEATURE_NAMES = (
    [f"AA_{aa}" for aa in AA_ORDER]
    + ["loop_len", "direction", "gly_count", "pro_count",
       "n_term_aa", "c_term_aa"]
)


def run_classifier(
    loops: List[Dict],
    labels: np.ndarray,
    n_folds: int = 5,
) -> Dict:
    """Train and evaluate random forest classifier with cross-validation.

    Only uses loops assigned to clusters (label != -1) from tight families.

    Returns dict with accuracy, per-family metrics, feature importances,
    confusion matrix.
    """
    # Filter to clustered loops only
    mask = labels != -1
    filtered_loops = [loops[i] for i in range(len(loops)) if mask[i]]
    filtered_labels = labels[mask]

    if len(filtered_loops) < 10:
        return {
            "error": f"Only {len(filtered_loops)} classified loops — need at least 10",
            "n_loops": len(filtered_loops),
        }

    # Check minimum class sizes
    unique, counts = np.unique(filtered_labels, return_counts=True)
    min_count = counts.min()
    if min_count < 2:
        return {
            "error": f"Smallest family has {min_count} member(s) — need at least 2",
            "n_loops": len(filtered_loops),
        }

    X = build_features(filtered_loops)
    y = filtered_labels

    # Adjust n_folds if classes are too small
    effective_folds = min(n_folds, min_count)
    if effective_folds < 2:
        effective_folds = 2

    # Random forest with cross-validation
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)
    y_pred = cross_val_predict(rf, X, y, cv=skf)

    # Train on full data for feature importances
    rf.fit(X, y)
    importances = rf.feature_importances_

    # Sort feature importances
    feat_imp = sorted(
        zip(FEATURE_NAMES, importances),
        key=lambda x: -x[1],
    )

    # Metrics
    acc = accuracy_score(y, y_pred)
    report_str = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred)

    return {
        "n_loops": len(filtered_loops),
        "n_classes": len(unique),
        "accuracy": acc,
        "n_folds": effective_folds,
        "classification_report": report_str,
        "confusion_matrix": cm,
        "class_labels": unique.tolist(),
        "feature_importances": feat_imp[:10],
        "all_feature_importances": feat_imp,
    }


def write_classifier_report(result: Dict, output_path: str) -> None:
    """Write classifier evaluation report as markdown."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Loop Path Classifier Report",
        "",
    ]

    if "error" in result:
        lines.extend([
            f"**Status:** Insufficient data",
            f"**Reason:** {result['error']}",
            f"**Classified loops available:** {result['n_loops']}",
            "",
            "The classifier requires more loop data with cluster assignments.",
            "Run the taxonomy pipeline on more PDB structures first.",
        ])
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        print(f"  Report written: {output_path}")
        return

    acc = result["accuracy"]
    n_loops = result["n_loops"]
    n_classes = result["n_classes"]
    n_folds = result["n_folds"]

    lines.extend([
        f"**Loops used:** {n_loops} (from tight families, noise excluded)",
        f"**Families:** {n_classes}",
        f"**Cross-validation:** {n_folds}-fold stratified",
        "",
        "## Overall Accuracy",
        "",
        f"**Accuracy: {acc:.1%}**",
        f"**Target: >60%**",
        f"**Verdict: {'PASS' if acc > 0.6 else 'below target'}**",
        "",
    ])

    # Per-family precision and recall
    report = result["classification_report"]
    lines.extend([
        "## Per-Family Precision and Recall",
        "",
        "| Family | Precision | Recall | F1 | Support |",
        "|--------|-----------|--------|----|---------|",
    ])
    for label in result["class_labels"]:
        key = str(label)
        if key in report:
            m = report[key]
            lines.append(
                f"| {label} | {m['precision']:.2f} | {m['recall']:.2f} "
                f"| {m['f1-score']:.2f} | {m['support']} |"
            )
    lines.append("")

    # Feature importances
    lines.extend([
        "## Top 10 Feature Importances",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ])
    for rank, (fname, imp) in enumerate(result["feature_importances"], 1):
        lines.append(f"| {rank} | {fname} | {imp:.3f} |")
    lines.append("")

    # Confusion matrix
    cm = result["confusion_matrix"]
    labels_list = result["class_labels"]
    lines.extend([
        "## Confusion Matrix",
        "",
        "| Predicted -> | " + " | ".join(str(l) for l in labels_list) + " |",
        "|" + "---|" * (len(labels_list) + 1),
    ])
    for i, true_label in enumerate(labels_list):
        row = " | ".join(str(cm[i, j]) for j in range(len(labels_list)))
        lines.append(f"| **{true_label}** | {row} |")
    lines.append("")

    # Verdict
    lines.extend([
        "## Verdict",
        "",
    ])
    if acc > 0.60:
        lines.append("Sequence-to-path-family mapping EXISTS. Loop family can be "
                      "predicted from amino acid sequence with >60% accuracy.")
    elif acc > 0.40:
        lines.append("Weak sequence-to-path signal detected. Accuracy above chance "
                      "but below the 60% target. More data or features may help.")
    else:
        lines.append("No significant sequence-to-path signal detected. "
                      "Loop path family appears independent of sequence composition.")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report written: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    print("Priority 5: Loop Path Classifier")
    print("=" * 55)
    print()

    output_dir = os.path.join(_PROJECT_ROOT, "output")
    report_path = os.path.join(output_dir, "loop_classifier_report.md")
    cache_dir = os.path.join(_PROJECT_ROOT, "data", "pdb_cache")

    # Build superpotential
    print("Building superpotential W(phi, psi)...")
    W_grid, phi_grid, psi_grid = build_superpotential()
    print()

    # --- Step 1: Extract loops (reuse taxonomy logic) ---
    print("Extracting loops from cached PDB structures...")
    from loops.taxonomy import _get_pdb_entries

    entries = _get_pdb_entries(cache_dir)
    all_loops: List[Dict] = []
    for pdb_id, chain_id in entries:
        pdb_path = os.path.join(cache_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            continue
        try:
            residues = extract_dihedrals_pdb(pdb_path, chain_id)
        except (ValueError, Exception):
            continue
        if len(residues) < 20:
            continue
        basins = _assign_basins_for_chain(residues)
        loops = extract_loops(residues, basins)
        for lp in loops:
            lp["pdb_id"] = pdb_id
        all_loops.extend(loops)

    print(f"  Total loops: {len(all_loops)}")

    # Filter to short loops only
    short_loops = [lp for lp in all_loops if lp["loop_len"] <= SHORT_MAX]
    print(f"  Short loops (<=7): {len(short_loops)}")

    if len(short_loops) < 10:
        print("  Insufficient short loops for classification.")
        result = {
            "error": f"Only {len(short_loops)} short loops available",
            "n_loops": len(short_loops),
        }
        write_classifier_report(result, report_path)
        print()
        print("Priority 5 complete (insufficient data).")
        return

    # --- Step 2: Cluster short loops to get family labels ---
    print("\nClustering short loops...")
    D = compute_distance_matrix(short_loops)
    labels, eps, n_clusters = cluster_loops(D)
    n_noise = int(np.sum(labels == -1))
    print(f"  eps={eps:.2f}, {n_clusters} clusters, {n_noise} noise")

    # Get stats to identify tight families
    stats = _compute_cluster_stats(
        short_loops, labels, W_grid, phi_grid, psi_grid
    )
    tight_ids = set(s["cluster_id"] for s in stats if s["tight"])
    print(f"  Tight families: {len(tight_ids)}")

    # Keep only loops from tight families
    tight_mask = np.array([labels[i] in tight_ids for i in range(len(short_loops))])
    tight_loops = [short_loops[i] for i in range(len(short_loops)) if tight_mask[i]]
    tight_labels = labels[tight_mask]
    print(f"  Loops in tight families: {len(tight_loops)}")

    # --- Step 3: Train and evaluate classifier ---
    print("\nRunning classifier with cross-validation...")
    result = run_classifier(tight_loops, tight_labels)

    if "error" in result:
        print(f"  {result['error']}")
    else:
        print(f"  Accuracy: {result['accuracy']:.1%} "
              f"(target >60%: {'PASS' if result['accuracy'] > 0.6 else 'BELOW'})")
        print(f"  Top features: "
              + ", ".join(f"{n}={v:.3f}" for n, v in result["feature_importances"][:5]))

    write_classifier_report(result, report_path)

    print()
    print("Priority 5 complete.")


if __name__ == "__main__":
    main()
