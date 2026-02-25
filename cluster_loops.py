"""
TorusFold Loop Clustering
=========================
Runs DBSCAN clustering on short loops from the AlphaFold pipeline output.
Stratifies by length, identifies tight families, reports coverage.

Usage:
  python cluster_loops.py [--sample N] [--full]

Reads: results/loops.csv
Writes: results/loop_taxonomy_report.md, results/loop_families.csv
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

try:
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
except ImportError:
    print("ERROR: scikit-learn required")
    print("  pip install scikit-learn")
    sys.exit(1)


def angular_diff(a, b):
    """Signed angular difference using atan2 (radians)."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def loop_feature_vector(row):
    """Build feature vector for DBSCAN clustering."""
    features = [
        row['delta_w'],
        row['torus_path_len'],
        row['length'],
    ]
    return np.array(features)


def classify_family_tightness(family_df, threshold_cv=30.0):
    """Check if a family is tight (low CV on |dW|)."""
    if len(family_df) < 3:
        return 'degenerate'
    dw_vals = family_df['delta_w'].values
    mean_dw = np.mean(dw_vals)
    if mean_dw < 1e-6:
        return 'degenerate'
    cv = np.std(dw_vals) / mean_dw * 100
    return 'tight' if cv < threshold_cv else 'catch-all'


def cluster_stratum(df, eps_values=[0.3, 0.5, 0.8], min_samples=5):
    """Run DBSCAN on a length stratum, try multiple eps values."""
    if len(df) < min_samples:
        return df.assign(family=-1), {}

    # Normalize features
    features = np.column_stack([
        df['delta_w'].values,
        df['torus_path_len'].values,
    ])

    # Standardize
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    stds[stds < 1e-10] = 1.0
    features_norm = (features - means) / stds

    best_result = None
    best_score = -1

    for eps in eps_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(features_norm)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        coverage = 1.0 - n_noise / len(labels)

        # Score: balance coverage with number of non-degenerate clusters
        score = coverage * min(n_clusters, 50) / 50

        if n_clusters > 0 and score > best_score:
            best_score = score
            best_result = {
                'labels': labels,
                'eps': eps,
                'n_clusters': n_clusters,
                'coverage': coverage,
            }

    if best_result is None:
        return df.assign(family=-1), {}

    result_df = df.copy()
    result_df['family'] = best_result['labels']

    # Analyze families
    families = {}
    for fam_id in sorted(set(best_result['labels'])):
        if fam_id == -1:
            continue
        fam_df = result_df[result_df['family'] == fam_id]
        tightness = classify_family_tightness(fam_df)
        dw_vals = fam_df['delta_w'].values
        dw_mean = float(np.mean(dw_vals))
        dw_cv = float(np.std(dw_vals) / dw_mean * 100) if dw_mean > 1e-6 else 999

        families[fam_id] = {
            'n': len(fam_df),
            'tightness': tightness,
            'dw_mean': dw_mean,
            'dw_std': float(np.std(dw_vals)),
            'dw_cv': dw_cv,
            'torus_len_mean': float(fam_df['torus_path_len'].mean()),
            'directions': dict(Counter(fam_df['direction'])),
        }

    return result_df, {
        'eps': best_result['eps'],
        'n_clusters': best_result['n_clusters'],
        'coverage': best_result['coverage'],
        'families': families,
    }


def run_classifier(df_clustered):
    """Train random forest to predict family from sequence features."""
    # Only use tight families with enough members
    family_counts = df_clustered[df_clustered['family'] >= 0]['family'].value_counts()
    valid_families = family_counts[family_counts >= 5].index.tolist()

    clf_df = df_clustered[df_clustered['family'].isin(valid_families)].copy()
    if len(clf_df) < 20 or len(valid_families) < 3:
        return None

    # Build features from residue sequence
    feature_cols = ['length', 'delta_w', 'torus_path_len']

    # Direction encoding
    clf_df = clf_df.copy()
    clf_df['dir_ab'] = (clf_df['direction'] == 'ab').astype(int)

    # Amino acid composition from residues column
    all_aas = set()
    for res in clf_df['residues']:
        all_aas.update(set(str(res)))

    for aa in sorted(all_aas):
        clf_df[f'AA_{aa}'] = clf_df['residues'].apply(
            lambda x: str(x).count(aa) / max(len(str(x)), 1)
        )

    feature_cols = ['length', 'dir_ab'] + [f'AA_{aa}' for aa in sorted(all_aas)]
    X = clf_df[feature_cols].values
    y = clf_df['family'].values

    # Cross-validation
    n_folds = min(3, min(Counter(y).values()))
    if n_folds < 2:
        return None

    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    try:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    except ValueError:
        return None

    # Feature importances from full fit
    rf.fit(X, y)
    importances = dict(zip(feature_cols, rf.feature_importances_))

    return {
        'accuracy': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'n_folds': n_folds,
        'n_samples': len(clf_df),
        'n_families': len(valid_families),
        'top_features': sorted(importances.items(), key=lambda x: -x[1])[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="TorusFold Loop Clustering")
    parser.add_argument("--sample", type=int, default=10000,
                        help="Sample size for short loops (default: 10000)")
    parser.add_argument("--full", action="store_true",
                        help="Use all short loops (may be slow)")
    parser.add_argument("--input", default="results/loops.csv",
                        help="Input loops CSV")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  TorusFold Loop Clustering")
    print("=" * 60)

    # Load data
    print(f"\nLoading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Total loops: {len(df):,}")

    # Stratify by length
    short = df[df['length'] <= 7].copy()
    medium = df[(df['length'] >= 8) & (df['length'] <= 10)].copy()
    long_ = df[(df['length'] >= 11) & (df['length'] <= 15)].copy()

    print(f"  Short (<=7):   {len(short):,}")
    print(f"  Medium (8-10): {len(medium):,}")
    print(f"  Long (11-15):  {len(long_):,}")

    # Sample short loops if needed
    if not args.full and len(short) > args.sample:
        print(f"\n  Sampling {args.sample:,} short loops (use --full for all)...")
        short_sample = short.sample(args.sample, random_state=42)
    else:
        short_sample = short

    # Also subsample medium/long
    med_sample = medium.sample(min(5000, len(medium)), random_state=42) if len(medium) > 5000 else medium
    long_sample = long_.sample(min(3000, len(long_)), random_state=42) if len(long_) > 3000 else long_

    results = {}

    # Cluster each stratum
    for name, sample_df in [("Short (<=7)", short_sample),
                             ("Medium (8-10)", med_sample),
                             ("Long (11-15)", long_sample)]:
        print(f"\n{'='*60}")
        print(f"  Clustering: {name} ({len(sample_df):,} loops)")
        print(f"{'='*60}")

        if len(sample_df) < 10:
            print("  Too few loops, skipping.")
            results[name] = None
            continue

        # Sub-stratify short loops by exact length
        if "Short" in name:
            all_families_df = []
            all_stats = {}
            family_offset = 0

            for length in sorted(sample_df['length'].unique()):
                len_df = sample_df[sample_df['length'] == length]
                print(f"\n  Length {length}: {len(len_df):,} loops")

                if len(len_df) < 20:
                    print(f"    Too few, skipping")
                    continue

                # Cluster by direction
                for direction in ['ab', 'ba']:
                    dir_df = len_df[len_df['direction'] == direction]
                    if len(dir_df) < 10:
                        continue

                    clustered, stats = cluster_stratum(
                        dir_df,
                        eps_values=[0.2, 0.3, 0.5, 0.8],
                        min_samples=max(5, len(dir_df) // 100)
                    )

                    if stats and stats['n_clusters'] > 0:
                        # Offset family IDs to be globally unique
                        clustered['family'] = clustered['family'].apply(
                            lambda x: x + family_offset if x >= 0 else -1
                        )
                        for fam_id in list(stats.get('families', {}).keys()):
                            stats['families'][fam_id + family_offset] = stats['families'].pop(fam_id)
                        family_offset += stats['n_clusters']

                        n_tight = sum(1 for f in stats['families'].values() if f['tightness'] == 'tight')
                        print(f"    {direction}: {stats['n_clusters']} clusters, "
                              f"{n_tight} tight, coverage={stats['coverage']:.1%}")

                        all_families_df.append(clustered)
                        for k, v in stats.get('families', {}).items():
                            all_stats[k] = v
                    else:
                        all_families_df.append(dir_df.assign(family=-1))
                        print(f"    {direction}: no clusters found")

            if all_families_df:
                combined = pd.concat(all_families_df, ignore_index=True)
                n_tight = sum(1 for f in all_stats.values() if f['tightness'] == 'tight')
                n_catchall = sum(1 for f in all_stats.values() if f['tightness'] == 'catch-all')
                n_degen = sum(1 for f in all_stats.values() if f['tightness'] == 'degenerate')
                total_in_clusters = sum(f['n'] for f in all_stats.values())
                coverage = total_in_clusters / len(combined) if len(combined) > 0 else 0

                tight_cvs = [f['dw_cv'] for f in all_stats.values() if f['tightness'] == 'tight']

                results[name] = {
                    'df': combined,
                    'n_clusters': len(all_stats),
                    'n_tight': n_tight,
                    'n_catchall': n_catchall,
                    'n_degenerate': n_degen,
                    'coverage': coverage,
                    'families': all_stats,
                    'tight_cv_median': float(np.median(tight_cvs)) if tight_cvs else 0,
                    'tight_cv_range': (float(min(tight_cvs)), float(max(tight_cvs))) if tight_cvs else (0, 0),
                }

                print(f"\n  TOTAL: {len(all_stats)} clusters, {n_tight} tight, "
                      f"coverage={coverage:.1%}")
                if tight_cvs:
                    print(f"  Tight |dW| CV: median={np.median(tight_cvs):.1f}%, "
                          f"range=[{min(tight_cvs):.1f}%, {max(tight_cvs):.1f}%]")
            else:
                results[name] = None
        else:
            # Medium/long: cluster as a whole
            clustered, stats = cluster_stratum(
                sample_df,
                eps_values=[0.3, 0.5, 0.8, 1.0],
                min_samples=max(5, len(sample_df) // 200)
            )
            if stats and stats['n_clusters'] > 0:
                n_tight = sum(1 for f in stats['families'].values() if f['tightness'] == 'tight')
                print(f"  {stats['n_clusters']} clusters, {n_tight} tight, "
                      f"coverage={stats['coverage']:.1%}")
                results[name] = {
                    'df': clustered,
                    'n_clusters': stats['n_clusters'],
                    'n_tight': n_tight,
                    'coverage': stats['coverage'],
                    'families': stats['families'],
                }
            else:
                print("  No clusters found.")
                results[name] = None

    # ── Run classifier on short loops ──
    print(f"\n{'='*60}")
    print("  Loop Family Classifier")
    print(f"{'='*60}")

    clf_result = None
    if results.get("Short (<=7)") and results["Short (<=7)"].get('df') is not None:
        clf_result = run_classifier(results["Short (<=7)"]['df'])
        if clf_result:
            print(f"  Accuracy: {clf_result['accuracy']:.1%} "
                  f"({clf_result['n_folds']}-fold CV, {clf_result['n_samples']} loops, "
                  f"{clf_result['n_families']} families)")
            print(f"  Top features:")
            for feat, imp in clf_result['top_features'][:5]:
                print(f"    {feat}: {imp:.3f}")
        else:
            print("  Insufficient data for classification.")

    # ── Save family assignments ──
    os.makedirs(args.output, exist_ok=True)

    if results.get("Short (<=7)") and results["Short (<=7)"].get('df') is not None:
        fam_csv = os.path.join(args.output, "loop_families.csv")
        results["Short (<=7)"]['df'].to_csv(fam_csv, index=False)
        print(f"\n  Saved family assignments: {fam_csv}")

    # ── Write Report ──
    report_path = os.path.join(args.output, "loop_taxonomy_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TorusFold Loop Taxonomy Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Total loops:** {len(df):,}\n")
        f.write(f"**Short loops sampled:** {len(short_sample):,} / {len(short):,}\n\n")

        f.write("## Summary\n\n")
        f.write("| Stratum | Loops | Clusters | Tight | Coverage |\n")
        f.write("|---|---|---|---|---|\n")

        for name in ["Short (<=7)", "Medium (8-10)", "Long (11-15)"]:
            r = results.get(name)
            if r:
                f.write(f"| {name} | {len(r.get('df', [])):,} | {r['n_clusters']} | "
                        f"{r['n_tight']} | {r['coverage']:.1%} |\n")
            else:
                f.write(f"| {name} | — | 0 | 0 | 0% |\n")

        f.write("\n")

        # Short loop detail
        sr = results.get("Short (<=7)")
        if sr and sr.get('families'):
            f.write("## Short Loop Families (<=7 residues)\n\n")
            f.write(f"- **Total clusters:** {sr['n_clusters']}\n")
            f.write(f"- **Tight families:** {sr['n_tight']}\n")
            f.write(f"- **Catch-all clusters:** {sr.get('n_catchall', 0)}\n")
            f.write(f"- **Degenerate (<3 members):** {sr.get('n_degenerate', 0)}\n")
            f.write(f"- **Coverage:** {sr['coverage']:.1%}\n")
            if sr.get('tight_cv_median'):
                f.write(f"- **Tight |ΔW| CV:** median={sr['tight_cv_median']:.1f}%, "
                        f"range=[{sr['tight_cv_range'][0]:.1f}%, {sr['tight_cv_range'][1]:.1f}%]\n")
            f.write("\n")

            f.write("### Tight Families Detail\n\n")
            f.write("| Family | N | |ΔW| Mean | |ΔW| CV% | Torus Len | Directions |\n")
            f.write("|---|---|---|---|---|---|\n")

            for fam_id in sorted(sr['families'].keys()):
                fam = sr['families'][fam_id]
                if fam['tightness'] != 'tight':
                    continue
                dirs = ', '.join(f"{k}:{v}" for k, v in fam['directions'].items())
                f.write(f"| {fam_id} | {fam['n']} | {fam['dw_mean']:.3f} | "
                        f"{fam['dw_cv']:.1f}% | {fam['torus_len_mean']:.3f} | {dirs} |\n")

            f.write("\n")

        # Classifier
        if clf_result:
            f.write("## Loop Family Classifier\n\n")
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Accuracy | {clf_result['accuracy']:.1%} |\n")
            f.write(f"| CV folds | {clf_result['n_folds']} |\n")
            f.write(f"| Loops used | {clf_result['n_samples']} |\n")
            f.write(f"| Families | {clf_result['n_families']} |\n\n")
            f.write("### Top Features\n\n")
            f.write("| Feature | Importance |\n|---|---|\n")
            for feat, imp in clf_result['top_features'][:10]:
                f.write(f"| {feat} | {imp:.3f} |\n")
            f.write("\n")

    print(f"\n  Report: {report_path}")

    # ── Scaling test advice ──
    if not args.full and len(short) > args.sample:
        print(f"\n  This used {args.sample:,} of {len(short):,} short loops.")
        print(f"  To scale up: python cluster_loops.py --sample 50000")
        print(f"  Full run:    python cluster_loops.py --full")

    print("\n  Done.")


if __name__ == "__main__":
    main()
