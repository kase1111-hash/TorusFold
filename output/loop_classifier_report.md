# Loop Path Classifier Report

**Loops used:** 52 (from tight families, noise excluded)
**Families:** 11
**Cross-validation:** 3-fold stratified

## Overall Accuracy

**Accuracy: 76.9%**
**Target: >60%**
**Verdict: PASS**

## Per-Family Precision and Recall

| Family | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| 1 | 1.00 | 1.00 | 1.00 | 5.0 |
| 6 | 1.00 | 1.00 | 1.00 | 3.0 |
| 8 | 0.46 | 0.67 | 0.55 | 9.0 |
| 9 | 0.86 | 1.00 | 0.92 | 12.0 |
| 15 | 1.00 | 1.00 | 1.00 | 3.0 |
| 20 | 0.75 | 1.00 | 0.86 | 3.0 |
| 23 | 0.00 | 0.00 | 0.00 | 4.0 |
| 29 | 1.00 | 0.75 | 0.86 | 4.0 |
| 34 | 1.00 | 0.67 | 0.80 | 3.0 |
| 35 | 1.00 | 1.00 | 1.00 | 3.0 |
| 39 | 0.00 | 0.00 | 0.00 | 3.0 |

## Top 10 Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | loop_len | 0.194 |
| 2 | c_term_aa | 0.107 |
| 3 | direction | 0.102 |
| 4 | n_term_aa | 0.087 |
| 5 | AA_GLY | 0.075 |
| 6 | gly_count | 0.058 |
| 7 | AA_THR | 0.056 |
| 8 | AA_ASN | 0.045 |
| 9 | AA_LEU | 0.044 |
| 10 | AA_PRO | 0.038 |

## Confusion Matrix

| Predicted -> | 1 | 6 | 8 | 9 | 15 | 20 | 23 | 29 | 34 | 35 | 39 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **6** | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **8** | 0 | 0 | 6 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| **9** | 0 | 0 | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **15** | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| **20** | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| **23** | 0 | 0 | 3 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **29** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| **34** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| **35** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| **39** | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |

## Verdict

Sequence-to-path-family mapping EXISTS. Loop family can be predicted from amino acid sequence with >60% accuracy.