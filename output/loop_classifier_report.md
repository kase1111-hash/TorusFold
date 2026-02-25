# Loop Path Classifier Report

**Loops used:** 40 (from tight families, noise excluded)
**Families:** 9
**Cross-validation:** 3-fold stratified

## Overall Accuracy

**Accuracy: 80.0%**
**Target: >60%**
**Verdict: PASS**

## Per-Family Precision and Recall

| Family | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| 1 | 1.00 | 1.00 | 1.00 | 5.0 |
| 9 | 0.80 | 1.00 | 0.89 | 12.0 |
| 20 | 1.00 | 1.00 | 1.00 | 3.0 |
| 23 | 0.17 | 0.25 | 0.20 | 4.0 |
| 29 | 1.00 | 0.75 | 0.86 | 4.0 |
| 32 | 1.00 | 0.67 | 0.80 | 3.0 |
| 34 | 1.00 | 0.67 | 0.80 | 3.0 |
| 35 | 1.00 | 1.00 | 1.00 | 3.0 |
| 39 | 1.00 | 0.33 | 0.50 | 3.0 |

## Top 10 Feature Importances

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | loop_len | 0.201 |
| 2 | direction | 0.137 |
| 3 | n_term_aa | 0.088 |
| 4 | c_term_aa | 0.074 |
| 5 | AA_THR | 0.064 |
| 6 | AA_GLY | 0.059 |
| 7 | gly_count | 0.058 |
| 8 | AA_PRO | 0.037 |
| 9 | AA_ASP | 0.032 |
| 10 | AA_ARG | 0.031 |

## Confusion Matrix

| Predicted -> | 1 | 9 | 20 | 23 | 29 | 32 | 34 | 35 | 39 |
|---|---|---|---|---|---|---|---|---|---|
| **1** | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **9** | 0 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **20** | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| **23** | 0 | 3 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **29** | 0 | 0 | 0 | 1 | 3 | 0 | 0 | 0 | 0 |
| **32** | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 |
| **34** | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 |
| **35** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| **39** | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 |

## Verdict

Sequence-to-path-family mapping EXISTS. Loop family can be predicted from amino acid sequence with >60% accuracy.