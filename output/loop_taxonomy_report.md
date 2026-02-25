# Loop Path Taxonomy Report

**Structures analyzed:** 218
**Total loops extracted:** 1344

## Loop Direction Summary

| Direction | Count |
|-----------|-------|
| alpha->beta | 683 |
| beta->alpha | 661 |

## Loop Length Distribution

| Metric | Value |
|--------|-------|
| Min | 1 |
| Max | 20 |
| Mean | 6.5 |
| Median | 5.0 |

## Long (11-15)

**Loops:** 158
**DBSCAN eps:** 0.30
**Clusters:** 2 (0 tight, 2 catch-all)
**Noise:** 150 (95%)
**Coverage:** 5%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 5 | 0.409 | 73.6% | 23.531 | 2.2% | catch-all | alpha->beta:5 |
| 1 | 3 | 1.375 | 53.6% | 28.412 | 0.7% | catch-all | alpha->beta:3 |

## Medium (8-10)

**Loops:** 208
**DBSCAN eps:** 0.25
**Clusters:** 2 (1 tight, 1 catch-all)
**Noise:** 202 (97%)
**Coverage:** 3%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 3 | 1.250 | 38.7% | 13.544 | 5.0% | catch-all | alpha->beta:3 |
| 1 | 3 | 0.691 | 14.1% | 12.408 | 2.3% | tight | beta->alpha:3 |

**Tight family |dW| CV range:** 14.1% - 14.1%

## Short (<=7)

**Loops:** 905
**DBSCAN eps:** 0.50
**Clusters:** 59 (19 tight, 40 catch-all)
**Noise:** 575 (64%)
**Coverage:** 36%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 26 | 0.469 | 67.5% | 7.082 | 5.5% | catch-all | alpha->beta:26 |
| 61 | 22 | 0.044 | 231.8% | 0.054 | 232.7% | catch-all | alpha->beta:1, beta->alpha:21 |
| 60 | 17 | 0.026 | 318.0% | 0.035 | 273.9% | catch-all | alpha->beta:3, beta->alpha:14 |
| 30 | 16 | 1.267 | 68.8% | 8.592 | 6.3% | catch-all | alpha->beta:16 |
| 7 | 15 | 1.540 | 57.5% | 6.100 | 18.6% | catch-all | alpha->beta:14, beta->alpha:1 |
| 16 | 15 | 3.066 | 55.9% | 6.228 | 5.8% | catch-all | alpha->beta:15 |
| 9 | 12 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:11, beta->alpha:1 |
| 45 | 10 | 2.259 | 9.8% | 0.916 | 12.7% | tight | beta->alpha:10 |
| 8 | 9 | 1.827 | 35.7% | 4.148 | 10.1% | catch-all | alpha->beta:8, beta->alpha:1 |
| 22 | 8 | 1.993 | 38.6% | 3.712 | 5.3% | catch-all | alpha->beta:8 |
| 31 | 7 | 1.016 | 64.3% | 7.157 | 18.5% | catch-all | beta->alpha:7 |
| 5 | 6 | 0.709 | 58.4% | 5.593 | 4.0% | catch-all | alpha->beta:6 |
| 25 | 6 | 0.810 | 98.3% | 7.636 | 11.9% | catch-all | alpha->beta:6 |
| 44 | 6 | 0.143 | 223.6% | 0.084 | 223.6% | catch-all | alpha->beta:4, beta->alpha:2 |
| 1 | 5 | 0.000 | 0.0% | 0.000 | 0.0% | tight | beta->alpha:5 |
| 14 | 5 | 0.879 | 62.3% | 7.655 | 6.6% | catch-all | alpha->beta:5 |
| 37 | 5 | 0.707 | 92.2% | 4.955 | 22.1% | catch-all | alpha->beta:4, beta->alpha:1 |
| 46 | 5 | 2.201 | 39.2% | 1.860 | 19.3% | catch-all | beta->alpha:5 |
| 17 | 4 | 0.445 | 93.3% | 6.539 | 8.4% | catch-all | beta->alpha:4 |
| 21 | 4 | 1.579 | 52.5% | 6.483 | 3.4% | catch-all | alpha->beta:3, beta->alpha:1 |
| 23 | 4 | 1.490 | 24.6% | 4.636 | 5.9% | tight | alpha->beta:3, beta->alpha:1 |
| 26 | 4 | 0.583 | 34.6% | 8.162 | 0.9% | catch-all | beta->alpha:4 |
| 27 | 4 | 0.772 | 102.4% | 9.205 | 4.2% | catch-all | alpha->beta:4 |
| 29 | 4 | 2.987 | 9.0% | 7.756 | 15.1% | tight | alpha->beta:4 |
| 43 | 4 | 1.199 | 35.0% | 3.617 | 4.6% | catch-all | beta->alpha:4 |
| 52 | 4 | 1.676 | 7.5% | 3.313 | 1.5% | tight | beta->alpha:4 |
| 54 | 4 | 0.413 | 50.2% | 6.056 | 2.6% | catch-all | alpha->beta:4 |
| 55 | 4 | 0.316 | 38.8% | 5.944 | 4.4% | catch-all | alpha->beta:4 |
| 58 | 4 | 0.563 | 95.3% | 4.132 | 4.9% | catch-all | beta->alpha:4 |
| 63 | 4 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:3 |
| 2 | 3 | 0.614 | 48.2% | 12.082 | 3.4% | catch-all | beta->alpha:3 |
| 6 | 3 | 0.352 | 36.9% | 8.692 | 1.8% | catch-all | alpha->beta:3 |
| 13 | 3 | 0.441 | 43.9% | 6.911 | 0.6% | catch-all | alpha->beta:3 |
| 15 | 3 | 0.658 | 48.7% | 3.589 | 9.2% | catch-all | alpha->beta:1, beta->alpha:2 |
| 18 | 3 | 0.355 | 70.6% | 14.003 | 2.4% | catch-all | alpha->beta:3 |
| 19 | 3 | 0.304 | 97.9% | 8.966 | 6.6% | catch-all | beta->alpha:3 |
| 20 | 3 | 2.276 | 23.6% | 5.796 | 1.4% | tight | beta->alpha:3 |
| 24 | 3 | 1.176 | 81.0% | 5.726 | 5.5% | catch-all | alpha->beta:3 |
| 32 | 3 | 0.877 | 29.3% | 5.860 | 5.9% | tight | beta->alpha:3 |
| 33 | 3 | 0.324 | 41.6% | 10.778 | 4.2% | catch-all | beta->alpha:3 |
| 34 | 3 | 3.825 | 15.7% | 2.677 | 5.5% | tight | beta->alpha:3 |
| 35 | 3 | 1.945 | 10.5% | 6.029 | 8.0% | tight | beta->alpha:3 |
| 36 | 3 | 0.895 | 31.7% | 7.679 | 6.3% | catch-all | beta->alpha:3 |
| 38 | 3 | 0.945 | 62.6% | 8.163 | 3.5% | catch-all | alpha->beta:3 |
| 39 | 3 | 0.283 | 14.8% | 8.803 | 16.2% | tight | alpha->beta:3 |
| 41 | 3 | 0.028 | 141.4% | 0.168 | 141.4% | catch-all | alpha->beta:1, beta->alpha:2 |
| 42 | 3 | 0.571 | 34.6% | 2.504 | 4.0% | catch-all | alpha->beta:3 |
| 47 | 3 | 0.223 | 28.5% | 2.836 | 3.1% | tight | alpha->beta:3 |
| 48 | 3 | 1.639 | 32.1% | 2.164 | 8.5% | catch-all | beta->alpha:3 |
| 49 | 3 | 1.319 | 1.0% | 3.588 | 4.3% | tight | beta->alpha:3 |
| 50 | 3 | 0.369 | 141.4% | 0.305 | 141.4% | catch-all | alpha->beta:1, beta->alpha:2 |
| 51 | 3 | 0.656 | 27.6% | 2.205 | 8.5% | tight | beta->alpha:3 |
| 53 | 3 | 2.442 | 2.8% | 3.174 | 1.7% | tight | beta->alpha:3 |
| 57 | 3 | 0.442 | 89.7% | 2.778 | 2.2% | catch-all | alpha->beta:3 |
| 59 | 3 | 1.736 | 45.9% | 4.710 | 6.0% | catch-all | beta->alpha:3 |
| 62 | 3 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:2 |
| 64 | 3 | 0.099 | 83.2% | 0.208 | 86.0% | catch-all | beta->alpha:3 |
| 65 | 3 | 1.085 | 8.9% | 2.965 | 1.6% | tight | alpha->beta:3 |
| 66 | 3 | 1.254 | 12.3% | 2.757 | 1.5% | tight | alpha->beta:3 |

**Tight family |dW| CV range:** 0.0% - 29.3%

## Comparison to Targets (from CLAUDE.md)

| Metric | Target | Observed |
|--------|--------|----------|
| Total loops | 1000+ | 1344 |
| Short tight families | ~30 | 19 |
| Short coverage | 100% | 36% |
| Short |dW| CV | <10% | 11.9% |
| Medium CV | ~24% | 14.1% |
