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
| 0 | 5 | 0.405 | 71.5% | 23.531 | 2.2% | catch-all | alpha->beta:5 |
| 1 | 3 | 1.452 | 47.9% | 28.412 | 0.7% | catch-all | alpha->beta:3 |

## Medium (8-10)

**Loops:** 208
**DBSCAN eps:** 0.25
**Clusters:** 2 (1 tight, 1 catch-all)
**Noise:** 202 (97%)
**Coverage:** 3%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 3 | 1.154 | 59.8% | 13.544 | 5.0% | catch-all | alpha->beta:3 |
| 1 | 3 | 0.943 | 10.0% | 12.408 | 2.3% | tight | beta->alpha:3 |

**Tight family |dW| CV range:** 10.0% - 10.0%

## Short (<=7)

**Loops:** 905
**DBSCAN eps:** 0.50
**Clusters:** 59 (23 tight, 36 catch-all)
**Noise:** 575 (64%)
**Coverage:** 36%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 26 | 0.454 | 69.5% | 7.082 | 5.5% | catch-all | alpha->beta:26 |
| 61 | 22 | 0.046 | 222.4% | 0.054 | 232.7% | catch-all | alpha->beta:1, beta->alpha:21 |
| 60 | 17 | 0.024 | 277.1% | 0.035 | 273.9% | catch-all | alpha->beta:3, beta->alpha:14 |
| 30 | 16 | 1.135 | 72.6% | 8.592 | 6.3% | catch-all | alpha->beta:16 |
| 7 | 15 | 1.409 | 58.5% | 6.100 | 18.6% | catch-all | alpha->beta:14, beta->alpha:1 |
| 16 | 15 | 3.001 | 55.0% | 6.228 | 5.8% | catch-all | alpha->beta:15 |
| 9 | 12 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:11, beta->alpha:1 |
| 45 | 10 | 2.010 | 7.7% | 0.916 | 12.7% | tight | beta->alpha:10 |
| 8 | 9 | 1.842 | 29.8% | 4.148 | 10.1% | tight | alpha->beta:8, beta->alpha:1 |
| 22 | 8 | 1.990 | 32.7% | 3.712 | 5.3% | catch-all | alpha->beta:8 |
| 31 | 7 | 0.960 | 63.7% | 7.157 | 18.5% | catch-all | beta->alpha:7 |
| 5 | 6 | 0.640 | 58.9% | 5.593 | 4.0% | catch-all | alpha->beta:6 |
| 25 | 6 | 0.798 | 96.2% | 7.636 | 11.9% | catch-all | alpha->beta:6 |
| 44 | 6 | 0.150 | 223.6% | 0.084 | 223.6% | catch-all | alpha->beta:4, beta->alpha:2 |
| 1 | 5 | 0.000 | 0.0% | 0.000 | 0.0% | tight | beta->alpha:5 |
| 14 | 5 | 0.905 | 58.9% | 7.655 | 6.6% | catch-all | alpha->beta:5 |
| 37 | 5 | 0.727 | 98.3% | 4.955 | 22.1% | catch-all | alpha->beta:4, beta->alpha:1 |
| 46 | 5 | 2.180 | 34.6% | 1.860 | 19.3% | catch-all | beta->alpha:5 |
| 17 | 4 | 0.448 | 86.2% | 6.539 | 8.4% | catch-all | beta->alpha:4 |
| 21 | 4 | 1.372 | 52.4% | 6.483 | 3.4% | catch-all | alpha->beta:3, beta->alpha:1 |
| 23 | 4 | 1.654 | 17.9% | 4.636 | 5.9% | tight | alpha->beta:3, beta->alpha:1 |
| 26 | 4 | 0.394 | 46.0% | 8.162 | 0.9% | catch-all | beta->alpha:4 |
| 27 | 4 | 0.727 | 89.2% | 9.205 | 4.2% | catch-all | alpha->beta:4 |
| 29 | 4 | 2.748 | 10.9% | 7.756 | 15.1% | tight | alpha->beta:4 |
| 43 | 4 | 1.197 | 26.2% | 3.617 | 4.6% | tight | beta->alpha:4 |
| 52 | 4 | 1.678 | 5.8% | 3.313 | 1.5% | tight | beta->alpha:4 |
| 54 | 4 | 0.440 | 42.2% | 6.056 | 2.6% | catch-all | alpha->beta:4 |
| 55 | 4 | 0.366 | 36.6% | 5.944 | 4.4% | catch-all | alpha->beta:4 |
| 58 | 4 | 0.526 | 87.2% | 4.132 | 4.9% | catch-all | beta->alpha:4 |
| 63 | 4 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:3 |
| 2 | 3 | 0.665 | 48.6% | 12.082 | 3.4% | catch-all | beta->alpha:3 |
| 6 | 3 | 0.285 | 27.5% | 8.692 | 1.8% | tight | alpha->beta:3 |
| 13 | 3 | 0.302 | 57.6% | 6.911 | 0.6% | catch-all | alpha->beta:3 |
| 15 | 3 | 0.912 | 25.0% | 3.589 | 9.2% | tight | alpha->beta:1, beta->alpha:2 |
| 18 | 3 | 0.337 | 80.6% | 14.003 | 2.4% | catch-all | alpha->beta:3 |
| 19 | 3 | 0.242 | 103.6% | 8.966 | 6.6% | catch-all | beta->alpha:3 |
| 20 | 3 | 2.275 | 27.3% | 5.796 | 1.4% | tight | beta->alpha:3 |
| 24 | 3 | 1.037 | 93.0% | 5.726 | 5.5% | catch-all | alpha->beta:3 |
| 32 | 3 | 0.858 | 35.4% | 5.860 | 5.9% | catch-all | beta->alpha:3 |
| 33 | 3 | 0.270 | 40.2% | 10.778 | 4.2% | catch-all | beta->alpha:3 |
| 34 | 3 | 3.671 | 11.3% | 2.677 | 5.5% | tight | beta->alpha:3 |
| 35 | 3 | 1.813 | 10.6% | 6.029 | 8.0% | tight | beta->alpha:3 |
| 36 | 3 | 0.743 | 44.3% | 7.679 | 6.3% | catch-all | beta->alpha:3 |
| 38 | 3 | 1.128 | 66.3% | 8.163 | 3.5% | catch-all | alpha->beta:3 |
| 39 | 3 | 0.307 | 27.8% | 8.803 | 16.2% | tight | alpha->beta:3 |
| 41 | 3 | 0.038 | 141.4% | 0.168 | 141.4% | catch-all | alpha->beta:1, beta->alpha:2 |
| 42 | 3 | 0.502 | 28.9% | 2.504 | 4.0% | tight | alpha->beta:3 |
| 47 | 3 | 0.487 | 5.4% | 2.836 | 3.1% | tight | alpha->beta:3 |
| 48 | 3 | 1.513 | 33.4% | 2.164 | 8.5% | catch-all | beta->alpha:3 |
| 49 | 3 | 1.209 | 1.3% | 3.588 | 4.3% | tight | beta->alpha:3 |
| 50 | 3 | 0.340 | 141.4% | 0.305 | 141.4% | catch-all | alpha->beta:1, beta->alpha:2 |
| 51 | 3 | 0.625 | 27.1% | 2.205 | 8.5% | tight | beta->alpha:3 |
| 53 | 3 | 2.267 | 3.9% | 3.174 | 1.7% | tight | beta->alpha:3 |
| 57 | 3 | 0.311 | 90.2% | 2.778 | 2.2% | catch-all | alpha->beta:3 |
| 59 | 3 | 1.586 | 46.3% | 4.710 | 6.0% | catch-all | beta->alpha:3 |
| 62 | 3 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:2 |
| 64 | 3 | 0.041 | 81.9% | 0.208 | 86.0% | catch-all | beta->alpha:3 |
| 65 | 3 | 1.113 | 6.7% | 2.965 | 1.6% | tight | alpha->beta:3 |
| 66 | 3 | 1.387 | 9.5% | 2.757 | 1.5% | tight | alpha->beta:3 |

**Tight family |dW| CV range:** 0.0% - 29.8%

## Comparison to Targets (from CLAUDE.md)

| Metric | Target | Observed |
|--------|--------|----------|
| Total loops | 1000+ | 1344 |
| Short tight families | ~30 | 23 |
| Short coverage | 100% | 36% |
| Short |dW| CV | <10% | 13.5% |
| Medium CV | ~24% | 10.0% |
