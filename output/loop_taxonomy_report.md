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
**Clusters:** 2 (1 tight, 1 catch-all)
**Noise:** 150 (95%)
**Coverage:** 5%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 5 | 0.048 | 59.8% | 23.531 | 2.2% | catch-all | alpha->beta:5 |
| 1 | 3 | 0.218 | 25.4% | 28.412 | 0.7% | tight | alpha->beta:3 |

**Tight family |dW| CV range:** 25.4% - 25.4%

## Medium (8-10)

**Loops:** 208
**DBSCAN eps:** 0.25
**Clusters:** 2 (1 tight, 1 catch-all)
**Noise:** 202 (97%)
**Coverage:** 3%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 3 | 0.121 | 45.1% | 13.544 | 5.0% | catch-all | alpha->beta:3 |
| 1 | 3 | 0.775 | 3.1% | 12.408 | 2.3% | tight | beta->alpha:3 |

**Tight family |dW| CV range:** 3.1% - 3.1%

## Short (<=7)

**Loops:** 905
**DBSCAN eps:** 0.50
**Clusters:** 57 (29 tight, 28 catch-all)
**Noise:** 563 (62%)
**Coverage:** 38%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 26 | 0.064 | 60.4% | 7.082 | 5.5% | catch-all | alpha->beta:26 |
| 59 | 22 | 0.003 | 305.9% | 0.054 | 232.7% | catch-all | alpha->beta:1, beta->alpha:21 |
| 12 | 21 | 0.867 | 15.0% | 2.838 | 8.3% | tight | alpha->beta:19, beta->alpha:2 |
| 58 | 17 | 0.004 | 331.1% | 0.035 | 273.9% | catch-all | alpha->beta:3, beta->alpha:14 |
| 30 | 16 | 0.100 | 80.4% | 8.592 | 6.3% | catch-all | alpha->beta:16 |
| 7 | 15 | 0.105 | 70.2% | 6.100 | 18.6% | catch-all | alpha->beta:14, beta->alpha:1 |
| 16 | 15 | 1.055 | 19.9% | 6.228 | 5.8% | tight | alpha->beta:15 |
| 9 | 12 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:11, beta->alpha:1 |
| 45 | 10 | 0.122 | 24.0% | 0.916 | 12.7% | tight | beta->alpha:10 |
| 8 | 9 | 0.955 | 9.4% | 4.148 | 10.1% | tight | alpha->beta:8, beta->alpha:1 |
| 22 | 8 | 0.984 | 8.9% | 3.712 | 5.3% | tight | alpha->beta:8 |
| 31 | 7 | 0.062 | 70.2% | 7.157 | 18.5% | catch-all | beta->alpha:7 |
| 5 | 6 | 0.046 | 81.8% | 5.593 | 4.0% | catch-all | alpha->beta:6 |
| 25 | 6 | 0.087 | 46.9% | 7.636 | 11.9% | catch-all | alpha->beta:6 |
| 44 | 6 | 0.022 | 223.6% | 0.084 | 223.6% | catch-all | alpha->beta:4, beta->alpha:2 |
| 1 | 5 | 0.000 | 0.0% | 0.000 | 0.0% | tight | beta->alpha:5 |
| 14 | 5 | 0.093 | 24.9% | 7.655 | 6.6% | tight | alpha->beta:5 |
| 37 | 5 | 0.801 | 4.3% | 4.955 | 22.1% | tight | alpha->beta:4, beta->alpha:1 |
| 46 | 5 | 0.175 | 37.4% | 1.860 | 19.3% | catch-all | beta->alpha:5 |
| 17 | 4 | 0.085 | 83.1% | 6.539 | 8.4% | catch-all | beta->alpha:4 |
| 21 | 4 | 0.080 | 70.7% | 6.483 | 3.4% | catch-all | alpha->beta:3, beta->alpha:1 |
| 23 | 4 | 0.845 | 2.1% | 4.636 | 5.9% | tight | alpha->beta:3, beta->alpha:1 |
| 26 | 4 | 0.090 | 20.7% | 8.162 | 0.9% | tight | beta->alpha:4 |
| 27 | 4 | 0.053 | 79.1% | 9.205 | 4.2% | catch-all | alpha->beta:4 |
| 29 | 4 | 0.179 | 32.2% | 7.756 | 15.1% | catch-all | alpha->beta:4 |
| 43 | 4 | 0.160 | 1.6% | 3.617 | 4.6% | tight | beta->alpha:4 |
| 52 | 4 | 0.353 | 1.2% | 3.313 | 1.5% | tight | beta->alpha:4 |
| 54 | 4 | 0.060 | 19.2% | 6.056 | 2.6% | tight | alpha->beta:4 |
| 55 | 4 | 0.061 | 94.6% | 5.944 | 4.4% | catch-all | alpha->beta:4 |
| 56 | 4 | 0.061 | 56.5% | 4.132 | 4.9% | catch-all | beta->alpha:4 |
| 61 | 4 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:3 |
| 2 | 3 | 0.405 | 26.4% | 12.082 | 3.4% | tight | beta->alpha:3 |
| 6 | 3 | 0.860 | 6.2% | 8.692 | 1.8% | tight | alpha->beta:3 |
| 13 | 3 | 0.078 | 15.4% | 6.911 | 0.6% | tight | alpha->beta:3 |
| 15 | 3 | 1.089 | 9.2% | 3.589 | 9.2% | tight | alpha->beta:1, beta->alpha:2 |
| 18 | 3 | 0.059 | 53.9% | 14.003 | 2.4% | catch-all | alpha->beta:3 |
| 19 | 3 | 0.064 | 48.8% | 8.966 | 6.6% | catch-all | beta->alpha:3 |
| 20 | 3 | 1.002 | 3.3% | 5.796 | 1.4% | tight | beta->alpha:3 |
| 24 | 3 | 0.096 | 55.0% | 5.726 | 5.5% | catch-all | alpha->beta:3 |
| 32 | 3 | 0.126 | 66.0% | 5.860 | 5.9% | catch-all | beta->alpha:3 |
| 33 | 3 | 0.032 | 71.8% | 10.778 | 4.2% | catch-all | beta->alpha:3 |
| 34 | 3 | 0.493 | 10.8% | 2.677 | 5.5% | tight | beta->alpha:3 |
| 35 | 3 | 0.075 | 29.6% | 6.029 | 8.0% | tight | beta->alpha:3 |
| 36 | 3 | 0.058 | 120.2% | 7.679 | 6.3% | catch-all | beta->alpha:3 |
| 38 | 3 | 0.963 | 6.3% | 8.163 | 3.5% | tight | alpha->beta:3 |
| 39 | 3 | 0.049 | 43.1% | 8.803 | 16.2% | catch-all | alpha->beta:3 |
| 41 | 3 | 0.013 | 141.4% | 0.168 | 141.4% | catch-all | alpha->beta:1, beta->alpha:2 |
| 42 | 3 | 0.078 | 17.9% | 2.504 | 4.0% | tight | alpha->beta:3 |
| 47 | 3 | 0.175 | 22.7% | 2.836 | 3.1% | tight | alpha->beta:3 |
| 48 | 3 | 0.152 | 8.8% | 2.164 | 8.5% | tight | beta->alpha:3 |
| 49 | 3 | 0.193 | 8.4% | 3.588 | 4.3% | tight | beta->alpha:3 |
| 50 | 3 | 0.024 | 141.4% | 0.305 | 141.4% | catch-all | alpha->beta:1, beta->alpha:2 |
| 51 | 3 | 0.043 | 41.2% | 2.205 | 8.5% | catch-all | beta->alpha:3 |
| 53 | 3 | 0.375 | 6.9% | 3.174 | 1.7% | tight | beta->alpha:3 |
| 57 | 3 | 0.075 | 124.8% | 4.710 | 6.0% | catch-all | beta->alpha:3 |
| 60 | 3 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:2 |
| 62 | 3 | 0.024 | 95.7% | 0.208 | 86.0% | catch-all | beta->alpha:3 |

**Tight family |dW| CV range:** 0.0% - 29.6%

## Comparison to Targets (from CLAUDE.md)

| Metric | Target | Observed |
|--------|--------|----------|
| Total loops | 1000+ | 1344 |
| Short tight families | ~30 | 29 |
| Short coverage | 100% | 38% |
| Short |dW| CV | <10% | 11.1% |
| Medium CV | ~24% | 3.1% |
