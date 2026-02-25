# Loop Path Taxonomy Report

**Structures analyzed:** 184
**Total loops extracted:** 1130

## Loop Direction Summary

| Direction | Count |
|-----------|-------|
| alpha->beta | 572 |
| beta->alpha | 558 |

## Loop Length Distribution

| Metric | Value |
|--------|-------|
| Min | 1 |
| Max | 20 |
| Mean | 6.5 |
| Median | 5.0 |

## Long (11-15)

**Loops:** 137
**DBSCAN eps:** 0.30
**Clusters:** 2 (1 tight, 1 catch-all)
**Noise:** 129 (94%)
**Coverage:** 6%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 5 | 0.048 | 59.8% | 23.531 | 2.2% | catch-all | alpha->beta:5 |
| 1 | 3 | 0.218 | 25.4% | 28.412 | 0.7% | tight | alpha->beta:3 |

**Tight family |dW| CV range:** 25.4% - 25.4%

## Medium (8-10)

**Loops:** 167
**DBSCAN eps:** 0.25
**Clusters:** 2 (1 tight, 1 catch-all)
**Noise:** 161 (96%)
**Coverage:** 4%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 3 | 0.121 | 45.1% | 13.544 | 5.0% | catch-all | alpha->beta:3 |
| 1 | 3 | 0.775 | 3.1% | 12.408 | 2.3% | tight | beta->alpha:3 |

**Tight family |dW| CV range:** 3.1% - 3.1%

## Short (<=7)

**Loops:** 762
**DBSCAN eps:** 0.50
**Clusters:** 54 (26 tight, 28 catch-all)
**Noise:** 469 (62%)
**Coverage:** 38%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 20 | 0.064 | 58.7% | 7.022 | 5.7% | catch-all | alpha->beta:20 |
| 12 | 16 | 0.863 | 15.3% | 2.849 | 8.9% | tight | alpha->beta:15, beta->alpha:1 |
| 24 | 15 | 0.107 | 74.6% | 8.521 | 5.7% | catch-all | alpha->beta:15 |
| 10 | 14 | 0.256 | 56.0% | 3.346 | 12.7% | catch-all | alpha->beta:1, beta->alpha:13 |
| 17 | 13 | 1.024 | 20.4% | 6.181 | 5.8% | tight | alpha->beta:13 |
| 54 | 12 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:2, beta->alpha:10 |
| 43 | 10 | 0.122 | 24.0% | 0.916 | 12.7% | tight | beta->alpha:10 |
| 7 | 9 | 0.080 | 83.7% | 5.552 | 11.9% | catch-all | alpha->beta:9 |
| 9 | 9 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:8, beta->alpha:1 |
| 55 | 8 | 0.000 | 0.0% | 0.000 | 0.0% | tight | beta->alpha:8 |
| 8 | 7 | 0.962 | 9.8% | 4.160 | 11.3% | tight | alpha->beta:7 |
| 25 | 7 | 0.062 | 70.2% | 7.157 | 18.5% | catch-all | beta->alpha:7 |
| 19 | 6 | 0.129 | 54.2% | 2.659 | 9.9% | catch-all | alpha->beta:6 |
| 21 | 6 | 0.081 | 67.5% | 2.786 | 16.0% | catch-all | alpha->beta:5, beta->alpha:1 |
| 53 | 6 | 0.001 | 223.6% | 0.011 | 223.6% | catch-all | beta->alpha:6 |
| 5 | 5 | 0.033 | 74.9% | 5.540 | 3.7% | catch-all | alpha->beta:5 |
| 44 | 5 | 0.175 | 37.4% | 1.860 | 19.3% | catch-all | beta->alpha:5 |
| 56 | 5 | 0.014 | 135.6% | 0.134 | 122.6% | catch-all | beta->alpha:5 |
| 1 | 4 | 0.000 | 0.0% | 0.000 | 0.0% | tight | beta->alpha:4 |
| 15 | 4 | 0.080 | 70.7% | 6.483 | 3.4% | catch-all | alpha->beta:3, beta->alpha:1 |
| 18 | 4 | 0.053 | 79.1% | 9.205 | 4.2% | catch-all | alpha->beta:4 |
| 22 | 4 | 0.179 | 32.2% | 7.756 | 15.1% | catch-all | alpha->beta:4 |
| 27 | 4 | 0.078 | 60.9% | 7.101 | 8.2% | catch-all | alpha->beta:4 |
| 36 | 4 | 0.799 | 4.8% | 4.521 | 16.5% | tight | alpha->beta:3, beta->alpha:1 |
| 38 | 4 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:3 |
| 42 | 4 | 0.160 | 1.6% | 3.617 | 4.6% | tight | beta->alpha:4 |
| 49 | 4 | 0.060 | 19.2% | 6.056 | 2.6% | tight | alpha->beta:4 |
| 50 | 4 | 0.061 | 94.6% | 5.944 | 4.4% | catch-all | alpha->beta:4 |
| 51 | 4 | 0.061 | 56.5% | 4.132 | 4.9% | catch-all | beta->alpha:4 |
| 57 | 4 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:3 |
| 2 | 3 | 0.405 | 26.4% | 12.082 | 3.4% | tight | beta->alpha:3 |
| 6 | 3 | 0.860 | 6.2% | 8.692 | 1.8% | tight | alpha->beta:3 |
| 13 | 3 | 0.078 | 15.4% | 6.911 | 0.6% | tight | alpha->beta:3 |
| 14 | 3 | 0.059 | 53.9% | 14.003 | 2.4% | catch-all | alpha->beta:3 |
| 16 | 3 | 0.096 | 55.0% | 5.726 | 5.5% | catch-all | alpha->beta:3 |
| 23 | 3 | 0.084 | 27.4% | 7.711 | 4.1% | tight | alpha->beta:3 |
| 26 | 3 | 0.091 | 23.4% | 8.159 | 1.0% | tight | beta->alpha:3 |
| 28 | 3 | 0.950 | 6.7% | 3.746 | 6.0% | tight | alpha->beta:3 |
| 29 | 3 | 0.126 | 66.0% | 5.860 | 5.9% | catch-all | beta->alpha:3 |
| 30 | 3 | 0.032 | 71.8% | 10.778 | 4.2% | catch-all | beta->alpha:3 |
| 31 | 3 | 0.112 | 54.3% | 6.738 | 7.4% | catch-all | beta->alpha:3 |
| 32 | 3 | 0.493 | 10.8% | 2.677 | 5.5% | tight | beta->alpha:3 |
| 33 | 3 | 0.075 | 29.6% | 6.029 | 8.0% | tight | beta->alpha:3 |
| 34 | 3 | 0.058 | 120.2% | 7.679 | 6.3% | catch-all | beta->alpha:3 |
| 35 | 3 | 0.963 | 6.3% | 8.163 | 3.5% | tight | alpha->beta:3 |
| 37 | 3 | 0.049 | 43.1% | 8.803 | 16.2% | catch-all | alpha->beta:3 |
| 40 | 3 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:2 |
| 41 | 3 | 0.078 | 17.9% | 2.504 | 4.0% | tight | alpha->beta:3 |
| 45 | 3 | 0.152 | 8.8% | 2.164 | 8.5% | tight | beta->alpha:3 |
| 46 | 3 | 0.193 | 8.4% | 3.588 | 4.3% | tight | beta->alpha:3 |
| 47 | 3 | 0.024 | 141.4% | 0.305 | 141.4% | catch-all | alpha->beta:1, beta->alpha:2 |
| 48 | 3 | 0.043 | 41.2% | 2.205 | 8.5% | catch-all | beta->alpha:3 |
| 52 | 3 | 0.075 | 124.8% | 4.710 | 6.0% | catch-all | beta->alpha:3 |
| 58 | 3 | 0.024 | 95.7% | 0.208 | 86.0% | catch-all | beta->alpha:3 |

**Tight family |dW| CV range:** 0.0% - 29.6%

## Comparison to Targets (from CLAUDE.md)

| Metric | Target | Observed |
|--------|--------|----------|
| Total loops | 1000+ | 1130 |
| Short tight families | ~30 | 26 |
| Short coverage | 100% | 38% |
| Short |dW| CV | <10% | 10.9% |
| Medium CV | ~24% | 3.1% |
