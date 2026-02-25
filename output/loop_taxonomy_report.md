# Loop Path Taxonomy Report

**Structures analyzed:** 14
**Total loops extracted:** 62

## Loop Direction Summary

| Direction | Count |
|-----------|-------|
| alpha->beta | 32 |
| beta->alpha | 30 |

## Loop Length Distribution

| Metric | Value |
|--------|-------|
| Min | 1 |
| Max | 18 |
| Mean | 6.3 |
| Median | 5.0 |

## Long (11-15)

**Loops:** 4
**DBSCAN eps:** 0.00
**Clusters:** 0 (0 tight, 0 catch-all)
**Noise:** 4 (100%)
**Coverage:** 0%

## Medium (8-10)

**Loops:** 14
**DBSCAN eps:** 0.50
**Clusters:** 0 (0 tight, 0 catch-all)
**Noise:** 14 (100%)
**Coverage:** 0%

## Short (<=7)

**Loops:** 41
**DBSCAN eps:** 0.50
**Clusters:** 2 (2 tight, 0 catch-all)
**Noise:** 35 (85%)
**Coverage:** 15%

| Family | N | |dW| mean | |dW| CV% | Path len mean | Path len CV% | Type | Directions |
|--------|---|----------|---------|---------------|-------------|------|------------|
| 0 | 3 | 0.159 | 1.3% | 3.523 | 1.0% | tight | beta->alpha:3 |
| 1 | 3 | 0.000 | 0.0% | 0.000 | 0.0% | tight | alpha->beta:1, beta->alpha:2 |

**Tight family |dW| CV range:** 0.0% - 1.3%

## Comparison to Targets (from CLAUDE.md)

| Metric | Target | Observed |
|--------|--------|----------|
| Total loops | 1000+ | 62 |
| Short tight families | ~30 | 2 |
| Short coverage | 100% | 15% |
| Short |dW| CV | <10% | 0.7% |
| Medium CV | ~24% | nan% |
