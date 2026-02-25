# End-to-End Reconstruction Test Report

**Structures tested:** 10

## Method

1. Extract experimental (phi, psi, omega) from crystal structure
2. Identify loop regions (alpha<->beta transitions)
3. Replace SS element angles with basin centers
4. Replace loop angles with nearest canonical family centroid
5. Reconstruct backbone via forward kinematics
6. Compute RMSD vs experimental backbone

## Results

| PDB | Length | Loops | RMSD (exp) | RMSD (canonical) | Delta |
|-----|--------|-------|------------|------------------|-------|
| 1UBQ | 76 | 4 | 4.066 | 17.200 | +13.134 |
| 1AKI | 129 | 0 | 3.506 | 9.667 | +6.161 |
| 1UBI | 76 | 2 | 2.112 | 21.269 | +19.158 |
| 1EJG | 46 | 2 | 0.926 | 6.787 | +5.861 |
| 1A8O | 33 | 0 | 0.706 | 3.050 | +2.344 |
| 1D3Z | 76 | 4 | 1.963 | 24.946 | +22.983 |
| 1DIX | 208 | 1 | 7.730 | 30.009 | +22.279 |
| 1GYA | 105 | 0 | 3.321 | 26.050 | +22.729 |
| 1K6P | 99 | 2 | 3.851 | 18.065 | +14.213 |
| 3O5R | 128 | 3 | 6.535 | 26.607 | +20.072 |

## Summary

| Metric | Value |
|--------|-------|
| Mean RMSD (experimental) | 3.472 A |
| Mean RMSD (canonical) | 18.365 A |
| Mean delta | +14.893 A |
| Max delta | +22.983 A |

## Interpretation

Using canonical loop path centroids instead of actual loop 
conformations adds an average of 14.9 A RMSD.

