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
| 1UBQ | 76 | 4 | 4.066 | 17.377 | +13.311 |
| 1AKI | 129 | 0 | 3.506 | 9.667 | +6.161 |
| 1UBI | 76 | 2 | 2.112 | 22.979 | +20.867 |
| 1EJG | 46 | 2 | 0.926 | 6.818 | +5.892 |
| 1A8O | 33 | 0 | 0.706 | 3.050 | +2.344 |
| 1D3Z | 76 | 4 | 1.963 | 26.757 | +24.794 |
| 1DIX | 208 | 1 | 7.730 | 29.519 | +21.789 |
| 1GYA | 105 | 0 | 3.321 | 26.050 | +22.729 |
| 1K6P | 99 | 2 | 3.851 | 18.059 | +14.208 |
| 3O5R | 128 | 3 | 6.535 | 32.076 | +25.541 |

## Summary

| Metric | Value |
|--------|-------|
| Mean RMSD (experimental) | 3.472 A |
| Mean RMSD (canonical) | 19.235 A |
| Mean delta | +15.764 A |
| Max delta | +25.541 A |

## Interpretation

Using canonical loop path centroids instead of actual loop 
conformations adds an average of 15.8 A RMSD.

