# PDB Validation Report: BPS/L

**Data source:** Cached PDB files (265 structures)
**Structures processed:** 265

## Filtered Statistics (X-ray, length >= 50)

**Quality filters applied:** exclude NMR (18 removed), exclude chains < 50 residues (19 removed)
**Structures after filtering:** 235

| Metric | Value |
|--------|-------|
| Mean BPS/L | 0.957 |
| Std BPS/L | 0.178 |
| CV | 18.6% |
| Median BPS/L | 0.971 |
| Min BPS/L | 0.381 |
| Max BPS/L | 1.685 |

## Unfiltered Statistics (all structures)

| Metric | Value |
|--------|-------|
| Mean BPS/L | 0.978 |
| Std BPS/L | 0.213 |
| CV | 21.7% |
| Median BPS/L | 0.973 |
| Min BPS/L | 0.381 |
| Max BPS/L | 2.018 |

## BPS/L Histogram (all structures)

```
  0.381-0.463 | # (2)
  0.463-0.544 | ## (4)
  0.544-0.626 | ## (4)
  0.626-0.708 | ####### (11)
  0.708-0.790 | ####### (11)
  0.790-0.872 | ############################# (44)
  0.872-0.954 | ############################ (43)
  0.954-1.035 | ######################################## (60)
  1.035-1.117 | ############################# (44)
  1.117-1.199 | ############## (21)
  1.199-1.281 | ### (5)
  1.281-1.363 | # (2)
  1.363-1.445 | ## (4)
  1.445-1.526 | # (2)
  1.526-1.608 | ## (4)
  1.608-1.690 | # (2)
  1.690-1.772 |  (0)
  1.772-1.854 |  (0)
  1.854-1.936 |  (1)
  1.936-2.018 |  (1)
```

## Secondary Structure Composition (computed from phi/psi)

| SS Type | Mean Fraction |
|---------|---------------|
| alpha | 39.2% |
| beta | 26.1% |
| other | 17.0% |

## Three-Level Backbone Decomposition

The Markov/Real ratio measures conformational coherence — how much
of backbone geometry is structured beyond what transition frequencies
alone predict. The ratio decreases monotonically with W smoothing.
Maximum separation occurs with the least-smoothed (realistic) W.

| W construction | Real | Markov | Shuffled | M/R | S/R |
|----------------|------|--------|----------|-----|-----|
| Realistic (KDE, no extra smoothing) | 0.957 | 1.086 | 1.359 | 1.13x | 1.42x |
| Smooth (KDE + sigma=60 deg) | 0.308 | 0.312 | 0.381 | 1.01x | 1.24x |

**Note:** The Markov/Real ratio decreases monotonically with smoothing. The smooth W used for the low BPS/L value acts as a low-pass filter isolating inter-basin transitions — since Markov chains preserve transition frequencies by construction, M/R collapses to ~1.0. The realistic W reveals intra-basin conformational coherence that Markov chains cannot reproduce.

## Fold-Class Breakdown (Realistic W)

BPS/L by SCOP-like fold class, classified from computed SS fractions.

| Fold class | N | Mean BPS/L | Std | CV% | Median | alpha% | beta% |
|------------|---|------------|-----|-----|--------|--------|-------|
| all-alpha | 30 | 0.782 | 0.210 | 26.8% | 0.814 | 72.2% | 5.9% |
| all-beta | 31 | 1.076 | 0.195 | 18.2% | 1.042 | 11.4% | 48.5% |
| alpha/beta | 145 | 0.953 | 0.140 | 14.7% | 0.951 | 40.8% | 25.4% |
| alpha+beta | 29 | 1.028 | 0.145 | 14.1% | 1.010 | 30.8% | 28.8% |

**Key separations (Cohen's d):**
- all-alpha vs all-beta: d = -1.45 (strong separation)
- all-alpha vs alpha/beta: d = -0.96 (large effect)
- all-beta vs alpha/beta: d = +0.72 (medium effect)

All-alpha proteins have the lowest BPS/L (0.782) — alpha helices are conformationally
smooth within the W landscape. All-beta proteins have the highest (1.076) — beta strands
traverse more varied W terrain. The 0.294 spread between fold-class means is 1.65x the
overall standard deviation, confirming fold-class separation persists with realistic W.

## Per-Structure Results

| # | Name | Length | BPS/L | alpha | beta | other | NMR |
|---|------|--------|-------|-------|------|-------|-----|
| 1 | 1A8O | 66 | 0.880 | 61.7% | 1.7% | 13.3% |  |
| 2 | 1AAF | 55 | 1.856 | 17.0% | 17.0% | 52.8% | Y |
| 3 | 1ACJ | 528 | 1.082 | 40.1% | 22.9% | 19.5% |  |
| 4 | 1AHQ | 52 | 0.843 | 42.0% | 22.0% | 10.0% |  |
| 5 | 1AKE | 214 | 0.805 | 53.3% | 18.9% | 14.6% |  |
| 6 | 1AKI | 129 | 1.010 | 44.9% | 11.8% | 21.3% |  |
| 7 | 1AKU | 147 | 0.894 | 44.8% | 24.1% | 13.8% |  |
| 8 | 1AQ7 | 223 | 1.151 | 17.6% | 31.7% | 22.2% |  |
| 9 | 1AS5 | 20 | 1.476 | 18.8% | 18.8% | 50.0% | Y |
| 10 | 1AY8 | 394 | 0.894 | 54.6% | 16.3% | 13.8% |  |
| 11 | 1AZ5 | 95 | 0.959 | 13.2% | 56.0% | 12.1% |  |
| 12 | 1B5S | 242 | 0.969 | 40.8% | 28.3% | 15.8% | Y |
| 13 | 1BE9 | 115 | 1.158 | 24.8% | 31.0% | 21.2% |  |
| 14 | 1BJU | 223 | 1.076 | 16.7% | 34.8% | 19.9% |  |
| 15 | 1BLU | 80 | 1.083 | 39.7% | 19.2% | 17.9% |  |
| 16 | 1BMA | 240 | 1.101 | 19.7% | 33.6% | 18.1% |  |
| 17 | 1BPI | 58 | 1.092 | 30.4% | 26.8% | 17.9% |  |
| 18 | 1CBN | 46 | 0.828 | 52.3% | 20.5% | 15.9% |  |
| 19 | 1CKB | 57 | 1.019 | 16.4% | 40.0% | 20.0% |  |
| 20 | 1CPH | 21 | 0.862 | 57.9% | 10.5% | 15.8% |  |
| 21 | 1CRN | 46 | 0.873 | 52.3% | 20.5% | 15.9% |  |
| 22 | 1CRR | 166 | 1.641 | 36.6% | 18.3% | 30.5% | Y |
| 23 | 1CZF | 335 | 1.163 | 11.1% | 48.0% | 15.6% |  |
| 24 | 1CZN | 169 | 0.883 | 44.3% | 28.7% | 14.4% |  |
| 25 | 1D0Q | 102 | 0.960 | 46.0% | 25.0% | 16.0% |  |
| 26 | 1D2F | 361 | 0.909 | 48.2% | 21.4% | 16.2% |  |
| 27 | 1D2S | 170 | 0.954 | 10.2% | 56.0% | 17.5% |  |
| 28 | 1D3G | 360 | 0.998 | 50.0% | 17.8% | 15.8% |  |
| 29 | 1D3Z | 76 | 0.657 | 27.0% | 41.9% | 10.8% | Y |
| 30 | 1D4A | 273 | 1.027 | 45.8% | 17.3% | 15.5% |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 261 | 6XLU | 1057 | 0.991 | 26.1% | 36.3% | 18.1% |  |
| 262 | 7DDO | 597 | 0.858 | 64.5% | 11.9% | 13.4% |  |
| 263 | 7PBL | 312 | 0.786 | 52.9% | 17.1% | 10.0% |  |
| 264 | 7TAA | 476 | 0.943 | 39.0% | 22.6% | 18.6% |  |
| 265 | 7TIM | 247 | 1.014 | 49.0% | 21.2% | 16.3% |  |
