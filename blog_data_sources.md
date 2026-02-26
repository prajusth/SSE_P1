# Blog Post Data Sources & Derivations

This document explains how each statistic in `blog.md` was derived from the experiment data. Not necessary for the project, but for clarity included. NOT TO BE SUBMITTED

## Source Files

| File | Description |
|------|-------------|
| `data/experiment_20260221_032827.csv` | Raw experiment data (2,520 runs) |
| `results/descriptive_stats.csv` | Per-configuration statistics |
| `results/normality_tests.csv` | Shapiro-Wilk test results |
| `results/provider_tests.csv` | Provider comparison statistics |
| `results/level_tests.csv` | Fast vs default comparison statistics |

---

## Results Section

### Overall Energy Consumption Table

**Source:** `data/experiment_20260221_032827.csv` (excluding warmup runs)

| Metric | Calculation |
|--------|-------------|
| Mean Energy | Average of `energy_joules` grouped by provider/level |
| Std Dev | Standard deviation of `energy_joules` grouped by provider/level |
| Mean Time | Average of `wall_time_s` grouped by provider/level |
| Compression Ratio | Average of `compression_ratio` grouped by provider/level |

**Derived values:**
```
7-Zip fast:    mean=52.00 J, std=64.74, time=3.32s, ratio=0.44
7-Zip default: mean=221.97 J, std=204.59, time=16.22s, ratio=0.39
gzip fast:     mean=40.34 J, std=51.14, time=5.33s, ratio=0.52
gzip default:  mean=66.32 J, std=68.92, time=8.45s, ratio=0.51
zstd fast:     mean=7.53 J, std=7.55, time=1.31s, ratio=0.54
zstd default:  mean=15.15 J, std=17.09, time=2.16s, ratio=0.53
```

### "12x less energy" Claim

**Calculation:**
```
zstd average = (7.53 + 15.15) / 2 = 11.34 J
7-Zip average = (52.00 + 221.97) / 2 = 136.99 J
gzip average = (40.34 + 66.32) / 2 = 53.33 J

Ratio: 136.99 / 11.34 = 12.08x (rounded to 12x)
Ratio: 53.33 / 11.34 = 4.70x (rounded to ~5x)
```

### Energy Scaling with File Size Table

**Source:** `results/descriptive_stats.csv`

**Calculation:** Average energy across all file types and both levels, grouped by provider and file size.

```
7-Zip:
  Small:  (fast: 1.90+4.53+4.73+1.89 + default: 26.36+12.29+13.45+27.49) / 8 = 11.58 J
  Medium: (fast: 13.51+35.75+41.50+14.14 + default: 197.21+100.87+117.41+230.82) / 8 = 93.90 J
  Large:  (fast: 63.72+170.01+204.23+68.07 + default: 500.24+347.88+471.37+618.14) / 8 = 305.46 J

gzip:
  Small:  3.03 J
  Medium: 27.02 J
  Large:  129.77 J

zstd:
  Small:  1.06 J
  Medium: 5.72 J
  Large:  27.22 J
```

### Energy per MB Values

**Source:** `results/descriptive_stats.csv`, column `epm_mean`

**Calculation:** Average of `epm_mean` across all file types and sizes for each provider/level.

```
zstd fast:     0.053 J/MB
zstd default:  0.084 J/MB
gzip fast:     0.193 J/MB
gzip default:  0.330 J/MB
7-Zip fast:    0.261 J/MB
7-Zip default: 1.473 J/MB
```

### Impact of File Type Table

**Source:** `results/descriptive_stats.csv`

**Calculation:** Average energy across all sizes and both levels, grouped by provider and file type.

```
7-Zip CSV:   (fast: 63.72+13.51+1.90 + default: 500.24+197.21+26.36) / 6 = 133.82 J
7-Zip Image: (fast: 170.01+35.75+4.53 + default: 347.88+100.87+12.29) / 6 = 111.89 J
7-Zip PDF:   (fast: 204.23+41.50+4.73 + default: 471.37+117.41+13.45) / 6 = 142.12 J
7-Zip Text:  (fast: 68.07+14.14+1.89 + default: 618.14+230.82+27.49) / 6 = 160.09 J
```

### Speed-Energy Tradeoff Table

**Calculation:** Percentage increase from fast to default.

```
7-Zip: (221.97 - 52.00) / 52.00 × 100 = 326.9% ≈ 327%
gzip:  (66.32 - 40.34) / 40.34 × 100 = 64.4% ≈ 64%
zstd:  (15.15 - 7.53) / 7.53 × 100 = 101.2% ≈ 101%
```

### Compression Ratio Improvements

**Calculation:** Percentage improvement in compression ratio.

```
7-Zip: (0.44 - 0.39) / 0.44 × 100 = 11.4% ≈ 11%
gzip:  (0.52 - 0.51) / 0.52 × 100 = 1.9% ≈ 2%
zstd:  (0.54 - 0.53) / 0.54 × 100 = 1.9% ≈ 2%
```

---

## Statistical Analysis Section

### Normality Testing

**Source:** `results/normality_tests.csv`

**Calculation:** Count of groups where `is_normal == True`

```
Total groups: 72
Normal (p > 0.05): 55
Percentage: 55/72 = 76.4%

Skewness range: min=-0.8795, max=2.6424
```

### Provider Comparisons - Effect Sizes

**Source:** `results/provider_tests.csv`

**Calculation:** Average Cohen's d for rows where `effect_type == "cohen_d"`, grouped by comparison.

```
7-Zip vs gzip: mean Cohen's d = 117.2 (13 tests with Cohen's d)
7-Zip vs zstd: mean Cohen's d = 172.8 (14 tests with Cohen's d)
gzip vs zstd:  mean Cohen's d = 191.7 (15 tests with Cohen's d)

Note: Some comparisons used Mann-Whitney U with CLES instead of Cohen's d
```

### Level Comparisons - Percentage Increase

**Source:** `results/level_tests.csv`

**Calculation:** Average of `pct_change` column, grouped by provider.

```
7-Zip: mean pct_change = 665.2%
gzip:  mean pct_change = 105.8% ≈ 106%
zstd:  mean pct_change = 76.2% ≈ 76%
```

### "Up to 1532% more energy"

**Source:** `results/level_tests.csv`, row for 7zip/text/medium

```
7-Zip text medium: pct_change = 1532.68%
Verification: (230.82 - 14.14) / 14.14 × 100 = 1532.7%
```

---

## Conclusion Section

### 1 TB Daily Backup Calculation

**Source:** Energy per MB values derived above.

```
7-Zip default: 1.473 J/MB × 1,000,000 MB = 1,473,000 J = 1,473 kJ/day ≈ 1,500 kJ/day
zstd fast:     0.053 J/MB × 1,000,000 MB = 53,000 J = 53 kJ/day ≈ 50 kJ/day

Daily savings: 1,473 - 53 = 1,420 kJ/day
Annual savings: 1,420 × 365 = 518,300 kJ = 518.3 MJ ≈ 520 MJ

Convert to kWh: 518.3 MJ / 3.6 MJ/kWh = 144 kWh
```
