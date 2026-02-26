# Zip Wars - Which Compression Tool is Burning Through Your Energy?

Every time a server rotates logs, a CI pipeline bundles an artifact, or a cloud service packages data for transfer, a compression algorithm consumes CPU cycles - and therefore energy. On a single machine, the impact is negligible. At infrastructure scale, it is not.

This project benchmarks the energy consumption of three widely used compression tools:

- 7-Zip (LZMA2)
- gzip (DEFLATE)
- Zstandard (zstd)

We measure energy usage across realistic workloads and analyze how input characteristics influence consumption. The results show that there is no universally best tool - energy efficiency depends strongly on the type of data and algorithmic design.

All scripts, raw data, and analysis code are included for full reproducibility.

---

## Requirements

- Ubuntu-based Linux system
- sudo privileges
- Python 3
- Internet connection (for dependency installation)

The setup script installs:

- 7-Zip
- gzip
- zstd
- Rust (if not present)
- EnergiBridge
- Python virtual environment

---

## Quick Start

### 1. Setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

---

### 2. Activate Environment

```bash
source .venv/bin/activate
```

---

### 3. Generate Test Data

```bash
python3 scripts/generate_test_data.py
```

---

### 4. Run Full Experiment

```bash
python3 scripts/run_experiment.py
```

Expected runtime:
- Approximately 4-6 hours
- Default configuration: 30 repetitions per workload

Energy measurements are collected using EnergiBridge.

---

### 5. Run Analysis

```bash
python3 analysis/analyze.py
python3 analysis/analyze_non_normals.py
```

This produces:
- Plots
- Statistical tests
- Normality and non-parametric analyses

---


## Notes

- The experiment is computationally intensive.
- Avoid running other heavy workloads during benchmarking.
- Results may vary depending on hardware.


