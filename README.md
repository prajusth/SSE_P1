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

- sudo privileges
- Python 3
- Internet connection (for dependency installation)

**Note:** This experiment was designed and tested exclusively on Ubuntu Linux. While some scripts might function on other distributions or operating systems, compatibility is not guaranteed.

The setup script installs:

- 7-Zip
- gzip
- zstd
- Rust (if not present)
- EnergiBridge
- Python virtual environment

---

## Quick Start

### 1. Elevate Privileges & Setup

The setup and execution require root access. Start a root shell before proceeding:

```bash
sudo -s
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

WARNING: This script automatically triggers Zen Mode, which kills background applications (browsers, Slack, Discord, Spotify, etc.) to minimize noise in energy measurements. Save your work before executing this command.

```bash
python3 scripts/run_experiment.py
```

Expected runtime on CPU:
- Approximately 12 hours (assuming default cooldown intervals in utils.py).
- Default configuration: 30 repetitions per workload

Energy measurements are collected using EnergiBridge.

---

### 5. Run Analysis

```bash
python3 analysis/analyze.py
python3 analysis/analyze_non_normals.py

```
Optional: If you need to see the large-format visualizations as presented in the blog, you can run:

```bash
python3 analysis/analyze_large.py

```
This produces:
- Plots
- Statistical tests
- Normality and non-parametric analyses

---


## Notes

- The experiment is computationally intensive.
- Do not run other applications or background workloads during the benchmark, as this will skew the energy measurements.
- Results may vary depending on hardware.


