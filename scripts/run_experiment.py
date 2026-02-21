"""
run_experiment.py — Main experiment runner.

This script:
  1. Enters "Zen mode" (kills background apps)
  2. Warms up the CPU (Fibonacci for 2 min)
  3. Loops through all (provider, level, file_type, size) combinations
  4. Shuffles the order each repetition (course requirement)
  5. For each combo: runs the compression wrapped in EnergiBridge
  6. Reads the EnergiBridge CSV to get energy in Joules
  7. Handles RAPL counter overflow (AMD wraps at 65536J)
  8. Writes all results to one CSV file

You can run the full experiment with:
    python3 scripts/run_experiment.py

  # Quick test (~5 min)
    python3 scripts/run_experiment.py -r 3 --types text --sizes small --skip-warmup

  # Dry run (just prints what would happen)
    python3 scripts/run_experiment.py --dry-run
"""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROVIDERS, FILE_TYPES, FILE_SIZES,
    NUM_REPS, NUM_WARMUP, COOLDOWN_SECS, CPU_WARMUP_SECS,
    ENERGIBRIDGE, RAPL_OVERFLOW,
    DATA_DIR, TEST_DATA_DIR,
    timestamp, zen_mode, check_energibridge,
    get_test_file_path, get_system_info, logger,
)
from warmup import warmup


# column names for the output CSV
COLUMNS = [
    "run_id",             # unique number for each run
    "provider",           # 7zip, gzip, zstd
    "level",              # fast, default
    "file_type",          # text, csv, pdf, image
    "file_size",          # small, medium, large
    "repetition",         # which rep (1-based)
    "is_warmup",          # True for the first NUM_WARMUP rounds
    "input_bytes",        # size of input file in bytes
    "output_bytes",       # size of compressed file in bytes
    "compression_ratio",  # output_bytes / input_bytes (lower = better)
    "wall_time_s",        # total wall clock time (includes EnergiBridge overhead)
    "energy_joules",      # energy measured by RAPL (corrected for overflow)
    "energy_per_mb",      # energy_joules / (input_bytes in MB)
    "rapl_overflow",      # True if we had to correct for RAPL overflow
]



def read_energy(csv_path):
    """
    read total energy from an EnergiBridge CSV file.

    EnergiBridge writes a CSV with periodic readings. Energy columns are
    monotonic counters (they count up from some starting value).
    Total energy = last reading - first reading.

    On AMD CPUs: column is "CPU_ENERGY (J)"
    On Intel CPUs: column is "PACKAGE_ENERGY (J)"
    On Mac: column is "SYSTEM_POWER (Watts)" — needs trapezoid integration

    Returns: (energy_joules, did_overflow)
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.warning(f"  Could not read {csv_path}: {e}")
        return None, False

    # try to find an energy counter column (AMD or Intel)
    energy_cols = [c for c in df.columns
                   if "energy" in c.lower() and "j" in c.lower()]

    if energy_cols:
        preferred = ["PACKAGE_ENERGY (J)", "CPU_ENERGY (J)"]
        col = None
        for name in preferred:
            if name in energy_cols:
                col = name
                break
        if col is None:
            col = energy_cols[0]  # fallback to whatever we found

        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) < 2:
            return None, False

        energy = float(vals.iloc[-1] - vals.iloc[0])

        # check for RAPL overflow (counter wrapped around)
        if energy < 0:
            energy += RAPL_OVERFLOW
            return round(energy, 4), True

        return round(energy, 4), False

    # Mac: SYSTEM_POWER in Watts — integrate with trapezoid rule
    power_cols = [c for c in df.columns
                  if "power" in c.lower() and "watt" in c.lower()]
    if power_cols and "Time" in df.columns:
        import numpy as np
        power = pd.to_numeric(df[power_cols[0]], errors="coerce").dropna()
        times = pd.to_numeric(df["Time"], errors="coerce").dropna() / 1000
        if len(power) >= 2:
            energy = float(np.trapezoid(power, times))
            return round(energy, 4), False

    return None, False


def run_one(provider, level, input_path, eb_csv_path, use_eb):
    """
    Compress one file and measure energy.

    Returns dict with: wall_time_s, energy_joules, output_bytes, rapl_overflow, compressed_path 
    [Or None if the run failed]
    """
    prov = PROVIDERS[provider]
    ext = prov["ext"]
    cmd_template = prov[level]

    # create temp dir for output after zipping (and for gzip input since it modifies in-place)
    tmpdir = tempfile.mkdtemp()

    try:
        # gzip is special: it compresses in-place, so we copy the file first
        if provider == "gzip":
            tmp_input = os.path.join(tmpdir, os.path.basename(input_path))
            shutil.copy2(input_path, tmp_input)
            cmd = cmd_template.format(input=tmp_input, output="")
            compressed_path = tmp_input + ".gz"
        else:
            output_path = os.path.join(tmpdir, f"compressed{ext}")
            cmd = cmd_template.format(input=input_path, output=output_path)
            compressed_path = output_path

        # wrap in EnergiBridge if available
        if use_eb:
            full_cmd = f"{ENERGIBRIDGE} --output {eb_csv_path} --summary {cmd}"
        else:
            full_cmd = cmd

        t0 = time.perf_counter()
        result = subprocess.run(
            full_cmd, shell=True,
            capture_output=True, text=True,
            timeout=600  # 10 min max per run
        )
        t1 = time.perf_counter()

        if result.returncode != 0:
            logger.warning(f"  FAILED: {result.stderr[:150]}")
            return None

        if os.path.exists(compressed_path):
            output_bytes = os.path.getsize(compressed_path)
        else:
            output_bytes = 0

        energy, overflow = None, False
        if use_eb and os.path.exists(eb_csv_path):
            energy, overflow = read_energy(eb_csv_path)

        return {
            "wall_time_s": round(t1 - t0, 6),
            "energy_joules": energy,
            "output_bytes": output_bytes,
            "rapl_overflow": overflow,
        }

    finally:
        # Clean up temp files
        shutil.rmtree(tmpdir, ignore_errors=True)

def run_experiment(reps, file_types, sizes, dry_run, use_eb, skip_warmup):
    ts = timestamp()
    output_file = DATA_DIR / f"experiment_{ts}.csv"
    eb_dir = DATA_DIR / f"energibridge_{ts}"
    eb_dir.mkdir(exist_ok=True)

    # to save system info for reproducibility
    with open(DATA_DIR / f"system_info_{ts}.json", "w") as f:
        json.dump(get_system_info(), f, indent=2)

    # to check if EnergiBridge is available before we start the long experiment
    if use_eb and not check_energibridge():
        logger.warning("EnergiBridge not found! Running time-only mode.")
        use_eb = False

    # to check if all test files are present before we start the long experiment
    for ft in file_types:
        for sz in sizes:
            path = get_test_file_path(ft, sz)
            if not os.path.exists(path):
                logger.error(f"Missing: {path}")
                logger.error("Run: python3 scripts/generate_test_data.py")
                sys.exit(1)

    # zen mode attempt 
    zen_mode()

    # to warm up CPU with Fibonacci
    if not skip_warmup:
        warmup(CPU_WARMUP_SECS)
    else:
        logger.info("Skipping CPU warm-up (--skip-warmup)")

    # each config in the configs list is: (provider, level, file_type, file_size)
    configs = []
    for prov in PROVIDERS:
        for level in ["fast", "default"]:
            for ft in file_types:
                for sz in sizes:
                    configs.append((prov, level, ft, sz))

    total_rounds = NUM_WARMUP + reps
    total_runs = len(configs) * total_rounds

    logger.info(f"\n{'='*50}")
    logger.info(f"--{len(configs)} configs x {total_rounds} rounds = {total_runs} total runs")
    logger.info(f"{'='*50}\n")

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()

        run_id = 0
        overflow_count = 0

        for round_num in range(total_rounds):
            is_warmup = round_num < NUM_WARMUP
            label = (f"warmup-{round_num + 1}" if is_warmup
                     else f"rep-{round_num - NUM_WARMUP + 1}/{reps}")

            logger.info(f"=== Round {round_num + 1}/{total_rounds} ({label}) ===")

            # shuffle order each round (course requirement: reduces time-bias)
            random.seed(round_num)
            shuffled = configs.copy()
            random.shuffle(shuffled)

            for prov, level, ft, sz in shuffled:
                run_id += 1

                if dry_run:
                    logger.info(f"  [DRY] {prov}/{level}/{ft}/{sz}")
                    continue

                logger.info(f"  [{prov:4s}] {level:7s} | {ft:5s} | {sz}")

                # get input file info
                input_path = get_test_file_path(ft, sz)
                input_bytes = os.path.getsize(input_path)
                input_mb = input_bytes / (1024 * 1024)

                # EnergiBridge CSV path for this run
                eb_csv = str(eb_dir / f"eb_{prov}_{level}_{ft}_{sz}_r{round_num}.csv")

                # run compression
                result = run_one(prov, level, input_path, eb_csv, use_eb)

                if result is None:
                    logger.warning("    Skipping failed run.")
                    continue

                # Track overflow count
                if result["rapl_overflow"]:
                    overflow_count += 1

                # Calculate derived metrics
                ratio = round(result["output_bytes"] / input_bytes, 4) if input_bytes > 0 else 0
                energy = result["energy_joules"]
                epm = round(energy / input_mb, 4) if energy and input_mb > 0 else None

                # Write row
                writer.writerow({
                    "run_id": run_id,
                    "provider": prov,
                    "level": level,
                    "file_type": ft,
                    "file_size": sz,
                    "repetition": round_num + 1,
                    "is_warmup": is_warmup,
                    "input_bytes": input_bytes,
                    "output_bytes": result["output_bytes"],
                    "compression_ratio": ratio,
                    "wall_time_s": result["wall_time_s"],
                    "energy_joules": energy,
                    "energy_per_mb": epm,
                    "rapl_overflow": result["rapl_overflow"],
                })
                f.flush()  # write immediately in case of crash

                # Cooldown between runs
                # Dynamic cooldown: longer runs get more settling time
                # Minimum COOLDOWN_SECS, or 50% of the run's wall time, whichever is greater
                cooldown = max(COOLDOWN_SECS, result["wall_time_s"] * 0.5)
                logger.info(f"    Cooldown: {cooldown:.0f}s")
                time.sleep(cooldown)

    logger.info(f"\n{'='*50}")
    logger.info(f"  Done! {run_id} runs completed.")
    logger.info(f"  RAPL overflows corrected: {overflow_count}")
    logger.info(f"  Data saved to: {output_file}")
    logger.info(f"{'='*50}")

def main():
    p = argparse.ArgumentParser(description="Compression energy experiment")
    p.add_argument("-r", "--reps", type=int, default=NUM_REPS,
                   help=f"Number of repetitions (default: {NUM_REPS})")
    p.add_argument("--types", nargs="+", default=FILE_TYPES,
                   choices=FILE_TYPES, help="File types to test")
    p.add_argument("--sizes", nargs="+", default=list(FILE_SIZES.keys()),
                   choices=FILE_SIZES.keys(), help="File sizes to test")
    p.add_argument("--dry-run", action="store_true",
                   help="Just print what would run, don't actually run")
    p.add_argument("--no-eb", action="store_true",
                   help="Skip EnergiBridge (time-only mode)")
    p.add_argument("--skip-warmup", action="store_true",
                   help="Skip CPU warm-up (for quick testing)")
    args = p.parse_args()

    run_experiment(
        reps=args.reps,
        file_types=args.types,
        sizes=args.sizes,
        dry_run=args.dry_run,
        use_eb=not args.no_eb,
        skip_warmup=args.skip_warmup,
    )


if __name__ == "__main__":
    main()
