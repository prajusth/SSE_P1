"""
utils.py — Configuration and shared helpers.

This is the "settings file" for the whole experiment.
"""

import subprocess, os, logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"           # where experiment CSVs go
RESULTS_DIR = PROJECT_ROOT / "results"     # where plots + stats go
TEST_DATA_DIR = PROJECT_ROOT / "test_data" # where generated files go


for d in [DATA_DIR, RESULTS_DIR, TEST_DATA_DIR]:
    d.mkdir(exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("compression")



# --- Compression tools ---
# Each tool has a "fast" and "default" compression command.
# {input} and {output} get replaced with actual file paths at runtime.
#
# Note: dropped "max" level because it takes a very long time and is not commonly used in practice.

PROVIDERS = {
    "7zip": {
        "label": "7-Zip",
        "ext": ".7z",
        "fast":    "7z a -t7z -mx=1 {output} {input}",
        "default": "7z a -t7z -mx=5 {output} {input}",
    },
    "gzip": {
        "label": "gzip",
        "ext": ".gz",
        # gzip works in-place (compresses the file itself), so we handle
        # this specially in run_experiment.py by copying the file first.
        # -k = keep original, -f = force overwrite
        "fast":    "gzip -1 -k -f {input}",
        "default": "gzip -6 -k -f {input}",
    },
    "zstd": {
        "label": "zstd",
        "ext": ".zst",
        # -f = force overwrite, -o = output path
        "fast":    "zstd -1 -f -o {output} {input}",
        "default": "zstd -3 -f -o {output} {input}",
    },
}

# --- File types we're compressing ---
# Each represents a real-world use case with different compressibility:
#   text  = highly compressible (server logs)
#   csv   = highly compressible (data science exports)
#   pdf   = barely compressible (already internally compressed)
#   image = highly compressible when raw (BMP format, like medical scans)

FILE_TYPES = ["text", "csv", "pdf", "image"]

# --- File sizes ---
# Three sizes to see how energy scales.
#   small  = email attachment
#   medium = software package / dataset
#   large  = video / database dump

FILE_SIZES = {
    "small":  10,    # ~10 MB
    "medium": 100,   # ~100 MB
    "large":  500,   # ~500 MB 
}


NUM_REPS = 30          # Number of measured repetitions (course says 30)
NUM_WARMUP = 5         # Extra runs at the start, discarded (course says do warmup)
COOLDOWN_SECS = 10     # Seconds to wait between runs (prevents tail energy bleed) NOTE: this is the minimum cooldown value
CPU_WARMUP_SECS = 300  # Seconds of CPU warmup before experiment starts
SEED = 42              # For reproducibility of generated data and shuffle order

# EnergiBridge binary name (must be on PATH)
ENERGIBRIDGE = "energibridge"

# RAPL overflow threshold for AMD CPUs.
# The CPU_ENERGY counter wraps around at 2^16 = 65536 Joules.
# We verified this by inspecting raw EnergiBridge CSVs.
RAPL_OVERFLOW = 65536

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def zen_mode():
    """
    Kill noisy background processes to reduce measurement noise.
    Course calls this 'Zen mode', to close everything except what you need.
    """
    noisy = [
        "firefox", "chromium", "chrome", "spotify", "slack",
        "discord", "teams", "zoom", "telegram", "update-manager", "brave"
    ]
    for proc in noisy:
        subprocess.run(["pkill", "-f", proc],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    logger.info("Zen mode: killed background processes.")


def check_energibridge():
    try:
        subprocess.run([ENERGIBRIDGE, "--help"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_test_file_path(file_type, size_label):

    extensions = {
        "text":  ".log",
        "csv":   ".csv",
        "pdf":   ".pdf",
        "image": ".bmp",
    }
    ext = extensions[file_type]
    return str(TEST_DATA_DIR / f"{file_type}_{size_label}{ext}")


def get_system_info():
    """Collect system info for the replication package."""
    info = {"timestamp": datetime.now().isoformat()}
    try:
        info["cpu"] = subprocess.check_output(
            ["lscpu"], text=True
        ).strip()
    except Exception:
        info["cpu"] = "unknown"
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        info["mem_gb"] = round(mem_bytes / (1024**3), 1)
    except Exception:
        info["mem_gb"] = "unknown"
    info["os"] = subprocess.check_output(["uname", "-a"], text=True).strip()

    # Record tool versions
    for name, cmd, flag in [("7z", "7z", "i"), ("gzip", "gzip", "--version"), ("zstd", "zstd", "--version")]:
        try:
            out = subprocess.check_output([cmd, flag], text=True, stderr=subprocess.STDOUT)
            info[f"{name}_version"] = out.split("\n")[1] if name == "7z" else out.split("\n")[0]
        except Exception:
            pass

    return info
