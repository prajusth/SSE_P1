
"""
analyze_non_normals.py — Visual inspection of the 17 groups that failed Shapiro-Wilk.

For each non-normal group, produces a panel with:
  - Histogram with KDE overlay (to see the shape)
  - Strip plot of individual data points (to spot clusters/outliers)
  - Annotated with W statistic, p-value, skewness, n

Output: results/non_normals.png (one big figure with all failing groups)
"""

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
RESULTS = PROJECT / "results"

RAPL_OVERFLOW = 65536


def find_latest(pattern):
    files = sorted(glob.glob(str(DATA_DIR / pattern)))
    return Path(files[-1]) if files else None


def z_filter(series):
    if len(series) < 3:
        return pd.Series(True, index=series.index)
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(True, index=series.index)
    return np.abs((series - series.mean()) / std) <= 3


def main():
    path = find_latest("experiment_*.csv")
    if not path or not path.exists():
        print("no data found")
        sys.exit(1)

    print(f"loading: {path}")
    df = pd.read_csv(path)

    # clean: same as analyze.py
    if "is_warmup" in df.columns:
        df = df[~df["is_warmup"]].copy()

    if "energy_joules" in df.columns:
        mask = df["energy_joules"] < 0
        if mask.any():
            df.loc[mask, "energy_joules"] += RAPL_OVERFLOW

    metric = "energy_joules"
    CONFIG = ["provider", "level", "file_type", "file_size"]

    # outlier removal
    keep = df.groupby(CONFIG)[metric].transform(z_filter)
    df = df[keep].copy()

    # find non-normal groups
    non_normals = []
    for name, group in df.groupby(CONFIG):
        vals = group[metric].dropna()
        if len(vals) < 3:
            continue
        W, p = stats.shapiro(vals)
        if p < 0.05:
            sk = float(stats.skew(vals))
            non_normals.append({
                "provider": name[0], "level": name[1],
                "file_type": name[2], "file_size": name[3],
                "W": round(W, 4), "p": p, "skew": round(sk, 2),
                "n": len(vals), "values": vals.values,
            })

    if not non_normals:
        print("all groups are normal! nothing to plot.")
        return

    print(f"\n{len(non_normals)} non-normal groups found:\n")
    for g in non_normals:
        print(f"  {g['provider']}/{g['level']}/{g['file_type']}/{g['file_size']}  "
              f"W={g['W']}  p={g['p']:.6f}  skew={g['skew']}  n={g['n']}")

    # plot: 2 columns per group (histogram + strip), wrap into rows
    n = len(non_normals)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)

    for idx, g in enumerate(non_normals):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        vals = g["values"]

        # histogram + kde
        ax.hist(vals, bins=12, density=True, alpha=0.5, color="#3498db", edgecolor="white")
        try:
            kde_x = np.linspace(vals.min() - vals.std(), vals.max() + vals.std(), 200)
            kde = stats.gaussian_kde(vals)
            ax.plot(kde_x, kde(kde_x), color="#e74c3c", linewidth=2)
        except Exception:
            pass

        # individual points as rug
        ax.plot(vals, np.zeros_like(vals) - 0.02 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else np.zeros_like(vals),
                "|", color="black", markersize=10, alpha=0.6)

        label = f"{g['provider']}/{g['level']}/{g['file_type']}/{g['file_size']}"
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.text(0.97, 0.95,
                f"W={g['W']}\np={g['p']:.4f}\nskew={g['skew']}\nn={g['n']}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        ax.set_xlabel("energy (J)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        sns.despine(ax=ax)

    # hide unused subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Non-normal groups ({n} of 72 failed Shapiro-Wilk at p<0.05)\n"
                 f"Blue = histogram, Red = KDE, Ticks = individual measurements",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = RESULTS / "non_normals.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()