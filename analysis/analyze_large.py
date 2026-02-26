"""
analyze_large.py - modified analyze.py and focused on `large` files only
"""

import argparse
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

# paths
PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

# plotting / constants
sns.set_theme(style="whitegrid", font_scale=1.1)
COLORS = {"7zip": "#e74c3c", "gzip": "#3498db", "zstd": "#2ecc71"}
LEVEL_ORDER = ["fast", "default"]
SIZE_ORDER = ["large"]
PROVIDER_ORDER = ["7zip", "gzip", "zstd"]
RAPL_OVERFLOW = 65536


def save_plot(fig, name, dpi=200):
    path = RESULTS / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {path}")


def _prov_order(df):
    return [p for p in PROVIDER_ORDER if p in df["provider"].unique()]

def _size_order(df):
    return [s for s in SIZE_ORDER if s in df["file_size"].unique()]


# --- effect size functions ---------------------------------------------------

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    nx, ny = len(a), len(b)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = np.var(a, ddof=1), np.var(b, ddof=1)
    denom = (nx - 1) * vx + (ny - 1) * vy
    div = nx + ny - 2
    if div <= 0 or denom <= 0:
        return 0.0
    return round(float((a.mean() - b.mean()) / np.sqrt(denom / div)), 4)


def cles(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.5
    return round(float((a[:, None] > b).mean()), 4)


# =============================================================================
#  STEP 1: CLEANING
# =============================================================================

def z_filter(series):
    if len(series) < 3:
        return pd.Series(True, index=series.index)
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(True, index=series.index)
    return np.abs((series - series.mean()) / std) <= 3


def clean_data(df):
    print("\n── step 1: cleaning data ──")
    before = len(df)

    if "is_warmup" in df.columns:
        df = df[~df["is_warmup"]].copy()
    print(f"  dropped warmups: {before - len(df)} rows")

    if "energy_joules" in df.columns:
        mask = df["energy_joules"] < 0
        if mask.any():
            df.loc[mask, "energy_joules"] += RAPL_OVERFLOW
            print(f"  fixed rapl wraparound: {mask.sum()} rows")

    metric = "energy_joules" if ("energy_joules" in df.columns and df["energy_joules"].notna().sum() > 10) else "wall_time_s"
    if metric == "wall_time_s":
        print("  ⚠ not enough energy samples, falling back to wall_time_s")

    # compute derived metrics
    if metric == "energy_joules" and "input_bytes" in df.columns:
        mb = df["input_bytes"] / (1024 * 1024)
        df["energy_per_mb"] = df["energy_joules"] / mb.replace(0, np.nan)

    # energy per percent compressed: how many joules per percentage point of size reduction
    # compression_ratio = output/input, so percent_saved = (1 - ratio) * 100
    if metric == "energy_joules" and "compression_ratio" in df.columns:
        pct_saved = (1 - df["compression_ratio"]) * 100
        df["energy_per_pct"] = df["energy_joules"] / pct_saved.replace(0, np.nan)

    # outlier removal per config group
    CONFIG = ["provider", "level", "file_type", "file_size"]
    n_before = len(df)
    keep = df.groupby(CONFIG)[metric].transform(z_filter)
    df = df[keep].copy()
    print(f"  removed outliers: {n_before - len(df)} rows")
    print(f"  final rows: {len(df)}")

    return df, metric


# =============================================================================
#  STEP 2: NORMALITY
# =============================================================================

def test_normality(df, metric):
    print("\n── step 2: normality tests (shapiro-wilk) ──")
    CONFIG = ["provider", "level", "file_type", "file_size"]
    rows = []

    for name, group in df.groupby(CONFIG):
        vals = group[metric].dropna()
        if len(vals) < 3:
            continue
        W, p = stats.shapiro(vals)
        sk = float(stats.skew(vals))
        rows.append({
            "provider": name[0], "level": name[1],
            "file_type": name[2], "file_size": name[3],
            "n": len(vals), "shapiro_W": round(float(W), 4),
            "p_value": float(p), "skewness": round(sk, 4),
            "is_normal": bool(p >= 0.05),
        })

    norm_df = pd.DataFrame(rows)
    total = len(norm_df)
    normal = int(norm_df["is_normal"].sum())
    print(f"  {normal}/{total} groups normal ({normal/total*100:.1f}%)")

    worst = norm_df.sort_values("p_value").head(6)
    if not worst.empty:
        print("\n  lowest p-values:")
        print(worst[["provider","level","file_type","file_size","n","p_value","skewness"]].to_string(index=False))

    norm_df.to_csv(RESULTS / "normality_tests.csv", index=False)
    return norm_df


def is_group_normal(norm_df, **kwargs):
    if norm_df is None or norm_df.empty:
        return True
    mask = np.ones(len(norm_df), dtype=bool)
    for k, v in kwargs.items():
        if k in norm_df.columns:
            mask &= (norm_df[k] == v)
    sub = norm_df[mask]
    return bool(sub["is_normal"].all()) if not sub.empty else True


# =============================================================================
#  STEP 3: STATISTICAL TESTS
# =============================================================================

def compare_two(x, y, x_norm, y_norm):
    x, y = pd.Series(x).dropna(), pd.Series(y).dropna()
    if len(x) < 3 or len(y) < 3:
        return None

    if x_norm and y_norm:
        stat, p = stats.ttest_ind(x, y, equal_var=False)
        effect = cohens_d(x, y)
        test_name, effect_name = "welch_t", "cohen_d"
    else:
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        effect = cles(x, y)
        test_name, effect_name = "mann_whitney", "cles"

    pct = ((y.mean() - x.mean()) / x.mean() * 100) if x.mean() != 0 else 0.0

    return {
        "test": test_name, "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6), "significant": bool(p < 0.05),
        "mean_a": round(float(x.mean()), 4), "mean_b": round(float(y.mean()), 4),
        "median_a": round(float(x.median()), 4), "median_b": round(float(y.median()), 4),
        "effect_size": effect, "effect_type": effect_name,
        "pct_change": round(float(pct), 2),
    }


def provider_tests(df, metric, norm_df):
    print("\n── step 3a: provider comparisons ──")
    CONFIG = ["level", "file_type", "file_size"]
    providers = sorted(df["provider"].unique())
    rows = []

    for name, group in df.groupby(CONFIG):
        level, ft, sz = name
        for i, a in enumerate(providers):
            for b in providers[i+1:]:
                x = group.loc[group["provider"] == a, metric]
                y = group.loc[group["provider"] == b, metric]
                res = compare_two(x, y,
                    is_group_normal(norm_df, provider=a, level=level, file_type=ft, file_size=sz),
                    is_group_normal(norm_df, provider=b, level=level, file_type=ft, file_size=sz))
                if res is None:
                    continue
                res.update({"comparison": f"{a} vs {b}", "level": level,
                            "file_type": ft, "file_size": sz,
                            "config": f"{level}/{ft}/{sz}"})
                rows.append(res)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        sig = int(result_df["significant"].sum())
        print(f"  {sig}/{len(result_df)} comparisons significant (p < 0.05)")
        result_df.to_csv(RESULTS / "provider_tests.csv", index=False)
        _print_summary(result_df)
    return result_df


def level_tests(df, metric, norm_df):
    print("\n── step 3b: level comparisons (fast vs default) ──")
    CONFIG = ["provider", "file_type", "file_size"]
    rows = []

    for name, group in df.groupby(CONFIG):
        prov, ft, sz = name
        x = group.loc[group["level"] == "fast", metric]
        y = group.loc[group["level"] == "default", metric]
        res = compare_two(x, y,
            is_group_normal(norm_df, provider=prov, level="fast", file_type=ft, file_size=sz),
            is_group_normal(norm_df, provider=prov, level="default", file_type=ft, file_size=sz))
        if res is None:
            continue
        res.update({"comparison": "fast vs default", "provider": prov,
                    "file_type": ft, "file_size": sz,
                    "config": f"{prov}/{ft}/{sz}"})
        rows.append(res)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        sig = int(result_df["significant"].sum())
        print(f"  {sig}/{len(result_df)} comparisons significant (p < 0.05)")
        result_df.to_csv(RESULTS / "level_tests.csv", index=False)
        _print_summary(result_df)
    return result_df


def _print_summary(df):
    cols = ["config", "comparison", "test", "p_value", "significant", "effect_size", "effect_type", "pct_change"]
    cols = [c for c in cols if c in df.columns]
    display = df[cols]
    if len(display) > 12:
        print(display.head(8).to_string(index=False))
        print(f"  ... ({len(display) - 8} more rows in CSV)")
    else:
        print(display.to_string(index=False))


# =============================================================================
#  STEP 4: PLOTS
# =============================================================================

# --- 4a: violin plots — one figure per (level, file_type) -------------------

def plot_violins(df, metric, ylabel):
    print("  violin plots...")
    sizes = _size_order(df)
    order = _prov_order(df)

    for level in LEVEL_ORDER:
        for ft in sorted(df["file_type"].unique()):
            sub = df[(df["level"] == level) & (df["file_type"] == ft)]
            if sub.empty:
                continue

            ncols = len(sizes)
            fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False, sharey=False)

            for j, sz in enumerate(sizes):
                ax = axes[0][j]
                ssub = sub[sub["file_size"] == sz]

                if len(ssub) < 3:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, color="gray")
                    continue

                sns.violinplot(data=ssub, x="provider", y=metric, hue="provider",
                    order=order, hue_order=order, palette=COLORS,
                    inner="box", cut=0, linewidth=1, legend=False, ax=ax)

                ax.set_title(sz, fontsize=12, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel(ylabel if j == 0 else "")
                sns.despine(ax=ax)

            fig.suptitle(f"{ft} — level={level}", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            save_plot(fig, f"violin_{level}_{ft}_large.png")


# --- 4b: energy per MB — bar chart, one subplot per (file_type, size) --------

def plot_energy_per_mb(df):
    if "energy_per_mb" not in df.columns or df["energy_per_mb"].notna().sum() < 10:
        print("  skipping energy_per_mb")
        return

    print("  energy per MB...")
    order = _prov_order(df)
    sizes = _size_order(df)

    for level in LEVEL_ORDER:
        sub = df[df["level"] == level]
        if sub.empty:
            continue

        file_types = sorted(sub["file_type"].unique())
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols,
            figsize=(10, 8), squeeze=False, sharey=False)

        for idx, ft in enumerate(file_types):
            i, j = divmod(idx, 2)
            ax = axes[i][j]
            ssub = sub[(sub["file_type"] == ft) & (sub["file_size"] == "large")]

            agg = ssub.groupby("provider")["energy_per_mb"].median().reindex(order)
            bars = ax.bar(agg.index, agg.values,
                color=[COLORS.get(p, "gray") for p in agg.index], edgecolor="white", width=0.6)

            for bar, val in zip(bars, agg.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

            ax.set_title(f"{ft} / large", fontsize=10, fontweight="bold")
            ax.set_ylabel("J/MB" if j == 0 else "")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=9)
            sns.despine(ax=ax)

        fig.suptitle(f"energy per MB — level={level}", fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_plot(fig, f"energy_per_mb_{level}_large.png")


# --- 4c: ratio vs energy scatter ---------------------------------------------

def plot_ratio_vs_energy(df, metric):
    print("  ratio vs energy scatter...")
    agg = df.groupby(["provider", "level", "file_type", "file_size"]).agg(
        energy=(metric, "median"), ratio=("compression_ratio", "median"),
    ).reset_index()
    if agg.empty:
        return

    markers = {"text": "o", "csv": "s", "pdf": "D", "image": "^"}
    size_map = {"small": 60, "medium": 120, "large": 200}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, level in zip(axes, LEVEL_ORDER):
        sub = agg[agg["level"] == level]
        for _, row in sub.iterrows():
            ax.scatter(row["ratio"], row["energy"],
                color=COLORS.get(row["provider"], "gray"),
                marker=markers.get(row["file_type"], "o"),
                s=size_map.get(row["file_size"], 100),
                edgecolors="black", linewidth=0.5, zorder=5)
        ax.set_xlabel("compression ratio (lower = better compression)")
        ax.set_ylabel("median energy (J)")
        ax.set_title(f"ratio vs energy — {level}")
        sns.despine(ax=ax)

    ax = axes[-1]
    for prov in _prov_order(agg):
        ax.scatter([], [], color=COLORS.get(prov), label=prov, s=80)
    for ft, m in markers.items():
        if ft in agg["file_type"].unique():
            ax.scatter([], [], color="gray", marker=m, label=f"type: {ft}", s=80)
    for sz, s in size_map.items():
        if sz in agg["file_size"].unique():
            ax.scatter([], [], color="gray", label=f"size: {sz}", s=s)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    save_plot(fig, "ratio_vs_energy_large.png")


# --- 4d: scaling -------------------------------------------------------------

def plot_scaling(df):
    if "energy_per_mb" not in df.columns or df["energy_per_mb"].notna().sum() < 10:
        return
    print("  scaling...")
    mb_map = {"large": 500}
    sizes = _size_order(df)
    if len(sizes) < 1:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, level in zip(axes, LEVEL_ORDER):
        sub = df[df["level"] == level]
        for prov in _prov_order(df):
            xs, ys = [], []
            for sz in sizes:
                vals = sub.loc[(sub["provider"] == prov) & (sub["file_size"] == sz), "energy_per_mb"]
                if len(vals) > 0:
                    xs.append(mb_map[sz])
                    ys.append(vals.median())
            if len(xs) >= 2:
                ax.plot(xs, ys, "o-", color=COLORS.get(prov), label=prov, linewidth=2, markersize=8)
        ax.set_xlabel("file size (MB)")
        ax.set_ylabel("median J/MB")
        ax.set_title(f"scaling — {level}")
        ax.legend(fontsize=9)
        sns.despine(ax=ax)
    plt.tight_layout()
    save_plot(fig, "scaling_large.png")


# --- 4e: EDP — provider on x-axis, one subplot per (file_type, size) --------

def plot_edp(df):
    if "energy_joules" not in df.columns or df["energy_joules"].notna().sum() < 10:
        return
    print("  edp...")
    tmp = df.copy()
    tmp["edp"] = tmp["energy_joules"] * tmp["wall_time_s"]
    order = _prov_order(tmp)

    for level in LEVEL_ORDER:
        sub = tmp[tmp["level"] == level]
        if sub.empty:
            continue

        file_types = sorted(sub["file_type"].unique())
        sizes = _size_order(sub)
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols,
            figsize=(10, 8), squeeze=False, sharey=False)

        for idx, ft in enumerate(file_types):
            i, j = divmod(idx, 2)
            ax = axes[i][j]
            ssub = sub[(sub["file_type"] == ft) & (sub["file_size"] == "large")]
            sz = "large"

            agg = ssub.groupby("provider")["edp"].median().reindex(order)
            bars = ax.bar(agg.index, agg.values,
                color=[COLORS.get(p, "gray") for p in agg.index], edgecolor="white", width=0.6)

            for bar, val in zip(bars, agg.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)

            ax.set_title(f"{ft} / {sz}", fontsize=10, fontweight="bold")
            ax.set_ylabel("EDP (J·s)" if j == 0 else "")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=9)
            sns.despine(ax=ax)

        fig.suptitle(f"energy delay product — level={level} (lower = better)",
            fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_plot(fig, f"edp_{level}_large.png")


# --- 4f: heatmaps -----------------------------------------------------------

def plot_heatmap(df, metric):
    print("  heatmaps...")

    for level in LEVEL_ORDER:
        sub = df[df["level"] == level]
        if sub.empty:
            continue

        sub = sub.copy()
        sub["config"] = sub["file_type"] + " / " + sub["file_size"]
        ft_order = sorted(sub["file_type"].unique())
        sz_order = _size_order(sub)
        config_order = [f"{ft} / {sz}" for ft in ft_order for sz in sz_order]
        config_order = [c for c in config_order if c in sub["config"].unique()]

        pivot = sub.groupby(["config", "provider"])[metric].median().unstack()
        pivot = pivot.reindex(index=config_order, columns=_prov_order(sub))

        fig, ax = plt.subplots(figsize=(7, max(5, len(config_order) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "median energy (J)"})
        ax.set_xlabel("provider")
        ax.set_ylabel("")
        ax.set_title(f"median energy (J) — level={level}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        save_plot(fig, f"heatmap_{level}_large.png")


def plot_level_heatmap(df, metric):
    print("  level comparison heatmap...")
    providers = _prov_order(df)
    ft_order = sorted(df["file_type"].unique())
    sz_order = _size_order(df)

    agg = df.groupby(["provider", "level", "file_type", "file_size"])[metric].median().reset_index()
    rows = []
    for prov in providers:
        for ft in ft_order:
            for sz in sz_order:
                fast_val = agg.loc[(agg["provider"]==prov) & (agg["level"]=="fast") &
                                   (agg["file_type"]==ft) & (agg["file_size"]==sz), metric]
                default_val = agg.loc[(agg["provider"]==prov) & (agg["level"]=="default") &
                                      (agg["file_type"]==ft) & (agg["file_size"]==sz), metric]
                if len(fast_val) > 0 and len(default_val) > 0:
                    f, d = float(fast_val.iloc[0]), float(default_val.iloc[0])
                    pct = round((d - f) / f * 100, 1) if f > 0 else 0
                    rows.append({"provider": prov, "config": f"{ft} / {sz}", "pct_increase": pct})

    if not rows:
        return
    rdf = pd.DataFrame(rows)
    config_order = [f"{ft} / {sz}" for ft in ft_order for sz in sz_order]
    config_order = [c for c in config_order if c in rdf["config"].unique()]
    pivot = rdf.pivot(index="config", columns="provider", values="pct_increase")
    pivot = pivot.reindex(index=config_order, columns=providers)

    fig, ax = plt.subplots(figsize=(7, max(5, len(config_order) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Oranges",
        linewidths=0.5, ax=ax, cbar_kws={"label": "% energy increase (fast → default)"})
    ax.set_xlabel("provider")
    ax.set_ylabel("")
    ax.set_title("energy cost of higher compression (%)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, "level_comparison_heatmap_large.png")


# --- 4g: efficiency — energy per percent compressed, bar chart ---------------

def plot_efficiency(df):
    if "energy_per_pct" not in df.columns or df["energy_per_pct"].notna().sum() < 10:
        print("  skipping efficiency (not enough data)")
        return

    print("  efficiency (J per % compressed)...")
    order = _prov_order(df)
    sizes = _size_order(df)

    for level in LEVEL_ORDER:
        sub = df[df["level"] == level]
        if sub.empty:
            continue

        file_types = sorted(sub["file_type"].unique())
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols,
            figsize=(10, 8), squeeze=False, sharey=False)

        for idx, ft in enumerate(file_types):
            i, j = divmod(idx, 2)
            ax = axes[i][j]
            ssub = sub[(sub["file_type"] == ft) & (sub["file_size"] == "large")]
            sz = "large"

            agg = ssub.groupby("provider")["energy_per_pct"].median().reindex(order)
            bars = ax.bar(agg.index, agg.values,
                color=[COLORS.get(p, "gray") for p in agg.index], edgecolor="white", width=0.6)

            for bar, val in zip(bars, agg.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

            ax.set_title(f"{ft} / {sz}", fontsize=10, fontweight="bold")
            ax.set_ylabel("J per % saved" if j == 0 else "")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=9)
            sns.despine(ax=ax)

        fig.suptitle(f"energy per % size reduction — level={level} (lower = more efficient)",
            fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_plot(fig, f"efficiency_{level}_large.png")


# --- 4h: compression ratio — 4x3 grid per level -----------------------------

def plot_compression_ratios(df):
    print("  compression ratios...")
    order = _prov_order(df)
    sizes = _size_order(df)

    for level in LEVEL_ORDER:
        sub = df[df["level"] == level]
        if sub.empty:
            continue

        file_types = sorted(sub["file_type"].unique())
        nrows, ncols = 2, 2
        fig, axes = plt.subplots(nrows, ncols,
            figsize=(10, 8), squeeze=False, sharey=False)

        for idx, ft in enumerate(file_types):
            i, j = divmod(idx, 2)
            ax = axes[i][j]
            ssub = sub[(sub["file_type"] == ft) & (sub["file_size"] == "large")]
            sz = "large"
            agg = ssub.groupby("provider")["compression_ratio"].median().reindex(order)
            bars = ax.bar(agg.index, agg.values,
                color=[COLORS.get(p, "gray") for p in agg.index], edgecolor="white", width=0.6)

            for bar, val in zip(bars, agg.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)

            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
            ax.set_ylim(0, 1.1)
            ax.set_title(f"{ft} / {sz}", fontsize=10, fontweight="bold")
            ax.set_ylabel("ratio (lower = better)" if j == 0 else "")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=9)
            sns.despine(ax=ax)

        fig.suptitle(f"compression ratio — level={level} (lower = better)",
            fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_plot(fig, f"compression_ratio_{level}_large.png")


# =============================================================================
#  STEP 5: DESCRIPTIVE STATS
# =============================================================================

def descriptive_stats(df, metric):
    print("\n── step 5: descriptive stats ──")
    groups = ["provider", "level", "file_type", "file_size"]
    desc = df.groupby(groups).agg(
        n=(metric, "count"), mean=(metric, "mean"), median=(metric, "median"),
        std=(metric, "std"), min=(metric, "min"), max=(metric, "max"),
        compression_ratio=("compression_ratio", "mean"),
    ).round(4).reset_index()

    if "energy_per_mb" in df.columns and df["energy_per_mb"].notna().sum() > 0:
        epm = df.groupby(groups)["energy_per_mb"].agg(
            epm_mean="mean", epm_median="median").round(4).reset_index()
        desc = desc.merge(epm, on=groups, how="left")

    if "energy_per_pct" in df.columns and df["energy_per_pct"].notna().sum() > 0:
        epp = df.groupby(groups)["energy_per_pct"].agg(
            epp_mean="mean", epp_median="median").round(4).reset_index()
        desc = desc.merge(epp, on=groups, how="left")

    desc.to_csv(RESULTS / "descriptive_stats.csv", index=False)
    compact = desc[["provider", "level", "file_type", "file_size", "n", "median", "std", "compression_ratio"]]
    print(compact.to_string(index=False))

    # also print the efficiency summary for the blog
    if "epp_median" in desc.columns:
        print("\n  efficiency summary (median J per % compressed, large files):")
        eff = desc[desc["file_size"] == "large"][["provider", "level", "file_type", "epp_median", "compression_ratio"]]
        if not eff.empty:
            print(eff.sort_values(["file_type", "level", "provider"]).to_string(index=False))

    return desc


# =============================================================================
#  MAIN
# =============================================================================

def find_latest(pattern):
    files = sorted(glob.glob(str(DATA_DIR / pattern)))
    return Path(files[-1]) if files else None


def main():
    p = argparse.ArgumentParser(description="analyze compression energy experiment")
    p.add_argument("--data-file", default=None)
    args = p.parse_args()

    path = Path(args.data_file) if args.data_file else find_latest("experiment_*.csv")
    if not path or not path.exists():
        print("no data found. run: python3 scripts/run_experiment.py")
        sys.exit(1)

    print(f"loading: {path}")
    df = pd.read_csv(path)
    print(f"  rows: {len(df)}")
    for col in ["provider", "level", "file_type", "file_size"]:
        if col in df.columns:
            print(f"  {col}: {list(df[col].unique())}")

    df, metric = clean_data(df)
    ylabel = "energy (J)" if "energy" in metric else "wall time (s)"

    norm_df = test_normality(df, metric)
    provider_tests(df, metric, norm_df)
    level_tests(df, metric, norm_df)

    print("\n── step 4: plots ──")
    plot_violins(df, metric, ylabel)
    plot_energy_per_mb(df)
    plot_ratio_vs_energy(df, metric)
    plot_scaling(df)
    plot_edp(df)
    plot_heatmap(df, metric)
    plot_level_heatmap(df, metric)
    plot_efficiency(df)
    plot_compression_ratios(df)

    descriptive_stats(df, metric)

    png_count = len(list(RESULTS.glob("*.png")))
    print(f"\n{'='*60}")
    print(f"  results: {RESULTS}/")
    print(f"  plots: {png_count} png files")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()