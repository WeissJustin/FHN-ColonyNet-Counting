#!/usr/bin/env python3
"""
CFU Count & Size Comparison Dashboard

Inputs:
  --CPSAM_csv     : CPSAM predicted counts CSV (petri_dish, count, cfu_area_px, ...)
  --Algo1_csv     : Optional KC Algorithm counts CSV (petri_dish, count, cfu_area_px, ...)
  --Algo2_csv     : Optional FHN ColonyNet counts CSV (petri_dish, count, cfu_area_px, ...)
  --GT_csv        : Ground-truth CSV. Test mode: (petri_dish, count).
                    Train mode: (petri_dish, count, cfu_area_px).
  --mode          : "test" or "train" (train enables size distribution plots)

Per-CFU CSVs have one row per detected CFU with columns:
  petri_dish / image_name — dish identifier
  count        — total CFU count for that dish (repeated on every row)
  cfu_area_px  — pixel area of this individual CFU

App-exported CSVs with extra columns such as image_name, dish_index, cfu_index,
mean_area_px, methods, classes, and total_runtime_sec are supported.

GT CSV (test):  petri_dish, count
GT CSV (train): petri_dish, count, cfu_area_px

Outputs (in out_dir):
  matched_rows.csv          — dish-level matched counts
  dashboard_counts.png      — count comparison plots
  dashboard_regression.png  — regression + overlay
  dashboard_regression_trimmed.png — regression + overlay, without 20 biggest errors
  dashboard_bland_altman.png— Bland–Altman plots
  dashboard_bland_altman_trimmed.png — Bland–Altman plots, without 20 biggest errors
  dashboard_sizes.png       — size distribution plots (train mode only)
  dashboard_size_analysis.png — advanced size analysis (train mode only)
  summary.txt               — console summary saved to file
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 16,
})

# =============================================================================
# Utilities
# =============================================================================

def safe_read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df


def parse_count_series(s: pd.Series, tm_value: float | None = None) -> pd.Series:
    s = s.astype(str).str.strip()
    is_tm = s.str.upper().eq("TM")
    out = pd.to_numeric(s.where(~is_tm, np.nan), errors="coerce")
    if tm_value is not None:
        out = out.where(~is_tm, float(tm_value))
    return out


def extract_match_key(file_name: str) -> str:
    """
    Normalise a petri_dish / file_name string to a canonical lowercase key
    for matching across CSVs.  Strips common suffixes like _output, _Top,
    _Bottom, _bak, file extensions, and whitespace.
    """
    s = str(file_name).strip()
    s = Path(s).stem  # drop extension
    # Remove common suffixes that differ between sources
    for suffix in ("_output", "_overlay", "_Top", "_Bottom", "_bak", "_expt", "_post", "__expt", "__post"):
        if s.lower().endswith(suffix.lower()):
            s = s[: -len(suffix)]
    return s.strip().lower()


def linregress_np(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    if x.size < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    return float(slope), float(intercept), float(r2)


def extract_dilu_class(name: str) -> float:
    m = re.search(r"dilu([0-3])(?=[^0-9]|$)", str(name), flags=re.IGNORECASE)
    return float(int(m.group(1))) if m else np.nan


def resolve_dish_names(df: pd.DataFrame, source_name: str) -> pd.Series:
    """
    Build a per-row dish identifier with fallback across commonly used columns.
    This supports app exports that include both image_name and petri_dish, or
    have one of them partially blank. If an x_suffix column is present, append
    it before the file extension when the chosen dish name does not already
    contain that suffix.
    """
    candidates = ("petri_dish", "file_name", "image_name", "dish", "sample")
    available = [c for c in candidates if c in df.columns]
    if not available:
        raise SystemExit(
            f"[{source_name}] Cannot find dish-name column. "
            f"Expected one of: {', '.join(candidates)}. Has: {list(df.columns)}"
        )

    dish_names = pd.Series("", index=df.index, dtype="object")
    for col in available:
        values = df[col].fillna("").astype(str).str.strip()
        fill_mask = dish_names.eq("") & values.ne("")
        dish_names.loc[fill_mask] = values.loc[fill_mask]

    if "x_suffix" in df.columns:
        suffixes = df["x_suffix"].fillna("").astype(str).str.strip()
        suffixes = suffixes.mask(suffixes.eq(""), "")

        def _apply_x_suffix(name: str, suffix: str) -> str:
            if not suffix:
                return name
            path = Path(str(name).strip())
            stem = path.stem
            # Avoid duplicating a suffix already present in the file name.
            if stem.lower().endswith(f"_{suffix}".lower()) or stem.lower().endswith(suffix.lower()):
                return str(name).strip()
            new_name = f"{stem}_{suffix}{path.suffix}"
            return new_name

        update_mask = dish_names.ne("") & suffixes.ne("")
        dish_names.loc[update_mask] = [
            _apply_x_suffix(name, suffix)
            for name, suffix in zip(dish_names.loc[update_mask], suffixes.loc[update_mask])
        ]

    if dish_names.eq("").any():
        missing = int(dish_names.eq("").sum())
        raise SystemExit(f"[{source_name}] Found {missing} row(s) without a usable dish identifier.")

    return dish_names


def aggregate_dish_counts(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Collapse repeated per-CFU rows into one row per dish.
    Counts are often repeated on every CFU row in app-style exports, so we pick
    the first non-null parsed value per dish and warn if multiple distinct
    numeric/TM values appear for the same dish.
    """
    parsed_counts = parse_count_series(df["count"])
    count_text = df["count"].fillna("").astype(str).str.strip()

    rows = []
    for match_key, sub in df.groupby("match_key", sort=False):
        numeric_counts = parsed_counts.loc[sub.index].dropna()
        if numeric_counts.empty:
            count_value = np.nan
        else:
            distinct_counts = np.unique(numeric_counts.to_numpy(dtype=float))
            if distinct_counts.size > 1:
                warnings.warn(
                    f"[{source_name}] Inconsistent counts for dish '{sub['petri_dish'].iloc[0]}': "
                    f"{distinct_counts.tolist()}. Using the first parsed value.",
                    stacklevel=2,
                )
            count_value = float(numeric_counts.iloc[0])

        if pd.isna(count_value):
            non_empty_text = count_text.loc[sub.index]
            non_empty_text = non_empty_text[non_empty_text.ne("")]
            if not non_empty_text.empty:
                warnings.warn(
                    f"[{source_name}] Could not parse count for dish '{sub['petri_dish'].iloc[0]}' "
                    f"from values {sorted(non_empty_text.unique().tolist())}.",
                    stacklevel=2,
                )

        rows.append({
            "match_key": match_key,
            "petri_dish": sub["petri_dish"].iloc[0],
            "count": count_value,
        })

    return pd.DataFrame(rows)


# =============================================================================
# CSV Loading & Aggregation
# =============================================================================

def load_prediction_csv(path: str, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a per-CFU prediction CSV.

    Returns
    -------
    dish_df : DataFrame with columns [petri_dish, count, match_key]
              One row per dish.
    cfu_df  : DataFrame with columns [petri_dish, cfu_area_px, match_key]
              One row per CFU (may be empty if cfu_area_px not present).
    """
    df = safe_read_csv(path)

    if "count" not in df.columns:
        raise SystemExit(f"[{source_name}] Missing 'count' column. Has: {list(df.columns)}")

    df["petri_dish"] = resolve_dish_names(df, source_name)
    df["match_key"] = df["petri_dish"].apply(extract_match_key)

    dish_df = aggregate_dish_counts(df, source_name)

    # CFU-level sizes
    cfu_df = pd.DataFrame()
    if "cfu_area_px" in df.columns:
        cfu_sub = df[["petri_dish", "match_key", "cfu_area_px"]].copy()
        cfu_sub["cfu_area_px"] = pd.to_numeric(cfu_sub["cfu_area_px"], errors="coerce")
        cfu_df = cfu_sub.dropna(subset=["cfu_area_px"]).copy()

    return dish_df, cfu_df


def load_gt_csv(path: str, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ground-truth CSV.

    Test mode:  petri_dish, count
    Train mode: petri_dish, count, cfu_area_px (per-CFU rows)
    """
    df = safe_read_csv(path)

    if "count" not in df.columns:
        raise SystemExit(f"[GT] Missing 'count' column. Has: {list(df.columns)}")

    df["petri_dish"] = resolve_dish_names(df, "GT")
    df["match_key"] = df["petri_dish"].apply(extract_match_key)

    dish_df = aggregate_dish_counts(df, "GT")

    cfu_df = pd.DataFrame()
    if mode == "train" and "cfu_area_px" in df.columns:
        cfu_sub = df[["petri_dish", "match_key", "cfu_area_px"]].copy()
        cfu_sub["cfu_area_px"] = pd.to_numeric(cfu_sub["cfu_area_px"], errors="coerce")
        cfu_df = cfu_sub.dropna(subset=["cfu_area_px"]).copy()

    return dish_df, cfu_df


def build_matched_table(
    gt_dish: pd.DataFrame,
    cpsam_dish: pd.DataFrame | None,
    algo1_dish: pd.DataFrame | None,
    algo2_dish: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Merge all sources on match_key.  Returns a DataFrame with columns:
      match_key, petri_dish, gt, cpsam, algo1, algo2, dilu_class
    """
    m = gt_dish[["match_key", "petri_dish", "count"]].rename(columns={"count": "gt"}).copy()

    for label, src_df in [("cpsam", cpsam_dish), ("algo1", algo1_dish), ("algo2", algo2_dish)]:
        if src_df is not None and not src_df.empty:
            sub = src_df[["match_key", "count"]].rename(columns={"count": label})
            # Drop duplicate match_keys in prediction (keep first)
            sub = sub.drop_duplicates(subset="match_key", keep="first")
            m = m.merge(sub, on="match_key", how="left")
        else:
            m[label] = np.nan

    m["dilu_class"] = m["petri_dish"].apply(extract_dilu_class)
    return m


# =============================================================================
# Plot: Count comparison histograms  (GT − Pred)
# =============================================================================

def plot_count_comparison_hist(ax, pred: np.ndarray, gt: np.ndarray, label: str,
                                color: str, bins: int = 40):
    diffs = gt - pred
    valid = np.isfinite(diffs)
    diffs = diffs[valid]
    if diffs.size == 0:
        ax.text(0.5, 0.5, f"{label}\nNo data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.hist(diffs, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(0, linestyle="--", linewidth=1.2, color="black")
    mae = float(np.mean(np.abs(diffs)))
    mean_d = float(np.mean(diffs))
    ax.set_title(f"GT − {label}")
    ax.set_xlabel("GT − Pred")
    ax.set_ylabel("Number of dishes")
    ax.text(0.98, 0.95, f"MAE={mae:.1f}\nBias={mean_d:.1f}\nn={diffs.size}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))


# =============================================================================
# Plot: Scatter + regression
# =============================================================================

def plot_scatter_regression(ax, pred: np.ndarray, gt: np.ndarray, label: str,
                            color: str, show_stats: bool = True):
    valid = np.isfinite(pred) & np.isfinite(gt)
    pred_v, gt_v = pred[valid], gt[valid]
    if pred_v.size < 2:
        ax.text(0.5, 0.5, f"{label}\nInsufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return np.nan, np.nan, np.nan

    ax.scatter(pred_v, gt_v, s=18, alpha=0.7, color=color, zorder=3)

    lo = float(min(pred_v.min(), gt_v.min()))
    hi = float(max(pred_v.max(), gt_v.max()))
    pad = 0.05 * (hi - lo + 1)
    lo -= pad
    hi += pad

    # y = x
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1, color="gray", label="y = x")

    # regression
    slope, intercept, r2 = linregress_np(pred_v, gt_v)
    if np.isfinite(slope):
        xs = np.linspace(lo, hi, 200)
        ax.plot(xs, slope * xs + intercept, linewidth=1.5, color=color,
                label=f"{label}: y={slope:.2f}x+{intercept:.1f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("GT count")
    ax.set_title(f"GT vs {label}" + (f"  (R²={r2:.3f})" if np.isfinite(r2) else ""))

    if show_stats:
        mae = float(np.mean(np.abs(gt_v - pred_v)))
        ax.text(0.02, 0.95, f"n={pred_v.size}\nMAE={mae:.1f}\nR²={r2:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    return slope, intercept, r2


def plot_regression_overlay(ax, matched: pd.DataFrame, sources: dict[str, str]):
    """
    Overlay regression lines from all sources on a single axis.
    sources: {column_name: color}
    """
    gt = matched["gt"].to_numpy(dtype=float)
    lo_global, hi_global = np.inf, -np.inf

    for col, color in sources.items():
        pred = matched[col].to_numpy(dtype=float)
        valid = np.isfinite(pred) & np.isfinite(gt)
        pv, gv = pred[valid], gt[valid]
        if pv.size < 2:
            continue

        lo_global = min(lo_global, pv.min(), gv.min())
        hi_global = max(hi_global, pv.max(), gv.max())

        ax.scatter(pv, gv, s=12, alpha=0.35, color=color, zorder=2)

        slope, intercept, r2 = linregress_np(pv, gv)
        if np.isfinite(slope):
            xs = np.linspace(pv.min(), pv.max(), 200)
            label_name = col.upper() if col != "cpsam" else "CPSAM"
            ax.plot(xs, slope * xs + intercept, linewidth=2, color=color,
                    label=f"{label_name}: R²={r2:.3f}", zorder=4)

    if np.isfinite(lo_global) and np.isfinite(hi_global):
        pad = 0.05 * (hi_global - lo_global + 1)
        lo_global -= pad
        hi_global += pad
        ax.plot([lo_global, hi_global], [lo_global, hi_global], "--",
                linewidth=1, color="gray", label="y = x", zorder=1)
        ax.set_xlim(lo_global, hi_global)
        ax.set_ylim(lo_global, hi_global)

    ax.set_xlabel("Predicted count")
    ax.set_ylabel("GT count")
    ax.set_title("Regression Overlay: All Methods vs GT")
    ax.legend(loc="upper left", fontsize=9, frameon=True)


# =============================================================================
# Plot: Bland–Altman
# =============================================================================

def plot_bland_altman(ax, pred: np.ndarray, gt: np.ndarray, pred_label: str,
                      gt_label: str, color: str, bins_hist: int = 30,
                      dilu_classes: np.ndarray | None = None):
    """
    Bland–Altman: x = mean of two, y = (gt - pred).
    Includes marginal histogram on right side.
    Optionally colours points by dilu_class.
    """
    valid = np.isfinite(pred) & np.isfinite(gt)
    p, g = pred[valid], gt[valid]
    if p.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    mean_val = 0.5 * (p + g)
    diff_val = g - p

    # Scatter (optionally coloured)
    if dilu_classes is not None:
        dc = dilu_classes[valid]
        dilu_colors = {0: "#1f77b4", 1: "#2ca02c", 2: "#ff7f0e", 3: "#d62728"}
        for d in sorted(dilu_colors):
            m = dc == d
            if np.any(m):
                ax.scatter(mean_val[m], diff_val[m], s=18, alpha=0.8,
                           c=dilu_colors[d], label=f"dilu{d}", zorder=3)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    else:
        ax.scatter(mean_val, diff_val, s=18, alpha=0.8, color=color, zorder=3)

    # Stats
    bias = float(np.mean(diff_val))
    sigma = float(np.std(diff_val, ddof=1)) if diff_val.size > 1 else 0.0
    loa = 1.96 * sigma

    ax.axhline(0, linestyle="-", linewidth=0.8, color="black", zorder=1)
    ax.axhline(bias, linestyle="--", linewidth=1.5, color="darkblue", zorder=2)
    ax.axhline(bias + loa, linestyle=":", linewidth=1.2, color="red", zorder=2)
    ax.axhline(bias - loa, linestyle=":", linewidth=1.2, color="red", zorder=2)

    ax.set_title(f"Bland–Altman: {pred_label} vs {gt_label}")
    ax.set_xlabel(f"Mean of {pred_label} & {gt_label}")
    ax.set_ylabel(f"{gt_label} − {pred_label}")

    ymax = max(np.percentile(np.abs(diff_val), 99), abs(bias) + loa, 1.0) * 1.15
    ax.set_ylim(-ymax, ymax)

    ax.text(0.02, 0.02,
            f"Bias={bias:.2f}  σ={sigma:.2f}  LoA=±{loa:.2f}  n={p.size}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    # Marginal histogram
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("right", size="18%", pad=0.08, sharey=ax)
    ax_hist.hist(diff_val, bins=bins_hist, orientation="horizontal",
                 alpha=0.4, color=color, edgecolor="none")
    ax_hist.axhline(bias, linestyle="--", linewidth=1.2, color="darkblue")
    ax_hist.axhline(bias + loa, linestyle=":", linewidth=1, color="red")
    ax_hist.axhline(bias - loa, linestyle=":", linewidth=1, color="red")
    ax_hist.set_xlabel("n")
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.grid(False)


# =============================================================================
# Plot: Size distribution (train mode)
# =============================================================================

def plot_size_histogram_comparison(ax, gt_sizes: np.ndarray, pred_sizes: np.ndarray,
                                    pred_label: str, color_pred: str, bins: int = 60):
    """Overlaid histograms of CFU pixel areas: GT vs one prediction source."""
    if gt_sizes.size == 0 and pred_sizes.size == 0:
        ax.text(0.5, 0.5, "No size data", ha="center", va="center", transform=ax.transAxes)
        return

    all_vals = np.concatenate([gt_sizes, pred_sizes])
    all_vals = all_vals[np.isfinite(all_vals) & (all_vals > 0)]
    if all_vals.size == 0:
        return

    bin_edges = np.linspace(0, np.percentile(all_vals, 99), bins + 1)

    if gt_sizes.size > 0:
        ax.hist(gt_sizes[gt_sizes > 0], bins=bin_edges, alpha=0.45,
                color="black", label=f"GT (n={gt_sizes.size})", edgecolor="none")
    if pred_sizes.size > 0:
        ax.hist(pred_sizes[pred_sizes > 0], bins=bin_edges, alpha=0.45,
                color=color_pred, label=f"{pred_label} (n={pred_sizes.size})", edgecolor="none")

    ax.set_xlabel("CFU area (px)")
    ax.set_ylabel("Count")
    ax.set_title(f"Size Distribution: GT vs {pred_label}")
    ax.legend(loc="upper right", fontsize=9)


def plot_size_cdf_comparison(ax, gt_sizes: np.ndarray, pred_sizes: np.ndarray,
                              pred_label: str, color_pred: str):
    """Empirical CDF of CFU areas: GT vs one prediction source."""
    def _ecdf(data):
        data = np.sort(data[np.isfinite(data) & (data > 0)])
        if data.size == 0:
            return np.array([]), np.array([])
        y = np.arange(1, data.size + 1) / data.size
        return data, y

    x_gt, y_gt = _ecdf(gt_sizes)
    x_pr, y_pr = _ecdf(pred_sizes)

    if x_gt.size > 0:
        ax.step(x_gt, y_gt, where="post", linewidth=1.5, color="black", label="GT")
    if x_pr.size > 0:
        ax.step(x_pr, y_pr, where="post", linewidth=1.5, color=color_pred, label=pred_label)

    ax.set_xlabel("CFU area (px)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"CDF: GT vs {pred_label}")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)


def plot_size_violin(ax, size_dict: dict[str, np.ndarray], colors: dict[str, str]):
    """
    Violin plot comparing CFU size distributions across all sources.
    """
    labels = []
    data = []
    cs = []
    for name, sizes in size_dict.items():
        s = sizes[np.isfinite(sizes) & (sizes > 0)]
        if s.size > 0:
            labels.append(f"{name}\n(n={s.size})")
            data.append(s)
            cs.append(colors.get(name, "gray"))

    if not data:
        ax.text(0.5, 0.5, "No size data", ha="center", va="center", transform=ax.transAxes)
        return

    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(cs[i])
        pc.set_alpha(0.5)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("darkred")

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CFU area (px)")
    ax.set_title("CFU Size Distributions (Violin)")


def plot_size_boxplot(ax, size_dict: dict[str, np.ndarray], colors: dict[str, str]):
    """Box plot comparing CFU size distributions."""
    labels = []
    data = []
    cs = []
    for name, sizes in size_dict.items():
        s = sizes[np.isfinite(sizes) & (sizes > 0)]
        if s.size > 0:
            labels.append(f"{name}\n(n={s.size})")
            data.append(s)
            cs.append(colors.get(name, "gray"))

    if not data:
        ax.text(0.5, 0.5, "No size data", ha="center", va="center", transform=ax.transAxes)
        return

    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, c in zip(bp["boxes"], cs):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)

    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("CFU area (px)")
    ax.set_title("CFU Size Distributions (Box, no outliers)")


def plot_size_qq(ax, gt_sizes: np.ndarray, pred_sizes: np.ndarray,
                 pred_label: str, color: str):
    """Q-Q plot: quantiles of pred sizes vs quantiles of GT sizes."""
    g = np.sort(gt_sizes[np.isfinite(gt_sizes) & (gt_sizes > 0)])
    p = np.sort(pred_sizes[np.isfinite(pred_sizes) & (pred_sizes > 0)])
    if g.size < 2 or p.size < 2:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    n_q = min(200, g.size, p.size)
    quantiles = np.linspace(0, 1, n_q)
    g_q = np.quantile(g, quantiles)
    p_q = np.quantile(p, quantiles)

    ax.scatter(g_q, p_q, s=12, alpha=0.7, color=color, zorder=3)
    lo = min(g_q.min(), p_q.min())
    hi = max(g_q.max(), p_q.max())
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1, color="gray", zorder=1)
    ax.set_xlabel("GT size quantiles (px)")
    ax.set_ylabel(f"{pred_label} size quantiles (px)")
    ax.set_title(f"Q-Q: {pred_label} vs GT")
    ax.set_aspect("equal", adjustable="datalim")


def plot_per_dish_mean_size_scatter(ax, matched: pd.DataFrame,
                                     cfu_dict: dict[str, pd.DataFrame],
                                     colors: dict[str, str]):
    """
    For each dish, compute mean CFU area and scatter pred-mean vs GT-mean.
    """
    gt_cfu = cfu_dict.get("GT")
    if gt_cfu is None or gt_cfu.empty:
        ax.text(0.5, 0.5, "No GT size data", ha="center", va="center",
                transform=ax.transAxes)
        return

    gt_mean = gt_cfu.groupby("match_key")["cfu_area_px"].mean().rename("gt_mean_area")

    for name, color in colors.items():
        if name == "GT":
            continue
        pred_cfu = cfu_dict.get(name)
        if pred_cfu is None or pred_cfu.empty:
            continue
        pred_mean = pred_cfu.groupby("match_key")["cfu_area_px"].mean().rename("pred_mean_area")
        joined = pd.DataFrame(gt_mean).join(pred_mean, how="inner").dropna()
        if joined.empty:
            continue

        ax.scatter(joined["gt_mean_area"], joined["pred_mean_area"],
                   s=20, alpha=0.7, color=color, label=name, zorder=3)

    # y = x
    all_vals = []
    for child in ax.collections:
        offsets = child.get_offsets()
        if offsets.size > 0:
            all_vals.append(offsets)
    if all_vals:
        pts = np.vstack(all_vals)
        lo = pts.min()
        hi = pts.max()
        ax.plot([lo, hi], [lo, hi], "--", linewidth=1, color="gray", zorder=1)

    ax.set_xlabel("GT mean CFU area (px)")
    ax.set_ylabel("Pred mean CFU area (px)")
    ax.set_title("Per-Dish Mean CFU Size: Pred vs GT")
    ax.legend(fontsize=9, loc="upper left")


def plot_size_ratio_hist(ax, gt_cfu: pd.DataFrame, pred_cfu: pd.DataFrame,
                         pred_label: str, color: str, bins: int = 30):
    """
    Histogram of per-dish mean-size ratio: pred_mean_area / gt_mean_area.
    """
    if gt_cfu is None or gt_cfu.empty or pred_cfu is None or pred_cfu.empty:
        ax.text(0.5, 0.5, "No size data", ha="center", va="center", transform=ax.transAxes)
        return

    gt_mean = gt_cfu.groupby("match_key")["cfu_area_px"].mean().rename("gt_mean")
    pred_mean = pred_cfu.groupby("match_key")["cfu_area_px"].mean().rename("pred_mean")
    joined = pd.DataFrame(gt_mean).join(pred_mean, how="inner").dropna()
    joined = joined[(joined["gt_mean"] > 0) & (joined["pred_mean"] > 0)]
    if joined.empty:
        ax.text(0.5, 0.5, "No overlapping dishes", ha="center", va="center", transform=ax.transAxes)
        return

    ratios = (joined["pred_mean"] / joined["gt_mean"]).to_numpy(dtype=float)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        ax.text(0.5, 0.5, "No valid ratios", ha="center", va="center", transform=ax.transAxes)
        return

    hi = max(2.5, float(np.percentile(ratios, 99)))
    bin_edges = np.linspace(0, hi, bins + 1)
    ax.hist(ratios, bins=bin_edges, color=color, alpha=0.7, edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("Pred / GT mean CFU area per dish")
    ax.set_ylabel("Number of dishes")
    ax.set_title(f"Per-Dish Mean-Size Ratio: {pred_label}")
    ax.text(0.98, 0.95,
            f"median={np.median(ratios):.2f}\nmean={np.mean(ratios):.2f}\nn={ratios.size}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))


def _cfu_sizes(cfu_df: pd.DataFrame | None) -> np.ndarray:
    if cfu_df is None or cfu_df.empty or "cfu_area_px" not in cfu_df.columns:
        return np.array([], dtype=float)
    arr = cfu_df["cfu_area_px"].to_numpy(dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def active_count_sources(matched: pd.DataFrame) -> list[str]:
    return [col for col in ("cpsam", "algo1", "algo2") if col in matched.columns and matched[col].notna().any()]


def active_regression_sources(matched: pd.DataFrame) -> list[str]:
    gt = matched["gt"].to_numpy(dtype=float)
    active = []
    for col in active_count_sources(matched):
        pred = matched[col].to_numpy(dtype=float)
        if np.isfinite(pred).sum() == 0:
            continue
        if np.count_nonzero(np.isfinite(pred) & np.isfinite(gt)) >= 2:
            active.append(col)
    return active


def active_size_sources(cfu_dict: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    active = [("CPSAM", "cpsam")]
    for label, key in (("KC Algorithm", "algo1"), ("FHN ColonyNet", "algo2")):
        cfu_df = cfu_dict.get(label)
        if cfu_df is not None and not cfu_df.empty:
            active.append((label, key))
    return active


def trim_biggest_errors(matched: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    trimmed = matched.copy()
    gt = trimmed["gt"].to_numpy(dtype=float)
    for col in ("cpsam", "algo1", "algo2"):
        if col not in trimmed.columns or not trimmed[col].notna().any():
            continue
        pred = trimmed[col].to_numpy(dtype=float)
        valid = np.isfinite(pred) & np.isfinite(gt)
        if valid.sum() <= top_n:
            continue
        valid_idx = np.flatnonzero(valid)
        diffs = gt[valid] - pred[valid]
        worst_local_idx = np.argsort(np.abs(diffs))[-top_n:]
        worst_global_idx = valid_idx[worst_local_idx]
        trimmed.loc[worst_global_idx, col] = np.nan
    return trimmed


def save_counts_dashboard(matched: pd.DataFrame, out_path: Path, labels: dict[str, str],
                          colors: dict[str, str]):
    gt = matched["gt"].to_numpy(dtype=float)
    active = active_count_sources(matched)

    fig, axes = plt.subplots(1, len(active), figsize=(7 * max(len(active), 1), 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, active):
        plot_count_comparison_hist(ax, matched[col].to_numpy(dtype=float), gt, labels[col], colors[col])
    fig.suptitle("Count Comparison Histograms", fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_regression_dashboard(matched: pd.DataFrame, out_path: Path, labels: dict[str, str],
                              colors: dict[str, str], title_suffix: str = ""):
    active = active_regression_sources(matched)
    if not active:
        raise SystemExit("No prediction columns contained enough matched numeric data for regression plotting.")
    if len(active) == 1:
        gt = matched["gt"].to_numpy(dtype=float)
        col = active[0]
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
        plot_scatter_regression(ax, matched[col].to_numpy(dtype=float), gt, labels[col], colors[col])
        fig.suptitle("Regression Analysis" + title_suffix, fontsize=16)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    ncols = max(2, len(active))
    fig = plt.figure(figsize=(7 * ncols, 10), constrained_layout=True)
    gs = GridSpec(2, ncols, figure=fig)
    gt = matched["gt"].to_numpy(dtype=float)

    for i, col in enumerate(active):
        ax = fig.add_subplot(gs[0, i])
        plot_scatter_regression(ax, matched[col].to_numpy(dtype=float), gt, labels[col], colors[col])

    overlay_ax = fig.add_subplot(gs[1, :])
    plot_regression_overlay(overlay_ax, matched, {col: colors[col] for col in active})
    fig.suptitle("Regression Analysis" + title_suffix, fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_bland_altman_dashboard(matched: pd.DataFrame, out_path: Path, labels: dict[str, str],
                                colors: dict[str, str], title_suffix: str = ""):
    active = active_count_sources(matched)
    fig, axes = plt.subplots(len(active), 1, figsize=(10, 5 * max(len(active), 1)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    gt = matched["gt"].to_numpy(dtype=float)
    dilu = matched["dilu_class"].to_numpy(dtype=float)

    for ax, col in zip(axes, active):
        plot_bland_altman(ax, matched[col].to_numpy(dtype=float), gt,
                          labels[col], "GT", colors[col], dilu_classes=dilu)
    fig.suptitle("Bland-Altman Analysis" + title_suffix, fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_size_dashboard(cfu_dict: dict[str, pd.DataFrame], out_path: Path, colors: dict[str, str]):
    gt_sizes = _cfu_sizes(cfu_dict.get("GT"))
    comparisons = active_size_sources(cfu_dict)
    fig, axes = plt.subplots(len(comparisons), 2,
                             figsize=(14, 5 * max(len(comparisons), 1)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row, (label, key) in enumerate(comparisons):
        pred_sizes = _cfu_sizes(cfu_dict.get(label))
        plot_size_histogram_comparison(axes[row, 0], gt_sizes, pred_sizes, label, colors[key])
        plot_size_cdf_comparison(axes[row, 1], gt_sizes, pred_sizes, label, colors[key])

    fig.suptitle("CFU Size Distributions", fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_size_analysis_dashboard(matched: pd.DataFrame, cfu_dict: dict[str, pd.DataFrame],
                                 out_path: Path, colors: dict[str, str]):
    size_dict = {name: _cfu_sizes(df) for name, df in cfu_dict.items()}
    active_predictions = active_size_sources(cfu_dict)
    fig = plt.figure(figsize=(16, 16), constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_size_violin(ax1, size_dict, colors)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_size_boxplot(ax2, size_dict, colors)

    ax3 = fig.add_subplot(gs[1, 0])
    plot_per_dish_mean_size_scatter(ax3, matched, cfu_dict, colors)

    qq_label, qq_key = active_predictions[0]
    ax4 = fig.add_subplot(gs[1, 1])
    plot_size_qq(ax4, size_dict.get("GT", np.array([])), size_dict.get(qq_label, np.array([])),
                 qq_label, colors[qq_key])

    ax5 = fig.add_subplot(gs[2, 0])
    plot_size_ratio_hist(ax5, cfu_dict.get("GT"), cfu_dict.get(qq_label), qq_label, colors[qq_key])

    ax6 = fig.add_subplot(gs[2, 1])
    text_lines = []
    for name in ("GT", "CPSAM", "KC Algorithm", "FHN ColonyNet"):
        arr = size_dict.get(name, np.array([]))
        if arr.size == 0:
            continue
        text_lines.append(
            f"{name}: n={arr.size}, mean={arr.mean():.1f}, median={np.median(arr):.1f}, p95={np.percentile(arr, 95):.1f}"
        )
    ax6.axis("off")
    ax6.text(0.01, 0.99, "\n".join(text_lines) if text_lines else "No size summary data",
             ha="left", va="top", fontsize=11,
             bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#cccccc"))

    fig.suptitle("Advanced CFU Size Analysis", fontsize=16)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_summary_lines(matched: pd.DataFrame, labels: dict[str, str]) -> list[str]:
    lines = []
    lines.append(f"Matched dishes: {len(matched)}")
    lines.append(f"GT dishes with numeric counts: {int(matched['gt'].notna().sum())}")

    gt = matched["gt"].to_numpy(dtype=float)
    for col in ("cpsam", "algo1", "algo2"):
        if col not in matched.columns or not matched[col].notna().any():
            continue
        pred = matched[col].to_numpy(dtype=float)
        valid = np.isfinite(pred) & np.isfinite(gt)
        if valid.sum() == 0:
            lines.append(f"{labels[col]}: no numeric overlaps")
            continue
        gt_valid = gt[valid]
        pred_valid = pred[valid]
        diffs = gt_valid - pred_valid
        mae = float(np.mean(np.abs(diffs)))
        bias = float(np.mean(diffs))
        _, _, r2 = linregress_np(pred_valid, gt_valid)
        lines.append(f"{labels[col]}: n={valid.sum()}, MAE={mae:.3f}, Bias(GT-Pred)={bias:.3f}, R^2={r2:.4f}")

        trim_n = min(20, diffs.size)
        if trim_n > 0 and diffs.size > trim_n:
            keep_mask = np.ones(diffs.size, dtype=bool)
            worst_idx = np.argsort(np.abs(diffs))[-trim_n:]
            keep_mask[worst_idx] = False
            gt_trim = gt_valid[keep_mask]
            pred_trim = pred_valid[keep_mask]
            diffs_trim = gt_trim - pred_trim
            mae_trim = float(np.mean(np.abs(diffs_trim)))
            bias_trim = float(np.mean(diffs_trim))
            _, _, r2_trim = linregress_np(pred_trim, gt_trim)
            lines.append(
                f"{labels[col]} without {trim_n} biggest errors: "
                f"n={gt_trim.size}, MAE={mae_trim:.3f}, Bias(GT-Pred)={bias_trim:.3f}, R^2={r2_trim:.4f}"
            )
    return lines


def build_top_error_lines(matched: pd.DataFrame, labels: dict[str, str], top_n: int = 20) -> list[str]:
    lines: list[str] = []
    for col in ("cpsam", "algo1", "algo2"):
        if col not in matched.columns or not matched[col].notna().any():
            continue

        sub = matched[["petri_dish", "gt", col]].copy()
        sub = sub.rename(columns={col: "pred"})
        sub = sub.dropna(subset=["gt", "pred"])
        if sub.empty:
            continue

        sub["error"] = sub["gt"] - sub["pred"]
        sub["abs_error"] = sub["error"].abs()
        sub = sub.sort_values(["abs_error", "petri_dish"], ascending=[False, True]).head(top_n)

        lines.append("")
        lines.append(f"Top {min(top_n, len(sub))} absolute errors: {labels[col]}")
        lines.append("petri_dish | GT | Pred | GT-Pred | abs_error")
        for _, row in sub.iterrows():
            lines.append(
                f"{row['petri_dish']} | "
                f"{row['gt']:.0f} | {row['pred']:.0f} | "
                f"{row['error']:.0f} | {row['abs_error']:.0f}"
            )
    return lines


def main():
    ap = argparse.ArgumentParser(description="Create count and size comparison dashboards.")
    ap.add_argument("--CPSAM_csv", required=True, help="CPSAM per-CFU CSV")
    ap.add_argument("--Algo1_csv", help="KC Algorithm per-CFU CSV")
    ap.add_argument("--Algo2_csv", help="FHN ColonyNet per-CFU CSV")
    ap.add_argument("--GT_csv", required=True, help="Ground-truth CSV")
    ap.add_argument("--mode", choices=["test", "train"], default="test",
                    help="Use train to enable size distribution plots")
    ap.add_argument("--out_dir", default="plot_dashboard_out", help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cpsam_dish, cpsam_cfu = load_prediction_csv(args.CPSAM_csv, "CPSAM")
    algo1_dish, algo1_cfu = (None, pd.DataFrame())
    algo2_dish, algo2_cfu = (None, pd.DataFrame())
    if args.Algo1_csv:
        algo1_dish, algo1_cfu = load_prediction_csv(args.Algo1_csv, "KC Algorithm")
    if args.Algo2_csv:
        algo2_dish, algo2_cfu = load_prediction_csv(args.Algo2_csv, "FHN ColonyNet")
    gt_dish, gt_cfu = load_gt_csv(args.GT_csv, args.mode)

    matched = build_matched_table(gt_dish, cpsam_dish, algo1_dish, algo2_dish)
    matched = matched.sort_values("petri_dish").reset_index(drop=True)
    excluded = matched[matched["gt"] > 300]
    if not excluded.empty:
        print(f"[Filter] Excluding {len(excluded)} dish(es) with GT > 300: {excluded['petri_dish'].tolist()}")
    matched = matched[matched["gt"] <= 300].reset_index(drop=True)
    matched.to_csv(out_dir / "matched_rows.csv", index=False)

    labels = {"cpsam": "CPSAM", "algo1": "KC Algorithm", "algo2": "FHN ColonyNet"}
    colors = {
        "cpsam": "#1f77b4",
        "algo1": "#2ca02c",
        "algo2": "#ff7f0e",
        "GT": "#000000",
        "CPSAM": "#1f77b4",
        "KC Algorithm": "#2ca02c",
        "FHN ColonyNet": "#ff7f0e",
    }

    active_cols = active_count_sources(matched)
    if not active_cols:
        raise SystemExit("No prediction columns contained numeric data after matching.")
    matched_trimmed = trim_biggest_errors(matched, top_n=20)

    save_counts_dashboard(matched, out_dir / "dashboard_counts.png", labels, colors)
    save_regression_dashboard(matched, out_dir / "dashboard_regression.png", labels, colors)
    save_bland_altman_dashboard(matched, out_dir / "dashboard_bland_altman.png", labels, colors)
    if active_regression_sources(matched_trimmed):
        save_regression_dashboard(
            matched_trimmed,
            out_dir / "dashboard_regression_trimmed.png",
            labels,
            colors,
            title_suffix=" (without 20 biggest errors)",
        )
        save_bland_altman_dashboard(
            matched_trimmed,
            out_dir / "dashboard_bland_altman_trimmed.png",
            labels,
            colors,
            title_suffix=" (without 20 biggest errors)",
        )

    cfu_dict = {"GT": gt_cfu, "CPSAM": cpsam_cfu, "KC Algorithm": algo1_cfu, "FHN ColonyNet": algo2_cfu}
    if args.mode == "train":
        save_size_dashboard(cfu_dict, out_dir / "dashboard_sizes.png", colors)
        save_size_analysis_dashboard(matched, cfu_dict, out_dir / "dashboard_size_analysis.png", colors)

    summary_lines = build_summary_lines(matched, labels)
    summary_lines.extend(build_top_error_lines(matched, labels, top_n=20))
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (out_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
