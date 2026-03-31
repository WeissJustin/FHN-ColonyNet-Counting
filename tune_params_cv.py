#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import re
from scipy.stats import wasserstein_distance

from countCFUAPP2 import TuningParams, predict_tuning_features


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_WORKER_ITEMS: List[Dict[str, Any]] = []
_WORKER_RGB_CACHE: Dict[int, np.ndarray] = {}
_WORKER_CACHE_MAX_ITEMS = 0
_CSV_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ParamSpec:
    name: str
    coarse: Tuple[Any, ...]
    fine_step: float
    lower: float
    upper: float
    kind: str  # int | float


# ── countCFUAPP2 parameter specs ──────────────────────────────────────────────
# Each ParamSpec carries:
#   coarse   : candidate grid for round 0 (broad exploration)
#   fine_step: step for refinement rounds (halved each successive round)
#   lower/upper: hard bounds enforced by sanitize_params
#   kind     : "int" or "float"
#
# Ranges are chosen based on APP2's pipeline semantics:
#  • hsv_v_*  : V channel in skimage rgb2hsv is [0,1]; colonies are dark (V≈0.5–0.7)
#  • h_ref_frac: 0.12 → h_ref ≈ 12% of the p10-p90 gray range; [0.04, 0.30] covers
#                sparse (low contrast) to dense (high contrast) plates
#  • open_disk_large: APP2 upsamples 2× before opening, so effective radius doubles;
#                keep upper bound at 28 to avoid erasing genuine colonies
#  • ws_thresh_abs_frac: fraction of dist-max for h_maxima; low = more splits,
#                high = conservative; default 0.05 is already conservative
#  • nonsmall_cluster_ecc_min: 0.99 by default (almost perfectly linear blobs);
#                widen range downward to 0.70 so the search can relax it

PARAM_SPECS: Dict[str, ParamSpec] = {
    # ── Block 2: HSV mask calibration ─────────────────────────────────────────
    "hsv_v_start": ParamSpec(
        "hsv_v_start", (0.50, 0.57, 0.635, 0.70, 0.76), 0.03, 0.30, 0.95, "float"),
    "hsv_v_step": ParamSpec(
        "hsv_v_step", (0.005, 0.008, 0.01, 0.015, 0.02), 0.003, 0.003, 0.10, "float"),
    "hsv_v_min": ParamSpec(
        "hsv_v_min", (0.20, 0.28, 0.35, 0.42, 0.49), 0.03, 0.10, 0.90, "float"),
    "hsv_s_min": ParamSpec(
        "hsv_s_min", (0.00, 0.03, 0.07, 0.12, 0.17), 0.02, 0.0, 0.50, "float"),
    "hsv_err_tol": ParamSpec(
        "hsv_err_tol", (0.0015, 0.003, 0.0065, 0.010, 0.015), 0.001, 5e-4, 0.10, "float"),
    # Controls how sensitive the imextendedmin reference is to plate contrast.
    # Smaller → lower h_ref → more pixels flagged as reference → tighter calibration.
    "h_ref_frac": ParamSpec(
        "h_ref_frac", (0.06, 0.09, 0.12, 0.16, 0.22), 0.02, 0.03, 0.40, "float"),
    # ── Block 3: Morphological cleanup ────────────────────────────────────────
    # hsv_extent_min: filters filamentary artefacts; default 0.23.
    # Too high → rejects valid small colonies; too low → keeps scratches.
    "hsv_extent_min": ParamSpec(
        "hsv_extent_min", (0.08, 0.14, 0.20, 0.27, 0.35), 0.03, 0.01, 0.80, "float"),
    # Small opening: disk(1) on sparse plates, disk(N) on dense ones (adaptive).
    # Coarse grid starts at 1 because APP2 already auto-selects 1 for sparse plates.
    "hsv_open_disk_small": ParamSpec(
        "hsv_open_disk_small", (1, 2, 3, 5, 7), 1, 1, 15, "int"),
    # Large opening (applied after 2× upsample, so effective radius = 2×value).
    "hsv_open_disk_large": ParamSpec(
        "hsv_open_disk_large", (6, 10, 14, 18, 22), 2, 3, 28, "int"),
    "min_object_area": ParamSpec(
        "min_object_area", (30, 60, 100, 150, 200, 250), 20, 10, 500, "int"),
    # ── Blocks 4: Colony / cluster classification ──────────────────────────────
    # small_area_quantile: separates "small" blobs (single colonies) from larger
    # blobs (potential clusters) using the per-image area distribution.
    "small_area_quantile": ParamSpec(
        "small_area_quantile", (0.55, 0.65, 0.75, 0.85, 0.90), 0.05, 0.30, 0.99, "float"),
    # small_cluster_extent_min: large-area blobs also need a minimum extent
    # (area/bbox) to be treated as clusters; very boxy = genuine cluster.
    "small_cluster_extent_min": ParamSpec(
        "small_cluster_extent_min", (0.05, 0.12, 0.20, 0.28, 0.38), 0.03, 0.01, 0.90, "float"),
    # nonsmall_cluster_ecc_min: highly eccentric (ecc→1) blobs are two or more
    # colonies fused end-to-end; default 0.99 is very tight.
    "nonsmall_cluster_ecc_min": ParamSpec(
        "nonsmall_cluster_ecc_min", (0.70, 0.80, 0.90, 0.95, 0.99), 0.03, 0.10, 0.999, "float"),
    "nonsmall_cluster_extent_min": ParamSpec(
        "nonsmall_cluster_extent_min", (0.05, 0.12, 0.20, 0.28, 0.38), 0.03, 0.01, 0.90, "float"),
    # colony_circularity_min: 4π·area/perimeter²; ~1 = perfect circle.
    # Default 0.09 is very permissive; typical single colonies are 0.5–0.9.
    # Allow lower bound 0.01 so the search can relax it for irregular colony shapes.
    "colony_circularity_min": ParamSpec(
        "colony_circularity_min", (0.03, 0.06, 0.09, 0.14, 0.20, 0.28), 0.03, 0.01, 0.80, "float"),
    # ── Block 5: Watershed ────────────────────────────────────────────────────
    # ws_thresh_abs_frac: h_maxima threshold = frac × dist_max. Lower → more
    # splits (aggressive); higher → fewer splits (conservative). Default 0.05.
    # Extend coarse grid below default to allow more aggressive splitting.
    "ws_thresh_abs_frac": ParamSpec(
        "ws_thresh_abs_frac", (0.02, 0.04, 0.06, 0.10, 0.15, 0.25), 0.02, 0.005, 0.90, "float"),
    # ws_gauss_sigma: Gaussian blur of distance map before h_maxima.
    # Larger → smoother peaks → fewer seeds → more conservative.
    "ws_gauss_sigma": ParamSpec(
        "ws_gauss_sigma", (0.3, 0.6, 1.0, 1.5, 2.0, 2.5), 0.2, 0.1, 5.0, "float"),
}


# Stage ordering follows the pipeline: upstream params (HSV mask) first,
# downstream params (watershed) last.  Optimising upstream first avoids
# searching downstream params on a suboptimal mask.
STAGES: List[Tuple[str, Sequence[str]]] = [
    (
        # Block 2: controls which pixels enter the candidate mask.
        # h_ref_frac and hsv_err_tol jointly determine how tightly the V
        # threshold is calibrated against the reference.
        "hsv",
        (
            "hsv_v_start",
            "hsv_v_step",
            "hsv_v_min",
            "hsv_s_min",
            "hsv_err_tol",
            "h_ref_frac",
        ),
    ),
    (
        # Block 3: morphological cleanup of the raw HSV mask.
        # min_object_area is here because it is applied twice (area-open before
        # and after morpho cleanup) and its primary effect is noise removal.
        "morpho",
        (
            "hsv_extent_min",
            "hsv_open_disk_small",
            "hsv_open_disk_large",
            "min_object_area",
        ),
    ),
    (
        # Block 4: which blobs become colonies vs clusters.
        "count_filter",
        (
            "small_area_quantile",
            "small_cluster_extent_min",
            "nonsmall_cluster_ecc_min",
            "nonsmall_cluster_extent_min",
            "colony_circularity_min",
        ),
    ),
    (
        # Block 5: how clusters are split into individual colony counts.
        "watershed",
        (
            "ws_thresh_abs_frac",
            "ws_gauss_sigma",
        ),
    ),
]
STAGE_NAMES = tuple(stage_name for stage_name, _ in STAGES)


def resolve_column_name(df: pd.DataFrame, requested: Optional[str], candidates: Sequence[str], required: bool) -> Optional[str]:
    if requested and requested in df.columns:
        return requested

    lower_map = {str(col).strip().lower(): col for col in df.columns}
    if requested and requested.strip().lower() in lower_map:
        return str(lower_map[requested.strip().lower()])

    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return str(lower_map[cand.lower()])

    if required:
        cols = ", ".join(map(str, df.columns))
        wanted = ", ".join(candidates)
        raise KeyError(
            f"Could not find a matching CSV column. "
            f"Requested={requested!r}. Available columns: [{cols}]. "
            f"Expected something like: [{wanted}]"
        )
    return None


def _init_worker(items_data: Optional[List[Dict[str, Any]]] = None, cache_max_items: int = 0) -> None:
    global _WORKER_ITEMS, _WORKER_RGB_CACHE, _WORKER_CACHE_MAX_ITEMS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    _WORKER_ITEMS = items_data or []
    _WORKER_RGB_CACHE = {}
    _WORKER_CACHE_MAX_ITEMS = max(0, int(cache_max_items))
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def append_row_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_WRITE_LOCK:
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def write_single_row_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _CSV_WRITE_LOCK:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_instance_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32, copy=False)


def mask_instance_areas(mask: np.ndarray) -> List[float]:
    flat = mask.reshape(-1)
    flat = flat[flat > 0]
    if flat.size == 0:
        return []
    _, counts = np.unique(flat, return_counts=True)
    return [float(c) for c in counts.tolist()]


def resolve_mask_path(mask_dir: Path, image_name: str, explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = mask_dir / explicit
        if p.exists():
            return p
    stem = Path(image_name).stem
    exact = mask_dir / image_name
    if exact.exists():
        return exact
    candidate_patterns = [
        stem + ".*",
        stem.replace("_expt", "") + ".*" if "_expt" in stem else None,
        stem.replace("__expt", "") + ".*" if "__expt" in stem else None,
    ]
    suffix_candidates = [
        f"{stem}_output_masks.*",
        f"{stem}_masks.*",
        f"{stem}_mask.*",
    ]
    if stem.endswith("_expt"):
        base = stem[:-5]
        suffix_candidates.extend(
            [
                f"{base}_output_masks.*",
                f"{base}_masks.*",
                f"{base}_mask.*",
            ]
        )

    seen = set()
    for pattern in [p for p in candidate_patterns + suffix_candidates if p]:
        for cand in sorted(mask_dir.glob(pattern)):
            if cand in seen:
                continue
            seen.add(cand)
            if cand.is_file() and cand.suffix.lower() in IMAGE_EXTS:
                return cand
    return None


def normalize_sample_key(name: str) -> str:
    stem = Path(str(name)).stem
    stem = re.sub(r"^DIFIP\\d+_", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_output_masks?$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_masks?$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_mask$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_expt$", "", stem, flags=re.IGNORECASE)
    return stem.strip().lower()


def sanitize_params(params: TuningParams) -> TuningParams:
    p = TuningParams(**asdict(params))

    def _clip(name: str) -> None:
        spec = PARAM_SPECS[name]
        value = getattr(p, name)
        if spec.kind == "int":
            value = int(round(float(value)))
            value = int(np.clip(value, spec.lower, spec.upper))
        else:
            value = float(np.clip(float(value), spec.lower, spec.upper))
        setattr(p, name, value)

    for name in PARAM_SPECS:
        _clip(name)

    # Enforce: the calibration loop must be able to take at least one step.
    # hsv_v_min must be below (hsv_v_start - hsv_v_step) so the first step
    # doesn't immediately exit the loop.
    p.hsv_v_step = max(float(p.hsv_v_step), 1e-3)
    p.hsv_v_min = min(float(p.hsv_v_min), float(p.hsv_v_start) - float(p.hsv_v_step) - 1e-4)
    p.hsv_v_min = max(float(p.hsv_v_min), 0.01)

    return p


def params_key(params: TuningParams) -> Tuple[Tuple[str, Any], ...]:
    p = sanitize_params(params)
    return tuple(sorted(asdict(p).items()))


def stratified_kfold_indices(items: Sequence[Dict[str, Any]], k: int, seed: int, n_bins: int = 6):
    n = len(items)
    if n < k:
        k = max(2, n)

    gts = np.array([int(item["gt_count"]) for item in items], dtype=float)
    if np.all(gts == gts[0]):
        idx = list(range(n))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        folds = [idx[i::k] for i in range(k)]
        for i in range(k):
            test = set(folds[i])
            train = [j for j in idx if j not in test]
            yield train, list(test)
        return

    edges = np.unique(np.quantile(gts, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) <= 2:
        bins = np.zeros(n, dtype=int)
    else:
        bins = np.digitize(gts, edges[1:-1], right=True)

    by_bin: Dict[int, List[int]] = {}
    for i, b in enumerate(bins):
        by_bin.setdefault(int(b), []).append(i)

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for b in sorted(by_bin):
        bucket = by_bin[b]
        rng.shuffle(bucket)
        for t, idx in enumerate(bucket):
            folds[t % k].append(idx)

    all_idx = list(range(n))
    for i in range(k):
        test = set(folds[i])
        train = [j for j in all_idx if j not in test]
        yield train, list(test)


def mean_abs(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()) if arr.size else 0.0


def weighted_mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    weights = 1.0 + np.sqrt(np.maximum(yt, 0.0)) / 12.0
    return float(np.sum(np.abs(yt - yp) * weights) / max(np.sum(weights), 1e-9))


def rmse(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(math.sqrt(np.mean((yt - yp) ** 2)))


def count_nmae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Normalised MAE with floor=10 (kept for reporting / backward compat)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.maximum(yt, 10.0)
    return float(np.mean(np.abs(yt - yp) / denom))


def count_log_mae(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """
    Mean absolute error in log1p space.

    Preferred primary count metric over count_nmae because it is equally
    calibrated across all count scales: predicting 2× too many gives the
    same penalty whether the true count is 5 or 100.  No arbitrary floor
    constant needed — log1p(0)=0 handles zero-count plates naturally.

    Typical values:
      perfect prediction  → 0.0
      2× over/under       → log(2) ≈ 0.69
      10× over/under      → log(10) ≈ 2.30
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(np.log1p(yp) - np.log1p(yt))))


def area_distribution_error(gt_areas: Sequence[float], pred_areas: Sequence[float]) -> float:
    if len(gt_areas) == 0 and len(pred_areas) == 0:
        return 0.0
    if len(gt_areas) == 0 or len(pred_areas) == 0:
        return 2.0
    gt_log = np.log1p(np.asarray(gt_areas, dtype=float))
    pred_log = np.log1p(np.asarray(pred_areas, dtype=float))
    return float(wasserstein_distance(gt_log, pred_log))


def area_mass_rel_error(gt_areas: Sequence[float], pred_areas: Sequence[float]) -> float:
    gt_sum = float(np.sum(gt_areas))
    pred_sum = float(np.sum(pred_areas))
    return abs(pred_sum - gt_sum) / max(gt_sum, 1.0)


def build_candidate_values(spec: ParamSpec, current_value: Any, refine_round: int, refine_span: int = 2) -> List[Any]:
    if refine_round <= 0:
        values = list(spec.coarse)
        values.append(current_value)
    else:
        step = spec.fine_step / float(refine_round + 1)
        if spec.kind == "int":
            step = max(1.0, round(step))
        span = max(1, int(refine_span))
        values = [current_value + offset * step for offset in range(-span, span + 1)]

    clipped: List[Any] = []
    for value in values:
        if spec.kind == "int":
            value = int(round(float(value)))
            value = int(np.clip(value, spec.lower, spec.upper))
        else:
            value = float(np.clip(float(value), spec.lower, spec.upper))
            value = round(value, 6)
        clipped.append(value)
    return list(dict.fromkeys(clipped))


def expand_edge_candidate_values(
    spec: ParamSpec,
    base_candidates: Sequence[Any],
    best_value: Any,
    refine_round: int,
    expansion_i: int,
) -> List[Any]:
    candidates = sorted(dict.fromkeys(base_candidates))
    if not candidates:
        return []

    lo = candidates[0]
    hi = candidates[-1]
    if best_value != lo and best_value != hi:
        return []

    if refine_round <= 0 and len(candidates) >= 2:
        gaps = [abs(float(b) - float(a)) for a, b in zip(candidates[:-1], candidates[1:]) if float(b) != float(a)]
        step = max(gaps) if gaps else float(spec.fine_step)
    elif len(candidates) >= 2:
        edge_neighbor = candidates[1] if best_value == lo else candidates[-2]
        step = abs(float(best_value) - float(edge_neighbor))
    else:
        step = float(spec.fine_step)

    if spec.kind == "int":
        step = max(1.0, round(step))
    else:
        step = max(float(spec.fine_step), float(step))

    direction = -1.0 if best_value == lo else 1.0
    n_new = 2
    new_values: List[Any] = []
    for offset_i in range(1, n_new + 1):
        value = float(best_value) + direction * step * float(expansion_i * n_new + offset_i)
        if spec.kind == "int":
            value = int(round(value))
            value = int(np.clip(value, spec.lower, spec.upper))
        else:
            value = float(np.clip(value, spec.lower, spec.upper))
            value = round(value, 6)
        if value not in candidates and value not in new_values:
            new_values.append(value)
    return new_values


def selection_score(metrics: Dict[str, float], fold_std_weight: float) -> float:
    return float(metrics["score"] + fold_std_weight * metrics.get("fold_score_std", 0.0))


def composite_score(
    count_err: float,
    area_emd: float,
    area_mass: float,
    *,
    count_weight: float,
    emd_weight: float,
    mass_weight: float,
) -> float:
    return float(
        count_weight * count_err
        + emd_weight * area_emd
        + mass_weight * area_mass
    )


def selected_stages(stage_names: Optional[Sequence[str]]) -> List[Tuple[str, Sequence[str]]]:
    if not stage_names:
        return list(STAGES)

    requested = set(stage_names)
    return [(stage_name, params) for stage_name, params in STAGES if stage_name in requested]


def _load_rgb_for_worker(item_idx: int, image_path: Path) -> np.ndarray:
    global _WORKER_RGB_CACHE, _WORKER_CACHE_MAX_ITEMS
    if _WORKER_CACHE_MAX_ITEMS <= 0:
        return load_rgb(image_path)

    rgb = _WORKER_RGB_CACHE.get(item_idx)
    if rgb is not None:
        # Refresh insertion order to approximate LRU eviction.
        _WORKER_RGB_CACHE.pop(item_idx, None)
        _WORKER_RGB_CACHE[item_idx] = rgb
        return rgb

    rgb = load_rgb(image_path)
    _WORKER_RGB_CACHE[item_idx] = rgb
    while len(_WORKER_RGB_CACHE) > _WORKER_CACHE_MAX_ITEMS:
        oldest_key = next(iter(_WORKER_RGB_CACHE))
        _WORKER_RGB_CACHE.pop(oldest_key, None)
    return rgb


def _predict_one(job: Tuple[int, str, float, Dict[str, Any], bool]):
    global _WORKER_ITEMS
    item_idx, dish_mode, target_area_px2, params_dict, use_blackhat = job
    item = _WORKER_ITEMS[item_idx]
    image_path = Path(item["image_path"])

    rgb = _load_rgb_for_worker(item_idx, image_path)

    gt_count = int(item["gt_count"])
    gt_areas = [float(a) for a in item["gt_areas"]]

    params = sanitize_params(TuningParams(**params_dict))
    effective_dish_mode = dish_mode
    if dish_mode == "auto" and image_path.stem.lower().endswith("_expt"):
        effective_dish_mode = "pre_cropped"
    pred = predict_tuning_features(
        rgb,
        dish_mode=effective_dish_mode,
        target_area_px2=target_area_px2,
        use_blackhat=bool(use_blackhat),
        params=params,
    )

    pred_count = int(pred["count"])
    pred_areas = [float(a) for a in pred["instance_areas"]]
    return {
        "filename": image_path.name,
        "gt_count": int(gt_count),
        "pred_count": pred_count,
        "gt_areas": gt_areas,
        "pred_areas": pred_areas,
        "methods": "|".join(pred.get("methods", [])),
        "classes": "|".join(pred.get("classes", [])),
        "mean_area": float(pred.get("mean_area", 0.0)),
    }


def _predict_one_cv(job: Tuple[int, int, str, float, Dict[str, Any], bool]):
    fold_i, item_idx, dish_mode, target_area_px2, params_dict, use_blackhat = job
    row = _predict_one((item_idx, dish_mode, target_area_px2, params_dict, use_blackhat))
    row["fold"] = int(fold_i)
    return row


def eval_params_cv(
    items: Sequence[Dict[str, Any]],
    k: int,
    seed: int,
    dish_mode: str,
    target_area_px2: float,
    use_blackhat: bool,
    params: TuningParams,
    chunksize: int,
    fold_csv_path: Optional[Path],
    eval_id: str,
    stage_name: str,
    param_name: str,
    round_i: int,
    executor: ProcessPoolExecutor,
    fold_std_weight: float,
    score_weight_count: float,
    score_weight_emd: float,
    score_weight_mass: float,
) -> Dict[str, float]:
    params = sanitize_params(params)
    params_dict = asdict(params)

    fold_indices = list(stratified_kfold_indices(items, k=k, seed=seed))
    all_results: List[Dict[str, Any]] = []
    fold_scores: List[float] = []
    fold_rows_map: Dict[int, List[Dict[str, Any]]] = {fold_i: [] for fold_i in range(1, len(fold_indices) + 1)}
    jobs = [
        (
            fold_i,
            j,
            dish_mode,
            float(target_area_px2),
            params_dict,
            bool(use_blackhat),
        )
        for fold_i, (_train_idx, test_idx) in enumerate(fold_indices, start=1)
        for j in test_idx
    ]
    effective_chunksize = chunksize
    if effective_chunksize <= 0:
        # Use the worker count, not os.cpu_count() — on SLURM os.cpu_count()
        # returns the whole machine's CPUs (128+), not the allocated ones, which
        # makes the auto-chunksize collapse to 1 and causes excessive dispatch overhead.
        _n_workers = executor._max_workers  # type: ignore[attr-defined]
        effective_chunksize = max(1, len(jobs) // max(4 * _n_workers, 1))

    _n_done = 0
    _total = len(jobs)
    _print_every = max(1, _total // 10)
    for row in executor.map(_predict_one_cv, jobs, chunksize=effective_chunksize):
        _n_done += 1
        if _n_done % _print_every == 0 or _n_done == _total:
            print(f"    progress: {_n_done}/{_total} images", flush=True)
        fold_i = int(row["fold"])
        fold_rows_map[fold_i].append(row)
        all_results.append(row)

    for fold_i in range(1, len(fold_indices) + 1):
        fold_rows = fold_rows_map[fold_i]
        if not fold_rows:
            continue

        gt_counts = [row["gt_count"] for row in fold_rows]
        pred_counts = [row["pred_count"] for row in fold_rows]
        fold_count_lmae = count_log_mae(gt_counts, pred_counts)   # primary count metric
        fold_count_nmae = count_nmae(gt_counts, pred_counts)       # reported for reference
        fold_area_emd = mean_abs(area_distribution_error(row["gt_areas"], row["pred_areas"]) for row in fold_rows)
        fold_area_mass = mean_abs(area_mass_rel_error(row["gt_areas"], row["pred_areas"]) for row in fold_rows)
        fold_score = composite_score(
            fold_count_lmae,   # log-MAE as primary count error
            fold_area_emd,
            fold_area_mass,
            count_weight=score_weight_count,
            emd_weight=score_weight_emd,
            mass_weight=score_weight_mass,
        )
        fold_scores.append(fold_score)

        if fold_csv_path is not None:
            append_row_csv(
                fold_csv_path,
                {
                    "timestamp": _now_iso(),
                    "eval_id": eval_id,
                    "stage": stage_name,
                    "param": param_name,
                    "round": round_i,
                    "fold": fold_i,
                    "fold_score": fold_score,
                    "fold_count_log_mae": fold_count_lmae,
                    "fold_count_nmae": fold_count_nmae,
                    "fold_area_emd": fold_area_emd,
                    "fold_area_mass_rel": fold_area_mass,
                    **params_dict,
                },
            )

    gt_counts = [row["gt_count"] for row in all_results]
    pred_counts = [row["pred_count"] for row in all_results]
    count_lmae = count_log_mae(gt_counts, pred_counts)   # primary
    count_nmae_ = count_nmae(gt_counts, pred_counts)      # for reporting
    area_emd = mean_abs(area_distribution_error(row["gt_areas"], row["pred_areas"]) for row in all_results)
    area_mass = mean_abs(area_mass_rel_error(row["gt_areas"], row["pred_areas"]) for row in all_results)
    score = composite_score(
        count_lmae,
        area_emd,
        area_mass,
        count_weight=score_weight_count,
        emd_weight=score_weight_emd,
        mass_weight=score_weight_mass,
    )

    return {
        "score": float(score),
        "count_log_mae": count_lmae,
        "count_nmae": count_nmae_,
        "count_wmae": weighted_mae(gt_counts, pred_counts),
        "count_rmse": rmse(gt_counts, pred_counts),
        "area_emd_mean": area_emd,
        "area_mass_rel_mean": area_mass,
        "fold_score_std": float(np.std(fold_scores)) if fold_scores else 0.0,
        "selection_score": float(score + fold_std_weight * (float(np.std(fold_scores)) if fold_scores else 0.0)),
        "n": len(all_results),
    }


def load_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    images_dir = Path(args.images_dir).expanduser().resolve()
    masks_dir = Path(args.masks_dir).expanduser().resolve()
    items: List[Dict[str, Any]] = []

    csv_lookup: Dict[str, Dict[str, Any]] = {}
    count_col: Optional[str] = None
    mask_col: Optional[str] = None
    if args.csv:
        df = pd.read_csv(args.csv)
        filename_col = resolve_column_name(
            df,
            args.filename_col,
            ("filename", "file", "file_name", "image", "image_name", "img", "sample", "sample_id"),
            required=True,
        )
        mask_col = resolve_column_name(
            df,
            args.mask_col,
            ("mask_filename", "mask", "mask_file", "mask_name", "instance_mask", "label"),
            required=False,
        )
        count_col = resolve_column_name(
            df,
            args.count_col,
            ("count", "gt_count", "num_cfu", "cfu_count"),
            required=False,
        )
        label_col = resolve_column_name(df, None, ("label",), required=False)
        dilu_col = resolve_column_name(df, None, ("dilu", "dilution"), required=False)
        suffix_col = resolve_column_name(df, None, ("x_suffix", "suffix"), required=False)

        for _, row in df.iterrows():
            keys = set()
            if filename_col and pd.notna(row[filename_col]):
                keys.add(normalize_sample_key(str(row[filename_col])))
            if label_col and dilu_col and pd.notna(row[label_col]) and pd.notna(row[dilu_col]):
                label = str(row[label_col]).strip()
                dilu = str(int(row[dilu_col])) if not isinstance(row[dilu_col], str) else str(row[dilu_col]).strip()
                keys.add(normalize_sample_key(f"{label}dilu{dilu}"))
                if suffix_col and pd.notna(row[suffix_col]):
                    keys.add(normalize_sample_key(f"{label}dilu{dilu}{str(row[suffix_col]).strip()}"))
            for key in keys:
                csv_lookup.setdefault(key, row.to_dict())

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        key = normalize_sample_key(image_path.name)
        row = csv_lookup.get(key)
        explicit_mask = None
        if row is not None and mask_col and mask_col in row and pd.notna(row[mask_col]):
            explicit_mask = str(row[mask_col])

        mask_path = resolve_mask_path(masks_dir, image_path.name, explicit_mask)
        if mask_path is None:
            continue

        mask = load_instance_mask(mask_path)
        gt_areas = mask_instance_areas(mask)
        mask_count = len(gt_areas)
        gt_count = mask_count
        if row is not None and count_col and count_col in row and pd.notna(row[count_col]):
            csv_count = int(row[count_col])
            if csv_count != mask_count:
                print(f"Warning: count mismatch for {image_path.name}: csv={csv_count}, mask={mask_count}. Using mask count.")

        items.append({
            "image_path": image_path,
            "mask_path": mask_path,
            "gt_count": gt_count,
            "gt_areas": gt_areas,
        })

    if args.max_images > 0 and len(items) > args.max_images:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(items))
        rng.shuffle(idx)
        items = [items[int(i)] for i in idx[:args.max_images]]

    return items


def log_eval(
    path: Path,
    eval_id: str,
    stage_name: str,
    param_name: str,
    round_i: int,
    value: Any,
    metrics: Dict[str, float],
    params: TuningParams,
) -> None:
    append_row_csv(
        path,
        {
            "timestamp": _now_iso(),
            "eval_id": eval_id,
            "stage": stage_name,
            "param": param_name,
            "round": round_i,
            "candidate_value": value,
            **metrics,
            **asdict(params),
        },
    )


def hierarchical_search(args: argparse.Namespace, items: Sequence[Dict[str, Any]]) -> Tuple[TuningParams, Dict[str, float]]:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trial_csv = out_dir / "trial_results.csv"
    fold_csv = out_dir / "fold_results.csv"
    best_csv = out_dir / "best_so_far.csv"

    best_params = sanitize_params(TuningParams())
    cache: Dict[Tuple[Tuple[str, Any], ...], Dict[str, float]] = {}
    eval_counter = itertools.count(1)
    cache_lock = threading.Lock()
    item_payload = [
        {
            "image_path": str(it["image_path"]),
            "gt_count": int(it["gt_count"]),
            "gt_areas": [float(a) for a in it["gt_areas"]],
        }
        for it in items
    ]
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(item_payload, args.worker_cache_images),
    ) as executor:
        def evaluate(params: TuningParams, stage_name: str, param_name: str, round_i: int, value: Any) -> Dict[str, float]:
            params = sanitize_params(params)
            key = params_key(params)
            with cache_lock:
                cached = cache.get(key)
                if cached is not None:
                    return cached
                eval_id = f"eval_{next(eval_counter):05d}"
            metrics = eval_params_cv(
                items=items,
                k=args.k,
                seed=args.seed,
                dish_mode=args.dish_mode,
                target_area_px2=args.target_area_px2,
                use_blackhat=args.use_blackhat,
                params=params,
                chunksize=args.chunksize,
                fold_csv_path=fold_csv,
                eval_id=eval_id,
                stage_name=stage_name,
                param_name=param_name,
                round_i=round_i,
                executor=executor,
                fold_std_weight=args.fold_std_weight,
                score_weight_count=args.score_weight_count,
                score_weight_emd=args.score_weight_emd,
                score_weight_mass=args.score_weight_mass,
            )
            with cache_lock:
                cache[key] = metrics
            log_eval(trial_csv, eval_id, stage_name, param_name, round_i, value, metrics, params)
            return metrics

        def evaluate_candidates_batch(
            stage_name: str,
            param_name: str,
            round_i: int,
            candidates: Sequence[Tuple[Any, TuningParams]],
        ) -> Dict[Any, Dict[str, float]]:
            results: Dict[Any, Dict[str, float]] = {}
            uncached: List[Tuple[Any, TuningParams]] = []

            for value, cand in candidates:
                key = params_key(cand)
                with cache_lock:
                    cached = cache.get(key)
                if cached is not None:
                    results[value] = cached
                else:
                    uncached.append((value, cand))

            if not uncached:
                return results

            outer_workers = max(1, min(len(uncached), int(args.candidate_parallelism)))
            if outer_workers == 1:
                for value, cand in uncached:
                    results[value] = evaluate(cand, stage_name, param_name, round_i, value)
            else:
                with ThreadPoolExecutor(max_workers=outer_workers) as thread_ex:
                    future_map = {
                        thread_ex.submit(evaluate, cand, stage_name, param_name, round_i, value): value
                        for value, cand in uncached
                    }
                    for fut in as_completed(future_map):
                        value = future_map[fut]
                        results[value] = fut.result()

            return results

        best_metrics = evaluate(best_params, "baseline", "baseline", 0, "default")
        append_row_csv(
            best_csv,
            {
                "timestamp": _now_iso(),
                "stage": "baseline",
                **best_metrics,
                **asdict(best_params),
            },
        )

        for stage_name, param_names in selected_stages(args.stages):
            print(f"\n=== Stage: {stage_name} ===")
            for round_i in range(args.stage_rounds):
                improved = False
                print(f"  Round {round_i + 1}/{args.stage_rounds}")
                for param_name in param_names:
                    spec = PARAM_SPECS[param_name]
                    current_value = getattr(best_params, param_name)
                    candidates = build_candidate_values(spec, current_value, refine_round=round_i, refine_span=args.refine_span)
                    local_best_value = current_value
                    local_best_params = best_params
                    local_best_metrics = best_metrics

                    print(f"    Tuning {param_name} with {len(candidates)} candidates")
                    seen_values = set()
                    expansion_i = 0
                    while True:
                        candidate_pairs: List[Tuple[Any, TuningParams]] = []
                        for value in candidates:
                            if value in seen_values:
                                continue
                            cand = TuningParams(**asdict(best_params))
                            setattr(cand, param_name, value)
                            cand = sanitize_params(cand)
                            candidate_pairs.append((value, cand))
                            seen_values.add(value)

                        if candidate_pairs:
                            batch_metrics = evaluate_candidates_batch(stage_name, param_name, round_i, candidate_pairs)

                            for value, cand in candidate_pairs:
                                metrics = batch_metrics[value]
                                print(
                                    f"      {param_name}={value} -> score={metrics['score']:.4f} "
                                    f"select={metrics['selection_score']:.4f} "
                                    f"count_lmae={metrics['count_log_mae']:.4f} "
                                    f"count_nmae={metrics['count_nmae']:.4f} area_emd={metrics['area_emd_mean']:.4f}"
                                )
                                cand_select = selection_score(metrics, args.fold_std_weight)
                                local_select = selection_score(local_best_metrics, args.fold_std_weight)
                                if (
                                    cand_select + 1e-12 < local_select
                                    or (
                                        abs(cand_select - local_select) <= 1e-12
                                        and metrics["score"] + 1e-12 < local_best_metrics["score"]
                                    )
                                ):
                                    local_best_value = value
                                    local_best_params = cand
                                    local_best_metrics = metrics

                        if expansion_i >= args.edge_expansions:
                            break

                        edge_values = expand_edge_candidate_values(
                            spec=spec,
                            base_candidates=sorted(seen_values),
                            best_value=local_best_value,
                            refine_round=round_i,
                            expansion_i=expansion_i,
                        )
                        if not edge_values:
                            break
                        expansion_i += 1
                        candidates = edge_values
                        print(f"      expanding edge for {param_name}: {edge_values}")

                    if local_best_params is not best_params:
                        improved = True
                        best_params = local_best_params
                        best_metrics = local_best_metrics
                        print(f"      -> improved {param_name} to {local_best_value}")
                        append_row_csv(
                            best_csv,
                            {
                                "timestamp": _now_iso(),
                                "stage": stage_name,
                                "round": round_i,
                                "param": param_name,
                                **best_metrics,
                                **asdict(best_params),
                            },
                        )
                if not improved:
                    print("  No further improvement in this stage.")
                    break

    return best_params, best_metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Hierarchical CV tuning for countCFUAPP.py with instance-mask-aware GT")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--mask_col", default="mask_filename")
    ap.add_argument("--count_col", default="count")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--dish_mode", default="auto", choices=["auto", "single", "double", "pre_cropped"])
    ap.add_argument("--use_blackhat", action="store_true", default=False, help="Enable blackhat pre-processing in countCFUAPP2 (boosts faint colony signal)")
    ap.add_argument("--target_area_px2", type=float, default=2245000.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=0, help="Jobs per worker dispatch; 0 chooses automatically")
    ap.add_argument("--stage_rounds", type=int, default=2, help="How many coarse-to-fine rounds to run per stage")
    ap.add_argument("--stages", nargs="+", choices=STAGE_NAMES, help="Restrict tuning to specific stages, e.g. --stages watershed hsv")
    ap.add_argument("--refine_span", type=int, default=3, help="How many fine-grid steps to test on each side of the current value in refinement rounds")
    ap.add_argument("--candidate_parallelism", type=int, default=1, help="How many parameter candidates to evaluate concurrently; increase cautiously because it raises memory use")
    ap.add_argument("--worker_cache_images", type=int, default=2, help="How many images each worker process keeps cached in memory; 0 disables worker-side image caching")
    ap.add_argument("--score_weight_count", type=float, default=0.55, help="Composite-score weight for count NMAE")
    ap.add_argument("--score_weight_emd", type=float, default=0.25, help="Composite-score weight for area distribution EMD")
    ap.add_argument("--score_weight_mass", type=float, default=0.20, help="Composite-score weight for relative area-mass error")
    ap.add_argument("--fold_std_weight", type=float, default=0.15, help="Penalty weight for fold-to-fold instability when selecting candidates")
    ap.add_argument("--edge_expansions", type=int, default=2, help="Extra outward search steps when the current best candidate sits on the edge of the tested range")
    ap.add_argument("--out_dir", default="tuning_out_hierarchical")
    args = ap.parse_args()

    total_score_weight = args.score_weight_count + args.score_weight_emd + args.score_weight_mass
    if total_score_weight <= 0:
        raise SystemExit("At least one score weight must be positive.")
    args.score_weight_count /= total_score_weight
    args.score_weight_emd /= total_score_weight
    args.score_weight_mass /= total_score_weight

    items = load_items(args)
    if not items:
        raise SystemExit("No matched image/mask pairs found.")

    print(f"Loaded {len(items)} image/mask pairs")
    print(f"Using {args.workers or (os.cpu_count() or 1)} workers")
    print(f"Candidate parallelism: {max(1, int(args.candidate_parallelism))}")
    print(
        "Score weights: "
        f"count={args.score_weight_count:g}, "
        f"emd={args.score_weight_emd:g}, "
        f"mass={args.score_weight_mass:g}"
    )
    if args.stages:
        print(f"Selected stages: {', '.join(args.stages)}")

    best_params, best_metrics = hierarchical_search(args, items)

    print("\nBest parameters:")
    for key, value in asdict(best_params).items():
        print(f"  {key}: {value}")

    print("\nBest metrics:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
