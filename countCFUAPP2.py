"""
countCFUAPP2.py  —  HSV-first CFU counter (clean rewrite)
Version: 2026-03-28

Pipeline overview
-----------------

BLOCK 1 · PREPROCESSING
    Standard enhancement identical to countCFUAPP:
        imadjust → locallapfilt → localcontrast → imlocalbrighten
    Lifts shadows, sharpens colony edges, and normalises brightness across
    the dish so the later steps see a consistent, high-contrast image.

    Optional BLACKHAT step (use_blackhat=True):
        blackhat = morph_close(gray, disk_r) - gray
    Blackhat highlights dark blobs on a bright background by computing
    "how much darker is each pixel than its local neighbourhood".
    Adding it back to the processed grayscale exaggerates the depth of
    each colony, making the V-channel dip more pronounced in the HSV step.
    Especially helpful for faint, low-contrast colonies.

BLOCK 2 · HSV ADAPTIVE MASKING
    Convert preprocessed RGB → HSV.
    A pixel is a CFU candidate when:
        V < v_thresh   (colony is darker than background)
        S < s_max      (colony is achromatic; agar is chromatic/pink/yellow)
    Colonies have S ≈ 0.05–0.17; preprocessed agar has S ≈ 0.38–0.51.
    S < 0.30 cleanly rejects coloured agar pixels that sneak through the V
    threshold (especially on golden/yellow agar plates).
    The V threshold is calibrated by stepping DOWN from hsv_v_start in fine
    steps of 0.01 until the false-positive rate (mask pixels not in reference)
    drops below hsv_err_tol, or the floor hsv_v_min is reached.
    Using FP-rate (not symmetric error) prevents the loop from overshooting
    into undercounting when it steps one too far.

BLOCK 3 · MORPHOLOGICAL CLEANUP
    1. Area open (min_object_area)         remove noise specks
    2. Extent filter (hsv_extent_min)      drop thin/filamentary artefacts
    3. Binary opening (disk_small)         smooth colony outlines
    4. Hole fill                           close interior gaps
    5. 2× upsample → open(disk_large) → 2× downsample
       (upsampling before the large opening avoids aliasing on small colonies)

BLOCK 4 · CANDIDATE FILTERING  (same thresholds as countCFUAPP)
    Label connected components, then split into:
      CLUSTERS — large, eccentric or boxy blobs (likely touching colonies)
          "small" path  →  area > Q75  AND  ecc > 0.65  AND  extent > 0.20
          "large" path  →  ecc > 0.99  AND  extent > 0.20
      COLONIES — remaining objects with circularity > 0.075
    This exactly mirrors the bwpropfilt chain in
    count_colonies_with_instances() from the original file.

BLOCK 5 · WATERSHED ON CLUSTERS
    For each cluster blob: distance transform + peak_local_max + watershed
    to estimate how many touching colonies it contains.
    Conservative parameters (min_distance, thresh_abs_frac) match the
    original conservative_watershed().

BLOCK 6 · COUNT + ROI
    Each isolated colony  → NumOfCFU = 1
    Each cluster          → NumOfCFU = watershed result
    ROI polygons/centroids built identically to the original.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hsv, rgb2gray
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
    h_maxima,
    h_minima,
    remove_small_objects,
)
from skimage.segmentation import watershed


# Preprocessing cache: keyed by MD5(image bytes) + blackhat flag.
# preprocess_expt is the slowest step (Perona-Malik + Laplacian pyramid) and
# its output depends only on the image content, not on TuningParams.  The tuner
# calls predict_tuning_features hundreds of times on the same images with
# different params, so caching the preprocessed result gives a large speedup.
_PREPROCESS_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
_PREPROCESS_CACHE_MAXSIZE = 128


@dataclass
class TuningParams:
    hsv_v_start: float = 0.3
    hsv_v_step: float = 0.05
    hsv_v_min: float = 0.35
    hsv_s_min: float = 0.00
    hsv_err_tol: float = 0.0061
    h_ref_frac: float = 0.12   # fraction of p10-p90 gray range used as imextendedmin h in reference
    hsv_extent_min: float = 0.23
    hsv_open_disk_small: int = 1
    hsv_open_disk_large: int = 16
    hsv_close_disk_large: int = 10  # per-blob closing radius for large blobs (area >= 8000)
    min_object_area: int = 90
    small_area_quantile: float = 0.75
    small_cluster_extent_min: float = 0.20
    nonsmall_cluster_ecc_min: float = 0.99
    nonsmall_cluster_extent_min: float = 0.20
    colony_circularity_min: float = 0.09
    ws_thresh_abs_frac: float = 0.02
    ws_gauss_sigma: float = 0.3
    uncountable_cutoff: int = 300
    uncountable_precheck_cutoff: int = 400


CLASS_LARGE_THRESH = 0.0043
CLASS_MEDIUM_THRESH = 0.0032
CLASS_HIGH_GRAD_LARGE_MULT = 15.0
CLASS_HIGH_GRAD_LARGE_ABS_MIN = 170_000.0


# ---------------------------------------------------------------------------
# Pre-set parameter profiles
# ---------------------------------------------------------------------------
# Default TuningParams() is tuned for small colonies (E. coli-scale, many per dish).
# Large-CFU mode handles fungi, slow growers, or low-dilution plates where
# colonies are large (radius 40–120 px), sparse, and their halos fool the
# adaptive-V calibration loop into over-tightening.
#
# Key differences vs. default:
#   hsv_err_tol  : 0.0061 → 0.04  — large colony halos look like "FP" to the
#                                    loop; must be far more lenient or V is
#                                    tightened until almost nothing is kept.
#   h_ref_frac   : 0.12 → 0.08    — smaller h → imextendedmin captures more of
#                                    each colony → fewer halo pixels counted as FP.
#   hsv_v_start  : 0.3 → 0.55     — large colonies may be bright-ringed; start
#                                    more liberal so the loop has room to work.
#   hsv_v_min    : 0.35 → 0.20    — allow V to fall further if needed.
#   min_object_area: 150 → 1500   — suppress agar noise while keeping large colonies.
#   uncountable_*: lowered         — 30 large colonies can already be TNTC.

LARGE_CFU_PARAMS = TuningParams(
hsv_v_start=0.635,
hsv_v_step=0.05,
hsv_v_min=0.35,
hsv_s_min=0.00,
hsv_err_tol=0.02,          # large colonies on dark agar have ~22% FP floor;
                            # tighter values can't converge and prevent detection
h_ref_frac=0.10,
hsv_extent_min=0.23,
hsv_open_disk_small=3,
hsv_open_disk_large=30,    # larger opening suppresses agar-texture noise better
hsv_close_disk_large=15,   # larger closing bridges wider holes inside large CFUs
min_object_area=90,       # raise floor to filter small agar-noise blobs
small_area_quantile=0.75,
small_cluster_extent_min=0.20,
nonsmall_cluster_ecc_min=0.99,
nonsmall_cluster_extent_min=0.20,
colony_circularity_min=0.09,
ws_thresh_abs_frac=0.05,
ws_gauss_sigma=1.0,
uncountable_cutoff=300,
uncountable_precheck_cutoff=400,
)



@dataclass
class ROI:
    Position: np.ndarray
    Center: np.ndarray
    Creator: str
    Shape: str
    NumOfCFU: int


def save_tiff_rgb(path: Union[str, Path], rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb is None or not isinstance(rgb, np.ndarray) or rgb.size == 0:
        raise ValueError(f"save_tiff_rgb: empty image for {path}")
    img = rgb
    if img.dtype != np.uint8:
        img = _to_uint8(_as_float01(img))
    if img.ndim == 2:
        ok = cv2.imwrite(str(path), img)
        if not ok:
            raise RuntimeError(f"Failed to write TIFF: {path}")
        return
    if img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise RuntimeError(f"Failed to write TIFF: {path}")
        return
    raise ValueError(f"save_tiff_rgb: unsupported shape {img.shape} for {path}")


def _as_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _gray_uint8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        if img.dtype != np.uint8:
            return _to_uint8(_as_float01(img))
        return img
    g = rgb2gray(_as_float01(img))
    return _to_uint8(g)


@lru_cache(maxsize=32)
def _disk_kernel(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


@lru_cache(maxsize=32)
def _disk(r: int) -> np.ndarray:
    return disk(r)


def imoverlay(rgb: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
    if rgb.ndim == 2:
        out = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    else:
        out = rgb.copy()
    m = mask.astype(bool)
    color = color.lower()
    if color == "red":
        out[m, 0] = 0
        out[m, 1] = 255
        out[m, 2] = 0
    elif color == "green":
        out[m, 0] = 0
        out[m, 1] = 255
        out[m, 2] = 0
    elif color == "blue":
        out[m, 0] = 0
        out[m, 1] = 0
        out[m, 2] = 255
    else:
        raise ValueError("color must be 'red', 'green', or 'blue'")
    return out


def bwpropfilt(bw: np.ndarray, prop: str, rng: Tuple[float, float]) -> np.ndarray:
    bw = bw.astype(bool)
    lab = label(bw, connectivity=2)
    keep = np.zeros_like(bw, dtype=bool)
    rmin, rmax = rng
    prop_key = prop.lower()
    for reg in regionprops(lab):
        area = float(reg.area)
        if prop_key == "area":
            val = area
        elif prop_key == "extent":
            minr, minc, maxr, maxc = reg.bbox
            bbox_area = float((maxr - minr) * (maxc - minc))
            val = area / bbox_area if bbox_area > 0 else 0.0
        elif prop_key == "eccentricity":
            val = float(reg.eccentricity)
        elif prop_key == "circularity":
            per = float(reg.perimeter) if reg.perimeter and reg.perimeter > 0 else 0.0
            val = (4.0 * np.pi * area / (per * per)) if per > 0 else 0.0
        else:
            raise ValueError(f"Unsupported property: {prop}")
        if (val >= rmin) and (val <= rmax):
            keep[lab == reg.label] = True
    return keep


def bwareaopen(bw: np.ndarray, min_size: int) -> np.ndarray:
    return remove_small_objects(bw.astype(bool), min_size=min_size, connectivity=2)


def imextendedmin(gray_u8: np.ndarray, h: int) -> np.ndarray:
    return h_minima(gray_u8, h=h).astype(bool)


def imadjust_approx(
    img: np.ndarray,
    low_in=(0.2, 0.3, 0.0),
    high_in=(0.6, 0.7, 1.0),
    gamma=1.0,
) -> np.ndarray:
    img = np.asarray(img)
    orig_dtype = img.dtype
    """
    Map RGB input to:
    RED -> 0.2, 0.6 -> 0,1
    GREEN -> 0.3, 0.7 -> 0,1
    BLUE -> 0.0, 1.0 -> 0,1 (no change)
    Gamma: 1 -> Responsible for nonlinear mapping. >1 darkens, <1 brightens.
    """
    # --- Convert to float [0,1] like MATLAB ---
    if orig_dtype == np.uint8:
        img01 = img.astype(np.float32) / 255.0
        scale_back = 255.0
    elif orig_dtype == np.uint16:
        img01 = img.astype(np.float32) / 65535.0
        scale_back = 65535.0
    elif np.issubdtype(orig_dtype, np.floating):
        img01 = img.astype(np.float32)
        scale_back = None
    else:
        raise TypeError(f"Unsupported dtype: {orig_dtype}")

    # --- Apply mapping ---
    if img01.ndim == 2:
        low = np.array(low_in[0], dtype=np.float32)
        high = np.array(high_in[0], dtype=np.float32)

        out = (img01 - low) / max(high - low, 1e-6)
        out = np.clip(out, 0.0, 1.0) ** gamma

    else:
        low = np.array(low_in, dtype=np.float32).reshape(1, 1, -1)
        high = np.array(high_in, dtype=np.float32).reshape(1, 1, -1)

        out = (img01 - low) / np.maximum(high - low, 1e-6)
        out = np.clip(out, 0.0, 1.0) ** gamma

    # --- Convert back like MATLAB ---
    if scale_back is None:
        return out.astype(orig_dtype)
    else:
        return np.clip(out * scale_back + 0.5, 0, scale_back).astype(orig_dtype)

def remove_background_rgb(
    img_rgb: np.ndarray,
    sigma: float = 80.0,
    output_mean: float = 140.0,
) -> np.ndarray:
    """
    Normalize the slowly-varying agar background so all dishes arrive at the
    subsequent pipeline with a uniform mid-gray background, regardless of the
    original agar color (yellow, pink, dark-red, white, …).

    Method (normalized Gaussian background estimation + per-channel division):
      1. For each color channel estimate the background B with a very large
         Gaussian (sigma >> colony radius).  The Gaussian is weighted by the
         valid-pixel mask (normalized convolution) so the zero-padded exterior
         does not pull the background estimate low near the dish edge.
      2. Divide: ratio = channel / B.  Ratio ≈ 1 on bare agar, < 1 where
         colonies absorb more light — independent of original agar color.
      3. Rescale by output_mean so the background lands at a fixed gray level
         (140) compatible with the downstream imadjust_approx fixed-range map.
      4. Restore the original zero mask (outside-dish pixels stay black).

    Parameters
    ----------
    img_rgb     : uint8 RGB, zero-padded outside the dish mask.
    sigma       : Gaussian sigma for background estimation.  80 px is safe for
                  colonies with radius ≤ ~40 px at typical dish sizes.
    output_mean : target background level in the returned uint8 image (0-255).
                  140 falls within imadjust_approx's expected input range.
    """
    valid = np.any(img_rgb > 0, axis=2).astype(np.float32)
    if float(valid.sum()) < 100:
        return img_rgb  # empty or near-empty crop — nothing to do

    # Use an eroded mask for background estimation only.
    # The dish inner rim casts a dark shadow ring ~20-30px wide inside the dish
    # boundary.  If rim pixels participate in the normalized-convolution background
    # estimate, the Gaussian spreads their low values inward, making the
    # background estimate too dark near the rim → ch/bg blows up → dark rim
    # ring appears as a large false-positive colony blob.
    # Eroding by 15px excludes the rim shadow zone from the bg numerator/
    # denominator while still computing a correct background value AT those
    # pixels (extrapolated from the brighter interior agar via the Gaussian).
    rim_margin = 15
    rim_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_margin + 1, 2 * rim_margin + 1))
    inner_valid = cv2.erode(valid.astype(np.uint8), rim_se, iterations=1).astype(np.float32)

    result = np.zeros_like(img_rgb)
    for c in range(3):
        ch = img_rgb[:, :, c].astype(np.float32)
        # Normalized convolution over the INNER valid region only, so the dark
        # rim shadow does not contaminate the background estimate.
        num = cv2.GaussianBlur(ch * inner_valid, (0, 0), sigma)
        den = cv2.GaussianBlur(inner_valid,       (0, 0), sigma) + 1e-6
        bg  = num / den  # background estimate in [0, 255] float

        # Guard against very dark background patches that cause blow-up.
        bg_safe = np.maximum(bg, 8.0)

        # Divide and rescale: background → output_mean, colonies stay darker.
        scaled = (ch / bg_safe) * output_mean
        result[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)

    # Restore dish mask: outside pixels must stay zero for downstream checks.
    result[valid < 0.5] = 0
    return result


def _mean_region_area(bw: np.ndarray) -> float:
    lab = label(bw, connectivity=2)
    counts = np.bincount(lab.ravel())
    if counts.size <= 1:
        return 0.0
    return float(np.mean(counts[1:]))


def _filter_rim_blobs(
    bw: np.ndarray,
    border_px: np.ndarray,
    n_border: int,
) -> np.ndarray:
    if n_border <= 0 or not np.any(bw):
        return bw
    lab_rim = label(bw, connectivity=2)
    max_label = int(lab_rim.max())
    if max_label == 0:
        return bw
    border_hits = np.bincount(lab_rim[border_px], minlength=max_label + 1)
    remove_labels = np.flatnonzero(border_hits[1:] / float(n_border) >= 0.20) + 1
    if remove_labels.size == 0:
        return bw
    bw = bw.copy()
    bw[np.isin(lab_rim, remove_labels)] = False
    return bw


def _crop_to_nonzero_bbox(rgb: np.ndarray, pad: int = 2) -> np.ndarray:
    if rgb.size == 0:
        return rgb
    m = np.any(rgb != 0, axis=2) if rgb.ndim == 3 else (rgb != 0)
    ys, xs = np.where(m)
    if ys.size == 0 or xs.size == 0:
        return rgb
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(rgb.shape[0], y1 + pad)
    x1 = min(rgb.shape[1], x1 + pad)
    return rgb[y0:y1, x0:x1].copy()


def derive_experiment_ids_for_name(name: Union[str, Path]) -> Tuple[str, Optional[str]]:
    """
    Return output IDs for a possible top/bottom pair name.

    Pair example:
      DIFIP1_1229Edilu3_1230Edilu3.tif
      -> ("DIFIP1_1229Edilu3", "DIFIP1_1230Edilu3")

    Single-name inputs return no bottom ID; if a false split happens later, the
    lower half is ignored instead of being saved as a misleading *_Bottom file.
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 3 and re.search(r"dilu\d+", parts[-2], re.IGNORECASE) and re.search(r"dilu\d+", parts[-1], re.IGNORECASE):
        prefix = "_".join(parts[:-2])
        return f"{prefix}_{parts[-2]}", f"{prefix}_{parts[-1]}"
    return stem, None


def _looks_pre_cropped(rgb: np.ndarray) -> bool:
    if rgb.size == 0:
        return False
    m = np.any(rgb != 0, axis=2) if rgb.ndim == 3 else (rgb != 0)
    h, w = m.shape
    if h < 16 or w < 16:
        return False

    nz_frac = float(np.count_nonzero(m)) / float(m.size)
    if nz_frac <= 0.02 or nz_frac >= 0.995:
        return False

    bw = max(4, int(round(min(h, w) * 0.02)))
    border = np.zeros_like(m, dtype=bool)
    border[:bw, :] = True
    border[-bw:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True

    border_zero_frac = float(np.count_nonzero(~m[border])) / float(np.count_nonzero(border))
    edge_touch_fracs = [
        float(np.mean(m[:bw, :])),
        float(np.mean(m[-bw:, :])),
        float(np.mean(m[:, :bw])),
        float(np.mean(m[:, -bw:])),
    ]
    max_edge_touch = max(edge_touch_fracs)

    return border_zero_frac >= 0.25 and max_edge_touch >= 0.10

def _get_expts_from_detectdish(
    rgb: np.ndarray,
    dish_mode: str,
    target_area_px2: float,
) -> List[Dict[str, np.ndarray]]:
    if dish_mode == "pre_cropped":
        return [{"expt": _crop_to_nonzero_bbox(rgb)}]
    if dish_mode == "auto" and _looks_pre_cropped(rgb):
        return [{"expt": _crop_to_nonzero_bbox(rgb)}]
    import DetectDish
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fixed_diameter_px = 2.0 * float(np.sqrt(float(target_area_px2) / np.pi))
    masked_bgr, ellipse_final, brightness, _ = DetectDish.detect_plate_rgb(
        bgr, fixed_diameter_px=fixed_diameter_px
    )
    if ellipse_final is None or masked_bgr is None:
        return [{"expt": rgb}]
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    if dish_mode == "single":
        return [{"expt": _crop_to_nonzero_bbox(masked_rgb)}]

    # --- Attempt to find a divider bar (half-dish split) ---
    split_result = None
    line_candidate = DetectDish.find_divider_line_dark(brightness, ellipse_final)
    if line_candidate is not None:
        p1, p2, line_score = line_candidate
        if dish_mode == "double" or line_score >= DetectDish.BAR_LINE_SCORE_THRESH_AUTO:
            split_result = DetectDish.mask_top_bottom_from_line(
                image_bgr=bgr,
                ellipse=ellipse_final,
                p1=p1,
                p2=p2,
                gap=DetectDish.BAR_SPLIT_MARGIN_PX,
                bar_half_width=DetectDish.BAR_MAX_THICK_PX // 2,
                brightness=brightness,
            )

    bar_extents = None
    if split_result is None:
        bar_extents = DetectDish.find_divider_y(
            brightness,
            ellipse_final,
            bar_q=DetectDish.BAR_Q,
            frac_thresh=DetectDish.BAR_FRAC_THRESH,
            min_thick=DetectDish.BAR_MIN_THICK_PX,
            max_thick=DetectDish.BAR_MAX_THICK_PX,
        )

    if split_result is None and bar_extents is None:
        return [{"expt": _crop_to_nonzero_bbox(masked_rgb)}]

    if split_result is None:
        y0_bar, y1_bar = bar_extents
        split_result = DetectDish.mask_top_bottom(
            image_bgr=bgr,
            ellipse=ellipse_final,
            y_bar_top=y0_bar,
            y_bar_bot=y1_bar,
            margin=DetectDish.BAR_SPLIT_MARGIN_PX,
            brightness=brightness,
        )

    top_bgr, bot_bgr = split_result
    top_rgb = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
    bot_rgb = cv2.cvtColor(bot_bgr, cv2.COLOR_BGR2RGB)

    top_crop = _crop_to_nonzero_bbox(top_rgb)
    bot_crop = _crop_to_nonzero_bbox(bot_rgb)

    # ------------------------------------------------------------------
    # Validate the split: each half must own a meaningful fraction of the
    # total dish area.  A genuine half-dish produces two halves of roughly
    # equal size (each ≈ 35–50 %).  A false-positive divider on a full
    # dish yields one tiny sliver and one nearly-full region.
    #
    # If either half is below the threshold, discard the split and return
    # the whole dish as a single experiment.
    # ------------------------------------------------------------------
    _MIN_HALF_FRAC = 0.20  # each half must be ≥ 20 % of total dish

    total_dish_nz = float(np.count_nonzero(np.any(masked_rgb > 0, axis=2)))
    if total_dish_nz > 0:
        top_nz = float(np.count_nonzero(np.any(top_crop > 0, axis=2)))
        bot_nz = float(np.count_nonzero(np.any(bot_crop > 0, axis=2)))
        top_frac = top_nz / total_dish_nz
        bot_frac = bot_nz / total_dish_nz
        if top_frac < _MIN_HALF_FRAC or bot_frac < _MIN_HALF_FRAC:
            # Split is lopsided → false divider on a full dish
            return [{"expt": _crop_to_nonzero_bbox(masked_rgb)}]

    return [{"expt": top_crop}, {"expt": bot_crop}]

# ===========================================================================
# BLOCK 1 — Preprocessing
# ===========================================================================

def preprocess_expt(
    expt: np.ndarray,
    use_blackhat: bool = False,
    blackhat_disk_r: int = 15,
    bg_sigma: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplified preprocessing: background normalisation → per-channel contrast
    stretch → CLAHE + bilateral filter for the gray reference.

    Parameters
    ----------
    expt            : RGB uint8 image (cropped dish / half-dish)
    use_blackhat    : if True, add a blackhat pass to the gray reference so
                      faint colony valleys are deepened before HSV calibration
    blackhat_disk_r : structuring element radius for the blackhat close op
    bg_sigma        : Gaussian sigma for remove_background_rgb.  80 px (default)
                      is safe for colonies with radius ≤ 40 px.  Use 150 for
                      large-CFU mode where colony radius can reach 60–120 px.

    Returns
    -------
    expt_adj  : preprocessed RGB uint8 — input to HSV masking (Block 2)
    gray_expt : grayscale uint8 — input to imextendedmin reference + size guess
    """
    _cache_key = (id(expt), expt.shape, expt.strides, use_blackhat, blackhat_disk_r, bg_sigma)
    _cached = _PREPROCESS_CACHE.get(_cache_key)
    if _cached is not None:
        return _cached

    # --- Step 1: background normalisation ---
    # Divides each channel by a large-Gaussian background estimate so all agar
    # colours (yellow, pink, dark-red, white) arrive at a uniform ~140 DN.
    # This makes the fixed imadjust_approx clip ranges consistent across plates.
    expt_bg = remove_background_rgb(expt, sigma=bg_sigma)

    # --- Step 2: per-channel contrast stretch ---
    # Maps agar (~140 DN ≈ 0.55 float) to near-white and clips colony pixels
    # to near-black.  After this step V_agar ≈ 0.87 and V_colony ≈ 0.0,
    # giving the HSV adaptive loop the widest separation to calibrate against.
    expt_adj = imadjust_approx(expt_bg)

    # --- Step 3: gray reference for HSV calibration ---
    # Convert the contrast-stretched image to grayscale, then apply:
    #   CLAHE  — adaptive histogram equalisation normalises local dynamic range
    #             so imextendedmin h values are consistent across plates with
    #             different agar textures or sparse vs dense colony coverage.
    #   Bilateral filter — edge-preserving smoothing flattens uniform agar
    #             regions while keeping colony-to-agar boundaries sharp,
    #             replacing Perona-Malik diffusion (20 iters) + Laplacian
    #             pyramid + two localcontrast passes + imsharpen.
    gray = cv2.cvtColor(expt_adj, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)
    gray_expt = cv2.bilateralFilter(gray_clahe, d=9, sigmaColor=25, sigmaSpace=25)

    # --- Optional blackhat ---
    if use_blackhat:
        se = _disk_kernel(blackhat_disk_r)
        blackhat = cv2.morphologyEx(gray_expt, cv2.MORPH_BLACKHAT, se)
        gray_expt = cv2.add(gray_expt, blackhat)

    _result = (expt_adj, gray_expt)
    if len(_PREPROCESS_CACHE) < _PREPROCESS_CACHE_MAXSIZE:
        _PREPROCESS_CACHE[_cache_key] = _result
    return _result


# ===========================================================================
# BLOCK 2 — HSV adaptive masking
# ===========================================================================


def hsv_mask_adaptive(
    expt_adj: np.ndarray,
    gray_expt: np.ndarray,
    params: TuningParams,
) -> np.ndarray:
    """
    Adapt the V threshold so the HSV mask agrees with the extended-minima
    reference derived from the preprocessed grayscale.

    Logic (verbatim from hsv_filter() in countCFUAPP, with configurable step):
      reference = imextendedmin(gray, h_ref) & (gray != 0)
      start at v_thresh = hsv_v_start
      while FP-rate > hsv_err_tol:
          v_thresh -= hsv_v_step   (default 0.01; APP1 used 0.05 — finer = more precise)
          if v_thresh < hsv_v_min: break

    Returns a raw boolean candidate mask (before morphological cleanup).
    """
    # Reference: dark blobs from extended minima, h adaptive to plate contrast.
    # Formula: 12% of the p10-p90 range, clamped to [10, 35].
    # This is strictly better than APP1's fixed h=35: low-contrast plates get
    # a smaller h (more sensitive minima), high-contrast plates get h up to 35.
    nz_pixels = gray_expt[gray_expt != 0]
    h_ref = int(np.clip(float(params.h_ref_frac) * (np.percentile(nz_pixels, 90) - np.percentile(nz_pixels, 10)), 10, 35))
    bw_ref = imextendedmin(gray_expt, h_ref) & (gray_expt != 0)

    dish_size = float(np.count_nonzero(gray_expt != 0))

    # Pre-compute HSV channels once — the loop only changes v_thresh.
    _hsv = rgb2hsv(_as_float01(expt_adj))
    _S = _hsv[..., 1]
    _V = _hsv[..., 2]
    _dish_nz = gray_expt != 0
    _s_min = float(params.hsv_s_min)

    v_thresh = float(params.hsv_v_start)
    mask = (_V < v_thresh) & (_S > _s_min) & _dish_nz

    # FP rate: fraction of dish pixels that are in mask but NOT in reference.
    # Unlike APP1's symmetric |mask - ref| / image_size (which counts FN too),
    # this only counts over-prediction (agar leakage). The loop can never be
    # tricked by missed colonies (FN) into stepping too far → prevents undercounting.
    fp_rate = float(np.count_nonzero(mask & ~bw_ref)) / dish_size if dish_size > 0 else 0.0

    while fp_rate > float(params.hsv_err_tol):
        v_thresh -= float(params.hsv_v_step)
        if v_thresh < float(params.hsv_v_min):
            break
        mask = (_V < v_thresh) & (_S > _s_min) & _dish_nz
        fp_rate = float(np.count_nonzero(mask & ~bw_ref)) / dish_size if dish_size > 0 else 0.0

    return mask


# ===========================================================================
# BLOCK 3 — Morphological cleanup
# ===========================================================================

def morpho_cleanup(bw: np.ndarray, params: TuningParams, dish_area: int = 0) -> np.ndarray:
    """
    Clean up the raw HSV candidate mask:

      1. Area open           — kill noise specks smaller than min_object_area
      2. Extent filter       — remove thin / filamentary artefacts
      3. Binary opening      — smooth colony silhouettes (small disk)
      4. Hole fill           — close interior holes (ring-shaped colonies)
      5. 2× upsample → large opening → 2× downsample
         Upsampling before the large morphological opening prevents small
         colonies from being erased by the kernel footprint.

    Returns a clean boolean mask ready for candidate filtering.
    """
    # 1 — remove tiny blobs
    bw = bwareaopen(bw, params.min_object_area)

    # 2 — smooth outlines with a light opening.
    # On sparse plates (< 8 % of dish covered) small colony edges appear as
    # thin arcs (4-6 px thick).  disk(3) opening destroys them before the gap-
    # repair in step 3 can save them.  Use disk(1) for sparse plates so those
    # arcs survive; keep disk(hsv_open_disk_small) for dense plates where real
    # thin-bridge noise between nearby colonies must be cut.
    _cov_pre = float(np.count_nonzero(bw)) / dish_area if dish_area > 0 else 1.0
    _open_r = 1 if _cov_pre < 0.08 else int(params.hsv_open_disk_small)
    bw = binary_opening(bw, footprint=_disk(_open_r))

    # 3 — close ring / arc gaps per blob, then fill enclosed holes.
    # The HSV mask often captures only the dark rim of a colony, leaving
    # C-shapes, arcs, and crescents.  binary_fill_holes requires a fully
    # enclosed hole, so open arcs are not filled.
    # Strategy: for each blob independently dilate → fill → erode back.
    # Adaptive kernel radius: small blobs are likely thin arc remnants with
    # wide gaps (20-30 px); they need a larger disk to bridge the gap.
    # Large blobs (area >= 8000) may have internal holes with thin channels
    # to the background; hsv_close_disk_large (default 10, 15 for large-CFU
    # mode) must be wide enough to seal those channels so fill_holes works.
    # Per-blob processing prevents merging of neighbouring colonies.
    bw_labeled, n_blobs = ndi.label(bw)
    bw_repaired = np.zeros_like(bw)
    blob_areas = np.bincount(bw_labeled.ravel())
    H, W = bw.shape
    blob_slices = ndi.find_objects(bw_labeled)  # O(1) bounding boxes, no full-image scan
    for i, sl in enumerate(blob_slices):
        if sl is None:
            continue
        label_i = i + 1
        area = int(blob_areas[label_i])
        r = 12 if area < 1500 else (6 if area < 8000 else int(params.hsv_close_disk_large))
        # Pad the bounding box by r so dilation/erosion have enough room.
        # Operating on the padded crop instead of the full image reduces cost
        # from O(H×W) to O((blob_bbox + 2r)²) — ~100–1000× faster per blob.
        r0 = max(0, sl[0].start - r);  r1 = min(H, sl[0].stop + r)
        c0 = max(0, sl[1].start - r);  c1 = min(W, sl[1].stop + r)
        blob_crop = bw_labeled[r0:r1, c0:c1] == label_i
        _gap_se = _disk(r)
        grown = binary_dilation(blob_crop, footprint=_gap_se)
        filled = ndi.binary_fill_holes(grown)
        shrunk = binary_erosion(filled, footprint=_gap_se)
        bw_repaired[r0:r1, c0:c1] |= blob_crop | shrunk
    bw = bw_repaired

    # 4 — fill interior holes so large dark-core colonies are treated as
    # compact blobs rather than donuts.
    bw = ndi.binary_fill_holes(bw)

    # 5 — keep only objects with reasonable extent (area / bbox area) AFTER
    # the repair above. This preserves genuine colonies that were holey in the
    # raw HSV mask while still removing filamentary scratches/smears.
    bw = bwpropfilt(bw, "Extent", (float(params.hsv_extent_min), 1.0))

    # 6 — upsample → large open → downsample  (avoids aliasing on small colonies)
    # Skip on sparse plates (coverage ≤ 10% of dish): small colonies are precious
    # and agar noise is minimal when the HSV calibration ran on a sparse reference.
    # Dense/noisy plates need this step to suppress medium-sized agar patches that
    # survived steps 1–5.
    coverage = float(np.count_nonzero(bw)) / dish_area if dish_area > 0 else 1.0
    if coverage > 0.10:
        h, w = bw.shape
        big = cv2.resize(bw.astype(np.uint8), None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_NEAREST) > 0
        big = binary_opening(big, footprint=_disk(int(params.hsv_open_disk_large)))
        bw = cv2.resize(big.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST) > 0

    # 7 — the large-scale smoothing can re-open tiny holes / islands after
    # resampling; do a final conservative repair and speck cleanup.
    bw = ndi.binary_fill_holes(bw)
    bw = bwareaopen(bw, params.min_object_area)


    return bw


# ===========================================================================
# BLOCK 4+5+6 — Candidate filtering, watershed, count/ROI
# ===========================================================================

def _ws_single_pass(
    region_mask: np.ndarray,
    h_frac: float,
    min_area: int,
) -> Tuple[np.ndarray, int]:
    """
    One h_maxima watershed pass. Returns (label_image, n_valid_segments).
    Does NOT produce a boundary map — call this only for counting/refinement.
    """
    dist = ndi.distance_transform_edt(region_mask).astype(np.float32)
    dist_s = ndi.gaussian_filter(dist, sigma=1.0)
    dmax = float(dist_s.max())
    if dmax < 3.0:
        return region_mask.astype(np.int32), 1

    h_val = max(1.0, h_frac * dmax)
    seed_mask = h_maxima(dist_s, h=h_val) & region_mask
    markers, n_seeds = ndi.label(seed_mask)
    if n_seeds <= 1:
        return region_mask.astype(np.int32), 1

    L = watershed(-dist_s, markers=markers, mask=region_mask).astype(np.int32)
    label_sizes = np.bincount(L.ravel())
    valid = np.flatnonzero(label_sizes[1:] >= min_area) + 1
    if len(valid) <= 1:
        return region_mask.astype(np.int32), 1

    new_L = np.zeros_like(L)
    for new_i, old_lbl in enumerate(valid, start=1):
        new_L[L == old_lbl] = new_i
    return new_L, len(valid)


def _conservative_watershed(
    region_mask: np.ndarray,
    params: TuningParams,
    ref_area: Optional[float] = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Segment a blob into individual colonies using h_maxima on the distance transform.

    h_maxima finds peaks with prominence >= h above every surrounding valley,
    so a single colony always yields exactly one seed regardless of boundary
    roughness, while two touching colonies produce two seeds.

    If ref_area is provided, a second refinement pass (at h_frac × 0.6) is run
    on any first-pass segment that is still much larger than one colony (> 1.8×
    ref_area) but not a truly uncountable mass (< 15× ref_area).  This catches
    tightly-packed groups where the first-pass saddle wasn't deep enough.

    Returns (label_image, n_colonies, boundary_mask).
    """
    region_mask = region_mask.astype(bool)
    if not np.any(region_mask):
        return np.zeros_like(region_mask, dtype=np.int32), 0, np.zeros_like(region_mask, dtype=bool)

    dist = ndi.distance_transform_edt(region_mask).astype(np.float32)
    dist_s = ndi.gaussian_filter(dist, sigma=float(params.ws_gauss_sigma))

    dmax = float(dist_s.max())
    if dmax < 3.0:
        return region_mask.astype(np.int32), 1, np.zeros_like(region_mask, dtype=bool)

    h_val = max(1.0, float(params.ws_thresh_abs_frac) * dmax)
    seed_mask = h_maxima(dist_s, h=h_val) & region_mask
    markers, n_seeds = ndi.label(seed_mask)

    h_frac_2 = float(params.ws_thresh_abs_frac) * 0.6  # 0.05 × 0.6 = 0.03
    min_area = int(params.min_object_area)

    if n_seeds <= 1:
        # First pass found no split. Try lower h if blob is larger than one
        # colony but not a hopeless fused mass (no saddle exists).
        blob_area = int(np.count_nonzero(region_mask))
        if (ref_area is not None
                and blob_area > ref_area * 1.8
                and blob_area < ref_area * 15):
            sub_L, sub_n = _ws_single_pass(region_mask, h_frac_2, min_area)
            if sub_n > 1:
                mx = ndi.maximum_filter(sub_L, size=3)
                mn = ndi.minimum_filter(sub_L, size=3)
                return sub_L, sub_n, region_mask & (mx != mn)
        return region_mask.astype(np.int32), 1, np.zeros_like(region_mask, dtype=bool)

    # First pass found multiple seeds — watershed into segments.
    L = watershed(-dist_s, markers=markers, mask=region_mask).astype(np.int32)
    label_sizes = np.bincount(L.ravel())
    valid = np.flatnonzero(label_sizes[1:] >= min_area) + 1

    if len(valid) <= 1:
        return region_mask.astype(np.int32), 1, np.zeros_like(region_mask, dtype=bool)

    # Refinement: for each segment still larger than ~2 colonies, try lower h.
    # This handles tightly-packed sub-clusters where the first-pass saddle
    # wasn't deep enough to reveal all the peaks.
    final_L = np.zeros_like(L, dtype=np.int32)
    next_lbl = 1
    total_count = 0
    for old_lbl in valid:
        seg = (L == old_lbl)
        seg_area = int(np.count_nonzero(seg))
        refined = False
        if (ref_area is not None
                and seg_area > ref_area * 1.8
                and seg_area < ref_area * 15):
            sub_L, sub_n = _ws_single_pass(seg, h_frac_2, min_area)
            if sub_n > 1:
                for sub_lbl in range(1, sub_n + 1):
                    sub_mask = (sub_L == sub_lbl) & seg
                    if np.any(sub_mask):
                        final_L[sub_mask] = next_lbl
                        next_lbl += 1
                total_count += sub_n
                refined = True
        if not refined:
            final_L[seg] = next_lbl
            next_lbl += 1
            total_count += 1

    if total_count <= 1:
        return region_mask.astype(np.int32), 1, np.zeros_like(region_mask, dtype=bool)

    mx = ndi.maximum_filter(final_L, size=3)
    mn = ndi.minimum_filter(final_L, size=3)
    return final_L, total_count, region_mask & (mx != mn)


def _recover_second_pass_blobs(
    bw_raw: np.ndarray,
    bw_clean: np.ndarray,
    gray_expt: np.ndarray,
    dish_area: int,
) -> np.ndarray:
    bw_clean_cov = np.count_nonzero(bw_clean) / dish_area if dish_area > 0 else 0.0
    if dish_area <= 0 or bw_clean_cov > 0.10:
        return bw_clean

    lost = bw_raw & ~bw_clean
    if not np.any(lost):
        return bw_clean

    bw_soft = imextendedmin(gray_expt, 8) & (gray_expt != 0)
    already_covered = binary_dilation(bw_clean, footprint=_disk(10))
    uncovered = bw_soft & lost & ~already_covered
    if not np.any(uncovered):
        return bw_clean

    labeled_lost, _ = ndi.label(lost)
    keep2 = np.unique(labeled_lost[uncovered])
    keep2 = keep2[keep2 > 0]
    if keep2.size == 0:
        return bw_clean

    bw_extra = np.isin(labeled_lost, keep2)
    H, W = bw_clean.shape
    pad = 10
    bw_clean = bw_clean | bw_extra

    labeled_final, _ = ndi.label(bw_clean)
    affected = np.unique(labeled_final[bw_extra])
    affected = affected[affected > 0]
    for i in affected:
        region = labeled_final == i
        rows, cols = np.where(region)
        r0 = max(rows.min() - pad, 0)
        r1 = min(rows.max() + pad + 1, H)
        c0 = max(cols.min() - pad, 0)
        c1 = min(cols.max() + pad + 1, W)
        crop = region[r0:r1, c0:c1]
        bw_clean[r0:r1, c0:c1] |= ndi.binary_fill_holes(crop)

    return bw_clean


def _expand_mask_to_dark_colony_halos(
    bw_clean: np.ndarray,
    expt_rgb: np.ndarray,
    gray_expt: np.ndarray,
    dish_area: int,
    large_cfu_mode: bool = False,
) -> np.ndarray:
    """
    Grow accepted CFU cores into the surrounding dark colony footprint for
    the final segmentation. This must happen before watershed so the drawn
    green fill and yellow watershed boundaries describe the same object.
    """
    bw_clean = bw_clean.astype(bool)
    if not np.any(bw_clean):
        return bw_clean

    # The old file (countCFUAPP22) had no halo expansion at all and performed
    # better for large CFUs.  The original grow_r=48 over-inflates large colony
    # masks, causing watershed to mis-split them or merge neighbours.
    # Use a small, conservative grow_r (12px) so the expansion captures dark
    # colony rims without running away into neighbouring colonies or agar shadow.
    if large_cfu_mode:
        grow_r_override = 16

    dish_mask = np.any(expt_rgb > 0, axis=2)
    if not np.any(dish_mask):
        return bw_clean

    gray_raw = _gray_uint8(expt_rgb).astype(np.float32)
    valid_f = dish_mask.astype(np.float32)
    bg_num = cv2.GaussianBlur(gray_raw * valid_f, (0, 0), sigmaX=34.0, sigmaY=34.0)
    bg_den = cv2.GaussianBlur(valid_f, (0, 0), sigmaX=34.0, sigmaY=34.0) + 1e-6
    local_bg = bg_num / bg_den
    dark_contrast = local_bg - gray_raw

    vals = dark_contrast[dish_mask]
    if vals.size < 100:
        return bw_clean

    contrast_pct = 58 if large_cfu_mode else 66
    contrast_thr = float(np.clip(np.percentile(vals, contrast_pct), 4.0, 14.0))
    dark_candidate = dish_mask & (dark_contrast >= contrast_thr)

    nz = gray_expt[dish_mask]
    if nz.size >= 100:
        gray_med = float(np.median(nz))
        gray_pct = 50 if large_cfu_mode else 42
        gray_thr = float(np.percentile(nz, gray_pct))
        if gray_thr < gray_med - 5.0:
            dark_candidate |= dish_mask & (gray_expt.astype(np.float32) <= gray_thr)

    coverage = float(np.count_nonzero(bw_clean)) / dish_area if dish_area > 0 else 0.0
    if large_cfu_mode:
        grow_r = grow_r_override  # conservative fixed radius set above
    elif coverage > 0.15:
        grow_r = 4
    elif coverage > 0.10:
        grow_r = 8
    elif coverage > 0.08:
        grow_r = 16
    else:
        grow_r = 34
    local_neighborhood = binary_dilation(bw_clean, footprint=_disk(grow_r))
    allowed = dark_candidate & local_neighborhood

    # Do not let expansion walk into leftover divider-bar shadows. Those are
    # usually broad, flat, horizontal components near a crop edge; if they are
    # connected to a nearby colony seed, unconstrained geodesic growth makes an
    # unnatural green shelf and gives watershed a bad boundary.
    lab_allowed = label(allowed, connectivity=2)
    protected_seed_zone = binary_dilation(bw_clean, footprint=_disk(8))
    h, w = allowed.shape
    for reg in regionprops(lab_allowed):
        minr, minc, maxr, maxc = reg.bbox
        comp_h = int(maxr - minr)
        comp_w = int(maxc - minc)
        if comp_h <= 0 or comp_w <= 0:
            continue
        touches_horizontal_edge = minr <= 2 or maxr >= h - 2
        broad_band = comp_w >= int(0.18 * w) and comp_w >= 4 * comp_h
        edge_band = touches_horizontal_edge and comp_w >= int(0.06 * w) and comp_w >= 2 * comp_h
        if broad_band or edge_band:
            comp = lab_allowed == reg.label
            allowed[comp & ~protected_seed_zone] = False

    expanded = bw_clean.copy()
    for _ in range(max(1, grow_r)):
        nxt = binary_dilation(expanded, footprint=_disk(1)) & allowed
        nxt |= bw_clean
        if np.array_equal(nxt, expanded):
            break
        expanded = nxt

    expanded = ndi.binary_fill_holes(expanded)
    expanded &= dish_mask

    # Final safety cap: expansion is for boundary recovery, not discovering
    # new large structures. If a broad dark background/bar component is
    # connected to a seed, keep the pixels nearest the accepted core and
    # discard the runaway halo.
    lab_exp = label(expanded, connectivity=2)
    if int(lab_exp.max()) > 0:
        dist_from_seed = ndi.distance_transform_edt(~bw_clean)
        capped = np.zeros_like(expanded, dtype=bool)
        for reg in regionprops(lab_exp):
            comp = lab_exp == reg.label
            seed_area = int(np.count_nonzero(comp & bw_clean))
            if seed_area <= 0:
                continue
            comp_area = int(reg.area)
            if large_cfu_mode:
                max_area = int(max(seed_area * 16, seed_area + 8000))
            else:
                max_area = int(max(seed_area * 9, seed_area + 2500))
            if comp_area <= max_area:
                capped |= comp
                continue

            rr, cc = np.where(comp)
            d = dist_from_seed[rr, cc]
            keep_n = min(max_area, rr.size)
            keep_idx = np.argpartition(d, keep_n - 1)[:keep_n]
            capped[rr[keep_idx], cc[keep_idx]] = True
        expanded = capped
    return expanded


def _remove_edge_bar_artifacts(bw: np.ndarray) -> np.ndarray:
    """
    Remove only very bar-like components touching the top/bottom crop edge.
    This targets divider remnants in half dishes without filtering central
    elongated CFU clusters.
    """
    bw = bw.astype(bool)
    if not np.any(bw):
        return bw
    h, w = bw.shape
    edge_px = max(6, int(round(0.025 * min(h, w))))
    lab = label(bw, connectivity=2)
    out = bw.copy()
    for reg in regionprops(lab):
        minr, minc, maxr, maxc = reg.bbox
        comp_h = int(maxr - minr)
        comp_w = int(maxc - minc)
        if comp_h <= 0 or comp_w <= 0:
            continue
        touches_horizontal_edge = minr <= edge_px or maxr >= h - edge_px
        if not touches_horizontal_edge:
            continue
        aspect = comp_w / max(1.0, float(comp_h))
        very_wide_band = comp_w >= int(0.08 * w) and aspect >= 3.0
        long_thin = (
            float(reg.eccentricity) >= 0.985
            and comp_w >= int(0.05 * w)
            and float(reg.extent) <= 0.55
        )
        if very_wide_band or long_thin:
            out[lab == reg.label] = False
    return out


def _has_dark_object_probe(expt: np.ndarray, params: TuningParams) -> bool:
    gray = _gray_uint8(cv2.GaussianBlur(expt, (0, 0), 2))
    if not np.any(gray != 0):
        return False
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    probe = (gray < thr) & (gray != 0)
    probe = bwareaopen(probe, max(25, int(params.min_object_area)))
    probe = bwpropfilt(probe, "Extent", (0.15, 1.0))
    probe = _remove_edge_bar_artifacts(probe)
    return bool(np.any(probe))


def _identify_border_pixels(window_u8: np.ndarray) -> int:
    w = np.asarray(window_u8).ravel()
    if w.size == 0:
        return 0
    ctr = w[w.size // 2]
    if ctr == 0 or ctr >= 250:
        return 0
    return int(np.count_nonzero(w >= 250) > 1)


def _identify_high_bdry_pixels(window_u8: np.ndarray) -> int:
    w = np.asarray(window_u8).ravel()
    if w.size == 0:
        return 0
    ctr = w[w.size // 2]
    if ctr == 0:
        return 0
    return int(np.count_nonzero(w) > 2)


def _gradient_hist_size_guess(gray_expt: np.ndarray) -> Tuple[str, float, float, float]:
    """
    First size guess from APP1's blur + Laplacian histogram signal.

    Returns (class, max_hist_bin_count, medium_threshold, large_threshold).
    """
    if gray_expt is None or gray_expt.size == 0:
        return "small", 0.0, 0.0, 0.0

    gray_expt = _gray_uint8(gray_expt)
    blur = cv2.GaussianBlur(
        gray_expt, (3, 3),
        sigmaX=0.5, sigmaY=0.5,
        borderType=cv2.BORDER_REPLICATE,
    )
    im_grad = np.abs(cv2.Laplacian(
        blur, cv2.CV_32F, ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    ))

    border_grad = (im_grad >= 250).astype(np.uint8)
    border_grad2 = ndi.generic_filter(
        border_grad,
        function=_identify_high_bdry_pixels,
        size=(3, 3),
        mode="nearest",
    )
    im_grad_final = im_grad.astype(np.float32) * (1.0 - border_grad2.astype(np.float32))

    border_med = ndi.generic_filter(
        im_grad,
        function=_identify_border_pixels,
        size=(3, 3),
        mode="nearest",
    )
    im_grad_final = im_grad_final * (1.0 - border_med.astype(np.float32))

    vals = np.sort(im_grad_final.reshape(-1))
    vals = vals[vals > 5]
    if vals.size == 0:
        vals = np.array([6.0], dtype=np.float32)

    bins = np.arange(float(vals.min()), float(vals.max()) + 5.0, 5.0)
    hist, _ = np.histogram(vals, bins=bins)

    sz_total = float(gray_expt.shape[0] * gray_expt.shape[1])
    large_check = sz_total * CLASS_LARGE_THRESH
    medium_check = sz_total * CLASS_MEDIUM_THRESH
    max_val = float(hist.max()) if hist.size else 0.0
    high_grad_large_check = max(
        large_check * CLASS_HIGH_GRAD_LARGE_MULT,
        CLASS_HIGH_GRAD_LARGE_ABS_MIN,
    )

    if max_val < large_check or max_val >= high_grad_large_check:
        return "large", max_val, medium_check, large_check
    if medium_check < max_val < large_check:
        return "medium", max_val, medium_check, large_check
    return "small", max_val, medium_check, large_check


def _count_colonies_with_instances(
    binary_image: np.ndarray,
    expt_rgb: np.ndarray,
    params: TuningParams,
    *,
    build_overlay: bool = True,
    build_rois: bool = True,
    large_cfu_mode: bool = False,
) -> Tuple[int, np.ndarray, List[ROI], List[ROI], np.ndarray]:
    """
    Count colonies in a binary mask using per-blob watershed.

    Applies _conservative_watershed to every blob — h_maxima on the distance
    transform decides whether it is a single colony (1 seed) or a cluster
    (>1 seeds). No fragile eccentricity/extent pre-classification needed.

    large_cfu_mode: when True, the ref_area refinement pass is disabled.
    Large colonies at 1.8× ref_area are genuine single colonies, not clusters;
    the refinement sub-split produces the overcounting seen on large-CFU plates.
    """
    bw = binary_image.astype(bool)
    bw = bwareaopen(bw, params.min_object_area)

    if not np.any(bw):
        return 0, expt_rgb, [], [], np.zeros(bw.shape, dtype=np.int32)

    # Filter out clear non-colony artefacts, but keep plausible cluster blobs
    # even when they are not very circular. A global circularity gate is too
    # aggressive here because merged CFUs are exactly the objects we want the
    # watershed to inspect next.
    lab0 = label(bw, connectivity=2)
    regs0 = regionprops(lab0)
    if regs0:
        areas0 = np.array([r.area for r in regs0], dtype=float)
        area_q = float(np.quantile(areas0, float(params.small_area_quantile)))
        keep_labels: List[int] = []
        for reg in regs0:
            area = float(reg.area)
            perim = float(reg.perimeter)
            circularity = 0.0 if perim <= 0.0 else float(4.0 * np.pi * area / (perim ** 2))
            extent = float(reg.extent)
            ecc = float(reg.eccentricity)

            keep_as_colony = circularity >= float(params.colony_circularity_min)
            keep_as_big_cluster = (
                area >= area_q
                and extent >= float(params.small_cluster_extent_min)
            )
            keep_as_elongated_cluster = (
                ecc >= float(params.nonsmall_cluster_ecc_min)
                and extent >= float(params.nonsmall_cluster_extent_min)
            )

            if keep_as_colony or keep_as_big_cluster or keep_as_elongated_cluster:
                keep_labels.append(reg.label)

        bw = np.isin(lab0, keep_labels)
        bw = bwareaopen(bw, params.min_object_area)

    overlay = imoverlay(expt_rgb, bw, "green") if build_overlay else expt_rgb

    colony_roi: List[ROI] = []
    cluster_roi: List[ROI] = []
    num_cfu = 0
    instance_labels = np.zeros(bw.shape, dtype=np.int32)
    next_label = 1

    lab = label(bw, connectivity=2)
    regs = regionprops(lab)

    # Pre-scan: compute ref_area as the median area of blobs where the first
    # watershed pass returns exactly 1 seed AND area <= 1.5× median blob area.
    # This gives a reliable single-colony size reference for the refinement pass.
    all_areas = np.array([r.area for r in regs], dtype=float)
    if large_cfu_mode:
        # Skip the refinement pre-scan entirely for large-CFU mode.
        # The refinement pass fires on blobs > 1.8× ref_area and tries to
        # sub-split them — on large-CFU plates those blobs are often genuine
        # single large colonies, not clusters, so the sub-split overcounts.
        # Conservative first-pass watershed (ws_thresh_abs_frac=0.05, σ=1.0)
        # is sufficient; the second pass only makes things worse here.
        ref_area: Optional[float] = None
    elif all_areas.size > 0:
        overall_med = float(np.median(all_areas))
        # Cap at 50 candidates sorted by area — the smallest blobs are the most
        # reliable single-colony references and the median stabilises well before
        # scanning every eligible blob.  Identical result on typical plates;
        # saves O(n_blobs) watershed passes on dense plates.
        candidates_pre = sorted(
            [rp for rp in regs if rp.area <= overall_med * 1.5],
            key=lambda rp: rp.area,
        )[:50]
        single_areas = []
        for reg_pre in candidates_pre:
            rm = (lab[reg_pre.bbox[0]:reg_pre.bbox[2],
                      reg_pre.bbox[1]:reg_pre.bbox[3]] == reg_pre.label)
            _, n_pre = _ws_single_pass(rm, float(params.ws_thresh_abs_frac),
                                       int(params.min_object_area))
            if n_pre == 1:
                single_areas.append(reg_pre.area)
        ref_area = float(np.median(single_areas)) if single_areas else None
    else:
        ref_area = None

    for reg in regs:
        minr, minc, maxr, maxc = reg.bbox
        region_mask = (lab[minr:maxr, minc:maxc] == reg.label)
        instance_view = instance_labels[minr:maxr, minc:maxc]

        L, n_colonies, boundary_local = _conservative_watershed(region_mask, params, ref_area)

        center = np.array([reg.centroid[1], reg.centroid[0]], dtype=float)

        poly_pos: Optional[np.ndarray] = None
        if build_rois:
            region_contours = find_contours(region_mask.astype(float), level=0.5)
            if not region_contours:
                continue
            cont = max(region_contours, key=len)
            step = max(1, len(cont) // 50)
            pts = cont[::step, :]
            if pts.size == 0:
                continue
            poly_pos = np.stack([pts[:, 1] + minc, pts[:, 0] + minr], axis=1).astype(float)

        if n_colonies <= 1:
            num_cfu += 1
            if build_rois and poly_pos is not None:
                colony_roi.append(ROI(Position=poly_pos, Center=center,
                                      Creator="Algorithm", Shape="Polygon", NumOfCFU=1))
            instance_view[region_mask] = next_label
            next_label += 1
        else:
            num_cfu += n_colonies
            if build_rois and poly_pos is not None:
                cluster_roi.append(ROI(Position=poly_pos, Center=center,
                                       Creator="Algorithm", Shape="Polygon", NumOfCFU=n_colonies))

            if build_overlay:
                overlay_view = overlay[minr:maxr, minc:maxc]
                overlay_view[binary_dilation(boundary_local, footprint=_disk(1))] = np.array([255, 255, 0], dtype=np.uint8)

            for local_label in range(1, int(L.max()) + 1):
                local_mask = (L == local_label)
                if np.any(local_mask):
                    instance_view[local_mask] = next_label
                    next_label += 1

    return num_cfu, overlay, colony_roi, cluster_roi, instance_labels


# ===========================================================================
# Per-sample driver
# ===========================================================================

def _count_one_expt_hsv(
    expt: np.ndarray,
    k: int,
    use_blackhat: bool = False,
    params: Optional[TuningParams] = None,
    build_rois: bool = True,
) -> Tuple[int, str, str, float, np.ndarray, List[ROI], List[ROI], np.ndarray]:
    """
    Run the HSV pipeline on a single cropped/masked dish image.

    Returns
    -------
    num_cfu, method, size_class, mean_area, out_image, colony_roi, cluster_roi,
    bw_raw (raw HSV mask before morpho_cleanup, as uint8 RGB for saving)
    """
    _auto_adapt = params is None
    params = params or TuningParams()

    # --- BLOCK 1: preprocess ---
    expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat)

    # --- Zero CFU fast-check (from original) ---
    expt_blur = cv2.GaussianBlur(expt, (0, 0), 4)
    gray0 = _gray_uint8(expt_blur)
    bw_min0 = imextendedmin(gray0, 60) & (gray0 != 0)
    bw_min0 = bwpropfilt(bw_min0, "Extent", (0.2, 1.0))
    if not np.any(bw_min0) and not _has_dark_object_probe(expt, params):
        return 0, "Zero", "Zero", 0.0, expt, [], [], np.zeros((*expt.shape[:2], 3), dtype=np.uint8)

    # --- Uncountable fast-check (Otsu on raw gray, same as original) ---
    gray_unc = _gray_uint8(expt)
    thr, _ = cv2.threshold(gray_unc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_unc = (gray_unc < thr) & (gray_unc != 0)
    bw_unc = _remove_edge_bar_artifacts(bw_unc)
    num_unc, unc_overlay, _, _, _ = _count_colonies_with_instances(
        bw_unc, expt, params, build_overlay=True, build_rois=False
    )
    if num_unc > int(params.uncountable_precheck_cutoff):
        bw_unc_rgb = cv2.cvtColor((bw_unc.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
        return int(num_unc), "Uncountable", "Uncountable", _mean_region_area(bw_unc), unc_overlay, [], [], bw_unc_rgb

    # --- Auto size-adapt: switch to large-CFU params if needed ---
    # Use APP1's blur + Laplacian gradient histogram as the first size guess.
    if _auto_adapt:
        _cfu_class, _grad_max, _medium_check, _large_check = _gradient_hist_size_guess(gray_expt)
        if _cfu_class == "large":
            params = LARGE_CFU_PARAMS
            # Re-preprocess with a larger background-estimation sigma so that large
            # colony bodies (radius 60–120 px) do not contaminate their own
            # background estimate.  sigma=80 is too close to the colony radius and
            # reduces HSV contrast at the colony core, causing fragmented masks and
            # watershed over-splitting.  sigma=150 keeps the normalisation benefit
            # (equalises agar colour) without touching large-colony signal.
            # This second preprocess call is a cache miss; it costs ~2 s but runs
            # only once per large-CFU image and is cached for subsequent calls.
            expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat, bg_sigma=150.0)
            print(f"  [expt {k}] size-adapt → large-CFU mode "
                  f"(gradient max {_grad_max:.1f}; large if max < {_large_check:.1f} "
                  f"or max >= {max(_large_check * CLASS_HIGH_GRAD_LARGE_MULT, CLASS_HIGH_GRAD_LARGE_ABS_MIN):.1f})")
        else:
            print(f"  [expt {k}] size-adapt → {_cfu_class}-CFU mode "
                  f"(gradient max {_grad_max:.1f}; medium if max > {_medium_check:.1f}, "
                  f"large if max < {_large_check:.1f} "
                  f"or max >= {max(_large_check * CLASS_HIGH_GRAD_LARGE_MULT, CLASS_HIGH_GRAD_LARGE_ABS_MIN):.1f})")

    # --- BLOCK 2: adaptive HSV mask ---
    bw_raw = hsv_mask_adaptive(expt_adj, gray_expt, params)

    # --- BLOCK 3: morphological cleanup ---
    dish_area = int(np.count_nonzero(gray_expt != 0))
    bw_clean = morpho_cleanup(bw_raw, params, dish_area=dish_area)

    # --- Rim filter: remove blobs that own ≥20% of the dish border zone ---
    # IMPORTANT: use the original expt (not gray_expt) for the dish mask.
    # The preprocessing filters (Gaussian, localcontrast, diffusion) spread
    # non-zero values into the outer black region, so gray_expt != 0 covers
    # nearly the whole image and produces no useful border zone.
    # The original expt has clean hard zeros outside the circular crop.
    dish_mask_orig = np.any(expt > 0, axis=2)
    _large_mode = params.hsv_err_tol > 0.05  # proxy for LARGE_CFU_PARAMS
    _rim_iters = 5 if _large_mode else 12
    border_px = dish_mask_orig & ~ndi.binary_erosion(dish_mask_orig, iterations=_rim_iters)
    n_border = int(np.count_nonzero(border_px))
    bw_clean = _filter_rim_blobs(bw_clean, border_px, n_border)

    # --- Second pass: recover blobs lost in morpho_cleanup ---
    # Only runs on sparse plates (bw_clean < 10% of dish).
    # Dark-agar plates (e.g. J111dilu0/J113dilu0) naturally have bw_clean > 10%
    # → the gate below skips the second pass for them automatically.
    #
    # Diagnostic confirmed: uncovered seeds land inside bw_raw (not mask_delta),
    # meaning the faint colonies passed the HSV threshold but were removed by
    # morpho_cleanup (e.g. too elongated for the opening, or fragmented).
    #
    # Fix: search in (bw_raw & ~bw_clean) — the "survived HSV but killed by
    # cleanup" zone.  Keep only blobs in that zone that contain an
    # imextendedmin(h=15) seed not already near an accepted colony.
    # Add them back WITHOUT re-running morpho_cleanup (it already rejected them).
    # Skipped for large-CFU mode (old file had no second pass; the shallow
    # h=8 seeds recover too much agar-texture noise on sparse large-CFU plates).
    dish_area = int(np.count_nonzero(gray_expt != 0))
    if not _large_mode:
        bw_clean = _recover_second_pass_blobs(bw_raw, bw_clean, gray_expt, dish_area)

    # Expand accepted cores to the visible dark colony footprint BEFORE
    # watershed. This keeps the green fill, post mask, and yellow split
    # boundaries in the same geometry.
    # Large-CFU mode: skip halo expansion entirely.  Large colony halos are
    # broad enough (~30–50 px) that even a small grow_r causes neighbouring
    # halos to merge into one connected component; watershed then over-splits
    # the merged mass.  APP22 had no halo expansion and performed better for
    # large colonies — replicate that here.  Small/medium plates are unaffected.
    if _large_mode:
        bw_segment = bw_clean
    else:
        bw_segment = _expand_mask_to_dark_colony_halos(bw_clean, expt, gray_expt, dish_area,
                                                        large_cfu_mode=False)
    bw_segment = _remove_edge_bar_artifacts(bw_segment)

    # --- BLOCKS 4+5+6: candidate filtering + watershed + count/ROI ---
    # Pass the ORIGINAL (non-preprocessed) image so the green overlay lands
    # on unaltered pixel colours, not the contrast-boosted version.
    num_cfu, out_image, colony_roi, cluster_roi, _ = _count_colonies_with_instances(
        bw_segment, expt, params, build_overlay=True, build_rois=build_rois,
        large_cfu_mode=_large_mode,
    )

    # Hard ceiling (same as original)
    if num_cfu > int(params.uncountable_cutoff):
        bw_post_rgb = cv2.cvtColor((bw_segment.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
        return int(num_cfu), "Uncountable", "Uncountable", _mean_region_area(bw_segment), out_image, colony_roi, cluster_roi, bw_post_rgb

    mean_area = _mean_region_area(bw_segment)

    if mean_area <= 150:
        size_class = "small"
    elif mean_area < 400:
        size_class = "medium"
    else:
        size_class = "large"

    bw_post_rgb = cv2.cvtColor((bw_segment.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
    return int(num_cfu), "HSV", size_class, float(mean_area), out_image, colony_roi, cluster_roi, bw_post_rgb


# ===========================================================================
# Public API  (mirrors countCFUAPP.count_cfu_app / predict_count_only)
# ===========================================================================

def predict_count_only_hsv(
    rgb: np.ndarray,
    dish_mode: str = "auto",
    target_area_px2: float = 2245000.0,
    use_blackhat: bool = False,
    params: Optional[TuningParams] = None,
) -> int:
    """Return total CFU count for an image (no output images saved)."""
    params = params or TuningParams()
    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))
    total = 0
    _large_mode = params.hsv_err_tol > 0.05  # proxy for LARGE_CFU_PARAMS; determined by caller
    for item in expts:
        expt = item["expt"]

        expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat,
                                               bg_sigma=150.0 if _large_mode else 80.0)

        expt_blur = cv2.GaussianBlur(expt, (0, 0), 4)
        gray0 = _gray_uint8(expt_blur)
        bw_min0 = imextendedmin(gray0, 60) & (gray0 != 0)
        bw_min0 = bwpropfilt(bw_min0, "Extent", (0.2, 1.0))
        if not np.any(bw_min0) and not _has_dark_object_probe(expt, params):
            continue

        gray_unc = _gray_uint8(expt)
        thr, _ = cv2.threshold(gray_unc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_unc = (gray_unc < thr) & (gray_unc != 0)
        bw_unc = _remove_edge_bar_artifacts(bw_unc)
        num_unc, _, _, _, _ = _count_colonies_with_instances(
            bw_unc, expt, params, build_overlay=False, build_rois=False
        )
        if num_unc > int(params.uncountable_precheck_cutoff):
            total += int(num_unc)
            continue

        bw_raw = hsv_mask_adaptive(expt_adj, gray_expt, params)
        dish_area = int(np.count_nonzero(gray_expt != 0))
        bw_clean = morpho_cleanup(bw_raw, params, dish_area=dish_area)

        dish_mask_orig = np.any(expt > 0, axis=2)
        _rim_iters = 5 if _large_mode else 12
        border_px = dish_mask_orig & ~ndi.binary_erosion(dish_mask_orig, iterations=_rim_iters)
        n_border = int(np.count_nonzero(border_px))
        bw_clean = _filter_rim_blobs(bw_clean, border_px, n_border)
        if not _large_mode:
            bw_clean = _recover_second_pass_blobs(bw_raw, bw_clean, gray_expt, dish_area)
        if _large_mode:
            bw_segment = bw_clean
        else:
            bw_segment = _expand_mask_to_dark_colony_halos(bw_clean, expt, gray_expt, dish_area,
                                                            large_cfu_mode=False)
        bw_segment = _remove_edge_bar_artifacts(bw_segment)

        num_cfu, _, _, _, _ = _count_colonies_with_instances(
            bw_segment, expt, params, build_overlay=False, build_rois=False,
            large_cfu_mode=_large_mode,
        )
        total += int(num_cfu)
    return total


def predict_tuning_features(
    rgb: np.ndarray,
    dish_mode: str = "auto",
    target_area_px2: float = 2245000.0,
    use_blackhat: bool = False,
    params: Optional[TuningParams] = None,
) -> Dict[str, Any]:
    """
    Run the full HSV pipeline silently and return features needed for grid-search tuning.

    This is a print-free mirror of _count_one_expt_hsv that also exposes
    per-instance pixel areas (required by tune_params_cv.py for the area-EMD
    and area-mass scores).

    Returns
    -------
    dict with keys:
        count          : int   — total predicted CFU count
        instance_areas : list[float] — pixel area of every predicted instance
        methods        : list[str]   — method per expt ("HSV", "Zero", "Uncountable")
        classes        : list[str]   — size class per expt
        mean_area      : float       — mean colony area across countable expts
    """
    _large_override = params
    _small_params = TuningParams()
    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))

    total_count = 0
    all_instance_areas: List[float] = []
    mean_areas: List[float] = []
    methods: List[str] = []
    classes: List[str] = []
    _binary_parts: List[np.ndarray] = []

    for k, item in enumerate(expts, start=1):
        expt = item["expt"]
        _large_cfu_mode = False

        # --- BLOCK 1: preprocess ---
        expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat)

        # --- Size-adapt: keep small params fixed, override only large mode ---
        _cfu_class, _grad_max, _medium_check, _large_check = _gradient_hist_size_guess(gray_expt)
        if _cfu_class == "large":
            params = _large_override or LARGE_CFU_PARAMS
            _large_cfu_mode = True
            # Re-preprocess with larger background sigma (same rationale as in
            # _count_one_expt_hsv — preserves large-colony HSV contrast).
            expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat, bg_sigma=150.0)
            print(f"  [expt {k}] size-adapt → large-CFU mode "
                  f"(gradient max {_grad_max:.1f}; large if max < {_large_check:.1f} "
                  f"or max >= {max(_large_check * CLASS_HIGH_GRAD_LARGE_MULT, CLASS_HIGH_GRAD_LARGE_ABS_MIN):.1f})")
        else:
            params = _small_params
            print(f"  [expt {k}] size-adapt → {_cfu_class}-CFU mode "
                  f"(gradient max {_grad_max:.1f}; medium if max > {_medium_check:.1f}, "
                  f"large if max < {_large_check:.1f} "
                  f"or max >= {max(_large_check * CLASS_HIGH_GRAD_LARGE_MULT, CLASS_HIGH_GRAD_LARGE_ABS_MIN):.1f})")

        # --- Zero-CFU fast-check ---
        expt_blur = cv2.GaussianBlur(expt, (0, 0), 4)
        gray0 = _gray_uint8(expt_blur)
        bw_min0 = imextendedmin(gray0, 60) & (gray0 != 0)
        bw_min0 = bwpropfilt(bw_min0, "Extent", (0.2, 1.0))
        if not np.any(bw_min0) and not _has_dark_object_probe(expt, params):
            methods.append("Zero")
            classes.append("Zero")
            continue

        # --- Uncountable fast-check (Otsu on raw gray) ---
        gray_unc = _gray_uint8(expt)
        thr, _ = cv2.threshold(gray_unc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_unc = (gray_unc < thr) & (gray_unc != 0)
        bw_unc = _remove_edge_bar_artifacts(bw_unc)
        num_unc, _, _, _, inst_unc = _count_colonies_with_instances(
            bw_unc, expt, params, build_overlay=False, build_rois=False
        )
        if num_unc > int(params.uncountable_precheck_cutoff):
            _binary_parts.append(bw_unc)
            total_count += int(num_unc)
            _, cnts = np.unique(inst_unc[inst_unc > 0], return_counts=True)
            all_instance_areas.extend(float(c) for c in cnts)
            methods.append("Uncountable")
            classes.append("Uncountable")
            continue

        # --- BLOCK 2: adaptive HSV mask ---
        bw_raw = hsv_mask_adaptive(expt_adj, gray_expt, params)

        # --- BLOCK 3: morphological cleanup ---
        dish_area = int(np.count_nonzero(gray_expt != 0))
        bw_clean = morpho_cleanup(bw_raw, params, dish_area=dish_area)

        # --- Rim filter ---
        # Old file used 5 erosion iterations (~10px border zone); new default
        # is 12 (~24px).  The wider zone trims or removes large colonies near
        # the dish rim.  Restore the narrower zone for large-CFU mode only.
        dish_mask_orig = np.any(expt > 0, axis=2)
        _rim_iters = 5 if _large_cfu_mode else 12
        border_px = dish_mask_orig & ~ndi.binary_erosion(dish_mask_orig, iterations=_rim_iters)
        n_border = int(np.count_nonzero(border_px))
        bw_clean = _filter_rim_blobs(bw_clean, border_px, n_border)

        # --- Second pass: recover blobs lost in morpho_cleanup on sparse plates ---
        # Old file had no second pass.  On sparse large-CFU plates (coverage <10%)
        # imextendedmin(h=8) recovers too many shallow agar-texture fragments;
        # skip for large-CFU mode to match old behaviour.
        if not _large_cfu_mode:
            bw_clean = _recover_second_pass_blobs(bw_raw, bw_clean, gray_expt, dish_area)
        if _large_cfu_mode:
            bw_segment = bw_clean
        else:
            bw_segment = _expand_mask_to_dark_colony_halos(
                bw_clean, expt, gray_expt, dish_area,
                large_cfu_mode=False,
            )
        bw_segment = _remove_edge_bar_artifacts(bw_segment)
        _binary_parts.append(bw_segment)

        # --- BLOCKS 4+5+6: count + instance labels ---
        num_cfu, _, _, _, inst_labels = _count_colonies_with_instances(
            bw_segment, expt, params, build_overlay=False, build_rois=False,
            large_cfu_mode=_large_cfu_mode,
        )

        # Hard ceiling
        if num_cfu > int(params.uncountable_cutoff):
            total_count += int(num_cfu)
            _, cnts = np.unique(inst_labels[inst_labels > 0], return_counts=True)
            all_instance_areas.extend(float(c) for c in cnts)
            methods.append("Uncountable")
            classes.append("Uncountable")
            continue

        total_count += int(num_cfu)
        _, cnts = np.unique(inst_labels[inst_labels > 0], return_counts=True)
        all_instance_areas.extend(float(c) for c in cnts)

        mean_area = _mean_region_area(bw_segment)
        mean_areas.append(float(mean_area))

        if mean_area <= 150:
            size_class = "small"
        elif mean_area < 400:
            size_class = "medium"
        else:
            size_class = "large"
        classes.append(size_class)
        methods.append("HSV")

    # Merge per-expt binary masks.  For single-expt (pre_cropped) images
    # there is exactly one part; for multi-expt we OR them at the first
    # part's resolution (approximate but sufficient for a spatial Dice check).
    if _binary_parts:
        combined_binary = _binary_parts[0].copy()
        for _part in _binary_parts[1:]:
            if _part.shape == combined_binary.shape:
                combined_binary |= _part
            else:
                _part_rs = cv2.resize(
                    _part.astype(np.uint8),
                    (combined_binary.shape[1], combined_binary.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                combined_binary |= _part_rs
    else:
        combined_binary = np.zeros(rgb.shape[:2], dtype=bool)

    return {
        "count": total_count,
        "instance_areas": all_instance_areas,
        "methods": methods,
        "classes": classes,
        "mean_area": float(np.mean(mean_areas)) if mean_areas else 0.0,
        "binary_mask": combined_binary,
    }


def count_cfu_app2(
    rgb: np.ndarray,
    top_cell: str,
    bot_cell: str,
    folder_dir: Union[str, Path],
    version: str,
    index: Any,  # kept for API compatibility with count_cfu_app
    dish_mode: str = "auto",
    target_area_px2: float = 2245000.0,
    save_which: str = "overlay",   # "overlay" | "expt" | "both"
    use_blackhat: bool = False,
    params: Optional[TuningParams] = None,
    return_metadata: bool = False,
) -> Any:
    """
    Count CFU on a single image using the HSV-first pipeline.

    Drop-in replacement for countCFUAPP.count_cfu_app; same signature
    except for the extra use_blackhat flag.

    dish_mode:
      pre_cropped — input is already one cropped experiment image
      single      — whole dish, no split
      double      — try top/bottom split; fall back to 1 if no divider
      auto        — same as double
    """
    # Do NOT resolve params here — pass None through so _count_one_expt_hsv
    # can detect it and apply the automatic large/small-CFU size adaptation.
    # If the caller supplied explicit params, they are passed through unchanged.
    folder_dir = Path(folder_dir)
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / version).mkdir(parents=True, exist_ok=True)

    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))

    out_paths: List[str] = []
    counts: List[int] = []

    n_expts = len(expts)
    for k, item in enumerate(expts, start=1):
        if n_expts >= 2 and k == 2 and not bot_cell:
            continue
        expt = item["expt"]
        num_cfu, _, _, _, out_image, _, _, bw_post_rgb = _count_one_expt_hsv(
            expt, k, use_blackhat=use_blackhat, params=params, build_rois=False
        )
        counts.append(int(num_cfu))

        if n_expts >= 2:
            expt_id = top_cell if k == 1 else bot_cell
        else:
            # Single / full dish: use the plain stem (top_cell without any suffix)
            expt_id = top_cell.removesuffix("_Top") if top_cell.endswith("_Top") else top_cell
        base = folder_dir / version / expt_id

        if save_which in ("overlay", "both"):
            out_tif = base.with_suffix(".tif")
            save_tiff_rgb(
                out_tif,
                out_image if (isinstance(out_image, np.ndarray) and out_image.size) else expt
            )
            out_paths.append(str(out_tif))

        if save_which in ("expt", "both"):
            expt_tif = base.with_name(base.name + "__expt").with_suffix(".tif")
            save_tiff_rgb(expt_tif, expt)
            out_paths.append(str(expt_tif))

        # Save the post-cleanup mask to the system temp dir so it never
        # appears in the user's output directory.
        post_tif = Path(tempfile.gettempdir()) / (base.name + "__post.tif")
        save_tiff_rgb(post_tif, bw_post_rgb)
        out_paths.append(str(post_tif))

    if return_metadata:
        return {"out_paths": out_paths, "counts": counts}

    return out_paths


def _predict_test_rows(
    rgb: np.ndarray,
    image_name: str,
    dish_mode: str = "auto",
    target_area_px2: float = 2245000.0,
    use_blackhat: bool = False,
    params: Optional[TuningParams] = None,
) -> List[Dict[str, Any]]:
    params = params or TuningParams()
    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))
    rows: List[Dict[str, Any]] = []
    image_stem = Path(image_name).stem
    top_id, bot_id = derive_experiment_ids_for_name(image_name)
    n_expts = len(expts)

    for k, item in enumerate(expts, start=1):
        expt = item["expt"]
        if n_expts >= 2:
            if k == 2 and not bot_id:
                continue
            petri_dish = top_id if k == 1 else bot_id
        else:
            petri_dish = image_stem

        features = predict_tuning_features(
            expt,
            dish_mode="pre_cropped",
            target_area_px2=float(target_area_px2),
            use_blackhat=use_blackhat,
            params=params,
        )

        count = int(features.get("count", 0))
        instance_areas = [float(a) for a in features.get("instance_areas", [])]
        methods = ",".join(str(v) for v in features.get("methods", []))
        classes = ",".join(str(v) for v in features.get("classes", []))
        mean_area = float(features.get("mean_area", 0.0))

        if not instance_areas:
            rows.append(
                {
                    "image_name": image_name,
                    "petri_dish": petri_dish,
                    "dish_index": k,
                    "count": count,
                    "cfu_index": "",
                    "cfu_area_px": "",
                    "mean_area_px": mean_area,
                    "methods": methods,
                    "classes": classes,
                }
            )
            continue

        for i, area in enumerate(instance_areas, start=1):
            rows.append(
                {
                    "image_name": image_name,
                    "petri_dish": petri_dish,
                    "dish_index": k,
                    "count": count,
                    "cfu_index": i,
                    "cfu_area_px": area,
                    "mean_area_px": mean_area,
                    "methods": methods,
                    "classes": classes,
                }
            )

    return rows


def _write_test_csv(rows: List[Dict[str, Any]], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "countCFUAPP2_test_results.csv"
    fieldnames = [
        "image_name",
        "petri_dish",
        "dish_index",
        "count",
        "cfu_index",
        "cfu_area_px",
        "mean_area_px",
        "methods",
        "classes",
        "total_runtime_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


# ===========================================================================
# CLI
# ===========================================================================

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _collect_images(path: Path) -> List[Path]:
    """Return a list of image files from a file path OR a directory."""
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    return [path]


if __name__ == "__main__":
    import argparse
    t0_total = time.perf_counter()

    parser = argparse.ArgumentParser(description="HSV-first CFU counter (countCFUAPP2)")
    parser.add_argument("--image",  required=True,
                        help="Input image (png/jpg/tif) OR a folder of images")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--version",         default="v2")
    parser.add_argument("--dish_mode",       default="auto",
                        choices=["auto", "single", "double", "pre_cropped"])
    parser.add_argument("--target_area_px2", type=float, default=2245000.0)
    parser.add_argument("--save_which",      default="overlay",
                        choices=["overlay", "expt", "both"])
    parser.add_argument("--blackhat",        action="store_true",
                        help="Enable blackhat preprocessing to enhance faint colonies")
    parser.add_argument("--Test", action="store_true",
                        help="Write a CSV in outdir with per-petridish counts and per-CFU areas")
    args = parser.parse_args()

    input_path = Path(args.image)
    image_files = _collect_images(input_path)
    if not image_files:
        raise RuntimeError(f"No images found at: {input_path}")

    all_paths: List[str] = []
    test_rows: List[Dict[str, Any]] = []
    csv_path = None
    for img_path in image_files:
        print(f"[input] {img_path.name}", flush=True)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[skip] Cannot read {img_path}")
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        stem = img_path.stem
        top_id, bot_id = derive_experiment_ids_for_name(img_path.name)
        paths = count_cfu_app2(
            rgb=img,
            top_cell=top_id,
            bot_cell=bot_id or "",
            folder_dir=args.outdir,
            version=args.version,
            index=None,
            dish_mode=args.dish_mode,
            target_area_px2=args.target_area_px2,
            save_which=args.save_which,
            use_blackhat=args.blackhat,
        )
        all_paths.extend(list(paths))
        if args.Test:
            new_rows = _predict_test_rows(
                rgb=img,
                image_name=img_path.name,
                dish_mode=args.dish_mode,
                target_area_px2=args.target_area_px2,
                use_blackhat=args.blackhat,
            )
            elapsed = float(time.perf_counter() - t0_total)
            for row in new_rows:
                row["total_runtime_sec"] = elapsed
            test_rows.extend(new_rows)
            csv_path = _write_test_csv(test_rows, Path(args.outdir))
            print(f"[CSV] {csv_path}  ({len(test_rows)} rows, elapsed {elapsed:.1f}s)")
    for p in all_paths:
        print(p)
    if args.Test:
        total_runtime_sec = float(time.perf_counter() - t0_total)
        if csv_path:
            print(csv_path)
        print(f"[OK] Total runtime: {total_runtime_sec:.3f} s")
