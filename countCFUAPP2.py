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
from medpy.filter.smoothing import anisotropic_diffusion
from scipy import ndimage as ndi
from skimage import exposure
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
    large_cfu_mode: bool = False
    mini_cfu_mode: bool = False


CLASS_LARGE_THRESH = 0.0043
CLASS_MEDIUM_THRESH = 0.0032
CLASS_HIGH_GRAD_LARGE_MULT = 15.0
CLASS_HIGH_GRAD_LARGE_ABS_MIN = 75_000.0

# Blob-size classifier thresholds (median Otsu-blob area in raw image, px²).
# Large colony radius ~60-130 px → area ~11 000–53 000 px².
# Small colony radius ~8-25 px → area ~200–2 000 px².
LARGE_BLOB_AREA_THRESH  = 4_000.0   # median area above this → large-CFU mode
MEDIUM_BLOB_AREA_THRESH = 1_000.0   # median area above this → medium (uses small params)


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
hsv_v_start=0.60,
hsv_v_step=0.05,
hsv_v_min=0.15,
hsv_s_min=-1.0,
hsv_err_tol=0.02,
h_ref_frac=0.10,
hsv_extent_min=0.15,
hsv_open_disk_small=1,
hsv_open_disk_large=18,
hsv_close_disk_large=20,
min_object_area=300,
small_area_quantile=0.75,
small_cluster_extent_min=0.20,
nonsmall_cluster_ecc_min=0.99,
nonsmall_cluster_extent_min=0.20,
colony_circularity_min=0.09,
ws_thresh_abs_frac=0.05,
ws_gauss_sigma=0.3,
uncountable_cutoff=300,
uncountable_precheck_cutoff=400,
large_cfu_mode=True,
mini_cfu_mode=False,
)


# ---------------------------------------------------------------------------
# Mini-CFU parameter profile
# ---------------------------------------------------------------------------
# Mini-CFU mode targets truly tiny colonies: radius ~3–10 px, area ~30–300 px².
# These are smaller than what the default small-CFU path handles reliably.
#
# Why these values differ from default:
#   min_object_area  : 90 → 12   — allow very small blobs through; circularity
#                                   and local-contrast filters do FP rejection instead.
#   hsv_open_disk_large: 16 → 5  — the 2× up-sample opening with disk(16) erases
#                                   colonies whose radius is < 8 px; disk(5) is safe.
#   hsv_close_disk_large: 10 → 3 — gap-repair dilation applied uniformly to all blobs
#                                   in mini mode; small r prevents merging neighbours.
#   h_ref_frac       : 0.12 → 0.06 — smaller h captures shallower dark spots as
#                                   the reference for V-threshold calibration, making
#                                   the HSV step sensitive to faint mini colonies.
#   colony_circularity_min: 0.09 → 0.35 — agar-texture FP are non-circular; mini
#                                   genuine colonies are compact blobs → stricter gate.
#   hsv_err_tol      : 0.0061 → 0.006 — similar tolerance; combined with circularity
#                                   and local-contrast post-filter we can afford to
#                                   let slightly more through the V loop.
MINI_CFU_PARAMS = TuningParams(
hsv_v_start=0.35,
hsv_v_step=0.02,
hsv_v_min=0.15,
hsv_s_min=0.00,
hsv_err_tol=0.006,
h_ref_frac=0.06,
hsv_extent_min=0.25,
hsv_open_disk_small=1,
hsv_open_disk_large=5,
hsv_close_disk_large=3,
min_object_area=12,
small_area_quantile=0.75,
small_cluster_extent_min=0.20,
nonsmall_cluster_ecc_min=0.99,
nonsmall_cluster_extent_min=0.20,
colony_circularity_min=0.35,
ws_thresh_abs_frac=0.04,
ws_gauss_sigma=0.5,
uncountable_cutoff=500,
uncountable_precheck_cutoff=600,
large_cfu_mode=False,
mini_cfu_mode=True,
)

# Mean blob area threshold for "mini" size-class label (px²).
# Detected colony distributions with mean area ≤ this value are labelled
# "mini" when mini mode is active; above it they are labelled "small".
MINI_AREA_THRESH = 200


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


def localcontrast_approx(
    img: np.ndarray,
    amount: float = 0.17,
    mid: float = 0.9,
    sigma: float = 3.0,
) -> np.ndarray:
    img01 = _as_float01(img)
    if img01.ndim == 3:
        gray = cv2.cvtColor(_to_uint8(img01), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = img01
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    d = gray - local_mean
    weight = np.exp(-((gray - mid) ** 2) / (2 * 0.25 ** 2))
    d_enhanced = d * (1.0 + amount * weight)
    out_gray = np.clip(local_mean + d_enhanced, 0.0, 1.0)
    if img01.ndim == 3:
        denom = np.maximum(gray, 1e-6)
        ratio = (out_gray / denom)[..., None]
        out = np.clip(img01 * ratio, 0.0, 1.0)
        return _to_uint8(out)
    return _to_uint8(out_gray)


def imlocalbrighten_approx(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    img01 = _as_float01(img)
    out = exposure.adjust_sigmoid(img01, cutoff=0.5, gain=5.0 * float(strength) + 1.0)
    return _to_uint8(out)


def gaussian_pyramid(img, levels):
    G = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        G.append(img)
    return G


def laplacian_pyramid(G):
    L = []
    for i in range(len(G) - 1):
        up = cv2.pyrUp(G[i + 1], dstsize=(G[i].shape[1], G[i].shape[0]))
        L.append(G[i] - up)
    L.append(G[-1])
    return L


def reconstruct_pyramid(L):
    current = L[-1]
    for i in range(len(L) - 2, -1, -1):
        current = cv2.pyrUp(current, dstsize=(L[i].shape[1], L[i].shape[0])) + L[i]
    return current


def phi(d, sigma, alpha):
    abs_d = np.abs(d)
    sign = np.sign(d)
    w = np.exp(-(abs_d / sigma) ** 2)
    return sign * (alpha * abs_d * w + abs_d * (1 - w))



def locallapfilt_approx(img, sigma=0.2, alpha=3.0, beta=0.5, levels=5):
    """
    Approximation of MATLAB locallapfilt using pyramid + local remapping.

    Parameters:
        sigma: contrast threshold
        alpha: detail enhancement
        beta: base compression
    """
    img = _as_float01(img)

    if img.ndim == 2:
        img = img[..., None]

    out_channels = []

    for c in range(img.shape[2]):
        I = img[..., c]

        # --- 1. Gaussian pyramid ---
        G = gaussian_pyramid(I, levels)

        # --- 2. Laplacian pyramid ---
        L = laplacian_pyramid(G)

        # --- 3. Modify coefficients ---
        L_mod = []

        for k in range(len(L) - 1):
            g0 = G[k]
            d = I - cv2.resize(g0, (I.shape[1], I.shape[0]))

            d_remap = phi(d, sigma, alpha)

            I_remap = cv2.resize(g0, (I.shape[1], I.shape[0])) + d_remap

            # recompute Gaussian + Laplacian locally
            G_remap = gaussian_pyramid(I_remap, levels)
            L_remap = laplacian_pyramid(G_remap)

            L_mod.append(L_remap[k])

        # top level (coarse tone)
        L_mod.append(beta * L[-1])

        # --- 4. Reconstruct ---
        I_out = reconstruct_pyramid(L_mod)

        out_channels.append(I_out)

    out = np.stack(out_channels, axis=-1)

    if out.shape[2] == 1:
        out = out[..., 0]

    return _to_uint8(np.clip(out, 0.0, 1.0))

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

def imdiffusefilt_pm(img):
    img01 = img.astype(float) / 255.0
    out = anisotropic_diffusion(img01, niter=20, kappa=20, gamma=0.1)
    return (out * 255).astype(np.uint8)


def imsharpen_approx(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return sharp


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
    _n_border: int,
) -> np.ndarray:
    if not border_px.any() or not np.any(bw):
        return bw
    lab_rim = label(bw, connectivity=2)
    max_label = int(lab_rim.max())
    if max_label == 0:
        return bw
    blob_sizes = np.bincount(lab_rim.ravel(), minlength=max_label + 1)
    border_hits = np.bincount(lab_rim[border_px].ravel(), minlength=max_label + 1)
    # Fraction of each blob's own pixels that fall inside the border zone.
    # Old logic divided by total border pixels (n_border), so small rim arcs
    # with e.g. 400px out of 200k border pixels = 0.2% never reached the 20%
    # threshold.  Per-blob fraction correctly identifies blobs that live inside
    # the rim shadow regardless of how large the border zone is.
    frac = border_hits[1:] / np.maximum(blob_sizes[1:], 1).astype(float)
    # Primary: blob ≥70% inside border → rim shadow artifact
    rim_flag = frac >= 0.70
    # Secondary: elongated arc (aspect elongation ≥0.60) with ≥50% in border
    # catches thin horizontal bar-shadow arcs that hug the rim
    arc_candidates = np.where((frac >= 0.50) & ~rim_flag)[0]
    for idx in arc_candidates:
        lbl = int(idx) + 1
        ys, xs = np.where(lab_rim == lbl)
        bb_h = int(ys.max() - ys.min() + 1)
        bb_w = int(xs.max() - xs.min() + 1)
        elongation = 1.0 - min(bb_h, bb_w) / max(bb_h, bb_w, 1)
        if elongation >= 0.60:
            rim_flag[idx] = True
    remove_labels = np.where(rim_flag)[0] + 1
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


def _crop_to_nonzero_bbox_with_offset(
    rgb: np.ndarray, pad: int = 2
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Like _crop_to_nonzero_bbox but also returns the (y0, x0) top-left offset
    of the crop within `rgb`.  Used so callers can embed crop-space masks back
    into the full-image canvas with correct spatial alignment."""
    if rgb.size == 0:
        return rgb, (0, 0)
    m = np.any(rgb != 0, axis=2) if rgb.ndim == 3 else (rgb != 0)
    ys, xs = np.where(m)
    if ys.size == 0 or xs.size == 0:
        return rgb, (0, 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(rgb.shape[0], y1 + pad)
    x1 = min(rgb.shape[1], x1 + pad)
    return rgb[y0:y1, x0:x1].copy(), (y0, x0)


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
    full_shape = rgb.shape[:2]

    def _make_item(src: np.ndarray) -> dict:
        crop, (y0, x0) = _crop_to_nonzero_bbox_with_offset(src)
        return {"expt": crop, "offset": (y0, x0), "full_shape": full_shape}

    if dish_mode == "pre_cropped":
        return [_make_item(rgb)]
    if dish_mode == "auto" and _looks_pre_cropped(rgb):
        return [_make_item(rgb)]
    import DetectDish
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fixed_diameter_px = 2.0 * float(np.sqrt(float(target_area_px2) / np.pi))
    masked_bgr, ellipse_final, brightness, _, erosion_px = DetectDish.detect_plate_rgb(
        bgr, fixed_diameter_px=fixed_diameter_px
    )
    if ellipse_final is None or masked_bgr is None:
        return [{"expt": rgb, "offset": (0, 0), "full_shape": full_shape}]
    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    if dish_mode == "single":
        return [_make_item(masked_rgb)]

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
                border_erosion_px=erosion_px,
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
        return [_make_item(masked_rgb)]

    if split_result is None:
        y0_bar, y1_bar = bar_extents
        split_result = DetectDish.mask_top_bottom(
            image_bgr=bgr,
            ellipse=ellipse_final,
            y_bar_top=y0_bar,
            y_bar_bot=y1_bar,
            margin=DetectDish.BAR_SPLIT_MARGIN_PX,
            brightness=brightness,
            border_erosion_px=erosion_px,
        )

    top_bgr, bot_bgr = split_result
    top_rgb = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
    bot_rgb = cv2.cvtColor(bot_bgr, cv2.COLOR_BGR2RGB)

    top_item = _make_item(top_rgb)
    bot_item = _make_item(bot_rgb)

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
        top_nz = float(np.count_nonzero(np.any(top_item["expt"] > 0, axis=2)))
        bot_nz = float(np.count_nonzero(np.any(bot_item["expt"] > 0, axis=2)))
        top_frac = top_nz / total_dish_nz
        bot_frac = bot_nz / total_dish_nz
        if top_frac < _MIN_HALF_FRAC or bot_frac < _MIN_HALF_FRAC:
            # Split is lopsided → false divider on a full dish
            return [_make_item(masked_rgb)]

    return [top_item, bot_item]

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
    Standard enhancement chain + optional blackhat.

    Parameters
    ----------
    expt          : RGB uint8 image (cropped dish / half-dish)
    use_blackhat  : if True, apply blackhat on the contrast-enhanced gray
                    and add the result back before HSV masking
    blackhat_disk_r: structuring element radius for the blackhat close op

    Returns
    -------
    expt_adj  : preprocessed RGB uint8 (used as input to HSV block)
    gray_expt : grayscale uint8 of the preprocessed image
                (optionally blackhat-boosted)
    """
    # Cache lookup — preprocessing is params-independent and expensive
    # (Perona-Malik 20 iters + Laplacian pyramid).  The tuner calls this
    # hundreds of times per image with different TuningParams; caching the
    # result gives a large speedup at zero cost to accuracy.
    # id(expt) is O(1) — no image copy. Safe because _WORKER_RGB_CACHE in the
    # tuner holds a live reference, so the same array object is reused for the
    # same image across all param evaluations within a worker process.
    _cache_key = (id(expt), expt.shape, expt.strides, use_blackhat, blackhat_disk_r, bg_sigma)
    _cached = _PREPROCESS_CACHE.get(_cache_key)
    if _cached is not None:
        return _cached

    # --- background normalisation: makes all agar colors look the same ---
    # Must run before imadjust_approx because imadjust uses fixed per-channel
    # clip ranges calibrated for a specific agar brightness.  After this step
    # the agar background is always near 140 DN, so imadjust sees consistent
    # input regardless of whether the agar was yellow, pink, dark-red, or white.
    # bg_sigma=80 is the default (small colonies, radius ≤40px).  For large-CFU
    # mode (radius 60–120px) sigma=150 is used so the colony signal does not
    # bleed into the background estimate and reduce HSV contrast.
    expt = remove_background_rgb(expt, sigma=bg_sigma)

    # --- standard chain (identical to _count_one_expt_full_logic) ---
    expt_adj = imadjust_approx(expt)                           # stretch contrast per channel
    expt_adj = locallapfilt_approx(expt_adj, sigma=0.2, alpha=0.5)  # edge-aware tone mapping
    expt_adj = localcontrast_approx(expt_adj, amount=0.17, mid=0.9) # boost local microcontrast
    expt_adj = imlocalbrighten_approx(expt_adj, strength=0.5)       # lift shadows
    # im2grayContrast without imhmax: imhmax(h=100) was compressing the dynamic
    # range from ~160 units down to ~68, losing faint colony signal. APP2 only
    # uses gray for imextendedmin and the dish mask — full range is essential.
    img01 = _as_float01(expt_adj)
    expt_f = np.clip((1.0 - img01) - img01, 0.0, 1.0)  # clip(1 - 2*img, 0, 1)
    expt_u8 = _to_uint8(expt_f)
    expt_u8 = localcontrast_approx(expt_u8, 0.18, 0.9)
    expt_u8 = imsharpen_approx(expt_u8)
    gray_expt = _to_uint8(rgb2gray(_as_float01(255 - expt_u8)))
    gray_expt = imdiffusefilt_pm(gray_expt)  # Perona-Malik smoothing (edges preserved)

    # --- optional blackhat ---
    if use_blackhat:
        se = _disk_kernel(blackhat_disk_r)
        blackhat = cv2.morphologyEx(gray_expt, cv2.MORPH_BLACKHAT, se)
        # Add back: darker colonies become even darker (their V dip is amplified)
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

    # Large-CFU mode: after remove_background_rgb(sigma=150) agar normalises to
    # V≈0.50; large colony interiors are darker (V≈0.15–0.45).  The V histogram
    # is bimodal (dark colonies vs agar) with Otsu threshold consistently near
    # V≈0.35–0.37.  The adaptive loop cannot be used here because FP from
    # agar-below-mean pixels floods the rate at any permissive threshold.
    if params.large_cfu_mode:
        _v_nz = ((_V[_dish_nz]) * 255).astype(np.uint8).reshape(-1, 1)
        _thr_v, _ = cv2.threshold(_v_nz, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return (_V < (_thr_v / 255.0)) & _dish_nz

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
    _open_r = 1 if (params.large_cfu_mode or _cov_pre < 0.08) else int(params.hsv_open_disk_small)
    bw = binary_opening(bw, footprint=_disk(_open_r))

    # 3 — close ring / arc gaps per blob, then fill enclosed holes.
    # The HSV mask often captures only the bright rim of a colony, leaving
    # C-shapes, arcs, and crescents.  binary_fill_holes requires a fully
    # enclosed hole, so open arcs are not filled.
    # Strategy: for each blob independently dilate → fill → erode back.
    # Per-blob processing prevents merging of neighbouring colonies.
    bw_labeled, _ = ndi.label(bw)
    bw_repaired = np.zeros_like(bw)
    blob_areas = np.bincount(bw_labeled.ravel())
    H, W = bw.shape
    blob_slices = ndi.find_objects(bw_labeled)
    for i, sl in enumerate(blob_slices):
        if sl is None:
            continue
        label_i = i + 1
        area = int(blob_areas[label_i])
        if params.large_cfu_mode or params.mini_cfu_mode:
            # large: uniform large r for halos. mini: uniform small r (params.hsv_close_disk_large=3)
            # so nearby tiny colonies are never fused by the gap-repair dilation.
            r = int(params.hsv_close_disk_large)
        else:
            r = 12 if area < 1500 else (6 if area < 8000 else int(params.hsv_close_disk_large))
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
    if params.large_cfu_mode:
        # Remove thin agar patches while keeping compact colonies.
        # disk=20 on 2× image = effective 10px in original space, which is smaller
        # than the smallest real colony (area≥662px → radius≥15px).
        h, w = bw.shape
        big = cv2.resize(bw.astype(np.uint8), None, fx=2.0, fy=2.0,
                         interpolation=cv2.INTER_NEAREST) > 0
        big = binary_opening(big, footprint=_disk(20))
        bw = cv2.resize(big.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST) > 0
    elif coverage > 0.10:
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




_EDGE_MAX_LARGE_THRESH  = 1100   # largest Canny edge fragment (px) in large-CFU plates
_N_LOCAL_BLOB_MAX_LARGE = 150    # plates with ≥150 σ=30 local blobs are small-CFU
_OTSU_N_MIN_LARGE       = 2      # need ≥2 Otsu blobs OR n_local ≥50 (catches 1182E)
_GRAY_STD_FALLBACK      = 40.0   # fallback: high std + moderate n_local → large (catches 1186N)
# cond3 thresholds: catches low-contrast large-CFU plates (e.g. DIFIP2 at dilu2)
# where edge_max is always ~750 regardless of CFU size, but n_local and gray_std
# are elevated compared to nearly-empty small-CFU plates at the same dilution.
_COND3_NLOC_LO  = 8     # moderate local blob count (more than empty plate)
_COND3_NLOC_HI  = 80    # below this → not an overcrowded/uncountable plate
_COND3_STD_MIN  = 20.0  # sufficient gray variance to confirm colony signal (true small-CFU plates always <16)


def _blob_size_class_guess(expt_raw: np.ndarray) -> Tuple[str, float, float, float]:
    """
    Classify colony size using four features on the raw (RGB) image.

    Primary rule (cond1): large-CFU plates have long continuous Canny edge fragments
      (edge_max ≥ 1100), not too many local-dark blobs (n_local < 150), and at least
      some dark Otsu blobs visible (otsu_n ≥ 2) or many σ=30 local blobs (n_local ≥ 50).

    Fallback rule (cond2): plates with extremely high gray variance (std ≥ 40) and a
      moderate blob count are large-CFU even when edge fragments are short (e.g. 1186N
      where many overlapping large colonies reduce edge fragment length).

    Low-contrast rule (cond3): catches large-CFU plates on camera setups that produce
      low-contrast images (edge_max always ~750) — e.g. DIFIP2 at dilu2.  A moderate
      n_local [20, 80) together with sufficient gray variance (≥25) indicates that real
      large colonies are present, distinguishing them from nearly-empty plates (n_local<20,
      low std) and overcrowded plates (n_local≥80, also excluded by this range).

    LARGE if cond1 OR cond2 OR cond3.
    Returns (class, edge_max, n_local, gray_std).
    """
    img = expt_raw
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    H, W = gray.shape
    bh, bw = int(H * 0.05), int(W * 0.05)

    def _rm(m: np.ndarray) -> np.ndarray:
        m2 = m.copy()
        m2[:bh, :] = 0; m2[-bh:, :] = 0
        m2[:, :bw] = 0; m2[:, -bw:] = 0
        return m2

    # Feature 1: largest Canny edge fragment
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 15, 50)
    edges = _rm(edges)
    n_e, _, stats_e, _ = cv2.connectedComponentsWithStats(edges // 255, connectivity=8)
    edge_max = int(max((stats_e[i, cv2.CC_STAT_AREA] for i in range(1, n_e)), default=0))

    # Feature 2: count of locally-dark blobs (σ=30 background subtraction, depth 12)
    gf = gray.astype(np.float32)
    blur30 = cv2.GaussianBlur(gf, (0, 0), 30)
    dark30 = _rm(((gf - blur30) < -12).astype(np.uint8))
    n_lc, _, stats_lc, _ = cv2.connectedComponentsWithStats(dark30, connectivity=8)
    n_local = int(sum(1 for i in range(1, n_lc) if stats_lc[i, cv2.CC_STAT_AREA] >= 100))

    # Feature 3: number of Otsu-threshold blobs ≥ 200 px²
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_o = _rm((gray < thr).astype(np.uint8))
    n_o, _, stats_o, _ = cv2.connectedComponentsWithStats(dark_o, connectivity=8)
    otsu_n = int(sum(1 for i in range(1, n_o) if stats_o[i, cv2.CC_STAT_AREA] >= 200))

    # Feature 4: gray std-dev inside the dish (nonzero pixels)
    nz = gray > 0
    gray_std = float(np.std(gray[nz])) if np.any(nz) else 0.0

    cond1 = (
        edge_max >= _EDGE_MAX_LARGE_THRESH
        and n_local < _N_LOCAL_BLOB_MAX_LARGE
        and (otsu_n >= _OTSU_N_MIN_LARGE or n_local >= 50)
    )
    cond2 = (n_local >= 40 and n_local < _N_LOCAL_BLOB_MAX_LARGE and gray_std >= _GRAY_STD_FALLBACK)
    cond3 = (n_local >= _COND3_NLOC_LO and n_local < _COND3_NLOC_HI and gray_std >= _COND3_STD_MIN)
    is_large = cond1 or cond2 or cond3
    return ("large" if is_large else "small"), float(edge_max), float(n_local), float(gray_std)


# ---------------------------------------------------------------------------
# Mini-mode local contrast filter
# ---------------------------------------------------------------------------

# How far to expand each blob to find its "ring" of surrounding background.
_MINI_CONTRAST_RING_PX: int = 7
# Minimum required (ring_mean − blob_mean) in gray_expt for a blob to be kept.
# In gray_expt, colonies are DARK (low value) and agar is BRIGHT (high value).
# The preprocessing amplifies local contrast, so genuine mini colonies show a
# clear dark-centre / bright-ring gradient.  Agar-texture patches do not.
_MINI_CONTRAST_MIN: float = 6.0


def _mini_local_contrast_filter(
    bw: np.ndarray,
    gray_expt: np.ndarray,
) -> np.ndarray:
    """
    Reject blobs that lack a genuine dark centre relative to local background.

    After the HSV + morphological pipeline, agar-texture FP that are round
    enough to pass the circularity gate are removed here: a real mini colony
    is a dark spot in gray_expt, so its surrounding ring is brighter by at
    least _MINI_CONTRAST_MIN DN.  Texture patches are locally flat → rejected.

    Blobs with no accessible ring pixels (e.g. at image border) are kept so
    we never silently drop edge colonies.
    """
    bw = bw.astype(bool)
    if not np.any(bw):
        return bw
    lab = label(bw, connectivity=2)
    keep = np.zeros_like(bw, dtype=bool)
    foot = _disk(_MINI_CONTRAST_RING_PX)
    for reg in regionprops(lab):
        blob = (lab == reg.label)
        ring = binary_dilation(blob, footprint=foot) & ~blob & (gray_expt != 0)
        if not np.any(ring):
            keep |= blob          # no ring available → keep conservatively
            continue
        blob_mean = float(np.mean(gray_expt[blob]))
        ring_mean = float(np.mean(gray_expt[ring]))
        # ring_mean > blob_mean ↔ colony core is darker than surroundings
        if (ring_mean - blob_mean) >= _MINI_CONTRAST_MIN:
            keep |= blob
    return keep


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
    _gray_std = float('inf')  # default: unknown → skip uniform-agar global close
    if _auto_adapt:
        _cfu_class, _edge_max, _n_loc, _gray_std = _blob_size_class_guess(expt)
        if _cfu_class == "large":
            params = LARGE_CFU_PARAMS
            expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat, bg_sigma=150.0)
            print(f"  [expt {k}] size-adapt → large-CFU mode "
                  f"(edge_max={_edge_max:.0f} n_local={_n_loc:.0f} gray_std={_gray_std:.1f})")
        else:
            print(f"  [expt {k}] size-adapt → small-CFU mode "
                  f"(edge_max={_edge_max:.0f} n_local={_n_loc:.0f} gray_std={_gray_std:.1f})")

    # --- BLOCK 2: adaptive HSV mask ---
    # Use the ORIGINAL image dish mask (not gray_expt) so that corner regions
    # that leak non-zero values due to the sigma=150 Gaussian background removal
    # are excluded before masking.  The preprocessed gray_expt != 0 covers the
    # entire image including zero-padded corners, causing 4 large FP corner blobs.
    dish_mask_orig = np.any(expt > 0, axis=2)
    bw_raw = hsv_mask_adaptive(expt_adj, gray_expt, params) & dish_mask_orig

    # --- BLOCK 3: morphological cleanup ---
    dish_area = int(np.count_nonzero(dish_mask_orig))
    bw_clean = morpho_cleanup(bw_raw, params, dish_area=dish_area)

    # --- Rim filter ---
    _large_mode = params.large_cfu_mode
    _rim_iters = 25 if _large_mode else 20
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
    bw_segment = _remove_edge_bar_artifacts(bw_segment) & dish_mask_orig

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
    _large_mode = params.large_cfu_mode
    for item in expts:
        expt = item["expt"]

        expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat,
                                               bg_sigma=150.0 if _large_mode else 80.0)

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

        dish_mask_orig = np.any(expt > 0, axis=2)
        bw_raw = hsv_mask_adaptive(expt_adj, gray_expt, params) & dish_mask_orig
        dish_area = int(np.count_nonzero(dish_mask_orig))
        bw_clean = morpho_cleanup(bw_raw, params, dish_area=dish_area)

        _rim_iters = 25 if _large_mode else 20
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
        bw_segment = _remove_edge_bar_artifacts(bw_segment) & dish_mask_orig

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
        _cfu_class, _edge_max, _n_loc, _gray_std = _blob_size_class_guess(expt)
        if _cfu_class == "large":
            params = _large_override or LARGE_CFU_PARAMS
            _large_cfu_mode = True
            expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat, bg_sigma=150.0)
            print(f"  [expt {k}] size-adapt → large-CFU mode "
                  f"(edge_max={_edge_max:.0f} n_local={_n_loc:.0f} gray_std={_gray_std:.1f})")
        else:
            params = _small_params
            print(f"  [expt {k}] size-adapt → small-CFU mode "
                  f"(edge_max={_edge_max:.0f} n_local={_n_loc:.0f} gray_std={_gray_std:.1f})")

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
        dish_mask_orig = np.any(expt > 0, axis=2)
        bw_raw = hsv_mask_adaptive(expt_adj, gray_expt, params) & dish_mask_orig

        # --- BLOCK 3: morphological cleanup ---
        dish_area = int(np.count_nonzero(dish_mask_orig))
        bw_clean = morpho_cleanup(bw_raw, params, dish_area=dish_area)

        # --- Rim filter ---
        # Large-CFU mode uses a narrower zone (5px) because large colonies often
        # grow right to the rim edge and a wide zone would clip them.
        # Small-CFU mode uses 20px to cover the full rim shadow depth.
        _rim_iters = 5 if _large_cfu_mode else 20
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
        bw_segment = _remove_edge_bar_artifacts(bw_segment) & dish_mask_orig
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
        # The mask is saved at *crop* dimensions (matching the overlay TIF) so
        # that the ColonyNet display in app.py can use it directly without any
        # resize.  A companion JSON sidecar records where this crop sits inside
        # the full source image so hybrid.py can re-embed it at the right offset.
        offset_y, offset_x = item.get("offset", (0, 0))
        full_H, full_W = item.get("full_shape", rgb.shape[:2])
        post_tif = Path(tempfile.gettempdir()) / (base.name + "__post.tif")
        save_tiff_rgb(post_tif, bw_post_rgb)
        out_paths.append(str(post_tif))

        import json as _json
        post_offset_json = Path(tempfile.gettempdir()) / (base.name + "__post_offset.json")
        with open(post_offset_json, "w", encoding="utf-8") as _f:
            _json.dump(
                {"y0": offset_y, "x0": offset_x, "full_H": full_H, "full_W": full_W},
                _f,
            )

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
