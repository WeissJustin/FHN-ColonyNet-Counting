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

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
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
    min_object_area: int = 150
    small_area_quantile: float = 0.75
    small_cluster_extent_min: float = 0.20
    nonsmall_cluster_ecc_min: float = 0.99
    nonsmall_cluster_extent_min: float = 0.20
    colony_circularity_min: float = 0.09
    ws_thresh_abs_frac: float = 0.02
    ws_gauss_sigma: float = 0.3
    uncountable_cutoff: int = 300
    uncountable_precheck_cutoff: int = 400


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


def _get_expts_from_detectdish(
    rgb: np.ndarray,
    dish_mode: str,
    target_area_px2: float,
) -> List[Dict[str, np.ndarray]]:
    if dish_mode == "pre_cropped":
        return [{"expt": rgb}]
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
    y_split = DetectDish.find_divider_y(
        brightness,
        ellipse_final,
        bar_q=DetectDish.BAR_Q,
        frac_thresh=DetectDish.BAR_FRAC_THRESH,
        min_thick=DetectDish.BAR_MIN_THICK_PX,
        max_thick=DetectDish.BAR_MAX_THICK_PX,
    )
    if y_split is None:
        return [{"expt": _crop_to_nonzero_bbox(masked_rgb)}]
    top_bgr, bot_bgr = DetectDish.mask_top_bottom(
        image_bgr=bgr,
        ellipse=ellipse_final,
        y_split=y_split,
        gap=DetectDish.BAR_SPLIT_MARGIN_PX,
    )
    top_rgb = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
    bot_rgb = cv2.cvtColor(bot_bgr, cv2.COLOR_BGR2RGB)
    return [{"expt": _crop_to_nonzero_bbox(top_rgb)}, {"expt": _crop_to_nonzero_bbox(bot_rgb)}]


# ===========================================================================
# BLOCK 1 — Preprocessing
# ===========================================================================

def preprocess_expt(
    expt: np.ndarray,
    use_blackhat: bool = False,
    blackhat_disk_r: int = 15,
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
    _cache_key = (id(expt), expt.shape, expt.strides, use_blackhat, blackhat_disk_r)
    _cached = _PREPROCESS_CACHE.get(_cache_key)
    if _cached is not None:
        return _cached

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
    # Large blobs are mostly solid; disk(4) is sufficient and avoids over-
    # expanding neighbours.  Per-blob processing prevents merging.
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
        r = 12 if area < 1500 else (6 if area < 8000 else 4)
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

    bw_soft = imextendedmin(gray_expt, 15) & (gray_expt != 0)
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


def _count_colonies_with_instances(
    binary_image: np.ndarray,
    expt_rgb: np.ndarray,
    params: TuningParams,
    *,
    build_overlay: bool = True,
    build_rois: bool = True,
) -> Tuple[int, np.ndarray, List[ROI], List[ROI], np.ndarray]:
    """
    Count colonies in a binary mask using per-blob watershed.

    Applies _conservative_watershed to every blob — h_maxima on the distance
    transform decides whether it is a single colony (1 seed) or a cluster
    (>1 seeds). No fragile eccentricity/extent pre-classification needed.
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
    if all_areas.size > 0:
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
        ref_area: Optional[float] = float(np.median(single_areas)) if single_areas else None

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
    params = params or TuningParams()

    # --- BLOCK 1: preprocess ---
    expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat)

    # --- Zero CFU fast-check (from original) ---
    expt_blur = cv2.GaussianBlur(expt, (0, 0), 4)
    gray0 = _gray_uint8(expt_blur)
    bw_min0 = imextendedmin(gray0, 60) & (gray0 != 0)
    bw_min0 = bwpropfilt(bw_min0, "Extent", (0.2, 1.0))
    if not np.any(bw_min0):
        return 0, "Zero", "Zero", 0.0, expt, [], [], np.zeros((*expt.shape[:2], 3), dtype=np.uint8)

    # --- Uncountable fast-check (Otsu on raw gray, same as original) ---
    gray_unc = _gray_uint8(expt)
    _, thr = cv2.threshold(gray_unc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_unc = (gray_unc < thr) & (gray_unc != 0)
    num_unc, _, _, _, _ = _count_colonies_with_instances(
        bw_unc, expt, params, build_overlay=False, build_rois=False
    )
    if num_unc > int(params.uncountable_precheck_cutoff):
        return int(num_unc), "Uncountable", "Uncountable", 0.0, np.array([]), [], [], np.zeros((*expt.shape[:2], 3), dtype=np.uint8)

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
    border_px = dish_mask_orig & ~ndi.binary_erosion(dish_mask_orig, iterations=5)
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
    dish_area = int(np.count_nonzero(gray_expt != 0))
    bw_clean = _recover_second_pass_blobs(bw_raw, bw_clean, gray_expt, dish_area)

    # --- BLOCKS 4+5+6: candidate filtering + watershed + count/ROI ---
    # Pass the ORIGINAL (non-preprocessed) image so the green overlay lands
    # on unaltered pixel colours, not the contrast-boosted version.
    num_cfu, out_image, colony_roi, cluster_roi, _ = _count_colonies_with_instances(
        bw_clean, expt, params, build_overlay=True, build_rois=build_rois
    )

    # Hard ceiling (same as original)
    if num_cfu > int(params.uncountable_cutoff):
        return int(num_cfu), "Uncountable", "Uncountable", 0.0, np.array([]), [], [], expt_adj, gray_expt

    mean_area = _mean_region_area(bw_clean)

    if mean_area <= 150:
        size_class = "small"
    elif mean_area < 400:
        size_class = "medium"
    else:
        size_class = "large"

    bw_post_rgb = cv2.cvtColor((bw_clean.astype(np.uint8) * 255), cv2.COLOR_GRAY2RGB)
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
    for item in expts:
        expt = item["expt"]

        expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat)

        expt_blur = cv2.GaussianBlur(expt, (0, 0), 4)
        gray0 = _gray_uint8(expt_blur)
        bw_min0 = imextendedmin(gray0, 60) & (gray0 != 0)
        bw_min0 = bwpropfilt(bw_min0, "Extent", (0.2, 1.0))
        if not np.any(bw_min0):
            continue

        gray_unc = _gray_uint8(expt)
        _, thr = cv2.threshold(gray_unc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_unc = (gray_unc < thr) & (gray_unc != 0)
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
        border_px = dish_mask_orig & ~ndi.binary_erosion(dish_mask_orig, iterations=5)
        n_border = int(np.count_nonzero(border_px))
        bw_clean = _filter_rim_blobs(bw_clean, border_px, n_border)
        bw_clean = _recover_second_pass_blobs(bw_raw, bw_clean, gray_expt, dish_area)

        num_cfu, _, _, _, _ = _count_colonies_with_instances(
            bw_clean, expt, params, build_overlay=False, build_rois=False
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
    params = params or TuningParams()
    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))

    total_count = 0
    all_instance_areas: List[float] = []
    mean_areas: List[float] = []
    methods: List[str] = []
    classes: List[str] = []

    for k, item in enumerate(expts, start=1):
        expt = item["expt"]

        # --- BLOCK 1: preprocess ---
        expt_adj, gray_expt = preprocess_expt(expt, use_blackhat=use_blackhat)

        # --- Zero-CFU fast-check ---
        expt_blur = cv2.GaussianBlur(expt, (0, 0), 4)
        gray0 = _gray_uint8(expt_blur)
        bw_min0 = imextendedmin(gray0, 60) & (gray0 != 0)
        bw_min0 = bwpropfilt(bw_min0, "Extent", (0.2, 1.0))
        if not np.any(bw_min0):
            methods.append("Zero")
            classes.append("Zero")
            continue

        # --- Uncountable fast-check (Otsu on raw gray) ---
        gray_unc = _gray_uint8(expt)
        _, thr = cv2.threshold(gray_unc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_unc = (gray_unc < thr) & (gray_unc != 0)
        num_unc, _, _, _, inst_unc = _count_colonies_with_instances(
            bw_unc, expt, params, build_overlay=False, build_rois=False
        )
        if num_unc > int(params.uncountable_precheck_cutoff):
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
        dish_mask_orig = np.any(expt > 0, axis=2)
        border_px = dish_mask_orig & ~ndi.binary_erosion(dish_mask_orig, iterations=5)
        n_border = int(np.count_nonzero(border_px))
        bw_clean = _filter_rim_blobs(bw_clean, border_px, n_border)

        # --- Second pass: recover blobs lost in morpho_cleanup on sparse plates ---
        bw_clean = _recover_second_pass_blobs(bw_raw, bw_clean, gray_expt, dish_area)

        # --- BLOCKS 4+5+6: count + instance labels ---
        num_cfu, _, _, _, inst_labels = _count_colonies_with_instances(
            bw_clean, expt, params, build_overlay=False, build_rois=False
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

        mean_area = _mean_region_area(bw_clean)
        mean_areas.append(float(mean_area))

        if mean_area <= 150:
            size_class = "small"
        elif mean_area < 400:
            size_class = "medium"
        else:
            size_class = "large"
        classes.append(size_class)
        methods.append("HSV")

    return {
        "count": total_count,
        "instance_areas": all_instance_areas,
        "methods": methods,
        "classes": classes,
        "mean_area": float(np.mean(mean_areas)) if mean_areas else 0.0,
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
    params = params or TuningParams()
    folder_dir = Path(folder_dir)
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / version).mkdir(parents=True, exist_ok=True)

    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))

    out_paths: List[str] = []
    counts: List[int] = []

    for k, item in enumerate(expts, start=1):
        expt = item["expt"]
        num_cfu, _, _, _, out_image, _, _, bw_post_rgb = _count_one_expt_hsv(
            expt, k, use_blackhat=use_blackhat, params=params, build_rois=False
        )
        counts.append(int(num_cfu))

        expt_id = top_cell if k == 1 else bot_cell
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

        # Save the post-cleanup mask into a hidden cache subfolder so it
        # doesn't clutter the output directory. The desktop app reads it
        # from there to seed the editable mask layer.
        post_cache_dir = base.parent / ".mask_cache"
        post_cache_dir.mkdir(parents=True, exist_ok=True)
        post_tif = post_cache_dir / (base.name + "__post.tif")
        save_tiff_rgb(post_tif, bw_post_rgb)
        out_paths.append(str(post_tif))

    if return_metadata:
        return {"out_paths": out_paths, "counts": counts}

    return out_paths


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
    args = parser.parse_args()

    input_path = Path(args.image)
    image_files = _collect_images(input_path)
    if not image_files:
        raise RuntimeError(f"No images found at: {input_path}")

    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[skip] Cannot read {img_path}")
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        stem = img_path.stem
        paths = count_cfu_app2(
            rgb=img,
            top_cell=stem,
            bot_cell=stem + "_Bottom",
            folder_dir=args.outdir,
            version=args.version,
            index=None,
            dish_mode=args.dish_mode,
            target_area_px2=args.target_area_px2,
            save_which=args.save_which,
            use_blackhat=args.blackhat,
        )
    for p in paths:
        print(p)
