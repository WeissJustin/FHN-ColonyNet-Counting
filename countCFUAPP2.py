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

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hsv, rgb2gray
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import disk, binary_opening, binary_erosion, binary_dilation, h_maxima
from skimage.segmentation import watershed

# ---------------------------------------------------------------------------
# Re-use ALL helpers, dataclasses and building blocks from countCFUAPP.
# We only add the new top-level pipeline here — no duplication.
# ---------------------------------------------------------------------------
from countCFUAPP import (
    # dataclasses
    TuningParams,
    ROI,
    # image helpers
    _as_float01,
    _to_uint8,
    _gray_uint8,
    imsharpen_approx,
    _disk_kernel,
    save_tiff_rgb,
    # preprocessing
    imadjust_approx,
    locallapfilt_approx,
    localcontrast_approx,
    imlocalbrighten_approx,
    imdiffusefilt_pm,
    # mask helpers
    imextendedmin,
    bwareaopen,
    bwpropfilt,
    imoverlay,
    # helpers
    _mean_region_area,
    # dish detection / splitting
    _get_expts_from_detectdish,
)


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

    return expt_adj, gray_expt


# ===========================================================================
# BLOCK 2 — HSV adaptive masking
# ===========================================================================

def _create_hsv_mask(rgb: np.ndarray, v_thresh: float, s_min: float) -> np.ndarray:
    """
    Simple HSV threshold: V < v_thresh AND S > s_min.
    Both V and S are in [0, 1] (skimage rgb2hsv convention).
    """
    hsv = rgb2hsv(_as_float01(rgb))
    S = hsv[..., 1]
    V = hsv[..., 2]
    return (V < v_thresh) & (S > s_min)


def hsv_mask_adaptive(
    expt_adj: np.ndarray,
    gray_expt: np.ndarray,
    params: TuningParams,
) -> np.ndarray:
    """
    Adapt the V threshold so the HSV mask agrees with the extended-minima
    reference derived from the preprocessed grayscale.

    Logic (verbatim from hsv_filter() in countCFUAPP, with finer step):
      reference = imextendedmin(gray, h_ref) & (gray != 0)
      start at v_thresh = hsv_v_start
      while pixel-error > hsv_err_tol:
          v_thresh -= 0.01   (finer than original 0.05)
          if v_thresh < hsv_v_min: break

    Returns a raw boolean candidate mask (before morphological cleanup).
    """
    # Reference: dark blobs from extended minima, h adaptive to plate contrast.
    # Formula: 12% of the p10-p90 range, clamped to [10, 35].
    nz_pixels = gray_expt[gray_expt != 0]
    h_ref = int(np.clip(0.12 * (np.percentile(nz_pixels, 90) - np.percentile(nz_pixels, 10)), 10, 35))
    bw_ref = imextendedmin(gray_expt, h_ref) & (gray_expt != 0)

    dish_size = float(np.count_nonzero(gray_expt != 0))

    v_thresh = float(params.hsv_v_start)
    mask = _create_hsv_mask(expt_adj, v_thresh, params.hsv_s_min)
    mask = mask & (gray_expt != 0)

    # FP rate: fraction of dish pixels that are in mask but NOT in reference.
    # Unlike the original symmetric |mask - ref| / image_size, this only counts
    # over-prediction (agar leakage). The loop can never be tricked by missed
    # colonies (FN) into stepping too far, preventing undercounting.
    fp_rate = float(np.count_nonzero(mask & ~bw_ref)) / dish_size if dish_size > 0 else 0.0

    while fp_rate > float(params.hsv_err_tol):
        v_thresh -= 0.01
        if v_thresh < float(params.hsv_v_min):
            break
        mask = _create_hsv_mask(expt_adj, v_thresh, params.hsv_s_min)
        mask = mask & (gray_expt != 0)
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
    bw = binary_opening(bw, footprint=disk(_open_r))

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
    for i in range(1, n_blobs + 1):
        blob = bw_labeled == i
        area = int(np.count_nonzero(blob))
        r = 12 if area < 1500 else (6 if area < 8000 else 4)
        _gap_se = disk(r)
        grown = binary_dilation(blob, footprint=_gap_se)
        filled = ndi.binary_fill_holes(grown)
        shrunk = binary_erosion(filled, footprint=_gap_se)
        bw_repaired |= blob | shrunk   # union keeps original pixels
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
        big = binary_opening(big, footprint=disk(int(params.hsv_open_disk_large)))
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
    valid = [lbl for lbl in range(1, n_seeds + 1)
             if np.count_nonzero(L == lbl) >= min_area]
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
    valid = [lbl for lbl in range(1, n_seeds + 1)
             if np.count_nonzero(L == lbl) >= min_area]

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


def _count_colonies_with_instances(
    binary_image: np.ndarray,
    expt_rgb: np.ndarray,
    params: TuningParams,
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

    overlay = imoverlay(expt_rgb, bw, "green")

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
        single_areas = []
        for reg_pre in regs:
            if reg_pre.area > overall_med * 1.5:
                continue
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

        L, n_colonies, boundary_local = _conservative_watershed(region_mask, params, ref_area)

        center = np.array([reg.centroid[1], reg.centroid[0]], dtype=float)

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
            colony_roi.append(ROI(Position=poly_pos, Center=center,
                                  Creator="Algorithm", Shape="Polygon", NumOfCFU=1))
            instance_labels[lab == reg.label] = next_label
            next_label += 1
        else:
            num_cfu += n_colonies
            cluster_roi.append(ROI(Position=poly_pos, Center=center,
                                   Creator="Algorithm", Shape="Polygon", NumOfCFU=n_colonies))

            bd_full = np.zeros(overlay.shape[:2], dtype=bool)
            bd_full[minr:maxr, minc:maxc] = binary_dilation(boundary_local, footprint=disk(1))
            overlay[bd_full] = np.array([255, 255, 0], dtype=np.uint8)

            for local_label in range(1, int(L.max()) + 1):
                local_mask = (L == local_label)
                if np.any(local_mask):
                    instance_labels[minr:maxr, minc:maxc][local_mask] = next_label
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
    num_unc, _, _, _, _ = _count_colonies_with_instances(bw_unc, expt, params)
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
    if n_border > 0 and np.any(bw_clean):
        lab_rim = label(bw_clean)
        for reg in regionprops(lab_rim):
            blob_mask = lab_rim == reg.label
            if float(np.count_nonzero(blob_mask & border_px)) / n_border >= 0.20:
                bw_clean[blob_mask] = False

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
    bw_clean_cov = np.count_nonzero(bw_clean) / dish_area if dish_area > 0 else 0.0
    print(f"[2ndpass k={k}] bw_clean coverage={bw_clean_cov:.3f}")
    if dish_area > 0 and bw_clean_cov <= 0.10:
        lost = bw_raw & ~bw_clean
        print(f"[2ndpass k={k}] bw_raw={np.count_nonzero(bw_raw)} bw_clean={np.count_nonzero(bw_clean)} lost={np.count_nonzero(lost)}")
        if np.any(lost):
            bw_soft = imextendedmin(gray_expt, 15) & (gray_expt != 0)
            already_covered = binary_dilation(bw_clean, footprint=disk(10))
            uncovered = bw_soft & lost & ~already_covered
            print(f"[2ndpass k={k}] bw_soft={np.count_nonzero(bw_soft)} uncovered_in_lost={np.count_nonzero(uncovered)}")
            if np.any(uncovered):
                labeled_lost, _ = ndi.label(lost)
                keep2 = np.unique(labeled_lost[uncovered])
                keep2 = keep2[keep2 > 0]
                print(f"[2ndpass k={k}] blobs_selected={keep2.size}")
                if keep2.size > 0:
                    bw_extra = np.zeros_like(bw_clean)
                    H, W = bw_clean.shape
                    pad = 10
                    for lbl in keep2:
                        bw_extra |= labeled_lost == lbl
                    print(f"[2ndpass k={k}] bw_extra pixels={np.count_nonzero(bw_extra)}")
                    bw_clean = bw_clean | bw_extra

                    # A hole can appear at the seam between a first-pass blob and a
                    # second-pass blob (each is convex individually but together they
                    # form a ring). Fix: for every blob in bw_clean that was touched
                    # by bw_extra, do a tight-crop binary_fill_holes so the hole is
                    # enclosed and guaranteed to be filled.
                    labeled_final, _ = ndi.label(bw_clean)
                    affected = np.unique(labeled_final[bw_extra])
                    affected = affected[affected > 0]
                    for i in affected:
                        region = labeled_final == i
                        rows, cols = np.where(region)
                        r0 = max(rows.min() - pad, 0); r1 = min(rows.max() + pad + 1, H)
                        c0 = max(cols.min() - pad, 0); c1 = min(cols.max() + pad + 1, W)
                        crop = region[r0:r1, c0:c1]
                        bw_clean[r0:r1, c0:c1] |= ndi.binary_fill_holes(crop)

    # --- BLOCKS 4+5+6: candidate filtering + watershed + count/ROI ---
    # Pass the ORIGINAL (non-preprocessed) image so the green overlay lands
    # on unaltered pixel colours, not the contrast-boosted version.
    num_cfu, out_image, colony_roi, cluster_roi, _ = _count_colonies_with_instances(
        bw_clean, expt, params
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
    for k, item in enumerate(expts, start=1):
        num_cfu, _, _, _, _, _, _, _, _ = _count_one_expt_hsv(
            item["expt"], k, use_blackhat=use_blackhat, params=params
        )
        total += int(num_cfu)
    return total


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
) -> List[str]:
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

    for k, item in enumerate(expts, start=1):
        expt = item["expt"]
        _, _, _, _, out_image, _, _, bw_post_rgb = _count_one_expt_hsv(
            expt, k, use_blackhat=use_blackhat, params=params
        )

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

        # Save mask after Block 3 (post morpho_cleanup) for debugging
        post_tif = base.with_name(base.name + "__post").with_suffix(".tif")
        save_tiff_rgb(post_tif, bw_post_rgb)
        out_paths.append(str(post_tif))

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
