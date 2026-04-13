# ------------------------------
# PETRI DISH DETECTION + MASKING
# ------------------------------
# Detects petri dish, masks everything outside, saves masked images.
# Uses contour-based ellipse detection + center refinement on edge score,
# while forcing a fixed circle diameter derived from DEFAULT_TARGET_AREA_PX2.
# Version 1.1: Also top & bottom dishes can be handled.
#
# Version: 1.1 (22-02-2026)
# ------------------------------

import argparse
import cv2
import numpy as np
import os
import glob
import re

# ------------------------------
# Defaults / User settings
# ------------------------------
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif")

DEFAULT_MAX_IMAGES = 100                 # set to 0 or negative to process all
DEFAULT_TARGET_AREA_PX2 = 2245000        # dish area in pixels^2

# Parameters used for Optimization of center position (after initial ellipse fit)
CENTER_ITERS = 40
CENTER_LR = 0.5
CENTER_FD_EPS = 3
CENTER_MAX_SHIFT_PX = 400.0

# Conservative size adaptation: allow the detected circle to be slightly smaller than
# fixed_diameter_px when strong edge evidence supports it (~5% of dishes).
# Only accepted if the edge score improves by at least CENTER_SHRINK_MIN_GAIN (10%).
# This high threshold leaves all normally-sized dishes untouched.
CENTER_SHRINK_CANDIDATES  = [0.99, 0.98, 0.97, 0.96, 0.95]  # try up to 5 % smaller
CENTER_SHRINK_MIN_GAIN    = 0.10   # require 10 % edge-score improvement to accept
CENTER_SHRINK_REFINE_ITERS = 12    # quick center re-tune iterations per candidate

# Edge scoring parameters for ellipse center optimization
EDGE_SCORE_MODE = "p75"     # "mean" | "median" | "p75"
SMOOTH_K = 7                # odd >=3; 0 disables | smoothing of edge scores on the ellipse rim to reduce noise sensitivity
N_SAMPLES = 720             # number of points to sample on the ellipse rim for edge scoring

# Params for Half bar search
BAR_Q = 15                 # Row-wise 10th percentile intensity (robust to colonies/background)
BAR_FRAC_THRESH = 0.3     # Row is "bar" if row_stat < median(row_stat) * BAR_FRAC_THRESH
BAR_MIN_THICK_PX = 10      # ignore tiny runs (noise)
BAR_MAX_THICK_PX = 120     # ignore huge dark regions (bad illumination / glove)
BAR_SPLIT_MARGIN_PX = 20   # don’t include the divider pixels in either half
BAR_BALANCE_WEIGHT = 1.35  # favor runs that split the dish into similarly sized halves
BAR_RELAXED_FRAC_MULT = 1.35
BAR_RELAXED_MIN_THICK = 6
BAR_RELAXED_MAX_THICK = 180
BAR_LINE_MAX_ABS_ANGLE_DEG = 20.0
BAR_LINE_MIN_LEN_FRAC = 0.35
BAR_LINE_SCORE_THRESH_AUTO = 1.15

# ── NEW: Dish border erosion ──────────────────────────────────────────
# Shrink the final ellipse mask inward by this many pixels so the
# bright/dark petri-dish rim is excluded from the output.
BORDER_EROSION_PX = 25          # 15-25 px works well for ~1700 px dishes

# ── NEW: Bar post-cleanup ────────────────────────────────────────────
# After the top/bottom split, scan rows near the cut boundary.
# Any row whose median brightness (inside the dish) is below
# BAR_CLEANUP_DARK_THRESH_FRAC × (dish median) is painted black.
BAR_CLEANUP_BAND_PX = 60        # how many rows from the seam to inspect
BAR_CLEANUP_DARK_THRESH_FRAC = 0.45   # row is "bar remnant" if < 45 % of dish median
BAR_ADAPTIVE_MARGIN_MULT = 1.9  # widen detected bar extent by this factor

# Grid-searched param fallbacks for initial ellipse detection
PARAM_SETS = [
    {"blur_ks": 15, "close_ks": 19},
    {"blur_ks": 7,  "close_ks": 19},
    {"blur_ks": 11, "close_ks": 19},
    {"blur_ks": 17, "close_ks": 19},
    {"blur_ks": 17, "close_ks": 21},
    {"blur_ks": 15, "close_ks": 23},
    {"blur_ks": 19, "close_ks": 25},
    {"blur_ks": 19, "close_ks": 27},
    {"blur_ks": 21, "close_ks": 29},
]


# ----------------------------
# Edge utility functions
# ----------------------------
def make_eroded_ellipse_mask(shape_hw, ellipse, erosion_px):
    """
    Draw a filled ellipse mask, then erode it inward by `erosion_px`
    so the dish rim is excluded.
    """
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)
    if erosion_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erosion_px + 1, 2 * erosion_px + 1),
        )
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask

def cleanup_bar_remnants(masked_half, brightness, ellipse,
                         seam_side="bottom",
                         band_px=BAR_CLEANUP_BAND_PX,
                         dark_frac=BAR_CLEANUP_DARK_THRESH_FRAC):
    """
    After splitting, inspect rows near the seam (the side that faced the
    divider bar).  Any row whose in-dish median brightness is suspiciously
    dark compared to the overall dish median is painted black → kills
    leftover bar pixels that the margin didn't catch.

    Parameters
    ----------
    masked_half : BGR image (already masked outside the dish)
    brightness  : single-channel brightness of the *original* image
    ellipse     : detected dish ellipse
    seam_side   : "bottom" if this is the TOP half (its dark seam is at
                  the bottom), "top" if this is the BOTTOM half.
    """
    h, w = masked_half.shape[:2]
    dish_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(dish_mask, ellipse, 255, -1)
    inside = dish_mask > 0

    # Overall dish median brightness (reference level)
    dish_vals = brightness[inside]
    if dish_vals.size == 0:
        return masked_half
    dish_median = float(np.median(dish_vals))
    if dish_median < 1.0:
        return masked_half

    thresh = dish_median * dark_frac

    # Identify the row range near the seam
    # For the TOP half, the seam is at the bottom of the visible area
    # For the BOTTOM half, the seam is at the top of the visible area
    half_mask = np.any(masked_half > 0, axis=2)  # rows that have content
    active_rows = np.where(half_mask.any(axis=1))[0]
    if active_rows.size == 0:
        return masked_half

    if seam_side == "bottom":
        # top half → seam is at the bottom edge of visible content
        scan_start = max(int(active_rows[-1]) - band_px, int(active_rows[0]))
        scan_end   = int(active_rows[-1]) + 1
    else:
        # bottom half → seam is at the top edge of visible content
        scan_start = int(active_rows[0])
        scan_end   = min(int(active_rows[0]) + band_px + 1, int(active_rows[-1]) + 1)

    # Walk row by row; black-out any that are still dark
    out = masked_half.copy()
    for y in range(scan_start, scan_end):
        row_mask = inside[y]
        if not np.any(row_mask):
            continue
        row_med = float(np.median(brightness[y, row_mask]))
        if row_med < thresh:
            out[y, :, :] = 0          # paint entire row black

    # Row medians can miss a partially slanted bar remnant because a row may
    # contain enough bright agar to lift the median. Remove dark connected
    # components in the seam band when they touch the cut side or span like a
    # bar, while leaving isolated round colonies near the seam alone.
    band_mask = np.zeros((h, w), dtype=bool)
    band_mask[scan_start:scan_end, :] = True
    dark_band = inside & band_mask & (brightness.astype(np.float32) < float(thresh))
    dark_band = cv2.morphologyEx(
        (dark_band.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
        iterations=1,
    ) > 0

    lab, n_lab = ndi_label_compat(dark_band)
    if n_lab > 0:
        if seam_side == "bottom":
            seam_rows = lab[max(scan_end - 3, scan_start):scan_end, :]
        else:
            seam_rows = lab[scan_start:min(scan_start + 3, scan_end), :]
        touch_labels = set(int(v) for v in np.unique(seam_rows) if v > 0)

        remove = np.zeros((h, w), dtype=bool)
        for lbl in range(1, n_lab + 1):
            ys, xs = np.where(lab == lbl)
            if ys.size == 0:
                continue
            comp_w = int(xs.max() - xs.min() + 1)
            comp_h = int(ys.max() - ys.min() + 1)
            spans_like_bar = comp_w >= int(0.18 * w) and comp_w >= 4 * max(1, comp_h)
            if lbl in touch_labels or spans_like_bar:
                remove |= (lab == lbl)
        out[remove] = 0
    return out


def ndi_label_compat(mask):
    """
    Small local wrapper so DetectDish stays SciPy-free; returns skimage-like
    labels using OpenCV connected components.
    """
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return lab, int(n - 1)

def ellipse_points(ellipse, n=360):
    """
    Sample n points on the perimeter of the given ellipse.
    """
    (cx, cy), (d1, d2), angle = ellipse
    a = d1 / 2.0
    b = d2 / 2.0
    theta = np.deg2rad(angle)

    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)

    cos_th, sin_th = np.cos(theta), np.sin(theta)
    xr = cos_th * x - sin_th * y + cx
    yr = sin_th * x + cos_th * y + cy
    return np.stack([xr, yr], axis=1)


def edge_map_from_brightness(brightness):
    """
    Compute edge magnitude map from brightness image using Sobel filters. Blur first to reduce noise sensitivity.
    """
    blur = cv2.GaussianBlur(brightness, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-8)
    return mag


def edge_score_for_ellipse(ellipse, edge_mag, mode="p75", n=360, smooth_k=0):
    """
    Compute an edge score for the given ellipse by sampling points on its perimeter and aggregating edge magnitudes.
    """
    pts = ellipse_points(ellipse, n=n)
    h, w = edge_mag.shape[:2]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    vals = edge_mag[ys, xs]

    # Optional smoothing of edge scores to reduce noise sensitivity
    if smooth_k and smooth_k >= 3 and len(vals) >= smooth_k:
        k = int(smooth_k)
        if k % 2 == 0:
            k += 1
        ker = np.ones(k, dtype=np.float32) / k
        vals = np.convolve(vals.astype(np.float32), ker, mode="same")

    # Aggregate edge magnitudes using the specified mode
    if mode == "mean":
        return float(vals.mean())
    if mode == "median":
        return float(np.median(vals))
    return float(np.percentile(vals, 75))

#----------------------------
# Center refinement with fixed circle diameter
#----------------------------

def snap_center_to_edges_fixed_circle(ellipse_init, brightness, fixed_diameter_px):
    """
    Force a fixed circle diameter (d1=d2=fixed_diameter_px) and optimize center (cx,cy)
    to maximize edge score on the rim.
    """
    edge_mag = edge_map_from_brightness(brightness)

    (cx0, cy0), (_, _), _ = ellipse_init
    cx, cy = float(cx0), float(cy0)

    d = float(fixed_diameter_px)
    angle = 0.0  # irrelevant for circle, optionally add later if needed for ellipse cases

    # Helper to clamp center shifts to a max amount to prevent divergence
    def clamp_center(cxx, cyy):
        dx = cxx - cx0
        dy = cyy - cy0
        r = float(np.hypot(dx, dy))
        if r <= CENTER_MAX_SHIFT_PX:
            return float(cxx), float(cyy)
        s = CENTER_MAX_SHIFT_PX / (r + 1e-9)
        return float(cx0 + s * dx), float(cy0 + s * dy)

    # Objective function for center optimization
    def obj(cxx, cyy):
        e = ((float(cxx), float(cyy)), (d, d), angle)
        return edge_score_for_ellipse(e, edge_mag, mode=EDGE_SCORE_MODE, n=N_SAMPLES, smooth_k=SMOOTH_K)

    # Initial best score at the original center
    best = (cx, cy, obj(cx, cy))

    # Gradient ascent iterations to refine center position
    for _ in range(CENTER_ITERS):
        f0 = obj(cx, cy)

        fxp = obj(cx + CENTER_FD_EPS, cy)
        fxm = obj(cx - CENTER_FD_EPS, cy)
        gx = (fxp - fxm) / (2.0 * CENTER_FD_EPS)

        fyp = obj(cx, cy + CENTER_FD_EPS)
        fym = obj(cx, cy - CENTER_FD_EPS)

        # Gradient descent step for center position
        gy = (fyp - fym) / (2.0 * CENTER_FD_EPS)

        cx_new = cx + CENTER_LR * gx
        cy_new = cy + CENTER_LR * gy

        # Clamp center shifts to prevent divergence
        cx_new, cy_new = clamp_center(cx_new, cy_new)

        f1 = obj(cx_new, cy_new)

        # backoff if worse
        if f1 < f0:
            cx_new = cx + 0.25 * CENTER_LR * gx
            cy_new = cy + 0.25 * CENTER_LR * gy
            cx_new, cy_new = clamp_center(cx_new, cy_new)
            f1 = obj(cx_new, cy_new)

        cx, cy = cx_new, cy_new

        # Update best score and corresponding center if improved
        if f1 > best[2]:
            best = (cx, cy, f1)

    # Phase 2: fine-grained pass starting from global best, smaller step + tighter FD_EPS.
    # Polishes the last few pixels of placement that the coarse phase may miss.
    cx, cy = best[0], best[1]
    for _ in range(20):
        f0 = obj(cx, cy)

        fxp = obj(cx + 1.0, cy)
        fxm = obj(cx - 1.0, cy)
        gx  = (fxp - fxm) / 2.0

        fyp = obj(cx, cy + 1.0)
        fym = obj(cx, cy - 1.0)
        gy  = (fyp - fym) / 2.0

        cx_new = cx + 0.1 * gx
        cy_new = cy + 0.1 * gy
        cx_new, cy_new = clamp_center(cx_new, cy_new)

        f1 = obj(cx_new, cy_new)
        if f1 < f0:
            cx_new = cx + 0.025 * gx
            cy_new = cy + 0.025 * gy
            cx_new, cy_new = clamp_center(cx_new, cy_new)
            f1 = obj(cx_new, cy_new)

        cx, cy = cx_new, cy_new
        if f1 > best[2]:
            best = (cx, cy, f1)

    # Final pixel-grid scan: try all ±2 px offsets from best, pick the maximum.
    # Ensures pixel-level precision after gradient ascent.
    cx_best, cy_best, _ = best
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cx_c, cy_c = clamp_center(cx_best + dx, cy_best + dy)
            s = obj(cx_c, cy_c)
            if s > best[2]:
                best = (cx_c, cy_c, s)

    cx_best, cy_best, score_best = best

    # ---- Conservative size adaptation ----------------------------------------
    # For the ~5 % of dishes that are slightly smaller than fixed_diameter_px,
    # try progressively smaller diameters. Only accept a smaller one if the edge
    # score improves by >= CENTER_SHRINK_MIN_GAIN (10 %) — a high bar that leaves
    # all normally-sized dishes completely untouched.
    d_adapted = d
    for factor in CENTER_SHRINK_CANDIDATES:
        d_try = float(d * factor)

        # Quick pre-check: is this diameter even promising with the current center?
        s_init = edge_score_for_ellipse(
            ((cx_best, cy_best), (d_try, d_try), angle),
            edge_mag, mode=EDGE_SCORE_MODE, n=N_SAMPLES, smooth_k=SMOOTH_K,
        )
        if s_init <= score_best * (1.0 + CENTER_SHRINK_MIN_GAIN):
            continue  # Not enough gain — try next smaller candidate

        # Promising candidate: quick center re-tune for this diameter
        cx_t, cy_t, s_t = cx_best, cy_best, s_init
        for _ in range(CENTER_SHRINK_REFINE_ITERS):
            sxp = edge_score_for_ellipse(((cx_t + 1.0, cy_t), (d_try, d_try), angle), edge_mag, mode=EDGE_SCORE_MODE, n=N_SAMPLES, smooth_k=SMOOTH_K)
            sxm = edge_score_for_ellipse(((cx_t - 1.0, cy_t), (d_try, d_try), angle), edge_mag, mode=EDGE_SCORE_MODE, n=N_SAMPLES, smooth_k=SMOOTH_K)
            syp = edge_score_for_ellipse(((cx_t, cy_t + 1.0), (d_try, d_try), angle), edge_mag, mode=EDGE_SCORE_MODE, n=N_SAMPLES, smooth_k=SMOOTH_K)
            sym = edge_score_for_ellipse(((cx_t, cy_t - 1.0), (d_try, d_try), angle), edge_mag, mode=EDGE_SCORE_MODE, n=N_SAMPLES, smooth_k=SMOOTH_K)
            cx_n, cy_n = clamp_center(cx_t + 0.1 * (sxp - sxm) / 2.0,
                                      cy_t + 0.1 * (syp - sym) / 2.0)
            s_n = edge_score_for_ellipse(((cx_n, cy_n), (d_try, d_try), angle), edge_mag, mode=EDGE_SCORE_MODE, n=N_SAMPLES, smooth_k=SMOOTH_K)
            if s_n > s_t:
                cx_t, cy_t, s_t = cx_n, cy_n, s_n

        # Final confirmation: must still clear the threshold after re-tuning
        if s_t > score_best * (1.0 + CENTER_SHRINK_MIN_GAIN):
            cx_best, cy_best, score_best = cx_t, cy_t, s_t
            d_adapted = d_try
            print(f"  [size-adapt] d×{factor:.2f} = {d_try:.1f}px accepted (score {s_t:.4f})")
            break  # take the most conservative (largest) passing candidate; stop here

    ellipse_best = ((float(cx_best), float(cy_best)), (d_adapted, d_adapted), angle)
    return ellipse_best, float(score_best)


# ----------------------------
# Initial ellipse detection (done via contour detection)
# ----------------------------
def detect_ellipse_with_thresholding(image_bgr, fixed_diameter_px):
    """
    Detect an initial ellipse by using multiple thresholding strategies (Otsu's & Canny & HoughCircles) and picking the best valid contour.
    """
    h, w = image_bgr.shape[:2]
    img_area = float(h * w)

    def try_with_params(p):
        """
        Thresholding strategy: 
        blur -> compute brightness -> Otsu's threshold (binary + inv) -> 
        contour detection + filtering by area/circularity -> pick best contour -> 
        convex hull -> fit ellipse, use current parameters. If none is found, use Canny edge-based contours as a fallback.
        If that doesn't work either use HoughCircles (ONLY AVAILABLE IF FIXED CIRCLE DIAMETER IS KNOWN).
        """
        blurred = cv2.GaussianBlur(image_bgr, (p["blur_ks"], p["blur_ks"]), 0)

        # Standard brightness weights
        brightness = (
            0.299 * blurred[:, :, 2] +
            0.587 * blurred[:, :, 1] +
            0.114 * blurred[:, :, 0]
        ).astype(np.uint8)

        #Kernel for contour cleaning, size from params
        kernel = np.ones((p["close_ks"], p["close_ks"]), np.uint8)

        def best_contour_from_thresh(thresh_img):
            """
             Given a binary thresholded image, find contours via Otsu's thresholding, filter by area and circularity,
             and return the best valid contour.
             """
            # Morph close to clean up noise and connect edges
            thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
            # Find contours
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            valid = []
            for cnt in contours:
                # Filter contours by area
                area = cv2.contourArea(cnt)
                if area <= 0:
                    continue

                area_frac = area / img_area
                if area_frac < 0.30 or area_frac > 0.90:
                    continue

                # Filter contours by circularity (4*pi*area/per^2)
                per = cv2.arcLength(cnt, True)
                if per <= 0:
                    continue

                circ = 4 * np.pi * area / (per ** 2)
                if not (0.75 < circ < 1.2):
                    continue

                valid.append(cnt)

            if not valid:
                return None

            return max(valid, key=cv2.contourArea)

        #Threshold brightness using Otsu's method (both binary and inverse) and try to find valid contours
        _, thresh_bin = cv2.threshold(brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_inv = cv2.threshold(brightness, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        cnt1 = best_contour_from_thresh(thresh_bin)
        cnt2 = best_contour_from_thresh(thresh_inv)

        chosen = None
        if cnt1 is None and cnt2 is None:
            # If no valid contour found with Otsu's thresholds, try Canny edge thresholds as a fallback

            # Auto Canny thresholds based on image median
            v = float(np.median(brightness))
            sigma1 = 0.33
            low1 = int(max(0, (1.0 - sigma1) * v))
            high1 = int(min(255, (1.0 + sigma1) * v))

            # A second, slightly different pair of thresholds to increase chances of finding contour
            sigma2 = 0.50
            low2 = int(max(0, (1.0 - sigma2) * v))
            high2 = int(min(255, (1.0 + sigma2) * v))

            edges1 = cv2.Canny(brightness, low1, high1)
            edges2 = cv2.Canny(brightness, low2, high2)

            # Thicken edges so morphology/contours behave more like binary masks
            small = np.ones((3, 3), np.uint8)
            edges1 = cv2.dilate(edges1, small, iterations=1)
            edges2 = cv2.dilate(edges2, small, iterations=1)

            cnt3 = best_contour_from_thresh(edges1)
            cnt4 = best_contour_from_thresh(edges2)

            # Pick the best contour among the two adaptive threshold results if Otsu's method failed
            if cnt3 is None and cnt4 is None:
                # Third fallback: HoughCircles constrained by expected radius from target area -> ONLY AVAILABLE IF FIXED DIAMETER AVAILABLE
                expected_r = int(round(float(fixed_diameter_px) * 0.5))
                r_min = max(5, int(round(0.90 * expected_r)))
                r_max = max(r_min + 1, int(round(1.10 * expected_r)))

                # Smoothing
                blur2 = cv2.GaussianBlur(brightness, (9, 9), 1.5)

                # Apply HoughCircles
                circles = cv2.HoughCircles(
                    blur2,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=min(h, w) * 0.25,
                    param1=120,          # Canny high threshold internally
                    param2=30,           # accumulator threshold
                    minRadius=r_min,
                    maxRadius=r_max,
                )

                if circles is None:
                    return None, None

                circles = np.round(circles[0, :]).astype(int)

                # pick the circle closest to image center
                cx_img = w * 0.5
                cy_img = h * 0.5
                best_c = None
                best_d = None
                for (ccx, ccy, rr) in circles:
                    dcen = (ccx - cx_img) ** 2 + (ccy - cy_img) ** 2
                    if best_d is None or dcen < best_d:
                        best_d = dcen
                        best_c = (ccx, ccy, rr)

                ccx, ccy, rr = best_c
                ellipse = ((float(ccx), float(ccy)), (float(2 * rr), float(2 * rr)), 0.0)
                return ellipse, brightness

            if cnt4 is None:
                chosen = cnt3
            elif cnt3 is None:
                chosen = cnt4
            else:
                chosen = cnt3 if cv2.contourArea(cnt3) >= cv2.contourArea(cnt4) else cnt4
        else:
            if cnt2 is None:
                chosen = cnt1
            elif cnt1 is None:
                chosen = cnt2
            else:
                chosen = cnt1 if cv2.contourArea(cnt1) >= cv2.contourArea(cnt2) else cnt2

        if chosen is None or len(chosen) < 5:
            return None, None

        # Fit ellipse to the convex hull of the chosen contour to get a smoother initial ellipse
        hull = cv2.convexHull(chosen)
        if hull is None or len(hull) < 5:
            return None, None

        try:
            # Fit ellipse
            ellipse = cv2.fitEllipse(hull)
            return ellipse, brightness
        except cv2.error:
            return None, None

    for p in PARAM_SETS:
        # Call function to try ellipse detection with the current set of parameters,
        # if none of the parameter sets yield a valid ellipse, return None
        ellipse, brightness = try_with_params(p)
        if ellipse is not None:
            return ellipse, brightness, p

    return None, None, None


def detect_plate_rgb(image_bgr, fixed_diameter_px):
    """
    Detect petri dish, mask everything outside (with border erosion).
    """
    h, w = image_bgr.shape[:2]

    ellipse_init, brightness, used_params = detect_ellipse_with_thresholding(
        image_bgr, fixed_diameter_px=fixed_diameter_px
    )
    if ellipse_init is None:
        return None, None, None, None

    ellipse_final, edge_score = snap_center_to_edges_fixed_circle(
        ellipse_init, brightness, fixed_diameter_px=fixed_diameter_px
    )

    (cx, cy), (d1, d2), _ = ellipse_final
    area = np.pi * (d1 * 0.5) * (d2 * 0.5)
    print(f"circle: cx={cx:.1f} cy={cy:.1f} d={d1:.1f} "
          f"area={area:.0f} edge={edge_score:.4f}")

    # ── NEW: eroded mask keeps the dish rim out ──
    mask = make_eroded_ellipse_mask((h, w), ellipse_final, BORDER_EROSION_PX)
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    return masked, ellipse_final, brightness, used_params
# ----------------------------
# Petri dish top & bottom detection
# ----------------------------
def derive_top_bottom_filenames(original_filename: str):
    """
    Derive top and bottom dish filenames from the original filename.
    Example:
      DIFIP1_473Ndilu0_475Ndilu0.tif -> (DIFIP1_473Ndilu0.tif, DIFIP1_475Ndilu0.tif)
    Rule:
      prefix_B_C.ext -> top=prefix_B.ext, bottom=prefix_C.ext  (uses last two underscore tokens)
    Fallback:
      <stem>_top.ext and <stem>_bottom.ext
    """
    base = os.path.basename(original_filename)
    stem, ext = os.path.splitext(base)
    parts = stem.split("_")

    if len(parts) >= 3:
        prefix = "_".join(parts[:-2])
        b = parts[-2]
        c = parts[-1]
        top = f"{prefix}_{b}{ext}"
        bottom = f"{prefix}_{c}{ext}"
        return top, bottom

    # fallback
    return f"{stem}_top{ext}", f"{stem}_bottom{ext}"


def find_divider_y(brightness, ellipse_final, bar_q=10, frac_thresh=0.35,
                   min_thick=20, max_thick=120):
    """
    Search for a horizontal black divider inside the ellipse.
    Returns y_split (int) or None if not found.
    """
    h, w = brightness.shape[:2]

    # ellipse mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse_final, 255, -1)
    inside = mask.astype(bool)

    # row-wise low-percentile statistic inside the dish
    row_stat = np.full(h, np.nan, dtype=np.float32)
    for y in range(h):
        m = inside[y]
        if np.any(m):
            row_stat[y] = np.percentile(brightness[y, m], bar_q)

    valid = ~np.isnan(row_stat)
    if not np.any(valid):
        return None

    med = float(np.median(row_stat[valid]))
    thr = med * float(frac_thresh)

    is_bar = valid & (row_stat < thr)

    # Collect contiguous dark runs that look like divider candidates.
    runs = []
    start = None
    for i, v in enumerate(is_bar):
        if v and start is None:
            start = i
        elif (not v) and (start is not None):
            end = i - 1
            length = end - start + 1
            if min_thick <= length <= max_thick:
                runs.append((start, end, length))
            start = None
    if start is not None:
        end = h - 1
        length = end - start + 1
        if min_thick <= length <= max_thick:
            runs.append((start, end, length))

    if not runs:
        return None

    # Score each run by:
    # 1) darkness (divider should be dark),
    # 2) thickness sanity,
    # 3) top/bottom area balance inside ellipse (halves often similar size).
    inside_counts = inside.sum(axis=1).astype(np.float32)
    csum = np.cumsum(inside_counts)
    total_inside = float(csum[-1]) if csum.size else 0.0
    if total_inside <= 1e-6:
        return None

    def prefix_sum(y_end_inclusive: int) -> float:
        if y_end_inclusive < 0:
            return 0.0
        if y_end_inclusive >= h:
            return float(csum[-1])
        return float(csum[y_end_inclusive])

    # For darkness normalization; guards divide-by-zero if thr becomes very small.
    denom_dark = max(1e-6, thr)

    best_score = None
    best_run = None
    for y0, y1, length in runs:
        y_split = int(round(0.5 * (y0 + y1)))

        top_area = prefix_sum(y0 - 1)
        bottom_area = total_inside - prefix_sum(y1)
        if top_area <= 0.0 or bottom_area <= 0.0:
            continue

        # 0..1 (1 = perfectly balanced).
        balance = min(top_area, bottom_area) / (max(top_area, bottom_area) + 1e-6)

        # Darker rows -> larger score.
        run_dark = float(np.nanmean(row_stat[y0:y1 + 1]))
        darkness = max(0.0, (thr - run_dark) / denom_dark)

        # Prefer runs near expected thickness scale to suppress tiny flicker bars.
        thick_score = min(1.0, float(length) / max(1.0, float(min_thick)))

        score = darkness + (BAR_BALANCE_WEIGHT * balance) + (0.25 * thick_score)
        if (best_score is None) or (score > best_score):
            best_score = score
            best_run = (y0, y1)

    if best_run is None:
        # Fallback: keep old behavior if scoring could not evaluate candidates.
        y0, y1, _ = max(runs, key=lambda r: r[2])
    else:
        y0, y1 = best_run

    return int(y0), int(y1)


def find_divider_y_balance_fallback(brightness, ellipse_final, bar_q=BAR_Q):
    """
    Fallback divider search:
    scan y around ellipse center and score candidates by
    1) top/bottom area balance and
    2) darkness on the divider row.
    """
    h, w = brightness.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse_final, 255, -1)
    inside = mask.astype(bool)

    row_stat = np.full(h, np.nan, dtype=np.float32)
    row_cnt = inside.sum(axis=1).astype(np.float32)
    for y in range(h):
        m = inside[y]
        if np.any(m):
            row_stat[y] = np.percentile(brightness[y, m], bar_q)

    valid_rows = np.where((~np.isnan(row_stat)) & (row_cnt > 0))[0]
    if valid_rows.size == 0:
        return None

    csum = np.cumsum(row_cnt)
    total_inside = float(csum[-1]) if csum.size else 0.0
    if total_inside <= 1e-6:
        return None

    med_dark = float(np.nanmedian(row_stat[valid_rows]))
    dark_norm = max(1e-6, med_dark)
    cy = int(round(float(ellipse_final[0][1])))
    band_half = max(30, int(0.35 * float(ellipse_final[1][1])))
    y_lo = max(int(valid_rows[0]), cy - band_half)
    y_hi = min(int(valid_rows[-1]), cy + band_half)
    if y_lo >= y_hi:
        y_lo, y_hi = int(valid_rows[0]), int(valid_rows[-1])

    best_y = None
    best_score = None
    for y in range(y_lo, y_hi + 1):
        if not np.isfinite(row_stat[y]) or row_cnt[y] <= 0:
            continue

        top_area = float(csum[y - 1]) if y > 0 else 0.0
        bottom_area = total_inside - float(csum[y])
        if top_area <= 0.0 or bottom_area <= 0.0:
            continue

        balance = min(top_area, bottom_area) / (max(top_area, bottom_area) + 1e-6)
        darkness = max(0.0, (med_dark - float(row_stat[y])) / dark_norm)
        center_prior = 1.0 - min(1.0, abs(y - cy) / (band_half + 1e-6))
        score = (1.45 * balance) + (0.55 * darkness) + (0.20 * center_prior)

        if best_score is None or score > best_score:
            best_score = score
            best_y = y

    if best_y is None:
        return None
    return int(best_y), int(best_y)


def looks_like_double_name(filename: str) -> bool:
    """
    True for names like DIFIP1_1025Ndilu2_1026Ndilu0.tif (unsplit pair-like names).
    """
    stem, _ = os.path.splitext(os.path.basename(filename))
    parts = stem.split("_")
    if len(parts) < 3:
        return False
    return bool(re.search(r"dilu\d+$", parts[-1], re.IGNORECASE) and re.search(r"dilu\d+$", parts[-2], re.IGNORECASE))


def split_balance_only_y(image_shape, ellipse_final):
    """
    Always-return fallback split based purely on top/bottom area balance inside ellipse.
    Returns y or None.
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse_final, 255, -1)
    inside_counts = (mask > 0).sum(axis=1).astype(np.float32)
    csum = np.cumsum(inside_counts)
    if csum.size == 0 or float(csum[-1]) <= 1e-6:
        return None

    total = float(csum[-1])
    cy = int(round(float(ellipse_final[0][1])))
    band_half = max(30, int(0.45 * float(ellipse_final[1][1])))
    y_lo = max(1, cy - band_half)
    y_hi = min(h - 2, cy + band_half)
    if y_lo >= y_hi:
        y_lo, y_hi = 1, h - 2

    best_y = None
    best_balance = -1.0
    for y in range(y_lo, y_hi + 1):
        top_area = float(csum[y - 1]) if y > 0 else 0.0
        bottom_area = total - float(csum[y])
        if top_area <= 0.0 or bottom_area <= 0.0:
            continue
        balance = min(top_area, bottom_area) / (max(top_area, bottom_area) + 1e-6)
        if balance > best_balance:
            best_balance = balance
            best_y = y

    if best_y is None:
        return None
    return int(best_y), int(best_y)


def _line_balance_score(h: int, w: int, ellipse, p1, p2):
    """
    Compute top/bottom area balance for a near-horizontal divider line.
    """
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return 0.0

    m = (y2 - y1) / dx
    b = y1 - m * x1

    dish_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(dish_mask, ellipse, 255, -1)
    dish = dish_mask > 0

    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    y_line = m * xx + b

    at = float((dish & (yy < y_line)).sum())
    ab = float((dish & (yy > y_line)).sum())
    if at <= 0.0 or ab <= 0.0:
        return 0.0
    return min(at, ab) / (max(at, ab) + 1e-6)


def find_divider_line_dark(brightness, ellipse_final):
    """
    Detect dominant dark divider line in dish center.
    Returns (p1, p2, score) or None.
    """
    h, w = brightness.shape[:2]
    cx, cy = float(ellipse_final[0][0]), float(ellipse_final[0][1])
    d_est = float(max(ellipse_final[1][0], ellipse_final[1][1]))

    dish_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(dish_mask, ellipse_final, 255, -1)
    inside = dish_mask > 0

    band_half = max(30, int(0.35 * d_est))
    y0 = max(0, int(round(cy)) - band_half)
    y1 = min(h, int(round(cy)) + band_half + 1)
    band = np.zeros((h, w), dtype=bool)
    band[y0:y1, :] = True

    vals = brightness[inside & band]
    if vals.size < 200:
        return None
    thr = float(np.percentile(vals, 15))
    dark = (brightness <= thr) & inside & band
    dark_u8 = (dark.astype(np.uint8) * 255)
    dark_u8 = cv2.morphologyEx(
        dark_u8, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
        iterations=2,
    )

    min_len = max(30, int(BAR_LINE_MIN_LEN_FRAC * d_est))
    lines = cv2.HoughLinesP(
        dark_u8, 1, np.pi / 180.0, threshold=35,
        minLineLength=min_len, maxLineGap=25
    )
    if lines is None:
        return None

    best = None
    for ln in lines[:, 0, :]:
        x1, y1l, x2, y2l = [int(v) for v in ln]
        dx = float(x2 - x1)
        dy = float(y2l - y1l)
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue

        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) > BAR_LINE_MAX_ABS_ANGLE_DEG:
            continue

        mid_y = 0.5 * (y1l + y2l)
        center_pen = abs(mid_y - cy) / max(1.0, 0.30 * d_est)
        if center_pen > 1.0:
            continue

        balance = _line_balance_score(h, w, ellipse_final, (x1, y1l), (x2, y2l))
        len_score = length / max(1.0, 0.65 * d_est)
        angle_score = 1.0 - min(1.0, abs(angle) / BAR_LINE_MAX_ABS_ANGLE_DEG)
        score = (0.95 * len_score) + (0.95 * balance) + (0.35 * angle_score) - (0.55 * center_pen)

        if (best is None) or (score > best[2]):
            best = ((x1, y1l), (x2, y2l), float(score))

    return best

def mask_top_bottom_from_line(image_bgr, ellipse, p1, p2,
                               gap=BAR_SPLIT_MARGIN_PX,
                               bar_half_width=BAR_MAX_THICK_PX // 2,
                               brightness=None):
    h, w = image_bgr.shape[:2]
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    if abs(dx) < 1e-6:
        return None

    m = (y2 - y1) / dx
    b = y1 - m * x1

    # ── Eroded dish mask ──
    dish_mask = make_eroded_ellipse_mask((h, w), ellipse, BORDER_EROSION_PX)
    dish = dish_mask > 0

    xx = np.arange(w, dtype=np.float32)[None, :]
    yy = np.arange(h, dtype=np.float32)[:, None]
    y_line = m * xx + b

    half = float(bar_half_width + gap)
    top    = dish & (yy < (y_line - half))
    bottom = dish & (yy > (y_line + half))

    top_mask    = (top.astype(np.uint8) * 255)
    bottom_mask = (bottom.astype(np.uint8) * 255)
    masked_top    = cv2.bitwise_and(image_bgr, image_bgr, mask=top_mask)
    masked_bottom = cv2.bitwise_and(image_bgr, image_bgr, mask=bottom_mask)

    # ── Post-cleanup ──
    if brightness is not None:
        masked_top    = cleanup_bar_remnants(
            masked_top, brightness, ellipse, seam_side="bottom")
        masked_bottom = cleanup_bar_remnants(
            masked_bottom, brightness, ellipse, seam_side="top")

    return masked_top, masked_bottom


def mask_top_bottom(image_bgr, ellipse, y_bar_top, y_bar_bot,
                    margin=BAR_SPLIT_MARGIN_PX, brightness=None):
    """
    Split the dish into top/bottom halves.
    • Uses an ADAPTIVE margin derived from the actual bar thickness.
    • Applies BORDER_EROSION_PX so the rim is excluded.
    • Runs cleanup_bar_remnants() on each half to scrub leftover dark rows.
    """
    h, w = image_bgr.shape[:2]

    # ── Adaptive margin: widen proportionally to bar thickness ──
    bar_thickness = max(1, y_bar_bot - y_bar_top)
    adaptive_margin = max(
        margin,
        int(round(bar_thickness * BAR_ADAPTIVE_MARGIN_MULT * 0.5)),
    )

    # ── Eroded dish mask (no rim) ──
    dish_mask = make_eroded_ellipse_mask((h, w), ellipse, BORDER_EROSION_PX)

    y_black_start = max(0, y_bar_top - adaptive_margin)
    y_black_end   = min(h, y_bar_bot + adaptive_margin + 1)

    top_mask = dish_mask.copy()
    top_mask[y_black_start:, :] = 0

    bottom_mask = dish_mask.copy()
    bottom_mask[:y_black_end, :] = 0

    masked_top    = cv2.bitwise_and(image_bgr, image_bgr, mask=top_mask)
    masked_bottom = cv2.bitwise_and(image_bgr, image_bgr, mask=bottom_mask)

    # ── Post-cleanup: scrub any leftover dark bar rows near the seam ──
    if brightness is not None:
        masked_top    = cleanup_bar_remnants(
            masked_top, brightness, ellipse, seam_side="bottom")
        masked_bottom = cleanup_bar_remnants(
            masked_bottom, brightness, ellipse, seam_side="top")

    return masked_top, masked_bottom


# ----------------------------
# Main function
# ----------------------------

def list_images(input_dir: str):
    """
    Helper function to list all image files in the input directory with specified extensions.
    """
    all_files = glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)
    return [f for f in all_files if f.lower().endswith(IMAGE_EXTENSIONS)]


def main():
    """
    Main function to parse arguments, process images in the input directory, detect petri dishes, mask them, and save results.
    """
    # Argument parsing
    ap = argparse.ArgumentParser(description="Detect petri dish and mask images.")
    ap.add_argument("--input_dir", required=True, help="Input folder containing images.")
    ap.add_argument("--out_dir", required=True, help="Output folder for masked images.")
    ap.add_argument("--max_images", type=int, default=DEFAULT_MAX_IMAGES,
                    help="Max images to process. Set <=0 to process all.")
    ap.add_argument("--target_area_px2", type=float, default=DEFAULT_TARGET_AREA_PX2,
                    help="Target dish area in pixels^2 (used to derive fixed circle diameter).")
    ap.add_argument("--split_halves", default="auto",
                    help="Split mode: auto | yes | no. auto decides per image.")
    args = ap.parse_args()

    # Input/output directory setup
    input_dir = os.path.expanduser(args.input_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Derive fixed circle diameter from target area
    target_area = float(args.target_area_px2)
    fixed_diameter_px = 2.0 * np.sqrt(target_area / np.pi)

    split_mode = str(args.split_halves).strip().lower()
    if split_mode in ("y", "yes", "true", "1"):
        split_mode = "yes"
    elif split_mode in ("n", "no", "false", "0"):
        split_mode = "no"
    elif split_mode not in ("auto", "yes", "no"):
        split_mode = "auto"
    split_halves = split_mode != "no"

    # Parse image paths from input directory, optionally limit to max_images
    image_paths = list_images(input_dir)
    if args.max_images is not None and args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    # Print summary of settings and number of images to process
    print(f"Input:  {input_dir}")
    print(f"Output: {out_dir}")
    print(f"Found {len(image_paths)} images to process.")
    print(f"Fixed diameter from area={target_area:.0f}: d={fixed_diameter_px:.2f}px")
    print(f"Split mode: {split_mode}")

    no_contour_count = 0
    used_fallback_count = 0
    unsplit_pair_name_count = 0
    unsplit_pair_files_run = []
    removed_stale_unsplit = 0
    line_split_count = 0

    # Process each image: detect petri dish, mask, and save result
    for i, path in enumerate(image_paths, 1):
        print(f"[{i}] Processing: {path}")
        image = cv2.imread(path)
        if image is None:
            print("Could not load image.")
            continue

        # Detect petri dish, mask, and get used parameters
        masked, ellipse_final, brightness, used_params = detect_plate_rgb(image, fixed_diameter_px=fixed_diameter_px)

        # If no ellipse (dish) detected, count and skip saving
        if ellipse_final is None:
            no_contour_count += 1
            continue

        # If the used parameters for initial ellipse detection are not the first/default set, count it as a fallback usage
        if used_params != PARAM_SETS[0]:
            used_fallback_count += 1

        # If halves are used, after fitting ellipse: Search for black bar in Ellipse.
        if split_halves:
            base_name = os.path.basename(path)
            is_pair_name = looks_like_double_name(base_name)
            split_this = (split_mode == "yes") or (split_mode == "auto" and is_pair_name)
            split_result = None

            # Dominant dark-line detector (handles small tilt without forcing all images).
            line_candidate = find_divider_line_dark(brightness, ellipse_final)
            if line_candidate is not None:
                p1, p2, line_score = line_candidate
                if split_mode == "yes" or line_score >= BAR_LINE_SCORE_THRESH_AUTO:
                    split_result = mask_top_bottom_from_line(
                        image_bgr=image,
                        ellipse=ellipse_final,
                        p1=p1,
                        p2=p2,
                        gap=BAR_SPLIT_MARGIN_PX,
                        bar_half_width=BAR_MAX_THICK_PX // 2,
                        brightness=brightness, 
                    )
                    if split_result is not None:
                        split_this = True
                        line_split_count += 1

            bar_extents = None
            if split_result is None and split_this:
                bar_extents = find_divider_y(
                    brightness,
                    ellipse_final,
                    bar_q=BAR_Q,
                    frac_thresh=BAR_FRAC_THRESH,
                    min_thick=BAR_MIN_THICK_PX,
                    max_thick=BAR_MAX_THICK_PX,
                )

            # Relaxed pass for hard cases.
            if split_result is None and split_this and bar_extents is None:
                bar_extents = find_divider_y(
                    brightness,
                    ellipse_final,
                    bar_q=BAR_Q,
                    frac_thresh=min(0.90, BAR_FRAC_THRESH * BAR_RELAXED_FRAC_MULT),
                    min_thick=BAR_RELAXED_MIN_THICK,
                    max_thick=BAR_RELAXED_MAX_THICK,
                )

            # Final fallback: choose y by area-balance + row darkness score.
            if split_result is None and split_this and bar_extents is None:
                bar_extents = find_divider_y_balance_fallback(brightness, ellipse_final, bar_q=BAR_Q)

            # If still not found but filename strongly indicates two dishes,
            # force split by geometric area balance.
            if split_result is None and split_this and bar_extents is None and is_pair_name:
                bar_extents = split_balance_only_y(brightness.shape[:2], ellipse_final)
                if bar_extents is not None:
                    y_ctr = (bar_extents[0] + bar_extents[1]) // 2
                    print(f"[INFO] Forced split by area-balance for pair-like file: {base_name} at y={y_ctr}")

            # When a line-based split is already prepared, refine its gap using the
            # row-stat bar extent if available.
            if split_result is not None and bar_extents is not None:
                y0_bar, y1_bar = bar_extents
                bar_half = max(BAR_SPLIT_MARGIN_PX, (y1_bar - y0_bar) // 2 + BAR_SPLIT_MARGIN_PX)
                p1_line, p2_line = (line_candidate[0], line_candidate[1]) if line_candidate is not None else (None, None)
                if p1_line is not None:
                    refined = mask_top_bottom_from_line(
                        image_bgr=image,
                        ellipse=ellipse_final,
                        p1=p1_line,
                        p2=p2_line,
                        gap=BAR_SPLIT_MARGIN_PX,
                        bar_half_width=bar_half,
                        brightness=brightness, 
                    )
                    if refined is not None:
                        split_result = refined

            if split_result is None and split_this and bar_extents is not None:
                y0_bar, y1_bar = bar_extents
                split_result = mask_top_bottom(
                    image_bgr=image,
                    ellipse=ellipse_final,
                    y_bar_top=y0_bar,
                    y_bar_bot=y1_bar,
                    margin=BAR_SPLIT_MARGIN_PX,
                    brightness=brightness, 
                )

            if split_result is not None:
                masked_top, masked_bottom = split_result
                top_name, bottom_name = derive_top_bottom_filenames(os.path.basename(path))

                save_path_top = os.path.join(out_dir, top_name)
                save_path_bottom = os.path.join(out_dir, bottom_name)

                cv2.imwrite(save_path_top, masked_top)
                cv2.imwrite(save_path_bottom, masked_bottom)

                # Cleanup stale unsplit file from older runs, if present.
                original_name = os.path.basename(path)
                if looks_like_double_name(original_name):
                    stale_path = os.path.join(out_dir, original_name)
                    if os.path.exists(stale_path):
                        try:
                            os.remove(stale_path)
                            removed_stale_unsplit += 1
                        except Exception:
                            print(f"[WARN] Could not remove stale unsplit file: {stale_path}")
                continue

        # Save image (normal single-mask case)
        filename = os.path.basename(path)
        save_path = os.path.join(out_dir, filename)
        cv2.imwrite(save_path, masked)
        if split_halves and looks_like_double_name(filename):
            unsplit_pair_name_count += 1
            unsplit_pair_files_run.append(filename)

    # Final Summary of results
    total = len(image_paths)
    print("\nDone.")
    print(f"No contour detected in {no_contour_count} out of {total} images.")
    if total > 0:
        print(f"Success rate: {(1 - no_contour_count / total) * 100:.2f}%")
        ok = max(0, total - no_contour_count)
        print(f"Used fallback params in {used_fallback_count} out of {ok} successful detections.")
    if split_halves:
        print(f"Used dark-line split: {line_split_count}")
        print(f"Removed stale unsplit pair-like files during run: {removed_stale_unsplit}")
        print(f"Unsplit pair-like filenames in current run: {unsplit_pair_name_count}")
        if unsplit_pair_files_run:
            preview_run = ", ".join(sorted(unsplit_pair_files_run)[:10])
            if len(unsplit_pair_files_run) > 10:
                preview_run += ", ..."
            print(f"[WARN] Current-run unsplit pair-like: {preview_run}")

        out_files = list_images(out_dir)
        double_name_files = [os.path.basename(p) for p in out_files if looks_like_double_name(p)]

        # Final stale cleanup pass: if split outputs already exist, remove pair-like original.
        removed_stale_post = 0
        for fn in double_name_files:
            top_name, bottom_name = derive_top_bottom_filenames(fn)
            p_top = os.path.join(out_dir, top_name)
            p_bottom = os.path.join(out_dir, bottom_name)
            p_orig = os.path.join(out_dir, fn)
            if os.path.exists(p_top) and os.path.exists(p_bottom):
                try:
                    os.remove(p_orig)
                    removed_stale_post += 1
                except Exception:
                    print(f"[WARN] Could not remove stale pair-like file in final cleanup: {p_orig}")

        if removed_stale_post > 0:
            print(f"Removed stale pair-like files in final cleanup: {removed_stale_post}")
            out_files = list_images(out_dir)
            double_name_files = [os.path.basename(p) for p in out_files if looks_like_double_name(p)]

        # If current run had zero unsplit pair-like files, any remaining pair-like outputs
        # are stale leftovers from older runs. Purge them so out_dir reflects current run.
        purged_leftovers = 0
        if unsplit_pair_name_count == 0 and double_name_files:
            for fn in list(double_name_files):
                p_orig = os.path.join(out_dir, fn)
                if os.path.exists(p_orig):
                    try:
                        os.remove(p_orig)
                        purged_leftovers += 1
                    except Exception:
                        print(f"[WARN] Could not purge stale leftover pair-like file: {p_orig}")
            if purged_leftovers > 0:
                print(f"Purged stale leftover pair-like files (not from current run): {purged_leftovers}")
                out_files = list_images(out_dir)
                double_name_files = [os.path.basename(p) for p in out_files if looks_like_double_name(p)]

        print(f"Possible unsplit pair-name outputs in out_dir: {len(double_name_files)}")
        if double_name_files:
            preview = ", ".join(sorted(double_name_files)[:10])
            if len(double_name_files) > 10:
                preview += ", ..."
            print(f"[WARN] Unsplit-like files: {preview}")

# Call main function
if __name__ == "__main__":
    main()
