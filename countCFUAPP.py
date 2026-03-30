"""
Colony counting logic translated from MATLAB to Python. Original code based on KC.
Version: 2026-03-18
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from skimage.feature import peak_local_max
import numpy as np
import cv2

from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from medpy.filter.smoothing import anisotropic_diffusion
from skimage import exposure
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import threshold_multiotsu
from skimage.morphology import (
    disk,
    diamond,
    remove_small_objects,
    binary_opening,
    binary_dilation,
    binary_erosion,
    reconstruction,
    h_minima,
    h_maxima,
)
from skimage.measure import label, regionprops, find_contours
from skimage.segmentation import watershed
from skimage.transform import radon, rotate



@dataclass
class TuningParams:
    # HSV mask
    hsv_v_start: float = 0.635
    hsv_v_step: float = 0.05
    hsv_v_min: float = 0.35
    hsv_s_min: float = 0.00
    hsv_err_tol: float = 0.0065
    hsv_extent_min: float = 0.23
    hsv_open_disk_small: int = 3
    hsv_open_disk_large: int = 16

    # Remove small objects (global)
    min_object_area: int = 100   # lowered from 150: keeps small colonies for APP2 watershed

    # Colony-vs-cluster filtering
    small_area_quantile: float = 0.75
    small_cluster_ecc_min: float = 0.65
    small_cluster_extent_min: float = 0.20
    nonsmall_cluster_ecc_min: float = 0.99
    nonsmall_cluster_extent_min: float = 0.20
    colony_circularity_min: float = 0.075

    # Watershed conservativeness
    ws_min_distance: int = 9
    ws_thresh_abs_frac: float = 0.05
    ws_gauss_sigma: float = 1.0

    # Branch-selection thresholds
    branch_count_high_cutoff: int = 900
    branch_count_low_cutoff: int = 200
    density_scr_min: int = 300
    density_scr_max: int = 1000
    cc_count_entropy_cutoff: int = 150
    class_large_thresh: float = 0.0043
    class_medium_thresh: float = 0.0032
    init_count_large_cutoff: int = 50
    init_count_small_cutoff: int = 150
    uncountable_cutoff: int = 300
    uncountable_precheck_cutoff: int = 400

    # Otsu-Canny / Grad-overlap morphology
    otsu_canny_low: int = 50
    otsu_canny_high: int = 150
    otsu_canny_open_disk: int = 3
    grad_overlap_thresh: int = 4
    grad_overlap_open_disk: int = 3

# ----------------------------
# Data structures (ROI format)
# ----------------------------

@dataclass
class ROI:
    Position: np.ndarray        # Nx2 float array of polygon points (x, y)
    Center: np.ndarray          # (2,) float [x, y]
    Creator: str                # "Algorithm"
    Shape: str                  # "Polygon"
    NumOfCFU: int               # colony/cluster CFU count


# ----------------------------
# Utility helpers
# ----------------------------
def save_tiff_rgb(path: Union[str, Path], rgb: np.ndarray) -> None:
    """
    Save an RGB uint8 image as TIFF using OpenCV (which expects BGR).
    Handles float images by converting to uint8.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if rgb is None or not isinstance(rgb, np.ndarray) or rgb.size == 0:
        raise ValueError(f"save_tiff_rgb: empty image for {path}")

    img = rgb
    if img.dtype != np.uint8:
        img = _to_uint8(_as_float01(img))

    # If grayscale, write directly
    if img.ndim == 2:
        ok = cv2.imwrite(str(path), img)
        if not ok:
            raise RuntimeError(f"Failed to write TIFF: {path}")
        return

    # RGB -> BGR for OpenCV
    if img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(path), bgr)
        if not ok:
            raise RuntimeError(f"Failed to write TIFF: {path}")
        return

    raise ValueError(f"save_tiff_rgb: unsupported shape {img.shape} for {path}")


def _as_float01(img: np.ndarray) -> np.ndarray:
    """Convert uint8 image to [0, 1]; otherwise cast to float32."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _gray_uint8(img: np.ndarray) -> np.ndarray:
    """Grayscale uint8 0..255."""
    if img.ndim == 2:
        if img.dtype != np.uint8:
            return _to_uint8(_as_float01(img))
        return img
    # RGB -> gray
    g = rgb2gray(_as_float01(img))
    return _to_uint8(g)


def _disk_kernel(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _diamond_kernel(r: int) -> np.ndarray:
    # approximate diamond structuring element using skimage then to uint8
    k = diamond(r).astype(np.uint8)
    return k



def imoverlay(rgb: np.ndarray, mask: np.ndarray, color: str) -> np.ndarray:
    """
    MATLAB-like imoverlay for RGB images.
    """
    if rgb.ndim == 2:
        out = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    else:
        out = rgb.copy()

    m = mask.astype(bool)

    color = color.lower()
    if color == "red":
        out[m, 0] = 0   # R
        out[m, 1] = 255
        out[m, 2] = 0
    elif color == "green":
        out[m, 0] = 0
        out[m, 1] = 255   # G
        out[m, 2] = 0
    elif color == "blue":
        out[m, 0] = 0
        out[m, 1] = 0
        out[m, 2] = 255
    else:
        raise ValueError("color must be 'red', 'green', or 'blue'")
    return out


def bwselect(bw: np.ndarray, x: int, y: int) -> np.ndarray:
    """
    MATLAB bwselect equivalent:
    return the connected component of bw that contains point (x,y).
    MATLAB uses (x,y) as (col,row). We assume x=col, y=row.
    """
    bw = bw.astype(bool)
    lab = label(bw, connectivity=2)
    if y < 0 or y >= lab.shape[0] or x < 0 or x >= lab.shape[1]:
        return np.zeros_like(bw, dtype=bool)
    target = lab[y, x]
    if target == 0:
        return np.zeros_like(bw, dtype=bool)
    return lab == target


def bwpropfilt(bw: np.ndarray, prop: str, rng: Tuple[float, float]) -> np.ndarray:
    """
    Rough equivalent of MATLAB bwpropfilt for select properties:
      - "Area"
      - "Extent"
      - "Eccentricity"
      - "Circularity"
    """
    bw = bw.astype(bool)
    lab = label(bw, connectivity=2)
    keep = np.zeros_like(bw, dtype=bool)

    rmin, rmax = rng
    for reg in regionprops(lab):
        area = float(reg.area)

        if prop.lower() == "area":
            val = area
        elif prop.lower() == "extent":
            # extent = area / bbox_area
            minr, minc, maxr, maxc = reg.bbox
            bbox_area = float((maxr - minr) * (maxc - minc))
            val = area / bbox_area if bbox_area > 0 else 0.0
        elif prop.lower() == "eccentricity":
            val = float(reg.eccentricity)
        elif prop.lower() == "circularity":
            # circularity = 4*pi*area / perimeter^2
            per = float(reg.perimeter) if reg.perimeter and reg.perimeter > 0 else 0.0
            val = (4.0 * np.pi * area / (per * per)) if per > 0 else 0.0
        else:
            raise ValueError(f"Unsupported property: {prop}")

        if (val >= rmin) and (val <= rmax):
            keep[lab == reg.label] = True

    return keep


def bwareaopen(bw: np.ndarray, min_size: int) -> np.ndarray:
    return remove_small_objects(bw.astype(bool), min_size=min_size, connectivity=2)


def imhmax(gray_u8: np.ndarray, h: int) -> np.ndarray:
    """
    MATLAB imhmax(gray, h) using morphological reconstruction:
      imhmax(I,h) = reconstruction(I-h, I)
    """
    I = gray_u8.astype(np.int16)
    seed = np.clip(I - h, 0, 255).astype(np.uint8)
    rec = reconstruction(
        seed.astype(np.float32),
        gray_u8.astype(np.float32),
        method="dilation"
    )
    return np.clip(rec, 0, 255).astype(np.uint8)


def imextendedmin(gray_u8: np.ndarray, h: int) -> np.ndarray:
    """
    MATLAB imextendedmin(gray, h) ~ h-minima transform.
    """
    # skimage h_minima expects float or uint; returns boolean mask
    return h_minima(gray_u8, h=h).astype(bool)


def localcontrast_approx(
    img: np.ndarray,
    amount: float = 0.17,
    mid: float = 0.9,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Parameters
    ----------
    img : input image (uint8 or float)
    amount : strength of contrast enhancement
    mid : intensity around which contrast is boosted
    sigma : scale of local neighborhood (Gaussian blur)
    """
    img01 = _as_float01(img)

    # --- handle grayscale vs RGB ---
    if img01.ndim == 3:
        gray = cv2.cvtColor(_to_uint8(img01), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray = img01

    # --- local mean (Gaussian blur) ---
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # --- local contrast ---
    d = gray - local_mean

    # --- weighting around "mid" intensity ---
    # pixels near 'mid' get more enhancement
    weight = np.exp(-((gray - mid) ** 2) / (2 * 0.25 ** 2))

    # --- apply contrast scaling ---
    d_enhanced = d * (1.0 + amount * weight)

    # --- reconstruct ---
    out_gray = local_mean + d_enhanced
    out_gray = np.clip(out_gray, 0.0, 1.0)

    # --- restore RGB if needed ---
    if img01.ndim == 3:
        # scale RGB channels proportionally
        denom = np.maximum(gray, 1e-6)
        ratio = (out_gray / denom)[..., None]
        out = np.clip(img01 * ratio, 0.0, 1.0)
        return _to_uint8(out)
    else:
        return _to_uint8(out_gray)

def imlocalbrighten_approx(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Approximation of MATLAB imlocalbrighten:
    Lift shadows using gamma / sigmoid.
    """
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
    L.append(G[-1])  # top level
    return L


def reconstruct_pyramid(L):
    current = L[-1]
    for i in range(len(L) - 2, -1, -1):
        current = cv2.pyrUp(current, dstsize=(L[i].shape[1], L[i].shape[0])) + L[i]
    return current


def phi(d, sigma, alpha):
    """Nonlinear remapping function"""
    abs_d = np.abs(d)
    sign = np.sign(d)

    # smooth transition
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


# ----------------------------
# Core translated functions
# ----------------------------

def enhance_binary_image(bi: np.ndarray) -> np.ndarray:
    # MATLAB:
    # BWsdil = imdilate(bi, disk(1));
    # BWdfill = imfill(BWsdil, 'holes');
    # BWfinal = imerode(BWdfill, diamond(1));
    # bi = imerode(BWfinal, diamond(1));
    bi = bi.astype(bool)
    bi = binary_dilation(bi, footprint=disk(1))
    bi = ndi.binary_fill_holes(bi)
    bi = binary_erosion(bi, footprint=diamond(1))
    bi = binary_erosion(bi, footprint=diamond(1))
    return bi.astype(bool)

def imdiffusefilt_pm(img):
    img01 = img.astype(float) / 255.0
    out = anisotropic_diffusion(img01, niter=20, kappa=20, gamma=0.1)
    return (out * 255).astype(np.uint8)

def imsharpen_approx(img):
    """
    MATLAB imsharpen approximation (unsharp masking)
    """
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return sharp


def im2grayContrast(preImage):
    """
    Python version of MATLAB im2grayContrast
    """
    hmax=100 #Which features to suppress.
    # --- to float ---
    img01 = _as_float01(preImage)

    # 1. invert
    inv = 1.0 - img01

    # 2. difference (contrast exaggeration)
    expt = inv - img01
    expt = np.clip(expt, 0.0, 1.0)

    # 3. local contrast
    expt_u8 = _to_uint8(expt)
    con = localcontrast_approx(expt_u8, 0.18, 0.9)  # use your improved version

    # 4. sharpen
    sharp = imsharpen_approx(con)

    # 5. invert again
    expt2 = 255 - sharp

    # 6. grayscale
    gray = _to_uint8(rgb2gray(_as_float01(expt2)))

    # 7. suppress small peaks
    gray = imhmax(gray, hmax)

    return gray



def identify_border_pixels(window_u8: np.ndarray) -> int:
    # NOTE: scipy.ndimage.generic_filter passes a flattened 1D window, not a 2D array.
    w = np.asarray(window_u8).ravel()
    if w.size == 0:
        return 0
    ctr = w[w.size // 2]
    if ctr == 0 or ctr >= 250:
        return 0
    return int(np.count_nonzero(w >= 250) > 1)



def identify_high_bdry_pixels(window_u8: np.ndarray) -> int:
    # NOTE: scipy.ndimage.generic_filter passes a flattened 1D window, not a 2D array.
    w = np.asarray(window_u8).ravel()
    if w.size == 0:
        return 0
    ctr = w[w.size // 2]
    if ctr == 0:
        return 0
    return int(np.count_nonzero(w) > 2)


def mask_petri_dish(binary_mask: np.ndarray, origin: np.ndarray) -> np.ndarray:
    binary_mask = binary_mask.astype(bool)
    masked = origin.copy()
    masked[~binary_mask] = 0

    # crop to bounding box of mask
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        return masked
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0) + 1
    return masked[minr:maxr, minc:maxc]


def line_detect_radon(binary_image: np.ndarray, masked_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    # MATLAB uses theta = 0:179; [R,xp]=radon(...)
    bw = binary_image.astype(bool)
    theta = np.arange(0, 180, 1)
    R = radon(bw.astype(float), theta=theta, circle=False)
    # get top peaks
    uniq = np.unique(R)
    uniq_sorted = np.sort(uniq)[::-1]
    top = uniq_sorted[:30] if uniq_sorted.size >= 30 else uniq_sorted
    # indices where R is in top
    peaks = np.isin(R, top)
    row_peak, col_peak = np.where(peaks)

    if uniq_sorted.size > 0 and uniq_sorted[0] > 1500:
        xp = np.arange(R.shape[0]) - (R.shape[0] // 2)
        xp_peak_offset = xp[row_peak]
        theta_peak = theta[col_peak]
        line_theta = int(np.ceil(np.mean(theta_peak) - 90))

        rotated = rotate(masked_image, angle=-line_theta, resize=False, preserve_range=True).astype(masked_image.dtype)

        centerX = int(np.ceil(rotated.shape[1] / 2))
        centerY = int(np.ceil(rotated.shape[0] / 2))

        x = 800.0  # from pol2cart(deg2rad(0),800)
        # find second edge
        second_pk_offset = xp_peak_offset[0]
        for i in range(1, len(xp_peak_offset)):
            if abs(abs(xp_peak_offset[0]) - abs(xp_peak_offset[i])) > 5:
                second_pk_offset = xp_peak_offset[i]
                break

        xCoord = np.array([centerX - x, centerX + x], dtype=float)
        yCoord1 = np.array([centerY - xp_peak_offset[0], centerY - xp_peak_offset[0]], dtype=float)
        xy_first = np.stack([xCoord, yCoord1], axis=1)

        max_len = float(np.hypot(xCoord[1] - xCoord[0], yCoord1[1] - yCoord1[0]))

        yCoord2 = np.array([centerY - second_pk_offset, centerY - second_pk_offset], dtype=float)
        xy_second = np.stack([xCoord, yCoord2], axis=1)

        return xy_first, xy_second, max_len, rotated

    xy_first = np.array([[np.nan, np.nan]])
    xy_second = np.array([[np.nan, np.nan]])
    return xy_first, xy_second, 0.0, masked_image


def find_samples(masked_image: np.ndarray) -> List[Dict[str, np.ndarray]]:
    # MATLAB:
    # grayMasked = im2gray(masked); BW=edge(canny,[0.185,0.4025]); dilate line se 7x; close disk(10)
    gray = _gray_uint8(masked_image)
    edges = cv2.Canny(gray, int(0.185 * 255), int(0.4025 * 255))

    bw = edges.astype(bool)
    # approximate strel line(5,90) and line(5,0)
    k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    bw_u8 = bw.astype(np.uint8) * 255
    for _ in range(7):
        bw_u8 = cv2.dilate(bw_u8, k_v)
        bw_u8 = cv2.dilate(bw_u8, k_h)

    bw_u8 = cv2.morphologyEx(bw_u8, cv2.MORPH_CLOSE, _disk_kernel(10))
    bw2 = bw_u8 > 0

    xy_first, xy_second, max_len, rotated = line_detect_radon(bw2, masked_image)

    # reorder if needed (match MATLAB swap logic)
    if xy_first.shape[0] >= 2 and xy_second.shape[0] >= 2:
        if (xy_first[0, 1] > xy_second[0, 1]) and (xy_first[1, 1] > xy_second[1, 1]):
            xy_first, xy_second = xy_second, xy_first

    h, w = rotated.shape[:2]
    if max_len > 1400:
        midline_correction = 50
        # MATLAB uses imcrop with [x y width height] in 1-based; we use slices
        y1 = int(min(xy_first[0, 1], xy_first[1, 1]) - midline_correction)
        y2 = int(max(xy_second[0, 1], xy_second[1, 1]) + midline_correction)
        y1 = max(y1, 0)
        y2 = min(y2, h)

        expt1 = rotated[0:y1, :, ...] if y1 > 0 else rotated[0:0, :, ...]
        expt2 = rotated[y2:h, :, ...] if y2 < h else rotated[h:h, :, ...]
        return [{"expt": expt1}, {"expt": expt2}]
    else:
        return [{"expt": rotated}]


def generate_bw_and_gray(class_flag: str, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    expt = rgb.copy()

    if class_flag == "large":
        # average filter 3x3
        expt = cv2.blur(expt, (3, 3))
        expt = locallapfilt_approx(expt, sigma=0.5, alpha=0.8)

        gray = im2grayContrast(expt)
        # multi-otsu with 3 classes (2 thresholds)
        t = threshold_multiotsu(gray, classes=3)
        quant = np.digitize(gray, bins=t).astype(np.uint8)  # 0..2

        # label2rgb(gray) then back to gray -> just map to levels
        levels = np.array([0, 127, 255], dtype=np.uint8)
        gray_q = levels[quant]

        # canny
        edges = cv2.Canny(gray_q, int(0.1563 * 255), int(0.4095 * 255))
        bw = enhance_binary_image(edges > 0)
        return bw, gray_q

    # "small" path
    expt = localcontrast_approx(expt, amount=0.17, mid=0.9)
    expt = imlocalbrighten_approx(expt, strength=0.5)
    expt = cv2.GaussianBlur(expt, (0, 0), 1.0)

    gray = im2grayContrast(expt)

    # tophat/bothat approximation (disk 10)
    se = _disk_kernel(10)
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    bot = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)
    con = cv2.subtract(cv2.add(gray, top), bot)

    edges = cv2.Canny(con, int(0.0063 * 255), int(0.14950 * 255))
    bw = enhance_binary_image(edges > 0)
    return bw, gray

def conservative_watershed(region_mask: np.ndarray, params: Optional[TuningParams] = None) -> Tuple[np.ndarray, int, np.ndarray]:
    params = params or TuningParams()

    region_mask = region_mask.astype(bool)
    if not np.any(region_mask):
        L = np.zeros_like(region_mask, dtype=np.int32)
        return L, 0, np.zeros_like(region_mask, dtype=bool)

    dist = ndi.distance_transform_edt(region_mask).astype(np.float32)
    dist_s = ndi.gaussian_filter(dist, sigma=float(params.ws_gauss_sigma))

    dmax = float(dist_s.max())
    if dmax < 3.0:
        L = region_mask.astype(np.int32)
        boundary = np.zeros_like(region_mask, dtype=bool)
        return L, 1, boundary

    # h_maxima finds peaks that are at least h above every surrounding valley.
    # This rejects boundary-roughness bumps (prominence ~1–3px) while reliably
    # finding one prominent dome per circular colony.
    # Reuses ws_thresh_abs_frac as the prominence fraction of the max DT value.
    h_val = max(1.0, float(params.ws_thresh_abs_frac) * dmax)
    seed_mask = h_maxima(dist_s, h=h_val) & region_mask
    markers, n_seeds = ndi.label(seed_mask)

    if n_seeds <= 1:
        # Single colony (or degenerate) — no split needed
        L = region_mask.astype(np.int32)
        boundary = np.zeros_like(region_mask, dtype=bool)
        return L, 1, boundary

    L = watershed(-dist_s, markers=markers, mask=region_mask).astype(np.int32)

    # Discard watershed segments too small to be a real colony
    valid = [lbl for lbl in range(1, n_seeds + 1)
             if np.count_nonzero(L == lbl) >= int(params.min_object_area)]

    if len(valid) <= 1:
        return region_mask.astype(np.int32), 1, np.zeros_like(region_mask, dtype=bool)

    # Remap to contiguous labels 1..N
    new_L = np.zeros_like(L)
    for new_i, old_lbl in enumerate(valid, start=1):
        new_L[L == old_lbl] = new_i

    mx = ndi.maximum_filter(new_L, size=3)
    mn = ndi.minimum_filter(new_L, size=3)
    boundary = region_mask & (mx != mn)

    return new_L, len(valid), boundary


def count_colonies_with_instances(
    binary_image: np.ndarray,
    expt_rgb: np.ndarray,
    class_flag: Optional[str],
    sample_id: int,
    params: Optional[TuningParams] = None,
) -> Tuple[int, np.ndarray, List[ROI], List[ROI], np.ndarray]:
    """
    Count colonies in a binary mask using per-blob watershed.

    Instead of pre-classifying blobs as "cluster" vs "colony" by shape metrics
    (fragile: a 2-colony cluster with low eccentricity would be missed entirely),
    we apply conservative_watershed to EVERY blob and let the distance-transform
    h_maxima decide how many colonies it contains.

      n_seeds == 1  →  single colony, count = 1
      n_seeds >  1  →  cluster, count = n_seeds, draw yellow watershed boundaries

    class_flag == "large" retains the small-elongated-object cleanup for the
    original pipeline; otherwise it is unused.
    """
    params = params or TuningParams()

    bw = binary_image.astype(bool)
    bw = bwareaopen(bw, params.min_object_area)

    if not np.any(bw):
        return 0, expt_rgb, [], [], np.zeros(bw.shape, dtype=np.int32)

    lab = label(bw, connectivity=2)
    regs = regionprops(lab)
    stats_area = np.array([r.area for r in regs], dtype=float)

    # class_flag == "large": remove small elongated noise before counting
    if class_flag == "large" and stats_area.size:
        avg_area = float(stats_area.mean())
        min_area = avg_area * 0.6
        keep_labels: List[int] = []
        for idx, reg in enumerate(regs):
            a = stats_area[idx]
            if a < min_area:
                if float(reg.eccentricity) > 0.85 or a < 25:
                    continue  # discard
            keep_labels.append(reg.label)
        bw = np.isin(lab, keep_labels)
        bw = bwareaopen(bw, params.min_object_area)

    # Filter out clear non-colony artefacts (scratches, filaments)
    bw = bwpropfilt(bw, "Circularity", (float(params.colony_circularity_min), 1.0))

    # Paint every accepted blob green up front; clusters get yellow boundaries later
    overlay = imoverlay(expt_rgb, bw, "green")

    colony_roi: List[ROI] = []
    cluster_roi: List[ROI] = []
    num_cfu = 0
    instance_labels = np.zeros(bw.shape, dtype=np.int32)
    next_label = 1

    lab2 = label(bw, connectivity=2)
    regs2 = regionprops(lab2)

    for reg in regs2:
        minr, minc, maxr, maxc = reg.bbox
        region_mask = (lab2[minr:maxr, minc:maxc] == reg.label)

        # Watershed decides how many colonies are in this blob
        L, n_colonies, boundary_local = conservative_watershed(region_mask, params=params)

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
            # Single colony
            num_cfu += 1
            colony_roi.append(ROI(Position=poly_pos, Center=center,
                                  Creator="Algorithm", Shape="Polygon", NumOfCFU=1))
            instance_labels[lab2 == reg.label] = next_label
            next_label += 1
        else:
            # Cluster: n_colonies touching colonies
            num_cfu += n_colonies
            cluster_roi.append(ROI(Position=poly_pos, Center=center,
                                   Creator="Algorithm", Shape="Polygon", NumOfCFU=n_colonies))

            # Yellow watershed boundary lines
            bd_full = np.zeros(overlay.shape[:2], dtype=bool)
            bd_full[minr:maxr, minc:maxc] = binary_dilation(boundary_local, footprint=disk(1))
            overlay[bd_full] = np.array([255, 255, 0], dtype=np.uint8)

            for local_label in range(1, int(L.max()) + 1):
                local_mask = (L == local_label)
                if np.any(local_mask):
                    instance_labels[minr:maxr, minc:maxc][local_mask] = next_label
                    next_label += 1

    return num_cfu, overlay, colony_roi, cluster_roi, instance_labels


def count_colonies(binary_image: np.ndarray, expt_rgb: np.ndarray, class_flag: Optional[str],
                  sample_id: int, params: Optional[TuningParams] = None
                  ) -> Tuple[int, np.ndarray, List[ROI], List[ROI]]:
    num_cfu, overlay, colony_roi, cluster_roi, _ = count_colonies_with_instances(
        binary_image,
        expt_rgb,
        class_flag,
        sample_id,
        params=params,
    )
    return num_cfu, overlay, colony_roi, cluster_roi

def predict_count_only(
    rgb: np.ndarray,
    dish_mode: str = "auto",
    target_area_px2: float = 2245000.0,
    params: Optional[TuningParams] = None,
) -> int:
    params = params or TuningParams()
    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))

    total = 0
    for k, item in enumerate(expts, start=1):
        expt = item["expt"]
        num_cfu, _, _, _, _, _, _ = _count_one_expt_full_logic(expt, k, params=params)
        total += int(num_cfu)

    return int(total)


def predict_tuning_features(
    rgb: np.ndarray,
    dish_mode: str = "auto",
    target_area_px2: float = 2245000.0,
    params: Optional[TuningParams] = None,
) -> Dict[str, Any]:
    params = params or TuningParams()
    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))

    total_count = 0
    all_instance_areas: List[float] = []
    methods: List[str] = []
    classes: List[str] = []
    mean_areas: List[float] = []

    for k, item in enumerate(expts, start=1):
        expt = item["expt"]
        num_cfu, method, size_class, mean_area, _, colony_roi, cluster_roi = _count_one_expt_full_logic(
            expt,
            k,
            params=params,
        )
        total_count += int(num_cfu)
        methods.append(str(method))
        classes.append(str(size_class))
        mean_areas.append(float(mean_area))
        all_instance_areas.extend(_roi_instance_areas(colony_roi, cluster_roi))

    return {
        "count": int(total_count),
        "instance_areas": [float(a) for a in all_instance_areas],
        "methods": methods,
        "classes": classes,
        "mean_area": float(np.mean(mean_areas)) if mean_areas else 0.0,
    }


def create_prediction_mask(rgb: np.ndarray, v_thresh: float, s_min: float = 0.1) -> np.ndarray:
    hsv = rgb2hsv(_as_float01(rgb))
    S = hsv[..., 1]
    V = hsv[..., 2]
    return (V < float(v_thresh)) & (S > float(s_min))


def hsv_filter(
    rgb: np.ndarray,
    gray_image_u8: np.ndarray,
    sample_id: int,
    params: Optional[TuningParams] = None
) -> Tuple[int, np.ndarray, float, List[ROI], List[ROI]]:

    params = params or TuningParams()

    if gray_image_u8.shape == rgb.shape[:2]:
        gray_expt = _gray_uint8(gray_image_u8)
    else:
        gray_expt = _gray_uint8(rgb)
    bw_min = imextendedmin(gray_expt, 35) & (gray_expt != 0)

    v_thresh = float(params.hsv_v_start)
    pred = create_prediction_mask(rgb, v_thresh, s_min=params.hsv_s_min) & (gray_expt != 0)

    err_map = pred.astype(np.int8) - bw_min.astype(np.int8)
    err = float(np.count_nonzero(err_map) / err_map.size)

    while err > float(params.hsv_err_tol):
        v_thresh -= float(params.hsv_v_step)
        if v_thresh < float(params.hsv_v_min):
            break
        pred = create_prediction_mask(rgb, v_thresh, s_min=params.hsv_s_min) & (gray_expt != 0)
        err_map = pred.astype(np.int8) - bw_min.astype(np.int8)
        err = float(np.count_nonzero(err_map) / err_map.size)

    bw_open = bwareaopen(pred, params.min_object_area)
    bw_open = bwpropfilt(bw_open, "Extent", (float(params.hsv_extent_min), 1.0))
    bw_open = binary_opening(bw_open, footprint=disk(int(params.hsv_open_disk_small)))
    bw_fill = ndi.binary_fill_holes(bw_open)

    bw_big = cv2.resize(bw_fill.astype(np.uint8), None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST) > 0
    bw_filt = binary_opening(bw_big, footprint=disk(int(params.hsv_open_disk_large)))
    bw_final = cv2.resize(bw_filt.astype(np.uint8), (bw_fill.shape[1], bw_fill.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

    hsv_count, overlay, colony_roi, cluster_roi = count_colonies(bw_final, rgb, None, sample_id, params=params)
    mean_area = _mean_region_area(bw_final)

    _ = count_nearest_cfu(pred)
    return hsv_count, overlay, mean_area, colony_roi, cluster_roi

def grad_overlap(rgb: np.ndarray, im_grad_final: np.ndarray, class_flag: str, thresh_ratio: float, sample_id: int, params: Optional[TuningParams] = None
                 ) -> Tuple[int, np.ndarray, float, List[ROI], List[ROI]]:
    # MATLAB:
    # imGradFinal = enhanceBinaryImage(imGradFinal);
    # imGradCV = imopen(imGradFinal, disk(3));
    # imGradCV = reshape(imGradCV > 4, H,W);
    # imGradCV = enhanceBinaryImage(imGradCV);
    # bwInit = imGradCV
    params = params or TuningParams()
    _ = thresh_ratio
    im_grad_bin = enhance_binary_image(im_grad_final.astype(bool))
    im_grad_cv = binary_opening(im_grad_bin, footprint=disk(int(params.grad_overlap_open_disk)))
    im_grad_cv = (im_grad_cv.astype(np.uint8) * 255) > int(params.grad_overlap_thresh)
    im_grad_cv = enhance_binary_image(im_grad_cv)

    bw_init = im_grad_cv
    grad_count, overlay, colony_roi, cluster_roi = count_colonies(bw_init, rgb, class_flag, sample_id, params=params)
    mean_area = _mean_region_area(bw_init)

    _ = count_nearest_cfu(bw_init)
    return grad_count, overlay, mean_area, colony_roi, cluster_roi


def _mean_region_area(bw: np.ndarray) -> float:
    props = regionprops(label(bw, connectivity=2))
    return float(np.mean([p.area for p in props])) if props else 0.0


def _polygon_area_xy(pos: np.ndarray) -> float:
    if pos is None or len(pos) < 3:
        return 0.0
    x = pos[:, 0].astype(np.float64)
    y = pos[:, 1].astype(np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _roi_instance_areas(colony_roi: List[ROI], cluster_roi: List[ROI]) -> List[float]:
    areas: List[float] = []
    for roi in colony_roi:
        areas.append(max(_polygon_area_xy(roi.Position), 0.0))
    for roi in cluster_roi:
        total_area = max(_polygon_area_xy(roi.Position), 0.0)
        n = max(int(roi.NumOfCFU), 1)
        per_instance = total_area / float(n)
        areas.extend([per_instance] * n)
    return areas


def generate_cfu_count(args: Sequence[Any], params: Optional[TuningParams] = None) -> Tuple[int, str, str, float, np.ndarray, List[ROI], List[ROI]]:
    """
    Port of MATLAB generateCFUCount(args)
    args layout (1-based MATLAB -> 0-based Python):
      0 prevCount
      1 ccCount
      2 densityScr
      3 threshRatio
      4 sampleID
      5 prevMethod
      6 prevClass
      7 prevColonyROI
      8 prevClusterROI
      9 RGB
      10 imGradFinal
      11 prevOutImage
    """
    params = params or TuningParams()
    prev_count = float(args[0])
    cc_count = float(args[1])
    density_scr = float(args[2])
    thresh_ratio = float(args[3])
    sample_id = int(args[4])
    prev_method = str(args[5])
    prev_class = str(args[6])
    prev_colony_roi = list(args[7]) if args[7] is not None else []
    prev_cluster_roi = list(args[8]) if args[8] is not None else []
    rgb = args[9]
    im_grad = args[10]
    prev_out = args[11]
    gray_image = _gray_uint8(rgb)

    if prev_count < 0:
        raise ValueError("The counter incorrectly states that there is a negative count of colonies")

    if prev_method == "Uncountable":
        return int(prev_count), "Uncountable", prev_class, 0.0, np.array([]), [], []

    class_flag = prev_class
    method = prev_method
    num = int(prev_count)
    mean_area = 0.0
    overlay = prev_out
    c_roi = prev_colony_roi
    cl_roi = prev_cluster_roi

    if prev_count >= int(params.branch_count_high_cutoff):
        if prev_method != "Canny":
            class_flag = prev_class
            bw, _ = generate_bw_and_gray(prev_class, rgb)
            num, overlay, c_roi, cl_roi = count_colonies(bw, rgb, prev_class, sample_id, params=params)
            mean_area = _mean_region_area(bw)
            method = "Canny"
    elif prev_count <= int(params.branch_count_low_cutoff):
        if prev_method != "HSV Filter":
            class_flag = "large"
            num, overlay, mean_area, c_roi, cl_roi = hsv_filter(rgb, gray_image, sample_id, params=params)
            method = "HSV Filter"
    elif (prev_method != "hjDE") and (prev_method != "Grad Overlap"):
        class_flag = "medium"
        density_bool = (density_scr > int(params.density_scr_min)) and (density_scr < int(params.density_scr_max))
        entropy_bool = cc_count > int(params.cc_count_entropy_cutoff)

        if density_bool or entropy_bool:
            # "Otsu-Canny"
            _, gray_expt = generate_bw_and_gray("large", rgb)
            best = cv2.Canny(gray_expt, int(params.otsu_canny_low), int(params.otsu_canny_high)) > 0
            ebest = enhance_binary_image(best)
            bw = binary_opening(ebest, footprint=disk(int(params.otsu_canny_open_disk)))
            num, overlay, c_roi, cl_roi = count_colonies(bw, rgb, None, sample_id, params=params)
            mean_area = _mean_region_area(bw)
            method = "Otsu-Canny"
        else:
            num, overlay, mean_area, c_roi, cl_roi = grad_overlap(
                rgb,
                im_grad,
                prev_class,
                thresh_ratio,
                sample_id,
                params=params,
            )
            method = "Grad Overlap"

    if num > int(params.uncountable_cutoff):
        return int(prev_count), "Uncountable", prev_class, 0.0, np.array([]), [], []

    return num, method, class_flag, mean_area, overlay, c_roi, cl_roi


def count_nearest_cfu(bw: np.ndarray) -> int:
    # MATLAB rangesearch radius=100 and score by neighbor counts
    bw = bw.astype(bool)
    lab = label(bw, connectivity=2)
    regs = regionprops(lab)
    if not regs:
        return 0

    pts = np.array([[r.centroid[1], r.centroid[0]] for r in regs], dtype=np.float32)  # (x,y)
    radius = 100.0
    tree = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, r=radius)

    density_scr = 0
    for n_idx in neighbors:
        n = len(n_idx)
        if n > 10:
            density_scr += 1
        elif n < 4:
            density_scr -= 1
    return int(density_scr)


# ----------------------------
# Petri dish detection (EXTERNAL: DetectDish.py)
# ----------------------------
# We no longer detect/mask the petri dish inside this file.
# Instead we call DetectDish.py, which does robust contour/ellipse detection,
# optional top/bottom divider detection, and masking.

def _crop_to_nonzero_bbox(rgb: np.ndarray, pad: int = 2) -> np.ndarray:
    """Crop RGB image to bounding box of non-zero pixels (any channel)."""
    if rgb.size == 0:
        return rgb
    m = np.any(rgb != 0, axis=2) if rgb.ndim == 3 else (rgb != 0)
    ys, xs = np.where(m)
    if ys.size == 0 or xs.size == 0:
        return rgb
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(rgb.shape[0], y1 + pad); x1 = min(rgb.shape[1], x1 + pad)
    return rgb[y0:y1, x0:x1].copy()


def _get_expts_from_detectdish(
    rgb: np.ndarray,
    dish_mode: str,
    target_area_px2: float,
) -> List[Dict[str, np.ndarray]]:
    """
    Returns a list of dicts with key 'expt' (same structure as find_samples output),
    using DetectDish.py for masking and (optionally) top/bottom splitting.

    dish_mode:
      - 'pre_cropped': input is already a single cropped experiment image; skip DetectDish
      - 'single': always return 1 expt (whole dish)
      - 'double': try to split into top/bottom; if no divider found, returns 1 expt
      - 'auto': same as 'double' but falls back to 1 expt if no divider found
    """
    if dish_mode == "pre_cropped":
        return [{"expt": rgb}]

    # Lazy import so the rest of the file can be imported without DetectDish installed.
    import DetectDish  # your improved dish detector script/module

    # DetectDish works in BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    fixed_diameter_px = 2.0 * float(np.sqrt(float(target_area_px2) / np.pi))
    masked_bgr, ellipse_final, brightness, _ = DetectDish.detect_plate_rgb(bgr, fixed_diameter_px=fixed_diameter_px)
    if ellipse_final is None or masked_bgr is None:
        # If dish detection fails, fall back to original image (no masking)
        return [{"expt": rgb}]

    masked_rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)

    # Single dish: return whole masked dish
    if dish_mode == "single":
        return [{"expt": _crop_to_nonzero_bbox(masked_rgb)}]

    # Auto/double: attempt to find divider + split
    y_split = DetectDish.find_divider_y(
        brightness,
        ellipse_final,
        bar_q=DetectDish.BAR_Q,
        frac_thresh=DetectDish.BAR_FRAC_THRESH,
        min_thick=DetectDish.BAR_MIN_THICK_PX,
        max_thick=DetectDish.BAR_MAX_THICK_PX,
    )

    if y_split is None:
        # no divider found -> one dish
        return [{"expt": _crop_to_nonzero_bbox(masked_rgb)}]

    top_bgr, bot_bgr = DetectDish.mask_top_bottom(
        image_bgr=bgr,
        ellipse=ellipse_final,
        y_split=y_split,
        gap=DetectDish.BAR_SPLIT_MARGIN_PX,
    )
    top_rgb = cv2.cvtColor(top_bgr, cv2.COLOR_BGR2RGB)
    bot_rgb = cv2.cvtColor(bot_bgr, cv2.COLOR_BGR2RGB)

    expts = [{"expt": _crop_to_nonzero_bbox(top_rgb)}, {"expt": _crop_to_nonzero_bbox(bot_rgb)}]

    # If user forced double, keep 2; if auto, keep 2 as well (since divider found)
    return expts

def _count_one_expt_full_logic(
    expt: np.ndarray,
    k: int,
    params: Optional[TuningParams] = None,
) -> Tuple[int, str, str, float, np.ndarray, List[ROI], List[ROI]]:
    """
    Run the SAME logic as count_cfu_app() for a single masked/split dish image (expt).

    Returns:
      num_cfu, method, size_class, mean_area, out_image, colony_roi, cluster_roi
    """
    params = params or TuningParams()
    sz_h, sz_w = expt.shape[:2]

    # Enhancement chain (same as count_cfu_app)
    expt_adj = imadjust_approx(expt)
    expt_adj = locallapfilt_approx(expt_adj, sigma=0.2, alpha=0.5)
    expt_adj = localcontrast_approx(expt_adj, amount=0.17, mid=0.9)
    expt_adj = imlocalbrighten_approx(expt_adj, strength=0.5)

    gray_expt = im2grayContrast(expt_adj)
    gray_expt = imdiffusefilt_pm(gray_expt)
    blur = cv2.GaussianBlur(gray_expt, (3, 3), sigmaX=0.5, sigmaY=0.5, borderType=cv2.BORDER_REPLICATE)
    im_grad = cv2.Laplacian(blur, cv2.CV_32F, ksize=3, borderType=cv2.BORDER_REPLICATE)
    im_grad = np.abs(im_grad)

    border_grad = (im_grad >= 250).astype(np.uint8)
    border_grad2 = ndi.generic_filter(border_grad, function=identify_high_bdry_pixels, size=(3, 3), mode="nearest")
    im_grad_final = im_grad.astype(np.float32) * (1.0 - border_grad2.astype(np.float32))

    border_med = ndi.generic_filter(im_grad, function=identify_border_pixels, size=(3, 3), mode="nearest")
    im_grad_final = im_grad_final * (1.0 - border_med.astype(np.float32))

    im_grad_vec = im_grad_final.reshape(-1)
    im_grad_sorted = np.sort(im_grad_vec)

    im_grad_filtered = im_grad_sorted[im_grad_sorted > 5]
    if im_grad_filtered.size == 0:
        im_grad_filtered = np.array([6.0], dtype=np.float32)

    bins = np.arange(im_grad_filtered.min(), im_grad_filtered.max() + 5, 5.0)
    values, _ = np.histogram(im_grad_filtered, bins=bins)

    large_thresh = float(params.class_large_thresh)
    medium_thresh = float(params.class_medium_thresh)
    sz_total = float(sz_h * sz_w)
    large_check = sz_total * large_thresh
    medium_check = sz_total * medium_thresh
    max_val = float(values.max()) if values.size else 0.0

    if max_val < large_check:
        class_flag = "large"
    elif (max_val > medium_check) and (max_val < large_check):
        class_flag = "medium"
    else:
        class_flag = "small"

    # Initial guess (regional minima)
    gray_u8 = gray_expt
    filtered = cv2.GaussianBlur(
    gray_u8,
    (31, 31),
    sigmaX=15,
    sigmaY=15,
    borderType=cv2.BORDER_REPLICATE
    )
    closed = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, _disk_kernel(15))
    eroded = ndi.grey_erosion(closed, size=(3, 3))
    local_min = (closed == eroded)
    bi_regional = binary_dilation(local_min, footprint=disk(5))
    bw_bkg = gray_u8 != 0
    bi_regional = bi_regional & bw_bkg
    init_count = int(np.max(label(bi_regional, connectivity=2)))

    # Zero CFU detection
    expt_blur = cv2.GaussianBlur(expt, (0, 0), 4)
    gray0 = _gray_uint8(expt_blur)
    bw_min0 = imextendedmin(gray0, 60) & (gray0 != 0)
    bw_min0 = bwpropfilt(bw_min0, "Extent", (0.2, 1.0))
    zero_flag = not np.any(bw_min0)

    # Uncountable check
    gray_unc = _gray_uint8(expt)
    _, thr = cv2.threshold(gray_unc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_unc = (gray_unc < thr) & (gray_unc != 0)
    num_unc, _, _, _ = count_colonies(bw_unc, expt, None, k, params=params)
    uncountable_flag = num_unc > int(params.uncountable_precheck_cutoff)

    if zero_flag:
        num_cfu = 0
        size_class = "Zero"
        method = "Zero"
        out_image = expt
        colony_roi: List[ROI] = []
        cluster_roi: List[ROI] = []
        mean_area = 0.0
        return num_cfu, method, size_class, mean_area, out_image, colony_roi, cluster_roi

    if uncountable_flag:
        num_cfu = int(num_unc)
        size_class = "Uncountable"
        method = "Uncountable"
        out_image = np.array([])
        colony_roi = []
        cluster_roi = []
        mean_area = 0.0
        return num_cfu, method, size_class, mean_area, out_image, colony_roi, cluster_roi

    # Reclassify based on initCount
    if class_flag != "large" and init_count < int(params.init_count_large_cutoff):
        class_flag = "large"
    elif class_flag != "small" and init_count > int(params.init_count_small_cutoff):
        class_flag = "small"

    # First run: generate BW + count
    bw, _ = generate_bw_and_gray(class_flag, expt)
    num_cfu, _, colony_roi, cluster_roi = count_colonies(bw, expt, class_flag, k, params=params)
    density_scr = count_nearest_cfu(bw)
    thresh_ratio = 0.0

    # Now apply the SAME "method selection" logic as count_cfu_app (generate_cfu_count twice)
    prev_method = ""
    prev_class = ""
    prev_out = np.array([], dtype=np.uint8)
    prev_colony_roi: List[ROI] = []
    prev_cluster_roi: List[ROI] = []

    args1 = [
        num_cfu, init_count, density_scr, thresh_ratio, k,
        prev_method, prev_class, prev_colony_roi, prev_cluster_roi,
        expt, im_grad_final, prev_out
    ]
    num_cfu, first_method, first_class, mean_area, out_image, prev_colony_roi, prev_cluster_roi = generate_cfu_count(args1, params=params)

    args2 = [
        num_cfu, init_count, density_scr, thresh_ratio, k,
        first_method, first_class, prev_colony_roi, prev_cluster_roi,
        expt, im_grad_final, out_image
    ]
    num_cfu, sample_method, _, _, out_image, colony_roi, cluster_roi = generate_cfu_count(args2, params=params)

    # Size class from mean area
    if mean_area <= 150:
        size_class = "small"
    elif 150 < mean_area < 400:
        size_class = "medium"
    else:
        size_class = "large"

    return int(num_cfu), str(sample_method), size_class, float(mean_area), out_image, colony_roi, cluster_roi

def count_cfu_app(
    rgb: np.ndarray,
    top_cell: str,
    bot_cell: str,
    folder_dir: Union[str, Path],
    version: str,
    index: Any,
    dish_mode: str = "auto",
    target_area_px2: float = 2245000.0,
    save_which: str = "overlay",   # overlay | expt | both,
    params: Optional[TuningParams] = None
) -> List[str]:
    """
    Count CFU on a single image, using DetectDish.py for petri-dish masking/splitting.

    Outputs TIFF images (no .mat):
      - overlay: <outdir>/<version>/<sample_id>.tif
      - expt:    <outdir>/<version>/<sample_id>__expt.tif

    dish_mode:
      - pre_cropped: input already represents one cropped experiment image
      - single: always 1 dish/sample
      - double: try split into top/bottom (if divider found); otherwise 1
      - auto:   same as double
    """
    params = params or TuningParams()
    folder_dir = Path(folder_dir)
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / version).mkdir(parents=True, exist_ok=True)

    # Detect + mask (and optionally split) using DetectDish.py
    expts = _get_expts_from_detectdish(rgb, dish_mode=dish_mode, target_area_px2=float(target_area_px2))

    out_paths: List[str] = []

    for k, item in enumerate(expts, start=1):
        expt = item["expt"]
        _, _, _, _, out_image, _, _ = _count_one_expt_full_logic(expt, k, params=params)

        # Save TIFFs (no .mat)
        expt_id = top_cell if k == 1 else bot_cell
        base = folder_dir / version / expt_id

        if save_which in ("overlay", "both"):
            out_tif = base.with_suffix(".tif")
            save_tiff_rgb(out_tif, out_image if (isinstance(out_image, np.ndarray) and out_image.size) else expt)
            out_paths.append(str(out_tif))

        if save_which in ("expt", "both"):
            expt_tif = base.with_name(base.name + "__expt").with_suffix(".tif")
            save_tiff_rgb(expt_tif, expt)
            out_paths.append(str(expt_tif))

    return out_paths


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse
    import cv2
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run CFU counter on a single image")
    parser.add_argument("image", help="Input image (png/jpg/tif)")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("--version", default="v1")
    # Petri dish handling (via DetectDish.py)
    parser.add_argument("--dish_mode", default="auto", choices=["auto", "single", "double", "pre_cropped"],
                        help="pre_cropped=skip DetectDish; single=force 1 dish; double=try split top/bottom; auto=double with fallback")
    parser.add_argument("--target_area_px2", type=float, default=2245000.0,
                        help="Dish area in px^2 used to derive fixed circle diameter for DetectDish")
    # Output
    parser.add_argument("--save_which", default="overlay", choices=["overlay", "expt", "both"],
                        help="Save overlay TIFF, cropped dish TIFF, or both")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read {args.image}")
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    stem = Path(args.image).stem
    out_paths = count_cfu_app(
        rgb=img,
        top_cell=stem,               # if split -> top uses stem, bottom uses stem + "_Bottom"
        bot_cell=stem + "_Bottom",
        folder_dir=args.outdir,
        version=args.version,
        index=Path(args.image).name,
        dish_mode=args.dish_mode,
        target_area_px2=args.target_area_px2,
        save_which=args.save_which,
        params=None
    )

    print("Saved:")
    for p in out_paths:
        print(p)
