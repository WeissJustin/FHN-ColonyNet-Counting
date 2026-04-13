# ------------------------------
# CPSAM FINE-TUNING + INFERENCE
# ------------------------------
# Fine-tunes Cellpose-SAM (CPSAM) on image/mask pairs, then runs inference on
# a folder of dish images. Saves green-tinted overlay TIFFs and a count/size CSV;
# Version 1.1: Overlay-only output with a mini size-distribution histogram.
#
# Version: 1.1 (04-13-2026)
# ------------------------------

import argparse
import os
import glob
import time
from pathlib import Path
from tqdm import tqdm
import csv
import cv2
import numpy as np
from cellpose import io, models, train

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# ------------------------------
# Image discovery + preprocessing
# ------------------------------

def list_images(folder: str):
    """
    Return supported image files in `folder`, sorted for deterministic runs.

    Only scans the top level of the directory. Training data recursion is
    handled separately by Cellpose's `load_train_test_data`.
    """
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(paths)


def ensure_three_channels(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to the 3-channel format expected by CPSAM.

    Grayscale inputs are replicated into three channels, single-channel stacks
    are repeated, and images with extra channels keep only the first three.
    Raises ValueError for unsupported shapes.
    """
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        return np.repeat(img, 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] >= 3:
        return img[..., :3]
    raise ValueError(f"Unsupported image shape for CPSAM: {img.shape}")


# ------------------------------
# CPSAM fine-tuning
# ------------------------------

def fine_tune_cpsam(train_dir: str,
                    test_dir: str | None,
                    model_name: str,
                    use_gpu: bool,
                    learning_rate: float,
                    weight_decay: float,
                    n_epochs: int,
                    train_batch_size: int,
                    img_filter: str,
                    mask_filter: str) -> str:
    """
    Fine-tune CPSAM on image/mask training pairs and return the saved model path.

    Parameters are passed directly to Cellpose's training API. If `test_dir` is
    omitted, the training directory is reused as the test directory, matching
    Cellpose's expected train/test data loading workflow.
    """
    io.logger_setup()

    if test_dir is None:
        test_dir = train_dir

    output = io.load_train_test_data(
        train_dir, test_dir,
        image_filter=img_filter,
        mask_filter=mask_filter,
        look_one_level_down=False
    )
    images, labels, _, test_images, test_labels, _ = output

    images = [ensure_three_channels(im) for im in images]
    test_images = [ensure_three_channels(im) for im in test_images]

    model = models.CellposeModel(gpu=use_gpu, pretrained_model="cpsam")

    model_path, _, _ = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        test_data=test_images,
        test_labels=test_labels,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        model_name=model_name,
        batch_size=train_batch_size,
        min_train_masks=0,
    )
    return model_path


# ------------------------------
# Region statistics
# ------------------------------

def compute_region_stats(masks: np.ndarray, um_per_px: float | None = None):
    """
    Compute per-instance size statistics from a labeled mask image.

    Parameters
    ----------
    masks : np.ndarray
        Integer label image where 0 is background and each positive value is
        one predicted CFU/region.
    um_per_px : float | None
        Optional pixel calibration. When provided, area and equivalent diameter
        are also reported in micrometer units.

    Returns
    -------
    stats : list[dict]
        One row per nonzero label with area and equivalent diameter fields.
    n_instances : int
        Number of nonzero labels.
    """
    labels = np.unique(masks)
    labels = labels[labels != 0]

    stats = []
    for lab in labels:
        area_px = int(np.sum(masks == lab))
        eq_diam_px = float(2.0 * np.sqrt(area_px / np.pi)) if area_px > 0 else 0.0

        row = {
            "label": int(lab),
            "area_px": area_px,
            "eq_diameter_px": eq_diam_px,
        }

        if um_per_px is not None:
            area_um2 = float(area_px * (um_per_px ** 2))
            eq_diam_um = float(eq_diam_px * um_per_px)
            row.update({
                "area_um2": area_um2,
                "eq_diameter_um": eq_diam_um,
            })
        else:
            row.update({
                "area_um2": "",
                "eq_diameter_um": "",
            })

        stats.append(row)

    return stats, int(len(labels))


# ------------------------------
# Overlay visualization helpers
# ------------------------------

def to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert a numeric image to uint8 using per-image min/max normalization.

    Already-uint8 images are returned unchanged. Constant images become zeros
    to avoid divide-by-zero and to make the output deterministic.
    """
    if img.dtype == np.uint8:
        return img
    imgf = img.astype(np.float32)
    ptp = float(np.ptp(imgf))
    if ptp < 1e-8:
        return np.zeros_like(imgf, dtype=np.uint8)
    out = (255.0 * (imgf - float(imgf.min())) / (ptp + 1e-8)).astype(np.uint8)
    return out


def draw_mini_histogram(
    overlay_rgb: np.ndarray,
    values: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    bins: int,
    x_max: float,
    title: str = "Size",
):
    """
    Draw a compact histogram panel directly into an RGB overlay image.

    The panel is clipped to the image bounds, uses `values` as a one-dimensional
    size metric array, and clips values above `x_max` before binning. This is a
    visualization-only helper; it does not modify count statistics.
    """
    H, W, _ = overlay_rgb.shape
    x0 = int(np.clip(x0, 0, W - 1))
    y0 = int(np.clip(y0, 0, H - 1))
    w = int(np.clip(w, 10, W - x0))
    h = int(np.clip(h, 10, H - y0))

    panel = overlay_rgb[y0:y0+h, x0:x0+w].copy()

    # Background box
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (30, 30, 30), thickness=-1)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (220, 220, 220), thickness=1)

    # Title
    cv2.putText(panel, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

    # If no values, write message
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        cv2.putText(panel, "no data", (6, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        overlay_rgb[y0:y0+h, x0:x0+w] = panel
        return

    # Clip
    vals = np.clip(vals, 0.0, float(x_max))

    # Histogram
    hist, edges = np.histogram(vals, bins=int(bins), range=(0.0, float(x_max)))
    hist = hist.astype(np.float32)
    maxc = float(hist.max()) if hist.size else 1.0
    if maxc < 1e-6:
        maxc = 1.0

    # Plot area
    pad_l, pad_r, pad_t, pad_b = 6, 6, 22, 10
    px0, py0 = pad_l, pad_t
    px1, py1 = w - pad_r, h - pad_b
    plot_w = max(1, px1 - px0)
    plot_h = max(1, py1 - py0)

    # Axes
    cv2.line(panel, (px0, py1), (px1, py1), (180, 180, 180), 1)
    cv2.line(panel, (px0, py0), (px0, py1), (180, 180, 180), 1)

    # Bars
    nb = hist.size
    for i in range(nb):
        x_left = int(px0 + (i / nb) * plot_w)
        x_right = int(px0 + ((i + 1) / nb) * plot_w)
        bar_h = int((hist[i] / maxc) * plot_h)
        y_top = py1 - bar_h
        cv2.rectangle(panel, (x_left, y_top), (max(x_left + 1, x_right - 1), py1), (120, 220, 120), thickness=-1)

    # x_max label
    cv2.putText(panel, f"0..{x_max:g}", (px0, h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)

    overlay_rgb[y0:y0+h, x0:x0+w] = panel


# ------------------------------
# CPSAM inference + overlay export
# ------------------------------

def run_inference(pretrained_model_path: str,
                  infer_dir: str,
                  out_dir: str,
                  use_gpu: bool,
                  flow_threshold: float,
                  cellprob_threshold: float,
                  min_size: int,
                  tm_threshold: int = 300,
                  um_per_px: float | None = None,
                  hist_metric: str = "eq_diameter_px",
                  hist_bins: int = 24,
                  hist_max_x: float = 300.0,
                  hist_w: int = 720,
                  hist_h: int = 480):
    """
    Run CPSAM inference, save overlay PNGs, and return CSV-ready result rows.

    Each input image is converted to three channels, segmented with the selected
    CPSAM model, summarized by instance area/equivalent diameter, and rendered
    as a green-tinted overlay. The mini histogram is drawn in the top-right of
    each overlay. CSV writing is handled by `main` so total runtime can be added
    uniformly after inference finishes.
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading CPSAM model: {pretrained_model_path}")
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained_model_path)

    paths = list_images(infer_dir)
    if not paths:
        raise RuntimeError(f"No images found in inference dir: {infer_dir}")

    print(f"[INFO] Found {len(paths)} images for inference")
    print(f"[INFO] Saving outputs to: {out_dir}")
    print("[INFO] Saving: overlays only (no mask TIFFs)")

    test_rows = []

    for p in tqdm(paths, desc="Segmenting images", unit="img"):
        img = io.imread(p)
        img = ensure_three_channels(img)

        masks, _, _ = model.eval(
            img,
            channel_axis=-1,
            normalize=True,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
        )

        stats, num_blobs = compute_region_stats(masks, um_per_px=um_per_px)

        count_out = "TM" if num_blobs > tm_threshold else num_blobs
        fname = Path(p).name
        image_stem = Path(p).stem
        mean_area_px = float(np.mean([r["area_px"] for r in stats])) if stats else 0.0
        if not stats:
            test_rows.append({
                "image_name": fname,
                "petri_dish": image_stem,
                "dish_index": 1,
                "count": int(num_blobs),
                "cfu_index": "",
                "cfu_area_px": "",
                "mean_area_px": mean_area_px,
                "methods": "CPSAM",
                "classes": "",
            })
        else:
            for i, r in enumerate(stats, start=1):
                test_rows.append({
                    "image_name": fname,
                    "petri_dish": image_stem,
                    "dish_index": 1,
                    "count": int(num_blobs),
                    "cfu_index": i,
                    "cfu_area_px": float(r["area_px"]),
                    "mean_area_px": mean_area_px,
                    "methods": "CPSAM",
                    "classes": "",
                })

        # ---- build overlay ----
        stem = Path(p).stem
        overlay_path = os.path.join(out_dir, f"{stem}_overlay.png")

        img8 = to_uint8(img)
        overlay = img8.copy()

        # Mask tint
        green = np.array([144, 238, 144], dtype=np.uint8)  # RGB light green
        alpha = 0.4
        mask_pixels = masks > 0
        overlay[mask_pixels] = ((1 - alpha) * overlay[mask_pixels] + alpha * green).astype(np.uint8)

        # Count text (top-left)
        cv2.putText(
            overlay,
            f"Count: {count_out}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Mini histogram (top-right)
        # Gather per-colony size values for this image
        if hist_metric == "area_px":
            vals = np.array([r["area_px"] for r in stats], dtype=np.float32)
            title = "Area (px)"
        elif hist_metric == "eq_diameter_px":
            vals = np.array([r["eq_diameter_px"] for r in stats], dtype=np.float32)
            title = "EqDiam (px)"
        elif hist_metric == "area_um2":
            vals = pd_to_float_array([r["area_um2"] for r in stats])
            title = "Area (um^2)"
        elif hist_metric == "eq_diameter_um":
            vals = pd_to_float_array([r["eq_diameter_um"] for r in stats])
            title = "EqDiam (um)"
        else:
            # fallback
            vals = np.array([r["eq_diameter_px"] for r in stats], dtype=np.float32)
            title = f"{hist_metric} (fallback)"

        H, W, _ = overlay.shape
        x0 = W - hist_w - 20
        y0 = 20
        draw_mini_histogram(
            overlay_rgb=overlay,
            values=vals,
            x0=x0,
            y0=y0,
            w=hist_w,
            h=hist_h,
            bins=hist_bins,
            x_max=hist_max_x,
            title=title,
        )

        io.imsave(overlay_path, overlay)

    print("[OK] Inference complete")
    return test_rows


def pd_to_float_array(vals):
    """
    Convert optional numeric CSV-style values to a float32 NumPy array.

    Empty strings and None values are skipped. This keeps histogram generation
    robust when micrometer calibration is not available.
    """
    out = []
    for v in vals:
        try:
            if v == "" or v is None:
                continue
            out.append(float(v))
        except Exception:
            continue
    return np.array(out, dtype=np.float32)


# ------------------------------
# Command-line interface + CSV export
# ------------------------------

def main():
    """
    Parse command-line options, optionally fine-tune CPSAM, run inference, and
    write the overlay summary CSV.

    The script supports two workflows:
    1. Fine-tune from `--train_dir`, then infer on `--infer_dir`.
    2. Use `--skip_train` with built-in `cpsam` or `--pretrained_model`.
    """
    t0_total = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dir", type=str, default=None,
                    help="Directory of training images + *_masks files (omit when --skip_train).")
    ap.add_argument("--test_dir", type=str, default=None,
                    help="Optional folder with test images+labels; if omitted uses train_dir.")
    ap.add_argument("--infer_dir", type=str, required=True,
                    help="Directory of images to segment (no masks needed).")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output folder for overlays + CSV (no mask TIFFs).")
    ap.add_argument("--model_name", type=str, default="cpsam_finetuned",
                    help="Name for the fine-tuned model.")
    ap.add_argument("--use_gpu", action="store_true",
                    help="Use GPU if available.")
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--n_epochs", type=int, default=100)
    ap.add_argument("--train_batch_size", type=int, default=1)
    ap.add_argument("--img_filter", type=str, default="_expt",
                    help="Suffix identifying training images (e.g. *_expt.*).")
    ap.add_argument("--mask_filter", type=str, default="_output_masks",
                    help="Suffix identifying training masks (e.g. *_output_masks.*).")
    ap.add_argument("--flow_threshold", type=float, default=0.4)
    ap.add_argument("--cellprob_threshold", type=float, default=0.0)
    ap.add_argument("--min_size", type=int, default=15)
    ap.add_argument("--tm_threshold", type=int, default=300,
                    help='If predicted count exceeds this value, write "TM" instead.')
    ap.add_argument("--skip_train", action="store_true",
                    help="Skip fine-tuning and just run built-in cpsam (or a provided model).")
    ap.add_argument("--pretrained_model", type=str, default=None,
                    help="If --skip_train, use this pretrained model path/name (default: cpsam).")
    ap.add_argument("--um_per_px", type=float, default=None,
                    help="Optional calibration: micrometers per pixel. Enables area_um2 and eq_diameter_um.")
    ap.add_argument("--hist_metric", default="eq_diameter_px",
                    choices=["eq_diameter_px", "area_px", "eq_diameter_um", "area_um2"],
                    help="Metric to show as mini histogram on overlay (top-right).")
    ap.add_argument("--hist_bins", type=int, default=24, help="Bins for mini histogram.")
    ap.add_argument("--hist_max_x", type=float, default=300.0, help="Max x for mini histogram (values clipped).")
    ap.add_argument("--hist_w", type=int, default=720, help="Mini histogram width in pixels.")
    ap.add_argument("--hist_h", type=int, default=480, help="Mini histogram height in pixels.")

    args = ap.parse_args()

    if args.skip_train:
        pretrained = args.pretrained_model or "cpsam"
        print(f"[INFO] Using pretrained model: {pretrained}")
    else:
        if args.train_dir is None:
            raise ValueError("--train_dir must be provided unless --skip_train is set.")

        pretrained = fine_tune_cpsam(
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            model_name=args.model_name,
            use_gpu=args.use_gpu,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            n_epochs=args.n_epochs,
            train_batch_size=args.train_batch_size,
            img_filter=args.img_filter,
            mask_filter=args.mask_filter,
        )
        print(f"[OK] Fine-tuned model saved at: {pretrained}")

    test_rows = run_inference(
        pretrained_model_path=pretrained,
        infer_dir=args.infer_dir,
        out_dir=args.out_dir,
        use_gpu=args.use_gpu,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        tm_threshold=args.tm_threshold,
        um_per_px=args.um_per_px,
        hist_metric=args.hist_metric,
        hist_bins=args.hist_bins,
        hist_max_x=args.hist_max_x,
        hist_w=args.hist_w,
        hist_h=args.hist_h,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    total_runtime_sec = float(time.perf_counter() - t0_total)
    for row in test_rows:
        row["total_runtime_sec"] = total_runtime_sec

    csv_out = os.path.join(args.out_dir, "countCFUAPP_test_results.csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"[OK] Saved count/size CSV to {csv_out}")
    print(f"[OK] Total runtime: {total_runtime_sec:.3f} s")
    print(f"[OK] Saved overlays to: {args.out_dir}")


if __name__ == "__main__":
    main()
