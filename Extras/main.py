#!/usr/bin/env python3
from __future__ import annotations

import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
WEB_JOBS = ROOT / "web_jobs"
STATIC_DIR = Path(__file__).resolve().parent / "static"
SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

app = FastAPI(title="CARA iPad Postprocessing")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class SessionState:
    session_id: str
    created_at: str
    rows: List[dict] = field(default_factory=list)
    saved: List[str] = field(default_factory=list)
    saved_local: List[str] = field(default_factory=list)


SESSIONS: Dict[str, SessionState] = {}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _save_uploads(files: List[UploadFile], input_dir: Path) -> List[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.filename).name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        p = input_dir / name
        with p.open("wb") as w:
            w.write(f.file.read())
        saved.append(p)
    return saved


def _preview_png(src: Path, dst: Path):
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        return
    def to_u8(x: np.ndarray) -> np.ndarray:
        if x.dtype == np.uint8:
            return x
        # Robust normalization for 16-bit/float microscope images.
        y = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
        return y.astype(np.uint8)
    if img.ndim == 2:
        img = to_u8(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = to_u8(img)
        rgb = img
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), rgb)


def _read_rgb_anydepth(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    def to_u8(x: np.ndarray) -> np.ndarray:
        if x.dtype == np.uint8:
            return x
        y = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
        return y.astype(np.uint8)
    if img.ndim == 2:
        img = to_u8(img)
        return np.stack([img, img, img], axis=-1)
    img = to_u8(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask_anydepth(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read mask: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return ((img > 0).astype(np.uint8) * 255)


def _stitch_side_by_side(a: np.ndarray, b: np.ndarray, gap: int = 12) -> np.ndarray:
    ha, hb = a.shape[0], b.shape[0]
    H = max(ha, hb)
    if ha < H:
        a = np.pad(a, ((0, H - ha), (0, 0), (0, 0)), mode="constant", constant_values=0)
    if hb < H:
        b = np.pad(b, ((0, H - hb), (0, 0), (0, 0)), mode="constant", constant_values=0)
    spacer = np.zeros((H, gap, 3), dtype=np.uint8)
    return np.concatenate([a, spacer, b], axis=1)


def _stitch_mask_side_by_side(a: np.ndarray, b: np.ndarray, gap: int = 12) -> np.ndarray:
    ha, hb = a.shape[0], b.shape[0]
    H = max(ha, hb)
    if ha < H:
        a = np.pad(a, ((0, H - ha), (0, 0)), mode="constant", constant_values=0)
    if hb < H:
        b = np.pad(b, ((0, H - hb), (0, 0)), mode="constant", constant_values=0)
    spacer = np.zeros((H, gap), dtype=np.uint8)
    return np.concatenate([a, spacer, b], axis=1)


def _extract_yellow_mask(rgb: np.ndarray) -> np.ndarray:
    mask = (
        (rgb[:, :, 0] >= 220)
        & (rgb[:, :, 1] >= 220)
        & (rgb[:, :, 2] <= 140)
    )
    return (mask.astype(np.uint8) * 255)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/session/create")
def create_session(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    session_id = uuid.uuid4().hex[:10]
    base = WEB_JOBS / "sessions" / session_id
    input_dir = base / "input"
    preview_dir = base / "preview"
    saved_dir = base / "saved"
    saved_dir.mkdir(parents=True, exist_ok=True)

    saved = _save_uploads(files, input_dir)
    if not saved:
        raise HTTPException(status_code=400, detail="No supported images uploaded")

    rows = []
    for p in saved:
        prev = preview_dir / f"{p.stem}.png"
        _preview_png(p, prev)
        rows.append({
            "image": p.name,
            "preview_url": f"/web_jobs/sessions/{session_id}/preview/{prev.name}",
        })

    SESSIONS[session_id] = SessionState(
        session_id=session_id,
        created_at=_now(),
        rows=rows,
        saved=[],
    )
    return {"session_id": session_id, "rows": rows}


@app.post("/api/session/from_app")
def create_session_from_app(payload: dict = Body(...)):
    rows_in = payload.get("rows") or []
    if not rows_in:
        raise HTTPException(status_code=400, detail="No rows provided")

    session_id = uuid.uuid4().hex[:10]
    base = WEB_JOBS / "sessions" / session_id
    preview_dir = base / "preview"
    saved_dir = base / "saved"
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in enumerate(rows_in):
        image_name = str(r.get("image") or "")
        src_path = Path(str(r.get("source_path") or ""))
        tif_paths = [Path(str(p)) for p in (r.get("tif_paths") or []) if str(p)]
        expt_paths = [Path(str(p)) for p in (r.get("expt_paths") or []) if str(p)]
        post_mask_paths = [Path(str(p)) for p in (r.get("post_mask_paths") or []) if str(p)]
        if not src_path.exists():
            continue
        try:
            orig_rgb = _read_rgb_anydepth(src_path)
        except Exception:
            continue
        orig_prev = preview_dir / f"row{i:04d}__original.png"
        cv2.imwrite(str(orig_prev), cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))

        if len(tif_paths) >= 2 and tif_paths[0].exists() and tif_paths[1].exists():
            m0 = _read_rgb_anydepth(tif_paths[0])
            m1 = _read_rgb_anydepth(tif_paths[1])
            masked_rgb = _stitch_side_by_side(m0, m1, gap=12)
        elif len(tif_paths) >= 1 and tif_paths[0].exists():
            masked_rgb = _read_rgb_anydepth(tif_paths[0])
        else:
            masked_rgb = orig_rgb.copy()
        masked_prev = preview_dir / f"row{i:04d}__masked.png"
        cv2.imwrite(str(masked_prev), cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2BGR))

        if len(expt_paths) >= 2 and expt_paths[0].exists() and expt_paths[1].exists():
            b0 = _read_rgb_anydepth(expt_paths[0])
            b1 = _read_rgb_anydepth(expt_paths[1])
            mask_base_rgb = _stitch_side_by_side(b0, b1, gap=12)
        elif len(expt_paths) >= 1 and expt_paths[0].exists():
            mask_base_rgb = _read_rgb_anydepth(expt_paths[0])
        else:
            mask_base_rgb = masked_rgb.copy()
        mask_base_prev = preview_dir / f"row{i:04d}__mask_base.png"
        cv2.imwrite(str(mask_base_prev), cv2.cvtColor(mask_base_rgb, cv2.COLOR_RGB2BGR))

        algo_mask_url = None
        yellow_mask_url = None
        try:
            if len(post_mask_paths) >= 2 and post_mask_paths[0].exists() and post_mask_paths[1].exists():
                pm0 = _read_mask_anydepth(post_mask_paths[0])
                pm1 = _read_mask_anydepth(post_mask_paths[1])
                algo_mask = _stitch_mask_side_by_side(pm0, pm1, gap=12)
            elif len(post_mask_paths) >= 1 and post_mask_paths[0].exists():
                algo_mask = _read_mask_anydepth(post_mask_paths[0])
            else:
                algo_mask = None
            if algo_mask is not None:
                algo_mask_prev = preview_dir / f"row{i:04d}__algo_mask.png"
                cv2.imwrite(str(algo_mask_prev), algo_mask)
                algo_mask_url = f"/web_jobs/sessions/{session_id}/preview/{algo_mask_prev.name}"
        except Exception:
            algo_mask_url = None

        try:
            if len(tif_paths) >= 2 and tif_paths[0].exists() and tif_paths[1].exists():
                ym0 = _extract_yellow_mask(_read_rgb_anydepth(tif_paths[0]))
                ym1 = _extract_yellow_mask(_read_rgb_anydepth(tif_paths[1]))
                yellow_mask = _stitch_mask_side_by_side(ym0, ym1, gap=12)
            elif len(tif_paths) >= 1 and tif_paths[0].exists():
                yellow_mask = _extract_yellow_mask(_read_rgb_anydepth(tif_paths[0]))
            else:
                yellow_mask = None
            if yellow_mask is not None:
                yellow_mask_prev = preview_dir / f"row{i:04d}__yellow_mask.png"
                cv2.imwrite(str(yellow_mask_prev), yellow_mask)
                yellow_mask_url = f"/web_jobs/sessions/{session_id}/preview/{yellow_mask_prev.name}"
        except Exception:
            yellow_mask_url = None

        rows.append({
            "image": image_name or src_path.name,
            "mask_name": f"{Path(image_name or src_path.name).stem}_masked.png",
            "source_path": str(src_path),
            "original_url": f"/web_jobs/sessions/{session_id}/preview/{orig_prev.name}",
            "masked_url": f"/web_jobs/sessions/{session_id}/preview/{masked_prev.name}",
            "mask_base_url": f"/web_jobs/sessions/{session_id}/preview/{mask_base_prev.name}",
            "algo_mask_url": algo_mask_url,
            "yellow_mask_url": yellow_mask_url,
        })

    if not rows:
        raise HTTPException(status_code=400, detail="No valid rows from app")

    SESSIONS[session_id] = SessionState(
        session_id=session_id,
        created_at=_now(),
        rows=rows,
        saved=[],
        saved_local=[],
    )
    return {"session_id": session_id, "rows": rows}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return asdict(s)


@app.post("/api/session/{session_id}/save")
def save_annotated(
    session_id: str,
    image_name: str = Form(...),
    file: UploadFile = File(...),
):
    s = SESSIONS.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    base = WEB_JOBS / "sessions" / session_id
    saved_dir = base / "saved"
    stem = Path(image_name).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = saved_dir / f"{stem}_annotated_{ts}.png"
    with out.open("wb") as w:
        w.write(file.file.read())

    rel = f"/web_jobs/sessions/{session_id}/saved/{out.name}"
    if rel not in s.saved:
        s.saved.append(rel)
    out_local = str(out.resolve())
    if out_local not in s.saved_local:
        s.saved_local.append(out_local)
    return {"ok": True, "saved_url": rel, "saved_local": out_local}


@app.get("/")
def root_page():
    return FileResponse(str(STATIC_DIR / "index.html"))


app.mount("/web_jobs", StaticFiles(directory=str(WEB_JOBS)), name="web_jobs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webui.main:app", host="0.0.0.0", port=8000, reload=True)
