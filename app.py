#!/usr/bin/env python3
# app.py  (TIFF outputs + Halves + Clean UI + Postprocessing tools + CSV save)
#
# FIXES / UPGRADES (starting from YOUR code above, but corrected & improved):
# 1) Center logo more in Controls ribbon (bottom; horizontally centered).
# 2) Zoom + pan with hover overlay buttons on the preview:
#    - When you hover over the preview, you see 3 buttons:
#        [+] zoom in, [-] zoom out, [↔] pan mode
#    - In pan mode: left-drag pans the image
#    - Double click resets view
#    - (wheel zoom is also supported)
# 3) Paint/Remove drag: click-and-drag to continuously paint/remove (no re-clicking).
# 4) Progress bar chunk is GREEN.
# 5) Select numbers larger (bigger font + thicker stroke).
# 6) Table white bug fix: strong stylesheet for QTableWidget/QTableView/viewport/items + explicit palette.
#
# IMPORTANT:
# - Your pasted code had a syntax-breaking stray character: `app = QApplication(sys.argv)å`
#   That is fixed.
# - Postprocessing is VISUAL ONLY (does not overwrite TIFF outputs).
# - CSV: one row per INPUT image:
#     Dataname = input stem
#     Count    = number of selection labels (yellow numbers) (masked+original)
#   "Save CSV" writes results.csv into output directory.
from PySide6.QtGui import QKeySequence, QShortcut
import sys
import csv
import re
import shutil
import shlex
import traceback
import time
import subprocess
import tempfile
import socket
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PySide6.QtWidgets import QComboBox
import numpy as np
import cv2
from dataclasses import dataclass, field
import copy
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool, QRect, QRectF, QPoint, QSize, QTimer, QModelIndex, QSettings, QByteArray
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QCursor, QPen, QIcon, QPainterPath, QPainter
try:
    from PySide6.QtSvg import QSvgRenderer as _QSvgRenderer
    _HAS_SVG = True
except ImportError:
    _HAS_SVG = False
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QGroupBox, QRadioButton, QButtonGroup, QSpinBox, QToolButton, QListView, QFrame,
    QDialog, QPlainTextEdit
)
from PySide6.QtWidgets import QSlider
from countCFUAPP2 import count_cfu_app2
from PySide6.QtWidgets import QCheckBox
import redesign_patch
redesign_patch.install()

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
VERSION_FOLDER = ""  # keep as in your code
GT_TOKEN_RE = re.compile(r"^([A-Za-z0-9]+)dilu(?:e)?(\d+)$", re.IGNORECASE)

# ── Direct GPU server defaults (all values are editable in the dialog UI) ────
GPU_HOST_DEFAULT   = "100.126.92.100"          # Tailscale IP of the GPU machine
GPU_PORT_DEFAULT   = 22
GPU_KEY_DEFAULT    = "~/.ssh/gpu_server_key"   # SSH private key (no passphrase)
GPU_USER_DEFAULT   = "justin"                  # username on the GPU machine
GPU_CPSAM_ROOT     = "/home/justin/Documents/pretrained"   # folder containing CPSAM.py
GPU_CONDA_ENV      = "cellpose"
GPU_CPSAM_SCRIPT   = "CPSAM.py"
GPU_CPSAM_MODEL    = "/home/justin/Documents/pretrained/cpsam_finetuned.zip"
# ─────────────────────────────────────────────────────────────────────────────

SHARED_SLIDER_QSS = """
QSlider {
    min-height: 18px;
}
QSlider::groove:horizontal {
    border: none;
    height: 4px;
    background: #1d2029;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #4781d1;
    border-radius: 2px;
}
QSlider::add-page:horizontal {
    background: #1d2029;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #f2f5fb;
    border: 1px solid rgba(0,0,0,100);
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::handle:horizontal:hover {
    border: 2px solid rgba(138,180,255,90);
}
"""

# Subtle badge-style QPushButton for use inside the title bar.
# Overrides the global QSS which would apply 12px radius + gradient.
_TITLEBAR_BTN_QSS = """
QPushButton {
    background: rgba(255,255,255,5);
    border: 1px solid rgba(255,255,255,20);
    border-radius: 7px;
    padding: 3px 10px;
    color: #9aa1ae;
    font-size: 12px;
    font-weight: 500;
}
QPushButton:hover {
    background: rgba(255,255,255,10);
    border: 1px solid rgba(255,255,255,38);
    color: #e7eaf0;
}
QPushButton:pressed {
    background: rgba(0,0,0,20);
    border: 1px solid rgba(255,255,255,15);
    color: #9aa1ae;
}
"""


class ColorComboBox(QComboBox):
    # Swatch colors keyed to item text
    _SWATCH = {
        "Blue":  "#4d9ef6",
        "Pink":  "#f06db0",
        "Red":   "#f06060",
        "Green": "#4ade80",
        "Black": "#666666",
    }

    def _make_swatch_icon(self, hex_color: str, size: int = 12) -> QIcon:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(QColor(hex_color))
        p.setPen(QColor(255, 255, 255, 40))
        p.drawEllipse(1, 1, size - 2, size - 2)
        p.end()
        return QIcon(pm)

    def showPopup(self):
        # Attach swatch icons to items on first show
        model = self.model()
        if model is not None:
            for i in range(model.rowCount()):
                txt = self.itemText(i)
                if txt in self._SWATCH and self.itemIcon(i).isNull():
                    self.setItemIcon(i, self._make_swatch_icon(self._SWATCH[txt]))
        super().showPopup()
        QTimer.singleShot(0, self._polish_popup)

    def _polish_popup(self):
        view = self.view()
        if view is None:
            return
        selection_model = view.selectionModel()
        if selection_model is not None:
            selection_model.clearSelection()
        view.setCurrentIndex(QModelIndex())
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setContentsMargins(4, 4, 4, 4)
        popup = view.window()
        popup.setContentsMargins(0, 0, 0, 0)
        # Design-palette dark styling
        BG = "#0d0f14"
        popup.setStyleSheet(
            f"background: {BG};"
            "border: 1px solid rgba(255,255,255,18);"
            "border-radius: 10px;"
        )
        view.setStyleSheet(
            f"""
            QListView, QListView::viewport {{
                background: {BG};
                border: none;
                outline: 0;
                padding: 4px;
            }}
            QListView::item {{
                color: #e7eaf0;
                font-family: "Geist", "Inter", "SF Pro Text", sans-serif;
                font-size: 13px;
                padding: 6px 10px;
                border-radius: 6px;
                min-height: 20px;
            }}
            QListView::item:hover {{
                background: rgba(255,255,255,6);
            }}
            QListView::item:selected {{
                background: rgba(138,180,255,18);
                color: #e7eaf0;
            }}
            """
        )
        pal = view.palette()
        pal.setColor(QPalette.Base,            QColor(BG))
        pal.setColor(QPalette.Window,          QColor(BG))
        pal.setColor(QPalette.Text,            QColor("#e7eaf0"))
        pal.setColor(QPalette.Highlight,       QColor(138, 180, 255, 40))
        pal.setColor(QPalette.HighlightedText, QColor("#e7eaf0"))
        view.setPalette(pal)
        row_count = view.model().rowCount() if view.model() is not None else 0
        if row_count > 0:
            row_h = max(view.sizeHintForRow(i) for i in range(row_count))
            total_h = row_h * row_count + 10
            popup.resize(popup.width(), total_h)
        view.viewport().update()


def _make_svg_icon(svg_body: str, size: int = 16, color: str = "#e7eaf0") -> QIcon:
    """Render a 24×24-viewBox SVG path snippet to a QIcon — renders at 2× for crispness."""
    if not _HAS_SVG:
        return QIcon()
    render_size = size * 2  # render at 2× then let Qt scale
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{render_size}" height="{render_size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round">{svg_body}</svg>'
    )
    pm = QPixmap(render_size, render_size)
    pm.fill(Qt.transparent)
    renderer = _QSvgRenderer(QByteArray(svg.encode()))
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(p)
    p.end()
    # Scale down to target size with smooth filter
    pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return QIcon(pm)


# SVG paths for tool buttons (viewBox 0 0 24 24)
_ICON_PAINT  = '<path d="M4 20l4-1 10-10-3-3L5 16l-1 4zM13 7l4 4"/>'
_ICON_SELECT = '<path d="M4 4l6 16 2-7 7-2L4 4z"/>'
_ICON_REMOVE = '<path d="M6 6l12 12M18 6L6 18"/>'


class KbdBadgeButton(QPushButton):
    """QPushButton that paints a small keyboard-shortcut badge in its right margin."""

    def __init__(self, text: str = "", kbd: str = "", parent=None):
        super().__init__(text, parent)
        self._kbd = kbd

    def setKbd(self, kbd: str):
        self._kbd = kbd
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._kbd:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        small_font = QFont(p.font())
        small_font.setPointSizeF(9.5)
        small_font.setBold(False)
        p.setFont(small_font)
        fm = p.fontMetrics()

        margin_r = 8
        pad_x = 6
        pad_y = 3
        tw = fm.horizontalAdvance(self._kbd)
        badge_w = tw + pad_x * 2
        badge_h = fm.height() + pad_y * 2
        bx = self.width() - badge_w - margin_r
        by = (self.height() - badge_h) // 2

        rect = QRectF(bx, by, badge_w, badge_h)
        p.setBrush(QColor(255, 255, 255, 10))
        p.setPen(QColor(255, 255, 255, 35))
        p.drawRoundedRect(rect, 4, 4)
        p.setPen(QColor("#5f6675"))
        p.drawText(rect, Qt.AlignCenter, self._kbd)
        p.end()


DEFAULT_DARK_APP_QSS = """
QMainWindow, QWidget { background: #07080b; color: #e7eaf0; }
QLabel { color: #e7eaf0; background: transparent; }

QGroupBox {
    border: 1px solid rgba(255,255,255,15);
    border-radius: 18px;
    margin-top: 10px;
    padding: 16px;
    background: #0e1118;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
    color: #9aa1ae;
    background: transparent;
}

QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {
    background: #0a0c11;
    border: 1px solid rgba(255,255,255,15);
    border-radius: 12px;
    padding: 8px 10px;
    color: #e7eaf0;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: none;
    background: #0a0c11;
    border-top-right-radius: 12px;
    border-bottom-right-radius: 12px;
}
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
    margin-right: 6px;
}

QPushButton {
    background: #0f1218;
    border: 1px solid rgba(255,255,255,31);
    border-radius: 12px;
    padding: 11px 14px;
    color: #e7eaf0;
    font-weight: 500;
}
QPushButton:hover { border: 1px solid rgba(138,180,255,115); background: #141921; }
QPushButton:pressed { background: #080a0f; }
QPushButton[primary="true"] {
    background: #192e52;
    border: 1px solid rgba(138,180,255,140);
    font-weight: 600;
}
QPushButton[primary="true"]:hover { background: #1e3660; border: 1px solid rgba(138,180,255,191); }
QPushButton[danger="true"] { background: #3a0f0f; border: 1px solid rgba(255,133,133,128); color: #ffd3d3; }
QPushButton[danger="true"]:hover { background: #441212; border: 1px solid rgba(255,133,133,165); }
QPushButton[success="true"] { background: #163a20; border: 1px solid rgba(140,230,164,140); color: #d6f5de; font-weight: 700; }
QPushButton[success="true"]:hover { background: #1a4426; border: 1px solid rgba(140,230,164,191); }

QProgressBar {
    border: 1px solid rgba(255,255,255,15);
    border-radius: 5px;
    background: #0a0c11;
    color: transparent;
    min-height: 10px;
    max-height: 10px;
    padding: 1px;
}
QProgressBar::chunk { background: #3dc66a; border-radius: 4px; }

QHeaderView::section {
    background: rgba(255,255,255,4);
    border: none;
    border-bottom: 1px solid rgba(255,255,255,13);
    padding: 6px;
    color: #5f6675;
}
QTableCornerButton::section { background: transparent; border: none; }
QTableWidget, QTableView, QTableWidget::viewport, QTableView::viewport {
    background: #0d0f14;
    color: #e7eaf0;
    border: 1px solid rgba(255,255,255,13);
    gridline-color: rgba(255,255,255,10);
    selection-background-color: rgba(138,180,255,22);
    selection-color: #e7eaf0;
}
QTableWidget::item { background: transparent; color: #e7eaf0; }
QTableWidget::item:selected { background: rgba(138,180,255,22); color: #e7eaf0; }
QRadioButton, QCheckBox { color: #e7eaf0; background: transparent; spacing: 8px; }
QRadioButton::indicator, QCheckBox::indicator { width: 16px; height: 16px; }
QRadioButton::indicator { border: 1px solid rgba(255,255,255,31); border-radius: 8px; background: #0a0c11; }
QRadioButton::indicator:checked { border: 1px solid rgba(138,180,255,191); background: #f2f5fb; }
QCheckBox::indicator { border: 1px solid rgba(255,255,255,31); border-radius: 4px; background: #0a0c11; }
QCheckBox::indicator:checked { border: 1px solid rgba(138,180,255,191); background: #1f3a6e; }
QSlider::groove:horizontal { border: none; height: 4px; background: #1d2029; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #4781d1; border-radius: 2px; }
QSlider::handle:horizontal { background: #f2f5fb; border: 1px solid rgba(0,0,0,100); width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }
"""

# ----------------------------
# IO + image helpers
# ----------------------------
def cv_read_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def cv_read_rgb_anydepth(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    if img.ndim == 2:
        if img.dtype == np.uint16:
            img8 = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img8 = img
        return np.stack([img8, img8, img8], axis=-1)

    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype == np.uint16:
            img8 = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img8 = img
        return cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)

    if img.ndim == 3 and img.shape[2] == 4:
        if img.dtype == np.uint16:
            img8 = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img8 = img
        return cv2.cvtColor(img8, cv2.COLOR_BGRA2RGB)

    raise ValueError(f"Unsupported image shape: {img.shape} for {path}")


def cv_read_mask_anydepth(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read mask image: {path}")
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pad_to_same_height(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ha, _ = a.shape[:2]
    hb, _ = b.shape[:2]
    H = max(ha, hb)

    def pad(img: np.ndarray, H: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == H:
            return img
        pad_h = H - h
        return np.pad(img, ((0, pad_h), (0, 0), (0, 0)), mode="constant", constant_values=0)

    return pad(a, H), pad(b, H)


def stitch_side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 12) -> np.ndarray:
    left = left.astype(np.uint8)
    right = right.astype(np.uint8)
    left, right = _pad_to_same_height(left, right)
    H = left.shape[0]
    spacer = np.zeros((H, gap, 3), dtype=np.uint8)
    return np.concatenate([left, spacer, right], axis=1)


def stitch_mask_side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 12) -> np.ndarray:
    left = (left > 0).astype(np.uint8) * 255
    right = (right > 0).astype(np.uint8) * 255
    ha, wa = left.shape[:2]
    hb, wb = right.shape[:2]
    H = max(ha, hb)

    def pad_mask(img: np.ndarray, H: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == H:
            return img
        pad_h = H - h
        return np.pad(img, ((0, pad_h), (0, 0)), mode="constant", constant_values=0)

    left = pad_mask(left, H)
    right = pad_mask(right, H)
    spacer = np.zeros((H, gap), dtype=np.uint8)
    return np.concatenate([left, spacer, right], axis=1)


def make_output_ids_for_image(img_path: str) -> Tuple[str, str]:
    stem = Path(img_path).stem
    parts = stem.split("_")
    if len(parts) >= 3 and re.search(r"dilu\d+", parts[-2], re.IGNORECASE) and re.search(r"dilu\d+", parts[-1], re.IGNORECASE):
        prefix = "_".join(parts[:-2])
        return f"{prefix}_{parts[-2]}", f"{prefix}_{parts[-1]}"
    return stem, ""


# ----------------------------
# Interactive image widget (Zoom/Pan overlay buttons + drag paint support)
# ----------------------------
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize

from PySide6.QtWidgets import QWidget, QToolButton, QSizePolicy
import numpy as np

class ZoomPanCanvas(QWidget):
    clicked = Signal(int, int)
    drag = Signal(int, int)
    drag_end = Signal()
    view_changed = Signal()
    empty_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        # IMPORTANT: prevents layout/pixmap feedback loops
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)

        self._rgb = None
        self._img_w = 0
        self._img_h = 0
        self._bytes_per_line = 0

        self._fit_scale = 1.0
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        # Count overlay lines: list of (text, color_hex, is_big)
        self._count_overlay_lines: list = []

        self._pan_mode = False
        self._panning = False
        self._left_down = False
        self._pan_start_pos = None
        self._pan_start_xy = None
        self._empty_anim_phase = 0.0
        self._empty_anim_timer = QTimer(self)
        self._empty_anim_timer.setInterval(40)
        self._empty_anim_timer.timeout.connect(self._advance_empty_animation)
        self._empty_anim_timer.start()

        # hover overlay buttons
        self.btn_zoom_in = QToolButton(self); self.btn_zoom_in.setText("+")
        self.btn_zoom_out = QToolButton(self); self.btn_zoom_out.setText("−")
        self.btn_pan = QToolButton(self); self.btn_pan.setText("↔"); self.btn_pan.setCheckable(True)
        self.btn_reset_view = QToolButton(self); self.btn_reset_view.setText("⌂")

        for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_pan, self.btn_reset_view):
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            b.hide()

        tool_css = """
        QToolButton {
            background: rgba(10,12,18,200);
            border: 1px solid rgba(255,255,255,31);
            color: #e7eaf0;
            border-radius: 10px;
            padding: 6px 10px;
            font-weight: 700;
        }
        QToolButton:hover {
            border: 1px solid rgba(138,180,255,150);
            background: rgba(20,24,34,220);
        }
        QToolButton:checked {
            border: 1px solid rgba(138,180,255,200);
            background: rgba(25,43,72,200);
        }
        QToolButton:pressed {
            background: rgba(10,12,18,230);
            border: 1px solid rgba(138,180,255,100);
        }
        """
        self.btn_zoom_in.setStyleSheet(tool_css)
        self.btn_zoom_out.setStyleSheet(tool_css)
        self.btn_pan.setStyleSheet(tool_css)
        self.btn_reset_view.setStyleSheet(tool_css)

        self.btn_zoom_in.clicked.connect(lambda: self.zoom_by(1.15))
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_by(1.0 / 1.15))
        self.btn_pan.toggled.connect(self.set_pan_mode)
        self.btn_reset_view.clicked.connect(self.reset_view)

    # prevent QLabel-like pixmap sizeHint behavior
    def sizeHint(self):
        return QSize(640, 480)

    def minimumSizeHint(self):
        return QSize(200, 200)

    def set_pan_mode(self, on: bool):
        self._pan_mode = bool(on)
        self.setCursor(Qt.OpenHandCursor if self._pan_mode else Qt.ArrowCursor)
        self.view_changed.emit()

    def set_rgb(self, rgb: np.ndarray):
        rgb_u8 = rgb.astype(np.uint8, copy=False)
        rgb_u8 = np.ascontiguousarray(rgb_u8)  # IMPORTANT
        self._rgb = rgb_u8
        self._img_h, self._img_w = rgb_u8.shape[:2]
        self._bytes_per_line = int(rgb_u8.strides[0])
        self.update()
        self.view_changed.emit()


    def clear_rgb(self):
        self._rgb = None
        self._img_w = 0
        self._img_h = 0
        self._bytes_per_line = 0
        self._count_overlay_lines = []
        self.update()
        self.view_changed.emit()

    def set_count_overlay(self, lines: list):
        """Set count overlay lines painted at bottom-left of canvas.
        lines: list of (text: str, color_hex: str, is_big: bool)
        """
        self._count_overlay_lines = lines
        self.update()

    def _paint_count_overlay(self, painter):
        """Paint count info overlay labels at bottom-left of the canvas."""
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        pad_x, pad_y = 18, 18
        gap = 5
        corner_r = 8
        px_h_big, px_w_big = 8, 14   # padding for the big count pill
        px_h_sm,  px_w_sm  = 5, 10   # padding for small lines

        big_font = QFont("Geist Mono")
        big_font.setStyleHint(QFont.Monospace)
        big_font.setPointSizeF(20.0)
        big_font.setBold(True)
        big_font.setLetterSpacing(QFont.AbsoluteSpacing, -0.3)

        small_font = QFont("Geist Mono")
        small_font.setStyleHint(QFont.Monospace)
        small_font.setPointSizeF(9.5)
        small_font.setBold(False)

        # Measure each line
        box_list = []
        for text, color_hex, is_big in self._count_overlay_lines:
            painter.setFont(big_font if is_big else small_font)
            fm = painter.fontMetrics()
            pw = px_w_big if is_big else px_w_sm
            ph = px_h_big if is_big else px_h_sm
            bw = fm.horizontalAdvance(text) + pw * 2
            bh = fm.height() + ph * 2
            box_list.append((text, color_hex, is_big, bw, bh, pw, ph))

        start_x = pad_x
        start_y = pad_y   # fixed top-left, zoom-independent

        y_off = 0
        for text, color_hex, is_big, bw, bh, pw, ph in box_list:
            painter.setFont(big_font if is_big else small_font)
            rect = QRectF(start_x, start_y + y_off, bw, bh)
            if is_big:
                # Amber pill — coin style matching .overlay-labels .big
                painter.setBrush(QColor(0, 0, 0, 140))
                painter.setPen(QColor(255, 214, 107, 60))   # subtle amber border
                painter.drawRoundedRect(rect, corner_r, corner_r)
                painter.setPen(QColor("#ffd66b"))
            else:
                painter.setBrush(QColor(0, 0, 0, 145))
                painter.setPen(QColor(255, 255, 255, 18))
                painter.drawRoundedRect(rect, corner_r, corner_r)
                painter.setPen(QColor(color_hex))
            painter.drawText(
                rect.adjusted(pw, ph, -pw, -ph),
                Qt.AlignVCenter | Qt.AlignLeft,
                text,
            )
            y_off += bh + gap

        painter.setBrush(Qt.NoBrush)

    def _advance_empty_animation(self):
        if self._rgb is not None:
            return
        self._empty_anim_phase = (self._empty_anim_phase + 0.02) % 1.0
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()
        self.view_changed.emit()

    def zoom_by(self, factor: float):
        self._zoom = float(np.clip(self._zoom * factor, 0.1, 12.0))
        self.update()
        self.view_changed.emit()

    def enterEvent(self, e):
        super().enterEvent(e)
        if self._rgb is not None:
            self._place_overlay_buttons()
            self.btn_zoom_in.show(); self.btn_zoom_out.show()
            self.btn_pan.show(); self.btn_reset_view.show()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self.btn_zoom_in.hide(); self.btn_zoom_out.hide()
        self.btn_pan.hide(); self.btn_reset_view.hide()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_overlay_buttons()
        self.update()
        self.view_changed.emit()

    def display_scale(self) -> float:
        if self._rgb is None or self._img_w <= 0 or self._img_h <= 0:
            return 1.0
        fit = min(self.width() / self._img_w, self.height() / self._img_h)
        return float(max(0.01, fit * self._zoom))

    def _place_overlay_buttons(self):
        pad = 10
        w = 38
        h = 32
        gap = 8
        x = self.width() - pad - w
        y = pad
        self.btn_zoom_in.setGeometry(QRect(x, y, w, h))
        self.btn_zoom_out.setGeometry(QRect(x, y + (h + gap), w, h))
        self.btn_pan.setGeometry(QRect(x, y + (h + gap) * 2, w, h))
        self.btn_reset_view.setGeometry(QRect(x, y + (h + gap) * 3, w, h))

    def wheelEvent(self, e):
        if self._rgb is None:
            e.ignore()
            return

        dy = e.angleDelta().y()
        if dy:
            self.zoom_by(1.1 if dy > 0 else 1.0 / 1.1)

        e.accept()  # IMPORTANT: stops gesture propagation that breaks painting

    def mouseDoubleClickEvent(self, e):
        self.reset_view()

    def mousePressEvent(self, e):
        if self._rgb is None:
            if e.button() == Qt.LeftButton:
                self.empty_clicked.emit()
            return
        if e.button() == Qt.LeftButton:
            self._left_down = True

            if self._pan_mode:
                self._panning = True
                self._pan_start_pos = e.pos()
                self._pan_start_xy = (self._pan_x, self._pan_y)
                self.setCursor(Qt.ClosedHandCursor)
                return

            xi, yi = self._map_to_image(e.position().x(), e.position().y())
            if xi is not None:
                self.clicked.emit(xi, yi)
                self.drag.emit(xi, yi)

    def mouseMoveEvent(self, e):
        if self._rgb is None:
            return

        if self._panning and self._pan_start_pos is not None and self._pan_start_xy is not None:
            dx = e.pos().x() - self._pan_start_pos.x()
            dy = e.pos().y() - self._pan_start_pos.y()
            self._pan_x = self._pan_start_xy[0] + dx
            self._pan_y = self._pan_start_xy[1] + dy
            self.update()
            return

        if self._left_down and (not self._pan_mode):
            xi, yi = self._map_to_image(e.position().x(), e.position().y())
            if xi is not None:
                self.drag.emit(xi, yi)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._left_down = False
            self.drag_end.emit()

        if self._panning:
            self._panning = False
            self._pan_start_pos = None
            self._pan_start_xy = None
            self.setCursor(Qt.OpenHandCursor if self._pan_mode else Qt.ArrowCursor)

    def _map_to_image(self, x: float, y: float):
        if self._rgb is None:
            return None, None

        w = self._img_w
        h = self._img_h
        if w <= 0 or h <= 0:
            return None, None

        fit = min(self.width() / w, self.height() / h)
        scale = fit * self._zoom

        disp_w = w * scale
        disp_h = h * scale

        off_x = (self.width() - disp_w) / 2 + self._pan_x
        off_y = (self.height() - disp_h) / 2 + self._pan_y

        xi = int(round((x - off_x) / max(scale, 1e-9)))
        yi = int(round((y - off_y) / max(scale, 1e-9)))

        if 0 <= xi < w and 0 <= yi < h:
            return xi, yi
        return None, None

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if self._rgb is None:
            self._paint_empty_state(painter)
            painter.end()
            return

        h, w = self._img_h, self._img_w
        fit = min(self.width() / w, self.height() / h)
        scale = fit * self._zoom

        disp_w = int(round(w * scale))
        disp_h = int(round(h * scale))
        off_x = int(round((self.width() - disp_w) / 2 + self._pan_x))
        off_y = int(round((self.height() - disp_h) / 2 + self._pan_y))

        qimg = QImage(self._rgb.data, w, h, self._bytes_per_line, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg).scaled(disp_w, disp_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter.drawPixmap(off_x, off_y, pm)

        if self._count_overlay_lines:
            self._paint_count_overlay(painter)

        painter.end()

    def _paint_empty_state(self, painter: QPainter):
        painter.setRenderHint(QPainter.Antialiasing, True)

        center_x = self.width() / 2.0
        center_y = self.height() / 2.0

        t = self._empty_anim_phase
        cycle = t * 2.0
        if cycle < 1.0:
            fade = cycle
        else:
            fade = 2.0 - cycle
        fade = max(0.0, min(1.0, fade))
        arrow_alpha = fade
        text_alpha = 1.0 - fade
        text_y = center_y - 12.0 * arrow_alpha
        arrow_y = center_y + 12.0 * text_alpha

        if text_alpha > 0.01:
            font = QFont(self.font())
            font.setPointSize(30)
            font.setWeight(QFont.DemiBold)
            painter.setFont(font)
            painter.setPen(QColor(255, 255, 255, int(round(255 * text_alpha))))
            text_rect = QRect(
                int(round(center_x - min(340.0, self.width() * 0.42))),
                int(round(text_y - 32)),
                int(round(min(680.0, self.width() * 0.84))),
                64,
            )
            painter.drawText(text_rect, Qt.AlignCenter, "Upload images")

        if arrow_alpha > 0.01:
            arrow_color = QColor(255, 255, 255, int(round(255 * arrow_alpha)))
            pen = QPen(arrow_color, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            shaft_top = QPoint(int(round(center_x)), int(round(arrow_y - 44)))
            shaft_bottom = QPoint(int(round(center_x)), int(round(arrow_y + 30)))
            painter.drawLine(shaft_bottom, shaft_top)
            painter.drawLine(
                shaft_top,
                QPoint(int(round(center_x - 24)), int(round(arrow_y - 20))),
            )
            painter.drawLine(
                shaft_top,
                QPoint(int(round(center_x + 24)), int(round(arrow_y - 20))),
            )
            tray_pen = QPen(QColor(255, 255, 255, int(round(170 * arrow_alpha))), 7, Qt.SolidLine, Qt.RoundCap)
            painter.setPen(tray_pen)
            tray_y = int(round(arrow_y + 48))
            painter.drawLine(
                QPoint(int(round(center_x - 34)), tray_y),
                QPoint(int(round(center_x + 34)), tray_y),
            )


# ----------------------------
# Postprocessing state
# ----------------------------
@dataclass
class AnnotationLabel:
    n: int
    x: int
    y: int


@dataclass
class PostprocessState:
    paint_mask_blue: Optional[np.ndarray] = None
    paint_mask_pink: Optional[np.ndarray] = None
    paint_mask_red: Optional[np.ndarray] = None
    paint_mask_green: Optional[np.ndarray] = None
    paint_mask_yellow: Optional[np.ndarray] = None
    paint_mask_black: Optional[np.ndarray] = None
    labels: List[AnnotationLabel] = field(default_factory=list)
    next_label: int = 1

    # undo history (list of snapshots)
    history: List[object] = field(default_factory=list)
    history_limit: int = 40

    def ensure_shape(self, h: int, w: int):
        if self.paint_mask_blue is None or self.paint_mask_blue.shape != (h, w) or self.paint_mask_blue.dtype != np.uint8:
            self.paint_mask_blue = np.zeros((h, w), dtype=np.uint8)
        if self.paint_mask_pink is None or self.paint_mask_pink.shape != (h, w) or self.paint_mask_pink.dtype != np.uint8:
            self.paint_mask_pink = np.zeros((h, w), dtype=np.uint8)
        if self.paint_mask_red is None or self.paint_mask_red.shape != (h, w) or self.paint_mask_red.dtype != np.uint8:
            self.paint_mask_red = np.zeros((h, w), dtype=np.uint8)
        if self.paint_mask_green is None or self.paint_mask_green.shape != (h, w) or self.paint_mask_green.dtype != np.uint8:
            self.paint_mask_green = np.zeros((h, w), dtype=np.uint8)
        if self.paint_mask_yellow is None or self.paint_mask_yellow.shape != (h, w) or self.paint_mask_yellow.dtype != np.uint8:
            self.paint_mask_yellow = np.zeros((h, w), dtype=np.uint8)
        if self.paint_mask_black is None or self.paint_mask_black.shape != (h, w) or self.paint_mask_black.dtype != np.uint8:
            self.paint_mask_black = np.zeros((h, w), dtype=np.uint8)

    def push_undo(self):
        """Snapshot current state before an edit."""
        snap = (
            None if self.paint_mask_blue is None else self.paint_mask_blue.copy(),
            None if self.paint_mask_pink is None else self.paint_mask_pink.copy(),
            None if self.paint_mask_red is None else self.paint_mask_red.copy(),
            None if self.paint_mask_green is None else self.paint_mask_green.copy(),
            None if self.paint_mask_yellow is None else self.paint_mask_yellow.copy(),
            None if self.paint_mask_black is None else self.paint_mask_black.copy(),
            [(lab.n, lab.x, lab.y) for lab in self.labels],
            int(self.next_label),
        )
        self.history.append(snap)
        if len(self.history) > self.history_limit:
            self.history.pop(0)

    def undo(self) -> bool:
        """Restore previous snapshot. Returns True if something was undone."""
        if not self.history:
            return False
        blue, pink, red, green, yellow, black, labs, next_label = self.history.pop()
        self.paint_mask_blue = blue
        self.paint_mask_pink = pink
        self.paint_mask_red = red
        self.paint_mask_green = green
        self.paint_mask_yellow = yellow
        self.paint_mask_black = black
        self.labels = [AnnotationLabel(n, x, y) for (n, x, y) in labs]
        self.next_label = int(next_label)
        return True

class CheckBoxHeader(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cb = QCheckBox(self)
        self.cb.setTristate(False)
        self.cb.setText("")
        self.cb.setStyleSheet("QCheckBox { background: transparent; spacing: 0px; padding: 0px; margin: 0px; }")
        self.cb.setFixedSize(18, 18)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 0, 0)
        lay.addWidget(self.cb)
        lay.addStretch(1)
        self.setStyleSheet("background: transparent;")
        self.cb.stateChanged.connect(self._on_state_changed)

    def _on_state_changed(self, s):
        # PySide6 can deliver either int or Qt.CheckState; normalize first.
        try:
            state = Qt.CheckState(s)
        except Exception:
            state = Qt.CheckState.Checked if int(s) == int(Qt.CheckState.Checked) else Qt.CheckState.Unchecked
        self.toggled.emit(state == Qt.CheckState.Checked)

    def set_checked(self, on: bool):
        self.cb.blockSignals(True)
        self.cb.setChecked(on)
        self.cb.blockSignals(False)

@dataclass
class ResultRow:
    source_image: str
    tif_paths: List[str]
    expt_paths: List[str] = field(default_factory=list)
    post_mask_paths: List[str] = field(default_factory=list)
    algorithm_counts: List[int] = field(default_factory=list)
    prefer_annotated: bool = False

    _orig_cache: Optional[np.ndarray] = None
    _masked_cache: Optional[np.ndarray] = None
    _algo_mask_cache: Optional[np.ndarray] = None
    masked_green_initialized: bool = False
    user_modified_mask: bool = False
    current_count: Optional[int] = None
    cpsam_count: Optional[int] = None

    pp_original: PostprocessState = field(default_factory=PostprocessState)
    pp_masked: PostprocessState = field(default_factory=PostprocessState)


# ----------------------------
# Worker
# ----------------------------
class WorkerSignals(QObject):
    progress = Signal(int, int)
    message = Signal(str)
    row_ready = Signal(object, int, int)   # (ResultRow, done, total)
    result = Signal(object)
    error = Signal(str)

class ProcessImagesTask(QRunnable):
    def __init__(self, image_paths: List[str], out_dir: str):
        super().__init__()
        self.image_paths = image_paths
        self.out_dir = out_dir
        self.signals = WorkerSignals()

    def run(self):
        try:
            rows: List[ResultRow] = []
            total = len(self.image_paths)

            safe_mkdir(Path(self.out_dir))
            dish_mode = "auto"

            for i, img_path in enumerate(self.image_paths, start=1):
                self.signals.message.emit(f"Processing {i}/{total}: {Path(img_path).name}")

                rgb = cv_read_rgb(img_path)
                top_id, bot_id = make_output_ids_for_image(img_path)

                result_meta = count_cfu_app2(
                    rgb=rgb,
                    top_cell=top_id,
                    bot_cell=bot_id,
                    folder_dir=self.out_dir,
                    version=VERSION_FOLDER,
                    index=Path(img_path).name,
                    dish_mode=dish_mode,
                    save_which="both",
                    use_blackhat=True,
                    return_metadata=True,
                )

                tif_paths = list(result_meta.get("out_paths", [])) if isinstance(result_meta, dict) else []
                algorithm_counts = [int(v) for v in result_meta.get("counts", [])] if isinstance(result_meta, dict) else []

                expt_paths = [p for p in tif_paths if "__expt" in Path(p).stem.lower()]
                post_mask_paths = [p for p in tif_paths if "__post" in Path(p).stem.lower()]
                overlay_paths = [p for p in tif_paths if "__expt" not in Path(p).stem.lower() and "__post" not in Path(p).stem.lower()]

                rr = ResultRow(
                    source_image=img_path,
                    tif_paths=overlay_paths,
                    expt_paths=expt_paths,
                    post_mask_paths=post_mask_paths,
                    algorithm_counts=algorithm_counts,
                )
                rows.append(rr)

                self.signals.row_ready.emit(rr, i, total)
                self.signals.progress.emit(i, total)

                self.signals.progress.emit(i, total)

            self.signals.message.emit("Done.")
            self.signals.result.emit(rows)

        except Exception:
            self.signals.error.emit(traceback.format_exc())


class SSHJobSignals(QObject):
    status   = Signal(str)
    progress = Signal(int)      # 0-100
    finished = Signal(object)   # list of local output paths
    error    = Signal(str)


class _ConnCheckerSignals(QObject):
    result = Signal(str, str)   # (state, detail)


class _ConnCheckerTask(QRunnable):
    """Off-main-thread task: checks Tailscale login then SSH-pings the GPU server.

    Emits result(state, detail) where state is one of:
      'no_tailscale' | 'not_logged_in' | 'server_down' | 'connected'
    """

    def __init__(self, host: str, port: int, user: str, key_file: str,
                 signals: _ConnCheckerSignals):
        super().__init__()
        self.host     = host
        self.port     = port
        self.user     = user
        self.key_file = str(Path(key_file).expanduser())
        self.signals  = signals

    def run(self):
        # ── 1. Tailscale login state ─────────────────────────────────────────
        try:
            r = subprocess.run(
                ["tailscale", "status", "--json"],
                capture_output=True, text=True, timeout=5
            )
            try:
                data = json.loads(r.stdout)
                backend = data.get("BackendState", "")
                if backend in ("NeedsLogin", "NoState", "Stopped"):
                    self.signals.result.emit("not_logged_in", backend)
                    return
            except (json.JSONDecodeError, ValueError):
                pass  # tailscale running but output unparseable — try SSH anyway
        except FileNotFoundError:
            self.signals.result.emit("no_tailscale", "tailscale binary not found")
            return
        except Exception:
            pass  # tailscale status check failed; try SSH anyway

        # ── 2. SSH ping ──────────────────────────────────────────────────────
        if not Path(self.key_file).exists():
            self.signals.result.emit("server_down",
                                     f"SSH key not found: {self.key_file}")
            return
        try:
            r = subprocess.run(
                [
                    "ssh",
                    "-p", str(self.port),
                    "-i", self.key_file,
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=5",
                    f"{self.user}@{self.host}",
                    "echo ok",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                self.signals.result.emit("connected", self.host)
            else:
                detail = (r.stderr or r.stdout or f"exit {r.returncode}").strip()
                self.signals.result.emit("server_down", detail)
        except Exception as exc:
            self.signals.result.emit("server_down", str(exc))


class _SetupSignals(QObject):
    """Signals for the background SSH key-setup runnable."""
    log_msg  = Signal(str)
    pub_key  = Signal(str)          # emits the public key text once generated
    finished = Signal(bool, str)    # (success, error_message)


class _SetupRunnable(QRunnable):
    """Generates an SSH key (if needed) then tests the connection.

    No password / paramiko required — key installation is done manually once
    via the public key shown in the wizard log.

    IMPORTANT: `signals` must be created and owned by the caller (main thread).
    """

    def __init__(self, host: str, port: int, username: str, key_path: "Path",
                 signals: "_SetupSignals"):
        super().__init__()
        self.host     = host
        self.port     = port
        self.username = username
        self.key_path = key_path
        self.signals  = signals

    def run(self):
        try:
            # ── Step 1: generate SSH key if absent ────────────────────────
            self.signals.log_msg.emit("[ 1 / 2 ]  Checking SSH key ...")
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            pub_path = Path(str(self.key_path) + ".pub")

            if not self.key_path.exists():
                self.signals.log_msg.emit("  Generating new ed25519 key ...")
                r = subprocess.run(
                    ["ssh-keygen", "-t", "ed25519", "-f", str(self.key_path), "-N", ""],
                    capture_output=True, text=True,
                )
                if r.returncode != 0:
                    raise RuntimeError(f"ssh-keygen failed:\n{r.stderr}")
                self.signals.log_msg.emit("  ✓ New key generated")
            else:
                self.signals.log_msg.emit("  ✓ Key already exists — reusing it")

            pubkey = pub_path.read_text().strip()
            self.signals.pub_key.emit(pubkey)

            # ── Step 2: test connection with key ──────────────────────────
            self.signals.log_msg.emit(
                f"\n[ 2 / 2 ]  Testing connection to {self.username}@{self.host} ..."
            )
            r = subprocess.run(
                [
                    "ssh",
                    "-p", str(self.port),
                    "-i", str(self.key_path),
                    "-o", "BatchMode=yes",
                    "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=12",
                    f"{self.username}@{self.host}",
                    "echo cara_ok",
                ],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode == 0 and "cara_ok" in r.stdout:
                self.signals.log_msg.emit("  ✓ Connection successful!")
                self.signals.log_msg.emit(
                    "\n✓ Setup complete — the app will connect automatically."
                )
                self.signals.finished.emit(True, "")
            else:
                detail = (r.stderr or r.stdout or f"exit {r.returncode}").strip()
                raise RuntimeError(
                    f"SSH connection test failed:\n{detail}\n\n"
                    "Make sure the public key shown above has been added to\n"
                    "~/.ssh/authorized_keys on the GPU machine, and that\n"
                    "Tailscale is running on both computers."
                )
        except Exception as exc:
            self.signals.finished.emit(False, str(exc))


class GPUSetupWizard(QDialog):
    """SSH key setup + connection test — no password required."""

    def __init__(self, parent, host: str, port: int, username: str, key_path: "Path"):
        super().__init__(parent)
        self.host     = host
        self.port     = port
        self.username = username
        self.key_path = Path(key_path).expanduser()
        self.setup_ok = False

        self.setWindowTitle("GPU Server — SSH Setup")
        self.resize(600, 480)
        self.setModal(True)

        lay = QVBoxLayout(self)

        intro = QLabel(
            "<b>One-time SSH key setup</b><br><br>"
            "The app will generate a secure key (if needed) and test the connection. "
            "If the key is new, copy the public key shown in the log and add it to "
            "<code>~/.ssh/authorized_keys</code> on your GPU machine, then click "
            "<b>Verify Connection</b> again."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        lay.addWidget(intro)

        form = QGridLayout()
        form.setColumnMinimumWidth(0, 140)
        self.host_edit = QLineEdit(host)
        self.user_edit = QLineEdit(username)
        self.host_edit.returnPressed.connect(self._run_setup)
        self.user_edit.returnPressed.connect(self._run_setup)
        form.addWidget(QLabel("Host / Tailscale IP"), 0, 0)
        form.addWidget(self.host_edit, 0, 1)
        form.addWidget(QLabel("Username"), 1, 0)
        form.addWidget(self.user_edit, 1, 1)
        key_note = QLabel(f"Key file: {self.key_path}")
        key_note.setStyleSheet("color: #5f6675; font-size: 11px;")
        form.addWidget(key_note, 2, 1)
        lay.addLayout(form)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(400)
        self.log.setMinimumHeight(160)
        lay.addWidget(self.log)

        btn_row = QWidget()
        blay = QHBoxLayout(btn_row)
        blay.setContentsMargins(0, 0, 0, 0)
        self.btn_setup = QPushButton("Generate Key && Verify Connection")
        self.btn_setup.setProperty("primary", True)
        btn_cancel = QPushButton("Cancel")
        blay.addWidget(self.btn_setup)
        blay.addStretch(1)
        blay.addWidget(btn_cancel)
        lay.addWidget(btn_row)

        self.btn_setup.clicked.connect(self._run_setup)
        btn_cancel.clicked.connect(self.reject)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _append_log(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_pub_key(self, pubkey: str):
        self.log.appendPlainText(
            "\n── Public key (add this to ~/.ssh/authorized_keys on the GPU machine) ──\n"
            f"{pubkey}\n"
            "────────────────────────────────────────────────────────────────────────"
        )

    def _on_setup_finished(self, success: bool, error_msg: str):
        if success:
            _s = QSettings("CARA", "CPSAMGPUServer")
            _s.setValue("gpu_host",   self.host_edit.text().strip())
            _s.setValue("gpu_user",   self.user_edit.text().strip())
            _s.setValue("gpu_key",    str(self.key_path))
            _s.setValue("setup_done", "1")
            _s.sync()
            self.setup_ok = True
            self.btn_setup.setText("Done — Close")
            self.btn_setup.setEnabled(True)
            self.btn_setup.clicked.disconnect()
            self.btn_setup.clicked.connect(self.accept)
        else:
            self.log.appendPlainText(f"\n✗  {error_msg}")
            self.btn_setup.setEnabled(True)

    # ── kick off ──────────────────────────────────────────────────────────────

    def _run_setup(self):
        host     = self.host_edit.text().strip()
        username = self.user_edit.text().strip()
        if not host or not username:
            QMessageBox.warning(self, "Missing fields",
                                "Please enter Host and Username.")
            return

        self.btn_setup.setEnabled(False)
        self.log.clear()

        self._signals = _SetupSignals()
        runnable = _SetupRunnable(
            host=host, port=self.port, username=username,
            key_path=self.key_path, signals=self._signals,
        )
        self._signals.log_msg.connect(self._append_log)
        self._signals.pub_key.connect(self._on_pub_key)
        self._signals.finished.connect(self._on_setup_finished)
        QThreadPool.globalInstance().start(runnable)


class DirectSSHCPSAMTask(QRunnable):
    """Upload images to a GPU machine via SSH key auth, run CPSAM directly
    (no SLURM / no sbatch), download results, then delete all remote temp files.

    Auth uses an SSH private key — no password, no Duo, no ControlMaster socket.
    One-time setup in Terminal:
        ssh-keygen -t ed25519 -f ~/.ssh/gpu_server_key
        ssh-copy-id -i ~/.ssh/gpu_server_key.pub username@<tailscale-ip>
    """

    def __init__(
        self,
        image_paths: List[str],
        local_out_dir: Path,
        host: str,
        port: int,
        username: str,
        key_file: str,
        remote_root: str,      # folder containing the CPSAM script on the GPU machine
        conda_env: str,
        cpsam_script: str,     # filename of the CPSAM script, e.g. "CPSAM.py"
        cpsam_model: str,      # path to model dir (relative to remote_root or absolute)
        signals: "SSHJobSignals",
        cleanup_remote: bool = True,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.local_out_dir = local_out_dir
        self.host = host
        self.port = int(port)
        self.username = username
        self.key_file = str(Path(key_file).expanduser())
        self.remote_root = remote_root.rstrip("/")
        self.conda_env = conda_env
        self.cpsam_script = cpsam_script
        self.cpsam_model = cpsam_model
        self.cleanup_remote = bool(cleanup_remote)
        self.signals = signals   # borrowed — caller (main thread) must keep alive

    # ── SSH / SCP helpers ────────────────────────────────────────────────────

    @staticmethod
    def _run_local(cmd: List[str], timeout: Optional[int] = None) -> Tuple[int, str, str]:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""

    def _common_ssh_opts(self) -> List[str]:
        """Options shared between ssh and scp calls."""
        return [
            "-i", self.key_file,
            "-o", "BatchMode=yes",                    # fail immediately if key rejected
            "-o", "StrictHostKeyChecking=accept-new", # auto-accept on first connect
            "-o", "ConnectTimeout=15",
        ]

    def _ssh(self, remote_cmd: str, timeout: Optional[int] = None) -> Tuple[int, str, str]:
        cmd = [
            "ssh",
            "-p", str(self.port),
            *self._common_ssh_opts(),
            f"{self.username}@{self.host}",
            remote_cmd,
        ]
        return self._run_local(cmd, timeout=timeout)

    def _scp_up(self, local_path: str, remote_dir: str) -> Tuple[int, str, str]:
        cmd = [
            "scp",
            "-P", str(self.port),        # scp uses capital -P for port
            *self._common_ssh_opts(),
            local_path,
            f"{self.username}@{self.host}:{remote_dir.rstrip('/')}/",
        ]
        return self._run_local(cmd)

    def _scp_down(self, remote_file: str, local_file: str) -> Tuple[int, str, str]:
        cmd = [
            "scp",
            "-P", str(self.port),
            *self._common_ssh_opts(),
            f"{self.username}@{self.host}:{remote_file}",
            local_file,
        ]
        return self._run_local(cmd)

    # ── main run ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
            # Use /tmp so we never need write permission inside the CPSAM root.
            # /tmp is world-writable on every Linux/macOS machine.
            remote_in  = f"/tmp/cara_cpsam_{job_id}_in"
            remote_out = f"/tmp/cara_cpsam_{job_id}_out"
            self.local_out_dir.mkdir(parents=True, exist_ok=True)

            # 1 — verify SSH connection works before doing any work
            self.signals.progress.emit(2)
            self.signals.status.emit("GPU: checking SSH connection ...")
            rc, _, err = self._ssh("echo connected", timeout=20)
            if rc != 0:
                raise RuntimeError(
                    f"SSH connection to {self.username}@{self.host} failed.\n\n"
                    "Make sure:\n"
                    "  • Tailscale is running on both machines\n"
                    "  • The SSH key file path is correct\n"
                    "  • The public key was copied to the server (ssh-copy-id)\n\n"
                    f"Details: {err}"
                )

            # 2 — verify remote_root exists before wasting time uploading
            self.signals.progress.emit(6)
            self.signals.status.emit("GPU: verifying remote paths ...")
            rc, _, err = self._ssh(
                f"test -d {shlex.quote(self.remote_root)}", timeout=15
            )
            if rc != 0:
                raise RuntimeError(
                    f"CPSAM root directory not found on the GPU server:\n"
                    f"  {self.remote_root}\n\n"
                    "Fix: open the GPU dialog and set 'CPSAM root' to the\n"
                    "folder that contains your CPSAM script on the server."
                )
            rc, _, err = self._ssh(
                f"test -f {shlex.quote(self.remote_root + '/' + self.cpsam_script)}",
                timeout=15,
            )
            if rc != 0:
                raise RuntimeError(
                    f"CPSAM script not found on the GPU server:\n"
                    f"  {self.remote_root}/{self.cpsam_script}\n\n"
                    "Fix: check the 'Script filename' field in the GPU dialog."
                )

            # 3a — create remote temp dirs (hidden with _ prefix)
            self.signals.progress.emit(10)
            self.signals.status.emit("GPU: creating remote temp dirs ...")
            rc, _, err = self._ssh(
                f"mkdir -p {shlex.quote(remote_in)} {shlex.quote(remote_out)}"
            )
            if rc != 0:
                raise RuntimeError(f"Could not create remote directories:\n{err}")

            # 3 — upload images  (10 → 40 %)
            total = len(self.image_paths)
            for i, p in enumerate(self.image_paths, 1):
                src = Path(p)
                pct = 10 + int(round(i / total * 30))
                self.signals.progress.emit(pct)
                self.signals.status.emit(f"GPU: uploading {i}/{total} — {src.name}")
                rc, _, err = self._scp_up(str(src), remote_in)
                if rc != 0:
                    raise RuntimeError(f"Upload failed for {src.name}:\n{err}")

            # 4 — run CPSAM directly (SSH call blocks until the GPU job finishes)
            self.signals.progress.emit(42)
            self.signals.status.emit(
                f"GPU: running CPSAM on {self.host} — waiting for results ..."
            )
            # Non-interactive SSH shells don't source ~/.bashrc, so conda is
            # not on PATH.  Source the conda init script explicitly first.
            # ~/miniconda3 and ~/anaconda3 are the two common install locations;
            # we try both and fall through silently if one doesn't exist.
            run_cmd = (
                "source ~/.bashrc 2>/dev/null; "
                "[ -f ~/miniconda3/etc/profile.d/conda.sh ] && "
                "  source ~/miniconda3/etc/profile.d/conda.sh; "
                "[ -f ~/anaconda3/etc/profile.d/conda.sh ] && "
                "  source ~/anaconda3/etc/profile.d/conda.sh; "
                f"cd {shlex.quote(self.remote_root)} && "
                f"conda run --no-capture-output -n {shlex.quote(self.conda_env)} "
                f"python -u {shlex.quote(self.cpsam_script)} "
                f"--skip_train "
                f"--infer_dir {shlex.quote(remote_in)} "
                f"--out_dir {shlex.quote(remote_out)} "
                f"--use_gpu "
                f"--pretrained_model {shlex.quote(self.cpsam_model)}"
            )
            rc, out, err = self._ssh(run_cmd, timeout=7200)  # 2-hour cap
            if rc != 0:
                raise RuntimeError(
                    f"CPSAM exited with an error (exit code {rc}).\n\n"
                    f"stderr:\n{err}\n\nstdout:\n{out}"
                )

            # 5 — list and download result files
            self.signals.progress.emit(88)
            self.signals.status.emit("GPU: downloading results ...")
            rc, ls_out, err = self._ssh(
                f"find {shlex.quote(remote_out)} -maxdepth 1 -type f"
            )
            if rc != 0:
                raise RuntimeError(f"Could not list remote output files:\n{err}")

            downloaded: List[str] = []
            for remote_fp in [ln.strip() for ln in ls_out.splitlines() if ln.strip()]:
                name = Path(remote_fp).name
                if not name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".csv", ".txt", ".npy")
                ):
                    continue
                clean_stem = Path(name).stem.removesuffix("_overlay")
                local_fp = str(self.local_out_dir / (clean_stem + Path(name).suffix))
                self.signals.status.emit(f"GPU: downloading {name} ...")
                rc, _, err = self._scp_down(remote_fp, local_fp)
                if rc != 0:
                    raise RuntimeError(f"Download failed for {name}:\n{err}")
                downloaded.append(local_fp)

            # 6 — delete all remote temp files (nothing stored permanently on GPU machine)
            self.signals.progress.emit(97)
            if self.cleanup_remote:
                self.signals.status.emit("GPU: deleting remote temp files ...")
                self._ssh(
                    f"rm -rf {shlex.quote(remote_in)} {shlex.quote(remote_out)}"
                )

            self.signals.progress.emit(100)
            self.signals.status.emit(
                f"GPU: done — {len(downloaded)} file(s) saved to local output folder."
            )
            self.signals.finished.emit(downloaded)

        except Exception:
            self.signals.error.emit(traceback.format_exc())


# ----------------------------
# Main window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FHN ColonyNet")
        self.resize(1320, 760)

        #self._apply_global_style()
        self._apply_dark_palette()  # extra fix for macOS white-table bugs

        self.thread_pool = QThreadPool.globalInstance()
        qss_path = Path(__file__).resolve().parent / "style.qss"
        self.dark_app_qss = qss_path.read_text(encoding="utf-8") if qss_path.exists() else DEFAULT_DARK_APP_QSS
        self.selected_images: List[str] = []
        self.results: List[ResultRow] = []
        self.gt_counts_index: Dict[Tuple[str, str, int], int] = {}
        self.gt_counts_rows: List[Tuple[str, str, int, int]] = []

        self.active_tool: Optional[str] = None
        self.paint_color = "green"  # "blue", "pink", "red", "green", or "black"
        self.preview_contrast_percent = 100
        self.current_out_dir: Optional[Path] = None
        self.csv_filename = "results.csv"
        self.webui_proc: Optional[subprocess.Popen] = None
        self.web_session_id: Optional[str] = None
        self._web_tmp_files: set = set()
        self.web_saved_seen: set[str] = set()
        # CPSAM output directories whose contents should be deleted on close
        # unless the user has explicitly saved (on_save clears this list).
        self._cpsam_temp_dirs: List[str] = []
        self._last_process_inputs: set[str] = set()
        self.web_sync_timer = QTimer(self)
        self.web_sync_timer.setInterval(3000)
        self.web_sync_timer.timeout.connect(self._poll_web_session_saved)

        # ---- Left panel controls
        self.btn_pick_folder = QPushButton("Select Folder…")
        self.lbl_or = QLabel("or")
        self.lbl_or.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setItalic(True)
        self.lbl_or.setFont(f)
        self.lbl_or.setStyleSheet("color: #777; padding: 2px 0;")

        self.btn_pick_files = QPushButton("Select Images…")
        self.lbl_selected = QLabel("No images selected.")
        self.lbl_selected.setWordWrap(True)

        self.btn_pick_out = QPushButton("Select Output Directory…")
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("Choose output directory")
        self.out_dir_edit.setReadOnly(True)

        self.btn_process_old = QPushButton("ColonyNet")
        self.btn_process_old.setProperty("primary", True)
        self.btn_process_cpsam = QPushButton("CPSAM")
        self.btn_process_cpsam.setProperty("primary", True)
        self.btn_web_ui_help = QPushButton("iPad Web UI")
        # Backward-compat alias to avoid touching legacy processing code paths.
        self.btn_process = self.btn_process_old

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setObjectName("StatusLabel")
        self._load_gt_counts_csv()

        # Logo bottom-left (CENTERED more in controls ribbon)
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self._load_logo()

        # ---- Right panel (preview toggle + preview + table + postprocessing bar)
        self.preview_mode_original = QRadioButton("Original Image")
        self.preview_mode_masked = QRadioButton("Masked Image")
        self.preview_mode_original.setProperty("seg", True)
        self.preview_mode_masked.setProperty("seg", True)
        self.preview_mode_masked.setChecked(True)
        self.preview_mode_group = QButtonGroup(self)
        self.preview_mode_group.addButton(self.preview_mode_original)
        self.preview_mode_group.addButton(self.preview_mode_masked)


        self.preview_label = ZoomPanCanvas()
        self.preview_label.setMinimumHeight(420)
        self.preview_label.setObjectName("PreviewCanvas")
        self.preview_label.setStyleSheet("")  # remove inline style

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Image", "TIFF outputs"])
        self.table.setColumnWidth(0, 40)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionsClickable(True)
        self.table.horizontalHeader().resizeSection(0, 40)
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem(""))  # blank header text
        self.table.setCornerButtonEnabled(False)
        self._table_updating_checks = False

        # Header checkbox (select all)
        self._hdr = CheckBoxHeader(self.table.horizontalHeader())
        self._hdr.show()
        self._hdr.toggled.connect(self._set_all_checks)
        self.table.itemChanged.connect(self._on_table_item_changed)
        self.table.horizontalHeader().sectionResized.connect(lambda *_: self._position_header_checkbox())
        self.table.horizontalHeader().geometriesChanged.connect(self._position_header_checkbox)
        self.table.horizontalScrollBar().valueChanged.connect(lambda *_: self._position_header_checkbox())
        self._position_header_checkbox()
        self._force_table_dark()
        # Postprocessing bar
        self.post_box = QGroupBox("Postprocessing")

        self.btn_tool_paint  = KbdBadgeButton("  Paint",  "P")
        self.btn_tool_select = KbdBadgeButton("  Select", "S")
        self.btn_tool_remove = KbdBadgeButton("  Remove", "R")
        self.btn_tool_paint.setIcon(_make_svg_icon(_ICON_PAINT,  15))
        self.btn_tool_select.setIcon(_make_svg_icon(_ICON_SELECT, 15))
        self.btn_tool_remove.setIcon(_make_svg_icon(_ICON_REMOVE, 15))
        for b in (self.btn_tool_paint, self.btn_tool_select, self.btn_tool_remove):
            b.setProperty("tool", True)
            b.setCheckable(True)
            b.setIconSize(QSize(15, 15))

        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self.tool_group.addButton(self.btn_tool_paint)
        self.tool_group.addButton(self.btn_tool_select)
        self.tool_group.addButton(self.btn_tool_remove)

        # Thickness slider
        self.thickness_slider = QSlider(Qt.Horizontal)
        self.thickness_slider.setRange(1, 100)
        self.thickness_slider.setValue(12)
        self.thickness_slider.setStyleSheet(SHARED_SLIDER_QSS)
        self.thickness_slider.setFixedHeight(22)

        self.thickness_value_lbl = QLabel("12 px")
        self.thickness_slider.valueChanged.connect(self.on_thickness_changed)

        # Paint overlay color selector
        self.color_combo = ColorComboBox()
        self.color_combo.setView(QListView())
        self.color_combo.view().setFrameShape(QFrame.NoFrame)
        self.color_combo.view().viewport().setAutoFillBackground(True)
        self.color_combo.view().setSpacing(0)
        self.color_combo.addItems(["Blue", "Pink", "Red", "Green", "Black"])
        self.color_combo.setCurrentText("Green")
        self.color_combo.currentTextChanged.connect(self.on_color_changed)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(100, 300)
        self.contrast_slider.setValue(self.preview_contrast_percent)
        self.contrast_slider.setStyleSheet(SHARED_SLIDER_QSS)
        self.contrast_slider.setFixedHeight(22)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        self.contrast_value_lbl = QLabel(f"{self.preview_contrast_percent}%")

        self.count_info_lbl = QLabel("")
        self.count_info_lbl.setWordWrap(True)
        self.count_info_lbl.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.count_info_lbl.setTextFormat(Qt.RichText)

        self.btn_clear_annotations = QPushButton("Clear Annotations")
        self.btn_clear_annotations.setProperty("danger", True)

        self.btn_update_count = KbdBadgeButton("Update Count", "⌘U")
        self.btn_update_count.setProperty("success", True)

        self.btn_save = KbdBadgeButton("Save", "⌘S")
        self.btn_save.setProperty("primary", True)


                # ---- Remove focus halos (prevents the “surrounding” outline bug)
        for b in (self.btn_tool_paint, self.btn_tool_select, self.btn_tool_remove,
                self.btn_pick_folder, self.btn_pick_files, self.btn_pick_out,
                self.btn_process_old, self.btn_process_cpsam, self.btn_web_ui_help,
                self.btn_clear_annotations, self.btn_update_count, self.btn_save):
            b.setFocusPolicy(Qt.NoFocus)

        self.preview_mode_original.setFocusPolicy(Qt.NoFocus)
        self.preview_mode_masked.setFocusPolicy(Qt.NoFocus)

        # Layout
        root = QWidget()
        self.setCentralWidget(root)
        root_vlay = QVBoxLayout(root)
        root_vlay.setContentsMargins(8, 8, 8, 8)
        root_vlay.setSpacing(8)

        title_bar = self._build_title_bar()
        root_vlay.addWidget(title_bar, 0)

        main_row = QWidget()
        main_layout = QHBoxLayout(main_row)
        main_layout.setContentsMargins(0, 0, 0, 0)

        left = self._build_left_panel()
        right = self._build_right_panel_with_post()

        main_layout.addWidget(left, 0)
        main_layout.addWidget(right, 1)
        root_vlay.addWidget(main_row, 1)

        # Signals
        self.btn_pick_folder.clicked.connect(self.on_pick_folder)
        self.btn_pick_files.clicked.connect(self.on_pick_files)
        self.btn_pick_out.clicked.connect(self.on_pick_out_dir)
        self.btn_process_old.clicked.connect(self.on_process)
        self.btn_process_cpsam.clicked.connect(self.on_open_cpsam_dialog)
        self.btn_web_ui_help.clicked.connect(self.on_web_ui_help)

        self.table.itemSelectionChanged.connect(self.on_table_select)
        self.preview_mode_original.toggled.connect(self.on_table_select)
        self.preview_mode_masked.toggled.connect(self.on_table_select)
        self.preview_label.view_changed.connect(self._update_tool_cursor)
        self.preview_label.empty_clicked.connect(self.on_preview_empty_clicked)

        # Click + drag (paint/remove)
        self.preview_label.clicked.connect(self.on_preview_clicked_once)
        self.preview_label.drag.connect(self.on_preview_drag)
        self.preview_label.drag_end.connect(self.on_preview_drag_end)

        self.btn_tool_paint.toggled.connect(lambda on: self._set_tool("paint" if on else None))
        self.btn_tool_select.toggled.connect(lambda on: self._set_tool("select" if on else None))
        self.btn_tool_remove.toggled.connect(lambda on: self._set_tool("remove" if on else None))
        self._last_paint_xy: Optional[Tuple[int, int]] = None
        # Coalesce rapid drag events: render at most once per ~16ms (~60fps)
        self._paint_redraw_pending = False
        self._paint_redraw_timer = QTimer(self)
        self._paint_redraw_timer.setSingleShot(True)
        self._paint_redraw_timer.setInterval(16)
        self._paint_redraw_timer.timeout.connect(self._flush_paint_redraw)
        self.btn_clear_annotations.clicked.connect(self.on_clear_annotations)
        self.btn_update_count.clicked.connect(self.on_update_count)
        self.btn_save.clicked.connect(self.on_save)
        # Undo shortcut (Cmd+Z on mac, Ctrl+Z elsewhere)
        self.undo_sc = QShortcut(QKeySequence.Undo, self)
        self.undo_sc.activated.connect(self.on_undo)
        # Save shortcut (Cmd+S on mac, Ctrl+S elsewhere)
        self.save_sc = QShortcut(QKeySequence.Save, self)
        self.save_sc.activated.connect(self.on_save)
        # Tool shortcuts: P = Paint, S = Select, R = Remove
        self.sc_paint_tool = QShortcut(QKeySequence("P"), self)
        self.sc_paint_tool.activated.connect(lambda: self.btn_tool_paint.setChecked(True))
        self.sc_select_tool = QShortcut(QKeySequence("S"), self)
        self.sc_select_tool.activated.connect(lambda: self.btn_tool_select.setChecked(True))
        self.sc_remove_tool = QShortcut(QKeySequence("R"), self)
        self.sc_remove_tool.activated.connect(lambda: self.btn_tool_remove.setChecked(True))
        # Cmd+U → Update Count
        self.update_sc = QShortcut(QKeySequence("Ctrl+U"), self)
        self.update_sc.activated.connect(self.on_update_count)
        # Drag throttling: avoid drawing too many circles per pixel
        self._last_drag_xy: Optional[Tuple[int, int]] = None
        self._apply_theme_visuals()
        # Start GPU server connection monitor (Tailscale + SSH ping)
        self._start_server_monitor()

    def _load_gt_counts_csv(self, folder: Optional[Path] = None):
        self.gt_counts_index = {}
        self.gt_counts_rows = []
        if folder is None:
            return
        csv_files = sorted(folder.glob("*.csv"))
        if not csv_files:
            if hasattr(self, "status_lbl"):
                self.status_lbl.setText("No CSV found in images folder (GT overlay disabled).")
            return
        csv_path = csv_files[0]
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_ci = {k.lower().strip(): v for k, v in row.items()}
                    label = (row_ci.get("label") or "").strip().upper()
                    dilu_txt = (row_ci.get("dilu") or "").strip()
                    count_txt = (row_ci.get("count") or "").strip()
                    x_suffix = (row_ci.get("x_suffix") or "").strip().lower()
                    if not label or not dilu_txt or not count_txt:
                        continue
                    try:
                        dilu = int(dilu_txt)
                        count = int(float(count_txt))
                    except ValueError:
                        continue
                    self.gt_counts_index[(label, dilu, x_suffix)] = count
                    self.gt_counts_rows.append((label, dilu, x_suffix, count))
        except Exception:
            self.gt_counts_index = {}
            self.gt_counts_rows = []
            if hasattr(self, "status_lbl"):
                self.status_lbl.setText(f"Failed to read {csv_path.name} (GT overlay disabled).")

    def _normalize_file_key(self, file_name: str) -> str:
        # Normalize names like "..._output.png" to the core sample id.
        stem = Path(file_name.strip()).stem.lower()
        if stem.endswith("_output"):
            stem = stem[:-7]
        return stem

    def _lookup_gt_count(self, label: str, dilu: int, x_suffix: str = "") -> Optional[int]:
        label = label.upper()
        x_suffix = x_suffix.strip().lower()
        # 1) exact match with x_suffix
        exact = self.gt_counts_index.get((label, dilu, x_suffix))
        if exact is not None:
            return exact
        # 2) fallback: try without x_suffix
        if x_suffix:
            fallback = self.gt_counts_index.get((label, dilu, ""))
            if fallback is not None:
                return fallback
        return None

    def _parse_label_dilu_tokens(self, source_image: str) -> List[Tuple[str, int, str]]:
        """Return list of (label, dilu, x_suffix) parsed from the filename stem."""
        stem = Path(source_image).stem
        parts = stem.split("_")
        tokens: List[Tuple[str, int, str]] = []
        for i, part in enumerate(parts):
            m = GT_TOKEN_RE.match(part.strip())
            if not m:
                continue
            label = m.group(1).upper()
            dilu = int(m.group(2))
            x_suffix = parts[i + 1].strip().lower() if i + 1 < len(parts) else ""
            tokens.append((label, dilu, x_suffix))
        return tokens

    def _get_count_overlays_for_row(self, row: ResultRow, current_count: Optional[int] = None) -> List[Dict[str, Optional[str]]]:
        tokens = self._parse_label_dilu_tokens(row.source_image)
        algo_counts = list(row.algorithm_counts or [])

        n_overlays = max(len(tokens), len(algo_counts), 1 if (tokens or algo_counts) else 0)
        if len(row.tif_paths) >= 2 or len(tokens) >= 2 or len(algo_counts) >= 2:
            positions = ["left"] * max(2, n_overlays)
        else:
            positions = ["left"]

        overlays: List[Dict[str, Optional[str]]] = []
        for idx, pos in enumerate(positions):
            token = tokens[idx] if idx < len(tokens) else None
            algo_count = algo_counts[idx] if idx < len(algo_counts) else None

            gt_text: Optional[str] = None
            diff_text: Optional[str] = None
            cpsam_diff_text: Optional[str] = None
            if token is not None:
                label, dilu, x_suffix = token
                gt_count = self._lookup_gt_count(label, dilu, x_suffix)
                if gt_count is not None:
                    gt_text = f"GT {label} dilu{dilu}: {gt_count}"
                    if algo_count is not None:
                        diff_text = f"ColonyNet Diff: {algo_count - gt_count:+d}"
                    if idx == 0 and row.cpsam_count is not None:
                        cpsam_diff_text = f"CPSAM Diff: {row.cpsam_count - gt_count:+d}"

            algo_text = f"ColonyNet: {algo_count}" if algo_count is not None else None

            # CPSAM count (from CSV) and manual live count — first overlay block only
            cpsam_text: Optional[str] = None
            count_text: Optional[str] = None
            if idx == 0:
                if row.cpsam_count is not None:
                    cpsam_text = f"CPSAM: {row.cpsam_count}"
                if current_count is not None:
                    count_text = f"Current Count: {current_count}"

            if gt_text is not None or cpsam_text is not None or algo_text is not None or diff_text is not None or cpsam_diff_text is not None or count_text is not None:
                overlays.append(
                    {
                        "position": pos,
                        "gt_text": gt_text,
                        "cpsam_text": cpsam_text,
                        "algo_text": algo_text,
                        "diff_text": diff_text,
                        "cpsam_diff_text": cpsam_diff_text,
                        "count_text": count_text,
                    }
                )
        return overlays

    # ---------- Look & feel ----------
    def _apply_dark_palette(self):
        # This helps on macOS where some widgets ignore stylesheet on first paint.
        p = self.palette()
        p.setColor(QPalette.Window, QColor(7, 8, 11))          # #07080b
        p.setColor(QPalette.Base, QColor(10, 12, 17))          # #0a0c11
        p.setColor(QPalette.AlternateBase, QColor(13, 15, 20)) # #0d0f14
        p.setColor(QPalette.Text, QColor(231, 234, 240))       # #e7eaf0
        p.setColor(QPalette.WindowText, QColor(231, 234, 240)) # #e7eaf0
        p.setColor(QPalette.Button, QColor(11, 13, 18))        # #0b0d12
        p.setColor(QPalette.ButtonText, QColor(231, 234, 240)) # #e7eaf0
        p.setColor(QPalette.Highlight, QColor(25, 46, 82))     # #192e52
        p.setColor(QPalette.HighlightedText, QColor(231, 234, 240))
        self.setPalette(p)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._position_header_checkbox()

    def _cleanup_web_tmp_files(self):
        for p in list(self._web_tmp_files):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
        self._web_tmp_files.clear()

    def _cleanup_web_sessions(self):
        root = Path(__file__).resolve().parent
        sessions_dir = root / "web_jobs" / "sessions"
        if sessions_dir.exists():
            for path in sessions_dir.iterdir():
                try:
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
                except Exception:
                    pass
        pycache_dir = root / "webui" / "__pycache__"
        if pycache_dir.exists():
            try:
                shutil.rmtree(pycache_dir, ignore_errors=True)
            except Exception:
                pass

    def closeEvent(self, e):
        try:
            self.web_sync_timer.stop()
            if self.webui_proc is not None and self.webui_proc.poll() is None:
                self.webui_proc.terminate()
                try:
                    self.webui_proc.wait(timeout=1.5)
                except Exception:
                    self.webui_proc.kill()
                    try:
                        self.webui_proc.wait(timeout=1.0)
                    except Exception:
                        pass
            self._cleanup_web_tmp_files()
            self._cleanup_web_sessions()
        except Exception:
            pass
        # Delete temporary CPSAM output directories (registered when GPU run
        # finishes; cleared by on_save so files survive an explicit save).
        for d in self._cpsam_temp_dirs:
            try:
                import shutil as _shutil
                if Path(d).is_dir():
                    _shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        super().closeEvent(e)

    def _position_header_checkbox(self):
        if not hasattr(self, "_hdr"):
            return
        hdr = self.table.horizontalHeader()
        x = hdr.sectionViewportPosition(0)
        w = hdr.sectionSize(0)
        h = hdr.height()
        self._hdr.setGeometry(x, 0, w, h)
        self._update_tool_cursor()

    def _set_all_checks(self, on: bool):
        self._table_updating_checks = True
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is None:
                continue
            item.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._table_updating_checks = False
        self._sync_header_checkbox()

    def _on_table_item_changed(self, item: QTableWidgetItem):
        if self._table_updating_checks:
            return
        if item.column() == 0:
            self._sync_header_checkbox()

    def _sync_header_checkbox(self):
        total = self.table.rowCount()
        if total <= 0:
            self._hdr.set_checked(False)
            return
        checked = 0
        for r in range(total):
            it = self.table.item(r, 0)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                checked += 1
        self._hdr.set_checked(checked == total)

    def _load_logo(self):
        try:
            base_dir = Path(__file__).resolve().parent
            candidates = [
                base_dir / "Logo2.png",
                base_dir / "logo.png",
                base_dir / "webui" / "static" / "logo.png",
                base_dir / "Logo.jpg",
            ]
            for logo_path in candidates:
                if not logo_path.exists():
                    continue
                pm = self._load_logo_transparent(logo_path)
                if pm is not None and not pm.isNull():
                    pm = pm.scaledToWidth(260, Qt.SmoothTransformation)
                    self.logo_label.setPixmap(pm)
                    self.logo_label.setToolTip(logo_path.name)
                    return
        except Exception:
            pass

    def _load_logo_transparent(self, logo_path: Path) -> "Optional[QPixmap]":
        """Load a logo and make its dark background transparent (simulates mix-blend-mode: screen)."""
        try:
            img = cv2.imread(str(logo_path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            b_ch = img[:, :, 0].astype(np.float32)
            g_ch = img[:, :, 1].astype(np.float32)
            r_ch = img[:, :, 2].astype(np.float32)
            # Weighted luminance — dark pixels become transparent
            lum = 0.299 * r_ch + 0.587 * g_ch + 0.114 * b_ch
            alpha = np.clip(lum / 50.0, 0.0, 1.0)
            rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
            rgba[:, :, 0] = img[:, :, 2]  # R
            rgba[:, :, 1] = img[:, :, 1]  # G
            rgba[:, :, 2] = img[:, :, 0]  # B
            rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
            h, w = rgba.shape[:2]
            qimg = QImage(rgba.data, w, h, int(rgba.strides[0]), QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg.copy())
        except Exception:
            return None
    def _stamp_line(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, radius: int, value: int):
        # Step small enough so strokes are continuous
        dx = x1 - x0
        dy = y1 - y0
        dist = int(max(1, round((dx*dx + dy*dy) ** 0.5)))

        # spacing: half radius is usually smooth; clamp to [1..radius]
        step = max(1, radius // 2)
        n = max(1, dist // step)

        for i in range(n + 1):
            t = i / max(n, 1)
            xi = int(round(x0 + t * dx))
            yi = int(round(y0 + t * dy))
            cv2.circle(mask, (xi, yi), radius, value, -1)

    def on_undo(self):
        row = self._current_row()
        if row is None:
            return

        want_masked = self.preview_mode_masked.isChecked()
        pp = row.pp_masked if want_masked else row.pp_original

        if pp.undo():
            self.on_table_select()
            self.status_lbl.setText("Undone.")
        else:
            self.status_lbl.setText("Nothing to undo.")

    # ---------- Layout ----------
    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("TitleBar")
        bar.setFixedHeight(44)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(10)

        # Connection status dot
        self.conn_dot_lbl = QLabel("●")
        self.conn_dot_lbl.setStyleSheet("color: #9aa1ae; font-size: 11px;")
        self.conn_dot_lbl.setFixedWidth(16)

        # Connection status text
        self.conn_status_lbl = QLabel("Checking server connection…")
        self.conn_status_lbl.setObjectName("TitleBarAppName")

        lay.addWidget(self.conn_dot_lbl)
        lay.addWidget(self.conn_status_lbl)

        lay.addStretch(1)

        # Settings gear
        btn_settings = QPushButton("⚙  Server Settings")
        btn_settings.setStyleSheet(_TITLEBAR_BTN_QSS)
        btn_settings.clicked.connect(self._open_settings_dialog)
        lay.addWidget(btn_settings)

        return bar

    def _update_title_bar_folder(self, folder_path: Optional[str]):
        if not hasattr(self, "title_bar_folder_lbl"):
            return
        if folder_path:
            parts = Path(folder_path).parts
            # Show last 2 path components for brevity
            crumb = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            self.title_bar_folder_lbl.setText(crumb)
        else:
            self.title_bar_folder_lbl.setText("No folder loaded")

    def _build_left_panel(self) -> QWidget:
        box = QGroupBox("Controls")
        box.setObjectName("ControlsPanel")
        box.setMinimumWidth(320)
        box.setMaximumWidth(360)
        layout = QVBoxLayout(box)

        layout.addWidget(self.btn_pick_folder)
        layout.addWidget(self.lbl_or)
        layout.addWidget(self.btn_pick_files)
        layout.addWidget(self.lbl_selected)

        layout.addSpacing(10)

        layout.addWidget(self.btn_pick_out)
        layout.addWidget(self.out_dir_edit)

        layout.addSpacing(12)
        proc_row = QWidget()
        proc_lay = QHBoxLayout(proc_row)
        proc_lay.setContentsMargins(0, 0, 0, 0)
        proc_lay.addWidget(self.btn_process_old)
        proc_lay.addWidget(self.btn_process_cpsam)
        proc_lay.addWidget(self.btn_web_ui_help)
        layout.addWidget(proc_row)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_lbl)

        layout.addStretch(1)

        # 1) center logo more in controls ribbon
        self.logo_label.setMinimumWidth(260)
        self.logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.logo_label, 0, Qt.AlignHCenter | Qt.AlignBottom)

        return box

    def _build_ssh_panel(self) -> QWidget:
        box = QGroupBox("GPU (SSH)")
        lay = QGridLayout(box)
        lay.addWidget(QLabel("Host"), 0, 0)
        lay.addWidget(QLabel("User"), 1, 0)
        lay.addWidget(QLabel("Password"), 2, 0)
        return box

    def _build_post_panel(self) -> QWidget:
        self.post_box = QGroupBox("Postprocessing")
        self.post_box.setObjectName("PostPanel")
        layout = QVBoxLayout(self.post_box)

        layout.addWidget(self.btn_tool_paint)
        layout.addWidget(self.btn_tool_select)
        layout.addWidget(self.btn_tool_remove)

        layout.addSpacing(10)

        thick_row = QWidget()
        thick_row.setStyleSheet("background: transparent;")
        thick_layout = QHBoxLayout(thick_row)
        thick_layout.setContentsMargins(0, 0, 0, 0)
        thick_layout.addWidget(QLabel("Thickness:"))
        thick_layout.addWidget(self.thickness_slider, 1)
        thick_layout.addWidget(self.thickness_value_lbl, 0)
        layout.addWidget(thick_row)

        color_row = QWidget()
        color_row.setStyleSheet("background: transparent;")
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(6)
        lbl_color = QLabel("Color")
        lbl_color.setStyleSheet("color: #5f6675; font-size: 11px;")
        color_layout.addWidget(lbl_color)
        color_layout.addWidget(self.color_combo, 1)
        # Color swatch — circle showing currently selected color
        self.color_swatch_lbl = QLabel()
        self.color_swatch_lbl.setObjectName("ColorSwatch")
        self.color_swatch_lbl.setFixedSize(14, 14)
        self._update_color_swatch()
        color_layout.addWidget(self.color_swatch_lbl)
        layout.addWidget(color_row)

        contrast_row = QWidget()
        contrast_row.setStyleSheet("background: transparent;")
        contrast_layout = QHBoxLayout(contrast_row)
        contrast_layout.setContentsMargins(0, 0, 0, 0)
        contrast_layout.addWidget(QLabel("Contrast:"))
        contrast_layout.addWidget(self.contrast_slider, 1)
        contrast_layout.addWidget(self.contrast_value_lbl, 0)
        layout.addWidget(contrast_row)

        layout.addSpacing(10)
        # Count info — wrapped in a styled panel
        self.count_info_lbl.setObjectName("CountInfoPanel")
        layout.addWidget(self.count_info_lbl)
        layout.addSpacing(6)
        layout.addWidget(self.btn_clear_annotations)
        layout.addWidget(self.btn_update_count)
        layout.addWidget(self.btn_save)

        layout.addStretch(1)
        return self.post_box

    def _force_table_dark(self):
        pal = self.table.palette()
        pal.setColor(QPalette.Base, QColor(13, 15, 20))          # #0d0f14
        pal.setColor(QPalette.Window, QColor(13, 15, 20))        # #0d0f14
        pal.setColor(QPalette.AlternateBase, QColor(10, 12, 17)) # #0a0c11
        pal.setColor(QPalette.Text, QColor(231, 234, 240))       # #e7eaf0
        pal.setColor(QPalette.WindowText, QColor(231, 234, 240)) # #e7eaf0
        pal.setColor(QPalette.Button, QColor(13, 15, 20))        # #0d0f14
        pal.setColor(QPalette.ButtonText, QColor(231, 234, 240)) # #e7eaf0
        pal.setColor(QPalette.Highlight, QColor(25, 46, 82))     # #192e52
        pal.setColor(QPalette.HighlightedText, QColor(231, 234, 240))
        self.table.setPalette(pal)
        self.table.viewport().setPalette(pal)
        self.table.setStyleSheet("")  # let global style.qss handle table

    def _refresh_widget_style(self, widget: QWidget):
        if widget is None:
            return
        style = widget.style()
        if style is None:
            return
        style.unpolish(widget)
        style.polish(widget)
        widget.update()

    def _apply_combo_popup_theme(self):
        view = self.color_combo.view() if hasattr(self, "color_combo") else None
        if view is None:
            return
        pal = view.palette()
        BG_D = "#0d0f14"
        pal.setColor(QPalette.Base,            QColor(BG_D))
        pal.setColor(QPalette.Window,          QColor(BG_D))
        pal.setColor(QPalette.Text,            QColor("#e7eaf0"))
        pal.setColor(QPalette.Highlight,       QColor(138, 180, 255, 40))
        pal.setColor(QPalette.HighlightedText, QColor("#e7eaf0"))
        view.setStyleSheet("")
        view.setPalette(pal)
        self._refresh_widget_style(view)

    def _apply_theme_visuals(self):
        app = QApplication.instance()
        self._apply_dark_palette()
        if app is not None:
            app.setStyleSheet(self.dark_app_qss)
        self.btn_web_ui_help.setProperty("primary", False)
        self.lbl_or.setStyleSheet("color: #777; padding: 2px 0;")
        self._refresh_widget_style(self.btn_web_ui_help)
        self._load_logo()
        self._force_table_dark()
        self._apply_combo_popup_theme()
        row = self._current_row()
        if row is not None and hasattr(self, "count_info_lbl"):
            self._update_count_info_lbl(row)

    # ── Server connection monitor ────────────────────────────────────────────

    def _start_server_monitor(self):
        """Start background connection polling. Call once after UI is ready."""
        self._conn_state          = "checking"
        self._conn_checker_running = False
        self._tailscale_login_shown = False
        self._conn_timer = QTimer(self)
        self._conn_timer.timeout.connect(self._on_conn_check_tick)
        self._conn_timer.start(6000)               # re-check every 6 s
        QTimer.singleShot(800, self._on_conn_check_tick)  # first check after 0.8 s

    def _on_conn_check_tick(self):
        if getattr(self, "_conn_checker_running", False):
            return  # previous check still running
        self._conn_checker_running = True
        _s   = QSettings("CARA", "CPSAMGPUServer")
        host = _s.value("gpu_host", GPU_HOST_DEFAULT)
        port = int(_s.value("gpu_port", str(GPU_PORT_DEFAULT)))
        user = _s.value("gpu_user", GPU_USER_DEFAULT)
        key  = _s.value("gpu_key",  GPU_KEY_DEFAULT)
        self._conn_checker_signals = _ConnCheckerSignals()
        self._conn_checker_signals.result.connect(self._on_conn_result)
        task = _ConnCheckerTask(host, port, user, key, self._conn_checker_signals)
        self.thread_pool.start(task)

    def _on_conn_result(self, state: str, detail: str):
        self._conn_checker_running = False
        self._update_conn_ui(state, detail)

    _CONN_DOT_COLOR = {
        "connected":     "#4ade80",   # green
        "not_logged_in": "#ffd66b",   # amber
        "server_down":   "#f06060",   # red
        "no_tailscale":  "#f06060",   # red
        "checking":      "#9aa1ae",   # gray
    }
    _CONN_LABEL = {
        "connected":     "Connected to GPU server",
        "not_logged_in": "Tailscale: sign in required",
        "server_down":   "No server connection",
        "no_tailscale":  "Tailscale not installed",
        "checking":      "Checking connection…",
    }

    def _update_conn_ui(self, state: str, detail: str):
        """Update title-bar dot + label. Must be called on the main thread."""
        prev = getattr(self, "_conn_state", None)
        self._conn_state = state

        color = self._CONN_DOT_COLOR.get(state, "#9aa1ae")
        label = self._CONN_LABEL.get(state, detail)

        if hasattr(self, "conn_dot_lbl"):
            self.conn_dot_lbl.setStyleSheet(
                f"color: {color}; font-size: 11px;")
        if hasattr(self, "conn_status_lbl"):
            self.conn_status_lbl.setText(label)
        # Pulse when newly connected
        if state == "connected" and prev != "connected":
            self._pulse_connected_dot()

        # Show Tailscale login dialog once on first detection
        if state == "not_logged_in" and not getattr(
                self, "_tailscale_login_shown", False):
            self._tailscale_login_shown = True
            QTimer.singleShot(0, self._show_tailscale_login_dialog)

    def _pulse_connected_dot(self):
        """Brief white→green pulse to celebrate a new connection."""
        pulses = ["#ffffff", "#4ade80", "#ffffff", "#4ade80", "#ffffff", "#4ade80"]
        for i, col in enumerate(pulses):
            QTimer.singleShot(i * 140, lambda c=col: (
                self.conn_dot_lbl.setStyleSheet(
                    f"color: {c}; font-size: 11px;")
                if hasattr(self, "conn_dot_lbl") else None
            ))

    # ── Tailscale login dialog ───────────────────────────────────────────────

    def _show_tailscale_login_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Tailscale Login Required")
        dlg.resize(440, 210)
        dlg.setModal(True)
        lay = QVBoxLayout(dlg)

        info = QLabel(
            "<b>Tailscale is not logged in.</b><br><br>"
            "Tailscale is needed to reach the GPU server. "
            "Click <b>Login</b> to open the sign-in page in your browser — "
            "the app will detect when you have logged in automatically."
        )
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        lay.addWidget(info)
        lay.addStretch(1)

        btn_row = QWidget()
        blay = QHBoxLayout(btn_row)
        blay.setContentsMargins(0, 0, 0, 0)
        btn_login = QPushButton("Open Tailscale Login")
        btn_login.setProperty("primary", True)
        btn_skip = QPushButton("Use without GPU")
        blay.addWidget(btn_login)
        blay.addStretch(1)
        blay.addWidget(btn_skip)
        lay.addWidget(btn_row)

        def _do_login():
            try:
                subprocess.Popen(["tailscale", "login"])
            except Exception as exc:
                QMessageBox.warning(dlg, "Error",
                                    f"Could not run 'tailscale login':\n{exc}")
            dlg.accept()

        btn_login.clicked.connect(_do_login)
        btn_skip.clicked.connect(dlg.accept)
        dlg.exec()

    # ── Server settings dialog ───────────────────────────────────────────────

    def _open_settings_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Server Settings")
        dlg.resize(520, 380)
        lay = QVBoxLayout(dlg)

        _s = QSettings("CARA", "CPSAMGPUServer")

        fields_def = [
            ("Host / Tailscale IP", "gpu_host",   GPU_HOST_DEFAULT,
             "Tailscale IP or hostname of the GPU machine"),
            ("SSH Port",            "gpu_port",    str(GPU_PORT_DEFAULT),
             "SSH port — usually 22"),
            ("Username",            "gpu_user",    GPU_USER_DEFAULT,
             "Your account name on the GPU machine"),
            ("SSH Key File",        "gpu_key",     GPU_KEY_DEFAULT,
             "Local path to SSH private key (e.g. ~/.ssh/gpu_server_key)"),
            ("CPSAM Root",          "gpu_root",    GPU_CPSAM_ROOT,
             "Folder containing CPSAM.py on the server"),
            ("Conda Env",           "gpu_env",     GPU_CONDA_ENV,
             "Conda environment that has cellpose installed"),
            ("Script Filename",     "gpu_script",  GPU_CPSAM_SCRIPT,
             "Name of the CPSAM script file, e.g. CPSAM.py"),
            ("Model Path",          "gpu_model",   GPU_CPSAM_MODEL,
             "Absolute path to the finetuned model on the server"),
        ]

        form = QGridLayout()
        form.setColumnMinimumWidth(0, 140)
        edits: Dict[str, QLineEdit] = {}
        for row_i, (label, key, default, tip) in enumerate(fields_def):
            lbl = QLabel(label)
            edit = QLineEdit(_s.value(key, default))
            edit.setToolTip(tip)
            form.addWidget(lbl,  row_i, 0)
            form.addWidget(edit, row_i, 1)
            edits[key] = edit
        lay.addLayout(form)
        lay.addStretch(1)

        note = QLabel(
            "Changes take effect immediately — the connection check will re-run."
        )
        note.setStyleSheet("color: #5f6675; font-size: 11px;")
        lay.addWidget(note)

        btn_row = QWidget()
        blay = QHBoxLayout(btn_row)
        blay.setContentsMargins(0, 4, 0, 0)
        btn_rekey  = QPushButton("Re-run SSH Setup…")
        btn_rekey.setToolTip(
            "Run the one-time setup wizard again (new machine or lost key)")
        btn_cancel = QPushButton("Cancel")
        btn_apply  = QPushButton("Apply && Close")
        btn_apply.setProperty("primary", True)
        blay.addWidget(btn_rekey)
        blay.addStretch(1)
        blay.addWidget(btn_cancel)
        blay.addWidget(btn_apply)
        lay.addWidget(btn_row)

        def _apply():
            for key, edit in edits.items():
                _s.setValue(key, edit.text().strip())
            _s.setValue("setup_done", "1")
            _s.sync()
            # Trigger an immediate recheck with the new settings
            self._conn_checker_running = False
            QTimer.singleShot(200, self._on_conn_check_tick)
            dlg.accept()

        def _rekey():
            _s.setValue("setup_done", "")
            _s.sync()
            dlg.accept()
            QTimer.singleShot(0, self._run_ssh_setup_wizard)

        btn_apply.clicked.connect(_apply)
        btn_cancel.clicked.connect(dlg.reject)
        btn_rekey.clicked.connect(_rekey)
        dlg.exec()

    def _run_ssh_setup_wizard(self):
        _s   = QSettings("CARA", "CPSAMGPUServer")
        host = _s.value("gpu_host", GPU_HOST_DEFAULT)
        port = int(_s.value("gpu_port", str(GPU_PORT_DEFAULT)))
        user = _s.value("gpu_user", GPU_USER_DEFAULT)
        key  = Path(_s.value("gpu_key", GPU_KEY_DEFAULT)).expanduser()
        wizard = GPUSetupWizard(self, host=host, port=port,
                                username=user, key_path=key)
        if wizard.exec() == QDialog.Accepted and wizard.setup_ok:
            self._conn_checker_running = False
            QTimer.singleShot(500, self._on_conn_check_tick)

    # ── CPSAM result callbacks (used by simplified on_open_cpsam_dialog) ─────

    def _on_cpsam_done(self, outputs):
        self.btn_process_cpsam.setEnabled(True)
        # progress bar: hold at 100 briefly, then reset
        self.progress.setValue(100)
        QTimer.singleShot(1200, lambda: self.progress.setValue(0))
        outs = list(outputs)
        local_out_dir = getattr(self, "_cpsam_local_out_dir", Path.cwd())
        image_outs = [o for o in outs if Path(o).suffix.lower() != ".csv"]
        self.status_lbl.setText(
            f"CPSAM done — {len(image_outs)} image(s) saved to {local_out_dir}")
        out_dir_str = str(local_out_dir)
        if out_dir_str not in self._cpsam_temp_dirs:
            self._cpsam_temp_dirs.append(out_dir_str)
        if outs:
            loaded = self._attach_cpsam_outputs_to_originals(outs)
            msg = f"CPSAM GPU run finished.\n{len(image_outs)} image(s) processed."
            if loaded:
                msg += f"\n{loaded} image(s) matched and shown in the table."
            QMessageBox.information(self, "GPU CPSAM finished", msg)
        else:
            QMessageBox.information(
                self, "GPU CPSAM finished",
                "Run completed but no output files were downloaded.\n"
                "Check the CPSAM script output for errors."
            )

    def _on_cpsam_error(self, tb: str):
        self.btn_process_cpsam.setEnabled(True)
        self.progress.setValue(0)
        self.status_lbl.setText("GPU CPSAM error — see details.")
        QMessageBox.critical(self, "GPU CPSAM error", tb)

    def _build_right_panel_with_post(self) -> QWidget:
        right_root = QWidget()
        root_layout = QHBoxLayout(right_root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(12)

        main = QWidget()
        main_layout = QVBoxLayout(main)

        mode_row = QWidget()
        mode_layout = QHBoxLayout(mode_row)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(10)
        mode_layout.addWidget(self.preview_mode_original)
        mode_layout.addWidget(self.preview_mode_masked)
        mode_layout.addStretch(1)

        main_layout.addWidget(mode_row, 0)
        main_layout.addWidget(self.preview_label, 3)
        main_layout.addWidget(self.table, 2)

        post = self._build_post_panel()
        post.setFixedWidth(220)

        root_layout.addWidget(main, 1)
        root_layout.addWidget(post, 0)

        return right_root

    # ---------- Core logic ----------
    def on_pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if not folder:
            return
        self._update_title_bar_folder(folder)
        paths = []
        for p in sorted(Path(folder).glob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                paths.append(str(p))
        self.set_selected_images(paths)

    def on_pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select images", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg)"
        )
        if not files:
            return
        self.set_selected_images(files)

    def on_preview_empty_clicked(self):
        dialog = QFileDialog(self, "Select images or a folder")
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilter("Images (*.tif *.tiff *.png *.jpg *.jpeg)")

        list_view = dialog.findChild(QListView, "listView")
        if list_view is not None:
            list_view.setSelectionMode(QListView.ExtendedSelection)

        tree_view = dialog.findChild(QListView, "treeView")
        if tree_view is not None:
            tree_view.setSelectionMode(QListView.ExtendedSelection)

        if not dialog.exec():
            return

        selected = dialog.selectedFiles()
        if not selected:
            return

        if len(selected) == 1 and Path(selected[0]).is_dir():
            folder = selected[0]
            paths = []
            for p in sorted(Path(folder).glob("*")):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    paths.append(str(p))
            self.set_selected_images(paths)
            return

        files = [p for p in selected if Path(p).is_file() and Path(p).suffix.lower() in SUPPORTED_EXTS]
        if files:
            self.set_selected_images(files)

    def on_pick_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output directory")
        if not folder:
            return
        self.out_dir_edit.setText(folder)
        self.current_out_dir = Path(folder)

    def set_selected_images(self, paths: List[str]):
        self.selected_images = paths
        if not paths:
            self.lbl_selected.setText("No images selected.")
            self._load_gt_counts_csv(None)
            self._update_title_bar_folder(None)
        else:
            self.lbl_selected.setText(f"{len(paths)} image(s) selected.\nFirst: {Path(paths[0]).name}")
            self._load_gt_counts_csv(Path(paths[0]).parent)
            self._update_title_bar_folder(str(Path(paths[0]).parent))

        # NEW: create placeholder ResultRow objects immediately
        self.results = [
            ResultRow(source_image=p, tif_paths=[])
            for p in paths
        ]

        # populate table immediately (TIFF outputs empty until processed)
        self.populate_table(self.results)
        self._force_table_dark()
        self.table.clearSelection()

        # show first image immediately
        try:
            rgb = cv_read_rgb(paths[0])
            self.preview_label.set_rgb(rgb)
            self.preview_label.reset_view()
        except Exception:
            self.preview_label.set_rgb(np.zeros((1, 1, 3), dtype=np.uint8))

    def on_process(self):
        paths_to_process = self._get_checked_or_all_source_paths()
        if not paths_to_process:
            QMessageBox.warning(self, "No input", "Select images or a folder first.")
            return
        self._last_process_inputs = set(paths_to_process)
        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output directory", "Select an output directory first.")
            return

        self.current_out_dir = Path(out_dir)

        self.btn_process.setEnabled(False)
        self.progress.setValue(0)
        self.status_lbl.setText("Starting…")

        # Keep a strong reference on self so that task.signals (a QObject) is
        # never destroyed on the background thread while the main thread still
        # has queued signal deliveries pending — that race causes SIGSEGV via
        # Shiboken's QRunnableWrapper destructor.
        self._active_task = ProcessImagesTask(
            image_paths=paths_to_process,
            out_dir=out_dir,
        )
        self._active_task.signals.progress.connect(self.on_progress)
        self._active_task.signals.message.connect(self.on_message)
        self._active_task.signals.result.connect(self.on_results)
        self._active_task.signals.error.connect(self.on_error)
        self._active_task.signals.row_ready.connect(self.on_row_ready)
        self.thread_pool.start(self._active_task)

    def on_progress(self, done: int, total: int):
        pct = int(round((done / max(total, 1)) * 100))
        self.progress.setValue(pct)

    def on_message(self, msg: str):
        self.status_lbl.setText(msg)

    def on_row_ready(self, rr: ResultRow, done: int, total: int):
        # Update existing placeholder entry (by source path)
        for idx, existing in enumerate(self.results):
            if existing.source_image == rr.source_image:
                self.results[idx] = rr
                self._update_table_row(idx, rr)
                break

        self.status_lbl.setText(f"Processed {done}/{total}: {Path(rr.source_image).name}")
        self._force_table_dark()
        current = self._current_row()
        if current is not None and current.source_image == rr.source_image:
            self.on_table_select()

    def _update_table_row(self, row_i: int, r: ResultRow):
        if row_i < 0 or row_i >= self.table.rowCount():
            return

        img_name = Path(r.source_image).name
        tifs = self._format_output_cell(r.tif_paths)

        # keep existing checkbox state if present, else default checked
        existing_chk = self.table.item(row_i, 0)
        if existing_chk is None:
            chk = QTableWidgetItem("")
            chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(row_i, 0, chk)

        self.table.setItem(row_i, 1, QTableWidgetItem(img_name))
        self.table.setItem(row_i, 2, QTableWidgetItem(tifs))

        self.table.viewport().update()

    def on_error(self, tb: str):
        self.btn_process.setEnabled(True)
        self.status_lbl.setText("Error.")
        QMessageBox.critical(self, "Processing error", tb)

    def on_results(self, rows: List[ResultRow]):
        self.btn_process.setEnabled(True)
        if not self.results:
            self.results = rows
        else:
            by_src = {r.source_image: r for r in rows}
            for i, existing in enumerate(self.results):
                upd = by_src.get(existing.source_image)
                if upd is not None:
                    self.results[i] = upd
        self.populate_table(self.results)
        self.status_lbl.setText(f"Processed {len(rows)} image(s).")
        self.table.clearSelection()

    _COLOR_SWATCH_MAP = {
        "blue":  "#4d9ef6",
        "pink":  "#f06db0",
        "red":   "#f06060",
        "green": "#4ade80",
        "black": "#555555",
    }

    def _update_color_swatch(self):
        if not hasattr(self, "color_swatch_lbl"):
            return
        c = self._COLOR_SWATCH_MAP.get(self.paint_color, "#4ade80")
        self.color_swatch_lbl.setStyleSheet(
            f"background: {c}; border-radius: 7px; border: 1px solid rgba(255,255,255,30);"
        )

    def on_color_changed(self, txt: str):
        c = txt.strip().lower()
        if c == "pink":
            self.paint_color = "pink"
        elif c == "red":
            self.paint_color = "red"
        elif c == "green":
            self.paint_color = "green"
        elif c == "black":
            self.paint_color = "black"
        else:
            self.paint_color = "blue"
        self._update_color_swatch()
        self.on_table_select()  # redraw preview immediately

    def populate_table(self, rows: List[ResultRow]):
        self._table_updating_checks = True
        self.table.setRowCount(0)
        for r in rows:
            row_i = self.table.rowCount()
            self.table.insertRow(row_i)

            img_name = Path(r.source_image).name
            tifs = self._format_output_cell(r.tif_paths)
            # checkbox item
            chk = QTableWidgetItem("")
            chk.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            chk.setCheckState(Qt.CheckState.Unchecked)  # default: unchecked
            self.table.setItem(row_i, 0, chk)

            self.table.setItem(row_i, 1, QTableWidgetItem(img_name))
            self.table.setItem(row_i, 2, QTableWidgetItem(tifs))
        # strong repaint to avoid white flash
        self.table.viewport().update()
        self.table.repaint()
        self._table_updating_checks = False
        self._sync_header_checkbox()

    def _friendly_output_name(self, path: str) -> str:
        p = Path(path)
        stem = p.stem
        low = stem.lower()
        if "_annotated" in low:
            base = stem.split("_annotated", 1)[0]
            return f"{base}_annotated{p.suffix}"
        if low.endswith("_top") or low.endswith("_bottom"):
            base = stem.rsplit("_", 1)[0]
            return f"{base}_masked{p.suffix}"
        return p.name

    def _sort_output_paths(self, paths: List[str]) -> List[str]:
        out = list(paths)
        out.sort(
            key=lambda p: (
                0 if "_annotated" in Path(p).stem.lower() else 1,
                0 if "_masked" in Path(p).stem.lower() else 1,
                Path(p).name.lower(),
            )
        )
        return out

    def _format_output_cell(self, paths: List[str]) -> str:
        if not paths:
            return ""
        srt = self._sort_output_paths(paths)
        primary = self._friendly_output_name(srt[0])
        extra = len(srt) - 1
        if extra > 0:
            return f"{primary} (+{extra})"
        return primary

    def _current_row(self) -> Optional[ResultRow]:
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            return None
        i = idxs[0].row()
        if i < 0 or i >= len(self.results):
            return None
        return self.results[i]

    def _get_checked_indices(self) -> List[int]:
        out: List[int] = []
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 0)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                out.append(i)
        return out

    def _get_checked_or_all_source_paths(self) -> List[str]:
        if self.results:
            checked = [i for i in self._get_checked_indices() if 0 <= i < len(self.results)]
            if checked:
                return [self.results[i].source_image for i in checked]
            return [r.source_image for r in self.results]
        if self.selected_images:
            checked = [i for i in self._get_checked_indices() if 0 <= i < len(self.selected_images)]
            if checked:
                return [self.selected_images[i] for i in checked]
            return list(self.selected_images)
        return []

    def _get_save_targets(self) -> List[ResultRow]:
        """
        Priority:
        1) checked rows in checkbox column
        2) if none checked, use currently selected rows
        3) if none selected, use all rows
        """
        checked_rows = []
        for i in range(self.table.rowCount()):
            it = self.table.item(i, 0)
            if it is not None and it.checkState() == Qt.CheckState.Checked:
                if 0 <= i < len(self.results):
                    checked_rows.append(self.results[i])

        if checked_rows:
            return checked_rows

        # fallback: selected rows
        idxs = self.table.selectionModel().selectedRows()
        out = []
        for idx in idxs:
            r = idx.row()
            if 0 <= r < len(self.results):
                out.append(self.results[r])
        if out:
            return out
        return list(self.results)

    def _get_base_images(self, row: ResultRow) -> Tuple[np.ndarray, np.ndarray]:
        if row._orig_cache is None:
            row._orig_cache = cv_read_rgb(row.source_image)

        if row._masked_cache is None:
            if row.tif_paths:
                annotated = [p for p in row.tif_paths if "_annotated" in Path(p).stem.lower()]
                base = [p for p in row.expt_paths if "_annotated" not in Path(p).stem.lower()]
                if row.prefer_annotated and annotated:
                    newest = max(annotated, key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0.0)
                    row._masked_cache = cv_read_rgb_anydepth(newest)
                elif len(base) >= 2:
                    a = cv_read_rgb_anydepth(base[0])
                    b = cv_read_rgb_anydepth(base[1])
                    row._masked_cache = stitch_side_by_side(a, b, gap=12)
                elif len(base) >= 1:
                    row._masked_cache = cv_read_rgb_anydepth(base[0])
                elif annotated:
                    newest = max(annotated, key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0.0)
                    row._masked_cache = cv_read_rgb_anydepth(newest)
                elif row.tif_paths:
                    # Fallback for older rows that only have overlay outputs.
                    base = [p for p in row.tif_paths if "_annotated" not in Path(p).stem.lower()]
                    if len(base) >= 2:
                        a = cv_read_rgb_anydepth(base[0])
                        b = cv_read_rgb_anydepth(base[1])
                        row._masked_cache = stitch_side_by_side(a, b, gap=12)
                    elif len(base) >= 1:
                        row._masked_cache = cv_read_rgb_anydepth(base[0])
            else:
                row._masked_cache = row._orig_cache.copy()

        if row._masked_cache is None:
            row._masked_cache = row._orig_cache.copy()

        return row._orig_cache, row._masked_cache

    def _get_algorithm_mask(self, row: ResultRow) -> Optional[np.ndarray]:
        if row._algo_mask_cache is not None:
            return row._algo_mask_cache

        if not row.post_mask_paths:
            return None

        try:
            if len(row.post_mask_paths) >= 2:
                a = cv_read_mask_anydepth(row.post_mask_paths[0])
                b = cv_read_mask_anydepth(row.post_mask_paths[1])
                row._algo_mask_cache = stitch_mask_side_by_side(a, b, gap=12)
            else:
                row._algo_mask_cache = (cv_read_mask_anydepth(row.post_mask_paths[0]) > 0).astype(np.uint8) * 255
        except Exception:
            row._algo_mask_cache = None

        return row._algo_mask_cache

    def _get_yellow_boundary_mask(self, row: ResultRow, h: int, w: int) -> Optional[np.ndarray]:
        cached = getattr(row, "_yellow_boundary_cache", None)
        if cached is not None and cached.shape == (h, w):
            return cached
        if not row.tif_paths:
            return None

        try:
            parts: List[np.ndarray] = []
            for p in row.tif_paths:
                rgb = cv_read_rgb_anydepth(p)
                mask = (
                    (rgb[:, :, 0] >= 220)
                    & (rgb[:, :, 1] >= 220)
                    & (rgb[:, :, 2] <= 140)
                )
                part = (mask.astype(np.uint8) * 255)
                parts.append(part)

            if not parts:
                return None
            if len(parts) >= 2:
                overlay = stitch_mask_side_by_side(parts[0], parts[1], gap=12)
            else:
                overlay = parts[0]
            if overlay.shape != (h, w):
                overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_NEAREST)
            row._yellow_boundary_cache = overlay
        except Exception:
            row._yellow_boundary_cache = None

        return row._yellow_boundary_cache

    def _ensure_masked_seed(self, row: ResultRow, h: int, w: int):
        pp = row.pp_masked
        pp.ensure_shape(h, w)
        if row.masked_green_initialized:
            return

        algo_mask = self._get_algorithm_mask(row)
        if algo_mask is not None and algo_mask.shape == (h, w):
            # ColonyNet binary mask
            pp.paint_mask_green = algo_mask.copy().astype(np.uint8)
            yellow_mask = self._get_yellow_boundary_mask(row, h, w)
            if yellow_mask is not None and yellow_mask.shape == (h, w):
                pp.paint_mask_yellow = yellow_mask.copy().astype(np.uint8)
        elif row.cpsam_count is not None and row.tif_paths:
            # CPSAM overlay: pure green [0,255,0] = colony pixels,
            # pure yellow [255,255,0] = boundary lines between touching instances.
            # PNG is lossless so exact-value detection is 100 % reliable.
            try:
                ov = cv_read_rgb_anydepth(row.tif_paths[0])
                if ov is not None and ov.ndim == 3:
                    ov = ov.astype(np.int32)
                    R, G, B = ov[:, :, 0], ov[:, :, 1], ov[:, :, 2]
                    green_mask  = ((R == 0)   & (G == 255) & (B == 0)  ).astype(np.uint8) * 255
                    yellow_mask = ((R == 255) & (G == 255) & (B == 0)  ).astype(np.uint8) * 255
                    if green_mask.shape != (h, w):
                        green_mask  = cv2.resize(green_mask,  (w, h), interpolation=cv2.INTER_NEAREST)
                        yellow_mask = cv2.resize(yellow_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    if np.any(green_mask):
                        pp.paint_mask_green  = green_mask
                    if np.any(yellow_mask):
                        pp.paint_mask_yellow = yellow_mask
            except Exception:
                pass
        else:
            yellow_mask = self._get_yellow_boundary_mask(row, h, w)
            if yellow_mask is not None and yellow_mask.shape == (h, w):
                pp.paint_mask_yellow = yellow_mask.copy().astype(np.uint8)
        row.masked_green_initialized = True

    def _compose_with_annotations(
        self,
        base: np.ndarray,
        pp: PostprocessState,
    ) -> np.ndarray:
        out = base.copy().astype(np.uint8)
        h, w = out.shape[:2]
        pp.ensure_shape(h, w)

        # paint overlay (keeps old colors by using two masks)
        if pp.paint_mask_blue is not None:
            m = pp.paint_mask_blue > 0
            if np.any(m):
                out[m] = np.array([0, 0, 255], dtype=np.uint8)  # blue (RGB)

        if pp.paint_mask_pink is not None:
            m = pp.paint_mask_pink > 0
            if np.any(m):
                out[m] = np.array([255, 0, 180], dtype=np.uint8)  # pink (RGB)
        if pp.paint_mask_red is not None:
            m = pp.paint_mask_red > 0
            if np.any(m):
                out[m] = np.array([255, 0, 0], dtype=np.uint8)  # red (RGB)
        if pp.paint_mask_green is not None:
            m = pp.paint_mask_green > 0
            if np.any(m):
                out[m] = np.array([0, 255, 0], dtype=np.uint8)  # green (RGB)
        if pp.paint_mask_yellow is not None:
            m = pp.paint_mask_yellow > 0
            if np.any(m):
                out[m] = np.array([255, 255, 0], dtype=np.uint8)  # yellow (RGB)
        if pp.paint_mask_black is not None:
            m = pp.paint_mask_black > 0
            if np.any(m):
                out[m] = np.array([0, 0, 0], dtype=np.uint8)  # black (RGB)

        # 4) Coin-style selection labels — amber circle, black border, dark centred number
        #    NOTE: `out` is an RGB array (cv_read_rgb), so color tuples are (R, G, B)
        for lab in pp.labels:
            x, y, n = int(lab.x), int(lab.y), int(lab.n)
            r_coin = 26
            amber  = (255, 214, 107)   # #ffd66b
            dark   = (26,  18,   2)    # #1a1202

            # Soft gaussian glow (box-shadow: 0 0 12px rgba(255,214,107,0.6))
            glow = np.zeros_like(out)
            cv2.circle(glow, (x, y), r_coin + 10, amber, -1)
            glow = cv2.GaussianBlur(glow, (0, 0), 9)
            out  = cv2.addWeighted(out, 1.0, glow, 0.60, 0)

            # Amber fill + hairline ring (box-shadow: 0 0 0 1.5px #000)
            cv2.circle(out, (x, y), r_coin, amber, -1)
            cv2.circle(out, (x, y), r_coin, (0, 0, 0), 2)

            # Bold number via PIL — system font (SF Pro / Helvetica Neue Bold)
            try:
                from PIL import Image as _PIm, ImageDraw as _PDr, ImageFont as _PFnt
                _fsize = r_coin + 4   # fill ~75% of the coin diameter
                if not hasattr(self, "_coin_font") or getattr(self, "_coin_font_sz", 0) != _fsize:
                    self._coin_font_sz = _fsize
                    self._coin_font = None
                    for _fp, _idx in [
                        ("/System/Library/Fonts/HelveticaNeue.ttc", 4),   # Black
                        ("/System/Library/Fonts/HelveticaNeue.ttc", 2),   # Bold fallback
                        ("/Library/Fonts/Arial Bold.ttf", 0),
                        ("/System/Library/Fonts/Helvetica.ttc", 0),
                        ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0),
                        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
                    ]:
                        try:
                            self._coin_font = _PFnt.truetype(_fp, _fsize, index=_idx)
                            break
                        except Exception:
                            pass
                _pil = _PIm.fromarray(out)
                _PDr.Draw(_pil).text((x, y), str(n), font=self._coin_font,
                                     fill=(dark[0], dark[1], dark[2]), anchor="mm")
                out = np.array(_pil)
            except Exception:
                fs, fw = 0.70, 2
                _txt = str(n)
                (tw, th), _ = cv2.getTextSize(_txt, cv2.FONT_HERSHEY_SIMPLEX, fs, fw)
                cv2.putText(out, _txt, (x - tw // 2, y + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, dark, fw, cv2.LINE_AA)

        return out

    def _build_count_mask_rgb(self, row: ResultRow, h: int, w: int) -> np.ndarray:
        self._ensure_masked_seed(row, h, w)
        pp = row.pp_masked
        pp.ensure_shape(h, w)
        out = np.zeros((h, w, 3), dtype=np.uint8)

        masks_and_colors = [
            (pp.paint_mask_blue, np.array([0, 0, 255], dtype=np.uint8)),
            (pp.paint_mask_pink, np.array([255, 0, 180], dtype=np.uint8)),
            (pp.paint_mask_red, np.array([255, 0, 0], dtype=np.uint8)),
            (pp.paint_mask_green, np.array([0, 255, 0], dtype=np.uint8)),
        ]
        for mask, color in masks_and_colors:
            if mask is None:
                continue
            m = mask > 0
            if np.any(m):
                out[m] = color
        return out

    def _count_connected_components(self, mask: Optional[np.ndarray]) -> int:
        if mask is None:
            return 0
        binary = (mask > 0).astype(np.uint8)
        if not np.any(binary):
            return 0
        n_labels, _ = cv2.connectedComponents(binary, connectivity=8)
        return max(0, int(n_labels) - 1)

    def _count_cfus_from_mask(self, row: ResultRow, h: int, w: int) -> int:
        self._ensure_masked_seed(row, h, w)
        pp = row.pp_masked
        pp.ensure_shape(h, w)
        count = 0
        # Green colonies (ColonyNet binary mask OR CPSAM extracted from overlay)
        # minus yellow watershed/boundary pixels — identical logic for both algorithms.
        green = pp.paint_mask_green
        if green is not None:
            yellow = pp.paint_mask_yellow
            if yellow is not None:
                green_masked = ((green > 0) & (yellow == 0)).astype(np.uint8)
            else:
                green_masked = (green > 0).astype(np.uint8)
            count += self._count_connected_components(green_masked)
        # Manually painted colours counted separately; black is an eraser (not counted)
        for mask in (pp.paint_mask_blue, pp.paint_mask_pink, pp.paint_mask_red):
            count += self._count_connected_components(mask)
        return int(count)

    def _apply_preview_contrast(self, img: np.ndarray) -> np.ndarray:
        alpha = max(1.0, float(self.preview_contrast_percent) / 100.0)
        if alpha <= 1.0001:
            return img

        img_f = img.astype(np.float32)
        adjusted = (img_f - 127.5) * alpha + 127.5
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def _update_count_info_lbl(self, row: "ResultRow"):
        overlays = self._get_count_overlays_for_row(row, current_count=row.current_count)
        if not overlays:
            self.count_info_lbl.setText("")
            if hasattr(self, "preview_label"):
                self.preview_label.set_count_overlay([])
            return

        is_dark = True
        if is_dark:
            dim_col   = "#9aa1ae"   # text-dim  — labels
            count_col = "#ffd66b"   # amber     — current count value
            algo_col  = "#e7eaf0"   # text      — ColonyNet
            cpsam_col = "#9aa1ae"   # text-dim  — CPSAM
            gt_col    = "#9aa1ae"   # text-dim  — GT
            diff_neg  = "#ff8585"   # danger
            diff_pos  = "#8ce6a4"   # success
        else:
            dim_col   = "#555555"
            count_col = "#0060aa"
            algo_col  = "#006600"
            cpsam_col = "#004488"
            gt_col    = "#8a7000"
            diff_neg  = "#cc0000"
            diff_pos  = "#007700"

        parts = []
        canvas_lines = []  # (text, color_hex, is_big)

        for ov in overlays:
            # Current Count — big amber number on canvas, styled row in panel
            if ov.get("count_text"):
                raw = ov["count_text"]          # e.g. "Current Count: 142"
                try:
                    n_str = raw.split(":")[-1].strip()
                    canvas_lines.append((f"{n_str} colonies", "#ffd66b", True))
                except Exception:
                    canvas_lines.append((raw, "#ffd66b", True))
                try:
                    n_str = raw.split(":")[-1].strip()
                except Exception:
                    n_str = raw
                parts.append(
                    f'<span style="color:{dim_col};">Current Count: </span>'
                    f'<b style="color:{count_col};">{n_str}</b>'
                )

            # ColonyNet algo count
            if ov.get("algo_text"):
                parts.append(f'<span style="color:{algo_col};">{ov["algo_text"]}</span>')
                canvas_lines.append((ov["algo_text"], "#e7eaf0", False))

            # CPSAM count
            if ov.get("cpsam_text"):
                parts.append(f'<span style="color:{cpsam_col};">{ov["cpsam_text"]}</span>')
                canvas_lines.append((ov["cpsam_text"], "#9aa1ae", False))

            # CPSAM diff vs GT
            if ov.get("cpsam_diff_text"):
                raw_cdiff = ov["cpsam_diff_text"]
                try:
                    cdiff_val = int(raw_cdiff.split(":")[-1].strip().replace("−", "-"))
                    cdc = diff_pos if cdiff_val >= 0 else diff_neg
                except Exception:
                    cdc = diff_neg
                parts.append(f'<span style="color:{cdc};">{raw_cdiff}</span>')
                canvas_lines.append((raw_cdiff, cdc, False))

            # Ground truth
            if ov.get("gt_text"):
                parts.append(f'<span style="color:{gt_col};">{ov["gt_text"]}</span>')
                canvas_lines.append((ov["gt_text"], "#9aa1ae", False))

            # Diff — colour depends on sign
            if ov.get("diff_text"):
                raw_diff = ov["diff_text"]
                try:
                    diff_val = int(raw_diff.split(":")[-1].strip().replace("−", "-"))
                    dc = diff_pos if diff_val >= 0 else diff_neg
                except Exception:
                    dc = diff_neg
                parts.append(f'<span style="color:{dc};">{raw_diff}</span>')
                canvas_lines.append((raw_diff, dc, False))

        html = "<br>".join(parts)
        font_style = "font-family:'Geist Mono','SF Mono','Fira Mono',monospace;font-size:11pt;line-height:1.8;"
        self.count_info_lbl.setText(f'<div style="{font_style}">{html}</div>')

        # Push to canvas overlay
        if hasattr(self, "preview_label"):
            self.preview_label.set_count_overlay(canvas_lines)

    def on_table_select(self):
        row = self._current_row()
        if row is None:
            return
        want_masked = self.preview_mode_masked.isChecked()
        try:
            original, masked = self._get_base_images(row)
            if want_masked:
                self._ensure_masked_seed(row, masked.shape[0], masked.shape[1])
                # Auto-compute current count on first view (or after CPSAM results arrive).
                if row.current_count is None and (row.post_mask_paths or row.tif_paths):
                    row.current_count = self._count_cfus_from_mask(
                        row, masked.shape[0], masked.shape[1]
                    )
                composed = self._compose_with_annotations(masked, row.pp_masked)
            else:
                composed = self._compose_with_annotations(original, row.pp_original)
            composed = self._apply_preview_contrast(composed)
            self.preview_label.set_rgb(composed)
        except Exception:
            self.preview_label.setText("Preview render failed.")
            self.preview_label.setPixmap(QPixmap())
        self._update_count_info_lbl(row)

    def _set_tool(self, tool: Optional[str]):
        if self.btn_tool_paint.isChecked():
            self.active_tool = "paint"
        elif self.btn_tool_select.isChecked():
            self.active_tool = "select"
        elif self.btn_tool_remove.isChecked():
            self.active_tool = "remove"
        else:
            self.active_tool = None
        self._update_tool_cursor()

    def on_thickness_changed(self, v: int):
        self.thickness_value_lbl.setText(f"{v} px")
        self._update_tool_cursor()

    def on_contrast_changed(self, v: int):
        self.preview_contrast_percent = int(v)
        self.contrast_value_lbl.setText(f"{v}%")
        self.on_table_select()

    def _make_brush_cursor(self) -> QCursor:
        scale = self.preview_label.display_scale()
        r_img = max(1, int(self.thickness_slider.value()))
        r = int(np.clip(round(r_img * scale), 4, 128))
        d = r * 2 + 6
        c = d // 2
        pm = QPixmap(d, d)
        pm.fill(Qt.transparent)

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(0, 0, 0), 3))
        painter.drawEllipse(c - r, c - r, 2 * r, 2 * r)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawEllipse(c - r, c - r, 2 * r, 2 * r)
        painter.end()

        return QCursor(pm, c, c)

    def _update_tool_cursor(self):
        if self.preview_label._pan_mode:
            self.preview_label.setCursor(Qt.OpenHandCursor)
            return
        if self.active_tool in ("paint", "select", "remove"):
            self.preview_label.setCursor(self._make_brush_cursor())
        else:
            self.preview_label.setCursor(Qt.ArrowCursor)

    # ---- Drag painting support ----
    def _apply_tool_at(self, x: int, y: int, is_drag: bool):
        if self.active_tool is None:
            return
        row = self._current_row()
        if row is None:
            return

        want_masked = self.preview_mode_masked.isChecked()
        thickness = int(self.thickness_slider.value())
        radius = max(1, thickness)

        original, masked = self._get_base_images(row)
        base = masked if want_masked else original
        pp = row.pp_masked if want_masked else row.pp_original
        h, w = base.shape[:2]
        pp.ensure_shape(h, w)
        if want_masked:
            self._ensure_masked_seed(row, h, w)
        # Push undo snapshot once per click (and once at drag start)
        if not is_drag:
            pp.push_undo()
        else:
            # on drag, push only on first drag point
            if self._last_paint_xy is None:
                pp.push_undo()
        # throttle repeated points that are identical (or extremely close) during drag
        if is_drag and self._last_drag_xy is not None:
            lx, ly = self._last_drag_xy
            if (lx - x) * (lx - x) + (ly - y) * (ly - y) <= 2:
                return
        self._last_drag_xy = (x, y)

        if self.active_tool in ("paint", "remove") and want_masked:
            row.user_modified_mask = True

        if self.active_tool == "paint":
            if self.paint_color == "pink":
                target_mask = pp.paint_mask_pink
            elif self.paint_color == "red":
                target_mask = pp.paint_mask_red
            elif self.paint_color == "green":
                target_mask = pp.paint_mask_green
            elif self.paint_color == "black":
                target_mask = pp.paint_mask_black
            else:
                target_mask = pp.paint_mask_blue

            if is_drag and self._last_paint_xy is not None:
                x0, y0 = self._last_paint_xy
                self._stamp_line(target_mask, x0, y0, x, y, radius, 255)
            else:
                cv2.circle(target_mask, (x, y), radius, 255, -1)

            self._last_paint_xy = (x, y)

        elif self.active_tool == "remove":
            # erase both colors smoothly
            for m in (pp.paint_mask_blue, pp.paint_mask_pink, pp.paint_mask_red, pp.paint_mask_green, pp.paint_mask_yellow, pp.paint_mask_black):
                if m is None:
                    continue
                if is_drag and self._last_paint_xy is not None:
                    x0, y0 = self._last_paint_xy
                    self._stamp_line(m, x0, y0, x, y, radius, 0)
                else:
                    cv2.circle(m, (x, y), radius, 0, -1)

            self._last_paint_xy = (x, y)

            # ALSO remove nearest selection label (only on click, not during drag)
            if (not is_drag) and pp.labels:
                best_i = None
                best_d2 = None
                for i, lab in enumerate(pp.labels):
                    d2 = (lab.x - x) ** 2 + (lab.y - y) ** 2
                    if best_d2 is None or d2 < best_d2:
                        best_d2 = d2
                        best_i = i
                # threshold: within radius-ish area
                if best_d2 is not None and best_d2 <= (max(18, radius) ** 2):
                    pp.labels.pop(best_i)

        elif self.active_tool == "select":
            # only place one label per click (not continuously during drag)
            if not is_drag:
                pp.labels.append(AnnotationLabel(pp.next_label, x, y))
                pp.next_label += 1

        if self.active_tool in ("paint", "remove"):
            # Throttle ALL paint/remove redraws (click + drag) through the timer.
            # This prevents the initial click from blocking the event loop and
            # ensures panning (which emits drag_end but never sets this flag)
            # does not trigger unnecessary renders.
            self._was_paint_drag = True
            if not self._paint_redraw_pending:
                self._paint_redraw_pending = True
                self._paint_redraw_timer.start()
        else:
            self.on_table_select()

    def on_preview_clicked_once(self, x: int, y: int):
        self._last_drag_xy = None
        self._last_paint_xy = None
        self._apply_tool_at(x, y, is_drag=False)

    def on_preview_drag(self, x: int, y: int):
        # 3) enable painting/removing while mouse is held down and moving
        if self.active_tool in ("paint", "remove"):
            self._apply_tool_at(x, y, is_drag=True)

    def _flush_paint_redraw(self):
        self._paint_redraw_pending = False
        self.on_table_select()

    def on_preview_drag_end(self):
        self._last_drag_xy = None
        self._last_paint_xy = None
        self._paint_redraw_timer.stop()
        self._paint_redraw_pending = False
        # Only flush if we were actually painting — panning emits drag_end too
        # and calling on_table_select() there just creates a render backlog.
        if getattr(self, "_was_paint_drag", False):
            self._was_paint_drag = False
            self.on_table_select()

    def on_clear_annotations(self):
        row = self._current_row()
        if row is None:
            return

        want_masked = self.preview_mode_masked.isChecked()
        pp = row.pp_masked if want_masked else row.pp_original

        # Nothing to clear → silently return
        has_paint = (
            (pp.paint_mask_blue is not None and bool(np.any(pp.paint_mask_blue))) or
            (pp.paint_mask_pink is not None and bool(np.any(pp.paint_mask_pink))) or
            (pp.paint_mask_red is not None and bool(np.any(pp.paint_mask_red))) or
            (pp.paint_mask_green is not None and bool(np.any(pp.paint_mask_green))) or
            (pp.paint_mask_yellow is not None and bool(np.any(pp.paint_mask_yellow))) or
            (pp.paint_mask_black is not None and bool(np.any(pp.paint_mask_black)))
        )
        has_labels = bool(pp.labels)
        if (not has_paint) and (not has_labels):
            return

        mode = "MASKED image" if want_masked else "ORIGINAL image"

        # Build a custom dialog (since QMessageBox has no built-in "Clear" button)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Clear annotations?")
        msg.setText(f"Clear all annotations for the current {mode}?")
        msg.setInformativeText(
            "This will remove:\n"
            "• Painted green regions\n"
            "• All numbered selections\n\n"
            "This action cannot be undone."
        )

        btn_clear = msg.addButton("Clear", QMessageBox.AcceptRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.setDefaultButton(btn_clear)

        msg.exec()

        # If user didn't click Clear, do nothing
        if msg.clickedButton() is not btn_clear:
            return

        # Clear annotations
        if pp.paint_mask_blue is not None:
            pp.paint_mask_blue[:] = 0
        if pp.paint_mask_pink is not None:
            pp.paint_mask_pink[:] = 0
        if pp.paint_mask_red is not None:
            pp.paint_mask_red[:] = 0
        if pp.paint_mask_green is not None:
            if want_masked:
                algo_mask = self._get_algorithm_mask(row)
                if algo_mask is not None and algo_mask.shape == pp.paint_mask_green.shape:
                    pp.paint_mask_green[:] = algo_mask
                    row.masked_green_initialized = True
                else:
                    pp.paint_mask_green[:] = 0
            else:
                pp.paint_mask_green[:] = 0
        if pp.paint_mask_yellow is not None:
            if want_masked:
                yellow_mask = self._get_yellow_boundary_mask(row, pp.paint_mask_yellow.shape[0], pp.paint_mask_yellow.shape[1])
                if yellow_mask is not None and yellow_mask.shape == pp.paint_mask_yellow.shape:
                    pp.paint_mask_yellow[:] = yellow_mask
                else:
                    pp.paint_mask_yellow[:] = 0
            else:
                pp.paint_mask_yellow[:] = 0
        if pp.paint_mask_black is not None:
            pp.paint_mask_black[:] = 0
        pp.labels = []
        pp.next_label = 1

        row.user_modified_mask = False
        row.current_count = None
        self.on_table_select()

    def on_update_count(self):
        row = self._current_row()
        if row is None:
            return
        _, masked = self._get_base_images(row)
        count = self._count_cfus_from_mask(row, masked.shape[0], masked.shape[1])
        row.current_count = count
        row.user_modified_mask = True
        self.on_table_select()

    # ---------- CSV ----------
    def _csv_path(self) -> Optional[Path]:
        if self.current_out_dir is None:
            txt = self.out_dir_edit.text().strip()
            if not txt:
                return None
            self.current_out_dir = Path(txt)
        return self.current_out_dir / self.csv_filename

    def on_save_csv(self):
        csv_path = self._csv_path()
        if csv_path is None:
            QMessageBox.warning(self, "No output directory", "Select an output directory first.")
            return

        try:
            targets = self._get_save_targets()
            if not targets:
                QMessageBox.warning(self, "Nothing selected", "Select or check at least one image row to save.")
                return

            # Build updates from currently selected/checked rows.
            updates: Dict[str, int] = {}
            for r in targets:
                dataname = Path(r.source_image).stem
                if not r.user_modified_mask and r.algorithm_counts:
                    # No manual edits — trust the algorithm's authoritative count
                    count = sum(r.algorithm_counts)
                else:
                    _, masked = self._get_base_images(r)
                    count = self._count_cfus_from_mask(r, masked.shape[0], masked.shape[1])
                updates[dataname] = int(count)

            # Merge with existing CSV instead of replacing everything.
            rows_out: List[Tuple[str, int]] = []
            existing_idx: Dict[str, int] = {}
            if csv_path.exists():
                with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dataname = (row.get("Dataname") or "").strip()
                        count_txt = (row.get("Count") or "").strip()
                        if not dataname:
                            continue
                        try:
                            old_count = int(float(count_txt)) if count_txt else 0
                        except ValueError:
                            old_count = 0
                        existing_idx[dataname] = len(rows_out)
                        rows_out.append((dataname, old_count))

            # Update existing lines and append new lines.
            for dataname, count in updates.items():
                if dataname in existing_idx:
                    rows_out[existing_idx[dataname]] = (dataname, count)
                else:
                    rows_out.append((dataname, count))

            safe_mkdir(csv_path.parent)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["Dataname", "Count"])
                for dataname, count in rows_out:
                    w.writerow([dataname, int(count)])

            self.status_lbl.setText(f"Saved CSV: {csv_path.name}")
            if not getattr(self, "_suppress_csv_popup", False):
                QMessageBox.information(self, "CSV saved", f"Saved:\n{csv_path}")

        except Exception:
            tb = traceback.format_exc()
            QMessageBox.critical(self, "CSV save error", tb)

    def on_save(self):
        """
        SAVE = saves CSV + writes annotated images to output folder.
        User explicitly saved — CPSAM temp files are now permanent.
        """
        self._cpsam_temp_dirs.clear()
        # Save CSV without its own popup (we show one combined popup at the end)
        self._suppress_csv_popup = True
        try:
            self.on_save_csv()
        finally:
            self._suppress_csv_popup = False

        out_dir = self._csv_path()
        if out_dir is None:
            return
        out_dir = out_dir.parent
        safe_mkdir(out_dir)

        try:
            # Save annotated versions for every input image
            targets = self._get_save_targets()
            if not targets:
                QMessageBox.warning(self, "Nothing selected", "Select or check at least one image row to save.")
                return

            for r in targets:
                _, masked = self._get_base_images(r)

                self._ensure_masked_seed(r, masked.shape[0], masked.shape[1])
                mask_ann = self._compose_with_annotations(masked, r.pp_masked)
                mask_rgb = self._build_count_mask_rgb(r, masked.shape[0], masked.shape[1])

                stem = Path(r.source_image).stem
                p2 = out_dir / f"{stem}_annotated.tiff"
                p3 = out_dir / f"{stem}_mask.tiff"
                cv2.imwrite(str(p2), cv2.cvtColor(mask_ann, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(p3), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))

            # Don't call _attach_cpsam_outputs_to_originals here: loading the saved
            # _annotated.tiff as the new base would burn green/yellow into the background
            # layer, making subsequent mask removal impossible.
            self.status_lbl.setText("Saved CSV + annotated images + masks")
            QMessageBox.information(self, "Saved", f"Saved CSV + annotated images + masks to:\n{out_dir}")

        except Exception:
            QMessageBox.critical(self, "Save error", traceback.format_exc())

    def on_open_cpsam_dialog(self):
        image_paths = self._get_ssh_input_paths()
        if not image_paths:
            QMessageBox.warning(self, "No input", "Select/check images first.")
            return

        # ── Guard: warn if server not connected ───────────────────────────────
        state = getattr(self, "_conn_state", "checking")
        if state != "connected":
            if state == "not_logged_in":
                msg = ("Tailscale is not logged in.\n\n"
                       "Sign in to Tailscale first, then try again.")
            elif state == "no_tailscale":
                msg = ("Tailscale is not installed.\n\n"
                       "Download Tailscale from tailscale.com and log in.")
            else:
                msg = ("The GPU server is not reachable right now.\n\n"
                       "Check that Tailscale is running and the server is on, "
                       "or open Server Settings to verify the connection details.")
            mb = QMessageBox(self)
            mb.setWindowTitle("No Server Connection")
            mb.setText(msg)
            mb.setIcon(QMessageBox.Warning)
            mb.addButton("Cancel", QMessageBox.RejectRole)
            btn_open_settings = mb.addButton("Open Settings", QMessageBox.ActionRole)
            mb.exec()
            if mb.clickedButton() is btn_open_settings:
                self._open_settings_dialog()
            return

        # ── Auto-trigger first-time SSH key setup if needed ───────────────────
        _s = QSettings("CARA", "CPSAMGPUServer")
        _setup_done = _s.value("setup_done", "") == "1"
        _saved_key  = Path(_s.value("gpu_key", GPU_KEY_DEFAULT)).expanduser()
        if not _setup_done or not _saved_key.exists():
            self._run_ssh_setup_wizard()
            # Re-read after wizard; if key still missing, bail
            _s.sync()
            _saved_key = Path(_s.value("gpu_key", GPU_KEY_DEFAULT)).expanduser()
            if not _saved_key.exists():
                return

        # ── Run directly using saved settings ─────────────────────────────────
        _s   = QSettings("CARA", "CPSAMGPUServer")
        host = _s.value("gpu_host",   GPU_HOST_DEFAULT)
        port = int(_s.value("gpu_port", str(GPU_PORT_DEFAULT)))
        user = _s.value("gpu_user",   GPU_USER_DEFAULT)
        key  = _s.value("gpu_key",    GPU_KEY_DEFAULT)
        root = _s.value("gpu_root",   GPU_CPSAM_ROOT)
        env  = _s.value("gpu_env",    GPU_CONDA_ENV)
        script = _s.value("gpu_script", GPU_CPSAM_SCRIPT)
        model  = _s.value("gpu_model",  GPU_CPSAM_MODEL)

        out_base = self._csv_path()
        self._cpsam_local_out_dir = (
            out_base.parent / "cpsam_out" if out_base else Path.cwd() / "cpsam_out"
        )
        self._cpsam_local_out_dir.mkdir(parents=True, exist_ok=True)

        self.status_lbl.setText(
            f"CPSAM: uploading {len(image_paths)} image(s) to GPU server…")
        self.btn_process_cpsam.setEnabled(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        # Keep signals on self so they outlive this method's stack frame
        # (same SIGSEGV-prevention pattern as before).
        self._cpsam_signals = SSHJobSignals()
        task = DirectSSHCPSAMTask(
            image_paths=image_paths,
            local_out_dir=self._cpsam_local_out_dir,
            host=host, port=port, username=user, key_file=key,
            remote_root=root, conda_env=env,
            cpsam_script=script, cpsam_model=model,
            signals=self._cpsam_signals,
            cleanup_remote=True,
        )
        self._cpsam_signals.status.connect(self.status_lbl.setText)
        self._cpsam_signals.progress.connect(self.progress.setValue)
        self._cpsam_signals.finished.connect(self._on_cpsam_done)
        self._cpsam_signals.error.connect(self._on_cpsam_error)
        self.thread_pool.start(task)

    def _attach_cpsam_outputs_to_originals(self, output_paths: List[str]) -> int:
        if not self.results:
            return 0

        # Parse CPSAM count CSV if present — keyed by image_name stem (lowercased).
        cpsam_counts: Dict[str, int] = {}
        for p in output_paths:
            if Path(p).name.lower() == "countcfuapp_test_results.csv":
                try:
                    import csv as _csv
                    with open(p, newline="", encoding="utf-8-sig") as _f:
                        for _row in _csv.DictReader(_f):
                            name = (_row.get("image_name") or "").strip()
                            cnt_str = (_row.get("count") or "").strip()
                            if name and cnt_str.isdigit():
                                stem_key = Path(name).stem.lower()
                                cpsam_counts[stem_key] = int(cnt_str)
                except Exception:
                    pass
                break

        # Build output index by stem — skip CSV files.
        outs = [Path(p) for p in output_paths if Path(p).suffix.lower() != ".csv"]
        out_stems = {p: p.stem.lower() for p in outs}

        matched_rows = 0
        first_matched_idx: Optional[int] = None
        for i, row in enumerate(self.results):
            src_stem = Path(row.source_image).stem.lower()

            matched = []
            for p in outs:
                s = out_stems[p]
                if s == src_stem or s.startswith(src_stem + "_"):
                    matched.append(str(p))

            if not matched:
                continue

            matched_rows += 1
            if first_matched_idx is None:
                first_matched_idx = i

            # Overlay images → tif_paths (for display).
            # The original source image is set as expt_paths so the unmodified
            # image is used as the editing base (identical to how ColonyNet works).
            matched.sort(
                key=lambda p: (
                    0 if "_annotated" in Path(p).stem.lower() else 1,
                    -(Path(p).stat().st_mtime if Path(p).exists() else 0.0),
                    Path(p).name.lower(),
                )
            )
            non_ann = [p for p in matched if "_annotated" not in Path(p).stem.lower()]
            if non_ann:
                row.prefer_annotated = False
                newest = max(
                    non_ann,
                    key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0.0,
                )
                row.tif_paths = [newest]
            else:
                row.prefer_annotated = True
                existing = [x for x in row.tif_paths if x not in matched]
                row.tif_paths = matched + existing

            # Use original source image as editing base (so erasing shows the real image).
            row.expt_paths = [row.source_image]

            row._masked_cache = None
            row.masked_green_initialized = False   # re-extract green/yellow from new overlay
            # Use CSV count directly as current_count so it matches CPSAM's own tally.
            # Fall back to pixel-recompute only when no CSV count is available.
            if src_stem in cpsam_counts:
                row.cpsam_count = cpsam_counts[src_stem]
                row.current_count = cpsam_counts[src_stem]
            else:
                row.current_count = None           # will be auto-computed on next select
            self._update_table_row(i, row)

        if matched_rows > 0:
            self.preview_mode_masked.setChecked(True)
            self._force_table_dark()
            if self.table.selectionModel().hasSelection():
                self.on_table_select()
            elif first_matched_idx is not None:
                self.table.selectRow(first_matched_idx)
                self.on_table_select()
        return matched_rows

    def _export_mask_for_web(self, r: "ResultRow", kind: str) -> Optional[str]:
        """Write current in-memory mask to the system temp dir and return its path, or None."""
        pp = r.pp_masked
        if kind == "green":
            mask = pp.paint_mask_green
        else:
            mask = pp.paint_mask_yellow
        if mask is None or not np.any(mask):
            return None
        try:
            import tempfile as _tempfile
            suffix = f"_web_{kind}.png"
            stem = Path(r.source_image).stem
            tmp_path = Path(_tempfile.gettempdir()) / f"{stem}{suffix}"
            cv2.imwrite(str(tmp_path), mask)
            self._web_tmp_files.add(str(tmp_path))
            return str(tmp_path)
        except Exception:
            return None

    def on_web_ui_help(self):
        self._cleanup_web_tmp_files()
        rows = []
        selected_paths = set(self._get_checked_or_all_source_paths())
        src_rows = self.results if self.results else [ResultRow(source_image=p, tif_paths=[]) for p in self.selected_images]
        for r in src_rows:
            src = str(Path(r.source_image))
            if not Path(src).exists():
                continue
            if selected_paths and src not in selected_paths:
                continue
            # New session should start from base mask, not previous *_annotated artifacts.
            base_tifs = [p for p in r.tif_paths if "_annotated" not in Path(p).stem.lower()]
            base_expts = [p for p in r.expt_paths if "_annotated" not in Path(p).stem.lower()]
            base_post_masks = [p for p in r.post_mask_paths if "_annotated" not in Path(p).stem.lower()]

            # If user has edited masks in-app, export the current in-memory state so
            # the webapp session shows the same masks the user is looking at.
            if r.masked_green_initialized:
                exported_green = self._export_mask_for_web(r, "green")
                exported_yellow = self._export_mask_for_web(r, "yellow")
                if exported_green:
                    base_post_masks = [exported_green]
                if exported_yellow:
                    # Build an overlay RGB PNG with green+yellow so the webapp
                    # can extract the yellow boundary mask from it.
                    try:
                        import tempfile as _tempfile
                        h, w = r.pp_masked.paint_mask_yellow.shape[:2]
                        overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                        gm = r.pp_masked.paint_mask_green > 0
                        overlay_rgb[gm] = [0, 255, 0]
                        ym = r.pp_masked.paint_mask_yellow > 0
                        overlay_rgb[ym] = [255, 255, 0]
                        tmp_overlay = Path(_tempfile.gettempdir()) / f"{Path(r.source_image).stem}_web_overlay.png"
                        cv2.imwrite(str(tmp_overlay), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                        self._web_tmp_files.add(str(tmp_overlay))
                        base_tifs = [str(tmp_overlay)]
                    except Exception:
                        pass

            _tokens = self._parse_label_dilu_tokens(src)
            _gt_count = None
            if _tokens:
                _label, _dilu, _x_suffix = _tokens[0]
                _gt_count = self._lookup_gt_count(_label, _dilu, _x_suffix)
            rows.append(
                {
                    "image": Path(src).name,
                    "source_path": src,
                    "tif_paths": base_tifs,
                    "expt_paths": base_expts,
                    "post_mask_paths": base_post_masks,
                    "algo_count": r.algorithm_counts[0] if r.algorithm_counts else None,
                    "cpsam_count": r.cpsam_count,
                    "current_count": r.current_count,
                    "gt_count": _gt_count,
                }
            )
        if not rows:
            QMessageBox.warning(self, "Web UI", "Select images first in app.py.")
            return

        try:
            if self.webui_proc is not None and self.webui_proc.poll() is None:
                self.webui_proc.terminate()
                try:
                    self.webui_proc.wait(timeout=1.5)
                except Exception:
                    self.webui_proc.kill()
            cmd = [sys.executable, "-m", "uvicorn", "webui.main:app", "--host", "0.0.0.0", "--port", "8000"]
            self.webui_proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).resolve().parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.8)
        except Exception:
            QMessageBox.warning(
                self,
                "Web UI start failed",
                "Could not start web UI server automatically.\n"
                "Run manually:\n"
                "cd \"1.5: Python Version\"\n"
                "pip install -r webui/requirements-web.txt\n"
                "python3 webui/main.py",
            )
            return

        if not self._wait_webui_ready():
            QMessageBox.warning(self, "Web UI", "Server did not become ready on port 8000.")
            return

        sid = self._create_web_session_from_app(rows)
        if not sid:
            QMessageBox.warning(self, "Web UI", "Could not create iPad session from current app rows.")
            return
        self.web_session_id = sid
        self.web_saved_seen = set()
        self.web_sync_timer.start()

        ip = self._guess_local_ip()
        mac_url = f"http://127.0.0.1:8000/?sid={sid}"
        ipad_url = f"http://{ip}:8000/?sid={sid}"
        self._show_ipad_qr_dialog(mac_url, ipad_url)

    def _show_ipad_qr_dialog(self, mac_url: str, ipad_url: str):
        dlg = QDialog(self)
        dlg.setWindowTitle("Scan On iPad")
        dlg.resize(420, 520)
        lay = QVBoxLayout(dlg)

        title = QLabel("Session ready. Scan this QR code with iPad camera")
        title.setWordWrap(True)
        lay.addWidget(title)

        qr_label = QLabel()
        qr_label.setAlignment(Qt.AlignCenter)
        qr_label.setMinimumHeight(320)
        qr_ok = False
        try:
            import io
            import qrcode
            from PIL import Image, ImageChops, ImageOps

            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(ipad_url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

            base_dir = Path(__file__).resolve().parent
            logo_candidates = [
                base_dir / "Logo.png",
                base_dir / "webui" / "static" / "logo.png",
                base_dir / "logo.png",
            ]
            logo_path = next((p for p in logo_candidates if p.exists()), None)
            if logo_path is not None:
                logo = Image.open(logo_path).convert("RGBA")
                logo = ImageOps.grayscale(logo).convert("RGBA")
                logo_size = max(48, qr_img.size[0] // 5)
                logo.thumbnail((logo_size, logo_size), Image.Resampling.LANCZOS)

                bg_size = logo.size[0] + 20, logo.size[1] + 20
                logo_bg = Image.new("RGBA", bg_size, (255, 255, 255, 255))
                bg_x = (qr_img.size[0] - bg_size[0]) // 2
                bg_y = (qr_img.size[1] - bg_size[1]) // 2
                qr_img.paste(logo_bg, (bg_x, bg_y), logo_bg)

                logo_alpha = logo.getchannel("A")
                logo_bw = ImageOps.autocontrast(logo.convert("L"))
                logo_fill = logo_bw.point(lambda v: 255 if v < 200 else 0)
                logo_mask = ImageChops.multiply(logo_alpha, logo_fill)
                logo_final = Image.new("RGBA", logo.size, (255, 255, 255, 0))
                logo_final.paste((0, 0, 0, 255), mask=logo_mask)

                logo_x = (qr_img.size[0] - logo.size[0]) // 2
                logo_y = (qr_img.size[1] - logo.size[1]) // 2
                qr_img = qr_img.convert("RGBA")
                qr_img.paste(logo_final, (logo_x, logo_y), logo_final)
                qr_img = qr_img.convert("RGB")
            buf = io.BytesIO()
            qr_img.save(buf, format="PNG")
            pm = QPixmap()
            if pm.loadFromData(buf.getvalue(), "PNG"):
                qr_label.setPixmap(pm.scaled(320, 320, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                qr_ok = True
        except Exception:
            qr_ok = False

        if not qr_ok:
            qr_label.setText("QR generation unavailable.\nInstall package: pip install qrcode[pil]")
        lay.addWidget(qr_label)

        mac_lbl = QLabel(f"Mac: {mac_url}")
        mac_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        ipad_lbl = QLabel(f"iPad: {ipad_url}")
        ipad_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(mac_lbl)
        lay.addWidget(ipad_lbl)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close)
        dlg.exec()

    def _guess_local_ip(self) -> str:
        ip = "<MAC_LOCAL_IP>"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
        return ip

    def _wait_webui_ready(self) -> bool:
        url = "http://127.0.0.1:8000/api/health"
        for _ in range(25):
            try:
                with urllib.request.urlopen(url, timeout=1.0) as resp:
                    if int(getattr(resp, "status", 200)) == 200:
                        return True
            except Exception:
                time.sleep(0.2)
        return False

    def _create_web_session_from_app(self, rows: List[dict]) -> Optional[str]:
        payload = json.dumps({"rows": rows}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/session/from_app",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            sid = data.get("session_id")
            if isinstance(sid, str) and sid:
                return sid
        except Exception:
            return None
        return None

    def _apply_webapp_annotated_to_row(self, row: "ResultRow", png_path: str) -> bool:
        """
        Read a webapp-saved annotated PNG (base image + overlay composite) and
        decode its pixel colors back into the row's pp_masked masks.
        Green pixels → paint_mask_green, yellow → paint_mask_yellow.
        Returns True if something changed.
        """
        try:
            bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
            if bgr is None:
                return False
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.int32)

            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

            # Decode exact paint colors — canvas uses pure integer RGB values so
            # thresholds can be very tight, avoiding false matches with image content.
            # Green  (0,255,0)   — algorithm seed + user green paint
            green_mask  = ((g > 240) & (r <  15) & (b <  15)).astype(np.uint8) * 255
            # Yellow (255,255,0) — algorithm watershed boundaries
            yellow_mask = ((r > 240) & (g > 240) & (b <  15)).astype(np.uint8) * 255
            # Blue   (0,0,255)
            blue_mask   = ((b > 240) & (r <  15) & (g <  15)).astype(np.uint8) * 255
            # Pink   (255,0,180)
            pink_mask   = ((r > 240) & (g <  15) & (b > 160) & (b < 200)).astype(np.uint8) * 255
            # Red    (255,0,0)
            red_mask    = ((r > 240) & (g <  15) & (b <  15)).astype(np.uint8) * 255

            # Get the actual base image dimensions so masks survive pp.ensure_shape()
            _, masked_base = self._get_base_images(row)
            H, W = masked_base.shape[:2]

            def _resize(m: np.ndarray) -> np.ndarray:
                if m.shape == (H, W):
                    return m
                return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

            green_mask  = _resize(green_mask)
            yellow_mask = _resize(yellow_mask)
            blue_mask   = _resize(blue_mask)
            pink_mask   = _resize(pink_mask)
            red_mask    = _resize(red_mask)

            # Initialize seed once so ensure_shape won't clobber us
            self._ensure_masked_seed(row, H, W)
            pp = row.pp_masked
            pp.paint_mask_green  = green_mask
            pp.paint_mask_yellow = yellow_mask
            pp.paint_mask_blue   = blue_mask
            pp.paint_mask_pink   = pink_mask
            pp.paint_mask_red    = red_mask
            row.user_modified_mask = True
            return True
        except Exception:
            return False

    def _poll_web_session_saved(self):
        sid = self.web_session_id
        if not sid:
            return
        url = f"http://127.0.0.1:8000/api/session/{sid}"
        try:
            with urllib.request.urlopen(url, timeout=1.2) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return

        saved_local = data.get("saved_local") or []
        if not isinstance(saved_local, list):
            return

        new_paths = []
        for p in saved_local:
            if not isinstance(p, str):
                continue
            if p in self.web_saved_seen:
                continue
            self.web_saved_seen.add(p)
            if Path(p).exists():
                new_paths.append(p)

        if not new_paths:
            return

        loaded = 0
        for png_path in new_paths:
            png_stem = Path(png_path).stem.lower()
            for row in self.results:
                src_stem = Path(row.source_image).stem.lower()
                if png_stem.startswith(src_stem):
                    if self._apply_webapp_annotated_to_row(row, png_path):
                        # Auto-update count so all painted colors (green/blue/pink/red) are reflected
                        try:
                            _, masked_base = self._get_base_images(row)
                            row.current_count = self._count_cfus_from_mask(
                                row, masked_base.shape[0], masked_base.shape[1]
                            )
                        except Exception:
                            pass
                        loaded += 1
                    break

        if loaded > 0:
            self.status_lbl.setText(f"iPad annotations synced: matched {loaded} image(s).")
            self._force_table_dark()
            if self.table.selectionModel().hasSelection():
                self.on_table_select()

    def _get_ssh_input_paths(self) -> List[str]:
        targets = self._get_save_targets()
        if targets:
            return [r.source_image for r in targets]
        return list(self.selected_images)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    base_dir = Path(__file__).resolve().parent
    app_icon_path = base_dir / "Logo.png"
    app_icon = QIcon()
    if app_icon_path.exists():
        src = QPixmap(str(app_icon_path))
        if not src.isNull():
            icon_size = 512
            rounded = QPixmap(icon_size, icon_size)
            rounded.fill(Qt.transparent)
            painter = QPainter(rounded)
            try:
                painter.setRenderHint(QPainter.Antialiasing, True)
                path = QPainterPath()
                outer_rect = QRectF(54.0, 54.0, 404.0, 404.0)
                radius = 110.0
                path.addRoundedRect(outer_rect, radius, radius)
                painter.fillPath(path, QColor(255, 255, 255))
                painter.setClipPath(path)
                target_rect = QRect(
                    int(round(outer_rect.x() + 58.0)),
                    int(round(outer_rect.y() + 58.0)),
                    int(round(outer_rect.width() - 116.0)),
                    int(round(outer_rect.height() - 116.0)),
                )
                scaled = src.scaled(target_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                draw_x = target_rect.x() + (target_rect.width() - scaled.width()) // 2
                draw_y = target_rect.y() + (target_rect.height() - scaled.height()) // 2
                painter.drawPixmap(draw_x, draw_y, scaled)
            finally:
                painter.end()
            app_icon = QIcon(rounded)
            app.setWindowIcon(app_icon)

    # Load QSS theme
    qss_path = Path(__file__).resolve().parent / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    w = MainWindow()
    if not app_icon.isNull():
        w.setWindowIcon(app_icon)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
