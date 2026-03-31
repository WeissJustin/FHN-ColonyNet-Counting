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
from PySide6.QtCore import Qt, QObject, Signal, QRunnable, QThreadPool, QRect, QRectF, QPoint, QTimer, QModelIndex
from PySide6.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QCursor, QPen, QIcon, QPainterPath
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

SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
VERSION_FOLDER = ""  # keep as in your code
GT_TOKEN_RE = re.compile(r"^([A-Za-z0-9]+)dilu(?:e)?(\d+)$", re.IGNORECASE)
ORCD_HOST_DEFAULT = "orcd-login.mit.edu"
ORCD_PORT_DEFAULT = 22
ORCD_CPSAM_ROOT = "/home/juweiss/UROP/CPSAM"
ORCD_IMAGES_APP_DIR = f"{ORCD_CPSAM_ROOT}/Images_App"
ORCD_OUTPUT_APP_DIR = f"{ORCD_CPSAM_ROOT}/Output_App"
ORCD_MUX_SOCKET = "~/.ssh/orcd_mux"
LIGHT_COMBO_ARROW_PATH = (Path(__file__).resolve().parent / "ui_assets" / "combo_arrow_dark.svg").as_posix()
SHARED_SLIDER_QSS = """
QSlider {
    min-height: 18px;
}
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #1f2937;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #4b5563;
    border-radius: 3px;
}
QSlider::add-page:horizontal {
    background: #1f2937;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f3f4f6;
    border: 1px solid #9ca3af;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
"""


class ColorComboBox(QComboBox):
    def showPopup(self):
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
        view.setContentsMargins(0, 0, 0, 0)
        view.viewport().setContentsMargins(0, 0, 0, 0)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        popup = view.window()
        base = view.palette().color(QPalette.Base)
        popup_pal = popup.palette()
        popup_pal.setColor(QPalette.Window, base)
        popup_pal.setColor(QPalette.Base, base)
        popup.setPalette(popup_pal)
        popup.setAutoFillBackground(True)
        popup.setContentsMargins(0, 0, 0, 0)
        popup.setStyleSheet(f"background: {base.name()}; border: none;")
        row_count = view.model().rowCount() if view.model() is not None else 0
        if row_count > 0:
            row_h = max(view.sizeHintForRow(i) for i in range(row_count))
            total_h = row_h * row_count + 2
            popup.resize(popup.width(), total_h)
        view.viewport().update()

DEFAULT_DARK_APP_QSS = """
QMainWindow, QWidget { background: #000000; color: #eaeaea; }

QGroupBox {
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    margin-top: 10px;
    padding: 10px;
    background: #000000;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #cfcfcf;
    background: transparent;
}
QLabel { color: #d8d8d8; background: transparent; }
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {
    background: #050505;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 8px 10px;
    color: #f0f0f0;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: none;
    background: #050505;
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
}
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
    margin-right: 6px;
}
QPushButton {
    background: #0b0b0b;
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    padding: 10px 12px;
    color: #f0f0f0;
}
QPushButton:hover { border: 1px solid #666666; background: #101010; }
QPushButton:pressed { background: #070707; }
QPushButton:checked { background: #1a1a1a; border: 1px solid #9a9a9a; font-weight: 600; }
QPushButton[primary="true"] { background: #1a1a1a; border: 1px solid #8a8a8a; font-weight: 700; }
QPushButton[primary="true"]:hover { background: #202020; border: 1px solid #b0b0b0; }
QPushButton[danger="true"] { background: #1a0505; border: 1px solid #6a2a2a; color: #ffd3d3; }
QPushButton[danger="true"]:hover { background: #220707; border: 1px solid #a04040; }
QPushButton[success="true"] { background: #051a05; border: 1px solid #2a6a2a; color: #b0ffb0; font-weight: 700; }
QPushButton[success="true"]:hover { background: #072207; border: 1px solid #40a040; }
QProgressBar {
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    text-align: center;
    background: #050505;
    color: #eaeaea;
    height: 18px;
}
QProgressBar::chunk { background: #00aa00; border-radius: 10px; }
QHeaderView::section {
    background: #050505;
    border: 1px solid #111111;
    padding: 6px;
    color: #dcdcdc;
}
QTableCornerButton::section { background: #050505; border: 1px solid #111111; }
QTableWidget, QTableView, QTableWidget::viewport, QTableView::viewport {
    background: #000000;
    color: #eaeaea;
    border: 1px solid #1f1f1f;
    gridline-color: #161616;
    selection-background-color: #1a1a1a;
    selection-color: #ffffff;
}
QTableWidget::item { background: #000000; color: #eaeaea; }
QTableWidget::item:selected { background: #1a1a1a; color: #ffffff; }
QRadioButton, QCheckBox { color: #eaeaea; background: transparent; spacing: 8px; }
QRadioButton::indicator, QCheckBox::indicator {
    width: 16px;
    height: 16px;
}
QRadioButton::indicator {
    border: 1px solid #6b7280;
    border-radius: 8px;
    background: #050505;
}
QRadioButton::indicator:checked {
    border: 1px solid #b8bec8;
    background: #f3f4f6;
}
QCheckBox::indicator {
    border: 1px solid #6b7280;
    border-radius: 4px;
    background: #050505;
}
QCheckBox::indicator:checked {
    border: 1px solid #9ca3af;
    background: #22c55e;
}
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #1f2937;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #4b5563;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #f3f4f6;
    border: 1px solid #9ca3af;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
"""
LIGHT_APP_QSS = """
* {
  font-family: "Inter", "SF Pro Display", "SF Pro Text", "Segoe UI", Arial;
  font-size: 13px;
  color: rgba(25,30,38,220);
}

QMainWindow, QWidget { background: #f4f5f7; }
QLabel { background: transparent; }

QPushButton:focus,
QRadioButton:focus,
QSpinBox:focus,
QTableWidget:focus {
  outline: none;
}

QGroupBox {
  border: 1px solid rgba(28,34,45,18);
  border-radius: 18px;
  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                              stop:0 rgba(255,255,255,235),
                              stop:1 rgba(246,248,251,245));
  margin-top: 10px;
  padding: 16px;
}

QGroupBox::title {
  subcontrol-origin: margin;
  left: 16px;
  top: 12px;
  padding: 0 8px;
  color: rgba(37,44,54,150);
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  background: transparent;
}

QGroupBox#ControlsPanel,
QGroupBox#PostPanel {
  border-radius: 22px;
  background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                              stop:0 rgba(255,255,255,242),
                              stop:1 rgba(244,246,249,247));
}

QPushButton {
  background: rgba(248,250,252,235);
  border: 1px solid rgba(88,102,119,42);
  border-radius: 14px;
  padding: 12px 14px;
  font-weight: 600;
}

QPushButton:hover {
  border: 1px solid rgba(88,102,119,64);
  background: rgba(252,253,255,245);
}

QPushButton:pressed {
  background: rgba(224,230,237,245);
  border: 1px solid rgba(88,102,119,82);
}

QPushButton[primary="true"] {
  background: rgba(218,228,238,240);
  border: 1px solid rgba(108,129,147,120);
  font-weight: 800;
}
QPushButton[primary="true"]:hover {
  background: rgba(211,224,236,250);
  border: 1px solid rgba(88,113,133,145);
}
QPushButton[primary="true"]:pressed {
  background: rgba(197,212,226,255);
  border: 1px solid rgba(71,93,111,170);
}

QPushButton[danger="true"] {
  background: rgba(255,92,92,18);
  border: 1px solid rgba(220,87,87,70);
  color: rgba(140,35,35,245);
  font-weight: 800;
}
QPushButton[danger="true"]:hover {
  background: rgba(255,92,92,24);
  border: 1px solid rgba(220,87,87,95);
}
QPushButton[danger="true"]:pressed {
  background: rgba(238,208,208,255);
  border: 1px solid rgba(200,70,70,115);
}
QPushButton[success="true"] {
  background: rgba(60,180,60,30);
  border: 1px solid rgba(60,160,60,120);
  color: rgba(20,100,20,245);
  font-weight: 800;
}
QPushButton[success="true"]:hover {
  background: rgba(60,180,60,50);
  border: 1px solid rgba(40,140,40,160);
}

QPushButton[tool="true"] {
  text-align: left;
  padding: 14px 14px;
  border-radius: 16px;
  background: rgba(249,251,253,220);
  border: 1px solid rgba(88,102,119,38);
  color: rgba(32,39,48,185);
}

QPushButton[tool="true"]:hover {
  background: rgba(240,245,249,240);
  border: 1px solid rgba(96,113,131,60);
}

QPushButton[tool="true"]:checked {
  background: rgba(214,225,235,180);
  border: 2px solid rgba(108,129,147,125);
  color: rgba(22,27,34,245);
  font-weight: 900;
}

QPushButton[tool="true"]:pressed {
  background: rgba(204,214,224,210);
  border: 2px solid rgba(96,113,131,95);
  padding: 14px 14px;
}

QLineEdit {
  background: rgba(255,255,255,190);
  border: 1px solid rgba(31,41,55,12);
  border-radius: 14px;
  padding: 10px 12px;
  color: rgba(20,24,31,230);
}

QComboBox, QSpinBox, QPlainTextEdit {
  background: rgba(255,255,255,190);
  border: 1px solid rgba(31,41,55,12);
  border-radius: 12px;
  padding: 6px 10px;
  color: rgba(20,24,31,230);
}

QComboBox::drop-down {
  subcontrol-origin: padding;
  subcontrol-position: top right;
  width: 28px;
  border-left: none;
  background: rgba(255,255,255,190);
  border-top-right-radius: 12px;
  border-bottom-right-radius: 12px;
}

QComboBox::down-arrow {
  image: url("__LIGHT_COMBO_ARROW_PATH__");
  width: 10px;
  height: 10px;
  margin-right: 6px;
}

QComboBox QAbstractItemView {
  background: rgba(255,255,255,248);
  border: none;
  selection-background-color: rgba(255,255,255,248);
  selection-color: rgba(20,24,31,230);
  outline: 0;
}

QComboBox QAbstractItemView::item {
  background: rgba(255,255,255,248);
  color: rgba(20,24,31,230);
  min-height: 20px;
}

QComboBox QAbstractItemView::item:selected {
  background: rgba(255,255,255,248);
  color: rgba(20,24,31,230);
}

QCheckBox::indicator {
  width: 16px;
  height: 16px;
  border: 1px solid #c9c9c9;
  border-radius: 3px;
  background: #f7f7f7;
}

QCheckBox::indicator:checked {
  image: url("ui_assets/checkmark_dark.svg");
  border: 1px solid #c9c9c9;
  border-radius: 3px;
  background: #f7f7f7;
}

QTableWidget::indicator {
  width: 16px;
  height: 16px;
  border: 1px solid #c9c9c9;
  border-radius: 3px;
  background: #f7f7f7;
}

QTableWidget::indicator:unchecked {
  border: 1px solid #c9c9c9;
  border-radius: 3px;
  background: #f7f7f7;
}

QTableWidget::indicator:checked {
  image: url("ui_assets/checkmark_dark.svg");
  border: 1px solid #c9c9c9;
  border-radius: 3px;
  background: #f7f7f7;
}

QSpinBox {
  padding: 6px 26px 6px 10px;
}

QSpinBox::up-button, QSpinBox::down-button {
  subcontrol-origin: border;
  width: 18px;
  border: none;
  background: rgba(31,41,55,6);
}
QSpinBox::up-button {
  subcontrol-position: top right;
  border-top-right-radius: 12px;
}
QSpinBox::down-button {
  subcontrol-position: bottom right;
  border-bottom-right-radius: 12px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
  background: rgba(31,41,55,10);
}
QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {
  background: rgba(31,41,55,16);
}
QSpinBox::up-arrow, QSpinBox::down-arrow {
  width: 8px;
  height: 8px;
}

QRadioButton[seg="true"] {
  padding: 10px 16px;
  border: 1px solid rgba(31,41,55,12);
  border-radius: 16px;
  background: rgba(255,255,255,120);
  color: rgba(46,53,63,150);
  font-weight: 700;
}
QRadioButton[seg="true"]::indicator { width: 0px; height: 0px; }
QRadioButton[seg="true"]:hover {
  border: 1px solid rgba(31,41,55,18);
  background: rgba(250,251,253,220);
}
QRadioButton[seg="true"]:checked {
  color: rgba(25,30,38,240);
  background: rgba(215,224,233,185);
  border: 1px solid rgba(108,129,147,115);
}

#PreviewCanvas {
  background: rgba(255,255,255,120);
  border: 1px solid rgba(31,41,55,10);
  border-radius: 18px;
}

QTableWidget {
  background: rgba(255,255,255,150);
  border: 1px solid rgba(31,41,55,6);
  border-radius: 10px;
  gridline-color: rgba(31,41,55,5);
  color: rgba(20,24,31,230);
}

QTableWidget::viewport {
  background: rgba(0,0,0,0);
}

QTableCornerButton::section {
  background: rgba(0,0,0,0);
  border: none;
}

QHeaderView::section {
  background: transparent;
  border: none;
  padding: 14px 10px;
  color: rgba(44,50,59,120);
  font-weight: 800;
  letter-spacing: 2px;
  text-transform: uppercase;
}

QTableWidget::item {
  padding: 9px 8px;
  border-bottom: 1px solid rgba(31,41,55,5);
  background: rgba(0,0,0,0);
  color: rgba(20,24,31,220);
}

QTableWidget::item:selected {
  background: rgba(214,225,235,170);
}

QProgressBar {
  border: 1px solid rgba(31,41,55,10);
  border-radius: 999px;
  text-align: center;
  background: rgba(255,255,255,130);
  color: rgba(20,24,31,220);
  height: 18px;
}
QProgressBar::chunk {
  background: #16a34a;
  border-radius: 999px;
}

QSlider::groove:horizontal {
  border: none;
  height: 6px;
  background: #1f2937;
  border-radius: 3px;
}
QSlider::sub-page:horizontal {
  background: #4b5563;
  border-radius: 3px;
}
QSlider::handle:horizontal {
  background: #f3f4f6;
  border: 1px solid #9ca3af;
  width: 16px;
  margin: -5px 0;
  border-radius: 8px;
}
""".replace("__LIGHT_COMBO_ARROW_PATH__", LIGHT_COMBO_ARROW_PATH)


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
    return f"{stem}_Top", f"{stem}_Bottom"


# ----------------------------
# Interactive image widget (Zoom/Pan overlay buttons + drag paint support)
# ----------------------------
from PySide6.QtCore import Qt, Signal, QRect, QPoint, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter
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

        for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_pan):
            b.setAutoRaise(True)
            b.setCursor(Qt.PointingHandCursor)
            b.hide()

        tool_css = """
        QToolButton {
            background: rgba(10,10,10,200);
            border: 1px solid rgba(180,180,180,80);
            color: white;
            border-radius: 10px;
            padding: 6px 10px;
            font-weight: 700;
        }
        QToolButton:hover {
            border: 1px solid rgba(255,255,255,160);
            background: rgba(20,20,20,220);
        }
        QToolButton:checked {
            border: 1px solid rgba(255,255,255,200);
            background: rgba(30,30,30,230);
        }
        """
        self.btn_zoom_in.setStyleSheet(tool_css)
        self.btn_zoom_out.setStyleSheet(tool_css)
        self.btn_pan.setStyleSheet(tool_css)

        self.btn_zoom_in.clicked.connect(lambda: self.zoom_by(1.15))
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_by(1.0 / 1.15))
        self.btn_pan.toggled.connect(self.set_pan_mode)

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
        self.update()
        self.view_changed.emit()

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
            self.btn_zoom_in.show(); self.btn_zoom_out.show(); self.btn_pan.show()

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self.btn_zoom_in.hide(); self.btn_zoom_out.hide(); self.btn_pan.hide()

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
        x = self.width() - pad - w
        y = pad
        self.btn_zoom_in.setGeometry(QRect(x, y, w, h))
        self.btn_zoom_out.setGeometry(QRect(x, y + h + 8, w, h))
        self.btn_pan.setGeometry(QRect(x, y + (h + 8) * 2, w, h))

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
                tif_paths.sort(key=lambda p: (("_Bottom" in Path(p).stem), Path(p).name.lower()))

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
    status = Signal(str)
    finished = Signal(object)  # list of local output paths
    error = Signal(str)


class SSHCPSAMTask(QRunnable):
    def __init__(
        self,
        image_paths: List[str],
        local_out_dir: Path,
        host: str,
        port: int,
        username: str,
        password: str,
        use_existing_images_app: bool,
        cleanup_uploaded_inputs: bool,
        mux_socket: str,
    ):
        super().__init__()
        self.image_paths = image_paths
        self.local_out_dir = local_out_dir
        self.host = host
        self.port = int(port)
        self.username = username
        self.password = password
        self.use_existing_images_app = bool(use_existing_images_app)
        self.cleanup_uploaded_inputs = bool(cleanup_uploaded_inputs)
        self.mux_socket = mux_socket
        self.signals = SSHJobSignals()

    @staticmethod
    def _run_local(cmd: List[str]) -> Tuple[int, str, str]:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = p.stdout or ""
        err = p.stderr or ""
        return p.returncode, out, err

    def _ssh(self, remote_cmd: str) -> Tuple[int, str, str]:
        target = f"{self.username}@{self.host}"
        cmd = [
            "ssh",
            "-p", str(self.port),
            "-S", self.mux_socket,
            "-o", "ControlMaster=no",
            target,
            remote_cmd,
        ]
        return self._run_local(cmd)

    def _scp_upload(self, local_path: str, remote_dir: str) -> Tuple[int, str, str]:
        target = f"{self.username}@{self.host}:{remote_dir.rstrip('/')}/"
        cmd = [
            "scp",
            "-P", str(self.port),
            "-o", f"ControlPath={self.mux_socket}",
            "-o", "ControlMaster=no",
            local_path,
            target,
        ]
        return self._run_local(cmd)

    def _scp_download(self, remote_file: str, local_file: str) -> Tuple[int, str, str]:
        src = f"{self.username}@{self.host}:{remote_file}"
        cmd = [
            "scp",
            "-P", str(self.port),
            "-o", f"ControlPath={self.mux_socket}",
            "-o", "ControlMaster=no",
            src,
            local_file,
        ]
        return self._run_local(cmd)

    def run(self):
        try:
            mux_path = str(Path(self.mux_socket).expanduser())
            target = f"{self.username}@{self.host}"
            rc, out, err = self._run_local(
                ["ssh", "-p", str(self.port), "-S", mux_path, "-O", "check", target]
            )
            if rc != 0:
                raise RuntimeError(
                    "No active SSH control socket.\n\n"
                    "Run this once in your terminal (and complete Duo):\n"
                    f"  ssh -M -S {mux_path} -fNT {target}\n\n"
                    f"Then retry in app.\nDetails:\n{err or out}"
                )
            self.mux_socket = mux_path

            job_id = datetime.now().strftime("job_%Y%m%d_%H%M%S")
            remote_base = ORCD_CPSAM_ROOT
            remote_images_root = ORCD_IMAGES_APP_DIR
            remote_out_root = ORCD_OUTPUT_APP_DIR
            remote_jobs_root = f"{remote_base}/Jobs_App"
            remote_logs_root = f"{remote_base}/logs"

            self.local_out_dir.mkdir(parents=True, exist_ok=True)
            self.signals.status.emit(f"SSH: using control socket {self.mux_socket}")

            rc, _out, err = self._ssh(
                f"mkdir -p {shlex.quote(remote_images_root)} {shlex.quote(remote_out_root)} "
                f"{shlex.quote(remote_jobs_root)} {shlex.quote(remote_logs_root)}",
            )
            if rc != 0:
                raise RuntimeError(f"Failed to create remote folders.\n{err}")

            if self.use_existing_images_app:
                remote_infer_dir = remote_images_root
                self.signals.status.emit(f"SSH: using existing remote infer dir: {remote_infer_dir}")
            else:
                remote_infer_dir = f"{remote_images_root}/{job_id}"
                rc, _out, err = self._ssh(f"mkdir -p {shlex.quote(remote_infer_dir)}")
                if rc != 0:
                    raise RuntimeError(f"Failed to create remote input dir.\n{err}")
                total = len(self.image_paths)
                for i, p in enumerate(self.image_paths, start=1):
                    src = Path(p)
                    self.signals.status.emit(f"SSH: upload {i}/{total} {src.name}")
                    rc, _out, err = self._scp_upload(str(src), remote_infer_dir)
                    if rc != 0:
                        raise RuntimeError(f"Upload failed for {src.name}\n{err}")

            remote_out_dir = f"{remote_out_root}/{job_id}"
            rc, _out, err = self._ssh(f"mkdir -p {shlex.quote(remote_out_dir)}")
            if rc != 0:
                raise RuntimeError(f"Failed to create remote output dir.\n{err}")

            sbatch_script = f"""
#SBATCH -J CPSAMApp
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
cd {shlex.quote(remote_base)}
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA=/orcd/software/community/001/rocky8/miniforge/23.11.0-0/bin/conda
"$CONDA" run -n cellpose python -u CPSAMMainTODO.py \\
  --skip_train \\
  --infer_dir {shlex.quote(remote_infer_dir)} \\
  --out_dir {shlex.quote(remote_out_dir)} \\
  --use_gpu \\
  --pretrained_model ./models/cpsam_finetuned
"""
            remote_sbatch = f"{remote_jobs_root}/job_{job_id}.sbatch"
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sbatch", encoding="utf-8") as tf:
                tf.write(sbatch_script)
                local_sbatch = tf.name
            rc, _out, err = self._scp_upload(local_sbatch, remote_jobs_root)
            Path(local_sbatch).unlink(missing_ok=True)
            if rc != 0:
                raise RuntimeError(f"Failed to upload sbatch script.\n{err}")
            # Uploaded with same basename; move to desired filename.
            uploaded_tmp = f"{remote_jobs_root}/{Path(local_sbatch).name}"
            rc, _out, err = self._ssh(
                f"mv {shlex.quote(uploaded_tmp)} {shlex.quote(remote_sbatch)}"
            )
            if rc != 0:
                raise RuntimeError(f"Failed to place sbatch script remotely.\n{err}")

            self.signals.status.emit("SSH: submitting sbatch job ...")
            rc, out, err = self._ssh(
                f"cd {shlex.quote(remote_base)} && sbatch {shlex.quote(remote_sbatch)}",
            )
            if rc != 0:
                raise RuntimeError(f"sbatch submission failed.\n{err}")
            m = re.search(r"Submitted batch job\s+(\d+)", out)
            if not m:
                raise RuntimeError(f"Could not parse SLURM job id from output:\n{out}\n{err}")
            slurm_id = m.group(1)
            self.signals.status.emit(f"SSH: submitted SLURM job {slurm_id}. Waiting ...")

            terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
            final_state = "UNKNOWN"
            while True:
                rc1, qout, _qerr = self._ssh(
                    f"squeue -j {shlex.quote(slurm_id)} -h -o %T"
                )
                state = qout.strip().splitlines()[0].strip().upper() if (rc1 == 0 and qout.strip()) else ""
                if state:
                    self.signals.status.emit(f"SSH: SLURM {slurm_id} state = {state}")
                    if state in terminal:
                        final_state = state
                        break
                else:
                    rc2, aout, _aerr = self._ssh(
                        f"sacct -j {shlex.quote(slurm_id)} -n -o State | head -n 1",
                    )
                    if rc2 == 0 and aout.strip():
                        state = aout.strip().split()[0].upper()
                        self.signals.status.emit(f"SSH: SLURM {slurm_id} state = {state}")
                        if state in terminal:
                            final_state = state
                            break
                time.sleep(8)

            if final_state != "COMPLETED":
                raise RuntimeError(
                    f"SLURM job {slurm_id} ended with state: {final_state}. "
                    f"Check logs under {remote_logs_root}."
                )

            downloaded = []
            self.signals.status.emit("SSH: downloading outputs ...")
            rc, out, err = self._ssh(f"find {shlex.quote(remote_out_dir)} -maxdepth 1 -type f")
            if rc != 0:
                raise RuntimeError(f"Failed listing remote output files.\n{err}")
            for remote_fp in [ln.strip() for ln in out.splitlines() if ln.strip()]:
                name = Path(remote_fp).name
                low = name.lower()
                if not low.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".csv", ".txt", ".npy", ".npz")):
                    continue
                local_fp = self.local_out_dir / name
                rc, _out, err = self._scp_download(remote_fp, str(local_fp))
                if rc != 0:
                    raise RuntimeError(f"Failed downloading {name}\n{err}")
                downloaded.append(str(local_fp))

            if (not self.use_existing_images_app) and self.cleanup_uploaded_inputs:
                self._ssh(f"rm -rf {shlex.quote(remote_infer_dir)}")
                self.signals.status.emit("SSH: cleaned remote uploaded input folder.")
            self.signals.status.emit(f"SSH: finished. Job {slurm_id} COMPLETED.")
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
        self.theme_mode = "dark"
        qss_path = Path(__file__).resolve().parent / "style.qss"
        self.dark_app_qss = qss_path.read_text(encoding="utf-8") if qss_path.exists() else DEFAULT_DARK_APP_QSS
        self.selected_images: List[str] = []
        self.results: List[ResultRow] = []
        self.gt_counts_index: Dict[Tuple[str, str, int], int] = {}
        self.gt_counts_rows: List[Tuple[str, str, int, int]] = []
        self.cpsam_watch_dir: Optional[Path] = None
        self.cpsam_seen_done_markers: set[str] = set()
        self.cpsam_watch_timer = QTimer(self)
        self.cpsam_watch_timer.setInterval(3000)
        self.cpsam_watch_timer.timeout.connect(self._poll_cpsam_done_markers)

        self.active_tool: Optional[str] = None
        self.paint_color = "green"  # "blue", "pink", "red", "green", or "black"
        self.preview_contrast_percent = 100
        self.current_out_dir: Optional[Path] = None
        self.csv_filename = "results.csv"
        self.webui_proc: Optional[subprocess.Popen] = None
        self.web_session_id: Optional[str] = None
        self.web_saved_seen: set[str] = set()
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
        self._load_gt_counts_csv()
        self.ssh_status_lbl = QLabel("SSH idle.")

        # Logo bottom-left (CENTERED more in controls ribbon)
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self._load_logo()
        self.btn_theme_toggle = QPushButton("Switch To Light Mode")

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

        self.btn_tool_paint = QPushButton("Paint")
        self.btn_tool_select = QPushButton("Select")
        self.btn_tool_remove = QPushButton("Remove")
        for b in (self.btn_tool_paint, self.btn_tool_select, self.btn_tool_remove):
            b.setProperty("tool", True)
            b.setCheckable(True)

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

        self.btn_update_count = QPushButton("Update Count")
        self.btn_update_count.setProperty("success", True)

        self.btn_save = QPushButton("Save")
        self.btn_save.setProperty("primary", True)

        # SSH CPSAM controls
        self.ssh_host_edit = QLineEdit()
        self.ssh_host_edit.setText(ORCD_HOST_DEFAULT)
        self.ssh_host_edit.setPlaceholderText("orcd-login.mit.edu")
        self.ssh_user_edit = QLineEdit()
        self.ssh_user_edit.setPlaceholderText("juweiss")
        self.ssh_pass_edit = QLineEdit()
        self.ssh_pass_edit.setPlaceholderText("Optional: password / interactive prompt")
        self.ssh_pass_edit.setEchoMode(QLineEdit.Password)
        self.ssh_use_existing_cb = QCheckBox("Use existing remote Images_App (no upload)")
        self.ssh_use_existing_cb.setChecked(False)
        self.ssh_cleanup_cb = QCheckBox("Delete uploaded remote input folder after run")
        self.ssh_cleanup_cb.setChecked(True)
        self.ssh_help_lbl = QLabel(
            "Runs your CPSAM SLURM job on MIT ORCD.\n"
            f"Port is fixed to {ORCD_PORT_DEFAULT}. Remote dirs are fixed to:\n"
            f"{ORCD_IMAGES_APP_DIR} and {ORCD_OUTPUT_APP_DIR}\n"
            f"Before running, open mux session once:\n"
            f"ssh -M -S {ORCD_MUX_SOCKET} -fNT juweiss@{ORCD_HOST_DEFAULT}"
        )
        self.ssh_help_lbl.setWordWrap(True)
        self.ssh_help_lbl.setStyleSheet("color: #9aa0a6;")
        self.btn_run_ssh = QPushButton("Run CPSAM (SSH)")
        self.btn_run_ssh.setProperty("primary", True)
        self.btn_run_ssh.clicked.connect(self.on_run_ssh)

                # ---- Remove focus halos (prevents the “surrounding” outline bug)
        for b in (self.btn_tool_paint, self.btn_tool_select, self.btn_tool_remove,
                self.btn_pick_folder, self.btn_pick_files, self.btn_pick_out,
                self.btn_process_old, self.btn_process_cpsam, self.btn_web_ui_help,
                self.btn_clear_annotations, self.btn_update_count, self.btn_save, self.btn_run_ssh, self.btn_theme_toggle):
            b.setFocusPolicy(Qt.NoFocus)

        self.preview_mode_original.setFocusPolicy(Qt.NoFocus)
        self.preview_mode_masked.setFocusPolicy(Qt.NoFocus)

        # Layout
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        left = self._build_left_panel()
        right = self._build_right_panel_with_post()

        main_layout.addWidget(left, 0)
        main_layout.addWidget(right, 1)

        # Signals
        self.btn_pick_folder.clicked.connect(self.on_pick_folder)
        self.btn_pick_files.clicked.connect(self.on_pick_files)
        self.btn_pick_out.clicked.connect(self.on_pick_out_dir)
        self.btn_process_old.clicked.connect(self.on_process)
        self.btn_process_cpsam.clicked.connect(self.on_open_cpsam_dialog)
        self.btn_web_ui_help.clicked.connect(self.on_web_ui_help)
        self.btn_theme_toggle.clicked.connect(self.on_toggle_theme)

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
        self.btn_clear_annotations.clicked.connect(self.on_clear_annotations)
        self.btn_update_count.clicked.connect(self.on_update_count)
        self.btn_save.clicked.connect(self.on_save)
        # Undo shortcut (Cmd+Z on mac, Ctrl+Z elsewhere)
        self.undo_sc = QShortcut(QKeySequence.Undo, self)
        self.undo_sc.activated.connect(self.on_undo)
        # Drag throttling: avoid drawing too many circles per pixel
        self._last_drag_xy: Optional[Tuple[int, int]] = None
        self._apply_theme_visuals()

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
            if token is not None:
                label, dilu, x_suffix = token
                gt_count = self._lookup_gt_count(label, dilu, x_suffix)
                shown = str(gt_count) if gt_count is not None else "N/A"
                gt_text = f"GT {label} dilu{dilu}: {shown}"
                if gt_count is not None and algo_count is not None:
                    diff_text = f"Diff: {algo_count - gt_count:+d}"

            algo_text = f"Algorithm: {algo_count}" if algo_count is not None else None

            # current_count goes on the first overlay block only
            count_text: Optional[str] = None
            if idx == 0 and current_count is not None:
                count_text = f"Count: {current_count}"

            if gt_text is not None or algo_text is not None or diff_text is not None or count_text is not None:
                overlays.append(
                    {
                        "position": pos,
                        "gt_text": gt_text,
                        "algo_text": algo_text,
                        "diff_text": diff_text,
                        "count_text": count_text,
                    }
                )
        return overlays

    # ---------- Look & feel ----------
    def _apply_dark_palette(self):
        # This helps on macOS where some widgets ignore stylesheet on first paint.
        p = self.palette()
        p.setColor(QPalette.Window, QColor(0, 0, 0))
        p.setColor(QPalette.Base, QColor(0, 0, 0))
        p.setColor(QPalette.AlternateBase, QColor(5, 5, 5))
        p.setColor(QPalette.Text, QColor(235, 235, 235))
        p.setColor(QPalette.WindowText, QColor(235, 235, 235))
        p.setColor(QPalette.Button, QColor(10, 10, 10))
        p.setColor(QPalette.ButtonText, QColor(235, 235, 235))
        p.setColor(QPalette.Highlight, QColor(30, 30, 30))
        p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        self.setPalette(p)

    def _apply_light_palette(self):
        p = self.palette()
        p.setColor(QPalette.Window, QColor("#f4f2ee"))
        p.setColor(QPalette.Base, QColor("#fffdfa"))
        p.setColor(QPalette.AlternateBase, QColor("#fbf8f2"))
        p.setColor(QPalette.Text, QColor("#17191e"))
        p.setColor(QPalette.WindowText, QColor("#17191e"))
        p.setColor(QPalette.Button, QColor("#f2ede5"))
        p.setColor(QPalette.ButtonText, QColor("#17191e"))
        p.setColor(QPalette.Highlight, QColor("#dfe6ea"))
        p.setColor(QPalette.HighlightedText, QColor("#14171b"))
        self.setPalette(p)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._position_header_checkbox()

    def _cleanup_web_sessions(self):
        sessions_dir = Path(__file__).resolve().parent / "web_jobs" / "sessions"
        if not sessions_dir.exists():
            return
        for path in sessions_dir.iterdir():
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
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
            self._cleanup_web_sessions()
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
            if self.theme_mode == "light":
                candidates = [
                    base_dir / "Logo.png",
                    base_dir / "webui" / "static" / "logo.png",
                    base_dir / "logo.png",
                    base_dir / "Logo2.png",
                ]
            else:
                candidates = [
                    base_dir / "Logo2.png",
                    base_dir / "logo.png",
                    base_dir / "webui" / "static" / "logo.png",
                    base_dir / "Logo.jpg",
                ]

            for logo_path in candidates:
                if not logo_path.exists():
                    continue
                pm = QPixmap(str(logo_path))
                if pm.isNull():
                    continue
                pm = pm.scaledToWidth(260, Qt.SmoothTransformation)
                self.logo_label.setPixmap(pm)
                self.logo_label.setToolTip(logo_path.name)
                return
        except Exception:
            pass
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
    def _build_left_panel(self) -> QWidget:
        box = QGroupBox("Controls")
        box.setObjectName("ControlsPanel")
        box.setMinimumWidth(320)
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
        layout.addWidget(self.btn_theme_toggle, 0, Qt.AlignHCenter)

        return box

    def _build_ssh_panel(self) -> QWidget:
        box = QGroupBox("GPU (SSH)")
        lay = QGridLayout(box)
        lay.addWidget(QLabel("Host"), 0, 0)
        lay.addWidget(self.ssh_host_edit, 0, 1)
        lay.addWidget(QLabel("User"), 1, 0)
        lay.addWidget(self.ssh_user_edit, 1, 1)
        lay.addWidget(QLabel("Password"), 2, 0)
        lay.addWidget(self.ssh_pass_edit, 2, 1)
        lay.addWidget(self.ssh_use_existing_cb, 3, 0, 1, 2)
        lay.addWidget(self.ssh_cleanup_cb, 4, 0, 1, 2)
        lay.addWidget(self.ssh_help_lbl, 5, 0, 1, 2)
        lay.addWidget(self.btn_run_ssh, 6, 0, 1, 2)
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
        color_layout.addWidget(QLabel("Color:"))
        color_layout.addWidget(self.color_combo)
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
        layout.addWidget(self.count_info_lbl)
        layout.addSpacing(6)
        layout.addWidget(self.btn_clear_annotations)
        layout.addWidget(self.btn_update_count)
        layout.addWidget(self.btn_save)

        layout.addStretch(1)
        return self.post_box

    def _force_table_dark(self):
        # Keep method name for compatibility; now applies current theme table style.
        pal = self.table.palette()
        if self.theme_mode == "light":
            pal.setColor(QPalette.Base, QColor("#f6f8fa"))
            pal.setColor(QPalette.Window, QColor("#f6f8fa"))
            pal.setColor(QPalette.AlternateBase, QColor("#fbf8f2"))
            pal.setColor(QPalette.Text, QColor("#17191e"))
            pal.setColor(QPalette.WindowText, QColor("#17191e"))
            pal.setColor(QPalette.Button, QColor("#f2ede5"))
            pal.setColor(QPalette.ButtonText, QColor("#17191e"))
            pal.setColor(QPalette.Highlight, QColor("#f6f8fa"))
            pal.setColor(QPalette.HighlightedText, QColor("#17191e"))
            style = """
                QTableWidget, QTableView, QTableWidget::viewport, QTableView::viewport {
                    background: #f6f8fa;
                    color: #17191e;
                    border: 1px solid #d9e5f0;
                    gridline-color: #d9e5f0;
                }
                QTableWidget::item {
                    background: #f6f8fa;
                    color: #17191e;
                    border-bottom: 1px solid #d9e5f0;
                }
                QTableWidget::item:selected { background: #dfe6ea; color: #14171b; }
                QHeaderView::section {
                    background: #d9e5f0;
                    color: #4c4338;
                    border: 1px solid #d9e5f0;
                }
            """
        else:
            pal.setColor(QPalette.Base, QColor(0, 0, 0))
            pal.setColor(QPalette.Window, QColor(0, 0, 0))
            pal.setColor(QPalette.AlternateBase, QColor(5, 5, 5))
            pal.setColor(QPalette.Text, QColor(235, 235, 235))
            pal.setColor(QPalette.WindowText, QColor(235, 235, 235))
            pal.setColor(QPalette.Button, QColor(0, 0, 0))
            pal.setColor(QPalette.ButtonText, QColor(235, 235, 235))
            pal.setColor(QPalette.Highlight, QColor(30, 30, 30))
            pal.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            style = """
                QTableWidget, QTableView, QTableWidget::viewport, QTableView::viewport {
                    background: #000000;
                    color: #eaeaea;
                    border: 1px solid #111111;
                    gridline-color: #111111;
                }
                QTableWidget::item {
                    background: #000000;
                    color: #eaeaea;
                    border-bottom: 1px solid #111111;
                }
                QTableWidget::item:selected { background: #1a1a1a; }
            """

        self.table.setPalette(pal)
        self.table.viewport().setPalette(pal)

        self.table.setStyleSheet(style)

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
        if self.theme_mode == "light":
            pal.setColor(QPalette.Base, QColor("#fffdfa"))
            pal.setColor(QPalette.Window, QColor("#fffdfa"))
            pal.setColor(QPalette.Text, QColor("#17191e"))
            pal.setColor(QPalette.Highlight, QColor("#dfe6ea"))
            pal.setColor(QPalette.HighlightedText, QColor("#14171b"))
            view.setStyleSheet(
                """
                QListView, QListView::viewport {
                    background: #fffdfa;
                    color: #17191e;
                    border: none;
                    outline: 0;
                    margin: 0;
                    padding: 0;
                }
                QListView::item {
                    background: #fffdfa;
                    color: #17191e;
                    min-height: 22px;
                }
                QListView::item:selected {
                    background: #fffdfa;
                    color: #17191e;
                }
                """
            )
        else:
            pal.setColor(QPalette.Base, QColor("#0b0b0b"))
            pal.setColor(QPalette.Window, QColor("#0b0b0b"))
            pal.setColor(QPalette.Text, QColor("#f0f0f0"))
            pal.setColor(QPalette.Highlight, QColor("#0b0b0b"))
            pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
            view.setStyleSheet(
                """
                QListView, QListView::viewport {
                    background: #0b0b0b;
                    color: #f0f0f0;
                    border: none;
                    outline: 0;
                    margin: 0;
                    padding: 0;
                }
                QListView::item {
                    background: #0b0b0b;
                    color: #f0f0f0;
                    min-height: 22px;
                }
                QListView::item:selected {
                    background: #0b0b0b;
                    color: #ffffff;
                }
                """
            )
        view.setPalette(pal)
        self._refresh_widget_style(view)

    def on_toggle_theme(self):
        if self.theme_mode == "dark":
            self.theme_mode = "light"
        else:
            self.theme_mode = "dark"
        self._apply_theme_visuals()

    def _apply_theme_visuals(self):
        app = QApplication.instance()
        if self.theme_mode == "light":
            self._apply_light_palette()
            if app is not None:
                app.setStyleSheet(LIGHT_APP_QSS)
            self.btn_web_ui_help.setProperty("primary", True)
            self.btn_theme_toggle.setText("Switch To Dark Mode")
            self.lbl_or.setStyleSheet("color: #6e6253; padding: 2px 0;")
            self.ssh_help_lbl.setStyleSheet("color: #6e6253;")
        else:
            self._apply_dark_palette()
            if app is not None:
                app.setStyleSheet(self.dark_app_qss)
            self.btn_web_ui_help.setProperty("primary", False)
            self.btn_theme_toggle.setText("Switch To Light Mode")
            self.lbl_or.setStyleSheet("color: #777; padding: 2px 0;")
            self.ssh_help_lbl.setStyleSheet("color: #9aa0a6;")
        self._refresh_widget_style(self.btn_web_ui_help)
        self._load_logo()
        self._force_table_dark()
        self._apply_combo_popup_theme()
        row = self._current_row()
        if row is not None and hasattr(self, "count_info_lbl"):
            self._update_count_info_lbl(row)

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
        else:
            self.lbl_selected.setText(f"{len(paths)} image(s) selected.\nFirst: {Path(paths[0]).name}")
            self._load_gt_counts_csv(Path(paths[0]).parent)

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

        task = ProcessImagesTask(
            image_paths=paths_to_process,
            out_dir=out_dir,
        )
        task.signals.progress.connect(self.on_progress)
        task.signals.message.connect(self.on_message)
        task.signals.result.connect(self.on_results)
        task.signals.error.connect(self.on_error)
        task.signals.row_ready.connect(self.on_row_ready)
        self.thread_pool.start(task)

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
            pp.paint_mask_green = algo_mask.copy().astype(np.uint8)
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

        # 4) larger selection counts (numbers)
        for lab in pp.labels:
            x, y, n = int(lab.x), int(lab.y), int(lab.n)
            cv2.circle(out, (x, y), 12, (255, 255, 0), -1)
            cv2.putText(
                out,
                str(n),
                (x + 20, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.1,
                (255, 255, 0),
                6,
                cv2.LINE_AA,
            )

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
        # Green colonies minus yellow watershed boundaries
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
            return
        is_dark = getattr(self, "theme_mode", "dark") != "light"
        if is_dark:
            bg = "#000000"
            gt_col = "#ffff00"
            algo_col = "#00ee00"
            diff_col = "#ff5555"
            count_col = "#00c8ff"
        else:
            bg = "#ffffff"
            gt_col = "#8a7000"
            algo_col = "#006600"
            diff_col = "#cc0000"
            count_col = "#0060aa"
        parts = []
        for ov in overlays:
            if ov.get("gt_text"):
                parts.append(f'<span style="color:{gt_col};">{ov["gt_text"]}</span>')
            if ov.get("algo_text"):
                parts.append(f'<span style="color:{algo_col};">{ov["algo_text"]}</span>')
            if ov.get("diff_text"):
                parts.append(f'<span style="color:{diff_col};">{ov["diff_text"]}</span>')
            if ov.get("count_text"):
                parts.append(f'<span style="color:{count_col};">{ov["count_text"]}</span>')
        html = "<br>".join(parts)
        self.count_info_lbl.setText(
            f'<div style="background:{bg};padding:6px;border-radius:4px;'
            f'font-size:13pt;font-weight:bold;">{html}</div>'
        )

    def on_table_select(self):
        row = self._current_row()
        if row is None:
            return
        want_masked = self.preview_mode_masked.isChecked()
        self._update_count_info_lbl(row)
        try:
            original, masked = self._get_base_images(row)
            if want_masked:
                self._ensure_masked_seed(row, masked.shape[0], masked.shape[1])
                composed = self._compose_with_annotations(masked, row.pp_masked)
            else:
                composed = self._compose_with_annotations(original, row.pp_original)
            composed = self._apply_preview_contrast(composed)
            self.preview_label.set_rgb(composed)
        except Exception:
            self.preview_label.setText("Preview render failed.")
            self.preview_label.setPixmap(QPixmap())

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

        self.on_table_select()

    def on_preview_clicked_once(self, x: int, y: int):
        self._last_drag_xy = None
        self._last_paint_xy = None
        self._apply_tool_at(x, y, is_drag=False)

    def on_preview_drag(self, x: int, y: int):
        # 3) enable painting/removing while mouse is held down and moving
        if self.active_tool in ("paint", "remove"):
            self._apply_tool_at(x, y, is_drag=True)

    def on_preview_drag_end(self):
        self._last_drag_xy = None
        self._last_paint_xy = None

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
        """
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

    def _build_cpsam_terminal_script(self, host: str, user: str, remote_root: str, image_paths: List[str], local_out_dir: Path) -> str:
        # One terminal script: login/2FA once (mux), upload, submit, poll, download.
        image_list = " \\\n  ".join(shlex.quote(str(Path(p))) for p in image_paths)
        return f"""set +H
set -euo pipefail

HOST={shlex.quote(host)}
USER={shlex.quote(user)}
REMOTE_ROOT={shlex.quote(remote_root)}
JOB_ID="app_$(date +%Y%m%d_%H%M%S)"
REMOTE_IN="$REMOTE_ROOT/Images_App/$JOB_ID"
REMOTE_OUT="$REMOTE_ROOT/Output_App/$JOB_ID"
REMOTE_LOGS="$REMOTE_ROOT/logs"
REMOTE_JOB="$REMOTE_ROOT/Jobs_App/$JOB_ID.sbatch"
LOCAL_OUT={shlex.quote(str(local_out_dir))}
MARKER="$LOCAL_OUT/.cpsam_done_$JOB_ID"
MUX="$HOME/.ssh/orcd_mux"
CONDA="/orcd/software/community/001/rocky8/miniforge/23.11.0-0/bin/conda"

mkdir -p "$LOCAL_OUT"
mkdir -p "$HOME/.ssh"

echo "[0/5] Login + Duo once (create/reuse SSH mux socket)..."
ssh -M -S "$MUX" -fNT "$USER@$HOST"
ssh -S "$MUX" "$USER@$HOST" "echo connected"

echo "[1/5] Ensure remote folders exist..."
ssh -S "$MUX" "$USER@$HOST" "mkdir -p '$REMOTE_IN' '$REMOTE_OUT' '$REMOTE_LOGS' '$REMOTE_ROOT/Jobs_App'"

echo "[2/5] Upload selected images to $REMOTE_IN ..."
for f in \\
  {image_list}
do
  scp -o ControlPath="$MUX" "$f" "$USER@$HOST:$REMOTE_IN/"
done

echo "[3/5] Create and submit sbatch ..."
ssh -S "$MUX" "$USER@$HOST" "(
printf '#\\041/bin/bash\\n'
cat <<'SBATCH'
#SBATCH -J CPSAMApp
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p mit_normal_gpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail
cd '$REMOTE_ROOT'
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

\"$CONDA\" run -n cellpose python -u CPSAMMainTODO.py \\
  --skip_train \\
  --infer_dir '$REMOTE_IN' \\
  --out_dir '$REMOTE_OUT' \\
  --use_gpu \\
  --pretrained_model ./models/cpsam_finetuned
SBATCH
) > '$REMOTE_JOB'"
JOB_NUM=$(ssh -S "$MUX" "$USER@$HOST" "cd '$REMOTE_ROOT' && sbatch '$REMOTE_JOB'" | awk '{{print $4}}')
echo "Submitted job: $JOB_NUM"
echo "Queue:"
ssh -S "$MUX" "$USER@$HOST" "squeue -j $JOB_NUM -o '%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R'"

echo "[4/5] Wait for completion ..."
while ssh -S "$MUX" "$USER@$HOST" "squeue -j $JOB_NUM -h | grep -q ."; do
  ssh -S "$MUX" "$USER@$HOST" "squeue -j $JOB_NUM -h -o 'state=%T elapsed=%M nodes=%D reason=%R' || true"
  sleep 10
done
echo "Queue done. Final accounting:"
ssh -S "$MUX" "$USER@$HOST" "sacct -j $JOB_NUM -n -o JobID,State,Elapsed,ExitCode | head -n 5 || true"

echo
echo "[5/5] Download outputs ..."
rsync -av --progress -e "ssh -S $MUX" "$USER@$HOST:$REMOTE_OUT/" "$LOCAL_OUT/"
touch "$MARKER"

echo
echo "Done. In app use: Select Folder -> $LOCAL_OUT"
echo "Done marker: $MARKER"
echo "Optional close mux: ssh -S $MUX -O exit $USER@$HOST"
"""

    def on_open_cpsam_dialog(self):
        image_paths = self._get_ssh_input_paths()
        if not image_paths:
            QMessageBox.warning(self, "No input", "Select/check images first.")
            return

        out_base = self._csv_path()
        if out_base is None:
            local_out_dir = Path.cwd() / "cpsam_terminal_out"
        else:
            local_out_dir = out_base.parent / "cpsam_terminal_out"
        local_out_dir.mkdir(parents=True, exist_ok=True)
        self.cpsam_watch_dir = local_out_dir
        self.cpsam_watch_timer.start()

        dlg = QDialog(self)
        dlg.setWindowTitle("CPSAM via SSH (Terminal Workflow)")
        dlg.resize(900, 620)
        lay = QVBoxLayout(dlg)

        form = QGridLayout()
        host_edit = QLineEdit(ORCD_HOST_DEFAULT)
        user_edit = QLineEdit("juweiss")
        root_edit = QLineEdit(ORCD_CPSAM_ROOT)
        form.addWidget(QLabel("Host"), 0, 0)
        form.addWidget(host_edit, 0, 1)
        form.addWidget(QLabel("User"), 1, 0)
        form.addWidget(user_edit, 1, 1)
        form.addWidget(QLabel("Remote root"), 2, 0)
        form.addWidget(root_edit, 2, 1)
        lay.addLayout(form)

        info = QLabel(
            "This workflow uses your regular terminal SSH (key/passphrase + Duo).\n"
            "1) Upload selected images to Images_App\n"
            "2) Submit job.sbatch on ORCD\n"
            "3) Download outputs back locally"
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #9aa0a6;")
        lay.addWidget(info)

        script_box = QPlainTextEdit()
        script_box.setReadOnly(True)
        lay.addWidget(script_box, 1)

        btn_row = QWidget()
        btn_lay = QHBoxLayout(btn_row)
        btn_lay.setContentsMargins(0, 0, 0, 0)
        btn_refresh = QPushButton("Refresh Script")
        btn_copy = QPushButton("Copy Script")
        btn_close = QPushButton("Close")
        btn_lay.addWidget(btn_refresh)
        btn_lay.addWidget(btn_copy)
        btn_lay.addStretch(1)
        btn_lay.addWidget(btn_close)
        lay.addWidget(btn_row)

        def refresh_script():
            script = self._build_cpsam_terminal_script(
                host=host_edit.text().strip() or ORCD_HOST_DEFAULT,
                user=user_edit.text().strip() or "juweiss",
                remote_root=root_edit.text().strip() or ORCD_CPSAM_ROOT,
                image_paths=image_paths,
                local_out_dir=local_out_dir,
            )
            script_box.setPlainText(script)

        btn_refresh.clicked.connect(refresh_script)
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(script_box.toPlainText()))
        btn_copy.clicked.connect(lambda: self.status_lbl.setText("CPSAM terminal script copied to clipboard."))
        btn_close.clicked.connect(dlg.accept)

        refresh_script()
        dlg.exec()

    def _poll_cpsam_done_markers(self):
        if self.cpsam_watch_dir is None or not self.cpsam_watch_dir.exists():
            return
        markers = sorted(self.cpsam_watch_dir.glob(".cpsam_done_*"))
        new_markers = [str(m) for m in markers if str(m) not in self.cpsam_seen_done_markers]
        if not new_markers:
            return

        self.cpsam_seen_done_markers.update(new_markers)

        paths = []
        for p in sorted(self.cpsam_watch_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                paths.append(str(p))
        if not paths:
            return

        loaded = self._attach_cpsam_outputs_to_originals(paths)
        if loaded <= 0:
            self.status_lbl.setText("CPSAM finished (no new matched outputs).")
            return
        self.status_lbl.setText(f"CPSAM results ready: matched {loaded} original image(s).")
        QMessageBox.information(
            self,
            "CPSAM finished",
            f"New CPSAM results detected from:\n{self.cpsam_watch_dir}\n\n"
            f"Matched to {loaded} original image row(s).",
        )

    def _attach_cpsam_outputs_to_originals(self, output_paths: List[str]) -> int:
        if not self.results:
            return 0

        # Build output index by stem for quick matching.
        outs = [Path(p) for p in output_paths]
        out_stems = {p: p.stem.lower() for p in outs}

        matched_rows = 0
        first_matched_idx: Optional[int] = None
        for i, row in enumerate(self.results):
            src_stem = Path(row.source_image).stem.lower()
            matched = []
            for p in outs:
                s = out_stems[p]
                if s == src_stem or s.startswith(src_stem + "_") or s.startswith(src_stem):
                    matched.append(str(p))

            if not matched:
                continue

            matched_rows += 1
            if first_matched_idx is None:
                first_matched_idx = i
            matched_has_non_annotated = any("_annotated" not in Path(p).stem.lower() for p in matched)
            matched.sort(
                key=lambda p: (
                    0 if "_annotated" in Path(p).stem.lower() else 1,
                    -(Path(p).stat().st_mtime if Path(p).exists() else 0.0),
                    Path(p).name.lower(),
                )
            )
            if matched_has_non_annotated:
                row.prefer_annotated = False
                non_ann = [p for p in matched if "_annotated" not in Path(p).stem.lower()]
                newest = max(
                    non_ann,
                    key=lambda p: Path(p).stat().st_mtime if Path(p).exists() else 0.0,
                )
                # Replace any older/non-CPSAM mask with the newest non-annotated result only.
                row.tif_paths = [newest]
            else:
                row.prefer_annotated = True
                existing = [x for x in row.tif_paths if x not in matched]
                row.tif_paths = matched + existing
            row._masked_cache = None
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
        """Write current in-memory mask to a temp PNG and return its path, or None."""
        pp = r.pp_masked
        if kind == "green":
            mask = pp.paint_mask_green
        else:
            mask = pp.paint_mask_yellow
        if mask is None or not np.any(mask):
            return None
        try:
            suffix = f"_web_{kind}.png"
            stem = Path(r.source_image).stem
            tmp_dir = Path(r.source_image).parent
            tmp_path = tmp_dir / f"{stem}{suffix}"
            cv2.imwrite(str(tmp_path), mask)
            return str(tmp_path)
        except Exception:
            return None

    def on_web_ui_help(self):
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
                    # Build a fake tif by stitching yellow into an overlay RGB PNG
                    try:
                        h, w = r.pp_masked.paint_mask_yellow.shape[:2]
                        overlay_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                        ym = r.pp_masked.paint_mask_yellow > 0
                        overlay_rgb[ym] = [255, 255, 0]
                        # also paint green so the base image is still useful
                        gm = r.pp_masked.paint_mask_green > 0
                        overlay_rgb[gm] = [0, 255, 0]
                        tmp_overlay = Path(r.source_image).parent / f"{Path(r.source_image).stem}_web_overlay.png"
                        cv2.imwrite(str(tmp_overlay), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                        base_tifs = [str(tmp_overlay)]
                    except Exception:
                        pass

            rows.append(
                {
                    "image": Path(src).name,
                    "source_path": src,
                    "tif_paths": base_tifs,
                    "expt_paths": base_expts,
                    "post_mask_paths": base_post_masks,
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
        Removes any pixels that were erased in the webapp (i.e. no longer green/yellow).
        Returns True if something changed.
        """
        try:
            bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
            if bgr is None:
                return False
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.int32)
            h, w = rgb.shape[:2]

            self._ensure_masked_seed(row, h, w)
            pp = row.pp_masked
            pp.ensure_shape(h, w)

            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

            # Green: high G, low R, low B
            green_mask = (g > 180) & (r < 80) & (b < 80)
            # Yellow: high R+G, low B
            yellow_mask = (r > 180) & (g > 180) & (b < 80)

            pp.paint_mask_green = green_mask.astype(np.uint8) * 255
            pp.paint_mask_yellow = yellow_mask.astype(np.uint8) * 255
            row.user_modified_mask = True
            row._masked_cache = None
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

    def on_run_ssh(self):
        host = self.ssh_host_edit.text().strip()
        user = self.ssh_user_edit.text().strip()
        password = self.ssh_pass_edit.text()
        port = ORCD_PORT_DEFAULT

        image_paths = self._get_ssh_input_paths()
        if not image_paths:
            QMessageBox.warning(self, "No input", "Select images first.")
            return
        if not host or not user:
            QMessageBox.warning(self, "Missing SSH fields", "Fill Host and User.")
            return

        out_base = self._csv_path()
        if out_base is None:
            local_out_dir = Path.cwd() / "cpsam_ssh_out"
        else:
            local_out_dir = out_base.parent / "cpsam_ssh_out"

        self.btn_run_ssh.setEnabled(False)
        self.ssh_status_lbl.setText("SSH: starting...")
        self.status_lbl.setText("Starting remote CPSAM...")

        task = SSHCPSAMTask(
            image_paths=image_paths,
            local_out_dir=local_out_dir,
            host=host,
            port=port,
            username=user,
            password=password,
            use_existing_images_app=self.ssh_use_existing_cb.isChecked(),
            cleanup_uploaded_inputs=self.ssh_cleanup_cb.isChecked(),
            mux_socket=ORCD_MUX_SOCKET,
        )
        task.signals.status.connect(self.on_ssh_status)
        task.signals.finished.connect(self.on_ssh_finished)
        task.signals.error.connect(self.on_ssh_error)
        self.thread_pool.start(task)

    def on_ssh_status(self, msg: str):
        self.ssh_status_lbl.setText(msg)
        self.status_lbl.setText(msg)

    def on_ssh_finished(self, outputs: object):
        self.btn_run_ssh.setEnabled(True)
        outs = list(outputs) if isinstance(outputs, list) else []
        self.ssh_status_lbl.setText(f"SSH done: {len(outs)} output file(s).")
        self.status_lbl.setText(self.ssh_status_lbl.text())
        if outs:
            QMessageBox.information(
                self,
                "SSH CPSAM finished",
                f"Completed.\nSaved {len(outs)} file(s) to:\n{Path(outs[0]).parent}",
            )
        else:
            QMessageBox.information(self, "SSH CPSAM finished", "Completed (no downloaded outputs).")

    def on_ssh_error(self, tb: str):
        self.btn_run_ssh.setEnabled(True)
        self.ssh_status_lbl.setText("SSH error.")
        self.status_lbl.setText("SSH error.")
        QMessageBox.critical(self, "SSH CPSAM error", tb)


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
