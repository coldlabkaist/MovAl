"""color_config.py – tiny JSON-backed colour store"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Optional
import colorsys
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QIcon, QPainter, QLinearGradient
from PyQt6.QtWidgets import (
    QWidget, QDialog, QLabel, QListWidget, QListWidgetItem, QPushButton,
    QColorDialog, QHBoxLayout, QVBoxLayout, QComboBox, QSlider, QMessageBox, QGridLayout
)

# ───────── 팔레트 정의 (10 hues × 4 value) ──────────
_HUES = [           # 컬럼 순서
    "gray", "pink", "red", "orange", "yellow",
    "lime", "green", "mint", "sky", "blue", "purple"
]

# 행(row) = 밝기(light → dark) 8단계
_PALETTE_MATRIX: list[list[str]] = [
    ["#FFFFFF", "#FDE7EF", "#FFDADA", "#FFE8CC", "#FFFBCF",
     "#F7FAE5", "#DEF5E1", "#E6FBF8", "#E7FAFD", "#E3F2FD", "#F1E6FA"],
    ["#F2F2F2", "#FACBDA", "#FFBABA", "#FFD7A8", "#FFF59D",
     "#E9F3D1", "#C8EACC", "#C9F8F3", "#CCF5FB", "#C9E4FF", "#E4D1F9"],
    ["#D9D9D9", "#F8A5C4", "#FF8A80", "#FFC07E", "#FFE863",
     "#D4E8A7", "#A4DEB0", "#A3F2EB", "#A4EAF8", "#9FC6FF", "#CEA8F3"],
    ["#BFBFBF", "#F06292", "#F44336", "#FFA726", "#FFEB3B",
     "#BFE17C", "#81C784", "#64FFDA", "#80DEEA", "#64B5F6", "#B388FF"],
    ["#9E9E9E", "#E91E63", "#D32F2F", "#F57C00", "#FBC02D",
     "#9CCC65", "#4CAF50", "#1DE9B6", "#26C6DA", "#2196F3", "#9C27B0"],
    ["#7A7A7A", "#C2185B", "#B71C1C", "#E65100", "#F57F17",
     "#7CB342", "#388E3C", "#00BFA5", "#00ACC1", "#1976D2", "#7B1FA2"],
    ["#545454", "#880E4F", "#8B0000", "#BF360C", "#C77900",
     "#558B2F", "#1B5E20", "#00796B", "#00838F", "#0D47A1", "#4A148C"],
    ["#2E2E2E", "#560027", "#4A0000", "#3E0B00", "#5D3B00",
     "#2D3C14", "#0D370E", "#004D40", "#005662", "#002171", "#311B92"],
]

_BASE_HEX: list[str] = [hx for row in _PALETTE_MATRIX for hx in row]
_COLS, _ROWS = len(_HUES), len(_PALETTE_MATRIX)   # 11, 8
_GRAY_HEX   = [row[0]   for row in _PALETTE_MATRIX]          # 8 단계 회색
_COLOR_HEX  = [hx for row in _PALETTE_MATRIX for hx in row[1:]]  # 회색을 제외한 전부

def _gray_color(idx: int) -> QColor:
    return QColor(_GRAY_HEX[idx % len(_GRAY_HEX)])

def _color_palette(idx: int) -> QColor:
    return QColor(_COLOR_HEX[idx % len(_COLOR_HEX)])

class ColorManager:
    """Lightweight global colour table (key → #RRGGBB)."""

    _colors: Dict[str, str] = {}

    # ───────────────── persist ─────────────────
    @classmethod
    def load(cls, path: Path | str | None = None) -> None:
        p = Path(path) if path else _CFG_PATH
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                cls._colors = json.load(f)
        else:
            cls._colors = {}

    @classmethod
    def save(cls, path: Path | str | None = None) -> None:
        p = Path(path) if path else _CFG_PATH
        with p.open("w", encoding="utf-8") as f:
            json.dump(cls._colors, f, ensure_ascii=False, indent=2)

    # ───────────────── helpers ─────────────────
    @classmethod
    def get(cls, key: str) -> Optional[QColor]:
        return QColor(cls._colors[key]) if key in cls._colors else None

    @classmethod
    def set(cls, key: str, colour: QColor) -> None:
        cls._colors[key] = colour.name(QColor.NameFormat.HexArgb)

    # ───────────────── file I/O dialogs ─────────────────
    @classmethod
    def save_as(cls, parent: QWidget) -> None:
        fn, _ = QFileDialog.getSaveFileName(parent, "Export colour config", "", "JSON (*.json)")
        if fn:
            cls.save(fn)

    @classmethod
    def load_from(cls, parent: QWidget) -> bool:
        fn, _ = QFileDialog.getOpenFileName(parent, "Import colour config", "", "JSON (*.json)")
        if fn:
            cls.load(fn)
            return True
        return False

    # ───────────────────────── 자동 팔레트 ─────────────────────────

    @classmethod
    def track_color(cls, track: str, idx: int) -> QColor:
        """
        • 사용자 지정(track:…) 값이 있으면 그대로  
        • 없으면 고정 팔레트에서 ‘행-우선’으로 순환
        """
        return cls.get(f"track:{(track)}") or _gray_color(idx*3%8)

    @classmethod
    def kp_color(cls, kp: str) -> QColor:
        if (c := cls.get(f"kp:{kp}")):
            return c
        from data_loader import DataLoader
        base_idx = DataLoader.kp_order.index(kp)
        return _color_palette((21+ base_idx*2 + base_idx//9*27)%72)

class ColorDialog(QDialog):
    """
    • 좌측 콤보 박스: ① Track ② Key-point  
    • 하단: 색상 피커, 투명도 슬라이더, 적용/초기화/Import/Export/닫기
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Configuration")
        self.resize(300, 200)

        # ---------- 위젯 ▲ ----------
        self.cmb_target = QComboBox()
        self.cmb_item   = QComboBox()
        self.lbl_color  = QLabel()
        self.lbl_color.setFixedSize(80, 24)
        self.lbl_color.setFrameShape(QLabel.Shape.Box)

        self.btn_reset  = QPushButton("Reset")
        self.sld_alpha  = QSlider(Qt.Orientation.Horizontal)
        self.sld_alpha.setRange(0, 255)
        self.sld_alpha.setValue(255)

        self.btn_imp = QPushButton("Import JSON …")
        self.btn_exp = QPushButton("Export JSON …")
        self.btn_close = QPushButton("Close")

        self.picker = PalettePicker()
        self.picker.colourSelected.connect(self._apply_live_color)

        # ---------- 배치 ▼ ----------
        top = QHBoxLayout()
        top.addWidget(self.cmb_target); top.addWidget(self.cmb_item)

        row = QHBoxLayout()
        row.addWidget(self.lbl_color)

        lay = QVBoxLayout(self)
        lay.addLayout(top); lay.addLayout(row)
        lay.addWidget(QLabel("Alpha (투명도):"))
        lay.addWidget(self.sld_alpha)
        lay.addStretch()
        foot = QHBoxLayout()
        foot.addWidget(self.btn_imp); foot.addWidget(self.btn_exp)
        foot.addStretch(); foot.addWidget(self.btn_close)
        lay.addLayout(foot)
        
        lay.insertWidget(1, self.picker)      # 기존 콤보 아래 삽입

        # ---------- 데이터 준비 ----------
        from data_loader import DataLoader    # 순환 임포트 방지용 지역 import

        # 1) DataFrame 안전 획득
        df = getattr(DataLoader, "data", None)

        # 2) track / kp 목록 준비
        self._tracks = list(df["track"].unique()) if df is not None else []
        self._kps    = getattr(DataLoader, "kp_order", [])

        # 콤보 기본 항목 준비
        self.cmb_target.addItems(["Track", "Key-point"])
        self.cmb_target.currentIndexChanged.connect(self._rebuild_item_box)
        self.cmb_item.currentIndexChanged.connect(self._update_preview)
        self._rebuild_item_box()

        # ---------- 시그널 ----------
        self.sld_alpha.valueChanged.connect(self._update_alpha)
        self.btn_imp.clicked.connect(self._import)
        self.btn_exp.clicked.connect(self._export)
        self.btn_close.clicked.connect(self.accept)

    # ------------------------------------------------ helpers
    def _key(self) -> str:
        typ = self.cmb_target.currentText()
        val = self.cmb_item.currentText()
        return ("track:" if typ == "Track" else "kp:") + val

    def _rebuild_item_box(self):
        self.cmb_item.blockSignals(True)
        self.cmb_item.clear()
        if self.cmb_target.currentText() == "Track":
            self.cmb_item.addItems(self._tracks or ["DefaultTrack"])
        else:
            self.cmb_item.addItems(self._kps or ["nose"])
        self.cmb_item.blockSignals(False)
        self.cmb_item.setCurrentIndex(0)
        self._update_preview()

    def _update_preview(self):
        col = ColorManager.get(self._key())
        if not col:
            # 기본색이면 새로 계산(가시적 표시)
            if self.cmb_target.currentText() == "Track":
                idx = self.cmb_item.currentIndex()
                col = ColorManager.track_color(self.cmb_item.currentText(), idx)
            else:
                col = ColorManager.kp_color(self.cmb_item.currentText())
        alpha = self.sld_alpha.value()
        col.setAlpha(alpha)
        pm = QPixmap(80, 24); pm.fill(col)
        self.lbl_color.setPixmap(pm)

    # ------------------------------------------------ actions
    def _choose_color(self):
        base = ColorManager.get(self._key()) or QColor("white")
        self.colorDlg.setCurrentColor(base)
        self.colorDlg.open()          # 비차단(Non-modal) 모드로 띄움

    def _apply_live_color(self, qcolor: QColor):
        qcolor.setAlpha(self.sld_alpha.value())       # 슬라이더 값 반영
        ColorManager.set(self._key(), qcolor)
        self._update_preview()
        self.parent().video_label.update()            # 즉시 재-페인트
        self.parent().update_keypoint_list()

    def _update_alpha(self):
        # 현재 선택된(또는 기본) 색을 불러와서 알파만 갱신
        col = ColorManager.get(self._key()) or self.lbl_color.palette().window().color()
        col.setAlpha(self.sld_alpha.value())
        self._apply_live_color(col)

    def _import(self):
        if ColorManager.load_from(self):
            self._update_preview()

    def _export(self):
        ColorManager.save_as(self)

class PalettePicker(QWidget):
    """
    ▣ 고정 팔레트형 색상 선택기
      • column  ▶ 동일 Hue 계열
      • row     ▶ 동일 밝기(Value) 계열
      • click   ▶ colourSelected(QColor) 시그널 발행
    """
    colourSelected = pyqtSignal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        g = QGridLayout(self)
        g.setSpacing(4)
        g.setContentsMargins(0, 0, 0, 0)

        for idx, hx in enumerate(_BASE_HEX):      # ← 새 상수 사용
            row, col = divmod(idx, _COLS)
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setStyleSheet(f"background:{hx}; border:1px solid #444;")
            btn.clicked.connect(lambda _=False, h=hx: self._emit(h))
            g.addWidget(btn, row, col)

        # 레이아웃 stretch 조정(수동 여백 무효화)
        for r in range(_ROWS):
            g.setRowStretch(r, 0)
        for c in range(_COLS):
            g.setColumnStretch(c, 0)

    def _emit(self, hex_):
        self.colourSelected.emit(QColor(hex_))