from __future__ import annotations
import colorsys
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QPoint, QPointF, QSize, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QBrush, QPixmap, QFontMetrics
from PyQt6.QtWidgets import QLabel, QMenu

from ..IO.data_loader import DataLoader
from utils.project import ProjectInformation

from labelary.controller import MouseController

CUTIE_COLOR_BASE = ["#ab1f24", "#36ae37", "#b9b917", "#063391", "#983a91",
                    "#20b6b5", "#c1c0bf", "#5c0d11", "#e71f19", "#60b630",
                    "#f4ba19", "#503390", "#ca4392", "#5eb7b7", "#f6bcbc"]

SKELETON_COLOR_SET = {"cutie_light" : (QColor("white"), 0.5),
                        "cutie_dark" : (QColor("black"), 0.4),
                        "white" : (QColor("white"), 1),
                        "black" : (QColor("black"), 1)}

class ClickableImageLabel(QLabel):
    node_selected = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.original_pixmap: Optional[QPixmap] = None
        self.transformed_pixmap: Optional[QPixmap] = None
        self.base_scale = 1.0
        self.current_scale = 1.0
        self.translation = QPoint(0, 0)

        self.clicked_points: List[Tuple[float, float]] = []
        self.csv_points: Dict[str, Dict[str, Tuple[float, float]]] = {}

        self.mouse_controller = None
        
        self.dragging_target: Optional[Tuple] = None
        self.click_enabled = True

        self.skeleton_model = None
        self.skeleton_color_mode = "cutie_light"
        self._track_color_idx: dict[str, int] = {}

        self.current_project = None
        self.video_loaded = False

        self.current_animal_num = None

    def setImage(self, pix: QPixmap, reset: bool):
        self.original_pixmap = pix

        if reset:
            self.base_scale   = min(self.width() / pix.width(),
                                    self.height() / pix.height())
            self.current_scale = 1.0
            self._updateTransformed()

            tw, th = self.transformed_pixmap.width(), self.transformed_pixmap.height()
            self.translation = QPoint((self.width()  - tw) // 2,
                                    (self.height() - th) // 2)
        else:
            self._updateTransformed()
        self.update()

    def load_skeleton_model(self, skeleton_model):
        self.skeleton_model = skeleton_model
        if self.current_project:
            self._track_color_idx = {nm: i for i, nm in enumerate(self.current_project.animals_name)}

    def set_skeleton_color_mode(self, color_mode):
        self.skeleton_color_mode = color_mode
        self.update()

    def _ensure_track_color_idx(self) -> None:
        if not self._track_color_idx and self.current_project:
            self._track_color_idx = {
                nm: i for i, nm in enumerate(self.current_project.animals_name)
            }
            for orig, mapped in DataLoader.track_mapping.items():
                if mapped in self._track_color_idx:
                    self._track_color_idx[orig] = self._track_color_idx[mapped]

    def setCSVPoints(self, coords: Dict):
        self._ensure_track_color_idx()
        self.csv_points = coords
        self.update()

    def _updateTransformed(self):
        if not self.original_pixmap:
            return
        w = int(self.original_pixmap.width() * self.base_scale * self.current_scale)
        h = int(self.original_pixmap.height() * self.base_scale * self.current_scale)
        self.transformed_pixmap = self.original_pixmap.scaled(
            QSize(w, h), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.transformed_pixmap:
            return

        p = QPainter(self)
        p.drawPixmap(self.translation, self.transformed_pixmap)
        act = self.base_scale * self.current_scale
        ow, oh = self.original_pixmap.width(), self.original_pixmap.height()

        self._paint_skeleton_model(p, ow, oh, act)

    def _paint_skeleton_model(self, painter: QPainter, ow: int, oh: int, act: float) -> None:
        self.current_animal_num = 0
        for track in self.current_project.animals_name:  
            self.current_animal_num += 1
            pts = self.csv_points.get(track, {})
            if not pts:
                continue

            edge_pen = QPen(self._skeleton_color(track), 2)
            edge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            edge_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(edge_pen)

            for edge in self.skeleton_model.edges:
                a, b = tuple(edge)
                if a not in pts or b not in pts:
                    continue

                p1 = QPointF(
                    pts[a][0] * ow * act + self.translation.x(),
                    pts[a][1] * oh * act + self.translation.y()
                )
                p2 = QPointF(
                    pts[b][0] * ow * act + self.translation.x(),
                    pts[b][1] * oh * act + self.translation.y()
                )
                painter.drawLine(p1, p2)

            for node_name, node in self.skeleton_model.nodes.items():
                if node_name not in pts:
                    continue

                cx = pts[node_name][0] * ow * act + self.translation.x()
                cy = pts[node_name][1] * oh * act + self.translation.y()
                vis = pts[node_name][2]

                r = 5 * (act**0.5)
                base_color = node.color
                selected = (self.mouse_controller.selected_node == (track, node_name))
                pen_width = node.thickness * (3 if selected else 1)
                pen = QPen(base_color, pen_width)
                brush = QBrush(base_color if node.filled else Qt.BrushStyle.NoBrush)
                painter.setPen(pen)
                painter.setBrush(brush)

                shape = node.shape.lower()
                if vis == 1:
                    d = r
                    painter.drawLine(QPointF(cx - d, cy - d), QPointF(cx + d, cy + d))
                    painter.drawLine(QPointF(cx - d, cy + d), QPointF(cx + d, cy - d))
                elif shape == "circle":
                    painter.drawEllipse(QPointF(cx, cy), r, r)
                elif shape == "square":
                    painter.drawRect(QRectF(cx - r, cy - r, 2*r, 2*r))
                elif shape == "text":
                    txt = node.text or node.name
                    painter.save()

                    font = painter.font()
                    font.setPixelSize(int(max(r * 3, 8)))
                    painter.setFont(font)
                    fm = QFontMetrics(font)
                    w  = fm.horizontalAdvance(txt)
                    h  = fm.height()

                    painter.drawText(QPointF(cx - w / 2, cy + h / 4), txt)
                    painter.restore()
                else:
                    painter.drawEllipse(QPointF(cx, cy), r, r)
                    
            if self.mouse_controller.selected_instance == track:
                xs = [nx * ow * act + self.translation.x() for nx, ny, vis in pts.values()]
                ys = [ny * oh * act + self.translation.y() for nx, ny, vis in pts.values()]
                if xs and ys:
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    box_pen = QPen(self._skeleton_color(track), 1, Qt.PenStyle.DashLine)
                    painter.setPen(box_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))

    def _pos_to_norm(self, pos: QPointF) -> tuple[float, float] | None:
        if not self.original_pixmap:
            return None
        act = self.base_scale * self.current_scale
        ow, oh = self.original_pixmap.width(), self.original_pixmap.height()
        x_img = (pos.x() - self.translation.x()) / (act * ow)
        y_img = (pos.y() - self.translation.y()) / (act * oh)
        if 0.0 <= x_img <= 1.0 and 0.0 <= y_img <= 1.0:
            return x_img, y_img
        return None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.original_pixmap:
            return

        pix = self.original_pixmap
        self.base_scale = min(self.width() / pix.width(),
                              self.height() / pix.height())
        self._updateTransformed()

        tw, th = self.transformed_pixmap.width(), self.transformed_pixmap.height()
        self.translation = QPoint((self.width() - tw) // 2,
                                  (self.height() - th) // 2)
        self.update()

    def _skeleton_color(self, track: str) -> QColor:
        idx = self._track_color_idx.get(track, 0)
        color = QColor(CUTIE_COLOR_BASE[idx % len(CUTIE_COLOR_BASE)])
        other, t = SKELETON_COLOR_SET[self.skeleton_color_mode]
        
        return QColor(round(color.red()*(1-t)+other.red()*t),
                  round(color.green()*(1-t)+other.green()*t),
                  round(color.blue()*(1-t)+other.blue()*t))
