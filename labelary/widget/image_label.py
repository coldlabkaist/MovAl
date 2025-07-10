from __future__ import annotations
import colorsys
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QPoint, QPointF, QSize, QRectF
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QBrush, QPixmap
from PyQt6.QtWidgets import QLabel

from ..data_loader import DataLoader

from labelary.controller import MouseController


class ClickableImageLabel(QLabel):
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

        self.mouse_controller = MouseController(self)
        self.installEventFilter(self.mouse_controller)

        self.dragging_target: Optional[Tuple] = None
        self.click_enabled = False

        self.skeleton_model = None

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

    def setCSVPoints(self, coords: Dict):
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

    def clear_all(self):
        self.clicked_points.clear()
        self.csv_points.clear()
        self.transformed_pixmap = None
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if not self.transformed_pixmap:
            return

        p = QPainter(self)
        p.drawPixmap(self.translation, self.transformed_pixmap)

        act = self.base_scale * self.current_scale
        ow, oh = self.original_pixmap.width(), self.original_pixmap.height()

        self._paint_skeleton(p, ow, oh, act)


    def _paint_skeleton(self, painter: QPainter, ow: int, oh: int, act: float) -> None:
        # ── 트랙(= 한 개체) 단위 순회 ─────────────────────────────
        for idx, (track, pts) in enumerate(self.csv_points.items()):
            if not pts:        # 좌표가 없으면 건너뜀
                continue

            # ── 1) 스켈레톤 “선” ────────────────────────────────
            edge_pen = QPen(self._skeleton_color(track, idx), 1)
            edge_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            edge_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(edge_pen)

            for edge in self.skeleton_model.edges:          # edge = frozenset({'A','B'})
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

            # ── 2) 노드(모양·색 반영) ───────────────────────────
            for node_name, node in self.skeleton_model.nodes.items():
                if node_name not in pts:
                    continue

                cx = pts[node_name][0] * ow * act + self.translation.x()
                cy = pts[node_name][1] * oh * act + self.translation.y()
                vis = pts[node_name][2]

                # 기본 반지름(또는 반쪽 변) : 스케일에 비례
                r = 10 * act

                # 펜·브러시 설정
                base_color = node.color
                pen   = QPen(base_color, node.thickness)
                brush = QBrush(base_color if node.filled else Qt.BrushStyle.NoBrush)
                painter.setPen(pen)
                painter.setBrush(brush)

                shape = node.shape.lower()
                if vis == 1:
                    painter.drawLine(QPointF(x - 5, y - 5), QPointF(x + 5, y + 5))
                    painter.drawLine(QPointF(x - 5, y + 5), QPointF(x + 5, y - 5))
                elif shape == "circle":
                    painter.drawEllipse(QPointF(cx, cy), r, r)
                elif shape == "square":
                    painter.drawRect(QRectF(cx - r, cy - r, 2*r, 2*r))
                elif shape == "text":
                    txt = node.text or node.name
                    fm  = painter.fontMetrics()
                    w   = fm.horizontalAdvance(txt)
                    h   = fm.height()
                    painter.drawText(QPointF(cx - w/2, cy + h/4), txt)
                else:   # 모르는 모양 → 원으로 대체
                    painter.drawEllipse(QPointF(cx, cy), r, r)


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

    def _skeleton_color(self, track: str, idx: int) -> QColor:# TODO 이 코드 정리할것 0708
        _gray_color = ["#FFFFFF", "#F2F2F2", "#D9D9D9", "#BFBFBF", "#9E9E9E", "#7A7A7A", "#545454", "#2E2E2E"]

        return QColor(_gray_color[idx*3%8])
