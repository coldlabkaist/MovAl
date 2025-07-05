from __future__ import annotations
import colorsys
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QPoint, QPointF, QSize
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QLabel

from ..data_loader import DataLoader
from ..color import ColorManager, ColorDialog

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

        # skeleton lines
        sk = DataLoader.skeleton_data
        for i, track in enumerate(sk):
            pts = self.csv_points.get(track, {})
            pen = QPen(self._skeleton_color(track, i), 2) #TODO : 이거 2개씩 먹일 이유가 없음 track, i
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen) 
            for a, b in sk[track]:
                if a in pts and b in pts:
                    x1 = pts[a][0] * ow * act + self.translation.x()
                    y1 = pts[a][1] * oh * act + self.translation.y()
                    x2 = pts[b][0] * ow * act + self.translation.x()
                    y2 = pts[b][1] * oh * act + self.translation.y()
                    p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # csv points
        df = DataLoader.data
        frame = getattr(self, "current_frame", 0) + 1
        for track, pts in self.csv_points.items():
            #print(track, pts)
            for kp, (nx, ny) in pts.items():
                x = nx * ow * act + self.translation.x()
                y = ny * oh * act + self.translation.y()
                vis = 2
                if df is not None:
                    m = (df["track"] == track) & (df["frame.idx"] == frame)
                    if m.any():
                        col = f"{kp}.visibility"
                        if col in df.columns and not df.loc[m, col].empty:
                            vis = int(df.loc[m, col].values[0])
                """color = self.getColorForKeypoint(kp)
                p.setPen(QPen(color, 5))"""
                color = self.getColorForKeypoint(kp)
                p.setBrush(color) 
                if color.alpha() != 255:
                    color = QColor(color)
                    color.setAlpha(color.alpha()/5+204)
                p.setPen(QPen(color, 1))

                if vis == 2:
                    p.drawEllipse(QPointF(x, y), 5, 5)
                else:
                    p.drawLine(QPointF(x - 5, y - 5), QPointF(x + 5, y + 5))
                    p.drawLine(QPointF(x - 5, y + 5), QPointF(x + 5, y - 5))

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

    def _skeleton_color(self, track: str, idx: int) -> QColor:# TODO 이 코드 정리할것
        return ColorManager.track_color(track, idx)

    def getColorForKeypoint(self, kp: str) -> QColor:
        return ColorManager.kp_color(kp)