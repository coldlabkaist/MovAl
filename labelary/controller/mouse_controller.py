from __future__ import annotations
from PyQt6.QtCore import QObject, QEvent, QPoint, Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from ..IO.data_loader import DataLoader

class MouseController(QObject):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label

        self._dragging = False
        self._last_pos = QPoint()

    def eventFilter(self, obj, event) -> bool:
        if obj is not self.label:
            return False

        mapping = {
            QEvent.Type.MouseButtonPress:   self._press,
            QEvent.Type.MouseMove:          self._move,
            QEvent.Type.MouseButtonRelease: self._release,
            QEvent.Type.Wheel:              self._wheel,
        }
        handler = mapping.get(event.type())
        return handler(event) if handler else False

    def update_label(self):
        frame = getattr(self.label, "current_frame", 0) + 1
        coords_dict = DataLoader.get_keypoint_coordinates_by_frame(frame)

        self.label.setCSVPoints(coords_dict)
        self.label.update()
        return True

    def _press(self, e: QMouseEvent) -> bool:
        pos = e.pos()

        if e.button() == Qt.MouseButton.RightButton:
            near = self._nearest_csv_kp(pos)
            if near and self.label.click_enabled:
                track, kp = near
                self.label.node_selected.emit(track, kp)
                frame  = getattr(self.label, "current_frame", 0) + 1
                nx, ny, vis = self.label.csv_points[track][kp]
                DataLoader.update_visibility(track, frame, kp, 1 if vis == 2 else 2)

                if self.label.dragging_target:
                    DataLoader.update_point(track, frame, kp, nx, ny)
                self._dragging = False
                return self.update_label()

            self._dragging, self._last_pos = True, pos
            self.label.dragging_target = None
            return self.update_label()

        if e.button() == Qt.MouseButton.LeftButton and self.label.click_enabled:
            near = self._nearest_csv_kp(pos)
            if near:
                track, kp = near
                self._dragging = True
                self.label.dragging_target = ("csv", track, kp)
                return self.update_label()
            act = self.label.base_scale * self.label.current_scale
            for i, (ox, oy) in enumerate(self.label.clicked_points):
                px = ox * act + self.label.translation.x()
                py = oy * act + self.label.translation.y()
                if (pos - QPoint(int(px), int(py))).manhattanLength() <= 10:
                    self._dragging = True
                    self.label.dragging_target = ("click", i)
                    return self.update_label()
        return False

    def _move(self, e: QMouseEvent) -> bool:
        if not self._dragging:
            return False
        pos = e.pos()
        act = self.label.base_scale * self.label.current_scale

        if e.buttons() & Qt.MouseButton.RightButton and self.label.dragging_target is None:
            delta = pos - self._last_pos

            new_tx = self.label.translation.x() + delta.x()
            new_ty = self.label.translation.y() + delta.y()
            new_tx, new_ty = self._get_clamped_translation(new_tx, new_ty)

            self.label.translation = QPoint(new_tx, new_ty)
            self._last_pos = pos
            self.label._updateTransformed()
            return True

        elif self.label.dragging_target:
            kind = self.label.dragging_target[0]
            if kind == "csv":
                _, track, kp = self.label.dragging_target
                frame = getattr(self.label,'current_frame',0)+1
                nx = (pos.x() - self.label.translation.x()) / (act * self.label.original_pixmap.width())
                ny = (pos.y() - self.label.translation.y()) / (act * self.label.original_pixmap.height())
                DataLoader.update_point(track, frame, kp, nx, ny)
            else:
                _, idx = self.label.dragging_target
                frame = getattr(self.label,'current_frame',0)+1 
                nx = (pos.x() - self.label.translation.x()) / act
                ny = (pos.y() - self.label.translation.y()) / act
                DataLoader.update_point(track, frame, kp, nx, ny)
            return self.update_label()

    def _release(self, _: QMouseEvent) -> bool:
        if not self._dragging:
            return False

        if self.label.dragging_target and self.label.dragging_target[0]=='csv':
            _,track,kp = self.label.dragging_target
            frame = getattr(self.label,'current_frame',0)+1
            nx,ny,_ = self.label.csv_points[track][kp]
            DataLoader.update_point(track, frame, kp, nx, ny)
        self._dragging=False #TODO
        self.label.dragging_target=None

        return self.update_label()

    def _wheel(self, e: QWheelEvent) -> bool:
        if not self.label.original_pixmap:
            return False

        cursor_pos = e.position().toPoint()
        old_act = self.label.base_scale * self.label.current_scale

        img_rel_x = (cursor_pos.x() - self.label.translation.x()) / (old_act * self.label.original_pixmap.width())
        img_rel_y = (cursor_pos.y() - self.label.translation.y()) / (old_act * self.label.original_pixmap.height())

        delta = e.angleDelta().y() or e.pixelDelta().y()
        factor = 1.1 if delta > 0 else 0.9

        new_scale = max(1.0, min(self.label.current_scale * factor, 10.0))
        if new_scale == self.label.current_scale:
            return False

        self.label.current_scale = new_scale
        self.label._updateTransformed() 

        new_pw = self.label.transformed_pixmap.width()
        new_ph = self.label.transformed_pixmap.height()
        new_tx = cursor_pos.x() - img_rel_x * new_pw
        new_ty = cursor_pos.y() - img_rel_y * new_ph
        new_tx, new_ty = self._get_clamped_translation(new_tx, new_ty)

        self.label.translation = QPoint(int(new_tx), int(new_ty))
        return self.update_label()

    def _nearest_csv_kp(self, pos: QPoint, thresh: int = 10):
        act = self.label.base_scale * self.label.current_scale
        ow, oh = self.label.original_pixmap.width(), self.label.original_pixmap.height()

        best, best_d = None, thresh + 1
        for track, pts in self.label.csv_points.items():
            for kp, (nx, ny, _) in pts.items():
                px = nx * ow * act + self.label.translation.x()
                py = ny * oh * act + self.label.translation.y()
                d = (pos - QPoint(int(px), int(py))).manhattanLength()
                if d < best_d:
                    best, best_d = (track, kp), d
        return best

    def _get_clamped_translation(self, new_tx, new_ty):
        vw, vh = self.label.width(),  self.label.height()
        pw, ph = self.label.transformed_pixmap.width(), self.label.transformed_pixmap.height()

        if pw >= vw:
            min_x, max_x = vw - pw, 0
        else:
            min_x = max_x = (vw - pw) / 2

        if ph >= vh:
            min_y, max_y = vh - ph, 0
        else:
            min_y = max_y = (vh - ph) / 2

        new_tx = max(min_x, min(max_x, new_tx))
        new_ty = max(min_y, min(max_y, new_ty))

        return (new_tx, new_ty)