from __future__ import annotations
from PyQt6.QtCore import QObject, QEvent, QPoint, Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent
from ..data_loader import DataLoader

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

    def _press(self, e: QMouseEvent) -> bool:
        pos = e.pos()
        # ---------- right click ----------
        if e.button() == Qt.MouseButton.RightButton:
            near = self._nearest_csv_kp(pos)
            if near and self.label.click_enabled:
                track, kp = near
                frame  = getattr(self.label, "current_frame", 0) + 1
                v_col  = f"{kp}.visibility"
                cur    = 2
                df = DataLoader.data
                if df is not None:
                    try:
                        cur = df.loc[(df["track"] == track) &
                                     (df["frame.idx"] == frame), v_col].iat[0]
                    except Exception:
                        pass

                DataLoader.update_visibility(track, frame, kp, 1 if cur == 2 else 2)
                if self.label.dragging_target and self.label.dragging_target[0]=='csv':
                    _,track,kp = self.label.dragging_target
                    frame = getattr(self.label,'current_frame',0)+1
                    nx,ny = self.label.csv_points[track][kp]
                    DataLoader.update_point(track, frame, kp, nx, ny)

                self.label.update()
                self._dragging = False
                return True

            self._dragging, self._last_pos = True, pos
            self.label.dragging_target = None
            return True

        # ---------- left click ----------
        if e.button() == Qt.MouseButton.LeftButton and self.label.click_enabled:
            near = self._nearest_csv_kp(pos)
            if near:
                track, kp = near
                self._dragging = True
                self.label.dragging_target = ("csv", track, kp)
                return True
            act = self.label.base_scale * self.label.current_scale
            for i, (ox, oy) in enumerate(self.label.clicked_points):
                px = ox * act + self.label.translation.x()
                py = oy * act + self.label.translation.y()
                if (pos - QPoint(int(px), int(py))).manhattanLength() <= 10:
                    self._dragging = True
                    self.label.dragging_target = ("click", i)
                    return True
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

        elif self.label.dragging_target:
            kind = self.label.dragging_target[0]
            if kind == "csv":
                _, track, kp = self.label.dragging_target
                nx = (pos.x() - self.label.translation.x()) / (act * self.label.original_pixmap.width())
                ny = (pos.y() - self.label.translation.y()) / (act * self.label.original_pixmap.height())
                self.label.csv_points[track][kp] = (nx, ny)
            else:
                _, idx = self.label.dragging_target
                nx = (pos.x() - self.label.translation.x()) / act
                ny = (pos.y() - self.label.translation.y()) / act
                self.label.clicked_points[idx] = (nx, ny)

        self.label._updateTransformed()
        self.label.update()
        return True

    def _release(self, _: QMouseEvent) -> bool:
        if not self._dragging:
            return False

        if self.label.dragging_target and self.label.dragging_target[0]=='csv':
            _,track,kp = self.label.dragging_target
            frame = getattr(self.label,'current_frame',0)+1
            nx,ny = self.label.csv_points[track][kp]
            DataLoader.update_point(track, frame, kp, nx, ny)
        self._dragging=False #TODO
        self.label.dragging_target=None

        return True

    def _wheel(self, e: QWheelEvent) -> bool:
        """
        확대·축소용 휠 이벤트 핸들러 (MouseController 전용 개정판).

        • 확대/축소 범위 : current_scale ∈ [1.0, 10.0]
        • 커서 아래 지점 고정
        • 이미지가 화면보다 작아질 때는 자동 센터링
        """
        # 1) 이미지가 없는 경우 처리 생략
        if not self.label.original_pixmap:
            return False

        # 2) 위젯 좌표계에서의 커서 위치
        cursor_pos = e.position().toPoint()

        # 3) 현재 전체 배율 (base_scale × current_scale)
        old_act = self.label.base_scale * self.label.current_scale

        # 4) 커서가 가리키는 (원본 이미지 기준) 정규화 좌표 [0 ~ 1]
        img_rel_x = (cursor_pos.x() - self.label.translation.x()) / (old_act * self.label.original_pixmap.width())
        img_rel_y = (cursor_pos.y() - self.label.translation.y()) / (old_act * self.label.original_pixmap.height())

        # 5) 스케일 변화량 계산 (각도 델타 ↔ 픽셀 델타 모두 지원)
        delta = e.angleDelta().y() or e.pixelDelta().y()
        factor = 1.1 if delta > 0 else 0.9

        new_scale = max(1.0, min(self.label.current_scale * factor, 10.0))
        if new_scale == self.label.current_scale:        # 변경 없으면 즉시 종료
            return False

        # 6) 스케일 반영 후 변환된 Pixmap 갱신
        self.label.current_scale = new_scale
        self.label._updateTransformed()                  # → transformed_pixmap 크기 갱신

        new_pw = self.label.transformed_pixmap.width()
        new_ph = self.label.transformed_pixmap.height()

        # 7) “커서 고정”을 위한 새 translation 계산
        new_tx = cursor_pos.x() - img_rel_x * new_pw
        new_ty = cursor_pos.y() - img_rel_y * new_ph

        # 8) 화면 밖으로 나가지 않도록 클램핑
        new_tx, new_ty = self._get_clamped_translation(new_tx, new_ty)

        # 9) 상태 반영 및 화면 갱신
        self.label.translation = QPoint(int(new_tx), int(new_ty))
        self.label.update()
        return True

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