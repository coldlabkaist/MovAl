from __future__ import annotations
import math
from PyQt6.QtCore import QObject, QEvent, QPoint, Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeySequence
from PyQt6.QtWidgets import QMenu
from ..IO.data_loader import DataLoader

class MouseController(QObject):
    def __init__(self, video_loader, video_viewer, kpt_list, parent=None):
        super().__init__(parent)
        self.video_loader = video_loader
        self.video_viewer = video_viewer
        self.kpt_list = kpt_list
        self.track_list = video_viewer.current_project.animals_name
        self.max_animals = video_viewer.current_project.num_animals

        self._dragging = False
        self._last_pos = QPoint()
        self.enable_control = True

        self.selected_instance: str | None = None
        self.selected_node: tuple[str, str] | None = None
        self.new_selection = False
        self._rotation_center_norm: tuple[float, float] | None = None
        self._rotation_center_px: tuple[float, float] | None = None
        self._rotation_start_angle: float | None = None
        self._rotation_source_points: dict[str, tuple[float, float, int]] = {}
        self._resize_center_norm: tuple[float, float] | None = None
        self._resize_source_points: dict[str, tuple[float, float, int]] = {}
        self._resize_start_distance_px: float | None = None
        self._node_hit_margin_px = 10
        self._instance_handle_thresh = 16
        self._resize_handle_thresh = 12

    def eventFilter(self, obj, event) -> bool:
        if obj is not self.video_viewer:
            return False
        if self.video_viewer.video_loaded == False:
            return False
        if self.enable_control == False:
            return False

        mapping = {
            QEvent.Type.MouseButtonPress:   self._press,
            QEvent.Type.MouseMove:          self._move,
            QEvent.Type.MouseButtonRelease: self._release,
            QEvent.Type.Wheel:              self._wheel,
        }
        handler = mapping.get(event.type())
        return handler(event) if handler else False

    def _sync_list_selection(self):
        if self.selected_instance is None:
            self.kpt_list.highlight(None, None)
            return
        track = str(self.selected_instance)
        kp    = self.selected_node[1] if self.selected_node else ""
        self.kpt_list.highlight(track, kp)
        self.kpt_list.update() 

    def _press(self, e: QMouseEvent) -> bool:
        if not self.video_viewer.click_enabled:
            return True
        pos = e.pos()
        # ---------- right click ----------
        if e.button() == Qt.MouseButton.RightButton:
            target_track = None
            target_kp = None
            near = self._nearest_csv_kp(pos)
            if near:
                target_track, target_kp = near
            else:
                target_track = self._instance_at_point(pos)

            if target_track is not None:
                if target_kp is not None:
                    self.selected_instance = target_track
                    self.selected_node = (target_track, target_kp)
                else:
                    self.selected_instance = target_track
                    self.selected_node = None
            else:
                self.selected_instance = None
                self.selected_node = None
            self.video_viewer.update()
            self._sync_list_selection()

            self.show_context_menu(e)

        # ---------- left click ----------
        if e.button() == Qt.MouseButton.LeftButton:
            near = self._nearest_csv_kp(pos)
            inside_track = near[0] if near else self._instance_at_point(pos)
            selected_resize_hit = (
                self._resize_handle_at_point(pos, self.selected_instance)
                if self.selected_instance is not None
                else None
            )
            rotation_track = self.selected_instance if self.selected_instance is not None else inside_track
            if rotation_track is not None and self._point_near_rotation_handle(pos, rotation_track):
                self.selected_instance = rotation_track
                self.selected_node = None
                self._dragging = True
                self.video_viewer.dragging_target = ("rotate_instance", rotation_track)
                self._start_instance_rotation(rotation_track, pos)
                self.video_viewer.update()
                self._sync_list_selection()
                return True

            if selected_resize_hit:
                track, corner = selected_resize_hit
                self.selected_instance = track
                self.selected_node = None
                self._dragging = True
                self.video_viewer.dragging_target = ("resize_instance", track, corner)
                self._start_instance_resize(track, corner)
                self.video_viewer.update()
                self._sync_list_selection()
                return True

            if self.selected_instance is not None and inside_track == self.selected_instance:
                selected_node_hit = self._nearest_csv_kp(pos, track=self.selected_instance)
                if selected_node_hit:
                    track, kp = selected_node_hit
                    self.selected_instance = track
                    self.selected_node = (track, kp)
                    self._dragging = True
                    self.video_viewer.dragging_target = ("csv", track, kp)
                else:
                    self.selected_node = None
                    track = self.selected_instance
                    self._dragging = True
                    self.video_viewer.dragging_target = ("instance", track)
                    self._last_pos = pos
                self.video_viewer.update()
                self._sync_list_selection()
                return True

            if near:
                track, kp = near
                self.selected_instance = track
                if self._point_near_csv_kp(pos, track, kp):
                    self.selected_node = (track, kp)
                    self._dragging = True
                    self.video_viewer.dragging_target = ("csv", track, kp)
                else:
                    self.selected_node = None
                    self._dragging = True
                    self.video_viewer.dragging_target = ("instance", track)
                    self._last_pos = pos
                self.video_viewer.update()
                self._sync_list_selection()
                return True
            if inside_track is not None:
                track = inside_track
                self.selected_instance = track
                self.selected_node = None
                self._dragging = True
                self.video_viewer.dragging_target = ("instance", track)
                self._last_pos = pos
                self.video_viewer.update()
                self._sync_list_selection()
                return True

            if self.selected_instance is not None or self.selected_node is not None:
                self.selected_instance = None
                self.selected_node = None
                self.video_viewer.update()
                self._sync_list_selection()
            self._dragging = True
            self.video_viewer.dragging_target = None
            self._last_pos = pos
            return True

        return False

    def _move(self, e: QMouseEvent) -> bool:
        if not self._dragging:
            return False
        pos = e.pos()
        act = self.video_viewer.base_scale * self.video_viewer.current_scale

        if self.video_viewer.dragging_target is None:
            delta = pos - self._last_pos
            new_tx = self.video_viewer.translation.x() + delta.x()
            new_ty = self.video_viewer.translation.y() + delta.y()
            new_tx, new_ty = self._get_clamped_translation(new_tx, new_ty)
            self.video_viewer.translation = QPoint(new_tx, new_ty)
            self._last_pos = pos

        elif self.video_viewer.dragging_target:
            kind = self.video_viewer.dragging_target[0]
            if kind == "csv":
                _, track, kp = self.video_viewer.dragging_target
                nx = (pos.x() - self.video_viewer.translation.x()) / (act * self.video_viewer.original_pixmap.width())
                ny = (pos.y() - self.video_viewer.translation.y()) / (act * self.video_viewer.original_pixmap.height())
                nx = max(0.0, min(nx, 1.0))
                ny = max(0.0, min(ny, 1.0))
                self.video_viewer.csv_points[track][kp] = (nx, ny, self.video_viewer.csv_points[track][kp][2])
            elif kind == "instance":
                _, track = self.video_viewer.dragging_target
                dx_norm = (pos.x() - self._last_pos.x()) / (act * self.video_viewer.original_pixmap.width())
                dy_norm = (pos.y() - self._last_pos.y()) / (act * self.video_viewer.original_pixmap.height())
                
                for kp, (nx, ny, vis) in self.video_viewer.csv_points.get(track, {}).items():
                    nx_new = max(0.0, min(nx + dx_norm, 1.0))
                    ny_new = max(0.0, min(ny + dy_norm, 1.0))
                    self.video_viewer.csv_points[track][kp] = (nx_new, ny_new, vis)
                self._last_pos = pos
            elif kind == "rotate_instance":
                _, track = self.video_viewer.dragging_target
                self._rotate_instance(track, pos)
            elif kind == "resize_instance":
                _, track, corner = self.video_viewer.dragging_target
                self._resize_instance(track, corner, pos)
            elif kind == "click":
                _, idx = self.video_viewer.dragging_target
                nx = (pos.x() - self.video_viewer.translation.x()) / act
                ny = (pos.y() - self.video_viewer.translation.y()) / act
                self.video_viewer.clicked_points[idx] = (nx, ny)

        self.video_viewer.update()
        return True

    def _release(self, _: QMouseEvent) -> bool:
        if not self._dragging:
            return False

        try:
            if self.video_viewer.dragging_target:
                kind = self.video_viewer.dragging_target[0]
                frame_idx = getattr(self.video_viewer, "current_frame", 0) 
                if kind == "csv":
                    _, track, kp = self.video_viewer.dragging_target
                    nx, ny, _ = self.video_viewer.csv_points[track][kp]
                    DataLoader.update_point(track, frame_idx, kp, nx, ny)
                elif kind in ("instance", "rotate_instance", "resize_instance"):
                    if kind == "resize_instance":
                        _, track, _ = self.video_viewer.dragging_target
                    else:
                        _, track = self.video_viewer.dragging_target
                    if track in self.video_viewer.csv_points:
                        for kp, (nx, ny, _) in self.video_viewer.csv_points[track].items():
                            DataLoader.update_point(track, frame_idx, kp, nx, ny)
        except KeyError:
            pass

        self._dragging = False
        self.video_viewer.dragging_target = None
        self._clear_rotation_state()
        self._clear_resize_state()
        return True

    def _wheel(self, e: QWheelEvent) -> bool:
        if not self.video_viewer.original_pixmap:
            return False

        cursor_pos = e.position().toPoint()
        old_act = self.video_viewer.base_scale * self.video_viewer.current_scale

        img_rel_x = (cursor_pos.x() - self.video_viewer.translation.x()) / (old_act * self.video_viewer.original_pixmap.width())
        img_rel_y = (cursor_pos.y() - self.video_viewer.translation.y()) / (old_act * self.video_viewer.original_pixmap.height())

        delta = e.angleDelta().y() or e.pixelDelta().y()
        factor = 1.1 if delta > 0 else 0.9

        new_scale = max(1.0, min(self.video_viewer.current_scale * factor, 10.0))
        if new_scale == self.video_viewer.current_scale: 
            return False

        self.video_viewer.current_scale = new_scale
        self.video_viewer._updateTransformed()

        new_pw = self.video_viewer.transformed_pixmap.width()
        new_ph = self.video_viewer.transformed_pixmap.height()

        new_tx = cursor_pos.x() - img_rel_x * new_pw
        new_ty = cursor_pos.y() - img_rel_y * new_ph
        new_tx, new_ty = self._get_clamped_translation(new_tx, new_ty)

        self.video_viewer.translation = QPoint(int(new_tx), int(new_ty))
        self.video_viewer.update()
        return True

    def show_context_menu(self, e):
        if self.enable_control == False:
            return False
        if hasattr(self, "_active_menu") and self._active_menu is not None:
            self._active_menu.close()
            self._active_menu.deleteLater()
            self._active_menu = None

        pos = e.pos()
        self.video_viewer._context_click_pos = pos 
        menu = QMenu(self.video_viewer)
        self._active_menu = menu

        act_add = menu.addAction("Add New Instance")
        act_add.setShortcut(QKeySequence("Ctrl+A"))
        act_add.setShortcutVisibleInContextMenu(True)

        act_delete = menu.addAction("Delete Instance")
        act_delete.setShortcut(QKeySequence("Delete"))
        act_delete.setShortcut(QKeySequence("Ctrl+D"))
        act_delete.setShortcutVisibleInContextMenu(True)

        act_change_num = menu.addMenu("Change Instance Number")
        for nm in range(self.max_animals):
            track_name = self.track_list[nm]
            act_nm = act_change_num.addAction(track_name)
            act_nm.setEnabled(track_name != self.selected_instance)
            act_nm.triggered.connect(
                lambda _=False, t=track_name: self._change_instance_number(t)
            )
            if nm<=9:
                act_nm.setShortcut(QKeySequence(f"Ctrl+{(nm+1)%10}"))
        act_delete.setShortcutVisibleInContextMenu(True)

        act_vis = menu.addAction("Change Visibility")
        act_vis.setShortcut(QKeySequence("Ctrl+V"))
        act_delete.setShortcutVisibleInContextMenu(True)

        if self.video_loader is not None:
            menu.addSeparator()
            act_prev_lbl = menu.addAction("Move to Previous Labeled Frame")
            act_next_lbl = menu.addAction("Move to Next Labeled Frame")
            act_prev_lbl.triggered.connect(lambda: self._move_labeled(-1))
            act_next_lbl.triggered.connect(lambda: self._move_labeled(+1))
            act_prev_lbl.setShortcut(QKeySequence("Ctrl+Left"))
            act_next_lbl.setShortcut(QKeySequence("Ctrl+Right"))

        act_add.setEnabled(self.video_viewer.current_animal_num <= self.max_animals)
        act_delete.setEnabled(self.selected_instance is not None)
        act_change_num.setEnabled(self.selected_instance is not None)
        act_vis.setEnabled(self.selected_node is not None)
        
        act_add.triggered.connect(self._add_new_skeleton_label)
        act_delete.triggered.connect(self._delete_selected_instance)
        act_vis.triggered.connect(self._toggle_selected_node_visibility)
        
        menu.aboutToHide.connect(lambda: setattr(self, "_active_menu", None))
        global_pt = e.globalPosition().toPoint()
        menu.popup(global_pt)
        return True

    def _nearest_csv_kp(self, pos: QPoint, track: str | None = None) -> tuple[str, str] | None:
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1

        best = None
        best_d = float("inf")
        for track_name, pts in self.video_viewer.csv_points.items():
            if track is not None and track_name != track:
                continue
            for kp, (nx, ny, vis) in pts.items():
                px, py = self._point_to_viewer_px(nx, ny, ow, oh)
                d = math.hypot(pos.x() - px, pos.y() - py)
                if d < best_d:
                    best = (track_name, kp)
                    best_d = d
        hit_radius = self._node_display_radius_px() + self._node_hit_margin_px
        return best if best is not None and best_d <= hit_radius else None

    def _point_near_csv_kp(self, pos: QPoint, track: str, kp: str) -> bool:
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1
        nx, ny, _ = self.video_viewer.csv_points.get(track, {}).get(kp, (0.0, 0.0, 0))
        px, py = self._point_to_viewer_px(nx, ny, ow, oh)
        d = math.hypot(pos.x() - px, pos.y() - py)
        return d <= (self._node_display_radius_px() + self._node_hit_margin_px)

    def _instance_at_point(self, pos: QPoint) -> str | None:
        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1

        for track, pts in self.video_viewer.csv_points.items():
            if not pts:
                continue
            min_x, max_x, min_y, max_y = self._instance_bounds_px(track, padding=max(8, int(6 * (act ** 0.5))))
            if min_x <= pos.x() <= max_x and min_y <= pos.y() <= max_y:
                return track
        return None

    def _instance_bounds_px(self, track: str, padding: int = 0) -> tuple[float, float, float, float]:
        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1
        pts = self.video_viewer.csv_points.get(track, {})
        xs = [nx * ow * act + self.video_viewer.translation.x() for nx, ny, vis in pts.values()]
        ys = [ny * oh * act + self.video_viewer.translation.y() for nx, ny, vis in pts.values()]
        if not xs or not ys:
            return 0.0, 0.0, 0.0, 0.0
        return min(xs) - padding, max(xs) + padding, min(ys) - padding, max(ys) + padding

    def _rotation_handle_px(self, track: str) -> tuple[float, float] | None:
        geom = self._rotation_geometry(track)
        if geom is None:
            return None
        return geom["handle"]

    def _point_near_rotation_handle(self, pos: QPoint, track: str) -> bool:
        geom = self._rotation_geometry(track)
        if geom is None:
            return False
        handle_x, handle_y = geom["handle"]
        return math.hypot(pos.x() - handle_x, pos.y() - handle_y) <= self._instance_handle_thresh

    def _resize_handle_at_point(self, pos: QPoint, track: str) -> tuple[str, str] | None:
        geom = self._rotation_geometry(track)
        if geom is None:
            return None
        for corner_name, (hx, hy) in geom["resize_handles"].items():
            if math.hypot(pos.x() - hx, pos.y() - hy) <= self._resize_handle_thresh:
                return track, corner_name
        return None

    def _rotation_geometry(self, track: str) -> dict[str, tuple[float, float]] | None:
        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1
        pts = self.video_viewer.csv_points.get(track, {})
        if not pts:
            return None

        xs = [nx * ow * act + self.video_viewer.translation.x() for nx, ny, vis in pts.values()]
        ys = [ny * oh * act + self.video_viewer.translation.y() for nx, ny, vis in pts.values()]
        if not xs or not ys:
            return None

        box_margin = 15.0
        lift = 14.0
        min_x, max_x = min(xs) - box_margin, max(xs) + box_margin
        min_y, max_y = min(ys) - box_margin, max(ys) + box_margin
        handle = ((min_x + max_x) / 2.0, min_y - lift)
        anchor = ((min_x + max_x) / 2.0, min_y)
        resize_handles = {
            "top_left": (min_x, min_y),
            "top_right": (max_x, min_y),
            "bottom_left": (min_x, max_y),
            "bottom_right": (max_x, max_y),
        }
        return {
            "box_min": (min_x, min_y),
            "box_max": (max_x, max_y),
            "handle": handle,
            "anchor": anchor,
            "resize_handles": resize_handles,
        }

    def _start_instance_rotation(self, track: str, pos: QPoint) -> None:
        pts = self.video_viewer.csv_points.get(track, {})
        if not pts:
            self._clear_rotation_state()
            return

        xs = [nx for nx, ny, vis in pts.values()]
        ys = [ny for nx, ny, vis in pts.values()]
        center_norm = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        center_px = self._norm_to_viewer_px(*center_norm)

        self._rotation_center_norm = center_norm
        self._rotation_center_px = center_px
        self._rotation_start_angle = math.atan2(pos.y() - center_px[1], pos.x() - center_px[0])
        self._rotation_source_points = {
            kp: (nx, ny, vis) for kp, (nx, ny, vis) in pts.items()
        }

    def _rotate_instance(self, track: str, pos: QPoint) -> None:
        if (
            self._rotation_center_norm is None
            or self._rotation_center_px is None
            or self._rotation_start_angle is None
        ):
            return

        center_nx, center_ny = self._rotation_center_norm
        center_px_x, center_px_y = self._rotation_center_px
        current_angle = math.atan2(pos.y() - center_px_y, pos.x() - center_px_x)
        delta_angle = current_angle - self._rotation_start_angle
        cos_a = math.cos(delta_angle)
        sin_a = math.sin(delta_angle)

        for kp, (src_x, src_y, vis) in self._rotation_source_points.items():
            dx = src_x - center_nx
            dy = src_y - center_ny
            rot_x = center_nx + dx * cos_a - dy * sin_a
            rot_y = center_ny + dx * sin_a + dy * cos_a
            self.video_viewer.csv_points[track][kp] = (
                max(0.0, min(rot_x, 1.0)),
                max(0.0, min(rot_y, 1.0)),
                vis,
            )

    def _clear_rotation_state(self) -> None:
        self._rotation_center_norm = None
        self._rotation_center_px = None
        self._rotation_start_angle = None
        self._rotation_source_points = {}

    def _start_instance_resize(self, track: str, corner: str) -> None:
        pts = self.video_viewer.csv_points.get(track, {})
        geom = self._rotation_geometry(track)
        if not pts or geom is None:
            self._clear_resize_state()
            return

        xs = [nx for nx, ny, vis in pts.values()]
        ys = [ny for nx, ny, vis in pts.values()]
        center_norm = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        center_px = self._norm_to_viewer_px(*center_norm)
        corner_px = geom["resize_handles"][corner]

        self._resize_center_norm = center_norm
        self._resize_source_points = {
            kp: (nx, ny, vis) for kp, (nx, ny, vis) in pts.items()
        }
        self._resize_start_distance_px = max(
            math.hypot(corner_px[0] - center_px[0], corner_px[1] - center_px[1]),
            1.0,
        )

    def _resize_instance(self, track: str, corner: str, pos: QPoint) -> None:
        if (
            self._resize_center_norm is None
            or not self._resize_source_points
            or self._resize_start_distance_px is None
        ):
            return

        center_nx, center_ny = self._resize_center_norm
        center_px_x, center_px_y = self._norm_to_viewer_px(center_nx, center_ny)
        current_distance = math.hypot(pos.x() - center_px_x, pos.y() - center_px_y)
        scale = max(current_distance / self._resize_start_distance_px, 0.1)

        for kp, (src_x, src_y, vis) in self._resize_source_points.items():
            dx = src_x - center_nx
            dy = src_y - center_ny
            scaled_x = center_nx + dx * scale
            scaled_y = center_ny + dy * scale
            self.video_viewer.csv_points[track][kp] = (
                max(0.0, min(scaled_x, 1.0)),
                max(0.0, min(scaled_y, 1.0)),
                vis,
            )

    def _clear_resize_state(self) -> None:
        self._resize_center_norm = None
        self._resize_source_points = {}
        self._resize_start_distance_px = None

    def _norm_to_viewer_px(self, nx: float, ny: float) -> tuple[float, float]:
        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1
        px = nx * ow * act + self.video_viewer.translation.x()
        py = ny * oh * act + self.video_viewer.translation.y()
        return px, py

    def _point_to_viewer_px(self, nx: float, ny: float, ow: int | None = None, oh: int | None = None) -> tuple[float, float]:
        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        ow = ow if ow is not None else (self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1)
        oh = oh if oh is not None else (self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1)
        px = nx * ow * act + self.video_viewer.translation.x()
        py = ny * oh * act + self.video_viewer.translation.y()
        return px, py

    def _node_display_radius_px(self) -> float:
        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        return 5.0 * (act ** 0.5)

    def _get_clamped_translation(self, new_tx: int, new_ty: int) -> tuple[int, int]:
        vw, vh = self.video_viewer.width(), self.video_viewer.height()
        pw = self.video_viewer.transformed_pixmap.width() if self.video_viewer.transformed_pixmap else 0
        ph = self.video_viewer.transformed_pixmap.height() if self.video_viewer.transformed_pixmap else 0

        if pw >= vw:
            min_x, max_x = vw - pw, 0
        else:
            min_x = max_x = (vw - pw) // 2
        if ph >= vh:
            min_y, max_y = vh - ph, 0
        else:
            min_y = max_y = (vh - ph) // 2

        new_tx = max(min_x, min(max_x, new_tx))
        new_ty = max(min_y, min(max_y, new_ty))
        return new_tx, new_ty

    
    def _add_new_skeleton_label(self):
        if not hasattr(self.video_viewer, "current_frame"):
            return
        frame_idx = self.video_viewer.current_frame
        present_tracks = self._tracks_in_frame(frame_idx)
        
        new_track = next((name for name in self.track_list
                      if name not in present_tracks), None)
        if new_track is None:
            return

        anchor_norm = None
        if hasattr(self, "_context_click_pos"):
            anchor_norm = self._pos_to_norm(self._context_click_pos)
        success = DataLoader.add_skeleton_instance(
            frame_idx=frame_idx,
            track_name=new_track,
            anchor_xy=anchor_norm
        )
        if not success:
            return

        coords = DataLoader.get_keypoint_coordinates_by_frame(frame_idx)
        self.video_viewer.setCSVPoints(coords)
        self.kpt_list.update_list_visibility(coords)
        self.video_viewer.update()

    def _tracks_in_frame(self, frame_idx: int):
        df = DataLoader.loaded_data
        if df is None or df.empty:
            return set()
        return set(df[df["frame_idx"] == frame_idx]["track"].unique())

    def _delete_selected_instance(self):
        if self.selected_instance is None:
            return
        frame_idx = getattr(self.video_viewer, "current_frame", 0) 
        track = self.selected_instance
        success = DataLoader.delete_instance(frame_idx, track)
        if not success:
            return
        coords = DataLoader.get_keypoint_coordinates_by_frame(frame_idx)
        self.video_viewer.setCSVPoints(coords)
        self.kpt_list.update_list_visibility(coords)
        self.selected_instance = None
        self.selected_node = None

    def _change_instance_number_by_idx(self, idx):
        if idx >= len(self.track_list):
            return
        if self.selected_instance == self.track_list[idx]:
            return
        self._change_instance_number(self.track_list[idx])

    def _change_instance_number(self, new_track: str):
        if self.selected_instance is None:
            return
        if new_track is None:
            from PyQt6.QtWidgets import QInputDialog
            items = [nm for nm in range(self.max_animals)
                     if nm != self.selected_instance]
            if not items:
                return
            ok = False
            new_track, ok = QInputDialog.getItem(
                self.video_viewer,
                "Change Instance Number",
                "Select new track name:",
                items, 0, False
            )
            if not ok:
                return

        frame_idx   = getattr(self.video_viewer, "current_frame", 0)
        old_track   = self.selected_instance
        if not DataLoader.swap_or_rename_instance(frame_idx, old_track, new_track):
            return

        coords = DataLoader.get_keypoint_coordinates_by_frame(frame_idx)
        self.video_viewer.setCSVPoints(coords)
        self.selected_instance = new_track
        self.selected_node = None
        self.video_viewer.update()
        self.kpt_list.update_list_visibility(coords)
        self._sync_list_selection()

    def _toggle_selected_node_visibility(self):
        if self.selected_node is None:
            return
        track, kp = self.selected_node
        frame_idx = getattr(self.video_viewer, "current_frame", 0)

        cur_vis = 2
        df = DataLoader.loaded_data if hasattr(DataLoader, "loaded_data") else None
        if df is not None:
            try:
                cur_vis = df.loc[(df["track"] == track) & (df["frame_idx"] == frame_idx), f"{kp}.visibility"].iat[0]
            except Exception:
                pass

        new_vis = 1 if cur_vis == 2 else 2
        DataLoader.update_kpt_visibility(track, frame_idx, kp, new_vis)
        if track in self.video_viewer.csv_points and kp in self.video_viewer.csv_points[track]:
            nx, ny, _ = self.video_viewer.csv_points[track][kp]
            self.video_viewer.csv_points[track][kp] = (nx, ny, new_vis)
        self.video_viewer.update()

    def _move_labeled(self, direction: int):
        if self.video_loader is None:
            return
        self.video_loader.move_to_labeled_frame(direction)

    def _pos_to_norm(self, pos: QPointF):
        return self.video_viewer._pos_to_norm(pos)
