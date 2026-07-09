from __future__ import annotations
import math
from PyQt6.QtCore import QObject, QEvent, QPoint, Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeySequence
from PyQt6.QtWidgets import QMenu
from ..IO.data_loader import DataLoader
from .edit_history import DEFAULT_EDIT_HISTORY_LIMIT, FrameEditHistory

class MouseController(QObject):
    def __init__(self, video_loader, video_viewer, kpt_list, parent=None, edit_history_limit=DEFAULT_EDIT_HISTORY_LIMIT):
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
        self._resize_anchor_norm: tuple[float, float] | None = None
        self._resize_initial_corner_norm: tuple[float, float] | None = None
        self._resize_source_points: dict[str, tuple[float, float, int]] = {}
        self._resize_start_distance_px: float | None = None
        self._node_hit_margin_px = 10
        self._instance_handle_thresh = 16
        self._resize_handle_thresh = 12
        self.edit_history = FrameEditHistory(limit=edit_history_limit)
        self._pending_edit_frame: int | None = None
        self._pending_edit_before = None
        self._history_data_version = getattr(DataLoader, "_label_version", None)

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

    def _current_label_frame(self) -> int | None:
        if not hasattr(self.video_viewer, "current_frame"):
            return None
        parent = getattr(self.video_loader, "parent", None)
        if parent is not None and hasattr(parent, "resolve_label_frame"):
            return parent.resolve_label_frame(self.video_viewer.current_frame)
        return int(getattr(self.video_viewer, "current_frame", 0))

    def _refresh_visible_overlay(self) -> None:
        parent = getattr(self.video_loader, "parent", None)
        if parent is not None and hasattr(parent, "refresh_frame_bound_views"):
            parent.refresh_frame_bound_views()
        else:
            frame_idx = self._current_label_frame()
            coords = (
                DataLoader.get_keypoint_coordinates_by_frame(frame_idx)
                if frame_idx is not None
                else {}
            )
            self.video_viewer.setCSVPoints(coords)
            self.kpt_list.update_list_visibility(coords)
        self.video_viewer.update()

    def clear_edit_history_if_frame_changed(self, frame_idx: int | None) -> None:
        data_version = getattr(DataLoader, "_label_version", None)
        if data_version != self._history_data_version:
            self.edit_history.clear()
            self._history_data_version = data_version
        self.edit_history.clear_if_frame_changed(frame_idx)

    def _begin_frame_edit(self, frame_idx: int | None = None) -> None:
        if self._pending_edit_before is not None:
            return
        if frame_idx is None:
            frame_idx = self._current_label_frame()
        if frame_idx is None:
            return
        self._pending_edit_frame = int(frame_idx)
        self._pending_edit_before = DataLoader.snapshot_frame(frame_idx)

    def _finish_frame_edit(self, frame_idx: int | None = None) -> None:
        if self._pending_edit_before is None:
            return
        if frame_idx is None:
            frame_idx = self._pending_edit_frame
        if frame_idx is None or frame_idx != self._pending_edit_frame:
            self._discard_pending_frame_edit()
            return
        after = DataLoader.snapshot_frame(frame_idx)
        self.edit_history.push(frame_idx, self._pending_edit_before, after)
        self._discard_pending_frame_edit()

    def _discard_pending_frame_edit(self) -> None:
        self._pending_edit_frame = None
        self._pending_edit_before = None

    def _is_history_drag_kind(self, kind: str) -> bool:
        return kind in ("csv", "instance", "rotate_instance", "resize_instance")

    def can_undo_current_frame(self) -> bool:
        return self.edit_history.can_undo(self._current_label_frame())

    def can_redo_current_frame(self) -> bool:
        return self.edit_history.can_redo(self._current_label_frame())

    def undo_current_frame_edit(self) -> bool:
        frame_idx = self._current_label_frame()
        edit = self.edit_history.pop_undo(frame_idx)
        if edit is None:
            return False
        DataLoader.restore_frame(edit.frame_idx, edit.before)
        self._history_data_version = getattr(DataLoader, "_label_version", None)
        self._normalize_selection_after_restore(edit.frame_idx)
        self._refresh_visible_overlay()
        self._sync_list_selection()
        return True

    def redo_current_frame_edit(self) -> bool:
        frame_idx = self._current_label_frame()
        edit = self.edit_history.pop_redo(frame_idx)
        if edit is None:
            return False
        DataLoader.restore_frame(edit.frame_idx, edit.after)
        self._history_data_version = getattr(DataLoader, "_label_version", None)
        self._normalize_selection_after_restore(edit.frame_idx)
        self._refresh_visible_overlay()
        self._sync_list_selection()
        return True

    def _normalize_selection_after_restore(self, frame_idx: int) -> None:
        coords = DataLoader.get_keypoint_coordinates_by_frame(frame_idx)
        if self.selected_instance not in coords:
            self.selected_instance = None
            self.selected_node = None
            return
        if self.selected_node is not None:
            track, kp = self.selected_node
            if track not in coords or kp not in coords[track]:
                self.selected_node = None

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
            if self._is_history_drag_kind(kind):
                self._begin_frame_edit()
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
                frame_idx = self._current_label_frame()
                if frame_idx is None:
                    raise KeyError
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
                if self._is_history_drag_kind(kind):
                    self._finish_frame_edit(frame_idx)
        except KeyError:
            self._discard_pending_frame_edit()

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
        menu = QMenu(self.video_viewer)
        self._active_menu = menu

        act_undo = menu.addAction("Undo")
        act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        act_undo.setShortcutVisibleInContextMenu(True)

        act_redo = menu.addAction("Redo")
        act_redo.setShortcut(QKeySequence("Ctrl+Shift+Z"))
        act_redo.setShortcutVisibleInContextMenu(True)

        menu.addSeparator()

        act_add = menu.addAction("Add New Instance")
        act_add.setShortcut(QKeySequence("Ctrl+A"))
        act_add.setShortcutVisibleInContextMenu(True)

        act_replace = menu.addAction("Replace Instance")
        act_replace.setShortcuts([QKeySequence("Ctrl+X"), QKeySequence("Ctrl+F")])
        act_replace.setShortcutVisibleInContextMenu(True)

        act_delete = menu.addAction("Delete Instance")
        act_delete.setShortcut(QKeySequence("Ctrl+D"))
        act_delete.setShortcutVisibleInContextMenu(True)

        act_change_num = menu.addMenu("Change Instance Number")
        selected_base_track = (
            DataLoader.get_base_track_name(self.selected_instance)
            if self.selected_instance is not None
            else None
        )
        for nm in range(self.max_animals):
            track_name = self.track_list[nm]
            act_nm = act_change_num.addAction(track_name)
            act_nm.setEnabled(track_name != selected_base_track)
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

        parent_dialog = getattr(self.video_loader, "parent", None)
        menu.addSeparator()
        act_toggle_auto = menu.addAction("Toggle Automatic Labeling")
        act_toggle_auto.setCheckable(True)
        act_toggle_auto.setShortcut(QKeySequence("Ctrl+T"))
        act_toggle_auto.setShortcutVisibleInContextMenu(True)

        act_auto_add = menu.addAction("Automatic Label Addition")
        act_auto_add.setShortcut(QKeySequence("Ctrl+E"))
        act_auto_add.setShortcutVisibleInContextMenu(True)

        act_auto_relabel = menu.addAction("Automatic Re-labeling")
        act_auto_relabel.setShortcut(QKeySequence("Ctrl+R"))
        act_auto_relabel.setShortcutVisibleInContextMenu(True)

        if parent_dialog is not None and hasattr(parent_dialog, "automatic_label_checkbox"):
            act_toggle_auto.setChecked(parent_dialog.automatic_label_checkbox.isChecked())
        else:
            act_toggle_auto.setEnabled(False)
            act_auto_add.setEnabled(False)
            act_auto_relabel.setEnabled(False)

        current_frame = self._current_label_frame()
        if current_frame is None:
            current_frame = -1
        act_undo.setEnabled(self.can_undo_current_frame())
        act_redo.setEnabled(self.can_redo_current_frame())
        act_add.setEnabled(DataLoader.frame_has_capacity_for_new_instance(current_frame))
        act_replace.setEnabled(self.selected_instance is not None)
        act_delete.setEnabled(self.selected_instance is not None)
        act_change_num.setEnabled(self.selected_instance is not None)
        act_vis.setEnabled(self.selected_node is not None)
        
        act_undo.triggered.connect(self.undo_current_frame_edit)
        act_redo.triggered.connect(self.redo_current_frame_edit)
        act_add.triggered.connect(lambda: self._add_new_skeleton_label(context_pos=pos))
        act_replace.triggered.connect(lambda: self._replace_selected_instance(context_pos=pos))
        act_delete.triggered.connect(self._delete_selected_instance)
        act_vis.triggered.connect(self._toggle_selected_node_visibility)
        if parent_dialog is not None:
            act_toggle_auto.triggered.connect(
                lambda _checked=False: parent_dialog.toggle_automatic_labeling()
            )
            act_auto_add.triggered.connect(parent_dialog.run_automatic_label_addition)
            act_auto_relabel.triggered.connect(parent_dialog.run_automatic_relabel)
        
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
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center_norm = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        corner_norms = {
            "top_left": (min_x, min_y),
            "top_right": (max_x, min_y),
            "bottom_left": (min_x, max_y),
            "bottom_right": (max_x, max_y),
        }
        opposite_corners = {
            "top_left": "bottom_right",
            "top_right": "bottom_left",
            "bottom_left": "top_right",
            "bottom_right": "top_left",
        }
        anchor_corner = opposite_corners.get(corner)
        if anchor_corner is None or corner not in corner_norms:
            self._clear_resize_state()
            return

        self._resize_center_norm = center_norm
        self._resize_anchor_norm = corner_norms[anchor_corner]
        self._resize_initial_corner_norm = corner_norms[corner]
        self._resize_source_points = {
            kp: (nx, ny, vis) for kp, (nx, ny, vis) in pts.items()
        }
        self._resize_start_distance_px = 1.0

    def _resize_instance(self, track: str, corner: str, pos: QPoint) -> None:
        if (
            self._resize_anchor_norm is None
            or self._resize_initial_corner_norm is None
            or not self._resize_source_points
        ):
            return

        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1
        anchor_nx, anchor_ny = self._resize_anchor_norm
        start_corner_nx, start_corner_ny = self._resize_initial_corner_norm
        current_nx = (pos.x() - self.video_viewer.translation.x()) / (act * ow)
        current_ny = (pos.y() - self.video_viewer.translation.y()) / (act * oh)
        base_dx = start_corner_nx - anchor_nx
        base_dy = start_corner_ny - anchor_ny

        scale_x = (current_nx - anchor_nx) / base_dx if abs(base_dx) > 1e-6 else 1.0
        scale_y = (current_ny - anchor_ny) / base_dy if abs(base_dy) > 1e-6 else 1.0
        scale_x = max(scale_x, 0.05)
        scale_y = max(scale_y, 0.05)

        for kp, (src_x, src_y, vis) in self._resize_source_points.items():
            dx = src_x - anchor_nx
            dy = src_y - anchor_ny
            scaled_x = anchor_nx + dx * scale_x
            scaled_y = anchor_ny + dy * scale_y
            self.video_viewer.csv_points[track][kp] = (
                max(0.0, min(scaled_x, 1.0)),
                max(0.0, min(scaled_y, 1.0)),
                vis,
            )

    def _clear_resize_state(self) -> None:
        self._resize_center_norm = None
        self._resize_anchor_norm = None
        self._resize_initial_corner_norm = None
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

    
    def _add_new_skeleton_label(self, track_name: str | None = None, context_pos: QPoint | None = None):
        frame_idx = self._current_label_frame()
        if frame_idx is None:
            return

        preferred_track = track_name
        if preferred_track is None and self.selected_instance is not None:
            preferred_track = DataLoader.get_base_track_name(self.selected_instance)

        new_track = DataLoader.resolve_new_instance_track(
            frame_idx,
            preferred_track=preferred_track,
        )
        if new_track is None:
            return

        anchor_norm = None
        if context_pos is not None:
            anchor_norm = self._pos_to_norm(context_pos)
        success = DataLoader.add_skeleton_instance(
            frame_idx=frame_idx,
            track_name=new_track,
            anchor_xy=anchor_norm
        )
        if not success:
            return

        self._refresh_visible_overlay()

    def _replace_selected_instance(self, context_pos: QPoint | None = None):
        if self.selected_instance is None:
            return
        track_name = DataLoader.get_base_track_name(self.selected_instance)
        frame_idx = self._current_label_frame()
        if frame_idx is None:
            return
        if not DataLoader.delete_instance(frame_idx, self.selected_instance):
            return

        self._refresh_visible_overlay()

        self.selected_instance = None
        self.selected_node = None
        self._add_new_skeleton_label(track_name=track_name, context_pos=context_pos)

    def _tracks_in_frame(self, frame_idx: int):
        return set(DataLoader.get_track_keys_for_frame(frame_idx))

    def _delete_selected_instance(self):
        if self.selected_instance is None:
            return
        frame_idx = self._current_label_frame()
        if frame_idx is None:
            return
        track = self.selected_instance
        success = DataLoader.delete_instance(frame_idx, track)
        if not success:
            return
        self._refresh_visible_overlay()
        self.selected_instance = None
        self.selected_node = None

    def _change_instance_number_by_idx(self, idx):
        if idx >= len(self.track_list):
            return
        if DataLoader.get_base_track_name(self.selected_instance) == self.track_list[idx]:
            return
        self._change_instance_number(self.track_list[idx])

    def _change_instance_number(self, new_track: str):
        if self.selected_instance is None:
            return
        if new_track is None:
            from PyQt6.QtWidgets import QInputDialog
            current_base = DataLoader.get_base_track_name(self.selected_instance)
            items = [name for name in self.track_list if name != current_base]
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

        frame_idx = self._current_label_frame()
        if frame_idx is None:
            return
        old_track   = self.selected_instance
        updated_instance_key = DataLoader.swap_or_rename_instance(frame_idx, old_track, new_track)
        if not updated_instance_key:
            return

        self._refresh_visible_overlay()
        self.selected_instance = updated_instance_key
        self.selected_node = None
        self._sync_list_selection()

    def _toggle_selected_node_visibility(self):
        if self.selected_node is None:
            return
        track, kp = self.selected_node
        frame_idx = self._current_label_frame()
        if frame_idx is None:
            return

        cur_vis = 2
        if track in self.video_viewer.csv_points and kp in self.video_viewer.csv_points[track]:
            cur_vis = self.video_viewer.csv_points[track][kp][2]

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
