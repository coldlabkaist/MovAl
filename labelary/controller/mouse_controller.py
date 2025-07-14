from __future__ import annotations
from PyQt6.QtCore import QObject, QEvent, QPoint, Qt, QPointF
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QKeySequence
from PyQt6.QtWidgets import QMenu
from ..IO.data_loader import DataLoader

class MouseController(QObject):
    def __init__(self, video_viewer, kpt_list, parent=None):
        super().__init__(parent)
        self.video_viewer = video_viewer
        self.kpt_list = kpt_list
        self.track_list = video_viewer.current_project.animals_name
        self.max_animals = video_viewer.current_project.num_animals

        # Internal state
        self._dragging = False
        self._last_pos = QPoint()
        self.enable_control = True

        # Selection state: currently selected instance (track) and node (track, keypoint)
        self.selected_instance: str | None = None
        self.selected_node: tuple[str, str] | None = None
        self.new_selection = False

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

            # Determine click target on right-click (for context menu)
            target_track = None
            target_kp = None
            # Check if clicked near a node
            near = self._nearest_csv_kp(pos)
            if near:
                target_track, target_kp = near
            else:
                # If not near a node, check if inside any instance bounding box
                target_track = self._instance_at_point(pos)

            # Update selection based on right-click position
            if target_track is not None:
                # If clicked on a node
                if target_kp is not None:
                    # Select that node and its instance
                    self.selected_instance = target_track
                    self.selected_node = (target_track, target_kp)
                else:
                    # Clicked on instance (bounding box) with no specific node
                    self.selected_instance = target_track
                    self.selected_node = None
                # If right-clicked on background (no target), do not change selection
            else:
                # Clicked on instance (bounding box) with no specific node
                self.selected_instance = None
                self.selected_node = None
            self.video_viewer.update()
            self._sync_list_selection()

            self.show_context_menu(e)

        # ---------- left click ----------
        if e.button() == Qt.MouseButton.LeftButton:
            # Determine click classification for left-click
            near = self._nearest_csv_kp(pos)
            inside_track = None
            if near:
                track, kp = near
                inside_track = track
            else:
                inside_track = self._instance_at_point(pos)

            # Drag priority: if something is selected and click is within that selection
            if self.selected_instance is not None and inside_track == self.selected_instance:
                # If near a node of the selected instance
                if near and near[0] == self.selected_instance:
                    # Select that node (could be same or another node of the instance)
                    track, kp = near
                    self.selected_instance = track
                    self.selected_node = (track, kp)
                    self._dragging = True
                    self.video_viewer.dragging_target = ("csv", track, kp)
                # If click inside selected instance's box but not near any node
                if not near:
                    # Deselect any specific node, keep instance selected
                    self.selected_node = None
                    # Start dragging the whole instance
                    track = self.selected_instance
                    self._dragging = True
                    self.video_viewer.dragging_target = ("instance", track)
                    # Store last position for instance dragging
                    self._last_pos = pos
                self.video_viewer.update()
                self._sync_list_selection()
                return True
            # If not clicking within the currently selected target, proceed with normal classification
            if near:
                # Click on a node (not necessarily of previously selected instance)
                track, kp = near
                # Update selection to this node and instance
                self.selected_instance = track
                self.selected_node = (track, kp)
                self._dragging = True
                self.video_viewer.dragging_target = ("csv", track, kp)
                self.video_viewer.update()
                self._sync_list_selection()
                return True
            if inside_track is not None:
                # Click inside an instance bounding box (no node nearby)
                track = inside_track
                # Update selection to this instance (no specific node)
                self.selected_instance = track
                self.selected_node = None
                # Start dragging the entire instance
                self._dragging = True
                self.video_viewer.dragging_target = ("instance", track)
                self._last_pos = pos
                self.video_viewer.update()
                self._sync_list_selection()
                return True

            # Click on background (no node or instance hit)
            # Clear any existing selection
            if self.selected_instance is not None or self.selected_node is not None:
                self.selected_instance = None
                self.selected_node = None
                self.video_viewer.update()
                self._sync_list_selection()
            # Start panning the background
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

        # Pan the view if dragging background (dragging_target is None)
        if self.video_viewer.dragging_target is None:
            # Compute translation delta
            delta = pos - self._last_pos
            new_tx = self.video_viewer.translation.x() + delta.x()
            new_ty = self.video_viewer.translation.y() + delta.y()
            new_tx, new_ty = self._get_clamped_translation(new_tx, new_ty)
            self.video_viewer.translation = QPoint(new_tx, new_ty)
            self._last_pos = pos

        elif self.video_viewer.dragging_target:
            kind = self.video_viewer.dragging_target[0]
            if kind == "csv":
                # Dragging a single keypoint (node)
                _, track, kp = self.video_viewer.dragging_target
                nx = (pos.x() - self.video_viewer.translation.x()) / (act * self.video_viewer.original_pixmap.width())
                ny = (pos.y() - self.video_viewer.translation.y()) / (act * self.video_viewer.original_pixmap.height())
                # Clamp coordinates between 0 and 1
                nx = max(0.0, min(nx, 1.0))
                ny = max(0.0, min(ny, 1.0))
                self.video_viewer.csv_points[track][kp] = (nx, ny, self.video_viewer.csv_points[track][kp][2])
            elif kind == "instance":
                # Dragging an entire instance (move all keypoints)
                _, track = self.video_viewer.dragging_target
                # Calculate movement delta in normalized coordinates
                dx_norm = (pos.x() - self._last_pos.x()) / (act * self.video_viewer.original_pixmap.width())
                dy_norm = (pos.y() - self._last_pos.y()) / (act * self.video_viewer.original_pixmap.height())
                # Update all keypoints for this track
                for kp, (nx, ny, vis) in self.video_viewer.csv_points.get(track, {}).items():
                    nx_new = max(0.0, min(nx + dx_norm, 1.0))
                    ny_new = max(0.0, min(ny + dy_norm, 1.0))
                    self.video_viewer.csv_points[track][kp] = (nx_new, ny_new, vis)
                # Update last position for next move delta
                self._last_pos = pos
            elif kind == "click":
                # Dragging a miscellaneous clicked point (not part of csv data)
                _, idx = self.video_viewer.dragging_target
                nx = (pos.x() - self.video_viewer.translation.x()) / act
                ny = (pos.y() - self.video_viewer.translation.y()) / act
                self.video_viewer.clicked_points[idx] = (nx, ny)

        # Redraw after movement
        #self.video_viewer._updateTransformed()
        self.video_viewer.update()
        return True

    def _release(self, _: QMouseEvent) -> bool:
        if not self._dragging:
            return False

        # Finalize dragging operations
        if self.video_viewer.dragging_target:
            kind = self.video_viewer.dragging_target[0]
            frame_idx = getattr(self.video_viewer, "current_frame", 0) + 1
            if kind == "csv":
                # Commit single keypoint position to data
                _, track, kp = self.video_viewer.dragging_target
                nx, ny, _ = self.video_viewer.csv_points[track][kp]
                DataLoader.update_point(track, frame_idx, kp, nx, ny)
            elif kind == "instance":
                # Commit all moved keypoints of the instance to data
                _, track = self.video_viewer.dragging_target
                if track in self.video_viewer.csv_points:
                    for kp, (nx, ny, _) in self.video_viewer.csv_points[track].items():
                        DataLoader.update_point(track, frame_idx, kp, nx, ny)
            # No commit needed for "click" points as they are not part of DataLoader

        # Reset dragging state
        self._dragging = False
        self.video_viewer.dragging_target = None
        return True

    def _wheel(self, e: QWheelEvent) -> bool:
        """
        확대·축소용 휠 이벤트 핸들러 (MouseController 전용 개정판).

        • 확대/축소 범위 : current_scale ∈ [1.0, 10.0]
        • 커서 아래 지점 고정
        • 이미지가 화면보다 작아질 때는 자동 센터링
        """
        # 1) 이미지가 없는 경우 처리 생략
        if not self.video_viewer.original_pixmap:
            return False

        # 2) 위젯 좌표계에서의 커서 위치
        cursor_pos = e.position().toPoint()

        # 3) 현재 전체 배율 (base_scale × current_scale)
        old_act = self.video_viewer.base_scale * self.video_viewer.current_scale

        # 4) 커서가 가리키는 (원본 이미지 기준) 정규화 좌표 [0 ~ 1]
        img_rel_x = (cursor_pos.x() - self.video_viewer.translation.x()) / (old_act * self.video_viewer.original_pixmap.width())
        img_rel_y = (cursor_pos.y() - self.video_viewer.translation.y()) / (old_act * self.video_viewer.original_pixmap.height())

        # 5) 스케일 변화량 계산 (각도 델타 ↔ 픽셀 델타 모두 지원)
        delta = e.angleDelta().y() or e.pixelDelta().y()
        factor = 1.1 if delta > 0 else 0.9

        new_scale = max(1.0, min(self.video_viewer.current_scale * factor, 10.0))
        if new_scale == self.video_viewer.current_scale:        # 변경 없으면 즉시 종료
            return False

        # 6) 스케일 반영 후 변환된 Pixmap 갱신
        self.video_viewer.current_scale = new_scale
        self.video_viewer._updateTransformed()                  # → transformed_pixmap 크기 갱신

        new_pw = self.video_viewer.transformed_pixmap.width()
        new_ph = self.video_viewer.transformed_pixmap.height()

        # 7) “커서 고정”을 위한 새 translation 계산
        new_tx = cursor_pos.x() - img_rel_x * new_pw
        new_ty = cursor_pos.y() - img_rel_y * new_ph

        # 8) 화면 밖으로 나가지 않도록 클램핑
        new_tx, new_ty = self._get_clamped_translation(new_tx, new_ty)

        # 9) 상태 반영 및 화면 갱신
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

        print("show context")

        pos = e.pos()
        self.video_viewer._context_click_pos = pos  # store click position for anchor (if needed by add_new_instance)
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

    def _nearest_csv_kp(self, pos: QPoint, thresh: int = 30) -> tuple[str, str] | None:
        """Find the nearest keypoint (node) within `thresh` pixels of the given position."""
        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1

        best = None
        best_d = thresh + 1
        for track, pts in self.video_viewer.csv_points.items():
            for kp, (nx, ny, vis) in pts.items():
                # Convert normalized coordinates to widget coordinates
                px = nx * ow * act + self.video_viewer.translation.x()
                py = ny * oh * act + self.video_viewer.translation.y()
                d = (pos - QPoint(int(px), int(py))).manhattanLength()
                if d < best_d:
                    best = (track, kp)
                    best_d = d
        return best if best_d <= thresh else None

    def _instance_at_point(self, pos: QPoint) -> str | None:
        """Return the track of the instance whose bounding box contains the position (if any)."""
        act = self.video_viewer.base_scale * self.video_viewer.current_scale
        ow = self.video_viewer.original_pixmap.width() if self.video_viewer.original_pixmap else 1
        oh = self.video_viewer.original_pixmap.height() if self.video_viewer.original_pixmap else 1

        for track, pts in self.video_viewer.csv_points.items():
            if not pts:  # skip tracks with no points in current frame
                continue
            # Compute bounding box of all keypoints for this track
            min_x = min(nx * ow * act + self.video_viewer.translation.x() for nx, ny, vis in pts.values())
            max_x = max(nx * ow * act + self.video_viewer.translation.x() for nx, ny, vis in pts.values())
            min_y = min(ny * oh * act + self.video_viewer.translation.y() for nx, ny, vis in pts.values())
            max_y = max(ny * oh * act + self.video_viewer.translation.y() for nx, ny, vis in pts.values())
            # Check if pos lies within this bounding box
            if min_x <= pos.x() <= max_x and min_y <= pos.y() <= max_y:
                return track
        return None

    def _get_clamped_translation(self, new_tx: int, new_ty: int) -> tuple[int, int]:
        vw, vh = self.video_viewer.width(), self.video_viewer.height()
        pw = self.video_viewer.transformed_pixmap.width() if self.video_viewer.transformed_pixmap else 0
        ph = self.video_viewer.transformed_pixmap.height() if self.video_viewer.transformed_pixmap else 0

        if pw >= vw:
            min_x, max_x = vw - pw, 0
        else:
            # Image smaller than view; center it
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
        frame_idx = self.video_viewer.current_frame + 1
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
        return set(df[df["frame.idx"] == frame_idx]["track"].unique())

    def _delete_selected_instance(self):
        if self.selected_instance is None:
            return
        frame_idx = getattr(self.video_viewer, "current_frame", 0) + 1
        track = self.selected_instance
        success = DataLoader.delete_instance(frame_idx, track)
        if not success:
            return
        # Refresh the displayed points for the current frame
        coords = DataLoader.get_keypoint_coordinates_by_frame(frame_idx)
        self.video_viewer.setCSVPoints(coords)
        self.kpt_list.update_list_visibility(coords)
        # Clear selection since instance is deleted
        self.selected_instance = None
        self.selected_node = None

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

        frame_idx   = getattr(self.video_viewer, "current_frame", 0) + 1
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
        frame_idx = getattr(self.video_viewer, "current_frame", 0) + 1

        cur_vis = 2
        df = DataLoader.loaded_data if hasattr(DataLoader, "loaded_data") else None
        if df is not None:
            try:
                cur_vis = df.loc[(df["track"] == track) & (df["frame.idx"] == frame_idx), f"{kp}.visibility"].iat[0]
            except Exception:
                pass

        new_vis = 1 if cur_vis == 2 else 2
        DataLoader.update_kpt_visibility(track, frame_idx, kp, new_vis)
        if track in self.video_viewer.csv_points and kp in self.video_viewer.csv_points[track]:
            nx, ny, _ = self.video_viewer.csv_points[track][kp]
            self.video_viewer.csv_points[track][kp] = (nx, ny, new_vis)
        self.video_viewer.update()
