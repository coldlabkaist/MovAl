from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog,
    QSlider, QListWidget, QFrame, QApplication, QDialog, QListWidgetItem, QTreeWidget, QMessageBox,
    QColorDialog, QTreeWidgetItem, QComboBox, QHeaderView, QStyledItemDelegate,
)
from .gui import UI_LabelaryDialog
from .IO.video_loader import VideoLoader
from .widget.image_label import ClickableImageLabel
from .IO.data_loader import DataLoader
from .IO.save_files import save_modified_data, quick_save_csv, export_current_labels_to_txt_snapshot
from .controller.keyboard_controller import KeyboardController
from .controller.mouse_controller import MouseController
from utils.skeleton import SkeletonModel
from pose.prepare_data import create_online_training_dataset
from pose.thread import TrainThread

from typing import Union, Optional, List
from datetime import datetime
from pathlib import Path
import pandas as pd
import sys

VIDEO_NAME_ROLE = int(Qt.ItemDataRole.UserRole) + 1

class LabelaryDialog(QDialog, UI_LabelaryDialog):
    def __init__(self, project, parent= None):
        super().__init__(parent)
        self.setupUi(self)

        self.project = project
        self._restoring_ui_state = False
        self.shortcuts_enabled = True
        self.is_video_paused = True
        self._clean_loaded_data_snapshot: Optional[pd.DataFrame] = None
        self._allow_dialog_reject = False
        self.auto_label_model = None
        self.auto_label_model_path: Optional[str] = None
        self.auto_label_model_mode: Optional[str] = None
        self.mini_training_thread: Optional[TrainThread] = None
        self.mini_training_run_context: Optional[dict] = None
        self._suppress_mini_training_feedback = False
        self.load_skeleton_model()
        self.load_video_combo()
        self.load_mode_combo()

        self.video_loader = VideoLoader(self, 
                                        self.skeleton_video_viewer, 
                                        self.kpt_list, 
                                        self.frame_slider, 
                                        self.frame_number_label,
                                        self.frame_jump_spin)
        DataLoader.parent = self
        DataLoader.max_animals = self.project.num_animals
        DataLoader.max_instances_per_id = self.project.get_max_instances_per_id()
        DataLoader.animals_name = self.project.animals_name
        self.skeleton_video_viewer.current_project = project

        self.install_controller()

        self.play_button.clicked.connect(self.play_or_pause)
        self.speed_spin.valueChanged.connect(self.set_playback_rate)
        self.frame_slider.valueChanged.connect(self.video_loader.move_to_frame)
        self.frame_jump_spin.valueChanged.connect(self.on_frame_jump_changed)
        self.frame_slider.sliderPressed.connect(self.on_frame_slider_pressed)
        self.frame_slider.sliderReleased.connect(self.on_frame_slider_released)
        self.load_data_button.clicked.connect(self.on_show_clicked)
        self.load_model_button.clicked.connect(self.browse_and_load_model)
        self.automatic_label_checkbox.toggled.connect(self.on_automatic_label_toggled)
        self.mini_training_button.clicked.connect(self.run_mini_training)
        self.skeleton_delay_spin.valueChanged.connect(self.on_skeleton_delay_changed)
        self.kpt_list.currentItemChanged.connect(self.on_keypoint_list_selection_changed)

        self.video_combo.currentIndexChanged.connect(self.update_label_combo)
        self.video_combo.currentIndexChanged.connect(self._on_video_selection_changed)
        self.label_combo.currentIndexChanged.connect(self._on_label_selection_changed)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.file_entry_idx = 0

        self.set_color_combo()
        self.color_combo.currentIndexChanged.connect(self.set_color_mode)
        self._restore_ui_state()

        self.save_csv_button.clicked.connect(self.open_quick_save_dialog)
        self.save_options_button.clicked.connect(self.open_save_dialog)
        self._refresh_model_button_state()
        self._refresh_mini_training_button_state()

    def load_skeleton_model(self):
        self.skeleton = SkeletonModel()
        try:
            self.skeleton.load_from_yaml(self.project.skeleton_yaml)
            DataLoader.load_skeleton_info(self.skeleton)
            self.skeleton_video_viewer.load_skeleton_model(self.skeleton)
        except Exception as e:
            QMessageBox.warning(
                self,
                "Skeleton Load Error",
                f"Skeleton settings file not loaded:\n{e}"
            )
            self.accept()
    
    def install_controller(self):
        mouse_controller = MouseController(self.video_loader, self.skeleton_video_viewer, self.kpt_list)
        self.mouse_controller = mouse_controller
        self.skeleton_video_viewer.mouse_controller = mouse_controller
        self.skeleton_video_viewer.installEventFilter(mouse_controller)
        self.kpt_list.mouse_controller = mouse_controller

        keyboard_controller = KeyboardController(self, self.video_loader, mouse_controller=mouse_controller)
        self.keyboard_controller = keyboard_controller
        QApplication.instance().installEventFilter(keyboard_controller)

    def load_video_combo(self):
        self.video_combo.clear()
        for file_entry in self.project.files:
            video_path = Path(file_entry.video)
            self.video_combo.addItem(video_path.name, video_path)
            index = self.video_combo.count() - 1
            self.video_combo.setItemData(index, file_entry.name, VIDEO_NAME_ROLE)

    def load_mode_combo(self):
        self.mode_combo.clear()
        for display_mode in ["video", "images", "davis", "contour"]:
            self.mode_combo.addItem(display_mode)
        preferred_mode = self.project.get_preferred_frame_mode()
        index = self.mode_combo.findText(preferred_mode, Qt.MatchFlag.MatchExactly)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 0)

    def get_skeleton_frame_delay(self) -> int:
        return int(self.skeleton_delay_spin.value())

    def resolve_label_frame(self, view_frame: int) -> Optional[int]:
        return int(view_frame) - self.get_skeleton_frame_delay()

    def resolve_view_frame(self, label_frame: int) -> Optional[int]:
        view_frame = int(label_frame) + self.get_skeleton_frame_delay()
        if self.video_loader.total_frames > 0 and not (0 <= view_frame < self.video_loader.total_frames):
            return None
        return view_frame

    def _shift_frame_indices_by_current_delay(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int]:
        shifted = df.copy()
        if "frame_idx" not in shifted.columns:
            return shifted, 0

        delay = self.get_skeleton_frame_delay()
        if delay == 0:
            return shifted, 0

        shifted["frame_idx"] = (
            pd.to_numeric(shifted["frame_idx"], errors="coerce").fillna(0).astype(int) + delay
        )
        shifted.reset_index(drop=True, inplace=True)
        return shifted, 0

    def commit_current_delay_to_loaded_data(self) -> int:
        if DataLoader.loaded_data is None:
            return 0

        delay = self.get_skeleton_frame_delay()
        if delay == 0:
            return 0

        shifted, dropped = self._shift_frame_indices_by_current_delay(DataLoader.loaded_data)
        if shifted.empty:
            shifted = shifted.reindex(columns=DataLoader.loaded_data.columns)
            DataLoader.loaded_data = shifted.reset_index(drop=True)
            DataLoader._bump_label_version()
        else:
            DataLoader.loaded_data = DataLoader._normalize_loaded_df(shifted)
            DataLoader._bump_label_version()

        self.skeleton_delay_spin.blockSignals(True)
        self.skeleton_delay_spin.setValue(0)
        self.skeleton_delay_spin.blockSignals(False)
        self.refresh_frame_bound_views()
        self._persist_ui_state(include_frame=True)
        return dropped

    def prompt_close_after_main_window_closed(self) -> None:
        if not self.isVisible():
            return

        reply = QMessageBox.question(
            self,
            "Close Labelary?",
            "MovAl main window has been closed.\n\nDo you want to close Labelary too?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.close()
            return

        self.raise_()
        self.activateWindow()

    def confirm_close_from_main_window(self) -> bool:
        if not self.isVisible():
            return True

        if self.mini_training_thread is not None and self.mini_training_thread.isRunning():
            thread = self.mini_training_thread
            reply = QMessageBox.question(
                self,
                "Mini training in progress",
                "Mini training is currently running.\n\n"
                "Stop training and close MovAl and Labelary?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self.raise_()
                self.activateWindow()
                return False

            self._suppress_mini_training_feedback = True
            thread.stop()
            thread.wait(5000)
            if thread.isRunning():
                self._suppress_mini_training_feedback = False
                QMessageBox.warning(
                    self,
                    "Stop failed",
                    "Mini training is still running, so MovAl cannot be closed yet.",
                )
                self.raise_()
                self.activateWindow()
                return False

        if not self._confirm_discard_unsaved_changes("closing MovAl and Labelary"):
            self.raise_()
            self.activateWindow()
            return False
        return True

    def refresh_frame_bound_views(self) -> None:
        if not getattr(self.skeleton_video_viewer, "video_loaded", False):
            self.skeleton_video_viewer.setCSVPoints({})
            self.kpt_list.clear()
            self.kpt_list.update_list_visibility({})
            return

        label_frame = self.resolve_label_frame(self.video_loader.current_frame)
        coords_dict = (
            DataLoader.get_keypoint_coordinates_by_frame(label_frame)
            if label_frame is not None
            else {}
        )
        self.skeleton_video_viewer.setCSVPoints(coords_dict)
        self.update_keypoint_list()
        self.kpt_list.update_list_visibility(coords_dict)

    def _snapshot_loaded_data(self) -> Optional[pd.DataFrame]:
        if DataLoader.loaded_data is None:
            return None
        df = DataLoader.loaded_data.copy()
        df = df.reset_index(drop=True)
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def mark_loaded_data_clean(self) -> None:
        self._clean_loaded_data_snapshot = self._snapshot_loaded_data()

    def has_unsaved_changes(self) -> bool:
        if self.get_skeleton_frame_delay() != 0:
            return True

        current = self._snapshot_loaded_data()
        baseline = self._clean_loaded_data_snapshot
        if current is None and baseline is None:
            return False
        if current is None or baseline is None:
            return True

        try:
            return not current.equals(baseline)
        except Exception:
            return True

    def _confirm_discard_unsaved_changes(self, action_text: str) -> bool:
        if not self.has_unsaved_changes():
            return True

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Unsaved changes")
        msg_box.setText(
            "There are unsaved changes in Labelary.\n\n"
            f"Do you want to save the current changes before {action_text}?"
        )
        msg_box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        msg_box.setDefaultButton(QMessageBox.StandardButton.Save)

        reply = msg_box.exec()
        if reply == QMessageBox.StandardButton.Save:
            self.open_quick_save_dialog()
            return not self.has_unsaved_changes()
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False

    def update_label_combo(self, video_index = None, set_text = None):
        files = self.project.files
        if not files:
            self.label_combo.clear()
            return

        if video_index is None:
            video_index = self.file_entry_idx
        if not (0 <= int(video_index) < len(files)):
            video_index = 0
        self.file_entry_idx = int(video_index)

        file_entry = files[self.file_entry_idx]
        saved_state = self.project.get_labelary_state()
        self.label_combo.blockSignals(True)
        self.label_combo.clear()

        for csv_path in file_entry.csv:
            p = Path(csv_path)
            self.label_combo.addItem(p.name, p)
        num_csv = len(file_entry.csv)
        for txt_path in file_entry.txt:
            p = Path(txt_path)
            self.label_combo.addItem(p.name, p)
        self.label_combo.addItem("Load inference result", "Load inference result")
        self.label_combo.addItem("Create new label", "Create new label")

        if set_text:
            target_stem = Path(set_text).stem
            default_idx = next(
                (i for i in range(self.label_combo.count())
                if isinstance(self.label_combo.itemData(i), Path)
                    and self.label_combo.itemData(i).stem == target_stem),
                self.label_combo.count() - 1
            )
        elif (
            saved_state.get("video_name") == file_entry.name
            and saved_state.get("label_name")
        ):
            default_idx = self._find_saved_label_index(
                saved_state.get("label_name"),
                saved_state.get("label_type"),
            )
        elif num_csv > 0:
            default_idx = num_csv - 1
        elif file_entry.txt:
            default_idx = 0
        else:
            default_idx = self.label_combo.count() - 1

        self.label_combo.setCurrentIndex(default_idx)
        self.label_combo.blockSignals(False)
        if not self._restoring_ui_state:
            self._persist_ui_state()

    def on_show_clicked(self):
        if getattr(self.skeleton_video_viewer, "video_loaded", False):
            if not self._confirm_discard_unsaved_changes("load another file"):
                return

        video_path = self.video_combo.currentData(Qt.ItemDataRole.UserRole)
        display_mode = self.mode_combo.currentText()
        if not self.video_loader.load_video(video_path, display_mode):
            return
        self.skeleton_video_viewer.video_loaded = True

        label_name = self.label_combo.currentText()
        if label_name == "Create new label":
            self.create_new_label()
        elif label_name == "Load inference result":
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select inference result directory",
                str(Path(self.project.project_dir)/"predicts")
            )
            if not dir_path:
                return
            if not Path(dir_path).exists():
                return
            self.load_txt(dir_path)
        else:
            label_path = Path(self.label_combo.currentData(Qt.ItemDataRole.UserRole))
            if label_path.is_dir():
                if not self.load_txt(label_path):
                    return
            elif label_path.suffix.lower() == ".csv":
                if not self.load_csv(label_path):
                    return
            else:
                QMessageBox.warning(
                    self,
                    "Unsupported Format",
                    f"Unsupported file/folder:\n{label_path}"
                )
                return

        self.mouse_controller.enable_control = True
        self.is_video_paused = True
        self._restore_saved_frame_index()
        self.refresh_frame_bound_views()
        self.mark_loaded_data_clean()
        self.auto_label_current_frame()
        self._persist_ui_state(include_frame=True)

    def load_csv(self, path):
        loaded = DataLoader.load_csv_data(path)
        if not loaded:
            DataLoader.loaded_data = None
            self.skeleton_video_viewer.setCSVPoints({})
            self.kpt_list.clear()
        return loaded

    def load_txt(self, path):
        inference_mode = self.label_combo.currentText() == "Load inference result"
        loaded = DataLoader.load_txt_data(path, inference_mode=inference_mode)
        if not loaded:
            DataLoader.loaded_data = None
            self.skeleton_video_viewer.setCSVPoints({})
            self.kpt_list.clear()
        return loaded

    def create_new_label(self):
        DataLoader.create_new_data()

    def play_or_pause(self):
        self.is_video_paused = self.video_loader.toggle_playback()
        self.mouse_controller.enable_control = self.is_video_paused
        self.speed_spin.setEnabled(self.mouse_controller.enable_control)
        self.frame_jump_spin.setEnabled(self.mouse_controller.enable_control)
    
    def set_playback_rate(self):
        self.video_loader.play_rate = self.speed_spin.value()

    def on_frame_slider_pressed(self):
        if not self.is_video_paused:
            self.video_loader.toggle_playback()
        self.video_loader.move_to_frame(self.frame_slider.value())

    def on_frame_slider_released(self):
        if not self.is_video_paused:
            self.video_loader.toggle_playback()
        self.auto_label_current_frame()

    def on_frame_jump_changed(self, frame_idx: int):
        if self.video_loader.total_frames <= 0:
            return
        if frame_idx == self.video_loader.current_frame:
            return
        if not self.is_video_paused:
            self.is_video_paused = self.video_loader.toggle_playback()
            self.mouse_controller.enable_control = self.is_video_paused
            self.speed_spin.setEnabled(self.mouse_controller.enable_control)
            self.frame_jump_spin.setEnabled(self.mouse_controller.enable_control)
        self.video_loader.move_to_frame(frame_idx, force=True)
        self.auto_label_current_frame()

    def update_csv_points_on_image(self):
        self.refresh_frame_bound_views()
        
    def update_keypoint_list(self):
        self.kpt_list.clear()
        visible_tracks = []
        label_frame = self.resolve_label_frame(self.video_loader.current_frame)
        if DataLoader.loaded_data is not None and label_frame is not None:
            visible_tracks = DataLoader.get_track_keys_for_frame(label_frame)
        self.kpt_list.build(
            self.project.animals_name,
            visible_tracks,
            DataLoader.kp_order,
            self.skeleton,
        )

    def set_color_combo(self):
        self.color_combo.clear()
        self.color_combo.addItem("cutie_light")
        self.color_combo.addItem("cutie_dark")
        self.color_combo.addItem("white")
        self.color_combo.addItem("black")

    def on_keypoint_list_selection_changed(self, current, _previous):
        if self.kpt_list.is_syncing_selection():
            return
        if self.mouse_controller is None:
            return

        instance_key, kp_name = self.kpt_list.get_item_selection(current)
        if instance_key is None:
            return

        self.mouse_controller.selected_instance = instance_key
        self.mouse_controller.selected_node = (
            (instance_key, kp_name) if kp_name else None
        )
        self.mouse_controller._sync_list_selection()
        self.skeleton_video_viewer.update()
        saved_color = self.project.get_labelary_state().get("color_mode")
        index = self.color_combo.findText(saved_color, Qt.MatchFlag.MatchExactly)
        self.color_combo.setCurrentIndex(index if index >= 0 else 0)

    def set_color_mode(self):
        color_mode = self.color_combo.currentText()
        self.skeleton_video_viewer.set_skeleton_color_mode(color_mode)
        if not self._restoring_ui_state:
            self._persist_ui_state()

    def _restore_ui_state(self) -> None:
        self._restoring_ui_state = True
        try:
            if self.video_combo.count() == 0:
                return

            saved_state = self.project.get_labelary_state()
            saved_video_name = saved_state.get("video_name")
            if saved_video_name:
                video_index = self._find_video_index(saved_video_name)
            else:
                video_index = 0

            self.file_entry_idx = video_index
            self.video_combo.setCurrentIndex(video_index)
            self.update_label_combo(video_index=video_index)

            saved_color = saved_state.get("color_mode")
            if saved_color:
                color_index = self.color_combo.findText(saved_color, Qt.MatchFlag.MatchExactly)
                if color_index >= 0:
                    self.color_combo.setCurrentIndex(color_index)
            self.skeleton_delay_spin.setValue(int(saved_state.get("skeleton_frame_delay", 0) or 0))
        finally:
            self._restoring_ui_state = False

        self.set_color_mode()

    def _find_video_index(self, video_name: str) -> int:
        for index in range(self.video_combo.count()):
            if self.video_combo.itemData(index, VIDEO_NAME_ROLE) == video_name:
                return index
        return 0

    def _find_saved_label_index(self, label_name: Optional[str], label_type: Optional[str]) -> int:
        if not label_name:
            return self.label_combo.count() - 1

        for index in range(self.label_combo.count()):
            data = self.label_combo.itemData(index, Qt.ItemDataRole.UserRole)
            if not isinstance(data, Path):
                continue
            if data.name != label_name:
                continue
            if label_type == "txt" and data.is_dir():
                return index
            if label_type == "csv" and data.suffix.lower() == ".csv":
                return index
            if label_type is None:
                return index
        return self.label_combo.count() - 1

    def _current_video_name(self) -> Optional[str]:
        video_name = self.video_combo.currentData(VIDEO_NAME_ROLE)
        return str(video_name) if video_name else None

    def _current_label_state(self) -> tuple[Optional[str], Optional[str]]:
        data = self.label_combo.currentData(Qt.ItemDataRole.UserRole)
        if isinstance(data, Path):
            if data.is_dir():
                return data.name, "txt"
            if data.suffix.lower() == ".csv":
                return data.name, "csv"
        return None, None

    def _persist_ui_state(self, *, include_frame: bool = False) -> None:
        if self._restoring_ui_state:
            return

        label_name, label_type = self._current_label_state()
        frame_index = None
        if include_frame and getattr(self.skeleton_video_viewer, "video_loaded", False):
            frame_index = self.video_loader.current_frame

        self.project.save_labelary_state(
            video_name=self._current_video_name(),
            label_name=label_name,
            label_type=label_type,
            frame_index=frame_index,
            skeleton_frame_delay=self.get_skeleton_frame_delay(),
            color_mode=self.color_combo.currentText() if self.color_combo.count() else None,
            mode=self.mode_combo.currentText(),
        )

    def _restore_saved_frame_index(self) -> None:
        saved_state = self.project.get_labelary_state()
        if saved_state.get("video_name") != self._current_video_name():
            return
        frame_index = int(saved_state.get("frame_index", 0) or 0)
        if 0 <= frame_index < self.video_loader.total_frames:
            self.video_loader.move_to_frame(frame_index, force=True)

    def _on_video_selection_changed(self, index: int) -> None:
        if index < 0:
            return
        self.file_entry_idx = index
        if not self._restoring_ui_state:
            self._persist_ui_state()

    def _on_label_selection_changed(self, index: int) -> None:
        if index < 0:
            return
        if not self._restoring_ui_state:
            self._persist_ui_state()

    def _on_mode_changed(self, mode: str) -> None:
        if self._restoring_ui_state:
            return
        self.project.set_preferred_frame_mode(mode)
        self._refresh_mini_training_button_state()

    def on_skeleton_delay_changed(self, _value: int) -> None:
        if getattr(self.skeleton_video_viewer, "video_loaded", False):
            self.refresh_frame_bound_views()
        if not self._restoring_ui_state:
            self._persist_ui_state(include_frame=True)

    def open_save_dialog(self):
        self.shortcuts_enabled = False
        try:
            save_modified_data(self)
        finally:
            self.shortcuts_enabled = True

    def open_quick_save_dialog(self):
        self.shortcuts_enabled = False
        try:
            quick_save_csv(self)
        finally:
            self.shortcuts_enabled = True

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def reject(self) -> None:
        if self._allow_dialog_reject:
            super().reject()
            return

        if not self._confirm_discard_unsaved_changes("close Labelary"):
            return

        self._persist_ui_state(include_frame=True)
        self._allow_dialog_reject = True
        try:
            super().reject()
        finally:
            self._allow_dialog_reject = False

    def on_automatic_label_toggled(self, checked: bool):
        if not checked:
            return
        if self.auto_label_model is None:
            QMessageBox.warning(self, "Model not loaded", "Load a model before enabling automatic labeling.")
            self.automatic_label_checkbox.blockSignals(True)
            self.automatic_label_checkbox.setChecked(False)
            self.automatic_label_checkbox.blockSignals(False)
            return
        self.auto_label_current_frame()

    def browse_model(self):
        start_dir = self._default_model_dir()
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO pose model",
            str(start_dir),
            "PyTorch model (*.pt);;All Files (*)"
        )
        if not model_path:
            return
        self._set_model_path_display(model_path)

    def browse_and_load_model(self):
        if self.auto_label_model is not None and self.auto_label_model_path is not None:
            self.unload_model()
            return

        raw_path = self._model_path_text()
        if raw_path:
            model_path = Path(raw_path).expanduser()
            if model_path.exists() and model_path.is_file():
                self.load_model()
                return

        self.browse_model()
        if self._model_path_text():
            self.load_model()

    def load_model(self):
        raw_path = self._model_path_text()
        if not raw_path:
            QMessageBox.warning(self, "No model selected", "Select a model file first.")
            return

        model_path = Path(raw_path).expanduser()
        if not model_path.exists() or not model_path.is_file():
            QMessageBox.warning(self, "Invalid model path", f"Model file not found:\n{model_path}")
            return

        try:
            from ultralytics import YOLO
        except ImportError:
            QMessageBox.critical(
                self,
                "Ultralytics not installed",
                "The ultralytics package is required to load YOLO models."
            )
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            model = YOLO(str(model_path))
            model_task = getattr(model, "task", None)
            if model_task not in (None, "pose"):
                raise ValueError(f"Expected a pose model, but got task='{model_task}'.")
        except Exception as e:
            QMessageBox.critical(self, "Model load failed", f"Failed to load model:\n{e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        resolved_path = str(model_path.resolve())
        self.auto_label_model = model
        self.auto_label_model_path = resolved_path
        self.auto_label_model_mode = self.mode_combo.currentText()
        self._set_model_path_display(resolved_path)
        self._refresh_model_button_state()

        QMessageBox.information(
            self,
            "Model loaded",
            f"Loaded model:\n{resolved_path}\n\nMode at load time: {self.auto_label_model_mode}"
        )
        self.automatic_label_checkbox.setChecked(True)
        self.auto_label_current_frame()

    def unload_model(self):
        self.auto_label_model = None
        self.auto_label_model_path = None
        self.auto_label_model_mode = None
        self.automatic_label_checkbox.blockSignals(True)
        self.automatic_label_checkbox.setChecked(False)
        self.automatic_label_checkbox.blockSignals(False)
        self._set_model_path_display("")
        self._refresh_model_button_state()

    def on_model_path_changed(self, text: str):
        new_path = text.strip()
        if self.auto_label_model_path is None:
            self._refresh_model_button_state()
            return

        try:
            current_path = str(Path(new_path).expanduser().resolve()) if new_path else ""
        except Exception:
            current_path = new_path

        if current_path != self.auto_label_model_path:
            self.auto_label_model = None
            self.auto_label_model_path = None
            self.auto_label_model_mode = None
        self._refresh_model_button_state()

    def _model_path_text(self) -> str:
        return self.model_path_edit.text().strip()

    def _set_model_path_display(self, text: str) -> None:
        self.model_path_edit.setText(text)
        self.model_path_edit.setToolTip(text)
        self.on_model_path_changed(text)

    def _default_model_dir(self) -> Path:
        candidates = [
            Path(self.project.project_dir) / "runs",
            Path.cwd() / "models",
            Path(self.project.project_dir),
            Path.cwd(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path.cwd()

    def _refresh_model_button_state(self):
        loaded = self.auto_label_model is not None and self.auto_label_model_path is not None
        self.load_model_button.setText("Cancel Model Load" if loaded else "Browse/Load Model")
        if loaded:
            self.load_model_button.setToolTip(
                f"Loaded for mode '{self.auto_label_model_mode}': {self.auto_label_model_path}\n"
                "Click to unload the current model."
            )
        else:
            self.load_model_button.setToolTip("")
        self._refresh_mini_training_button_state()

    def _refresh_mini_training_button_state(self):
        if self.mini_training_thread is not None and self.mini_training_thread.isRunning():
            self.mini_training_button.setEnabled(False)
            self.mini_training_button.setText("Mini Training...")
        else:
            self.mini_training_button.setEnabled(True)
            self.mini_training_button.setText("Run Mini Training")

        frame_mode = self.video_loader.frame_display_mode if getattr(self.skeleton_video_viewer, "video_loaded", False) else self.mode_combo.currentText()
        self.mini_training_button.setToolTip(
            "Export current in-memory labels to a timestamped snapshot under runs/, "
            "build a separate online dataset, run short fine-tuning, and hot-load the resulting best.pt "
            f"using the current frame mode '{frame_mode}'."
        )

    def _resolve_base_model_path(self) -> Optional[Path]:
        candidates = []
        if self.auto_label_model_path:
            candidates.append(self.auto_label_model_path)

        text_path = self._model_path_text()
        if text_path:
            candidates.append(text_path)

        for raw_path in candidates:
            try:
                model_path = Path(raw_path).expanduser().resolve()
            except Exception:
                continue
            if model_path.exists() and model_path.is_file():
                return model_path
        return None

    def _current_frame_mode(self) -> str:
        if getattr(self.skeleton_video_viewer, "video_loaded", False):
            return self.video_loader.frame_display_mode
        return self.mode_combo.currentText()

    def _write_mini_training_config(self, dataset_dir: Path, run_name: str) -> Path:
        target_config_path = Path(self.project.project_dir) / "runs" / f"{run_name}_config.yaml"
        return self.project.write_training_config_yaml(
            dataset_dir=dataset_dir,
            target_path=target_config_path,
        )

    def run_mini_training(self):
        if self.mini_training_thread is not None and self.mini_training_thread.isRunning():
            return

        if DataLoader.loaded_data is None or DataLoader.loaded_data.empty:
            QMessageBox.warning(self, "No labels loaded", "Load and review labels before starting mini training.")
            return

        model_path = self._resolve_base_model_path()
        if model_path is None:
            QMessageBox.warning(self, "No model selected", "Load a base model or choose a valid model file first.")
            return

        try:
            current_video = self.video_combo.currentData(Qt.ItemDataRole.UserRole)
            if current_video is None:
                raise ValueError("Current video is not selected.")

            current_video_name = self._current_video_name()
            if not current_video_name:
                raise ValueError("Current project video name is not available.")
            snapshot_dir = export_current_labels_to_txt_snapshot(self)
            dataset_dir, split_counts = create_online_training_dataset(
                self.project,
                frame_type=self._current_frame_mode(),
                label_dirs={current_video_name: snapshot_dir},
            )

            run_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            run_name = f"mini_training_{run_stamp}"
            config_path = self._write_mini_training_config(dataset_dir, run_name)
            output_dir = Path(self.project.project_dir) / "runs" / run_name
        except Exception as e:
            QMessageBox.critical(self, "Mini training setup failed", f"Failed to prepare training inputs:\n{e}")
            return

        command = [
            "yolo",
            "pose",
            "train",
            f"model={model_path.as_posix()}",
            f"data={config_path.as_posix()}",
            f"epochs={int(self.mini_training_epochs_spin.value())}",
            f"project={(Path(self.project.project_dir) / 'runs').as_posix()}",
            f"name={run_name}",
            "exist_ok=False",
        ]

        self.mini_training_run_context = {
            "run_name": run_name,
            "output_dir": output_dir,
            "dataset_dir": dataset_dir,
            "snapshot_dir": snapshot_dir,
            "config_path": config_path,
            "split_counts": split_counts,
        }

        self.mini_training_thread = TrainThread(command)
        self.mini_training_thread.finished_signal.connect(self.on_mini_training_finished)
        self._refresh_mini_training_button_state()
        self.mini_training_thread.start()

        QMessageBox.information(
            self,
            "Mini training started",
            "Started quick fine-tuning with the current reviewed labels.\n\n"
            f"Snapshot: {snapshot_dir}\n"
            f"Dataset: {dataset_dir}\n"
            f"Train/Val/Test: {split_counts['train']}/{split_counts['val']}/{split_counts['test']}\n"
            f"Output: {output_dir}"
        )

    def on_mini_training_finished(self):
        context = self.mini_training_run_context or {}
        thread = self.mini_training_thread
        was_stopped = bool(thread and thread.was_stopped)
        self.mini_training_thread = None
        self._refresh_mini_training_button_state()

        if was_stopped:
            if not self._suppress_mini_training_feedback:
                QMessageBox.information(
                    self,
                    "Mini training stopped",
                    "Mini training was stopped before completion.",
                )
            self._suppress_mini_training_feedback = False
            return

        if self._suppress_mini_training_feedback:
            self._suppress_mini_training_feedback = False
            return

        best_model_path = Path(context.get("output_dir", "")) / "weights" / "best.pt"
        if not best_model_path.exists():
            QMessageBox.critical(
                self,
                "Mini training failed",
                "Training finished, but best.pt was not created.\n"
                f"Expected path:\n{best_model_path}"
            )
            return

        self._set_model_path_display(str(best_model_path.resolve()))
        self.load_model()

    def auto_label_current_frame(self):
        if not self.automatic_label_checkbox.isChecked():
            return
        if self.auto_label_model is None:
            return
        if not getattr(self.skeleton_video_viewer, "video_loaded", False):
            return

        current_mode = self.video_loader.frame_display_mode
        if self.auto_label_model_mode and current_mode != self.auto_label_model_mode:
            QMessageBox.warning(
                self,
                "Mode mismatch",
                "The loaded model was loaded for a different display mode.\n"
                "Reload the model for the current mode before using automatic labeling."
            )
            self.automatic_label_checkbox.blockSignals(True)
            self.automatic_label_checkbox.setChecked(False)
            self.automatic_label_checkbox.blockSignals(False)
            return

        frame_idx = self.resolve_label_frame(self.video_loader.current_frame)
        if frame_idx is None:
            return
        if DataLoader.frame_has_labels(frame_idx):
            return

        frame_source = self.video_loader.get_current_frame_source()
        if frame_source is None:
            return

        try:
            instances = self.predict_current_frame(frame_source)
        except Exception as e:
            QMessageBox.critical(self, "Auto labeling failed", f"Failed to run inference:\n{e}")
            return

        if not instances:
            return

        if DataLoader.add_auto_labeled_frame(frame_idx, instances):
            self.update_csv_points_on_image()
            self.skeleton_video_viewer.update()
            self.kpt_list.update()

    def predict_current_frame(self, frame_source) -> list[dict]:
        confidence_threshold = float(self.auto_label_confidence_spin.value())
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = self.auto_label_model.predict(
                source=frame_source,
                conf=confidence_threshold,
                verbose=False,
                save=False,
            )
        finally:
            QApplication.restoreOverrideCursor()

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or boxes.cls is None or keypoints.xyn is None:
            return []

        cls_ids = boxes.cls.cpu().tolist()
        det_scores = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(cls_ids)
        keypoint_xy = keypoints.xyn.cpu().tolist()
        keypoint_conf_tensor = getattr(keypoints, "conf", None)
        keypoint_conf = keypoint_conf_tensor.cpu().tolist() if keypoint_conf_tensor is not None else None

        if len(keypoint_xy) != len(cls_ids):
            return []

        expected_kpts = len(DataLoader.kp_order)
        detections_by_class: dict[int, list[dict]] = {}
        per_class_limit = self.project.get_max_instances_per_id()
        for det_idx, cls_val in enumerate(cls_ids):
            class_idx = int(cls_val)
            if not (0 <= class_idx < len(self.project.animals_name)):
                continue

            kp_xy = keypoint_xy[det_idx]
            if len(kp_xy) != expected_kpts:
                raise ValueError(
                    f"Model predicted {len(kp_xy)} keypoints, but the project expects {expected_kpts}."
                )

            score = float(det_scores[det_idx]) if det_idx < len(det_scores) else 0.0
            if score < confidence_threshold:
                continue

            kp_conf_row = None
            if keypoint_conf is not None and det_idx < len(keypoint_conf):
                kp_conf_row = keypoint_conf[det_idx]

            kp_map: dict[str, tuple[float, float, int]] = {}
            for kp_idx, kp_name in enumerate(DataLoader.kp_order):
                x, y = kp_xy[kp_idx]
                x = max(0.0, min(float(x), 1.0))
                y = max(0.0, min(float(y), 1.0))
                conf = None
                if kp_conf_row is not None and kp_idx < len(kp_conf_row):
                    conf = kp_conf_row[kp_idx]
                vis = 2 if conf is None or float(conf) > 0.0 else 1
                kp_map[kp_name] = (x, y, vis)

            detections_by_class.setdefault(class_idx, []).append({
                "score": score,
                "track": self.project.animals_name[class_idx],
                "keypoints": kp_map,
            })

        instances: list[dict] = []
        for class_idx in sorted(detections_by_class):
            sorted_items = sorted(
                detections_by_class[class_idx],
                key=lambda item: item["score"],
                reverse=True,
            )[:per_class_limit]
            for slot_idx, item in enumerate(sorted_items, start=1):
                record = {
                    "track": item["track"],
                    "keypoints": item["keypoints"],
                    "instance_score": item["score"],
                }
                if per_class_limit > 1:
                    record["instance_id"] = slot_idx
                instances.append(record)

        return instances

    def closeEvent(self, event) -> None:
        if self.mini_training_thread is not None and self.mini_training_thread.isRunning():
            thread = self.mini_training_thread
            reply = QMessageBox.question(
                self,
                "Mini training in progress",
                "Mini training is currently running.\n\n"
                "Stop training and close Labelary?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self._suppress_mini_training_feedback = True
            thread.stop()
            thread.wait(5000)
            if thread.isRunning():
                self._suppress_mini_training_feedback = False
                QMessageBox.warning(
                    self,
                    "Stop failed",
                    "Mini training is still running, so Labelary cannot be closed yet.",
                )
                event.ignore()
                return

        if not self._confirm_discard_unsaved_changes("close Labelary"):
            event.ignore()
            return

        self._persist_ui_state(include_frame=True)
        self._allow_dialog_reject = True
        try:
            super().closeEvent(event)
        finally:
            self._allow_dialog_reject = False

def run_labelary_with_project(current_project, parent=None):
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = LabelaryDialog(current_project, parent) 
    dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    return dlg
