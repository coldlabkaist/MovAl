from PyQt6.QtCore import Qt, QTimer, QThread, QEvent, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon, QPainter, QPen
from PyQt6.QtWidgets import (
    QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog,
    QSlider, QListWidget, QFrame, QApplication, QDialog, QListWidgetItem, QTreeWidget, QMessageBox,
    QColorDialog, QTreeWidgetItem, QComboBox, QHeaderView, QStyledItemDelegate, QPlainTextEdit,
    QAbstractSpinBox, QLineEdit,
)
from .gui import UI_LabelaryDialog
from .IO.video_loader import VideoLoader
from .widget.image_label import ClickableImageLabel
from .IO.data_loader import DataLoader
from .IO.save_files import save_modified_data, quick_save_csv, export_loaded_data_to_txt_dir
from .controller.keyboard_controller import KeyboardController
from .controller.mouse_controller import MouseController
from utils.skeleton import SkeletonModel
from pose.prepare_data import create_online_training_dataset
from pose.thread import TrainThread

from typing import Union, Optional, List
from datetime import datetime
from pathlib import Path
import gc
import pandas as pd
import sys

VIDEO_NAME_ROLE = int(Qt.ItemDataRole.UserRole) + 1
LABEL_ACTION_LOAD_INFERENCE_TXT = "action:load_inference_txt"
LABEL_ACTION_LOAD_INFERENCE_CSV = "action:load_inference_csv"
LABEL_ACTION_CREATE_NEW = "action:create_new_label"


class MiniTrainingLogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mini Training")
        self.resize(760, 460)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Idle")
        status_font = QFont()
        status_font.setPointSize(11)
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        layout.addWidget(self.status_label)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.log_view = QPlainTextEdit(self)
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, 1)

        button_row = QHBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.hide_button = QPushButton("Hide")
        self.hide_button.clicked.connect(self.hide)
        button_row.addWidget(self.stop_button)
        button_row.addStretch(1)
        button_row.addWidget(self.hide_button)
        layout.addLayout(button_row)

    def reset(self) -> None:
        self.status_label.setText("Preparing mini training...")
        self.summary_label.setText("")
        self.log_view.clear()
        self.stop_button.setEnabled(True)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(text)

    def append_log(self, text: str) -> None:
        clean = str(text).rstrip()
        if not clean:
            return
        self.log_view.appendPlainText(clean)
        scroll_bar = self.log_view.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def closeEvent(self, event) -> None:
        self.hide()
        event.ignore()


class MiniTrainingSetupWorker(QThread):
    progress = pyqtSignal(int, int, str)
    success = pyqtSignal(dict)
    cancelled = pyqtSignal()
    failure = pyqtSignal(str)

    def __init__(
        self,
        project,
        snapshot_df: pd.DataFrame,
        current_video_name: str,
        frame_mode: str,
        run_stamp: str,
    ):
        super().__init__()
        self.project = project
        self.snapshot_df = snapshot_df
        self.current_video_name = str(current_video_name)
        self.frame_mode = str(frame_mode)
        self.run_stamp = str(run_stamp)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True
        self.requestInterruption()

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested or self.isInterruptionRequested()

    def _raise_if_cancelled(self) -> None:
        if self._is_cancel_requested():
            raise InterruptedError()

    def _report_progress(self, done: int, total: int, message: str) -> None:
        self.progress.emit(int(done), int(total), str(message))

    def run(self) -> None:
        try:
            self._raise_if_cancelled()

            run_name = f"mini_training_{self.run_stamp}"
            project_dir = Path(self.project.project_dir)
            snapshot_dir = (
                project_dir
                / "runs"
                / "online_label_exports"
                / self.current_video_name
                / f"txt_snapshot_{self.run_stamp}"
            )

            self.progress.emit(0, 0, "Exporting reviewed labels snapshot...")
            export_loaded_data_to_txt_dir(
                snapshot_dir,
                df=self.snapshot_df,
                clear_existing=True,
            )

            self._raise_if_cancelled()
            self.progress.emit(0, 0, "Preparing training dataset...")
            dataset_dir, split_counts = create_online_training_dataset(
                self.project,
                frame_type=self.frame_mode,
                label_dirs={self.current_video_name: snapshot_dir},
                progress_callback=self._report_progress,
                should_cancel=self._is_cancel_requested,
            )

            self._raise_if_cancelled()
            self.progress.emit(0, 0, "Writing YOLO config...")
            config_path = self.project.write_training_config_yaml(
                dataset_dir=dataset_dir,
                target_path=project_dir / "runs" / f"{run_name}_config.yaml",
            )

            self.success.emit(
                {
                    "run_name": run_name,
                    "output_dir": project_dir / "runs" / run_name,
                    "dataset_dir": dataset_dir,
                    "snapshot_dir": snapshot_dir,
                    "config_path": config_path,
                    "split_counts": split_counts,
                }
            )
        except InterruptedError:
            self.cancelled.emit()
        except Exception as err:
            self.failure.emit(str(err))

class LabelaryDialog(QDialog, UI_LabelaryDialog):
    def __init__(self, project, parent= None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
        )

        self.project = project
        self._restoring_ui_state = False
        self.shortcuts_enabled = True
        self.is_video_paused = True
        self._clean_loaded_data_snapshot: Optional[pd.DataFrame] = None
        self._editor_focus_snapshots: dict[object, object] = {}
        self._save_operation_active = False
        self._allow_dialog_reject = False
        self.auto_label_model = None
        self.auto_label_model_path: Optional[str] = None
        self.auto_label_model_mode: Optional[str] = None
        self.mini_training_setup_worker: Optional[MiniTrainingSetupWorker] = None
        self.mini_training_thread: Optional[TrainThread] = None
        self.mini_training_run_context: Optional[dict] = None
        self.mini_training_log_dialog: Optional[MiniTrainingLogDialog] = None
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
        self.minimize_button.clicked.connect(self.showMinimized)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
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
        self._install_editor_focus_release_handlers()

        self.save_csv_button.clicked.connect(self.open_quick_save_dialog)
        self.save_options_button.clicked.connect(self.open_save_dialog)
        self._refresh_model_button_state()
        self._refresh_mini_training_button_state()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _refresh_fullscreen_button_text(self) -> None:
        if self.isFullScreen():
            self.fullscreen_button.setText("Exit Full Screen")
        else:
            self.fullscreen_button.setText("Full Screen")

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
        self._refresh_fullscreen_button_text()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._refresh_fullscreen_button_text()

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

    def _iter_editor_focus_widgets(self) -> list[object]:
        widgets: list[object] = [
            self.model_path_edit,
            self.auto_label_confidence_spin,
            self.mini_training_epochs_spin,
            self.speed_spin,
            self.frame_jump_spin,
            self.skeleton_delay_spin,
        ]
        spin_editors = [
            widget.lineEdit()
            for widget in widgets
            if isinstance(widget, QAbstractSpinBox) and widget.lineEdit() is not None
        ]
        return widgets + spin_editors

    def _install_editor_focus_release_handlers(self) -> None:
        for widget in self._iter_editor_focus_widgets():
            widget.installEventFilter(self)

    def _resolve_editor_focus_target(self, obj) -> Optional[object]:
        if isinstance(obj, QAbstractSpinBox):
            return obj
        if isinstance(obj, QLineEdit):
            parent = obj.parentWidget()
            if isinstance(parent, QAbstractSpinBox):
                return parent
            return obj
        return None

    def _snapshot_editor_focus_value(self, editor) -> None:
        if isinstance(editor, QAbstractSpinBox):
            self._editor_focus_snapshots[editor] = editor.value()
        elif isinstance(editor, QLineEdit):
            self._editor_focus_snapshots[editor] = editor.text()

    def _restore_editor_focus_value(self, editor) -> None:
        if editor not in self._editor_focus_snapshots:
            return
        snapshot = self._editor_focus_snapshots[editor]
        if isinstance(editor, QAbstractSpinBox):
            editor.setValue(snapshot)
        elif isinstance(editor, QLineEdit):
            editor.setText(str(snapshot))

    def _commit_editor_focus_value(self, editor) -> None:
        if isinstance(editor, QAbstractSpinBox):
            editor.interpretText()
        self._snapshot_editor_focus_value(editor)

    def _release_editor_focus(self, editor) -> None:
        editor.clearFocus()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.OtherFocusReason)

    def eventFilter(self, obj, event):
        editor = self._resolve_editor_focus_target(obj)
        if editor is not None:
            if event.type() == QEvent.Type.FocusIn:
                self._snapshot_editor_focus_value(editor)
            elif event.type() == QEvent.Type.KeyPress:
                if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self._commit_editor_focus_value(editor)
                    self._release_editor_focus(editor)
                    return True
                if event.key() == Qt.Key.Key_Escape:
                    self._restore_editor_focus_value(editor)
                    self._release_editor_focus(editor)
                    return True
        return super().eventFilter(obj, event)
    
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

        if self._mini_training_is_active():
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
            if not self._stop_active_mini_training():
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
            if hasattr(self, "mouse_controller"):
                self.mouse_controller.clear_edit_history_if_frame_changed(None)
            self.skeleton_video_viewer.setCSVPoints({})
            self.kpt_list.clear()
            self.kpt_list.update_list_visibility({})
            return

        label_frame = self.resolve_label_frame(self.video_loader.current_frame)
        if hasattr(self, "mouse_controller"):
            self.mouse_controller.clear_edit_history_if_frame_changed(label_frame)
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
        self.label_combo.addItem("Load inference TXT result", LABEL_ACTION_LOAD_INFERENCE_TXT)
        self.label_combo.addItem("Load inference CSV result", LABEL_ACTION_LOAD_INFERENCE_CSV)
        self.label_combo.addItem("Create new label", LABEL_ACTION_CREATE_NEW)

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

        label_action = self.label_combo.currentData(Qt.ItemDataRole.UserRole)
        if label_action == LABEL_ACTION_CREATE_NEW:
            self.create_new_label()
            self._apply_loaded_label_delay(inference_mode=False)
        elif label_action == LABEL_ACTION_LOAD_INFERENCE_TXT:
            dir_path = QFileDialog.getExistingDirectory(
                self,
                "Select inference TXT result directory",
                str(Path(self.project.project_dir)/"predicts")
            )
            if not dir_path:
                return
            if not Path(dir_path).exists():
                return
            if not self.load_txt(dir_path, inference_mode=True):
                return
        elif label_action == LABEL_ACTION_LOAD_INFERENCE_CSV:
            csv_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select inference CSV result",
                str(Path(self.project.project_dir) / "predicts"),
                "CSV Files (*.csv)",
            )
            if not csv_path:
                return
            if not Path(csv_path).exists():
                return
            if not self.load_csv(csv_path, inference_mode=True):
                return
        else:
            label_path = Path(self.label_combo.currentData(Qt.ItemDataRole.UserRole))
            if label_path.is_dir():
                if not self.load_txt(label_path, inference_mode=False):
                    return
            elif label_path.suffix.lower() == ".csv":
                if not self.load_csv(label_path, inference_mode=False):
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

    def _apply_loaded_label_delay(self, *, inference_mode: bool) -> None:
        delay_value = -1 if inference_mode else 0
        self.skeleton_delay_spin.blockSignals(True)
        self.skeleton_delay_spin.setValue(delay_value)
        self.skeleton_delay_spin.blockSignals(False)

    def load_csv(self, path, *, inference_mode: bool = False):
        loaded = DataLoader.load_csv_data(path, inference_mode=inference_mode)
        if not loaded:
            DataLoader.loaded_data = None
            self.skeleton_video_viewer.setCSVPoints({})
            self.kpt_list.clear()
            return loaded
        self._apply_loaded_label_delay(inference_mode=inference_mode)
        return loaded

    def load_txt(self, path, *, inference_mode: bool = False):
        loaded = DataLoader.load_txt_data(path, inference_mode=inference_mode)
        if not loaded:
            DataLoader.loaded_data = None
            self.skeleton_video_viewer.setCSVPoints({})
            self.kpt_list.clear()
            return loaded
        self._apply_loaded_label_delay(inference_mode=inference_mode)
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
        self._run_guarded_save_action(save_modified_data)

    def open_quick_save_dialog(self):
        self._run_guarded_save_action(quick_save_csv)

    def _set_save_controls_enabled(self, enabled: bool) -> None:
        self.save_csv_button.setEnabled(enabled)
        self.save_options_button.setEnabled(enabled)

    def _run_guarded_save_action(self, save_func) -> None:
        if self._save_operation_active:
            return

        self._save_operation_active = True
        self._set_save_controls_enabled(False)
        self.shortcuts_enabled = False
        try:
            save_func(self)
        finally:
            self.shortcuts_enabled = True
            self._set_save_controls_enabled(True)
            self._save_operation_active = False

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.toggle_fullscreen()
                event.accept()
                return
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

    def toggle_automatic_labeling(self) -> None:
        self.automatic_label_checkbox.toggle()

    def _clear_auto_label_selection(self) -> None:
        if not hasattr(self, "mouse_controller") or self.mouse_controller is None:
            return
        self.mouse_controller.selected_instance = None
        self.mouse_controller.selected_node = None
        self.mouse_controller._sync_list_selection()

    def _refresh_after_auto_label_change(self) -> None:
        self._clear_auto_label_selection()
        self.update_csv_points_on_image()
        self.skeleton_video_viewer.update()
        self.kpt_list.update()

    def _prepare_auto_label_request(
        self,
        *,
        require_enabled: bool,
        warn_if_unavailable: bool,
        clear_checkbox_on_mode_mismatch: bool,
    ) -> Optional[tuple[int, object]]:
        if require_enabled and not self.automatic_label_checkbox.isChecked():
            return None
        if self.auto_label_model is None:
            if warn_if_unavailable:
                QMessageBox.warning(
                    self,
                    "Model not loaded",
                    "Load a model before running automatic labeling.",
                )
            return None
        if not getattr(self.skeleton_video_viewer, "video_loaded", False):
            if warn_if_unavailable:
                QMessageBox.warning(
                    self,
                    "Video not loaded",
                    "Load a video before running automatic labeling.",
                )
            return None

        current_mode = self.video_loader.frame_display_mode
        if self.auto_label_model_mode and current_mode != self.auto_label_model_mode:
            QMessageBox.warning(
                self,
                "Mode mismatch",
                "The loaded model was loaded for a different display mode.\n"
                "Reload the model for the current mode before using automatic labeling."
            )
            if clear_checkbox_on_mode_mismatch:
                self.automatic_label_checkbox.blockSignals(True)
                self.automatic_label_checkbox.setChecked(False)
                self.automatic_label_checkbox.blockSignals(False)
            return None

        frame_idx = self.resolve_label_frame(self.video_loader.current_frame)
        if frame_idx is None:
            return None

        frame_source = self.video_loader.get_current_frame_source()
        if frame_source is None:
            if warn_if_unavailable:
                QMessageBox.warning(
                    self,
                    "Frame unavailable",
                    "The current frame could not be prepared for automatic labeling.",
                )
            return None
        return int(frame_idx), frame_source

    def _predict_current_frame_instances(self, frame_source) -> Optional[list[dict]]:
        try:
            return self.predict_current_frame(frame_source)
        except Exception as e:
            QMessageBox.critical(self, "Auto labeling failed", f"Failed to run inference:\n{e}")
            return None

    def run_automatic_label_addition(self) -> None:
        context = self._prepare_auto_label_request(
            require_enabled=False,
            warn_if_unavailable=True,
            clear_checkbox_on_mode_mismatch=False,
        )
        if context is None:
            return

        frame_idx, frame_source = context
        instances = self._predict_current_frame_instances(frame_source)
        if instances is None:
            return
        if not instances:
            return

        if DataLoader.add_auto_labeled_frame(frame_idx, instances, merge_mode="append"):
            self._refresh_after_auto_label_change()

    def run_automatic_relabel(self) -> None:
        context = self._prepare_auto_label_request(
            require_enabled=False,
            warn_if_unavailable=True,
            clear_checkbox_on_mode_mismatch=False,
        )
        if context is None:
            return

        frame_idx, frame_source = context
        had_existing_labels = DataLoader.frame_has_labels(frame_idx)
        if had_existing_labels:
            reply = QMessageBox.question(
                self,
                "Replace current skeletons?",
                "This will remove every skeleton on the current frame and replace them with automatic labels.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        instances = self._predict_current_frame_instances(frame_source)
        if instances is None:
            return

        changed = DataLoader.add_auto_labeled_frame(
            frame_idx,
            instances,
            merge_mode="replace",
        )
        if not changed:
            return

        self._refresh_after_auto_label_change()
        if had_existing_labels and not instances:
            QMessageBox.information(
                self,
                "Automatic re-labeling complete",
                "No detections were produced, so the existing skeletons were cleared.",
            )

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
        setup_running = (
            self.mini_training_setup_worker is not None
            and self.mini_training_setup_worker.isRunning()
        )
        train_running = (
            self.mini_training_thread is not None
            and self.mini_training_thread.isRunning()
        )
        frame_mode = self.video_loader.frame_display_mode if getattr(self.skeleton_video_viewer, "video_loaded", False) else self.mode_combo.currentText()

        self.mini_training_button.setEnabled(True)
        if setup_running:
            self.mini_training_button.setText("Preparing Mini Training...")
            self.mini_training_button.setToolTip(
                "Mini training setup is running. Click to reopen the log window."
            )
        elif train_running:
            self.mini_training_button.setText("Show Mini Training Log")
            self.mini_training_button.setToolTip(
                "Mini training is running. Click to reopen the log window; this will not start another training job."
            )
        else:
            self.mini_training_button.setText("Run Mini Training")
            self.mini_training_button.setToolTip(
                "Export current in-memory labels to a timestamped snapshot under runs/, "
                "build a separate online dataset, run short fine-tuning, and hot-load the resulting best.pt "
                f"using the current frame mode '{frame_mode}'."
            )

    def _mini_training_is_active(self) -> bool:
        setup_running = (
            self.mini_training_setup_worker is not None
            and self.mini_training_setup_worker.isRunning()
        )
        train_running = (
            self.mini_training_thread is not None
            and self.mini_training_thread.isRunning()
        )
        return setup_running or train_running

    def _release_auto_label_model_for_mini_training(self) -> None:
        if self.auto_label_model is None:
            return
        self.auto_label_model = None
        self.auto_label_model_path = None
        self.auto_label_model_mode = None
        self.automatic_label_checkbox.blockSignals(True)
        self.automatic_label_checkbox.setChecked(False)
        self.automatic_label_checkbox.blockSignals(False)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._refresh_model_button_state()

    def _confirm_release_auto_label_model_for_mini_training(self) -> Optional[bool]:
        if self.auto_label_model is None:
            return False

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setWindowTitle("Free GPU memory?")
        msg_box.setText(
            "A YOLO model is currently loaded for automatic labeling.\n\n"
            "Mini training needs GPU memory. Keeping the model loaded may cause "
            "training to fail with CUDA out-of-memory.\n\n"
            "Choose how to continue."
        )
        unload_button = msg_box.addButton("Unload Model and Train", QMessageBox.ButtonRole.AcceptRole)
        keep_button = msg_box.addButton("Keep Model Loaded", QMessageBox.ButtonRole.ActionRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg_box.setDefaultButton(unload_button)
        msg_box.exec()

        clicked = msg_box.clickedButton()
        if clicked == unload_button:
            return True
        if clicked == keep_button:
            return False
        if clicked == cancel_button:
            return None
        return None

    def _mini_training_failed_from_oom(self, output_text: str) -> bool:
        text = (output_text or "").lower()
        oom_markers = (
            "cuda out of memory",
            "torch.cuda.outofmemoryerror",
            "outofmemoryerror",
            "cublas_status_alloc_failed",
            "cudnn_status_alloc_failed",
            "failed to allocate",
            "not enough memory",
        )
        return any(marker in text for marker in oom_markers)

    def _ensure_mini_training_log_dialog(self) -> MiniTrainingLogDialog:
        if self.mini_training_log_dialog is None:
            self.mini_training_log_dialog = MiniTrainingLogDialog(self)
            self.mini_training_log_dialog.stop_button.clicked.connect(
                self.stop_mini_training
            )
        return self.mini_training_log_dialog

    def _show_mini_training_log_dialog(self) -> None:
        dialog = self._ensure_mini_training_log_dialog()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _append_mini_training_log(self, message: str) -> None:
        dialog = self._ensure_mini_training_log_dialog()
        stamp = datetime.now().strftime("%H:%M:%S")
        dialog.append_log(f"[{stamp}] {message}")

    def _snapshot_loaded_labels_dataframe(self) -> pd.DataFrame:
        if DataLoader.loaded_data is None or DataLoader.loaded_data.empty:
            raise ValueError("Load and review labels before starting mini training.")

        snapshot_df = DataLoader.loaded_data.copy(deep=True)
        for level_name in list(snapshot_df.index.names):
            if level_name:
                drop_level = level_name in snapshot_df.columns
                snapshot_df = snapshot_df.reset_index(level=level_name, drop=drop_level)
        snapshot_df.reset_index(drop=True, inplace=True)
        snapshot_df = snapshot_df.loc[:, ~snapshot_df.columns.duplicated()]
        return snapshot_df

    def _stop_active_mini_training(self, wait_ms: int = 5000) -> bool:
        setup_worker = self.mini_training_setup_worker
        if setup_worker is not None and setup_worker.isRunning():
            setup_worker.request_cancel()
            if not setup_worker.wait(wait_ms):
                return False

        thread = self.mini_training_thread
        if thread is not None and thread.isRunning():
            thread.stop()
            if not thread.wait(wait_ms):
                return False

        return True

    def stop_mini_training(self) -> None:
        if not self._mini_training_is_active():
            return

        reply = QMessageBox.question(
            self,
            "Stop mini training",
            "Mini training is currently running.\n\nStop it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        dialog = self._ensure_mini_training_log_dialog()
        dialog.set_status("Stopping mini training...")
        dialog.stop_button.setEnabled(False)
        self._append_mini_training_log("Stop requested by user.")
        if not self._stop_active_mini_training():
            dialog.set_status("Mini training is still stopping...")
            dialog.stop_button.setEnabled(True)
            QMessageBox.warning(
                self,
                "Still stopping",
                "Mini training is still shutting down. Please wait a moment and try again.",
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

    def run_mini_training(self):
        if self._mini_training_is_active():
            self._show_mini_training_log_dialog()
            return

        if DataLoader.loaded_data is None or DataLoader.loaded_data.empty:
            QMessageBox.warning(self, "No labels loaded", "Load and review labels before starting mini training.")
            return

        model_path = self._resolve_base_model_path()
        if model_path is None:
            QMessageBox.warning(self, "No model selected", "Load a base model or choose a valid model file first.")
            return

        try:
            current_video_name = self._current_video_name()
            if not current_video_name:
                raise ValueError("Current project video name is not available.")
            release_auto_label_model = self._confirm_release_auto_label_model_for_mini_training()
            if release_auto_label_model is None:
                return
            run_stamp = datetime.now().strftime("%y%m%d_%H%M%S")
            frame_mode = self._current_frame_mode()
            snapshot_df = self._snapshot_loaded_labels_dataframe()
            requested_epochs = int(self.mini_training_epochs_spin.value())
        except Exception as e:
            QMessageBox.critical(self, "Mini training setup failed", f"Failed to prepare training inputs:\n{e}")
            return

        self.mini_training_run_context = {
            "base_model_path": str(model_path),
            "requested_epochs": requested_epochs,
            "frame_mode": frame_mode,
            "current_video_name": current_video_name,
            "run_stamp": run_stamp,
            "release_auto_label_model": bool(release_auto_label_model),
            "kept_auto_label_model": bool(self.auto_label_model is not None and not release_auto_label_model),
        }

        dialog = self._ensure_mini_training_log_dialog()
        dialog.reset()
        dialog.set_status("Preparing mini training...")
        dialog.set_summary(
            f"Video: {current_video_name} | Mode: {frame_mode} | Epochs: {requested_epochs}"
        )
        self._append_mini_training_log(f"Base model: {model_path}")
        self._append_mini_training_log(f"Video: {current_video_name}")
        self._append_mini_training_log(f"Frame mode: {frame_mode}")
        self._append_mini_training_log("Preparing label snapshot and training dataset...")
        self._show_mini_training_log_dialog()

        self.mini_training_setup_worker = MiniTrainingSetupWorker(
            self.project,
            snapshot_df=snapshot_df,
            current_video_name=current_video_name,
            frame_mode=frame_mode,
            run_stamp=run_stamp,
        )
        self.mini_training_setup_worker.progress.connect(self.on_mini_training_setup_progress)
        self.mini_training_setup_worker.success.connect(self.on_mini_training_setup_success)
        self.mini_training_setup_worker.cancelled.connect(self.on_mini_training_setup_cancelled)
        self.mini_training_setup_worker.failure.connect(self.on_mini_training_setup_failure)
        self._refresh_mini_training_button_state()
        self.mini_training_setup_worker.start()

    def on_mini_training_setup_progress(self, done: int, total: int, message: str) -> None:
        dialog = self._ensure_mini_training_log_dialog()
        dialog.set_status(message)
        if total > 0:
            dialog.set_summary(f"{message} ({done}/{total})")
        self._append_mini_training_log(message)

    def on_mini_training_setup_success(self, context: dict) -> None:
        self.mini_training_setup_worker = None
        base_context = self.mini_training_run_context or {}
        base_context.update(context)
        self.mini_training_run_context = base_context

        model_path = Path(base_context["base_model_path"])
        run_name = str(base_context["run_name"])
        config_path = Path(base_context["config_path"])
        output_dir = Path(base_context["output_dir"])
        split_counts = base_context["split_counts"]
        requested_epochs = int(base_context["requested_epochs"])

        dialog = self._ensure_mini_training_log_dialog()
        dialog.set_status("Mini training is running...")
        dialog.set_summary(
            "Train/Val/Test: "
            f"{split_counts['train']}/{split_counts['val']}/{split_counts['test']} | "
            f"Output: {output_dir}"
        )
        self._append_mini_training_log(f"Snapshot: {base_context['snapshot_dir']}")
        self._append_mini_training_log(f"Dataset: {base_context['dataset_dir']}")
        self._append_mini_training_log(
            f"Split counts - train: {split_counts['train']}, val: {split_counts['val']}, test: {split_counts['test']}"
        )
        self._append_mini_training_log(f"Config: {config_path}")

        command = [
            "yolo",
            "pose",
            "train",
            f"model={model_path.as_posix()}",
            f"data={config_path.as_posix()}",
            f"epochs={requested_epochs}",
            f"project={(Path(self.project.project_dir) / 'runs').as_posix()}",
            f"name={run_name}",
            "batch=1",
            "cache=False",
            "workers=0",
            "exist_ok=False",
        ]

        if base_context.get("release_auto_label_model") and self.auto_label_model is not None:
            self._append_mini_training_log("Auto-label model was unloaded to free GPU memory before training.")
            self._release_auto_label_model_for_mini_training()
        elif base_context.get("kept_auto_label_model"):
            self._append_mini_training_log("Auto-label model remains loaded during mini training by user choice.")

        self._append_mini_training_log("Starting YOLO mini training...")
        self._append_mini_training_log("Command: " + " ".join(command))

        self.mini_training_thread = TrainThread(command)
        self.mini_training_thread.log_signal.connect(self.on_mini_training_thread_log)
        self.mini_training_thread.finished_signal.connect(self.on_mini_training_finished)
        self._refresh_mini_training_button_state()
        self.mini_training_thread.start()

    def on_mini_training_setup_cancelled(self) -> None:
        self.mini_training_setup_worker = None
        self.mini_training_run_context = None
        dialog = self._ensure_mini_training_log_dialog()
        dialog.set_status("Mini training stopped.")
        dialog.set_summary("")
        dialog.stop_button.setEnabled(False)
        self._append_mini_training_log("Mini training setup was cancelled.")
        self._refresh_mini_training_button_state()

        if self._suppress_mini_training_feedback:
            self._suppress_mini_training_feedback = False
            return

        QMessageBox.information(
            self,
            "Mini training stopped",
            "Mini training was stopped before training began.",
        )

    def on_mini_training_setup_failure(self, message: str) -> None:
        self.mini_training_setup_worker = None
        self.mini_training_run_context = None
        dialog = self._ensure_mini_training_log_dialog()
        dialog.set_status("Mini training setup failed.")
        dialog.set_summary("")
        dialog.stop_button.setEnabled(False)
        self._append_mini_training_log(f"Setup failed: {message}")
        self._show_mini_training_log_dialog()
        self._refresh_mini_training_button_state()

        if self._suppress_mini_training_feedback:
            self._suppress_mini_training_feedback = False
            return

        QMessageBox.critical(
            self,
            "Mini training setup failed",
            f"Failed to prepare training inputs:\n{message}",
        )

    def on_mini_training_thread_log(self, message: str) -> None:
        if not message:
            return
        dialog = self._ensure_mini_training_log_dialog()
        if dialog.status_label.text() != "Mini training is running...":
            dialog.set_status("Mini training is running...")
        self._append_mini_training_log(message)

    def on_mini_training_finished(self):
        context = self.mini_training_run_context or {}
        thread = self.mini_training_thread
        was_stopped = bool(thread and thread.was_stopped)
        exit_code = None if thread is None else thread.exit_code
        self.mini_training_thread = None
        self._refresh_mini_training_button_state()

        dialog = self._ensure_mini_training_log_dialog()
        dialog.stop_button.setEnabled(False)

        if was_stopped:
            dialog.set_status("Mini training stopped.")
            dialog.set_summary("")
            self._append_mini_training_log("Mini training was stopped before completion.")
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

        if exit_code not in (None, 0):
            output_text = "" if thread is None else thread.output_text
            oom_failed = self._mini_training_failed_from_oom(output_text)
            if oom_failed:
                kept_model_loaded = bool(context.get("kept_auto_label_model"))
                extra_note = (
                    "\n\nYou chose to keep the auto-label model loaded, which may have contributed "
                    "to GPU memory pressure. Try again with 'Unload Model and Train' selected."
                    if kept_model_loaded
                    else ""
                )
                dialog.set_status("Mini training failed: insufficient GPU memory.")
                self._append_mini_training_log(
                    "Training failed because GPU memory was exhausted even with batch=1, cache=False, and workers=0."
                )
                self._show_mini_training_log_dialog()
                QMessageBox.critical(
                    self,
                    "GPU memory is insufficient",
                    "Mini training could not run on the current GPU memory budget.\n\n"
                    "MovAl used a conservative mini-training configuration "
                    "(batch=1, cache=False, workers=0)."
                    f"{extra_note}\n\n"
                    "This GPU is not suitable for mini training with the current model/data settings. "
                    "Use a GPU with more VRAM, a smaller model, or fewer/lower-resolution training frames.",
                )
            else:
                dialog.set_status("Mini training failed.")
                self._append_mini_training_log(f"Training failed with exit code {exit_code}.")
                self._show_mini_training_log_dialog()
                QMessageBox.critical(
                    self,
                    "Mini training failed",
                    f"YOLO training exited with code {exit_code}.\n\n"
                    "Check the mini training log for details.",
                )
            self.mini_training_run_context = None
            return

        best_model_path = Path(context.get("output_dir", "")) / "weights" / "best.pt"
        if not best_model_path.exists():
            dialog.set_status("Mini training failed.")
            self._append_mini_training_log(
                f"Training finished, but best.pt was not created: {best_model_path}"
            )
            self._show_mini_training_log_dialog()
            QMessageBox.critical(
                self,
                "Mini training failed",
                "Training finished, but best.pt was not created.\n"
                f"Expected path:\n{best_model_path}"
            )
            self.mini_training_run_context = None
            return

        dialog.set_status("Mini training completed.")
        dialog.set_summary(f"Best model: {best_model_path}")
        self._append_mini_training_log(f"Training completed. Loading best model: {best_model_path}")
        self._set_model_path_display(str(best_model_path.resolve()))
        self.load_model()
        self.mini_training_run_context = None

    def auto_label_current_frame(self):
        context = self._prepare_auto_label_request(
            require_enabled=True,
            warn_if_unavailable=False,
            clear_checkbox_on_mode_mismatch=True,
        )
        if context is None:
            return
        frame_idx, frame_source = context
        if DataLoader.frame_has_labels(frame_idx):
            return

        instances = self._predict_current_frame_instances(frame_source)
        if instances is None:
            return

        if not instances:
            return

        if DataLoader.add_auto_labeled_frame(frame_idx, instances, merge_mode="empty_only"):
            self._refresh_after_auto_label_change()

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
        if self._mini_training_is_active():
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
            if not self._stop_active_mini_training():
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
