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
from .IO.save_files import save_modified_data, export_current_labels_to_txt_snapshot
from .controller.keyboard_controller import KeyboardController
from .controller.mouse_controller import MouseController
from utils.skeleton import SkeletonModel
from pose.prepare_data import create_online_training_dataset
from pose.thread import TrainThread

from typing import Union, Optional, List
from datetime import datetime
from pathlib import Path
import yaml
import sys

class LabelaryDialog(QDialog, UI_LabelaryDialog):
    def __init__(self, project, parent= None):
        super().__init__(parent)
        self.setupUi(self)

        self.project = project
        self._restoring_ui_state = False
        self.auto_label_model = None
        self.auto_label_model_path: Optional[str] = None
        self.auto_label_model_mode: Optional[str] = None
        self.mini_training_thread: Optional[TrainThread] = None
        self.mini_training_run_context: Optional[dict] = None
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

        self.video_combo.currentIndexChanged.connect(self.update_label_combo)
        self.video_combo.currentIndexChanged.connect(self._on_video_selection_changed)
        self.label_combo.currentIndexChanged.connect(self._on_label_selection_changed)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.file_entry_idx = 0

        self.set_color_combo()
        self.color_combo.currentIndexChanged.connect(self.set_color_mode)
        self._restore_ui_state()

        self.save_button.clicked.connect(self.open_save_dialog)
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
        for video in self.project.get_video_list():
            p = Path(video)
            self.video_combo.addItem(p.name, p)

    def load_mode_combo(self):
        self.mode_combo.clear()
        for display_mode in ["images", "davis", "contour"]:
            self.mode_combo.addItem(display_mode)
        preferred_mode = self.project.get_preferred_frame_mode()
        index = self.mode_combo.findText(preferred_mode, Qt.MatchFlag.MatchExactly)
        self.mode_combo.setCurrentIndex(index if index >= 0 else 1)

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
        self.update_keypoint_list()
        self.update_csv_points_on_image()
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
        current_frame = self.video_loader.current_frame
        coords_dict = DataLoader.get_keypoint_coordinates_by_frame(current_frame)
        self.skeleton_video_viewer.setCSVPoints(coords_dict)
        self.kpt_list.update_list_visibility(coords_dict)
        
    def update_keypoint_list(self):
        self.kpt_list.clear()
        if DataLoader.loaded_data is None:
            return
        tracks = list(self.project.animals_name)
        self.kpt_list.build(tracks, DataLoader.kp_order, self.skeleton)

    def set_color_combo(self):
        self.color_combo.clear()
        self.color_combo.addItem("cutie_light")
        self.color_combo.addItem("cutie_dark")
        self.color_combo.addItem("white")
        self.color_combo.addItem("black")
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
        finally:
            self._restoring_ui_state = False

        self.set_color_mode()

    def _find_video_index(self, video_name: str) -> int:
        for index in range(self.video_combo.count()):
            video_path = self.video_combo.itemData(index, Qt.ItemDataRole.UserRole)
            if isinstance(video_path, Path) and video_path.stem == video_name:
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
        video_path = self.video_combo.currentData(Qt.ItemDataRole.UserRole)
        if isinstance(video_path, Path):
            return video_path.stem
        return None

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

    def open_save_dialog(self):
        save_modified_data(self)

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
        self.load_model_button.setText("Reload Model" if loaded else "Browse/Load Model")
        if loaded:
            self.load_model_button.setToolTip(
                f"Loaded for mode '{self.auto_label_model_mode}': {self.auto_label_model_path}"
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
        base_config_path = Path(self.project.project_dir) / "runs" / "training_config.yaml"
        if not base_config_path.exists():
            raise FileNotFoundError(f"Training config not found:\n{base_config_path}")

        with base_config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        config["train"] = (dataset_dir / "train").as_posix()
        config["val"] = (dataset_dir / "val").as_posix()
        config["test"] = (dataset_dir / "test").as_posix()

        target_config_path = Path(self.project.project_dir) / "runs" / f"{run_name}_config.yaml"
        with target_config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

        return target_config_path

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

            current_video_name = Path(current_video).stem
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
        self.mini_training_thread = None
        self._refresh_mini_training_button_state()

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

        frame_idx = self.video_loader.current_frame
        if DataLoader.frame_has_labels(frame_idx):
            return

        frame_path = self.video_loader.get_current_frame_path()
        if not frame_path:
            return

        try:
            instances = self.predict_current_frame(frame_path)
        except Exception as e:
            QMessageBox.critical(self, "Auto labeling failed", f"Failed to run inference:\n{e}")
            return

        if not instances:
            return

        if DataLoader.add_auto_labeled_frame(frame_idx, instances):
            self.update_csv_points_on_image()
            self.skeleton_video_viewer.update()
            self.kpt_list.update()

    def predict_current_frame(self, frame_path: str) -> list[dict]:
        confidence_threshold = float(self.auto_label_confidence_spin.value())
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = self.auto_label_model.predict(
                source=frame_path,
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
        best_by_class: dict[int, dict] = {}
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
            prev = best_by_class.get(class_idx)
            if prev is not None and prev["score"] >= score:
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

            best_by_class[class_idx] = {
                "score": score,
                "track": self.project.animals_name[class_idx],
                "keypoints": kp_map,
            }

        return [
            {
                "track": item["track"],
                "keypoints": item["keypoints"],
            }
            for _, item in sorted(best_by_class.items())
        ]

    def closeEvent(self, event) -> None:
        self._persist_ui_state(include_frame=True)
        super().closeEvent(event)

def run_labelary_with_project(current_project, parent=None):
    app = QApplication.instance() or QApplication(sys.argv)
    dlg = LabelaryDialog(current_project, parent) 
    dlg.exec()  
    return 
