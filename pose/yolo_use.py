from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QScrollArea, QGridLayout,
    QDialog, QLineEdit, QMessageBox, QSpinBox, QFileDialog, QGroupBox, QFormLayout,
    QCheckBox, QComboBox, QDoubleSpinBox, QRadioButton, QListWidget, QFrame,
    QListWidgetItem, QAbstractItemView
)
from PyQt6.QtCore import Qt
import os
from .thread import TrainThread, InferenceThread
from .task_state import pose_execution_state
from utils.runtime_locks import is_project_compression_running
from datetime import datetime
from pathlib import Path
import re
import pandas as pd
from typing import Optional, Union
from utils.csv_interpolation import interpolate_pose_csv
from utils.skeleton.skeleton_model import SkeletonModel
from utils.ui_theme import build_list_widget_stylesheet
from utils.txt_conversion import parse_txt_pose_detections, resolve_frame_track_data

class BrowseOnlyLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.setAcceptDrops(False)
        self.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, False)

    def keyPressEvent(self, event):
        event.ignore()

    def inputMethodEvent(self, event):
        event.ignore()

    def dragEnterEvent(self, event):
        event.ignore()

    def dragMoveEvent(self, event):
        event.ignore()

    def dropEvent(self, event):
        event.ignore()


DISABLED_COMBOBOX_STYLESHEET = (
    "QComboBox:disabled { color: #8a8f98; background-color: #eceff3; border-color: #c6ccd6; }"
)

class YOLODialog(QDialog):
    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Train Config")
        self.setFixedSize(1000, 800)

        self.current_project = current_project
        self.train_thread = None
        self._training_running = False

        main_layout = QVBoxLayout(self)

        config_group = QGroupBox("Training Config")
        config_layout = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv8", "YOLOv11", "use pretrained model"])

        self.size_combo = QComboBox()
        self.size_combo.addItems(["n", "s", "m", "l", "x"])

        config_layout.addRow("Model", self.model_combo)
        config_layout.addRow("Model Size", self.size_combo)
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        middle_layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.create_group_box("Hyper Parameters", {
            "batch": 32, "epochs": 400, "imgsz": 640, "patience": 100,
            "lr0": 0.001, "optimizer": "AdamW", "weight_decay": 0.0005,
            "cos_lr": True, "amp": True, "lrf": 0.001, "momentum": 0.937,
            "dropout": 0.35
        }))
        left_layout.addWidget(self.create_group_box("Loss Design", {
            "box": 7.5, "cls": 0.5, "dfl": 1.5, "pose": 18,
            "kobj": 5, "nbs": 64
        }))

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.create_group_box("Training Options", {
            "cache": False, "device": 0, "workers": 8, "pretrained": True, "deterministic": True, "fraction": 1
        }))
        right_layout.addWidget(self.create_group_box("Augmentation", {
            "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "degrees": 90,
            "scale": 0.5, "shear": 0.0, "translate": 0.1, "flipud": 0.3,
            "fliplr": 0.5, "erasing": 0.4, "crop_fraction": 0.1
        }))

        middle_layout.addLayout(left_layout, 1)
        middle_layout.addLayout(right_layout, 1)
        main_layout.addLayout(middle_layout)

        self.run_btn = QPushButton("Run Training")
        self.run_btn.setProperty("primary", True)
        self.run_btn.clicked.connect(self._on_run_button_clicked)
        self.run_btn.setFixedHeight(30)
        main_layout.addWidget(self.run_btn)

        pose_execution_state.busy_changed.connect(self._on_pose_task_busy_changed)
        self._on_pose_task_busy_changed(
            pose_execution_state.is_busy(),
            pose_execution_state.active_task() or "",
        )

    def _on_run_button_clicked(self):
        active_task = (pose_execution_state.active_task() or "").lower()
        if pose_execution_state.is_busy() and active_task == "training":
            self._stop_training()
            return
        if self._is_training_running():
            self._stop_training()
            return
        self.run_train()

    def _is_training_running(self) -> bool:
        if self._training_running or (self.train_thread is not None and self.train_thread.isRunning()):
            return True
        return pose_execution_state.is_busy() and (pose_execution_state.active_task() or "").lower() == "training"

    def _set_training_parameter_controls_enabled(self, enabled: bool):
        widgets = [self.model_combo, self.size_combo]
        for box in self.findChildren(QGroupBox):
            widgets.append(box)
        for widget in widgets:
            if widget is not None:
                widget.setEnabled(enabled)

    def _stop_training(self):
        active_task = (pose_execution_state.active_task() or "").lower()
        if not self._is_training_running() and active_task != "training":
            return
        print("[Training] Stop requested by user.", flush=True)
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Stopping training...")
        if self.train_thread is not None:
            self.train_thread.stop()
            return

        owner = pose_execution_state.active_owner()
        if owner is not None and owner is not self:
            owner_thread = getattr(owner, "train_thread", None)
            if owner_thread is not None and owner_thread.isRunning():
                stop_fn = getattr(owner_thread, "stop", None)
                if callable(stop_fn):
                    stop_fn()
                    return

    def create_group_box(self, title, params):
        group = QGroupBox(title)
        group.setObjectName(title)
        form = QFormLayout()

        for key, default in params.items():
            if key == "optimizer":
                widget = QComboBox()
                widget.addItems(["SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"])
                widget.setCurrentText(default)

            elif key == "device":
                widget = QLineEdit()
                widget.setText(str(default)) 

            elif isinstance(default, bool):
                widget = QCheckBox()
                widget.setChecked(default)

            elif isinstance(default, str):
                widget = QLineEdit()
                widget.setText(default)

            elif isinstance(default, int):
                widget = QSpinBox()
                widget.setMinimum(0)
                widget.setMaximum(10000)
                widget.setSingleStep(1)
                widget.setValue(default)

            else: 
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setMinimum(0.0)
                widget.setMaximum(10000)
                widget.setSingleStep(0.0001)
                widget.setValue(float(default))

            form.addRow(key, widget)

        group.setLayout(form)
        return group
            
    def run_train(self):
        import os

        if is_project_compression_running():
            QMessageBox.information(
                self,
                "Compression in progress",
                "Project compression is running. Please wait until it finishes.",
            )
            return

        if pose_execution_state.is_busy():
            running = (pose_execution_state.active_task() or "pose task").lower()
            if running == "training":
                QMessageBox.information(self, "Training in progress", "Training is already running.")
            else:
                QMessageBox.information(
                    self,
                    "Pose task already running",
                    f"Another pose task is running ({running}).\n"
                    "Please wait until it finishes.",
                )
            return

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")

        model_type = self.model_combo.currentText().lower()

        if model_type in ("yolov11", "yolov8"):
            model_size = self.size_combo.currentText()
            model_name = f"yolo11{model_size}-pose.pt" if model_type == "yolov11" else f"{model_type}{model_size}-pose.pt"
            model_path = os.path.join(models_dir, model_name)
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Error", f"{model_path} not found!\nPlease download the model first.")
                return
        elif model_type == "use pretrained model":
            model_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select pretrained model file",
                str(Path(self.current_project.project_dir)/"runs"),
                "PyTorch model (*.pt);;All Files (*)"
            )
            if not model_path:
                QMessageBox.warning(
                    self,
                    "No model selected",
                    "Please select a pre-trained model"
                )
                return
            if not Path(model_path).exists():
                QMessageBox.warning(
                    self,
                    "No model selected",
                    "Please select a pre-trained model"
                )
        else:
            QMessageBox.critical(
                self,
                "Invalid model type",
                f"Unknown model_type: {model_type}"
            )
            return

        try:
            yaml_path = self.current_project.write_training_config_yaml().as_posix()
        except Exception as err:
            QMessageBox.critical(self, "Training config error", f"Failed to generate training config:\n{err}")
            return

        params = {
            "model": model_path,
            "data": yaml_path,
        }
        group_names = [
            "Hyper Parameters",
            "Loss Design",
            "Training Options",
            "Augmentation",
        ]

        for group_name in group_names:
            group = self.findChild(QGroupBox, group_name)
            if not group:
                continue

            layout = group.layout()
            for i in range(layout.rowCount()):
                label_widget = layout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget()
                field_widget = layout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget()

                key = label_widget.text()

                if isinstance(field_widget, QLineEdit):
                    params[key] = field_widget.text()
                elif isinstance(field_widget, QComboBox):
                    params[key] = field_widget.currentText()
                elif isinstance(field_widget, QCheckBox):
                    params[key] = field_widget.isChecked()
                elif isinstance(field_widget, (QSpinBox, QDoubleSpinBox)):
                    value = field_widget.value()
                    params[key] = int(value) if value == int(value) else value

        ts = datetime.now()
        ts_date = ts.strftime("%y%m%d")
        ts_time = ts.strftime("%H%M%S")
        params["project"] = os.path.join(self.current_project.project_dir, "runs")
        params["name"]    = f"train_{ts_date}_{ts_time}"

        command = ["yolo", "pose", "train"]
        for key, value in params.items():
            if value in ["", "None"]:
                continue
            if isinstance(value, bool):
                if value:
                    command.append(f"{key}=True")
            else:
                command.append(f"{key}={value}")

        print("Execute Command:", command)

        if not pose_execution_state.acquire("training", owner=self):
            QMessageBox.information(
                self,
                "Pose task already running",
                "Another pose task is running. Please try again later.",
            )
            return

        pose_execution_state.update_progress(0, 0, "Training running...")
        self._training_running = True
        self.train_thread = TrainThread(command)
        self.train_thread.finished_signal.connect(self._on_train_finished)
        self._set_training_parameter_controls_enabled(False)
        self.train_thread.start()
        self._on_pose_task_busy_changed(True, "training")

    def _on_train_finished(self):
        was_stopped = self.train_thread.was_stopped if self.train_thread is not None else False
        pose_execution_state.release(owner=self)
        self.train_thread = None
        self._training_running = False
        self._set_training_parameter_controls_enabled(True)
        self._on_pose_task_busy_changed(
            pose_execution_state.is_busy(),
            pose_execution_state.active_task() or "",
        )
        if was_stopped:
            print("[Training] Stopped.", flush=True)
            QMessageBox.information(self, "Stopped", "Training stopped.")
        else:
            print("[Training] Completed.", flush=True)
            QMessageBox.information(self, "Done", "Training Completed")

    def _on_pose_task_busy_changed(self, busy: bool, task_name: str):
        if not hasattr(self, "run_btn"):
            return

        active = (task_name or "").lower()
        if busy and active == "training":
            self._set_training_parameter_controls_enabled(False)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Stop Training")
            return

        if busy:
            self._set_training_parameter_controls_enabled(False)
            self.run_btn.setEnabled(False)
            if active == "inference":
                self.run_btn.setText("Run Training (Disabled during inference)")
            else:
                self.run_btn.setText("Run Training")
        else:
            self._set_training_parameter_controls_enabled(True)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run Training")

    def closeEvent(self, event):
        if self._is_training_running():
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)

class YoloInferenceDialog(QDialog):
    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.current_project = current_project
        self.animals_name = current_project.animals_name
        self.command_queue = []
        self.current_run_item = None
        self.infer_thread = None
        self._inference_running = False
        self._stop_requested = False
        self._completed_commands = 0
        self._total_commands = 0
        self.build_ui()
        pose_execution_state.busy_changed.connect(self._on_pose_task_busy_changed)
        self._on_pose_task_busy_changed(
            pose_execution_state.is_busy(),
            pose_execution_state.active_task() or "",
        )

    def build_ui(self):
        self.setWindowTitle("YOLO Pose Inference")
        self.setMinimumSize(720, 700)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(self.build_model_row())
        main_layout.addLayout(self.build_track_row())

        self.grid = QGridLayout()
        self.video_group  = self.build_video_group()
        self.target_group = self.build_target_group()
        self.inference_group = self.create_params_group(
            "Inference Config",
            {"imgsz": 640, "conf": 0.5, "iou": 0.7,
             "augment": False, "half": False, "device": "None"}
        )
        self.visualization_group = self.build_visualization_group()

        self.grid.addWidget(self.video_group, 0, 0)
        self.grid.addWidget(self.target_group, 0, 1)
        self.grid.addWidget(self.inference_group, 1, 0)
        self.grid.addWidget(self.visualization_group, 1, 1)

        self.grid.setRowStretch(0, 5)
        self.grid.setRowStretch(1, 1)
        self.grid.setColumnStretch(0, 3)
        self.grid.setColumnStretch(1, 1)

        main_layout.addLayout(self.grid)
        self.run_btn = QPushButton("Run Inference", clicked=self._on_run_button_clicked)
        self.run_btn.setProperty("primary", True)
        self.run_btn.setFixedHeight(30)
        main_layout.addWidget(self.run_btn)

    def _on_run_button_clicked(self):
        active_task = (pose_execution_state.active_task() or "").lower()
        if pose_execution_state.is_busy() and active_task == "inference":
            self._stop_inference()
            return
        if self._inference_running:
            self._stop_inference()
            return
        self.run_inference()

    def build_model_row(self):
        row = QHBoxLayout()
        self.model_line_edit = BrowseOnlyLineEdit()
        self.model_line_edit.setPlaceholderText("Select Model")
        self.model_browse_btn = QPushButton("Browse Model", clicked=self.select_model)
        self.model_browse_btn.setProperty("primary", True)
        row.addWidget(self.model_line_edit)
        row.addWidget(self.model_browse_btn)
        return row

    def build_track_row(self):
        row = QHBoxLayout()
        self.inference_radio = QRadioButton("Inference", checked=True)
        self.tracking_radio = QRadioButton("Tracking")
        self.track_method_combo = QComboBox(enabled=False)
        self.track_method_combo.addItems(["botsort", "bytetrack"])
        self.track_method_combo.setStyleSheet(DISABLED_COMBOBOX_STYLESHEET)

        for w in (self.inference_radio, self.tracking_radio):
            w.toggled.connect(self.update_mode)
            row.addWidget(w)
        row.addWidget(QLabel("Tracking Method:"))
        row.addWidget(self.track_method_combo)
        self.update_mode()
        return row

    def build_video_group(self):
        group = QGroupBox("Video Selection")
        form  = QFormLayout(group)
        mode_box = QWidget()
        mode_lay = QHBoxLayout(mode_box)
        mode_lay.setContentsMargins(0,0,0,0)
        self.image_radio = QRadioButton("image frames")
        self.video_radio = QRadioButton("video", checked=True)
        self._source_mode_ready = False
        for w in (self.image_radio, self.video_radio):
            mode_lay.addWidget(w)
            w.toggled.connect(self.update_source_mode_ui)
        form.addRow("Source Mode :", mode_box)

        self.image_section = self.build_image_section()
        form.addRow(self.image_section)
        self.video_section = self.build_video_section()
        form.addRow(self.video_section)
        self.update_source_mode_ui()
        self._source_mode_ready = True
        return group

    def build_image_section(self):
        sect = QWidget()
        lay = QVBoxLayout(sect)
        lay.setContentsMargins(0,0,0,0)
        self.image_mode_combo = QComboBox()
        self.image_mode_combo.addItems(["images", "davis", "contour"])
        preferred_mode = self.current_project.get_preferred_frame_mode()
        preferred_index = self.image_mode_combo.findText(preferred_mode, Qt.MatchFlag.MatchExactly)
        if preferred_index >= 0:
            self.image_mode_combo.setCurrentIndex(preferred_index)
        self.image_mode_combo.currentTextChanged.connect(self._save_image_mode)
        lay.addWidget(self.image_mode_combo)
        self.image_source_hint = QLabel(
            "Only items shown in this list will be included in inference."
        )
        self.image_source_hint.setWordWrap(True)
        lay.addWidget(self.image_source_hint)
        self.image_source_list = self._create_source_list_widget()
        self._populate_image_source_list()
        lay.addWidget(self.image_source_list)

        button_row = QHBoxLayout()
        self.exclude_image_sources_btn = QPushButton("Exclude Selected", clicked=self.exclude_selected_image_sources)
        self.keep_only_image_sources_btn = QPushButton("Keep Only Selected", clicked=self.keep_only_selected_image_sources)
        button_row.addWidget(self.exclude_image_sources_btn)
        button_row.addWidget(self.keep_only_image_sources_btn)
        self.reset_image_sources_btn = QPushButton("Reset From Project", clicked=self.reset_image_sources)
        button_row.addWidget(self.reset_image_sources_btn)
        lay.addLayout(button_row)
        return sect

    def _save_image_mode(self, image_mode: str) -> None:
        if self.image_radio.isChecked():
            self.current_project.set_preferred_frame_mode(image_mode)

    def build_video_section(self):
        sect = QWidget()
        sect.hide()
        lay  = QVBoxLayout(sect)
        lay.setContentsMargins(0,0,0,0)
        self.load_video_btn = QPushButton("Load Video...", clicked=self.load_videos)
        self.video_source_hint = QLabel(
            "Only items shown in this list will be included in inference."
        )
        self.video_source_hint.setWordWrap(True)
        self.loaded_list = self._create_source_list_widget()
        self._populate_loaded_videos_from_project()
        self.exclude_selected_videos_btn = QPushButton(
            "Exclude Selected",
            clicked=self.exclude_selected_videos,
        )
        self.include_only_selected_videos_btn = QPushButton(
            "Keep Only Selected",
            clicked=self.include_only_selected_videos,
        )
        self.clear_video_btn = QPushButton("Clear List", clicked=self.clear_videos)
        lay.addWidget(self.load_video_btn)
        lay.addWidget(self.video_source_hint)
        lay.addWidget(self.loaded_list)
        action_row = QHBoxLayout()
        action_row.addWidget(self.exclude_selected_videos_btn)
        action_row.addWidget(self.include_only_selected_videos_btn)
        action_row.addWidget(self.clear_video_btn)
        lay.addLayout(action_row)
        return sect

    def build_target_group(self):
        group = QGroupBox("Inference Target")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setContentsMargins(6, 6, 6, 6)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(2) 
        self.target_checks = []
        for name in self.animals_name:
            cb = QCheckBox(name, checked=True)
            self.target_checks.append(cb)
            vbox.addWidget(cb)
        vbox.addStretch()  

        scroll = QScrollArea(frameShape=QFrame.Shape.NoFrame, widgetResizable=True)
        scroll.setObjectName("inferenceTargetScroll")
        scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(container)
        form.addRow(scroll)
        return group

    def build_visualization_group(self):
        group = QGroupBox("Visualization")
        form = QFormLayout(group)

        self.show_tracking_checkbox = QCheckBox(checked=False)
        self.save_media_checkbox = QCheckBox(checked=False)
        self.save_txt_checkbox = QCheckBox(checked=False)
        self.convert_txt_to_csv_checkbox = QCheckBox(checked=True)
        self.csv_output_mode_combo = QComboBox()
        self.csv_output_mode_combo.addItems(["Raw CSV only", "Raw + interpolated CSV"])
        self.csv_output_mode_combo.setStyleSheet(DISABLED_COMBOBOX_STYLESHEET)

        form.addRow(QLabel("show tracking result"), self.show_tracking_checkbox)
        form.addRow(QLabel("save image/video"), self.save_media_checkbox)
        form.addRow(QLabel("save result as txt"), self.save_txt_checkbox)
        form.addRow(QLabel("save result as csv"), self.convert_txt_to_csv_checkbox)
        form.addRow(QLabel("csv output mode"), self.csv_output_mode_combo)

        self.save_txt_checkbox.toggled.connect(self._update_visualization_option_states)
        self.convert_txt_to_csv_checkbox.toggled.connect(self._update_visualization_option_states)
        self._update_visualization_option_states()
        return group

    def create_params_group(self, title, params: dict):
        group = QGroupBox(title)
        form  = QFormLayout(group)
        for key, default in params.items():
            widget = (
                QCheckBox(checked=default)                              if isinstance(default, bool)  else
                QDoubleSpinBox(decimals=4, maximum=1e4, value=default)  if isinstance(default, float) else
                QSpinBox(maximum=1e4, value=default)                    if isinstance(default, int)   else
                QLineEdit(text=str(default))
            )
            form.addRow(QLabel(key), widget)
        return group

    def select_model(self):
        model_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select pretrained model file",
                str(Path(self.current_project.project_dir)/"runs"),
                "PyTorch model (*.pt);;All Files (*)"
            )
        if not model_path:
            QMessageBox.warning(
                self,
                "No model selected",
                "Please select a model"
            )
            return
        if not Path(model_path).exists():
            QMessageBox.warning(
                self,
                "No model selected",
                "Please select a model"
            )
        self.model_line_edit.setText(model_path)
            
    def update_mode(self):
        if self.tracking_radio.isChecked():
            self.track_method_combo.setEnabled(True)
        else:
            self.track_method_combo.setEnabled(False)

    def update_source_mode_ui(self):
        if self.image_radio.isChecked():
            self.image_section.show()
            self.video_section.hide()
        else:
            self.image_section.hide()
            self.video_section.show()

        self._update_visualization_option_states()

        if getattr(self, "_source_mode_ready", False):
            if self.video_radio.isChecked():
                self.current_project.set_preferred_frame_mode("video")
            else:
                self.current_project.set_preferred_frame_mode(self.image_mode_combo.currentText())

    def clear_videos(self):
        self.loaded_list.clear()

    def _create_source_list_widget(self) -> QListWidget:
        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        list_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        list_widget.setStyleSheet(build_list_widget_stylesheet())
        return list_widget

    def _populate_image_source_list(self) -> None:
        self.image_source_list.clear()
        for file_entry in self.current_project.files:
            item = QListWidgetItem(str(file_entry.name))
            item.setData(Qt.ItemDataRole.UserRole, file_entry.name)
            item.setToolTip(str(file_entry.video))
            self.image_source_list.addItem(item)

    def _item_user_value(self, item: QListWidgetItem) -> str:
        value = item.data(Qt.ItemDataRole.UserRole)
        if value is None:
            return item.text()
        return str(value)

    def _exclude_selected_items(self, list_widget: QListWidget) -> None:
        selected_rows = sorted(
            {list_widget.row(item) for item in list_widget.selectedItems()},
            reverse=True,
        )
        for row in selected_rows:
            list_widget.takeItem(row)

    def _keep_only_selected_items(self, list_widget: QListWidget) -> None:
        selected_items = list_widget.selectedItems()
        if not selected_items:
            return

        kept_values = [self._item_user_value(item) for item in selected_items]
        list_widget.clear()
        self._restore_list_items(list_widget, kept_values)

    def _restore_list_items(self, list_widget: QListWidget, values: list[str]) -> None:
        if list_widget is self.image_source_list:
            self._add_image_source_items(values)
            return
        if list_widget is self.loaded_list:
            self._add_loaded_video_paths(values)
            return
        raise ValueError("Unsupported source list widget")

    def _project_image_source_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for file_entry in self.current_project.files:
            source_name = str(file_entry.name)
            if source_name in seen:
                continue
            seen.add(source_name)
            names.append(source_name)
        return names

    def _add_image_source_items(self, source_names: list[str]) -> None:
        existing = {
            self._item_user_value(self.image_source_list.item(i))
            for i in range(self.image_source_list.count())
        }
        project_entries = {
            str(file_entry.name): file_entry
            for file_entry in self.current_project.files
        }
        for source_name in source_names:
            if source_name in existing:
                continue
            file_entry = project_entries.get(str(source_name))
            if file_entry is None:
                continue
            item = QListWidgetItem(str(file_entry.name))
            item.setData(Qt.ItemDataRole.UserRole, str(file_entry.name))
            item.setToolTip(str(file_entry.video))
            self.image_source_list.addItem(item)
            existing.add(str(source_name))

    def reset_image_sources(self) -> None:
        self.image_source_list.clear()
        self._add_image_source_items(self._project_image_source_names())

    def exclude_selected_image_sources(self) -> None:
        self._exclude_selected_items(self.image_source_list)

    def keep_only_selected_image_sources(self) -> None:
        self._keep_only_selected_items(self.image_source_list)

    def clear_image_sources(self) -> None:
        self.image_source_list.clear()

    def _normalized_video_source_key(self, source: Union[str, Path]) -> str:
        try:
            return str(Path(source).expanduser().resolve())
        except Exception:
            return str(Path(source))

    def _project_video_source_paths(self) -> list[str]:
        sources: list[str] = []
        seen: set[str] = set()
        for file_entry in self.current_project.files:
            path_obj = Path(file_entry.video)
            if not path_obj.exists():
                continue
            normalized = self._normalized_video_source_key(path_obj)
            if normalized in seen:
                continue
            seen.add(normalized)
            sources.append(str(path_obj))
        return sources

    def _build_video_display_labels(self, path_texts: list[str]) -> dict[str, str]:
        path_objects = [Path(path_text) for path_text in path_texts]
        labels = {path_text: path_obj.stem for path_text, path_obj in zip(path_texts, path_objects)}
        duplicate_groups: dict[str, list[tuple[str, Path]]] = {}

        for path_text, path_obj in zip(path_texts, path_objects):
            duplicate_groups.setdefault(path_obj.stem, []).append((path_text, path_obj))

        for duplicates in duplicate_groups.values():
            if len(duplicates) <= 1:
                continue

            resolved: dict[str, str] = {}
            depth = 1
            while len(resolved) < len(duplicates):
                candidate_map: dict[str, list[str]] = {}
                for path_text, path_obj in duplicates:
                    parents = [part for part in path_obj.parts[:-1] if part not in (path_obj.anchor,)]
                    if parents:
                        tail = parents[-depth:]
                        prefix = "/".join(tail)
                        candidate = f"{prefix}/{path_obj.stem}" if prefix else path_obj.stem
                    else:
                        candidate = path_obj.stem
                    candidate_map.setdefault(candidate, []).append(path_text)

                collisions = False
                for candidate, keys in candidate_map.items():
                    if len(keys) == 1:
                        resolved[keys[0]] = candidate
                    else:
                        collisions = True

                if not collisions:
                    break

                depth += 1
                if depth > max((len(Path(path_text).parts) for path_text, _ in duplicates), default=1):
                    break

            for path_text, path_obj in duplicates:
                labels[path_text] = resolved.get(path_text, path_obj.name)

        return labels

    def _refresh_loaded_video_labels(self) -> None:
        path_texts = [
            self._item_user_value(self.loaded_list.item(i))
            for i in range(self.loaded_list.count())
        ]
        labels = self._build_video_display_labels(path_texts)
        for i in range(self.loaded_list.count()):
            item = self.loaded_list.item(i)
            path_text = self._item_user_value(item)
            item.setText(labels.get(path_text, Path(path_text).stem))
            item.setToolTip(path_text)

    def _add_loaded_video_item(self, path: Union[str, Path]) -> None:
        path_text = str(path)
        item = QListWidgetItem(Path(path_text).stem)
        item.setData(Qt.ItemDataRole.UserRole, path_text)
        item.setToolTip(path_text)
        self.loaded_list.addItem(item)

    def _add_loaded_video_paths(self, paths: list[Union[str, Path]]) -> None:
        existing = {
            self._normalized_video_source_key(self._item_user_value(self.loaded_list.item(i)))
            for i in range(self.loaded_list.count())
        }
        for raw_path in paths:
            path_text = str(raw_path)
            normalized = self._normalized_video_source_key(path_text)
            if normalized in existing:
                continue
            existing.add(normalized)
            self._add_loaded_video_item(path_text)
        self._refresh_loaded_video_labels()

    def _populate_loaded_videos_from_project(self) -> None:
        self.loaded_list.clear()
        self._add_loaded_video_paths(self._project_video_source_paths())

    def exclude_selected_videos(self) -> None:
        self._exclude_selected_items(self.loaded_list)
        self._refresh_loaded_video_labels()

    def include_only_selected_videos(self) -> None:
        self._keep_only_selected_items(self.loaded_list)
        self._refresh_loaded_video_labels()

    def load_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select video files",
            self.current_project.project_dir, 
            "Video (*.mp4 *.avi *.mov)"
        )
        valid_paths = [path for path in paths if path]
        self._add_loaded_video_paths(valid_paths)

    def get_params_from_group(self, group):
        params = {}
        layout = group.layout()
        for i in range(layout.rowCount()):
            label = layout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text()
            field_item = layout.itemAt(i, QFormLayout.ItemRole.FieldRole)
            field = field_item.widget()
            if isinstance(field, QLineEdit):
                params[label] = field.text()
            elif isinstance(field, (QSpinBox, QDoubleSpinBox)):
                params[label] = field.value()
            elif isinstance(field, QCheckBox):
                params[label] = field.isChecked()
        return params
        
    def get_video_list(self):
        if self.image_radio.isChecked():
            source_names: list[str] = [
                self.image_source_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.image_source_list.count())
            ]
            if not source_names:
                return None
            image_mode = self.image_mode_combo.currentText() 
            base_dir = Path(self.current_project.project_dir)
            if image_mode in ("davis", "contour"):
                sources = [
                    (name, base_dir / "frames" / name / "visualization" / image_mode)
                    for name in source_names
                ]
            else:
                sources = [
                    (name, base_dir / "frames" / name / "images")
                    for name in source_names
                ]
            return sources
        elif self.video_radio.isChecked():
            count = self.loaded_list.count()
            if count == 0:
                return None
            sources = []
            for i in range(count):
                src = Path(self._item_user_value(self.loaded_list.item(i)))
                sources.append((src.stem, src))
            return sources
        raise
        
    def get_inference_target(self):
        classes = [idx for idx, cb in enumerate(self.target_checks) if cb.isChecked()]
        max_det = len(classes) * self.current_project.get_max_instances_per_id()
        return classes, max_det

    def _predicts_dir(self) -> Path:
        return Path(self.current_project.project_dir) / "predicts"

    def _make_inference_run_root_name(self, ts: datetime) -> str:
        return f"run_{ts.strftime('%y%m%d_%H%M%S')}"

    def _inference_run_root_dir(self, run_root_name: str) -> Path:
        return self._predicts_dir() / run_root_name

    def _inference_source_dir(self, run_root_name: str, source_name: str) -> Path:
        return self._inference_run_root_dir(run_root_name) / source_name

    def _inference_source_labels_dir(self, run_root_name: str, source_name: str) -> Path:
        return self._inference_source_dir(run_root_name, source_name) / "labels"

    def _inference_source_csv_path(self, run_root_name: str, source_name: str) -> Path:
        return self._inference_run_root_dir(run_root_name) / f"{source_name}.csv"

    def _inference_source_interpolated_csv_path(self, run_root_name: str, source_name: str) -> Path:
        return self._inference_run_root_dir(run_root_name) / f"{source_name}.interpolated.csv"

    def _dir_has_txt_files(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return next(path.glob("*.txt"), None) is not None

    def _resolve_inference_txt_dir(self, run_root_name: str, source_name: str) -> Optional[Path]:
        source_dir = self._inference_source_dir(run_root_name, source_name)
        if self._dir_has_txt_files(source_dir):
            return source_dir

        labels_dir = self._inference_source_labels_dir(run_root_name, source_name)
        if self._dir_has_txt_files(labels_dir):
            return labels_dir
        return None

    def _remove_dir_if_empty(self, path: Path) -> None:
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except OSError:
            pass

    def _flatten_inference_txt_outputs(self, run_root_name: str, source_name: str) -> None:
        source_dir = self._inference_source_dir(run_root_name, source_name)
        labels_dir = self._inference_source_labels_dir(run_root_name, source_name)
        if not labels_dir.is_dir():
            return

        for txt_path in sorted(labels_dir.glob("*.txt"), key=lambda p: self._extract_frame_number(p.name)):
            destination = source_dir / txt_path.name
            if destination.exists():
                destination.unlink()
            txt_path.replace(destination)

        self._remove_dir_if_empty(labels_dir)

    def _delete_inference_txt_outputs(self, run_root_name: str, source_name: str) -> None:
        source_dir = self._inference_source_dir(run_root_name, source_name)
        labels_dir = self._inference_source_labels_dir(run_root_name, source_name)

        for txt_dir in (labels_dir, source_dir):
            if not txt_dir.is_dir():
                continue
            for txt_path in txt_dir.glob("*.txt"):
                txt_path.unlink(missing_ok=True)

        self._remove_dir_if_empty(labels_dir)
        self._remove_dir_if_empty(source_dir)

    def _finalize_inference_outputs(
        self,
        run_root_name: str,
        source_name: str,
        keep_txt: bool,
        want_csv: bool,
        csv_output_mode: str,
    ) -> None:
        txt_dir = self._resolve_inference_txt_dir(run_root_name, source_name)
        converted = False

        if want_csv and txt_dir is not None:
            csv_path = self._convert_txt_result_to_csv(run_root_name, source_name, txt_dir=txt_dir)
            converted = True
            if (
                csv_path is not None
                and csv_output_mode == "Raw + interpolated CSV"
            ):
                interpolate_pose_csv(
                    csv_path,
                    self._inference_source_interpolated_csv_path(run_root_name, source_name),
                )

        if keep_txt:
            self._flatten_inference_txt_outputs(run_root_name, source_name)
        elif not want_csv or converted or txt_dir is None:
            self._delete_inference_txt_outputs(run_root_name, source_name)

    def _update_visualization_option_states(self):
        if hasattr(self, "show_tracking_checkbox"):
            if self.image_radio.isChecked():
                self.show_tracking_checkbox.setChecked(False)
                self.show_tracking_checkbox.setEnabled(False)
            else:
                self.show_tracking_checkbox.setEnabled(True)

        if hasattr(self, "convert_txt_to_csv_checkbox"):
            self.convert_txt_to_csv_checkbox.setEnabled(True)
        if hasattr(self, "csv_output_mode_combo") and hasattr(self, "convert_txt_to_csv_checkbox"):
            self.csv_output_mode_combo.setEnabled(self.convert_txt_to_csv_checkbox.isChecked())

    def run_inference(self):
        if is_project_compression_running():
            QMessageBox.information(
                self,
                "Compression in progress",
                "Project compression is running. Please wait until it finishes.",
            )
            return

        if pose_execution_state.is_busy():
            running = (pose_execution_state.active_task() or "pose task").lower()
            if running != "inference":
                QMessageBox.information(
                    self,
                    "Pose task already running",
                    f"Another pose task is running ({running}).\n"
                    "Please wait until it finishes.",
                )
            else:
                QMessageBox.information(self, "Inference in progress", "Inference is already running.")
            return

        model_path = self.model_line_edit.text()
        infer_params = self.get_params_from_group(self.inference_group)
        sources = self.get_video_list()
        classes, max_det = self.get_inference_target()
        if not model_path:
            QMessageBox.warning(self, "Warning", "Select model.")
            return
        if not sources:
            QMessageBox.warning(self, "Warning", "Select videos.")
            return
        if not classes:
            QMessageBox.warning(self, "Warning", "Select target.")
            return
        classes = ",".join(str(c) for c in classes)
        infer_params["classes"] = classes
        infer_params["max_det"] = max_det

        self.command_queue = []
        self.current_run_item = None

        ts = datetime.now()
        run_root_name = self._make_inference_run_root_name(ts)
        base_out = self._inference_run_root_dir(run_root_name)
        keep_txt = self.save_txt_checkbox.isChecked()
        want_csv = self.convert_txt_to_csv_checkbox.isChecked()
        csv_output_mode = self.csv_output_mode_combo.currentText()
        need_txt_output = keep_txt or want_csv

        def norm(p):
            return str(p).replace("\\", "/")
        
        model_path = norm(model_path)
        base_out.mkdir(parents=True, exist_ok=True)
        base_out = norm(base_out)

        for name, src in sources:
            src = norm(src)

            if self.tracking_radio.isChecked():
                tracker_name = self.track_method_combo.currentText() + ".yaml"
                cmd = [
                    "yolo", "track",
                    f"model={model_path}",
                    f"tracker={tracker_name}",
                    f"source={src}",
                    f"project={base_out}",
                    f"name={name}",
                ]
            else:
                cmd = [
                    "yolo", "pose", "predict",
                    f"model={model_path}",
                    f"source={src}",
                    f"project={base_out}",
                    f"name={name}",
                ]

            for k, v in infer_params.items():
                if v in ["", "None"]:
                    continue
                if isinstance(v, bool):
                    if v:
                        cmd.append(f"{k}=True")
                else:
                    cmd.append(f"{k}={v}")

            if self.show_tracking_checkbox.isChecked():
                cmd.append("show=True")
            if self.save_media_checkbox.isChecked():
                cmd.append("save=True")
            if need_txt_output:
                cmd.append("save_txt=True")

            self.command_queue.append(
                {
                    "command": cmd,
                    "run_root_name": run_root_name,
                    "source_name": name,
                    "keep_txt": keep_txt,
                    "want_csv": want_csv,
                    "csv_output_mode": csv_output_mode,
                }
            )

        if not self.command_queue:
            QMessageBox.warning(self, "Warning", "No valid inference command was generated.")
            return

        if not pose_execution_state.acquire("inference", owner=self):
            QMessageBox.information(
                self,
                "Pose task already running",
                "Another pose task is running. Please try again later.",
            )
            return

        self._inference_running = True
        self._stop_requested = False
        self._completed_commands = 0
        self._total_commands = len(self.command_queue)
        self._set_inference_config_controls_enabled(False)
        self._on_pose_task_busy_changed(True, "inference")
        pose_execution_state.update_progress(
            self._completed_commands,
            self._total_commands,
            f"Inference queued ({self._completed_commands}/{self._total_commands})",
        )

        try:
            self.run_next_command()
        except Exception as err:
            self._finish_inference_run(success=False)
            QMessageBox.critical(
                self,
                "Inference start failed",
                f"Failed to start inference:\n{err}",
            )

    def run_next_command(self):
        if self._stop_requested:
            self.command_queue.clear()
            self._finish_inference_run(success=False, cancelled=True)
            return

        if not self.command_queue:
            print("All inference tasks completed.")
            self._finish_inference_run()
            return

        self.current_run_item = self.command_queue.pop(0)
        command = self.current_run_item["command"]
        source_name = self.current_run_item.get("source_name", "")
        pose_execution_state.update_progress(
            self._completed_commands,
            self._total_commands,
            f"Inference running: {source_name} ({self._completed_commands + 1}/{self._total_commands})",
        )
        print("Executing:", command)

        self.infer_thread = InferenceThread(command)
        self.infer_thread.finished_signal.connect(self._on_inference_command_finished)
        self.infer_thread.start()

    def _on_inference_command_finished(self):
        thread = self.infer_thread
        self.infer_thread = None

        if self._stop_requested or (thread is not None and thread.was_stopped):
            print("[Inference] Stop confirmed. Cleaning up queued tasks.", flush=True)
            self.command_queue.clear()
            self.current_run_item = None
            self._finish_inference_run(success=False, cancelled=True)
            return

        run_item = self.current_run_item or {}
        run_root_name = run_item.get("run_root_name")
        source_name = run_item.get("source_name")
        keep_txt = bool(run_item.get("keep_txt"))
        want_csv = bool(run_item.get("want_csv"))
        csv_output_mode = str(run_item.get("csv_output_mode", "Raw CSV only"))
        if run_root_name and source_name and (keep_txt or want_csv):
            try:
                self._finalize_inference_outputs(
                    run_root_name,
                    source_name,
                    keep_txt,
                    want_csv,
                    csv_output_mode,
                )
            except Exception as err:
                QMessageBox.warning(
                    self,
                    "Inference output finalization failed",
                    f"Run: {run_root_name}\nSource: {source_name}\n{err}",
                )
        self._completed_commands += 1
        pose_execution_state.update_progress(
            self._completed_commands,
            self._total_commands,
            f"Inference completed {self._completed_commands}/{self._total_commands}",
        )
        self.current_run_item = None
        self.run_next_command()

    def _set_inference_config_controls_enabled(self, enabled: bool):
        widgets = [
            getattr(self, "model_line_edit", None),
            getattr(self, "model_browse_btn", None),
            getattr(self, "inference_radio", None),
            getattr(self, "tracking_radio", None),
            getattr(self, "track_method_combo", None),
            getattr(self, "video_group", None),
            getattr(self, "target_group", None),
            getattr(self, "inference_group", None),
            getattr(self, "visualization_group", None),
        ]
        for widget in widgets:
            if widget is not None:
                widget.setEnabled(enabled)
        if enabled and hasattr(self, "track_method_combo"):
            self.update_mode()

    def _stop_inference(self):
        active_task = (pose_execution_state.active_task() or "").lower()
        if not self._inference_running and active_task != "inference":
            return
        print("[Inference] Stop requested by user.", flush=True)
        self._stop_requested = True
        self.command_queue.clear()
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Stopping inference...")
        pose_execution_state.update_progress(
            self._completed_commands,
            self._total_commands,
            "Stopping inference...",
        )
        if self.infer_thread is not None and self.infer_thread.isRunning():
            self.infer_thread.stop()
            return

        owner = pose_execution_state.active_owner()
        if owner is not None and owner is not self:
            owner_thread = getattr(owner, "infer_thread", None)
            if owner_thread is not None and owner_thread.isRunning():
                stop_fn = getattr(owner_thread, "stop", None)
                if callable(stop_fn):
                    stop_fn()
                    return

        self._finish_inference_run(success=False, cancelled=True)

    def _finish_inference_run(self, success: bool = True, cancelled: bool = False):
        if not self._inference_running:
            return

        self._inference_running = False
        self._completed_commands = self._total_commands
        self._stop_requested = False

        pose_execution_state.release(owner=self)
        self._set_inference_config_controls_enabled(True)
        self._on_pose_task_busy_changed(
            pose_execution_state.is_busy(),
            pose_execution_state.active_task() or "",
        )
        if cancelled:
            print("[Inference] Stopped.", flush=True)
            QMessageBox.information(self, "Stopped", "Inference stopped.")
        elif success:
            print("[Inference] Completed.", flush=True)
            QMessageBox.information(self, "Done", "Inference Completed")

    def _on_pose_task_busy_changed(self, busy: bool, task_name: str):
        if not hasattr(self, "run_btn"):
            return

        active = (task_name or "").lower()
        if busy and active == "inference":
            self._set_inference_config_controls_enabled(False)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Stop Inference")
            return

        if busy:
            self._set_inference_config_controls_enabled(False)
            self.run_btn.setEnabled(False)
            if active == "training":
                self.run_btn.setText("Run Inference (Disabled during training)")
            else:
                self.run_btn.setText("Run Inference")
        else:
            self._set_inference_config_controls_enabled(True)
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run Inference")

    def closeEvent(self, event):
        if self._inference_running:
            self.hide()
            event.ignore()
            return
        super().closeEvent(event)

    @staticmethod
    def _extract_frame_number(filename: str) -> int:
        match = re.search(r"_(\d+)\.txt$", filename)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\.txt$", filename)
        return int(match.group(1)) if match else -1

    def _project_kpt_names(self) -> list[str]:
        skeleton_model = SkeletonModel()
        skeleton_model.load_from_dict(self.current_project.skeleton_data)
        _, _, kpt_names = skeleton_model.create_training_config()
        return list(kpt_names)

    def _convert_txt_result_to_csv(
        self,
        run_root_name: str,
        source_name: str,
        txt_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        if txt_dir is None:
            txt_dir = self._resolve_inference_txt_dir(run_root_name, source_name)
        if txt_dir is None or not txt_dir.is_dir():
            return None

        txt_files = sorted(
            txt_dir.glob("*.txt"),
            key=lambda p: self._extract_frame_number(p.name),
        )
        if not txt_files:
            return None

        kpt_names = self._project_kpt_names()
        if not kpt_names:
            return None

        rows = []
        per_id_limit = self.current_project.get_max_instances_per_id()
        include_instance_id = per_id_limit > 1

        for idx, txt_path in enumerate(txt_files):
            with txt_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            detections, frame_has_instance_id = parse_txt_pose_detections(lines)
            include_instance_id = include_instance_id or frame_has_instance_id

            frame_num = self._extract_frame_number(txt_path.name)
            if frame_num < 0:
                frame_num = idx + 1

            track_data, frame_uses_instance_id = resolve_frame_track_data(
                detections,
                keypoint_count=len(kpt_names),
                per_id_limit=per_id_limit,
            )
            include_instance_id = include_instance_id or frame_uses_instance_id

            for track_id, kpt_data, remapped_id in track_data:
                track_name = (
                    self.animals_name[track_id]
                    if 0 <= track_id < len(self.animals_name)
                    else f"track_{track_id}"
                )
                row = {
                    "track": track_name,
                    "frame_idx": frame_num,
                    "instance.score": 0.9,
                }
                for kp in range(len(kpt_names)):
                    base = kp * 3
                    if base + 2 < len(kpt_data):
                        x, y, conf = kpt_data[base:base + 3]
                    else:
                        x, y, conf = 0.0, 0.0, 0.0
                    kp_name = kpt_names[kp]
                    row[f"{kp_name}.x"] = x
                    row[f"{kp_name}.y"] = y
                    row[f"{kp_name}.score"] = conf
                if remapped_id is not None:
                    row["instance.id"] = remapped_id
                rows.append(row)

        if not rows:
            return None

        columns = ["track", "frame_idx", "instance.score"]
        for name in kpt_names:
            columns += [f"{name}.x", f"{name}.y", f"{name}.score"]
        if include_instance_id:
            columns.append("instance.id")

        df = pd.DataFrame(rows, columns=columns)
        csv_path = self._inference_source_csv_path(run_root_name, source_name)
        df.to_csv(csv_path, index=False)
        return csv_path
