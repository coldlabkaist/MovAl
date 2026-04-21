from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QScrollArea, QSizePolicy, QGridLayout,
    QDialog, QLineEdit, QApplication, QMessageBox, QSpinBox, QFileDialog, QGroupBox, QFormLayout,
    QCheckBox, QComboBox, QDoubleSpinBox, QRadioButton, QListWidget, QListWidgetItem, QFrame, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt
import subprocess
import os
from .thread import TrainThread, InferenceThread
import sys
from datetime import datetime
from pathlib import Path
import re
import pandas as pd
from utils.skeleton.skeleton_model import SkeletonModel

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

class YOLODialog(QDialog):
    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Train Config")
        self.setFixedSize(1000, 800)

        self.current_project = current_project

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

        run_btn = QPushButton("Run Train")
        run_btn.clicked.connect(self.run_train)
        run_btn.setFixedHeight(30)
        main_layout.addWidget(run_btn)

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

        command = "yolo pose train"
        for key, value in params.items():
            if value in ["", "None"]:
                continue
            if isinstance(value, bool):
                if value:
                    command += f" {key}=True"
            else:
                command += f" {key}={value}"

        print("Execute Command:", command)

        self.train_thread = TrainThread(command)
        self.train_thread.finished_signal.connect(
            lambda: QMessageBox.information(self, "Done", "Training Completed")
        )
        self.train_thread.start()

class YoloInferenceDialog(QDialog):
    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.current_project = current_project
        self.animals_name = current_project.animals_name
        self.command_queue = []
        self.current_run_item = None
        self.build_ui()

    def build_ui(self):
        self.setWindowTitle("YOLO Pose Inference")
        self.setMinimumSize(600, 620)

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
        run_btn = QPushButton("Run", clicked=self.run_inference)
        run_btn.setFixedHeight(30)
        main_layout.addWidget(run_btn)

    def build_model_row(self):
        row = QHBoxLayout()
        self.model_line_edit = BrowseOnlyLineEdit()
        self.model_line_edit.setPlaceholderText("Select Model")
        browse_btn = QPushButton("Browse", clicked=self.select_model)
        row.addWidget(self.model_line_edit)
        row.addWidget(browse_btn)
        return row

    def build_track_row(self):
        row = QHBoxLayout()
        self.inference_radio = QRadioButton("Inference", checked=True)
        self.tracking_radio = QRadioButton("Tracking")
        self.track_method_combo = QComboBox(enabled=False)
        self.track_method_combo.addItems(["botsort", "bytetrack"])

        for w in (self.inference_radio, self.tracking_radio):
            w.toggled.connect(self.update_mode)
            row.addWidget(w)
        row.addWidget(QLabel("Tracking Method:"))
        row.addWidget(self.track_method_combo)
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

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(2)
        self.video_checks = []
        for fe in self.current_project.files:
            stem = Path(fe.video).stem
            cb = QCheckBox(stem)
            vbox.addWidget(cb)
            self.video_checks.append((cb, stem))
        vbox.addStretch()
        scroll = QScrollArea(frameShape=QFrame.Shape.NoFrame, widgetResizable=True)
        scroll.setWidget(container)
        lay.addWidget(scroll)

        button_row = QHBoxLayout()
        self.select_all_images_btn = QPushButton("Select All", clicked=self.select_all_image_sources)
        self.deselect_all_images_btn = QPushButton("Deselect All", clicked=self.deselect_all_image_sources)
        button_row.addWidget(self.select_all_images_btn)
        button_row.addWidget(self.deselect_all_images_btn)
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
        self.loaded_list = QListWidget()
        seen = set()
        for fe in self.current_project.files:
            path_obj = Path(fe.video)
            if not path_obj.exists():
                continue
            path = str(path_obj)
            if path in seen:
                continue
            seen.add(path)
            self.loaded_list.addItem(path)
        self.clear_video_btn = QPushButton("Clear List", clicked=self.clear_videos)
        lay.addWidget(self.load_video_btn)
        lay.addWidget(self.loaded_list)
        lay.addWidget(self.clear_video_btn)
        return sect

    def build_target_group(self):
        group = QGroupBox("Inference Target")
        form = QFormLayout(group)
        form.setSpacing(4)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0,0,0,0)
        vbox.setSpacing(2) 
        self.target_checks = []
        for name in self.animals_name:
            cb = QCheckBox(name, checked=True)
            self.target_checks.append(cb)
            vbox.addWidget(cb)
        vbox.addStretch()  

        scroll = QScrollArea(frameShape=QFrame.Shape.NoFrame, widgetResizable=True)
        scroll.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(container)
        form.addRow(scroll)
        return group

    def build_visualization_group(self):
        group = QGroupBox("Visualization")
        form = QFormLayout(group)

        self.show_tracking_checkbox = QCheckBox(checked=False)
        self.save_media_checkbox = QCheckBox(checked=False)
        self.save_txt_checkbox = QCheckBox(checked=True)
        self.convert_txt_to_csv_checkbox = QCheckBox(checked=True)

        form.addRow(QLabel("show tracking result"), self.show_tracking_checkbox)
        form.addRow(QLabel("save image/video"), self.save_media_checkbox)
        form.addRow(QLabel("save txt"), self.save_txt_checkbox)
        form.addRow(QLabel("convert txt to csv"), self.convert_txt_to_csv_checkbox)

        self.save_txt_checkbox.toggled.connect(self._update_visualization_option_states)
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

    def select_all_image_sources(self):
        for cb, _ in self.video_checks:
            cb.setChecked(True)

    def deselect_all_image_sources(self):
        for cb, _ in self.video_checks:
            cb.setChecked(False)

    def load_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select video files",
            self.current_project.project_dir, 
            "Video (*.mp4 *.avi *.mov)"
        )
        for p in paths:
            if p and not any(p == self.loaded_list.item(i).text() for i in range(self.loaded_list.count())):
                self.loaded_list.addItem(p)

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
            selected_names: list[str] = [
                stem for cb, stem in self.video_checks if cb.isChecked()
            ]
            if not selected_names:
                return None
            image_mode = self.image_mode_combo.currentText() 
            base_dir = Path(self.current_project.project_dir)
            if image_mode in ("davis", "contour"):
                sources = [
                    (name, base_dir / "frames" / name / "visualization" / image_mode)
                    for name in selected_names
                ]
            else:
                sources = [
                    (name, base_dir / "frames" / name / "images")
                    for name in selected_names
                ]
            return sources
        elif self.video_radio.isChecked():
            count = self.loaded_list.count()
            if count == 0:
                return None
            sources = []
            for i in range(count):
                src = Path(self.loaded_list.item(i).text())
                sources.append((src.stem, src))
            return sources
        raise
        
    def get_inference_target(self):
        classes = [idx for idx, cb in enumerate(self.target_checks) if cb.isChecked()]
        max_det = len(classes)
        return classes, max_det

    def _update_visualization_option_states(self):
        if hasattr(self, "show_tracking_checkbox"):
            if self.image_radio.isChecked():
                self.show_tracking_checkbox.setChecked(False)
                self.show_tracking_checkbox.setEnabled(False)
            else:
                self.show_tracking_checkbox.setEnabled(True)

        if hasattr(self, "save_txt_checkbox") and hasattr(self, "convert_txt_to_csv_checkbox"):
            if self.save_txt_checkbox.isChecked():
                self.convert_txt_to_csv_checkbox.setEnabled(True)
            else:
                self.convert_txt_to_csv_checkbox.setChecked(False)
                self.convert_txt_to_csv_checkbox.setEnabled(False)

    def run_inference(self):
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
        ts_date = ts.strftime("%y%m%d")
        ts_time = ts.strftime("%H%M%S")
        base_out= os.path.join(self.current_project.project_dir, "predicts")

        def norm(p):
            return str(p).replace("\\", "/")
        
        model_path = norm(model_path)
        base_out = norm(base_out)

        for name, src in sources:
            src = norm(src)

            run_name = f"predict__{name}_{ts_date}_{ts_time}"

            if self.tracking_radio.isChecked():
                tracker_name = self.track_method_combo.currentText() + ".yaml"
                cmd = [
                    "yolo", "track",
                    f"model={model_path}",
                    f"tracker={tracker_name}",
                    f"source={src}",
                    f"project={base_out}",
                    f"name={run_name}",
                ]
            else:
                cmd = [
                    "yolo", "pose", "predict",
                    f"model={model_path}",
                    f"source={src}",
                    f"project={base_out}",
                    f"name={run_name}",
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
            if self.save_txt_checkbox.isChecked():
                cmd.append("save_txt=True")

            self.command_queue.append(
                {
                    "command": cmd,
                    "run_name": run_name,
                }
            )

        self.run_next_command()

    def run_next_command(self):
        if not self.command_queue:
            print("All inference tasks completed.")
            return

        self.current_run_item = self.command_queue.pop(0)
        command = self.current_run_item["command"]
        print("Executing:", command)

        self.infer_thread = InferenceThread(command)
        self.infer_thread.finished.connect(self._on_inference_command_finished)
        self.infer_thread.start()

    def _on_inference_command_finished(self):
        run_item = self.current_run_item or {}
        run_name = run_item.get("run_name")
        if (
            run_name
            and self.save_txt_checkbox.isChecked()
            and self.convert_txt_to_csv_checkbox.isChecked()
        ):
            try:
                self._convert_txt_result_to_csv(run_name)
            except Exception as err:
                QMessageBox.warning(
                    self,
                    "TXT to CSV conversion failed",
                    f"Run: {run_name}\n{err}",
                )
        self.current_run_item = None
        self.run_next_command()

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

    def _convert_txt_result_to_csv(self, run_name: str):
        run_dir = Path(self.current_project.project_dir) / "predicts" / run_name
        txt_dir = run_dir / "labels"
        if not txt_dir.is_dir():
            return

        txt_files = sorted(
            txt_dir.glob("*.txt"),
            key=lambda p: self._extract_frame_number(p.name),
        )
        if not txt_files:
            return

        kpt_names = self._project_kpt_names()
        if not kpt_names:
            return

        rows = []
        has_instance_id = False

        for idx, txt_path in enumerate(txt_files):
            with txt_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            detections = []
            for line in lines:
                items = line.strip().split()
                if len(items) < 6:
                    continue
                try:
                    track_id = int(float(items[0]))
                except Exception:
                    continue
                raw = items[5:]

                if len(raw) % 3 == 1:
                    try:
                        instance_id = int(float(raw[-1]))
                        kpt_data = list(map(float, raw[:-1]))
                        has_instance_id = True
                    except Exception:
                        continue
                else:
                    instance_id = None
                    try:
                        kpt_data = list(map(float, raw))
                    except Exception:
                        continue

                remapped_id = instance_id if instance_id is not None else ""
                detections.append((track_id, remapped_id, kpt_data))

            frame_num = self._extract_frame_number(txt_path.name)
            if frame_num < 0:
                frame_num = idx + 1

            track_data = {}
            for track_id, remapped_id, kpt_data in detections:
                key = (track_id, remapped_id if remapped_id != "" else None)
                if key not in track_data:
                    track_data[key] = (kpt_data, remapped_id)
                    continue
                prev, rid = track_data[key]
                for kp in range(min(len(kpt_names), len(prev) // 3, len(kpt_data) // 3)):
                    if kpt_data[kp * 3 + 2] > prev[kp * 3 + 2]:
                        prev[kp * 3:kp * 3 + 3] = kpt_data[kp * 3:kp * 3 + 3]
                track_data[key] = (prev, rid)

            for (track_id, _), (kpt_data, remapped_id) in track_data.items():
                track_name = (
                    self.animals_name[track_id]
                    if 0 <= track_id < len(self.animals_name)
                    else f"track_{track_id}"
                )
                row = [track_name, frame_num, 0.9]
                for kp in range(len(kpt_names)):
                    base = kp * 3
                    if base + 2 < len(kpt_data):
                        x, y, conf = kpt_data[base:base + 3]
                    else:
                        x, y, conf = 0.0, 0.0, 0.0
                    row.extend([x, y, conf])
                if has_instance_id:
                    row.append(remapped_id)
                rows.append(row)

        if not rows:
            return

        columns = ["track", "frame_idx", "instance.score"]
        for name in kpt_names:
            columns += [f"{name}.x", f"{name}.y", f"{name}.score"]
        if has_instance_id:
            columns.append("instance.id")

        df = pd.DataFrame(rows, columns=columns)
        csv_path = run_dir / "inference_result.csv"
        df.to_csv(csv_path, index=False)
