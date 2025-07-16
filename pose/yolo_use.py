from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog, QLineEdit, QApplication, QMessageBox, QSpinBox, QFileDialog, QGroupBox, QFormLayout,
    QCheckBox, QComboBox, QDoubleSpinBox, QRadioButton
)
from PyQt6.QtCore import Qt
import subprocess
import os
from .thread import TrainThread, InferenceThread
import sys
from datetime import datetime
import yaml
from pathlib import Path

class YOLODialog(QDialog):
    def __init__(self, current_project, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Train Config")
        self.setFixedSize(1000, 800)

        self.current_project = current_project
        self.yaml_path = os.path.join(current_project.project_dir, "runs", "training_config.yaml")

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
                project_root,
                "PyTorch model (*.pt);;All Files (*)"
            )
            if not model_path:
                QMessageBox.warning(
                    self,
                    "No model selected",
                    "Please select a pre-trained model"
                )
                return
        else:
            QMessageBox.critical(
                self,
                "Invalid model type",
                f"Unknown model_type: {model_type}"
            )
            return

        yaml_path = self.yaml_path
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
        self.setWindowTitle("YOLO Inference Config")
        self.setFixedSize(800, 450)

        self.current_project = current_project

        main_layout = QVBoxLayout(self)

        model_layout = QHBoxLayout()
        self.model_line_edit = QLineEdit()
        model_btn = QPushButton("Browse")
        model_btn.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_line_edit)
        model_layout.addWidget(model_btn)
        main_layout.addLayout(model_layout)

        track_mode_layout = QHBoxLayout()
        self.inference_radio = QRadioButton("Inference")
        self.tracking_radio = QRadioButton("Tracking")
        self.inference_radio.setChecked(True) 
        self.inference_radio.toggled.connect(self.update_mode)
        self.tracking_radio.toggled.connect(self.update_mode)

        self.track_method_combo = QComboBox()
        self.track_method_combo.addItems(["botsort", "bytetrack"])
        self.track_method_combo.setEnabled(False) 

        track_mode_layout.addWidget(self.inference_radio)
        track_mode_layout.addWidget(self.tracking_radio)
        track_mode_layout.addWidget(QLabel("Tracking Method:"))
        track_mode_layout.addWidget(self.track_method_combo)

        main_layout.addLayout(track_mode_layout)

        inference_params = {
            "source": "",
            "imgsz": 640,
            "conf": 0.5,
            "iou": 0.7,
            "max_det": 15,
            "augment": False,
            "classes": "",
            "half": False,
            "device": "None"
        }
        self.inference_group = self.create_group_box("Inference Config", inference_params, is_inference=True)

        visualization_params = {
            "show": False,
            "save": True,
            "save_txt": True,
            "show_labels": True,
            "show_conf": True,
            "show_boxes": True,
            "kpt_radius": 4
        }
        self.visualization_group = self.create_group_box("Visualization", visualization_params)

        middle_layout = QHBoxLayout()
        middle_layout.addWidget(self.inference_group, 1)
        middle_layout.addWidget(self.visualization_group, 1)
        main_layout.addLayout(middle_layout)

        run_btn = QPushButton("Run")
        run_btn.setFixedHeight(30)
        run_btn.clicked.connect(self.run_inference)
        main_layout.addWidget(run_btn)

    def create_group_box(self, title, params, is_inference=False):
        group = QGroupBox(title)
        form = QFormLayout()

        for key, default in params.items():
            if is_inference and key == "source":
                h_layout = QHBoxLayout()
                line_edit = QLineEdit()
                btn = QPushButton("Select Data")
                btn.clicked.connect(lambda _, le=line_edit: self.select_source(le))
                h_layout.addWidget(line_edit)
                h_layout.addWidget(btn)
                form.addRow(QLabel(key), h_layout)
            else:
                if isinstance(default, bool):
                    widget = QCheckBox()
                    widget.setChecked(default)
                elif isinstance(default, float):
                    widget = QDoubleSpinBox()
                    widget.setDecimals(4)
                    widget.setMaximum(10000)
                    widget.setValue(default)
                elif isinstance(default, int):
                    widget = QSpinBox()
                    widget.setMaximum(10000)
                    widget.setValue(default)
                else:
                    widget = QLineEdit()
                    widget.setText(str(default))

                form.addRow(QLabel(key), widget)

        group.setLayout(form)
        return group

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pt *.onnx *.engine)")
        if file_path:
            self.model_line_edit.setText(file_path)

    def select_source(self, line_edit):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Video Files", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_paths:
            line_edit.setText(";".join(file_paths))
            
    def update_mode(self):
        if self.tracking_radio.isChecked():
            self.track_method_combo.setEnabled(True)
        else:
            self.track_method_combo.setEnabled(False)

    def get_params_from_group(self, group):
        params = {}
        layout = group.layout()
        for i in range(layout.rowCount()):
            label = layout.itemAt(i, QFormLayout.ItemRole.LabelRole).widget().text()
            field_item = layout.itemAt(i, QFormLayout.ItemRole.FieldRole)

            if label == "source":
                h_layout = field_item.layout()
                line_edit = h_layout.itemAt(0).widget() 
                params[label] = line_edit.text()
                continue

            field = field_item.widget()

            if isinstance(field, QLineEdit):
                params[label] = field.text()
            elif isinstance(field, (QSpinBox, QDoubleSpinBox)):
                params[label] = field.value()
            elif isinstance(field, QCheckBox):
                params[label] = field.isChecked()

        return params
    
    def run_inference(self):
        model_path = self.model_line_edit.text()
        infer_params = self.get_params_from_group(self.inference_group)
        vis_params = self.get_params_from_group(self.visualization_group)

        if "classes" in infer_params:
            classes_val = infer_params["classes"]
            if classes_val.strip():
                infer_params["classes"] = ",".join(classes_val.replace(" ", "").split(","))

        sources = infer_params.get("source", "")
        sources = [s.strip() for s in sources.split(";") if s.strip()]
        self.command_queue = [] 

        ts = datetime.now()
        ts_date = ts.strftime("%y%m%d")
        ts_time = ts.strftime("%H%M%S")
        base_out= os.path.join(self.current_project.project_dir, "predicts")

        for src in sources:
            if self.tracking_radio.isChecked():
                tracker_name = self.track_method_combo.currentText() + ".yaml"
                command = f"yolo track model={model_path} tracker={tracker_name} source={src}"
            else:
                command = f"yolo pose predict model={model_path} source={src}"
            command += f" project={base_out} name=predict_{ts_date}_{ts_time}"
            for k, v in infer_params.items():
                if k == "source" or v in ["", "None"]:
                    continue
                if isinstance(v, bool):
                    if v:
                        command += f" {k}=True"
                else:
                    command += f" {k}={v}"

            for k in ["show", "save", "save_txt"]:
                if vis_params.get(k, False):
                    command += f" {k}=True"

            self.command_queue.append(command)

        self.run_next_command()

    
    def run_next_command(self):
        if not self.command_queue:
            print("All inference tasks completed.")
            return

        command = self.command_queue.pop(0)
        print("▶Executing:", command)

        self.infer_thread = InferenceThread(command)
        self.infer_thread.finished.connect(self.run_next_command) 
        self.infer_thread.start()



