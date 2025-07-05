from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog, QLineEdit, QApplication, QMessageBox, QSpinBox, QFileDialog
)
from PyQt6.QtGui import QClipboard
from PyQt6.QtCore import Qt
import subprocess
import shutil
import os

class MainInstallDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Installation Manager")
        self.setFixedSize(200, 150)

        layout = QVBoxLayout()

        self.one_click_btn = QPushButton("One-Click Install")
        self.manual_cutie_btn = QPushButton("Manual Install (Cutie)")
        self.manual_yolo_btn = QPushButton("Manual Install (YOLO)")

        layout.addWidget(self.one_click_btn)
        layout.addWidget(self.manual_cutie_btn)
        layout.addWidget(self.manual_yolo_btn)

        self.setLayout(layout)

        self.one_click_btn.clicked.connect(self.on_one_click)
        self.manual_cutie_btn.clicked.connect(self.on_manual_cutie)
        self.manual_yolo_btn.clicked.connect(self.on_manual_yolo)

    def on_one_click(self):
        from installation_manager import OneClickInstallDialog
        dialog = OneClickInstallDialog(self)
        dialog.exec()
        self.accept()

    def on_manual_cutie(self):
        dialog = CutieInstallDialog(self)
        dialog.exec()

    def on_manual_yolo(self):
        dialog = YoloInstallDialog(self)
        dialog.exec()

class CutieInstallDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cutie Installation Guide")
        self.setFixedSize(650, 500)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("<b>GitHub:</b>"))
        layout.addLayout(self.add_command_row("https://github.com/hkchengrex/Cutie"))

        layout.addWidget(QLabel("<b>Prerequisites:</b>"))
        layout.addWidget(QLabel("• Python 3.8+"))
        layout.addWidget(QLabel("• PyTorch 1.12+ and torchvision"))

        layout.addWidget(QLabel("<b>Clone Cutie repository:</b>"))
        layout.addLayout(self.add_command_row("git clone https://github.com/hkchengrex/Cutie.git"))
        
        layout.addWidget(QLabel("<b>Install with pip:</b>"))
        layout.addLayout(self.add_command_row("cd Cutie"))
        layout.addLayout(self.add_command_row("pip install -e ."))

        layout.addWidget(QLabel("<b>Download pretrained models:</b>"))
        layout.addLayout(self.add_command_row("python cutie/utils/download_models.py"))
        
        note = QLabel("⚠️ Upgrade pip with: <i>pip install --upgrade pip</i>")
        layout.addWidget(note)

        self.setLayout(layout)

    def add_command_row(self, text: str):
        layout = QHBoxLayout()
        field = QLineEdit()
        field.setText(text)
        field.setReadOnly(True)

        copy_button = QPushButton("Copy")
        copy_button.setFixedWidth(60)
        copy_button.clicked.connect(lambda: self.copy_to_clipboard(text))

        layout.addWidget(field)
        layout.addWidget(copy_button)
        return layout

    def copy_to_clipboard(self, text):
        clipboard: QClipboard = QApplication.clipboard()
        clipboard.setText(text)

class YoloInstallDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Install Guide")
        self.setFixedSize(420, 260) 

        layout = QVBoxLayout()

        req_label = QLabel("<b>Requirements:</b><br>"
                           "Python >= 3.8 / PyTorch >= 1.8 / Recommend the latest version of Ultralytics")
        req_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        req_label.setWordWrap(True)
        layout.addWidget(req_label)
        layout.addSpacing(10)

        install_title = QLabel("<b>-Install-</b>")
        layout.addWidget(install_title)

        install_layout = QHBoxLayout()
        self.install_cmd = QLineEdit("pip install ultralytics")
        self.install_cmd.setReadOnly(True)
        install_copy_btn = QPushButton("Copy")
        install_copy_btn.setFixedWidth(60)
        install_copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.install_cmd.text()))
        install_layout.addWidget(self.install_cmd)
        install_layout.addWidget(install_copy_btn)
        layout.addLayout(install_layout)

        update_title = QLabel("<b>-Update-</b>")
        layout.addWidget(update_title)

        update_layout = QHBoxLayout()
        self.update_cmd = QLineEdit("pip install -U ultralytics")
        self.update_cmd.setReadOnly(True)
        update_copy_btn = QPushButton("Copy")
        update_copy_btn.setFixedWidth(60)
        update_copy_btn.clicked.connect(lambda: self.copy_to_clipboard(self.update_cmd.text()))
        update_layout.addWidget(self.update_cmd)
        update_layout.addWidget(update_copy_btn)
        layout.addLayout(update_layout)

        self.download_btn = QPushButton("Download Pose Models")
        self.download_btn.setFixedHeight(35)
        self.download_btn.clicked.connect(self.download_models)
        layout.addWidget(self.download_btn)

        self.setLayout(layout)

    def copy_to_clipboard(self, text):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

    def download_models(self):
        force_reinstall_yolo = False
        yolo_model_dir = "models"
        if os.path.isdir(yolo_model_dir):
            ans = QMessageBox.question(
                self,
                "Existing Directory Found",
                "A YOLO models already exists.\n"
                "Do you want to delete it and perform a reinstallation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                print("YOLO reinstallation skipped by user.")
            else:
                force_reinstall_yolo = True

        if os.path.isdir(yolo_model_dir):
            if force_reinstall_yolo:
                print("· Deleting existing YOLO models …")
                shutil.rmtree(yolo_model_dir, onerror=_force_remove)
            else:
                print("· Repository already exists. Skipping installation.")
                return
        print("Downloading Models...")

        models = [
            'yolov8n-pose.pt',
            'yolov8s-pose.pt',
            'yolov8m-pose.pt',
            'yolov8l-pose.pt',
            'yolov8x-pose.pt',
            'yolo11n-pose.pt',
            'yolo11s-pose.pt',
            'yolo11m-pose.pt',
            'yolo11l-pose.pt',
            'yolo11x-pose.pt'
        ]

        cwd = os.getcwd()
        weights_dir = os.path.join(cwd, yolo_model_dir)
        os.makedirs(weights_dir, exist_ok=True)

        from ultralytics import YOLO
        for model in models:
            model_path = os.path.join(weights_dir, model)
            print(f". Downloading {model}...")
            YOLO(model)
            os.rename(model, model_path)
        print(". All models downloaded")

def _force_remove(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)
