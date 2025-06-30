from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QDialog, QLineEdit, QApplication, QMessageBox, QSpinBox, QFileDialog
)
from PyQt6.QtGui import QClipboard
import subprocess
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

    def on_manual_cutie(self):
        dialog = CutieInstallDialog(self)
        dialog.exec()

    def on_manual_yolo(self):
        #dialog = YoloInstallDialog(self)
        #dialog.exec()
        pass

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