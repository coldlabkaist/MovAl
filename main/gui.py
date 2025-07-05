from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMainWindow, QLineEdit, QFileDialog, QMessageBox, QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from utils.project import ProjectInformation
from pathlib import Path
from typing import Union, Optional
import os

class MainWindow(QMainWindow):
    def __init__(self, controller=None):
        super().__init__()
        self.setWindowTitle("Move Altogether: MoVal")
        self.setGeometry(100, 100, 500, 400) 

        self.controller = controller

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        outer_layout = QVBoxLayout()
        central_widget.setLayout(outer_layout)

        title_label = QLabel("Welcome to MovAl")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer_layout.addWidget(title_label)

        proj_bar = QHBoxLayout()
        proj_bar.addWidget(QLabel("Current project:", self))
        self.proj_name = QLineEdit(self)
        self.proj_name.setReadOnly(True)
        self.proj_name.setPlaceholderText("No project loaded")
        proj_bar.addWidget(self.proj_name, 1) 
        btn_load_yaml = QPushButton("Load YAML…", self)
        btn_load_yaml.clicked.connect(self.on_load_yaml_clicked)
        proj_bar.addWidget(btn_load_yaml)
        outer_layout.addLayout(proj_bar)

        self.current_project = None
        self.controller.set_main_window_project = self.on_load_yaml_clicked

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)  # (Plain/Sunken/Raised)
        line.setFixedHeight(2)
        outer_layout.addSpacing(5)
        outer_layout.addWidget(line)
        outer_layout.addSpacing(5)

        main_layout = QHBoxLayout()
        outer_layout.addLayout(main_layout)

        self.button_layout = QVBoxLayout()
        self.button_layout.setSpacing(5) 
        self.button_layout.setContentsMargins(5, 5, 5, 5) 
        main_layout.addLayout(self.button_layout)
        main_layout.insertStretch(0, 1)
        main_layout.addStretch(1) 

        right_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "background_image.png"))
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaledToWidth(380, Qt.TransformationMode.SmoothTransformation))

        right_layout.addWidget(self.image_label)

        main_layout.addLayout(right_layout, 2)
        
        self.setup_buttons()

    def setup_buttons(self):
        installation_label = QLabel("Installation (Cutie / YOLO)")
        installation_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(installation_label)
        installation_btn = QPushButton("Installation")
        installation_btn.setFixedHeight(25)
        installation_btn.setMinimumWidth(120)
        installation_btn.clicked.connect(self.controller.run_installation)
        self.button_layout.addWidget(installation_btn)

        step1_label = QLabel("Step 1")
        step1_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(step1_label)
        create_project_btn = QPushButton("Create Project")
        create_project_btn.setFixedHeight(25)
        create_project_btn.setMinimumWidth(120)
        create_project_btn.clicked.connect(self.controller.run_project_manager)
        self.button_layout.addWidget(create_project_btn)

        step2_label = QLabel("Step 2")
        step2_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(step2_label)
        preprocess_btn = QPushButton("Preprocess")
        preprocess_btn.setFixedHeight(25)
        preprocess_btn.setMinimumWidth(120)
        preprocess_btn.clicked.connect(self.controller.run_video_preprocess)
        self.button_layout.addWidget(preprocess_btn)

        step3_label = QLabel("Step 3")
        step3_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(step3_label)
        label_btn = QPushButton("Labelary")
        label_btn.setFixedHeight(25)
        label_btn.setMinimumWidth(120)
        label_btn.clicked.connect(self.controller.run_labelary)
        self.button_layout.addWidget(label_btn)
        
        step4_label = QLabel("Step 4")
        step4_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(step4_label)
        pose_btn = QPushButton("Pose Estimation")
        pose_btn.setFixedHeight(25)
        pose_btn.setMinimumWidth(120)
        pose_btn.clicked.connect(self.controller.run_pose_estimation)
        self.button_layout.addWidget(pose_btn)

        optional_label = QLabel("Additional Tools")
        optional_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(optional_label)
        extract_btn = QPushButton("Data Extract (txt_to_csv)")
        extract_btn.setFixedHeight(25)
        extract_btn.setMinimumWidth(120)
        extract_btn.clicked.connect(self.controller.data_extract)
        self.button_layout.addWidget(extract_btn)

    def on_load_yaml_clicked(
        self,
        checked: bool = False,
        path: Optional[Union[str, Path]] = None
    ) -> None:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select project YAML",
                "",
                "YAML files (*.yaml *.yml)"
            )
            if not path:
                return
        else:
            path = str(path)

        try:
            self.current_project = ProjectInformation.from_yaml(path)
            self.controller.current_project = self.current_project
            self.proj_name.setText(self.current_project.title or Path(path).stem)

        except FileNotFoundError as fnf:
            QMessageBox.warning(self, "File not found", str(fnf))
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))