from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMainWindow
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import os

class MainWindow(QMainWindow):
    def __init__(self, controller=None):
        super().__init__()
        self.setWindowTitle("Move Altogether: MoVal")
        self.setGeometry(100, 100, 600, 330) 

        self.controller = controller

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        self.button_layout = QVBoxLayout()
        self.button_layout.setSpacing(5) 
        self.button_layout.setContentsMargins(5, 5, 5, 5) 
        main_layout.addLayout(self.button_layout, 1)

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
        title_label = QLabel("Welcome to MovAl")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_layout.addWidget(title_label)

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
        pose_btn = QPushButton("Pose Estimation")
        pose_btn.setFixedHeight(25)
        pose_btn.setMinimumWidth(120)
        pose_btn.clicked.connect(self.controller.run_pose_estimation)
        self.button_layout.addWidget(pose_btn)

        optional_label = QLabel("Optional")
        optional_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.button_layout.addWidget(optional_label)
        extract_btn = QPushButton("Data Extract")
        extract_btn.setFixedHeight(25)
        extract_btn.setMinimumWidth(120)
        extract_btn.clicked.connect(self.controller.data_extract)
        self.button_layout.addWidget(extract_btn)

