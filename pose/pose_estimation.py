from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
from pose.prepare_data import PrepareDataDialog
from pose.yolo_use import YOLODialog, YoloInferenceDialog

class PoseEstimationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pose Estimation Options")
        self.setFixedSize(350, 220)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Step 1"))
        data_btn = QPushButton("Prepare datasets")
        data_btn.setFixedHeight(40)
        data_btn.clicked.connect(self.open_prepare_data)
        layout.addWidget(data_btn)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Step 2"))
        train_btn = QPushButton("Train Model")
        train_btn.setFixedHeight(40)
        train_btn.clicked.connect(self.train_yolo)
        layout.addWidget(train_btn)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Step 3"))
        inf_btn = QPushButton("Pose estimation")
        inf_btn.setFixedHeight(40)
        inf_btn.clicked.connect(self.pose_estimation)
        layout.addWidget(inf_btn)

        self.setLayout(layout)

    def open_prepare_data(self):
        dialog = PrepareDataDialog(self)
        dialog.exec()

    def train_yolo(self):
        dialog = YOLODialog(self)
        dialog.exec()
        
    def pose_estimation(self):
        dialog = YoloInferenceDialog(self)
        dialog.exec()