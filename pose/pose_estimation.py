from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
from pose.data_converter import DataConverter
from pose.yolo_use import YOLODialog

class PoseEstimationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pose Estimation Options")
        self.setFixedSize(350, 220)

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Step 1"))
        slp_btn = QPushButton("Prepare datasets")
        slp_btn.setFixedHeight(40)
        slp_btn.clicked.connect(self.open_data_convert)
        layout.addWidget(slp_btn)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Step 2"))
        yolo_btn = QPushButton("Pose estimation")
        yolo_btn.setFixedHeight(40)
        yolo_btn.clicked.connect(self.open_yolo)
        layout.addWidget(yolo_btn)

        self.setLayout(layout)

    def open_data_convert(self):
        dialog = DataConverter(self)
        dialog.exec()

    def open_yolo(self):
        dialog = YOLODialog(self)
        dialog.exec()