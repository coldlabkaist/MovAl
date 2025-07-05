from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel
from .segment import CutieDialog
from .cutie_based_contour import ContourDialog

class PreprocessDialog(QDialog):
    def __init__(self, parent=None, current_project = None):
        super().__init__(parent)
        self.current_project = current_project
        
        self.setWindowTitle("Preprocess Options")
        self.setFixedSize(350, 220) 

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Step 1"))
        segment_btn = QPushButton("Segment")
        segment_btn.setFixedHeight(40) 
        segment_btn.clicked.connect(self.open_segment)
        layout.addWidget(segment_btn)

        layout.addSpacing(10)
        layout.addWidget(QLabel("Step 2 (optional)"))
        contour_btn = QPushButton("Contour")
        contour_btn.setFixedHeight(40)
        contour_btn.clicked.connect(self.open_contour)
        layout.addWidget(contour_btn)

        self.setLayout(layout)

    def open_segment(self):
        dialog = CutieDialog(self)
        dialog.exec()

    def open_contour(self):
        dialog = ContourDialog(self)
        dialog.exec()
