from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox
from pathlib import Path
from .segment import CutieDialog
from .cutie_based_contour import BatchContourProcessor, VideoMultiSelectDialog

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
        base = Path(self.current_project.project_dir) / "frames"
        if not base.exists():
            QMessageBox.critical(self, "Error", f"'frames' directory not found:\n{base}")
            return
        dlg = VideoMultiSelectDialog(self, self.current_project)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dlg.selected_names()
        if not selected:
            QMessageBox.information(self, "Contour", "Please select at least one video.")
            return
        reply = QMessageBox.question(
            self,
            "Generate Contours",
            f"Selected {len(selected)} Video(s) for contour generation.\n",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        processor = BatchContourProcessor(self, self.current_project, max_threads=4, include_only=selected)
        processor.any_error.connect(lambda msg: QMessageBox.critical(self, "Error", msg))
        processor.progress.connect(lambda done, total: print(f"[Batch] {done}/{total} videos finished"))
        processor.all_done.connect(lambda: QMessageBox.information(self, "Batch", "All contours finished."))
        processor.start()