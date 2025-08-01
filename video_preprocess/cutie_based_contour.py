from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton,
    QTextEdit, QProgressBar, QLabel,
    QDialog, QLineEdit, QMessageBox, QSpinBox, QProgressBar
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QThreadPool
import os
import glob
from .thread import ContourWorker
from pathlib import Path

class BatchContourProcessor(QObject):
    all_done   = pyqtSignal()       
    any_error  = pyqtSignal(str) 
    progress   = pyqtSignal(int,int) 

    def __init__(self, parent, current_project, max_threads: int = 4):
        super().__init__(parent)
        self.parent = parent

        self.current_project = current_project
        self.pool            = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max_threads)

        self._total  = 0
        self._done   = 0
        self._threads: list[ContourWorker] = []

    def start(self):
        base = Path(self.current_project.project_dir) / "frames"
        if not base.exists():
            self.any_error.emit(f"'frame' directory not found:\n{base}")
            return

        workspaces = [p for p in base.iterdir() if p.is_dir()]
        if not workspaces:
            self.any_error.emit(f"No workspace folders in:\n{base}")
            return

        self._total = len(workspaces)
        self.progress.emit(0, self._total)

        for ws in workspaces:
            try:
                self._launch_worker(ws)
            except Exception as e:
                self.any_error.emit(f"{ws.name}: {e}")

    def _launch_worker(self, workspace_path: Path):
        video_name  = workspace_path.name
        masks_path  = workspace_path / "masks"
        seg_path    = workspace_path / "visualization" / "davis"
        output_dir  = workspace_path / "visualization" / "contour"

        masks      = sorted(str(p) for p in masks_path.glob("*.png"))
        seg_frames = sorted(str(p) for p in seg_path.glob("*.jpg"))

        if not masks or not seg_frames:
            raise FileNotFoundError("No masks or segmented frames found.")

        if len(masks) != len(seg_frames):
            print(f"{video_name}: #masks ≠ #frames ({len(masks)} vs {len(seg_frames)})\nCheck files if necessary.")

        worker = ContourWorker(video_name, seg_frames, masks, str(output_dir))

        def _done_callback(vn, w=worker):
            self._threads.remove(w)
            self._on_worker_done(vn)

        worker.finished.connect(lambda vn=video_name: _done_callback(vn))
        worker.finished.connect(worker.deleteLater)

        self._threads.append(worker)  
        worker.start()

    def _on_worker_done(self, video_name: str):
        self._done += 1

        QMessageBox.information(
            self.parent,
            "Done",
            f"{video_name} contour saved."
        )

        self.progress.emit(self._done, self._total)
        if self._done == self._total:
            self.all_done.emit()