from PyQt6.QtWidgets import (
    QVBoxLayout, QPushButton,
    QTextEdit, QProgressBar, QLabel,
    QDialog, QLineEdit, QMessageBox, QSpinBox, QProgressBar,
    QCheckBox, QDialogButtonBox, QHBoxLayout, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt, QObject, pyqtSignal
import os
import glob
from .thread import ContourWorker
from pathlib import Path
from typing import Optional, List

class VideoMultiSelectDialog(QDialog):
    def __init__(self, parent, current_project):
        super().__init__(parent)
        self.setWindowTitle("Select Videos for Contour")
        self.setMinimumSize(420, 480)
        self._checks: list[QCheckBox] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select videos to generate contours."))

        base = Path(current_project.project_dir) / "frames"
        names = sorted([p.name for p in base.iterdir() if p.is_dir()]) if base.exists() else []

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        box = QWidget(); v = QVBoxLayout(box); v.setContentsMargins(6,6,6,6); v.setSpacing(4)
        for name in names:
            cb = QCheckBox(name); cb.setChecked(True)
            v.addWidget(cb)
            self._checks.append(cb)
        v.addStretch(1)
        scroll.setWidget(box)
        layout.addWidget(scroll, 1)

        tools = QHBoxLayout()
        btn_all   = QPushButton("Select All")
        btn_none  = QPushButton("Select None")
        btn_inv   = QPushButton("Select Invert")
        tools.addWidget(btn_all); tools.addWidget(btn_none); tools.addWidget(btn_inv)
        layout.addLayout(tools)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(btns)

        btn_all.clicked.connect(lambda: [cb.setChecked(True) for cb in self._checks])
        btn_none.clicked.connect(lambda: [cb.setChecked(False) for cb in self._checks])
        btn_inv.clicked.connect(lambda: [cb.setChecked(not cb.isChecked()) for cb in self._checks])
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def selected_names(self) -> list[str]:
        return [cb.text() for cb in self._checks if cb.isChecked()]

class BatchContourProcessor(QObject):
    all_done   = pyqtSignal()       
    any_error  = pyqtSignal(str) 
    progress   = pyqtSignal(int,int) 

    def __init__(self, parent, current_project, max_threads: int = 4, include_only: Optional[List[str]] = None):
        super().__init__(parent)
        self.parent = parent

        self.current_project = current_project
        self._max_parallel   = max_threads
        self._include_only   = set(include_only) if include_only else None

        self._total  = 0
        self._done   = 0
        self._active: list[ContourWorker] = []
        self._pending: list[Path] = []

    def start(self):
        base = Path(self.current_project.project_dir) / "frames"
        if not base.exists():
            self.any_error.emit(f"'frame' directory not found:\n{base}")
            return

        workspaces = [p for p in base.iterdir() if p.is_dir()]
        if self._include_only is not None:
            name_set = self._include_only
            workspaces = [p for p in workspaces if p.name in name_set]
        if not workspaces:
            self.any_error.emit(f"No workspace folders in:\n{base}")
            msg = f"No workspace folders in:\n{base}" if self._include_only is None \
                  else "No selected videos to process."
            self.any_error.emit(msg)
            return

        print(f"[Batch] Starting contour for {len(workspaces)} videos with max {self._max_parallel} threads.")

        self._total = len(workspaces)
        #self.progress.emit(0, self._total)

        self._pending = workspaces.copy()
        self._active = []
        self._fill_slots()

    def _launch_worker(self, workspace_path: Path) -> ContourWorker:
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
            if w in self._active:
                self._active.remove(w)
            self._on_worker_done(vn)

        worker.finished.connect(lambda vn=video_name: _done_callback(vn))
        worker.finished.connect(worker.deleteLater)
        return worker

    def _on_worker_done(self, video_name: str):
        self._done += 1
        try:
            print(f"[Contour] {video_name} done")
        except Exception:
            pass

        self.progress.emit(self._done, self._total)
        self._fill_slots()
        if self._done == self._total and not self._pending and not self._active:
            self.all_done.emit()

    def _fill_slots(self):
        while self._pending and len(self._active) < self._max_parallel:
            ws = self._pending.pop(0)
            try:
                worker = self._launch_worker(ws)
                self._active.append(worker)
                worker.start()
            except Exception as e:
                self.any_error.emit(f"{ws.name}: {e}")
                self._done += 1
                self.progress.emit(self._done, self._total)