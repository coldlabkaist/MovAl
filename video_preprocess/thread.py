from PyQt6.QtCore import QThread, pyqtSignal
import subprocess
from .contour import ContouredVideoProduction
import sys

class ContourWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, video_name, seg_frames, masks, output_dir):
        super().__init__()
        self.video_name = video_name
        self.seg_frames = seg_frames
        self.masks = masks
        self.output_dir = output_dir

    def run(self):
        ContouredVideoProduction(
            output_video_name=self.video_name,
            segmented_frames=self.seg_frames,
            masks=self.masks,
            output_dir=self.output_dir,
            progress_callback=self.report_progress
        )
        self.finished.emit()

    def report_progress(self, value):
        self.progress.emit(value)
        