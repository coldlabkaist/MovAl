from PyQt6.QtCore import QThread, pyqtSignal
import subprocess
import sys
import os
import cv2

class ContourWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, video_name, seg_frames, masks, output_dir, throttle_ms: int = 0):
        super().__init__()
        self.video_name = video_name
        self.seg_frames = seg_frames
        self.masks = masks
        self.output_dir = output_dir
        self.throttle_ms = int(throttle_ms) if throttle_ms else 0

    def run(self):
        # Limit library-internal threading so 4 workers ≈ 4 cores total
        try:
            cv2.setNumThreads(1)
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
        except Exception:
            pass

        # Also constrain common math backends in this process
        for env_key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            os.environ[env_key] = os.environ.get(env_key, "1") or "1"

        # Import after env/thread limits so NumPy/OpenBLAS pick them up
        from .contour import ContouredVideoProduction

        try:
            ContouredVideoProduction(
                output_video_name=self.video_name,
                segmented_frames=self.seg_frames,
                masks=self.masks,
                output_dir=self.output_dir,
                progress_callback=self.report_progress
            )
        except Exception as e:
            try:
                print(f"[ContourWorker] {self.video_name} failed: {e}")
            except Exception:
                pass
        finally:
            self.finished.emit()

    def report_progress(self, value):
        self.progress.emit(value)
        