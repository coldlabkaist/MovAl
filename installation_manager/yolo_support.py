import os
import shutil
import stat
import subprocess
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QMessageBox, QProgressBar, QTextEdit, QVBoxLayout


ULTRALYTICS_PACKAGE_SPEC = "ultralytics"
YOLO_MODEL_DIR = "models"
YOLO_POSE_MODELS = [
    "yolov8n-pose.pt",
    "yolov8s-pose.pt",
    "yolov8m-pose.pt",
    "yolov8l-pose.pt",
    "yolov8x-pose.pt",
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo11l-pose.pt",
    "yolo11x-pose.pt",
]


def _force_remove(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _emit_log(log_fn, message: str) -> None:
    if callable(log_fn):
        log_fn(message)


def install_ultralytics_package(python_executable: str, *, upgrade: bool, log_fn=None) -> None:
    command = [python_executable, "-m", "pip", "install"]
    if upgrade:
        command.append("-U")
    command.append(ULTRALYTICS_PACKAGE_SPEC)
    _emit_log(log_fn, f"Running: {' '.join(command)}")
    subprocess.check_call(command)


def download_yolo_pose_models(
    *,
    force_reinstall: bool = False,
    yolo_model_dir: str = YOLO_MODEL_DIR,
    log_fn=None,
) -> None:
    if os.path.isdir(yolo_model_dir):
        if force_reinstall:
            _emit_log(log_fn, "Deleting existing YOLO models directory...")
            shutil.rmtree(yolo_model_dir, onerror=_force_remove)
        else:
            _emit_log(log_fn, "YOLO models directory already exists. Skipping download.")
            return

    _emit_log(log_fn, "Downloading YOLO pose models...")

    cwd = os.getcwd()
    weights_dir = os.path.join(cwd, yolo_model_dir)
    os.makedirs(weights_dir, exist_ok=True)

    from ultralytics import YOLO

    for model in YOLO_POSE_MODELS:
        target_path = os.path.join(weights_dir, model)
        downloaded_path = os.path.join(cwd, model)

        _emit_log(log_fn, f"Downloading {model}...")
        YOLO(model)

        if os.path.exists(downloaded_path):
            os.replace(downloaded_path, target_path)
            continue

        if os.path.exists(target_path):
            continue

        raise FileNotFoundError(
            f"Downloaded model file could not be found for '{model}'. "
            "The Ultralytics download location may have changed."
        )

    _emit_log(log_fn, "All YOLO pose models downloaded.")


class YoloUpdateWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(bool)

    def __init__(self, refresh_models: bool = False, parent=None):
        super().__init__(parent)
        self.python = sys.executable
        self.refresh_models = refresh_models

    def run(self):
        steps = [("Updating ultralytics package...", self.update_ultralytics)]
        if self.refresh_models:
            steps.append(("Refreshing YOLO pose models...", self.refresh_yolo_models))

        try:
            total = max(len(steps), 1)
            for index, (message, action) in enumerate(steps, 1):
                self.log.emit(message)
                action()
                self.progress.emit(int((index / total) * 100))
            self.done.emit(True)
        except Exception as err:
            self.log.emit(f"error : {err}")
            self.done.emit(False)

    def update_ultralytics(self):
        install_ultralytics_package(self.python, upgrade=True, log_fn=self.log.emit)

    def refresh_yolo_models(self):
        download_yolo_pose_models(force_reinstall=True, log_fn=self.log.emit)


class YoloUpdateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update YOLO")
        self.setFixedSize(320, 220)

        refresh_models = False
        if os.path.isdir(YOLO_MODEL_DIR):
            answer = QMessageBox.question(
                self,
                "Refresh local models?",
                "Do you also want to redownload the local YOLO pose models?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            refresh_models = answer == QMessageBox.StandardButton.Yes

        layout = QVBoxLayout(self)
        self.log_view = QTextEdit(readOnly=True)
        self.bar = QProgressBar()
        self.bar.setValue(0)
        layout.addWidget(self.log_view)
        layout.addWidget(self.bar)

        self.worker = YoloUpdateWorker(refresh_models=refresh_models, parent=self)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.bar.setValue)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def append_log(self, text: str):
        self.log_view.append(text)

    def on_done(self, ok: bool):
        self.bar.setValue(100)
        if ok:
            QMessageBox.information(
                self,
                "YOLO Update Complete",
                "Ultralytics has been successfully updated.",
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Error",
                "An error occurred during the YOLO update.\nPlease check the log.",
            )
