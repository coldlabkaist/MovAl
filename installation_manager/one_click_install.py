from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QTextEdit, QProgressBar, QDialog, QVBoxLayout, 
    QMessageBox, QTextEdit, QProgressBar
)
import subprocess
import os
import sys
import shutil
import stat

class OneClickInstallDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("One-Click Install")
        self.setFixedSize(300, 200)

        vbox = QVBoxLayout(self)
        self.log_view = QTextEdit(readOnly=True)
        self.bar = QProgressBar()
        self.bar.setValue(0)
        vbox.addWidget(self.log_view)
        vbox.addWidget(self.bar)

        force_reinstall_cutie = False
        cutie_dir = "Cutie"
        if os.path.isdir(cutie_dir):
            ans = QMessageBox.question(
                self,
                "Existing Directory Found",
                "A Cutie directory already exists.\n"
                "Do you want to delete it and perform a reinstallation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                self.log_view.append("Cutie directory reinstallation skipped by user.")
            else:
                force_reinstall_cutie = True

        force_reinstall_yolo = False
        yolo_model_dir = "models"
        if os.path.isdir(yolo_model_dir):
            ans = QMessageBox.question(
                self,
                "Existing Directory Found",
                "A YOLO models already exists.\n"
                "Do you want to delete it and perform a reinstallation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                self.log_view.append("YOLO model reinstallation skipped by user.")
            else:
                force_reinstall_yolo = True

        self.worker = OneClickWorker(
            force_reinstall_cutie = force_reinstall_cutie, 
            force_reinstall_yolo = force_reinstall_yolo, 
            parent=self
        )
        self.worker.log.connect(self.append_log) 
        self.worker.progress.connect(self.bar.setValue)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def append_log(self, txt: str):
        self.log_view.append(txt)

    def on_done(self, ok: bool):
        self.bar.setValue(100)
        if ok:
            QMessageBox.information(
                self,
                "Installation Complete",
                "Cutie has been successfully installed.",
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Error",
                "An error occurred during installation.\nPlease check the log.",
            )

class OneClickWorker(QThread):
    log      = pyqtSignal(str)
    progress = pyqtSignal(int)
    done     = pyqtSignal(bool)

    def __init__(self, force_reinstall_cutie=False, force_reinstall_yolo=False, parent=None):
        super().__init__(parent)
        self.cutie_url = "https://github.com/hkchengrex/Cutie.git"
        self.cutie_dir = "Cutie"
        self.yolo_model_dir = "models"
        self.python   = sys.executable
        self.force_reinstall_cutie = force_reinstall_cutie
        self.force_reinstall_yolo = force_reinstall_yolo

    def run(self):
        try:
            steps_cutie = [
                ("Cloning repository …",        self.clone_repo_cutie),
                ("Installing package …",        self.pip_install_cutie),
                ("Downloading models …",        self.download_models_cutie),
            ]
            steps_yolo = [
                ("Installing package …",        self.pip_install_ultralytics),
                ("Downloading models …",        self.download_models_yolo),
            ]
            n = len(steps_cutie) + len(steps_yolo)

            for i, (msg, fn) in enumerate(steps_cutie, 1):
                self.log.emit(msg)
                fn()
                self.progress.emit(int(i/n*100))
            for i, (msg, fn) in enumerate(steps_yolo, 1):
                self.log.emit(msg)
                fn()
                self.progress.emit(int(i+len(steps_cutie)/n*100))
            self.done.emit(True)

        except Exception as e:
            self.log.emit(f"error : {e}")
            self.done.emit(False)

    def clone_repo_cutie(self):
        if os.path.isdir(self.cutie_dir):
            if self.force_reinstall_cutie:
                self.log.emit("· Deleting existing Cutie directory …")
                shutil.rmtree(self.cutie_dir, onerror=_force_remove)
            else:
                self.log.emit("· Repository already exists. Skipping installation.")
                return

        self.log.emit("· Cloning Cutie repository …")
        subprocess.check_call(["git", "clone", "--depth", "1",
                               self.cutie_url, self.cutie_dir])

    def pip_install_cutie(self):
        subprocess.check_call([self.python, "-m", "pip", "install", "-e", self.cutie_dir])

    def download_models_cutie(self):
        if self.force_reinstall_cutie:
            script = os.path.join(self.cutie_dir, "cutie", "utils", "download_models.py")
            subprocess.check_call([self.python, script])

    def pip_install_ultralytics(self):
        subprocess.check_call([self.python, "-m", "pip", "install", "ultralytics"])

    def download_models_yolo(self):
        if os.path.isdir(self.yolo_model_dir):
            if self.force_reinstall_yolo:
                self.log.emit("· Deleting existing YOLO models …")
                shutil.rmtree(self.yolo_model_dir, onerror=_force_remove)
            else:
                self.log.emit("· Repository already exists. Skipping installation.")
                return
        self.log.emit("Downloading Models...")

        models = [
            'yolov8n-pose.pt',
            'yolov8s-pose.pt',
            'yolov8m-pose.pt',
            'yolov8l-pose.pt',
            'yolov8x-pose.pt',
            'yolo11n-pose.pt',
            'yolo11s-pose.pt',
            'yolo11m-pose.pt',
            'yolo11l-pose.pt',
            'yolo11x-pose.pt'
        ]

        cwd = os.getcwd()
        weights_dir = os.path.join(cwd, self.yolo_model_dir)
        os.makedirs(weights_dir, exist_ok=True)

        from ultralytics import YOLO
        for model in models:
            model_path = os.path.join(weights_dir, model)
            self.log.emit(f". Downloading {model}...")
            YOLO(model)
            os.rename(model, model_path)
        self.log.emit(". All models downloaded")


def _force_remove(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)
