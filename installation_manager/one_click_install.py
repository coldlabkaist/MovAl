import os
import shutil
import stat
import subprocess
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QDialog, QMessageBox, QProgressBar, QTextEdit, QVBoxLayout

from .yolo_support import download_yolo_pose_models, install_ultralytics_package


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
            answer = QMessageBox.question(
                self,
                "Existing Directory Found",
                "A Cutie directory already exists.\n"
                "Do you want to delete it and perform a reinstallation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                force_reinstall_cutie = True
            else:
                self.log_view.append("Cutie directory reinstallation skipped by user.")

        force_reinstall_yolo = False
        yolo_model_dir = "models"
        if os.path.isdir(yolo_model_dir):
            answer = QMessageBox.question(
                self,
                "Existing Directory Found",
                "YOLO models already exist.\n"
                "Do you want to delete them and perform a reinstallation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                force_reinstall_yolo = True
            else:
                self.log_view.append("YOLO model reinstallation skipped by user.")

        self.worker = OneClickWorker(
            force_reinstall_cutie=force_reinstall_cutie,
            force_reinstall_yolo=force_reinstall_yolo,
            parent=self,
        )
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
                "Installation Complete",
                "Cutie and YOLO dependencies have been successfully installed.",
            )
            self.accept()
        else:
            QMessageBox.critical(
                self,
                "Error",
                "An error occurred during installation.\nPlease check the log.",
            )


class OneClickWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int)
    done = pyqtSignal(bool)

    def __init__(self, force_reinstall_cutie=False, force_reinstall_yolo=False, parent=None):
        super().__init__(parent)
        self.cutie_url = "https://github.com/hkchengrex/Cutie.git"
        self.cutie_dir = "Cutie"
        self.yolo_model_dir = "models"
        self.python = sys.executable
        self.force_reinstall_cutie = force_reinstall_cutie
        self.force_reinstall_yolo = force_reinstall_yolo

    def run(self):
        try:
            steps_cutie = [
                ("Cloning repository...", self.clone_repo_cutie),
                ("Installing package...", self.pip_install_cutie),
                ("Downloading models...", self.download_models_cutie),
            ]
            steps_yolo = [
                ("Installing package...", self.pip_install_ultralytics),
                ("Downloading models...", self.download_models_yolo),
            ]
            total = len(steps_cutie) + len(steps_yolo)

            for index, (message, action) in enumerate(steps_cutie, 1):
                self.log.emit(message)
                action()
                self.progress.emit(int((index / total) * 100))

            for index, (message, action) in enumerate(steps_yolo, 1):
                self.log.emit(message)
                action()
                self.progress.emit(int(((len(steps_cutie) + index) / total) * 100))

            self.done.emit(True)
        except Exception as err:
            self.log.emit(f"error : {err}")
            self.done.emit(False)

    def clone_repo_cutie(self):
        if os.path.isdir(self.cutie_dir):
            if self.force_reinstall_cutie:
                self.log.emit("Deleting existing Cutie directory...")
                shutil.rmtree(self.cutie_dir, onerror=_force_remove)
            else:
                self.log.emit("Cutie repository already exists. Skipping clone.")
                return

        self.log.emit("Cloning Cutie repository...")
        subprocess.check_call(["git", "clone", "--depth", "1", self.cutie_url, self.cutie_dir])

    def pip_install_cutie(self):
        subprocess.check_call([self.python, "-m", "pip", "install", "-e", self.cutie_dir])

    def download_models_cutie(self):
        if self.force_reinstall_cutie:
            script = os.path.join(self.cutie_dir, "cutie", "utils", "download_models.py")
            subprocess.check_call([self.python, script])

    def pip_install_ultralytics(self):
        install_ultralytics_package(self.python, upgrade=False, log_fn=self.log.emit)

    def download_models_yolo(self):
        download_yolo_pose_models(
            force_reinstall=self.force_reinstall_yolo,
            yolo_model_dir=self.yolo_model_dir,
            log_fn=self.log.emit,
        )


def _force_remove(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)
