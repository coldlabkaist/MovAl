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

        force_reinstall = False
        repo_dir = "Cutie"
        if os.path.isdir(repo_dir):
            ans = QMessageBox.question(
                self,
                "Existing Directory Found",
                "A Cutie directory already exists.\n"
                "Do you want to delete it and perform a reinstallation?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                self.log_view.append("Reinstallation skipped by user.")
                self.bar.setValue(100)
                return 
            force_reinstall = True

        self.worker = OneClickWorker(force_reinstall=force_reinstall, parent=self)
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

    def __init__(self, force_reinstall=False, parent=None):
        super().__init__(parent)
        self.repo_url = "https://github.com/hkchengrex/Cutie.git"
        self.repo_dir = "Cutie"
        self.python   = sys.executable
        self.force_reinstall = force_reinstall

    def run(self):
        try:
            steps = [
                ("Cloning repository …",        self.clone_repo),
                ("Installing package …",        self.pip_install),
                ("Downloading models …",        self.download_models),
            ]
            n = len(steps)
            for i, (msg, fn) in enumerate(steps, 1):
                self.log.emit(msg)
                fn()
                self.progress.emit(int(i/n*100))
            self.done.emit(True)
        except Exception as e:
            self.log.emit(f"error : {e}")
            self.done.emit(False)

    def clone_repo(self):
        if os.path.isdir(self.repo_dir):
            if self.force_reinstall:
                self.log.emit("· Deleting existing Cutie directory …")
                shutil.rmtree(self.repo_dir, onerror=_force_remove)
            else:
                self.log.emit("· Repository already exists. Skipping installation.")
                raise RuntimeError("Installation skipped by user")

        self.log.emit("· Cloning Cutie repository …")
        subprocess.check_call(["git", "clone", "--depth", "1",
                               self.repo_url, self.repo_dir])

    def pip_install(self):
        subprocess.check_call([self.python, "-m", "pip", "install", "-e", self.repo_dir])

    def download_models(self):
        script = os.path.join(self.repo_dir, "cutie", "utils", "download_models.py")
        subprocess.check_call([self.python, script])

def _force_remove(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)
