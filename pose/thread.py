from PyQt6.QtCore import QThread, pyqtSignal
import subprocess
import sys

class TrainThread(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            subprocess.run(self.command, shell=True, check=True)
        except Exception:
            cmd_list = self.command if isinstance(self.command, (list, tuple)) else str(self.command).split()
            subprocess.run(cmd_list, shell=False, check=True)
        self.finished_signal.emit()
        
class InferenceThread(QThread):
    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        try:
            process = subprocess.Popen(
                self.command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, encoding='utf-8'
            )
        except Exception:
            cmd_list = self.command if isinstance(self.command, (list, tuple)) else str(self.command).split()
            process = subprocess.Popen(
                cmd_list, shell=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, encoding='utf-8'
            )

        for line in iter(process.stdout.readline, ''):
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()

        for line in iter(process.stderr.readline, ''):
            if line:
                sys.stderr.write(line)
                sys.stderr.flush()

        process.stdout.close()
        process.stderr.close()
        process.wait()
