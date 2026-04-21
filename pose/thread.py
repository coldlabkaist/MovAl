from PyQt6.QtCore import QThread, pyqtSignal
import subprocess
import sys
import os
import shlex

def _to_cmd_list(command):
    if isinstance(command, (list, tuple)):
        return list(command)
    return shlex.split(str(command), posix=False)

def _make_env():
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env

class TrainThread(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, command):
        super().__init__()
        self.command = command
        self._process = None
        self._stop_requested = False
        self.exit_code = None

    @property
    def was_stopped(self) -> bool:
        return self._stop_requested

    def stop(self):
        self._stop_requested = True
        process = self._process
        if process is None:
            return

        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        except Exception:
            pass

    def run(self):
        cmd_list = _to_cmd_list(self.command)
        env = _make_env()

        process = None
        rc = None
        try:
            process = subprocess.Popen(
                cmd_list,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace", 
                env=env,
            )
            self._process = process

            # stdout
            for line in iter(process.stdout.readline, ''):
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()

            # stderr
            for line in iter(process.stderr.readline, ''):
                if line:
                    sys.stderr.write(line)
                    sys.stderr.flush()

            process.stdout.close()
            process.stderr.close()
            rc = process.wait()
            self.exit_code = rc

            if rc != 0 and not self._stop_requested:
                sys.stderr.write(f"\n[TrainThread] YOLO exited with code {rc}\n")
                sys.stderr.flush()
        finally:
            self._process = None
            self.finished_signal.emit()

class InferenceThread(QThread):
    def __init__(self, command):
        super().__init__()
        self.command = command

    def run(self):
        cmd_list = _to_cmd_list(self.command)
        env = _make_env()

        process = subprocess.Popen(
            cmd_list,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
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
        rc = process.wait()

        if rc != 0:
            sys.stderr.write(f"\n[InferenceThread] YOLO exited with code {rc}\n")
            sys.stderr.flush()
