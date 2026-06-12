from PyQt6.QtCore import QThread, pyqtSignal
import subprocess
import sys
import os
import shlex
import locale
import re

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


_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")


def _decode_console_bytes(data: bytes) -> str:
    encodings = []
    preferred = locale.getpreferredencoding(False)
    if preferred:
        encodings.append(preferred)
    encodings.extend(["utf-8", "cp949", "utf-8-sig"])

    for encoding in dict.fromkeys(encodings):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _sanitize_console_text(text: str) -> str:
    clean = _ANSI_OSC_RE.sub("", text)
    clean = _ANSI_CSI_RE.sub("", clean)
    clean = clean.replace("\r", "")
    return clean


def _iter_clean_lines(stream):
    for raw_line in iter(stream.readline, b""):
        if not raw_line:
            continue
        yield _sanitize_console_text(_decode_console_bytes(raw_line)).rstrip("\n")

class TrainThread(QThread):
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str)

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
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
                env=env,
            )
            self._process = process

            for line in _iter_clean_lines(process.stdout):
                if line:
                    sys.stdout.write(f"{line}\n")
                    sys.stdout.flush()
                    self.log_signal.emit(line)

            process.stdout.close()
            rc = process.wait()
            self.exit_code = rc

            if rc != 0 and not self._stop_requested:
                message = f"[TrainThread] YOLO exited with code {rc}"
                sys.stderr.write(f"\n{message}\n")
                sys.stderr.flush()
                self.log_signal.emit(message)
        finally:
            self._process = None
            self.finished_signal.emit()

class InferenceThread(QThread):
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
                text=False,
                bufsize=0,
                env=env,
            )
            self._process = process

            for line in _iter_clean_lines(process.stdout):
                if line:
                    sys.stdout.write(f"{line}\n")
                    sys.stdout.flush()

            for line in _iter_clean_lines(process.stderr):
                if line:
                    sys.stderr.write(f"{line}\n")
                    sys.stderr.flush()

            process.stdout.close()
            process.stderr.close()
            rc = process.wait()
            self.exit_code = rc

            if rc != 0 and not self._stop_requested:
                sys.stderr.write(f"\n[InferenceThread] YOLO exited with code {rc}\n")
                sys.stderr.flush()
        finally:
            self._process = None
            self.finished_signal.emit()
