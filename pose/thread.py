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
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)")


def _decode_console_bytes(data: bytes) -> str:
    encodings = ["utf-8", "utf-8-sig"]
    preferred = locale.getpreferredencoding(False)
    if preferred:
        encodings.append(preferred)
    encodings.append("cp949")

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
    """Yield records separated by either newline or carriage return."""
    pending = b""
    while True:
        chunk = os.read(stream.fileno(), 4096)
        if not chunk:
            break
        pending += chunk
        records = re.split(br"[\r\n]+", pending)
        pending = records.pop()
        for raw_record in records:
            if raw_record:
                yield _sanitize_console_text(_decode_console_bytes(raw_record))
    if pending:
        yield _sanitize_console_text(_decode_console_bytes(pending))


_OUTPUT_TAIL_LIMIT = 200


def _append_output_tail(lines, line) -> None:
    lines.append(line)
    if len(lines) > _OUTPUT_TAIL_LIMIT:
        del lines[: len(lines) - _OUTPUT_TAIL_LIMIT]


class TrainThread(QThread):
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self._process = None
        self._stop_requested = False
        self.exit_code = None
        self.output_lines = []

    @property
    def output_text(self) -> str:
        return "\n".join(self.output_lines)

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
        self.output_lines.clear()

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
                    _append_output_tail(self.output_lines, line)
                    sys.stdout.write(f"{line}\n")
                    sys.stdout.flush()
                    self.log_signal.emit(line)

            process.stdout.close()
            rc = process.wait()
            self.exit_code = rc

            if rc != 0 and not self._stop_requested:
                message = f"[TrainThread] YOLO exited with code {rc}"
                _append_output_tail(self.output_lines, message)
                sys.stderr.write(f"\n{message}\n")
                sys.stderr.flush()
                self.log_signal.emit(message)
        finally:
            self._process = None
            self.finished_signal.emit()

class InferenceThread(QThread):
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self._process = None
        self._stop_requested = False
        self.exit_code = None
        self.output_lines = []

    @property
    def output_text(self) -> str:
        return "\n".join(self.output_lines)

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
        self.output_lines.clear()

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
                    _append_output_tail(self.output_lines, line)
                    sys.stdout.write(f"{line}\n")
                    sys.stdout.flush()
                    self.log_signal.emit(line)

            process.stdout.close()
            rc = process.wait()
            self.exit_code = rc

            if rc != 0 and not self._stop_requested:
                message = f"[InferenceThread] YOLO exited with code {rc}"
                _append_output_tail(self.output_lines, message)
                sys.stderr.write(f"\n{message}\n")
                sys.stderr.flush()
        finally:
            self._process = None
            self.finished_signal.emit()


class FunctionProgressThread(QThread):
    progress = pyqtSignal(int, int, str)
    success = pyqtSignal()
    failure = pyqtSignal(str)

    def __init__(self, function):
        super().__init__()
        self._function = function
        self.error_text = None

    def run(self):
        self.error_text = None
        try:
            self._function(self.progress.emit)
            self.success.emit()
        except Exception as err:
            self.error_text = str(err)
            self.failure.emit(str(err))
