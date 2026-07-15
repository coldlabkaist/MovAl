from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, pyqtSignal


class DataSplitState(QObject):
    busy_changed = pyqtSignal(bool)
    progress_changed = pyqtSignal(int, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._running = False
        self._done = 0
        self._total = 0
        self._message = ""

    def set_running(self, running: bool) -> None:
        running = bool(running)
        with self._lock:
            changed = self._running != running
            self._running = running
            if not running:
                self._done = 0
                self._total = 0
                self._message = ""
        if changed:
            self.busy_changed.emit(running)

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def update_progress(self, done: int, total: int, message: str) -> None:
        with self._lock:
            if not self._running:
                return
            self._done = max(0, int(done))
            self._total = max(0, int(total))
            self._message = str(message or "")
            snapshot = (self._done, self._total, self._message)
        self.progress_changed.emit(*snapshot)

    def snapshot(self) -> tuple[int, int, str]:
        with self._lock:
            return self._done, self._total, self._message


data_split_state = DataSplitState()


def set_data_split_running(running: bool) -> None:
    data_split_state.set_running(running)


def is_data_split_running() -> bool:
    return data_split_state.is_running()


def update_data_split_progress(done: int, total: int, message: str) -> None:
    data_split_state.update_progress(done, total, message)
