from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QObject, pyqtSignal


class PoseExecutionState(QObject):
    busy_changed = pyqtSignal(bool, str)
    progress_changed = pyqtSignal(str, int, int, str)

    def __init__(self) -> None:
        super().__init__()
        self._busy = False
        self._active_task: Optional[str] = None
        self._owner: Any = None
        self._done = 0
        self._total = 0
        self._message = ""

    def is_busy(self) -> bool:
        return self._busy

    def active_task(self) -> Optional[str]:
        return self._active_task

    def acquire(self, task_name: str, owner: Any = None) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._active_task = task_name
        self._owner = owner
        self._done = 0
        self._total = 0
        self._message = ""
        self.busy_changed.emit(True, task_name)
        self.progress_changed.emit(task_name, self._done, self._total, self._message)
        return True

    def release(self, owner: Any = None) -> None:
        if not self._busy:
            return
        if owner is not None and self._owner is not owner:
            return

        prev_task = self._active_task or ""
        self._busy = False
        self._active_task = None
        self._owner = None
        self._done = 0
        self._total = 0
        self._message = ""
        self.busy_changed.emit(False, prev_task)
        self.progress_changed.emit("", 0, 0, "")

    def owned_by(self, owner: Any) -> bool:
        return self._busy and self._owner is owner

    def update_progress(self, done: int, total: int, message: str = "") -> None:
        if not self._busy or self._active_task is None:
            return
        self._done = max(0, int(done))
        self._total = max(0, int(total))
        self._message = str(message or "")
        self.progress_changed.emit(self._active_task, self._done, self._total, self._message)


pose_execution_state = PoseExecutionState()
