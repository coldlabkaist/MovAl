from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DEFAULT_EDIT_HISTORY_LIMIT = 10


@dataclass
class FrameEdit:
    frame_idx: int
    before: pd.DataFrame
    after: pd.DataFrame


class FrameEditHistory:
    def __init__(self, limit: int = DEFAULT_EDIT_HISTORY_LIMIT):
        self.limit = max(1, int(limit))
        self._undo_stack: list[FrameEdit] = []
        self._redo_stack: list[FrameEdit] = []
        self._frame_idx: int | None = None

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def clear_if_frame_changed(self, frame_idx: int | None) -> None:
        if frame_idx != self._frame_idx:
            self.clear()
            self._frame_idx = frame_idx

    def push(self, frame_idx: int, before: pd.DataFrame, after: pd.DataFrame) -> bool:
        before = before.reset_index(drop=True)
        after = after.reset_index(drop=True)
        if before.equals(after):
            return False

        self.clear_if_frame_changed(frame_idx)
        self._undo_stack.append(FrameEdit(frame_idx, before.copy(deep=True), after.copy(deep=True)))
        if len(self._undo_stack) > self.limit:
            self._undo_stack = self._undo_stack[-self.limit:]
        self._redo_stack.clear()
        return True

    def can_undo(self, frame_idx: int | None) -> bool:
        return bool(self._undo_stack) and frame_idx == self._frame_idx

    def can_redo(self, frame_idx: int | None) -> bool:
        return bool(self._redo_stack) and frame_idx == self._frame_idx

    def pop_undo(self, frame_idx: int | None) -> FrameEdit | None:
        if not self.can_undo(frame_idx):
            return None
        edit = self._undo_stack.pop()
        self._redo_stack.append(edit)
        return edit

    def pop_redo(self, frame_idx: int | None) -> FrameEdit | None:
        if not self.can_redo(frame_idx):
            return None
        edit = self._redo_stack.pop()
        self._undo_stack.append(edit)
        return edit
