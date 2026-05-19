from __future__ import annotations

import threading

_split_state_lock = threading.Lock()
_split_running = False


def set_data_split_running(running: bool) -> None:
    global _split_running
    with _split_state_lock:
        _split_running = bool(running)


def is_data_split_running() -> bool:
    with _split_state_lock:
        return _split_running
