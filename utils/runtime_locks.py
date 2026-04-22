from __future__ import annotations

import threading

_compression_state_lock = threading.Lock()
_compression_running = False


def try_acquire_project_compression_lock() -> bool:
    global _compression_running
    with _compression_state_lock:
        if _compression_running:
            return False
        _compression_running = True
        return True


def release_project_compression_lock() -> None:
    global _compression_running
    with _compression_state_lock:
        _compression_running = False


def is_project_compression_running() -> bool:
    with _compression_state_lock:
        return _compression_running
