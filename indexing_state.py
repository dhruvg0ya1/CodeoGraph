"""
indexing_state.py

Thread-safe in-memory store for live indexing job progress.
FastAPI's BackgroundTasks run in the same process but different threads,
so we use a simple dict + lock rather than anything async.
"""

import threading
from typing import Optional

_lock = threading.Lock()

# One state dict per namespace; only one job per namespace can run at a time.
_states: dict[str, dict] = {}


def init_state(namespace: str, total_files: int = 0) -> None:
    """Create or reset state for a namespace before starting a job."""
    with _lock:
        _states[namespace] = {
            "namespace": namespace,
            "status": "running",       # idle | running | done | error
            "total_files": total_files,
            "processed_files": 0,
            "total_chunks": 0,
            "indexed_chunks": 0,
            "current_file": "",
            "error": None,
        }


def update_state(namespace: str, **kwargs) -> None:
    """Partial update — only provided keys are changed."""
    with _lock:
        if namespace in _states:
            _states[namespace].update(kwargs)


def get_state(namespace: str) -> Optional[dict]:
    with _lock:
        return dict(_states.get(namespace, {}))


def set_error(namespace: str, error: str) -> None:
    with _lock:
        if namespace in _states:
            _states[namespace]["status"] = "error"
            _states[namespace]["error"] = error


def set_done(namespace: str) -> None:
    with _lock:
        if namespace in _states:
            _states[namespace]["status"] = "done"
