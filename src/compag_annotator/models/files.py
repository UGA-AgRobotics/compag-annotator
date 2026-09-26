"""Atomic local metadata and process-safe locks; no checkpoint deserialization."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
import uuid

from ..providers.protocol import check_cancel

_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def file_lock(path, cancel=None):
    """Thread + advisory process lock, automatically released after process death."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(str(path.resolve()), threading.Lock())
    while not lock.acquire(timeout=.1):
        check_cancel(cancel)
    try:
        with path.open("a+") as stream:
            while True:
                check_cancel(cancel)
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.1)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
    finally:
        lock.release()
