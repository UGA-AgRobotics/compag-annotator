"""Cancellable subprocesses; provider processes never inherit core imports."""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time
import uuid

from .protocol import PROTOCOL_VERSION, ProviderError, check_cancel, emit


def terminate(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=3)


def run_command(args, *, env=None, cwd=None, progress=None, cancel=None, timeout=3600):
    """No shell; log progress from pip/bootstrap while allowing cancellation."""
    check_cancel(cancel)
    env = dict(os.environ if env is None else env)
    # Source-mode callers often set PYTHONPATH=src. Letting it leak into pip
    # makes provider checks see the core's distribution and incompatible deps.
    # Provider bootstrap must inspect only its own isolated environment.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    process = subprocess.Popen(list(map(str, args)), cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    messages = queue.Queue()

    def read():
        for line in process.stdout:
            messages.put(line.rstrip())
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    lines = collections.deque(maxlen=100)
    start = time.monotonic()
    download_label = "Dependency download"
    try:
        while process.poll() is None or reader.is_alive() or not messages.empty():
            check_cancel(cancel)
            if time.monotonic() - start > timeout:
                raise ProviderError("Runtime operation timed out")
            try:
                line = messages.get(timeout=.1)
                lines.append(line)
                if line.lstrip().startswith("Downloading "):
                    download_label = line.strip()[:250]
                raw = re.fullmatch(r"Progress (\d+) of (\d+)", line.strip())
                event = {"stage": "runtime_install", "message": line}
                if raw:
                    event.update(download_bytes=int(raw[1]), download_total_bytes=int(raw[2]), download_label=download_label)
                emit(progress, **event)
            except queue.Empty:
                pass
        if process.returncode:
            raise ProviderError("Runtime command failed: " + "\n".join(lines)[-6000:])
        return "\n".join(lines)
    finally:
        terminate(process)
        process.stdout.close()


def worker_environment(data_dir):
    env = os.environ.copy()
    # Authentication is never needed by inference or training.
    for key in list(env):
        if key.startswith(("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_", "COMET_", "CLEARML_", "NEPTUNE_")):
            env.pop(key, None)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "WANDB_MODE": "disabled", "COMET_MODE": "DISABLED", "YOLO_AUTOINSTALL": "false",
                "YOLO_OFFLINE": "true", "YOLO_CONFIG_DIR": str(Path(data_dir) / "ultralytics-settings"),
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg",
                "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


class WorkerClient:
    def __init__(self, python, data_dir):
        self.events = queue.Queue()
        self.logs = collections.deque(maxlen=80)
        self.process = subprocess.Popen([str(python), "-I", "-B", "-u", str(Path(__file__).with_name("worker.py"))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, cwd=data_dir, env=worker_environment(data_dir),
                                        start_new_session=True)
        self.mutex = threading.Lock()
        for stream, target in [(self.process.stdout, self.events), (self.process.stderr, None)]:
            threading.Thread(target=self._read, args=(stream, target), daemon=True).start()

    def _read(self, stream, target):
        for line in stream:
            if target is None:
                self.logs.append(line.rstrip())
            else:
                try:
                    target.put(json.loads(line))
                except ValueError:
                    self.logs.append(line.rstrip())

    def request(self, operation, payload, *, progress=None, cancel=None, timeout=900):
        with self.mutex:
            request_id = uuid.uuid4().hex
            request = {"protocol": PROTOCOL_VERSION, "request_id": request_id, "operation": operation, **payload}
            try:
                check_cancel(cancel)
                self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
                self.process.stdin.flush()
                deadline = time.monotonic() + timeout
                while True:
                    check_cancel(cancel)
                    if time.monotonic() > deadline:
                        raise ProviderError("Provider timed out; reduce workload or increase worker_timeout explicitly")
                    try:
                        event = self.events.get(timeout=.1)
                    except queue.Empty:
                        if self.process.poll() is not None:
                            raise ProviderError("Provider process exited: " + "\n".join(self.logs)[-3000:])
                        continue
                    if event.get("protocol") != PROTOCOL_VERSION or event.get("request_id") != request_id:
                        raise ProviderError("Invalid or stale worker response")
                    if event["type"] == "progress":
                        emit(progress, **event["data"])
                    elif event["type"] == "result":
                        return event["data"]
                    elif event["type"] == "error":
                        raise ProviderError(event["message"])
                    else:
                        raise ProviderError("Unknown worker event")
            except BaseException:
                self.close()
                raise

    def close(self):
        terminate(self.process)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
