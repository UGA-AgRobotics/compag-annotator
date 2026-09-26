"""Versioned, JSON-only worker protocol and original-image prompt validation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Protocol

PROTOCOL_VERSION = 1
Progress = Callable[[dict], None]
Cancel = Callable[[], bool]


class ProviderError(RuntimeError):
    pass


class CancelledError(ProviderError):
    pass


class Provider(Protocol):
    def infer(self, request: dict, progress: Progress) -> dict: ...
    def unload(self) -> None: ...


class AutomaticProposalProvider(Provider, Protocol):
    """Optional capability; SAM3 is deliberately not advertised as implementing it."""
    def generate_proposals(self, request: dict, progress: Progress) -> dict: ...


def check_cancel(cancel=None):
    if cancel is not None and cancel():
        raise CancelledError("Operation cancelled")


def emit(progress=None, **event):
    if progress is not None:
        progress(event)


def sha256_file(path, cancel=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check_cancel(cancel)
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_prompts(width, height, points=None, labels=None, box=None):
    """Refactored from A's finite/bounded validation, without its crop limit/classes."""
    points = [] if points is None else points
    labels = [] if labels is None else labels
    if not isinstance(points, (list, tuple)) or not isinstance(labels, (list, tuple)):
        raise ValueError("Points and labels must be arrays")
    if len(points) != len(labels):
        raise ValueError("Supply one 0/1 label per point")
    if any(type(v) is not int or v not in (0, 1) for v in labels):
        raise ValueError("Prompt labels mean foreground (1) or background (0), not classes")
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in point):
            raise ValueError("Each point needs two finite coordinates")
        x, y = point
        if not 0 <= x < width or not 0 <= y < height:
            raise ValueError("Point lies outside the original image")
        normalized.append([float(x), float(y)])
    if len({tuple(p) for p in normalized}) != len(normalized):
        raise ValueError("Remove duplicate or contradictory points")
    if box is not None:
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError("Box needs four finite coordinates")
        x0, y0, x1, y1 = box
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError("Box lies outside the original image or has zero area")
        box = list(map(float, box))
    if not points and box is None or points and 1 not in labels and box is None:
        raise ValueError("Supply a positive point or a box")
    return normalized, list(labels), box


def normalize_device(device):
    import re
    if device == "cpu":
        return device
    if isinstance(device, int) or isinstance(device, str) and device.isdecimal():
        device = f"cuda:{device}"
    if device == "cuda":
        device = "cuda:0"
    if not isinstance(device, str) or re.fullmatch(r"cuda:\d+", device) is None:
        raise ValueError("Choose cpu or one CUDA device (for example cuda:0)")
    return device
