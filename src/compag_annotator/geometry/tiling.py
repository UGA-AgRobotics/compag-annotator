"""Deterministic original-image tile plans and exact crop-to-image mask mapping.

Dimensions are width × height; RLE sizes remain [height,width]. The shared tiler
never pads or resizes pixels. Providers record their encoder transforms outside
this plan and must return crop-coordinate masks before mapping them here.
"""
from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from numbers import Real

from . import MAX_PIXELS, _dimensions, _integer
from .rle import counts_from_intervals, foreground_intervals, geometry_rle

# Public preflight bounds, deliberately checked before generating origins/tiles.
MAX_TILES = 10_000
DEFAULT_TILING = {"mode": "tiled", "preset": "512", "width": 512, "height": 512,
                  "overlap": {"mode": "percent", "x": 25, "y": 25}}


class TilingError(ValueError):
    """Invalid plan inputs with UI field names mapped to actionable messages."""
    def __init__(self, fields):
        self.fields = dict(fields)
        super().__init__("; ".join(f"{key}: {value}" for key, value in self.fields.items()))


def _positive(value, field, errors):
    try:
        value = _integer(value, field)
        if value <= 0:
            raise ValueError("must be a positive integer")
        return value
    except ValueError:
        errors[field] = "Must be a positive integer (not a boolean or fractional number)"
        return None


def _normalize(config):
    if config is None:
        config = DEFAULT_TILING
    if not isinstance(config, dict):
        raise TilingError({"config": "Tiling settings must be an object"})
    errors = {}
    allowed = {"mode", "preset", "width", "height", "overlap", "overlap_x", "overlap_y", "stride_x", "stride_y"}
    for key in config.keys() - allowed:
        errors[str(key)] = "Unknown tiling field; model imgsz is separate from source crops"
    mode = config.get("mode", "tiled")
    if mode not in ("whole", "tiled"):
        errors["mode"] = "Choose whole or tiled"
    preset = config.get("preset", "512")
    if preset not in ("256", "512", "1024", "custom"):
        errors["preset"] = "Choose 256, 512, 1024 or custom"
    default_size = int(preset) if preset in ("256", "512", "1024") else 512
    width = _positive(config.get("width", default_size), "width", errors)
    height = _positive(config.get("height", default_size), "height", errors)
    if preset in ("256", "512", "1024"):
        for field, value in (("width", width), ("height", height)):
            if value is not None and value != int(preset):
                errors[field] = "Dimensions must match the selected preset; choose custom for a different size"
    if width is not None and height is not None and width * height > MAX_PIXELS:
        errors["width"] = errors["height"] = f"Requested tile exceeds the {MAX_PIXELS}-pixel geometry limit"
    overlap = config.get("overlap", {"mode": "percent", "x": 25, "y": 25})
    if not isinstance(overlap, dict):
        errors["overlap"] = "Overlap must be an object with mode, x and y"
        overlap = {}
    for key in overlap.keys() - {"mode", "x", "y"}:
        errors[f"overlap.{key}"] = "Unknown overlap field"
    overlap_mode = overlap.get("mode", "percent")
    if overlap_mode not in ("percent", "pixels"):
        errors["overlap.mode"] = "Choose percent or pixels"
    values, resolved = {}, {}
    for axis, size in (("x", width), ("y", height)):
        value = overlap.get(axis, 25 if overlap_mode == "percent" else 0)
        if overlap_mode == "percent":
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < 100 or not math.isfinite(value):
                errors[f"overlap.{axis}"] = "Percentage must be finite and satisfy 0 <= value < 100"
                continue
            value = int(value) if float(value).is_integer() else float(value)
            if size is not None:
                resolved[axis] = int(Decimal(size) * Decimal(str(value)) // Decimal(100))
        elif overlap_mode == "pixels":
            try:
                value = _integer(value, f"overlap.{axis}")
            except ValueError:
                errors[f"overlap.{axis}"] = "Pixel overlap must be an integer"
                continue
            if value < 0 or size is not None and value >= size:
                errors[f"overlap.{axis}"] = "Require 0 <= overlap < the corresponding tile dimension"
                continue
            resolved[axis] = value
        values[axis] = value
    if errors:
        raise TilingError(errors)
    result = {"mode": mode, "preset": preset, "width": width, "height": height,
              "overlap": {"mode": overlap_mode, **values},
              "overlap_x": resolved["x"], "overlap_y": resolved["y"],
              "stride_x": width - resolved["x"], "stride_y": height - resolved["y"]}
    # Persisted effective settings can be replayed, but never silently reinterpreted.
    for field in ("overlap_x", "overlap_y", "stride_x", "stride_y"):
        if field in config:
            try:
                value = _integer(config[field], field)
            except ValueError:
                errors[field] = "Resolved values must be integers"
                continue
            if value != result[field]:
                errors[field] = "Recorded effective value disagrees with the requested overlap settings"
    if errors:
        raise TilingError(errors)
    return result


def _axis_count(length, size, stride):
    far = max(0, length - size)
    return far // stride + 1 + int(far % stride != 0)


def _origins(length, size, stride):
    far = max(0, length - size)
    values = list(range(0, far + 1, stride))
    if values[-1] != far:
        values.append(far)
    return values


def plan_tiles(image_width, image_height, config=None, *, image_sha256=None):
    """Create an immutable-by-value plan with edge-aligned row-major origins.

    config=None means tiled 512×512 with 25% per-axis overlap. Whole mode keeps
    the selected tile controls as settings but emits exactly one full image box.
    Use returned crops, not configured width/height, for a small image's shape.
    """
    errors = {}
    image_width = _positive(image_width, "image_width", errors)
    image_height = _positive(image_height, "image_height", errors)
    if image_width is not None and image_height is not None and image_width * image_height > MAX_PIXELS:
        errors["image_width"] = errors["image_height"] = f"Image exceeds the {MAX_PIXELS}-pixel geometry limit"
    if image_sha256 is not None:
        if not isinstance(image_sha256, str) or len(image_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in image_sha256):
            errors["image_sha256"] = "Image SHA-256 must be 64 hexadecimal characters or null"
        else:
            image_sha256 = image_sha256.lower()
    try:
        normalized = _normalize(config)
    except TilingError as exc:
        errors.update(exc.fields)
    if errors:
        raise TilingError(errors)
    width, height = normalized["width"], normalized["height"]
    sx, sy = normalized["stride_x"], normalized["stride_y"]
    count = 1 if normalized["mode"] == "whole" else _axis_count(image_width, width, sx) * _axis_count(image_height, height, sy)
    if count > MAX_TILES:
        raise TilingError({"tile_count": f"Plan requires {count} tiles; limit is {MAX_TILES}. Reduce overlap or increase crop dimensions."})
    image = {"width": image_width, "height": image_height, "sha256": image_sha256}
    identity = {"schema_version": 1, "policy": "edge_aligned_no_padding/v1", "image": image, "config": normalized}
    plan_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if normalized["mode"] == "whole":
        boxes = [[0, 0, image_width, image_height]]
    else:
        boxes = [[x, y, min(image_width, x + width), min(image_height, y + height)]
                 for y in _origins(image_height, height, sy) for x in _origins(image_width, width, sx)]
    tiles = [{"id": f"tile-{plan_hash}-{index:05d}", "box": b,
              "valid_box": [0, 0, b[2] - b[0], b[3] - b[1]], "padding": [0, 0, 0, 0]}
             for index, b in enumerate(boxes)]
    return {"schema_version": 1, "image": image, "config": normalized,
            "tiles": tiles, "tile_count": len(tiles), "plan_hash": plan_hash}


def _four_integers(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"Tile {name} requires four integers")
    return [_integer(value, f"tile {name}") for value in values]


def map_tile_geometry(geometry, tile, image_width, image_height):
    """Embed a crop-local mask/polygon as full-image RLE without a full array.

    Padding order is [left,top,right,bottom]; shared plans use zeros. Explicitly
    described padded inputs are cropped to valid_box before translation. No
    resize/encoder-coordinate inversion is guessed. Padding-only masks fail.
    """
    image_width, image_height = _dimensions(image_width, image_height)
    if not isinstance(tile, dict):
        raise ValueError("Tile descriptor must be an object")
    x0, y0, x1, y1 = _four_integers(tile.get("box"), "box")
    if not 0 <= x0 < x1 <= image_width or not 0 <= y0 < y1 <= image_height:
        raise ValueError("Tile crop must lie inside the original image")
    crop_width, crop_height = x1 - x0, y1 - y0
    left, top, right, bottom = _four_integers(tile.get("padding", [0, 0, 0, 0]), "padding")
    if min(left, top, right, bottom) < 0:
        raise ValueError("Tile padding cannot be negative")
    source_width, source_height = _dimensions(crop_width + left + right, crop_height + top + bottom)
    valid = _four_integers(tile.get("valid_box", [left, top, left + crop_width, top + crop_height]), "valid_box")
    if valid != [left, top, left + crop_width, top + crop_height]:
        raise ValueError("Tile valid_box must agree with crop dimensions and padding")
    rle = geometry_rle(geometry, source_width, source_height)
    def translated():
        for start, end in foreground_intervals(rle["counts"]):
            first = max(start // source_height, left)
            last = min((end - 1) // source_height, left + crop_width - 1)
            for column in range(first, last + 1):
                low = max(start - column * source_height, top)
                high = min(end - column * source_height, top + crop_height)
                if low < high:
                    offset = (x0 + column - left) * image_height + y0 - top
                    yield offset + low, offset + high
    counts = counts_from_intervals(translated(), image_width * image_height)
    if len(counts) == 1:
        raise ValueError("Tile mask has no foreground inside the valid crop; padding is excluded")
    return {"type": "mask", "rle": {"size": [image_height, image_width], "counts": counts}}
