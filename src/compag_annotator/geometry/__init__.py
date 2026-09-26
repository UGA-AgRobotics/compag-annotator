"""Original-image geometry with exact COCO raster semantics.

Coordinates describe pixel edges, boxes have exclusive right/bottom edges.
Polygon rasterization is pycocotools' COCO polygon rasterization. Canonical RLE
uses the full image, background-first runs in column-major order. No ML imports.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from pycocotools import mask as coco_mask
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

# Bound allocations before passing untrusted geometry to native libraries.
MAX_PIXELS = 100_000_000
MAX_VERTICES = 1_000_000
MAX_VECTOR_RUNS = 100_000


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _dimensions(width, height):
    width, height = _integer(width, "width"), _integer(height, "height")
    if min(width, height) <= 0 or width * height > MAX_PIXELS:
        raise ValueError(f"Image dimensions must be positive and at most {MAX_PIXELS} pixels")
    return width, height


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("Coordinates must be finite numbers")
    return float(value)


def encode_rle(mask):
    """Encode binary pixels without changing holes, components or empty masks.

    The run transition algorithm is adapted from A's metrics.encode_mask;
    cropping was removed because this API binds masks to the full image.
    """
    raster = np.asarray(mask)
    if raster.ndim != 2 or not np.isin(raster, [0, 1]).all():
        raise ValueError("Mask must be a two-dimensional binary raster")
    _dimensions(raster.shape[1], raster.shape[0])
    flat = raster.astype(bool).ravel(order="F")
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [flat.size]))).tolist()
    if flat[0]:
        counts.insert(0, 0)
    return {"size": list(raster.shape), "counts": counts}


def _compressed_counts(encoded, total):
    """Validate COCO's signed/differential LEB128 before any C decoder call."""
    if isinstance(encoded, bytes):
        try:
            encoded = encoded.decode("ascii")
        except UnicodeError as exc:
            raise ValueError("Invalid compressed COCO RLE") from exc
    counts, position, accumulated = [], 0, 0
    if not encoded or len(encoded) > 7 * (total + 1):
        raise ValueError("Invalid compressed COCO RLE length")
    while position < len(encoded):
        value, shift = 0, 0
        while True:
            if position >= len(encoded) or shift > 35:
                raise ValueError("Truncated or oversized compressed RLE run")
            code = ord(encoded[position]) - 48
            position += 1
            if not 0 <= code <= 63:
                raise ValueError("Invalid compressed COCO RLE character")
            value |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0 or value > total or accumulated + value > total:
            raise ValueError("Invalid compressed COCO RLE run length")
        counts.append(value)
        accumulated += value
        if len(counts) > 2 * total + 1:
            raise ValueError("Too many RLE runs")
    return counts


def decode_rle(rle):
    """Decode checked uncompressed or official compressed COCO RLE to bool."""
    if not isinstance(rle, dict) or not isinstance(rle.get("size"), (list, tuple)) or len(rle["size"]) != 2:
        raise ValueError("RLE requires size [height,width] and counts")
    height, width = rle["size"]
    width, height = _dimensions(width, height)
    total = width * height
    counts = rle.get("counts")
    if isinstance(counts, (str, bytes)):
        counts = _compressed_counts(counts, total)
    if not isinstance(counts, (list, tuple)) or not counts or len(counts) > 2 * total + 1:
        raise ValueError("Invalid RLE counts")
    counts = [_integer(v, "RLE run") for v in counts]
    if min(counts) < 0 or sum(counts) != total:
        raise ValueError("RLE runs must be nonnegative and sum to the image size")
    flat = np.zeros(total, dtype=bool)
    position = 0
    for index, count in enumerate(counts):
        if index % 2:
            flat[position:position + count] = True
        position += count
    return flat.reshape((height, width), order="F")


def validate_geometry(geometry, width, height):
    """Return a fresh canonical dictionary; reject rather than silently repair."""
    width, height = _dimensions(width, height)
    if not isinstance(geometry, dict):
        raise ValueError("Geometry must be an object")
    kind = geometry.get("type")
    if kind == "mask":
        return {"type": "mask", "rle": canonical_rle(geometry.get("rle"), width, height)}
    if kind == "box":
        values = geometry.get("xyxy")
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError("Box requires xyxy with four coordinates")
        x0, y0, x1, y1 = map(_number, values)
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError("Box must have positive area and be inside the image")
        return {"type": "box", "xyxy": [x0, y0, x1, y1]}
    if kind == "polygon":
        points = geometry.get("points")
        if not isinstance(points, (list, tuple)) or not 3 <= len(points) <= MAX_VERTICES:
            raise ValueError("Polygon requires between 3 and 1000000 vertices")
        result = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Polygon points must be [x,y] pairs")
            x, y = map(_number, point)
            if not 0 <= x <= width or not 0 <= y <= height:
                raise ValueError("Polygon coordinates must be inside the image")
            result.append([x, y])
        if result[0] == result[-1]:
            result.pop()
        if len(result) < 3 or len({tuple(p) for p in result}) != len(result):
            raise ValueError("Polygon has repeated vertices or too few distinct points")
        polygon = Polygon(result)
        if not polygon.is_valid or polygon.area <= 0:
            raise ValueError("Polygon is degenerate or self-intersecting; explicitly repair it")
        return {"type": "polygon", "points": result}
    raise ValueError("Unsupported geometry type")


def geometry_mask(geometry, width, height):
    geometry = validate_geometry(geometry, width, height)
    if geometry["type"] == "mask":
        return decode_rle(geometry["rle"])
    if geometry["type"] == "box":
        x0, y0, x1, y1 = geometry["xyxy"]
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    else:
        points = geometry["points"]
    rles = coco_mask.frPyObjects([np.asarray(points).ravel().tolist()], height, width)
    return coco_mask.decode(coco_mask.merge(rles)).astype(bool)


def geometry_bbox(geometry, width, height):
    geometry = validate_geometry(geometry, width, height)
    if geometry["type"] == "box":
        return geometry["xyxy"]
    if geometry["type"] == "polygon":
        points = np.asarray(geometry["points"])
        return [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]
    return mask_metadata(geometry, width, height)["bbox"]


def _pixel_polygon(mask):
    # Union horizontal pixel runs, never one polygon per foreground pixel.
    rectangles = []
    for y, row in enumerate(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        if len(rectangles) + len(edges) // 2 > MAX_VECTOR_RUNS:
            raise ValueError("Mask is too complex for bounded polygon conversion; retain exact COCO RLE or per-instance PNG")
        rectangles.extend(box(int(start), y, int(stop), y + 1) for start, stop in edges.reshape(-1, 2))
    return unary_union(rectangles)


def _loss_report(before, after, **metadata):
    intersection = int(np.count_nonzero(before & after))
    union = int(np.count_nonzero(before | after))
    changed = int(np.count_nonzero(before ^ after))
    return {"lossy": bool(changed), "changed_pixels": changed,
            "added_pixels": int(np.count_nonzero(after & ~before)),
            "removed_pixels": int(np.count_nonzero(before & ~after)),
            "original_area": int(before.sum()), "converted_area": int(after.sum()),
            "iou": intersection / union if union else 1.0,
            "rasterization": "coco", **metadata}


def polygon_for_yolo(geometry, width, height, allow_lossy=False):
    """Return ORIGINAL-coordinate points and a per-instance conversion report.

    YOLO export/training share this one converter. Strict masks must be a simple
    polygon with an exact COCO pixel round trip. Approximation fills holes or
    uses the convex hull of ALL components; it never splits or drops instances.
    Boxes remain detection labels even with loss consent.
    """
    geometry = validate_geometry(geometry, width, height)
    if geometry["type"] == "box":
        raise ValueError("Box-only annotations are not segmentation ground truth; create a mask or polygon")
    original = geometry_mask(geometry, width, height)
    if not original.any():
        raise ValueError("Segmentation has no foreground pixels at original resolution")
    components, holes, method = 1, 0, "original_polygon"
    if geometry["type"] == "polygon":
        points = geometry["points"]
    else:
        polygon = _pixel_polygon(original)
        pieces = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
        components = len(pieces)
        holes = sum(len(piece.interiors) for piece in pieces)
        if components != 1 or holes:
            if not allow_lossy:
                raise ValueError(f"YOLO segmentation cannot preserve {components} components and {holes} holes; explicit allow_lossy required")
            polygon = polygon.convex_hull if components != 1 else Polygon(polygon.exterior)
            method = "all_components_convex_hull" if components != 1 else "fill_holes"
        else:
            method = "pixel_edge_polygon"
        points = [[float(x), float(y)] for x, y in polygon.exterior.coords[:-1]]
    converted = geometry_mask({"type": "polygon", "points": points}, width, height)
    report = _loss_report(original, converted, components=components, holes=holes, method=method,
                          allow_lossy=bool(allow_lossy))
    if report["lossy"] and not allow_lossy:
        raise ValueError(f"YOLO polygon changes {report['changed_pixels']} pixels; explicit allow_lossy required")
    return points, report


def tile_boxes(width, height, tile_size=1024, overlap=128):
    """Row-major edge-anchored tiles adapted from A.original_axis_positions."""
    width, height = _dimensions(width, height)
    tile_size, overlap = _integer(tile_size, "tile_size"), _integer(overlap, "overlap")
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    def positions(length):
        far_edge = max(0, length - tile_size)
        values = list(range(0, far_edge + 1, tile_size - overlap))
        if values[-1] != far_edge:
            values.append(far_edge)
        return values
    return [[x, y, min(x + tile_size, width), min(y + tile_size, height)]
            for y in positions(height) for x in positions(width)]


def validate_yolo_rows(lines):
    """Reject per-image segment rows Ultralytics would silently deduplicate.

    Upstream verify_image_label applies unique(class, float32 xywh), even when
    two different instance polygons have the same bounding box. Training and
    export callers should run this AFTER normalizing/serializing one image's
    rows. There is no safe automatic fix that preserves instance identities.
    """
    seen = {}
    for number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 7 or (len(fields) - 1) % 2:
            raise ValueError("YOLO segment rows require a class and at least three points")
        try:
            class_index = int(fields[0])
            points = np.array(fields[1:], dtype=np.float32).reshape(-1, 2)
        except (ValueError, OverflowError) as exc:
            raise ValueError("Invalid YOLO class/coordinates") from exc
        if class_index < 0 or str(class_index) != fields[0] or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError("Invalid normalized YOLO segmentation row")
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        center, size = (minimum + maximum) / 2, maximum - minimum
        key = (class_index, *center.tolist(), *size.tolist())
        if key in seen:
            raise ValueError(f"YOLO loader would merge rows {seen[key]} and {number}: same class and float32 bounding box; resolve these instances explicitly")
        seen[key] = number


def deduplicate(proposals, width, height, iou=.8):
    """Stable score-ordered class-aware mask NMS; never invent mask unions.

    Unassigned model classes use their model identity/index for comparisons.
    Bare unassigned proposals are retained because their class is unknown.
    """
    iou = _number(iou)
    if not 0 < iou <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    prepared = []
    for index, proposal in enumerate(proposals):
        mask = geometry_mask(proposal["geometry"], width, height)
        if not mask.any():
            raise ValueError("Proposal contains no foreground pixels")
        key = proposal.get("class_id")
        if key is None:
            model_index = proposal.get("model_class_index")
            key = ("unassigned", proposal.get("source", {}).get("model_id"), model_index) if model_index is not None else ("unknown", index)
        score = _number(proposal.get("score", 0))
        xs = np.flatnonzero(mask.any(axis=0))
        ys = np.flatnonzero(mask.any(axis=1))
        bbox = (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)
        x0, y0, x1, y1 = bbox
        # A's bbox-local overlap contract avoids retaining one full scene per object.
        cropped = mask[y0:y1, x0:x1].copy()
        prepared.append((index, score, key, cropped, proposal, bbox, int(cropped.sum())))
    def overlap(left, right):
        a, b = left[5], right[5]
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        if x0 >= x1 or y0 >= y1:
            return 0.0
        intersection = int(np.count_nonzero(
            left[3][y0-a[1]:y1-a[1], x0-a[0]:x1-a[0]] &
            right[3][y0-b[1]:y1-b[1], x0-b[0]:x1-b[0]]))
        return intersection / (left[6] + right[6] - intersection)
    kept = []
    for row in sorted(prepared, key=lambda r: -r[1]):
        if any(row[2] == previous[2] and overlap(row, previous) >= iou for previous in kept):
            continue
        kept.append(row)
    return [row[4] for row in sorted(kept, key=lambda r: r[0])]


# Import after the core validation definitions to keep the shared helpers small.
from .rle import canonical_rle, mask_metadata, mask_union
from .tiling import TilingError, map_tile_geometry, plan_tiles

__all__ = ["validate_geometry", "geometry_mask", "geometry_bbox", "encode_rle",
           "decode_rle", "polygon_for_yolo", "validate_yolo_rows", "tile_boxes", "deduplicate",
           "TilingError", "plan_tiles", "map_tile_geometry", "mask_union", "mask_metadata"]
