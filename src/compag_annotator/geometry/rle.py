"""Exact run-level mask operations, with memory proportional to encoded runs.

Runs use COCO's column-major full-image coordinate system. Polygon rasterization
uses the same native COCO rasterizer as the editor without decoding its raster.
These helpers never load a model or recover/add pixels outside a requested union.
"""
from __future__ import annotations

from collections.abc import Iterable
from heapq import merge
from itertools import islice

from . import _compressed_counts, _dimensions, _integer, coco_mask, validate_geometry

MAX_MERGE_PARENTS = 1000


def checked_counts(rle, width=None, height=None):
    """Validate encoded dimensions and counts before any allocation/geometry use."""
    if not isinstance(rle, dict) or not isinstance(rle.get("size"), (list, tuple)) or len(rle["size"]) != 2:
        raise ValueError("RLE requires size [height,width] and counts")
    source_height, source_width = rle["size"]
    source_width, source_height = _dimensions(source_width, source_height)
    if width is not None or height is not None:
        width, height = _dimensions(width, height)
        if (source_width, source_height) != (width, height):
            raise ValueError("Mask dimensions do not match the image")
    total = source_width * source_height
    counts = rle.get("counts")
    if isinstance(counts, (str, bytes)):
        counts = _compressed_counts(counts, total)
    if not isinstance(counts, (list, tuple)) or not counts or len(counts) > 2 * total + 1:
        raise ValueError("Invalid RLE counts")
    values = []
    position = 0
    for count in counts:
        count = _integer(count, "RLE run")
        if count < 0 or position + count > total:
            raise ValueError("RLE runs must be nonnegative and sum to the image size")
        position += count
        values.append(count)
    if position != total:
        raise ValueError("RLE runs must be nonnegative and sum to the image size")
    return source_width, source_height, values


def foreground_intervals(counts):
    """Yield ordered half-open foreground intervals in Fortran-flat coordinates."""
    position = 0
    for index, count in enumerate(counts):
        end = position + count
        if index % 2 and count:
            yield position, end
        position = end


def counts_from_intervals(intervals, total):
    """Union sorted foreground intervals into canonical background-first runs."""
    counts = []
    previous_end = 0
    current_start = current_end = None
    for start, end in intervals:
        if not 0 <= start < end <= total:
            raise ValueError("Foreground interval lies outside the image")
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            counts.extend((current_start - previous_end, current_end - current_start))
            previous_end = current_end
            current_start, current_end = start, end
    if current_start is not None:
        counts.extend((current_start - previous_end, current_end - current_start))
        previous_end = current_end
    if previous_end < total or not counts:
        counts.append(total - previous_end)
    return counts


def canonical_rle(rle, width=None, height=None, *, require_nonempty=True):
    width, height, counts = checked_counts(rle, width, height)
    counts = counts_from_intervals(foreground_intervals(counts), width * height)
    if require_nonempty and len(counts) == 1:
        raise ValueError("Empty masks are not annotation instances")
    return {"size": [height, width], "counts": counts}


def geometry_rle(geometry, width, height, *, allow_boxes=False):
    """Rasterize vectors directly to runs; retain raster topology exactly."""
    width, height = _dimensions(width, height)
    if not isinstance(geometry, dict):
        raise ValueError("Geometry must be an object")
    kind = geometry.get("type")
    if kind == "mask":
        return canonical_rle(geometry.get("rle"), width, height)
    if kind == "box" and not allow_boxes:
        raise ValueError("Box-only annotations cannot be merged or mapped as segmentation masks; create an explicit mask first")
    validated = validate_geometry(geometry, width, height)
    if kind == "polygon":
        points = validated["points"]
    elif kind == "box":
        x0, y0, x1, y1 = validated["xyxy"]
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    else:
        raise ValueError("Expected polygon or mask geometry")
    encoded = coco_mask.frPyObjects([[v for point in points for v in point]], height, width)[0]
    return canonical_rle(encoded, width, height)


def mask_union(geometries, width, height):
    """Return exact pixel OR of 2..1000 polygon/mask parents as canonical RLE.

    Image identity, live revisions, class choice, history and user authorization
    belong to the caller. Boxes and empty instances are always rejected. No
    parent is changed and no label/review/model metadata is inferred here.
    """
    width, height = _dimensions(width, height)
    if isinstance(geometries, (str, bytes, dict)) or not isinstance(geometries, Iterable):
        raise ValueError("Merge requires a sequence of mask or polygon geometries")
    parents = list(islice(iter(geometries), MAX_MERGE_PARENTS + 1))
    if not 2 <= len(parents) <= MAX_MERGE_PARENTS:
        raise ValueError(f"Merge requires between 2 and {MAX_MERGE_PARENTS} parents")
    streams = []
    for geometry in parents:
        rle = geometry_rle(geometry, width, height)
        streams.append(foreground_intervals(rle["counts"]))
    counts = counts_from_intervals(merge(*streams), width * height)
    return {"type": "mask", "rle": {"size": [height, width], "counts": counts}}


def mask_metadata(geometry, width, height):
    """Return exact foreground pixel area and exclusive raster bbox, without decode.

    Vector polygons/boxes use canonical COCO pixel rasterization. For continuous
    vector bounds use geometry_bbox instead. Empty instances remain invalid.
    """
    rle = geometry_rle(geometry, width, height, allow_boxes=True)
    height, width = rle["size"]
    area = 0
    xmin, ymin, xmax, ymax = width, height, 0, 0
    for start, end in foreground_intervals(rle["counts"]):
        first_x, first_y = divmod(start, height)
        last_x, last_y = divmod(end - 1, height)
        area += end - start
        xmin, xmax = min(xmin, first_x), max(xmax, last_x + 1)
        if first_x != last_x:
            ymin, ymax = 0, height
        else:
            ymin, ymax = min(ymin, first_y), max(ymax, last_y + 1)
    return {"area": area, "bbox": [xmin, ymin, xmax, ymax]}
