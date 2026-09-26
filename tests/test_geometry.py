"""Synthetic geometry QA; no human-review or provider claims."""
import copy
import json

import numpy as np
import pytest
from pycocotools import mask as coco_mask

from compag_annotator.geometry import (decode_rle, deduplicate, encode_rle, geometry_bbox,
                                       geometry_mask, polygon_for_yolo, tile_boxes, validate_geometry)


def mask_geometry(mask):
    return {"type": "mask", "rle": encode_rle(mask)}


@pytest.mark.parametrize("shape", [(1, 1), (1, 37), (29, 1), (31, 27), (100, 103)])
@pytest.mark.parametrize("kind", ["empty", "full", "random", "checkerboard"])
def test_rle_real_pycocotools_roundtrip(shape, kind):
    rng = np.random.default_rng(419)
    if kind == "empty":
        mask = np.zeros(shape, bool)
    elif kind == "full":
        mask = np.ones(shape, bool)
    elif kind == "random":
        mask = rng.random(shape) > .65
    else:
        mask = np.indices(shape).sum(axis=0) % 2 == 0
    encoded = encode_rle(mask)
    assert sum(encoded["counts"]) == mask.size
    assert np.array_equal(decode_rle(json.loads(json.dumps(encoded))), mask)
    official = coco_mask.frPyObjects(encoded, *shape)
    assert np.array_equal(coco_mask.decode(official), mask)
    assert int(coco_mask.area(official)) == int(mask.sum())
    compressed = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
    assert np.array_equal(decode_rle(compressed), mask)
    compressed["counts"] = compressed["counts"].decode("ascii")
    assert np.array_equal(decode_rle(compressed), mask)


def test_rle_column_major_not_row_major():
    mask = np.array([[1, 0, 0], [1, 0, 1]], bool)
    assert encode_rle(mask) == {"size": [2, 3], "counts": [0, 2, 3, 1]}


@pytest.mark.parametrize("rle", [None, {}, {"size": [2, 2]}, {"size": [0, 2], "counts": [0]},
    {"size": [True, 2], "counts": [2]}, {"size": [2., 2], "counts": [4]},
    {"size": [2, 2], "counts": [True, 3]}, {"size": [2, 2], "counts": [-1, 5]},
    {"size": [2, 2], "counts": [5]}, {"size": [2, 2], "counts": [2]},
    {"size": [2, 2], "counts": []}, {"size": [2, 2], "counts": ""},
    {"size": [2, 2], "counts": "P"}, {"size": [2, 2], "counts": "z"},
    {"size": [2, 2], "counts": "9"}, {"size": [2, 2], "counts": "PPPPPPPPPP0"},
    {"size": [2, 2], "counts": b"\xff"}, {"size": [1000000000, 1000000000], "counts": [1]}])
def test_reject_bad_rle_before_native_decoder(rle):
    with pytest.raises(ValueError):
        decode_rle(rle)


@pytest.mark.parametrize("mask", [np.ones((1, 1, 1)), np.array([[2]]), np.array([[np.nan]]), np.empty((0, 2))])
def test_bad_encode(mask):
    with pytest.raises(ValueError):
        encode_rle(mask)


@pytest.mark.parametrize("geometry", [
    {"type": "box", "xyxy": [0, 0, 0, 4]}, {"type": "box", "xyxy": [-1, 0, 4, 4]},
    {"type": "box", "xyxy": [0, 0, 11, 4]}, {"type": "box", "xyxy": [0, 0, float("nan"), 4]},
    {"type": "polygon", "points": [[0, 0], [4, 4], [0, 4], [4, 0]]},
    {"type": "polygon", "points": [[0, 0], [2, 2], [4, 4]]},
    {"type": "polygon", "points": [[0, 0], [2, 2], [4, 0], [2, 2]]},
    {"type": "polygon", "points": [[0, 0], [2, 2], [11, 0]]},
    {"type": "polygon", "points": [[0, 0], [2, 2], [float("inf"), 0]]},
    {"type": "polygon", "points": [[True, 0], [2, 2], [3, 0]]},
    {"type": "ellipse"}, mask_geometry(np.zeros((10, 10), bool)), mask_geometry(np.ones((9, 10), bool))])
def test_reject_invalid_annotations(geometry):
    with pytest.raises(ValueError):
        validate_geometry(geometry, 10, 10)


def test_polygon_box_bounds_and_coco_rasterization():
    points = [[1.25, 2.25], [7.4, 2.25], [7.4, 8.7], [1.25, 8.7]]
    geometry = {"type": "polygon", "points": points + [points[0]]}
    before = copy.deepcopy(geometry)
    validated = validate_geometry(geometry, 12, 11)
    assert geometry == before
    assert validated["points"] == points
    official = coco_mask.merge(coco_mask.frPyObjects([np.array(points).ravel().tolist()], 11, 12))
    assert np.array_equal(geometry_mask(geometry, 12, 11), coco_mask.decode(official))
    assert geometry_bbox(geometry, 12, 11) == [1.25, 2.25, 7.4, 8.7]
    assert np.array_equal(geometry_mask({"type": "box", "xyxy": [1.25, 2.25, 7.4, 8.7]}, 12, 11), coco_mask.decode(official))


def test_holes_components_exact_native():
    mask = np.zeros((31, 37), bool)
    mask[2:17, 3:21] = True
    mask[5:12, 7:15] = False
    mask[23:27, 29:33] = True
    geometry = validate_geometry(mask_geometry(mask), 37, 31)
    assert np.array_equal(geometry_mask(geometry, 37, 31), mask)
    assert geometry_bbox(geometry, 37, 31) == [3, 2, 33, 27]
    with pytest.raises(ValueError, match="components.*holes"):
        polygon_for_yolo(geometry, 37, 31)
    points, report = polygon_for_yolo(geometry, 37, 31, allow_lossy=True)
    after = geometry_mask({"type": "polygon", "points": points}, 37, 31)
    assert report["components"] == 2 and report["holes"] == 1
    assert report["method"] == "all_components_convex_hull"
    assert report["removed_pixels"] == 0
    assert report["changed_pixels"] == np.count_nonzero(mask ^ after)
    assert report["iou"] == np.count_nonzero(mask & after) / np.count_nonzero(mask | after)
    assert np.array_equal(geometry_mask(geometry, 37, 31), mask)


def test_yolo_fill_holes_reports_area():
    mask = np.ones((7, 9), bool)
    mask[2:5, 2:6] = False
    _, report = polygon_for_yolo(mask_geometry(mask), 9, 7, True)
    assert report["method"] == "fill_holes"
    assert report["added_pixels"] == 12 and report["removed_pixels"] == 0
    assert report["iou"] == 51 / 63


@pytest.mark.parametrize("mask", [np.ones((1, 1), bool), np.ones((9, 7), bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]], bool)])
def test_simple_mask_to_polygon_strict_exact(mask):
    height, width = mask.shape
    points, report = polygon_for_yolo(mask_geometry(mask), width, height)
    assert report["iou"] == 1 and report["changed_pixels"] == 0
    assert np.array_equal(geometry_mask({"type": "polygon", "points": points}, width, height), mask)


def test_diagonal_touching_is_multiple_components():
    mask = np.eye(3, dtype=bool)
    with pytest.raises(ValueError, match="components"):
        polygon_for_yolo(mask_geometry(mask), 3, 3)


def test_yolo_points_are_original_pixels_and_box_never_segmentation():
    geometry = {"type": "polygon", "points": [[1, 1], [29, 1], [29, 19], [1, 19]]}
    points, report = polygon_for_yolo(geometry, 30, 20)
    assert points == geometry["points"] and max(p[0] for p in points) == 29
    for allow in (True, False):
        with pytest.raises(ValueError, match="Box-only"):
            polygon_for_yolo({"type": "box", "xyxy": [1, 1, 29, 19]}, 30, 20, allow)
    with pytest.raises(ValueError, match="foreground"):
        polygon_for_yolo({"type": "polygon", "points": [[0, 0], [.01, 0], [0, .01]]}, 30, 20)


@pytest.mark.parametrize("width,height,size,overlap", [(1, 1, 512, 128), (300, 200, 512, 128), (1300, 1100, 512, 128), (100, 100, 25, 0), (37, 43, 16, 15)])
def test_tiles_complete_row_major_edge_anchored(width, height, size, overlap):
    boxes = tile_boxes(width, height, size, overlap)
    coverage = np.zeros((height, width), bool)
    assert len({tuple(b) for b in boxes}) == len(boxes)
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        coverage[y0:y1, x0:x1] = True
    assert coverage.all()
    assert boxes[-1] == [max(0, width - size), max(0, height - size), width, height]
    assert [(b[1], b[0]) for b in boxes] == sorted((b[1], b[0]) for b in boxes)


@pytest.mark.parametrize("args", [(0, 1), (True, 5), (10, 10, 0, 0), (10, 10, 5, 5), (10, 10, 5, -1)])
def test_invalid_tiles(args):
    with pytest.raises(ValueError):
        tile_boxes(*args)


def test_dedup_uses_masks_classes_scores_no_union_no_mutation():
    a = np.eye(8, dtype=bool)
    b = np.fliplr(a)
    proposals = [{"id": "low", "class_id": "one", "score": .3, "geometry": mask_geometry(a)},
                 {"id": "different-mask", "class_id": "one", "geometry": mask_geometry(b)},
                 {"id": "high", "class_id": "one", "score": .9, "geometry": mask_geometry(a)},
                 {"id": "other-class", "class_id": "two", "geometry": mask_geometry(a)}]
    before = copy.deepcopy(proposals)
    output = deduplicate(proposals, 8, 8)
    assert [a["id"] for a in output] == ["different-mask", "high", "other-class"]
    assert proposals == before and all(a in proposals for a in output)


def test_unassigned_model_classes_not_collapsed():
    geometry = mask_geometry(np.ones((2, 3), bool))
    proposals = [{"class_id": None, "geometry": geometry, "model_class_index": 1},
                 {"class_id": None, "geometry": geometry, "model_class_index": 2},
                 {"class_id": None, "geometry": geometry}, {"class_id": None, "geometry": geometry}]
    assert len(deduplicate(proposals, 3, 2)) == 4
    with pytest.raises(ValueError):
        deduplicate(proposals, 3, 2, float("nan"))


def test_offset_cropped_dedup_matches_full_image_reference():
    masks = []
    for x0, y0, x1, y1 in [(1, 1, 20, 30), (2, 2, 21, 31), (40, 20, 55, 40), (40, 21, 55, 41)]:
        mask = np.zeros((50, 60), bool)
        mask[y0:y1, x0:x1] = True
        masks.append(mask)
    proposals = [{"id": str(i), "class_id": "one", "score": i / 10, "geometry": mask_geometry(mask)} for i, mask in enumerate(masks)]
    threshold = .8
    kept = []
    for index in reversed(range(len(masks))):
        if not any(np.count_nonzero(masks[index] & masks[j]) / np.count_nonzero(masks[index] | masks[j]) >= threshold for j in kept):
            kept.append(index)
    assert [a["id"] for a in deduplicate(proposals, 60, 50, threshold)] == [str(i) for i in sorted(kept)]


def test_all_3_by_3_masks_official_compressed_roundtrip():
    for number in range(512):
        mask = np.array([(number >> bit) & 1 for bit in range(9)], dtype=np.uint8).reshape(3, 3)
        compressed = coco_mask.encode(np.asfortranarray(mask))
        assert np.array_equal(decode_rle(compressed), mask)


def test_yolo_upstream_bbox_dedup_guard():
    from compag_annotator.geometry import validate_yolo_rows
    first = '0 0 0 1 0 1 1'
    second = '0 0 0 0 1 1 1'
    with pytest.raises(ValueError, match='would merge rows 1 and 2'):
        validate_yolo_rows([first, second])
    validate_yolo_rows([first, '1 0 0 0 1 1 1'])
    validate_yolo_rows([first, '0 .1 .1 .8 .1 .8 .8'])
    validate_yolo_rows([])
