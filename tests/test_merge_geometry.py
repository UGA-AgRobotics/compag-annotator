"""Addendum exact manual union and metadata, with no model/GPU requirements."""
import copy
import json
import tracemalloc

import numpy as np
import pytest
from pycocotools import mask as coco_mask

import compag_annotator.geometry as geometry
from compag_annotator.geometry import (decode_rle, encode_rle, geometry_mask, map_tile_geometry,
                                       mask_metadata, mask_union, polygon_for_yolo, validate_geometry)
from compag_annotator.formats import export_annotations, import_annotations


def masked(mask):
    return {"type": "mask", "rle": encode_rle(mask)}


def assert_official(merged, expected):
    h, w = expected.shape
    official = coco_mask.frPyObjects(merged["rle"], h, w)
    assert np.array_equal(coco_mask.decode(official), expected)
    meta = mask_metadata(merged, w, h)
    box = coco_mask.toBbox(official).astype(int).tolist()
    assert meta == {"area": int(coco_mask.area(official)), "bbox": [box[0], box[1], box[0] + box[2], box[1] + box[3]]}
    assert merged["rle"] == encode_rle(expected)


@pytest.mark.parametrize("kind", ["adjacent", "overlap", "disconnected", "nested", "identical"])
def test_union_exact_pixels_area_bbox_and_no_input_mutation(kind):
    a = np.zeros((37, 41), bool)
    b = np.zeros_like(a)
    a[3:20, 4:21] = True
    if kind == "adjacent":
        b[3:20, 21:32] = True
    elif kind == "overlap":
        b[10:30, 10:31] = True
    elif kind == "disconnected":
        b[29:35, 31:39] = True
    elif kind == "nested":
        b[6:11, 8:14] = True
    else:
        b[:] = a
    parents = [masked(a), masked(b)]
    before = copy.deepcopy(parents)
    merged = mask_union(parents, 41, 37)
    assert_official(merged, a | b)
    assert parents == before
    assert set(merged) == {"type", "rle"}  # no invented class, confidence, human actor or IDs
    assert mask_union(list(reversed(parents)), 41, 37) == merged


def test_union_preserves_holes_disconnected_parts_and_never_adds_bridges():
    a = np.zeros((40, 50), bool)
    a[2:30, 2:30] = True
    a[7:25, 7:25] = False
    b = np.zeros_like(a)
    b[31:38, 41:48] = True
    merged = mask_union([masked(a), masked(b)], 50, 40)
    assert_official(merged, a | b)
    assert not decode_rle(merged["rle"])[7:25, 7:25].any()
    with pytest.raises(ValueError, match="components.*holes"):
        polygon_for_yolo(merged, 50, 40)
    _, report = polygon_for_yolo(merged, 50, 40, allow_lossy=True)
    assert report["components"] == 2 and report["holes"] == 1
    assert report["added_pixels"] > 0 and report["removed_pixels"] == 0


def test_union_only_fills_hole_pixels_actually_present_in_other_parent():
    ring = np.ones((11, 13), bool)
    ring[2:9, 3:10] = False
    addition = np.zeros_like(ring)
    addition[4:7, 5:8] = True
    merged = mask_union([masked(ring), masked(addition)], 13, 11)
    assert_official(merged, ring | addition)
    assert mask_metadata(merged, 13, 11)["area"] == int(ring.sum()) + 9


def test_cross_tile_fragments_merge_in_original_coordinates():
    left = np.ones((9, 11), bool)
    right = np.ones((9, 11), bool)
    left[2:7, 3:9] = False
    right[2:7, :4] = False
    first = map_tile_geometry(masked(left), {"box": [5, 7, 16, 16]}, 31, 29)
    second = map_tile_geometry(masked(right), {"box": [12, 7, 23, 16]}, 31, 29)
    merged = mask_union([first, second], 31, 29)
    expected = np.zeros((29, 31), bool)
    expected[7:16, 5:16] |= left
    expected[7:16, 12:23] |= right
    assert_official(merged, expected)
    assert mask_metadata(merged, 31, 29)["area"] < int(left.sum() + right.sum())


def test_polygon_and_mask_union_rasterizes_with_coco_without_bbox_fill():
    polygon = {"type": "polygon", "points": [[1.1, 2.3], [17.2, 3.8], [6.3, 21.1]]}
    mask = np.zeros((27, 31), bool)
    mask[20:25, 23:28] = True
    parents = [polygon, masked(mask)]
    merged = mask_union(parents, 31, 27)
    assert_official(merged, geometry_mask(polygon, 31, 27) | mask)


@pytest.mark.parametrize("parents", [[], [{"type": "mask", "rle": {"size": [2, 3], "counts": [0, 6]}}], None, {}, "mask"])
def test_union_requires_parent_sequence_and_at_least_two(parents):
    with pytest.raises(ValueError):
        mask_union(parents, 3, 2)


def test_union_rejects_boxes_empty_mismatched_and_invalid_geometries():
    valid = masked(np.ones((5, 7), bool))
    bad = [{"type": "box", "xyxy": [0, 0, 2, 2]}, masked(np.ones((4, 7), bool)),
           masked(np.zeros((5, 7), bool)), {"type": "mask", "rle": {"size": [5, 7], "counts": [-1, 36]}},
           {"type": "polygon", "points": [[1, 1], [4, 4], [1, 4], [4, 1]]}]
    for parent in bad:
        with pytest.raises(ValueError):
            mask_union([valid, parent], 7, 5)
    with pytest.raises(ValueError, match="Box-only"):
        map_tile_geometry(bad[0], {"box": [0, 0, 7, 5]}, 7, 5)


def test_merge_parent_resource_limit_is_explicit():
    parent = masked(np.ones((1, 1), bool))
    with pytest.raises(ValueError, match="1000"):
        mask_union((parent for _ in range(1001)), 1, 1)
    assert mask_union([parent] * 1000, 1, 1) == parent


@pytest.mark.parametrize("shape", [(1, 1), (1, 43), (39, 1), (23, 31)])
def test_metadata_all_edges_asymmetric_and_foreground_runs_cross_columns(shape):
    h, w = shape
    rng = np.random.default_rng(612)
    for mask in [np.ones(shape, bool), rng.random(shape) > .7]:
        mask[0, 0] = True
        mask[-1, -1] = True
        result = masked(mask)
        assert_official(result, mask)


def test_metadata_vector_pixel_area_not_continuous_area():
    box = {"type": "box", "xyxy": [1.2, 2.2, 8.4, 9.4]}
    expected = geometry_mask(box, 12, 13)
    assert mask_metadata(box, 12, 13) == mask_metadata(masked(expected), 12, 13)


def test_compressed_and_noncanonical_zero_runs_union_exactly():
    a = np.array([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    b = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    compressed = coco_mask.encode(np.asfortranarray(a))
    compressed["counts"] = compressed["counts"].decode("ascii")
    merged = mask_union([{"type": "mask", "rle": compressed}, masked(b)], 3, 2)
    assert merged == {"type": "mask", "rle": {"size": [2, 3], "counts": [0, 6]}}
    noncanonical = {"type": "mask", "rle": {"size": [2, 3], "counts": [0, 1, 0, 1, 4, 0]}}
    assert validate_geometry(noncanonical, 3, 2) == {"type": "mask", "rle": {"size": [2, 3], "counts": [0, 2, 4]}}


def test_dense_100_megapixel_merge_metadata_validation_allocate_no_rasters(monkeypatch):
    parent = {"type": "mask", "rle": {"size": [10000, 10000], "counts": [0, 100_000_000]}}
    def forbidden(*args, **kwargs):
        pytest.fail("Full raster allocation/decoding in exact RLE operation")
    monkeypatch.setattr(geometry, "decode_rle", forbidden)
    monkeypatch.setattr(geometry.np, "zeros", forbidden)
    monkeypatch.setattr(geometry.np, "nonzero", forbidden)
    tracemalloc.start()
    try:
        merged = mask_union([parent] * 100, 10000, 10000)
        assert merged == parent
        assert mask_metadata(merged, 10000, 10000) == {"area": 100_000_000, "bbox": [0, 0, 10000, 10000]}
        assert validate_geometry(merged, 10000, 10000) == parent
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000


def test_random_multi_parent_union_matches_numpy_pixel_or():
    rng = np.random.default_rng(601)
    for _ in range(50):
        height, width = map(int, rng.integers(1, 31, size=2))
        parents, expected = [], np.zeros((height, width), bool)
        for _ in range(int(rng.integers(2, 8))):
            mask = rng.random((height, width)) > .7
            mask[0, 0] = True
            parents.append(masked(mask))
            expected |= mask
        assert_official(mask_union(parents, width, height), expected)


def test_merged_topology_native_json_and_coco_roundtrip(tmp_path):
    ring = np.ones((20, 30), bool)
    ring[3:17, 3:27] = False
    fragment = np.zeros_like(ring)
    fragment[8:12, 11:15] = True
    merged = mask_union([masked(ring), masked(fragment)], 30, 20)
    assert json.loads(json.dumps(merged)) == merged
    document = {"schema_version": 1, "classes": [{"id": "one", "name": "Leaf"}],
                "images": [{"id": "image", "name": "synthetic.png", "width": 30, "height": 20}],
                "annotations": [{"id": "child", "image_id": "image", "class_id": "one", "geometry": merged,
                                  "status": "accepted", "review_actor": "automated_qa", "human_verified": False}]}
    root = tmp_path / "coco"
    export_annotations("coco", document, root, include_images=False)
    restored = import_annotations("coco", root, document["images"], document["classes"])
    assert restored["annotations"][0]["geometry"] == merged
    assert restored["annotations"][0]["human_verified"] is False
    with pytest.raises(ValueError, match="holes"):
        export_annotations("yolo_seg", document, tmp_path / "strict", include_images=False)
