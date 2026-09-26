"""Addendum CPU geometry tests: actual source crops, never model inference."""
import copy
import json
from decimal import Decimal

import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco_mask

import compag_annotator.geometry as geometry
from compag_annotator.geometry import (TilingError, decode_rle, encode_rle, geometry_mask,
                                       map_tile_geometry, mask_metadata, plan_tiles, tile_boxes)


def settings(width=512, height=512, *, preset="custom", mode="tiled", overlap_mode="percent", x=25, y=25):
    return {"mode": mode, "preset": preset, "width": width, "height": height,
            "overlap": {"mode": overlap_mode, "x": x, "y": y}}


def masked(array):
    return {"type": "mask", "rle": encode_rle(array)}


@pytest.mark.parametrize("width,height,preset", [(256, 256, "256"), (512, 512, "512"),
                                                 (1024, 1024, "1024"), (640, 384, "custom"), (513, 385, "custom")])
@pytest.mark.parametrize("image_width,image_height", [(2048, 1536), (1579, 1163), (117, 83)])
def test_actual_crop_presets_and_custom_cover_whole_original(width, height, preset, image_width, image_height):
    config = settings(width, height, preset=preset)
    plan = plan_tiles(image_width, image_height, config)
    source = Image.new("L", (image_width, image_height))
    coverage = np.zeros((image_height, image_width), dtype=np.uint16)
    origins = []
    ids = []
    for tile in plan["tiles"]:
        x0, y0, x1, y1 = tile["box"]
        crop = source.crop(tile["box"])
        assert crop.size == (min(width, image_width), min(height, image_height))
        assert tile["valid_box"] == [0, 0, crop.width, crop.height]
        assert tile["padding"] == [0, 0, 0, 0]
        coverage[y0:y1, x0:x1] += 1
        origins.append((y0, x0))
        ids.append(tile["id"])
    assert coverage.min() >= 1
    assert origins == sorted(origins) and len(set(origins)) == len(origins)
    assert len(set(ids)) == plan["tile_count"] == len(plan["tiles"])
    assert plan["tiles"][-1]["box"] == [max(0, image_width - width), max(0, image_height - height), image_width, image_height]
    assert plan["config"]["overlap_x"] == width // 4
    assert plan["config"]["overlap_y"] == height // 4
    assert plan["config"]["stride_x"] == width - width // 4
    assert plan["config"]["stride_y"] == height - height // 4


def test_defaults_and_whole_image_mode_are_honest():
    default = plan_tiles(2048, 1536)
    assert default["config"]["width"] == default["config"]["height"] == 512
    assert default["config"]["overlap_x"] == default["config"]["overlap_y"] == 128
    assert default["tile_count"] == 20
    whole = plan_tiles(2048, 1536, {"mode": "whole"})
    assert whole["tile_count"] == 1 and whole["tiles"][0]["box"] == [0, 0, 2048, 1536]
    assert whole["tiles"][0]["valid_box"] == [0, 0, 2048, 1536]
    assert whole["config"]["width"] == 512  # stored tiled control, not an encoder size


@pytest.mark.parametrize("width,height,x,y", [(513, 385, 25, 25), (513, 385, 33.3, 10.5), (1000, 7, .3, 99.99), (256, 1024, 0, 0)])
def test_percent_resolution_is_floor_per_axis(width, height, x, y):
    config = settings(width, height, x=x, y=y)
    actual = plan_tiles(width, height, config)["config"]
    assert actual["overlap_x"] == int(Decimal(width) * Decimal(str(x)) // 100)
    assert actual["overlap_y"] == int(Decimal(height) * Decimal(str(y)) // 100)


def test_percentage_and_pixel_modes_can_resolve_same_grid():
    percent = plan_tiles(1579, 1163, settings(513, 385))
    pixels = plan_tiles(1579, 1163, settings(513, 385, overlap_mode="pixels", x=128, y=96))
    assert [tile["box"] for tile in percent["tiles"]] == [tile["box"] for tile in pixels["tiles"]]
    assert percent["plan_hash"] != pixels["plan_hash"]  # authoritative input mode is part of provenance
    zero = plan_tiles(1024, 512, settings(256, 256, overlap_mode="pixels", x=0, y=0))
    assert zero["tile_count"] == 8
    assert {t["box"][0] for t in zero["tiles"]} == {0, 256, 512, 768}


@pytest.mark.parametrize("width,height", [(512, 512), (896, 512), (897, 512), (1024, 1024), (1, 1)])
def test_edge_origins_never_repeat_and_old_api_unchanged(width, height):
    actual = plan_tiles(width, height, settings(512, 512, preset="512"))
    assert [row["box"] for row in actual["tiles"]] == tile_boxes(width, height, 512, 128)
    assert tile_boxes(1500, 700) == [[0, 0, 1024, 700], [476, 0, 1500, 700]]


@pytest.mark.parametrize("config,field", [
    ({"width": True}, "width"), ({"height": 0}, "height"), ({"width": 512.0}, "width"),
    ({"preset": "custom", "width": -3}, "width"), ({"mode": "viewport"}, "mode"),
    ({"preset": 512}, "preset"), ({"preset": "256", "width": 512}, "width"),
    ({"preset": "custom", "width": 100000, "height": 100000}, "width"),
    ({"overlap": {"mode": "percent", "x": 100, "y": 25}}, "overlap.x"),
    ({"overlap": {"mode": "percent", "x": float("nan"), "y": 25}}, "overlap.x"),
    ({"overlap": {"mode": "percent", "x": float("inf"), "y": 25}}, "overlap.x"),
    ({"overlap": {"mode": "percent", "x": True, "y": 25}}, "overlap.x"),
    ({"overlap": {"mode": "pixels", "x": 25.5, "y": 25}}, "overlap.x"),
    ({"overlap": {"mode": "pixels", "x": -1, "y": 25}}, "overlap.x"),
    ({"overlap": {"mode": "pixels", "x": 512, "y": 25}}, "overlap.x"),
    ({"overlap": {"mode": "other", "x": 25, "y": 25}}, "overlap.mode"),
    ({"overlap": []}, "overlap"), ({"overlap": {"z": 1}}, "overlap.z"),
    ({"imgsz": 640}, "imgsz"), ([], "config"), ({"stride_x": 1}, "stride_x"),
    ({"overlap_y": False}, "overlap_y")])
def test_config_field_errors(config, field):
    with pytest.raises(TilingError) as error:
        plan_tiles(2048, 1536, config)
    assert field in error.value.fields
    assert isinstance(json.loads(json.dumps(error.value.fields))[field], str)


@pytest.mark.parametrize("width,height,sha,field", [(True, 3, None, "image_width"), (3, 3.5, None, "image_height"),
    (0, 2, None, "image_width"), (100000, 100000, None, "image_width"), (3, 3, "invalid", "image_sha256")])
def test_image_field_errors(width, height, sha, field):
    with pytest.raises(TilingError) as error:
        plan_tiles(width, height, image_sha256=sha)
    assert field in error.value.fields


def test_excessive_grid_is_rejected_before_origin_allocation(monkeypatch):
    from compag_annotator.geometry import tiling
    monkeypatch.setattr(tiling, "_origins", lambda *args: pytest.fail("Origins allocated before count validation"))
    with pytest.raises(TilingError) as error:
        plan_tiles(10000, 10000, settings(2, 2, overlap_mode="pixels", x=1, y=1))
    assert "tile_count" in error.value.fields
    assert "10000" in error.value.fields["tile_count"]


def test_plan_identity_serialization_replay_and_input_isolation():
    config = settings(513, 385, x=25., y=25.)
    before = copy.deepcopy(config)
    first = plan_tiles(2048, 1536, config, image_sha256="a" * 64)
    assert config == before
    replay = plan_tiles(2048, 1536, json.loads(json.dumps(first["config"])), image_sha256="A" * 64)
    assert replay == first
    assert plan_tiles(2048, 1536, config, image_sha256="b" * 64)["plan_hash"] != first["plan_hash"]
    assert plan_tiles(2049, 1536, config, image_sha256="a" * 64)["plan_hash"] != first["plan_hash"]
    config["width"] = 640
    assert first["config"]["width"] == 513
    first["config"]["overlap"]["x"] = 0
    assert config["overlap"]["x"] == 25
    persisted = copy.deepcopy(replay["config"])
    persisted["overlap_x"] += 1
    with pytest.raises(TilingError, match="Recorded effective value"):
        plan_tiles(2048, 1536, persisted)


def test_actual_preset_changes_source_grid_not_model_imgsz():
    plans = [plan_tiles(2048, 1536, {"preset": preset}) for preset in ["256", "512", "1024"]]
    assert len({p["tile_count"] for p in plans}) == 3
    assert [p["tiles"][0]["box"] for p in plans] == [[0, 0, 256, 256], [0, 0, 512, 512], [0, 0, 1024, 1024]]
    assert all("imgsz" not in p["config"] for p in plans)


def test_exact_crop_mapping_holes_components_and_image_edges():
    plan = plan_tiles(1579, 1163, settings(513, 385))
    tile = plan["tiles"][-1]
    local = np.ones((385, 513), bool)
    local[40:170, 30:330] = False
    local[250:300] = False
    before = masked(local)
    input_copy = copy.deepcopy(before)
    result = map_tile_geometry(before, tile, 1579, 1163)
    expected = np.zeros((1163, 1579), bool)
    x0, y0, x1, y1 = tile["box"]
    expected[y0:y1, x0:x1] = local
    assert np.array_equal(decode_rle(result["rle"]), expected)
    assert result["rle"] == encode_rle(expected)
    assert before == input_copy
    assert mask_metadata(result, 1579, 1163) == {"area": int(local.sum()), "bbox": [x0, y0, x1, y1]}
    official = coco_mask.frPyObjects(result["rle"], 1163, 1579)
    assert np.array_equal(coco_mask.decode(official), expected)


def test_small_image_valid_crop_not_preset_shape():
    tile = plan_tiles(13, 7)["tiles"][0]
    mask = np.ones((7, 13), bool)
    assert map_tile_geometry(masked(mask), tile, 13, 7) == masked(mask)
    with pytest.raises(ValueError, match="dimensions"):
        map_tile_geometry(masked(np.ones((512, 512), bool)), tile, 13, 7)


def test_explicit_padding_pixels_are_excluded_without_resize():
    tile = {"box": [3, 4, 8, 8], "valid_box": [2, 1, 7, 5], "padding": [2, 1, 3, 2]}
    local = np.ones((7, 10), bool)
    local[2:4, 3:5] = False
    mapped = map_tile_geometry(masked(local), tile, 17, 13)
    expected = np.zeros((13, 17), bool)
    expected[4:8, 3:8] = local[1:5, 2:7]
    assert mapped["rle"] == encode_rle(expected)
    local[1:5, 2:7] = False
    with pytest.raises(ValueError, match="padding is excluded"):
        map_tile_geometry(masked(local), tile, 17, 13)


def test_polygon_mapping_matches_original_coco_raster_contract():
    polygon = {"type": "polygon", "points": [[.2, 1.4], [7.6, 2.1], [4.1, 9.3]]}
    local = geometry_mask(polygon, 10, 11)
    result = map_tile_geometry(polygon, {"box": [12, 8, 22, 19]}, 35, 31)
    expected = np.zeros((31, 35), bool)
    expected[8:19, 12:22] = local
    assert result["rle"] == encode_rle(expected)


@pytest.mark.parametrize("tile", [{}, {"box": [-1, 0, 3, 2]}, {"box": [0, 0, 31, 2]},
    {"box": [0, 0, 0, 2]}, {"box": [0, False, 3, 2]}, {"box": [0, 0, 3., 2]},
    {"box": [0, 0, 3, 2], "valid_box": [0, 0, 2, 2]},
    {"box": [0, 0, 3, 2], "padding": [0, 0, -1, 0]},
    {"box": [0, 0, 3, 2], "padding": [0, 0, 0, 1], "valid_box": [0, 0, 3, 3]}])
def test_malformed_tile_descriptors_rejected(tile):
    with pytest.raises(ValueError):
        map_tile_geometry(masked(np.ones((2, 3), bool)), tile, 30, 20)


def test_mapping_compressed_rle_and_zero_run_normalization():
    local = np.array([[True, False, True], [True, True, False]])
    compressed = coco_mask.encode(np.asfortranarray(local, dtype=np.uint8))
    tile = {"box": [4, 5, 7, 7]}
    assert map_tile_geometry({"type": "mask", "rle": compressed}, tile, 20, 15) == map_tile_geometry(masked(local), tile, 20, 15)
    zero_runs = {"type": "mask", "rle": {"size": [2, 3], "counts": [0, 1, 0, 1, 4]}}
    normalized = {"type": "mask", "rle": {"size": [2, 3], "counts": [0, 2, 4]}}
    assert map_tile_geometry(zero_runs, tile, 20, 15) == map_tile_geometry(normalized, tile, 20, 15)


def test_mapping_never_decodes_or_allocates_a_full_image(monkeypatch):
    local = masked(np.array([[1, 0, 1], [1, 1, 0]], bool))
    expected = {"area": 4, "bbox": [9997, 9998, 10000, 10000]}
    def forbidden(*args, **kwargs):
        pytest.fail("Eager raster allocation/decoding in run-level mapping")
    monkeypatch.setattr(geometry, "decode_rle", forbidden)
    monkeypatch.setattr(geometry.np, "zeros", forbidden)
    monkeypatch.setattr(geometry.np, "nonzero", forbidden)
    mapped = map_tile_geometry(local, {"box": [9997, 9998, 10000, 10000]}, 10000, 10000)
    assert mapped["rle"]["size"] == [10000, 10000]
    assert len(mapped["rle"]["counts"]) < 12
    assert mask_metadata(mapped, 10000, 10000) == expected
    assert geometry.validate_geometry(mapped, 10000, 10000) == mapped
    assert geometry.geometry_bbox(mapped, 10000, 10000) == expected["bbox"]


def test_random_asymmetric_crop_mappings_match_pixel_paste():
    rng = np.random.default_rng(309)
    for _ in range(80):
        height, width = map(int, rng.integers(1, 25, size=2))
        x, y = map(int, rng.integers(0, 20, size=2))
        local = rng.random((height, width)) > .65
        local[0, 0] = True
        mapped = map_tile_geometry(masked(local), {"box": [x, y, x + width, y + height]}, 50, 50)
        expected = np.zeros((50, 50), bool)
        expected[y:y + height, x:x + width] = local
        assert mapped["rle"] == encode_rle(expected)


def test_huge_percentage_is_a_field_error_not_float_overflow():
    with pytest.raises(TilingError) as error:
        plan_tiles(20, 20, settings(x=10 ** 1000))
    assert "overlap.x" in error.value.fields


def test_random_padding_clipping_and_translation_match_pixel_paste():
    rng = np.random.default_rng(780)
    for _ in range(40):
        crop_width, crop_height = map(int, rng.integers(1, 16, size=2))
        left, top, right, bottom = map(int, rng.integers(0, 6, size=4))
        x0, y0 = map(int, rng.integers(0, 15, size=2))
        local = rng.random((crop_height + top + bottom, crop_width + left + right)) > .55
        local[top, left] = True
        tile = {"box": [x0, y0, x0 + crop_width, y0 + crop_height],
                "valid_box": [left, top, left + crop_width, top + crop_height], "padding": [left, top, right, bottom]}
        mapped = map_tile_geometry(masked(local), tile, 40, 40)
        expected = np.zeros((40, 40), bool)
        expected[y0:y0 + crop_height, x0:x0 + crop_width] = local[top:top + crop_height, left:left + crop_width]
        assert mapped["rle"] == encode_rle(expected)
