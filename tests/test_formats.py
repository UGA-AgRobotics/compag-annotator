"""Real exchange-library tests with synthetic assets only."""
import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import yaml

from compag_annotator.formats import export_annotations, import_annotations, FormatError
from compag_annotator.geometry import encode_rle, geometry_mask


@pytest.fixture
def document(tmp_path):
    images = []
    for index, role in enumerate(["train", "validation", "test"]):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (37, 31), (index * 40, 30, 80)).save(path)
        images.append({"id": f"image-{index}", "name": "duplicate name image ü.png", "width": 37,
                       "height": 31, "path": str(path), "role": role})
    classes = [{"id": "class-0", "name": "leaf <&> ✓", "order": 0},
               {"id": "class-1", "name": "fruit", "order": 1},
               {"id": "class-2", "name": "stem", "order": 2}]
    polygon = {"type": "polygon", "points": [[2, 2], [17, 2], [17, 19], [2, 19]]}
    annotations = [{"id": f"object-{i}", "image_id": f"image-{i}", "class_id": f"class-{i}",
                    "geometry": copy.deepcopy(polygon), "status": "accepted", "human_verified": False,
                    "review_actor": "automated_qa", "source": {"kind": "manual"}} for i in range(3)]
    return {"schema_version": 1, "classes": classes, "images": images, "annotations": annotations}


def topology(document):
    document = copy.deepcopy(document)
    mask = np.zeros((31, 37), bool)
    mask[2:21, 2:21] = True
    mask[5:12, 5:12] = False
    mask[25:29, 30:35] = True
    document["annotations"][0]["geometry"] = {"type": "mask", "rle": encode_rle(mask)}
    return document, mask


def export_import(format_name, document, tmp_path, **kwargs):
    output = tmp_path / f"export-{format_name}"
    manifest = export_annotations(format_name, document, output, **kwargs)
    result = import_annotations(format_name, output, document["images"], document["classes"])
    return output, manifest, result


def assert_masks_equal(before, after):
    for source in before:
        matching = [a for a in after if a["class_id"] == source["class_id"] and a["image_id"] == source["image_id"]]
        assert len(matching) == 1
        assert np.array_equal(geometry_mask(source["geometry"], 37, 31), geometry_mask(matching[0]["geometry"], 37, 31))


def test_coco_exact_masks_real_api(document, tmp_path):
    document, mask = topology(document)
    before = copy.deepcopy(document)
    root, manifest, result = export_import("coco", document, tmp_path)
    assert_masks_equal(document["annotations"], result["annotations"])
    coco = COCO(str(root / "annotations.json"))
    annotation = coco.anns[1]
    assert annotation["iscrowd"] == 0
    assert isinstance(annotation["segmentation"]["counts"], str)
    assert annotation["bbox"] == [2., 2., 33., 27.]
    assert annotation["area"] == int(mask.sum())
    assert np.array_equal(coco.annToMask(annotation), mask)
    assert document == before and not manifest["lossy"]
    for annotation in result["annotations"]:
        assert annotation["status"] == "proposal"
        assert annotation["human_verified"] is False and annotation["review_actor"] is None
    assert {c["id"] for c in result["classes"]} == {c["id"] for c in document["classes"]}


def test_external_coco_uncompressed_and_multi_polygons(document, tmp_path):
    first = document["images"][0]
    mask = np.eye(31, 37, dtype=bool)
    coco = {"images": [{"id": 12, "file_name": first["name"], "width": 37, "height": 31}],
            "categories": [{"id": 99, "name": "new class"}], "annotations": [
                {"id": 1, "image_id": 12, "category_id": 99, "segmentation": encode_rle(mask), "iscrowd": 1},
                {"id": 2, "image_id": 12, "category_id": 99,
                 "segmentation": [[0, 0, 3, 0, 3, 3, 0, 3], [8, 8, 11, 8, 11, 11, 8, 11]], "area": 999}]}
    path = tmp_path / "external.json"
    path.write_text(json.dumps(coco))
    result = import_annotations("coco", path, [first], [])
    assert len(result["classes"]) == 1 and result["classes"][0]["name"] == "new class"
    assert np.array_equal(geometry_mask(result["annotations"][0]["geometry"], 37, 31), mask)
    assert result["annotations"][0]["source"]["iscrowd"] == 1
    assert geometry_mask(result["annotations"][1]["geometry"], 37, 31).sum() == 18
    assert "recomputed" in result["loss_report"][0]["reason"]


@pytest.mark.parametrize("format_name", ["yolo_seg", "labelme", "png"])
def test_representable_roundtrip_and_portable_files(format_name, document, tmp_path):
    before = copy.deepcopy(document)
    root, manifest, result = export_import(format_name, document, tmp_path)
    assert_masks_equal(document["annotations"], result["annotations"])
    assert document == before
    assert len(list((root / "images").glob("*.png"))) == 3
    for filename in manifest["files"]:
        assert not Path(filename).is_absolute() and ".." not in Path(filename).parts
        assert (root / filename).is_file()
    text = "".join(p.read_text() for p in root.rglob("*") if p.suffix in {".json", ".yaml", ".txt"})
    assert str(tmp_path) not in text
    assert "automated_qa" in (root / "manifest.json").read_text()


def test_yolo_seg_strict_all_errors_and_explicit_loss(document, tmp_path):
    document, mask = topology(document)
    document["annotations"][1]["geometry"] = {"type": "box", "xyxy": [1, 1, 9, 9]}
    destination = tmp_path / "blocked"
    with pytest.raises(FormatError) as error:
        export_annotations("yolo_seg", document, destination)
    assert {r["annotation_id"] for r in error.value.loss_report} == {"object-0", "object-1"}
    assert not destination.exists() and not list(tmp_path.glob(".compag-export-*"))
    with pytest.raises(FormatError, match="Box-only"):
        export_annotations("yolo_seg", document, destination, allow_lossy=True)
    document["annotations"].pop(1)
    root, manifest, result = export_import("yolo_seg", document, tmp_path, allow_lossy=True)
    report = next(r for r in manifest["loss_report"] if r["annotation_id"] == "object-0")
    assert report["changed_pixels"] > 0 and report["removed_pixels"] == 0 and 0 < report["iou"] < 1
    after = geometry_mask(next(a for a in result["annotations"] if a["image_id"] == "image-0")["geometry"], 37, 31)
    assert report["changed_pixels"] == np.count_nonzero(mask ^ after)
    assert manifest["lossy"]


def test_yolo_dataset_split_and_negative_images(document, tmp_path):
    document["annotations"] = []
    root, manifest, result = export_import("yolo_seg", document, tmp_path)
    assert not result["annotations"]
    dataset = yaml.safe_load((root / "dataset.yaml").read_text())
    assert list(dataset["names"]) == [0, 1, 2]
    for index, split in enumerate(["train", "val", "test"]):
        assert (root / dataset[split]).read_text().strip() == f"./images/image-{index}.png"
        assert not (root / "labels" / f"image-{index}.txt").read_text()
    assert manifest["split_counts"] == {"train": 1, "val": 1, "test": 1}


@pytest.mark.parametrize("format_name", ["yolo_box", "voc"])
def test_boxes_roundtrip_and_segmentation_loss_gate(format_name, document, tmp_path):
    with pytest.raises(ValueError, match="allow_lossy"):
        export_annotations(format_name, document, tmp_path / "blocked")
    for annotation in document["annotations"]:
        annotation["geometry"] = {"type": "box", "xyxy": [0, 0, 37, 31]}
    root, manifest, result = export_import(format_name, document, tmp_path)
    assert_masks_equal(document["annotations"], result["annotations"])
    if format_name == "voc":
        xml = (root / "annotations/image-0.xml").read_text()
        assert "<xmin>1</xmin>" in xml and "<xmax>37</xmax>" in xml
        assert "leaf &lt;&amp;&gt; ✓" in xml
    document, _ = topology(document)
    manifest = export_annotations(format_name, document, tmp_path / "lossy", allow_lossy=True)
    assert manifest["lossy"] and manifest["loss_report"][0]["changed_pixels"] > 0


def test_voc_fractional_coordinates_require_consent(document, tmp_path):
    for annotation in document["annotations"]:
        annotation["geometry"] = {"type": "box", "xyxy": [1.2, 2.2, 17.3, 23.4]}
    with pytest.raises(FormatError):
        export_annotations("voc", document, tmp_path / "strict")
    root, manifest, result = export_import("voc", document, tmp_path, allow_lossy=True)
    assert manifest["lossy"]
    assert result["annotations"][0]["geometry"]["xyxy"] == [1, 2, 18, 24]


def test_labelme_group_disconnected_components_roundtrip(document, tmp_path):
    mask = np.zeros((31, 37), bool)
    mask[2:8, 2:8] = True
    mask[20:27, 28:34] = True
    document["annotations"][0]["geometry"] = {"type": "mask", "rle": encode_rle(mask)}
    root, manifest, result = export_import("labelme", document, tmp_path)
    shapes = json.loads((root / "images/image-0.json").read_text())["shapes"]
    assert len(shapes) == 2 and shapes[0]["group_id"] == shapes[1]["group_id"]
    assert len(result["annotations"]) == 3
    assert_masks_equal(document["annotations"], result["annotations"])
    assert not manifest["lossy"]


def test_labelme_flags_holes_and_unsupported_shape(document, tmp_path):
    doc, _ = topology(document)
    with pytest.raises(FormatError, match="holes"):
        export_annotations("labelme", doc, tmp_path / "strict")
    _, manifest, _ = export_import("labelme", doc, tmp_path, allow_lossy=True)
    assert manifest["lossy"] and manifest["loss_report"][0]["added_pixels"] == 49
    labelme = {"imagePath": "image-0.png", "imageWidth": 37, "imageHeight": 31, "shapes": [
        {"shape_type": "rectangle", "points": [[10, 10], [1, 2]], "label": "fruit", "group_id": 5, "flags": {"occluded": True}},
        {"shape_type": "circle", "points": [[5, 5], [8, 8]], "label": "fruit"}]}
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps(labelme))
    with pytest.raises(FormatError, match="circle"):
        import_annotations("labelme", path, document["images"], document["classes"])
    imported = import_annotations("labelme", path, document["images"], document["classes"], options={"allow_lossy": True})
    annotation = imported["annotations"][0]
    assert annotation["geometry"]["xyxy"] == [1, 2, 10, 10]
    assert annotation["source"]["labelme"]["flags"] == {"occluded": True}
    assert annotation["source"]["labelme"]["group_id"] == 5
    assert len(imported["loss_report"]) == 1


def test_png_per_instance_preserves_overlap_holes_components(document, tmp_path):
    document, _ = topology(document)
    document["annotations"][1]["image_id"] = "image-0"
    root, manifest, result = export_import("png", document, tmp_path)
    assert not manifest["lossy"]
    assert_masks_equal(document["annotations"], result["annotations"])
    assert len(list((root / "masks").rglob("*.png"))) == 3


@pytest.mark.parametrize("variant", ["semantic", "instance_id"])
def test_png_single_plane_overlap_policy(variant, document, tmp_path):
    document["annotations"][1]["image_id"] = "image-0"
    with pytest.raises(FormatError, match="conflict policy"):
        export_annotations("png", document, tmp_path / "strict", png_variant=variant)
    for policy in ["first", "last"]:
        manifest = export_annotations("png", document, tmp_path / policy, png_variant=variant, conflict_policy=policy)
        assert manifest["lossy"]
        lost_id = "object-1" if policy == "first" else "object-0"
        assert any(row["annotation_id"] == lost_id and row["removed_pixels"] == 255 for row in manifest["loss_report"])
        result = import_annotations("png", tmp_path / policy, document["images"], document["classes"])
        assert len(result["annotations"]) == 2
        assert any("no visible" in row["reason"] for row in result["loss_report"])


def test_png_semantic_instance_identity_loss(document, tmp_path):
    document["annotations"][1]["image_id"] = "image-0"
    document["annotations"][1]["class_id"] = "class-0"
    document["annotations"][1]["geometry"] = {"type": "polygon", "points": [[25, 20], [35, 20], [35, 29], [25, 29]]}
    with pytest.raises(FormatError, match="merge instances"):
        export_annotations("png", document, tmp_path / "strict", png_variant="semantic")
    _, manifest, result = export_import("png", document, tmp_path, png_variant="semantic", allow_lossy=True)
    assert manifest["lossy"] and len(result["annotations"]) == 2
    assert any("identities" in row["reason"] for row in manifest["loss_report"])


@pytest.mark.parametrize("variant", ["semantic", "instance_id"])
def test_png_uint16_ids_above_255(variant, document, tmp_path):
    image = document["images"][0]
    document["images"] = [image]
    document["classes"] = [{"id": f"c{i}", "name": f"class {i}"} for i in range(300)]
    document["annotations"] = []
    for i in range(300):
        x, y = i % 30, i // 30
        document["annotations"].append({"id": f"a{i}", "image_id": image["id"], "class_id": f"c{i}",
                                         "geometry": {"type": "box", "xyxy": [x, y, x + 1, y + 1]}})
    root, manifest, result = export_import("png", document, tmp_path, png_variant=variant)
    with Image.open(root / "masks/image-0.png") as image:
        array = np.asarray(image)
        assert int(array.max()) == 300 and len(np.unique(array)) == 301
    assert len(result["annotations"]) == 300
    assert not manifest["lossy"]


def test_standalone_png_import(document, tmp_path):
    path = tmp_path / "mask.png"
    array = np.eye(31, 37, dtype=np.uint8) * 255
    Image.fromarray(array).save(path)
    result = import_annotations("png", path, document["images"], document["classes"], options={"image_id": "image-0", "class_id": "class-0"})
    assert np.array_equal(geometry_mask(result["annotations"][0]["geometry"], 37, 31), array != 0)


@pytest.mark.parametrize("format_name", ["coco", "yolo_seg", "voc", "labelme", "png"])
def test_empty_document_and_no_image_copies(format_name, tmp_path):
    root, manifest, result = export_import(format_name, {"classes": [], "images": [], "annotations": []}, tmp_path, include_images=False)
    assert result["annotations"] == [] and not manifest["annotation_count"]


@pytest.mark.parametrize("format_name", ["coco", "yolo_seg", "labelme", "png"])
def test_unassigned_and_invalid_geometry_block_exports(format_name, document, tmp_path):
    document["annotations"][0]["class_id"] = None
    with pytest.raises(ValueError, match="unassigned"):
        export_annotations(format_name, document, tmp_path / "blocked")
    assert not (tmp_path / "blocked").exists()


def test_no_clobber_atomic_failure_symlinks_and_malicious_ids(document, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("original")
    with pytest.raises(ValueError, match="never overwritten"):
        export_annotations("coco", document, output)
    assert (output / "keep.txt").read_text() == "original"
    empty = tmp_path / "empty"
    empty.mkdir()
    export_annotations("coco", document, empty, include_images=False)
    bad = copy.deepcopy(document)
    bad["images"][0]["path"] = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError):
        export_annotations("coco", bad, tmp_path / "failed")
    assert not (tmp_path / "failed").exists() and not list(tmp_path.glob(".compag-export-*"))
    link = tmp_path / "link"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_annotations("coco", document, link)
    document["images"][0]["id"] = "../escape"
    with pytest.raises(ValueError, match="identifiers"):
        export_annotations("coco", document, tmp_path / "evil")


def test_coco_ambiguity_traversal_nan_and_duplicate_keys(document, tmp_path):
    path = tmp_path / "bad.json"
    body = {"images": [{"id": 1, "file_name": document["images"][0]["name"], "width": 37, "height": 31}], "categories": [], "annotations": []}
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="ambiguous"):
        import_annotations("coco", path, document["images"], [])
    result = import_annotations("coco", path, document["images"], [], options={"image_mapping": {document["images"][0]["name"]: "image-1"}})
    assert result["annotations"] == []
    for evil in ["../secret.png", "/etc/passwd", "C:\\data\\image.png"]:
        body["images"][0]["file_name"] = evil
        path.write_text(json.dumps(body))
        with pytest.raises(ValueError):
            import_annotations("coco", path, document["images"], [])
    for evil in ['{"annotations": [], "annotations": []}', '{"area": NaN}']:
        path.write_text(evil)
        with pytest.raises(ValueError):
            import_annotations("coco", path, document["images"], [])


@pytest.mark.parametrize("line", ["0 .1 .1 .2 .2", "-1 .1 .1 .9 .1 .9 .9", "1.5 .1 .1 .9 .1 .9 .9", "0 nan .1 .9 .1 .9 .9", "0 -.1 .1 .9 .1 .9 .9", "8 .1 .1 .9 .1 .9 .9", "0 .1 .1 .9 .1 .9"])
def test_yolo_rejects_malformed_or_box_rows(line, document, tmp_path):
    root = tmp_path / "labels"
    root.mkdir()
    (root / "image-0.txt").write_text(line)
    with pytest.raises(ValueError):
        import_annotations("yolo_seg", root, document["images"], document["classes"])


def test_png_rejects_mapping_escape_and_unknown_values(document, tmp_path):
    root, _, _ = export_import("png", document, tmp_path, png_variant="instance_id")
    mapping = json.loads((root / "mapping.json").read_text())
    mapping["objects"][0]["file_name"] = "../secret.png"
    (root / "mapping.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="traversal"):
        import_annotations("png", root, document["images"], document["classes"])
    mapping["objects"][0]["file_name"] = "masks/image-0.png"
    (root / "mapping.json").write_text(json.dumps(mapping))
    Image.fromarray(np.full((31, 37), 999, dtype=np.uint16)).save(root / "masks/image-0.png")
    with pytest.raises(ValueError, match="absent"):
        import_annotations("png", root, document["images"], document["classes"])


def test_exif_oriented_export_pixels(tmp_path):
    source = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (12, 7), "red")
    exif = Image.Exif()
    exif[274] = 6
    image.save(source, exif=exif)
    document = {"images": [{"id": "oriented", "name": "original.jpg", "width": 7, "height": 12, "path": str(source)}], "classes": [], "annotations": []}
    export_annotations("coco", document, tmp_path / "export")
    with Image.open(tmp_path / "export/images/oriented.png") as output:
        assert output.size == (7, 12) and output.getexif().get(274) is None


def test_actual_ultralytics_label_loader(document, tmp_path):
    """Optional provider-runtime check; never installs/downloads models or runs GPU."""
    runtime = os.environ.get("COMPAG_YOLO_TEST_PYTHON")
    if not runtime:
        pytest.skip("Set COMPAG_YOLO_TEST_PYTHON to an installed Ultralytics runtime for real loader validation")
    root = tmp_path / "yolo-loader"
    export_annotations("yolo_seg", document, root)
    script = '''
import sys
from pathlib import Path
import torch
from ultralytics.data.utils import verify_image_label
assert not torch.cuda.is_initialized(), "Loader import initialized CUDA"
root = Path(sys.argv[1])
for index in range(3):
    image = str(root / "images" / f"image-{index}.png")
    label = str(root / "labels" / f"image-{index}.txt")
    result = verify_image_label((image, label, "", False, 3, 0, 0, False))
    assert result[0] == image, result[-1]
    assert len(result[3]) == 1 and result[1].shape == (1, 5), result
assert not torch.cuda.is_initialized(), "Label verification initialized CUDA"
print("real Ultralytics verify_image_label accepted every segmentation row")
'''
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "YOLO_CONFIG_DIR": str(tmp_path / "yolo-settings")}
    result = subprocess.run([runtime, "-c", script, str(root)], capture_output=True, text=True, timeout=60, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr


def test_yolo_class_ids_survive_target_rename(document, tmp_path):
    root = tmp_path / "yolo"
    export_annotations("yolo_seg", document, root)
    classes = copy.deepcopy(document["classes"])
    classes[0]["name"] = "renamed leaf"
    result = import_annotations("yolo_seg", root, document["images"], classes)
    assert len(result["classes"]) == 3
    assert result["annotations"][0]["class_id"] == "class-0"
    assert result["classes"][0]["name"] == "renamed leaf"
    mapping = json.loads((root / "class_map.json").read_text())
    mapping["classes"][0]["name"] = "inconsistent mapping"
    (root / "class_map.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="disagree"):
        import_annotations("yolo_seg", root, document["images"], classes)


def test_xml_entities_and_unsafe_yaml_are_rejected(document, tmp_path):
    xml = tmp_path / "malicious.xml"
    xml.write_text('<!DOCTYPE annotation [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><annotation><filename>&xxe;</filename></annotation>')
    with pytest.raises(ValueError, match="Unsafe"):
        import_annotations("voc", xml, document["images"], document["classes"])
    root = tmp_path / "yaml"
    root.mkdir()
    (root / "dataset.yaml").write_text('names: !!python/object/apply:os.system ["exit 7"]')
    with pytest.raises(ValueError, match="Unsafe"):
        import_annotations("yolo_seg", root, document["images"], document["classes"])


def test_export_symlink_nested_path_and_import_mask_symlink(document, tmp_path):
    safe = tmp_path / "safe"
    safe.mkdir()
    link = tmp_path / "linked-parent"
    link.symlink_to(safe, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_annotations("coco", document, link / "export")
    root, _, _ = export_import("png", document, tmp_path)
    image = root / "masks/image-0/object-0.png"
    original = tmp_path / "original-mask.png"
    image.rename(original)
    image.symlink_to(original)
    with pytest.raises(ValueError):
        import_annotations("png", root, document["images"], document["classes"])


def test_explicit_class_mapping_and_unknown_class_policy(document, tmp_path):
    root = tmp_path / "labels"
    root.mkdir()
    (root / "dataset.yaml").write_text('names:\n  0: foreign-class\n')
    (root / "image-0.txt").write_text('0 .1 .1 .9 .1 .9 .9 .1 .9\n')
    with pytest.raises(ValueError, match="Unknown"):
        import_annotations("yolo_seg", root, document["images"], document["classes"], options={"create_classes": False})
    result = import_annotations("yolo_seg", root, document["images"], document["classes"], options={"class_mapping": {"0": "class-2"}, "create_classes": False})
    assert result["annotations"][0]["class_id"] == "class-2"
    assert len(result["classes"]) == 3


def test_full_image_brush_mask_export_geometry_consistency(document, tmp_path):
    width, height = 1103, 709
    image = document["images"][0]
    image.update(width=width, height=height)
    Image.new("RGB", (width, height)).save(image["path"])
    document["images"] = [image]
    document["annotations"] = [document["annotations"][0]]
    mask = np.zeros((height, width), bool)
    mask[20:600, 40:1000] = True
    mask[100:550, 100:800] = False
    mask[680:700, 1070:1100] = True
    document["annotations"][0]["geometry"] = {"type": "mask", "rle": encode_rle(mask)}
    root, _, result = export_import("coco", document, tmp_path)
    assert np.array_equal(geometry_mask(result["annotations"][0]["geometry"], width, height), mask)
    official = COCO(str(root / "annotations.json"))
    assert np.array_equal(official.annToMask(official.anns[1]), mask)


def test_coco_box_scope_and_reject_wrong_rle_size(document, tmp_path):
    for annotation in document["annotations"]:
        annotation["geometry"] = {"type": "box", "xyxy": [1.25, 2.75, 20.5, 24.5]}
    root, _, result = export_import("coco", document, tmp_path)
    for annotation in result["annotations"]:
        assert annotation["geometry"] == {"type": "box", "xyxy": [1.25, 2.75, 20.5, 24.5]}
    body = json.loads((root / "annotations.json").read_text())
    body["annotations"][0]["segmentation"] = {"size": [1, 1], "counts": [0, 1]}
    (root / "annotations.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="dimensions"):
        import_annotations("coco", root, document["images"], document["classes"])


def test_single_class_yolo_empty_negative_and_strict_mask(document, tmp_path):
    document["classes"] = document["classes"][:1]
    document["annotations"] = document["annotations"][:1]
    mask = np.zeros((31, 37), bool)
    mask[2:29, 2:11] = True
    mask[23:29, 2:35] = True
    document["annotations"][0]["geometry"] = {"type": "mask", "rle": encode_rle(mask)}
    root, manifest, result = export_import("yolo_seg", document, tmp_path)
    assert_masks_equal(document["annotations"], result["annotations"])
    assert len(manifest["class_mapping"]) == 1
    assert not (root / "labels/image-1.txt").read_text()


def test_yolo_export_blocks_distinct_instances_with_same_loader_bbox(document, tmp_path):
    document["annotations"] = document["annotations"][:2]
    first, second = document["annotations"]
    second["image_id"], second["class_id"] = first["image_id"], first["class_id"]
    first["geometry"] = {"type": "polygon", "points": [[0, 0], [10, 0], [10, 10]]}
    second["geometry"] = {"type": "polygon", "points": [[0, 0], [0, 10], [10, 10]]}
    with pytest.raises(FormatError, match="would merge"):
        export_annotations("yolo_seg", document, tmp_path / "blocked")


def test_arbitrary_yolo_yaml_filename(document, tmp_path):
    root = tmp_path / "external-yolo"
    (root / "labels").mkdir(parents=True)
    (root / "chosen-labels.yml").write_text('names:\n  0: stem\n')
    (root / "labels/image-0.txt").write_text('0 .1 .1 .9 .1 .9 .9 .1 .9\n')
    result = import_annotations("yolo_seg", root / "chosen-labels.yml", document["images"], document["classes"])
    assert result["annotations"][0]["class_id"] == "class-2"


def test_png_identifier_concatenation_cannot_overwrite_another_mask(document, tmp_path):
    document["images"] = document["images"][:2]
    document["images"][0]["id"], document["images"][1]["id"] = "a_b", "a"
    document["annotations"] = document["annotations"][:2]
    document["annotations"][0].update(id="c", image_id="a_b")
    document["annotations"][1].update(id="b_c", image_id="a")
    root, _, result = export_import("png", document, tmp_path)
    assert len(result["annotations"]) == 2
    assert (root / "masks/a_b/c.png").is_file()
    assert (root / "masks/a/b_c.png").is_file()


@pytest.mark.parametrize("format_name", ["coco", "yolo_seg", "yolo_box", "voc", "labelme", "png"])
def test_import_class_mapping_dropdown_contract_all_formats(format_name, document, tmp_path):
    if format_name in {"yolo_box", "voc"}:
        for annotation in document["annotations"]:
            annotation["geometry"] = {"type": "box", "xyxy": [2, 2, 17, 19]}
    root = tmp_path / "class-contract"
    export_annotations(format_name, document, root)
    # Category/index/value keys are strings in the JSON options sent by the UI.
    if format_name in {"voc", "labelme"}:
        mapping = {document["classes"][0]["name"]: "class-2"}
    else:
        source_index = "0" if format_name.startswith("yolo") else "1"
        mapping = {source_index: "class-2", document["classes"][0]["name"]: "class-1"}
    before = copy.deepcopy(document)
    result = import_annotations(format_name, root, document["images"], document["classes"],
                                options={"class_mapping": mapping, "create_classes": False})
    first = next(a for a in result["annotations"] if a["image_id"] == "image-0")
    assert first["class_id"] == "class-2"
    assert result["classes"] == document["classes"] and document == before
    assert all(a["status"] == "proposal" and a["human_verified"] is False for a in result["annotations"])
    mapping = {key: "missing-project-class" for key in mapping}
    with pytest.raises(ValueError, match="Class mapping"):
        import_annotations(format_name, root, document["images"], document["classes"], options={"class_mapping": mapping})


def test_bare_yolo_txt_honors_explicit_index_mapping(document, tmp_path):
    path = tmp_path / "image-0.txt"
    path.write_text("0 .1 .1 .9 .1 .9 .9 .1 .9\n")
    result = import_annotations("yolo_seg", path, document["images"], document["classes"],
                                options={"class_mapping": {"0": "class-2"}, "create_classes": False})
    assert result["annotations"][0]["class_id"] == "class-2"


def test_yolo_class_names_uses_source_order_then_dropdown_mapping(document, tmp_path):
    path = tmp_path / "image-0.txt"
    path.write_text("1 .1 .1 .9 .1 .9 .9 .1 .9\n")
    result = import_annotations("yolo_seg", path, document["images"], document["classes"],
                                options={"class_names": ["foreign first", "foreign second"],
                                         "class_mapping": {"0": "class-1", "1": "class-2"}, "create_classes": False})
    assert result["annotations"][0]["class_id"] == "class-2" and len(result["classes"]) == 3


def test_export_class_mapping_is_derived_from_project_class_order(document, tmp_path):
    document["classes"][0]["order"] = 2
    document["classes"][1]["order"] = 0
    document["classes"][2]["order"] = 1
    manifest = export_annotations("yolo_seg", document, tmp_path / "ordered-export")
    assert [(c["index"], c["class_id"]) for c in manifest["class_mapping"]] == [(0, "class-1"), (1, "class-2"), (2, "class-0")]


@pytest.mark.parametrize("format_name", ["coco", "yolo_seg", "png"])
def test_import_mapping_supports_stable_source_id(format_name, document, tmp_path):
    root = tmp_path / "stable-source-ids"
    export_annotations(format_name, document, root)
    result = import_annotations(format_name, root, document["images"], document["classes"],
                                options={"class_mapping": {"class-0": "class-2", document["classes"][0]["name"]: "class-1"}})
    assert next(a for a in result["annotations"] if a["image_id"] == "image-0")["class_id"] == "class-2"


def test_coco_image_mapping_accepts_external_numeric_id(document, tmp_path):
    path = tmp_path / "external-image-id.json"
    path.write_text(json.dumps({"images": [{"id": 55, "file_name": "unknown-source.png", "width": 37, "height": 31}],
                                "categories": [{"id": 9, "name": "stem"}], "annotations": [
                                    {"id": 1, "image_id": 55, "category_id": 9, "bbox": [1, 1, 8, 8]}]}))
    result = import_annotations("coco", path, document["images"], document["classes"],
                                options={"image_mapping": {"55": "image-1"}})
    assert result["annotations"][0]["image_id"] == "image-1"
