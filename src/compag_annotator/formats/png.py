"""PNG instance masks and 16-bit semantic/instance planes plus mapping.json."""
from collections import defaultdict

import numpy as np
from PIL import Image

from ..geometry import encode_rle, geometry_mask
from ..geometry import _loss_report
from ._common import FormatError, child, read_json, write_json

VARIANTS = {"per_instance": "per_instance", "class": "semantic", "semantic": "semantic",
            "instance": "instance_id", "instance_id": "instance_id"}


def export(root, classes, images, annotations, manifest, *, png_variant="per_instance", conflict_policy="error", allow_lossy=False, **options):
    if png_variant not in VARIANTS:
        raise ValueError("PNG variant must be per_instance, semantic/class or instance_id")
    variant = VARIANTS[png_variant]
    if conflict_policy not in {"error", "first", "last"}:
        raise ValueError("PNG conflict policy must be error, first or last")
    if len(classes) > 65535 and variant == "semantic":
        raise ValueError("Semantic PNG supports at most 65535 classes plus background")
    by_image = defaultdict(list)
    for annotation in annotations:
        by_image[annotation["image_id"]].append(annotation)
    class_values = {c["id"]: index + 1 for index, c in enumerate(classes)}
    mapping = {"schema_version": 1, "variant": variant, "background": 0,
               "conflict_policy": conflict_policy,
               "classes": [{"id": c["id"], "name": c["name"], "value": class_values[c["id"]]} for c in classes],
               "images": [{"id": im["id"], "file_name": f"images/{im['id']}.png", "width": im["width"], "height": im["height"]} for im in images],
               "objects": []}
    (root / "masks").mkdir()
    for image in images:
        objects = by_image[image["id"]]
        if len(objects) > 65535 and variant == "instance_id":
            raise ValueError("Instance-ID PNG supports at most 65535 instances per image")
        if variant == "per_instance":
            for annotation in objects:
                mask = geometry_mask(annotation["geometry"], image["width"], image["height"])
                if not mask.any():
                    raise ValueError("Cannot export an empty per-instance mask")
                filename = f"masks/{image['id']}/{annotation['id']}.png"
                (root / "masks" / str(image["id"])).mkdir(exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255).save(root / filename)
                mapping["objects"].append({"image_id": image["id"], "annotation_id": annotation["id"],
                                           "class_id": annotation["class_id"], "file_name": filename, "value": 255})
            continue
        if variant == "semantic":
            class_groups = defaultdict(list)
            for annotation in objects:
                class_groups[annotation["class_id"]].append(annotation)
            for class_id, group in class_groups.items():
                if len(group) > 1:
                    report = {"image_id": image["id"], "annotation_ids": [a["id"] for a in group],
                              "lossy": True, "reason": "Semantic PNG merges instance identities within a class",
                              "class_id": class_id, "instances_before": len(group), "instances_after": 1}
                    if not allow_lossy:
                        raise FormatError("Semantic PNG requires allow_lossy to merge instances of one class", loss_report=[report])
                    manifest["loss_report"].append(report)
        plane = np.zeros((image["height"], image["width"]), dtype=np.uint16)
        masks, values = [], []
        for index, annotation in enumerate(objects):
            mask = geometry_mask(annotation["geometry"], image["width"], image["height"])
            value = class_values[annotation["class_id"]] if variant == "semantic" else index + 1
            masks.append(mask)
            values.append(value)
            overlap = mask & (plane != 0)
            if overlap.any() and conflict_policy == "error":
                raise FormatError("Overlapping PNG instances require an explicit first/last conflict policy",
                                  loss_report=[{"annotation_id": annotation["id"], "overlap_pixels": int(overlap.sum()), "reason": "Single-plane overlap"}])
            plane[mask & (plane == 0) if conflict_policy == "first" else mask] = value
        for annotation, mask, value in zip(objects, masks, values):
            visible = (plane == value) & mask
            report = {"annotation_id": annotation["id"], **_loss_report(mask, visible), "method": f"png_{conflict_policy}_overlap"}
            if report["lossy"]:
                manifest["loss_report"].append(report)
        filename = f"masks/{image['id']}.png"
        Image.fromarray(plane).save(root / filename)
        emitted = set()
        for annotation, value in zip(objects, values):
            if value in emitted:
                continue
            emitted.add(value)
            mapping["objects"].append({"image_id": image["id"], "annotation_id": annotation["id"],
                                       "class_id": annotation["class_id"], "file_name": filename, "value": value})
    write_json(root / "mapping.json", mapping)
    manifest["png_variant"] = variant
    manifest["conflict_policy"] = conflict_policy
    manifest["warnings"].append("PNG retains raster pixels; authored vector coordinates and project history are not represented.")


def _read_mask(path, width, height):
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (width, height) or len(image.getbands()) != 1:
            raise ValueError("Mask must be a single-channel PNG matching the image dimensions")
        array = np.asarray(image).copy()
    if not np.issubdtype(array.dtype, np.integer) and array.dtype != bool:
        raise ValueError("Mask PNG must contain integer values")
    if array.min() < 0 or array.max() > 65535:
        raise ValueError("PNG mask values must be in [0,65535]")
    return array


def load(path, context):
    if path.is_file() and path.suffix.lower() == ".png":
        image = context.image(external_id=context.options.get("image_id"))
        class_id = context.options.get("class_id")
        if not any(c["id"] == class_id for c in context.classes):
            raise ValueError("A standalone binary PNG needs options.class_id")
        array = _read_mask(path, image["width"], image["height"])
        if not np.isin(array, [0, 1, 255]).all():
            raise ValueError("Standalone binary PNG must contain only 0, 1 or 255")
        context.add(image, class_id, {"type": "mask", "rle": encode_rle(array != 0)}, external_id=path.name)
        return
    root = path if path.is_dir() else path.parent
    body = read_json(root / "mapping.json" if path.is_dir() else path)
    if body.get("schema_version") != 1 or body.get("variant") not in {"per_instance", "semantic", "instance_id"} or body.get("background") != 0:
        raise ValueError("Unsupported PNG mapping schema, variant or background")
    classes, images = {}, {}
    for label in body["classes"]:
        if label["id"] in classes:
            raise ValueError("Duplicate class ID in PNG mapping")
        classes[label["id"]] = context.label(label["name"], label.get("value"), label["id"])
    for row in body["images"]:
        if row["id"] in images:
            raise ValueError("Duplicate image ID in PNG mapping")
        images[row["id"]] = context.image(row["file_name"], row["id"], row["width"], row["height"])
    by_file = defaultdict(list)
    for row in body["objects"]:
        if row.get("image_id") not in images or row.get("class_id") not in classes:
            raise ValueError("PNG object mapping refers to an unknown image or class")
        if type(row.get("value")) is not int or not 1 <= row["value"] <= 65535:
            raise ValueError("PNG mapped values must be integers in [1,65535]")
        by_file[row["file_name"]].append(row)
    for filename, rows in by_file.items():
        image = images[rows[0]["image_id"]]
        if any(row["image_id"] != rows[0]["image_id"] for row in rows):
            raise ValueError("One PNG mask cannot map to different images")
        array = _read_mask(child(root, filename), image["width"], image["height"])
        if len({row["value"] for row in rows}) != len(rows):
            raise ValueError("Duplicate PNG value mapping")
        permitted = {0, *(row["value"] for row in rows)}
        if not set(np.unique(array).tolist()) <= permitted:
            raise ValueError("PNG contains values absent from mapping metadata")
        for row in rows:
            mask = array == row["value"]
            if not mask.any():
                context.losses.append({"external_id": row.get("annotation_id"), "lossy": True,
                                       "reason": "Mapped instance has no visible pixels in the single-plane PNG"})
                continue
            context.add(image, classes[row["class_id"]], {"type": "mask", "rle": encode_rle(mask)},
                        external_id=row.get("annotation_id"), metadata={"png_variant": body["variant"]})
    if body["variant"] == "semantic":
        context.losses.append({"lossy": True, "reason": "Semantic PNG has class regions; original instance identities are unavailable"})
