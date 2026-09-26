"""The shared YOLO export used by exchange and segmentation training snapshots."""
import math
import re

import yaml

from ..geometry import geometry_bbox, polygon_for_yolo, validate_yolo_rows
from ._common import FormatError, conversion_loss, read_text, read_json, child, write_json


def export(root, classes, images, annotations, manifest, *, format_name, allow_lossy=False, **options):
    by_image = {image["id"]: image for image in images}
    class_indices = {label["id"]: i for i, label in enumerate(classes)}
    lines = {image["id"]: [] for image in images}
    reports, failures = [], []
    for annotation in annotations:
        image = by_image[annotation["image_id"]]
        width, height = image["width"], image["height"]
        geometry = annotation["geometry"]
        try:
            if format_name == "yolo_seg":
                points, report = polygon_for_yolo(geometry, width, height, allow_lossy=allow_lossy)
                values = [v / (width if axis == 0 else height) for point in points for axis, v in enumerate(point)]
                reports.append({"annotation_id": annotation["id"], **report})
            else:
                x0, y0, x1, y1 = geometry_bbox(geometry, width, height)
                values = [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height), (x1 - x0) / width, (y1 - y0) / height]
                if geometry["type"] != "box":
                    report = conversion_loss(annotation, image, {"type": "box", "xyxy": [x0, y0, x1, y1]}, "Detection boxes do not represent segmentation")
                    reports.append(report)
                    if not allow_lossy:
                        raise ValueError("Segmentation-to-box conversion requires explicit allow_lossy")
            # 17 digits avoid introducing extra coordinate loss during serialization.
            lines[image["id"]].append(str(class_indices[annotation["class_id"]]) + " " + " ".join(format(v, ".17g") for v in values))
        except ValueError as exc:
            failures.append({"annotation_id": annotation["id"], "reason": str(exc)})
    if format_name == "yolo_seg":
        for image_id, rows in lines.items():
            try:
                validate_yolo_rows(rows)
            except ValueError as exc:
                failures.append({"annotation_id": ",".join(a["id"] for a in annotations if a["image_id"] == image_id),
                                 "image_id": image_id, "reason": str(exc)})
    if failures:
        raise FormatError("YOLO export blocked for annotations: " + "; ".join(f"{r['annotation_id']}: {r['reason']}" for r in failures), loss_report=failures)
    manifest["loss_report"].extend(reports)
    (root / "labels").mkdir()
    for iid, rows in lines.items():
        (root / "labels" / f"{iid}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    split_images = {"train": [], "val": [], "test": []}
    for image in images:
        split = {"validation": "val", "test": "test"}.get(image.get("role"), "train")
        split_images[split].append(f"./images/{image['id']}.png")
    dataset = {"names": {i: label["name"] for i, label in enumerate(classes)}, "nc": len(classes)}
    for split, filenames in split_images.items():
        (root / f"{split}.txt").write_text("\n".join(filenames) + ("\n" if filenames else ""), encoding="utf-8")
        dataset[split] = f"{split}.txt"
    # No 'path' field: recent Ultralytics resolves relative entries from YAML's parent.
    (root / "dataset.yaml").write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(root / "class_map.json", {"schema_version": 1, "classes": manifest["class_mapping"]})
    manifest["dataset_yaml"] = "dataset.yaml"
    manifest["split_counts"] = {k: len(v) for k, v in split_images.items()}
    manifest["warnings"].append("YOLO training rasterization/resampling may differ from COCO pixel-edge rasterization; reported conversion IoU is at original resolution using COCO.")
    if not split_images["val"]:
        manifest["warnings"].append("No validation images; training coordinator must enforce its validation policy.")


def _class_mapping(root, context, yaml_path=None):
    if yaml_path is None:
        candidates = [root / name for name in ("dataset.yaml", "data.yaml", "dataset.yml", "data.yml") if (root / name).is_file()]
        if len(candidates) > 1:
            raise ValueError("Multiple YOLO dataset YAML files; import the intended YAML file explicitly")
        yaml_path = candidates[0] if candidates else None
    if (root / "class_map.json").is_file():
        rows = read_json(root / "class_map.json")["classes"]
        if [row["index"] for row in rows] != list(range(len(rows))):
            raise ValueError("Invalid YOLO class map indices")
        if yaml_path is not None:
            data = yaml.safe_load(read_text(yaml_path))
            if not isinstance(data, dict):
                raise ValueError("YOLO dataset YAML must be a mapping")
            names = data.get("names")
            expected = {row["index"]: row["name"] for row in rows}
            if (dict(enumerate(names)) if isinstance(names, list) else names) != expected:
                raise ValueError("YOLO class_map.json and dataset.yaml disagree")
        return {row["index"]: context.label(row["name"], row["index"], row.get("class_id")) for row in rows}
    if yaml_path is not None:
        data = yaml.safe_load(read_text(yaml_path))
        if not isinstance(data, dict):
            raise ValueError("YOLO dataset YAML must be a mapping")
        names = data.get("names")
        if isinstance(names, list):
            names = dict(enumerate(names))
        if not isinstance(names, dict) or any(type(k) is not int for k in names) or sorted(names) != list(range(len(names))):
            raise ValueError("YOLO names must define contiguous class indices from zero")
        # Never execute a YAML download directive or follow arbitrary dataset paths.
        if "download" in data:
            context.losses.append({"lossy": False, "reason": "YAML download directive ignored; imports never execute scripts"})
        return {index: context.label(name, index) for index, name in names.items()}
    supplied = context.options.get("class_names")
    if supplied is not None:
        if not isinstance(supplied, list):
            raise ValueError("YOLO class_names must be an ordered list")
        return {i: context.label(name, i) for i, name in enumerate(supplied)}
    if not context.classes:
        raise ValueError("YOLO requires dataset.yaml, class_map.json or explicit class_names")
    return {i: context.label(c["name"], i, c["id"]) for i, c in enumerate(list(context.classes))}


def load(path, context):
    root = path if path.is_dir() else path.parent
    if root.name == "labels" and any((root.parent / name).is_file() for name in ("dataset.yaml", "data.yaml", "dataset.yml", "data.yml", "class_map.json")):
        root = root.parent
    mapping = _class_mapping(root, context, path if path.is_file() and path.suffix.lower() in {".yaml", ".yml"} else None)
    labels = root / "labels" if (root / "labels").is_dir() else root
    if path.is_file() and path.suffix == ".txt":
        files = [path]
    else:
        files = sorted(labels.rglob("*.txt"))
        if labels == root:
            files = [p for p in files if p.name not in {"train.txt", "val.txt", "test.txt", "classes.txt"}]
    seen = set()
    for file in files:
        child(root, file.relative_to(root).as_posix())
        image = context.image(file.name)
        if image["id"] in seen:
            raise ValueError("Multiple YOLO label files map to the same image")
        seen.add(image["id"])
        width, height = image["width"], image["height"]
        for line_number, line in enumerate(read_text(file).splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if not re.fullmatch(r"0|[1-9][0-9]*", fields[0]):
                raise ValueError("YOLO class index must be a nonnegative integer")
            index = int(fields[0])
            if index not in mapping:
                raise ValueError("YOLO class index is absent from its class map")
            try:
                values = list(map(float, fields[1:]))
            except ValueError as exc:
                raise ValueError("Malformed YOLO coordinates") from exc
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError("YOLO coordinates must be finite and normalized to [0,1]")
            if context.format == "yolo_seg":
                if len(values) < 6 or len(values) % 2:
                    raise ValueError("YOLO segmentation requires at least three points; detection rows are not masks")
                geometry = {"type": "polygon", "points": [[values[i] * width, values[i + 1] * height] for i in range(0, len(values), 2)]}
            else:
                if len(values) != 4:
                    raise ValueError("YOLO detection requires center-x center-y width height")
                x, y, w, h = values
                coords = [(x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height]
                # Undo floating roundoff only (not an out-of-bounds label repair).
                coords = [0.0 if -1e-9 < v < 0 else float(limit) if limit < v < limit + 1e-9 else v
                          for v, limit in zip(coords, [width, height, width, height])]
                geometry = {"type": "box", "xyxy": coords}
            context.add(image, mapping[index], geometry, external_id=f"{file.name}:{line_number}")
