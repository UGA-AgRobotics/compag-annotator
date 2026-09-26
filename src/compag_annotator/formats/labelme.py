"""LabelMe polygons, rectangles, flags, and grouped disconnected components."""
import copy
import numpy as np
from shapely.geometry import Polygon

from ..geometry import encode_rle, geometry_mask, validate_geometry
from ..geometry import _pixel_polygon, _loss_report
from ._common import FormatError, read_json, write_json, child, relative


def export(root, classes, images, annotations, manifest, *, allow_lossy=False, **options):
    names = {c["id"]: c["name"] for c in classes}
    if len(set(names.values())) != len(names):
        raise ValueError("LabelMe requires unique class names or explicit class remapping")
    by_image = {im["id"]: [] for im in images}
    for annotation in annotations:
        by_image[annotation["image_id"]].append(annotation)
    for image in images:
        shapes = []
        reserved_groups = {a.get("source", {}).get("labelme", {}).get("group_id")
                           for a in by_image[image["id"]]}
        next_group = 1
        for ordinal, annotation in enumerate(by_image[image["id"]], 1):
            geometry = annotation["geometry"]
            metadata = annotation.get("source", {}).get("labelme", {})
            flags = copy.deepcopy(metadata.get("flags", {}))
            group = metadata.get("group_id")
            entries = []
            if geometry["type"] == "box":
                x0, y0, x1, y1 = geometry["xyxy"]
                entries = [("rectangle", [[x0, y0], [x1, y1]])]
            elif geometry["type"] == "polygon":
                entries = [("polygon", geometry["points"])]
            else:
                mask = geometry_mask(geometry, image["width"], image["height"])
                polygon = _pixel_polygon(mask)
                components = [polygon] if polygon.geom_type == "Polygon" else list(polygon.geoms)
                holes = sum(len(p.interiors) for p in components)
                if holes and not allow_lossy:
                    raise FormatError(f"LabelMe annotation {annotation['id']} has {holes} holes; explicit allow_lossy required",
                                      loss_report=[{"annotation_id": annotation["id"], "holes": holes, "reason": "LabelMe polygon shapes cannot represent holes"}])
                converted = np.zeros_like(mask)
                for component in components:
                    points = [[float(x), float(y)] for x, y in Polygon(component.exterior).exterior.coords[:-1]]
                    converted |= geometry_mask({"type": "polygon", "points": points}, image["width"], image["height"])
                    entries.append(("polygon", points))
                report = {"annotation_id": annotation["id"], **_loss_report(mask, converted),
                          "components": len(components), "holes": holes, "method": "grouped_pixel_edge_polygons"}
                if report["lossy"] and not allow_lossy:
                    raise FormatError("LabelMe pixel conversion requires allow_lossy", loss_report=[report])
                manifest["loss_report"].append(report)
                if len(components) > 1 and group is None:
                    while next_group in reserved_groups:
                        next_group += 1
                    group = next_group
                    reserved_groups.add(group)
            for shape_type, points in entries:
                shapes.append({"label": names[annotation["class_id"]], "points": points,
                               "group_id": group, "shape_type": shape_type, "flags": flags,
                               "description": metadata.get("description", "")})
        # JSON resides with images so imagePath is portable without parent traversal.
        write_json(root / "images" / f"{image['id']}.json",
                   {"version": "5.0.1", "flags": {}, "shapes": shapes, "imagePath": f"{image['id']}.png",
                    "imageData": None, "imageHeight": image["height"], "imageWidth": image["width"]})


def load(path, context):
    root = path if path.is_dir() else path.parent
    files = sorted(root.rglob("*.json")) if path.is_dir() else [path]
    seen_images = set()
    for file in files:
        if file.name in {"manifest.json", "class_map.json", "mapping.json"}:
            continue
        child(root, file.relative_to(root).as_posix())
        body = read_json(file)
        if "shapes" not in body:
            raise ValueError("LabelMe file requires a shapes array")
        name = relative(body.get("imagePath"))
        image = context.image(name, width=body["imageWidth"], height=body["imageHeight"])
        if image["id"] in seen_images:
            raise ValueError("Multiple LabelMe files map to the same image")
        seen_images.add(image["id"])
        groups = {}
        for index, shape in enumerate(body["shapes"]):
            kind, points = shape.get("shape_type", "polygon"), shape.get("points")
            if kind == "polygon":
                geometry = {"type": "polygon", "points": points}
            elif kind == "rectangle":
                if not isinstance(points, list) or len(points) != 2 or any(not isinstance(p, list) or len(p) != 2 for p in points):
                    raise ValueError("LabelMe rectangles require two corners")
                if any(type(value) not in (int, float) for point in points for value in point):
                    raise ValueError("LabelMe rectangle coordinates must be finite numbers")
                array = np.asarray(points, dtype=float)
                geometry = {"type": "box", "xyxy": [*array.min(axis=0).tolist(), *array.max(axis=0).tolist()]}
            else:
                if not context.options.get("allow_lossy", False):
                    raise FormatError(f"Unsupported LabelMe shape: {kind}; enable allow_lossy to skip explicitly")
                context.losses.append({"external_id": f"{file.name}:{index}", "lossy": True,
                                       "reason": f"Unsupported LabelMe {kind} shape skipped"})
                continue
            geometry = validate_geometry(geometry, image["width"], image["height"])
            class_id = context.label(shape["label"])
            group = shape.get("group_id")
            if group is not None and (type(group) not in (int, str)):
                raise ValueError("LabelMe group_id must be an integer, string or null")
            flags = shape.get("flags", {})
            if not isinstance(flags, dict):
                raise ValueError("LabelMe flags must be a mapping")
            key = (class_id, type(group).__name__, group) if group is not None else ("shape", index)
            groups.setdefault(key, []).append((geometry, {"group_id": group, "flags": flags,
                                                         "description": shape.get("description", ""), "shape_index": index}))
        for entries in groups.values():
            metadata = copy.deepcopy(entries[0][1])
            if len(entries) == 1:
                geometry = entries[0][0]
            else:
                mask = np.zeros((image["height"], image["width"]), dtype=bool)
                for geometry, _ in entries:
                    mask |= geometry_mask(geometry, image["width"], image["height"])
                geometry = {"type": "mask", "rle": encode_rle(mask)}
                metadata["shape_flags"] = [entry[1]["flags"] for entry in entries]
                if any(entry[1]["flags"] != metadata["flags"] for entry in entries):
                    context.losses.append({"lossy": True, "reason": "Grouped shapes retain per-shape flags in provenance; later mask edits/export may merge flags"})
            class_id = context.label(body["shapes"][entries[0][1]["shape_index"]]["label"])
            context.add(image, class_id, geometry, external_id=f"{file.name}:{metadata['shape_index']}", metadata={"labelme": metadata})
