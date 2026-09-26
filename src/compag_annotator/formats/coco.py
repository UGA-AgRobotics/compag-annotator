"""COCO instance JSON with exact RLE and explicit non-crowd instances."""
import numpy as np
from pycocotools import mask as coco_mask

from ..geometry import decode_rle, encode_rle, geometry_mask, validate_geometry
from ._common import read_json, relative, write_json


def export(root, classes, images, annotations, manifest, **options):
    image_ids = {image["id"]: i + 1 for i, image in enumerate(images)}
    category_ids = {label["id"]: i + 1 for i, label in enumerate(classes)}
    by_image = {image["id"]: image for image in images}
    body = {"info": {"description": "COMPAG Annotator COCO instance export", "version": "1"},
            "images": [{"id": image_ids[im["id"]], "file_name": f"images/{im['id']}.png",
                        "width": im["width"], "height": im["height"], "compag_id": im["id"]} for im in images],
            "categories": [{"id": category_ids[c["id"]], "name": c["name"], "supercategory": "",
                            "compag_id": c["id"]} for c in classes], "annotations": []}
    for index, annotation in enumerate(annotations):
        image = by_image[annotation["image_id"]]
        geometry = annotation["geometry"]
        row = {"id": index + 1, "image_id": image_ids[annotation["image_id"]],
               "category_id": category_ids[annotation["class_id"]], "compag_id": annotation["id"],
               "iscrowd": int(annotation.get("source", {}).get("iscrowd", 0))}
        if row["iscrowd"] not in (0, 1):
            raise ValueError("COCO iscrowd must be 0 or 1")
        if geometry["type"] == "box":
            x0, y0, x1, y1 = geometry["xyxy"]
            row.update(bbox=[x0, y0, x1 - x0, y1 - y0], area=(x1 - x0) * (y1 - y0), segmentation=[])
        else:
            raster = geometry_mask(geometry, image["width"], image["height"])
            if not raster.any():
                raise ValueError(f"Annotation {annotation['id']} has no raster foreground")
            compressed = coco_mask.encode(np.asfortranarray(raster, dtype=np.uint8))
            row["bbox"] = coco_mask.toBbox(compressed).tolist()
            row["area"] = int(coco_mask.area(compressed))
            if geometry["type"] == "polygon" and not row["iscrowd"]:
                row["segmentation"] = [np.asarray(geometry["points"]).ravel().tolist()]
            else:
                row["segmentation"] = {"size": compressed["size"], "counts": compressed["counts"].decode("ascii")}
        body["annotations"].append(row)
    write_json(root / "annotations.json", body)


def load(path, context):
    path = path / "annotations.json" if path.is_dir() else path
    body = read_json(path)
    categories, images = {}, {}
    for row in body.get("categories", []):
        key = row["id"]
        if type(key) is not int or key in categories:
            raise ValueError("COCO category IDs must be unique integers")
        categories[key] = context.label(row["name"], key, row.get("compag_id"))
    for row in body.get("images", []):
        key = row["id"]
        if type(key) is not int or key in images:
            raise ValueError("COCO image IDs must be unique integers")
        relative(row["file_name"])
        external_id = key if str(key) in context.options.get("image_mapping", {}) else row.get("compag_id", key)
        images[key] = context.image(row["file_name"], external_id, row["width"], row["height"])
    seen = set()
    for row in body.get("annotations", []):
        key = row.get("id")
        if type(key) is not int or key in seen:
            raise ValueError("COCO annotation IDs must be unique integers")
        seen.add(key)
        if row.get("image_id") not in images or row.get("category_id") not in categories:
            raise ValueError("COCO annotation references an unknown image or category")
        if type(row.get("iscrowd", 0)) is not int or row.get("iscrowd", 0) not in (0, 1):
            raise ValueError("Invalid COCO crowd flag")
        image = images[row["image_id"]]
        width, height = image["width"], image["height"]
        segmentation = row.get("segmentation")
        if isinstance(segmentation, dict):
            mask = decode_rle(segmentation)
            if mask.shape != (height, width):
                raise ValueError("COCO RLE dimensions disagree with image dimensions")
            geometry = {"type": "mask", "rle": encode_rle(mask)}
        elif isinstance(segmentation, list) and segmentation:
            polygons = []
            for points in segmentation:
                if not isinstance(points, list) or len(points) < 6 or len(points) % 2:
                    raise ValueError("Malformed COCO polygon")
                polygon = validate_geometry({"type": "polygon", "points": [points[i:i + 2] for i in range(0, len(points), 2)]}, width, height)
                polygons.append(polygon)
            if len(polygons) == 1:
                geometry = polygons[0]
            else:
                raster = np.zeros((height, width), dtype=bool)
                for polygon in polygons:
                    raster |= geometry_mask(polygon, width, height)
                geometry = {"type": "mask", "rle": encode_rle(raster)}
        elif segmentation is None or segmentation == []:
            bbox = row.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4 or any(type(v) not in (int, float) for v in bbox):
                raise ValueError("COCO annotation without segmentation requires a bbox")
            x, y, w, h = bbox
            geometry = {"type": "box", "xyxy": [x, y, x + w, y + h]}
        else:
            raise ValueError("Unsupported COCO segmentation representation")
        annotation = context.add(image, categories[row["category_id"]], geometry, external_id=key,
                                 metadata={"iscrowd": row.get("iscrowd", 0)})
        # Recompute derived fields; never trust stale area/bbox in third-party files.
        if geometry["type"] != "box":
            raster = geometry_mask(geometry, width, height)
            actual_area = int(raster.sum())
            if "area" in row and row["area"] != actual_area:
                context.losses.append({"annotation_id": annotation["id"], "lossy": False,
                                       "reason": "COCO area recomputed from segmentation", "source_area": row["area"], "area": actual_area})
