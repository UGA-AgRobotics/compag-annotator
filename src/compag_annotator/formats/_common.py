"""Shared validation and exchange helpers (no provider or project imports)."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
from PIL import Image, ImageOps

from ..geometry import geometry_mask, validate_geometry
from ..geometry import _dimensions, _loss_report

MAX_TEXT_BYTES = 64 * 1024 * 1024
FORMATS = {"coco", "yolo_seg", "yolo_box", "voc", "labelme", "png"}


class FormatError(ValueError):
    """A conversion failure whose report can be presented in the UI."""
    def __init__(self, message, *, loss_report=None):
        super().__init__(message)
        self.loss_report = loss_report or []


def identifier(value):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("IDs used in exchange filenames must be portable identifiers")
    return value


def relative(value):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError("Expected a safe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError("Absolute paths and traversal are forbidden in annotation files")
    return path.as_posix()


def child(root, value):
    path = Path(root) / relative(value)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Annotation path escapes the import/export folder")
    if any(p.is_symlink() for p in (path, *path.parents) if p != Path(root).parent):
        raise ValueError("Symlinks are not supported in annotation packages")
    return path


def read_text(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_TEXT_BYTES:
        raise ValueError("Annotation file is a symlink or exceeds the 64 MiB limit")
    return path.read_text(encoding="utf-8-sig")


def read_json(path):
    def invalid_constant(value):
        raise ValueError("Non-finite JSON values are forbidden")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(read_text(path), parse_constant=invalid_constant, object_pairs_hook=unique_pairs)


def write_json(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_document(document):
    if not isinstance(document, dict) or document.get("schema_version", 1) != 1:
        raise ValueError("Unsupported document schema")
    classes = copy.deepcopy(document.get("classes", []))
    images = copy.deepcopy(document.get("images", []))
    annotations = copy.deepcopy(document.get("annotations", []))
    for label in classes:
        identifier(label["id"])
        if not isinstance(label.get("name"), str) or not label["name"].strip():
            raise ValueError("Classes require nonempty names")
    if len({str(c["id"]) for c in classes}) != len(classes):
        raise ValueError("Duplicate class IDs")
    classes.sort(key=lambda c: c.get("order", 0))
    class_ids = {c["id"] for c in classes}
    for image in images:
        identifier(image["id"])
        _dimensions(image["width"], image["height"])
        if not isinstance(image.get("name"), str):
            raise ValueError("Images require a name")
    by_image = {image["id"]: image for image in images}
    if len(by_image) != len(images):
        raise ValueError("Duplicate image IDs")
    if len({str(a["id"]) for a in annotations}) != len(annotations):
        raise ValueError("Duplicate annotation IDs")
    for annotation in annotations:
        identifier(annotation["id"])
        if annotation.get("image_id") not in by_image:
            raise ValueError("Annotation references an unknown image")
        if annotation.get("class_id") not in class_ids:
            raise ValueError(f"Annotation {annotation['id']} has an unknown or unassigned class")
        image = by_image[annotation["image_id"]]
        annotation["geometry"] = validate_geometry(annotation["geometry"], image["width"], image["height"])
    return classes, images, annotations


def base_manifest(format_name, classes, images, annotations, include_images, allow_lossy):
    return {"schema_version": 1, "format": format_name, "include_images": bool(include_images),
            "allow_lossy": bool(allow_lossy), "annotation_count": len(annotations),
            "image_count": len(images),
            "class_mapping": [{"index": i, "class_id": c["id"], "name": c["name"]}
                              for i, c in enumerate(classes)],
            "images": [{"id": im["id"], "name": Path(im["name"]).name,
                        "file_name": f"images/{im['id']}.png", "width": im["width"], "height": im["height"]}
                       for im in images],
            "review_scope": [{"id": a["id"], "status": a.get("status", "draft"),
                              "human_verified": a.get("human_verified") is True,
                              "review_actor": a.get("review_actor")} for a in annotations],
            "loss_report": [], "warnings": ["Exchange formats do not preserve full project history; use native backup for that."]}


def copy_images(root, images):
    (root / "images").mkdir(exist_ok=True)
    for image in images:
        if not image.get("path"):
            raise ValueError(f"Image {image['id']} has no readable source path")
        with Image.open(image["path"]) as opened:
            if getattr(opened, "n_frames", 1) != 1 or opened.mode not in {"1", "L", "LA", "P", "RGB", "RGBA"}:
                raise ValueError("Exchange image copies require single-frame 8-bit images")
            oriented = ImageOps.exif_transpose(opened)
            if oriented.size != (image["width"], image["height"]):
                raise ValueError(f"Image {image['id']} oriented dimensions do not match geometry")
            # Portable lossless display pixels; native backups preserve raw originals.
            oriented.convert("RGBA" if "A" in oriented.getbands() else "RGB").save(root / "images" / f"{image['id']}.png")


def conversion_loss(annotation, image, converted, reason):
    before = geometry_mask(annotation["geometry"], image["width"], image["height"])
    after = geometry_mask(converted, image["width"], image["height"])
    return {"annotation_id": annotation["id"], **_loss_report(before, after),
            "lossy": True, "reason": reason, "geometry_type_loss": True}


class ImportContext:
    def __init__(self, images, classes, options, format_name):
        self.images = copy.deepcopy(list(images))
        self.classes = copy.deepcopy(list(classes))
        self.options = options or {}
        self.format = format_name
        self.annotations = []
        self.losses = []
        if len({c["id"] for c in self.classes}) != len(self.classes):
            raise ValueError("Duplicate target class IDs")
        if len({im["id"] for im in self.images}) != len(self.images):
            raise ValueError("Duplicate target image IDs")
        for im in self.images:
            _dimensions(im["width"], im["height"])

    def image(self, name=None, external_id=None, width=None, height=None):
        if name is not None:
            relative(name)
        if width is not None or height is not None:
            _dimensions(width, height)
        mapping = self.options.get("image_mapping", {})
        explicit = mapping.get(str(external_id), mapping.get(name))
        candidates = []
        if explicit is not None:
            candidates = [im for im in self.images if im["id"] == explicit]
        else:
            if external_id is not None:
                candidates = [im for im in self.images if str(im["id"]) == str(external_id)]
            if not candidates and name:
                name = relative(name)
                candidates = [im for im in self.images if im.get("name") == name or f"images/{im['id']}.png" == name]
                if not candidates:
                    basename = PurePosixPath(name).name
                    candidates = [im for im in self.images if Path(im.get("name", "")).name == basename or f"{im['id']}.png" == basename]
                if not candidates:
                    stem = PurePosixPath(name).stem
                    candidates = [im for im in self.images if str(im["id"]) == stem or Path(im.get("name", "")).stem == stem]
        if len(candidates) != 1:
            raise ValueError("Image mapping is missing or ambiguous; provide options.image_mapping")
        image = candidates[0]
        if width is not None and (width != image["width"] or height != image["height"]):
            raise ValueError("Imported dimensions do not match the oriented image")
        return image

    def label(self, name, external_id=None, stable_id=None):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Imported class name must be nonempty")
        mapping = self.options.get("class_mapping", {})
        explicit = (mapping[str(external_id)] if external_id is not None and str(external_id) in mapping
                    else mapping[str(stable_id)] if stable_id is not None and str(stable_id) in mapping
                    else mapping.get(name))
        if explicit is not None:
            candidates = [c for c in self.classes if c["id"] == explicit]
        else:
            candidates = [c for c in self.classes if stable_id is not None and c["id"] == stable_id]
            if not candidates:
                candidates = [c for c in self.classes if c["name"] == name]
        if len(candidates) > 1 or explicit is not None and not candidates:
            raise ValueError("Class mapping is invalid or ambiguous")
        if not candidates:
            if self.options.get("create_classes", True) is not True:
                raise ValueError(f"Unknown imported class: {name}")
            class_id = str(uuid5(NAMESPACE_URL, "compag-import-class:" + name))
            if any(c["id"] == class_id for c in self.classes):
                class_id = str(uuid4())
            candidate = {"id": class_id, "name": name, "color": "#4f86f7", "shortcut": None,
                         "archived": False, "order": len(self.classes)}
            self.classes.append(candidate)
            candidates = [candidate]
        return candidates[0]["id"]

    def add(self, image, class_id, geometry, *, external_id=None, metadata=None):
        geometry = validate_geometry(geometry, image["width"], image["height"])
        source = {"kind": "import", "format": self.format}
        if external_id is not None:
            source["external_id"] = str(external_id)
        if metadata:
            source.update(metadata)
        annotation = {"id": str(uuid4()), "image_id": image["id"], "class_id": class_id,
                      "geometry": geometry, "status": "proposal", "revision": 0,
                      "source": source, "human_verified": False, "review_actor": None}
        self.annotations.append(annotation)
        return annotation

    def result(self):
        return {"annotations": self.annotations, "classes": self.classes, "loss_report": self.losses}
