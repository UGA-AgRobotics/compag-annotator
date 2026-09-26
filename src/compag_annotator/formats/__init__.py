"""Local annotation exchange. Native project archives belong to the core.

Exports are built in a private temporary directory, then promoted atomically.
The destination must be missing or empty. Formats never run ML or mutate a
project. Imports return proposals, never fabricated human review.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from xml.etree.ElementTree import ParseError

from defusedxml.common import DefusedXmlException
from yaml import YAMLError

from . import coco, labelme, png, voc, yolo
from ._common import (FORMATS, FormatError, ImportContext, base_manifest,
                      copy_images, prepare_document, write_json)

ADAPTERS = {"coco": coco, "yolo_seg": yolo, "yolo_box": yolo,
            "voc": voc, "labelme": labelme, "png": png}


def export_annotations(format_name, document, output_dir, *, include_images=True,
                       allow_lossy=False, png_variant="per_instance", conflict_policy="error"):
    if format_name not in FORMATS:
        raise ValueError("Unsupported exchange format; native archives use project backup")
    classes, images, annotations = prepare_document(document)
    destination = Path(output_dir).absolute()
    if destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
        raise ValueError("Export destination must not contain symlinks")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Export destination must be missing or empty; existing files are never overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".compag-export-", dir=destination.parent))
    try:
        manifest = base_manifest(format_name, classes, images, annotations, include_images, allow_lossy)
        ADAPTERS[format_name].export(stage, classes, images, annotations, manifest,
                                    format_name=format_name, allow_lossy=allow_lossy,
                                    png_variant=png_variant, conflict_policy=conflict_policy)
        if include_images:
            copy_images(stage, images)
        manifest["files"] = sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()) + ["manifest.json"]
        manifest["lossy"] = any(row.get("lossy", False) for row in manifest["loss_report"])
        write_json(stage / "manifest.json", manifest)
        # POSIX rename refuses to replace nonempty directories if a concurrent writer appeared.
        os.rename(stage, destination)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def import_annotations(format_name, path, images, classes, *, options=None):
    if format_name not in FORMATS:
        raise ValueError("Unsupported exchange format; native archives use project restore")
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Import paths must not contain symlinks")
    if not path.exists():
        raise ValueError("Annotation import path does not exist")
    context = ImportContext(images, classes, options, format_name)
    try:
        ADAPTERS[format_name].load(path, context)
        return context.result()
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(f"Malformed {format_name} annotation structure") from exc
    except (DefusedXmlException, ParseError, YAMLError) as exc:
        raise ValueError(f"Unsafe or malformed {format_name} annotation document") from exc


__all__ = ["export_annotations", "import_annotations", "FormatError"]
