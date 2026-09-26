"""Torch-free automatic-proposal settings and crop contract.

The coordinator owns external tiling, durable tile results and mapping to the
image. A provider call processes exactly one whole image or canonical crop.
"""
from copy import deepcopy
import math

SAM3_AUTOMATIC_REASON = (
    "SAM3 automatic class-agnostic generation is unsupported in this adapter: "
    "the pinned image integration implements point/box interactivity, and no "
    "automatic point-grid adapter has been implemented and independently tested. "
    "Use SAM2 Generate masks; SAM3 local-weight point/box assistance remains available."
)

AMG_DEFAULTS = {
    "points_per_side": 64, "points_per_batch": 512, "pred_iou_thresh": .8,
    "stability_score_thresh": .88, "stability_score_offset": 1.0,
    "box_nms_thresh": .7, "crop_n_layers": 0, "min_mask_region_area": 0,
    "mask_threshold": 0.0, "multimask_output": True, "precision": "float32",
}
AMG_CONTROLS = {
    "points_per_side": {"type": "integer", "min": 1, "max": 64},
    "points_per_batch": {"type": "integer", "min": 1, "max": 512},
    "pred_iou_thresh": {"type": "number", "min": 0, "max": 1},
    "stability_score_thresh": {"type": "number", "min": 0, "max": 1},
    "stability_score_offset": {"type": "number", "min": 0, "max": 10},
    "box_nms_thresh": {"type": "number", "min": 0, "max": 1},
    "mask_threshold": {"type": "number", "min": -32, "max": 32},
    "crop_n_layers": {"type": "integer", "min": 0, "max": 0,
                      "reason": "Additional internal crops require a separately measured advanced profile; use the shared external tile plan."},
    "min_mask_region_area": {"type": "integer", "min": 0, "max": 0,
                             "reason": "Topology-changing cleanup is disabled; the optional CUDA cleanup extension is not built."},
    "multimask_output": {"type": "boolean"},
    "precision": {"type": "enum", "values": ["float32", "float16", "bfloat16"]},
}
_EXECUTION_CONTROLS = {
    "max_image_pixels": {"type": "integer", "min": 1, "max": 100_000_000},
    "reserve_free_bytes": {"type": "integer", "min": 0, "max": 2**50},
    "cache_image_bytes": {"type": "integer", "min": 0, "max": 2**50},
    "worker_timeout": {"type": "number", "min": .1, "max": 86400},
}


class ProposalSettingsError(ValueError):
    """Validation before worker/model allocation, with actionable field errors."""
    def __init__(self, fields):
        self.fields = fields
        super().__init__("; ".join(f"{key}: {value}" for key, value in fields.items()))


def automatic_capability(provider):
    if provider != "sam2":
        return {"supported": False, "status": "unsupported",
                "reason": SAM3_AUTOMATIC_REASON if provider == "sam3" else
                "Class-agnostic SAM generation requires SAM2; YOLO uses class-aware prediction."}
    return {"supported": True, "status": "implemented_pending_real_automatic_acceptance",
            "method": "SAM2AutomaticMaskGenerator.generate", "requires_classes": False,
            "requires_yolo": False, "coordinates": "crop_local",
            "defaults": deepcopy(AMG_DEFAULTS), "controls": deepcopy(AMG_CONTROLS),
            "presets": {"smoke": {"points_per_side": 8}, "normal": {"points_per_side": 16}, "paper": {"points_per_side": 64, "points_per_batch": 512, "pred_iou_thresh": .8, "stability_score_thresh": .88}},
            "internal_crop_layers_supported": [0], "topology_cleanup_supported": False,
            "quality_is_class_probability": False,
            "help": "Masks are proposals requiring review; no preset guarantees complete object recall."}


def proposal_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ProposalSettingsError({"settings": "Expected an object"})
    effective = {**AMG_DEFAULTS, **(settings or {})}
    controls = {**AMG_CONTROLS, **_EXECUTION_CONTROLS}
    errors = {}
    for key, value in effective.items():
        rule = controls.get(key)
        if rule is None:
            errors[key] = "Unsupported automatic-generation setting"
            continue
        kind = rule["type"]
        if kind == "boolean":
            valid = type(value) is bool
        elif kind == "enum":
            valid = value in rule["values"]
        else:
            valid = (type(value) is int if kind == "integer" else type(value) in (int, float))
            valid = valid and rule["min"] <= value <= rule["max"] and math.isfinite(value)
        if not valid:
            errors[key] = rule.get("reason") or f"Expected {kind} within {rule}"
    if errors:
        raise ProposalSettingsError(errors)
    return deepcopy(effective)


def validate_crop_box(crop_box, width=None, height=None):
    if crop_box is None:
        return None
    if not isinstance(crop_box, (list, tuple)) or len(crop_box) != 4 or any(type(v) is not int for v in crop_box):
        raise ProposalSettingsError({"crop_box": "Expected four integer canonical pixel edges [x0,y0,x1,y1]"})
    x0, y0, x1, y1 = crop_box
    if not (0 <= x0 < x1 and 0 <= y0 < y1) or (width is not None and x1 > width) or (height is not None and y1 > height):
        raise ProposalSettingsError({"crop_box": "Crop must be nonempty and entirely inside the canonical image"})
    return list(crop_box)


def canonical_image_size(path, settings):
    """Read dimensions without allocating an image or importing an ML runtime."""
    from PIL import Image
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Multipage images require an explicit frame import")
        width, height = source.size
        if width * height > settings.get("max_image_pixels", 100_000_000):
            raise ValueError("Image exceeds the configured full-image memory limit; no implicit resizing is performed")
        if source.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
    return width, height


def amg_filter_metadata(settings):
    return {
        "predicted_iou": {"threshold": settings["pred_iou_thresh"], "comparison": ">", "enabled": settings["pred_iou_thresh"] > 0},
        "stability": {"threshold": settings["stability_score_thresh"], "comparison": ">=", "enabled": settings["stability_score_thresh"] > 0},
        "box_nms": {"iou_threshold": settings["box_nms_thresh"], "scope": "within this provider input crop", "ranking": "predicted_mask_iou"},
        "internal_crop_edge": {"upstream_enabled": True, "tolerance_pixels": 20,
                               "effect": "With crop_n_layers=0, the internal crop equals the input image; input-edge masks are exempt."},
        "application_tile_edge_suppression": False, "cross_tile_suppression": False,
        "topology_cleanup": False, "heavy_boundary_recovery": False,
        "suppression_counts": None, "counts_reason": "The official generate API does not expose per-filter removal counts.",
    }
