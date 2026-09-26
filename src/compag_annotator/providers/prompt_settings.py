"""Shared settings for the validated SAM2/SAM3 point and box APIs."""
from copy import deepcopy
from .automatic import AMG_CONTROLS, _EXECUTION_CONTROLS, ProposalSettingsError
import math

PROMPT_DEFAULTS = {"multimask_output": True, "mask_threshold": 0.0, "precision": "float32"}

def prompt_settings(settings=None):
    if settings is not None and not isinstance(settings, dict):
        raise ProposalSettingsError({"settings": "Expected an object"})
    effective = {**PROMPT_DEFAULTS, **(settings or {})}
    controls = {**{k: AMG_CONTROLS[k] for k in PROMPT_DEFAULTS}, **_EXECUTION_CONTROLS}
    controls["boundary_quality"] = {"type": "boolean"}
    errors = {}
    for key, value in effective.items():
        rule = controls.get(key)
        if rule is None:
            errors[key] = "Unsupported point/box setting; automatic grid/IoU/NMS controls are SAM2 generation only"
            continue
        kind = rule["type"]
        valid = (type(value) is bool if kind == "boolean" else value in rule["values"] if kind == "enum" else
                 (type(value) is int if kind == "integer" else type(value) in (int, float)) and
                 rule["min"] <= value <= rule["max"] and math.isfinite(value))
        if not valid:
            errors[key] = f"Expected {kind} within {rule}"
    if errors:
        raise ProposalSettingsError(errors)
    return deepcopy(effective)


def native_boundary_quality(logits, predicted_iou, threshold):
    """Native interactive-head IoU and stability at threshold +/- 1 logit."""
    import numpy as np
    high=int(np.count_nonzero(logits > threshold+1.0))
    low=int(np.count_nonzero(logits > threshold-1.0))
    return {'predicted_iou':float(predicted_iou),'stability_score':high/low if low else 0.0,
            'quality_source':'native_interactive_iou_and_full_resolution_logits',
            'stability_offset':1.0}
