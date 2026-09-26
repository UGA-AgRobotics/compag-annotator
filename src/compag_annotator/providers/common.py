"""Worker-only image, geometry and device helpers. ML imports stay inside calls."""
from contextlib import nullcontext
import math
import time

from .protocol import normalize_device, sha256_file


def read_image(request):
    import numpy as np
    from PIL import Image, ImageOps
    path = request["image_path"]
    if sha256_file(path) != request["image_sha256"]:
        raise ValueError("Image changed while inference was queued")
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Multipage images require an explicit frame import")
        limit = int(request.get("settings", {}).get("max_image_pixels", 100_000_000))
        if source.width * source.height > limit:
            raise ValueError("Image exceeds the configured full-image memory limit; no implicit resizing is performed")
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    return image, np.asarray(image)


def device_context(device, settings):
    import torch
    device = normalize_device(device)
    precision = settings.get("precision", "float32")
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("Precision must be float32, float16 or bfloat16")
    if device == "cpu":
        if precision != "float32":
            raise ValueError("CPU provider execution requires float32; no implicit precision fallback")
        return device, nullcontext()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose CPU where supported or repair the GPU runtime")
    index = int(device.split(":")[1])
    if index >= torch.cuda.device_count():
        raise ValueError("Selected CUDA device does not exist")
    torch.cuda.set_device(index)
    reserve = int(settings.get("reserve_free_bytes", 256 * 1024**2))
    if reserve < 0:
        raise ValueError("Memory reserve must be nonnegative")
    free, _ = torch.cuda.mem_get_info(index)
    if free < reserve:
        # Previous tiles can leave almost all device memory in PyTorch's idle
        # allocator cache. Driver-free memory alone would reject the next tile
        # even though that cache is reusable. Reclaim only under pressure, then
        # measure again; reserved-minus-allocated is not all releasable memory.
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(index)
        if free < reserve:
            raise RuntimeError(
                f"Insufficient free GPU memory on cuda:{index}: {free / 1024**2:.0f} MiB free "
                f"after releasing unused cache; {reserve / 1024**2:.0f} MiB required by the memory reserve. "
                "Unload another model or lower explicit settings."
            )
    if precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected CUDA device does not support bfloat16")
    return device, (nullcontext() if precision == "float32" else torch.autocast("cuda", dtype=getattr(torch, precision)))


def mask_record(mask, request, score, model_class_index=None, tile=None):
    import numpy as np
    from compag_annotator.geometry import encode_rle
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Provider returned an invalid mask dimension")
    if not math.isfinite(float(score)):
        raise ValueError("Provider returned a nonfinite score")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    source = {"kind": request["model"]["provider"], "model_id": request["model"]["id"],
              "model_sha256": request["model"]["sha256"]}
    if tile is not None:
        source["tile"] = tile
    result = {"geometry": {"type": "mask", "rle": encode_rle(mask)},
              "bbox": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
              "source": source, "score": float(score), "status": "proposal", "human_verified": False,
              "review_actor": None, "class_id": None}
    if model_class_index is not None:
        result["model_class_index"] = int(model_class_index)
    else:
        result["score_kind"] = "predicted_mask_iou; not a class probability"
    return result


def result_envelope(request, width, height, annotations, *, alternatives=None, timings=None, **extra):
    model = request["model"]
    return {"annotations": annotations, "alternatives": alternatives or [], "model_id": model["id"],
            "model_sha256": model["sha256"], "provider": model["provider"],
            "device": request["device"], "precision": request["settings"].get("precision", "float32"),
            "loaded_checkpoint_sha256": model["sha256"],
            "image_sha256": request["image_sha256"], "coverage": {"width": width, "height": height, "full_image": True},
            "timings": timings or {}, **extra}


class ExactImageCache:
    """One encoded image; exact pixel equality plus checkpoint/preprocessing identity.

    Refactors the A worker's bounded exact-image cache concept. Prompt state is
    deliberately excluded. A hit must satisfy both the hash binding and pixels.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self.pixels = None
        self.binding = None

    def matches(self, pixels, binding):
        import numpy as np
        return (self.binding == binding and self.pixels is not None and pixels.shape == self.pixels.shape
                and pixels.dtype == self.pixels.dtype and np.array_equal(pixels, self.pixels))

    def remember(self, pixels, binding, limit):
        self.clear()
        if pixels.nbytes <= limit:
            self.pixels, self.binding = pixels.copy(), binding
            return True
        return False


def state_bytes(value):
    if isinstance(value, dict):
        return sum(state_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(state_bytes(v) for v in value)
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return value.numel() * value.element_size()
    return getattr(value, "nbytes", 0)


def clone_state(value):
    if isinstance(value, dict):
        return {k: clone_state(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_state(v) for v in value)
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    return value


def timed_sync(device):
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize(device)
    return time.perf_counter()
