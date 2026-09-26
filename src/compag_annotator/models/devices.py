"""Live device discovery inside a provider's interpreter, without loading weights."""
from __future__ import annotations


def inspect_devices():
    import torch
    devices = [{"id": "cpu", "name": "CPU", "kind": "cpu"}]
    error = None
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append({"id": f"cuda:{index}", "name": props.name, "kind": "gpu",
                                "total_memory_bytes": props.total_memory})
    except Exception as exc:
        error = f"GPU discovery failed: {type(exc).__name__}: {exc}"
    return {"devices": devices, "cuda_available": len(devices) > 1,
            "torch_cuda": torch.version.cuda, "torch_version": str(torch.__version__), "error": error,
            "checkpoint_loaded": False}
