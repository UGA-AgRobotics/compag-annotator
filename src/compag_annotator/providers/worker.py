"""Standalone child entry point. Protocol stdout is separate from upstream logs.

Invoked by an explicitly selected Python with -I. Blocking network here is a
privacy guard, not a sandbox against a malicious trusted pickle.
"""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

# The core is intentionally not installed in the optional runtime. Resolve only
# this installed package's code, never the working directory or PYTHONPATH.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from compag_annotator.providers.protocol import PROTOCOL_VERSION, identity, sha256_file


def deny_network(event, args):
    if event == "socket.getaddrinfo":
        raise RuntimeError("Provider workers are offline; install/download assets through Model Manager")
    if event == "socket.connect":
        address = args[1]
        if isinstance(address, tuple):
            raise RuntimeError("Network access is disabled in provider workers")


def probe(provider):
    from compag_annotator.models.catalog import PROVIDERS
    import compag_annotator.geometry
    import torch
    import torchvision
    spec = PROVIDERS[provider]
    if provider == "sam2":
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    elif provider == "sam3":
        import inspect
        import sam3
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model.sam3_image import Sam3Image
        expected = {"checkpoint_path", "load_from_HF", "enable_inst_interactivity"}
        if not expected <= set(inspect.signature(build_sam3_image_model).parameters):
            raise RuntimeError("SAM3 runtime lacks the pinned local interactive image API")
        if not (Path(sam3.__file__).parent / "assets/bpe_simple_vocab_16e6.txt.gz").is_file():
            raise RuntimeError("SAM3 vocabulary asset missing")
        if not callable(getattr(Sam3Processor, "set_image", None)) or not callable(getattr(Sam3Image, "predict_inst", None)):
            raise RuntimeError("SAM3 image processor/interactive prediction API missing")
    else:
        from ultralytics import YOLO
    # CUDA availability reports do not initialize a model or perform a GPU job.
    return {"provider": provider, "python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "distribution_version": importlib.metadata.version(spec["distribution"]),
            "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
            "cuda_visibility": "disabled_by_environment" if __import__("os").environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"} else "default_or_selected",
            "scope": "still_image_interactivity" if provider == "sam3" else "provider_runtime",
            "real_checkpoint_tested": False}


def runtime_identity(provider):
    from compag_annotator.models.catalog import PROVIDERS
    spec = PROVIDERS[provider]
    distribution = importlib.metadata.distribution(spec["distribution"])
    raw = distribution.read_text("direct_url.json")
    direct = json.loads(raw) if raw else {}
    source_matches = spec["source_revision"] in direct.get("url", "")
    if provider == "yolo":
        source_matches = distribution.version == spec["version"]
    import torch
    import torchvision
    return {"source_revision": spec["source_revision"] if source_matches else "external_runtime_unverified",
            "distribution_version": distribution.version, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "python": sys.version.split()[0]}


class Worker:
    def __init__(self):
        self.adapter = None
        self.binding = None

    def release(self):
        if self.adapter:
            self.adapter.unload()
        self.adapter, self.binding = None, None
        import gc
        gc.collect()
        if "torch" in sys.modules:
            import torch
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()

    def execute(self, request, progress):
        if request["protocol"] != PROTOCOL_VERSION:
            raise ValueError("Worker protocol version mismatch")
        operation = request["operation"]
        if operation == "devices":
            from compag_annotator.models.devices import inspect_devices
            return inspect_devices()
        if operation == "probe":
            return probe(request["provider"])
        if operation == "unload":
            self.release()
            return {"state": "unloaded"}
        if operation not in {"infer", "train", "generate_proposals"}:
            raise ValueError("Unsupported worker operation")
        record = request["model"]
        if operation == "generate_proposals":
            from compag_annotator.providers.automatic import automatic_capability, proposal_settings, canonical_image_size, validate_crop_box
            capability = automatic_capability(record["provider"])
            if not capability["supported"]:
                raise ValueError(capability["reason"])
            request = {**request, "settings": proposal_settings(request["settings"])}
            width, height = canonical_image_size(request["image_path"], request["settings"])
            validate_crop_box(request.get("crop_box"), width, height)
        if operation == "infer" and record["provider"] in {"sam2", "sam3"}:
            from compag_annotator.providers.prompt_settings import prompt_settings
            request = {**request, "settings": prompt_settings(request["settings"])}
        if not record.get("trusted"):
            raise PermissionError("Explicit trust is required before loading a checkpoint")
        digest = sha256_file(record["path"])
        if digest != record["sha256"]:
            raise ValueError("Checkpoint bytes changed since registration; register and approve it again")
        binding = identity({"sha256": digest, "path": record["path"], "architecture": record["architecture"],
                            "provider": record["provider"], "device": request["device"],
                            "precision": request["settings"].get("precision", "float32")})
        start = time.perf_counter()
        cold = self.binding != binding
        if cold:
            self.release()
            progress({"stage": "model_loading", "provider": record["provider"], "model_sha256": digest})
            if record["provider"] == "sam2":
                from compag_annotator.providers.sam2 import SAM2Provider as Adapter
            elif record["provider"] == "sam3":
                from compag_annotator.providers.sam3 import SAM3Provider as Adapter
            elif record["provider"] == "yolo":
                from compag_annotator.providers.yolo import YOLOProvider as Adapter
            else:
                raise ValueError("Unsupported provider")
            self.adapter = Adapter(record, request["device"], request["settings"])
            self.binding = binding
        load_seconds = time.perf_counter() - start
        if operation == "train":
            if record["provider"] != "yolo":
                raise ValueError("Training is available only for YOLO segmentation")
            result = self.adapter.train(request, progress)
            result["runtime"] = runtime_identity(record["provider"])
            return result
        result = (self.adapter.generate_proposals(request, progress) if operation == "generate_proposals"
                  else self.adapter.infer(request, progress))
        if sha256_file(record["path"]) != digest or sha256_file(request["image_path"]) != request["image_sha256"]:
            self.release()
            raise ValueError("Checkpoint or image changed during inference; result discarded")
        result["timings"].update(load_seconds=load_seconds, cold_model_load=cold,
                                  worker_total_seconds=time.perf_counter()-start)
        result["loaded_checkpoint_sha256"] = digest
        result["runtime"] = runtime_identity(record["provider"])
        if request["device"].startswith("cuda"):
            import torch
            free, total = torch.cuda.mem_get_info(request["device"])
            result["memory"] = {"device_free_bytes": free, "device_total_bytes": total,
                                "allocated_bytes": torch.cuda.memory_allocated(request["device"]),
                                "reserved_bytes": torch.cuda.memory_reserved(request["device"])}
        return result


def main():
    protocol_output = sys.stdout
    sys.addaudithook(deny_network)
    worker = Worker()
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            request_id = request["request_id"]

            def send(kind, **value):
                protocol_output.write(json.dumps({"protocol": PROTOCOL_VERSION, "request_id": request_id,
                                                  "type": kind, **value}, allow_nan=False) + "\n")
                protocol_output.flush()

            with contextlib.redirect_stdout(sys.stderr):
                result = worker.execute(request, lambda event: send("progress", data=event))
            send("result", data=result)
        except Exception as exc:
            worker.release()
            protocol_output.write(json.dumps({"protocol": PROTOCOL_VERSION, "request_id": request.get("request_id"),
                                               "type": "error", "message": f"{type(exc).__name__}: {exc}"}) + "\n")
            protocol_output.flush()
    worker.release()


if __name__ == "__main__":
    main()
