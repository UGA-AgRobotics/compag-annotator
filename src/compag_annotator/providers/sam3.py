"""SAM3 image-only interactivity with a local checkpoint and strict key coverage."""
from .common import ExactImageCache, clone_state, device_context, mask_record, read_image, result_envelope, timed_sync, state_bytes
from .protocol import identity, validate_prompts


def strict_sam3_load(model, checkpoint_path):
    """Pinned upstream key translation, with failure instead of warning on gaps.

    The upstream loader uses strict=False and prints missing keys. We require
    all state keys for the selected image+interactive model, including buffers.
    Video-only weights cannot satisfy this contract.
    """
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("SAM3 checkpoint must be a state dictionary")
    if isinstance(checkpoint.get("model"), dict):
        checkpoint = checkpoint["model"]
    state = {k[len("detector."):]: v for k, v in checkpoint.items() if k.startswith("detector.")}
    state.update({"inst_interactive_predictor.model." + k[len("tracker."):]: v
                  for k, v in checkpoint.items() if k.startswith("tracker.")})
    if not state or not any(k.startswith("inst_interactive_predictor.") for k in state):
        raise ValueError("Checkpoint lacks SAM3 image/instance-interactivity weights; SAM3.1/video is not interchangeable")
    required = model.state_dict()
    missing = sorted(set(required) - set(state))
    unexpected = sorted(set(state) - set(required))
    mismatched = [k for k in set(required) & set(state) if required[k].shape != state[k].shape]
    if missing or unexpected or mismatched:
        raise ValueError(f"SAM3 checkpoint incompatible: {len(missing)} missing, {len(unexpected)} unexpected, "
                         f"{len(mismatched)} shape mismatches; keys={(missing+unexpected+mismatched)[:12]}")
    model.load_state_dict(state, strict=True)


class SAM3Provider:
    def __init__(self, record, device, settings):
        if not record.get("trusted"):
            raise PermissionError("Explicit checkpoint trust is required for SAM3")
        if device == "cpu":
            raise ValueError("SAM3 CUDA image runtime has no validated CPU mode")
        from pathlib import Path
        import sam3
        import sam3.model_builder as builder
        from sam3.model.sam3_image_processor import Sam3Processor
        device_context(device, settings)
        if not Path(record["path"]).is_file():
            raise ValueError("SAM3 local weights are missing")
        bpe = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe.is_file():
            raise ValueError("SAM3 tokenizer asset missing; reinstall the pinned runtime")
        previous = builder._load_checkpoint
        builder._load_checkpoint = strict_sam3_load
        try:
            # Passing 'cpu' avoids upstream's device=='cuda' special case; move
            # explicitly to the selected indexed device after verified loading.
            self.model = builder.build_sam3_image_model(bpe_path=str(bpe), checkpoint_path=record["path"],
                load_from_HF=False, enable_inst_interactivity=True, device="cpu", compile=False).to(device).eval()
        finally:
            builder._load_checkpoint = previous
        self.processor = Sam3Processor(self.model, device=device)
        self.cache = ExactImageCache()
        self.state = None
        self.record = record

    def infer(self, request, progress):
        import numpy as np
        import torch
        device, autocast = device_context(request["device"], request["settings"])
        # Decoder threshold changes do not invalidate image embeddings.
        self.model.inst_interactive_predictor.mask_threshold = request["settings"].get("mask_threshold", 0.0)
        self.model.inst_interactive_predictor._transforms.mask_threshold = self.model.inst_interactive_predictor.mask_threshold
        started = timed_sync(device)
        image, pixels = read_image(request)
        width, height = image.size
        points, labels, box = validate_prompts(width, height, request.get("points"), request.get("labels"), request.get("box"))
        binding = identity({"checkpoint": self.record["sha256"], "image": request["image_sha256"],
                            "preprocess": "official_sam3_rgb_1008", "device": device,
                            "precision": request["settings"].get("precision", "float32")})
        hit = self.cache.matches(pixels, binding)
        with torch.inference_mode(), autocast:
            if not hit:
                self.state = self.processor.set_image(image, state=None)
                self.cache.remember(pixels, binding, int(request["settings"].get("cache_image_bytes", 512 * 1024**2)) - state_bytes(self.state))
            encoded = timed_sync(device)
            progress({"stage": "sam_prompt", "encoding_cache_hit": hit})
            masks, scores, _ = self.model.predict_inst(clone_state(self.state),
                point_coords=np.asarray(points, dtype=np.float32) if points else None,
                point_labels=np.asarray(labels, dtype=np.int32) if labels else None,
                box=np.asarray(box, dtype=np.float32) if box else None,
                multimask_output=request["settings"].get("multimask_output", True), return_logits=request["settings"].get("boundary_quality", False))
        ended = timed_sync(device)
        records = []
        for mask, score in zip(masks, scores):
            if mask.shape != (height, width):
                raise RuntimeError("SAM3 returned masks outside original-image coordinates")
            quality = request["settings"].get("boundary_quality", False)
            threshold = request["settings"].get("mask_threshold", 0.0)
            row = mask_record(mask > threshold if quality else mask, request, score)
            if row and quality:
                from .prompt_settings import native_boundary_quality
                row.update(native_boundary_quality(mask, score, threshold))
            if row:
                records.append(row)
        records.sort(key=lambda r: -r["score"])
        if self.cache.pixels is None:
            self.state = None
        return result_envelope(request, width, height, records[:1], alternatives=records[1:],
                               timings={"image_encoding_seconds": encoded-started, "prompt_seconds": ended-encoded},
                               settings=request["settings"], encoding_cache_hit=hit, preprocessing="official_sam3_rgb_1008")

    def unload(self):
        self.cache.clear()
        self.state = self.model = self.processor = None
