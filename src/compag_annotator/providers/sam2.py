"""Pinned official SAM2 image predictor. No optional CUDA mask cleanup is used."""
from .common import ExactImageCache, device_context, mask_record, read_image, result_envelope, timed_sync, state_bytes
from .protocol import identity, validate_prompts
from .automatic import proposal_settings, validate_crop_box, amg_filter_metadata


def generation_settings(settings=None):
    """Pure recipe validation: no ML imports; 64 points/side, 512/batch by default (COMPAG paper).

    Smoke checks must explicitly pass points_per_side=8. Additional internal
    crop layers and topology-changing cleanup are unsupported in this profile.
    """
    return proposal_settings(settings)


class SAM2Provider:
    def __init__(self, record, device, settings):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from compag_annotator.models.catalog import get_entry
        device_context(device, settings)
        self.record = record
        self.model = build_sam2(get_entry("sam2", record["architecture"])["config"],
                               ckpt_path=record["path"], device=device, apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(self.model, max_hole_area=0, max_sprinkle_area=0)
        self.cache = ExactImageCache()

    def infer(self, request, progress):
        import numpy as np
        import torch
        settings = request["settings"]
        device, autocast = device_context(request["device"], settings)
        self.predictor.mask_threshold = settings.get("mask_threshold", 0.0)
        self.predictor._transforms.mask_threshold = self.predictor.mask_threshold
        started = timed_sync(device)
        image, pixels = read_image(request)
        width, height = image.size
        points, labels, box = validate_prompts(width, height, request.get("points"), request.get("labels"), request.get("box"))
        binding = identity({"checkpoint": self.record["sha256"], "image": request["image_sha256"],
                            "preprocess": "official_sam2_rgb_1024_no_cleanup", "device": device,
                            "precision": settings.get("precision", "float32")})
        hit = self.cache.matches(pixels, binding)
        with torch.inference_mode(), autocast:
            if not hit:
                self.predictor.set_image(pixels)
                self.cache.remember(pixels, binding, int(settings.get("cache_image_bytes", 512 * 1024**2)) - state_bytes(self.predictor._features))
            encoded = timed_sync(device)
            progress({"stage": "sam_prompt", "encoding_cache_hit": hit})
            masks, scores, _ = self.predictor.predict(
                point_coords=np.asarray(points, dtype=np.float32) if points else None,
                point_labels=np.asarray(labels, dtype=np.int32) if labels else None,
                box=np.asarray(box, dtype=np.float32) if box else None,
                multimask_output=settings.get("multimask_output", True), return_logits=settings.get("boundary_quality", False),
                normalize_coords=True)
        ended = timed_sync(device)
        records = []
        for mask, score in zip(masks, scores):
            if mask.shape != (height, width):
                raise RuntimeError("SAM2 output is not in original-image coordinates")
            quality = settings.get("boundary_quality", False)
            threshold = settings.get("mask_threshold", 0.0)
            row = mask_record(mask > threshold if quality else mask, request, score)
            if row and quality:
                from .prompt_settings import native_boundary_quality
                row.update(native_boundary_quality(mask, score, threshold))
            if row is not None:
                records.append(row)
        records.sort(key=lambda r: -r["score"])
        if self.cache.pixels is None:
            self.predictor.reset_predictor()
        return result_envelope(request, width, height, records[:1], alternatives=records[1:],
                               timings={"image_encoding_seconds": encoded-started, "prompt_seconds": ended-encoded},
                               settings=request["settings"], encoding_cache_hit=hit, preprocessing="official_sam2_rgb_1024_no_cleanup",
                               limitations=["CUDA connected-component hole/sprinkle cleanup is disabled"])

    def generate_proposals(self, request, progress):
        """Run the pinned official AMG on one canonical image/crop, without classes.

        generate() returns RLE directly; only one candidate is decoded at a time
        for exact mask metadata. The coordinator owns any cross-tile mapping.
        """
        import math
        import numpy as np
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from compag_annotator.geometry import decode_rle

        settings = generation_settings(request["settings"])
        request = {**request, "settings": settings}
        device, autocast = device_context(request["device"], settings)
        started = timed_sync(device)
        image, pixels = read_image(request)
        original_width, original_height = image.size
        crop_box = validate_crop_box(request.get("crop_box"), original_width, original_height)
        x0, y0, x1, y1 = crop_box or [0, 0, original_width, original_height]
        pixels = np.ascontiguousarray(pixels[y0:y1, x0:x1])
        width, height = x1-x0, y1-y0
        encoder_size = int(self.model.image_size)
        transform = {"crop_box": [x0, y0, x1, y1], "original_width": original_width,
                     "original_height": original_height, "coordinate_space": "crop_local",
                     "valid_box": [0, 0, width, height], "padding": [0, 0, 0, 0],
                     "encoder_size": encoder_size, "encoder_resize": [encoder_size, encoder_size],
                     "encoder_tensor_shape": [1, 3, encoder_size, encoder_size],
                     "encoder_scale_xy": [encoder_size / width, encoder_size / height],
                     "encoder_padding": [0, 0, 0, 0], "resize_mode": "bilinear_square_resize",
                     "output_to_image": {"translation_xy": [x0, y0], "scale_xy": [1, 1]},
                     "orientation": "EXIF_transposed_RGB"}
        options = {key: settings[key] for key in (
            "points_per_side", "points_per_batch", "pred_iou_thresh", "stability_score_thresh",
            "stability_score_offset", "mask_threshold", "box_nms_thresh", "crop_n_layers",
            "min_mask_region_area", "multimask_output")}
        options.update(output_mode="uncompressed_rle", use_m2m=False,
                       crop_nms_thresh=.7, crop_overlap_ratio=.40, crop_n_points_downscale_factor=2)
        # AMG owns a separate predictor but shares the already verified model.
        # Release interactive image features before allocating a generation crop.
        self.cache.clear()
        self.predictor.reset_predictor()
        generator = SAM2AutomaticMaskGenerator(self.model, **options)
        prepared = timed_sync(device)
        work = {"input_crops": 1, "internal_crops": 1, "points": settings["points_per_side"]**2,
                "prompt_batches": math.ceil(settings["points_per_side"]**2 / settings["points_per_batch"]),
                "masks_per_point": 3 if settings["multimask_output"] else 1}
        progress({"stage": "sam2_automatic_generation", "completed_crops": 0, "total_crops": 1,
                  "crop_box": [x0, y0, x1, y1], "work": work})
        try:
            with torch.inference_mode(), autocast:
                candidates = generator.generate(pixels)
            generated = timed_sync(device)
        finally:
            generator.predictor.reset_predictor()
        records, empty = [], 0
        for candidate in candidates:
            mask = decode_rle(candidate["segmentation"])
            if mask.shape != (height, width):
                raise RuntimeError("SAM2 automatic output is not in the requested crop coordinates")
            area = int(np.count_nonzero(mask))
            if area != candidate["area"]:
                raise RuntimeError("SAM2 automatic mask area disagrees with its exact RLE")
            row = mask_record(mask, request, candidate["predicted_iou"])
            if row is None:
                empty += 1
                continue
            stability = float(candidate["stability_score"])
            if not math.isfinite(stability):
                raise RuntimeError("SAM2 automatic generator returned a nonfinite stability score")
            row.update(area=area, quality={"predicted_iou": float(candidate["predicted_iou"]),
                                          "stability_score": stability,
                                          "point_coords": candidate["point_coords"],
                                          "provider_bbox_xywh": candidate["bbox"],
                                          "internal_crop_box_xywh": candidate["crop_box"]})
            row["source"].update(operation="generate_proposals", transform=transform,
                                 settings_hash=identity(settings))
            records.append(row)
        ended = timed_sync(device)
        progress({"stage": "sam2_automatic_generation", "completed_crops": 1, "total_crops": 1,
                  "proposals": len(records), "provider_candidates": len(candidates)})
        return result_envelope(
            request, width, height, records,
            coverage={"width": width, "height": height, "full_image": crop_box is None},
            timings={"image_preparation_seconds": prepared-started, "generation_seconds": generated-prepared,
                     "geometry_seconds": ended-generated, "recovery_seconds": 0.0},
            transform=transform, settings=settings, settings_hash=identity(settings), generator_options=options,
            filters=amg_filter_metadata(settings), work=work,
            proposal_count=len(records), provider_candidates=len(candidates), empty_masks_removed=empty,
            encoding_cache_hit=False, preprocessing="official_sam2_amg_rgb_square_resize_no_cleanup",
            limitations=["Quality scores are not class probabilities; generated masks require review.",
                         "AMG retains its upstream quality, box-NMS and internal crop-edge filters.",
                         "No cross-tile suppression, topology cleanup or heavy boundary recovery is performed.",
                         "Automatic image encodings are not cached across calls; the verified model is reused."])

    def unload(self):
        self.cache.clear()
        self.predictor.reset_predictor()
        self.predictor = self.model = None
