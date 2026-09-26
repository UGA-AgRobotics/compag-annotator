"""Real Ultralytics segmentation training and original-coordinate inference."""
from pathlib import Path
import math
import os
import time

from .common import device_context, mask_record, read_image, result_envelope, timed_sync
from .protocol import sha256_file


def validate_tiled_prediction(settings, width, height, image_sha256):
    """Validate the additive tile plan independently of historical tile_size."""
    from compag_annotator.geometry import plan_tiles
    if not isinstance(settings["tiling"], dict):
        raise ValueError("tiling must be a processing configuration object")
    if any(key in settings for key in ("tile_size", "overlap", "deduplicate_iou")):
        raise ValueError("Choose settings.tiling or historical tile_size/overlap settings; cross-tile deduplicate_iou is unsupported for a new plan")
    for name, default, limit in (("imgsz", 640, 4096), ("max_det", 300, 10000)):
        value = settings.get(name, default)
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"{name} must be an integer from 1 to {limit}")
    if settings.get("imgsz", 640) % 32:
        raise ValueError("YOLO imgsz must be divisible by 32; source crop width/height have no such restriction")
    for name, default in (("confidence", .25), ("iou", .7)):
        value = settings.get(name, default)
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite number in [0,1]")
    plan=plan_tiles(width, height, settings["tiling"], image_sha256=image_sha256)
    if settings.get('tile_padding'):
        from compag_annotator.training.tiles import padded_plan, TILING
        expected=plan_tiles(width,height,TILING,image_sha256=image_sha256)
        if settings['tile_padding']!='constant_bottom_right_pixel' or plan['config']!=expected['config'] or settings.get('imgsz')!=512:
            raise ValueError('The trained padding recipe requires 512px crops, stride 512 and Image size 512')
        plan=padded_plan(width,height,image_sha256)
    if 'stitching' in settings:
        s=settings['stitching']
        if not isinstance(s,dict) or s.get('method')!='class_aware_bbox_nms' or type(s.get('iou')) not in (int,float) or not 0<s['iou']<=1:
            raise ValueError('Choose class_aware_bbox_nms with finite IoU in (0,1]')
    return plan


def recording_predictor_class():
    """Observe actual preprocessing without changing pinned predictor behavior.

    LetterBox.get_params is the pinned upstream parameter calculator. Tensor
    dimensions come from the real preprocess result, not from user crop sizes.
    """
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.data.augment import LetterBox

    class RecordingSegmentationPredictor(SegmentationPredictor):
        def pre_transform(self, images):
            transformed = super().pre_transform(images)
            same_shapes = len({value.shape for value in images}) == 1
            letterbox = LetterBox(self.imgsz, auto=same_shapes and self.args.rect and
                                  (self.model.format == "pt" or (getattr(self.model, "dynamic", False)
                                                               and self.model.format != "imx")),
                                  stride=self.model.stride)
            self.compag_transforms = []
            for original, result in zip(images, transformed):
                params = letterbox.get_params({"img": original})
                resized_width, resized_height = map(int, params["new_unpad"])
                left, top, right, bottom = [int(params[key]) for key in ("left", "top", "right", "bottom")]
                if result.shape[:2] != (resized_height + top + bottom, resized_width + left + right):
                    raise RuntimeError("YOLO observed letterbox shape differs from the pinned transform parameters")
                self.compag_transforms.append({
                    "source_size": [int(original.shape[1]), int(original.shape[0])],
                    "resized_size": [resized_width, resized_height], "padding": [left, top, right, bottom],
                    "scale_xy": list(map(float, params["ratio"])), "interpolation": "opencv_INTER_LINEAR",
                    "normalization": "RGB uint8 divided by 255"})
            return transformed

        def preprocess(self, images):
            tensor = super().preprocess(images)
            for transform in self.compag_transforms:
                transform["tensor_shape"] = list(map(int, tensor.shape))
            return tensor

    return RecordingSegmentationPredictor


def configure_ultralytics():
    import shutil
    import matplotlib
    from ultralytics import settings
    from ultralytics.utils import callbacks, USER_CONFIG_DIR
    disabled = {k: False for k in ("sync", "hub", "wandb", "clearml", "comet", "dvc", "mlflow",
                                   "neptune", "raytune", "tensorboard", "platform") if k in settings}
    if disabled:
        settings.update(disabled)
    # Only local, default progress/checkpoint callbacks are used. No cloud SDKs.
    callbacks.add_integration_callbacks = lambda instance: None
    # Upstream font helper otherwise downloads Arial during dataset validation.
    font = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("Arial.ttf", "Arial.Unicode.ttf"):
        target = USER_CONFIG_DIR / name
        if not target.exists():
            shutil.copyfile(font, target)


def local_dataset(dataset_yaml):
    """Preflight the shared exporter's YAML; never execute YAML download hooks."""
    import yaml
    path = Path(dataset_yaml).resolve(strict=True)
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or "download" in data:
        raise ValueError("Dataset YAML must be a local snapshot without download scripts")
    root = Path(data.get("path") or path.parent)
    if not root.is_absolute():
        root = path.parent / root
    root = root.resolve(strict=True)
    names = data.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise ValueError("Dataset requires contiguous class names")
    names = {int(k): v for k, v in names.items()}
    if sorted(names) != list(range(len(names))) or any(not isinstance(v, str) or not v.strip() for v in names.values()):
        raise ValueError("Dataset class indices must be contiguous from zero")
    inventory = {}
    for split in ("train", "val"):
        values = data.get(split)
        if not values:
            raise ValueError("Supply a separate validation split; training-only mode is not enabled")
        values = values if isinstance(values, list) else [values]
        images = []
        for value in values:
            if not isinstance(value, str) or "://" in value:
                raise ValueError("Dataset split paths must be local")
            entry = Path(value)
            if not entry.is_absolute():
                entry = root / entry
            entry = entry.resolve(strict=True)
            if entry.is_dir():
                images.extend(p for p in entry.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
            elif entry.suffix.lower() == ".txt":
                for line in entry.read_text().splitlines():
                    if not line.strip():
                        continue
                    if "://" in line:
                        raise ValueError("Image lists may not contain URLs")
                    item = Path(line.strip())
                    images.append((entry.parent / item if not item.is_absolute() else item).resolve(strict=True))
            else:
                raise ValueError("Split must be an image directory or image-list text file")
        images = sorted(set(p.resolve() for p in images))
        if not images:
            raise ValueError(f"The {split} split has no images")
        inventory[split] = images
    if set(inventory["train"]) & set(inventory["val"]):
        raise ValueError("Training and validation paths overlap")
    train_hashes = {sha256_file(p) for p in inventory["train"]}
    if train_hashes & {sha256_file(p) for p in inventory["val"]}:
        raise ValueError("Training and validation contain duplicate image bytes")
    instances = {"train": 0, "val": 0}
    for split, images in inventory.items():
        for image in images:
            # Match Ultralytics img2label_paths, including its last /images/ rule.
            image_text = str(image)
            marker = os.sep + "images" + os.sep
            if marker in image_text:
                label_text = (os.sep + "labels" + os.sep).join(image_text.rsplit(marker, 1))
            else:
                label_text = image_text
            labels = Path(label_text).with_suffix(".txt")
            if not labels.is_file():
                raise ValueError(f"Missing label file for {image.name}; reviewed negatives need an explicit empty file")
            for line_number, line in enumerate(labels.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) < 7 or len(fields) % 2 != 1:
                    raise ValueError(f"Label {labels.name}:{line_number} is not a segmentation polygon (box-only targets are invalid)")
                values = list(map(float, fields))
                if any(not math.isfinite(v) for v in values) or not values[0].is_integer() or int(values[0]) not in names:
                    raise ValueError("Invalid segmentation class index or nonfinite coordinates")
                if any(not 0 <= v <= 1 for v in values[1:]):
                    raise ValueError("Segmentation polygon coordinates must be normalized to [0,1]")
                instances[split] += 1
    if not instances["train"]:
        raise ValueError("Training requires at least one segmentation instance")
    return path, names, inventory


def train_settings(settings):
    allowed = {"epochs", "imgsz", "batch", "device", "seed", "workers", "patience", "lr0", "optimizer",
               "rect", "cache", "close_mosaic", "mosaic", "mixup", "cutmix", "copy_paste", "scale", "translate",
               "fliplr", "flipud", "degrees", "shear", "perspective", "deterministic", "amp"}
    metadata = {"class_mapping", "qa_smoke", "round_id", "dataset_id", "worker_timeout", "full_instance", "resume",
                "resume_checkpoint", "precision", "reserve_free_bytes", "class_version", "snapshot_sha256", "training_layout"}
    unknown = set(settings) - allowed - metadata
    if unknown:
        raise ValueError(f"Unsupported training settings: {sorted(unknown)}")
    result = {"epochs": 10, "imgsz": 640, "batch": 1, "device": "cpu", "seed": 0,
              "workers": 0, "amp": False, "deterministic": True, "cache": False,
              "plots": False, "save": True, "val": True, "verbose": False}
    result.update({k: v for k, v in settings.items() if k in allowed})
    if settings.get('training_layout',{}).get('mode')=='tiles_512' and result['imgsz']!=512:
        raise ValueError('Tile-trained models require Image size 512')
    for key in ("epochs", "imgsz", "batch"):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer; automatic memory probing is not enabled")
    if result["imgsz"] % 32:
        raise ValueError("Image size must be divisible by 32")
    if result["amp"]:
        raise ValueError("Automatic mixed-precision checks can request extra weights; use amp=False")
    if settings.get("full_instance"):
        result.update(mosaic=0, mixup=0, cutmix=0, copy_paste=0, scale=0, translate=0, degrees=0, shear=0, perspective=0)
    return result


def bind_training_location(trainer, output_dir, dataset_yaml):
    """Run after upstream check_resume and before BaseTrainer creates any files."""
    output_dir = Path(output_dir).resolve()
    trainer.args.project = str(output_dir)
    trainer.args.name = "run"
    trainer.args.save_dir = str(output_dir / "run")
    trainer.args.exist_ok = False
    trainer.args.data = str(Path(dataset_yaml).resolve())


def validate_resume_state(checkpoint, options):
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch", -1) < 0 or checkpoint.get("optimizer") is None:
        raise ValueError("Checkpoint is not resumable: optimizer/epoch state is absent. Start a new fine-tuning job instead")
    original = checkpoint.get("train_args", {})
    original_epochs = original.get("epochs")
    if not isinstance(original_epochs, int) or checkpoint["epoch"] + 1 >= original_epochs:
        raise ValueError("Checkpoint training already finished; start a new fine-tuning job instead of resume")
    if options["epochs"] != original_epochs:
        raise ValueError(f"Resume must retain the original {original_epochs} total epochs; use fine-tuning to change the schedule")
    if original.get("amp", False):
        raise ValueError("Resume of externally mixed-precision training is not supported by the offline preset")


class YOLOProvider:
    def __init__(self, record, device, settings):
        if not record.get("trusted"):
            raise PermissionError("Loading a serialized YOLO checkpoint requires explicit trust")
        from ultralytics import YOLO
        configure_ultralytics()
        device_context(device, settings)
        self.record = record
        self.model = YOLO(record["path"], task="segment")
        if self.model.task != "segment" or getattr(self.model.model, "task", "segment") != "segment":
            raise ValueError("This checkpoint is not a segmentation model")
        # Task=segment must not disguise a detect-only head.
        head = self.model.model.model[-1]
        if not hasattr(head, "proto"):
            raise ValueError("Checkpoint has no instance-segmentation mask head")

    def infer(self, request, progress):
        import numpy as np
        from compag_annotator.geometry import tile_boxes, deduplicate
        settings = request["settings"]
        device, _ = device_context(request["device"], settings)
        if settings.get("precision", "float32") not in {"float32", "float16"}:
            raise ValueError("YOLO inference supports float32 or float16")
        started = timed_sync(device)
        image, pixels = read_image(request)
        width, height = image.size
        plan = validate_tiled_prediction(settings, width, height, request["image_sha256"]) if "tiling" in settings else None
        tiled = plan["config"]["mode"] == "tiled" if plan else bool(settings.get("tile_size"))
        if plan:
            from compag_annotator.geometry import map_tile_geometry, mask_metadata
            tiles = [tile["box"] for tile in plan["tiles"]]
            # Only the new path adds observation; historical jobs keep their API.
            self.model.predictor = None
            predictor_class = recording_predictor_class()
        else:
            tiles = tile_boxes(width, height, int(settings["tile_size"]), int(settings.get("overlap", 128))) if tiled else [[0, 0, width, height]]
        proposals = []
        tile_receipts = []
        for index, tile in enumerate(tiles):
            x0, y0, x1, y1 = tile
            progress({"stage": "yolo_inference", "tile": index + 1, "tiles": len(tiles)})
            if settings.get('tile_padding'):
                from compag_annotator.training.tiles import padded_crop
                crop=padded_crop(image,(x0,y0,x1,y1))
            else:
                crop = image.crop((x0, y0, x1, y1))
            output = self.model.predict(source=crop, device=device, imgsz=int(settings.get("imgsz", 640)),
                                        conf=float(settings.get("confidence", .25)), iou=float(settings.get("iou", .7)),
                                        max_det=int(settings.get("max_det", 300)), retina_masks=True,
                                        half=settings.get("precision", "float32") == "float16",
                                        save=False, verbose=False, augment=False,
                                        **({"predictor": predictor_class} if plan else {}))[0]
            if plan:
                transforms = getattr(self.model.predictor, "compag_transforms", None)
                if not transforms or len(transforms) != 1:
                    raise RuntimeError("YOLO did not report the actual crop preprocessing transform")
                transform = {**transforms[0], "crop_box": tile, "original_width": width,
                             "original_height": height, "output_coordinate_space": "canonical_full_image",
                             "output_to_image": {"translation_xy": [x0, y0], "scale_xy": [1, 1]}}
                tile_receipts.append({**plan["tiles"][index], "transform": transform,
                                      "proposals_returned": len(output.boxes)})
            if output.masks is None:
                if len(output.boxes):
                    raise RuntimeError("Segmentation model returned detections without masks")
                continue
            masks = output.masks.data.cpu().numpy()
            classes = output.boxes.cls.cpu().numpy()
            scores = output.boxes.conf.cpu().numpy()
            if len(masks) != len(classes):
                raise RuntimeError("YOLO mask and class counts differ")
            for mask, cls, score in zip(masks, classes, scores):
                if mask.shape != (crop.height, crop.width):
                    raise RuntimeError("YOLO retina mask differs from original crop dimensions")
                if plan:
                    if not (mask[:y1-y0,:x1-x0] > .5).any():
                        continue  # Padding-only foreground has no original-image object.
                    row = mask_record(mask > .5, request, score, int(cls), tile)
                    if row:
                        row["geometry"] = map_tile_geometry(row["geometry"], plan["tiles"][index], width, height)
                        row.update(mask_metadata(row["geometry"], width, height))
                        row["source"].update(tile_id=plan["tiles"][index]["id"], plan_hash=plan["plan_hash"],
                                             transform=transform)
                else:
                    full = np.zeros((height, width), dtype=bool)
                    full[y0:y1, x0:x1] = mask > .5
                    row = mask_record(full, request, score, int(cls), tile if tiled else None)
                if row:
                    proposals.append(row)
            progress({"stage": "yolo_inference", "completed_tiles": index+1, "total_tiles": len(tiles),
                      "proposals": len(proposals)})
        if tiled and plan is None:
            # Group model indices explicitly: class_id remains unassigned.
            grouped = {}
            for row in proposals:
                grouped.setdefault(row["model_class_index"], []).append(row)
            proposals = [row for rows in grouped.values() for row in deduplicate(rows, width, height, iou=float(settings.get("deduplicate_iou", .8)))]
        suppression=None
        if plan and settings.get('stitching'):
            from compag_annotator.geometry.stitching import stitch_predictions
            proposals,suppression=stitch_predictions(proposals,settings['stitching']['iou'])
        ended = timed_sync(device)
        names = {str(k): v for k, v in self.model.names.items()}
        return result_envelope(request, width, height, proposals, timings={"inference_seconds": ended-started},
                               class_names=names, preprocessing="official_yolo_letterbox_retina_masks", tiles=tiles,
                               settings=settings, **({"tile_plan": plan, "tile_receipts": tile_receipts,
                                   "filters": {"cross_tile_suppression": bool(suppression), "automatic_union": False,
                                               "backend_per_crop_filtering": "Pinned Ultralytics confidence/IoU/max_det postprocessing",
                                               "confidence": settings.get("confidence", .25),
                                               "iou": settings.get("iou", .7), "max_det": settings.get("max_det", 300),
                                               "suppression_counts": suppression}} if plan else {}))

    def train(self, request, progress):
        path, expected_names, inventory = local_dataset(request["dataset_yaml"])
        options = train_settings(request["settings"])
        output_dir = Path(request["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / "run"
        if run_dir.exists():
            raise ValueError("Training output already exists; select a new job output folder, including for resume")
        resume = bool(request["settings"].get("resume"))
        if resume:
            validate_resume_state(self.model.ckpt, options)
            expected = (output_dir / "resume-input" / "last.pt").resolve()
            if Path(self.record["path"]).resolve() != expected:
                raise ValueError("Resume requires the manager's isolated checkpoint copy")
        from .training_updates import training_batch_policy, TrainingUpdateAudit, require_training_updates
        from compag_annotator.storage.files import atomic
        options, policy = training_batch_policy(options, len(inventory["train"]), resume=resume,
            checkpoint_args=self.model.ckpt.get("train_args", {}) if resume else None)
        from ultralytics.models.yolo.segment.train import SegmentationTrainer

        class BoundSegmentationTrainer(SegmentationTrainer):
            def check_resume(self, overrides):
                super().check_resume(overrides)
                bind_training_location(self, output_dir, path)

            def _setup_train(self):
                super()._setup_train()
                self.compag_updates = TrainingUpdateAudit(self.optimizer, self.model)
                self.compag_training_policy = {**policy, "effective_batch": self.batch_size,
                    "effective_nbs": self.args.nbs, "batches_per_epoch": len(self.train_loader),
                    "warmup_epochs": self.args.warmup_epochs,
                    "warmup_iterations": self._get_warmup_iterations(len(self.train_loader))}
                progress({"stage": "training_configuration", "optimization": self.compag_training_policy,
                          "setup_label": "Training configured; actual weight updates will be checked."})

        def epoch(trainer):
            metrics = {str(k): float(v) for k, v in (getattr(trainer, "metrics", {}) or {}).items()
                       if isinstance(v, (float, int)) and math.isfinite(float(v))}
            progress({"stage": "training", "epoch": min(trainer.epoch + 1, trainer.epochs), "epochs": trainer.epochs, "metrics": metrics,
                      "final_validation": trainer.epoch >= trainer.epochs,
                      "optimization": trainer.compag_updates.counts(),
                      "last_checkpoint": str(trainer.last) if Path(trainer.last).is_file() else None})

        self.model.add_callback("on_fit_epoch_end", epoch)
        self.model.add_callback("on_train_start", lambda trainer: progress({"stage": "training_started", "epochs": trainer.epochs}))
        start = time.monotonic()
        try:
            metrics = self.model.train(trainer=BoundSegmentationTrainer, data=str(path), project=str(output_dir), name="run",
                                       save_dir=str(run_dir), exist_ok=False,
                                       resume=self.record["path"] if resume else False, **options)
            trainer = self.model.trainer
            optimization = {**trainer.compag_training_policy, **trainer.compag_updates.finish()}
            atomic(output_dir / "optimization.json", optimization)
            progress({"stage": "training_updates_checked", "optimization": optimization})
            require_training_updates(optimization)
        finally:
            audit = getattr(getattr(self.model, "trainer", None), "compag_updates", None)
            if audit is not None:
                audit.close()
        trainer = self.model.trainer
        checkpoint = Path(trainer.best)
        basis = "best_validation_checkpoint"
        if not checkpoint.is_file():
            checkpoint, basis = Path(trainer.last), "last_checkpoint; best absent"
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1024:
            raise RuntimeError("Training did not produce a valid checkpoint file")
        actual_names = {int(k): v for k, v in self.model.names.items()}
        if actual_names != expected_names:
            raise RuntimeError("Saved model class names differ from the training snapshot")
        metric_values = getattr(metrics, "results_dict", {}) if metrics is not None else {}
        metric_values = {str(k): float(v) for k, v in metric_values.items() if math.isfinite(float(v))}
        progress({"stage": "training_checkpoint_saved", "checkpoint": str(checkpoint)})
        return {"checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                "task": "segmentation", "class_names": {str(k): v for k, v in actual_names.items()},
                "metrics": metric_values, "selection_basis": basis,
                "probe_image": str(inventory["val"][0]), "training_seconds": time.monotonic()-start,
                "last_checkpoint": str(Path(trainer.last).resolve()), "settings": options,
                "optimization": optimization,
                "resume": resume, "resume_input_sha256": self.record["sha256"] if resume else None}

    def unload(self):
        self.model = None
