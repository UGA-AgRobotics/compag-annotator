"""Authoritative ModelManager API, with no eager Torch/provider imports."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
import weakref
import threading
from copy import deepcopy

from .catalog import PROVIDERS, catalog as provider_catalog, get_entry
from .naming import dated_model
from .download import download_checkpoint, validate_checkpoint_file
from .files import file_lock, write_json
from .runtime import install_runtime, runtime_python
from ..providers.process import WorkerClient
from ..providers.automatic import automatic_capability, canonical_image_size, proposal_settings, validate_crop_box
from ..providers.protocol import ProviderError, check_cancel, emit, normalize_device, sha256_file

_MANAGERS = weakref.WeakSet()


def _close_all():
    for manager in list(_MANAGERS):
        manager._close_worker()


atexit.register(_close_all)


class ModelManager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.data_dir / "models.json"
        self._worker = None
        self._worker_key = None
        self._state = "idle"
        self._error = None
        self._device_cache = {}
        self._device_probe_lock = threading.Lock()
        # Same user, all managers/data roots: memory-heavy operations serialize.
        self._gpu_lock = Path(tempfile.gettempdir()) / f"compag-annotator-{os.getuid()}" / "gpu.lock"
        _MANAGERS.add(self)

    def catalog(self) -> list:
        return provider_catalog()

    def _read(self):
        if not self.registry_path.exists():
            return {"schema_version": 1, "models": []}
        data = json.loads(self.registry_path.read_text())
        if data.get("schema_version") != 1 or not isinstance(data.get("models"), list):
            raise ValueError("Unsupported or corrupt model registry")
        return data

    def models(self) -> list:
        rows = [dated_model(row) for row in self._read()["models"]]
        for row in rows:
            row["missing"] = not Path(row["path"]).is_file()
        return rows

    def _model(self, model_id):
        for row in self.models():
            if row["id"] == model_id:
                if row["missing"]:
                    raise ValueError("Checkpoint is missing; register its new location")
                return row
        raise ValueError("Unknown model id")

    def _save_record(self, record):
        with file_lock(self.data_dir / "models.lock"):
            data = self._read()
            data["models"] = [r for r in data["models"] if r["id"] != record["id"]] + [record]
            write_json(self.registry_path, data)

    def register(self, provider, path, *, architecture=None, trust=False, class_mapping=None) -> dict:
        if type(trust) is not bool:
            raise ValueError("Checkpoint trust must be an explicit boolean")
        if provider not in PROVIDERS:
            raise ValueError("Unknown provider")
        path = Path(path).expanduser().resolve(strict=True)
        if path.is_dir():
            # Only one supported canonical SAM3 image checkpoint; never search a
            # directory recursively or inspect every pickle to guess a version.
            if provider != "sam3" or not (path / "sam3.pt").is_file():
                raise ValueError("Choose a checkpoint file (SAM3 directory must contain sam3.pt)")
            path = path / "sam3.pt"
        validate_checkpoint_file(path)
        if architecture is None:
            candidates = [row for row in self.catalog() if row["provider"] == provider and row.get("filename") == path.name]
            if candidates:
                architecture = candidates[0]["architecture"]
            elif provider == "yolo":
                architecture = "custom-seg"
            else:
                raise ValueError("Select the exact checkpoint architecture explicitly")
        entry = get_entry(provider, architecture)
        if class_mapping is not None:
            if not isinstance(class_mapping, dict) or any(not str(k).isdigit() or not isinstance(v, (str, type(None))) for k, v in class_mapping.items()):
                raise ValueError("Class mapping must map nonnegative model indices to class ids or null")
            class_mapping = {str(k): v for k, v in class_mapping.items()}
        digest = sha256_file(path)
        with file_lock(self.data_dir / "models.lock"):
            data = self._read()
            prior = next((r for r in data["models"] if r["sha256"] == digest and r["provider"] == provider
                          and r["architecture"] == entry["architecture"] and r["path"] == str(path)), None)
            row = {"id": prior["id"] if prior else str(uuid.uuid4()), "provider": provider,
                   "architecture": entry["architecture"], "name": path.name, "path": str(path), "sha256": digest,
                   "size_bytes": path.stat().st_size, "trusted": bool(trust or prior and prior.get("trusted")),
                   "source_revision": PROVIDERS[provider]["source_revision"], "license": entry["license"],
                   "license_url": entry["license_url"], "class_mapping": class_mapping if class_mapping is not None else (prior or {}).get("class_mapping", {}),
                   "class_names": (prior or {}).get("class_names", {}), "created_at": (prior or {}).get("created_at", time.time()),
                   "task": "segmentation", "validation_status": (prior or {}).get("validation_status", "registered_unverified"),
                   "preprocessing": "official_provider_full_image", "tested_devices": (prior or {}).get("tested_devices", [])}
            for key in ("training_layout", "prediction_settings", "metric_scope", "trained_at",
                        "checkpoint_name", "parent_model_id", "training_settings"):
                if prior and key in prior: row[key] = prior[key]
            row = dated_model(row)
            data["models"] = [r for r in data["models"] if r["id"] != row["id"]] + [row]
            write_json(self.registry_path, data)
        return row

    def install(self, provider, architecture, *, consent=False, download_weights=False, progress=None, cancel=None) -> dict:
        entry = get_entry(provider, architecture)
        if consent is not True:
            raise PermissionError("Explicit installation consent is required")
        if type(download_weights) is not bool:
            raise ValueError("download_weights must be a boolean")
        # Reject before runtime setup/network, even if the caller asks both.
        if provider == "sam3" and download_weights:
            raise ValueError("SAM3 checkpoint downloads are prohibited; add authorized local weights")
        self._state, self._error = "installing", None
        try:
            receipt = install_runtime(self.data_dir, provider, consent=consent, progress=progress, cancel=cancel)
            result = {"provider": provider, "architecture": entry["architecture"], "runtime": receipt,
                      "state": "weights_missing", "model": None}
            if download_weights:
                destination = self.data_dir / "checkpoints" / provider / entry["filename"]
                download_entry = entry
                # Preserve idempotence for the initial installer receipt format:
                # a previous official-source registration can bind a cached SAM2
                # file even before per-download sidecars were introduced.
                if destination.exists() and not entry.get("sha256"):
                    known = next((r for r in self.models() if r["path"] == str(destination)
                                  and r.get("download_source") == entry["url"] and r["trusted"]
                                  and r["provider"] == provider and r["architecture"] == entry["architecture"]), None)
                    if known:
                        download_entry = {**entry, "sha256": known["sha256"]}
                download = download_checkpoint(download_entry, destination, consent=consent, progress=progress, cancel=cancel)
                row = self.register(provider, destination, architecture=entry["architecture"], trust=True)
                row.update(download_source=entry["url"], digest_authority=entry.get("digest_authority"))
                self._save_record(row)
                result.update(model=row, model_id=row["id"], download=download, state="ready_unverified")
            return result
        except Exception as exc:
            self._error = str(exc)
            raise
        finally:
            self._state = "idle" if self._error is None else "failed"

    def _close_worker(self):
        if self._worker is not None:
            self._worker.close()
        self._worker = self._worker_key = None

    def _client(self, provider, *, persistent=False):
        python = runtime_python(self.data_dir, provider)
        if python is None:
            raise ProviderError(f"{provider} runtime is not installed; install it through Models & AI")
        for manager in list(_MANAGERS):
            if manager is not self:
                manager._close_worker()
        key = (provider, str(python))
        if not persistent or self._worker_key != key or self._worker is None or self._worker.process.poll() is not None:
            self._close_worker()
            self._worker = WorkerClient(python, self.data_dir)
            self._worker_key = key
        return self._worker

    def _infer_record(self, record, image_path, points, labels, box, device, settings, progress, cancel, *, persistent=True):
        if not record["trusted"]:
            raise PermissionError("Trust this checkpoint explicitly before loading; a hash does not make pickle safe")
        image_path = Path(image_path).expanduser().resolve(strict=True)
        payload = {"model": record, "image_path": str(image_path), "image_sha256": sha256_file(image_path, cancel),
                   "points": points, "labels": labels, "box": box, "device": device, "settings": settings}
        client = self._client(record["provider"], persistent=persistent and record["provider"] in {"sam2", "sam3"})
        result = client.request("infer", payload, progress=progress, cancel=cancel, timeout=float(settings.get("worker_timeout", 900)))
        if result.get("loaded_checkpoint_sha256") != record["sha256"] or result.get("model_id") != record["id"]:
            self._close_worker()
            raise ProviderError("Worker loaded a different checkpoint; result discarded")
        if result.get("image_sha256") != payload["image_sha256"] or not result.get("coverage", {}).get("full_image"):
            raise ProviderError("Worker image binding/coverage mismatch")
        return result

    def infer(self, model_id, image_path, *, points=None, labels=None, box=None, device="cpu", settings=None, progress=None, cancel=None) -> dict:
        from copy import deepcopy
        settings = deepcopy(settings or {})
        device = normalize_device(device)
        record = self._model(model_id)
        if record["provider"] in {"sam2", "sam3"}:
            from ..providers.prompt_settings import prompt_settings
            settings = prompt_settings(settings)
        if record["provider"] == "yolo":
            from ..training.tiles import resolve_prediction_settings
            settings = resolve_prediction_settings(record, settings)
        if record["provider"] == "yolo" and "tiling" in settings:
            from ..providers.yolo import validate_tiled_prediction
            width, height = canonical_image_size(image_path, settings)
            validate_tiled_prediction(settings, width, height, None)
        with file_lock(self._gpu_lock, cancel):
            self._state, self._error = "busy", None
            try:
                check_cancel(cancel)
                result = self._infer_record(record, image_path, points, labels, box, device, settings, progress, cancel)
                record.update(validation_status="inference_tested", last_tested_at=time.time(),
                              tested_devices=sorted(set(record.get("tested_devices", []) + [device])))
                if result.get("class_names"):
                    record["class_names"] = result["class_names"]
                record["preprocessing"] = result.get("preprocessing", record["preprocessing"])
                record["last_runtime"] = result.get("runtime", {})
                record["last_device"] = device
                record["last_precision"] = settings.get("precision", "float32")
                self._save_record(record)
                return result
            except Exception as exc:
                self._error = str(exc)
                self._close_worker()
                raise
            finally:
                if record["provider"] == "yolo":
                    self._close_worker()
                self._state = "idle" if self._error is None else "failed"

    def generate_proposals(self, model_id, image_path, *, crop_box=None, settings=None,
                           device="cpu", progress=None, cancel=None) -> dict:
        """Generate unassigned SAM2 proposals in crop-local pixel coordinates.

        No tile loop, class assignment, boundary recovery or durable layer lives
        here. The coordinator maps each completed crop through shared geometry.
        A cancelled worker returns no successful/complete result for that crop.
        """
        record = self._model(model_id)
        capability = automatic_capability(record["provider"])
        if not capability["supported"]:
            raise ValueError(capability["reason"])
        if not record["trusted"]:
            raise PermissionError("Trust this checkpoint explicitly before loading; a hash does not make pickle safe")
        settings = proposal_settings(settings)
        device = normalize_device(device)
        if device == "cpu" and settings["precision"] != "float32":
            raise ValueError("CPU provider execution requires float32; no implicit precision fallback")
        check_cancel(cancel)
        image_path = Path(image_path).expanduser().resolve(strict=True)
        width, height = canonical_image_size(image_path, settings)
        crop_box = validate_crop_box(crop_box, width, height)
        resolved_box = crop_box or [0, 0, width, height]
        expected_coverage = {"width": resolved_box[2] - resolved_box[0],
                             "height": resolved_box[3] - resolved_box[1], "full_image": crop_box is None}
        with file_lock(self._gpu_lock, cancel):
            self._state, self._error = "busy", None
            try:
                check_cancel(cancel)
                payload = {"model": record, "image_path": str(image_path), "image_sha256": sha256_file(image_path, cancel),
                           "crop_box": crop_box, "device": device, "settings": settings}
                result = self._client(record["provider"], persistent=True).request(
                    "generate_proposals", payload, progress=progress, cancel=cancel,
                    timeout=float(settings.get("worker_timeout", 900)))
                check_cancel(cancel)
                if (result.get("loaded_checkpoint_sha256") != record["sha256"] or
                        result.get("model_sha256") != record["sha256"] or result.get("model_id") != record["id"]):
                    raise ProviderError("Worker loaded a different checkpoint; result discarded")
                transform = result.get("transform", {})
                if (result.get("image_sha256") != payload["image_sha256"] or
                        result.get("coverage") != expected_coverage or
                        transform.get("crop_box") != resolved_box or
                        transform.get("original_width") != width or transform.get("original_height") != height or
                        result.get("settings") != settings):
                    raise ProviderError("Worker proposal image/crop/settings binding mismatch; result discarded")
                if not isinstance(result.get("annotations"), list):
                    raise ProviderError("Worker did not return proposal records")
                for row in result["annotations"]:
                    if (row.get("class_id") is not None or row.get("status") != "proposal" or
                            row.get("human_verified") is not False or row.get("review_actor") is not None or
                            row.get("geometry", {}).get("rle", {}).get("size") !=
                            [expected_coverage["height"], expected_coverage["width"]]):
                        raise ProviderError("Worker returned invalid class/review/coordinate proposal semantics")
                tests = record.setdefault("capability_tests", {})
                tests["generate_proposals"] = {"tested_at": time.time(), "device": device,
                                               "precision": settings["precision"],
                                               "coverage": expected_coverage, "proposal_count": len(result["annotations"])}
                record["last_runtime"] = result.get("runtime", {})
                self._save_record(record)
                return result
            except Exception as exc:
                self._error = str(exc)
                self._close_worker()
                raise
            finally:
                self._state = "idle" if self._error is None else "failed"

    def train(self, model_id, dataset_yaml, output_dir, settings, *, progress=None, cancel=None) -> dict:
        settings = dict(settings)
        device = normalize_device(settings.get("device", "cpu"))
        settings["device"] = device
        record = self._model(model_id)
        if record["provider"] != "yolo":
            raise ValueError("Only YOLO instance-segmentation training is supported")
        if not record["trusted"]:
            raise PermissionError("Trust the selected checkpoint before training")
        if settings.get("resume") and not settings.get("resume_checkpoint"):
            raise ValueError("Resume requires a registered, trusted resume_checkpoint path; use a new output directory")
        if settings.get("resume_checkpoint"):
            resume_path = Path(settings["resume_checkpoint"]).resolve(strict=True)
            # Must be a registered trusted checkpoint, never unpickle an arbitrary
            # path supplied in an advanced settings dictionary.
            resumed = next((r for r in self.models() if r["provider"] == "yolo" and r["trusted"] and Path(r["path"]) == resume_path), None)
            if resumed is None:
                raise PermissionError("Register and trust the last checkpoint before resuming")
            record = resumed
            settings["resume"] = True
        dataset_yaml = Path(dataset_yaml).resolve(strict=True)
        output_dir = Path(output_dir).expanduser().resolve()
        with file_lock(self._gpu_lock, cancel):
            self._state, self._error = "training", None
            try:
                self._close_worker()
                resume_source = None
                if settings.get("resume"):
                    resume_source = Path(record["path"])
                    if (output_dir / "run").exists():
                        raise ValueError("Resume needs a new job output directory; previous run artifacts are preserved")
                    destination = output_dir / "resume-input" / "last.pt"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with resume_source.open("rb") as source, destination.open("xb") as target:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            check_cancel(cancel)
                            target.write(block)
                        target.flush()
                        os.fsync(target.fileno())
                    if sha256_file(destination, cancel) != record["sha256"]:
                        raise ProviderError("Resume checkpoint changed during copy")
                    record = {**record, "path": str(destination.resolve())}
                client = self._client("yolo", persistent=False)
                result = client.request("train", {"model": record, "dataset_yaml": str(dataset_yaml),
                    "output_dir": str(output_dir), "device": device, "settings": settings},
                    progress=progress, cancel=cancel, timeout=float(settings.get("worker_timeout", 24 * 3600)))
                self._close_worker()
                check_cancel(cancel)
                if resume_source and sha256_file(resume_source, cancel) != record["sha256"]:
                    raise ProviderError("Original resume checkpoint changed; result cannot be accepted")
                checkpoint = Path(result["checkpoint_path"]).resolve(strict=True)
                if not checkpoint.is_relative_to(output_dir):
                    raise ProviderError("Trainer returned a checkpoint outside its output directory")
                measured = sha256_file(checkpoint, cancel)
                if measured != result["checkpoint_sha256"]:
                    raise ProviderError("Training artifact changed before load validation")
                candidate = {**record, "id": str(uuid.uuid4()), "path": str(checkpoint), "sha256": measured,
                             "architecture": "custom-seg", "trusted": True}
                emit(progress, stage="fresh_checkpoint_probe", checkpoint_sha256=measured)
                # A distinct OS process must load the saved artifact, not the old
                # model or the training process's in-memory weights.
                probe = self._infer_record(candidate, result["probe_image"], None, None, None, device,
                                           {"imgsz": settings.get("imgsz", 640), "worker_timeout": settings.get("worker_timeout", 900)},
                                           progress, cancel, persistent=False)
                self._close_worker()
                if probe.get("class_names") != result["class_names"]:
                    raise ProviderError("Fresh checkpoint class names do not match the training snapshot")
                mapping = settings.get("class_mapping", {})
                if mapping and set(map(str, mapping)) != set(result["class_names"]):
                    raise ValueError("Training class mapping must cover every trained class")
                row = self.register("yolo", checkpoint, architecture="custom-seg", trust=True, class_mapping=mapping)
                row.update(parent_model_id=model_id, class_names=result["class_names"], validation_status="fresh_checkpoint_inference_tested",
                           tested_devices=[device], dataset_sha256=sha256_file(dataset_yaml), training_settings=settings,
                           effective_training_settings=result.get("settings", {}), optimization=result.get("optimization"),
                           metrics=result["metrics"], selection_basis=result["selection_basis"])
                if settings.get('training_layout',{}).get('mode')=='tiles_512':
                    from compag_annotator.training.tiles import prediction_recipe
                    row['training_layout']=deepcopy(settings['training_layout'])
                    row['prediction_settings']=prediction_recipe(settings['training_layout'])
                    row['metric_scope']='tile'
                row["last_runtime"] = probe.get("runtime", {})
                from datetime import datetime, timezone
                row["trained_at"] = datetime.now(timezone.utc).isoformat()
                row = dated_model(row)
                self._save_record(row)
                result.update(model_id=row["id"], model=row, class_mapping=mapping,
                              load_probe={"success": True, "fresh_process": True, "loaded_checkpoint_sha256": probe["loaded_checkpoint_sha256"],
                                          "coverage": probe["coverage"], "proposal_count": len(probe["annotations"]), "timings": probe["timings"]})
                return result
            except Exception as exc:
                self._error = str(exc)
                raise
            finally:
                self._close_worker()
                self._state = "idle" if self._error is None else "failed"

    def unload(self) -> dict:
        with file_lock(self._gpu_lock):
            self._close_worker()
        self._state = "idle"
        return {"state": "unloaded"}

    def status(self) -> dict:
        rows = self.models()
        providers = {}
        for provider in PROVIDERS:
            runtime_error = None
            try:
                python = runtime_python(self.data_dir, provider)
            except (ValueError, OSError) as exc:
                python, runtime_error = None, str(exc)
            local = [r for r in rows if r["provider"] == provider and not r["missing"]]
            providers[provider] = {"installed": python is not None, "external": bool(os.environ.get(f"COMPAG_{provider.upper()}_PYTHON")),
                                   "state": "failed" if runtime_error else "not_installed" if python is None else "weights_missing" if not local else "trust_required" if not any(r["trusted"] for r in local) else "ready_unverified",
                                   "error": runtime_error,
                                   "model_count": len(local), "inference_tested": any("tested" in r["validation_status"] for r in local),
                                   "capabilities": list(PROVIDERS[provider]["capabilities"]),
                                   "capability_details": {"generate_proposals": automatic_capability(provider)},
                                   "automatic_proposals_tested": any("generate_proposals" in r.get("capability_tests", {}) for r in local)}
        return {"state": self._state, "error": self._error, "providers": providers,
                "loaded_provider": self._worker_key[0] if self._worker_key else None,
                "manual_annotation_available": True, "gpu_policy": "one controlled provider operation per user"}

    def devices(self, provider="sam2", *, refresh=False) -> dict:
        if provider not in PROVIDERS:
            raise ValueError("Unknown provider")
        if type(refresh) is not bool:
            raise ValueError("refresh must be a boolean")
        fallback = {"provider": provider, "devices": [{"id": "cpu", "name": "CPU", "kind": "cpu"}],
                    "cuda_available": False, "checkpoint_loaded": False}
        try:
            python = runtime_python(self.data_dir, provider)
            if python is None:
                return {**fallback, "status": "not_installed", "message": "Install this model's runtime in Models & AI to detect usable GPUs."}
            receipt = self.data_dir / "runtimes" / provider / "installed.json"
            key = (provider, str(python), python.stat().st_mtime_ns,
                   receipt.stat().st_mtime_ns if receipt.exists() else None, os.environ.get("CUDA_VISIBLE_DEVICES"))
            with self._device_probe_lock:
                cached = self._device_cache.get(provider)
                if not refresh and cached and cached[0] == key and time.monotonic() - cached[1] < 30:
                    return deepcopy(cached[2])
                # A separate short-lived worker must not unload a running model
                # or wait for an image-generation job's GPU lock.
                client = WorkerClient(python, self.data_dir)
                try:
                    result = client.request("devices", {}, timeout=30)
                finally:
                    client.close()
                result.update(provider=provider, status="ready" if not result.get("error") else "failed")
                result["message"] = result.get("error") or ("Choose CPU or an available GPU." if result["cuda_available"] else
                    "No CUDA GPU is available to this model runtime. Check the NVIDIA driver, WSL GPU access, and the CUDA-enabled runtime.")
                self._device_cache[provider] = (key, time.monotonic(), deepcopy(result))
                return result
        except (OSError, ValueError, RuntimeError) as exc:
            return {**fallback, "status": "failed", "message": f"Device detection failed: {exc}"}

    def doctor(self) -> dict:
        result = self.status()
        result["checks"] = []
        with file_lock(self._gpu_lock):
            self._close_worker()
            for provider in PROVIDERS:
                try:
                    python = runtime_python(self.data_dir, provider)
                except (ValueError, OSError) as exc:
                    result["checks"].append({"provider": provider, "status": "failed", "error": str(exc)})
                    continue
                if python is None:
                    result["checks"].append({"provider": provider, "status": "not_installed"})
                    continue
                client = WorkerClient(python, self.data_dir)
                try:
                    result["checks"].append({"provider": provider, "status": "runtime_import_ok", **client.request("probe", {"provider": provider}, timeout=120)})
                except Exception as exc:
                    result["checks"].append({"provider": provider, "status": "failed", "error": str(exc)})
                finally:
                    client.close()
        result["checkpoint_testing"] = "Runtime import/device checks do not prove real-checkpoint inference"
        return result
