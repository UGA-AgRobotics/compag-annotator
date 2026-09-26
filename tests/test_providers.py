"""Provider unit/protocol tests. Doubles here are NOT real-model validation."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pytest

from compag_annotator.models.catalog import catalog, get_entry
from compag_annotator.models.download import download_checkpoint, validate_checkpoint_file
from compag_annotator.models.manager import ModelManager
from compag_annotator.providers.common import ExactImageCache, mask_record, read_image
from compag_annotator.providers.protocol import CancelledError, normalize_device, sha256_file, validate_prompts
from compag_annotator.providers.yolo import local_dataset, train_settings


def checkpoint(path):
    """Structural PyTorch ZIP fixture, intentionally not loadable ML weights."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("fixture/data.pkl", b"explicit unit-test fixture; not model weights")
        archive.writestr("fixture/version", b"3")
    return path


def image(path, color="white", size=(71, 53)):
    from PIL import Image
    Image.new("RGB", size, color).save(path)
    return path


def test_core_manager_import_does_not_import_torch(tmp_path):
    code = "from compag_annotator.models.manager import ModelManager; from compag_annotator.providers.sam2 import generation_settings; assert generation_settings({})['points_per_side'] == 64; import sys; assert not any(n in sys.modules for n in ['torch','sam2','sam3','ultralytics'])"
    env = os.environ.copy()
    if os.environ.get("COMPAG_TEST_INSTALLED_PYTHON"):
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
    assert ModelManager(tmp_path).status()["manual_annotation_available"]


def test_catalog_is_copy_and_versions_are_exact():
    rows = catalog()
    assert len(rows) == 9
    assert all(len(r["source_revision"]) == 40 for r in rows)
    assert get_entry("sam2", "small")["config"] == "configs/sam2.1/sam2.1_hiera_s.yaml"
    assert get_entry("sam2", "base_plus")["config"].endswith("b+.yaml")
    assert get_entry("yolo", "yolo26n-seg.pt")["sha256"] == "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"
    assert not get_entry("sam3", "sam3-image")["checkpoint_download_allowed"]
    rows[0]["capabilities"].clear()
    assert catalog()[0]["capabilities"]
    with pytest.raises(ValueError):
        get_entry("sam3", "sam3.1")


@pytest.mark.parametrize("points,labels,box", [([[1, 2]], [2], None), ([[100, 2]], [1], None),
    ([[1, float("nan")]], [1], None), ([[1, 2]], [0], None), ([[1, 2]], [], None),
    ([[1, 2], [1, 2]], [0, 1], None), ([], [], [0, 0, 0, 4]), ([], [], None)])
def test_invalid_prompts(points, labels, box):
    with pytest.raises(ValueError):
        validate_prompts(100, 80, points, labels, box)


def test_positive_negative_and_box_prompts_full_image():
    assert validate_prompts(2400, 1900, [[1800, 1300], [1900, 1400]], [1, 0])[1] == [1, 0]
    assert validate_prompts(2400, 1900, box=[100, 100, 2200, 1800])[2] == [100., 100., 2200., 1800.]
    assert validate_prompts(100, 80, [[2, 3]], [0], [0, 0, 10, 10])[1] == [0]


@pytest.mark.parametrize("source,expected", [("cpu", "cpu"), ("cuda", "cuda:0"), (0, "cuda:0"), ("1", "cuda:1"), ("cuda:2", "cuda:2")])
def test_device_mapping(source, expected):
    assert normalize_device(source) == expected


@pytest.mark.parametrize("value", ["mps", "cuda:0,1", "auto", "0;echo", -1])
def test_invalid_device(value):
    with pytest.raises(ValueError):
        normalize_device(value)


def test_registration_hashes_without_loading_and_requires_trust(tmp_path, monkeypatch):
    model = checkpoint(tmp_path / "local.pt")
    manager = ModelManager(tmp_path / "state")
    row = manager.register("yolo", model, class_mapping={0: "uuid-a", 1: "uuid-b", 2: "uuid-c"})
    assert row["sha256"] == sha256_file(model)
    assert row["validation_status"] == "registered_unverified"
    assert not row["trusted"]
    assert row["class_mapping"]["2"] == "uuid-c"
    monkeypatch.setattr(manager, "_client", lambda *a, **kw: pytest.fail("Untrusted weights reached a worker"))
    with pytest.raises(PermissionError):
        manager.infer(row["id"], image(tmp_path / "im.png"))
    trusted = manager.register("yolo", model, trust=True)
    assert trusted["id"] == row["id"] and trusted["trusted"]
    assert trusted["class_mapping"] == row["class_mapping"]
    assert len(ModelManager(tmp_path / "state").models()) == 1


def test_wrong_architecture_and_sam3_directory(tmp_path):
    path = checkpoint(tmp_path / "sam3.pt")
    manager = ModelManager(tmp_path / "state")
    with pytest.raises(ValueError):
        manager.register("sam2", path)
    with pytest.raises(ValueError):
        manager.register("sam3", path, architecture="sam3.1", trust=True)
    row = manager.register("sam3", tmp_path, architecture="sam3-image", trust=True)
    assert row["path"] == str(path)


@pytest.mark.parametrize("body", [b"", b"<!DOCTYPE html><html>login</html>", b"<html>Forbidden</html>", b"PK broken file" * 5,
                                   b"version https://git-lfs.github.com/spec/v1\nnot downloaded"])
def test_bad_checkpoint_headers(tmp_path, body):
    path = tmp_path / "bad.pt"
    path.write_bytes(body)
    with pytest.raises(ValueError):
        validate_checkpoint_file(path)


def test_sam3_install_download_refused_before_runtime_or_network(tmp_path, monkeypatch):
    import compag_annotator.models.manager as module
    monkeypatch.setattr(module, "install_runtime", lambda *a, **kw: pytest.fail("SAM3 network/install invoked"))
    manager = ModelManager(tmp_path)
    with pytest.raises(ValueError, match="prohibited"):
        manager.install("sam3", "sam3-image", consent=True, download_weights=True)
    with pytest.raises(ValueError, match="local-only"):
        download_checkpoint(get_entry("sam3", "sam3-image"), tmp_path / "sam3.pt", consent=True)


class Response(io.BytesIO):
    def __init__(self, data, kind="application/octet-stream", declared=None):
        super().__init__(data)
        self.headers = {"Content-Type": kind, "Content-Length": str(len(data) if declared is None else declared)}


def download_double(monkeypatch, data, kind="application/octet-stream", declared=None):
    import urllib.request
    class Opener:
        def open(self, request, timeout):
            return Response(data, kind, declared)
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())


def test_download_verification_progress_and_idempotence(tmp_path, monkeypatch):
    payload = checkpoint(tmp_path / "fixture.pt").read_bytes()
    download_double(monkeypatch, payload)
    entry = {**get_entry("yolo", "yolo26n-seg"), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    progress = []
    out = tmp_path / "downloaded.pt"
    result = download_checkpoint(entry, out, consent=True, progress=progress.append)
    assert result["sha256"] == entry["sha256"] and out.read_bytes() == payload
    assert progress[-1]["stage"] == "download_verified"
    assert download_checkpoint(entry, out, consent=True)["cached"]
    assert not list(tmp_path.glob("*.part"))


def test_download_failure_leaves_no_checkpoint(tmp_path, monkeypatch):
    payload = checkpoint(tmp_path / "fixture.pt").read_bytes()
    entry = {**get_entry("yolo", "yolo26n-seg"), "size_bytes": len(payload), "sha256": "0" * 64}
    download_double(monkeypatch, payload)
    out = tmp_path / "download.pt"
    with pytest.raises(ValueError, match="checksum"):
        download_checkpoint(entry, out, consent=True)
    assert not out.exists() and not list(tmp_path.glob("*.part"))
    with pytest.raises(PermissionError):
        download_checkpoint(entry, out)
    with pytest.raises(CancelledError):
        download_checkpoint(entry, out, consent=True, cancel=lambda: True)


def test_download_interruption_and_html(tmp_path, monkeypatch):
    entry = {**get_entry("sam2", "small"), "size_bytes": None}
    download_double(monkeypatch, b"<html>login</html>", kind="text/html")
    with pytest.raises(ValueError, match="error/login"):
        download_checkpoint(entry, tmp_path / "a.pt", consent=True)
    download_double(monkeypatch, b"partial", declared=1024)
    with pytest.raises(ValueError, match="interrupted"):
        download_checkpoint(entry, tmp_path / "b.pt", consent=True)
    assert not list(tmp_path.glob("*.part"))


def test_exact_cache_is_pixel_model_and_preprocessing_bound():
    pixels = np.zeros((30, 40, 3), dtype=np.uint8)
    cache = ExactImageCache()
    assert cache.remember(pixels, ("model", "rgb"), 10000)
    assert cache.matches(pixels.copy(), ("model", "rgb"))
    assert not cache.matches(pixels, ("new-model", "rgb"))
    assert not cache.matches(pixels, ("model", "bgr"))
    pixels[0, 0, 0] = 1
    assert not cache.matches(pixels, ("model", "rgb"))
    assert not cache.remember(pixels, (), 1)


def test_full_mask_holes_components_and_unassigned_class():
    from compag_annotator.geometry import decode_rle
    mask = np.zeros((777, 1001), dtype=bool)
    mask[10:300, 100:400] = True
    mask[40:90, 120:170] = False
    mask[500:720, 800:1000] = True
    request = {"model": {"provider": "sam2", "id": "model", "sha256": "abc"}}
    row = mask_record(mask, request, .8)
    assert np.array_equal(mask, decode_rle(row["geometry"]["rle"]))
    assert row["bbox"] == [100, 10, 1000, 720]
    assert row["class_id"] is None and not row["human_verified"] and row["status"] == "proposal"


def test_image_identity_and_exif(tmp_path):
    from PIL import Image
    path = tmp_path / "oriented.jpg"
    im = Image.new("RGB", (80, 40))
    exif = im.getexif()
    exif[274] = 6
    im.save(path, exif=exif)
    out, pixels = read_image({"image_path": str(path), "image_sha256": sha256_file(path)})
    assert out.size == (40, 80) and pixels.shape == (80, 40, 3)
    with pytest.raises(ValueError, match="changed"):
        read_image({"image_path": str(path), "image_sha256": "wrong"})


def dataset(tmp_path):
    import yaml
    for name in ("train", "val"):
        (tmp_path / name).mkdir()
    image(tmp_path / "train" / "a.png", "red")
    image(tmp_path / "val" / "b.png", "blue")
    (tmp_path / "train" / "a.txt").write_text("0 .1 .1 .9 .1 .9 .9 .1 .9\n")
    (tmp_path / "val" / "b.txt").write_text("1 .1 .1 .8 .1 .8 .8 .1 .8\n")
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump({"path": str(tmp_path), "train": "train", "val": "val", "names": ["apple", "pear", "flower"]}))
    return path


def test_dataset_validation_local_split_names(tmp_path):
    path = dataset(tmp_path)
    _, names, images = local_dataset(path)
    assert names == {0: "apple", 1: "pear", 2: "flower"}
    assert images["train"][0].name == "a.png"
    (tmp_path / "val" / "b.png").write_bytes((tmp_path / "train" / "a.png").read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        local_dataset(path)


def test_dataset_no_download_hook_or_validation_leak(tmp_path):
    path = dataset(tmp_path)
    original = path.read_text()
    path.write_text(original + "\ndownload: echo unsafe\n")
    with pytest.raises(ValueError, match="download scripts"):
        local_dataset(path)
    path.write_text(original.replace("val: val", "val: train"))
    with pytest.raises(ValueError, match="overlap"):
        local_dataset(path)


def test_train_settings_no_extra_model_download_or_hidden_overrides():
    options = train_settings({"epochs": 1, "imgsz": 64, "batch": 1, "full_instance": True})
    assert options["amp"] is False and options["mosaic"] == 0 and options["translate"] == 0
    for bad in ({"amp": True}, {"batch": -1}, {"imgsz": 63}, {"data": "another-dataset"}):
        with pytest.raises(ValueError):
            train_settings(bad)


def test_real_worker_protocol_missing_runtime_is_error_and_no_model(tmp_path):
    """Actual child process, base Python has no SAM3. This is a protocol test."""
    from compag_annotator.providers.process import WorkerClient
    from compag_annotator.providers.protocol import ProviderError
    client = WorkerClient(sys.executable, tmp_path)
    try:
        with pytest.raises(ProviderError):
            client.request("probe", {"provider": "sam3"}, timeout=15)
    finally:
        client.close()
    assert client.process.poll() is not None


def test_process_cancel_terminates_child(tmp_path):
    from compag_annotator.providers.process import run_command
    start = time.monotonic()
    with pytest.raises(CancelledError):
        run_command([sys.executable, "-c", "import time; time.sleep(60)"], cancel=lambda: time.monotonic()-start > .2)
    assert time.monotonic()-start < 5


def test_training_does_not_register_success_without_fresh_probe(tmp_path, monkeypatch):
    manager = ModelManager(tmp_path / "state")
    base = manager.register("yolo", checkpoint(tmp_path / "base.pt"), trust=True)
    data = dataset(tmp_path)
    output = tmp_path / "training"
    output.mkdir()
    saved = checkpoint(output / "best.pt")
    sessions = []

    class Double:
        def request(self, operation, payload, **kwargs):
            sessions.append((operation, payload))
            if operation == "train":
                return {"checkpoint_path": str(saved), "checkpoint_sha256": sha256_file(saved),
                        "class_names": {"0": "apple", "1": "pear", "2": "flower"},
                        "probe_image": str(tmp_path / "val" / "b.png"), "metrics": {}, "selection_basis": "test_double"}
            raise RuntimeError("deliberately failed fresh probe")
    monkeypatch.setattr(manager, "_client", lambda *a, **kw: Double())
    with pytest.raises(RuntimeError, match="failed fresh probe"):
        manager.train(base["id"], data, output, {"epochs": 1, "imgsz": 64})
    assert len(manager.models()) == 1
    assert [s[0] for s in sessions] == ["train", "infer"]
    assert sessions[1][1]["model"]["id"] != base["id"]
    assert sessions[1][1]["model"]["path"] == str(saved)


def test_stale_checkpoint_result_is_rejected(tmp_path, monkeypatch):
    manager = ModelManager(tmp_path / "state")
    row = manager.register("yolo", checkpoint(tmp_path / "base.pt"), trust=True)
    class Double:
        def request(self, *args, **kwargs):
            return {"loaded_checkpoint_sha256": "stale", "model_id": row["id"]}
    monkeypatch.setattr(manager, "_client", lambda *a, **kw: Double())
    with pytest.raises(RuntimeError, match="different checkpoint"):
        manager.infer(row["id"], image(tmp_path / "test.png"))
    assert manager.models()[0]["validation_status"] == "registered_unverified"


def test_train_provenance_is_not_forwarded_to_backend():
    options = train_settings({"epochs": 1, "class_version": 3, "snapshot_sha256": "a" * 64})
    assert "class_version" not in options and "snapshot_sha256" not in options


def test_resume_rebinds_all_old_output_locations_before_file_creation(tmp_path):
    from types import SimpleNamespace
    from compag_annotator.providers.yolo import bind_training_location
    trainer = SimpleNamespace(args=SimpleNamespace(project="old/project", name="old-run", save_dir="old/output", data="old/data.yaml", exist_ok=True))
    bind_training_location(trainer, tmp_path / "new-job", tmp_path / "original-snapshot.yaml")
    assert trainer.args.project == str(tmp_path / "new-job")
    assert trainer.args.save_dir == str(tmp_path / "new-job" / "run")
    assert trainer.args.name == "run" and not trainer.args.exist_ok
    assert trainer.args.data == str(tmp_path / "original-snapshot.yaml")


def test_finished_or_incompatible_resume_does_not_fall_back_to_new_training():
    from compag_annotator.providers.yolo import validate_resume_state
    valid = {"epoch": 0, "optimizer": {"state": {}}, "train_args": {"epochs": 3, "amp": False}}
    validate_resume_state(valid, {"epochs": 3})
    for ckpt, options in [(None, {"epochs": 3}), ({**valid, "optimizer": None}, {"epochs": 3}),
                          ({**valid, "epoch": 2}, {"epochs": 3}), (valid, {"epochs": 5})]:
        with pytest.raises(ValueError):
            validate_resume_state(ckpt, options)


def test_resume_checkpoint_is_exclusive_byte_identical_copy(tmp_path, monkeypatch):
    manager = ModelManager(tmp_path / "state")
    source = checkpoint(tmp_path / "original-last.pt")
    initial = source.read_bytes()
    registered = manager.register("yolo", source, trust=True)
    data = dataset(tmp_path)
    output = tmp_path / "new-job"
    called = []
    class Double:
        def request(self, operation, payload, **kwargs):
            called.append(payload)
            raise RuntimeError("unit-test stops before training")
    monkeypatch.setattr(manager, "_client", lambda *a, **kw: Double())
    with pytest.raises(RuntimeError, match="stops before"):
        manager.train(registered["id"], data, output, {"epochs": 3, "resume_checkpoint": str(source)})
    copied = output / "resume-input" / "last.pt"
    assert source.read_bytes() == copied.read_bytes() == initial
    assert called[0]["model"]["path"] == str(copied)
    assert called[0]["settings"]["resume"] is True
    assert called[0]["dataset_yaml"] == str(data)
    assert len(manager.models()) == 1


def test_resume_rejects_unregistered_or_ambiguous_inputs(tmp_path):
    manager = ModelManager(tmp_path / "state")
    registered = manager.register("yolo", checkpoint(tmp_path / "base.pt"), trust=True)
    data = dataset(tmp_path)
    unregistered = checkpoint(tmp_path / "other.pt")
    with pytest.raises(PermissionError, match="Register and trust"):
        manager.train(registered["id"], data, tmp_path / "job", {"resume_checkpoint": str(unregistered)})
    with pytest.raises(ValueError, match="resume_checkpoint"):
        manager.train(registered["id"], data, tmp_path / "job", {"resume": True})


def test_box_only_rows_and_missing_negative_files_block_training(tmp_path):
    data = dataset(tmp_path)
    label = tmp_path / "train" / "a.txt"
    label.write_text("0 .5 .5 .8 .8\n")
    with pytest.raises(ValueError, match="box-only"):
        local_dataset(data)
    label.unlink()
    with pytest.raises(ValueError, match="Missing label"):
        local_dataset(data)


def test_sam3_strict_key_loader_rejects_missing_unexpected_and_shape(monkeypatch):
    """Explicit fake Torch/state dictionaries test validation, never ML execution."""
    from types import SimpleNamespace
    from compag_annotator.providers.sam3 import strict_sam3_load
    class Model:
        def state_dict(self):
            return {"backbone.weight": np.zeros((2, 3)), "inst_interactive_predictor.model.head": np.zeros(4)}
        def load_state_dict(self, state, strict):
            assert strict
            self.loaded = state
    valid = {"detector.backbone.weight": np.ones((2, 3)), "tracker.head": np.ones(4)}
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **kw: valid))
    model = Model()
    strict_sam3_load(model, "unit-fixture")
    assert len(model.loaded) == 2
    for bad in [{"tracker.head": np.ones(4)}, {**valid, "detector.extra": np.ones(1)}, {**valid, "tracker.head": np.ones(3)}]:
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *a, **kw: bad))
        with pytest.raises(ValueError, match="incompatible"):
            strict_sam3_load(Model(), "unit-fixture")


def test_worker_network_guard_blocks_checkpoint_requests():
    from compag_annotator.providers.worker import deny_network
    with pytest.raises(RuntimeError, match="offline"):
        deny_network("socket.getaddrinfo", ())
    with pytest.raises(RuntimeError, match="disabled"):
        deny_network("socket.connect", (None, ("example.com", 443)))
    deny_network("socket.connect", (None, "/tmp/local-ipc"))


def test_sam2_exact_pinned_set_image_signature_and_repeated_prompt_cache(tmp_path, monkeypatch):
    """Contract double uses the inspected SAM2 signature (no image_format kwarg)."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import compag_annotator.providers.sam2 as adapter
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(adapter, "device_context", lambda device, settings: ("cpu", nullcontext()))
    path = image(tmp_path / "test.png", size=(91, 73))
    record = {"id": "test", "provider": "sam2", "sha256": "fixture"}
    calls = []
    class Predictor:
        _transforms = SimpleNamespace(mask_threshold=0.0)
        _features = {"embedding": np.zeros(10)}
        def set_image(self, image):
            calls.append(("set_image", image.shape))
        def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None,
                    multimask_output=True, return_logits=False, normalize_coords=True):
            calls.append(("predict", point_labels.tolist(), box))
            masks = np.zeros((1, 73, 91), dtype=bool)
            masks[0, 10:40, 20:60] = True
            return masks, np.array([.9]), None
        def reset_predictor(self):
            calls.append(("reset",))
    provider = adapter.SAM2Provider.__new__(adapter.SAM2Provider)
    provider.record, provider.predictor, provider.cache = record, Predictor(), ExactImageCache()
    request = {"image_path": str(path), "image_sha256": sha256_file(path), "model": record,
               "points": [[25, 20], [60, 55]], "labels": [1, 0], "device": "cpu", "settings": {}}
    first = provider.infer(request, lambda event: None)
    second = provider.infer({**request, "points": [[30, 25], [65, 50]]}, lambda event: None)
    assert first["coverage"] == {"width": 91, "height": 73, "full_image": True}
    assert not first["encoding_cache_hit"] and second["encoding_cache_hit"]
    assert len([row for row in calls if row[0] == "set_image"]) == 1
    assert [row[1] for row in calls if row[0] == "predict"] == [[1, 0], [1, 0]]


@pytest.mark.parametrize("succeed", [False, True])
def test_adapter_repair_is_atomic_and_preserves_existing_runtime(tmp_path, monkeypatch, succeed):
    import compag_annotator.models.runtime as runtime
    python = tmp_path / "runtimes" / "sam2" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("existing runtime sentinel")
    site = python.parent.parent / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    pth = site / "compag_adapter_deps.pth"
    pth.write_text("previous-overlay\n")
    receipt = {"source_revision": "old", "lock_hash": "old-lock"}
    receipt_path = tmp_path / "runtimes" / "sam2" / "installed.json"
    receipt_path.write_text(json.dumps(receipt))
    lock = tmp_path / "adapters.txt"
    lock.write_text("explicit fixture lock")
    monkeypatch.setattr(runtime, "run_command", lambda *a, **kw: "")
    class Client:
        def __init__(self, *args):
            pass
        def request(self, *args, **kwargs):
            if not succeed:
                raise RuntimeError("fixture import failed")
            return {"geometry_import": True}
        def close(self):
            pass
    monkeypatch.setattr(runtime, "WorkerClient", Client)
    if succeed:
        result = runtime._repair_adapters(tmp_path, "sam2", python, receipt, lock, "new-lock", None, None)
        assert result["lock_hash"] == "new-lock" and pth.read_text() != "previous-overlay\n"
        assert json.loads(receipt_path.read_text())["adapter_repair"] == "atomic_overlay"
    else:
        with pytest.raises(RuntimeError, match="fixture import failed"):
            runtime._repair_adapters(tmp_path, "sam2", python, receipt, lock, "new-lock", None, None)
        assert pth.read_text() == "previous-overlay\n"
        assert json.loads(receipt_path.read_text()) == receipt
    assert python.read_text() == "existing runtime sentinel"


def test_packaged_provider_locks_have_geometry_dependencies_and_hashes():
    import compag_annotator.models.runtime as runtime
    directory = Path(runtime.__file__).with_name("requirements")
    for path in directory.glob("provider-*-*.txt"):
        text = path.read_text()
        assert "shapely==2.1.2 \\\n    --hash=sha256:" in text
        assert "pycocotools==2.0.10 \\\n    --hash=sha256:" in text
        assert text.count("shapely==") == 1 and text.count("pycocotools==") == 1


def test_sam3_image_lock_does_not_install_video_decoder():
    import compag_annotator.models.runtime as runtime
    path = Path(runtime.__file__).with_name("requirements") / "provider-sam3-cu128.txt"
    text = path.read_text()
    assert "decord==" not in text
    assert "torch==2.10.0+cu128" in text and "torchvision==0.25.0+cu128" in text


def test_sam3_repair_checks_exact_source_prefix_and_every_pin(tmp_path):
    import copy
    from compag_annotator.models.runtime import _sam3_image_repair_eligible
    from compag_annotator.models.catalog import PROVIDERS
    python = tmp_path / "venv" / "bin" / "python"
    lock = tmp_path / "lock.txt"
    lock.write_text("numpy==1.26.4 \\\n    --hash=sha256:fixture\ntorch==2.10.0+cu128\n")
    good = {"versions": {"sam3": "0.1.0", "numpy": "1.26.4", "torch": "2.10.0+cu128"},
            "python": [3, 12], "prefix": str(tmp_path / "venv"), "source": {"url": PROVIDERS["sam3"]["source_url"]}}
    assert _sam3_image_repair_eligible(good, python, lock)
    for key, value in [("python", [3, 11]), ("prefix", str(tmp_path / "other")), ("source", {"url": "different-source"})]:
        bad = {**good, key: value}
        assert not _sam3_image_repair_eligible(bad, python, lock)
    bad = copy.deepcopy(good)
    bad["versions"]["torch"] = "2.7.1+cu128"
    assert not _sam3_image_repair_eligible(bad, python, lock)


@pytest.mark.parametrize("probe_passes", [False, True])
def test_sam3_repair_requires_pip_check_and_real_import_probe_before_receipt(tmp_path, monkeypatch, probe_passes):
    import compag_annotator.models.runtime as runtime
    root = tmp_path / "runtimes" / "sam3"
    root.mkdir(parents=True)
    python = root / "venv" / "bin" / "python"
    commands = []
    def run(args, **kwargs):
        commands.append(list(map(str, args)))
        return "sam3==0.1.0"
    monkeypatch.setattr(runtime, "run_command", run)
    class Client:
        def __init__(self, *args):
            pass
        def request(self, operation, payload, **kwargs):
            assert operation == "probe" and payload == {"provider": "sam3"}
            if not probe_passes:
                raise RuntimeError("image API unavailable")
            return {"real_checkpoint_tested": False}
        def close(self):
            pass
    monkeypatch.setattr(runtime, "WorkerClient", Client)
    inventory = {"decord": {"version": "0.6.0", "wheel": "Tag: cp36-cp36m-manylinux2010_x86_64"}}
    if probe_passes:
        receipt = runtime._finish_sam3_image_runtime(tmp_path, python, inventory, "lock", "adapters", None, None)
        assert receipt["omitted_optional_dependencies"] == ["decord"]
        assert receipt["upstream_source_modified"] is False
        assert receipt["repair"]["removed"] == ["decord==0.6.0"]
        assert not receipt["real_checkpoint_tested"]
    else:
        with pytest.raises(RuntimeError, match="image API unavailable"):
            runtime._finish_sam3_image_runtime(tmp_path, python, inventory, "lock", "adapters", None, None)
        assert not (root / "installed.json").exists()
    assert commands[0][-3:] == ["uninstall", "--yes", "decord"]
    assert commands[1][-2:] == ["pip", "check"]
    assert all("torch" not in command for args in commands for command in args)


def test_runtime_commands_do_not_inherit_core_pythonpath(tmp_path, monkeypatch):
    from compag_annotator.providers.process import run_command
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "core-only"))
    monkeypatch.setenv("PIP_CACHE_DIR", str(tmp_path / "shared-cache"))
    code = "import os,json;print(json.dumps({'pythonpath':os.getenv('PYTHONPATH'),'cache':os.getenv('PIP_CACHE_DIR')}))"
    result = json.loads(run_command([sys.executable, "-c", code]))
    assert result == {"pythonpath": None, "cache": str(tmp_path / "shared-cache")}


def test_generation_recipe_and_capability_are_pure_immutable_and_honest():
    from compag_annotator.providers.sam2 import generation_settings
    source = {"points_per_side": 8}
    recipe = generation_settings(source)
    assert source == {"points_per_side": 8}
    assert recipe["points_per_batch"] == 512 and recipe["crop_n_layers"] == recipe["min_mask_region_area"] == 0
    assert generation_settings()["points_per_side"] == 64
    assert generation_settings(recipe) == recipe
    rows = {r["provider"]: r for r in catalog()}
    assert "generate_proposals" in rows["sam2"]["capabilities"]
    detail = rows["sam2"]["capability_details"]["generate_proposals"]
    assert detail["supported"] and not detail["requires_classes"] and not detail["requires_yolo"]
    detail["defaults"]["points_per_side"] = 999
    assert catalog()[0]["capability_details"]["generate_proposals"]["defaults"]["points_per_side"] == 64
    sam3 = rows["sam3"]["capability_details"]["generate_proposals"]
    assert not sam3["supported"] and "point-grid" in sam3["reason"]
    assert "points" in rows["sam3"]["capabilities"] and "box" in rows["sam3"]["capabilities"]
    assert not rows["sam3"]["checkpoint_download_allowed"]


@pytest.mark.parametrize("field,value", [
    ("points_per_side", True), ("points_per_side", 8.5), ("points_per_side", 0), ("points_per_side", 65),
    ("points_per_side", 10**400), ("points_per_batch", 513), ("points_per_batch", False),
    ("pred_iou_thresh", float("nan")), ("stability_score_thresh", float("inf")), ("box_nms_thresh", -1),
    ("crop_n_layers", 1), ("min_mask_region_area", 1), ("multimask_output", 1),
    ("precision", "auto"), ("worker_timeout", 0), ("classes", []), ("tiling", {}), ("use_m2m", True),
])
def test_generation_recipe_rejects_invalid_or_unsupported_controls(field, value):
    from compag_annotator.providers.sam2 import generation_settings
    from compag_annotator.providers.automatic import ProposalSettingsError
    with pytest.raises(ProposalSettingsError) as error:
        generation_settings({field: value})
    assert field in error.value.fields


@pytest.mark.parametrize("crop", [[0, 0, 0, 4], [0, 0, 101, 4], [0, -1, 2, 4],
                                  [True, 0, 2, 4], [0, 0, 2.5, 4], [0, 0, 4]])
def test_generation_crop_rejects_invalid_canonical_pixel_edges(crop):
    from compag_annotator.providers.automatic import validate_crop_box
    with pytest.raises(ValueError, match="crop_box"):
        validate_crop_box(crop, 100, 80)


def automatic_double(tmp_path, monkeypatch, *, size=(37, 29), fail=False):
    """Exact public AMG signature double; tests never claim real masks/weights."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from compag_annotator.geometry import encode_rle
    import compag_annotator.providers.sam2 as adapter
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setattr(adapter, "device_context", lambda device, settings: ("cpu", nullcontext()))
    calls = {"resets": [], "generated_pixels": [], "masks": []}

    class Generator:
        def __init__(self, model, points_per_side=32, points_per_batch=64, pred_iou_thresh=.8,
                     stability_score_thresh=.95, stability_score_offset=1., mask_threshold=0.,
                     box_nms_thresh=.7, crop_n_layers=0, crop_nms_thresh=.7, crop_overlap_ratio=512/1500,
                     crop_n_points_downscale_factor=1, point_grids=None, min_mask_region_area=0,
                     output_mode="binary_mask", use_m2m=False, multimask_output=True):
            calls["options"] = {key: value for key, value in locals().items() if key not in {"self", "model"}}
            self.predictor = SimpleNamespace(reset_predictor=lambda: calls["resets"].append("automatic"))

        def generate(self, image):
            calls["generated_pixels"].append(image.copy())
            assert image.flags.c_contiguous and image.dtype == np.uint8
            if fail:
                raise RuntimeError("explicit generation failure double")
            height, width = image.shape[:2]
            mask = np.ones((height, width), dtype=bool)  # Includes every input edge.
            mask[2:-2, 2:-2] = False
            mask[height//2, width//2] = True  # Disconnected island inside the hole.
            calls["masks"].append(mask)
            return [{"segmentation": encode_rle(mask), "area": int(mask.sum()),
                     "predicted_iou": .93, "stability_score": .97,
                     "bbox": [0, 0, width-1, height-1], "crop_box": [0, 0, width, height],
                     "point_coords": [[width/2, height/2]]}]

    monkeypatch.setitem(sys.modules, "sam2.automatic_mask_generator", SimpleNamespace(SAM2AutomaticMaskGenerator=Generator))
    path = image(tmp_path / "automatic.png", size=size)
    record = {"id": "explicit-unit-double", "provider": "sam2", "sha256": "fixture"}
    provider = adapter.SAM2Provider.__new__(adapter.SAM2Provider)
    provider.record = record
    provider.model = SimpleNamespace(image_size=1024)
    provider.cache = ExactImageCache()
    provider.cache.remember(np.ones((2, 2), dtype=bool), "interactive", 100)
    provider.predictor = SimpleNamespace(reset_predictor=lambda: calls["resets"].append("interactive"))
    request = {"image_path": str(path), "image_sha256": sha256_file(path), "model": record,
               "device": "cpu", "settings": {"points_per_side": 8}}
    return provider, request, calls


@pytest.mark.parametrize("crop", [None, [3, 4, 34, 26]])
def test_automatic_exact_crop_rle_topology_filters_and_mapping(tmp_path, monkeypatch, crop):
    from compag_annotator.geometry import decode_rle, map_tile_geometry
    provider, request, calls = automatic_double(tmp_path, monkeypatch)
    events = []
    result = provider.generate_proposals({**request, "crop_box": crop}, events.append)
    box = crop or [0, 0, 37, 29]
    width, height = box[2]-box[0], box[3]-box[1]
    assert calls["generated_pixels"][0].shape == (height, width, 3)
    assert calls["options"]["output_mode"] == "uncompressed_rle"
    assert calls["options"]["crop_n_layers"] == calls["options"]["min_mask_region_area"] == 0
    assert not calls["options"]["use_m2m"]
    row = result["annotations"][0]
    assert np.array_equal(decode_rle(row["geometry"]["rle"]), calls["masks"][0])
    assert row["area"] == int(calls["masks"][0].sum()) and row["bbox"] == [0, 0, width, height]
    assert row["class_id"] is None and row["review_actor"] is None and not row["human_verified"]
    assert "model_class_index" not in row and row["quality"]["stability_score"] == .97
    assert result["coverage"] == {"width": width, "height": height, "full_image": crop is None}
    transform = result["transform"]
    assert transform["crop_box"] == box and transform["encoder_tensor_shape"] == [1, 3, 1024, 1024]
    assert transform["output_to_image"]["translation_xy"] == box[:2]
    assert result["filters"]["suppression_counts"] is None
    assert not result["filters"]["application_tile_edge_suppression"]
    assert not result["filters"]["heavy_boundary_recovery"]
    assert result["work"]["points"] == 64 and result["work"]["prompt_batches"] == 1
    assert events[0]["completed_crops"] == 0 and events[-1]["completed_crops"] == 1
    assert provider.cache.pixels is None and calls["resets"] == ["interactive", "automatic"]
    tile = {"box": box, "valid_box": [0, 0, width, height], "padding": [0, 0, 0, 0]}
    mapped = decode_rle(map_tile_geometry(row["geometry"], tile, 37, 29)["rle"])
    expected = np.zeros((29, 37), dtype=bool)
    expected[box[1]:box[3], box[0]:box[2]] = calls["masks"][0]
    assert np.array_equal(mapped, expected)
    json.dumps(result, allow_nan=False)


def test_automatic_exception_resets_features_without_success(tmp_path, monkeypatch):
    provider, request, calls = automatic_double(tmp_path, monkeypatch, fail=True)
    events = []
    with pytest.raises(RuntimeError, match="generation failure"):
        provider.generate_proposals(request, events.append)
    assert provider.cache.pixels is None and calls["resets"] == ["interactive", "automatic"]
    assert all(event["completed_crops"] == 0 for event in events)


def proposal_response(payload):
    """Empty output is a valid protocol double, never actual model evidence."""
    from compag_annotator.providers.automatic import canonical_image_size
    width, height = canonical_image_size(payload["image_path"], payload["settings"])
    box = payload.get("crop_box") or [0, 0, width, height]
    return {"annotations": [], "model_id": payload["model"]["id"],
            "model_sha256": payload["model"]["sha256"], "loaded_checkpoint_sha256": payload["model"]["sha256"],
            "image_sha256": payload["image_sha256"], "settings": payload["settings"],
            "coverage": {"width": box[2]-box[0], "height": box[3]-box[1], "full_image": payload.get("crop_box") is None},
            "transform": {"crop_box": box, "original_width": width, "original_height": height}, "timings": {}}


def test_manager_generation_routes_serial_worker_and_keeps_capability_tests_separate(tmp_path, monkeypatch):
    manager = ModelManager(tmp_path / "state")
    model = manager.register("sam2", checkpoint(tmp_path / "sam.pt"), architecture="small", trust=True)
    path = image(tmp_path / "test.png")
    calls = []
    class Client:
        def request(self, operation, payload, **kwargs):
            calls.append((operation, payload, kwargs))
            return proposal_response(payload)
    monkeypatch.setattr(manager, "_client", lambda provider, **kwargs: Client())
    result = manager.generate_proposals(model["id"], path, crop_box=[4, 5, 40, 30], settings={"points_per_side": 8})
    assert calls[0][0] == "generate_proposals"
    assert calls[0][1]["crop_box"] == [4, 5, 40, 30] and calls[0][1]["settings"]["points_per_batch"] == 512
    assert result["coverage"] == {"width": 36, "height": 25, "full_image": False}
    updated = manager.models()[0]
    assert updated["validation_status"] == "registered_unverified"  # Does not claim prompted-inference success.
    assert updated["capability_tests"]["generate_proposals"]["proposal_count"] == 0


@pytest.mark.parametrize("field", ["image", "crop", "settings", "checkpoint", "dimensions"])
def test_manager_rejects_stale_generation_binding(tmp_path, monkeypatch, field):
    manager = ModelManager(tmp_path / "state")
    model = manager.register("sam2", checkpoint(tmp_path / "sam.pt"), architecture="small", trust=True)
    class Client:
        def request(self, operation, payload, **kwargs):
            result = proposal_response(payload)
            if field == "image": result["image_sha256"] = "stale"
            if field == "crop": result["transform"]["crop_box"] = [0, 0, 10, 10]
            if field == "settings": result["settings"] = {}
            if field == "checkpoint": result["loaded_checkpoint_sha256"] = "stale"
            if field == "dimensions": result["coverage"]["width"] += 1
            return result
    monkeypatch.setattr(manager, "_client", lambda *args, **kwargs: Client())
    with pytest.raises(RuntimeError, match="mismatch|different checkpoint"):
        manager.generate_proposals(model["id"], image(tmp_path / "test.png"), crop_box=[1, 2, 60, 40])
    assert "capability_tests" not in manager.models()[0]


def test_manager_auto_unsupported_untrusted_invalid_and_cancel_before_runtime(tmp_path, monkeypatch):
    manager = ModelManager(tmp_path / "state")
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid/unsupported request must not open a model runtime")
    monkeypatch.setattr(manager, "_client", forbidden)
    local = checkpoint(tmp_path / "local.pt")
    sam3 = manager.register("sam3", local, architecture="sam3-image", trust=True)
    with pytest.raises(ValueError, match="SAM3 automatic.*unsupported"):
        manager.generate_proposals(sam3["id"], "does-not-exist")
    sam2 = manager.register("sam2", local, architecture="small")
    with pytest.raises(PermissionError, match="Trust"):
        manager.generate_proposals(sam2["id"], "does-not-exist")
    sam2 = manager.register("sam2", local, architecture="small", trust=True)
    with pytest.raises(ValueError, match="crop_n_layers"):
        manager.generate_proposals(sam2["id"], "does-not-exist", settings={"crop_n_layers": 1})
    with pytest.raises(CancelledError):
        manager.generate_proposals(sam2["id"], "does-not-exist", cancel=lambda: True)
    with pytest.raises(ValueError, match="CPU.*float32"):
        manager.generate_proposals(sam2["id"], "does-not-exist", settings={"precision": "bfloat16"})


def test_worker_generation_dispatch_and_revalidation_without_loading_ml(tmp_path, monkeypatch):
    import compag_annotator.providers.sam2 as sam2
    import compag_annotator.providers.worker as worker_module
    model = {"id": "explicit-unit-double", "provider": "sam2", "architecture": "sam2.1_hiera_small",
             "path": str(checkpoint(tmp_path / "fixture.pt")), "trusted": True}
    model["sha256"] = sha256_file(model["path"])
    path = image(tmp_path / "test.png")
    calls = []
    class Adapter:
        def __init__(self, record, device, settings): calls.append("load")
        def generate_proposals(self, request, progress):
            calls.append("generate")
            return proposal_response(request)
        def infer(self, *args): pytest.fail("Automatic requests must not route to prompted infer")
        def unload(self): calls.append("unload")
    monkeypatch.setattr(sam2, "SAM2Provider", Adapter)
    monkeypatch.setattr(worker_module, "runtime_identity", lambda provider: {"kind": "explicit-unit-double"})
    worker = worker_module.Worker()
    request = {"protocol": 1, "operation": "generate_proposals", "model": model,
               "image_path": str(path), "image_sha256": sha256_file(path), "device": "cpu", "settings": {}}
    result = worker.execute(request, lambda event: None)
    assert calls == ["load", "generate"] and result["settings"]["points_per_side"] == 64
    assert result["timings"]["cold_model_load"]
    with pytest.raises(ValueError, match="points_per_side"):
        worker.execute({**request, "settings": {"points_per_side": False}}, lambda event: None)
    assert calls == ["load", "generate"]
    worker.release()


def yolo_inference_double(tmp_path, monkeypatch, size):
    """Fake predictor output; shared real tile planner/mapping remain under test."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import compag_annotator.providers.yolo as adapter
    monkeypatch.setattr(adapter, "device_context", lambda device, settings: ("cpu", nullcontext()))
    monkeypatch.setattr(adapter, "recording_predictor_class", lambda: "explicit-predictor-double")
    calls = []
    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def cpu(self): return self
        def numpy(self): return self.data
    class Boxes:
        cls, conf = Tensor([0]), Tensor([.9])
        def __len__(self): return 1
    class Model:
        names = {0: "test class"}
        predictor = None
        def predict(self, source, **kwargs):
            width, height = source.size
            calls.append({"size": source.size, **kwargs})
            self.predictor = SimpleNamespace(compag_transforms=[{
                "source_size": [width, height], "tensor_shape": [1, 3, 640, 640],
                "explicit_test_double": True}])
            return [SimpleNamespace(masks=SimpleNamespace(data=Tensor(np.ones((1, height, width)))), boxes=Boxes())]
    provider = adapter.YOLOProvider.__new__(adapter.YOLOProvider)
    provider.model = Model()
    path = image(tmp_path / "yolo.png", size=size)
    request = {"image_path": str(path), "image_sha256": sha256_file(path), "device": "cpu", "settings": {},
               "model": {"id": "explicit-unit-double", "provider": "yolo", "sha256": "fixture"}}
    return provider, request, calls


@pytest.mark.parametrize("config", [
    {"preset": "256"}, {"preset": "512"}, {"preset": "1024"},
    {"preset": "custom", "width": 640, "height": 384},
    {"preset": "custom", "width": 513, "height": 385}, {"mode": "whole"},
])
def test_yolo_real_rectangular_plan_crop_sizes_and_full_image_mapping(tmp_path, monkeypatch, config):
    from compag_annotator.geometry import plan_tiles, mask_metadata
    provider, request, calls = yolo_inference_double(tmp_path, monkeypatch, (1536, 2048))
    events = []
    result = provider.infer({**request, "settings": {"tiling": config, "imgsz": 640}}, events.append)
    plan = plan_tiles(1536, 2048, config, image_sha256=request["image_sha256"])
    assert result["tile_plan"] == plan
    assert len(calls) == plan["tile_count"] == len(result["annotations"])
    for call, tile, row, receipt in zip(calls, plan["tiles"], result["annotations"], result["tile_receipts"]):
        x0, y0, x1, y1 = tile["box"]
        assert call["size"] == (x1-x0, y1-y0) and call["imgsz"] == 640
        assert row["geometry"]["rle"]["size"] == [2048, 1536]
        assert mask_metadata(row["geometry"], 1536, 2048) == {"area": (x1-x0)*(y1-y0), "bbox": tile["box"]}
        assert row["source"]["tile_id"] == tile["id"] and receipt["transform"]["crop_box"] == tile["box"]
    assert not result["filters"]["cross_tile_suppression"] and not result["filters"]["automatic_union"]
    assert events[-1]["completed_tiles"] == plan["tile_count"]
    json.dumps(result, allow_nan=False)


def test_yolo_new_plan_retains_near_duplicates_but_legacy_grid_unchanged(tmp_path, monkeypatch):
    provider, request, calls = yolo_inference_double(tmp_path, monkeypatch, (513, 400))
    plan = {"preset": "512", "overlap": {"mode": "pixels", "x": 0, "y": 0}}
    result = provider.infer({**request, "settings": {"tiling": plan}}, lambda event: None)
    assert len(result["annotations"]) == 2  # Almost identical masks, shifted by one pixel.
    assert [row["bbox"] for row in result["annotations"]] == [[0, 0, 512, 400], [1, 0, 513, 400]]
    legacy = provider.infer({**request, "settings": {"tile_size": 512, "overlap": 0}}, lambda event: None)
    assert len(legacy["annotations"]) == 1 and "tile_plan" not in legacy


@pytest.mark.parametrize("settings", [{"tiling": None}, {"tiling": {}, "tile_size": 512},
    {"tiling": {}, "deduplicate_iou": .8}, {"tiling": {}, "imgsz": 513},
    {"tiling": {}, "confidence": True}, {"tiling": {}, "max_det": 1.5}])
def test_new_yolo_plan_rejects_ambiguous_or_invalid_controls_before_prediction(tmp_path, monkeypatch, settings):
    provider, request, calls = yolo_inference_double(tmp_path, monkeypatch, (73, 59))
    with pytest.raises(ValueError):
        provider.infer({**request, "settings": settings}, lambda event: None)
    assert not calls
