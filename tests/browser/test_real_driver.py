"""Offline unit checks only; no actual acceptance project or provider is contacted."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pycocotools import mask as coco


@pytest.fixture
def driver():
    path = Path(__file__).resolve().parents[2] / "scripts" / "real_ui_addendum_acceptance.py"
    spec = importlib.util.spec_from_file_location("real_ui_driver_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_driver_plan_does_not_contact_server_or_create_output(driver, tmp_path, capsys):
    output = tmp_path / "not-created"
    assert (
        driver.main(
            ["--url", "http://127.0.0.1:9", "--pid", "offline", "--iid", "offline", "--output", str(output)]
        )
        == 0
    )
    assert "no server contact" in capsys.readouterr().out.lower()
    assert not output.exists()
    with pytest.raises(SystemExit):
        driver.arguments(
            [
                "--url",
                "https://remote.example",
                "--pid",
                "offline",
                "--iid",
                "offline",
                "--output",
                str(output),
            ]
        )


def test_driver_exact_pixel_helpers_with_small_unit_fixture(driver):
    # Synthetic pixels are restricted to this unit fixture; the real driver has
    # no data-generation path and rejects explicit synthetic provider evidence.
    pixels = np.zeros((21, 23), dtype=np.uint8)
    pixels[2:15, 2:15] = 1
    pixels[5:10, 5:10] = 0
    pixels[17:20, 18:22] = 1
    encoded = coco.encode(np.asfortranarray(pixels))
    encoded["counts"] = encoded["counts"].decode("ascii")
    decoded = driver.decode_mask(encoded, 23, 21)
    assert np.array_equal(decoded, pixels)
    receipt = driver.pixel_receipt(decoded)
    assert receipt["area"] == 156 and receipt["bbox"] == [2, 2, 22, 20]
    x, y = driver.interior_point(decoded)
    assert decoded[int(y), int(x)]
    with pytest.raises(RuntimeError, match="dimensions|full-image"):
        driver.decode_mask(encoded, 22, 21)


def test_driver_blocks_provider_and_cross_project_mutations(driver, tmp_path):
    instance = object.__new__(driver.Acceptance)
    instance.origin = "http://127.0.0.1:9"
    instance.base = "/api/projects/target"
    instance.image_base = instance.base + "/images/source"
    instance.args = SimpleNamespace(execute=True)
    instance.output = tmp_path
    instance.stage = "offline_route_policy_unit_test"
    instance.record = {"mutations": [], "blocked_requests": [], "inference_training_weight_requests": 0}

    class Route:
        def __init__(self, path, method="POST"):
            self.request = SimpleNamespace(url=instance.origin + path, method=method, post_data_json={})
            self.allowed = self.blocked = False

        def continue_(self):
            self.allowed = True

        def abort(self):
            self.blocked = True

    for suffix in ("generate", "predict", "train", "boundary/annotation"):
        route = Route(instance.base + "/" + suffix)
        instance.guard_request(route)
        assert route.blocked and not route.allowed
    for path in ("/api/models/install", "/api/projects/other/selection"):
        route = Route(path)
        instance.guard_request(route)
        assert route.blocked
    allowed = Route(instance.base + "/merge")
    instance.guard_request(allowed)
    assert allowed.allowed and not allowed.blocked
    instance.args.execute = False
    readonly = Route(instance.base + "/merge")
    instance.guard_request(readonly)
    assert readonly.blocked
    assert instance.record["inference_training_weight_requests"] == 4
