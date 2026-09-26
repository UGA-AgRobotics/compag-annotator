"""Synthetic dataset evidence only: no model inference, training, or human UAT."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image
import pytest

from compag_annotator.core.catalog import Catalog
from compag_annotator.core.projects import Project
from compag_annotator.core.selection import merge_selected
from compag_annotator.geometry import encode_rle, geometry_mask
from compag_annotator.storage.files import atomic, digest, uid
from compag_annotator.training.snapshots import build_snapshot


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_addendum_training_path.py"
SPEC = importlib.util.spec_from_file_location("dataset_evidence", SCRIPT)
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def add(project, iid, geometry, cid, status="accepted", source=None):
    old = {r["id"] for r in project.annotations()}
    project.annotate(iid, {"geometry": geometry, "class_id": cid, "status": status,
                           "source": source or {"kind": "manual", "synthetic_fixture": True}}, "automated_qa")
    return next(r["id"] for r in project.annotations() if r["id"] not in old)


def complete(project, iid):
    project.update_image(iid, {"complete": True, "attest": True}, "automated_qa")


def polygon():
    return {"type": "polygon", "points": [[5, 5], [16, 5], [16, 16], [5, 16]]}


@pytest.fixture
def scene(tmp_path):
    data = tmp_path / "app"
    catalog = Catalog(data)
    base = catalog.create("Synthetic six-image fixture", [{"name": "Old category"}])
    paths = []
    for i in range(6):
        path = tmp_path / f"fixture-{i}.png"
        Image.new("RGB", (64, 48), (25 + i * 15, 60, 110)).save(path)
        paths.append(path)
    assert not base.add_images(paths)["errors"]
    images = base.state()["images"]
    cid = base.state()["classes"][0]["id"]
    for index, image in enumerate(images):
        if index >= 4:
            base.update_image(image["id"], {"role": "validation"}, "automated_qa")
        add(base, image["id"], polygon(), cid)
        complete(base, image["id"])
    base.plan_rounds({"count": 2, "confirm": True}, "automated_qa")
    old_snapshots = []
    for round_record in base.state()["rounds"]:
        rid = round_record["id"]
        base.round_action(rid, {"action": "start"}, "automated_qa")
        base.round_action(rid, {"action": "finish"}, "automated_qa")
        jid, mid = uid(), uid()
        job_dir = data / "jobs" / jid
        snapshot = build_snapshot(base, job_dir / "dataset", {"round_id": rid, "qa_smoke": True})
        old_snapshots.append(snapshot)
        # Persisted receipt fixtures, never actual trained checkpoints.
        weights = job_dir / "fixture_weights.bin"
        weights.write_bytes(b"explicit synthetic receipt fixture; never load as model")
        model = {"id": mid, "path": str(weights), "sha256": digest(weights),
                 "class_mapping": snapshot["receipt"]["class_mapping"],
                 "training_settings": {"round_id": rid, "snapshot_sha256": snapshot["snapshot_sha256"]},
                 "synthetic_fixture": True}
        with base.edit(None, "synthetic_historical_receipt", "automated_qa") as (db, state):
            state["models"].append(model)
            selected = next(r for r in state["rounds"] if r["id"] == rid)
            selected.update(trained_model_id=mid, dataset_snapshot_sha256=snapshot["snapshot_sha256"], training_status="complete")
        atomic(job_dir / "job.json", {"id": jid, "kind": "train", "status": "complete", "synthetic_fixture": True,
            "payload": {"project_id": base.state()["id"], "round_id": rid},
            "result": {"model_id": mid, "dataset_snapshot": str(job_dir / "dataset/snapshot.json"),
                       "snapshot_sha256": snapshot["snapshot_sha256"], "checkpoint_path": str(weights),
                       "checkpoint_sha256": digest(weights), "class_mapping": model["class_mapping"]}})

    generated = catalog.create("Synthetic generated masks", [{"name": "Reviewed arbitrary category"}])
    generated.add_images([paths[4]])
    iid = generated.state()["images"][0]["id"]
    source_cid = generated.state()["classes"][0]["id"]
    layer, generation_job = uid(), uid()
    with generated.edit(None, "synthetic_generation_layer", "automated_qa") as (db, state):
        state["generation_layers"].append({"id": layer, "image_id": iid, "status": "complete"})
    atomic(data / "jobs" / generation_job / "job.json", {"id": generation_job, "kind": "generate", "status": "complete",
        "synthetic_fixture": True, "payload": {"project_id": generated.state()["id"], "layers": [{"id": layer, "image_id": iid}]}})

    def mask_add(mask, status="accepted"):
        aid = add(generated, iid, {"type": "mask", "rle": encode_rle(mask)}, source_cid, status,
                  {"kind": "sam2", "operation": "automatic_mask_generation", "job_id": generation_job,
                   "synthetic_fixture": True})
        with generated.edit(None, "synthetic_generation_membership", "automated_qa") as (db, state):
            row = generated.get_annotation(aid, full=False)
            row["layer_id"] = layer
            generated._put(db, row)
        return aid

    first = np.zeros((48, 64), bool)
    first[2:16, 2:16] = True
    first[6:11, 6:11] = False
    second = np.zeros_like(first)
    second[24:32, 24:36] = True
    parents = [mask_add(first), mask_add(second)]
    merge_selected(generated, {"image_id": iid, "annotation_ids": parents,
                               "expected_revision": generated.state()["revision"], "confirm": True, "accept": True}, "automated_qa")
    child = next(r["id"] for r in generated.annotations() if r["status"] == "accepted")
    other = np.zeros_like(first)
    other[35:44, 42:55] = True
    kept = mask_add(other)
    omitted = [mask_add(other, status) for status in ("draft", "proposal", "rejected")]
    deleted = mask_add(other, "proposal")
    generated.annotate(iid, {}, "automated_qa", aid=deleted, delete=True)
    return {"data": data, "base": base, "generated": generated, "matched": images[4],
            "child": child, "kept": kept, "parents": parents, "omitted": omitted, "deleted": deleted,
            "source_class": source_cid, "old_snapshots": old_snapshots, "expected_mask": first | second,
            "generation_job": generation_job}


def run(scene, output, **kwargs):
    return evidence.check(scene["data"], scene["base"].state()["id"], scene["generated"].state()["id"], output, **kwargs)


def test_native_clone_fixed_validation_exact_rows_classes_losses_and_preservation(scene, tmp_path):
    before = evidence.tree_hashes(scene["data"])
    mapping = {scene["source_class"]: {"id": uuid.uuid4().hex, "name": "Arbitrary Ω label", "color": "#aa4411"}}
    result = run(scene, tmp_path / "evidence", mapping=mapping)
    assert result["passed"] and result["source_bytes_unchanged"]
    assert evidence.tree_hashes(scene["data"]) == before
    assert result["new_snapshot_status"] == "UNTRAINED" and not result["generated_masks_trained"]
    assert result["epochs_run"] == result["new_training_jobs"] == 0
    assert result["loader"]["status"] == "not_requested"
    assert result["class_mapping"] == mapping
    assert result["snapshot"]["class_mapping"]["0"] == mapping[scene["source_class"]]["id"]
    assert result["merged_child_ids"] == [scene["child"]]
    ids = [r["id"] for r in result["snapshot"]["annotation_versions"]]
    assert ids.count(scene["child"]) == ids.count(scene["kept"]) == 1
    assert not set(ids).intersection(scene["parents"] + scene["omitted"] + [scene["deleted"]])
    matched = scene["matched"]["id"]
    assert matched in result["snapshot"]["validation_image_ids"]
    assert matched not in result["snapshot"]["train_image_ids"]
    assert all(row["image_id"] == matched for row in result["snapshot"]["annotation_versions"] if row["id"] in (scene["child"], scene["kept"]))
    clone = Project(result["clone_path"])
    child = clone.get_annotation(scene["child"])
    assert np.array_equal(geometry_mask(child["geometry"], 64, 48), scene["expected_mask"])
    assert child["review_actor"] == "automated_qa" and not child["human_verified"]
    assert clone.state()["active_model_id"] is None
    loss = next(r for r in result["conversion_report"] if r["annotation_id"] == scene["child"])
    assert loss["lossy"] and loss["changed_pixels"] > 0
    assert len(result["old_model_snapshot_bindings"]) == 2
    assert all(c["unchanged"] for c in result["preserved_inputs"].values())
    saved = json.loads((tmp_path / "evidence/evidence.json").read_text())
    assert saved == result


@pytest.mark.parametrize("change,match", [
    ("not_accepted", "merged child"),
    ("different_original", "original-SHA"),
    ("training_role", "FIXED VALIDATION"),
    ("wrong_union", "exact union"),
    ("missing_generation_job", "completed job"),
    ("canonical_mismatch", "canonical image bytes"),
])
def test_fail_closed_without_mutating_inputs(scene, tmp_path, change, match):
    base, generated = scene["base"], scene["generated"]
    if change == "not_accepted":
        generated.annotate(generated.state()["images"][0]["id"], {"status": "draft"}, "automated_qa", aid=scene["child"])
    elif change == "different_original":
        with generated.edit(None, "synthetic_corruption", "automated_qa") as (db, state):
            state["images"][0]["sha256"] = "f" * 64
    elif change == "training_role":
        base.update_image(scene["matched"]["id"], {"role": "train"}, "automated_qa")
    elif change == "wrong_union":
        mask = np.zeros((48, 64), bool)
        mask[20:35, 20:40] = True
        generated.annotate(generated.state()["images"][0]["id"], {"geometry": {"type": "mask", "rle": encode_rle(mask)}, "status": "accepted"},
                           "automated_qa", aid=scene["child"])
        with generated.edit(None, "synthetic_unrecorded_correction", "automated_qa") as (db, state):
            row = generated.get_annotation(scene["child"], full=False)
            row.pop("manually_edited")
            generated._put(db, row)
    elif change == "missing_generation_job":
        path = scene["data"] / "jobs" / scene["generation_job"] / "job.json"
        job = evidence.read_json(path)
        job["status"] = "failed"
        atomic(path, job)
    elif change == "canonical_mismatch":
        with generated.edit(None, "synthetic_corruption", "automated_qa") as (db, state):
            state["images"][0]["normalized_sha256"] = "a" * 64
    before = evidence.tree_hashes(scene["data"])
    with pytest.raises(ValueError, match=match):
        run(scene, tmp_path / "failed")
    assert evidence.tree_hashes(scene["data"]) == before
    report = evidence.read_json(tmp_path / "failed/evidence.json")
    assert not report["passed"] and report["source_bytes_unchanged"]
    assert not (tmp_path / "failed/prepared_snapshot").exists()


def test_brush_corrected_child_uses_current_accepted_mask_not_parent_union(scene, tmp_path):
    generated = scene["generated"]
    original = generated.get_annotation(scene["child"])
    corrected = scene["expected_mask"].copy()
    corrected[2:5, 2:16] = False
    corrected[18:22, 50:58] = True
    geometry = {"type": "mask", "rle": encode_rle(corrected)}
    generated.annotate(original["image_id"], {"geometry": geometry}, "automated_qa", aid=scene["child"])
    draft = generated.get_annotation(scene["child"])
    assert draft["status"] == "draft" and draft["manually_edited"] is True
    generated.annotate(original["image_id"], {"status": "accepted"}, "automated_qa", aid=scene["child"])
    assert all(generated.get_annotation(aid)["status"] == "superseded" for aid in scene["parents"])
    before = evidence.tree_hashes(scene["data"])
    result = run(scene, tmp_path / "brush_corrected")
    assert result["passed"] and evidence.tree_hashes(scene["data"]) == before
    clone = Project(result["clone_path"])
    child = clone.get_annotation(scene["child"])
    assert child["geometry"] == geometry
    assert child["geometry"] != original["geometry"]
    assert np.array_equal(geometry_mask(child["geometry"], 64, 48), corrected)
    trace = result["source_lineage"][scene["child"]]["merge_geometry"]
    assert trace["manually_edited"] and not trace["equals_parent_union"]
    assert trace["parent_union_sha256"] == evidence.object_hash(original["geometry"])
    assert trace["current_canonical_sha256"] == evidence.object_hash(geometry)
    assert trace["current_accepted_geometry_used"]
    assert result["canonical_preexport_sha256"][scene["child"]] == evidence.object_hash(geometry)
    assert [r["id"] for r in result["snapshot"]["annotation_versions"]].count(scene["child"]) == 1
    assert result["new_snapshot_status"] == "UNTRAINED" and not result["generated_masks_trained"]


def test_refuses_overlapping_output_and_overwrite(scene, tmp_path):
    with pytest.raises(ValueError, match="separate"):
        run(scene, scene["data"] / "output")
    with pytest.raises(ValueError, match="fresh"):
        run(scene, tmp_path)
    assert not (scene["data"] / "output").exists()


@pytest.mark.parametrize("bad", [{}, {"wrong-source": {"id": uuid.uuid4().hex, "name": "New"}}])
def test_mapping_requires_exact_source_ids(scene, tmp_path, bad):
    with pytest.raises(ValueError, match="exactly"):
        run(scene, tmp_path / "bad-mapping", mapping=bad)


def test_no_reuse_of_old_class_id(scene, tmp_path):
    mapping = {scene["source_class"]: {"id": scene["base"].state()["classes"][0]["id"], "name": "New"}}
    with pytest.raises(ValueError, match="new and distinct"):
        run(scene, tmp_path / "old-class", mapping=mapping)


def test_invalidates_report_if_source_changes_during_run(scene, tmp_path, monkeypatch):
    original = evidence.build_snapshot

    def simultaneous_external_edit(*args, **kwargs):
        result = original(*args, **kwargs)
        # Simulate another process editing the input while verification runs.
        atomic(scene["base"].path / "concurrent-change.json", {"synthetic_fixture": True})
        return result

    monkeypatch.setattr(evidence, "build_snapshot", simultaneous_external_edit)
    with pytest.raises(ValueError, match="Input bytes changed"):
        run(scene, tmp_path / "concurrent")
    report = evidence.read_json(tmp_path / "concurrent/evidence.json")
    assert not report["passed"] and not report["source_bytes_unchanged"]
    assert report["changed_inputs"] == [str(scene["base"].path)]


def test_synthetic_core_merged_training_label_once_no_real_generated_training(scene, tmp_path):
    """Separate TRAIN-label regression; never moves the real validation image."""
    project = scene["base"]
    image = project.state()["images"][0]
    iid, cid = image["id"], project.state()["classes"][0]["id"]
    first = add(project, iid, {"type": "polygon", "points": [[22, 10], [32, 10], [32, 22], [22, 22]]}, cid)
    second = add(project, iid, {"type": "polygon", "points": [[32, 10], [44, 10], [44, 22], [32, 22]]}, cid)
    merge_selected(project, {"image_id": iid, "annotation_ids": [first, second], "confirm": True, "accept": True,
                             "expected_revision": project.state()["revision"]}, "automated_qa")
    child = next(r for r in project.annotations(iid) if r["source"].get("operation") == "exact_pixel_union")
    complete(project, iid)
    old_binding = copy.deepcopy(project.state()["models"])
    old_hashes = {r["snapshot_sha256"] for r in scene["old_snapshots"]}
    snapshot = build_snapshot(project, tmp_path / "synthetic_train_snapshot",
                              {"round_id": project.state()["rounds"][1]["id"], "qa_smoke": True, "allow_lossy": False})
    ids = [r["id"] for r in snapshot["receipt"]["annotation_versions"]]
    assert ids.count(child["id"]) == 1 and first not in ids and second not in ids
    assert iid in snapshot["receipt"]["train_image_ids"]
    assert len((tmp_path / "synthetic_train_snapshot/labels/train" / (iid + ".txt")).read_text().splitlines()) == 2
    assert not any(r["lossy"] for r in snapshot["receipt"]["conversion_report"])
    assert snapshot["snapshot_sha256"] not in old_hashes
    assert project.state()["models"] == old_binding


def test_optional_actual_ultralytics_cpu_label_loader(scene, tmp_path):
    runtime = os.environ.get("COMPAG_YOLO_TEST_PYTHON")
    if not runtime:
        pytest.skip("Set COMPAG_YOLO_TEST_PYTHON to an installed runtime; no downloads or model calls")
    result = run(scene, tmp_path / "real_cpu_loader", yolo_python=runtime)
    loader = result["loader"]
    assert loader["status"] == "passed" and loader["backend"] == "ultralytics.verify_image_label"
    assert not loader["cuda_initialized"] and not loader["model_loaded"] and loader["epochs_run"] == 0
    assert len(loader["images"]) == 6 and all(row["accepted"] for row in loader["images"])
