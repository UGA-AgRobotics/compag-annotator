"""Boundary software regressions; synthetic masks and direct grants, no ML jobs."""
import copy
import json

import numpy as np
from PIL import Image
import pytest

from compag_annotator.boundary.preparation import BoundaryPreparation
from compag_annotator.core.generation import identity
from compag_annotator.core.projects import Project, ConflictError
from compag_annotator.geometry import encode_rle, mask_metadata
from compag_annotator.jobs.runner import AwaitingReview
from compag_annotator.providers.protocol import CancelledError


def mask(x0=20, y0=20, x1=30, y1=30):
    pixels = np.zeros((80, 100), dtype=bool)
    pixels[y0:y1, x0:x1] = True
    return {"type": "mask", "rle": encode_rle(pixels)}


@pytest.fixture
def scene(tmp_path):
    project = Project.create(tmp_path / "project", "Boundary software fixtures", [])
    images = []
    for number, color in enumerate(("white", "black")):
        path = tmp_path / f"image-{number}.png"
        Image.new("RGB", (100, 80), color).save(path)
        images.append(path)
    project.add_images(images)
    return project, BoundaryPreparation(), [image["id"] for image in project.state()["images"]]


def generated_row(project, image_id, *, geometry=None, class_id=None):
    """Persist a synthetic generator-shaped row; never assert actual SAM output.

    Tests exercise BoundaryPreparation through its server-created grant, not
    through the public endpoint's generation-membership eligibility check.
    """
    geometry = geometry or mask()
    previous = {row["id"] for row in project.annotations(image_id)}
    project.annotate(image_id, {
        "expected_revision": project.state()["revision"], "class_id": class_id,
        "status": "proposal", "geometry": geometry,
        "source": {"kind": "sam2", "operation": "automatic_mask_generation", "tile": [0, 0, 30, 30],
                   "generation_id": "synthetic-generation", "model_id": "synthetic-model"},
    }, "automated_qa")
    row = next(row for row in project.annotations(image_id) if row["id"] not in previous)
    with project.edit(None, "synthetic_generation_fixture", "automated_qa") as (db, state):
        row.update(layer_id="synthetic-layer", generation_signature=identity(geometry),
                   generated_revision=row["revision"], **mask_metadata(geometry, 100, 80))
        project._put(db, row)
    return row["id"]


class CandidateDouble:
    """Synthetic provider results; no runtime, checkpoint, training or GPU calls."""
    def __init__(self, primary=None, alternatives=None, error=None):
        self.primary = [mask(x1=33)] if primary is None else primary
        self.alternatives = [] if alternatives is None else alternatives
        self.error = error
        self.calls = 0

    def models(self):
        return [{"id": "synthetic-model", "provider": "sam2", "trusted": True}]

    def infer(self, *args, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return {
            "annotations": [{"geometry": copy.deepcopy(value), "predicted_iou": .95, "stability_score": .96} for value in self.primary],
            "alternatives": [{"geometry": copy.deepcopy(value), "predicted_iou": .95, "stability_score": .96} for value in self.alternatives],
            "loaded_checkpoint_sha256": "synthetic-checkpoint-identity",
            "timings": {"prompt_seconds": 0.0},
        }


def stage(project, boundary, annotation_id, manager=None):
    manager = manager or CandidateDouble()
    grant = boundary.authorize(project, job_kind="boundary_annotation", job_id="synthetic-job",
                               annotation_ids=[annotation_id], explicit_opt_in=True, limit=1)
    with pytest.raises(AwaitingReview) as event:
        boundary.prepare(project, manager, {
            "phase": "automatic_mask_annotation", "boundary_opt_in": True,
            "boundary_model_id": "synthetic-model", "device": "cpu",
        }, lambda event: None, lambda: False, authorization=grant)
    assert event.value.result["model_calls"] == manager.calls == 1
    return next(p for p in project.state()["boundary_proposals"]
                if p["annotation_id"] == annotation_id and p["status"] == "pending")


def review(project, boundary, proposal, decision):
    return boundary.review(project, {
        "proposal_id": proposal["id"], "decision": decision, "attest": True,
        "expected_revision": project.state()["revision"],
    }, "automated_qa")


def suggestion(project, proposal):
    return next(p for p in project.state()["boundary_proposals"] if p["id"] == proposal["id"])


def history(project, image_id, action):
    reloaded = Project(project.path)
    reloaded.history(image_id, {"action": action, "expected_revision": reloaded.state()["revision"]}, "automated_qa")
    return reloaded


@pytest.mark.parametrize("valid_primary", [True, False])
def test_primary_and_alternatives_are_both_considered(scene, valid_primary):
    project, boundary, images = scene
    aid = generated_row(project, images[0])
    original = project.get_annotation(aid)
    valid, invalid = mask(x1=33), mask(70, 60, 80, 70)
    manager = CandidateDouble(primary=[valid if valid_primary else invalid],
                              alternatives=[invalid if valid_primary else valid])
    proposal = stage(project, boundary, aid, manager)
    assert proposal["geometry"] == valid
    assert project.get_annotation(aid) == original  # Staging never applies geometry.


def test_primary_neighbour_capture_is_rejected_before_alternative_selection(scene):
    project, boundary, images = scene
    aid = generated_row(project, images[0])
    generated_row(project, images[0], geometry=mask(30, 20, 33, 30))
    manager = CandidateDouble(primary=[mask(x1=33)], alternatives=[mask()])
    proposal = stage(project, boundary, aid, manager)
    assert proposal["geometry"] == mask()
    assert proposal["neighbour_conflicts"] == 1


@pytest.mark.parametrize("assigned", [False, True])
def test_accept_metadata_tombstone_and_boundary_history_survive_restart(scene, assigned):
    project, boundary, images = scene
    cid = None
    if assigned:
        project.class_change({"action": "create", "name": "Object", "expected_revision": project.state()["revision"]}, "automated_qa")
        cid = project.state()["classes"][0]["id"]
    aid = generated_row(project, images[0], class_id=cid)
    original = project.get_annotation(aid)
    proposal = stage(project, boundary, aid)
    review(project, boundary, proposal, "accept")
    current = project.get_annotation(aid)
    assert current["geometry"] == mask(x1=33)
    assert current["area"] == 130 and current["bbox"] == [20, 20, 33, 30]
    assert current["status"] == ("accepted" if assigned else "draft")
    assert current["class_id"] == cid and current["human_verified"] is False
    assert current["geometry_review_actor"] == "automated_qa" and current["manually_edited"]
    assert "generation_signature" not in current
    assert current["source"]["boundary_reviews"][-1]["model_sha256"] == "synthetic-checkpoint-identity"
    tombstones = project.state()["images"][0]["generation_tombstones"]["synthetic-layer"]
    assert tombstones[original["generation_signature"]] == {"annotation_id": aid, "reason": "boundary_accept"}
    assert suggestion(project, proposal)["annotation_revision"] == current["revision"]

    project = history(project, images[0], "undo")
    restored = project.get_annotation(aid)
    assert restored["geometry"] == original["geometry"] and restored["generation_signature"] == original["generation_signature"]
    assert not project.state()["images"][0].get("generation_tombstones")
    assert suggestion(project, proposal)["status"] == "pending"
    assert suggestion(project, proposal)["annotation_revision"] == restored["revision"]

    project = history(project, images[0], "redo")
    assert project.get_annotation(aid)["geometry"] == mask(x1=33)
    assert suggestion(project, proposal)["status"] == "accepted"
    assert suggestion(project, proposal)["annotation_revision"] == project.get_annotation(aid)["revision"]
    assert project.state()["images"][0]["generation_tombstones"]["synthetic-layer"] == tombstones


@pytest.mark.parametrize("decision,expected_status", [("retain", "retained"), ("exclude", "excluded")])
def test_retain_and_exclude_have_their_own_atomic_undo_redo(scene, decision, expected_status):
    project, boundary, images = scene
    aid = generated_row(project, images[0])
    original = project.get_annotation(aid)
    proposal = stage(project, boundary, aid)
    review(project, boundary, proposal, decision)
    assert project.get_annotation(aid) == original
    assert suggestion(project, proposal)["status"] == expected_status
    assert bool(project.state()["images"][0].get("training_excluded")) == (decision == "exclude")
    with project.connection() as db:
        latest = db.execute("SELECT action,previous,current FROM history ORDER BY seq DESC LIMIT 1").fetchone()
    assert latest[0] == "boundary_" + decision
    assert json.loads(latest[1])["boundary_proposals"][0]["status"] == "pending"
    assert json.loads(latest[2])["boundary_proposals"][0]["status"] == expected_status

    project = history(project, images[0], "undo")
    assert project.get_annotation(aid)["geometry"] == original["geometry"]
    assert not project.state()["images"][0].get("training_excluded")
    assert suggestion(project, proposal)["status"] == "pending"
    assert suggestion(project, proposal)["annotation_revision"] == project.get_annotation(aid)["revision"]
    project = history(project, images[0], "redo")
    assert project.get_annotation(aid)["geometry"] == original["geometry"]
    assert suggestion(project, proposal)["status"] == expected_status
    assert bool(project.state()["images"][0].get("training_excluded")) == (decision == "exclude")


@pytest.mark.parametrize("change", ["edit", "delete", "reject", "removed_image"])
def test_stale_retain_dismisses_without_geometry_and_stale_accept_exclude_fail(scene, change):
    project, boundary, images = scene
    aid = generated_row(project, images[0])
    proposal = stage(project, boundary, aid)
    if change == "removed_image":
        with project.edit(None, "synthetic_remove_image", "automated_qa") as (db, state):
            project.image(state, images[0])["removed"] = True
    else:
        updates = {"geometry": mask(21, 21, 29, 29)} if change == "edit" else {"status": "rejected"} if change == "reject" else {}
        project.annotate(images[0], {**updates, "expected_revision": project.state()["revision"]},
                         "automated_qa", aid, delete=change == "delete")
    unchanged = project.annotations(full=True)
    previous = project.state()
    for decision in ("accept", "exclude"):
        with pytest.raises(ValueError, match="stale.*retain"):
            review(project, boundary, proposal, decision)
        assert project.state() == previous and project.annotations(full=True) == unchanged
    review(project, boundary, proposal, "retain")
    dismissed = suggestion(project, proposal)
    assert dismissed["status"] == "dismissed_stale" and dismissed["decision"] == "retain"
    assert dismissed["annotation_revision"] == proposal["annotation_revision"]
    assert dismissed["geometry"] == proposal["geometry"]  # Preserved only as historical evidence.
    assert project.annotations(full=True) == unchanged
    assert not project.state()["images"][0].get("training_excluded")
    assert not any(p["status"] == "pending" for p in project.state()["boundary_proposals"])
    if change != "removed_image":
        project = history(project, images[0], "undo")
        assert suggestion(project, proposal)["status"] == "pending"
        if change != "delete":
            assert suggestion(project, proposal)["annotation_revision"] != project.get_annotation(aid)["revision"]
        else:
            assert not project.annotations(images[0])
        project = history(project, images[0], "redo")
        assert suggestion(project, proposal)["status"] == "dismissed_stale"
        assert {row["id"] for row in project.annotations()} == {row["id"] for row in unchanged}


def test_stale_dismissal_releases_training_preflight_without_running_training(scene, monkeypatch, tmp_path):
    from compag_annotator.web.service import Service
    import compag_annotator.web.service as service_module
    project, boundary, images = scene
    aid = generated_row(project, images[0])
    proposal = stage(project, boundary, aid)
    project.annotate(images[0], {"geometry": mask(x1=31), "expected_revision": project.state()["revision"]}, "automated_qa", aid)
    service = Service.__new__(Service)
    service.qa_mode = True
    service.project = lambda pid: project
    class SnapshotReached(Exception):
        pass
    def stop_at_snapshot(*args, **kwargs):
        raise SnapshotReached("Software-only preflight reached; no snapshot or training executed")
    monkeypatch.setattr(service_module, "build_snapshot", stop_at_snapshot)
    body = {"project_id": project.state()["id"], "qa_smoke": True}
    with pytest.raises(AwaitingReview):
        service.work("train", dict(body), tmp_path / "unused", lambda event: None, lambda: False)
    review(project, boundary, proposal, "retain")
    with pytest.raises(SnapshotReached):
        service.work("train", dict(body), tmp_path / "unused", lambda event: None, lambda: False)


@pytest.mark.parametrize("decision", ["accept", "retain", "exclude"])
def test_history_failure_rolls_back_geometry_decision_exclusion_and_tombstones(scene, monkeypatch, decision):
    project, boundary, images = scene
    aid = generated_row(project, images[0])
    proposal = stage(project, boundary, aid)
    previous, rows = project.state(), project.annotations(full=True)
    def history_failure(*args, **kwargs):
        assert kwargs["state"]["boundary_proposals"][0]["status"] != "pending"
        raise RuntimeError("Explicit synthetic history failure")
    monkeypatch.setattr(project, "_history", history_failure)
    with pytest.raises(RuntimeError, match="history failure"):
        review(project, boundary, proposal, decision)
    assert project.state() == previous and project.annotations(full=True) == rows


def test_boundary_history_does_not_restore_other_images_decisions(scene):
    project, boundary, images = scene
    first = stage(project, boundary, generated_row(project, images[0]))
    second = stage(project, boundary, generated_row(project, images[1]))
    review(project, boundary, first, "exclude")
    review(project, boundary, second, "retain")
    other = suggestion(project, second)
    project = history(project, images[0], "undo")
    assert suggestion(project, first)["status"] == "pending"
    assert suggestion(project, second) == other
    project = history(project, images[0], "redo")
    assert suggestion(project, first)["status"] == "excluded"
    assert suggestion(project, second) == other


def test_review_requires_attestation_and_current_project_revision(scene):
    project, boundary, images = scene
    proposal = stage(project, boundary, generated_row(project, images[0]))
    before = project.state()
    body = {"proposal_id": proposal["id"], "decision": "retain", "expected_revision": before["revision"]}
    with pytest.raises(ValueError, match="Explicit review"):
        boundary.review(project, body, "automated_qa")
    with pytest.raises(ConflictError):
        boundary.review(project, {**body, "attest": True, "expected_revision": before["revision"]-1}, "automated_qa")
    assert project.state() == before


@pytest.mark.parametrize("error_type", [RuntimeError, InterruptedError, CancelledError])
def test_inference_failure_and_cancellation_keep_existing_attempt_receipts(scene, error_type):
    project, boundary, images = scene
    aid = generated_row(project, images[0])
    original = project.get_annotation(aid)
    error = error_type("Explicit software failure")
    manager = CandidateDouble(error=error)
    grant = boundary.authorize(project, job_kind="boundary_annotation", job_id="synthetic-job",
                               annotation_ids=[aid], explicit_opt_in=True)
    with pytest.raises(error_type) as caught:
        boundary.prepare(project, manager, {"phase": "automatic_mask_annotation", "boundary_opt_in": True,
                         "boundary_model_id": "synthetic-model"}, lambda event: None, lambda: False, authorization=grant)
    assert caught.value.result["model_calls"] == boundary.model_calls == manager.calls == 1
    assert caught.value.result["original_masks_preserved"]
    assert project.get_annotation(aid) == original and project.state()["boundary_proposals"] == []
