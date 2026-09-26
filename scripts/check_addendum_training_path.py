#!/usr/bin/env python3
"""Prepare UNTRAINED validation-path evidence from accepted generated masks.

Only a fresh output directory is written. Source projects are byte-copied before
opening SQLite; the base copy is then cloned with Catalog.backup/restore. Run
after review/merge/accept has finished and while the input projects are idle.
No model is instantiated, no job is submitted, and no epochs are run.

Optional --class-mapping JSON is {source_class_id: {id: NEW_UUID, name: LABEL,
color: OPTIONAL_HEX_COLOR}}. It must cover exactly the accepted source classes.
Without it, reproducible UUID5 IDs and deliberately different labels are used.
New classes precede existing classes, exercising changed model-index mapping.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

# Also support a source checkout without requiring an editable installation.
_source = Path(__file__).resolve().parents[1] / "src"
if (_source / "compag_annotator").is_dir():
    sys.path.insert(0, str(_source))

from compag_annotator.core.catalog import Catalog
from compag_annotator.core.projects import Project, clean_name
from compag_annotator.geometry import mask_union, polygon_for_yolo, validate_geometry
from compag_annotator.storage.files import atomic, digest, json_bytes, safe_child
from compag_annotator.training.snapshots import build_snapshot


def require(condition, message):
    if not condition:
        raise ValueError(message)


def object_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def tree_hashes(path):
    """File bytes only: reading changes atime on some filesystems."""
    path = Path(path)
    require(path.exists(), f"Missing evidence input: {path}")
    if path.is_file():
        require(not path.is_symlink(), f"Symbolic link is not an isolated input: {path}")
        return {".": digest(path)}
    result = {}
    for entry in sorted(path.rglob("*")):
        require(not entry.is_symlink(), f"Symbolic link is not an isolated input: {entry}")
        if entry.is_file():
            result[str(entry.relative_to(path))] = digest(entry)
    return result


def copy_project(path, destination, expected):
    # Never open source SQLite: even a normal SELECT can checkpoint a WAL on
    # close. Never hard-link project files, because the clone is writable.
    shutil.copytree(path, destination, copy_function=shutil.copy2)
    require(tree_hashes(destination) == expected, "Source changed while being copied; finish review and retry in a fresh directory")
    return Project(destination)


def class_mapping(rows, base, generated, supplied=None):
    source_ids = sorted({r["class_id"] for r in rows})
    source_classes = {c["id"]: c for c in generated["classes"] if not c["archived"]}
    require(all(cid in source_classes for cid in source_ids), "Accepted source class is missing or archived")
    mapping = supplied if supplied is not None else {
        cid: {"id": uuid.uuid5(uuid.NAMESPACE_URL, f"compag-evidence/{generated['id']}/{cid}").hex,
              "name": f"Evidence category {index + 1}", "color": "#36c8aa"}
        for index, cid in enumerate(source_ids)
    }
    require(isinstance(mapping, dict) and set(mapping) == set(source_ids), "Class mapping must cover exactly the accepted source class IDs")
    existing_ids = {c["id"] for c in base["classes"] + generated["classes"]}
    names = {c["name"].casefold() for c in base["classes"] if not c["archived"]}
    new_ids, result = set(), {}
    for source_id in source_ids:
        item = mapping[source_id]
        require(isinstance(item, dict) and {"id", "name"} <= item.keys(), "Each mapping needs an exact new UUID and label")
        cid = item["id"]
        require(isinstance(cid, str), "New class ID must be a UUID string")
        uuid.UUID(cid)
        require(cid not in existing_ids | new_ids, "Mapping IDs must be new and distinct")
        name = clean_name(item["name"])
        require(name == item["name"] and name.casefold() not in names, "Mapping labels must be exact and distinct from existing active labels")
        color = item.get("color", "#36c8aa")
        require(isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color), "Invalid mapping color")
        names.add(name.casefold())
        new_ids.add(cid)
        result[source_id] = {"id": cid, "name": name, "color": color}
    return result


def generated_rows(project, data_dir):
    """Require accepted canonical masks and trace each merge to persisted jobs."""
    state = project.state()
    rows = {r["id"]: r for r in project.annotations()}
    accepted = [r for r in rows.values() if r["status"] == "accepted"]
    require(accepted, "No accepted generated annotations yet")
    merged = [r for r in accepted if r["source"].get("operation") == "exact_pixel_union"]
    require(merged, "Accept an actual merged child before preparing evidence")
    image_ids = {r["image_id"] for r in accepted}
    require(len(image_ids) == 1, "Accepted generated rows must belong to exactly one original image")
    image = Project.image(state, next(iter(image_ids)))
    layers = {layer["id"]: layer for layer in state.get("generation_layers", [])}
    lineage, job_paths = {}, set()

    def visit(row, ancestry=()):
        aid = row["id"]
        require(aid not in ancestry, "Cyclic merge provenance")
        if aid in lineage:
            return project.geometry(row)
        require(row["image_id"] == image["id"], "Merge provenance crosses original images")
        geometry = project.geometry(row)
        require(geometry.get("type") == "mask" and isinstance(geometry.get("rle", {}).get("counts"), list),
                "Accepted/generated geometry must be canonical uncompressed COCO RLE")
        validate_geometry(geometry, image["width"], image["height"])
        source = row["source"]
        merge_geometry = None
        if source.get("operation") == "exact_pixel_union":
            parents = source.get("parents", [])
            require(len(parents) >= 2 and len({p["id"] for p in parents}) == len(parents), "Merge needs distinct recorded parents")
            geometries = []
            for parent in parents:
                item = rows.get(parent["id"])
                require(item is not None and item["status"] == "superseded" and item.get("superseded_by") == aid,
                        "Merged parent is missing, active, or superseded by a different child")
                geometries.append(visit(item, (*ancestry, aid)))
            parent_union = mask_union(geometries, image["width"], image["height"])
            unchanged_union = parent_union == geometry
            require(unchanged_union or row.get("manually_edited") is True,
                    "Merged child differs from the exact union without a recorded manual correction")
            # Brush corrections are authoritative accepted geometry. Parent OR
            # remains provenance, never a replacement for the corrected child.
            merge_geometry = {"parent_union_sha256": object_hash(parent_union),
                              "current_canonical_sha256": object_hash(geometry),
                              "equals_parent_union": unchanged_union,
                              "manually_edited": row.get("manually_edited") is True,
                              "current_accepted_geometry_used": True}
        else:
            require(source.get("operation") == "automatic_mask_generation" and source.get("kind") in ("sam2", "sam3"),
                    "Accepted row lacks automatic-generation lineage")
            layer = layers.get(row.get("layer_id"))
            require(layer is not None and layer["image_id"] == image["id"], "Generation layer is missing or belongs to another image")
            job_path = safe_child(data_dir / "jobs", source["job_id"]) / "job.json"
            job = read_json(job_path)
            require(job["kind"] == "generate" and job["status"] == "complete" and job["payload"]["project_id"] == state["id"],
                    "Generation provenance must refer to a completed job for this project")
            require(any(p["id"] == layer["id"] and p["image_id"] == image["id"] for p in job["payload"]["layers"]),
                    "Generation job does not contain the recorded layer and original image")
            job_paths.add(job_path.parent)
        lineage[aid] = {"row_sha256": object_hash(row), "canonical_sha256": object_hash(geometry),
                        "revision": row["revision"], "status": row["status"], "source": copy.deepcopy(source),
                        "review_actor": row.get("review_actor"), "human_verified": row.get("human_verified"),
                        "manually_edited": row.get("manually_edited", False)}
        if merge_geometry is not None:
            lineage[aid]["merge_geometry"] = merge_geometry
        return geometry

    full = []
    for row in accepted:
        require(row.get("review_actor") in ("human", "automated_qa"), "Accepted annotation lacks recorded review")
        full.append({**row, "geometry": visit(row)})
    return full, image, lineage, job_paths, dict(Counter(r["status"] for r in rows.values()))


def old_training(data_dir, base):
    jobs = []
    for path in sorted((data_dir / "jobs").glob("*/job.json")):
        job = read_json(path)
        if job["kind"] == "train" and job["payload"].get("project_id") == base["id"]:
            require(job["status"] == "complete", "Base must have only completed training jobs")
            jobs.append((path, job))
    require(len(jobs) == 2 and len(base["models"]) == 2, "Expected exactly two existing completed training jobs and two base models")
    records, protected, snapshots = [], set(), {}
    models = {m["id"]: m for m in base["models"]}
    for path, job in jobs:
        result = job["result"]
        snapshot_path = Path(result["dataset_snapshot"]).resolve()
        snapshot = read_json(snapshot_path)
        snapshot_sha = digest(snapshot_path)
        model = models[result["model_id"]]
        round_id = job["payload"]["round_id"]
        round_record = next(r for r in base["rounds"] if r["id"] == round_id)
        require(snapshot_sha == result["snapshot_sha256"] == model["training_settings"]["snapshot_sha256"]
                == round_record["dataset_snapshot_sha256"], "Existing model/round/job snapshot binding differs")
        require(snapshot["settings"]["round_id"] == round_id and round_record["trained_model_id"] == model["id"],
                "Existing round/model binding differs")
        require(snapshot["class_mapping"] == model["class_mapping"] == result["class_mapping"], "Existing class mapping binding differs")
        require(digest(result["checkpoint_path"]) == result["checkpoint_sha256"] == model["sha256"], "Existing checkpoint hash differs")
        for relative, sha in snapshot["file_hashes"].items():
            require(digest(safe_child(snapshot_path.parent, relative)) == sha, "Existing dataset file hash differs")
        require(round_id not in snapshots, "Existing jobs must belong to distinct rounds")
        snapshots[round_id] = snapshot
        protected.update((path.parent, snapshot_path.parent, Path(result["checkpoint_path"]).resolve()))
        records.append({"job_id": job["id"], "job_sha256": digest(path), "model_id": model["id"],
                        "model_record_sha256": object_hash(model), "checkpoint_sha256": model["sha256"],
                        "round_id": round_id, "snapshot_path": str(snapshot_path), "snapshot_sha256": snapshot_sha,
                        "class_mapping": model["class_mapping"], "binding_verified": True})
    return records, protected, snapshots


def cpu_loader(runtime, dataset, receipt, output):
    """Use the real backend's label parser, with CUDA initialization forbidden."""
    request = output / "loader_request.json"
    items = []
    for split, ids in (("train", receipt["train_image_ids"]), ("val", receipt["validation_image_ids"])):
        for iid in ids:
            images = list((dataset / "images" / split).glob(iid + ".*"))
            require(len(images) == 1, "Snapshot image path is ambiguous")
            label = dataset / "labels" / split / (iid + ".txt")
            lines = label.read_text().splitlines()
            items.append({"image": str(images[0]), "label": str(label), "count": len(lines),
                          "classes": [int(line.split()[0]) for line in lines]})
    atomic(request, {"items": items, "nc": len(receipt["classes"])})
    result_path = output / "loader_result.json"
    script = r'''
import json, socket, sys
from pathlib import Path
def forbidden(*args, **kwargs):
    raise RuntimeError("Network and CUDA initialization are forbidden in dataset verification")
socket.socket.connect = forbidden
socket.create_connection = forbidden
import torch
if torch.cuda.is_initialized():
    raise RuntimeError("CUDA was initialized before label verification")
torch.cuda.init = forbidden
torch.cuda._lazy_init = forbidden
import ultralytics
from ultralytics.data.utils import verify_image_label
request = json.loads(Path(sys.argv[1]).read_text())
rows = []
for item in request["items"]:
    result = verify_image_label((item["image"], item["label"], "", False, request["nc"], 0, 0, False))
    if result[0] != item["image"] or result[1].shape != (item["count"], 5) or len(result[3]) != item["count"]:
        raise RuntimeError("Backend rejected or dropped segmentation rows: " + str(result[-1]))
    if result[1][:, 0].astype(int).tolist() != item["classes"]:
        raise RuntimeError("Backend class indices differ")
    rows.append({**item, "accepted": True, "message": result[-1]})
if torch.cuda.is_initialized():
    raise RuntimeError("CUDA was initialized during label verification")
Path(sys.argv[2]).write_text(json.dumps({"status": "passed", "backend": "ultralytics.verify_image_label",
    "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
    "cuda_initialized": False, "epochs_run": 0, "model_loaded": False, "images": rows}, indent=2))
'''
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "YOLO_CONFIG_DIR": str(output / "yolo-settings"),
           "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false", "WANDB_MODE": "disabled",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([str(runtime), "-c", script, str(request), str(result_path)],
                               capture_output=True, text=True, timeout=60, env=env)
    (output / "loader_stdout.txt").write_text(completed.stdout)
    (output / "loader_stderr.txt").write_text(completed.stderr)
    require(completed.returncode == 0, "Real CPU backend loader failed; see loader_stderr.txt")
    return {**read_json(result_path), "runtime": str(runtime), "runtime_sha256": digest(runtime)}


def training_ast_audit(source_dir, baseline):
    """Report, never assume, whether relevant function ASTs match a git base."""
    paths = {
        "src/compag_annotator/providers/yolo.py": {"train", "train_settings", "bind_training_location"},
        "src/compag_annotator/models/manager.py": {"train"},
        "src/compag_annotator/training/snapshots.py": {"build_snapshot", "readiness"},
    }

    def functions(text):
        result = {}

        def visit(node, prefix=""):
            for child in ast.iter_child_nodes(node):
                name = prefix
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = prefix + child.name + "."
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        result[name[:-1]] = hashlib.sha256(ast.dump(child, include_attributes=False).encode()).hexdigest()
                visit(child, name)
        visit(ast.parse(text))
        return result

    audit = {"baseline": baseline, "files": {}, "status": "checked"}
    for relative, wanted in paths.items():
        try:
            current = functions((source_dir / relative).read_text())
            old = subprocess.run(["git", "show", f"{baseline}:{relative}"], cwd=source_dir,
                                 text=True, capture_output=True, check=True, timeout=10)
            previous = functions(old.stdout)
            keys = {k for k in current.keys() | previous.keys() if k.rsplit(".", 1)[-1] in wanted}
            audit["files"][relative] = {k: {"baseline_sha256": previous.get(k), "current_sha256": current.get(k),
                                            "unchanged": current.get(k) == previous.get(k)} for k in sorted(keys)}
        except (OSError, subprocess.SubprocessError, SyntaxError) as exc:
            audit["status"] = "partly_unavailable"
            audit["files"][relative] = {"error": str(exc)}
    return audit


def check(data_dir, base_project_id, generated_project_id, output_dir, *, mapping=None,
          yolo_python=None, baseline_ref="5d98629"):
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    require(data_dir.is_dir() and (data_dir / "projects.json").is_file(), "Choose an existing application data directory")
    require(not output_dir.exists(), "Choose a fresh output directory")
    require(not output_dir.is_relative_to(data_dir) and not data_dir.is_relative_to(output_dir), "Output must be separate from the source data directory")
    records = read_json(data_dir / "projects.json")
    paths = []
    for pid in (base_project_id, generated_project_id):
        found = [r for r in records if r["id"] == pid]
        require(len(found) == 1, "Project ID must resolve to exactly one catalog record")
        path = Path(found[0]["path"]).resolve()
        require(not output_dir.is_relative_to(path) and not path.is_relative_to(output_dir), "Output must be separate from input projects")
        paths.append(path)
    require(base_project_id != generated_project_id and paths[0] != paths[1], "Base and generated projects must be different")
    output_dir.mkdir(parents=True)
    protected = {}
    report = {"schema": "compag-addendum-dataset-evidence/v1", "passed": False,
              "scope": "Automated validation-image dataset integration; not training or human UAT",
              "new_snapshot_status": "UNTRAINED", "new_training_jobs": 0, "epochs_run": 0,
              "generated_masks_trained": False, "human_uat": False,
              "base_project_id": base_project_id, "generated_project_id": generated_project_id,
              "allow_lossy": True, "loss_consent": "Explicit automated conversion test only",
              "loader": {"status": "not_requested", "verified": False}}

    def protect(path):
        path = Path(path).resolve()
        protected.setdefault(str(path), tree_hashes(path))

    try:
        for path in [data_dir / "projects.json", *paths]:
            protect(path)
        inputs = output_dir / "input_copies"
        inputs.mkdir()
        base = copy_project(paths[0], inputs / "base", protected[str(paths[0])])
        generated = copy_project(paths[1], inputs / "generated", protected[str(paths[1])])
        base_state, generated_state = base.state(), generated.state()
        require(base_state["id"] == base_project_id and generated_state["id"] == generated_project_id, "Catalog and project IDs disagree")
        base_images = [i for i in base_state["images"] if not i.get("removed")]
        require(len(base_images) == 6, "Expected the preserved six-image base QA project")
        rounds = [r for r in base_state["rounds"] if r["number"] == 2]
        require(len(rounds) == 1 and rounds[0]["status"] == "review_complete", "Base round two must exist with complete review")
        round_id = rounds[0]["id"]
        old, old_paths, old_snapshots = old_training(data_dir, base_state)
        for path in old_paths:
            protect(path)
        report["old_model_snapshot_bindings"] = old
        old_snapshot = old_snapshots[round_id]
        accepted, source_image, lineage, generation_jobs, statuses = generated_rows(generated, data_dir)
        for path in generation_jobs:
            protect(path)
        matches = [i for i in base_images if i["sha256"] == source_image["sha256"]]
        require(len(matches) == 1, "Generated image needs one exact original-SHA match in the base project")
        matched = matches[0]
        require(matched["role"] == "validation" and matched["id"] in old_snapshot["validation_image_ids"],
                "Matching original image must retain its FIXED VALIDATION role")
        require(all(matched[k] == source_image[k] for k in ("width", "height", "normalized_sha256")),
                "Original match has different oriented dimensions or canonical image bytes")
        for project, image in ((base, matched), (generated, source_image)):
            require(digest(safe_child(project.path, image["path"])) == image["normalized_sha256"], "Canonical image hash differs")
            original = Path(image["original"]) if image["storage"] == "reference" else safe_child(project.path, image["original"])
            protect(original) if image["storage"] == "reference" else None
            require(digest(original) == image["sha256"], "Original image hash differs")
        selected_mapping = class_mapping(accepted, base_state, generated_state, mapping)
        report.update(source_status_counts=statuses, source_lineage=lineage, class_mapping=selected_mapping,
                      matched_image={"base_image_id": matched["id"], "generated_image_id": source_image["id"],
                                     "name": matched["name"], "original_sha256": matched["sha256"],
                                     "normalized_sha256": matched["normalized_sha256"], "role": "validation",
                                     "width": matched["width"], "height": matched["height"]})
        isolated_catalog = Catalog(output_dir / "catalog")
        archive = isolated_catalog.backup(base, output_dir / "base.native.zip", include_images=True)
        clone = isolated_catalog.restore(archive["artifact"])
        before = clone.annotations(full=True)
        imported_ids = {row["id"] for row in accepted}
        require(not imported_ids.intersection(row["id"] for row in before), "Source annotation IDs collide with base annotation IDs")
        with clone.edit(None, "dataset_evidence_import", "automated_qa") as (db, state):
            state["name"] = "Untrained generated-mask dataset evidence"
            new_classes = [{**value, "order": index, "archived": False, "shortcut": ""}
                           for index, value in enumerate(selected_mapping.values())]
            state["classes"] = new_classes + state["classes"]
            for index, label in enumerate(state["classes"]):
                label["order"] = index
            state["class_version"] += 1
            db.execute("DELETE FROM annotations WHERE image_id=?", (matched["id"],))
            for original in accepted:
                row = copy.deepcopy(original)
                row.update(image_id=matched["id"], class_id=selected_mapping[original["class_id"]]["id"],
                           revision=state["revision"] + 1, status="accepted", review_actor="automated_qa", human_verified=False)
                row["source"]["dataset_evidence_import"] = {"project_id": generated_project_id,
                    "image_id": source_image["id"], "annotation_id": original["id"], "revision": original["revision"],
                    "canonical_sha256": object_hash(original["geometry"])}
                row["geometry"] = clone._geometry(original["geometry"], matched)
                clone._put(db, row)
            Project.image(state, matched["id"]).update(complete=False, review_actor=None)
        clone.update_image(matched["id"], {"complete": True, "attest": True}, "automated_qa")
        after = clone.annotations(full=True)
        untouched = lambda rows: {r["id"]: r for r in rows if r["image_id"] != matched["id"]}
        require(untouched(before) == untouched(after), "Annotations on another photo changed")
        canonical = {r["id"]: object_hash(r["geometry"]) for r in after if r["id"] in imported_ids}
        require(canonical == {r["id"]: object_hash(r["geometry"]) for r in accepted}, "Canonical geometry changed during native-clone import")
        clone_state = clone.state()
        require(clone_state["rounds"] == base_state["rounds"], "Base round history changed in clone")
        for image in clone_state["images"]:
            original = next(i for i in base_state["images"] if i["id"] == image["id"])
            require(all(image[k] == original[k] for k in ("role", "group_id", "sha256", "normalized_sha256")), "Original split or image identity changed")
        settings = {"round_id": round_id, "qa_smoke": True, "allow_lossy": True, "boundary_opt_in": False,
                    "evidence_only": True}
        dataset = output_dir / "prepared_snapshot"
        snapshot = build_snapshot(clone, dataset, settings)
        receipt = snapshot["receipt"]
        for key in ("image_ids", "train_image_ids", "validation_image_ids", "image_groups"):
            require(receipt[key] == old_snapshot[key], f"Fixed cumulative snapshot membership changed: {key}")
        actual_ids = [r["id"] for r in receipt["annotation_versions"]]
        expected = {r["id"] for r in after if r["status"] == "accepted" and r["image_id"] in receipt["image_ids"]}
        require(len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == expected, "Snapshot duplicates or omits accepted instances")
        require(imported_ids <= set(actual_ids), "Imported accepted child missing from snapshot")
        require(all(r["status"] == "accepted" and r["id"] in imported_ids for r in after if r["image_id"] == matched["id"]),
                "Unaccepted/superseded/old QA rows entered the matched image")
        for source_id, value in selected_mapping.items():
            require(value["id"] in receipt["class_mapping"].values(), "New stable class missing from snapshot")
            require(any(c["id"] == value["id"] and c["name"] == value["name"] for c in receipt["classes"]), "New class label changed")
        index_by_id = {cid: index for index, cid in receipt["class_mapping"].items()}
        lines = []
        for row in after:
            if row["image_id"] != matched["id"]:
                continue
            points, _ = polygon_for_yolo(row["geometry"], matched["width"], matched["height"], allow_lossy=True)
            values = [v for x, y in points for v in (float(x) / matched["width"], float(y) / matched["height"])]
            lines.append(index_by_id[row["class_id"]] + " " + " ".join(format(v, ".17g") for v in values))
        label = dataset / "labels" / "val" / (matched["id"] + ".txt")
        require(label.read_text().splitlines() == lines, "Actual validation labels differ from the audited shared converter")
        require(canonical == {r["id"]: object_hash(clone.get_annotation(r["id"])["geometry"]) for r in accepted},
                "Pre-export canonical geometry was mutated by conversion")
        for model in clone_state["models"]:
            original = next(m for m in base_state["models"] if m["id"] == model["id"])
            require(model["training_settings"] == original["training_settings"] and model["class_mapping"] == original["class_mapping"],
                    "Old model metadata was rebound to the new snapshot")
            require(model["training_settings"]["snapshot_sha256"] != snapshot["snapshot_sha256"], "Prepared snapshot unexpectedly bound to a trained model")
        report.update(native_backup=archive, clone_project_id=clone_state["id"], clone_path=str(clone.path),
                      round_id=round_id, canonical_preexport_sha256=canonical,
                      replaced_base_annotation_ids=[r["id"] for r in before if r["image_id"] == matched["id"]],
                      imported_annotation_ids=sorted(imported_ids),
                      merged_child_ids=[r["id"] for r in accepted if r["source"].get("operation") == "exact_pixel_union"],
                      snapshot_sha256=snapshot["snapshot_sha256"], snapshot=receipt,
                      validation_label_path=str(label), validation_label_sha256=digest(label),
                      conversion_report=receipt["conversion_report"],
                      training_path_limitation="Imported generated masks occur only in fixed validation labels. No generated mask was trained. The separate synthetic core test covers a merged training-label row.",
                      checks={"canonical_preexport_equal": True, "unrelated_photo_annotations_unchanged": True,
                              "fixed_splits_unchanged": True, "same_cumulative_round_two": True,
                              "accepted_child_once": True, "no_unaccepted_or_superseded_imports": True,
                              "old_models_bound_to_old_snapshots": True, "new_snapshot_untrained": True})
        if yolo_python is not None:
            report["loader"] = cpu_loader(Path(yolo_python).absolute(), dataset, receipt, output_dir)
        report["training_function_ast"] = training_ast_audit(Path(__file__).resolve().parents[1], baseline_ref)
        report["tool_sha256"] = digest(__file__)
        report["passed"] = True
    except BaseException as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "loss_report"):
            report["failure_loss_report"] = exc.loss_report
        raise
    finally:
        differences = []
        for path, before_hashes in protected.items():
            try:
                after_hashes = tree_hashes(path)
            except (ValueError, OSError) as exc:
                after_hashes = {"error": str(exc)}
            if after_hashes != before_hashes:
                differences.append(path)
            report.setdefault("preserved_inputs", {})[path] = {"before": before_hashes, "after": after_hashes,
                                                               "unchanged": before_hashes == after_hashes}
        report["source_bytes_unchanged"] = not differences
        if differences:
            report.update(passed=False, preservation_error="Input changed during verification; evidence is invalid", changed_inputs=differences)
        report["output_file_sha256"] = tree_hashes(output_dir)
        atomic(output_dir / "evidence.json", report)
        if differences:
            raise ValueError("Input bytes changed during verification; see evidence.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--base-project-id", required=True)
    parser.add_argument("--generated-project-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--class-mapping", type=Path, help="Exact source-class-ID to new UUID/label JSON mapping")
    parser.add_argument("--yolo-python", type=Path, help="Already installed Ultralytics Python; CPU label verification only")
    parser.add_argument("--baseline-ref", default="5d98629", help="Read-only git AST comparison base")
    args = parser.parse_args()
    try:
        result = check(args.data_dir, args.base_project_id, args.generated_project_id, args.output_dir,
                       mapping=read_json(args.class_mapping) if args.class_mapping else None,
                       yolo_python=args.yolo_python, baseline_ref=args.baseline_ref)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    print(json.dumps({"passed": result["passed"], "snapshot_status": result["new_snapshot_status"],
                      "generated_masks_trained": False, "evidence": str(args.output_dir.resolve() / "evidence.json")}))


if __name__ == "__main__":
    main()
