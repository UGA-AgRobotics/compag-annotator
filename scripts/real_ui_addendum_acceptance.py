#!/usr/bin/env python3
"""UI acceptance on an existing, completed real-SAM proposal project.

No images, masks, model results or labels are fabricated. All edits are performed
through the installed browser UI; API access is read-only verification. Requires
Playwright/Chromium, numpy and pycocotools in the invoking Python environment.

Default invocation prints a plan without contacting the server. --preflight is
read-only. --execute is explicit permission to modify the supplied QA project
AFTER its coordinator has declared generation complete and the new UI ready.
Use a fresh private output directory outside the source checkout. Failures retain
screenshots, a baseline and a mutation journal; no automatic cleanup hides them.
QA classes describe the test only, not scientific labels or human ground truth.
Export completion rejects remaining proposals as automated QA and marks this QA
image complete. --continue-from resumes that step after saved acceptance evidence;
it never repeats class creation or geometry edits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import traceback
from urllib.parse import urlsplit
import zipfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def decode_mask(geometry, width, height):
    """Independent COCO decoder; never uses the app's union/brush implementation."""
    import numpy as np
    from pycocotools import mask as coco

    rle = geometry.get("rle") if geometry.get("type") == "mask" else geometry
    require(
        isinstance(rle, dict) and rle.get("size") == [height, width], "Expected a full-image canonical mask"
    )
    counts = rle.get("counts")
    if isinstance(counts, list):
        require(
            all(type(n) is int and n >= 0 for n in counts) and sum(counts) == width * height,
            "Invalid canonical RLE counts",
        )
        encoded = coco.frPyObjects(rle, height, width)
    else:
        require(isinstance(counts, str), "Unsupported COCO RLE representation")
        encoded = {"size": [height, width], "counts": counts.encode("ascii")}
    raster = coco.decode(encoded)
    require(raster.shape == (height, width), "Decoded dimensions disagree with the source image")
    return np.asarray(raster, dtype=bool)


def pixel_receipt(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    require(len(xs) > 0, "A selected real mask is empty")
    return {
        "area": int(mask.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "pixel_sha256": digest(mask.tobytes(order="C")),
        "size": list(mask.shape),
    }


def interior_point(mask, exclude=None):
    """Choose an actual foreground point; no mask is drawn or synthesized here."""
    import numpy as np

    valid = mask if exclude is None else mask & ~exclude
    ys, xs = np.nonzero(valid)
    require(len(xs), "No distinct clickable foreground remains in this real mask")
    best, score = None, -1
    # Bounded samples favor thick foreground over sub-pixel display fragments.
    for index in np.linspace(0, len(xs) - 1, min(384, len(xs)), dtype=int):
        x, y = int(xs[index]), int(ys[index])
        radius = min(12, x, y, mask.shape[1] - x - 1, mask.shape[0] - y - 1)
        patch = valid[y - radius : y + radius + 1, x - radius : x + radius + 1]
        value = int(patch.sum()) if patch.all() else int(patch.sum()) / (patch.size + 1)
        if value > score:
            best, score = (x + 0.5, y + 0.5), value
    return best


def semantic_row(row):
    geometry = row.get("geometry", {})
    return {
        "id": row["id"],
        "class_id": row.get("class_id"),
        "status": row["status"],
        "geometry": geometry.get("sha256") or digest(json.dumps(geometry, sort_keys=True).encode()),
        "human_verified": row.get("human_verified", False),
        "review_actor": row.get("review_actor"),
        "source": row.get("source"),
    }


class Acceptance:
    def __init__(self, args, browser):
        self.args = args
        self.output = args.output.resolve()
        self.base = "/api/projects/" + args.project_id
        self.image_base = self.base + "/images/" + args.image_id
        self.origin = args.url.rstrip("/")
        self.deadline = time.monotonic() + args.timeout_seconds
        self.record = {
            "scope": "REAL_GENERATED_ROWS_INSTALLED_UI_AUTOMATED_QA",
            "human_uat": "AWAITING_HUMAN_UAT",
            "scientific_ground_truth": False,
            "review_actor": "automated_qa",
            "passed": False,
            "started_at": utc_now(),
            "project_id": args.project_id,
            "image_id": args.image_id,
            "server": self.origin,
            "stages": [],
            "screenshots": [],
            "mutations": [],
            "failures": [],
            "browser_errors": [],
            "blocked_requests": [],
            "inference_training_weight_requests": 0,
        }
        self.context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
        self.context.set_default_timeout(args.action_timeout_seconds * 1000)
        self.context.route("**/*", self.guard_request)
        self.page = self.context.new_page()
        self.page.on("pageerror", lambda error: self.record["browser_errors"].append(str(error)))
        self.geometry_responses = set()
        self.page.on("response", self.observe_response)
        self.stage = "initialization"
        self.token = None

    def save(self):
        write_json(self.output / "evidence.json", self.record)

    def get(self, path, binary=False):
        require(path.startswith("/api/"), "Verification can only read a local API path")
        require(time.monotonic() < self.deadline, "Acceptance wall-time limit reached")
        response = self.context.request.get(self.origin + path)
        try:
            require(response.ok, f"GET {path} failed with HTTP {response.status}")
            return response.body() if binary else response.json()
        finally:
            response.dispose()

    def scene(self):
        return self.get(self.image_base + "/annotations")["annotations"]

    def project(self):
        return self.get(self.base)

    def geometry(self, row):
        return self.get(row.get("geometry_url") or self.base + "/annotations/" + row["id"] + "/geometry")

    def raster(self, row):
        return decode_mask(self.geometry(row), self.image["width"], self.image["height"])

    def observe_response(self, response):
        path = urlsplit(response.url).path
        if response.status == 200 and path.endswith("/geometry"):
            self.geometry_responses.add(path)

    def guard_request(self, route):
        request = route.request
        parsed = urlsplit(request.url)
        if parsed.scheme not in {"http", "https"}:
            route.continue_()
            return
        same_origin = (
            parsed.netloc == urlsplit(self.origin).netloc and parsed.scheme == urlsplit(self.origin).scheme
        )
        if not same_origin:
            self.record["blocked_requests"].append(
                {"method": request.method, "path": parsed.path, "reason": "external origin"}
            )
            route.abort()
            return
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            route.continue_()
            return
        path = parsed.path
        allowed_posts = {
            self.base + "/classes",
            self.base + "/selection",
            self.base + "/merge/preview",
            self.base + "/merge",
            self.image_base + "/history",
            self.image_base + "/complete",
            self.base + "/exports",
        }
        annotation_patch = request.method == "PATCH" and re.fullmatch(
            re.escape(self.base) + r"/annotations/[a-zA-Z0-9_-]+", path
        )
        allowed = self.args.execute and (
            request.method == "POST" and path in allowed_posts or annotation_patch
        )
        if not allowed:
            self.record["blocked_requests"].append(
                {
                    "method": request.method,
                    "path": path,
                    "reason": "outside authorized UI acceptance mutations",
                }
            )
            if re.search(r"/(generate|assist|predict|train|models)(/|$)", path):
                self.record["inference_training_weight_requests"] += 1
            self.save()
            route.abort()
            return
        body = request.post_data_json or {}
        if "geometry" in body:
            body = {
                **body,
                "geometry": {
                    "type": body["geometry"].get("type"),
                    "json_sha256": digest(json.dumps(body["geometry"], sort_keys=True).encode()),
                },
            }
        self.record["mutations"].append(
            {"at": utc_now(), "stage": self.stage, "method": request.method, "path": path, "body": body}
        )
        self.save()
        route.continue_()

    @contextmanager
    def step(self, name):
        self.stage = name
        entry = {"name": name, "started_at": utc_now(), "passed": False}
        self.record["stages"].append(entry)
        start = time.monotonic()
        self.save()
        try:
            require(time.monotonic() < self.deadline, "Acceptance wall-time limit reached")
            yield entry
            require(not self.record["browser_errors"], "Browser reported uncaught errors")
            require(not self.record["blocked_requests"], "The browser attempted an unauthorized request")
            entry["passed"] = True
        finally:
            entry["wall_seconds"] = round(time.monotonic() - start, 3)
            self.save()

    def screenshot(self, name):
        target = self.output / (name + ".png")
        self.page.screenshot(path=str(target), full_page=True)
        self.record["screenshots"].append(
            {"file": target.name, "stage": self.stage, "sha256": digest(target.read_bytes())}
        )
        self.save()

    def wait_until(self, predicate, message, seconds=None):
        end = min(self.deadline, time.monotonic() + (seconds or self.args.action_timeout_seconds))
        while time.monotonic() < end:
            if predicate():
                return
            self.page.wait_for_timeout(80)
        raise TimeoutError(message)

    def saved(self):
        from playwright.sync_api import expect

        expect(self.page.get_by_text("All changes saved", exact=True)).to_be_visible()

    def ui_request(self, path, action, method="POST"):
        with self.page.expect_response(
            lambda r: urlsplit(r.url).path == path and r.request.method == method
        ) as response:
            action()
        response = response.value
        require(response.ok, f"UI {method} {path} failed: HTTP {response.status}: {response.text()[:1000]}")
        return response.json()

    def open_editor(self, reload=False):
        from playwright.sync_api import expect

        if reload:
            self.page.reload()
        else:
            self.page.goto(self.origin)
        expect(self.page.get_by_role("heading", name="Projects", exact=True)).to_be_visible()
        card = self.page.locator("article").filter(
            has=self.page.get_by_role("heading", name=self.initial_project["name"], exact=True)
        )
        if card.count() > 1:
            card = card.filter(has_text=self.initial_project["path"])
        require(card.count() == 1, "Cannot uniquely identify the requested project in the real project list")
        card.get_by_role("button", name="Open workspace", exact=True).click()
        self.page.get_by_role("button", name="Annotate " + self.image["name"], exact=True).click()
        self.saved()
        expect(self.page.get_by_role("button", name="Merge selected masks", exact=True)).to_be_visible()
        self.page.get_by_role("button", name="Fit", exact=True).click()
        self.page.wait_for_function(
            "() => {const c=document.querySelector('canvas.annotation-canvas');return c?.width>0 && c.getContext('2d').getImageData(Math.floor(c.width/2),Math.floor(c.height/2),1,1).data[3]>0;}"
        )
        self.wait_until(
            lambda: self.page.locator(".object-row").count() == len(self.scene()),
            "Object list did not load the real scene",
        )

    def select_ids(self, ids):
        self.page.get_by_role("button", name="Clear selection", exact=True).click()
        for annotation_id in ids:
            row = self.page.locator(f'.object-row[data-object-id="{annotation_id}"]')
            row.get_by_role("checkbox", name=re.compile("Include object .* in selection")).check()
        require(
            set(self.selected_ids()) == set(ids), "Object-list selection did not match the requested real IDs"
        )

    def selected_ids(self):
        return self.page.locator(".object-row").evaluate_all(
            "rows => rows.filter(r => r.querySelector('input[id^=object-select-]')?.checked).map(r=>r.dataset.objectId)"
        )

    def canvas_point(self, original):
        box = self.page.get_by_label("Full image annotation canvas", exact=True).bounding_box()
        require(box and box["width"] > 50 and box["height"] > 50, "Full-image canvas is not visible")
        scale = min((box["width"] - 50) / self.image["width"], (box["height"] - 50) / self.image["height"])
        return (
            box["x"] + (box["width"] - self.image["width"] * scale) / 2 + original[0] * scale,
            box["y"] + (box["height"] - self.image["height"] * scale) / 2 + original[1] * scale,
        )

    def preflight(self):
        with self.step("read_only_real_generation_preflight") as result:
            session = self.get("/api/session")
            self.token = session.get("token")
            require(
                session.get("qa_mode") is True and session.get("actor") == "automated_qa",
                "Server must run in automated QA mode",
            )
            self.record["served_asset_sha256"] = {}
            for name in ("editor.js", "canvas.js", "selection.js", "masks.js", "generation.js", "tiling.js"):
                response = self.context.request.get(self.origin + "/assets/" + name)
                try:
                    require(response.ok, "Installed UI is missing " + name)
                    self.record["served_asset_sha256"][name] = digest(response.body())
                finally:
                    response.dispose()
            self.initial_project = self.project()
            require(
                self.initial_project.get("classes") == [],
                "Start only after generation and before any classes are created; failure reruns need coordinator review",
            )
            self.image = next(
                (i for i in self.initial_project["images"] if i["id"] == self.args.image_id), None
            )
            require(self.image is not None, "Requested image is not in this project")
            require(
                self.image["width"] * self.image["height"] <= self.args.max_image_pixels,
                "Image exceeds the declared verification memory bound",
            )
            self.initial_rows = self.scene()
            require(
                len(self.initial_rows) >= 4,
                "Need at least four actual proposal rows, with different source tiles",
            )
            if self.args.expected_proposals is not None:
                require(
                    len(self.initial_rows) == self.args.expected_proposals,
                    "Proposal count differs from the coordinator-provided count",
                )
            self.initial_by_id = {row["id"]: row for row in self.initial_rows}
            require(len(self.initial_by_id) == len(self.initial_rows), "Duplicate canonical annotation IDs")
            for row in self.initial_rows:
                source = row.get("source", {})
                require(re.fullmatch(r"[a-zA-Z0-9_-]+", row["id"]), "Unsupported annotation ID")
                require(
                    row["status"] == "proposal"
                    and row.get("class_id") is None
                    and not row.get("human_verified"),
                    "Initial rows must be unassigned, unreviewed real proposals",
                )
                require(
                    source.get("kind") == "sam2" and source.get("operation") == "automatic_mask_generation",
                    "Only genuine automatic SAM2 proposal rows are eligible",
                )
                require(
                    not any(source.get(key) for key in ("provider_mock", "mock", "synthetic", "qa_fixture")),
                    "Synthetic proposal sources are forbidden in this driver",
                )
                require(
                    not re.search(
                        r"mock|synthetic|fixture",
                        json.dumps(
                            {
                                "model_id": source.get("model_id"),
                                "quality": source.get("quality"),
                                "transform": source.get("transform"),
                            },
                            sort_keys=True,
                        ),
                        re.I,
                    ),
                    "Explicit fixture provider evidence is forbidden in this real-row driver",
                )
                require(
                    re.fullmatch(r"[0-9a-f]{64}", source.get("model_sha256", "")),
                    "Real proposal checkpoint identity is missing",
                )
            models = {m["id"]: m for m in self.get("/api/models")}
            for row in self.initial_rows:
                model = models.get(row["source"].get("model_id"))
                require(
                    model
                    and model.get("provider") == "sam2"
                    and model.get("trusted") is True
                    and not model.get("missing")
                    and model.get("sha256") == row["source"]["model_sha256"],
                    "Real proposal model is not a trusted available matching checkpoint",
                )
            self.layers = [
                layer
                for layer in self.get(self.base + "/generation_layers")
                if layer["image_id"] == self.image["id"]
            ]
            require(
                self.layers
                and all(
                    l["status"] == "complete" and l["completed_tiles"] == l["total_tiles"]
                    for l in self.layers
                ),
                "Generation layers must all be complete before UI mutations",
            )
            jobs = self.get("/api/jobs")
            self.initial_jobs = {j["id"]: j for j in jobs}
            require(
                not any(
                    j["payload"].get("project_id") == self.args.project_id
                    and j["status"] in ("queued", "running")
                    for j in jobs
                ),
                "Project still has active work; wait for the coordinator",
            )
            for layer in self.layers:
                job = self.initial_jobs.get(layer.get("last_execution_job_id") or layer["job_id"])
                require(
                    job and job["kind"] == "generate" and job["status"] == "complete",
                    "Layer is not bound to a completed generation job",
                )
                require(
                    job["payload"].get("boundary_opt_in") is False,
                    "Expected completed generation with boundary recovery off",
                )
                require(
                    all(
                        r["source"]["model_sha256"] == layer["model_sha256"]
                        for r in self.initial_rows
                        if r.get("layer_id") == layer["id"]
                    ),
                    "Layer and proposal model identities differ",
                )
            self.boundary_before = self.get("/api/doctor")["boundary_invocations"]
            self.media_sha = digest(self.get(self.image_base + "/media", binary=True))
            require(
                self.media_sha == self.image["normalized_sha256"],
                "Served canonical image bytes do not match project identity",
            )
            write_json(self.output / "initial-project.json", self.initial_project)
            write_json(self.output / "initial-scene.json", self.initial_rows)
            result.update(
                proposals=len(self.initial_rows),
                classes_before_generation_review=0,
                generation_layers=[
                    {
                        "id": l["id"],
                        "tiles": l["total_tiles"],
                        "plan_hash": l["plan_hash"],
                        "model_sha256": l["model_sha256"],
                    }
                    for l in self.layers
                ],
                image={
                    "width": self.image["width"],
                    "height": self.image["height"],
                    "normalized_sha256": self.media_sha,
                },
                boundary_invocations=self.boundary_before,
            )

    def choose_cross_tile_parents(self):
        import numpy as np

        with self.step("choose_actual_cross_tile_parents") as result:
            candidates = []
            for index, first in enumerate(self.initial_rows):
                a = first.get("bbox") or first["geometry"].get("bbox")
                if not a:
                    continue
                for second in self.initial_rows[index + 1 :]:
                    if first.get("layer_id") != second.get("layer_id") or first["source"].get(
                        "tile"
                    ) == second["source"].get("tile"):
                        continue
                    b = second.get("bbox") or second["geometry"].get("bbox")
                    if not b:
                        continue
                    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                        0, min(a[3], b[3]) - max(a[1], b[1])
                    )
                    if overlap:
                        candidates.append(
                            (
                                overlap / max(1, min(first.get("area", 1), second.get("area", 1))),
                                first,
                                second,
                            )
                        )
            candidates.sort(key=lambda item: item[0], reverse=True)
            require(
                candidates,
                "No bounding-overlap pair from different real tiles; no synthetic fallback is allowed",
            )
            for _, first, second in candidates[: self.args.max_candidate_pairs]:
                a, b = self.raster(first), self.raster(second)
                intersection = int(np.logical_and(a, b).sum())
                if intersection and np.any(a & ~b) and np.any(b & ~a):
                    self.parents = [first, second]
                    self.parent_masks = [a, b]
                    self.union = a | b
                    break
            else:
                raise RuntimeError(
                    "No non-identical, mutually contributing cross-tile pair within the bounded search; coordinator must inspect real candidates"
                )
            others = [
                row for row in reversed(self.initial_rows) if row["id"] not in {r["id"] for r in self.parents}
            ]
            self.third, self.fourth = others[:2]
            self.changed_original_ids = {r["id"] for r in self.parents + [self.third, self.fourth]}
            for label, row in zip(("parent-a", "parent-b"), self.parents):
                write_json(self.output / (label + "-geometry.json"), self.geometry(row))
            result.update(
                parent_ids=[r["id"] for r in self.parents],
                layer_id=first["layer_id"],
                source_tiles=[r["source"]["tile"] for r in self.parents],
                source_tile_ids=[r["source"]["tile_id"] for r in self.parents],
                intersection_pixels=intersection,
                exact_union=pixel_receipt(self.union),
            )

    def canvas_and_list_selection(self):
        import numpy as np

        with self.step("installed_canvas_click_multiselect_and_object_list") as result:
            self.open_editor()
            self.screenshot("01-real-unassigned-proposals")
            self.page.get_by_role("button", name="Select (V)", exact=True).click()
            self.page.get_by_role("button", name="Clear selection", exact=True).click()
            # Reverse scene order is the UI's front-to-back hit order. First
            # choose a real visible foreground; then another point outside it.
            top = self.initial_rows[-1]
            top_mask = self.raster(top)
            first_point = interior_point(top_mask)
            self.page.mouse.click(*self.canvas_point(first_point))
            self.wait_until(
                lambda: len(self.selected_ids()) == 1, "Canvas click did not select an actual mask"
            )
            selected_first = self.selected_ids()[0]
            require(selected_first in self.initial_by_id, "Canvas selected a non-baseline annotation")
            first_mask = (
                top_mask if selected_first == top["id"] else self.raster(self.initial_by_id[selected_first])
            )
            second_point = None
            first_box = pixel_receipt(first_mask)["bbox"]
            for other in reversed(self.initial_rows):
                box = other.get("bbox") or other["geometry"].get("bbox")
                if other["id"] == selected_first or (box and box == first_box):
                    continue
                other_mask = self.raster(other)
                if np.any(other_mask & ~first_mask):
                    second_point = interior_point(other_mask, first_mask)
                    break
            require(second_point is not None, "Cannot reach a second real object outside the first mask")
            self.page.keyboard.down("Shift")
            try:
                self.page.mouse.click(*self.canvas_point(second_point))
            finally:
                self.page.keyboard.up("Shift")
            self.wait_until(
                lambda: len(self.selected_ids()) == 2, "Shift-click did not add another real object"
            )
            clicked = self.selected_ids()
            require(all(i in self.initial_by_id for i in clicked), "Unexpected selection identities")
            require(
                self.project()["revision"] == self.initial_project["revision"],
                "Selection unexpectedly mutated annotations",
            )
            self.screenshot("02-real-canvas-multiselection")
            self.select_ids([r["id"] for r in self.parents])
            self.screenshot("03-cross-tile-parent-selection")
            result.update(
                canvas_selected_ids=clicked,
                image_points=[first_point, second_point],
                list_selected_parent_ids=self.selected_ids(),
            )

    def create_classes(self):
        from playwright.sync_api import expect

        with self.step("create_three_classes_after_generation") as result:
            self.classes = []
            for label, color, row in zip(
                ("A", "B", "C"), ("#71ad83", "#c19148", "#a078b4"), self.parents + [self.third]
            ):
                self.select_ids([row["id"]])
                self.page.get_by_role("button", name="Create class and assign", exact=True).click()
                dialog = self.page.get_by_role("dialog", name="Create class and assign", exact=True)
                name = self.args.class_prefix + " " + label
                dialog.get_by_label("Class name", exact=True).fill(name)
                dialog.get_by_label("Class color", exact=True).fill(color)
                with self.page.expect_response(
                    lambda r: urlsplit(r.url).path == self.base + "/selection" and r.request.method == "POST"
                ) as assignment:
                    created = self.ui_request(
                        self.base + "/classes",
                        lambda: dialog.get_by_role(
                            "button", name="Create class and assign", exact=True
                        ).click(),
                    )
                require(assignment.value.ok, "In-place creation succeeded but assignment failed")
                expect(dialog).not_to_be_visible()
                self.saved()
                state = self.project()
                cls = next(c for c in state["classes"] if c["name"] == name)
                self.classes.append(cls)
                current = next(r for r in self.scene() if r["id"] == row["id"])
                require(
                    current["class_id"] == cls["id"]
                    and current["status"] == "draft"
                    and not current.get("human_verified"),
                    "Class assignment must be separate from acceptance",
                )
                require(len(created["classes"]) == len(self.classes), "Unexpected class creation count")
            require(len(self.project()["classes"]) == 3, "The UI did not create exactly three classes")
            result.update(
                classes=[{"id": c["id"], "name": c["name"]} for c in self.classes],
                annotations_remain_drafts=True,
            )
            self.screenshot("04-three-qa-classes")

    def merge(self):
        import numpy as np
        from playwright.sync_api import expect

        with self.step("exact_cross_tile_merge_preview_and_confirmation") as result:
            ids = [r["id"] for r in self.parents]
            self.select_ids(ids)
            preview = self.ui_request(
                self.base + "/merge/preview",
                lambda: self.page.get_by_role("button", name="Merge selected masks", exact=True).click(),
            )
            self.preview = preview
            require(
                np.array_equal(
                    decode_mask(preview["geometry"], self.image["width"], self.image["height"]), self.union
                ),
                "Preview is not the exact full-resolution parent union",
            )
            require(
                preview["area"] == int(self.union.sum())
                and preview["bbox"] == pixel_receipt(self.union)["bbox"],
                "Union metadata is inconsistent with exact pixels",
            )
            require(
                set(preview["parent_ids"]) == set(ids) and preview["class_conflict"] is True,
                "Parent identity / class conflict preview is missing",
            )
            dialog = self.page.get_by_role("dialog", name="Preview exact mask union", exact=True)
            expect(
                dialog.get_by_text("Exact union ready. Inspect the result before confirming.", exact=True)
            ).to_be_visible()
            require(
                dialog.get_by_label("Merged object class", exact=True).input_value() == "",
                "Conflicting parent classes were silently resolved",
            )
            dialog.get_by_label("Merged object class", exact=True).select_option(self.classes[0]["id"])
            self.screenshot("05-real-exact-union-preview")
            self.ui_request(
                self.base + "/merge",
                lambda: dialog.get_by_role("button", name="Merge as draft", exact=True).click(),
            )
            expect(dialog).not_to_be_visible()
            self.saved()
            rows = self.scene()
            new = [row for row in rows if row["id"] not in self.initial_by_id]
            require(len(new) == 1, "Merge did not create exactly one new identity")
            self.child = new[0]
            require(
                self.child["status"] == "draft" and not self.child.get("human_verified"),
                "Merge inherited acceptance",
            )
            require(
                set(p["id"] for p in self.child["source"]["parents"]) == set(ids),
                "Merged lineage lost the real parents",
            )
            require(
                all(next(r for r in rows if r["id"] == i)["status"] == "superseded" for i in ids),
                "Parents were not atomically superseded",
            )
            require(
                np.array_equal(self.raster(self.child), self.union),
                "Persisted merge differs from the exact union",
            )
            write_json(self.output / "merged-geometry-before-brush.json", self.geometry(self.child))
            result.update(
                child_id=self.child["id"],
                parent_revisions=preview["parent_revisions"],
                union=pixel_receipt(self.union),
                class_id=self.child["class_id"],
                inference_performed=False,
            )

    def brush(self):
        import numpy as np

        with self.step("full_image_brush_edit_on_actual_merged_pixels") as result:
            self.select_ids([self.child["id"]])
            self.page.get_by_role("button", name="Fit", exact=True).click()
            geometry_path = self.base + "/annotations/" + self.child["id"] + "/geometry"
            self.wait_until(
                lambda: geometry_path in self.geometry_responses,
                "Editor has not hydrated the selected merged mask",
            )
            point = interior_point(self.union)
            radius = 8
            self.page.get_by_role("button", name="Subtract mask pixels (E)", exact=True).click()
            self.page.get_by_label("Brush radius", exact=True).fill(str(radius))
            self.ui_request(
                self.base + "/annotations/" + self.child["id"],
                lambda: self.page.mouse.click(*self.canvas_point(point), delay=180),
                method="PATCH",
            )
            self.saved()
            self.child = next(r for r in self.scene() if r["id"] == self.child["id"])
            self.brushed = self.raster(self.child)
            removed = self.union & ~self.brushed
            require(
                np.any(removed) and not np.any(self.brushed & ~self.union),
                "Subtract brush did not remove real foreground exclusively",
            )
            ys, xs = np.nonzero(removed)
            distance = np.sqrt((xs + 0.5 - point[0]) ** 2 + (ys + 0.5 - point[1]) ** 2)
            require(
                float(distance.max()) <= radius + 2,
                "Brush changed pixels outside its bounded source-image neighborhood",
            )
            require(self.brushed.any(), "Brush unexpectedly erased the entire instance")
            write_json(self.output / "merged-geometry-after-brush.json", self.geometry(self.child))
            result.update(
                image_point=point,
                radius_px=radius,
                removed_pixels=int(removed.sum()),
                max_changed_distance_px=float(distance.max()),
                after=pixel_receipt(self.brushed),
            )
            self.screenshot("06-real-brush-edit")

    def delete_undo_reload(self):
        import numpy as np

        with self.step("delete_then_persistent_undo_after_browser_reload") as result:
            self.select_ids([self.child["id"]])
            before = semantic_row(self.child)
            self.page.get_by_role("button", name="Delete selected", exact=True).click()
            dialog = self.page.get_by_role("dialog", name="Delete selected objects?", exact=True)
            self.ui_request(
                self.base + "/selection",
                lambda: dialog.get_by_role("button", name="Delete selected", exact=True).click(),
            )
            self.saved()
            require(
                self.child["id"] not in {r["id"] for r in self.scene()},
                "Deleted child remains in the canonical scene",
            )
            require(
                all(
                    next(r for r in self.scene() if r["id"] == p["id"])["status"] == "superseded"
                    for p in self.parents
                ),
                "Deleting a merged child resurrected its parents",
            )
            self.screenshot("07-deleted-child")
            self.open_editor(reload=True)
            self.ui_request(
                self.image_base + "/history",
                lambda: self.page.get_by_role("button", name="Undo", exact=True).click(),
            )
            self.saved()
            restored = next(r for r in self.scene() if r["id"] == self.child["id"])
            require(
                semantic_row(restored) == before and np.array_equal(self.raster(restored), self.brushed),
                "Persistent undo lost identity, class, provenance or exact brushed pixels",
            )
            self.open_editor(reload=True)
            require(
                np.array_equal(self.raster(restored), self.brushed),
                "Geometry did not survive a second browser reopen",
            )
            self.select_ids([self.child["id"]])
            self.screenshot("08-restored-after-reload")
            result.update(
                restored_annotation_id=self.child["id"],
                pixel_receipt=pixel_receipt(self.brushed),
                persistent_server_history=True,
            )

    def accept_qa_examples(self):
        with self.step("explicit_automated_acceptance_of_three_real_instances") as result:
            self.select_ids([self.fourth["id"]])
            self.page.get_by_label("Class for selection / paint", exact=True).select_option(
                self.classes[1]["id"]
            )
            self.ui_request(
                self.base + "/selection",
                lambda: self.page.get_by_role("button", name="Assign class", exact=True).click(),
            )
            self.saved()
            self.accepted_ids = [self.child["id"], self.fourth["id"], self.third["id"]]
            self.select_ids(self.accepted_ids)
            self.ui_request(
                self.base + "/selection",
                lambda: self.page.get_by_role("button", name="Accept selected", exact=True).click(),
            )
            self.saved()
            current = {r["id"]: r for r in self.scene()}
            self.accepted = {i: current[i] for i in self.accepted_ids}
            require(
                all(
                    r["status"] == "accepted"
                    and r.get("review_actor") == "automated_qa"
                    and not r.get("human_verified")
                    for r in self.accepted.values()
                ),
                "Explicit QA acceptance was mislabeled as human verification",
            )
            require(
                {r["class_id"] for r in self.accepted.values()} == {c["id"] for c in self.classes},
                "Expected one active accepted example per QA class",
            )
            require(
                next(i for i in self.project()["images"] if i["id"] == self.image["id"])["complete"] is False,
                "Partial QA review was incorrectly marked image-complete",
            )
            self.accepted_pixels = {i: pixel_receipt(self.raster(r)) for i, r in self.accepted.items()}
            result.update(
                accepted_ids=self.accepted_ids,
                pixel_receipts=self.accepted_pixels,
                human_verified=False,
                image_complete=False,
            )
            self.screenshot("09-explicit-qa-acceptance")

    def restore_export_continuation(self):
        """Read and verify prior real-row evidence before any further mutation."""
        with self.step("verify_saved_acceptance_before_export_continuation") as result:
            previous_path = self.args.continue_from.resolve()
            previous_bytes = previous_path.read_bytes()
            previous = json.loads(previous_bytes)
            self.qa_rejected_ids = set()
            if previous.get("continuation_source"):
                # Preserve a completed bulk rejection if its verifier stopped
                # before image completion. Never repeat the mutation.
                require(
                    previous["project_id"] == self.args.project_id
                    and previous["image_id"] == self.args.image_id,
                    "Review continuation belongs to another project/image",
                )
                require(
                    any(
                        f["error"] == "QA rejection altered geometry or source identity"
                        for f in previous.get("failures", [])
                    ),
                    "Unsupported review continuation failure",
                )
                mutations = previous["mutations"]
                require(
                    len(mutations) == 1
                    and mutations[0]["method"] == "POST"
                    and mutations[0]["path"] == self.base + "/selection"
                    and mutations[0]["body"].get("action") == "reject",
                    "Review continuation contains unexpected mutations",
                )
                self.qa_rejected_ids = set(mutations[0]["body"]["annotation_ids"])
                self.record["prior_review_evidence"] = {
                    "path": str(previous_path),
                    "sha256": digest(previous_bytes),
                    "rejected_count": len(self.qa_rejected_ids),
                }
                reference = previous["continuation_source"]
                previous_path = Path(reference["evidence"])
                previous_bytes = previous_path.read_bytes()
                require(digest(previous_bytes) == reference["sha256"], "Prior evidence changed")
                previous = json.loads(previous_bytes)
            require(
                previous["project_id"] == self.args.project_id and previous["image_id"] == self.args.image_id,
                "Continuation evidence belongs to another project/image",
            )
            require(
                previous.get("scope") == self.record["scope"]
                and not previous.get("browser_errors")
                and not previous.get("blocked_requests"),
                "Invalid prior real UI evidence",
            )
            stages = {s["name"]: s for s in previous["stages"]}
            accepted_stage = stages["explicit_automated_acceptance_of_three_real_instances"]
            require(
                accepted_stage["passed"] is True
                and stages["delete_then_persistent_undo_after_browser_reload"]["passed"] is True,
                "Continuation requires successful acceptance and persistent undo evidence",
            )
            require(
                any(
                    f["stage"] == "accepted_coco_export"
                    and f["error"] == "COCO image scope differs from the requested original"
                    for f in previous.get("failures", [])
                ),
                "Continuation only handles the incomplete-image export gate",
            )
            session = self.get("/api/session")
            self.token = session.get("token")
            require(
                session.get("qa_mode") is True and session.get("actor") == "automated_qa",
                "Continuation requires automated QA mode",
            )
            self.initial_project = json.loads((previous_path.parent / "initial-project.json").read_text())
            self.initial_rows = json.loads((previous_path.parent / "initial-scene.json").read_text())
            self.initial_by_id = {r["id"]: r for r in self.initial_rows}
            self.image = next(i for i in self.initial_project["images"] if i["id"] == self.args.image_id)
            self.media_sha = self.image["normalized_sha256"]
            self.boundary_before = stages["read_only_real_generation_preflight"]["boundary_invocations"]
            require(
                self.get("/api/doctor")["boundary_invocations"] == self.boundary_before,
                "Boundary baseline changed since prior acceptance",
            )
            require(
                digest(self.get(self.image_base + "/media", binary=True)) == self.media_sha,
                "Original image changed since prior acceptance",
            )
            current_project = self.project()
            self.classes = current_project["classes"]
            expected_classes = stages["create_three_classes_after_generation"]["classes"]
            require(
                {c["id"]: c["name"] for c in self.classes} == {c["id"]: c["name"] for c in expected_classes},
                "QA classes changed",
            )
            require(
                not next(i for i in current_project["images"] if i["id"] == self.image["id"])["complete"],
                "Image is already complete; inspect before repeating export continuation",
            )
            rows = self.scene()
            current = {r["id"]: r for r in rows}
            self.accepted_ids = accepted_stage["accepted_ids"]
            self.accepted_pixels = accepted_stage["pixel_receipts"]
            self.accepted = {aid: current[aid] for aid in self.accepted_ids}
            child_id = stages["exact_cross_tile_merge_preview_and_confirmation"]["child_id"]
            self.child = current[child_id]
            parent_ids = stages["choose_actual_cross_tile_parents"]["parent_ids"]
            self.changed_original_ids = set(parent_ids) | set(self.accepted_ids)
            require(set(current) == set(self.initial_by_id) | {child_id}, "Annotation identities changed")
            require(
                {r["id"] for r in rows if r["status"] == "accepted"} == set(self.accepted_ids),
                "Accepted set changed since prior UI run",
            )
            require(
                all(current[aid]["status"] == "superseded" for aid in parent_ids),
                "Merged parent state changed",
            )
            for aid, row in self.accepted.items():
                require(
                    row.get("review_actor") == "automated_qa" and not row.get("human_verified"),
                    "Accepted review actor changed",
                )
                require(
                    pixel_receipt(self.raster(row)) == self.accepted_pixels[aid],
                    "Accepted pixels changed since prior run",
                )
            for aid, row in self.initial_by_id.items():
                if aid not in self.changed_original_ids:
                    expected = semantic_row(row)
                    if aid in self.qa_rejected_ids:
                        expected["status"] = "rejected"
                    require(semantic_row(current[aid]) == expected, "Unrelated proposal changed")
            require(
                not self.qa_rejected_ids.intersection(self.changed_original_ids),
                "Recorded rejection includes a protected accepted object or parent",
            )
            jobs = self.get("/api/jobs")
            require(
                not any(
                    j["payload"].get("project_id") == self.args.project_id
                    and j["status"] in {"running", "queued"}
                    for j in jobs
                ),
                "Project has active jobs",
            )
            self.initial_jobs = {j["id"]: j for j in jobs}
            self.record["continuation_source"] = {
                "evidence": str(previous_path),
                "sha256": digest(previous_bytes),
                "prior_mutation_count": len(previous["mutations"]),
                "passed_stages": [s["name"] for s in previous["stages"] if s["passed"]],
            }
            write_json(self.output / "initial-project.json", self.initial_project)
            write_json(self.output / "initial-scene.json", self.initial_rows)
            result.update(
                accepted_ids=self.accepted_ids,
                corrected_child_id=child_id,
                accepted_pixels_unchanged=True,
                original_image_unchanged=True,
            )
            self.open_editor()

    def complete_qa_image(self):
        """Explicit QA disposition makes accepted-only exchange export eligible."""
        from playwright.sync_api import expect

        with self.step("explicit_qa_rejection_of_remaining_proposals_and_image_completion") as result:
            before = {r["id"]: r for r in self.scene()}
            pending_ids = {aid for aid, row in before.items() if row["status"] in {"proposal", "draft"}}
            self.qa_rejected_ids = getattr(self, "qa_rejected_ids", set()) | pending_ids
            if pending_ids:
                self.page.get_by_role("button", name="Select visible objects", exact=True).click()
                for aid in self.accepted_ids:
                    self.page.locator(f'.object-row[data-object-id="{aid}"]').get_by_role(
                        "checkbox", name=re.compile("Include object .* in selection")
                    ).uncheck()
                require(
                    set(self.selected_ids()) == pending_ids,
                    "QA bulk rejection selection includes an accepted object or misses a proposal",
                )
                self.page.get_by_role("button", name="Reject selected", exact=True).click()
                dialog = self.page.get_by_role("dialog", name="Reject selected objects?", exact=True)
                self.ui_request(
                    self.base + "/selection",
                    lambda: dialog.get_by_role("button", name="Reject selected", exact=True).click(),
                )
                expect(dialog).not_to_be_visible()
                self.saved()
            rows = {r["id"]: r for r in self.scene()}
            for aid in self.qa_rejected_ids:
                expected = {**semantic_row(before[aid]), "status": "rejected"}
                require(
                    semantic_row(rows[aid]) == expected, "QA rejection altered geometry or source identity"
                )
            for aid in self.accepted_ids:
                require(
                    semantic_row(rows[aid]) == semantic_row(before[aid]),
                    "QA rejection touched an accepted object",
                )
                require(
                    pixel_receipt(self.raster(rows[aid])) == self.accepted_pixels[aid],
                    "QA rejection changed accepted pixels",
                )
            self.page.get_by_role("button", name="Mark reviewed", exact=True).click()
            dialog = self.page.get_by_role("dialog", name="Confirm image review", exact=True)
            dialog.get_by_role("checkbox", name="I reviewed all objects", exact=False).check()
            self.ui_request(
                self.image_base + "/complete",
                lambda: dialog.get_by_role("button", name="Confirm full image review", exact=True).click(),
            )
            expect(dialog).not_to_be_visible()
            self.saved()
            image = next(i for i in self.project()["images"] if i["id"] == self.image["id"])
            require(
                image["complete"] is True
                and image.get("review_actor") == "automated_qa"
                and not image.get("human_verified"),
                "QA image did not become complete without human verification",
            )
            self.screenshot("09b-complete-automated-qa-image")
            result.update(
                rejected_ids=sorted(self.qa_rejected_ids),
                rejected_count=len(self.qa_rejected_ids),
                accepted_ids=self.accepted_ids,
                image_complete=True,
                human_verified=False,
            )

    def export(self, fmt):
        from playwright.sync_api import expect

        with self.step("accepted_coco_export" if fmt == "coco" else "lossless_native_backup") as result:
            self.page.get_by_role("navigation").get_by_role("button", name="Images", exact=True).click()
            self.page.get_by_role("checkbox", name="Select " + self.image["name"], exact=True).check()
            self.page.get_by_role("navigation").get_by_role("button", name="Export", exact=True).click()
            self.page.get_by_label("Format", exact=True).select_option(fmt)
            self.page.get_by_label("Image scope", exact=True).select_option(
                "selected" if fmt == "coco" else "all"
            )
            self.page.get_by_role("checkbox", name="Reviewed annotations only", exact=True).check()
            self.page.get_by_role("checkbox", name="Include image copies", exact=True).check()
            self.screenshot("10-coco-export-settings" if fmt == "coco" else "12-native-backup-settings")
            job = self.ui_request(
                self.base + "/exports",
                lambda: self.page.get_by_role(
                    "button", name="Create downloadable export", exact=True
                ).click(),
            )

            def finished():
                current = self.get("/api/jobs/" + job["id"])
                if current["status"] in {"failed", "cancelled", "interrupted"}:
                    raise RuntimeError("Export failed: " + str(current.get("error")))
                return current["status"] == "complete"

            self.wait_until(finished, "Export did not complete", self.args.export_timeout_seconds)
            job = self.get("/api/jobs/" + job["id"])
            link = self.page.locator(f'[data-job-id="{job["id"]}"]').get_by_role(
                "link", name="Download artifact", exact=True
            )
            expect(link).to_be_visible(timeout=self.args.action_timeout_seconds * 1000)
            with self.page.expect_download() as download:
                link.click()
            download = download.value
            artifact = self.output / ("accepted-coco.zip" if fmt == "coco" else "project-native.zip")
            download.save_as(str(artifact))
            require(download.failure() is None, "Browser download failed")
            artifact_sha = digest(artifact.read_bytes())
            if job["result"].get("sha256"):
                require(
                    job["result"]["sha256"] == artifact_sha,
                    "Downloaded native artifact checksum differs from server receipt",
                )
            result.update(
                job_id=job["id"], artifact=artifact.name, sha256=artifact_sha, bytes=artifact.stat().st_size
            )
            if fmt == "coco":
                self.verify_coco(artifact, result)
            else:
                self.verify_native(artifact, result)
            self.screenshot("11-coco-download-ready" if fmt == "coco" else "13-native-download-ready")

    def verify_coco(self, artifact, result):
        with zipfile.ZipFile(artifact) as archive:
            body = json.loads(archive.read("annotations.json"))
            scope = json.loads(archive.read("compag_export_scope.json"))
            require(
                scope["draft_inclusive"] is False and scope["boundary_recovery_invoked"] is False,
                "COCO export did not honor accepted-only/no-recovery settings",
            )
            require(
                {i["compag_id"] for i in body["images"]} == {self.image["id"]},
                "COCO image scope differs from the requested original",
            )
            require(
                {a["compag_id"] for a in body["annotations"]} == set(self.accepted_ids),
                "COCO included unaccepted/superseded rows or omitted an accepted row",
            )
            categories = {c["id"]: c["compag_id"] for c in body["categories"]}
            for annotation in body["annotations"]:
                aid = annotation["compag_id"]
                pixels = decode_mask(annotation["segmentation"], self.image["width"], self.image["height"])
                require(
                    pixel_receipt(pixels) == self.accepted_pixels[aid],
                    "COCO RLE pixels differ from the accepted canonical mask",
                )
                require(
                    categories[annotation["category_id"]] == self.accepted[aid]["class_id"],
                    "COCO class identity mapping changed",
                )
                require(annotation["area"] == self.accepted_pixels[aid]["area"], "COCO area is stale")
            image_bytes = archive.read(body["images"][0]["file_name"])
            # Exported PNG encoding may differ; compare decoded original pixels.
            from PIL import Image
            from io import BytesIO
            import numpy as np

            exported = np.asarray(Image.open(BytesIO(image_bytes)).convert("RGB"))
            original = np.asarray(
                Image.open(BytesIO(self.get(self.image_base + "/media", binary=True))).convert("RGB")
            )
            require(np.array_equal(exported, original), "COCO export altered original image pixels")
            result.update(
                accepted_annotations=len(body["annotations"]),
                exact_canonical_pixels=True,
                original_image_pixels_unchanged=True,
            )

    def verify_native(self, artifact, result):
        with zipfile.ZipFile(artifact) as archive:
            manifest = json.loads(archive.read("native_manifest.json"))
            require(
                manifest.get("model_weights_included") is False,
                "Native backup unexpectedly includes model weights",
            )
            require(
                not any(
                    Path(name).suffix.lower() in {".pt", ".pth", ".onnx", ".safetensors"}
                    for name in archive.namelist()
                ),
                "Unexpected model weight member in native archive",
            )
            for name, checksum in manifest["files"].items():
                require(digest(archive.read(name)) == checksum, "Native manifest checksum mismatch: " + name)
            database = self.output / "native-inspection.sqlite3"
            database.write_bytes(archive.read("project.sqlite3"))
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
                state = json.loads(connection.execute("SELECT body FROM state WHERE id=1").fetchone()[0])
                rows = [
                    json.loads(row[0])
                    for row in connection.execute(
                        "SELECT body FROM annotations WHERE image_id=?", (self.image["id"],)
                    )
                ]
                history = connection.execute(
                    "SELECT action, actor, undone FROM history WHERE image_id=? ORDER BY seq",
                    (self.image["id"],),
                ).fetchall()
                events = connection.execute("SELECT action, actor FROM events ORDER BY seq").fetchall()
            expected = {r["id"]: semantic_row(r) for r in self.scene()}
            require(
                {r["id"]: semantic_row(r) for r in rows} == expected,
                "Native backup did not preserve complete live proposal/review state",
            )
            require(
                {c["id"] for c in state["classes"]} == {c["id"] for c in self.classes},
                "Native class identities changed",
            )
            for row in rows:
                if row["id"] in self.accepted:
                    geometry = row["geometry"]
                    if geometry.get("ref"):
                        geometry = json.loads(archive.read(geometry["ref"]))
                    require(
                        pixel_receipt(decode_mask(geometry, self.image["width"], self.image["height"]))
                        == self.accepted_pixels[row["id"]],
                        "Native accepted mask differs from canonical pixels",
                    )
            require(
                any(action == "merge_instances" and actor == "automated_qa" for action, actor, _ in history),
                "Native backup lost merge QA history",
            )
            # A new edit after Undo intentionally clears the redo branch. The
            # immutable events still record the deletion and its undo.
            require(
                ("selection_delete", "automated_qa") in events and ("undo", "automated_qa") in events,
                "Native backup lost deletion/undo event history",
            )
            if self.qa_rejected_ids:
                require(
                    ("selection_reject", "automated_qa") in events,
                    "Native backup lost the automated QA rejection event",
                )
            require(
                not any(r.get("human_verified") for r in rows),
                "Native backup manufactured human verification",
            )
            result.update(
                full_project_backup=True,
                accepted_ids=self.accepted_ids,
                retained_proposals=sum(r["status"] == "proposal" for r in rows),
                superseded_ids=[r["id"] for r in rows if r["status"] == "superseded"],
                history_entries=len(history),
                event_entries=len(events),
                exact_canonical_pixels=True,
                weights_included=False,
            )

    def final_persistence(self):
        with self.step("final_reload_and_no_inference_audit") as result:
            self.open_editor(reload=True)
            rows = self.scene()
            current = {r["id"]: r for r in rows}
            for aid, baseline in self.initial_by_id.items():
                if aid not in self.changed_original_ids:
                    expected = semantic_row(baseline)
                    if aid in self.qa_rejected_ids:
                        expected.update(status="rejected")
                    require(
                        semantic_row(current[aid]) == expected,
                        "An unrelated original proposal changed",
                    )
            for aid in self.accepted_ids:
                require(
                    current[aid]["status"] == "accepted"
                    and current[aid].get("review_actor") == "automated_qa"
                    and not current[aid].get("human_verified"),
                    "Accepted QA state did not survive browser reopen",
                )
                require(
                    pixel_receipt(self.raster(current[aid])) == self.accepted_pixels[aid],
                    "Accepted geometry changed after export/reopen",
                )
            require(
                digest(self.get(self.image_base + "/media", binary=True)) == self.media_sha,
                "Original canonical image changed during UI acceptance",
            )
            require(
                self.get("/api/doctor")["boundary_invocations"] == self.boundary_before,
                "UI acceptance invoked boundary recovery",
            )
            jobs = self.get("/api/jobs")
            created = [
                j
                for j in jobs
                if j["id"] not in self.initial_jobs and j["payload"].get("project_id") == self.args.project_id
            ]
            require(
                len(created) == 2
                and all(j["kind"] == "export" and j["status"] == "complete" for j in created),
                "Acceptance created work beyond its two exports",
            )
            require(
                not self.record["inference_training_weight_requests"], "Forbidden provider request attempted"
            )
            require(not any(r.get("human_verified") for r in rows), "A row claims human verification")
            require(
                next(i for i in self.project()["images"] if i["id"] == self.image["id"])["complete"],
                "Completed QA review did not persist",
            )
            self.select_ids(self.accepted_ids)
            self.page.set_viewport_size({"width": 1366, "height": 768})
            self.page.get_by_role("button", name="Fit", exact=True).click()
            self.screenshot("14-real-final-laptop")
            write_json(self.output / "final-scene.json", rows)
            write_json(self.output / "final-project.json", self.project())
            result.update(
                accepted=len(self.accepted_ids),
                proposals=sum(r["status"] == "proposal" for r in rows),
                rejected=sum(r["status"] == "rejected" for r in rows),
                superseded=sum(r["status"] == "superseded" for r in rows),
                image_complete=True,
                boundary_calls_added=0,
                inference_jobs_added=0,
                training_jobs_added=0,
                exports=[j["id"] for j in created],
                original_bytes_unchanged=True,
            )

    def run(self):
        try:
            if self.args.continue_from:
                self.restore_export_continuation()
            else:
                self.preflight()
                self.choose_cross_tile_parents()
                if self.args.preflight:
                    self.record.update(preflight_passed=True, executed_mutations=False, finished_at=utc_now())
                    return
                self.canvas_and_list_selection()
                self.create_classes()
                self.merge()
                self.brush()
                self.delete_undo_reload()
                self.accept_qa_examples()
            self.complete_qa_image()
            self.export("coco")
            self.export("native")
            self.final_persistence()
            self.record.update(passed=True, finished_at=utc_now(), executed_mutations=True)
        except BaseException as error:
            message = str(error).replace(self.token or "\0", "<redacted-session-token>")
            self.record["failures"].append(
                {
                    "stage": self.stage,
                    "error": message,
                    "traceback": traceback.format_exc().replace(
                        self.token or "\0", "<redacted-session-token>"
                    ),
                }
            )
            self.record["finished_at"] = utc_now()
            try:
                self.screenshot("FAILURE-" + re.sub("[^a-zA-Z0-9_-]", "-", self.stage))
                (self.output / "failure-page-text.txt").write_text(self.page.locator("body").inner_text())
            except Exception as capture_error:
                self.record["failure_capture_error"] = str(capture_error)
            raise
        finally:
            self.save()
            self.context.close()


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url", required=True, help="Coordinator-announced installed QA server URL (loopback only)"
    )
    parser.add_argument("--project-id", "--pid", dest="project_id", required=True)
    parser.add_argument("--image-id", "--iid", dest="image_id", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh private evidence folder outside this source checkout",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute", action="store_true", help="Run authorized UI mutations only after coordinator readiness"
    )
    mode.add_argument(
        "--preflight", action="store_true", help="Read-only generation/candidate checks; no browser UI edits"
    )
    parser.add_argument("--headed", action="store_true", help="Show the Chromium window during acceptance")
    parser.add_argument(
        "--continue-from",
        type=Path,
        help="Prior real UI evidence.json with accepted instances but an incomplete-image COCO export failure; continues review/export only",
    )
    parser.add_argument("--expected-proposals", type=int)
    parser.add_argument("--class-prefix", default="AUTOMATED QA region")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--action-timeout-seconds", type=int, default=90)
    parser.add_argument("--export-timeout-seconds", type=int, default=240)
    parser.add_argument("--max-image-pixels", type=int, default=100_000_000)
    parser.add_argument("--max-candidate-pairs", type=int, default=48)
    args = parser.parse_args(argv)
    if args.continue_from and not args.execute:
        parser.error(
            "--continue-from requires --execute and coordinator authorization for remaining QA review"
        )
    url = urlsplit(args.url)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or url.username
        or url.password
        or url.path not in {"", "/"}
        or url.query
        or url.fragment
    ):
        parser.error("Use a plain loopback HTTP server origin without credentials, path, query or fragment")
    if not all(re.fullmatch(r"[a-zA-Z0-9_-]+", item) for item in (args.project_id, args.image_id)):
        parser.error("Project and image IDs must be plain application identifiers")
    if (
        not 1 <= args.timeout_seconds <= 3600
        or not 1 <= args.action_timeout_seconds <= 300
        or not 1 <= args.export_timeout_seconds <= 600
    ):
        parser.error("Timeouts must remain within the declared bounded ranges")
    if not 1 <= args.max_candidate_pairs <= 256 or not 1 <= args.max_image_pixels <= 100_000_000:
        parser.error("Use bounded candidate and image-pixel limits")
    if not args.class_prefix.strip() or len(args.class_prefix) > 100:
        parser.error("Choose a nonempty QA class prefix of at most 100 characters")
    source_root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(source_root):
        parser.error("Private real-image evidence must be written outside the source checkout")
    return args


def main(argv=None):
    args = arguments(argv)
    if not (args.execute or args.preflight):
        print(
            "Plan only: no server contact. After coordinator readiness, use --preflight for read-only actual-proposal checks or --execute for class creation, real-mask UI selection/merge/brush/history, rejection of remaining proposals, completed QA image review and verified COCO/native exports. No inference, training or weight operations. All review is automated QA, never human UAT."
        )
        return 0
    args.output.resolve().mkdir(parents=True, exist_ok=False)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=not args.headed)
            try:
                Acceptance(args, browser).run()
            finally:
                browser.close()
    except BaseException as error:
        evidence = args.output.resolve() / "evidence.json"
        if not evidence.exists():
            write_json(
                evidence,
                {
                    "passed": False,
                    "stage": "browser_startup",
                    "human_uat": "AWAITING_HUMAN_UAT",
                    "executed_mutations": False,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    print(
        "Read-only preflight passed."
        if args.preflight
        else "Real generated-row UI acceptance passed (automated QA only)."
    )
    print("Evidence: " + str(args.output.resolve() / "evidence.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
