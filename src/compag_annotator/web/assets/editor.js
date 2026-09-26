import { addAssistPoint, removeLastAssistPoint, assistDrafts, assistBatchDialog } from "./assist-review.js";
import { loadMaskDisplay, flushMaskDisplay } from "./mask-display.js";
import { samSettingsDialog } from "./sam-settings.js";
import {
  generateDialog,
  generationLayers,
  layerControls,
} from "./generation.js";
import {
  activeSelection,
  selectionAction,
  selectionToolbar,
  createClassDialog,
  annotationBoundaryDialog,
  reviewAnnotationBoundary,
} from "./selection.js";
import {
  el,
  button,
  input,
  select,
  field,
  check,
  row,
  badge,
  details,
  dialog,
  confirmAction,
  notify,
  projectPath,
} from "./ui.js";
import { t } from "./i18n.js";
import { reviewStatus, needsReview } from "./review-status.js";
import { editorGuide, focusControl } from "./workflow-guide.js";
import { AnnotationCanvas } from "./canvas.js";
import { clone, bbox, rasterize, encodeRLE } from "./geometry.js";
export class Editor {
  constructor(app, imageId) {
    this.app = app;
    this.project = app.project;
    this.image =
      this.project.images.find((i) => i.id === imageId) ||
      this.project.images[0];
    this.base = projectPath(this.project.id);
    this.path = this.base + `/images/${this.image.id}`;
    this.objects = [];
    this.objectOrder = new Map();
    this.revision = this.image.revision;
    this.alive = true;
    this.busy = false;
    this.failed = null;
    this.promptState = {
      mode: "independent",
      points: [],
      labels: [],
      box: null,
      model_id: "",
      class_id: this.project.classes.find((c) => !c.archived)?.id || null,
    };
    this.previewIndex = 0;
    this.events = new AbortController();
  }
  async mount() {
    const app = this.app,
      p = this.project;
    this.saveStatus = el(
      "span",
      { class: "editor-status", "aria-live": "polite" },
      "Loading annotations…",
    );
    this.coordinates = el("span");
    this.inspector = el("aside", {
      class: "inspector",
      "aria-label": "Annotation inspector",
    });
    this.workflowGuide = el("div", { class: "editor-guide", dataset: { testid: "editor-next-step" } });
    const canvas = el("canvas", {
      class: "annotation-canvas",
      tabIndex: 0,
      "aria-label": "Full image annotation canvas",
      "aria-describedby": "canvas-guide",
    });
    this.canvas = canvas;
    canvas.dataset.mode = "select";
    this.zoomLabel = el("span", {}, "100%");
    this.hint = el(
      "div",
      { class: "tool-hint" },
      "Select mode · Click an object · Shift/Ctrl-click adds to selection",
    );
    const tools = [
      ["select", "↖", "Select (V)"],
      ["paint", "◒", "Paint class (L)"],
      ["edit", "◇", "Edit geometry (G)"],
      ["pan", "✥", "Pan (H)"],
      ["box", "▭", "Draw box (B)"],
      ["polygon", "⬡", "Draw polygon (P)"],
      ["brush-add", "◉", "Add mask pixels (D)"],
      ["brush-subtract", "◌", "Subtract mask pixels (E)"],
      ["positive", "+", "SAM positive point"],
      ["negative", "−", "SAM negative point"],
      ["prompt-box", "⊡", "SAM box prompt"],
    ];
    this.toolButtons = new Map();
    const toolbar = el(
      "div",
      { class: "tools", role: "toolbar", "aria-label": "Drawing tools" },
      tools.map(([tool, symbol, label]) => {
        const b = button(symbol, () => this.setTool(tool), "", {
          "aria-label": label,
          title: label,
          "aria-pressed": String(tool === "select"),
        });
        this.toolButtons.set(tool, b);
        return b;
      }),
    );
    this.layer = new AnnotationCanvas(canvas, {
      change: (o, g) => this.saveGeometry(o, g),
      geometry: (o) =>
        this.app.api.get(
          o.geometry_url || this.base + `/annotations/${o.id}/geometry`,
        ),
      paint: (o) => {
        if (!this.promptState.class_id)
          throw Error("Choose or create a class before using Paint class.");
        return selectionAction(
          this,
          this.paintAccept ? "assign_accept" : "assign",
          this.promptState.class_id,
          [o.id],
        );
      },
      select: () => {
        this.invalidateAssist();
        this.renderInspector();
        const focused = this.inspector.querySelector(
          `[data-object-id="${this.layer.selected}"]`,
        );
        focused?.scrollIntoView({ block: "nearest" });
      },
      prompt: (point, label) => {
        addAssistPoint(this, point, label);
      },
      boxPrompt: (box) => {
        if (this.promptState.mode === "independent") {
          notify("For a box prompt, choose Refine one object in SAM assistance.", true);
          return;
        }
        this.promptState.box = box;
        this.invalidateAssist();
        this.syncPrompts();
        this.renderInspector();
      },
      error: (m) => notify(m, true),
      view: (s, point) => {
        this.zoomLabel.textContent = `${Math.round(s * 100)}%`;
        this.coordinates.textContent = point
          ? `${Math.round(point[0])}, ${Math.round(point[1])} px`
          : "";
      },
    });
    this.layer.classes = p.classes;
    loadMaskDisplay(this);
    const imageSelect = select(
      "editor-image",
      p.images.map((i) => [i.id, i.name]),
      this.image.id,
      {
        "aria-label": "Current image",
        onchange: (e) => app.annotate(e.target.value),
      },
    );
    this.root = el(
      "div",
      {
        class: "editor-root",
        style: "display:flex;flex-direction:column;min-height:0;height:100%",
      },
      el(
        "header",
        { class: "editor-header" },
        row(
          button("←", () => this.adjacent(-1), "quiet", {
            "aria-label": "Previous image",
          }),
          imageSelect,
          button("→", () => this.adjacent(1), "quiet", {
            "aria-label": "Next image",
          }),
          badge(`${this.image.width} × ${this.image.height}`),
        ),
        row(
          button("Generate masks", () => generateDialog(app, this.image.id)),
          button("Undo", () => this.history("undo"), "", {
            title: "Undo (Ctrl+Z)",
          }),
          button("Redo", () => this.history("redo"), "", {
            title: "Redo (Ctrl+Shift+Z)",
          }),
          button("Mark reviewed", () => this.complete(), "primary"),
        ),
      ),
      this.workflowGuide,
      el(
        "div",
        { class: "editor-layout" },
        toolbar,
        el(
          "div",
          { class: "canvas-wrap" },
          canvas,
          this.hint,
          el(
            "div",
            { class: "canvas-hud" },
            button("−", () => this.layer.zoom(0.8), "", {
              "aria-label": "Zoom out",
            }),
            this.zoomLabel,
            button("+", () => this.layer.zoom(1.25), "", {
              "aria-label": "Zoom in",
            }),
            button("Fit", () => this.layer.fit()),
            button("1:1", () => {
              this.layer.zoom(1 / this.layer.transform.scale);
            }),
          ),
        ),
        this.inspector,
      ),
      el(
        "footer",
        { class: "editor-footer" },
        el("span", { id: "canvas-guide" }, t("editorHint")),
        row(this.saveStatus, this.coordinates),
      ),
    );
    window.addEventListener(
      "beforeunload",
      (e) => {
        if (this.busy || this.failed || this.layer.draft.length) {
          e.preventDefault();
          e.returnValue = "";
        }
      },
      { signal: this.events.signal },
    );
    window.addEventListener("keydown", (e) => this.key(e), {
      signal: this.events.signal,
    });
    queueMicrotask(async () => {
      try {
        await Promise.all([
          this.loadScene(),
          this.layer.setImage(
            this.image,
            this.image.media_url || this.path + "/media",
          ),
        ]);
        const result = await app.api.get("/api/models");
        this.models = Array.isArray(result) ? result : result.models || [];
        if (this.alive) {
          this.renderInspector();
          await generationLayers(this);
        }
      } catch (e) {
        this.setStatus(e.message, true);
        notify(e.message, true);
      }
    });
    return this.root;
  }
  async adjacent(delta) {
    const list = this.project.images,
      index = list.findIndex((i) => i.id === this.image.id),
      next = list[index + delta];
    if (next) await this.app.annotate(next.id);
  }
  key(e) {
    if (e.target.closest("input,textarea,select,dialog")) return;
    if (e.ctrlKey && !e.altKey && !e.shiftKey && (e.code === "KeyD" || e.key.toLowerCase() === "d")) {
      if (e.target.isContentEditable) return;
      e.preventDefault();
      removeLastAssistPoint(this);
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      this.history(e.shiftKey ? "redo" : "undo").catch((e) =>
        notify(e.message, true),
      );
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      if (this.layer.draft.length) {
        this.layer.draft.pop();
        this.layer.render();
      } else this.deleteSelected().catch((e) => notify(e.message, true));
      e.preventDefault();
      return;
    }
    const tool = {
      v: "select",
      g: "edit",
      l: "paint",
      h: "pan",
      b: "box",
      p: "polygon",
      d: "brush-add",
      e: "brush-subtract",
    }[e.key.toLowerCase()];
    if (tool) {
      this.setTool(tool);
      return;
    }
    const cls = this.project.classes.find(
      (c) => !c.archived && c.shortcut && c.shortcut === e.key,
    );
    if (cls) {
      this.promptState.class_id = cls.id;
      this.invalidateAssist();
      this.renderInspector();
    }
  }
  setTool(tool) {
    if (this.busy || this.failed) {
      notify("Finish saving or resolve the save error first.", true);
      return;
    }
    this.layer.setTool(tool);
    for (const [key, b] of this.toolButtons)
      b.setAttribute("aria-pressed", String(key === tool));
    const hints = {
      select:
        "Select mode · Click an object · Shift/Ctrl-click toggles selection · Alt-click cycles overlaps",
      paint:
        "Paint class mode · Choose a class, then click a mask to label it · Acceptance is a separate option",
      edit: "Edit geometry mode · Drag a shape or vertex · Double-click an edge to add · Shift-click a vertex to remove",
      pan: "Drag to move the full image",
      box: "Drag from one corner to the opposite corner",
      polygon: "Click vertices · Enter or click the first vertex to finish",
      "brush-add":
        "Paint missing mask pixels · Existing components and holes stay intact",
      "brush-subtract": "Erase mask pixels · Use this to create holes",
      positive: this.promptState.mode === "independent"
        ? "Click once inside each object · Ctrl+D removes the last point · Right-drag to pan"
        : "Click inside the object · Ctrl+D removes the last point · Right-drag to pan",
      negative:
        "Click an area to exclude · Ctrl+D removes the last point · Right-drag to pan",
      "prompt-box": "Drag a box around the object · Then preview with SAM",
    };
    this.hint.textContent = t(hints[tool]);
    this.canvas.dataset.mode = tool;
    this.renderInspector();
  }
  setStatus(message, error = false) {
    this.saveStatus.textContent = t(message);
    this.saveStatus.className = `editor-status ${error ? "error" : this.busy ? "saving" : ""}`;
  }
  async loadScene() {
    const seq = (this.loadSeq = (this.loadSeq || 0) + 1);
    const data = await this.app.api.get(this.path + "/annotations");
    if (!this.alive || seq !== this.loadSeq) return;
    const selected = this.layer.selected;
    const priorIds = new Set(this.objects.map((o) => o.id));
    this.objects = data.annotations || [];
    this.sceneLoaded = true;
    for (const object of this.objects)
      if (!this.objectOrder.has(object.id))
        this.objectOrder.set(object.id, this.objectOrder.size);
    this.objects.sort(
      (a, b) => this.objectOrder.get(a.id) - this.objectOrder.get(b.id),
    );
    this.revision = data.revision;
    this.layer.objects = this.objects;
    this.layer.selected = this.objects.some((o) => o.id === selected)
      ? selected
      : (this.busy
          ? this.objects.find((o) => !priorIds.has(o.id))?.id
          : null) || null;
    const activeIds = new Set(
      this.objects
        .filter((o) => !["rejected", "superseded"].includes(o.status))
        .map((o) => o.id),
    );
    this.layer.selection = new Set(
      [...this.layer.selection].filter((id) => activeIds.has(id)),
    );
    if (this.layer.selected && !activeIds.has(this.layer.selected))
      this.layer.selected = null;
    if (this.layer.selected) this.layer.selection.add(this.layer.selected);
    this.layer.pruneHiddenSelection();
    this.layer.maskCache.clear();
    this.layer.render();
    this.renderInspector();
    if (!this.busy) this.setStatus("saved");
  }
  async mutate(action, localDraft = null) {
    if (this.busy) throw Error("A save is already in progress.");
    if (this.failed) throw Error("Retry or reload the failed save first.");
    this.busy = true;
    this.layer.locked = true;
    this.invalidateAssist();
    this.setStatus("saving");
    try {
      await action();
      await this.loadScene();
      await this.app.refreshProject();
      this.project = this.app.project;
      this.layer.classes = this.project.classes;
      this.image =
        this.app.project.images.find((i) => i.id === this.image.id) ||
        this.image;
      this.failed = null;
      this.setStatus("saved");
      return true;
    } catch (e) {
      this.failed = { action, draft: localDraft, error: e };
      this.setStatus(
        e.status === 409 ? "conflict" : `Not saved: ${e.message}`,
        true,
      );
      if (localDraft)
        this.layer.preview = {
          id: "failed-local-draft",
          geometry: localDraft,
          class_id: this.promptState.class_id,
          status: "draft",
        };
      notify(e.message, true);
      return false;
    } finally {
      this.busy = false;
      this.layer.locked = !!this.failed;
      if (!this.failed) this.setStatus("saved");
      this.renderInspector();
      this.layer.render();
    }
  }
  async saveGeometry(object, geometry) {
    const expected = this.revision,
      source = object?.source || { kind: "manual" };
    await this.mutate(
      () =>
        object
          ? this.app.api.patch(this.base + `/annotations/${object.id}`, {
              geometry,
              status: "draft",
              expected_revision: expected,
            })
          : this.app.api.post(this.path + "/annotations", {
              class_id: this.promptState.class_id,
              geometry,
              status: "draft",
              source,
              expected_revision: expected,
            }),
      geometry,
    );
  }
  async updateSelected(patch) {
    const o = this.objects.find((o) => o.id === this.layer.selected);
    if (!o) throw Error("Select an object first.");
    await this.mutate(() =>
      this.app.api.patch(this.base + `/annotations/${o.id}`, {
        ...patch,
        expected_revision: this.revision,
      }),
    );
  }
  async deleteSelected() {
    if (activeSelection(this).length > 1)
      return selectionAction(this, "delete");
    const o = this.objects.find((o) => o.id === this.layer.selected);
    if (!o) return;
    if (
      !(await confirmAction(
        "Delete object?",
        "Delete this object from the image? Persistent undo is available.",
        "Delete object",
      ))
    )
      return;
    await this.mutate(() =>
      this.app.api.delete(this.base + `/annotations/${o.id}`, {
        expected_revision: this.revision,
      }),
    );
  }
  async history(action) {
    if (this.layer.draft.length) {
      if (action === "undo") {
        this.layer.draft.pop();
        this.layer.render();
        return;
      }
      throw Error("Finish or cancel the polygon before redo.");
    }
    await this.mutate(() =>
      this.app.api.post(this.path + "/history", {
        action,
        expected_revision: this.revision,
      }),
    );
  }
  async complete() {
    if (this.busy || this.failed || this.layer.draft.length)
      throw Error(
        "Finish drawing and save edits before marking the image reviewed.",
      );
    if (!this.sceneLoaded)
      throw Error("Wait for the saved annotations to load before confirming review.");
    if (this.assistController || this.candidates?.length)
      throw Error("Save or cancel the SAM preview before confirming the image review.");
    if (this.objects.some(needsReview)) return this.pendingReviewDialog();
    const attestation = check("attest", "reviewAttestation");
    const d = dialog(
      "Confirm image review",
      el(
        "div",
        {},
        el(
          "p",
          {},
          `${this.objects.filter((o) => o.status === "accepted").length} accepted objects. Review every object and reject unwanted proposals first.`,
        ),
        attestation,
        button(
          "Confirm full image review",
          async () => {
            if (!attestation.querySelector("input").checked)
              throw Error("Confirm that you reviewed the entire image.");
            let pendingReview = false;
            const saved = await this.mutate(async () => {
              try {
                await this.app.api.post(this.path + "/complete", {
                  complete: true,
                  attest: true,
                  expected_revision: this.revision,
                });
              } catch (error) {
                // This specific validation rejects only the completion request.
                // No annotation was edited; refresh saved state without locking
                // editing as if an annotation save had failed. Other failures
                // (including conflicts and network errors) retain recovery.
                if (error.status !== 400 || error.message !==
                    "Accept or reject all draft/proposed objects before marking this image complete") throw error;
                pendingReview = true;
              }
            });
            if (saved) {
              d.close();
              if (pendingReview) this.pendingReviewDialog();
            }
          },
          "primary",
        ),
      ),
    );
  }
  pendingReviewDialog() {
    const pending = this.objects.filter(needsReview),
      unassigned = pending.filter(o => !o.class_id).length;
    if (!pending.length) {
      notify("Review state refreshed. Check the full image, then use Mark reviewed again.");
      return;
    }
    const d = dialog("Image review is not finished", el("div", {},
      el("p", {}, "Your annotations are saved. Mark reviewed finishes the whole image; it does not accept its masks for you."),
      el("p", { class: "inline-warning", role: "status" },
        `${pending.length} ${pending.length === 1 ? "mask still needs" : "masks still need"} a review decision, including any hidden masks.`),
      el("ul", {},
        el("li", {}, `${pending.length - unassigned} labeled: inspect each mask, then choose Accept selected or Reject selected.`),
        unassigned > 0 && el("li", {}, `${unassigned} unassigned: choose or create a class and Assign class before accepting, or reject unwanted masks.`)),
      el("p", {}, "Show remaining masks reveals the pending objects and selects one. Accept masks you have checked; reject unwanted ones. Repeat until Pending review is 0, then use Mark reviewed again."),
      row(button("Show remaining masks", () => {
        // Run after the dialog restores focus to its previous control.
        d.addEventListener("close", () => { if (this.alive) this.reviewPending(true); }, { once: true });
        d.close();
      }, "primary"), button("Continue editing", () => d.close())),
    ));
    return d;
  }
  syncPrompts(redraw = true) {
    this.layer.points = this.promptState.points;
    this.layer.labels = this.promptState.labels;
    this.layer.promptBox = this.promptState.box;
    this.layer.numberedPoints = this.promptState.mode === "independent";
    if (redraw) this.layer.render();
  }
  signature() {
    return JSON.stringify([
      this.image.id,
      this.image.sha256,
      this.revision,
      this.promptState.mode === "independent" ? null : this.layer.selected,
      this.promptState,
      this.app.device,
    ]);
  }
  invalidateAssist(redraw = true) {
    if (this.candidates?.length || this.assistController)
      this.assistMessage =
        "Previous preview discarded. Request a preview for the current prompts.";
    this.assistController?.abort();
    this.assistController = null;
    this.candidates = [];
    this.batchPreview = false;
    this.batchJobId = null;
    this.layer.previews = [];
    this.layer.preview = null;
    this.assistSignature = null;
    if (redraw) {
      this.layer.render();
      this.renderGuide();
    }
  }
  async assist() {
    if (this.busy || this.failed)
      throw Error("Save all edits before requesting a preview.");
    if (!this.promptState.model_id)
      throw Error("Select a registered SAM model.");
    if (!this.promptState.points.length && !this.promptState.box)
      throw Error("Place a point or draw a prompt box first.");
    const independent = this.promptState.mode === "independent";
    if (independent && (this.promptState.box || this.promptState.labels.some(label => label !== 1)))
      throw Error("One mask per point uses positive points only. Clear prompts or choose Refine one object.");
    if (independent) this.layer.select(null);
    this.invalidateAssist();
    const signature = this.signature(),
      controller = new AbortController();
    this.assistController = controller;
    this.assistSignature = signature;
    this.assistMessage = "Submitting preview…";
    this.renderInspector();
    try {
      const job = await this.app.api.post(
        this.path + "/assist",
        {
          ...clone(this.promptState),
          annotation_id: independent ? undefined : this.layer.selected || undefined,
          expected_revision: this.revision,
          device: this.app.device,
        },
        controller.signal,
      );
      this.assistJob = job;
      const complete = await this.app.api.waitJob(job, {
        signal: controller.signal,
        onProgress: (j) => {
          if (this.alive) {
            this.assistMessage = `${j.status}: ${j.progress?.stage || "SAM preview"}`;
            this.renderInspector();
          }
        },
      });
      if (
        !this.alive ||
        signature !== this.signature() ||
        this.assistController !== controller
      ) {
        this.assistMessage = t("staleAssist");
        return;
      }
      const result = complete.result || {};
      this.assistJob = complete;
      const binding = result.binding;
      if (!binding)
        throw Error("SAM response is missing its image and prompt binding.");
      if (independent && (result.mode !== "independent" || binding.mode !== "independent"))
        throw Error("SAM returned a different prompt mode. Preview discarded.");
      if (
        JSON.stringify([
          binding.image_id,
          binding.image_sha256,
          binding.project_revision,
          binding.annotation_id || null,
          binding.model_id,
          binding.class_id || null,
          binding.points || [],
          binding.labels || [],
          binding.box || null,
        ]) !==
        JSON.stringify([
          this.image.id,
          this.image.sha256,
          this.revision,
          independent ? null : this.layer.selected || null,
          this.promptState.model_id,
          this.promptState.class_id || null,
          this.promptState.points,
          this.promptState.labels,
          this.promptState.box,
        ])
      )
        throw Error(t("staleAssist"));
      this.candidates = [
        ...(result.annotations || []),
        ...(independent ? [] : result.alternatives || []),
      ].map((c, i) => ({
        ...(c.geometry ? c : { geometry: c }),
        id: `preview-${i}`,
        source: c.source || {
          kind: result.provider || "sam2",
          model_id: result.model_id || this.promptState.model_id,
        },
        class_id: this.promptState.class_id,
        status: "proposal",
      }));
      if (!this.candidates.length)
        throw Error(
          "The provider returned no mask candidates. Adjust the prompts.",
        );
      this.previewIndex = 0;
      this.batchPreview = independent;
      this.batchJobId = independent ? complete.id : null;
      this.layer.preview = independent ? null : this.candidates[0];
      this.layer.previews = independent ? this.candidates : [];
      this.assistMessage = independent
        ? `${this.candidates.length} independent masks ready from ${result.requested_points} points. ${result.empty_points?.length || 0} points returned no mask. Previews are not saved yet; add them together as drafts, then review.`
        : "Preview ready. Inspect an alternative, then add it as a draft.";
      this.layer.render();
    } catch (e) {
      if (e.name !== "AbortError") {
        this.assistMessage = e.message;
        notify(e.message, true);
      }
    } finally {
      if (this.assistController === controller) this.assistController = null;
      if (this.alive) this.renderInspector();
    }
  }
  async acceptPreview() {
    if (this.assistSignature !== this.signature())
      throw Error(t("staleAssist"));
    if (this.batchPreview) {
      const jobId = this.batchJobId, count = this.candidates.length;
      const saved = await this.mutate(() => this.app.api.post(this.path + `/assist/${jobId}/apply`, {
        expected_revision: this.revision,
      }));
      if (saved) {
        this.promptState.points = [];
        this.promptState.labels = [];
        this.promptState.box = null;
        this.syncPrompts();
        this.assistMessage = `${count} separate masks saved as drafts. They still need review${this.layer.hideAssigned ? "; assigned masks may be hidden by your display filter" : ""}. Undo restores the whole addition.`;
        this.renderInspector();
      }
      return;
    }
    const candidate = this.candidates?.[this.previewIndex];
    if (!candidate) throw Error("Choose a current preview first.");
    const id = this.layer.selected,
      geometry = clone(candidate.geometry),
      source = { ...clone(candidate.source), operation: "sam_assist", assist_job_id: this.assistJob?.id };
    await this.mutate(
      () =>
        id
          ? this.app.api.patch(this.base + `/annotations/${id}`, {
              geometry,
              status: "draft",
              class_id: this.promptState.class_id,
              source,
              expected_revision: this.revision,
            })
          : this.app.api.post(this.path + "/annotations", {
              geometry,
              status: "draft",
              class_id: this.promptState.class_id,
              source,
              expected_revision: this.revision,
            }),
      geometry,
    );
  }
  refreshVisibility() {
    this.layer.pruneHiddenSelection();
    this.invalidateAssist();
    this.layer.render();
    this.renderInspector();
  }
  reviewPending(selectNext = false) {
    const pending = this.objects.filter(needsReview),
      index = pending.findIndex(o => o.id === this.layer.selected),
      next = pending[(index + 1) % pending.length];
    this.setTool("select");
    this.layer.hideAssigned = this.layer.hideGeneratedSAM = false;
    this.layer.reviewPendingOnly = true;
    this.layer.showProposals = this.layer.showUnassigned = true;
    this.layer.hiddenClasses.clear();
    this.layer.hiddenObjects.clear();
    this.layer.hiddenLayers.clear();
    this.refreshVisibility();
    if (selectNext && next) this.layer.select(next.id);
    focusControl(this.inspector, selectNext && next?.class_id
      ? "[data-review-action=accept]" : "[name=selection-class]");
  }
  renderGuide() {
    if (!this.workflowGuide || !this.layer) return;
    this.workflowGuide.replaceChildren(editorGuide(this, {
      generate: () => generateDialog(this.app, this.image.id),
      sam: () => {
        if (!this.candidates?.length && !this.assistController) this.setTool("positive");
        focusControl(this.inspector, "[name=sam-model]");
      },
      review: () => this.reviewPending(),
      nextPending: () => this.reviewPending(true),
    }));
  }
  renderInspector() {
    if (!this.inspector) return;
    this.renderGuide();
    const clsOptions = [
        ["", "Unassigned"],
        ...this.project.classes
          .filter((c) => !c.archived)
          .map((c) => [c.id, c.name]),
      ],
      selected = this.objects.find((o) => o.id === this.layer.selected);
    const activeClass = select(
      "active-class",
      [...clsOptions, ["__new", "Create new class…"]],
      this.promptState.class_id || "",
      {
        onchange: (e) => {
          if (e.target.value === "__new") {
            e.target.value = this.promptState.class_id || "";
            createClassDialog(this);
            return;
          }
          this.promptState.class_id = e.target.value || null;
          this.invalidateAssist();
        },
      },
    );
    const objectList = el(
      "div",
      { class: "object-list" },
      this.objects.map((o, index) => {
        const visible = input(`object-visible-${o.id}`, "", "checkbox", {
          "aria-label": `Show object ${index + 1}`,
          checked: !this.layer.hiddenObjects.has(o.id),
          onchange: (e) => {
            e.target.checked
              ? this.layer.hiddenObjects.delete(o.id)
              : this.layer.hiddenObjects.add(o.id);
            this.refreshVisibility();
          },
        });
        return el(
          "div",
          { class: "object-row", dataset: { objectId: o.id } },
          visible,
          input(`object-select-${o.id}`, "", "checkbox", {
            "aria-label": `Include object ${index + 1} in selection`,
            checked: this.layer.selection.has(o.id),
            disabled: o.status === "superseded" || !this.layer.visible(o),
            onchange: () => this.layer.select(o.id, true),
          }),
          button(
            el(
              "span",
              {},
              el("span", {
                class: "class-swatch",
                style: `background:${this.layer.color(o)}`,
              }),
              `${this.project.classes.find((c) => c.id === o.class_id)?.name || "Unassigned"} ${index + 1}`,
              el("small", {}, ` · ${reviewStatus(o)}`),
              !this.layer.visible(o) && el("small", {}, " · Hidden"),
            ),
            () => {
              this.layer.select(o.id);
            },
            this.layer.selection.has(o.id) ? "active" : "",
            {
              "aria-label": `Select object ${index + 1}`,
              "aria-pressed": String(this.layer.selection.has(o.id)),
              disabled: !this.layer.visible(o),
              title: `Object ${o.id} · ${o.status}`,
            },
          ),
        );
      }),
    );
    const selectedPanel =
      selected && !["rejected", "superseded"].includes(selected.status)
        ? el(
            "section",
            {},
            el("h2", {}, "Selected object"),
            field(
              "Object class",
              select("object-class", clsOptions, selected.class_id || "", {
                onchange: async (e) => {
                  await this.updateSelected({
                    class_id: e.target.value || null,
                    status: "draft",
                  });
                },
              }),
            ),
            row(
              badge(selected.geometry.type),
              badge(reviewStatus(selected)),
              badge(
                selected.human_verified
                  ? "Human verified"
                  : selected.review_actor === "automated_qa"
                    ? "Automated QA"
                    : "Unverified",
              ),
            ),
            el(
              "p",
              {},
              `Source: ${selected.source?.kind || "unknown"}${selected.source?.model_id ? " · " + selected.source.model_id : ""}`,
            ),
            row(
              button(
                "Accept object",
                () => {
                  if (!selected.class_id)
                    throw Error("Assign a class before accepting.");
                  return this.updateSelected({ status: "accepted" });
                },
                "primary",
              ),
              button("Reject", () =>
                this.updateSelected({ status: "rejected" }),
              ),
            ),
            row(
              button("Delete", () => this.deleteSelected(), "danger"),
              selected.geometry.type !== "mask" &&
                button("Convert to mask", async () => {
                  if (
                    await confirmAction(
                      "Convert geometry to mask?",
                      "Rasterize this shape at original image resolution for brush editing? Undo can restore the vector shape.",
                      "Convert to mask",
                    )
                  )
                    await this.saveGeometry(selected, {
                      type: "mask",
                      rle: encodeRLE(
                        rasterize(
                          selected.geometry,
                          this.image.width,
                          this.image.height,
                        ),
                      ),
                    });
                }),
            ),
            button("Refine with SAM", () => {
              this.promptState.mode = "single";
              this.promptState.points = [];
              this.promptState.labels = [];
              this.promptState.box = this.layer.bounds(selected);
              this.invalidateAssist();
              this.syncPrompts();
              this.setTool("positive");
            }),
            details("Object provenance", selected),
          )
        : selected
          ? el(
              "section",
              {},
              el("h2", {}, "Inactive object history"),
              badge(selected.status),
              el(
                "small",
                {},
                "This object is excluded from the active scene. Persistent Undo can restore the recorded operation.",
              ),
              selected.status === "rejected" &&
                button(
                  "Delete rejected object",
                  () => selectionAction(this, "delete", null, [selected.id]),
                  "danger",
                ),
              details("Object provenance", selected),
            )
          : el("p", {}, "Select an object on the image or in the list.");
    const brushSize = input(
        "brush-size",
        String(this.layer.brushRadius),
        "range",
        {
          min: 1,
          max: 150,
          oninput: (e) => {
            this.layer.brushRadius = Number(e.target.value);
            brushText.textContent = `${e.target.value} px radius`;
          },
        },
      ),
      brushText = el("small", {}, `${this.layer.brushRadius} px radius`);
    const samModels = (this.models || []).filter((m) =>
      ["sam2", "sam3"].includes(m.provider),
    );
    const sam = el(
      "section",
      {},
      el("h2", {}, "SAM assistance"),
      button(`Label & accept SAM Assist drafts (${assistDrafts(this).length})…`, () => assistBatchDialog(this), "", {
        disabled: !assistDrafts(this).length || this.busy || !!this.failed,
        "data-testid": "review-assist-drafts",
      }),
      field("Prompt mode", select("sam-prompt-mode", [
        ["independent", "One mask per point · multiple objects"],
        ["single", "Refine one object · points / box"],
      ], this.promptState.mode, { onchange: (e) => {
        this.promptState.mode = e.target.value;
        this.promptState.points = [];
        this.promptState.labels = [];
        this.promptState.box = null;
        this.invalidateAssist();
        this.syncPrompts();
        this.renderInspector();
      }})),
      this.promptState.mode === "independent" && el("small", {}, "Choose SAM positive point (+), click once inside each object anywhere in the image, then preview all points together. Each point produces a separate mask; the masks are not merged. There is no fixed point-count limit. More points take longer; processing stays sequential and can be cancelled."),
      el("small", {}, "Hold the right mouse button and drag to pan the image. Press Ctrl+D to remove the last SAM point; repeat to remove earlier points. Saved masks stay unchanged."),
      field(
        "Model",
        select(
          "sam-model",
          [
            ["", "Choose a local SAM model"],
            ...samModels.map((m) => [m.id, m.name || m.architecture || m.id]),
          ],
          this.promptState.model_id,
          {
            onchange: (e) => {
              this.promptState.model_id = e.target.value;
              this.invalidateAssist();
            },
          },
        ),
      ),
      !samModels.length &&
        el(
          "p",
          {},
          "Add SAM2 or local SAM3 weights in Models & AI. Manual tools remain available.",
        ),
      el(
        "small",
        {},
        `${this.promptState.labels.filter((x) => x === 1).length} positive · ${this.promptState.labels.filter((x) => x === 0).length} negative${this.promptState.box ? " · box prompt" : ""}`,
      ),
      row(
        button("SAM settings", () => samSettingsDialog(this)),
        button(this.promptState.mode === "independent" ? `Preview ${this.promptState.points.length} masks` : "Preview mask", () => this.assist(), "", {
          disabled: !!this.assistController,
        }),
        this.assistController &&
          button("Cancel SAM request", async () => {
            if (this.assistJob)
              await this.app.api.post(`/api/jobs/${this.assistJob.id}/cancel`);
            this.invalidateAssist();
            this.renderInspector();
          }),
        button("Clear prompts", () => {
          this.promptState.points = [];
          this.promptState.labels = [];
          this.promptState.box = null;
          this.invalidateAssist();
          this.syncPrompts();
          this.renderInspector();
        }),
        this.promptState.points.length > 0 && button("Remove last point", () => removeLastAssistPoint(this)),
      ),
      el(
        "p",
        { "aria-live": "polite" },
        this.assistMessage ||
          (this.promptState.mode === "independent"
            ? "Each positive point represents a separate object. Use Refine one object for negative points or boxes."
            : "Positive means object pixels; negative excludes pixels."),
      ),
      this.candidates?.length &&
        el(
          "div",
          {},
          !this.batchPreview && field(
            "Alternative preview",
            select(
              "sam-alternative",
              this.candidates.map((c, i) => [
                String(i),
                `Candidate ${i + 1}${Number.isFinite(c.score) ? " · score " + c.score.toFixed(3) : ""}`,
              ]),
              String(this.previewIndex),
              {
                onchange: (e) => {
                  this.previewIndex = Number(e.target.value);
                  this.layer.preview = this.candidates[this.previewIndex];
                  this.layer.render();
                },
              },
            ),
          ),
          button(this.batchPreview ? `Add all ${this.candidates.length} masks as drafts` : "Use preview as draft", () => this.acceptPreview(), "primary"),
          button("Cancel preview", async () => {
            if (
              this.assistJob &&
              ["queued", "running"].includes(this.assistJob.status)
            )
              await this.app.api.post(`/api/jobs/${this.assistJob.id}/cancel`);
            this.invalidateAssist();
            this.assistMessage = "Preview cancelled.";
            this.renderInspector();
          }),
          el(
            "small",
            {},
            "SAM scores are mask-quality estimates, not class probabilities.",
          ),
        ),
    );
    const maskFill = check("show-mask-fill", "Show mask fill", this.layer.showMaskFill);
    maskFill.querySelector("input").addEventListener("change", (e) => {
      this.layer.showMaskFill = e.target.checked;
      this.layer.render();
    });
    const layers = el(
      "section",
      {},
      el("h2", {}, "Layers & display"),
      maskFill,
      el("small", {}, "Turn off the translucent tint over saved objects. Masks remain selectable; hover or select to see their bounds. SAM previews and brush feedback remain visible."),
      field(
        "Overlay opacity",
        input("opacity", String(this.layer.opacity), "range", {
          min: 0,
          max: 1,
          step: 0.05,
          oninput: (e) => {
            this.layer.opacity = Number(e.target.value);
            this.layer.render();
          },
        }),
      ),
      ...this.project.classes.map((c) => {
        const control = check(
          `visible-${c.id}`,
          c.name,
          !this.layer.hiddenClasses.has(c.id),
        );
        control.querySelector("input").addEventListener("change", (e) => {
          e.target.checked
            ? this.layer.hiddenClasses.delete(c.id)
            : this.layer.hiddenClasses.add(c.id);
          this.refreshVisibility();
        });
        return control;
      }),
      ...["Proposals", "Rejected", "Superseded", "Unassigned"].map((label) => {
        const key = "show" + label,
          control = check(
            `show-${label}`,
            `Show ${label.toLowerCase()}`,
            this.layer[key],
          );
        control.querySelector("input").addEventListener("change", (e) => {
          this.layer[key] = e.target.checked;
          this.refreshVisibility();
        });
        return control;
      }),
    );
    const failed =
      this.failed &&
      el(
        "section",
        { class: "inline-warning", "data-save-recovery": "true" },
        el("strong", {}, "Changes not saved"),
        el("p", {}, this.failed.error.message),
        row(
          button("Retry save", async () => {
            const { action, draft, error } = this.failed;
            if (error.status === 409)
              throw Error("Reload the current revision before editing again.");
            this.failed = null;
            this.layer.preview = null;
            await this.mutate(action, draft);
          }),
          button("Reload saved state", async () => {
            if (
              await confirmAction(
                "Discard unsaved edit?",
                "The local failed edit will be discarded. Saved history remains available.",
                "Reload",
              )
            ) {
              this.failed = null;
              this.layer.locked = false;
              this.layer.preview = null;
              await this.loadScene();
            }
          }),
        ),
      );
    this.inspector.replaceChildren(
      el(
        "section",
        {},
        el("h2", {}, "Annotation"),
        field("Class for new objects", activeClass),
        field("Brush radius", brushSize),
        brushText,
        this.layer.tool === "polygon" &&
          button("Finish polygon", () => this.layer.finishPolygon()),
      ),
      failed || "",
      selectionToolbar(this),
      el(
        "section",
        {},
        row(el("h2", {}, "Objects"), badge(String(this.objects.length))),
        el(
          "div",
          { class: "object-counts" },
          ...[
            [
              "Proposals",
              this.objects.filter((o) => o.status === "proposal").length,
            ],
            [
              "Unassigned",
              this.objects.filter(
                (o) =>
                  !o.class_id && !["rejected", "superseded"].includes(o.status),
              ).length,
            ],
            [
              "Labeled drafts",
              this.objects.filter((o) => o.class_id && o.status === "draft")
                .length,
            ],
            [
              "Accepted",
              this.objects.filter((o) => o.status === "accepted").length,
            ],
            [
              "Rejected",
              this.objects.filter((o) => o.status === "rejected").length,
            ],
            [
              "Superseded",
              this.objects.filter((o) => o.status === "superseded").length,
            ],
          ].map(([label, count]) => el("span", {}, `${label}: ${count}`)),
        ),
        objectList,
        selectedPanel,
      ),
      layerControls(this),
      el(
        "section",
        {},
        el("h2", {}, "Annotation tile-edge recovery"),
        el("small", {}, "Off by default · explicit scope and review required"),
        row(
          button("Set recovery scope", () => annotationBoundaryDialog(this)),
          button("Review tile-edge changes", () =>
            reviewAnnotationBoundary(this),
          ),
        ),
      ),
      sam,
      layers,
    );
  }
  async canLeave() {
    if (this.busy) {
      notify("Wait for the current save to finish.", true);
      return false;
    }
    if (this.failed || this.layer.draft.length)
      return confirmAction(
        "Leave unsaved work?",
        "The local draft has not been saved. Leave and discard it?",
        "Discard draft",
      );
    return true;
  }
  destroy() {
    flushMaskDisplay(this);
    this.alive = false;
    this.events.abort();
    this.assistController?.abort();
    this.layer.destroy();
  }
}
