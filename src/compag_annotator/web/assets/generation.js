import { DevicePicker } from "./devices.js";
import {
  el,
  button,
  input,
  select,
  field,
  check,
  row,
  badge,
  dialog,
  form,
  notify,
  projectPath,
} from "./ui.js";
import { TilingPanel } from "./tiling.js";
const list = (value, key) =>
  Array.isArray(value) ? value : value?.[key] || [];
export async function generateDialog(
  app,
  imageId = app.currentImageId,
  initialScope = "current",
) {
  if (!app.project?.images.length)
    throw Error("Import an image before generating masks.");
  const p = app.project,
    base = projectPath(p.id);
  const [modelsResult, layersResult, catalogResult] = await Promise.all([
    app.api.get("/api/models"),
    app.api.get(base + "/generation_layers"),
    app.api.get("/api/models/catalog"),
  ]);
  const models = list(modelsResult, "models"),
    layers = list(layersResult, "layers"),
    catalog = list(catalogResult, "catalog");
  const defaults = catalog.find(c => c.provider === "sam2")?.capability_details?.generate_proposals?.defaults || {};
  const current = p.images.find((i) => i.id === imageId) || p.images[0],
    selected = [...app.selectedImages].filter((id) =>
      p.images.some((i) => i.id === id),
    );
  const scope = select(
    "generation-scope",
    [
      ["current", `Current image · ${current.name}`],
      ["selected", `Selected images (${selected.length})`],
      ...p.rounds.map((r) => [
        `round:${r.id}`,
        `Round ${r.number} (${r.image_ids.length} images)`,
      ]),
    ],
    initialScope,
  );
  const model = select(
    "generation-model",
    [
      ["", "Choose a local SAM2 model"],
      ...models
        .filter((m) => ["sam2", "sam3"].includes(m.provider))
        .map((m) => [m.id, m.name || m.architecture || m.id]),
    ],
    models.find((m) => m.provider === "sam2")?.id || "",
  );
  const devices = new DevicePicker(app, models.find(m => m.id === model.value)?.provider || "sam2");
  model.addEventListener("change", () => devices.load(models.find(m => m.id === model.value)?.provider || "sam2"));
  const reason = el("p", { role: "status" }),
    batch = check(
      "batch-confirm",
      "I confirm generation for every image in this batch",
    ),
    mode = select("layer-mode", [
      ["new", "Create a new proposal layer"],
      ["replace_unreviewed", "Replace unreviewed proposals in one layer"],
    ]);
  const replace = select("replace-layer", [
    ["", "Choose a proposal layer"],
    ...layers
      .filter((l) => l.image_id === current.id)
      .map((l, i) => [
        l.id,
        `Layer ${i + 1} · ${l.completed_tiles}/${l.total_tiles} tiles · ${l.status}`,
      ]),
  ]);
  const density = select(
    "sampling-preset",
    [
      ["8", "Quick inspection · 8 × 8 points per tile"],
      ["16", "Standard · 16 × 16 points per tile"],
      ["32", "Dense · 32 × 32 points per tile"],
      ["64", "Default settings · 64 × 64 points per tile"],
      ["custom", "Custom point grid"],
    ],
    String(defaults.points_per_side ?? 64),
  );
  const points = input("points_per_side", defaults.points_per_side ?? 64, "number", {
    min: 1,
    max: 64,
    step: 1,
    required: true,
  });
  density.addEventListener("change", () => {
    points.disabled = density.value !== "custom";
    if (density.value !== "custom") points.value = density.value;
    work();
  });
  points.disabled = true;
  const crop = input("crop_n_layers", "0", "number", {
      min: 0,
      max: 0,
      step: 1,
      readOnly: true,
    }),
    workNote = el("p", { class: "inline-warning" });
  const boundary = check(
    "boundary_opt_in",
    "Enable tile-edge recovery for this generation job only",
    false,
  );
  const boundaryLimit = input("boundary_limit", "3", "number", {
    min: 1,
    max: 20,
    step: 1,
  });
  const panel = new TilingPanel(
    app,
    current,
    p.processing_defaults,
    async () => {
      if (app.editor?.alive) {
        app.editor.project = app.project;
        await app.editor.loadScene();
      }
    },
  );
  const getIds = () =>
    scope.value === "current"
      ? [current.id]
      : scope.value === "selected"
        ? selected
        : p.rounds.find((r) => r.id === scope.value.slice(6))?.image_ids || [];
  const updateScope = () => {
    const ids = getIds();
    batch.hidden = ids.length <= 1;
    replace.closest("label").hidden = mode.value !== "replace_unreviewed";
    panel.setImage(p.images.find((i) => i.id === ids[0]) || current);
  };
  scope.addEventListener("change", updateScope);
  mode.addEventListener("change", updateScope);
  function capability() {
    const record = models.find((m) => m.id === model.value),
      spec = catalog.find((c) => c.provider === record?.provider);
    const cap =
      record?.capability_details?.generate_proposals ||
      spec?.capability_details?.generate_proposals;
    return {
      record,
      supported:
        !!record && record.provider === "sam2" && cap?.supported !== false,
      reason: cap?.reason,
    };
  }
  let generateButton;
  const updateModel = () => {
    const cap = capability();
    if (generateButton) generateButton.disabled = !cap.supported;
    reason.textContent = cap.supported
      ? "Automatic class-agnostic masks. No classes or YOLO model are required. Masks begin unassigned and unreviewed."
      : cap.reason ||
        (cap.record?.provider === "sam3"
          ? "SAM3 automatic generation is unavailable in this adapter. Its point and box assistance remains available."
          : "Register a trusted local SAM2 checkpoint in Models & AI before generating.");
  };
  model.addEventListener("change", updateModel);
  updateModel();
  function work() {
    const n = Number(points.value),
      k = Number(crop.value);
    workNote.textContent = `${n * n} point prompts per crop. Internal crop layers: ${k}. Additional internal crops would compound the work of the external grid. This runtime supports 0 internal crop layers; use the shared tile controls. Small-region cleanup is off by default.`;
  }
  points.addEventListener("input", work);
  crop.addEventListener("input", work);
  work();
  let submissionIdentity, requestKey;
  const f = form(
    [
      el(
        "p",
        {},
        "Generate masks on the original image, then choose which objects matter. Review and label proposals at full resolution; generation does not certify completeness.",
      ),
      row(
        field("Generation scope", scope),
        field("Automatic mask model", model),
      ),
      reason,
      batch,
      panel.root,
      row(field("Proposal layer", mode), field("Layer to replace", replace)),
      el(
        "small",
        {},
        "Replace targets only unreviewed, unedited proposals in the chosen layer. Accepted and manually edited objects stay protected. To continue an interrupted layer, use Resume in the layer list or Retry in Jobs.",
      ),
      field("Sampling density", density),
      field("Points per side", points),
      workNote,
      el("p", {}, "Default settings use 64 × 64 points per tile and 512 points per batch. Choose your SAM2 model in Models & AI. Adjust the advanced settings for your images and device; a smaller batch uses less memory."),
      el(
        "details",
        {},
        el("summary", {}, "Advanced automatic mask settings"),
        row(
          field(
            "Points per batch",
            input("points_per_batch", defaults.points_per_batch ?? 512, "number", {
              min: 1,
              max: 512,
              step: 1,
            }),
          ),
          field("Internal crop layers", crop),
        ),
        row(
          field(
            "Predicted IoU threshold",
            input("pred_iou_thresh", defaults.pred_iou_thresh ?? .8, "number", {
              min: 0,
              max: 1,
              step: 0.01,
            }),
          ),
          field(
            "Stability threshold",
            input("stability_score_thresh", defaults.stability_score_thresh ?? .88, "number", {
              min: 0,
              max: 1,
              step: 0.01,
            }),
          ),
          field(
            "Box NMS threshold",
            input("box_nms_thresh", "0.7", "number", {
              min: 0,
              max: 1,
              step: 0.01,
            }),
          ),
        ),
        row(
          field("Stability score offset", input("stability_score_offset", defaults.stability_score_offset ?? 1, "number", { min: 0, max: 10, step: .1 })),
          field("Mask logit threshold", input("mask_threshold", defaults.mask_threshold ?? 0, "number", { min: -32, max: 32, step: .1 })),
        ),
        check("multimask_output", "Generate alternative masks per point", defaults.multimask_output ?? true),
        el("small", {}, "Internal crops are disabled (0 layers), so internal crop overlap and point downscaling have no effect. Use the shared tile controls for full-image coverage. All masks that pass the selected filters are retained."),
        field(
          "Small-region cleanup area (px²)",
          input("min_mask_region_area", "0", "number", {
            min: 0,
            max: 0,
            step: 1,
            readOnly: true,
          }),
          "Cleanup is disabled in this runtime: the optional topology-changing extension is not installed. Components and holes are preserved.",
        ),
        field(
          "Precision",
          select(
            "precision",
            [
              ["float32", "Float32"],
              ["bfloat16", "BFloat16"],
              ["float16", "Float16"],
            ],
            "float32",
          ),
        ),
      ),
      devices.root,
      el(
        "details",
        {},
        el(
          "summary",
          {},
          "Optional annotation tile-edge recovery · off by default",
        ),
        boundary,
        field("Maximum recovery attempts", boundaryLimit),
        el(
          "p",
          {},
          "Scope: only proposals from this generation job touching internal crop seams. Whole-image generation has no internal seam and is not applicable. Recovered pixels are staged for comparison and explicit review; no hidden recovery runs during selection, labeling or merging.",
        ),
      ),
    ],
    async (v) => {
      if (!capability().supported) throw Error(reason.textContent);
      const image_ids = getIds();
      if (!image_ids.length) throw Error("Choose at least one image.");
      if (image_ids.length > 1 && !v["batch-confirm"])
        throw Error("Confirm the batch scope before generating.");
      if (
        v["layer-mode"] === "replace_unreviewed" &&
        (image_ids.length !== 1 || !v["replace-layer"])
      )
        throw Error(
          "Choose one image and the specific proposal layer to replace.",
        );
      await panel.validate();
      const settings = {
        points_per_side: Number(points.value),
        points_per_batch: v.points_per_batch,
        pred_iou_thresh: v.pred_iou_thresh,
        stability_score_thresh: v.stability_score_thresh,
        box_nms_thresh: v.box_nms_thresh,
        stability_score_offset: v.stability_score_offset,
        mask_threshold: v.mask_threshold,
        multimask_output: v.multimask_output,
        crop_n_layers: v.crop_n_layers,
        min_mask_region_area: v.min_mask_region_area,
        precision: v.precision,
      };
      const device = devices.value();
      if (device === "cpu" && settings.precision !== "float32") throw Error("Choose Float32 for CPU.");
      const request = {
        image_ids,
        model_id: model.value,
        tiling: panel.value(),
        settings,
        device,
        batch_confirm: image_ids.length > 1,
        layer_mode: mode.value,
        ...(mode.value === "replace_unreviewed"
          ? { replace_layer_id: replace.value }
          : {}),
        boundary_opt_in: v.boundary_opt_in,
        boundary_limit: v.boundary_limit,
        expected_revision: app.project.revision,
      };
      const signature = JSON.stringify(request);
      if (signature !== submissionIdentity) {
        submissionIdentity = signature;
        requestKey = crypto.randomUUID();
      }
      const job = await app.api.post(base + "/generate", {
        ...request,
        request_key: requestKey,
      });
      d.close();
      app.showJob(job);
    },
    "Generate masks",
  );
  generateButton = f.querySelector("button[type=submit]");
  updateModel();
  const d = dialog("Generate masks", f);
  d.classList.add("wide-dialog");
  d.addEventListener("close", () => panel.destroy());
  updateScope();
  return d;
}

export async function generationLayers(editor) {
  const layers = list(
    await editor.app.api.get(editor.base + "/generation_layers"),
    "layers",
  ).filter((l) => l.image_id === editor.image.id);
  if (!editor.alive) return;
  editor.generationLayers = layers;
  editor.renderInspector();
}
export function layerControls(editor) {
  const layers = editor.generationLayers || [];
  return el(
    "section",
    {},
    row(
      el("h2", {}, "Proposal layers"),
      button("Refresh layers", async () => {
        await editor.app.refreshProject();
        editor.project = editor.app.project;
        await editor.loadScene();
        await generationLayers(editor);
      }),
    ),
    !layers.length &&
      el("small", {}, "No automatic mask layers on this image yet."),
    layers.map((layer, index) => {
      const visible = check(
        `layer-${layer.id}`,
        `Show layer ${index + 1}`,
        !editor.layer.hiddenLayers.has(layer.id),
      );
      visible.querySelector("input").addEventListener("change", (e) => {
        e.target.checked
          ? editor.layer.hiddenLayers.delete(layer.id)
          : editor.layer.hiddenLayers.add(layer.id);
        editor.refreshVisibility();
      });
      return el(
        "div",
        { class: "generation-layer" },
        row(visible, badge(layer.status)),
        el(
          "small",
          {},
          `${layer.completed_tiles} / ${layer.total_tiles} tiles saved · ${layer.created_at || ""}`,
        ),
        layer.status !== "complete" &&
          el(
            "p",
            {},
            "Partial results are available for review; coverage is incomplete.",
          ),
        layer.resume_available === false &&
          el(
            "p",
            { class: "inline-warning" },
            layer.resume_unavailable_reason ||
              "Resume is unavailable for this restored layer. Start a new generation explicitly.",
          ),
        row(
          button(`Select layer ${index + 1}`, () => {
            editor.layer.selection = new Set(
              editor.objects
                .filter(
                  (o) =>
                    o.layer_id === layer.id &&
                    editor.layer.visible(o) &&
                    !["rejected", "superseded"].includes(o.status),
                )
                .map((o) => o.id),
            );
            editor.layer.selected = [...editor.layer.selection][0] || null;
            editor.layer.callbacks.select?.(editor.layer.selected);
            editor.layer.render();
          }),
          ["failed", "cancelled", "interrupted", "partial"].includes(
            layer.status,
          ) &&
            button(
              "Resume unfinished tiles",
              async () => {
                if (layer.resume_available === false)
                  throw Error(
                    layer.resume_unavailable_reason ||
                      "Resume is unavailable for this restored layer. Start a new generation explicitly.",
                  );
                const belongs = (job) =>
                  job.kind === "generate" &&
                  job.payload?.project_id === editor.project.id &&
                  job.payload?.layers?.some(
                    (item) =>
                      item.id === layer.id && item.image_id === editor.image.id,
                  ) &&
                  (!layer.generation_id ||
                    job.payload?.generation_id === layer.generation_id);
                let jobId = layer.last_execution_job_id || layer.job_id;
                if (!jobId) {
                  const jobs = list(
                    await editor.app.api.get("/api/jobs"),
                    "jobs",
                  );
                  jobId = jobs.find(
                    (job) =>
                      belongs(job) &&
                      ["failed", "cancelled", "interrupted"].includes(
                        job.status,
                      ),
                  )?.id;
                }
                if (!jobId)
                  throw Error(
                    "No matching generation job is available for this project and layer.",
                  );
                const job = await editor.app.api.get(
                  `/api/jobs/${encodeURIComponent(jobId)}`,
                );
                if (!belongs(job))
                  throw Error(
                    "Resume blocked: this generation job does not belong to the current project, image and layer.",
                  );
                if (
                  !["failed", "cancelled", "interrupted"].includes(job.status)
                )
                  throw Error(
                    "Only an interrupted, cancelled or failed generation can resume.",
                  );
                editor.app.showJob(
                  await editor.app.api.post(
                    `/api/jobs/${encodeURIComponent(jobId)}/retry`,
                  ),
                );
              },
              "",
              { disabled: layer.resume_available === false },
            ),
        ),
      );
    }),
  );
}
