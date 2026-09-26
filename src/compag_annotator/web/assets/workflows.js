import { DevicePicker } from "./devices.js";
import { trainingDataMessage, trainingReadinessPanel } from "./training-readiness.js";
import { generateDialog } from "./generation.js";
import {
  el,
  button,
  input,
  select,
  field,
  check,
  card,
  row,
  heading,
  badge,
  details,
  form,
  dialog,
  confirmAction,
  notify,
  projectPath,
} from "./ui.js";
import { array, modelOptions } from "./models.js";
import { AnnotationCanvas } from "./canvas.js";
export async function roundsPage(app) {
  const p = app.project,
    base = projectPath(p.id);
  const eligible = p.images.filter(
    (i) =>
      i.role === "pool" &&
      !p.rounds.some(
        (r) => r.status !== "planned" && r.image_ids.includes(i.id),
      ),
  );
  const ordered = eligible.slice(),
    preview = el("div"),
    manual = el("div", { class: "data-list" });
  function renderOrder() {
    manual.replaceChildren(
      ...ordered.map((image, index) =>
        el(
          "div",
          { class: "data-row" },
          row(
            el("span", {}, `${index + 1}. ${image.name}`),
            button(
              "↑",
              () => {
                [ordered[index - 1], ordered[index]] = [
                  ordered[index],
                  ordered[index - 1],
                ];
                renderOrder();
              },
              "",
              { disabled: index === 0, "aria-label": `Move ${image.name} up` },
            ),
            button(
              "↓",
              () => {
                [ordered[index + 1], ordered[index]] = [
                  ordered[index],
                  ordered[index + 1],
                ];
                renderOrder();
              },
              "",
              {
                disabled: index === ordered.length - 1,
                "aria-label": `Move ${image.name} down`,
              },
            ),
          ),
        ),
      ),
    );
  }
  renderOrder();
  const order = select("order", [
    ["import", "Import order"],
    ["shuffle", "Seeded shuffle"],
    ["manual", "Manual order"],
  ]);
  const orderPanel = el(
    "div",
    { hidden: true },
    el("h3", {}, "Image order"),
    manual,
  );
  order.addEventListener("change", () => {
    orderPanel.hidden = order.value !== "manual";
    preview.replaceChildren();
  });
  const planner = form(
    [
      el(
        "p",
        {},
        "Rounds organize image review. Epochs control training passes over a frozen dataset. Rounds are optional.",
      ),
      row(
        field(
          "Number of rounds",
          input("count", "1", "number", {
            min: 1,
            max: Math.max(1, eligible.length),
            required: true,
          }),
        ),
        field("Order", order),
      ),
      field("Shuffle seed", input("seed", "42", "number", { step: 1 })),
      orderPanel,
    ],
    async (v) => {
      const body = {
        count: v.count,
        order: v.order,
        seed: v.seed,
        ...(v.order === "manual"
          ? { image_ids: ordered.map((i) => i.id) }
          : {}),
      };
      const result = await app.api.post(base + "/rounds/preview", body);
      preview.replaceChildren(
        el("h3", {}, "Preview image allocation"),
        ...(result.rounds || []).map((r) =>
          el(
            "div",
            { class: "data-row" },
            el(
              "strong",
              {},
              `Round ${r.number} · ${r.image_ids.length} images`,
            ),
            el(
              "ul",
              {},
              r.image_ids.map((id) =>
                el(
                  "li",
                  {},
                  p.images.find((i) => i.id === id)?.name ||
                    "Image unavailable",
                ),
              ),
            ),
          ),
        ),
        button(
          "Confirm round plan",
          async () => {
            await app.api.post(base + "/rounds", {
              ...body,
              confirm: true,
              expected_revision: app.project.revision,
            });
            await app.refreshProject();
            await app.navigate("rounds");
          },
          "primary",
        ),
      );
    },
    "Preview distribution",
  );
  const roundCards = p.rounds.map((r) =>
    el(
      "article",
      { class: "data-row" },
      row(
        el("h3", {}, `Round ${r.number}`),
        badge(r.status),
        badge(`${r.image_ids.length} images`),
      ),
      el(
        "p",
        {},
        `Model: ${p.models.find((m) => m.id === r.model_id)?.name || (r.model_id ? "Bound model" : "No model bound")}`,
      ),
      el(
        "ul",
        {},
        r.image_ids.map((id) => {
          const image = p.images.find((i) => i.id === id);
          return el(
            "li",
            {},
            button(
              image?.name || "Image unavailable",
              () => {
                app.selectedImages = new Set(r.image_ids);
                app.annotate(id);
              },
              "quiet",
            ),
            image?.complete && badge("Reviewed", "good"),
          );
        }),
      ),
      row(
        r.status === "planned" &&
          button(
            "Start round",
            async () => {
              await app.api.post(base + `/rounds/${r.id}`, {
                action: "start",
                expected_revision: app.project.revision,
              });
              await app.refreshProject();
              await app.navigate("rounds");
            },
            "primary",
          ),
        r.status === "in_progress" &&
          button(
            "Finish review",
            async () => {
              if (
                await confirmAction(
                  "Finish round review?",
                  "All images must have explicit complete review. Training is optional and is not started by this action.",
                  "Finish review",
                )
              ) {
                await app.api.post(base + `/rounds/${r.id}`, {
                  action: "finish",
                  expected_revision: app.project.revision,
                });
                await app.refreshProject();
                await app.navigate("rounds");
              }
            },
            "primary",
          ),
        button("Generate round masks", () => {
          app.selectedImages = new Set(r.image_ids);
          return generateDialog(app, r.image_ids[0], `round:${r.id}`);
        }),
        button("Predict this round", () => {
          app.selectedImages = new Set(r.image_ids);
          return app.predictDialog();
        }),
      ),
      details("Round provenance", r),
    ),
  );
  return el(
    "div",
    {},
    heading(
      "Review rounds",
      "Choose a schedule that fits your dataset. Every image stays at full resolution.",
    ),
    el(
      "div",
      { class: "columns" },
      card(
        "Plan rounds",
        eligible.length
          ? planner
          : el(
              "p",
              {},
              "No eligible review-pool images. Import images and assign pool roles first.",
            ),
        preview,
      ),
      card(
        "Current rounds",
        roundCards.length
          ? roundCards
          : el(
              "p",
              {},
              "You can annotate, export and train without planning rounds.",
            ),
      ),
    ),
  );
}
export async function trainingPage(app) {
  const p = app.project,
    base = projectPath(p.id), navigation = app.navSeq;
  const [modelResult, boundaryResult, jobsResult] =
    await Promise.all([
      app.api.get("/api/models"),
      app.api.get(base + "/boundary/proposals"),
      app.api.get("/api/jobs"),
    ]);
  const models = array(modelResult, "models"),
    proposals = array(boundaryResult, "proposals"),
    jobs = array(jobsResult, "jobs").filter((j) => j.kind === "train");
  const readinessPanel = el("div", { "aria-live": "polite", tabIndex: -1 });
  let alive = true, activeCheck = null;
  const current = () => alive && app.navSeq === navigation && app.route === "train" && app.project?.id === p.id;
  if (!current()) return el("div");
  const previousCleanup = app.cleanup;
  app.cleanup = () => {
    alive = false;
    activeCheck?.controller.abort();
    previousCleanup?.();
  };
  function showReadiness(report) {
    readinessPanel.replaceChildren(trainingReadinessPanel(app, report, async () => {
      await app.refreshProject();
      try { await checkReadiness(); }
      catch (error) { if (error.name !== "AbortError") throw error; }
    }));
  }
  async function checkReadiness(settings = null) {
    if (!current()) throw new DOMException("View changed", "AbortError");
    const round = settings?.round_id ?? setup.elements.round_id.value;
    const lossy = settings?.allow_lossy ?? setup.elements.allow_lossy.checked;
    const query = new URLSearchParams({ allow_lossy: String(lossy) });
    query.set("training_mode", settings?.training_mode ?? setup.elements.training_mode.value);
    if (round) query.set("round_id", round);
    const key = query.toString();
    if (activeCheck?.key === key) return activeCheck.promise;
    activeCheck?.controller.abort();
    const entry = { key, controller: new AbortController() };
    activeCheck = entry;
    const elapsed = el("small", { "aria-live": "off" });
    readinessPanel.replaceChildren(card("Data readiness",
      el("p", { "data-testid": "training-check-progress" }, "Checking reviewed images and mask geometry…"),
      el("progress", { "aria-label": "Checking training data" }),
      el("p", {}, "Large images or many masks can take longer. You can choose training settings while this check runs. Training will wait for validation to finish."),
      elapsed,
    ));
    const started = performance.now();
    const timer = setInterval(() => {
      elapsed.textContent = `Checking for ${Math.floor((performance.now() - started) / 1000)} seconds…`;
    }, 1000);
    entry.controller.signal.addEventListener("abort", () => clearInterval(timer), { once: true });
    entry.promise = (async () => {
      try {
        const report = await app.api.get(base + "/training/readiness?" + key, entry.controller.signal);
        if (!current() || activeCheck !== entry) throw new DOMException("View changed", "AbortError");
        showReadiness(report);
        return report;
      } catch (error) {
        if (current() && activeCheck === entry && !entry.controller.signal.aborted) {
          readinessPanel.replaceChildren(card("Data readiness",
            el("p", { role: "alert" }, "The data check could not finish. " + error.message),
            el("p", {}, "Your training settings are kept. Retry the check before starting training."),
            button("Retry data check", async () => {
              try { await checkReadiness(); }
              catch (error) { if (error.name !== "AbortError") throw error; }
            }),
          ));
        }
        throw error;
      } finally {
        clearInterval(timer);
        if (activeCheck === entry) activeCheck = null;
      }
    })();
    return entry.promise;
  }
  const devices = new DevicePicker(app, "yolo");
  const setup = form(
    [
      field("Training images", select("training_mode", [
        ["tiles_512", "512 × 512 tiles (recommended for small objects)"],
        ["whole", "Whole images"],
      ], "tiles_512")),
      el("p", {"data-testid":"training-layout-help"}, "Tile training crops reviewed originals into 512 × 512 images with stride 512, covers the edges, and clips masks into each crop. Every crop keeps its original image's dataset role. New models remember this layout for full-image prediction. Validation metrics are measured on tiles."),
      field(
        "Base / previous segmentation model",
        select(
          "model_id",
          [["", "Choose a segmentation model"], ...modelOptions(models)],
          p.active_model_id || "",
        ),
      ),
      row(
        field(
          "Epochs",
          input("epochs", "10", "number", { min: 1, required: true }),
        ),
        field(
          "Image size",
          input("imgsz", "512", "number", {
            min: 32,
            step: 32,
            required: true,
          }),
        ),
        field(
          "Batch size",
          input("batch", "4", "number", { min: 1, required: true }),
        ),
      ),
      devices.root,
      field(
        "Round binding (optional)",
        select("round_id", [
          ["", "Cumulative reviewed dataset"],
          ...p.rounds.map((r) => [r.id, `Round ${r.number}`]),
        ]),
      ),
      el(
        "details",
        {},
        el("summary", {}, "Advanced settings"),
        field("Reproducible seed", input("seed", "42", "number", { step: 1 })),
        check(
          "allow_lossy",
          "Allow approximate polygon conversion with a quantified loss report",
        ),
        check(
          "boundary_opt_in",
          "Check/repair tile-edge masks before training (slower)",
        ),
        field(
          "Boundary preparation SAM model",
          select("boundary_model_id", [
            ["", "Choose only for optional preparation"],
            ...models
              .filter((m) => ["sam2", "sam3"].includes(m.provider))
              .map((m) => [m.id, m.name || m.architecture || m.id]),
          ]),
        ),
        el(
          "p",
          {},
          "Off by default. Any proposed geometry changes require review before you explicitly retry training. This check never runs during annotation or export.",
        ),
        check(
          "auto_activate",
          "Use a successfully validated new model for subsequent inference",
        ),
      ),
      app.resumeJobId &&
        el(
          "p",
          { class: "inline-warning" },
          `Resume requested from job ${app.resumeJobId}. The backend must find a valid resumable checkpoint.`,
        ),
      el(
        "p",
        {},
        "Training uses a frozen cumulative snapshot of complete reviewed originals. Tile selection never splits one original across training and validation. Incomplete images and unreviewed proposals are excluded. Box-only annotations do not qualify as segmentation targets.",
      ),
      el("p", {}, "Small datasets are handled automatically. New training jobs adapt the effective batch to the available images and verify that learnable weights actually change. Check validation results before using the model."),
    ],
    async (v) => {
      // A resumed run uses its original frozen dataset, not today's project edits.
      if (!app.resumeJobId) {
        let checked;
        try { checked = await checkReadiness(v); }
        catch (error) { if (error.name === "AbortError") return; throw error; }
        if (!checked.ready) {
          readinessPanel.focus();
          throw Error(trainingDataMessage(checked));
        }
      }
      if (!v.model_id) throw Error("Select a segmentation model.");
      if (
        v.allow_lossy &&
        !(await confirmAction(
          "Allow approximate training geometry?",
          "Unsupported mask topology may be approximated. Inspect the conversion report and affected objects in the job result.",
          "Allow approximation",
        ))
      )
        return;
      const job = await app.api.post(base + "/train", {
        ...v,
        device: devices.value(),
        round_id: v.round_id || undefined,
        qa_smoke: !!app.api.session.qa_mode,
        resume_job_id: app.resumeJobId || undefined,
      });
      app.resumeJobId = null;
      app.showJob(job);
    },
    app.resumeJobId ? "Resume training" : "Train model",
  );
  const syncTrainingLayout = () => {
    const tiled = setup.elements.training_mode.value === "tiles_512";
    setup.elements.imgsz.readOnly = tiled;
    if (tiled) setup.elements.imgsz.value = "512";
  };
  setup.elements.training_mode.addEventListener("change", syncTrainingLayout);
  syncTrainingLayout();
  if(app.resumeJobId){
    const frozen=jobs.find(j=>j.id===app.resumeJobId)?.payload;
    setup.elements.training_mode.value=frozen?.training_mode || "whole";
    setup.elements.training_mode.disabled=true;
    if(frozen?.imgsz)setup.elements.imgsz.value=frozen.imgsz;
    setup.elements.imgsz.readOnly=true;
    setup.querySelector('[data-testid="training-layout-help"]').textContent="Resume keeps the original frozen images, labels, input size and layout. Choose a new training job to use 512 × 512 tiles.";
  }
  for (const name of ["round_id", "allow_lossy", "training_mode"]) {
    setup.elements[name].addEventListener("change", () =>
      checkReadiness().catch(() => {}));
  }
  // Mount the settings immediately; full-resolution geometry validation runs
  // independently and must still finish before any new training submission.
  queueMicrotask(() => {
    if (current()) {
      if(app.resumeJobId) readinessPanel.replaceChildren(card("Frozen dataset",el("p",{},"Resume verifies and reuses the original snapshot. Current project edits and new tile settings are excluded.")));
      else checkReadiness().catch(() => {});
    }
  });
  const boundary = proposals
    .filter(
      (v) =>
        v.status === "pending" && v.context !== "automatic_mask_annotation",
    )
    .map((proposal) =>
      el(
        "article",
        { class: "data-row" },
        el("h3", {}, proposal.title || "Proposed mask preparation change"),
        details("Proposal and affected geometry", proposal),
        button("Compare & review", () => boundaryDialog(app, proposal)),
      ),
    );
  return el(
    "div",
    {},
    heading(
      "Train",
      "Turn cumulative reviewed annotations into a new local segmentation model.",
    ),
    el(
      "div",
      { class: "columns" },
      readinessPanel,
      card("Training setup", setup),
    ),
    boundary.length &&
      card(
        "Training preparation · review required",
        el(
          "p",
          {},
          "Review each proposed change. Accepted geometry is protected until you choose a decision. Start training again explicitly after resolving these cases.",
        ),
        boundary,
      ),
    card(
      "Training history",
      jobs.length
        ? jobs.map((j) =>
            el(
              "div",
              { class: "data-row" },
              row(badge(j.status), el("small", {}, j.id)),
              details("Settings, metrics & results", j),
              button("Open job logs", () => app.navigate("jobs")),
              j.status === "complete" &&
                button("Review model for activation", () =>
                  app.navigate("models"),
                ),
            ),
          )
        : el("p", {}, "Training has not run."),
    ),
  );
}
export async function boundaryDialog(app, proposal, onReviewed = null) {
  const base = projectPath(app.project.id),
    image = app.project.images.find(
      (i) => i.id === (proposal.image_id || proposal.image?.id),
    );
  const original = proposal.annotation_id
    ? await app.api.get(
        base + `/annotations/${proposal.annotation_id}/geometry`,
      )
    : null;
  const canvases = [];
  const compare = el("div", { class: "columns" });
  if (image) {
    for (const [label, geometry] of [
      [
        proposal.original_geometry ? "Original at proposal time" : "Current",
        proposal.original_geometry ||
          original ||
          proposal.before?.geometry ||
          proposal.before,
      ],
      [
        "Proposed",
        proposal.proposed_geometry ||
          proposal.after?.geometry ||
          proposal.geometry,
      ],
    ])
      if (geometry?.type) {
        const canvas = el("canvas", {
          class: "annotation-canvas",
          tabIndex: 0,
          "aria-label": `${label} boundary geometry`,
          style: "height:260px;background:#202b28",
        });
        compare.append(el("div", {}, el("h3", {}, label), canvas));
        const layer = new AnnotationCanvas(canvas);
        layer.locked = true;
        layer.classes = app.project.classes;
        canvases.push({ layer, geometry });
      }
  }
  const comparisonStatus = el(
    "p",
    { role: "status" },
    "Loading exact geometry comparison…",
  );
  const decisions = [];
  let comparisonReady = false;
  const d = dialog(
    onReviewed
      ? "Review annotation recovery geometry"
      : "Review training preparation geometry",
    el(
      "div",
      {},
      el(
        "p",
        {},
        proposal.reason ||
          "Compare the current object with its proposed replacement.",
      ),
      compare,
      comparisonStatus,
      details("Affected objects and provenance", proposal),
      row(
        ...[
          ["accept", "Accept proposed geometry"],
          ["retain", "Retain original"],
          ["exclude", "Exclude from training"],
        ].map(([decision, label]) => {
          const control = button(
            label,
            async () => {
              if (!comparisonReady)
                throw Error(
                  "Wait for the exact image and geometry comparison before reviewing.",
                );
              await app.api.post(base + "/boundary/review", {
                proposal_id: proposal.id || proposal.proposal_id,
                decision,
                attest: true,
                expected_revision: app.project.revision,
              });
              d.close();
              await app.refreshProject();
              if (onReviewed) await onReviewed();
              else await app.navigate("train");
            },
            decision === "accept" ? "primary" : "",
            { disabled: true },
          );
          decisions.push({ decision, control });
          return control;
        }),
      ),
    ),
  );
  d.style.width = "min(900px,calc(100vw - 40px))";
  d.addEventListener("close", () =>
    canvases.forEach(({ layer }) => layer.destroy()),
  );
  for (const { layer, geometry } of canvases) {
    await layer.setImage(
      image,
      image.media_url || base + `/images/${image.id}/media`,
    );
    layer.objects = [
      {
        id: "boundary-preview",
        geometry,
        class_id: proposal.class_id,
        status: "proposal",
      },
    ];
    layer.render();
  }
  if (!image || !canvases.length) {
    comparisonStatus.textContent =
      "The image or original geometry is unavailable. Close this dialog and relink the source image before review.";
    return;
  }
  comparisonReady = true;
  comparisonStatus.textContent =
    "Geometry comparison ready. Inspect both masks before choosing a decision.";
  for (const { decision, control } of decisions)
    control.disabled = decision === "accept" && !proposal.geometry;
}
export { exportPage } from "./export.js";
