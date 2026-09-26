import { DevicePicker } from "./devices.js";
import { TilingPanel } from "./tiling.js";
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
  notify,
  projectPath,
  confirmAction,
} from "./ui.js";
const stateLabel = (value) =>
  ({
    not_installed: "Not installed",
    weights_missing: "Runtime ready · weights required",
    ready_unverified: "Weights registered · not yet tested",
    trust_required: "Checkpoint trust required",
    registered_unverified: "Registered · not yet tested",
    awaiting_real_checkpoint_test: "Real checkpoint test pending",
    failed: "Failed",
    busy: "Busy",
    idle: "Idle",
  })[value] || value;
export const array = (result, key) =>
  Array.isArray(result) ? result : result?.[key] || [];
export function modelOptions(models, provider = "yolo") {
  return models
    .filter((m) => m.provider === provider)
    .map((m) => [m.id, m.name || m.architecture || m.id]);
}
export async function modelsPage(app) {
  const [catalogResult, modelsResult, status] = await Promise.all([
    app.api.get("/api/models/catalog"),
    app.api.get("/api/models"),
    app.api.get("/api/models/status"),
  ]);
  const catalog = array(catalogResult, "catalog"),
    models = array(modelsResult, "models");
  const providers = ["sam2", "sam3", "yolo"].map((provider) => {
    const entries = catalog.filter((c) => c.provider === provider),
      s = status[provider] || status.providers?.[provider] || {};
    const title = {
      sam2: "SAM2 · automatic masks, points & boxes",
      sam3: "SAM3 · local weights",
      yolo: "YOLO · instance segmentation",
    }[provider];
    return card(
      title,
      el(
        "p",
        {},
        provider === "sam3"
          ? "Install the optional runtime, then add your authorized local image checkpoint. The application never downloads SAM3 weights."
          : provider === "sam2"
            ? "Choose a SAM2.1 model for class-agnostic automatic masks and interactive correction. Small is a balanced starting preset."
            : "Predict object proposals and train segmentation models using your reviewed annotations.",
      ),
      provider !== "yolo" &&
        el(
          "p",
          { class: "capability-note" },
          entries[0]?.capability_details?.generate_proposals?.supported
            ? "Automatic masks supported · unassigned proposals require review. Point and box correction is available separately."
            : entries[0]?.capability_details?.generate_proposals?.reason ||
                "Automatic mask capability has not been reported by this provider.",
        ),
      row(
        badge(stateLabel(s.status || s.state || "See provider status")),
        button("Add existing checkpoint", () =>
          registerDialog(app, provider, entries),
        ),
      ),
      ...entries.map((entry) =>
        el(
          "div",
          { class: "data-row", style: "margin-top:12px" },
          el(
            "h3",
            {},
            entry.name || entry.label || entry.architecture || entry.id,
          ),
          el(
            "p",
            {},
            entry.description ||
              entry.status ||
              "Review runtime and checkpoint details before installing.",
          ),
          row(
            entry.source_url &&
              el(
                "a",
                {
                  href: entry.source_url,
                  target: "_blank",
                  rel: "noopener noreferrer",
                },
                "Official source",
              ),
            entry.license_url &&
              el(
                "a",
                {
                  href: entry.license_url,
                  target: "_blank",
                  rel: "noopener noreferrer",
                },
                entry.license || "License",
              ),
          ),
          details("Source, license & compatibility", entry),
          button(
            provider === "sam3" ? "Install runtime" : "Install / download",
            () => installDialog(app, entry, provider),
          ),
        ),
      ),
      !entries.length &&
        el(
          "p",
          {},
          "No installation entries returned by the provider catalog. See device diagnostics for details.",
        ),
    );
  });
  return el(
    "div",
    {},
    heading(
      "Models & AI",
      "Optional local assistance. Manual annotation works without a model.",
      app.project &&
        button("Roll back active model", async () => {
          if (
            await confirmAction(
              "Roll back model?",
              "Restore the previous active model and its saved class mapping?",
              "Roll back",
            )
          ) {
            await app.api.post(
              projectPath(app.project.id) + "/models/rollback",
              { confirm: true, expected_revision: app.project.revision },
            );
            await app.refreshProject();
            notify("Previous model activation restored.");
          }
        }),
      button("Unload providers", async () => {
        const result = await app.api.post("/api/models/unload");
        notify(result.message || "Provider unload completed.");
        await app.navigate("models");
      }),
    ),
    details("Device & provider status", status),
    el("div", { class: "stack" }, providers),
    card(
      "Registered models",
      models.length
        ? models.map((m) =>
            el(
              "article",
              { class: "data-row" },
              row(
                el("h3", {}, m.name || m.architecture || m.id),
                badge(m.provider),
                badge(m.status || m.validation_status || "Registered"),
              ),
              m.saved_time_label && el("p", {class:"model-saved-time"}, m.saved_time_label),
              el(
                "p",
                {},
                `Model identity: ${m.sha256 || m.model_sha256 || "See metadata"}`,
              ),
              details("Classes, provenance & validation", m),
              row(
                button("Load and test", () => testDialog(app, m)),
                m.provider === "yolo" &&
                  button("Activate for project", () => activateDialog(app, m)),
                button("Inspect metadata", () =>
                  dialog("Model metadata", details("Model record", m, true)),
                ),
              ),
            ),
          )
        : el(
            "p",
            {},
            "No checkpoints registered. Add existing weights or install a selected official checkpoint.",
          ),
    ),
  );
}
function installDialog(app, entry, provider) {
  const consent = check(
    "consent",
    "I reviewed the source and applicable license terms and approve this installation.",
  );
  const download = check(
    "download_weights",
    "Download this selected checkpoint",
    provider !== "sam3",
  );
  if (provider === "sam3") download.querySelector("input").disabled = true;
  const d = dialog(
    "Install optional AI component",
    el(
      "div",
      {},
      details("Source and installation details", entry, true),
      el(
        "p",
        {},
        provider === "sam3"
          ? "Only runtime code is installed. Obtain authorized SAM3 image weights yourself through the official access process."
          : "This installs a separate provider runtime. Download progress, validation failures and cancellation are available in Jobs.",
      ),
      consent,
      download,
      button(
        "Start installation",
        async () => {
          if (!consent.querySelector("input").checked)
            throw Error("Review and approve the installation first.");
          const job = await app.api.post("/api/models/install", {
            provider,
            architecture: entry.architecture || entry.id,
            consent: true,
            download_weights:
              provider !== "sam3" && download.querySelector("input").checked,
          });
          d.close();
          app.showJob(job);
        },
        "primary",
      ),
    ),
  );
}
function registerDialog(app, provider, entries) {
  const architectures = entries.map((e) => [
    e.architecture || e.id,
    e.name || e.label || e.architecture || e.id,
  ]);
  const d = dialog(
    provider === "sam3" ? "Add local SAM3 weights" : "Add existing checkpoint",
    form(
      [
        field(
          "Provider",
          select("provider", ["sam2", "sam3", "yolo"], provider),
        ),
        field(
          "Architecture",
          architectures.length
            ? select("architecture", [
                ["", "Auto / explicit model metadata"],
                ...architectures,
              ])
            : input("architecture", "", "text", {
                placeholder: "Optional architecture identifier",
              }),
        ),
        field(
          "Local checkpoint path",
          input("path", "", "text", { required: true }),
          "Enter the full file or supported model directory path on this computer.",
        ),
        check("approved", "I approve reading this local checkpoint"),
        check(
          "trust",
          "I trust the checkpoint source. Some model formats can execute code during loading.",
        ),
        el(
          "details",
          {},
          el("summary", {}, "Model class mapping (optional)"),
          field(
            "Model index → class name or ID (JSON)",
            el("textarea", {
              name: "class_mapping",
              placeholder: '{"0":"Object"}',
            }),
          ),
        ),
        provider === "sam3" &&
          el(
            "p",
            {},
            el(
              "a",
              {
                href: "https://github.com/facebookresearch/sam3",
                target: "_blank",
                rel: "noopener noreferrer",
              },
              "Official SAM3 access and setup instructions",
            ),
          ),
      ],
      async (v) => {
        if (!v.approved || !v.trust)
          throw Error(
            "Explicit path approval and checkpoint trust are required.",
          );
        const result = await app.api.post("/api/models/register", {
          ...v,
          architecture: v.architecture || undefined,
          class_mapping: v.class_mapping
            ? JSON.parse(v.class_mapping)
            : undefined,
        });
        d.close();
        notify(
          `Registered model ${result.name || result.architecture || result.id}. Run Load and test to verify inference.`,
        );
        await app.navigate("models");
      },
      "Register checkpoint",
    ),
  );
}
function testDialog(app, model) {
  if (!app.project?.images.length)
    throw Error("Open a project and import an image before testing a model.");
  const devices = new DevicePicker(app, model.provider);
  const d = dialog(
    "Load and test model",
    form(
      [
        field(
          "Test image",
          select(
            "image_id",
            app.project.images.map((i) => [i.id, i.name]),
          ),
        ),
        devices.root,
      ],
      async (v) => {
        const j = await app.api.post(`/api/models/${model.id}/test`, {
          ...v,
          device: devices.value(),
          project_id: app.project.id,
        });
        d.close();
        app.showJob(j);
      },
      "Run real inference test",
    ),
  );
}
export function activateDialog(app, model) {
  if (!app.project) throw Error("Open a project first.");
  const p = app.project,
    names = model.class_names || model.classes || model.names || {},
    existing = model.class_mapping || {};
  const pairs = Array.isArray(names)
    ? names.map((name, i) => [String(i), name])
    : Object.entries(names);
  if (!pairs.length)
    for (const [index, id] of Object.entries(existing))
      pairs.push([
        index,
        p.classes.find((c) => c.id === id)?.name || `Model class ${index}`,
      ]);
  const rows = el("div", { class: "stack" }),
    controls = [];
  function addMapping(index = "", name = "") {
    const idx = input("model-index", index, "number", {
      min: 0,
      step: 1,
      required: true,
      "aria-label": "Model class number",
    });
    const target = select(
      "project-class",
      [
        ["", "Leave unassigned"],
        ...p.classes.filter((c) => !c.archived).map((c) => [c.id, c.name]),
      ],
      existing[index] || "",
      { "aria-label": `Project class for ${name || "model output"}` },
    );
    controls.push({ idx, target });
    rows.append(
      el(
        "div",
        { class: "data-row" },
        name && el("strong", {}, name),
        row(field("Model class number", idx), field("Project class", target)),
      ),
    );
  }
  pairs.forEach(([index, name]) => addMapping(index, String(name)));
  const d = dialog(
    "Activate model for this project",
    form(
      [
        el("p", {"data-testid":"activation-model-name"}, "Selected model: ", el("strong", {}, model.name || model.architecture || model.id)),
        model.saved_time_label && el("p", {}, model.saved_time_label),
        el(
          "p",
          {},
          "Map each model output to a project class explicitly. Unassigned outputs stay unassigned. Roll back can restore a previous activation.",
        ),
        !pairs.length &&
          el(
            "p",
            {},
            "No model class names have been inspected yet. Load and test the model to inspect its classes, or add the known model class numbers below.",
          ),
        rows,
        button("Add output mapping", () => addMapping(String(controls.length))),
      ],
      async () => {
        const mapping = Object.fromEntries(
          controls
            .filter((r) => r.target.value)
            .map((r) => [r.idx.value, r.target.value]),
        );
        await app.api.post(projectPath(p.id) + "/models/activate", {
          model_id: model.id,
          mapping,
          confirm: true,
          expected_revision: app.project.revision,
        });
        d.close();
        await app.refreshProject();
        notify("Model activated for subsequent inference.");
        await app.navigate("models");
      },
      "Activate model",
    ),
  );
}
export async function predictDialog(app) {
  if (!app.project) throw Error("Open a project first.");
  const models = array(await app.api.get("/api/models"), "models"),
    p = app.project,
    ids = [...app.selectedImages];
  const options = modelOptions(models);
  if (!options.length)
    throw Error("Register a YOLO segmentation model in Models & AI first.");
  const tileDefault={mode:"tiled",preset:"512",width:512,height:512,overlap:{mode:"pixels",x:0,y:0}};
  const modelSelect=select("model_id",options,p.active_model_id || options[0][0]);
  const match=check("use_training_layout","Use the model's saved training layout",true);
  const matchInput=match.querySelector("input");
  const layoutNote=el("p",{"data-testid":"prediction-layout-note"});
  const size=input("imgsz",512,"number",{min:32,max:4096,step:32,required:true});
  const panel = new TilingPanel(
    app,
    p.images.find((i) => i.id === ids[0]) || p.images[0],
    tileDefault,
  );
  const devices = new DevicePicker(app, "yolo");
  const d = dialog(
    "Predict image proposals",
    form(
      [
        field(
          "Model",
          modelSelect,
        ),
        field(
          "Images",
          select("scope", [
            ["selected", `Selected images (${ids.length})`],
            ["pool", "All incomplete pool / train images"],
          ]),
        ),
        devices.root,
        match,
        layoutNote,
        el(
          "details",
          {},
          el("summary", {}, "Tiling & inference settings"),
          panel.root,
          field("Model input size",size),
          field(
            "Confidence",
            input("conf", "0.25", "number", { min: 0, max: 1, step: 0.05 }),
          ),
        ),
        check("boundary_opt_in","Boundary check · slower contextual SAM recovery (optional)",false),
        field("Boundary SAM model",select("boundary_model_id",[["","Choose SAM for optional boundary checking"],...models.filter(m=>["sam2","sam3"].includes(m.provider)).map(m=>[m.id,m.name||m.architecture])])),
        field("Maximum boundary objects this job",input("boundary_limit",3,"number",{min:1,max:20,step:1})),
        el("p",{},"Proposals are placed on the full image and duplicate boxes are suppressed per class (IoU 0.5). Boundary check is OFF unless selected. It examines up to the chosen number of seam objects, preserves originals, and stages replacements/merges for your review. Deferred objects are listed in the job result."),
      ],
      async (v) => {
        const image_ids =
          v.scope === "selected"
            ? ids
            : p.images
                .filter(
                  (i) => ["pool", "train"].includes(i.role) && !i.complete,
                )
                .map((i) => i.id);
        if (!image_ids.length)
          throw Error("Select images or choose an eligible pool.");
        if(v.boundary_opt_in && !v.boundary_model_id)throw Error("Choose an installed SAM model for Boundary check.");
        await panel.validate();
        const j = await app.api.post(projectPath(p.id) + "/predict", {
          image_ids,
          model_id: v.model_id,
          device: devices.value(),
          boundary_opt_in:v.boundary_opt_in,
          boundary_model_id:v.boundary_model_id,
          boundary_limit:v.boundary_limit,
          settings: {
            tiling: panel.value(),
            confidence: v.conf,
            imgsz:v.imgsz,
            use_training_layout:matchInput.checked && !matchInput.disabled,
            stitching:{method:"class_aware_bbox_nms",iou:0.5},
            ...(matches512(panel.value(),v.imgsz) ? {tile_padding:"constant_bottom_right_pixel"}:{}),
          },
          expected_revision: app.project.revision,
        });
        d.close();
        app.showJob(j);
      },
      "Create proposals",
    ),
  );
  function matches512(value,imgsz){return value.mode==="tiled" && value.width===512 && value.height===512 && value.overlap.x===0 && value.overlap.y===0 && Number(imgsz)===512;}
  function syncLayout(){
    const model=models.find(m=>m.id===modelSelect.value), recipe=model?.prediction_settings;
    const available=model?.training_layout?.mode==="tiles_512" && recipe;
    matchInput.disabled=!available;
    const bound=available && matchInput.checked;
    if(bound){
      panel.mode.value="tiled";panel.preset.value="512";panel.width.value=panel.height.value=512;
      panel.overlapMode.value="pixels";panel.x.value=panel.y.value=0;size.value=512;
    }
    panel.sync();
    for(const c of panel.root.querySelectorAll("input,select,button"))c.disabled=!!bound;
    if(!bound)panel.sync();
    size.readOnly=!!bound;
    layoutNote.textContent=bound ? "Matched to this model: 512 × 512 source tiles, stride 512, input size 512, edge coverage and full-image reconstruction."
      :available ? "Custom override: changing source scale can reduce accuracy. The job records these settings." : "This older/base model has no saved tile-training layout. 512 × 512 tiles are offered here; training a new model on this layout is a separate step.";
    panel.schedule();
  }
  modelSelect.addEventListener("change",syncLayout);matchInput.addEventListener("change",syncLayout);syncLayout();
  d.classList.add("wide-dialog");
  d.addEventListener("close", () => panel.destroy());
}
