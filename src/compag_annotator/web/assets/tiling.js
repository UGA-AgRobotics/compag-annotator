import {
  el,
  input,
  select,
  field,
  row,
  button,
  notify,
  projectPath,
  dialog,
} from "./ui.js";
export const defaultTiling = () => ({
  mode: "tiled",
  preset: "512",
  width: 512,
  height: 512,
  overlap: { mode: "percent", x: 25, y: 25 },
});

// Every preview comes from the same server tiler used by generation/prediction.
export class TilingPanel {
  constructor(app, image, initial, onSaved = () => {}) {
    this.app = app;
    this.image = image;
    this.onSaved = onSaved;
    this.alive = true;
    const value = initial || app.project.processing_defaults || defaultTiling();
    this.mode = select(
      "processing-mode",
      [
        ["whole", "Whole image"],
        ["tiled", "Overlapping tiles"],
      ],
      value.mode,
    );
    this.preset = select(
      "tile-preset",
      [
        ["256", "256 × 256"],
        ["512", "512 × 512"],
        ["1024", "1024 × 1024"],
        ["custom", "Custom rectangle"],
      ],
      String(value.preset),
    );
    this.width = input("tile-width", value.width, "number", {
      min: 1,
      step: 1,
      required: true,
    });
    this.height = input("tile-height", value.height, "number", {
      min: 1,
      step: 1,
      required: true,
    });
    this.overlapMode = select(
      "overlap-mode",
      [
        ["percent", "Percent per axis"],
        ["pixels", "Pixels per axis"],
      ],
      value.overlap.mode,
    );
    this.x = input("overlap-x", value.overlap.x, "number", {
      min: 0,
      step: "any",
      required: true,
    });
    this.y = input("overlap-y", value.overlap.y, "number", {
      min: 0,
      step: "any",
      required: true,
    });
    this.summary = el(
      "p",
      { class: "tiling-summary", role: "status" },
      "Preparing tile preview…",
    );
    this.error = el("p", {
      class: "inline-warning",
      role: "alert",
      hidden: true,
    });
    this.canvas = el("canvas", {
      class: "tile-preview",
      width: 680,
      height: 290,
      "aria-label": "Full image tile grid with shaded overlap",
    });
    this.tiles = el(
      "div",
      {},
      row(
        field("Tile size", this.preset),
        field("Width (px)", this.width),
        field("Height (px)", this.height),
      ),
      row(
        field("Overlap units", this.overlapMode),
        field("Horizontal overlap", this.x),
        field("Vertical overlap", this.y),
      ),
    );
    this.root = el(
      "section",
      { class: "tiling-panel" },
      el("h3", {}, "Image processing"),
      field("Processing area", this.mode),
      this.tiles,
      this.error,
      this.summary,
      this.canvas,
      el(
        "small",
        {},
        "Shaded intersections show overlap. Tile dimensions refer to original image pixels; edge crops can be smaller. Preview reflects the server plan.",
      ),
      row(
        button("Refresh tile preview", () => this.validate()),
        button("Save as project default", () => this.save()),
      ),
      el(
        "small",
        {},
        initial || app.project.processing_defaults
          ? "Changes here affect this new job. Save explicitly to use them as the project default for future jobs."
          : "This older project has no saved processing default. The displayed 512 px / 25% choice applies only when you start a new job or save it explicitly.",
      ),
    );
    for (const control of [
      this.mode,
      this.preset,
      this.width,
      this.height,
      this.overlapMode,
      this.x,
      this.y,
    ])
      control.addEventListener("input", () => {
        this.plan = null;
        this.sync();
        this.schedule();
      });
    this.sync();
    this.schedule();
  }
  sync() {
    this.tiles.hidden = this.mode.value === "whole";
    const custom = this.preset.value === "custom";
    this.width.disabled = this.height.disabled =
      !custom || this.mode.value === "whole";
    if (!custom) this.width.value = this.height.value = this.preset.value;
    this.x.max =
      this.overlapMode.value === "percent"
        ? "99.999"
        : String(Number(this.width.value) - 1);
    this.y.max =
      this.overlapMode.value === "percent"
        ? "99.999"
        : String(Number(this.height.value) - 1);
    this.x.step = this.y.step =
      this.overlapMode.value === "percent" ? "any" : "1";
  }
  value() {
    const value = {
      mode: this.mode.value,
      preset: this.preset.value,
      width: Number(this.width.value),
      height: Number(this.height.value),
      overlap: {
        mode: this.overlapMode.value,
        x: Number(this.x.value),
        y: Number(this.y.value),
      },
    };
    if (![value.width, value.height].every((v) => Number.isInteger(v) && v > 0))
      throw Error("Enter positive whole-pixel tile dimensions.");
    if (value.mode === "tiled") {
      const limit =
        value.overlap.mode === "percent"
          ? [100, 100]
          : [value.width, value.height];
      if (
        ![value.overlap.x, value.overlap.y].every(
          (v, i) =>
            Number.isFinite(v) &&
            v >= 0 &&
            v < limit[i] &&
            (value.overlap.mode === "percent" || Number.isInteger(v)),
        )
      )
        throw Error(
          "Overlap must be nonnegative and smaller than the tile on each axis (less than 100% for percentages). Pixel overlap must be a whole number.",
        );
    }
    return value;
  }
  setImage(image) {
    this.image = image;
    this.plan = null;
    this.schedule();
  }
  schedule() {
    clearTimeout(this.timer);
    this.summary.textContent = "Tile settings changed. Updating preview…";
    this.timer = setTimeout(() => this.validate().catch(() => {}), 180);
  }
  async validate() {
    clearTimeout(this.timer);
    this.controller?.abort();
    const seq = (this.seq = (this.seq || 0) + 1);
    this.controller = new AbortController();
    try {
      this.error.hidden = true;
      const config = this.value(),
        image = this.image;
      if (!image) throw Error("Import an image before previewing a tile grid.");
      const signature = JSON.stringify([image.id, config]);
      const plan = await this.app.api.post(
        projectPath(this.app.project.id) + "/processing/preview",
        { image_id: image.id, tiling: config },
        this.controller.signal,
      );
      if (
        !this.alive ||
        seq !== this.seq ||
        signature !== JSON.stringify([this.image?.id, this.value()])
      )
        throw new DOMException("Preview superseded", "AbortError");
      this.plan = plan;
      const c = plan.config;
      this.summary.textContent = `${plan.image.width} × ${plan.image.height} px · ${plan.tile_count} ${plan.tile_count === 1 ? "tile" : "tiles"} · overlap ${c.overlap_x} × ${c.overlap_y} px · stride ${c.stride_x} × ${c.stride_y} px`;
      if (c.mode === "whole")
        this.summary.textContent = `${plan.image.width} × ${plan.image.height} px · 1 tile · Whole image, no internal seams (overlap not applied)`;
      this.draw(plan);
      return plan;
    } catch (error) {
      if (error.name !== "AbortError" && this.alive && seq === this.seq) {
        this.plan = null;
        this.summary.textContent =
          "Preview unavailable. Correct the settings before starting.";
        this.error.textContent = error.message;
        this.error.hidden = false;
        this.canvas
          .getContext("2d")
          .clearRect(0, 0, this.canvas.width, this.canvas.height);
      }
      throw error;
    }
  }
  draw(plan) {
    const ctx = this.canvas.getContext("2d"),
      { width, height } = plan.image,
      scale = Math.min(
        (this.canvas.width - 24) / width,
        (this.canvas.height - 24) / height,
      ),
      x = (this.canvas.width - width * scale) / 2,
      y = (this.canvas.height - height * scale) / 2;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.fillStyle = "#162b29";
    ctx.fillRect(x, y, width * scale, height * scale);
    for (const tile of plan.tiles) {
      const b = tile.box;
      ctx.fillStyle = "rgba(153,205,170,.17)";
      ctx.strokeStyle = "rgba(208,240,220,.55)";
      ctx.fillRect(
        x + b[0] * scale,
        y + b[1] * scale,
        (b[2] - b[0]) * scale,
        (b[3] - b[1]) * scale,
      );
      ctx.strokeRect(
        x + b[0] * scale,
        y + b[1] * scale,
        (b[2] - b[0]) * scale,
        (b[3] - b[1]) * scale,
      );
    }
  }
  async save() {
    await this.validate();
    this.app.project = await this.app.api.post(
      projectPath(this.app.project.id) + "/processing",
      { tiling: this.value(), expected_revision: this.app.project.revision },
    );
    await this.onSaved();
    notify(
      "Processing default saved for future jobs. Existing jobs keep their settings.",
    );
  }
  destroy() {
    this.alive = false;
    clearTimeout(this.timer);
    this.controller?.abort();
  }
}
export function processingDialog(app) {
  if (!app.project?.images.length)
    throw Error("Open a project with an image to preview processing defaults.");
  const image =
    app.project.images.find((i) => i.id === app.currentImageId) ||
    app.project.images[0];
  const panel = new TilingPanel(app, image),
    d = dialog("Project processing defaults", panel.root);
  d.classList.add("wide-dialog");
  d.addEventListener("close", () => panel.destroy());
  return d;
}
