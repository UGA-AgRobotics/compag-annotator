import { el, button, select, field } from "./ui.js";

// Every option comes from the selected provider's live, isolated Torch runtime.
export class DevicePicker {
  constructor(app, provider) {
    this.app = app;
    this.sequence = 0;
    this.preferred = app.device || "cpu";
    this.control = select("device", [[this.preferred, "Detecting devices…"]], this.preferred, { disabled: true });
    this.note = el("small", { role: "status" });
    this.root = el("div", { class: "device-picker" },
      field("Device", this.control),
      button("Refresh devices", () => this.load(this.provider, true)), this.note);
    this.control.addEventListener("change", () => {
      this.preferred = this.control.value;
      app.device = this.preferred;
      try { localStorage.setItem("compag-device", this.preferred); } catch (_) { /* Optional browser preference. */ }
    });
    this.load(provider);
  }
  async load(provider, refresh = false) {
    this.provider = provider;
    const sequence = ++this.sequence;
    this.loading = true;
    this.control.disabled = true;
    this.note.textContent = "Detecting CPU and GPUs in this model runtime…";
    try {
      const report = await this.app.api.get(`/api/models/devices?provider=${encodeURIComponent(provider)}&refresh=${refresh}`);
      if (sequence !== this.sequence) return;
      this.available = new Set(report.devices.filter(d => provider !== "sam3" || d.id !== "cpu").map(d => d.id));
      this.control.replaceChildren(...report.devices.map(d => el("option", { value: d.id, disabled: provider === "sam3" && d.id === "cpu" },
        d.kind === "gpu" ? `GPU ${d.id.split(":")[1]} · ${d.name} (${d.id})` : provider === "sam3" ? "CPU · unsupported by SAM3 runtime" : "CPU")));
      if (!report.devices.some(d => d.id === this.preferred)) {
        this.control.prepend(el("option", { value: this.preferred, disabled: true }, `${this.preferred} · unavailable`));
      }
      this.control.value = this.preferred;
      this.note.textContent = report.message || "Choose CPU or GPU for this job.";
      if (!this.available.has(this.preferred)) this.note.textContent += " Select an available device; no automatic CPU fallback.";
      this.control.disabled = false;
    } catch (error) {
      if (sequence !== this.sequence) return;
      this.available = new Set();
      this.note.textContent = `Device detection failed: ${error.message}. Refresh to retry.`;
    } finally {
      if (sequence === this.sequence) this.loading = false;
    }
  }
  value() {
    if (this.loading) throw Error("Wait for device detection to finish.");
    if (!this.available?.has(this.control.value)) throw Error("Select an available CPU or GPU, or refresh devices.");
    return this.control.value;
  }
}
