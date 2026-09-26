import { el, input, select, field, check, form, dialog } from "./ui.js";
import { DevicePicker } from "./devices.js";
export function samSettingsDialog(editor) {
  const model = editor.models.find(m => m.id === editor.promptState.model_id);
  if (!model) throw Error("Choose a SAM model first.");
  const saved = editor.promptState.settings || {};
  const devices = new DevicePicker(editor.app, model.provider);
  const content = form([
    el("p", {}, `${model.provider.toUpperCase()} point/box settings. These shared controls affect the actual decoder. Grid density, stability and NMS belong to SAM2 automatic generation.`),
    devices.root,
    field("Mask logit threshold", input("mask_threshold", saved.mask_threshold ?? 0, "number", { min: -32, max: 32, step: .1 }), "Higher values keep fewer pixels; 0 is the common SAM2/SAM3 default."),
    check("multimask_output", "Offer alternative masks", saved.multimask_output ?? true),
    field("Precision", select("precision", [["float32", "Float32"], ["bfloat16", "BFloat16 (GPU)"], ["float16", "Float16 (GPU)"]], saved.precision || "float32")),
    el("p", {}, "CPU requires Float32. SAM3 currently requires a CUDA GPU and your local weights. Settings changes invalidate an older preview."),
  ], async values => {
    const device = devices.value();
    if (device === "cpu" && values.precision !== "float32") throw Error("Choose Float32 for CPU.");
    editor.app.device = device;
    editor.promptState.settings = { mask_threshold: values.mask_threshold, multimask_output: values.multimask_output, precision: values.precision };
    editor.invalidateAssist(); editor.renderInspector(); modal.close();
  }, "Apply SAM settings");
  const modal = dialog("SAM settings", content);
  return modal;
}
