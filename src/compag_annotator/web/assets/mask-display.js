import { el, input, field, button, row } from "./ui.js";

const hex = value => typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value);
export function loadMaskDisplay(editor) {
  // Keep rc17 per-mask colors, but ignore its removed overlap-highlight option.
  editor.displayKey = `compag-mask-display:${editor.project.id}:${editor.image.id}`;
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(editor.displayKey) || "{}"); } catch { /* use defaults */ }
  editor.layer.maskColors = new Map(Object.entries(saved?.colors || {}).filter(([, c]) => hex(c)));
  editor.layer.defaultMaskColor = hex(saved?.defaultColor) ? saved.defaultColor : null;
  window.addEventListener("pagehide", () => flushMaskDisplay(editor), { signal: editor.events.signal });
}
export function flushMaskDisplay(editor) {
  clearTimeout(editor.displaySaveTimer);
  if (!editor.displayDirty) return;
  editor.displayDirty = false;
  try {
    localStorage.setItem(editor.displayKey, JSON.stringify({
      colors: Object.fromEntries(editor.layer.maskColors), defaultColor: editor.layer.defaultMaskColor,
    }));
  } catch { /* The current display still works without browser storage. */ }
}
function update(editor, commit = false) {
  editor.displayDirty = true;
  clearTimeout(editor.displaySaveTimer);
  if (commit) flushMaskDisplay(editor);
  else editor.displaySaveTimer = setTimeout(() => flushMaskDisplay(editor), 150);
  editor.layer.requestRender();
}
export function maskDisplayControls(editor) {
  const layer = editor.layer;
  const selected = editor.objects.filter(o => layer.selection.has(o.id) && layer.visible(o) &&
    ["mask", "polygon"].includes(o.geometry.type) && !["rejected", "superseded"].includes(o.status));
  const all = input("all-mask-color", layer.defaultMaskColor || "#39b9e6", "color", {
    oninput: e => {
      layer.defaultMaskColor = e.target.value; layer.maskColors.clear(); update(editor);
    },
    onchange: () => { flushMaskDisplay(editor); editor.renderInspector(); },
  });
  return el("div", { class: "mask-colors", dataset: { testid: "mask-colors" } },
    field("Color for all masks", all),
    el("small", {}, "Choose a color to see saved masks clearly, including masks made earlier. Display only; classes and exports stay unchanged."),
    button("Reset all mask colors", () => {
      layer.defaultMaskColor = null; layer.maskColors.clear(); update(editor, true); editor.renderInspector();
    }, "quiet"),
    selected.length > 0 && el("div", {},
      field(`Display color for ${selected.length} selected masks`, input("selected-mask-color", layer.color(selected[0]), "color", {
        oninput: e => { selected.forEach(o => layer.maskColors.set(o.id, e.target.value)); update(editor); },
        onchange: () => { flushMaskDisplay(editor); editor.renderInspector(); },
      })),
      row(button("Reset selected mask colors", () => {
        selected.forEach(o => layer.maskColors.delete(o.id)); update(editor, true); editor.renderInspector();
      }, "quiet")),
    ),
  );
}
