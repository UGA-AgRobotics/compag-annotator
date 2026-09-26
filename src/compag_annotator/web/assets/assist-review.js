import { el, button, select, field, check, row, dialog, notify } from "./ui.js";

// Explicit markers support new assists; the fallback recognizes saved legacy
// point/box masks without treating automatic generation layers as assistance.
export function isSAMAssist(object) {
  const source = object.source || {};
  return object.geometry?.type === "mask" && ["sam2", "sam3"].includes(source.kind) &&
    (["sam_assist", "independent_point_assist"].includes(source.operation) ||
      (!source.operation && !object.layer_id));
}
export const assistDrafts = editor => editor.objects.filter(o => o.status === "draft" && isSAMAssist(o));

// Prompt placement is synchronous and never scans or loads saved masks.
export function addAssistPoint(editor, point, label) {
  if (!editor.alive || editor.busy || editor.failed) return;
  if (editor.promptState.mode === "independent" && label !== 1)
    throw Error("For negative points, choose Refine one object in SAM assistance.");
  editor.promptState.points.push(point);
  editor.promptState.labels.push(label);
  editor.invalidateAssist(false);
  editor.syncPrompts(false);
  editor.layer.requestRender();
  editor.renderInspector();
}

export function removeLastAssistPoint(editor) {
  if (!editor.alive || editor.busy || editor.failed || !editor.promptState.points.length) return;
  editor.promptState.points.pop();
  editor.promptState.labels.pop();
  editor.invalidateAssist(false);
  editor.syncPrompts(false);
  editor.layer.requestRender();
  editor.renderInspector();
}

export function assistBatchDialog(editor) {
  const drafts = assistDrafts(editor), revision = editor.revision;
  if (!drafts.length) throw Error("There are no saved SAM Assist drafts in this image.");
  const selected = drafts.filter(o => editor.layer.selection.has(o.id) && editor.layer.visible(o));
  const scope = select("assist-review-scope", [
    ["selected", `Selected SAM Assist drafts (${selected.length})`],
    ["image", `All SAM Assist drafts in this image (${drafts.length})`],
  ], selected.length ? "selected" : "image");
  const classes = editor.project.classes.filter(c => !c.archived);
  const target = select("assist-review-class", [["", "Choose a class"], ...classes.map(c => [c.id, c.name])], "");
  const confirmed = check("assist-review-confirm", "I have inspected these masks and want to accept them with the chosen class.");
  const summary = el("p", { role: "status", dataset: { testid: "assist-review-summary" } });
  const chosen = () => scope.value === "selected" ? selected : drafts;
  const apply = button("Assign class and accept drafts", async () => {
    if (!editor.alive || revision !== editor.revision) throw Error("The image changed. Close this dialog and review the current masks again.");
    if (!confirmed.querySelector("input").checked || !target.value) throw Error("Choose a class and confirm mask review.");
    const objects = chosen(), ids = objects.map(o => o.id);
    if (!ids.length) throw Error("Choose SAM Assist drafts to review.");
    const saved = await editor.mutate(() => editor.app.api.post(editor.base + "/selection", {
      image_id: editor.image.id, annotation_ids: ids,
      action: "assist_assign_accept", class_id: target.value, confirm: true,
      parent_revisions: Object.fromEntries(objects.map(o => [o.id, o.revision])), expected_revision: revision,
    }));
    if (saved) {
      d.close();
      notify(`${ids.length} SAM Assist drafts assigned and accepted. Undo restores the whole batch. Review the rest of the image before Mark reviewed.`);
    }
  }, "primary");
  const update = (reset = false) => {
    if (reset) confirmed.querySelector("input").checked = false;
    const objects = chosen(), hidden = objects.filter(o => !editor.layer.visible(o)).length;
    summary.textContent = `${objects.length} drafts will be assigned to one class and accepted as separate instances. ${hidden} are currently hidden. Existing class labels in this batch will be replaced. Other objects and images remain unchanged. Undo restores the entire batch.`;
    apply.disabled = !objects.length || !target.value || !confirmed.querySelector("input").checked;
  };
  scope.addEventListener("change", () => update(true));
  target.addEventListener("change", () => update(true));
  confirmed.addEventListener("change", () => update());
  const d = dialog("Review SAM Assist drafts together", el("div", {},
    field("Drafts to review", scope), summary,
    field("Assign every mask to class", target),
    !classes.length && el("p", {}, "Create a class in Labels or the annotation class picker, then return here."),
    el("p", {}, "This accepts masks only. It does not mark the whole image reviewed."),
    confirmed, row(apply, button("Cancel", () => d.close())),
  ));
  update();
  return d;
}
