import { el, button, check, row, dialog, projectPath, notify } from "./ui.js";

export function incompatibleTrainingObjects(report) {
  return [...new Map((report.errors || [])
    .filter(e => e.annotation_id && ["invalid_annotation_geometry", "box_only_annotation"].includes(e.code))
    .map(e => [e.annotation_id, e])).values()];
}

export function trainingExclusionActions(app, report, refresh) {
  const incompatible = incompatibleTrainingObjects(report), excluded = report.excluded_annotations || [];
  return el("div", { "data-testid": "training-exclusions" },
    incompatible.length > 0 && button(`Exclude incompatible objects (${incompatible.length})…`,
      () => exclusionDialog(app, report, incompatible, "exclude", refresh)),
    excluded.length > 0 && el("p", {},
      `${excluded.length} objects excluded from new training/validation labels. Original annotations are kept.`),
    excluded.length > 0 && button("Review training exclusions…",
      () => exclusionDialog(app, report, excluded, "restore", refresh)),
  );
}

export function exclusionDialog(app, report, objects, action, refresh) {
  const pid = app.project.id, restoring = action === "restore";
  const choices = objects.map(o => ({
    object: o,
    control: check("training-object-" + o.annotation_id,
      `${o.image_name || o.image_id} · ${o.class_name || "Object"} · ID ${o.annotation_id}`),
  }));
  const consent = check("training-exclusion-consent", restoring
    ? "Include the selected objects again and recheck their geometry before training."
    : "I understand omitted objects will be treated as background in YOLO training and validation, which may affect model quality and metrics.");
  let busy = false, failed = false;
  const status = el("p", { role: "alert", hidden: true });
  const selected = () => choices.filter(c => c.control.querySelector("input").checked).map(c => c.object);
  const update = () => {
    apply.disabled = busy || failed || !selected().length || !consent.querySelector("input").checked;
    for (const c of d.querySelectorAll("input,button")) if (c !== apply) c.disabled = busy;
  };
  const apply = el("button", { type: "button", class: "primary", onclick: async () => {
    if (apply.disabled) return;
    busy = true; update();
    try {
      if (app.project?.id !== pid) throw Error("The active project changed. Refresh readiness first.");
      const items = selected();
      await app.api.post(projectPath(pid) + "/training/exclusions", {
        action, confirm: true, expected_revision: report.project_revision,
        annotation_ids: items.map(o => o.annotation_id),
        annotation_revisions: Object.fromEntries(items.map(o => [o.annotation_id, o.annotation_revision])),
      });
      d.close();
      notify(restoring ? "Selected objects restored for training. Rechecking geometry." :
        "Selected objects excluded from future training/validation labels. Original annotations kept; rechecking readiness.");
      await refresh();
    } catch (error) {
      failed = true; status.hidden = false;
      status.textContent = error.message + " Close this dialog and refresh readiness before trying again.";
    } finally { busy = false; update(); }
  } }, restoring ? "Restore selected objects" : "Exclude selected objects");
  const d = dialog(restoring ? "Restore training objects" : "Exclude objects from training", el("div", {},
    el("p", {}, restoring ? "Restore individual exclusions. Unsupported geometry will block strict YOLO training again."
      : "The selected objects will be omitted from future YOLO training and validation labels. Their masks, classes, review status and annotation exports stay unchanged. Lossy conversion stays off unless you separately enable it."),
    !restoring && el("p", { class: "inline-warning" },
      "Omitted objects are seen as background by YOLO. To avoid partially labeled images, you can instead set the whole image's role to excluded in Images. Images with no remaining accepted objects are left out of new training snapshots."),
    row(button("Select all", () => { for (const c of choices) c.control.querySelector("input").checked = true; update(); }),
      button("Clear selection", () => { for (const c of choices) c.control.querySelector("input").checked = false; update(); })),
    el("div", { class: "training-object-list" }, choices.map(({object,control}) => el("div", {},
      control, el("small", {}, object.message || object.reason)))),
    consent, status, row(apply, button("Cancel", () => d.close())),
  ));
  d.addEventListener("change", update);
  d.addEventListener("cancel", e => { if (busy) e.preventDefault(); });
  update();
  return d;
}
