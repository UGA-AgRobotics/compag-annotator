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
  confirmAction,
  notify,
} from "./ui.js";
import { AnnotationCanvas } from "./canvas.js";
import { reviewCounts } from "./review-status.js";
import { maskDisplayControls } from "./mask-display.js";
export const activeSelection = (editor) =>
  editor.objects.filter(
    (o) =>
      editor.layer.selection.has(o.id) &&
      editor.layer.visible(o) &&
      !["rejected", "superseded"].includes(o.status),
  );
export async function selectionAction(
  editor,
  action,
  classId = editor.promptState.class_id,
  ids = activeSelection(editor).map((o) => o.id),
) {
  if (!ids.length) throw Error("Select one or more active objects first.");
  if (
    ["accept", "assign_accept"].includes(action) &&
    action === "assign_accept" &&
    !classId
  )
    throw Error("Choose a class before assigning and accepting.");
  if (
    ["reject", "delete"].includes(action) &&
    !(await confirmAction(
      `${action === "delete" ? "Delete" : "Reject"} selected objects?`,
      `${ids.length} objects will ${action === "delete" ? "leave the active annotation set" : "be marked rejected"}. Original images are preserved. Undo restores this change.`,
      action === "delete" ? "Delete selected" : "Reject selected",
    ))
  )
    return;
  await editor.mutate(() =>
    editor.app.api.post(editor.base + "/selection", {
      image_id: editor.image.id,
      annotation_ids: ids,
      action,
      ...(["assign", "assign_accept"].includes(action)
        ? { class_id: classId || null }
        : {}),
      confirm: ["reject", "delete"].includes(action),
      expected_revision: editor.revision,
    }),
  );
}
export function createClassDialog(editor, assign = false) {
  const ids = activeSelection(editor).map((o) => o.id),
    old = new Set(editor.project.classes.map((c) => c.id));
  if (assign && !ids.length)
    throw Error("Select objects before creating and assigning a class.");
  const d = dialog(
    assign ? "Create class and assign" : "Create new class",
    form(
      [
        field(
          "Class name",
          input("class-name", "", "text", { required: true, maxLength: 120 }),
        ),
        row(
          field("Class color", input("class-color", "#73ab83", "color")),
          field(
            "Class shortcut",
            input("class-shortcut", "", "text", { maxLength: 1 }),
          ),
        ),
        el(
          "p",
          {},
          assign
            ? `Assign the new class to ${ids.length} selected instances. They remain separate and need explicit acceptance.`
            : "The new class becomes available immediately on this image.",
        ),
      ],
      async (v) => {
        const saved = await editor.mutate(() =>
          editor.app.api.post(editor.base + "/classes", {
            action: "create",
            name: v["class-name"],
            color: v["class-color"],
            shortcut: v["class-shortcut"] || "",
            expected_revision: editor.revision,
          }),
        );
        if (!saved) return;
        const created = editor.project.classes.find((c) => !old.has(c.id));
        if (!created)
          throw Error(
            "Class was saved but could not be selected. Refresh the image before assigning.",
          );
        editor.promptState.class_id = created.id;
        editor.layer.classes = editor.project.classes;
        editor.renderInspector();
        d.close();
        if (assign) await selectionAction(editor, "assign", created.id, ids);
      },
      assign ? "Create class and assign" : "Create class",
    ),
  );
  return d;
}
export function selectionToolbar(editor) {
  const objects = activeSelection(editor),
    chosen = editor.objects.filter(
      (o) => editor.layer.selection.has(o.id) && editor.layer.visible(o) && o.status !== "superseded",
    ),
    n = chosen.length,
    allActive = n > 0 && objects.length === n;
  const classControl = select(
    "selection-class",
    [
      ["", "Unassigned"],
      ...editor.project.classes
        .filter((c) => !c.archived)
        .map((c) => [c.id, c.name]),
      ["__new", "Create new class…"],
    ],
    editor.promptState.class_id || "",
    {
      onchange: (e) => {
        if (e.target.value === "__new") {
          e.target.value = editor.promptState.class_id || "";
          createClassDialog(editor, n > 0);
        } else {
          editor.promptState.class_id = e.target.value || null;
          editor.invalidateAssist();
        }
      },
    },
  );
  const combined = check(
    "paint-accept",
    "Paint class and accept each clicked object",
    editor.paintAccept || false,
  );
  combined
    .querySelector("input")
    .addEventListener("change", (e) => (editor.paintAccept = e.target.checked));
  const hideAssigned = check(
    "hide-assigned-masks",
    "Hide assigned masks",
    editor.layer.hideAssigned,
  );
  hideAssigned.querySelector("input").addEventListener("change", (e) => {
    editor.layer.hideAssigned = e.target.checked;
    if (e.target.checked) editor.layer.reviewPendingOnly = false;
    editor.refreshVisibility();
  });
  const hideGeneratedSAM = check(
    "hide-generated-sam-masks",
    "Hide generated SAM masks",
    editor.layer.hideGeneratedSAM,
  );
  hideGeneratedSAM.querySelector("input").addEventListener("change", (e) => {
    editor.layer.hideGeneratedSAM = e.target.checked;
    if (e.target.checked) editor.layer.reviewPendingOnly = false;
    editor.refreshVisibility();
  });
  const counts = reviewCounts(editor.objects),
    hidden = reviewCounts(editor.objects.filter(o => !editor.layer.visible(o)));
  return el(
    "section",
    { class: "selection-panel" },
    row(el("h2", {}, "Selection"), badge(`${n} selected`)),
    maskDisplayControls(editor),
    hideGeneratedSAM,
    el("small", {}, "Hide masks from Generate masks, including labeled or reviewed ones. Manual objects and new SAM point/box masks stay visible unless another filter hides them."),
    hideAssigned,
    el("small", {}, "Hide labeled masks to select remaining objects. Uncheck to show them again; saved labels are unchanged."),
    el("p", { "aria-live": "polite" },
      `Pending review: ${counts.pending} · Accepted: ${counts.accepted} · Rejected: ${counts.rejected}`,
    ),
    el("small", {}, `Hidden: ${hidden.pending} still need review · ${hidden.accepted} accepted · ${hidden.rejected} rejected. Assigning a class does not accept a mask.`),
    editor.layer.reviewPendingOnly
      ? button("Show all active masks", () => {
          editor.layer.reviewPendingOnly = false;
          editor.refreshVisibility();
        })
      : counts.pending > 0 && button(`Review ${counts.pending} pending ${counts.pending === 1 ? "mask" : "masks"}`, () => {
          editor.layer.hideAssigned = false;
          editor.layer.hideGeneratedSAM = false;
          editor.layer.reviewPendingOnly = true;
          editor.layer.showProposals = editor.layer.showUnassigned = true;
          editor.layer.hiddenClasses.clear();
          editor.layer.hiddenObjects.clear();
          editor.layer.hiddenLayers.clear();
          editor.refreshVisibility();
        }),
    row(
      button("Clear selection", () => editor.layer.select(null)),
      button("Select visible objects", () => {
        editor.layer.selection = new Set(
          editor.objects
            .filter(
              (o) =>
                editor.layer.visible(o) &&
                !["rejected", "superseded"].includes(o.status),
            )
            .map((o) => o.id),
        );
        editor.layer.selected = [...editor.layer.selection][0] || null;
        editor.layer.callbacks.select?.(editor.layer.selected);
        editor.layer.render();
      }),
    ),
    field("Class for selection / paint", classControl),
    row(
      button("Assign class", () => selectionAction(editor, "assign"), "", {
        disabled: !allActive,
      }),
      button(
        "Create class and assign",
        () => createClassDialog(editor, true),
        "",
        { disabled: !allActive },
      ),
    ),
    row(
      button("Accept selected", () => selectionAction(editor, "accept"), "", {
        disabled: !allActive,
        "data-review-action": "accept",
      }),
      button(
        "Assign and accept",
        () => selectionAction(editor, "assign_accept"),
        "",
        { disabled: !allActive },
      ),
    ),
    editor.layer.tool === "paint" && combined,
    row(
      button("Merge selected masks", () => mergeDialog(editor), "primary", {
        disabled: !allActive || n < 2,
      }),
      button("Reject selected", () => selectionAction(editor, "reject"), "", {
        disabled: !allActive,
      }),
      button(
        "Delete selected",
        () =>
          selectionAction(
            editor,
            "delete",
            null,
            chosen.map((o) => o.id),
          ),
        "danger",
        { disabled: !n },
      ),
    ),
    el(
      "small",
      {},
      "Shift/Ctrl-click toggles selection. Alt-click cycles overlaps. The list offers checkboxes. Assigning a class alone does not accept geometry.",
    ),
    editor.layer.overlaps?.length > 1 &&
      field(
        "Objects under last pointer",
        select(
          "overlap-choice",
          editor.layer.overlaps.map((id) => {
            const i = editor.objects.findIndex((o) => o.id === id);
            return [
              id,
              `Object ${i + 1} · ${editor.project.classes.find((c) => c.id === editor.objects[i]?.class_id)?.name || "Unassigned"}`,
            ];
          }),
          editor.layer.selected || "",
          { onchange: (e) => editor.layer.select(e.target.value) },
        ),
      ),
  );
}
export async function mergeDialog(editor) {
  const parents = activeSelection(editor);
  if (parents.length < 2)
    throw Error("Select at least two masks or polygons from this image.");
  if (parents.some((o) => o.geometry.type === "box"))
    throw Error(
      "Boxes are not segmentation masks. Explicitly convert each box to a mask before merging.",
    );
  const request = {
    image_id: editor.image.id,
    annotation_ids: parents.map((o) => o.id),
    expected_revision: editor.revision,
  };
  const preview = await editor.app.api.post(
    editor.base + "/merge/preview",
    request,
  );
  const options = [
    ...(preview.class_conflict ? [["", "Choose the merged class…"]] : []),
    ["__unassigned", "Unassigned"],
    ...editor.project.classes
      .filter((c) => !c.archived)
      .map((c) => [c.id, c.name]),
  ];
  const target = select(
    "merged-class",
    options,
    preview.class_conflict ? "" : preview.proposed_class_id || "__unassigned",
  );
  const canvas = el("canvas", {
      class: "annotation-canvas merge-preview",
      tabIndex: 0,
      "aria-label": "Exact merged mask preview",
    }),
    layer = new AnnotationCanvas(canvas);
  layer.locked = true;
  layer.classes = editor.project.classes;
  const status = el(
    "p",
    { role: "status" },
    "Loading full-image union preview…",
  );
  let ready = false;
  const commit = async (accept) => {
    if (!ready) throw Error("Wait for the exact merge preview.");
    if (!target.value)
      throw Error("Choose a target class or explicitly choose Unassigned.");
    if (accept && target.value === "__unassigned")
      throw Error("Choose a class before accepting the merged object.");
    if (editor.revision !== request.expected_revision)
      throw Error(
        "The image changed. Close this dialog and preview the merge again.",
      );
    const saved = await editor.mutate(() =>
      editor.app.api.post(editor.base + "/merge", {
        ...request,
        parent_revisions: preview.parent_revisions,
        confirm: true,
        class_id: target.value === "__unassigned" ? null : target.value,
        accept,
      }),
    );
    if (saved) {
      d.close();
      editor.setTool("select");
    }
  };
  const draft = button("Merge as draft", () => commit(false), "primary", {
      disabled: true,
    }),
    accept = button("Merge and accept", () => commit(true), "", {
      disabled: true,
    });
  const d = dialog(
    "Preview exact mask union",
    el(
      "div",
      {},
      el(
        "p",
        {},
        `${parents.length} parents · exact union area ${preview.area.toLocaleString()} px². Holes and disconnected components are preserved.`,
      ),
      el(
        "p",
        {},
        "Merge combines only existing pixels. It does not recover missing pixels between fragments. Use brush correction or Refine with SAM separately after merging.",
      ),
      canvas,
      status,
      preview.class_conflict &&
        el(
          "p",
          { class: "inline-warning" },
          "Parents have different assigned classes. Choose the target class explicitly.",
        ),
      field("Merged object class", target),
      el(
        "small",
        {},
        "The new object records parent history. Parents are superseded together; Undo restores their identities and review state. A draft merge does not inherit verification.",
      ),
      row(draft, accept),
    ),
  );
  d.classList.add("wide-dialog");
  d.addEventListener("close", () => layer.destroy());
  await layer.setImage(
    editor.image,
    editor.image.media_url || editor.path + "/media",
  );
  if (!d.isConnected) return;
  layer.objects = [
    {
      id: "union-preview",
      geometry: preview.geometry,
      class_id: preview.proposed_class_id,
      status: "draft",
    },
  ];
  layer.render();
  ready = true;
  draft.disabled = accept.disabled = false;
  status.textContent =
    "Exact union ready. Inspect the result before confirming.";
  target.addEventListener("change", () => {
    layer.objects[0].class_id =
      target.value === "__unassigned" ? null : target.value;
    layer.render();
  });
}
export async function annotationBoundaryDialog(editor) {
  const models = (editor.models || []).filter((m) =>
      ["sam2", "sam3"].includes(m.provider),
    ),
    layers = editor.generationLayers || [],
    selected = activeSelection(editor);
  const eligible = selected.filter(
    (o) => o.status === "proposal" && o.source?.touches_internal_seam,
  );
  const scope = select("recovery-scope", [
    ["selection", `Selected proposals (${eligible.length} at internal seams)`],
    ...layers.map((l, i) => [l.id, `Proposal layer ${i + 1} · ${l.status}`]),
  ]);
  const opt = check(
    "recovery-opt-in",
    "Enable annotation tile-edge recovery for this chosen scope",
    false,
  );
  const d = dialog(
    "Annotation tile-edge recovery",
    form(
      [
        el(
          "p",
          {},
          "Optional and off by default. Only generated proposals touching internal tile seams are eligible. Whole-image runs and physical image edges are not applicable. Results are staged for explicit comparison.",
        ),
        field("Recovery scope", scope),
        field(
          "Recovery model",
          select("recovery-model", [
            ["", "Choose a local SAM model"],
            ...models.map((m) => [m.id, m.name || m.architecture || m.id]),
          ]),
        ),
        field(
          "Maximum recovery attempts",
          input("recovery-limit", "3", "number", { min: 1, max: 20, step: 1 }),
        ),
        opt,
      ],
      async (v) => {
        if (!v["recovery-opt-in"])
          throw Error("Explicitly enable recovery for the chosen scope.");
        if (!v["recovery-model"]) throw Error("Choose a recovery model.");
        if (scope.value === "selection" && !eligible.length)
          throw Error(
            "No selected proposals touch an internal crop seam. Recovery is not applicable.",
          );
        const job = await editor.app.api.post(
          editor.base + "/boundary/annotation",
          {
            image_id: editor.image.id,
            ...(scope.value === "selection"
              ? { annotation_ids: eligible.map((o) => o.id) }
              : { layer_id: scope.value }),
            model_id: v["recovery-model"],
            explicit_opt_in: true,
            limit: v["recovery-limit"],
            expected_revision: editor.revision,
          },
        );
        d.close();
        editor.app.showJob(job);
      },
      "Start scoped recovery",
    ),
  );
}
export async function reviewAnnotationBoundary(editor) {
  const results = await editor.app.api.get(editor.base + "/boundary/proposals"),
    proposals = (
      Array.isArray(results) ? results : results.proposals || []
    ).filter(
      (p) =>
        p.image_id === editor.image.id && (!p.status || p.status === "pending"),
    );
  if (!proposals.length) {
    notify("No pending tile-edge proposals for this image.");
    return;
  }
  const { boundaryDialog } = await import("./workflows.js");
  const d = dialog(
    "Review annotation tile-edge proposals",
    el(
      "div",
      {},
      proposals.map((proposal, i) =>
        row(
          el("span", {}, `Proposal ${i + 1}`),
          button("Compare & review", async () => {
            d.close();
            await boundaryDialog(editor.app, proposal, async () => {
              await editor.app.refreshProject();
              editor.project = editor.app.project;
              await editor.loadScene();
            });
          }),
        ),
      ),
    ),
  );
}
