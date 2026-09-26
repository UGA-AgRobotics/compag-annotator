import { el, button, row } from "./ui.js";
import { needsReview, reviewCounts } from "./review-status.js";

const steps = ["Create masks", "Label & review", "Confirm image", "Export / Train"];

function guide(title, description, actions = [], step = null) {
  return el("section", { class: "workflow-guide", "aria-label": "Next step" },
    step !== null && el("ol", { class: "workflow-steps", "aria-label": "Annotation workflow" },
      steps.map((label, index) => el("li", {
        ...(index === step ? { "aria-current": "step" } : {}),
      }, `${index + 1}. ${label}`))),
    el("div", { class: "workflow-content" },
      el("div", {}, el("strong", { "aria-live": "polite" }, title), el("p", {}, description)),
      actions.length > 0 && row(...actions)),
  );
}

export function focusControl(root, selector) {
  const target = root.querySelector(selector);
  target?.scrollIntoView({ block: "nearest" });
  target?.focus({ preventScroll: true });
}

export function nextReviewImage(project, currentId) {
  const images = project.images || [], index = images.findIndex(i => i.id === currentId);
  return [...images.slice(index + 1), ...images.slice(0, index + 1)]
    .find(i => i.id !== currentId && !i.complete && !i.missing && i.role !== "excluded");
}

export function editorGuide(editor, actions) {
  const counts = reviewCounts(editor.objects),
    unassigned = editor.objects.filter(o => needsReview(o) && !o.class_id).length,
    next = nextReviewImage(editor.project, editor.image.id),
    navigate = (label, route) => button(label, () => editor.app.navigate(route));
  let card;
  if (editor.failed) {
    card = guide("Next: resolve the unsaved edit", "Your last change was not saved. Use Retry save, or reload the saved state before continuing.", [
      button("Show save recovery", () => focusControl(editor.inspector, "[data-save-recovery] button")),
    ]);
  } else if (editor.busy) {
    card = guide("Saving your changes…", "Wait for the save to finish. The next step will update automatically.");
  } else if (!editor.sceneLoaded) {
    card = guide("Loading saved annotations…", "The next step will appear after the image's annotations load.");
  } else if (editor.assistController) {
    card = guide("Creating SAM previews…", "When previews are ready, inspect them and add the masks you want as drafts.", [], 0);
  } else if (editor.candidates?.length) {
    card = guide("Next: save the SAM preview as a draft", editor.batchPreview
      ? `${editor.candidates.length} separate mask previews are ready. Inspect them, then save them as drafts. Check their class labels and explicitly accept or reject each mask afterward.`
      : "Inspect the preview or choose an alternative in SAM assistance, then save it as a draft. This does not accept the mask.", [
      button(editor.batchPreview ? "Save SAM previews as drafts" : "Save SAM preview as draft", () => editor.acceptPreview(), "primary"),
      button("Inspect SAM previews", actions.sam),
    ], 0);
  } else if (counts.pending) {
    card = guide(unassigned ? "Next: label and review the masks" : "Next: accept or reject the labeled masks",
      `${counts.pending} saved ${counts.pending === 1 ? "mask still needs" : "masks still need"} review${unassigned ? `; ${unassigned} have no class. Assign a class to masks you want to keep, then inspect and Accept selected. Reject selected removes unwanted masks from the active set.` : ". Their class labels are saved. Inspect each mask: Accept selected keeps a correct mask; Reject selected excludes an unwanted mask."} Labeling alone does not accept a mask. Use Review next pending mask to select one, including hidden objects.`, [
      button("Review next pending mask", actions.nextPending, "primary"),
      button("Open labeling & review", actions.review),
    ], 1);
  } else if (editor.image.complete) {
    card = guide("Image review saved — choose what to do next", next
      ? `Continue with ${next.name}, or export the reviewed images now. Training is optional and requires separate reviewed validation images.`
      : "You can export your reviewed annotations now. Training is optional and requires separate reviewed validation images. Use Images to check the full project.", [
      next && button("Review next image", () => editor.app.annotate(next.id), "primary"),
      navigate("Export annotations", "export"), navigate("Prepare training (optional)", "train"),
    ].filter(Boolean), 3);
  } else if (counts.accepted || counts.rejected) {
    card = guide("Next: confirm the full image review",
      `${counts.accepted} accepted · ${counts.rejected} rejected · no pending masks. Check the whole image for missing objects. When finished, mark it reviewed and confirm the review dialog.`, [
      button("Confirm image review…", () => editor.complete(), "primary"),
    ], 2);
  } else {
    card = guide("Next: create masks or draw objects", "Use Generate masks for automatic proposals, SAM assistance for point prompts, or the drawing tools. Then assign classes and review the objects. For a genuinely empty image, check the full image before using Mark reviewed.", [
      button("Generate initial masks", actions.generate, "primary"),
      button("Use SAM point prompts", actions.sam),
    ], 0);
  }
  if (editor.generationLayers?.some(layer => layer.status !== "complete"))
    card.append(el("p", { class: "workflow-note" }, "An automatic mask layer has incomplete coverage. Inspect its status in Proposal layers or Jobs, and check for missing objects before confirming the image."));
  return card;
}

export function imagesGuide(app) {
  const p = app.project, next = nextReviewImage(p),
    count = p.images.filter(i => i.complete).length;
  if (!p.images.length)
    return guide("Next: import your images", "Choose Import images, select files or a folder, then use Add images to project. After import, open an image to annotate it.");
  return guide(next ? "Next: open an image and review its objects" : "Next: check project images or export reviewed work",
    `${count} of ${p.images.length} images marked reviewed. Open an image to create masks, label and accept or reject them, then confirm the full image review. Export and training are separate choices; training is optional.`, [
    next && button("Continue image review", () => app.annotate(next.id), "primary"),
    count > 0 && button("Export reviewed annotations", () => app.navigate("export")),
    !next && button("Check project images", () => focusControl(app.main, "[name=role-filter]")),
  ].filter(Boolean));
}

// Job history describes a completed operation, not the current review state.
export async function openJobContext(app, job, route = "images") {
  const pid = job.payload?.project_id;
  if (!pid) throw Error("This job has no project reference. Open the intended project from Projects.");
  if (app.project?.id !== pid) await app.openProject(pid);
  if (app.project?.id !== pid) return; // Navigation may have been declined.
  if (route === "annotate") {
    await app.refreshProject();
    const iid = job.payload?.image_id || job.payload?.layers?.[0]?.image_id || job.payload?.image_ids?.[0];
    if (!app.project.images.some(i => i.id === iid)) {
      await app.navigate("images");
      return;
    }
    return app.annotate(iid);
  }
  return app.navigate(route);
}

export function jobGuide(app, job) {
  const kind = job.kind, status = job.status,
    open = (label, route) => button(label, () => openJobContext(app, job, route), "primary");
  if (status === "awaiting_review")
    return guide("Next: review the proposed changes", "Use the review button below to inspect the proposed changes and make an explicit decision before continuing.");
  if (["generate", "predict"].includes(kind)) {
    if (status === "complete")
      return guide("Next: review the saved mask proposals", "Open the annotation results, inspect the masks, assign classes, then accept or reject them. Confirm the full image review when finished. A completed generation job does not mean the annotations are reviewed.", kind === "predict" ? [open("Review predicted masks", "annotate")] : []);
    if (["failed", "cancelled", "interrupted"].includes(status))
      return guide("Next: check the job before continuing", "Some results may be saved, but coverage can be incomplete. Read the error and resume or retry if appropriate; inspect all results before marking any image reviewed.");
    return guide("Mask generation is in progress", "Wait for the job to finish. Then open its annotation results to label and review the masks.");
  }
  if (status !== "complete") return null;
  if (kind === "assist")
    return guide("SAM preview request completed", "Previews are not accepted annotations. In the editor, save the previews you want as drafts, then label and review them. If you already saved them, continue reviewing; if you left without saving, place the points and request a new preview.", [open("Return to this image", "annotate")]);
  if (kind === "import_images" || kind === "import_annotations")
    return guide("Next: open the project images", "Check the import summary for files that could not be added. Open an image to create or inspect its annotations, then label and review objects.", [open("Open imported images", "images")]);
  if (kind === "install_model")
    return guide("Next: choose the model in your workflow", "Check Models & AI for readiness. Use a ready SAM model in Generate masks or SAM assistance; use a ready YOLO model in Train or Predict selected. Refresh devices to choose CPU or a detected GPU.", [button("Check model readiness", () => app.navigate("models"), "primary")]);
  if (kind === "train")
    return guide("Next: inspect the trained model", "Check the training results and model status in Models & AI. To use a trained model for new predictions, select or explicitly activate it there, then choose images and Predict selected. Review the new predictions before accepting them.");
  if (kind === "export")
    return guide("Next: download and check the export", "Use Download artifact below, extract the archive and read its conversion report before using the annotations in another tool.");
  return null;
}
