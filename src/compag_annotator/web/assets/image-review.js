import { el, button, check, card, row, badge, dialog, notify, projectPath } from "./ui.js";
import { needsReview } from "./review-status.js";

export function imageReviewPlan(project, image, objects) {
  const active = objects.filter(o => !["rejected", "superseded"].includes(o.status));
  const classes = new Set(project.classes.filter(c => !c.archived).map(c => c.id));
  return {
    pending: active.filter(needsReview),
    accepted: active.filter(o => o.status === "accepted").length,
    unlabeled: active.filter(o => !classes.has(o.class_id)).length,
    negative: active.length === 0,
    eligibleRole: ["pool", "train", "validation"].includes(image.role),
    rejected: objects.filter(o => o.status === "rejected").length,
  };
}

export function imageTrainingSummary(project) {
  const eligible = project.images.filter(i => !i.removed);
  const training = eligible.filter(i => ["pool", "train"].includes(i.role));
  const validation = eligible.filter(i => i.role === "validation");
  return {
    training: training.length, validation: validation.length,
    reviewedTraining: training.filter(i => i.complete).length,
    reviewedValidation: validation.filter(i => i.complete).length,
  };
}

export function imagesTrainingPanel(app, savingRoles = false) {
  const counts = imageTrainingSummary(app.project);
  return card("Prepare images for training",
    el("p", {}, "1. Choose Dataset role under each image: train (or pool) teaches the model; validation checks it."),
    el("p", {}, "2. Use Confirm review under each chosen image after inspecting its objects. Choosing a role alone does not confirm review."),
    el("p", { "data-testid": "images-training-summary", role: "status" },
      savingRoles ? "Saving dataset role…" :
        `Reviewed images: ${counts.reviewedTraining} training · ${counts.reviewedValidation} validation. Roles selected: ${counts.training} train/pool · ${counts.validation} validation.`),
    el("p", {}, "3. Continue to Train to check the reviewed dataset, choose a model and start training. Other unfinished images may stay in the project; they are excluded. Geometry or model setup issues are shown in Train."),
    button("Continue to Train", () => app.navigate("train"), "primary", { disabled: savingRoles }),
  );
}

export async function confirmImageReview(app, imageId, onSaved) {
  const pid = app.project.id, base = projectPath(pid), path = base + `/images/${imageId}`;
  const [project, scene] = await Promise.all([app.api.get(base), app.api.get(path + "/annotations")]);
  if (app.project?.id !== pid || app.route !== "images") return;
  if (scene.revision !== project.revision) throw Error("The project changed while loading. Open Confirm review again.");
  const image = project.images.find(i => i.id === imageId);
  if (!image) throw Error("This image is no longer in the project.");
  const plan = imageReviewPlan(project, image, scene.annotations);
  const pendingConsent = plan.pending.length ? check("accept-pending-image-review",
    `I inspected all ${plan.pending.length} pending labeled objects and accept them with their current classes.`) : null;
  const attestation = check("attest-image-review", "reviewAttestation");
  const negative = plan.negative ? check("confirm-negative-image-review",
    "This is an intentional negative image with no target objects.") : null;
  const status = el("p", { role: "alert", class: "inline-warning", hidden: true });
  const checked = control => !control || control.querySelector("input").checked;
  let busy = false, failed = false;
  const blocked = !!image.missing || !plan.eligibleRole || plan.unlabeled > 0;
  const update = () => {
    save.disabled = busy || failed || blocked || !checked(attestation) || !checked(pendingConsent) || !checked(negative);
    for (const control of d.querySelectorAll("button,input")) {
      if (control !== save) control.disabled = busy;
    }
  };
  const save = el("button", { type: "button", class: "primary", onclick: async () => {
    if (save.disabled) return;
    busy = true; update();
    try {
      if (app.project?.id !== pid) throw Error("The active project changed. Reopen this image's review.");
      await app.api.post(path + "/review", {
        expected_revision: project.revision, expected_role: image.role,
        annotation_revisions: Object.fromEntries(scene.annotations.map(o => [o.id, o.revision])),
        attest: checked(attestation), accept_pending: !!pendingConsent && checked(pendingConsent),
        confirm_negative: !!negative && checked(negative),
      });
      d.close();
      await onSaved();
      notify(`${image.name}: review confirmed for ${image.role}. Use Continue to Train to check readiness.`);
    } catch (error) {
      failed = true;
      status.hidden = false;
      status.textContent = error.message + " Reload review status before confirming again.";
    } finally { busy = false; update(); }
  } }, "Confirm image for training");
  const d = dialog("Confirm image review for training", el("div", {},
    el("h3", {}, image.name), row(badge(`Dataset role: ${image.role}`)),
    el("p", {}, `${plan.accepted} accepted · ${plan.pending.length} pending review · ${plan.rejected} rejected.`),
    el("p", {}, "Confirm only after inspecting the whole image and every object. Labeled drafts are not accepted automatically. Use Open in Annotate to inspect, correct or reject objects first."),
    plan.unlabeled > 0 && el("p", { class: "inline-warning" },
      `${plan.unlabeled} objects need an active class or rejection. Resolve them in Annotate before confirming.`),
    !plan.eligibleRole && el("p", { class: "inline-warning" }, "Choose train, pool or validation in Images first."),
    image.missing && el("p", { class: "inline-warning" }, "Relink the missing image before confirming review."),
    plan.negative && el("p", { class: "inline-warning" }, "There are no kept objects. Confirm an empty image only if it truly has no target objects; it will be used as a negative example."),
    pendingConsent, attestation, negative, status,
    el("p", {}, "This records review only. Train checks the dataset and model separately; training does not start here. Undo in Annotate can reverse this confirmation."),
    row(save, button("Open in Annotate", async () => { d.close(); await app.annotate(imageId); }),
      button("Reload review status", async () => { if (busy) return; d.close(); await confirmImageReview(app, imageId, onSaved); })),
  ));
  for (const c of [pendingConsent, attestation, negative]) c?.addEventListener("change", update);
  d.addEventListener("cancel", e => { if (busy) e.preventDefault(); });
  update();
  return d;
}
