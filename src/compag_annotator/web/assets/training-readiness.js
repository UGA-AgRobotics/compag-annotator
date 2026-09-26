import { el, button, badge, card, row, details, notify } from "./ui.js";
import { trainingExclusionActions } from "./training-exclusions.js";

function imageRoleGuidance(app, report) {
  const roles = report.image_role_counts;
  if (!roles) return null;
  const codes = new Set((report.errors || []).map((error) => error.code));
  const missingValidation = codes.has("missing_validation_images") && !roles.validation;
  const missingTraining = codes.has("missing_training_images") && !roles.pool && !roles.train;
  if (!missingValidation && !missingTraining) return null;
  return el("div", { class: "inline-warning", "data-testid": "training-role-guidance" },
    el("h3", {}, "Choose image roles before training"),
    el("p", {}, missingValidation
      ? "No image has the validation role. Imported images start as pool; reviewed pool images count as training images."
      : "No image has a training role. Keep at least one image as train or pool, separate from validation."),
    el("p", {}, "Mark reviewed confirms your annotations. It does not choose a dataset role."),
    el("p", {}, `Current roles: ${roles.pool} pool, ${roles.train} train, ${roles.validation} validation.`),
    el("ol", {},
      el("li", {}, "Open Images. Under one image, change the Dataset role dropdown to validation. This image is used to check the model, not to teach it."),
      el("li", {}, "Keep a different image as train (or pool) to teach the model. Use at least one accepted segmentation object in the training image. Related images must stay in the same role."),
      el("li", {}, "Role changes save automatically. Already reviewed images keep their review status; you do not need to annotate them again. Finish review only for images marked Needs review."),
      el("li", {}, "Use Confirm review under each chosen image if it still needs review, then Continue to Train. You need at least 1 ready training image and 1 ready validation image; resolve any other reported issues before starting."),
    ),
    button("Set image roles", async () => {
      await app.navigate("images");
      app.main.querySelector('[aria-label^="Dataset role for "]')?.focus();
      notify("Choose Dataset role under each image: validation for checking the model, train or pool for teaching it. Changes save automatically. Then use Confirm review under each chosen image and Continue to Train.");
    }, "primary"),
  );
}

function imageReviewGuidance(app, report) {
  const images = (report.excluded || []).filter(i => i.reason === "image_incomplete" &&
    (i.role === "validation" || i.role === "train" || (i.role === "pool" && !report.image_role_counts?.train)));
  if (!images.length) return null;
  return el("div", { class: "inline-warning", "data-testid": "training-review-guidance" },
    el("h3", {}, "Finish review for your chosen images"),
    el("p", {}, "Their dataset roles are saved, but full-image review is not confirmed. In Images, use Confirm review under each chosen image. Accept labeled drafts only after inspecting them, and explicitly confirm the whole image."),
    el("ul", {}, images.map(i => el("li", {}, `${i.name || i.image_id} · ${i.role} · review not confirmed`))),
    button("Confirm images in Images", () => app.navigate("images"), "primary"),
  );
}

export function trainingDataMessage(report) {
  return [
    "Training data not ready.",
    report.summary,
    ...(report.errors || []).map((error) => error.message || error.reason),
  ].filter(Boolean).join(" ");
}

export function trainingReadinessPanel(app, report, refresh) {
  return card(
    "Data readiness",
    badge(report.ready ? "Ready for dataset validation" : "Action needed",
      report.ready ? "good" : "warn"),
    el("p", { "data-testid": "training-data-summary" }, report.summary ||
      `Ready images: ${report.training_images || 0} training, ${report.validation_images || 0} validation.`),
    report.training_layout?.mode === "tiles_512" && el("p", {"data-testid":"training-tile-summary"},
      `512 × 512 dataset: ${report.training_tiles || 0} training tiles, ${report.validation_tiles || 0} validation tiles. Metrics describe tiles; original-image roles stay separate.`),
    imageRoleGuidance(app, report),
    imageReviewGuidance(app, report),
    el("ul", {}, (report.errors || []).map((error) =>
      el("li", {}, error.message || error.reason))),
    trainingExclusionActions(app, report, refresh),
    el("p", { class: report.ready ? "" : "inline-warning" }, report.minimum_requirements?.message ||
      "Use separate, fully reviewed training and validation images."),
    row(
      button("Review images", () => app.navigate("images")),
      button("Refresh readiness", refresh),
    ),
    el("h3", {}, "How to prepare validation images"),
    el("ol", {},
      el("li", {}, "Open Images and set a separate image's Dataset role to validation. Keep another image as pool or train."),
      el("li", {}, "Annotate the image, assign classes, and accept or reject every object."),
      el("li", {}, "Use Confirm review in Images, or Mark reviewed in Annotate, and explicitly confirm full image review. Continue to Train; readiness refreshes automatically."),
    ),
    el("h3", {}, "Accepted training instances by class"),
    el("div", { class: "data-list" }, app.project.classes.filter((c) => !c.archived).map((c) =>
      row(el("span", {}, c.name), badge(String(report.training_instances_per_class?.[c.id] ?? 0))))),
    !!report.warnings?.length && el("ul", {}, report.warnings.map((warning) => el("li", {}, warning))),
    !!report.excluded?.length && el("details", {},
      el("summary", {}, `Images not ready or outside this training selection (${report.excluded.length})`),
      el("ul", {}, report.excluded.map((image) =>
        el("li", {}, `${image.name || image.image_id} (${image.role || "image"}): ${image.message || image.reason}`))),
    ),
    details("Detailed validation & exclusions", report),
    el("p", {}, "Training and validation are checked again before starting. Increasing epochs does not replace missing reviewed images."),
  );
}
