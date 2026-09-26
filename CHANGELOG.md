# Changelog

## 1.0.0rc24

- GitHub preparation: synchronized current install/version documentation and bundled help; added English release notes, publication guide and a public verification summary. No executable application changes in this packaging step.

- YOLO best.pt models now have date/minute/timezone labels and a short model ID in lists, training/prediction selectors and activation confirmation. New model names and UTC completion timestamps are saved. Older models use their recorded registration time, clearly labeled. Re-registration preserves naming provenance. Checkpoint paths, model IDs and all training/inference settings are unchanged.


## 1.0.0rc23

- Added exact 512px source-tile training, original-image/group split lineage, fragment validation and immutable tile manifests. Old snapshots and whole-image API requests remain compatible.
- Bound new trained models to 512px prediction crops/input size, edge coverage, bottom/right padding and class-aware full-image box NMS.
- Adapted the preserved contextual boundary policy: bounded context sizes, native SAM quality, coverage/area/neighbour gates, ambiguity reporting, explicit review of suggested fragment merges, and persistent atomic Undo/Redo. Boundary recovery remains OFF by default; prediction has an explicit opt-in.
- Added unit/browser regressions and a bounded real YOLO/SAM integration checker. No live-project retraining or automatic model activation.


## 1.0.0rc22

- Add explicit, reversible per-object exclusions for incompatible YOLO geometry in future training and validation snapshots, shared by all supported YOLO segmenters.
- Keep annotation geometry, classes, review and exports intact; retain snapshot exclusion provenance and reject stale selections atomically. Omit all-excluded images rather than treating them as negative examples.
- Explain the background-label effect before confirmation, keep lossy conversion off, and offer selective restoration. Skip excluded objects during optional training boundary preparation.
- Open Train settings without waiting for full-resolution mask geometry validation; show the pending check and elapsed time in Data readiness.
- Reuse an in-flight check with matching settings and retain all checks before training submission.
- Keep settings on validation errors, offer inline retry, and ignore stale results after navigation or settings changes.

## 1.0.0rc21

- Add Confirm review on Images with explicit consent for pending labeled objects and whole-image review; empty images require negative-image confirmation.
- Save the confirmation atomically with revision checks and persistent Undo, without loading mask geometry or changing class labels.
- Show role selection separately from review completion, update image cards after role saves, and add Continue to Train with fresh readiness.
- Explain selected-but-unreviewed images on Train; preserve training, geometry, split and model validation.

## 1.0.0rc20

- Restore right-button dragging to pan in all annotation tools, including SAM points.
- Use Ctrl+D to remove the last temporary SAM point; preserve the Remove last point button and exclude forms and dialogs from the shortcut.
- Update editor hints and offline help to match the mouse and keyboard controls.

## 1.0.0rc19

- Right-click in either SAM point tool removes the last temporary prompt point; repeated clicks step backward through the points.
- Share the existing Remove last point action, discard stale previews, and keep saved masks, box prompts and other tools unchanged.
- Keep point removal lightweight and document the shortcut in the editor and offline help.

## 1.0.0rc18

- Restore direct SAM point placement without saved-mask scans, geometry fetches or duplicate warnings.
- Replace overlap computation with simple shared/individual saved-mask colors near Selection; retain existing per-mask colors and ignore legacy overlap settings.
- Coalesce prompt, pan, brush-feedback and mask-loading redraw requests per animation frame; debounce browser preference writes.
- Preserve unrestricted prompt counts, draft batch acceptance, persistent Undo/Redo and annotation/export data.

## 1.0.0rc17

- Warn before independent point prompts inside existing SAM Assist masks, including hidden masks; allow deliberate overlap or refinement.
- Add browser-local selected-mask colors and a configurable overlap highlight without changing annotations or exports.
- Add confirmed, atomic class assignment and acceptance for selected/all SAM Assist drafts in the current image, with persistent Undo/Redo.
- Remove the fixed SAM point-count limit; independent prompts remain sequential and cancellable.

## 1.0.0rc16

- Bound new YOLO training batches and gradient accumulation by the actual dataset size across every supported segmentation architecture.
- Measure real optimizer calls and learnable-weight changes; prevent successful model registration when no effective update occurred.
- Record requested/effective settings and update evidence in training results and model metadata. Preserve historical checkpoints and resume schedules.

## 1.0.0rc15

- Guide users with missing dataset roles from Train to the labeled Dataset role dropdowns in Images.
- Explain pool/train/validation, automatic role saving, and preservation of completed review when roles change.
- Keep role assignments, training rules, model settings and annotations under the existing explicit user controls.

## 1.0.0rc14

- Explain incomplete image review with exact pending-mask counts and direct Show remaining masks navigation.
- Reveal and select the next pending mask without changing its label or review decision.
- Keep editing available after the server's incomplete-review validation; preserve genuine save/conflict recovery and unsaved SAM previews.

## 1.0.0rc13

- Add contextual next-step guidance for SAM previews, saved proposals, labeling, review confirmation, next images, export and optional training.
- Guide job results back to their owning project/image and keep failed saves and incomplete coverage explicit.
- Preserve all annotation, training, model, device and export behavior; guidance navigates or invokes existing explicit actions.

## 1.0.0rc12

- Explain training/validation readiness with current counts, existing minimum requirements, excluded image names and explicit review steps.
- Recheck readiness before new training submissions without changing roles, annotations, split rules, GPU settings or resume snapshots.
- Distinguish accepted training-instance counts from combined training/validation counts.

## 1.0.0rc11

Release preparation: readable dialogs across system themes, repaired current browser contracts, dynamic installed-wheel CI, strict reproducible staging, approved AGPL-3.0-only licensing/authors, optional-provider security pins and honest evidence indexing. See docs/UPDATE_RC11_EN.md.

## 1.0.0rc10

- Make annotation dataset export the main workflow, with COCO default and task-specific format choices.
- Preview real counts, skipped items and existing splits; distinguish reviewed datasets from labeled drafts.
- Produce small history-free datasets with documentation, filename mapping, actual ZIP sizes and an inline download.
- Keep full project backup separate from training exports.

## 1.0.0rc9

- Add a display filter for automatic SAM masks, preserving manually created and new prompted masks.
- Add a switch to disable saved mask fill while retaining selection and editing feedback.
- Keep generation, annotations, review decisions and exports unchanged.

## 1.0.0rc8

- Make application-authored interface and offline help English-only.
- Omit the old Persian quick guide and local Persian reports from new installation packages, preserving local originals.
- Preserve user-entered text, SAM assistance, review controls and existing settings.

## 1.0.0rc7

- Add independent point prompts for multiple SAM objects, simultaneous previews, and atomic draft saving with undo.
- Preserve single-object point/box refinement and local-only SAM3 weights.
- Distinguish pending review, accepted and rejected masks, including hidden masks, with a pending-review view.

## 1.0.0rc6

- Add an optional Hide assigned masks filter next to review selection controls.
- Keep hidden objects out of active selection and canvas hits, preserving saved annotations and history.
- Preserve existing per-class and per-object visibility controls.

## 1.0.0rc5 — GPU cache preflight and general SAM labels

- Reclaim idle CUDA allocator cache only under memory pressure and recheck free memory before rejecting work.
- Preserve the error for genuine shortages and include the selected device, free memory and required reserve.
- Label the SAM automatic preset Default settings; preserve all numerical defaults and frozen generation recipes.

## 1.0.0rc4 — MPO import and project deletion

- Accept multi-picture JPEG/MPO primary images without resizing; retain source bytes, frame metadata and visible import notes.
- Add permanent project deletion with scope preview, exact-name/revision/path confirmation and active-job guards.
- Erase project and job data while keeping external originals, other projects and checksum-verified shared models.
- Cover cancellation, stale confirmation, overlapping/referenced projects, shared models, rollback and incomplete-purge reporting.

## 1.0.0rc3 — image import and removal

- Explicit selection → add → view workflow; subfolders on by default and file-level feedback.
- Bounded sequential uploads, visible completion and immediate image-list refresh, including approved local paths.
- Remove on image cards with confirmation; preserved originals/history, protected started rounds, mutable planned allocations updated.
- Duplicate progress and display-name accounting; completed upload staging is cleaned even for mixed-validity batches.
- Removed images stay outside new export/training data and active class remapping.

## 1.0.0rc1 — unreleased candidate

New local application packaging for general 2-D image annotation with user-defined labels, exact mask geometry, explicit review, optional local model providers, cumulative segmentation training and flexible rounds.

Packaging introduces a user-space Ubuntu installer, verified local Python bootstrap input, desktop launcher/icon, CLI launch/help/version/doctor/backup commands, packaged offline help and public staging/content/artifact checks. Core dependencies remain independent of Torch and model weights.

### Addendum 01 — integration in progress

The same unreleased candidate is being extended with actual configurable source tiles (whole image, 256/512/1024 square presets and independent Custom width × height), overlap/grid preview and immutable job settings. New projects start at 512×512 with 25% overlap; existing job grids and training snapshots remain preserved. Encoder size and training image size remain separate controls.

The required SAM2 automatic-proposal workflow starts without classes or YOLO. It adds durable full-image generation layers, class creation after generation, distinct selection/paint/prompt/geometry modes, multi-selection, bulk assignment/review, explicit exact-mask merge and persistent correction/delete/undo. Merge preserves holes/components and lineage; optional refinement is a separate operation. SAM3 automatic generation remains unavailable until a compatible adapter is implemented and tested, independently of point/box assistance.

The former training-only heavy boundary restriction is superseded by two explicitly opted-in contexts: training preparation and automatic-mask annotation. Recovery remains OFF by default, scoped per job and staged for review; unrelated operations cannot trigger it. User guides, offline help and the addendum requirement map reflect this policy.

These entries describe the incremental implementation target, not passed addendum acceptance. Final CPU/API/browser/real-provider results, installed wheel/sdist checks and human UAT must be recorded against the frozen source. Both authorized training jobs are already consumed; this addendum does not authorize another training run. No final artifact has been delivered, so the version remains 1.0.0rc1.

This is a candidate, not an approved public release. Capability/test evidence and remaining human, provider, native OS and ownership gates are recorded separately. No performance parity or speedup claim is made for the preserved research workflow.

- Integration corrections: persistent generation-decision tombstones prevent overlap-tile resurrection on resume; recovery decisions participate in undo/redo and stale suggestions can be dismissed safely; restored copies cannot resume another project's job.
