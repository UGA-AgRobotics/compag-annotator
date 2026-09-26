# User guide

The Addendum 01 workflow below is being integrated into this release candidate. Its real-provider and installed-UI checks remain pending; see the release acceptance records before treating a feature as validated.

## Projects and images

**Projects** creates, opens and renames projects, and provides backup/restore. Restore into a new location so the existing project remains available. Backup archives normally exclude model weights. A backup contains your data and must not be treated as a public diagnostic.

**Delete project** on a project card permanently erases its folder and project job outputs after you type its exact name. External originals and shared models are kept. The confirmation lists the scope; files placed inside the project folder are included. Finish active jobs first. Other projects and originals they reference are protected. This has no Undo.

**Images → Import images**: (1) Choose image files or a folder, or drop them into the window. (2) Check the selected count and the skipped-file details; subfolders are included by default. Nothing is added until you click **Add N images to project**. (3) Wait for the import summary, then click **View images**. The project list refreshes automatically. Large selections use sequential batches (up to 100 files and a target of 64 MiB per batch); a single image may be up to 255 MiB for browser upload. Duplicate files are reported as already in the project. Corrupt files are listed separately, while valid files still import.

**Remove** on each image card asks for confirmation, then removes that image and its annotations from the active dataset and new exports. It keeps original files and archived project history; it is not a disk-cleanup command. A removed image can be imported again as a new image. Images in started review rounds are protected. Finish or cancel this project's running jobs before removal. Planned round allocations are updated when an image is removed.

For folders on the computer running COMPAG, the approved local-path form is an alternative. It also shows an import summary and **View images**. Copy storage keeps portable project-owned files. Reference storage requires originals to remain in their source locations.

Review the summary for corrupt files, unsupported formats, duplicates and disk usage. JPEG (including MPO), PNG, BMP, WebP and the supported TIFF subset are the baseline. Multipage and unusual high-bit-depth inputs require an explicit supported conversion; do not assume they are accepted. Originals are preserved; EXIF orientation defines the displayed coordinate system. Object coordinates use the full oriented image, independent of zoom or thumbnail resolution.

Image roles distinguish review pool, training, fixed validation, test and excluded data. Related images and duplicate content should stay in one split. Search and filters help find unfinished work. Removing an image from a project should not delete a reference-mode original.

To prepare training on **Images**, choose **Dataset role** under each chosen image: **train** (or **pool**) for teaching the model, and **validation** for checking it. Role changes save automatically. Next, use **Confirm review** under each chosen image. The dialog shows accepted and pending object counts. After inspecting the full image, explicitly confirm its review; if there are labeled drafts/proposals, separately confirm accepting them with their existing classes. Unassigned or archived-class objects must be labeled or rejected in **Annotate** first. An empty image requires an additional intentional-negative confirmation. No boxes are ticked automatically.

Confirmation records acceptance and full-image review together, preserving masks, classes and rejected objects. **Undo** in Annotate reverses the confirmation and reopens image review. If another edit or role change occurs, reload the dialog and inspect its updated counts before confirming again. Click **Continue to Train** to load fresh readiness. At least one different reviewed training and validation image are needed; unfinished images may stay excluded. Geometry, split and model setup checks still apply, and training starts only when you choose **Train model**.

## Labels

Start with zero classes if desired. Generate and edit masks first; use **Create new class…** inside the editor to give selected masks a name and color afterward. **Assign class** applies a class to every selected instance without merging or accepting them. For example, five leaf masks assigned to “Leaf” remain five objects. Only a clearly named combined assign-and-accept action performs both steps.

Add, rename, recolor or reorder classes in **Labels**. Class identities remain stable through display-name changes. Preview the effects of archiving or merging used classes before confirming. Merging classes remaps labels; it does not merge object masks. A trained model keeps the class mapping it learned; adding a label does not teach an existing model that label.

Unassigned, rejected, background and review status are not automatically classes. A negative SAM click excludes pixels; it is not a negative object class.

## Annotate and review

Use the full-image canvas to draw polygons and boxes and correct masks. Choose a distinct active tool: **Select** selects masks, **Paint class** deliberately labels them with the chosen class, **SAM prompt** places model prompts, and **Edit geometry** changes shapes. A selection click must not also label or prompt. Hover highlights the mask; click selects it and focuses its object-list entry. Shift/Ctrl-click toggles additional selections. Use the object-list selection controls when a keyboard is inconvenient; check the selection count and use **Clear selection** before changing scope.

For overlapping or nested masks, use the overlap-cycle control or select the desired object in the list. Clicking inside a bounding box but outside its actual mask is not a mask hit. Zoom/pan changes the view, not selection geometry. Unassigned proposals have a neutral style and filter; proposal, unassigned, labeled-draft, accepted, rejected and superseded counts represent different states.

Select an object for brush add/erase, polygon/vertex correction, SAM refinement, or rejection/deletion. **Hide** only changes display; **Reject** dismisses a proposal; **Delete** removes an instance from the active saved/export/training set while preserving undo history. Confirm bulk or approved-data deletion. Draw a missing object manually when needed. Undo/redo reverses edits across restart; watch the save state before leaving. Class visibility and overlay opacity never change saved geometry or training eligibility.

Enable **Hide assigned masks** under **Selection** to hide objects that already have a class and automatically hide each mask after labeling. Turn it off to show those objects again for acceptance or correction. Hidden objects cannot intercept clicks or remain in the active selection. For one class, use **Layers & display**; for one object, use its **Show object** checkbox. These are view-only filters, reset when opening an image, and do not delete annotations or change exports.

Accepting an object and confirming an image's completeness are separate actions. Resolve relevant unassigned/draft proposals, or explicitly dismiss non-target proposals, then inspect the entire image for missing objects before completion. Resolving all returned candidates does not prove the model found every object. A deliberately reviewed empty image is different from an unannotated or filtered image. An imported annotation, class assignment or successful prediction is not evidence of human review. Substantive edits reopen completeness review; previous immutable training snapshots retain their historical state.

SAM assistance also supports **One mask per point · multiple objects**: click one positive point in each desired object, then **Preview N masks** to see separate masks together. Use **Add all N masks as drafts** to save the group in one undoable action. Empty results are reported. For positive/negative points or a box describing one object, choose **Refine one object · points / box**.

Hold the **right mouse button and drag to pan the image**, including while placing SAM points. Press **Ctrl+D** in the annotation editor to remove the last SAM point. Repeat to remove earlier points, or use **Remove last point** in the panel. The shortcut does not run while typing in a form or using a dialog. This removes only temporary prompt points, never saved masks or a box prompt. An existing preview is discarded when prompts change; preview again before saving. **Space + drag** and the **Pan** tool also move the image.

There is no fixed SAM point-count limit. Independent points are processed sequentially, with progress and cancellation. More points require more time and saved-preview memory; this is not a promise of unlimited hardware capacity. Validation still rejects duplicate/contradictory points and coordinates outside the image.

SAM point placement is direct: you can click on objects that already have masks, including hidden masks. No duplicate-mask dialog or scan of saved mask pixels runs when you place a point. Use **Refine with SAM** when you want to correct an existing mask.

Use **Label & accept SAM Assist drafts (N)…** in SAM assistance to review selected drafts or all saved SAM Assist drafts in the current image. Choose an existing class, inspect the count (including hidden drafts), and explicitly confirm the review. The batch retains separate instances, leaves other annotations alone, and supports persistent **Undo/Redo**. It does not mark the full image complete.

At the top of **Selection**, choose **Color for all masks** to recolor saved masks, including masks generated earlier. It also becomes the display color for new saved masks on this image. Select one or more masks to use a separate **Display color**. **Reset selected mask colors** restores the shared display color (or class color); **Reset all mask colors** restores class colors for every object. These preferences stay in this browser for the project image and do not change annotations, training or exports. **Show mask fill** still turns saved fills on/off. The separate overlap highlight was removed to keep review responsive; legacy overlap settings are ignored while individual saved colors are retained.


The Selection panel distinguishes **Labeled · needs review**, **Accepted** and **Rejected**, including hidden-object counts. **Review N pending masks** reveals unfinished objects and hides completed decisions; **Show all active masks** restores the active view. Hiding a mask never accepts it.

SAM assistance previews a mask from positive/negative points or a box. Refine the prompts, inspect the preview and accept it with your selected class. A cancelled or stale response must not attach to another image or overwrite accepted work. Exact masks retain holes and disconnected parts; some export formats cannot.

Keyboard controls shown in the editor are authoritative for the installed version. Scroll zooms; Space-drag pans; Ctrl/⌘+Z undoes and Ctrl/⌘+Shift+Z redoes. Shift/Ctrl-click toggles mask selection in **Select**. Use the visible geometry tool for vertex editing so a selection gesture cannot unexpectedly remove a vertex. Focus a form field to type class names without triggering drawing shortcuts. Object-list controls provide a non-keyboard alternative.

## Generate masks across a full image

With a ready SAM2 provider, choose **Generate masks / Segment everything**. No YOLO checkpoint, predefined class or research dataset is needed. Masks are class-agnostic proposals requiring review; the workflow name promises neither complete object discovery nor correct geometry. SAM3 automatic generation is a separate capability and currently unavailable until a compatible adapter is implemented and tested; use SAM2 for this workflow.

The default scope is the current **full image**, including areas outside the viewport. A selected-image/folder batch needs explicit confirmation. In **Processing / Tiling**, preview whole-image processing or the grid below. Check original dimensions, planned tile count, overlap and the job summary before starting. Saving project defaults does not start generation; overrides apply to the new job only.

| Setting | Meaning |
|---|---|
| Whole image | One full-image input under the recorded model transform |
| 256×256, 512×512, 1024×1024 | Actual source-image crop sizes in width × height pixels |
| Custom width × height | Independent positive integer dimensions, including 640×384 and 513×385 |
| Overlap | Percentage or advanced per-axis pixels; the active mode determines overlap and stride |

New projects start at 512×512 with 25% overlap per axis: 128 pixels overlap and 384 pixels stride. At 25%, the other square presets resolve to 64 or 256 pixels overlap. Percentage overlap is rounded down separately on each axis; pixel overlap must be nonnegative and smaller than that axis's tile dimension. The final tile aligns to the far edge without a duplicate origin; that last overlap can exceed the nominal value. Smaller images use their valid crop without external padding. The preview covers the entire image without gaps.

The canvas remains the full oriented original. Tile outlines never cut saved masks or cross-tile edits. Source crops differ from the model's encoder resizing/padding and from YOLO **image size**; consult the job's recorded transforms. Choosing a larger crop does not guarantee native-scale model processing, equivalent masks or faster execution. Changing an inference grid does not alter accepted masks, old jobs or training crops. Training remains full-image unless a separate training-crop plan is explicitly selected.

Choose sampling density and prompt batch size; advanced quality/stability and duplicate-filter settings belong to generation. More prompts may use more time/memory without guaranteeing recall. Additional internal SAM crop layers are separate from external tiles and start at zero. Topology-changing cleanup is off. SAM's own filters may remove candidates; heavy boundary recovery OFF does not disable those provider filters. The app must not discard all tile-edge fragments or automatically merge neighboring proposals.

**Jobs** records completed/total tiles, returned proposal counts, timings and failures. Cancelled or failed work stays visibly partial. Resume the same generation's unfinished tiles; repeated Start requests must not append duplicate copies. For another run, choose a new proposal layer or replace only the chosen unreviewed layer. Accepted/edited annotations and rejection/merge history remain protected. Nonidentical overlapping alternatives remain accessible; identical masks may share a representative with all source lineage retained.

## Merge fragments, then refine if needed

Select at least two active mask-capable instances from the **same image**, including fragments from different tiles, then choose **Merge selected masks**. Inspect the preview: the result contains exactly the pixels present in any selected parent. Holes that remain empty and disconnected parts are preserved. The area is recomputed, so overlapping pixels count once. Box-only objects cannot be merged into a mask without a separate explicit mask-creation step.

Confirm the proposed class. Conflicting parent classes require an explicit target class or an explicit unassigned result. Merging creates one new draft instance and supersedes its parents; it does not inherit human approval or invent a combined confidence score. Accept the reviewed result separately, unless you deliberately choose a clearly labeled **Merge and accept** action. Undo restores the parents and their review state; redo restores the merged operation. Neither path should leave child and parents active together.

Merge needs no model and makes no model calls. It cannot fill missing pixels between fragments. Use a **separate** brush correction or **Refine merged mask with SAM** preview to add/remove pixels, then review that change. Native masks and COCO RLE preserve disconnected unions and holes; strict YOLO segmentation can block them. Never accept an unreported bridge, filled hole or automatic split into different instances just to satisfy export.

## Optional tile-boundary check

Heavy boundary recovery is **OFF by default**. It is available only for explicitly authorized jobs in three scoped contexts: **training preparation**, **automatic-mask annotation**, and explicit **Boundary check** in YOLO prediction. In Generate masks, the advanced **Check/repair tile-boundary masks after generation (slower)** option starts unchecked. You can instead generate first, then choose **Run boundary check on these proposals…** for selected proposals or the current generated image layer. Inspect the scope, attempt limit and resource warning before confirming.

Each new job needs its own explicit consent; enabling annotation recovery does not enable training recovery, future unrelated jobs or background work. Import, image opening, labeling, manual edits, prompting, merging, ordinary prediction, round transitions and export never trigger it automatically. Basic geometry checks and cheap seam flags remain enabled, as do normal manual SAM prompts. A physical photo edge is not an internal tile seam; whole-image input with no internal crop seams may be not applicable.

The check uses relevant source-image context and stages proposed changes for review, including unassigned masks. It cannot silently infer a class, join touching objects or overwrite accepted geometry. Review old/new masks and affected IDs, then accept or retain the original. A failed/unresolved check keeps the original editable proposal with a warning; it does not promise completeness or silently switch the model, crop plan or image size. See [Training and rounds](TRAINING_AND_ROUNDS.md) for the separate training option.

## Models, rounds, training and exports

Set up optional providers in **Models & AI**, and use **Load and test** before relying on a registered checkpoint. A runtime installed successfully can still lack weights or fail on a particular device. See [Model setup](MODEL_SETUP.md).

**Rounds** splits the eligible pool into a chosen number of batches. It is optional. **Train** shows data readiness and creates a cumulative snapshot from explicitly reviewed data. Read [Training and rounds](TRAINING_AND_ROUNDS.md) before selecting model activation or boundary preparation.

Train opens its settings while the **Data readiness** panel checks reviewed images and mask geometry. Large full-resolution masks may take time; the panel shows an indeterminate indicator and elapsed time, not an invented percentage. You can adjust settings while it runs. **Train model** waits for the required check and never bypasses validation. A failed check offers **Retry data check** without clearing settings. Leaving Train stops waiting in the browser; server-side validation may finish separately.

In **Export**, choose the scope, image inclusion and format. Read [Export formats](EXPORT_FORMATS.md) for mask/topology limits. An export does not improve or alter the project annotations. **Reviewed annotations only** requires explicitly completed images and accepted objects; finish image review or deliberately choose a draft-inclusive export. A scope with no completed images is rejected with an explanation. **Native backup** is a complete project backup, including history and unreviewed state, regardless of this filter.

### Exclude incompatible masks from training

If strict YOLO conversion reports disconnected components, holes or other incompatible geometry, choose **Exclude incompatible objects (N)…** in Train. The dialog lists the affected image, class, object ID and reason. Check individual objects, or use **Select all**, read and check the acknowledgement, then choose **Exclude selected objects**. Readiness checks the remaining objects. Approximate conversion is not enabled by this action.

These exclusions apply to future training and validation snapshots for all supported YOLO segmenters. Original masks, classes, acceptance decisions and annotation exports stay intact. **Review training exclusions…** lets you restore selected objects. The annotation list also shows **Excluded from training**. Existing snapshots and resumed training retain their original dataset; start a new training job to use a new selection.

YOLO sees omitted objects as background; this can affect learning and validation metrics. This is not an ignore-region feature. If you want to avoid partially labeled images, instead set the whole image's **Dataset role** to **excluded** in Images. If every accepted object is excluded, that image is omitted from the new snapshot rather than counted as a negative image. At least one separate, fully reviewed training image with a remaining segmentation object and one fully reviewed validation image are still required. Exclusion is reversible and never starts training automatically.

## Jobs and recovery

**Jobs** shows queued/running/failed/cancelled work and recorded progress. A failed or interrupted job is not complete. Cancelling a job preserves accepted annotations; a provider may finish its current operation before it observes cancellation. Resume training only from an available, supported checkpoint. Retrying an export or import creates new work that should be checked before reuse.

If another view has changed the same object, refresh and resolve the version conflict rather than repeatedly overwriting it. After an unexpected shutdown, reopen the project and inspect interrupted jobs. Use **Help / Settings** for offline guidance, version and diagnostics. Review a diagnostic export for personal information before sharing it.

### Retry, backups and old recovery suggestions

A cancelled generation keeps completed tiles. Deleting, rejecting or replacing a generated mask records a reversible decision, so an identical mask from a later overlapping tile does not resurrect it during resume. Undo/redo restores these decisions with the geometry. A deliberately new proposal layer remains a new prediction set requiring review.

A native backup preserves masks, classes, generation metadata and edit history. Job execution artifacts are not part of that backup. In a restored copy, start a new generation; Resume cannot launch the source project's job.

If a tile-edge suggestion became stale after an edit, deletion or merge, choose **Retain** to dismiss it while preserving the current annotation. Stale geometry cannot be accepted or used to exclude an image. Accept, Retain and Exclude decisions have persistent undo/redo.

MPO files: one full-resolution, EXIF-oriented primary picture is imported per file. Additional embedded pictures stay in the unchanged original; the summary explains this. No HDR gain-map rendering is applied. See UPDATE_RC4_EN.md for details.

## Mask display controls

In Selection, use **Hide generated SAM masks** to hide masks originating in automatic generation while keeping new manual and SAM point/box objects available. In Layers & display, turn off **Show mask fill** to remove the translucent tint while keeping objects selectable. Hover/selection bounds remain visible, as do temporary SAM previews and brush feedback. These display controls do not change saved annotations or review decisions.

## Simple training export

Open Export → choose the training format and reviewed or labeled-draft data → Check export → Create dataset ZIP → Download dataset ZIP. Image copies are optional. Counts, skipped objects, existing splits and conversion notes are shown before export. Datasets exclude the project database and Undo/Redo history. The separate Project backup section preserves the full workspace. See the training annotation export update guide for format details.
