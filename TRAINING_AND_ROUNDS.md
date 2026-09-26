# Training and rounds

Rounds organize review work. Epochs control repeated passes through a training dataset. You can annotate and export without either.

## Plan review batches

Choose eligible pool images and a round count `R` between 1 and the pool size `N`. Fixed validation, test and excluded images stay outside this pool. Import order, a saved shuffle seed or a manual order defines the sequence. The planner distributes the remainder over the first batches: 20 images in 3 rounds gives 7/7/6; 10 in 3 gives 4/3/3. Inspect the preview before confirming.

Finish a round after actual image review. Annotation-only completion means **review complete; training not run**. Add new images or replan future work explicitly; completed history and past model bindings must remain preserved.

## Prepare valid segmentation data

The readiness panel reports complete reviewed images, instances per class, negative images, unsupported geometry, class balance, group/split coverage and exclusions. The normal snapshot uses only live accepted segmentation objects from explicitly completed eligible training images. Unassigned/draft/rejected/deleted/superseded masks and partial images are not ground truth or an implicit background class. Bounding boxes alone are not segmentation targets; draw/refine a polygon or mask first.

SAM2 automatic proposals, manual masks and reviewed edits use the same project state. You may generate with zero classes, then create classes and label/accept selected instances. Class assignment alone does not confirm geometry or completeness. After a merge, the accepted child enters the next snapshot once and superseded parents do not; undo restores the appropriate parents. Changes reopen completeness review and require a new readiness check. Class edits rebuild the next job's explicit class map; existing snapshots and trained models keep their original mapping. Hiding/filtering proposals never creates a reviewed negative image.

Keep original-image groups together before cropping or augmentation. Validation is fixed and separate; never reuse training images as validation just to satisfy a loader. Supply a valid validation split when the backend requires it. Test data do not choose the active model. A tiny smoke test shows that the software works, not that the model generalizes.

Select the base or previous segmentation model, epochs, image size, batch, device and seed. Later-round training uses all eligible reviewed training data so far. The snapshot fixes the annotation versions, class map, image/group splits, conversion report, source hashes and settings; subsequent edits belong to the next job.

The same strict YOLO segmentation converter is used for export and training. Holes or disconnected components may block conversion. If an approximate route is offered, inspect the affected IDs and changed-area/IoU report before explicitly enabling it. Native masks remain intact.

**New training defaults to 512 × 512 source tiles with stride 512 and input size 512.** Select Training images → Whole images to keep whole-image training for a new job. Last tiles anchor to the far image edge, so they can overlap earlier tiles; images smaller than 512 are padded at bottom/right with the bottom-right source pixel. Labels are clipped in source pixels. Strict topology checks also apply to each fragment: holes, disconnected pieces or unrepresentable polygons are reported, never silently repaired. Existing per-object exclusions remain available.

All crops retain the original image/group's training or validation role. Snapshots record original images, annotation revisions, tile origins, padding, label conversions and hashes. Empty crops from fully reviewed images have explicit empty labels. Validation metrics for tile training are **tile-level**, not measured full-image metrics. A one-image training split is still a very small dataset even if it produces many correlated tiles.

A new tile-trained model remembers its layout. In Predict selected, **Use the model's saved training layout** starts enabled when a saved tile recipe is available: source crop 512, stride 512, network input 512, full-image coordinate mapping, and per-class full-image box NMS at IoU 0.5. Changing confidence does not change the layout. Disable the matching option only to make an explicit custom override. Old models/snapshots are not relabeled as tile-trained. **Start a new training job** to learn at the new source scale; resuming an old job keeps its frozen dataset. Generic SAM tiling defaults remain separate.

Clipped seam fragments are not joined by box overlap alone. For contextual seam repair, explicitly enable **Boundary check**, choose a SAM model and a maximum number of objects (default 3, range 1–20). It may use up to three distinct 512/1024/1536 contexts per YOLO seed, or one per SAM-only seed, with a 32-pixel margin. Native SAM IoU/stability, 90% target/fragment preservation, area growth and neighbour protection gate suggestions. Distinct suitable alternatives remain unresolved. Only scoped, unreviewed, unedited same-class seam fragments can be proposed for merging. Reviewed neighbours remain protected. Inspect the union of original fragments against the suggestion; accepting applies the replacement and supersedes companions atomically, with persistent Undo/Redo. Nothing is applied automatically. Deferred objects and actual calls/timings appear in the job report.

## Explicit boundary contexts

Heavy boundary checking/recovery is **OFF by default**, including normal training. It is allowed only with explicit job-level consent in **training preparation**, **automatic-mask annotation**, or the explicitly selected **Boundary check** in YOLO prediction. It never starts automatically on import, image opening, manual edits/labeling, SAM prompting, merging, ordinary prediction, round transitions or export. Basic geometry checks, cheap seam flags and provider-internal candidate filters remain distinct from heavy recovery.

For applicable tile-edge data, explicitly enable the advanced **Check/repair tile-edge masks before training** option in the training job. Suggestions are staged for your review. Accept, retain the original or exclude the affected data, then explicitly retry training from the approved state. Cancellation leaves accepted geometry intact. This option is not required for manual annotations or for first training.

The separate automatic-mask workflow offers an unchecked **Check/repair tile-boundary masks after generation (slower)** option or **Run boundary check on these proposals…** afterward. It can work on unassigned proposals without a YOLO checkpoint. Confirm the selected proposals/current layer, attempt limit and resource use for that job. A physical image edge is not an internal seam; without relevant internal tile/crop boundaries the check may be not applicable. Failed/unresolved checks preserve original editable geometry. Review each proposed replacement before applying it.

Neither context grants permission for the other or for later jobs. Every new job starts OFF; a deliberately saved profile must still be visibly presented and confirmed. A saved browser checkbox or migration is not authorization. Manual merge stays an exact pixel union with no recovery call; optional SAM refinement is a separate, reviewed geometry change.

Prepared full-image inputs do not guarantee that backend crops/augmentations always keep objects whole. Inspect augmentation settings; no universal ignore-region promise is made for YOLO label text.

## Train, activate and continue

Strict conversion can reject masks with disconnected components or holes. **Exclude incompatible objects (N)…** on Train lets you select affected objects, acknowledge their treatment as background, and omit them from future training/validation labels without changing the annotation geometry or enabling lossy conversion. **Review training exclusions…** restores selected objects. All supported YOLO segmenters use the same snapshot selection. Snapshot receipts record exclusions; existing snapshots and resumed jobs are unchanged. Annotation exports continue to contain the original objects.

Exclusions are not ignore regions: omitted objects are seen as background and can affect learning and validation metrics. To avoid partial labeling, exclude the entire image using its dataset role. An image with all accepted instances excluded is omitted rather than counted as an intentional negative. Minimum reviewed training/validation and remaining-instance requirements still apply. Optional training boundary preparation skips excluded objects; annotation boundary tools remain separate.

Start the job and watch actual logs/epochs in **Jobs**. Cancel when needed; checkpoint-based resume depends on the backend and available checkpoint. A dataset file or an empty checkpoint is not successful training. Success requires a saved segmenter with the expected class map and a real load/inference probe.

New YOLO training jobs adapt effective batch and gradient accumulation to small datasets. Actual optimizer steps and learnable-weight changes are verified before a model can be registered; inspect the weight-update check and validation results in Jobs. This shared policy applies to all supported YOLO segmenters. Existing resume schedules are preserved; replace an unsuitable old short run with a new job from the original base model. See [short-run update checks](docs/YOLO_TRAINING_UPDATES.md).

Review model provenance and validation metrics, then approve activation (or deliberately select automatic activation for that job). The next inference must load that exact checkpoint, with its recorded hash and model-to-project mapping. Proposals do not overwrite accepted geometry. Rollback selects the preceding active model; it does not alter past annotations.

## Release acceptance, separate from ordinary work

The base demonstration uses six full images: four review-pool images over two rounds and two separately reviewed validation images; at least three classes; two short real YOLO training jobs; fresh next-round inference; native/COCO/YOLO checks and restart. Both authorized training jobs have already been consumed. Do not start a third training run for the addendum; the coordinator must validate the changed annotation-to-snapshot path and identify which prior bindings remain reusable. Automated QA is recorded as `automated_qa`, never human acceptance. Human review remains **AWAITING_HUMAN_UAT** until a person performs it. Addendum automatic-generation, merge-to-dataset and two-context boundary acceptance remain pending final evidence; prior prompting/training success alone does not pass them.

## Training readiness and missing validation images

The Train page reports reviewed training and validation image counts, accepted
training instances by class, excluded images, and steps to fix each blocker.
It rechecks the selected round and polygon-conversion setting before starting;
blocked data stays on Train without creating a failed training job. Refresh
readiness preserves the model, device, epoch, batch and other form settings.
Resuming a saved training job still uses its original frozen snapshot.

The existing execution minimum is one fully reviewed training image with at
least one accepted segmentation instance and one separate fully reviewed
validation image. This minimum does not establish model accuracy. Increasing
epochs or adding masks to a training image does not create validation data.

In Images, change the Dataset role of a separate image to `validation`. Annotate
and review all objects, accept or reject every proposal, then choose **Mark
reviewed** and confirm full image review. Keep at least one other reviewed image
as `pool` or `train`. Related image groups stay in one role. No role, annotation,
review attestation, or dataset split is changed automatically.
