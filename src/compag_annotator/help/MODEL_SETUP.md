# Optional model setup

You can use manual tools without any model. Open **Models & AI** when you want assistance. Runtime installation, available weights, registration, loading and a successful real test are separate states. Device availability and speed depend on the installed runtime and hardware.

## SAM2

The setup flow offers SAM2.1 Tiny, Small, Base Plus and Large with matching configurations. Small is the proposed initial preset; Tiny is a lower-resource choice. Inspect the actual compatibility status before downloading a preset. Runtime setup and the selected official checkpoint require explicit consent and display progress. Use an existing checkpoint to avoid another download.

The [official SAM2 project](https://github.com/facebookresearch/sam2) documents the image predictor and model/config pairs. [Installation guidance](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) explains dependencies and optional CUDA postprocessing. A missing extension or untested CPU route must be reported; manual annotation remains available.

### Automatic proposals and source tiles

SAM2 is the required provider for **Generate masks / Segment everything**. It needs neither YOLO nor classes: generate first, create classes in the editor, assign them to selected proposals, then review and accept. Generation is distinct from positive/negative point or box refinement. Its installed full-image acceptance is pending Addendum 01 evidence; earlier prompt tests do not establish this capability.

The adapter targets the official [`SAM2AutomaticMaskGenerator.generate`](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/automatic_mask_generator.py), at the source revision pinned in the model catalog. It samples point prompts, filters mask quality/stability and applies duplicate filtering. These are provider operations, not semantic classification or the app's heavy boundary recovery. The provider's internal crop-edge filter can affect candidates when additional internal crops are used; OFF does not disable every edge calculation. No app rule should discard every external tile-edge fragment.

Choose **Whole image** or real source crops: 256×256, 512×512, 1024×1024 or independent Custom width × height. New projects start at 512×512 with 25% overlap. Preview the complete grid and actual crop dimensions before running. Source crop size differs from encoder resizing/padding and YOLO `imgsz`; full-image coordinates and the actual provider transform belong to the job record. Model preprocessing does not change the original stored image.

The initial application profile uses 16 sampling points per side, 32 points per batch, predicted-quality threshold 0.8 and stability threshold 0.9; the smaller smoke profile uses 8 points per side. These are product settings, not a claim of optimal recall or upstream defaults. Advanced settings are recorded with the job. External tiling initially uses zero extra internal SAM crop layers. Topology-changing small-region cleanup is off, preserving returned holes/components. Review a separately offered cleanup preview before allowing geometry changes.

More samples, overlap and internal crop layers compound work. Watch actual completed/total tiles, proposal counts, effective filters, load/generation/grouping/recovery timings and warnings. Cancellation/failure yields a clearly partial layer; retry must retain completed tile receipts without duplicating proposals. On memory failure, explicitly choose a smaller prompt batch or another plan; no silent image/grid/model/precision change is acceptable. A quality score is not a calibrated class probability, and the workflow does not guarantee every physical object is found.

## SAM3

Install the optional runtime separately. Obtain an authorized image checkpoint yourself using the [official SAM3 access instructions](https://github.com/facebookresearch/sam3), then choose **Add local SAM3 weights** and confirm that you trust the file. The app must not download these weights, sign in, accept access terms for you or substitute unrelated video/multiplex weights.

The image adapter uses the selected checkpoint and disables upstream automatic download. The pinned [image builder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model_builder.py) exposes optional interactive capability separately. Missing/mismatched weights cannot count as a successful test. Report a missing checkpoint, unsupported capability and untested device independently. Consult the compatibility table for the previously tested local checkpoint; that result does not validate every checkpoint or device.

**SAM3 automatic generation is currently unavailable:** no compatible automatic/point-grid adapter has been validated for this release. Choose SAM2 for automatic proposals. SAM3 point/box assistance remains a separate capability with its own real evidence. A text prompt such as “everything” is not proof of exhaustive class-agnostic segmentation, and a mock or button is not a provider test.

SAM3 has a custom [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). Separate processes do not establish license compatibility; public integration remains subject to the owner review.

## YOLO segmentation

The requested catalog includes YOLO26 nano/small segmentation presets; see the [official segmentation documentation](https://docs.ultralytics.com/tasks/segment/). A compatibility alternative must be explicitly identified and validated. Inspect classes and map them to your project before inference. Unknown model classes remain unassigned; they are not all mapped to your first label.

Download a selected official base model only after consent, or register a trusted local segmentation checkpoint. A box-only model is not a segmentation model. Training on your reviewed data creates a new checkpoint with its own saved mapping; approve activation before using it in the next round.

Ultralytics presents [AGPL-3.0 and Enterprise licensing routes](https://www.ultralytics.com/license). This candidate proposes an AGPL-compatible source release subject to source rights and owner approval; no blanket proprietary redistribution permission is claimed.

## Privacy, trust and failure recovery

Heavy boundary recovery is **OFF by default** and requires explicit consent for each scoped job in **training preparation**, **automatic-mask annotation**, or explicit **Boundary check** in YOLO prediction. It is not normal AMG filtering. In the automatic workflow, choose the unchecked after-generation option or a later selected-proposal/layer check, then review proposed geometry changes. It does not automatically run from import, image opening, manual edits, labels, prompts, merge, ordinary inference, rounds or export. It must not infer classes or require a private/YOLO model for the SAM-only workflow. Manual editing, exact merge, deletion and export remain available without any optional provider.

Provider runtimes are isolated from the small core environment and from one another. They are installed from provider-specific requirements and pinned sources. No model weights belong in the source wheel. Image editing remains local and works offline after installation; downloads need a connection or user-supplied local assets.

Only select model files from sources you trust. Some checkpoint formats can execute code when loaded; a checksum or subprocess does not eliminate that risk. Review source, license, required disk space and device before installing. Cancel or retry failed downloads through the job interface. A corrupt, missing or mismatched checkpoint is a failure, not a placeholder success. Unload a model when you need to release its resources.

The release compatibility table lives in `docs/PROVIDER_COMPATIBILITY.md` when completed by the provider owner. Do not interpret planned catalog entries as individually tested models.

## Identifying trained models

YOLO best.pt model labels include date, hour/minute, the application host timezone, and a short model ID (for same-minute runs). New runs save their UTC completion time. Older models use the existing saved/registration timestamp, explicitly labeled; absent dates are never guessed from copied checkpoint files. The selected model name is repeated in activation confirmation. Names also appear in Train and Predict selected. Checkpoint files and activation IDs stay unchanged.
