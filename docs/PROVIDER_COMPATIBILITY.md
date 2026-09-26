# Provider compatibility and integration contract

The base measurements below are historical. For rc11 security pins and validation scope, see DEPENDENCY_SECURITY.md and ../ACCEPTANCE_RESULTS.json. Model sources/Torch pins are unchanged; dependency-overlay validation is not a fresh complete provider installation.

Base status at 2026-09-24: 59 provider unit/protocol tests passed, including execution
without a caller-supplied `PYTHONPATH`. Coordinator-controlled installed-wheel
acceptance passed SAM2.1 Small full-image GPU point/negative/box prompting and
CPU prompting, both selected runtime/checkpoint installs, two real three-epoch
YOLO segmentation training jobs, and fresh-checkpoint next-round inference.
The training acceptance budget is exhausted; no further training is authorized.
SAM3's repaired isolated runtime passes `pip check` and offline image-API imports
with CUDA hidden. Separately, coordinator-controlled installed-wheel acceptance
passed actual SAM3 GPU bfloat16 prompting with an authorized local checkpoint.
These automated checks do not establish native Ubuntu coverage or human UAT.
No checkpoint or private acceptance path is distributed in this repository.

The SAM3 checkpoint test passed strict state validation and positive/negative
point and box prompts on a full 3024 x 4032 image. Both calls returned nonempty
masks in full-image coordinates and the loaded checkpoint SHA-256 matched its
registration. The box call reused the image encoding. Observed end-to-end times
were 17.96 seconds for cold point prompting and 3.95 seconds for the warm box
call on the acceptance machine; these are observations, not performance or
accuracy guarantees. The checkpoint remained user-supplied and local-only, with
no checkpoint download. Compatibility is established for this tested image
checkpoint and pinned runtime, not every SAM3 checkpoint or device.

## Exact versions and authoritative sources

| Provider | Source revision / package | Python | Torch / torchvision | Runtime variants |
|---|---|---|---|---|
| SAM2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4`, SAM-2 1.0 | 3.12 | 2.7.1 / 0.22.1 | CPU, CUDA 12.8 |
| SAM3 image | `2345a4ad109ac29c569da749c91d84f10dc08c40`, sam3 0.1.0 | 3.12 | 2.10.0 / 0.25.0 | CUDA 12.8 only |
| YOLO | Ultralytics 8.4.162; release commit `91392240568a261d2cd6e37b1684cbfedf3cd1e5` | 3.12 | 2.7.1 / 0.22.1 | CPU, CUDA 12.8 |

Dependency locks are `locks/provider-{provider}-{cpu|cu128}.txt`, mirrored in
`src/compag_annotator/models/requirements/` for installed-wheel operation. They
include exact transitive versions and distribution SHA-256 hashes. The small
`provider-adapters.txt` lock supplies Shapely 2.1.2 and pycocotools 2.0.10 used by
the shared geometry API for repairs; these are also included directly in every
provider lock for fresh installation. The runtime probe imports the shared
geometry module so missing adapter dependencies fail during setup.
No optional environment inherits the core's packages.
Runtime installations preserve a resolved package receipt after `pip check`.

The SAM3 image lock deliberately omits Decord. The inspected Decord 0.6.0 release
wheel declares `cp36-cp36m-manylinux2010_x86_64` internally and failed Python 3.12
`pip check`, despite its installable filename. In the pinned SAM3 code, Decord is
an optional notebooks dependency and is imported lazily only in video-file or
video-frame dataset loading. The still-image builder, processor and interactive
predictor import without that decoder; no upstream source patch or wheel
retagging is applied. Video loading remains outside this application's scope.
See the [pinned video-only import](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/utils/sam2_utils.py)
and [upstream dependency declaration](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/pyproject.toml).

The upstream code/API was inspected at exact revisions:

- [SAM2 build and strict checkpoint loading](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/build_sam.py),
  [predictor](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py),
  [checkpoint URL source](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/checkpoints/download_ckpts.sh),
  [installation limitations](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/INSTALL.md).
- [SAM3 builder](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model_builder.py),
  [image interactivity](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_image.py),
  [processor](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/sam3/model/sam3_image_processor.py),
  [SAM License](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/LICENSE).
- [Ultralytics segmentation](https://docs.ultralytics.com/tasks/segment/),
  [dataset syntax](https://docs.ultralytics.com/datasets/segment/),
  [pinned model API](https://github.com/ultralytics/ultralytics/blob/91392240568a261d2cd6e37b1684cbfedf3cd1e5/ultralytics/engine/model.py),
  [pinned trainer/resume code](https://github.com/ultralytics/ultralytics/blob/91392240568a261d2cd6e37b1684cbfedf3cd1e5/ultralytics/engine/trainer.py),
  [official release asset metadata](https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0),
  [licensing](https://www.ultralytics.com/license).
- [Official Torch/torchvision version pairs](https://pytorch.org/get-started/previous-versions/).

The inspected source archive SHA-256 values (source code, not weights) were:

| Source archive | Measured SHA-256 |
|---|---|
| SAM2 pinned GitHub archive | `fe93082a71a885a427894b1eab76341768781b6b58a298a0717e03862097d137` |
| SAM3 pinned GitHub archive | `2aa67466a5e9c6552bf130df02168b243b4631c0eea3caec49397193a548efa0` |

SAM2 is Apache-2.0 with component notices; SAM3 uses its custom SAM License;
Ultralytics follows AGPL-3.0 or separately obtained Enterprise terms. These
runtime mechanisms do not resolve ownership or redistribution clearance.

## Catalog and weights

SAM2.1 Tiny, Small, Base Plus and Large each bind the matching upstream config.
Small is the recommended initial preset. Their official URLs are recorded in
`models/catalog.py`; upstream does not publish a digest there. Downloads record
a measured hash and source receipt, without claiming an official checksum.
Only the user's selected checkpoint is downloaded.

| YOLO checkpoint | Bytes | Official release SHA-256 |
|---|---:|---|
| YOLO26n-seg | 6,719,965 | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` |
| YOLO26s-seg | 23,467,933 | `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4` |
| YOLO11n-seg compatibility option | 6,182,636 | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |
| YOLO11s-seg compatibility option | 20,669,228 | `1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3` |

SAM3 has no checkpoint download URL in the catalog. A request to install SAM3
with `download_weights=True` fails before runtime installation/network access.
The user may choose a local `.pt` or a directory containing `sam3.pt`. The
architecture must be `sam3-image`; SAM3.1/multiplex/video are not interchangeable.
An inference child has outbound networking disabled and passes both the explicit
checkpoint path and `load_from_HF=False`, with instance interactivity enabled.
The tokenizer vocabulary is supplied by the pinned source package.

All local checkpoint registration hashes and inspects only the serialization
header/ZIP directory. It never imports Torch or unpickles a file. `trust=True`
is required before real loading, including local SAM3. A subprocess and a hash
are not a security sandbox for a malicious trusted artifact. Official consented
catalog downloads are registered as trusted official-source artifacts.

## Stable public Python API

`models.manager.ModelManager(data_dir: Path)` implements the integration contract
without signature deviations:

```python
catalog() -> list
status() -> dict
models() -> list
register(provider, path, *, architecture=None, trust=False, class_mapping=None) -> dict
install(provider, architecture, *, consent=False, download_weights=False,
        progress=None, cancel=None) -> dict
infer(model_id, image_path, *, points=None, labels=None, box=None, device="cpu",
      settings=None, progress=None, cancel=None) -> dict
train(model_id, dataset_yaml, output_dir, settings, *, progress=None, cancel=None) -> dict
unload() -> dict
doctor() -> dict
```

`install` returns `{provider, architecture, runtime, state, model, model_id?}`.
Successful checkpoint downloads are already registered: use `result['model']['id']`.
Runtime-only setup is available as:

```python
from compag_annotator.models.runtime import install_runtime
receipt = install_runtime(data_dir, "sam2", consent=True, flavor="cu128",
                          progress=print, cancel=lambda: False)
```

`install_runtime` is idempotent. It can add the small adapter dependency lock to
an existing matching runtime without reinstalling Torch or downloading weights.
The known initial-lock upgrade stages an immutable package overlay, switches its
Python site path atomically, runs the geometry/provider probe, and updates the
receipt only on success. On a failed probe the previous path and receipt are
restored. Unknown lock changes require explicit runtime replacement; they are
not silently upgraded. All installation/repair calls require consent.
For the failed initial SAM3 install, the same `install_runtime` call verifies
the environment prefix, Python version, exact upstream source URL and every
current locked package version before removing only optional `decord==0.6.0`.
It then requires `pip check`, geometry imports, builder/processor/interactivity
imports, and the tokenizer asset check to pass before writing the success receipt.
The receipt declares `still_image_interactivity`, omitted dependencies and
`upstream_source_modified=false`. Torch, source packages and checkpoints are
preserved. This repair passed in the actual acceptance runtime with CUDA hidden,
no weights loaded, and no GPU or training job. Installer commands strip inherited
`PYTHONPATH`/`PYTHONHOME` while preserving the shared `PIP_CACHE_DIR`, preventing
source-tree core metadata from contaminating optional-runtime package checks.
An interrupted install has no success receipt and can be retried. Installation
logs stream real subprocess output; download events contain actual byte counts.
Partial downloads are removed on interruption; retries restart the transfer.
Network reads have a 20-second timeout; cancellation stops between reads.

Optional environment settings:

| Setting | Meaning |
|---|---|
| `COMPAG_RUNTIME_PYTHON` | Explicit Python 3.12 bootstrap interpreter; otherwise the core's 3.12 interpreter or `python3.12` is used |
| `COMPAG_TORCH_FLAVOR` | `cu128` (default) or `cpu`; SAM3 requires `cu128` |
| `PIP_CACHE_DIR` | Shared pip wheel/download cache; does not change runtime or project locations |
| `COMPAG_SAM2_PYTHON`, `COMPAG_SAM3_PYTHON`, `COMPAG_YOLO_PYTHON` | Explicit optional external runtime for local testing; never auto-discovered from historical installations |

External runtimes are not modified by the installer. Their actual versions are
reported; a source revision is marked unverified unless its package provenance
matches. A normal standalone install needs none of these external overrides.

## Worker behavior, memory and coordinates

Each worker runs an explicitly chosen interpreter with `-I -B`, loading only
the installed application's provider code. Protocol version 1 exchanges JSON
lines with request IDs. Events are `progress`, `result`, or `error`; upstream
logs are redirected to stderr. Stale/wrong checkpoint/image bindings fail.
Workers do not assign project classes, human review, or image completeness.

YOLO inference uses a fresh process; training and its mandatory saved-checkpoint
inference probe use different processes. SAM uses a controlled persistent worker
for repeated clicks. One user-wide advisory lock serializes provider operations,
and the application job queue must also serialize GPU work. Switching providers
or training unloads the current worker. Cancellation terminates the entire child
process group; it preserves any already-written training checkpoint.

Images are EXIF-oriented RGB; original bytes are never edited. The default
image limit is 100 million pixels. Every inference returns full-image coverage,
canonical masks, derived exclusive-edge boxes, loaded checkpoint hash, image
hash, provider/runtime metadata, and timings. Masks preserve holes and disconnected
components; no largest-contour simplification is used.

SAM cache hits require equal image bytes/identity, model hash, preprocessing,
precision, device and exact decoded pixels. The cache retains one image and its
encoding; prompt-dependent state is not cached. The default 512 MiB cache budget
counts image pixels and encoded tensors. Exceeding it discards the cache after
the current result. `settings['cache_image_bytes']` changes the budget explicitly.
SAM3 clones per-prompt state. No quality/accuracy parity with the research path
is claimed.

SAM2 uses the exact pinned `set_image(image)` signature (there is no
`image_format` keyword in that revision), `build_sam2(..., apply_postprocessing=False)` and
`SAM2ImagePredictor(..., max_hole_area=0, max_sprinkle_area=0)`. Its optional CUDA
connected-component extension is not compiled. This is recorded in results;
there is no claim of equivalent hole/sprinkle cleanup. The coordinator's real
CPU checkpoint test passed on the full-size acceptance image; it does not
establish timing on other hardware.

SAM3's upstream image loader only warns about missing keys. The adapter keeps
its detector/tracker translation but validates every required key and shape,
then loads strictly. Missing, unexpected, random or incompatible image weights
fail rather than reporting readiness. This strict path is unit-tested with
explicit doubles and separately passed the coordinator's real local-checkpoint
load and full-image GPU point/box acceptance described above.

YOLO preserves `retina_masks=True` output in original coordinates. Optional
`tile_size`/`overlap` settings map every crop mask back to the full image and use
the shared exact-mask, class-aware duplicate function. Nearby instances are
never merged by invented unions. No automatic recovery is imported or called.

Device strings are `cpu`, `cuda`, `cuda:N`, or a single numeric GPU index.
`float32` is default; explicit `precision='float16'` is supported for CUDA YOLO,
and SAM also supports explicit CUDA `bfloat16`. CPU requires float32. Insufficient
memory or unsupported devices fail visibly without hidden precision/resolution
fallback. `reserve_free_bytes` defaults to 256 MiB before provider work.
When driver-free CUDA memory falls below that reserve, the worker releases its
unused PyTorch allocator cache and measures the selected device again. This
avoids rejecting a later tile because an earlier tile populated the idle cache.
If the rechecked memory is still too low, the error reports free and required
MiB. Sufficient free memory leaves the cache in place. Model weights, precision,
point-grid density, batch size, thresholds and active tensors are unchanged.

## Training, saved-checkpoint validation and resume

The coordinator supplies the immutable snapshot produced by the shared YOLO
converter. Provider preflight rejects network/download-hook YAML, missing or
overlapping train/validation splits, duplicate image bytes across splits,
noncontiguous class indices, box-only labels, invalid normalized polygons, and
implicit negative images without a label file. It does not manufacture reviews
or create an alternative segmentation converter.

Core training controls are `epochs`, `imgsz` (multiple of 32), `batch` (positive
integer), `device`, and `seed`. Supported advanced options are enumerated in
`providers.yolo.train_settings`. `class_mapping`, `class_version`,
`snapshot_sha256`, `round_id`, `dataset_id`, and `qa_smoke` are metadata and are
never forwarded as Ultralytics options. `worker_timeout` defaults to 24 hours
for training and 900 seconds for inference. The coordinator's acceptance budget
is separate from production settings.

Automatic batch sizing and automatic mixed-precision probing are disabled;
the latter can cause upstream to download another model. `amp=True` fails
explicitly. External/cloud integration callbacks and telemetry are disabled.
Matplotlib's bundled DejaVu font is copied into the private provider settings
directory to avoid upstream Arial downloads. Validation remains enabled with a
distinct split; no final test split is used for checkpoint selection.

`full_instance=True` disables cutting geometric augmentations in this prepared
full-image preset. Otherwise ordinary upstream augmentations apply; no universal
complete-instance guarantee is made.

The returned training result contains actual `checkpoint_path`,
`checkpoint_sha256`, `model_id`, `task='segmentation'`, `class_names`,
`class_mapping`, `metrics`, `selection_basis`, `last_checkpoint`, and
`load_probe`. Success requires a second process to load the saved checkpoint,
perform inference on a separate validation image and return its matching hash
and names. No model is registered as successful when this probe fails. A probe
with zero detections can prove execution, not model quality. Activation and
rollback remain coordinator responsibilities.

Resume is explicit and isolated:

1. Register and trust the saved `last.pt` that still contains optimizer/epoch state.
2. Pass `settings['resume_checkpoint']` with that registered path, the original
   immutable snapshot, the original total epoch count, and a **new job output**.
3. The manager makes an exclusive byte-identical `resume-input/last.pt` copy in
   the new output directory. It verifies the source and copy hashes.
4. The pinned trainer's `check_resume` restores checkpoint arguments. The adapter
   rebinds `project`, `name`, `save_dir`, and `data` before BaseTrainer creates
   output directories. It passes the copied checkpoint path, not boolean resume,
   avoiding upstream's fallback from a non-resumable artifact to new training.
5. Completed/stripped checkpoints, missing optimizer state, a changed epoch
   schedule, or unsupported mixed-precision resume fail with an instruction to
   start a new fine-tuning job. The prior run and source checkpoint are preserved.

Resume isolation and rejection paths are unit-tested. No additional actual
training job was run by the provider owner; real training and resume acceptance
are coordinator-controlled and must stay within the task's two-job cap.

## Reuse and verification scope

The preserved A worker and SAM-assist source were read, not modified or imported.
Finite original-coordinate prompt validation, exact-image cache binding and
prompt-state separation were refactored as concepts into the provider modules.
Hardcoded research paths, labels, Slurm allocation checks, crop limits, binary
classifiers and automatic recovery were removed from this implementation.
The ordinary tiling/RLE/duplicate APIs are the geometry owner's shared module.

`tests/test_providers.py` covers lazy imports, registry identities/trust, prompt
validation, full-image masks with holes/components, EXIF, download consent and
integrity failures/cancellation, cache identity, local snapshot validation,
process cancellation, strict SAM3 state validation, worker network denial,
stale checkpoint rejection, failed fresh checkpoint probe, and resume isolation.
All artificial checkpoint/state fixtures are explicitly marked as test doubles.
They never count as actual SAM2/SAM3/YOLO inference or training evidence.

Integration needs: package `models/requirements/*.txt`, run the small adapter
dependency repair in preexisting runtimes, retain serial job scheduling, preserve
immutable training snapshot provenance, and record each real-provider/device
test separately. The selected SAM2, YOLO and local SAM3 acceptance checks above
have passed. Actual resume training, other catalog checkpoints and untested
device combinations remain outside the completed acceptance evidence. The final
wheel must include the repaired runtime installer and mirrored dependency locks;
the SAM3 image-runtime repair did not change the inference API used by the
successful installed-wheel test.

The coordinator's first installed-wheel SAM2 GPU attempt exposed an unsupported
`image_format` keyword before a mask was returned. That call was corrected and
an exact-signature regression test added. That failed attempt is not counted as
real-provider success. The corrected installed-wheel retry subsequently passed
GPU/CPU prompting and verified the checkpoint hash and full-image coverage.

## Addendum 01 automatic generation and actual source tiles

The SAM2 adapter now exposes the pinned official `SAM2AutomaticMaskGenerator.generate` over each actual source crop, without YOLO or predefined classes. Its 108 provider software tests pass. The installed application additionally completed one full 3024×4032 natural image with 88 actual 512×512 crops and one with 84 actual 640×384 crops, producing 420 and 421 canonical proposals. Every returned crop/encoder transform, checkpoint binding and exact crop-to-full-image mask mapping was verified. Initial observed whole-job times were 107.45 and 110.61 seconds; these functional checks establish neither accuracy nor comparative speed. Final corrected-candidate timing and source bindings are in `ADDENDUM_MEASURED_RESULTS.json`.

The source crop size is separate from the SAM2 1024×1024 encoder resize and YOLO `imgsz`. New `settings.tiling` requests use the common rectangular planner, including an explicitly empty default config, and preserve nonidentical overlapping/nested proposals. Historical `tile_size`/`overlap` jobs retain their previous duplicate policy. Application-wide automatic union/recovery is never part of generation.

SAM3 automatic generation is **UNSUPPORTED** by this validated adapter; the UI explains that limit and offers SAM2. The previously validated local-weight SAM3 positive/negative-point and box capabilities remain separate. No SAM3 checkpoint was downloaded.
