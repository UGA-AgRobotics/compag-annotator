# Architecture

COMPAG Annotator uses a small Python/FastAPI core and a bundled browser interface. This keeps the base install independent of Torch, checkpoints, Node and hosted services. The browser talks only to the local process; static assets and offline help ship in the wheel.

## Main boundaries

| Area | Responsibility |
|---|---|
| CLI / application | Loopback launch, random per-launch capability token, QA provenance, safe startup |
| core / storage | Stable class/object identities, revisions, SQLite project state, files, image identity and backups |
| geometry / formats | Original-image coordinates, exact masks, strict conversion, fidelity reports |
| models / providers | Isolated runtime setup, checkpoint identity, trust, device and lifecycle |
| jobs / training | Durable background work, immutable reviewed snapshots, training and activation |
| boundary | Scoped, explicitly authorized training-preparation or automatic-mask-annotation suggestions, followed by review |
| web | Session, API validation, bundled editor and user workflows |
| help / assets | Offline help and application icon, independent of provider packages |

The canonical geometry distinguishes polygons, boxes and exact COCO-style uncompressed RLE masks. Image orientation is normalized for coordinates while original bytes remain preserved. Large masks are requested separately from scene metadata. Export/training use the same conversion rules.

Stable project class IDs differ from contiguous model class indices. Models and training snapshots save their mapping; renaming/reordering display labels does not rewrite learned classes. Inference produces unreviewed proposals. Human and automated QA actors are distinct.

Model runtimes use subprocesses to contain dependency conflicts and manage resources. A subprocess is not a security sandbox. Runtime locks and one controlled GPU owner avoid unbounded CUDA contexts. Core CLI help/version/doctor must not import Torch or load weights. SAM3 always uses local user-supplied weights with upstream auto-download disabled.

Project changes are revision-checked and persisted atomically. Backups copy a coherent native project and are written outside the live project. Training snapshots bind reviewed annotation versions; activation binds a real load-probed checkpoint. Heavy recovery is guarded centrally and cannot run from ordinary import, edit, prompting, inference, round or export paths.

## Addendum 01 integration

Implementation and real evidence are being integrated; [ADDENDUM_01_IMPLEMENTATION.md](ADDENDUM_01_IMPLEMENTATION.md) maps required behavior to modules, API routes and tests without treating this architecture description as validation.

The shared geometry tiler plans real source crops in canonical full-image coordinates. New-project defaults are 512×512 with 25% overlap per axis. Whole-image mode, three square presets and rectangular/non-multiple-of-32 custom sizes share integer validation, resolved overlap/stride, deterministic unique origins and a far-edge-aligned final crop. A smaller image has a smaller valid crop; no external padding is required. Manifests bind image hash, plan hash, crop IDs/boxes/valid regions and provider transforms. Width × height in the UI is distinguished from height × width in arrays. SAM AMG and tiled YOLO use the same plan; encoder resizing and optional training crops remain independent.

The SAM2 worker uses real `SAM2AutomaticMaskGenerator.generate` on each actual crop, returning crop-local exact masks and genuine provider metadata. The core maps these into canonical coordinates and saves per-tile receipts in a durable generation layer. Stable IDs, request idempotency and unfinished-tile resume preserve partial completion without duplicate append. New generations or replacement of explicitly selected unreviewed layers cannot overwrite accepted/edited work or revive rejected/superseded objects. Provider quality/NMS/crop filters are recorded separately from application grouping and optional heavy recovery. Nonidentical overlapping masks are not automatically unioned or erased by box overlap. SAM3 automatic capability is unavailable until independently implemented/tested; prompting remains separate.

Zero-class projects store unassigned proposals as `class_id=null`. The editor separates Select, Paint class, SAM prompt and Edit geometry; lazy exact-mask hit testing and object-list controls support overlapping/multiple selections without decoding every full-image mask on every pointer move. Bulk assignment preserves instance identities and does not confer review. Exact merge validates same-image live parent revisions, computes canonical pixel OR and area/box, resolves class conflicts explicitly, and atomically creates a draft child while superseding parents. History supports undo/redo across restart. Merge calls no provider; subsequent refinement is a separate reviewed change. Active exports and eligible snapshots exclude superseded parents, drafts, rejected/deleted and unassigned objects.

The central boundary policy permits only `training_preparation` and `automatic_mask_annotation`, with recovery disabled by default and explicit authorization for the job's actual kind/scope. The worker must reject a forged client phase or reused checkbox consent. Automatic annotation can request a post-generation check or a later selected-proposal/layer check. A physical image edge is not an internal seam. Contextual model calls/neighbor checks stage proposed replacements with stale-revision protection; they never silently apply changes or grant completeness. Routine validation/seam flags remain cheap and always available. A failed check preserves the original proposal. Training and annotation consent never enable each other or unrelated later work.

The canonical project state feeds existing rounds, native backup, exact COCO RLE and loss-aware YOLO export/training. New UI training uses 512px source tiles with original-image split lineage; whole-image API compatibility and immutable old snapshots remain. Tile-trained model records bind the inference recipe. Contextual boundary recovery is a separate opt-in review stage. Masks retain holes/components; strict polygon-only conversion blocks unsupported merged topology instead of inventing bridges or splitting instance identity. One controlled provider owner schedules GPU work; cancellation and settings changes invalidate incompatible prompt/crop/model caches. Limits must be visible; resource failure must not silently alter a recorded recipe.

## Security boundary

The CLI binds only `127.0.0.1`; no remote-bind switch is provided. HTTP mutations require the local session token, with Host/Origin validation. Tokens are not put in URLs or printed. Server-side file selection needs explicit user approval. Archive, XML/YAML, filename and image limits are enforced at ingress. Diagnostics redact personal paths and secrets.

## Installation and release

The user installer creates immutable per-wheel environments and switches a `current` link after validation. The previous application environment remains available for rollback. Runtime archives require a verified SHA-256; there is no downloaded shell bootstrap. Projects/models live separately and are preserved by removal.

The public staging tool uses an allowlist and actual content checks, reads wheel/sdist entries without unsafe extraction, and compares packaged files with source hashes. It cannot certify ownership or classify every possible private sentence. Human rights/privacy review is still mandatory before publication. No Git remote or upload is created.
