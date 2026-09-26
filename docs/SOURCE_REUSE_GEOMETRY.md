# Geometry and exchange implementation / integration notes

## Scope and preservation

This component owns `src/compag_annotator/geometry/`, `src/compag_annotator/formats/`,
`tests/test_geometry.py`, and `tests/test_formats.py`. Native project archives,
review decisions, storage, training orchestration, and provider runtimes belong
to the application core and their respective owners.

The integration contract, complete v1 master task, supplied baseline audit,
baseline README/reference/external-assets manifest, and relevant original code
were read. The preserved baseline's complete `SHA256SUMS` verification passed
before and after implementation. No preserved file was imported as a running
module or modified. No scientific fixtures, private paths, learned constants,
model weights, GPU jobs, commits, remotes, or publication were introduced.

| Baseline source relative to A_BASELINE | Adaptation |
| --- | --- |
| `baseline_src/revision_experiments/sam3_ignore_v2/original_tiling.py`, `original_axis_positions` | Reused the deterministic stride progression and explicit final far-edge position. New `tile_boxes` accepts arbitrary positive tile sizes and overlaps, clips small images, and returns original-coordinate row-major boxes. Fixed research settings, name rules, padding, and plan identity were not copied. |
| `baseline_src/revision_experiments/sam3_image_al/metrics.py`, `encode_mask` | Reused the NumPy transition-index / difference algorithm for background-first column-major runs. New `encode_rle` preserves the full image rather than tight cropping. It supports empty binary masks as a codec operation; empty annotation instances remain invalid. |
| Same file, `decode_mask` and `_intersection` | Adapted integer/count validation, exclusive pixel boxes, and cropped mask intersection. New decoder accepts full-image canonical RLE and independently validates official compressed RLE before allocating. Class-aware NMS retains original proposals and compares cropped exact masks without unions. |
| Research pipeline, recovery and overlap imports | Audited as context, excluded from public dependency paths. No scientific policy, heavy recovery, GPU overlap cache, or research import is used. |
| Common annotation exchange | New adapters rather than importing research exporters. All use the same canonical validation and rasterization functions. |

Audited source identities:

- `original_tiling.py`: SHA-256 `84ebff4c3c367e8362406796146dd026e029f3ea986ba61cf00d441c261b9ac3`.
- `sam3_image_al/metrics.py`: SHA-256 `2aa1fbafd24859251eba602e0005a6e53fde9929892f901f1769b3f7520051ea`.

These are provenance identities, not a statement that the unlicensed supplied
snapshot is cleared for public redistribution. The coordinator's rights gate
continues to apply.

## Geometry contract

- Original oriented image coordinates; `(x,y)` describes pixel edges. Bounds
  include `width`/`height` at the exclusive right/bottom edges. Boxes are
  zero-based exclusive `xyxy`. Authored polygon vertices remain vectors.
- Polygons rasterize with the actual pycocotools COCO polygon implementation.
  Display libraries may rasterize edge pixels differently; the canonical
  contract is COCO, not browser fill, PIL draw, or OpenCV fillPoly.
- Canonical masks are full-image `{size:[height,width],counts:[...]}` with
  background-first, column-major COCO runs. Holes and disconnected components
  are preserved exactly. `geometry_mask` accepts this representation directly
  for brush edits and returns a boolean array.
- The codec also accepts official compressed COCO counts in bytes or ASCII;
  signed differential runs, length, total pixels, truncation and character
  bounds are checked in Python before decoding. Canonical validation returns
  fresh uncompressed data and never repairs geometry silently.
- Allocation guard: at most 100,000,000 pixels; at most 1,000,000 authored polygon
  vertices. Polygonization additionally permits at most 100,000 horizontal
  foreground runs. More complex masks remain supported by RLE and PNG; the
  polygon converter fails with an actionable error instead of an unbounded
  Shapely allocation. No approximate reduction is automatic.
- `tile_boxes(width,height,tile_size=1024,overlap=128)` covers every pixel;
  small axes are clipped, with no fake full-size crop or duplicate far edge.
- `deduplicate` uses exact mask IoU, class identity and stable score ordering.
  Different unassigned model classes remain distinct; entirely unknown classes
  are conservatively retained. Bounding-box overlap alone never suppresses an
  object. Source proposals and masks are unchanged.

### Shared YOLO conversion

`polygon_for_yolo(geometry,width,height,allow_lossy=False)` returns **original
pixel points**, never normalized points, plus a JSON-serializable report.
Training/export normalize each x by width and each y by height. Fields include
`lossy`, `changed_pixels`, `added_pixels`, `removed_pixels`, `original_area`,
`converted_area`, `iou`, `components`, `holes`, `method`, `allow_lossy` and
`rasterization: "coco"`.

Strict mask conversion requires a single simple pixel-edge polygon, no holes,
and an exact COCO raster round trip. Four-connected components that meet only
at a corner remain separate. With explicit consent, one component's holes are
filled, or the convex hull of **all** components is used. Conversion never
retains only the largest component or splits one instance into several labels.
Box-only objects always fail, including with loss consent. Tiny vector shapes
with no foreground pixels also fail. No native geometry is modified.

**Additional training integration requirement:** call
`validate_yolo_rows(lines)` on every image's fully serialized segmentation rows
before writing the labels. Ultralytics `verify_image_label` deduplicates by
class plus float32 bounding box, which can silently remove distinct polygons
with identical bounds. The shared guard blocks that case and names the rows;
the exchange exporter already calls it. The coordinator's direct per-instance
converter route must also call it. This adds a guard without changing the
agreed `polygon_for_yolo` return type.

Reported pixel fidelity uses original-resolution COCO rasterization. Actual
Ultralytics masks use OpenCV and resizing/resampling; downstream training
rasterization is not claimed to be pixel-identical to the native mask.

## Exchange capabilities and losses

| Format | Implemented import/export | Exact subset / explicit limits |
| --- | --- | --- |
| COCO | Images, categories, annotations; vector polygons; compressed/uncompressed RLE; box-only rows; arbitrary external IDs mapped to stable project classes | Masks retain holes/components with exact RLE. RLE does not imply crowd: default `iscrowd=0`; explicit imported crowd flags survive. Area/segmentation boxes are calculated with pycocotools. Multiple polygon parts import as one union mask, not multiple objects. Stale area is reported and recomputed. |
| YOLO segmentation | Per-image TXT, `dataset.yaml`, split image lists, `class_map.json`; explicit normalized imports | Same audited converter as training. Strict topology and loader dedup guards. Empty label files support negative images. Dataset YAML does not create a train/validation overlap. A missing validation split is reported, not fabricated. |
| YOLO detection | Five-column normalized box TXT and dataset/class mapping | Exact boxes within floating precision. Segmentation-to-box conversion requires `allow_lossy=True` and quantifies pixel change. Never accepted as segmentation rows. |
| Pascal VOC XML | Polygon/mask projection with consent; box import/export; class names and difficult/truncated flags | One-based inclusive integer XML coordinates, translated to zero-based exclusive native boxes. Segmentation and fractional box conversion require explicit loss consent. Class names must be unique; XML is safely escaped and parsed with defusedxml. |
| LabelMe JSON | Polygon and rectangle shapes, class labels, group IDs and shape flags; grouped disconnected components | Separate components share a numeric group ID and import as one mask. Holes require consent to fill. Unsupported circle/line/point shapes fail unless explicitly opted into skipping, with a report. Per-shape flags from grouped imports remain in provenance; editing/converting the group can merge flags. Authored vectors follow the canonical COCO raster contract, not LabelMe's drawing rasterizer. |
| PNG `per_instance` | One binary 8-bit PNG per instance plus `mapping.json` | Exact canonical raster pixels, including overlaps, holes and components. Vector authoring coordinates are not represented; native backups preserve those. |
| PNG `semantic` / `class` | One uint16 class-value map per image plus metadata | At most 65,535 classes plus zero background. Multiple instances of one class require explicit `allow_lossy` to merge identities. Every overlap needs `conflict_policy='first'` or `'last'`; default `error` blocks it. |
| PNG `instance_id` / `instance` | One uint16 instance-value map per image plus metadata | At most 65,535 instances per image; no wrap at 255. Overlap requires explicit first/last policy and reports removed pixels per affected object. Fully occluded records import with a visible loss report, not a fabricated empty annotation. |
| Native | Core-owned project backup/restore | Required for IDs/history/review/rounds/models and lossless original vector/raster storage. Not implemented a second time in formats. |

Exports write a `manifest.json` with relative filenames, explicit class-index
mapping, reviewed/draft scope, conversion reports, warnings and counts. Image
copies are unique-ID-named PNG derivatives using oriented, lossless 8-bit display
pixels; private source paths never enter the exchange metadata. Original bytes
remain the responsibility of native backups. Callers choose the annotation
scope: these adapters do not silently discard drafts or infer training readiness.

`export_annotations` accepts a missing or empty destination only. Work is staged
and promoted atomically; failures leave no partial output and existing files are
never overwritten. `include_images=False` retains relative intended filenames
without reading source images. Copies reject multipage/high-bit-depth input;
the core should supply its supported oriented derivative.

Imports return `{annotations,classes,loss_report}`; `classes` is the complete
updated class list. Imported annotations always start as `proposal`, revision
zero, `human_verified=False`, `review_actor=None`, with import provenance. No
exchange field can manufacture human review. External IDs stay in `source`;
annotation IDs are new to avoid colliding with existing project history.

Input may be a format file or an unpacked package directory. Archive validation
and extraction remain the coordinator's responsibility. Paths in packages must
be relative and traversal/symlink-free; XML DTD/entities and executable YAML are
rejected. JSON/text files are bounded to 64 MiB. YAML download directives are
never executed. Ambiguous duplicate basenames fail unless explicitly mapped.

Import options:

- `image_mapping`: external image ID/name to target image ID.
- `class_mapping`: a JSON object with string source keys and existing target
  project class IDs as values. Selection priority is external category/index/
  value first, source stable ID second when present, and source name third.
  COCO uses the category `id` or `compag_id`; YOLO uses its zero-based model
  index or `class_map.json` `class_id`; PNG uses mapping class `value` or `id`;
  VOC/LabelMe use the source class name. Omit an entry for automatic matching.
  Empty/unassigned/skip target IDs are not supported. Multiple sources may map
  to the same target class. Bare YOLO TXT imports also honor explicit mappings.
- `create_classes=False`: block unknown classes instead of adding them.
- `class_names`: explicit ordered YOLO names when no YAML/map is available.
- `allow_lossy=True`: explicit supported LabelMe shape skipping.
- Standalone binary PNG: `image_id` and `class_id`. Multivalue PNG requires the
  documented mapping metadata, preventing guessed class/instance assignment.

The UI sends these options as a JSON string in the multipart `options` field
alongside `file` and `format` to `POST /api/projects/{pid}/imports`. When no
explicit class mapping applies, source stable identity is matched first, then
an exact class name; otherwise a new class is created only if `create_classes`
is true (the default). Ambiguous names and invalid target IDs fail. With no YOLO
class table or `class_names`, indices refer to the supplied project's class
list order; the UI should provide explicit names/mappings for external data.

Export has **no class-mapping option**. It emits the document's classes sorted
by `order`, assigning contiguous zero-based indices in `manifest.class_mapping`
and YOLO output (COCO categories and PNG class values are one-based). Import
dropdown values must be actual existing project IDs, not those export indices.

## Verification and dependency handoff

Core dependencies are already in the integration contract: NumPy, Pillow,
Shapely, pycocotools, PyYAML and defusedxml. No additional core dependency,
Torch import, private research package or model runtime is required.

Focused tests use Python 3.12.7, NumPy 2.5.3, Pillow 12.3.0, Shapely 2.1.2,
pycocotools 2.0.11, PyYAML 6.0.3, defusedxml 0.7.1, and pytest 9.1.1. The real
pycocotools extension, `COCO.annToMask`, area/bbox operations and compressed
encoding are exercised; these tests are not mocks. All 512 binary 3x3 masks
are checked against the official encoder, alongside asymmetric/random/full/
empty rasters, topology, malformed RLE, full-image masks larger than one tile,
class mappings, overlap policies, 300 PNG IDs, XML/YAML/path protections,
EXIF copies and atomic/no-clobber behavior.

Updated focused result with the supplied provider runtime: **143 passed, zero
skipped**, including the real Ultralytics loader and class-mapping contract tests
for every mandatory format. Geometry implementation was unchanged during this
follow-up; fixes are confined to format mapping and tests.

Run:

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_geometry.py tests/test_formats.py
```

The provider-loader test is gated on
`COMPAG_YOLO_TEST_PYTHON`, an explicitly supplied installed Ultralytics runtime.
It calls actual `verify_image_label`, with no model loading, weights or training.
The supplied runtime passed with Python 3.12.7, Ultralytics 8.4.162, Torch
2.7.1+cu128, Torchvision 0.22.1+cu128 and NumPy 1.26.4. The test disables GPU
visibility and asserts CUDA is uninitialized before and after label loading.
The CUDA-enabled package build is not a GPU execution claim. Without the
variable this test is **skipped**, not claimed passed.

The loader test prepares geometry in the core test environment and validates
labels in the provider subprocess. It does not prove provider-runtime geometry
imports. A separate probe found pycocotools and Shapely missing there; the provider owner
is adding/probing the pinned geometry dependencies. This is separate from the
successful actual label-loader result. Integration/browser/human UAT and any
real-provider/GPU claims belong to their owners.

Upstream pycocotools currently emits a NumPy `__array__(copy=...)` deprecation
warning during decode; masks still pass exact equality checks. No warning was
hidden by production code.

## Primary sources checked

- [Official COCO mask API](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/mask.py): array ordering, RLE, polygons, boxes and crowd semantics.
- [Official COCO C implementation](https://raw.githubusercontent.com/cocodataset/cocoapi/master/common/maskApi.c): signed differential compressed RLE validation.
- [Ultralytics segmentation dataset specification](https://docs.ultralytics.com/datasets/segment/): one polygon row per instance, normalized coordinates and class YAML.
- [Ultralytics dataset loader](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py): real label validation, float32 bbox deduplication, YAML path behavior and downstream rasterization distinctions.
- [Official LabelMe example](https://github.com/wkentaro/labelme/blob/main/examples/instance_segmentation/data_annotated/2011_000003.json): shapes, labels, groups and flags.

Moving documentation URLs are references, not runtime installation pins. The
provider owner pins Ultralytics separately; no rolling source is installed by
these adapters.
