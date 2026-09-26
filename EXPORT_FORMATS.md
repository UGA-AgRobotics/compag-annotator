# Export and import formats

Choose a training format in Export, select Reviewed dataset or Labeled objects (includes drafts), then Check export → Create dataset ZIP → Download dataset ZIP. The preview shows actual counts and skipped items. Image copies default OFF. Training ZIPs include README.md and image_index.json, and exclude the project database and Undo/Redo history. Full Native backup is a separate expandable section; it ignores dataset scope and review filters.

Choose scope (whole project, selected images, round or cumulative eligible data), reviewed-only selection and whether to copy images. Draft-inclusive exports must be identified as drafts. Inspect the conversion report before using a dataset. Exports do not modify accepted annotations or run providers/boundary recovery.

| Format | Supported purpose | Fidelity and review limits |
|---|---|---|
| Native project archive | Backup and restore into a new location | Preserves project identity, geometry, review history, classes, rounds and model metadata. Weights excluded by default. Verify linked-image inclusion before depending on portability. |
| COCO instances | Polygon/RLE instance exchange | Use RLE for exact raster masks, including holes and disconnected components. RLE alone does not imply a crowd annotation. External COCO does not preserve every application history field. |
| YOLO segmentation | Per-image polygons plus dataset YAML/class map | Strict conversion blocks unrepresentable topology. Explicit approximation requires a loss report. Box-only objects are not segmentation targets. |
| YOLO detection | Normalized bounding boxes | Mask shape, holes and components are not represented. This is a box export, not segmentation training data. |
| Pascal VOC | XML bounding boxes | Box-only scope; preserve documented coordinate convention. Fine mask detail is lost. |
| LabelMe | Supported polygon/rectangle shapes | Arbitrary shape types and native mask topology may be unsupported. Read grouping/flag and conversion reports. |
| PNG plus mapping metadata | Per-instance binary or supported class/instance-ID planes | Separate instance masks preserve overlaps. A single plane cannot encode overlapping instances without an explicit conflict policy. Use sufficient bit depth; IDs must not wrap. |

Native archives and exact COCO masks are the preferred paths when topology matters. YOLO's [documented segmentation format](https://docs.ultralytics.com/datasets/segment/) stores a class index and normalized polygon coordinates per object. It does not provide arbitrary hole/component semantics. The [COCO mask API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py) defines RLE, area and bounding-box operations.

A strict failure should identify affected objects. Repair the annotations, choose a faithful format, or deliberately approve an available approximate conversion after inspecting changed pixels/IoU. Never assume keeping only the largest contour is lossless. Training uses the same converter as export.

Imports retain source provenance and start unreviewed unless you explicitly attest the applicable scope. Check image dimensions, coordinate orientation and class mapping before acceptance. Use copies and relative image paths for portable exports; duplicate basenames are disambiguated by image identity.

Test evidence for each adapter belongs in `ACCEPTANCE_RESULTS.json`. A format listed here is the intended supported scope, not a claim that every variant has passed an external loader or round trip.
