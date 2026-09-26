# Training annotation exports — 1.0.0rc10

Open Export, choose a training format and annotation selection, then click **Check export**. Review actual image, object, class and split counts, including skipped objects. Click **Create dataset ZIP**, then **Download dataset ZIP** on the same page. Jobs also retains the download.

Formats: COCO instance JSON/RLE; YOLO segmentation polygons; YOLO detection boxes; PNG semantic masks, separate binary instance masks or instance-ID planes; Pascal VOC XML; LabelMe JSON. Each option explains the task and files it produces. COCO is the default for exact SAM raster masks.

**Reviewed dataset** includes accepted objects only on images explicitly marked reviewed. **Labeled objects** also includes current labeled drafts and proposals, without accepting them. Unassigned or archived-class objects, rejected/superseded objects, removed images and images marked excluded are omitted with counts. Incomplete images with no labeled objects are never exported as background-only training images. Draft exports are marked as drafts and require completeness review before training.

Image copies default OFF for a small annotation-only ZIP. Enable copies for a portable dataset: lossless oriented PNG images match the annotation coordinates. image_index.json maps original names to stable exported names. Every dataset includes README.md, mapping/provenance and export_summary.json. Dataset roles are preserved; no automatic train/validation/test splitting occurs. Missing validation and mixed image groups are reported.

Bounding-box and semantic identity conversions require the visible explicit option. Approximate polygon conversion is OFF by default. Single-plane overlaps block export unless the user selects a conflict policy. Strict geometry conversion errors remain visible; COCO or separate-instance PNG avoids flat-polygon restrictions.

**Project backup — continue editing later** is a separate expandable section. It retains the entire project and its history; dataset selection controls do not apply to it. Training datasets contain no project database, Undo/Redo history or model weights. Existing projects, masks, settings and model runtimes are unchanged. No AI or training is invoked by export.
