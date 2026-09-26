# Source reuse and replacement map

All historical references below are relative to the approved baseline source root. Historical source/data remain outside the public tree. No private path, fixed image identity or historical model binding is needed to install the new application.

| Application area | Relative source candidates | Relationship / verification |
|---|---|---|
| Full-image review concepts | `baseline_src/revision_experiments/sam3_ignore_v2/web.py`, `static/` | Class-agnostic UI/API adaptation; UI owner must record exact reused code separately before claiming direct reuse. |
| Annotation review/history | `baseline_src/revision_experiments/sam3_ignore_v2/registry.py`, `corrections.py`, `per_image_review.py` | Transaction/review concepts carried into a new stable-class schema. Original binary policy is preserved separately, not relabeled. |
| SAM prompt/worker lifecycle | `baseline_src/revision_experiments/sam3_ignore_v2/sam_assist.py`, `sam_assist_batch.py`, `sam_worker.py` | Provider isolation and prompt-validation concepts; exact implementation evidence belongs to provider owner. |
| Mask/RLE/tiling | `baseline_src/revision_experiments/sam3_ignore_v2/original_tiling.py`, `overlap_*.py`, `recovery.py` | Geometry owner records exact extraction/adaptation in `docs/SOURCE_REUSE_GEOMETRY.md`; do not infer verbatim reuse from this summary. |
| Flexible rounds | `baseline_src/revision_experiments/sam3_ignore_v2/al_lifecycle.py` | Fixed historical schedule replaced with arbitrary eligible image counts and user-selected round count. |
| Universal training | `baseline_src/revision_experiments/sam3_image_al/training.py`, `sam3_ignore_v2/decision_policy.py` | Historical binary feature/classifier path is not a dependency. New cumulative YOLO segmentation snapshots and activation. |
| Packaging / CLI / installer / release scanner / offline help | No baseline packaging existed in the audited inventory | New implementation; tests in `tests/test_packaging.py`. No historical source was copied into these files. |

New packaging does not depend on private runtimes, old model bundles, research manifests or historical environment listings. Baseline source rights still need owner approval. Detailed reuse evidence from each module owner supersedes “candidate” relationships above; final coordinator review must close any remaining direct-reuse attribution gaps.
