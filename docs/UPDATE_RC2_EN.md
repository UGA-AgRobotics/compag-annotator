# 1.0.0rc2: devices, progress, night theme and SAM settings

This incremental update leaves projects, annotations, checkpoints and existing job recipes intact. It does not start training or publish anything. Old partial generation jobs retain their saved tiles; the existing implementation-hash guard requires an explicit new job after a provider-code update rather than silently mixing versions.

## Devices

Generate masks, Train, model testing, prediction and the editor's **SAM settings** use detected CPU/GPU choices. Discovery runs in the selected model provider's interpreter without loading checkpoints or unloading an active model. Refresh devices rechecks availability; the list is cached for at most 30 seconds. GPU indices respect CUDA_VISIBLE_DEVICES. Choose the named GPU to submit `cuda:N`; CPU submits `cpu`. There is no automatic CPU fallback when a remembered GPU is absent. Training checks the YOLO environment, not the SAM environment.

A GPU needs a working NVIDIA driver, WSL GPU access where applicable, and the CUDA provider runtime. A CPU-only Torch runtime cannot offer CUDA even if hardware exists. A missing runtime displays setup guidance. SAM3's pinned image adapter requires CUDA: CPU remains visible but unavailable. Its weights remain local-only. Each heavy model operation still follows the existing serial GPU policy.

## Progress and theme

Jobs displays the downloaded byte percentage when the server provides the total; otherwise it shows received bytes and indeterminate progress. Dependency downloads show their individual percentages. Runtime setup shows completed steps out of eight, explicitly **not a time estimate**. Validation must succeed before the runtime reaches 100%. A failed job is still shown as failed even if its download previously completed.

Choose **Theme → Dark / night** in the header. This changes interface colors only, not photograph pixels, annotation class colors or exports. Theme and last selected device are remembered in this browser for this local address; a different browser/port has separate preferences.

## SAM2 paper defaults

Source: user-supplied `COMPAG_D_26_02120_MAJOR_REVISION.pdf`, page 5 (AMG table), page 11 (Table 2), page 20 (Supplementary Table S1).

| Automatic-generation parameter | New-job default |
| --- | --- |
| points_per_side | 64 |
| points_per_batch | 512 |
| pred_iou_thresh | 0.80 |
| stability_score_thresh | 0.88 |
| crop_n_layers | 0 |
| crop_n_points_downscale_factor | 2 (inactive with 0 internal crop layers) |
| crop_overlap_ratio | 0.40 (inactive with 0 internal crop layers) |
| Largest-mask exclusion | Off: largest masks retained |

The paper used **SAM2.1 Hiera Large**. This update does not replace the selected checkpoint; Large is available in Models & AI. The app defaults to the paper's AMG numeric recipe, not its entire research pipeline. Research XGBoost gates, post-merge 500-mask cap, multiscale merging and preprocessing are not introduced. Existing external tiling defaults remain unchanged (512×512, 25% overlap); the paper's 512 stride is a separate explicit user choice. Internal crops and topology cleanup remain disabled. Heavy boundary recovery remains off unless separately opted in.

Change sampling density and the advanced batch, quality, stability offset, NMS, logit threshold, alternative masks and precision controls before generation. Batch 512 is memory intensive, particularly for whole-image output; reduce it explicitly on smaller GPUs or CPU. A lower batch changes scheduling, while fewer points changes sampling. CPU requires Float32. Box NMS 0.7, stability offset 1, logit threshold 0, alternatives on and Float32 are retained adapter defaults; the paper does not specify them. No accuracy equivalence to the paper is claimed.

## SAM2 / SAM3 point and box assistance

In Annotate, select the model, then **SAM settings**. Both support editable logit threshold (default 0), alternative masks (on) and precision (Float32). These values reach the actual predictor. Changing threshold updates decoder binarization while preserving reusable image embeddings. Changing settings invalidates an older preview. The selected device is used for the preview.

SAM3 uses its pinned real `predict_inst` / `SAM3InteractiveImagePredictor` interface. SAM2 automatic point-grid, IoU, stability and NMS settings have no equivalent implemented SAM3 automatic path, so they are not advertised as working SAM3 controls. Unsupported point/box settings are rejected. Automatic SAM3 masks remain unavailable; prompted masks work with local weights.

## Updating another computer

Stop the previous app after its jobs finish. Extract the new portable archive and run from its extracted folder:

```sh
sh scripts/install.sh --wheel dist/compag_annotator-1.0.0rc2-py3-none-any.whl --download-python
~/.local/bin/compag-annotator serve --no-browser --port 8765
```

Open `http://127.0.0.1:8765` manually in a normal browser. Keep the same data directory as before; if you previously used `--data-dir`, supply that same directory. The installer updates application code with rollback support; it does not replace your projects or model runtimes. Custom installation prefixes must be supplied again. The archive contains no datasets or model weights.
