# Troubleshooting

| Problem | What to do |
|---|---|
| Application command is missing | Open the applications menu or `~/.local/bin/compag-annotator`. Re-run the installer with the same destinations if installation failed. |
| Browser does not open | Run `compag-annotator serve --no-browser`, then open the printed loopback address. Keep the process running. No Internet is required. |
| Requested port is busy | Omit `--port`; the default selects an available loopback port. |
| Core dependency/import failure | Run `compag-annotator doctor`. Reinstall the release wheel in its isolated environment. Do not install model packages into the core environment. |
| Python 3.12 / venv is missing | Follow INSTALL_UBUNTU.md for an existing interpreter or verified local runtime archive. Do not use sudo on the installer. |
| Offline install cannot find a dependency | Prepare all pinned wheels for Python 3.12 and the same OS/architecture, then use `--wheelhouse`. |
| Image is missing | Relink the explicitly referenced original, or restore a backup with image copies. |
| Image format is rejected | Read the import error; multipage/high-bit-depth variants are not assumed compatible. Convert a copy using a trusted tool and preserve the original. |
| Edit reports a conflict | Refresh the image/project, inspect the saved state and reapply the intended edit. |
| Mask preview belongs to old prompts | Cancel and request a new preview on the current image/model; never force-apply stale output. |
| Optional AI is unavailable | Continue manual editing; inspect runtime, checkpoint, architecture, trust and device status in Models & AI. |
| GPU runs out of memory | Cancel, unload unused models and choose a smaller supported batch/model. Any change of device or settings must be explicit. |
| Training is blocked | Read readiness exclusions: complete review, segmentation geometry, class mapping and separate valid splits are required. |
| Training waits for boundary review | Review staged changes, accept/retain/exclude, then explicitly retry training. |
| YOLO export rejects a mask | Choose native/COCO, repair topology, or explicitly approve a documented approximate conversion. |
| Job was interrupted on restart | Inspect Jobs. Retry supported work; resume training only from a valid supported checkpoint. |
| Uninstall refuses a changed launcher | Preserve or restore the altered file; the remover deliberately avoids deleting unknown files. |

`doctor --json` prints a redacted, metadata-only report. It never loads/downloads weights and is not a GPU benchmark. Review any diagnostic bundle before sharing. Private images, model files, tokens and project archives do not belong in public issue reports.
