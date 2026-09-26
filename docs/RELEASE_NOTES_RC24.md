# COMPAG Annotator 1.0.0rc24

An Ubuntu application for local full-image annotation and instance-segmentation
workflows, with arbitrary classes and optional model assistance.

## Included

- Manual drawing/correction, mask selection/merging, class assignment, review and
  persistent undo/redo; image/project management and guided next steps.
- Optional SAM2 automatic full-image mask proposals and point/box assistance;
  independent object prompts, configurable source tiles and CPU/GPU selection.
- Optional SAM3 point/box assistance with user-supplied local weights. SAM3
  automatic mask generation is not supported by the current adapter.
- Real YOLO segmentation training from cumulative reviewed data, separate fixed
  validation images, 512 × 512 tile training, matching tiled prediction and
  full-image mask reconstruction. Models retain their training layout.
- Optional contextual boundary checks, OFF by default, with explicit confirmation
  of proposed changes. Masks incompatible with strict YOLO polygons can be
  explicitly excluded from training.
- Reviewed-annotation exports, native project backups, dark theme, device
  selection and model installation progress.
- Dated Best model names showing date, hour/minute, timezone and short identity
  in model lists, activation, training and prediction.

## Install

Download `compag-annotator-1.0.0rc24-ubuntu-x86_64.tar.gz` and `SHA256SUMS` from this
release's assets. Extract the bundle, open its folder in a terminal and run:

```sh
sha256sum -c SHA256SUMS
sh scripts/install.sh --download-python \
  --wheel dist/compag_annotator-1.0.0rc24-py3-none-any.whl --launch
```

`--download-python` explicitly permits downloading the pinned Python runtime when
needed. If Python 3.12 is already available, omit it or select `--python`.
Installation needs an internet connection for dependencies unless an offline
wheelhouse is supplied. The bundle does not include model weights, GPU drivers
or provider runtimes. Add optional models through **Models & AI**; SAM3 weights
are supplied locally by the user. See INSTALL_UBUNTU.md for details.

## Validation and status

This is a **prerelease**, tested on Ubuntu 22.04 WSL2 x86-64. The public verification
summary distinguishes current software tests from historical real-model tests.
Native Ubuntu desktop and genuine human acceptance remain outstanding. Small
synthetic training tests demonstrate execution, not useful accuracy on new data.
Review KNOWN_LIMITATIONS.md and docs/DEPENDENCY_SECURITY.md before use.

This GitHub preparation corrects stale installation/version documentation and
packages the rc24 application without changing its executable code. The rc24
model-naming implementation preserves checkpoint files, identifiers and model
settings. Private projects, datasets, weights and development history are excluded.

License: **AGPL-3.0-only**. Third-party providers and weights retain separate terms.
