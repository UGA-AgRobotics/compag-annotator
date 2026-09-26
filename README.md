# COMPAG Annotator

A local Ubuntu application for drawing, correcting and reviewing objects in full images, with your own classes. Manual annotation does not need a model, a cloud account or a GPU. Optional local AI assistance and segmentation training are separate installations.

**Version: 1.0.0rc24 — release candidate.** Licensed under **AGPL-3.0-only** with the author approval recorded in [LICENSE_REVIEW.md](LICENSE_REVIEW.md). This is a prerelease tested on Ubuntu 22.04 WSL2; native Ubuntu desktop and genuine human acceptance remain outstanding. See [ACCEPTANCE_RESULTS.json](ACCEPTANCE_RESULTS.json) for current tests and clearly identified inherited real-model evidence. The prepared package is a prerelease; see [release notes](docs/RELEASE_NOTES_RC24.md).

## Install and open

Download `compag-annotator-1.0.0rc24-ubuntu-x86_64.tar.gz` and `SHA256SUMS` from the GitHub Release assets. The automatically generated GitHub source archive does not contain the built installer wheel. Extract the Ubuntu bundle and, from its directory, run:

```sh
sha256sum -c SHA256SUMS
sh scripts/install.sh --wheel dist/compag_annotator-1.0.0rc24-py3-none-any.whl --launch
```

If Python 3.12 is missing, add `--download-python` to explicitly allow the pinned, verified user-space runtime download. See [Ubuntu installation](INSTALL_UBUNTU.md) for the source/digest, offline installation, updates and removal. Open **COMPAG Annotator** in the applications menu, or run `compag-annotator`. The application opens in your browser and keeps work on your computer.

## Installing from repository source

For developers with Python 3.12 (including venv) already installed:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[dev]'
.venv/bin/python -m build --outdir dist
sh scripts/install.sh --python "$(pwd)/.venv/bin/python" \
  --wheel dist/compag_annotator-1.0.0rc24-py3-none-any.whl --launch
```

The source checkout includes tests, GitHub Actions, license notices and offline help.
Optional SAM/YOLO runtimes and model weights are installed separately in **Models & AI**.
The app can detect compatible CPU/GPU devices after those runtimes are installed;
the installer does not install GPU drivers. SAM3 requires user-supplied local weights.

## Start with your images

1. In **Projects**, create a project. You can start with zero classes and add them after generating masks.
2. In **Images**, import files or a folder. Copying images into the project is the simplest portable choice.
3. In **Annotate**, draw/edit manually, or use SAM2 **Generate masks / Segment everything**. Choose whole-image processing or real source tiles: 256×256, 512×512, 1024×1024 or independent Custom width × height. New projects start at 512×512 with 25% overlap; preview the full grid before starting.
4. In **Select**, click masks or select several from the object list. Create a class in the editor and assign it to the selection. Assignment keeps separate instances and does not accept them. Explicit **Merge selected masks** previews the exact union of existing pixels; use separate brush/SAM refinement to add missing pixels. Review, accept/reject, delete or undo as needed.
5. Confirm image completeness only when you have checked the whole image and resolved relevant drafts/unassigned proposals. An empty image can be complete after deliberate review.
6. Use **Export** for a native backup, COCO, YOLO or another supported format. Read the conversion report before accepting any approximation; disconnected merged masks and holes can block strict YOLO segmentation conversion.

The **Next step** guide follows the current image: save SAM previews as drafts, label and review masks, explicitly confirm the whole image, then choose the next image, export, or optional training. The guide never accepts objects or changes dataset roles automatically. See [the workflow guide](docs/WORKFLOW_GUIDANCE.md).

**Generate masks** returns proposals for review, not guaranteed complete objects or semantic labels. It needs a ready SAM2 provider but no YOLO checkpoint or predefined classes. SAM3 automatic generation is unavailable until a compatible adapter is implemented and tested; its separate point/box assistance uses user-supplied local weights. Crop size is distinct from model encoder resizing and YOLO image size.

**Rounds** organizes eligible images; **Train** uses cumulative accepted segmentation data from explicitly completed images. Heavy boundary recovery is **OFF by default** and allowed only with explicit consent for a scoped job in **training preparation**, **automatic-mask annotation**, or the **Boundary check** option in YOLO prediction. Review proposed changes before applying them. Import, opening images, ordinary prediction, manual edits, prompting, merging, rounds and export never enable it automatically.

The prior Addendum 01 results remain documented in [the requirement map](ADDENDUM_01_IMPLEMENTATION.md). See [the rc2 update guide](docs/UPDATE_RC2_EN.md) for detected CPU/GPU selection, download/setup progress, dark theme and editable SAM settings.

## Help and status

- [Quick start](docs/QUICKSTART.md) · [User guide](USER_GUIDE.md)
- [Models](MODEL_SETUP.md) · [Training and rounds](TRAINING_AND_ROUNDS.md) · [Export formats](EXPORT_FORMATS.md)
- [Troubleshooting](TROUBLESHOOTING.md) · [Known limitations](KNOWN_LIMITATIONS.md)
- [Development and checks](DEVELOPMENT.md) · [Architecture](ARCHITECTURE.md)
- [License review](LICENSE_REVIEW.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Publication checklist](PUBLICATION_CHECKLIST.md)

The **Help** screen is bundled for offline use. This candidate covers 2-D still images and instance segmentation; video, medical-image formats, 3-D and shared remote servers are outside v1.

## Tile training and boundary checking

For small objects, Train defaults to **512 × 512 tiles**. A new model stores this layout and Predict selected reuses it, then reconstructs masks in full-image coordinates. Optional **Boundary check** uses contextual SAM inference to propose seam repairs/merges for your confirmation. It is slower and OFF by default. See [Training and rounds](TRAINING_AND_ROUNDS.md) and [rc23 implementation/evidence](docs/TILED_PIPELINE_RC23.md).

## Publishing this candidate

See [the GitHub upload guide](docs/GITHUB_RELEASE.md) for repository contents, release assets and prerelease tagging. Current package checks are in [the release verification summary](docs/RELEASE_VERIFICATION_RC24.json).
