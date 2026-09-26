# COMPAG Annotator

COMPAG Annotator is a local Ubuntu app for drawing, correcting, and reviewing objects in full images, using your own classes. Manual annotation works without a model, a cloud account, or a GPU. Local AI assistance and segmentation training are optional and installed separately.

**Current version:** 1.0.0rc24 (release candidate)

**License:** AGPL-3.0-only. The author's approval is recorded in `LICENSE_REVIEW.md`.

This is a prerelease. It has been tested on Ubuntu 22.04 under WSL2, but not yet on a native Ubuntu desktop, and it hasn't gone through acceptance testing with real users. Current test results are in `ACCEPTANCE_RESULTS.json`, where results carried over from earlier real-model runs are marked as such. See the release notes for more.

## Install and open

Download `compag-annotator-1.0.0rc24-ubuntu-x86_64.tar.gz` and `SHA256SUMS` from the GitHub Release assets. Don't use the source archive that GitHub generates automatically, since it doesn't include the built installer wheel. Extract the bundle and run this from its folder:

```bash
sha256sum -c SHA256SUMS
sh scripts/install.sh --wheel dist/compag_annotator-1.0.0rc24-py3-none-any.whl --launch
```

If Python 3.12 isn't installed, add `--download-python`. This lets the installer download a pinned, verified Python runtime into your user space. The Ubuntu installation guide covers the download source and digest, offline installs, updates, and uninstalling.

To open the app, pick COMPAG Annotator from the applications menu or run `compag-annotator`. It opens in your browser, and your work stays on your computer.

## Installing from source

For developers who already have Python 3.12 (with venv):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[dev]'
.venv/bin/python -m build --outdir dist
sh scripts/install.sh --python "$(pwd)/.venv/bin/python" \
  --wheel dist/compag_annotator-1.0.0rc24-py3-none-any.whl --launch
```

The source checkout includes tests, GitHub Actions workflows, license notices, and the offline help. Optional SAM/YOLO runtimes and model weights are installed separately from the Models & AI screen. Once they're installed, the app can detect compatible CPU/GPU devices. The installer doesn't install GPU drivers. SAM3 needs local weights that you provide yourself.

## Getting started with your images

1. **Projects:** Create a project. You don't need any classes to start; you can add them after generating masks.
2. **Images:** Import files or a whole folder. Copying the images into the project is the easiest way to keep it portable.
3. **Annotate:** Draw and edit by hand, or use SAM2 **Generate masks** / **Segment everything**. You can process the whole image or split it into real source tiles: 256×256, 512×512, 1024×1024, or a custom width × height. New projects default to 512×512 tiles with 25% overlap. Preview the full tile grid before you start.
4. **Select:** Click masks on the image or pick several from the object list. Create a class in the editor and assign it to the selection. Assigning a class keeps the instances separate and doesn't accept them. **Merge selected masks** shows a preview of the exact union of the existing pixels; to add missing pixels, refine separately with the brush or SAM. Review, accept, reject, delete, or undo as needed.
5. **Confirm the image:** Only mark an image as complete once you've checked all of it and dealt with any relevant drafts or unassigned proposals. An empty image can be marked complete too, as long as you've actually reviewed it.
6. **Export:** Save a native backup, or export to COCO, YOLO, or another supported format. Read the conversion report before accepting any approximation. Merged masks with disconnected parts or holes can block strict YOLO segmentation export.

The **Next step** guide follows the image you're working on: save SAM previews as drafts, label and review masks, confirm the whole image, then move on to the next image, export, or train if you want to. It never accepts objects or changes dataset roles for you. See the workflow guide for more.

**Generate masks** gives you proposals to review. They aren't guaranteed to be complete objects and don't come with labels. It needs a working SAM2 provider, but no YOLO checkpoint or predefined classes. Automatic mask generation with SAM3 isn't available yet, because a compatible adapter still has to be implemented and tested. SAM3 point/box assistance does work with your own local weights. Also note that crop size is a separate setting from the model's encoder resizing and from the YOLO image size.

**Rounds** organizes eligible images, and **Train** uses all accepted segmentation data from images you've marked as complete. Heavy boundary recovery is off by default. It only runs if you explicitly allow it for a specific job: training preparation, automatic-mask annotation, or the Boundary check option in YOLO prediction. Review the proposed changes before applying them. Importing, opening images, regular prediction, manual edits, prompting, merging, rounds, and export never turn it on by themselves.

Results from Addendum 01 are still documented in the requirement map. The rc2 update guide covers CPU/GPU device selection, download and setup progress, the dark theme, and editable SAM settings.

## Help and status

- Quick start, User guide
- Models, Training and rounds, Export formats
- Troubleshooting, Known limitations
- Development and checks, Architecture
- License review, Third-party notices, Publication checklist

The Help screen is bundled with the app, so it works offline. This release covers 2-D still images and instance segmentation. Video, medical image formats, 3-D, and shared remote servers aren't part of v1.

## Tile training and boundary checking

For small objects, Train uses 512 × 512 tiles by default. A new model remembers this tile layout, and **Predict selected** reuses it, then puts the masks back together in full-image coordinates. The optional **Boundary check** runs SAM with the surrounding context to suggest fixes or merges along tile seams, which you then confirm. It's slower and off by default. See Training and rounds, and the rc23 notes for implementation details and evidence.

## Publishing this release

The GitHub upload guide explains what goes in the repository, which release assets to attach, and how to tag a prerelease. Current package checks are in the release verification summary.