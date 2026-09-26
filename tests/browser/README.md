# Browser verification

Run with the core test dependencies, Playwright and its Chromium browser installed:

```sh
PYTHONPATH=src .venv/bin/python -m pytest tests/browser -q
```

The fixtures start a separate local app with `--qa-mode`, a temporary data folder,
and a dynamically allocated loopback port. They create synthetic images only.
All review assertions require `automated_qa` and `human_verified=false`.
No test installs a provider, downloads a checkpoint, or launches GPU work.

The real browser/API cases cover:

- New-project classes, Unicode and HTML-safe display.
- Full-image box creation/resizing and polygon creation/vertex add/delete.
- Persistent autosave/history, explicit image review and reopening.
- Exact mask brush holes and disconnected components, undo/redo, layers and zoom/pan.
- File and directory uploads, filename-based manual round ordering, remainder allocation,
  round start, import class dropdowns and downloadable native export.
- Reachability of Models, Train, Rounds, Export, Jobs and offline English Help.
- Project-global revision conflicts with retained local failed-edit previews.
- Live before/after comparison and review of a synthetic staged boundary proposal.

Two tests explicitly intercept provider endpoints. They validate the SAM positive /
negative / box protocol, response binding, alternative selection and stale rejection;
and installation/training/prediction form payloads including SAM3 download prohibition,
default-off boundary preparation and explicit opt-in model selection. These tests do
**not** demonstrate real SAM or YOLO operation.

Screenshots are written to pytest temporary directories, including an editor view at
1366×768. Browser test success is not human acceptance or provider validation. The
coordinator owns built-wheel, real-provider and human-UAT release gates.

The UI uses native ES modules and a bundled stylesheet/icon. Runtime operation needs
no npm, CDN, hosted font, third-party script, or browser-external image asset. DOM text
is rendered with text nodes through the English translation layer. Exact masks use
uncompressed COCO column-major RLE; paint updates preserve one instance's topology.
# Installed-wheel browser validation

Set `COMPAG_TEST_INSTALLED_PYTHON` to the Python executable in a separately
installed application, then run this suite normally. The server starts outside
the source checkout with `PYTHONPATH` removed. The test controller still uses its
development browser dependencies. This remains automated QA, not human UAT.


## Addendum 01 browser verification

`test_addendum.py` extends the original suite. Its mock-provider fixture runs
`provider_qa_server.py` with a separate temporary data directory and a dynamic
loopback port. The replacement is explicitly a synthetic provider: it cannot
load weights, train, infer, install anything, or use a GPU. Generation jobs,
tile planning, crop-to-image mapping, per-tile receipts, class changes, selection,
merge, boundary policy, review and persistent history use the production APIs.
The installed-Python environment flag applies to both server fixtures.

Coverage includes:

- Empty-class projects through class-agnostic generation on a 1536×2048 image;
  three Unicode/HTML-safe classes created in place; selection assignment versus
  acceptance; exact union of overlapping masks, a retained hole and disconnected
  components; conflicting class choice; superseded parents; browser reopening,
  bulk delete/reject and persistent undo/redo.
- Server previews for 256, 512, 1024, 640×384 and 513×385 rectangles with percent
  and per-axis pixel overlap, invalid overlap feedback, a 1537×2051 source,
  saved project defaults and immutable prior job plans.
- Deliberately interrupted synthetic generation, durable partial results,
  retry/resume into the same layer without duplicate IDs, explicit replacement
  of unreviewed proposals and preservation of a reviewed object.
- Exact click tests inside holes and nested masks, overlap cycling, multi-select,
  distinct Select and Paint class modes, optional paint-and-accept, and repeated
  pointer movement without repeated geometry requests. Display and hit testing
  retain compressed RLE and cached paths; only the selected brush mask uses a
  full-resolution editable raster.
- Filename-based rounds through explicit batch generation confirmation; normal
  project processing settings; disabled SAM3 automatic capability with its real
  provider reason; boundary recovery off by default.
- Explicit annotation recovery scope on a whole-image layer, correctly reported
  as not applicable with zero model calls; a clearly synthetic staged replacement
  through live before/after review, leaving an unassigned result as a draft.
- A stale merge preview receives the real conflict response, retains its local
  review dialog and creates neither a child nor partially superseded parents.

These are automated UI/API and synthetic-provider results, never evidence of
real SAM mask quality or human review. Real-image/provider acceptance remains a
separate coordinator-owned gate. Screenshot filenames are listed in
`UI_HANDOFF.json`; test runs write them only to their temporary output folders.
