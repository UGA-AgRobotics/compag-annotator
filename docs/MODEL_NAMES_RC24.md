# Model names in rc24

Only YOLO checkpoints named `best.pt` receive dated labels, for example
`Best · 2026-09-26 01:13 EDT · c508aedf`.

The date and time use the app host's local timezone. The short model ID distinguishes
runs registered in the same minute. Names appear in Models & AI, Activate model,
Train, and Predict selected. Official model names and other custom filenames keep
their names.

New training completions persist their UTC completion time and dated name.
Existing models use the saved registration time, labelled Saved / registered;
the app does not invent a training time from file modification dates. Missing or
invalid dates are shown as Date unavailable. Reading an older model list does
not rewrite its registry. Checkpoint filenames, file contents, paths, identifiers,
project activation, training settings and inference behavior stay unchanged.

Validation: 122 targeted naming/provider/browser tests passed. Tests cover legacy
records, unchanged checkpoint and registry bytes, same-minute identities, invalid
timestamps, persisted new completion names, re-registration, and activation plus
training/prediction selectors. The training test is an explicit protocol double;
it is not a real model fit. Browser tests use an isolated project and structural
fixture files, never the user's project. Python lint and JavaScript syntax passed.
Installed-wheel verification is recorded in the outer release evidence.
