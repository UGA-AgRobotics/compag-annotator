# GitHub preparation: 1.0.0rc24

The current publication entry point is docs/RELEASE_VERIFICATION_RC24.json. Installation filenames, the README, bundled installation help and current version labels have been synchronized with rc24. The clean repository includes release notes and GitHub upload instructions; release assets are built separately. Executable application files remain identical to the previously installed rc24. No remote publication, new model fit, production-data changes or new accuracy claim is part of this preparation. All reports below describe earlier implementation stages and their original evidence.

# Incremental release: 1.0.0rc24

YOLO best.pt entries now display their saved date, hour, minute, timezone and a short model ID in model cards, activation confirmation, training and prediction selectors. New completed training runs persist a UTC completion timestamp. Existing models use the recorded registration timestamp and identify that source explicitly. File paths, model IDs, weights, active bindings and prediction settings remain unchanged. Targeted naming/provider/browser checks passed (122 tests); no real training or inference was run for this naming update. See docs/MODEL_NAMES_RC24.md and the outer evidence/FINAL_VERIFICATION.json. Earlier reports below are historical.

# Incremental release: 1.0.0rc14

Incomplete image review now gets actionable counts and navigation instead of failed-save recovery. The frontend prevents premature confirmation and handles the matching server validation without changing backend review rules. Current checks and preservation receipts are in the outer FINAL_VERIFICATION.json. The following reports are retained historical evidence.

# Incremental release: 1.0.0rc13

This update adds UI workflow guidance only. See `docs/WORKFLOW_GUIDANCE.md`. The reports below are historical baseline evidence. Current rc13 installed-package checks are recorded in the outer `FINAL_VERIFICATION.json`; no new real-model run or human acceptance is claimed.

# Incremental release: 1.0.0rc12

This update changes training readiness guidance and preflight UI only. See `docs/TRAINING_READINESS_UPDATE.md`. The following rc11 report is retained as baseline evidence; its test counts are historical, not new rc12 measurements.

# Release candidate 1.0.0rc11

Prepared for local review under the author-approved AGPL-3.0-only license. The five supplied authors are recorded in CITATION.cff and LICENSE_REVIEW.md. No remote repository, registry upload or public release has been created.

## Corrections

Light-theme dialogs now have explicit light rendering and background, independent of the operating system theme. The shipped browser tests follow current English/export/import/review controls. Versioned artifacts derive from project metadata; CI tests an installed wheel. The public scanner permits only the reviewed report, prevents stale self-hashing manifests and refuses empty-history success. Current evidence and inherited model/training results are clearly separated.

Pillow, Hydra and packaging/bootstrap tools in optional provider setups have been updated to patched versions with published hashes. The base installer upgrades pip only in its new private application environment. An old provider runtime cannot be relabelled patched through adapter-only repair. Existing user runtimes, data and weights remain unchanged.

## Measured checks

The complete installed security candidate passed 601 core tests, 38 browser tests and 11 JavaScript tests; Python lint passed. The four app/system theme combinations achieved dialog text contrast of about 13.11:1 in light mode and 14.29:1 in dark mode. Installer upgrade rc10-to-rc11, rollback, bad-digest refusal, idempotence and uninstall preserved a real synthetic annotated project byte-for-byte and a separate model sentinel.

A copy of the existing 785-mask audit project on the native WSL filesystem assigned two objects in 0.474 seconds, undid in 0.072 seconds and redid in 0.077 seconds. These are direct component timings, not end-to-end UI latency. Persistent history was retained; original data was unchanged. Large native backups remain expected because they retain history.

The patched SAM2/SAM3/YOLO package environments passed dependency checks, API imports and CUDA lock resolution using existing read-only ML binaries. These are not fresh complete provider installations. Prior full-image real-model and two-fit evidence remains inherited. No new fit occurred. Consult ACCEPTANCE_RESULTS.json for the current status of any separately budgeted real-weight smoke test.

## Release qualifications

Keep this version labelled prerelease. Native Ubuntu 22.04/24.04 desktop and genuine human acceptance remain unperformed. SAM3 automatic generation is unsupported. Tiny historical training fixtures do not establish useful accuracy. Version-specific advisory queries covered 127 package/version entries with no unavailable entries; Torch pins retain advisories and the trusted-model limitations remain explicit in docs/DEPENDENCY_SECURITY.md. No vulnerability-free claim is made.

The clean public tree excludes raw logs, personal paths, model files, images, runtime environments and private Git history. Final outer checksums and FINAL_VERIFICATION.json bind the delivered tree and built wheel/sdist; inspect them before any future upload. Private evidence and the Persian handoff report are outside the public source.
