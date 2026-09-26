# Development and release checks

Use Python 3.12. Keep ML providers in separate environments; ordinary CI never trains models or downloads weights. Node 22 runs the small JavaScript test suite. Browser binaries are development dependencies only.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install '.[test,dev]'
.venv/bin/python -m ruff check src tests scripts
node --experimental-default-type=module --test tests/js/*.test.js
.venv/bin/python -m build --outdir dist
release_wheel=$(.venv/bin/python scripts/release_paths.py --field wheel)
release_sdist=$(.venv/bin/python scripts/release_paths.py --field sdist)
.venv/bin/python scripts/stage_public.py --source . \
  --artifact "$release_wheel" --artifact "$release_sdist" --require-artifacts \
  --report ../sanitization-report.json
```

These are reproducible commands, not a claim they have run on the reader's machine. Actual evidence is indexed by ACCEPTANCE_RESULTS.json. GPU/model and human acceptance are separate.

## Test the built wheel

```sh
python3.12 -m venv ../installed-check
../installed-check/bin/python -m pip install "$release_wheel[test]"
../installed-check/bin/python -m playwright install chromium
export COMPAG_TEST_INSTALLED_PYTHON="$(cd ../installed-check && pwd)/bin/python"
export CUDA_VISIBLE_DEVICES=-1
"$COMPAG_TEST_INSTALLED_PYTHON" -m pytest -q tests --ignore=tests/browser -o pythonpath=
"$COMPAG_TEST_INSTALLED_PYTHON" -m pytest -q tests/browser -o pythonpath=
```

The override removes source imports in CLI and browser subprocess tests. Optional CPU YOLO dataset-loader tests use COMPAG_YOLO_TEST_PYTHON only when a prepared runtime is supplied; otherwise those checks explicitly skip. No fit occurs. The CI workflow executes against the installed wheel on Ubuntu 22.04 and 24.04 runners; this local preparation does not claim a remote CI run or native desktop acceptance.

## Stage a release

Keep raw logs, datasets, models and runtimes outside source. The scanner uses an exact allowlist, rejects credential patterns and private absolute paths, and inspects actual wheel/sdist bytes. Generated source manifests exclude their own hash and outer checksums so restaging cannot silently preserve stale hashes.

```sh
.venv/bin/python scripts/stage_public.py --source . \
  --destination ../public_release \
  --artifact "$release_wheel" --artifact "$release_sdist" --require-artifacts \
  --report ../public-stage-report.json
```

Staging excludes `.git`. For a repository with real commits, also use `--history`; an empty history now fails rather than certifying zero objects. Do not copy private development history into a public repository. The public tree is prepared for a new repository; review author identity and all intended commits before a separate authorized push.

Build from the frozen staging tree, scan again against it, and record outer SHA256SUMS and artifact identities. No report inside a wheel can contain that wheel's own final hash. The outer verification receipt supplies that binding. Any later source edit invalidates it.

Test installer idempotence, a real rc10-to-rc11 upgrade, rollback, bad-digest refusal and uninstall preservation in a disposable HOME, including spaces and Unicode. Use docs/HUMAN_UAT.md for the remaining genuine human and native-desktop checks. Never convert automated QA decisions into human approval.
