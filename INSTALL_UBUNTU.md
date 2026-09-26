# Install on Ubuntu

Target platforms are Ubuntu 24.04 and 22.04, x86-64, with Python 3.12 and a local browser. Native clean-install acceptance for each platform remains **NOT_RUN** until recorded by the coordinator. A development or WSL run is not native Ubuntu acceptance.

## Normal install

Extract the prepared release archive into a folder you own. Verify its `SHA256SUMS` against the independently supplied release manifest. The installer needs Python 3.10+ to run and Python 3.12 with `venv`/`ensurepip` for the application.

```sh
sha256sum -c SHA256SUMS
sh scripts/install.sh --wheel dist/compag_annotator-1.0.0rc24-py3-none-any.whl --launch
```

Use `--python /path/to/python3.12` to select an existing interpreter. The installer checks its version and environment support. It runs without `sudo`, creates an isolated application environment and installs the menu entry and command. It does not change system Python, install a driver, or download model weights. Installing dependency wheels needs Internet unless you supply an offline wheelhouse.

The default application code directory is `$XDG_DATA_HOME/compag-annotator-app`, falling back to `~/.local/share/compag-annotator-app`. The command is `~/.local/bin/compag-annotator`; the desktop entry goes in `$XDG_DATA_HOME/applications` (default `~/.local/share/applications`). If the command is not on PATH, use its full path or the applications menu. Sign out/in if the menu has not refreshed.

## Python on Ubuntu 22.04 or a minimal installation

If Python 3.12 is missing, explicitly allow the installer to download and set up the pinned runtime:

```sh
sh scripts/install.sh --download-python \
  --wheel dist/compag_annotator-1.0.0rc24-py3-none-any.whl --launch
```

`--download-python` is consent to download and execute the verified CPython runtime, subject to its licenses. No download occurs without that flag. The installer still runs using the system's Python 3.10+ and never needs root or changes system Python. Keep at least the download size plus 768 MiB free for runtime extraction and installation.

The pin in `locks/python-bootstrap.json` identifies **CPython 3.12.14**, Astral's immutable **20260924** release, and the **x86_64-unknown-linux-gnu install_only** archive. Its expected size is **66,890,910 bytes** and SHA-256 is:

```text
5eae8cf79dd47fc2496a4fccc892936be831ce7a84d984b2299dfb1cdb592682
```

The digest was verified against both the [official release API](https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/20260924) and [asset metadata](https://api.github.com/repos/astral-sh/python-build-standalone/releases/assets/586643535). See the [release page](https://github.com/astral-sh/python-build-standalone/releases/tag/20260924) and [archive documentation](https://gregoryszorc.com/docs/python-build-standalone/main/distributions.html). This pins upstream release metadata; no detached-signature or attestation verification is claimed.

The installer displays source, size, digest and license information. It streams to a temporary file, rejects non-archive responses, enforces the exact byte count and SHA-256, then extracts with traversal/link/size checks. Only the verified runtime is executed. Redirects must use HTTPS on permitted GitHub release hosts; there is no rolling latest lookup, shell bootstrap, model download or privileged command. Cancellation and network failure remove the partial download; retry the same command. A verified cached archive supports an offline repeat. A corrupt cache fails visibly; remove that specific cached archive and retry or use the local-archive route below.

The runtime, its unchanged notices and its verified archive stay beneath the application installation's `runtimes` directory. Updates retain earlier environments for rollback. The installed helper also receives a copy of `python-bootstrap.json`. Runtime binaries are not bundled into the application's source, wheel or sdist. See [third-party notices](THIRD_PARTY_NOTICES.md) before redistributing a runtime separately.

### Existing interpreter or offline archive

The existing `--python /path/to/python3.12` route remains available. To bring the pinned archive from another connected machine, obtain the exact asset named in the lock and verify it against the digest above, then run:

```sh
python3 scripts/install_ubuntu.py \
  --python-archive /path/to/cpython-3.12.14+20260924-x86_64-unknown-linux-gnu-install_only.tar.gz \
  --python-sha256 5eae8cf79dd47fc2496a4fccc892936be831ce7a84d984b2299dfb1cdb592682 \
  --wheel dist/compag_annotator-1.0.0rc24-py3-none-any.whl
```

Other trusted standalone Python 3.12 archives require their own independently verified upstream digest. Select one runtime route at a time. The application interpreter must provide `venv` and `ensurepip`. For a completely offline first install, supply the dependency wheelhouse described below as well.

### Actual bootstrap coverage

The pinned archive was downloaded, hash-verified, extracted and executed, and a built application wheel installed, on **Ubuntu 22.04.5 x86-64 under WSL2**, starting the installer with system Python **3.10.12**. The installed environment uses the downloaded **3.12.14** runtime, without the private development interpreter. Core import/doctor, packaged resources, cached offline repeat and a first-download network failure were exercised. Existing runtime notice files were preserved byte-for-byte. Native Ubuntu 22.04/24.04 desktop acceptance remains **NOT_RUN**; WSL is not evidence of a native desktop or GPU installation. Details are in `docs/PYTHON_BOOTSTRAP_RESULTS.json`.

## Offline install

On a connected Python 3.12 Ubuntu machine of the same architecture, prepare a wheelhouse from the actual release wheel:

```sh
python3.12 -m pip download --only-binary=:all: --dest wheelhouse pip==26.2.1 dist/compag_annotator-1.0.0rc24-py3-none-any.whl
```

Transfer the wheelhouse, release wheel and verified runtime if needed. Then add `--wheelhouse /path/to/wheelhouse` to the installer. It uses `--no-index`; missing dependency wheels fail with an actionable package error. Keep dependency license notices with any redistributed wheelhouse. Core editing and bundled help do not need a network after installation. Optional providers require separately prepared environments and authorized weights.

## Update and roll back

Back up each project through **Projects → Backup** or:

```sh
compag-annotator backup /path/to/project --output /path/to/new-backup.zip
```

Close the running app. Install the next release wheel with the same options/destinations. Each wheel has its own environment. Only a successfully installed and checked environment becomes current; the previous environment is retained. There is no automatic update or model replacement.

```sh
python3 scripts/install_ubuntu.py --rollback
```

Code rollback does not roll back a project's schema or its data. If a future release migrates a project, restore its pre-update backup into a new folder before opening with an older app. Keep the original backup and project until you verify restoration.

## Uninstall

Close the application, then run:

```sh
sh scripts/uninstall.sh
```

If the original release directory is unavailable, use `python3 ~/.local/share/compag-annotator-app/uninstall_ubuntu.py`. Custom installs need the same `--prefix` supplied at installation. The remover checks its marker and launcher identities and refuses unexpected files. Projects, backups, settings, caches and model storage are preserved. Remove those separately only after deciding what you want to keep; the script has no purge switch.

## Installation options

`--prefix`, `--bin-dir` and `--desktop-dir` allow user-owned destinations. Paths with spaces are supported. Update using the same destinations. Install paths with symlink parents, line breaks or percent characters are rejected to keep filesystem and desktop-entry behavior unambiguous. Concurrent installation/removal is blocked by a lock. Keep at least 256 MiB free for core installation plus space for images, backups and any optional runtimes/models.

The installer updates pip to 26.2.1 inside the new application environment before installing the application. Include that wheel in an offline wheelhouse; system Python and the preceding installed version remain unchanged.
