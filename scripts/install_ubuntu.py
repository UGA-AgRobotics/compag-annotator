#!/usr/bin/env python3
"""User-space installer; no privileged commands or downloaded bootstrap scripts."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
import urllib.error
import urllib.parse
import urllib.request
import zipfile

MARKER = "compag-annotator-user-install-v1"
VERSION = "1.0.0rc10"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def location(value):
    path = Path(value).expanduser().absolute()
    if any(c in str(path) for c in ("\n", "\r", "\0", "%")):
        raise ValueError("Install paths cannot contain line breaks, NUL or percent characters.")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Choose an installation path without symbolic-link parents.")
    return path


def atomic_write(path, text, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".compag-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        if os.path.lexists(name):
            os.unlink(name)


def desktop_quote(value):
    # Desktop string escaping is applied before Exec argument escaping.
    text = str(value)
    if any(c in text for c in ("\n", "\r", "%", "\0")):
        raise ValueError("Unsupported character in desktop launcher path.")
    text = "".join("\\" + c if c in '\\"`$' else c for c in text)
    return '"' + text.replace("\\", "\\\\") + '"'


def desktop_entry(command, icon):
    return (
        "[Desktop Entry]\nType=Application\nVersion=1.0\nName=COMPAG Annotator\n"
        "Comment=Annotate and review images locally\n"
        f"Exec={desktop_quote(command)}\n"
        f"Icon={str(icon).replace(chr(92), chr(92) * 2)}\n"
        "Terminal=false\nCategories=Graphics;\nStartupNotify=false\n"
    )


def command_wrapper(prefix):
    return "#!/bin/sh\nexec " + shlex.quote(str(prefix / "current/bin/python")) + " -m compag_annotator.cli \"$@\"\n"


def _run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def check_python(executable):
    result = _run([executable, "-I", "-c", "import sys,venv,ensurepip; print('.'.join(map(str,sys.version_info[:2])))"],
                  capture_output=True, text=True)
    if result.stdout.strip() != "3.12":
        raise ValueError("The application installer requires Python 3.12 with venv and ensurepip.")
    return Path(executable).absolute()


def extract_python(archive, destination):
    """Bounded extraction with links created last, after every member is checked."""
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) > 100000 or sum(x.size for x in members) > 2 * 1024**3:
            raise ValueError("Python runtime archive is larger than the allowed extraction limit.")
        entries = {}
        links = set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "python" or "\\" in member.name:
                raise ValueError("Unsafe Python runtime archive path.")
            if member.name in entries or not (member.isdir() or member.isfile() or member.issym() or member.islnk()):
                raise ValueError("Unsupported or repeated Python runtime archive member.")
            entries[member.name] = member
            if member.issym() or member.islnk():
                links.add(path)
                raw = PurePosixPath(member.linkname)
                target = (destination / path.parent / raw) if member.issym() else (destination / raw)
                if raw.is_absolute() or not target.resolve().is_relative_to(destination.resolve() / "python"):
                    raise ValueError("Python archive link escapes the runtime.")
        for name in entries:
            if any(parent in links for parent in PurePosixPath(name).parents):
                raise ValueError("Python archive contains a file beneath a link.")
        for member in members:
            path = destination / member.name
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, path.open("xb") as target:
                    shutil.copyfileobj(source, target)
                path.chmod(0o755 if member.mode & 0o111 else 0o644)
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if member.issym():
                path.symlink_to(member.linkname)
            elif member.islnk():
                target = destination / member.linkname
                if target.is_symlink() or not target.is_file():
                    raise ValueError("Invalid Python runtime hard link.")
                os.link(target, path)


def bootstrap_lock_path():
    """Source release keeps the pin in locks; installed helper has a local copy."""
    adjacent = Path(__file__).resolve().with_name("python-bootstrap.json")
    return adjacent if adjacent.is_file() else Path(__file__).resolve().parent.parent / "locks/python-bootstrap.json"


def load_bootstrap_pin():
    pin = json.loads(bootstrap_lock_path().read_text(encoding="utf-8"))
    version, tag = pin.get("python_version", ""), pin.get("release_tag", "")
    if (pin.get("schema_version") != 1 or pin.get("provider") != "astral-sh/python-build-standalone"
            or not re.fullmatch(r"3\.12\.\d+", version) or not re.fullmatch(r"\d{8}", tag)
            or pin.get("target") != "x86_64-unknown-linux-gnu" or pin.get("archive_layout") != "install_only"
            or not re.fullmatch(r"[a-f0-9]{64}", pin.get("sha256", ""))
            or type(pin.get("size_bytes")) is not int or not 0 < pin["size_bytes"] <= 256 * 1024**2):
        raise ValueError("Invalid pinned Python bootstrap metadata; restore the release's locks/python-bootstrap.json.")
    expected_name = f"cpython-{version}+{tag}-x86_64-unknown-linux-gnu-install_only.tar.gz"
    expected_url = "https://github.com/astral-sh/python-build-standalone/releases/download/" + tag + "/" + urllib.parse.quote(expected_name)
    if pin.get("asset_name") != expected_name or pin.get("asset_url") != expected_url:
        raise ValueError("Python bootstrap source does not match the pinned official release asset.")
    return pin


def _check_bootstrap_host():
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise ValueError("The pinned Python download supports Linux x86-64 only. Choose a compatible existing --python instead.")
    libc, _ = platform.libc_ver()
    if libc != "glibc":
        raise ValueError("The pinned Linux runtime requires glibc, as provided by supported Ubuntu systems.")


class _PythonDownloadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if (target.scheme != "https" or target.hostname not in {
                "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"}
                or target.username or target.password or target.port not in {None, 443}):
            raise ValueError("Python download redirected outside the permitted HTTPS release hosts.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_python_download(url):
    opener = urllib.request.build_opener(_PythonDownloadRedirect())
    request = urllib.request.Request(url, headers={"User-Agent": "COMPAG-Annotator-bootstrap/1.0",
                                                   "Accept": "application/octet-stream",
                                                   "Accept-Encoding": "identity"})
    return opener.open(request, timeout=30)


def download_python(prefix, pin, *, consent=False):
    """Stream only the pinned runtime, then verify before making it available."""
    if consent is not True:
        raise ValueError("Python download requires explicit consent via --download-python.")
    _check_bootstrap_host()
    print(f"Python download consent supplied: CPython {pin['python_version']}, release {pin['release_tag']}.", flush=True)
    print(f"Source: {pin['asset_url']}\nDownload size: {pin['size_bytes']:,} bytes\nSHA-256: {pin['sha256']}", flush=True)
    print("License: Python Software Foundation plus bundled third-party notices. See THIRD_PARTY_NOTICES.md.", flush=True)
    cache = Path(prefix) / "runtimes" / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / pin["asset_name"]
    if archive.exists() or archive.is_symlink():
        if archive.is_symlink() or archive.stat().st_size != pin["size_bytes"] or digest(archive) != pin["sha256"]:
            raise ValueError("Cached Python archive failed size/SHA-256 verification. Remove that cached archive and retry, or supply a verified local archive.")
        print("Verified cached Python archive; no runtime download needed.", flush=True)
        return archive
    if shutil.disk_usage(cache).free < pin["size_bytes"] + 768 * 1024**2:
        raise ValueError("Free at least the runtime download size plus 768 MiB for Python extraction and installation.")
    fd, temporary_name = tempfile.mkstemp(prefix=".python-download-", suffix=".partial", dir=cache)
    temporary = Path(temporary_name)
    received, next_report = 0, 8 * 1024**2
    identity = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as target, _open_python_download(pin["asset_url"]) as response:
            if response.status != 200:
                raise ValueError("Python runtime download did not return a complete successful response.")
            kind = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if kind in {"text/html", "application/xhtml+xml", "application/json", "text/plain"}:
                raise ValueError("Python runtime download returned a page or error document, not an archive.")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) != pin["size_bytes"]):
                raise ValueError("Python runtime Content-Length differs from the pinned asset size.")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                if received == 0 and not chunk.startswith(b"\x1f\x8b"):
                    raise ValueError("Python runtime download is not a gzip archive.")
                received += len(chunk)
                if received > pin["size_bytes"]:
                    raise ValueError("Python runtime download exceeds the pinned asset size.")
                identity.update(chunk)
                target.write(chunk)
                if received >= next_report:
                    print(f"Python download: {received:,} / {pin['size_bytes']:,} bytes", flush=True)
                    next_report = received + 8 * 1024**2
            target.flush()
            os.fsync(target.fileno())
        if received != pin["size_bytes"] or identity.hexdigest() != pin["sha256"]:
            raise ValueError("Python runtime size/SHA-256 mismatch; nothing was extracted or executed.")
        os.replace(temporary, archive)
        print(f"Python download verified: {received:,} bytes.", flush=True)
        return archive
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        # Redirect URLs can carry temporary credentials. Do not print them.
        raise ValueError("Python runtime download failed or was interrupted. Check your connection and retry --download-python, or use --python-archive with --python-sha256 offline.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def install_python_archive(archive, expected_sha256, prefix, *, expected_version=None):
    if not expected_sha256 or not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("A Python archive needs --python-sha256 from an independently verified upstream source.")
    identity = digest(archive)
    if identity != expected_sha256.lower():
        raise ValueError("Python archive SHA-256 mismatch; nothing was executed.")
    runtime = prefix / "runtimes" / identity
    created = False
    if runtime.is_symlink():
        raise ValueError("The managed Python runtime directory cannot be a symbolic link.")
    if not runtime.exists():
        temporary = prefix / "runtimes" / (".extract-" + uuid.uuid4().hex)
        temporary.mkdir(parents=True)
        try:
            extract_python(archive, temporary)
            if not (temporary / "python/bin/python3.12").is_file():
                raise ValueError("The archive does not contain a standalone Python 3.12 runtime.")
            temporary.rename(runtime)
            created = True
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    try:
        executable = check_python(runtime / "python/bin/python3.12")
        if expected_version:
            result = _run([executable, "-I", "-c", "import platform; print(platform.python_version())"], capture_output=True, text=True)
            if result.stdout.strip() != expected_version:
                raise ValueError("Extracted Python version does not match the pinned bootstrap version.")
        return executable
    except BaseException:
        if created:
            shutil.rmtree(runtime)
        raise


def prepare_python(args, prefix):
    if args.download_python:
        if args.python or args.python_archive or args.python_sha256:
            raise ValueError("Use --download-python by itself, or choose an existing interpreter/local archive instead.")
        pin = load_bootstrap_pin()
        archive = download_python(prefix, pin, consent=args.download_python)
        return install_python_archive(archive, pin["sha256"], prefix, expected_version=pin["python_version"])
    if args.python_archive:
        return install_python_archive(args.python_archive, args.python_sha256, prefix)
    executable = args.python or shutil.which("python3.12")
    if not executable:
        raise ValueError("Python 3.12 was not found. Consent to the pinned runtime with --download-python, use --python PATH, or provide --python-archive and --python-sha256. See INSTALL_UBUNTU.md.")
    return check_python(executable)


def read_manifest(prefix):
    path = prefix / "installation.json"
    if not path.exists():
        return None
    if path.is_symlink():
        raise ValueError("Invalid installation marker.")
    record = json.loads(path.read_text())
    if record.get("schema") != MARKER or record.get("prefix") != str(prefix):
        raise ValueError("This folder is not a recognized COMPAG Annotator installation.")
    return record


@contextlib.contextmanager
def install_lock(prefix):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lockpath = prefix.parent / ("." + prefix.name + ".install.lock")
    fd = os.open(lockpath, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another installation or uninstall is still running.") from exc
        yield


def switch_current(prefix, target):
    if not re.fullmatch(r"[a-f0-9]{16}", target):
        raise ValueError("Invalid saved release identifier.")
    if not (prefix / "releases" / target / "bin/python").is_file():
        raise ValueError("The selected installed version is incomplete.")
    current = prefix / "current"
    if current.exists() and not current.is_symlink():
        raise ValueError("The current-version pointer is not managed by this installer.")
    link = prefix / (".current-" + uuid.uuid4().hex)
    try:
        link.symlink_to(Path("releases") / target)
        os.replace(link, current)
    finally:
        if link.is_symlink():
            link.unlink()


def install(args):
    prefix = location(args.prefix)
    with install_lock(prefix):
        old = read_manifest(prefix)
        if args.rollback:
            if not old or not old.get("previous"):
                raise ValueError("No previous installed version is available.")
            switch_current(prefix, old["previous"])
            old["current"], old["previous"] = old["previous"], old["current"]
            atomic_write(prefix / "installation.json", json.dumps(old, indent=2) + "\n")
            print("Previous application version restored. Projects and models were preserved.")
            return
        if old is None and prefix.exists() and any(prefix.iterdir()):
            raise ValueError("Choose an empty folder; the install location contains unmanaged files.")
        wheel = Path(args.wheel).absolute() if args.wheel else None
        if wheel is None:
            matches = sorted((Path(__file__).resolve().parent.parent / "dist").glob("compag_annotator-*.whl"))
            if len(matches) != 1:
                raise ValueError("Choose the release wheel with --wheel FILE.")
            wheel = matches[0]
        if not wheel.is_file() or not re.fullmatch(r"compag_annotator-[A-Za-z0-9_.+]+-py3-none-any.whl", wheel.name):
            raise ValueError("Select the built COMPAG Annotator wheel.")
        with zipfile.ZipFile(wheel) as archive:
            if "compag_annotator/cli.py" not in archive.namelist():
                raise ValueError("The selected wheel does not contain the application.")
        wheel_hash = digest(wheel)
        if args.wheel_sha256 and wheel_hash != args.wheel_sha256.lower():
            raise ValueError("Application wheel SHA-256 mismatch.")
        prefix.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(prefix).free < 256 * 1024**2 + wheel.stat().st_size * 4:
            raise ValueError("Free at least 256 MiB for the application environment and retry.")
        bin_dir = location(args.bin_dir)
        desktop_dir = location(args.desktop_dir)
        launcher = bin_dir / "compag-annotator"
        desktop = desktop_dir / "compag-annotator.desktop"
        for item in (launcher, desktop):
            if item.exists() or item.is_symlink():
                if not old or str(item) not in old.get("owned_files", {}) or digest(item) != old["owned_files"][str(item)]:
                    raise ValueError("An existing launcher was changed or belongs to another installation; choose another destination.")
        if old and (old["bin_dir"] != str(bin_dir) or old["desktop_dir"] != str(desktop_dir)):
            raise ValueError("Updates must use the installation's original launcher directories.")
        if old is None:
            # A resumable marker makes a failed first installation retryable.
            old = {"schema": MARKER, "prefix": str(prefix), "bin_dir": str(bin_dir),
                   "desktop_dir": str(desktop_dir), "current": None, "previous": None, "owned_files": {}}
            atomic_write(prefix / "installation.json", json.dumps(old, indent=2) + "\n")
        interpreter = prepare_python(args, prefix)
        # A newly selected managed runtime must not silently reuse an environment
        # that still points at a previously supplied external interpreter.
        release_identity = wheel_hash
        if args.download_python:
            release_identity += ":python:" + load_bootstrap_pin()["sha256"]
        release = hashlib.sha256(release_identity.encode()).hexdigest()[:16] if args.download_python else wheel_hash[:16]
        environment = prefix / "releases" / release
        completed = environment / ".compag-install-complete"
        if not completed.is_file():
            if environment.exists():
                shutil.rmtree(environment)
            environment.parent.mkdir(exist_ok=True)
            try:
                _run([interpreter, "-m", "venv", environment])
                command = [environment / "bin/python", "-m", "pip", "--disable-pip-version-check", "install", "--only-binary=:all:"]
                if args.wheelhouse:
                    command += ["--no-index", "--find-links", Path(args.wheelhouse).absolute()]
                # Upgrade only this new application venv before processing its packages.
                # Offline installations must supply the same pip wheel in their wheelhouse.
                _run(command + ["pip==26.2.1"])
                command += [wheel]
                _run(command)
                _run([environment / "bin/python", "-I", "-c",
                      "import compag_annotator.cli,fastapi,uvicorn,numpy,PIL,shapely,pycocotools; "
                      "import sys; assert 'torch' not in sys.modules; "
                      "from importlib.resources import files; "
                      "assert files('compag_annotator').joinpath('help/index.html').is_file(); "
                      "assert files('compag_annotator').joinpath('web/assets/index.html').is_file()"])
                _run([environment / "bin/python", "-m", "compag_annotator.cli", "--version"])
                atomic_write(completed, wheel_hash + "\n")
            except BaseException:
                if environment.exists():
                    shutil.rmtree(environment)
                raise
        elif completed.read_text().strip() != wheel_hash:
            raise ValueError("Installed release identity mismatch.")
        icon = prefix / "icon.svg"
        icon_text = _run([environment / "bin/python", "-I", "-c",
                          "from importlib.resources import files; print(files('compag_annotator').joinpath('assets/icon.svg').read_text())"],
                         capture_output=True, text=True).stdout
        payloads = {launcher: command_wrapper(prefix), desktop: desktop_entry(launcher, icon)}
        originals = {path: path.read_bytes() if path.exists() else None for path in payloads}
        previous = old["current"] if old and old["current"] != release else (old or {}).get("previous")
        try:
            atomic_write(icon, icon_text, 0o644)
            for path, text in payloads.items():
                atomic_write(path, text, 0o755 if path == launcher else 0o644)
            for name in ("install_ubuntu.py", "uninstall_ubuntu.py"):
                source = Path(__file__).absolute().parent / name
                if source.exists() and source != prefix / name:
                    atomic_write(prefix / name, source.read_text(), 0o755)
            pin_source = bootstrap_lock_path()
            if pin_source.is_file() and pin_source != prefix / "python-bootstrap.json":
                atomic_write(prefix / "python-bootstrap.json", pin_source.read_text(), 0o644)
            record = {"schema": MARKER, "prefix": str(prefix), "bin_dir": str(bin_dir), "desktop_dir": str(desktop_dir),
                      "current": release, "previous": previous, "wheel_sha256": wheel_hash,
                      "owned_files": {str(path): digest(path) for path in payloads}}
            switch_current(prefix, release)
            atomic_write(prefix / "installation.json", json.dumps(record, indent=2) + "\n")
        except BaseException:
            for path, contents in originals.items():
                if contents is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_write(path, contents.decode(), 0o755 if path == launcher else 0o644)
            if old and old.get("current"):
                switch_current(prefix, old["current"])
            elif (prefix / "current").is_symlink():
                (prefix / "current").unlink()
            raise
        print("Installed COMPAG Annotator. Open it from the applications menu.")
        print(f"Command: {launcher}")
        print("Projects and optional models have their own storage and are preserved during updates.")
        if args.launch:
            subprocess.Popen([str(launcher)], start_new_session=True)


def make_parser():
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    parser = argparse.ArgumentParser(description="Install a local release wheel without changing system Python.")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--wheel-sha256", help="Expected digest from the release SHA256SUMS")
    parser.add_argument("--wheelhouse", type=Path, help="Offline directory containing all pinned dependency wheels")
    parser.add_argument("--download-python", action="store_true", help="Consent to downloading, verifying and installing the pinned official CPython 3.12 Linux x86-64 runtime (size/license in INSTALL_UBUNTU.md)")
    parser.add_argument("--python", help="Existing Python 3.12 executable with venv/ensurepip")
    parser.add_argument("--python-archive", type=Path, help="Locally supplied standalone Python 3.12 install_only tar.gz")
    parser.add_argument("--python-sha256", help="Independently verified upstream SHA-256 for that runtime archive")
    parser.add_argument("--prefix", type=Path, default=data / "compag-annotator-app")
    parser.add_argument("--bin-dir", type=Path, default=Path.home() / ".local/bin")
    parser.add_argument("--desktop-dir", type=Path, default=data / "applications")
    parser.add_argument("--rollback", action="store_true", help="Restore the preceding installed application environment")
    parser.add_argument("--launch", action="store_true")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    os.umask(0o077)
    try:
        if sys.version_info < (3, 10):
            raise ValueError("Run the installer with Python 3.10 or newer.")
        if os.geteuid() == 0:
            raise ValueError("Run as your ordinary desktop user, without sudo.")
        if sum(bool(x) for x in (args.python, args.python_archive, args.download_python)) > 1 or (args.download_python and args.python_sha256):
            raise ValueError("Choose exactly one runtime route: --python, --python-archive with digest, or --download-python.")
        install(args)
        return 0
    except KeyboardInterrupt:
        print("Installation cancelled. No incomplete Python download was promoted.", file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.CalledProcessError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"Installation stopped: {exc}", file=sys.stderr)
        print("Any previously selected application version remains available; no project deletion was requested.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
