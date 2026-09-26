"""Packaging/CLI and hostile-input checks; no ML runtimes or checkpoints."""
from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.request
import zipfile

import pytest
from compag_annotator import __version__

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


installer = script("install_ubuntu")
uninstaller = script("uninstall_ubuntu")
scanner = script("stage_public")


def package_environment():
    settings = dict(os.environ)
    if os.environ.get("COMPAG_TEST_INSTALLED_PYTHON"):
        settings.pop("PYTHONPATH", None)
    else:
        settings["PYTHONPATH"] = str(ROOT / "src")
    return settings


def invoke(*args, cwd=None, env=None):
    settings = package_environment()
    settings.update(env or {})
    return subprocess.run([sys.executable, "-m", "compag_annotator.cli", *args],
                          cwd=cwd, env=settings, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("option", ["--help", "--version", "doctor"])
def test_cli_without_weights(option, tmp_path):
    args = [option] if option != "doctor" else ["doctor", "--data-dir", str(tmp_path / "data"), "--json"]
    result = invoke(*args, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert ("usage:" if option == "--help" else __version__) in result.stdout
    assert str(ROOT) not in result.stdout
    if option == "doctor":
        report = json.loads(result.stdout)
        assert report["core_ready"] is True
        assert report["provider_probe"] == "metadata_only"


def test_cli_import_and_doctor_never_import_ml(tmp_path):
    code = """
import importlib.abc,json,sys
class NoML(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in {'torch','torchvision','ultralytics','sam2','sam3'}:
            raise AssertionError('ML import attempted')
sys.meta_path.insert(0,NoML())
from compag_annotator.cli import doctor
from pathlib import Path
result=doctor(Path(sys.argv[1]))
assert result['core_ready']
assert not {'torch','ultralytics','sam2','sam3'} & sys.modules.keys()
print(json.dumps(result))
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            env=package_environment(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("args", [["serve", "--qa-mode"], ["--qa-mode", "serve"], ["--qa-mode"]])
def test_qa_flag_survives_subcommand(args):
    from compag_annotator.cli import parser
    assert parser().parse_args(args).qa_mode is True


@pytest.mark.parametrize("args", [["serve", "--host", "0.0.0.0"], ["serve", "--port", "65536"], ["serve", "--port", "-1"]])
def test_reject_remote_binding_and_invalid_port(args):
    assert invoke(*args).returncode == 2


def test_actual_loopback_server_qa_actor(tmp_path):
    log = tmp_path / "server.log"
    with log.open("w+") as stream:
        process = subprocess.Popen([sys.executable, "-u", "-m", "compag_annotator.cli", "serve", "--no-browser", "--qa-mode", "--data-dir", str(tmp_path / "appdata")],
                                   env=package_environment(), stdout=stream, stderr=stream)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            deadline = time.monotonic() + 15
            url = None
            while time.monotonic() < deadline:
                text = log.read_text()
                for line in text.splitlines():
                    if "http://127.0.0.1:" in line:
                        url = line.split(": ", 1)[1]
                if url:
                    try:
                        with opener.open(url + "/api/session", timeout=0.5) as response:
                            session = json.load(response)
                        break
                    except OSError:
                        pass
                if process.poll() is not None:
                    pytest.fail("Server exited: " + text)
                time.sleep(0.1)
            else:
                pytest.fail("Server did not become ready")
            assert session["actor"] == "automated_qa"
            assert session["token"] not in log.read_text()
            with opener.open(url + "/", timeout=2) as response:
                assert b"COMPAG" in response.read()
            for asset in ("index.html", "USER_GUIDE.md", "ADDENDUM_01_IMPLEMENTATION.md"):
                with opener.open(url + "/help/" + asset, timeout=2) as response:
                    assert response.read() == (ROOT / "src/compag_annotator/help" / asset).read_bytes()
        finally:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_backup_uses_real_native_format_preserves_project(tmp_path):
    from compag_annotator.cli import backup_project
    from compag_annotator.core.projects import Project
    from compag_annotator.core.catalog import Catalog
    project = Project.create(tmp_path / "project", "Fixture", [{"name":n} for n in ("One", "Two", "Three")])
    before = project.state()
    output = tmp_path / "native.zip"
    backup_project(project.path, output)
    assert project.state() == before
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(archive.read("native_manifest.json"))
        assert manifest["model_weights_included"] is False
        assert "project.sqlite3" in archive.namelist()
    restored = Catalog(tmp_path / "restored").restore(output)
    assert restored.state()["classes"] == before["classes"]
    assert restored.state()["id"] != before["id"]
    with pytest.raises(ValueError, match="already exists"):
        backup_project(project.path, output)
    with pytest.raises(ValueError, match="outside"):
        backup_project(project.path, project.path / "recursive.zip")


def test_diagnostics_redact_nested_paths_and_secret():
    from compag_annotator.cli import _redact
    personal = "/" + "home" + "/" + "sample-user" + "/private/checkpoint"
    result = _redact({"model_path":personal,"nested":[{"message":personal + " password=sample-secret-value"}]})
    serialized = json.dumps(result)
    assert personal not in serialized
    assert "sample-secret-value" not in serialized


def test_metadata_is_small_pinned_and_licensed():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["version"] == __version__
    assert metadata["project"]["requires-python"] == ">=3.12"
    assert metadata["project"]["license"] == "AGPL-3.0-only"
    assert "Private :: Do Not Upload" not in metadata["project"]["classifiers"]
    deps = metadata["project"]["dependencies"]
    assert all("==" in d for d in deps)
    assert not any(any(n in d.lower() for n in ("torch", "ultralytics", "sam2", "sam3")) for d in deps)
    for asset in ("assets/icon.svg", "help/index.html", "web/assets/index.html", "models/requirements/provider-sam2-cpu.txt"):
        assert (ROOT / "src/compag_annotator" / asset).is_file()


def test_help_uses_only_local_resources():
    page = (ROOT / "src/compag_annotator/help/index.html").read_text()
    assert '<script src="http' not in page
    assert 'href="http' not in page
    for key in ("Projects", "Images", "Annotate", "Export", "Models &amp; AI", "Train", "Rounds"):
        assert key in page


def test_application_owned_text_is_english_only():
    package = ROOT / "src/compag_annotator"
    # Check authored resources only, never the user's classes, names or annotations.
    arabic_script = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u0870-\u089f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]")
    failures = []
    for path in package.rglob("*"):
        if path.suffix in {".py", ".js", ".html", ".css", ".md", ".json", ".svg", ".txt"}:
            if arabic_script.search(path.read_text(encoding="utf-8")):
                failures.append(path.relative_to(package).as_posix())
    assert not failures, failures
    assert not (package / "help/README_FA.md").exists()


def test_archived_local_guides_are_not_distributed(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fixture'\n")
    for name in ("README_FA.md", "FINAL_REPORT_FA.md"):
        (tmp_path / name).write_text("Archived local report")
    records = scanner.source_files(tmp_path)
    assert {record["path"] for record in records} == {"pyproject.toml"}
    for name in ("README_FA.md", "FINAL_REPORT_FA.md"):
        assert (tmp_path / name).is_file()
        assert not scanner.allowed_source(name)
    assert "exclude README_FA.md FINAL_REPORT_FA.md" in (ROOT / "MANIFEST.in").read_text()


def test_addendum_offline_guides_match_public_text():
    for source in ("USER_GUIDE.md", "TRAINING_AND_ROUNDS.md", "MODEL_SETUP.md",
                   "KNOWN_LIMITATIONS.md", "docs/QUICKSTART.md",
                   "ADDENDUM_01_IMPLEMENTATION.md"):
        public = ROOT / source
        packaged = ROOT / "src/compag_annotator/help" / public.name
        assert packaged.read_bytes() == public.read_bytes(), source


def test_help_navigation_targets_are_packaged_local_files():
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.targets.extend(value for name, value in attrs if name == "href")

    directory = ROOT / "src/compag_annotator/help"
    parser = Links()
    parser.feed((directory / "index.html").read_text())
    assert parser.targets
    for target in parser.targets:
        if target == "/":  # The existing application entry point.
            continue
        assert ":" not in target and not target.startswith("/")
        path = (directory / target).resolve()
        assert path.is_relative_to(directory.resolve()) and path.is_file(), target


def test_current_editor_and_help_assets_match_wheel_and_sdist_rules():
    """Inspect packaging selection without building a release or importing ML."""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package = ROOT / "src/compag_annotator"
    patterns = metadata["tool"]["setuptools"]["package-data"]["compag_annotator"]
    selected = {path for pattern in patterns for path in package.glob(pattern) if path.is_file()}
    assets = {path for folder in ("help", "web/assets", "assets")
              for path in (package / folder).rglob("*") if path.is_file()}
    assert package / "help/ADDENDUM_01_IMPLEMENTATION.md" in assets
    assert assets <= selected, sorted(str(path.relative_to(package)) for path in assets - selected)
    assert all(scanner.allowed_source(path.relative_to(ROOT).as_posix()) for path in assets)
    # Check declarations without requiring the isolated build backend in core.
    # Actual artifact inclusion is checked separately against source hashes.
    manifest = (ROOT / "MANIFEST.in").read_text().splitlines()
    assert "include *.md LICENSE *.json" in manifest
    assert "recursive-include src/compag_annotator/help *.md" in manifest
    assert "recursive-include src/compag_annotator *.py *.html *.css *.js *.svg" in manifest
    assert {path.suffix for path in assets} <= {".md", ".html", ".css", ".js", ".svg"}


def test_shell_launcher_quotes_paths_and_arguments(tmp_path):
    prefix = tmp_path / "app with spaces ' and $dollar"
    executable = prefix / "current/bin/python"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    executable.chmod(0o755)
    launcher = tmp_path / "launcher"
    launcher.write_text(installer.command_wrapper(prefix))
    launcher.chmod(0o755)
    result = subprocess.run([launcher, "--data-dir", "two words", "literal$(value)"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["-m", "compag_annotator.cli", "--data-dir", "two words", "literal$(value)"]
    desktop = installer.desktop_entry(launcher, tmp_path / "icon.svg")
    assert "Terminal=false" in desktop and f'Exec="{launcher}"' in desktop


def test_desktop_escape_and_reject_percent():
    assert installer.desktop_quote('a"b$c`d\\e') == '"a\\\\"b\\\\$c\\\\`d\\\\\\\\e"'
    with pytest.raises(ValueError):
        installer.desktop_quote("bad%f")


def test_python_version_checked_before_use(tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nprintf '3.11\\n'\n")
    fake.chmod(0o755)
    with pytest.raises(ValueError, match="Python 3.12"):
        installer.check_python(fake)


def runtime_archive(path, members):
    with tarfile.open(path, "w:gz") as archive:
        for name, content, link in members:
            info = tarfile.TarInfo(name)
            if link:
                info.type = tarfile.SYMTYPE
                info.linkname = link
                archive.addfile(info)
            else:
                info.size = len(content)
                info.mode = 0o755
                archive.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("members", [
    [("../escape", b"x", None)],
    [("python/bin/python3.12", b"", "../../../escape")],
    [("python/bin", b"", "lib"), ("python/bin/file", b"x", None)],
])
def test_bootstrap_rejects_traversal_and_link_writes(tmp_path, members):
    archive = tmp_path / "runtime.tar.gz"
    runtime_archive(archive, members)
    with pytest.raises(ValueError):
        installer.extract_python(archive, tmp_path / "extract")
    assert not (tmp_path / "escape").exists()


def test_bootstrap_accepts_internal_relative_link(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    runtime_archive(archive, [("python/bin/python3.12", b"fixture", None), ("python/bin/python3", b"", "python3.12")])
    target = tmp_path / "extract"
    installer.extract_python(archive, target)
    assert (target / "python/bin/python3").read_bytes() == b"fixture"


def test_bootstrap_digest_mismatch_does_not_extract(tmp_path):
    archive = tmp_path / "runtime.tar.gz"
    archive.write_bytes(b"not a runtime")
    args = installer.make_parser().parse_args(["--python-archive", str(archive), "--python-sha256", "0" * 64])
    with pytest.raises(ValueError, match="mismatch"):
        installer.prepare_python(args, tmp_path / "prefix")
    assert not (tmp_path / "prefix").exists()


def test_install_lock_excludes_parallel_mutation(tmp_path):
    with installer.install_lock(tmp_path / "prefix"):
        with pytest.raises(ValueError, match="still running"):
            with installer.install_lock(tmp_path / "prefix"):
                pass


def test_rollback_switches_current_without_moving_environment(tmp_path):
    prefix = tmp_path / "prefix"
    for name in ("a" * 16, "b" * 16):
        python = prefix / "releases" / name / "bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("fixture")
    record = {"schema":installer.MARKER,"prefix":str(prefix),"current":"b" * 16,"previous":"a" * 16}
    (prefix / "installation.json").write_text(json.dumps(record))
    installer.switch_current(prefix, record["current"])
    args = installer.make_parser().parse_args(["--prefix", str(prefix), "--rollback"])
    installer.install(args)
    assert (prefix / "current").resolve() == prefix / "releases" / ("a" * 16)
    assert json.loads((prefix / "installation.json").read_text())["previous"] == "b" * 16


def managed_install_fixture(tmp_path):
    prefix = tmp_path / "code"
    prefix.mkdir()
    command = tmp_path / "bin/compag-annotator"
    command.parent.mkdir()
    command.write_text("owned launcher")
    record = {"schema":installer.MARKER,"prefix":str(prefix),"bin_dir":str(command.parent),"desktop_dir":str(tmp_path / "desktop"),"owned_files":{str(command):installer.digest(command)}}
    (prefix / "installation.json").write_text(json.dumps(record))
    return prefix, command


def test_uninstall_preserves_projects_and_refuses_unknown_files(tmp_path):
    prefix, command = managed_install_fixture(tmp_path)
    project = tmp_path / "projects/data.txt"
    project.parent.mkdir()
    project.write_text("keep")
    (prefix / "unexpected.txt").write_text("do not remove")
    with pytest.raises(ValueError, match="extra files"):
        uninstaller.uninstall(prefix)
    assert command.exists()
    (prefix / "unexpected.txt").unlink()
    uninstaller.uninstall(prefix)
    assert not prefix.exists() and not command.exists()
    assert project.read_text() == "keep"


def test_uninstall_refuses_changed_launcher(tmp_path):
    prefix, command = managed_install_fixture(tmp_path)
    command.write_text("user edit")
    with pytest.raises(ValueError, match="modified"):
        uninstaller.uninstall(prefix)
    assert command.read_text() == "user edit"


def fixture_source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="compag-annotator"\n')
    (root / "README.md").write_text("Synthetic fixture.\n")
    return root


def test_staging_copies_allowlist_and_creates_manifest(tmp_path):
    root = fixture_source(tmp_path)
    (root / ".git").mkdir()
    (root / ".git/secret").write_text("excluded history content")
    records = scanner.source_files(root)
    destination = tmp_path / "public"
    scanner.stage(root, destination, records)
    manifest = json.loads((destination / "SOURCE_MANIFEST.json").read_text())
    assert manifest["publication_authorized"] is False
    assert (destination / "README.md").read_bytes() == (root / "README.md").read_bytes()
    assert not (destination / ".git").exists()


def test_addendum_allowlist_is_exact_and_stages_scanned_contents(tmp_path):
    root = fixture_source(tmp_path)
    name = "ADDENDUM_01_IMPLEMENTATION.md"
    public = root / name
    public.write_bytes((ROOT / name).read_bytes())
    assert scanner.allowed_source(name)
    for unknown in ("ADDENDUM_02_IMPLEMENTATION.md", "ADDENDUM_01_PRIVATE.md", "PRIVATE_NOTES.md"):
        assert not scanner.allowed_source(unknown)
    with pytest.raises(scanner.ScanError, match="private"):
        scanner.allowed_source("private/" + name)
    records = scanner.source_files(root)
    destination = tmp_path / "public"
    scanner.stage(root, destination, records)
    assert (destination / name).read_bytes() == public.read_bytes()
    public.write_text("/" + "home" + "/sample-user/annotation-data")
    with pytest.raises(scanner.ScanError, match="private_absolute_path"):
        scanner.source_files(root)


@pytest.mark.parametrize("kind", ["path", "token", "binary", "model"])
def test_content_scan_rejects_private_payloads_even_text_extension(kind):
    values = {
        "path":("README.md", ("/" + "home" + "/sample-user/data").encode()),
        "token":("README.md", ("ghp_" + "A" * 32).encode()),
        "binary":("README.md", b"\x00binary"),
        "model":("data.json", json.dumps({"learner" + "_model_param":{}}).encode()),
    }
    with pytest.raises(scanner.ScanError):
        scanner.scan_content(*values[kind])


def test_json_dataset_is_not_public_by_default():
    data = json.dumps({"images":[{"name":"example"}]}).encode()
    with pytest.raises(scanner.ScanError, match="dataset"):
        scanner.scan_content("fixture.json", data)


def test_source_rejects_symlink_unknown_and_private_folder(tmp_path):
    root = fixture_source(tmp_path)
    (root / "README.md").unlink()
    (root / "README.md").symlink_to(root / "pyproject.toml")
    with pytest.raises(scanner.ScanError, match="regular"):
        scanner.source_files(root)
    (root / "README.md").unlink()
    (root / "payload.bin").write_bytes(b"x")
    with pytest.raises(scanner.ScanError, match="allowlisted"):
        scanner.source_files(root)
    (root / "payload.bin").unlink()
    (root / "private").mkdir()
    with pytest.raises(scanner.ScanError, match="private"):
        scanner.source_files(root)


def test_source_changed_after_scan_fails_staging(tmp_path):
    root = fixture_source(tmp_path)
    records = scanner.source_files(root)
    (root / "README.md").write_text("changed")
    with pytest.raises(scanner.ScanError, match="changed"):
        scanner.stage(root, tmp_path / "public", records)
    assert not (tmp_path / "public").exists()


def fixture_wheel(tmp_path, extra=None):
    path = tmp_path / "compag_annotator-1.0.0rc10-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("cli.py", "web/assets/index.html", "help/index.html", "assets/icon.svg"):
            archive.writestr("compag_annotator/" + name, "fixture")
        if extra:
            archive.writestr(*extra)
    return path


def test_scan_inspects_actual_wheel_content(tmp_path):
    artifact = fixture_wheel(tmp_path)
    result = scanner.scan_artifact(artifact)
    assert len(result["members"]) == 4
    assert result["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    secret = ("hf_" + "B" * 32).encode()
    artifact = fixture_wheel(tmp_path, ("compag_annotator/extra.py", secret))
    with pytest.raises(scanner.ScanError, match="token"):
        scanner.scan_artifact(artifact)


def test_wheel_missing_resource_and_source_mismatch_fail(tmp_path):
    wheel = fixture_wheel(tmp_path)
    with pytest.raises(scanner.ScanError, match="identity_mismatch"):
        scanner.scan_artifact(wheel, [{"path":"src/compag_annotator/cli.py","sha256":"0" * 64}])
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("compag_annotator/cli.py", "fixture")
    with pytest.raises(scanner.ScanError, match="required"):
        scanner.scan_artifact(wheel)


def test_sdist_content_and_traversal_checks(tmp_path):
    artifact = tmp_path / "compag_annotator-1.0.0rc10.tar.gz"
    runtime_archive(artifact, [("compag_annotator-1.0.0rc10/README.md", b"fixture", None)])
    assert scanner.scan_artifact(artifact)["members"][0]["bytes"] == 7
    runtime_archive(artifact, [("../escape", b"fixture", None)])
    with pytest.raises(scanner.ScanError, match="unsafe"):
        scanner.scan_artifact(artifact)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_source_bound_artifact_requires_new_help_and_editor_assets(tmp_path, kind):
    """Archive fixtures test the scanner, not a claim of a real release build."""
    names = ["src/compag_annotator/cli.py", "src/compag_annotator/assets/icon.svg",
             "src/compag_annotator/web/assets/index.html", "src/compag_annotator/web/assets/editor.js",
             "src/compag_annotator/help/index.html", "src/compag_annotator/help/ADDENDUM_01_IMPLEMENTATION.md"]
    if kind == "sdist":
        names.append("ADDENDUM_01_IMPLEMENTATION.md")
    contents = {name:(ROOT / name).read_bytes() for name in names}
    records = [{"path":name, "sha256":hashlib.sha256(data).hexdigest()} for name, data in contents.items()]
    artifact = tmp_path / ("compag_annotator-1.0.0rc10-py3-none-any.whl" if kind == "wheel"
                           else "compag_annotator-1.0.0rc10.tar.gz")

    def write_archive(omit=None, replace=None):
        members = {name:(replace if replace is not None and name.endswith("editor.js") else data)
                   for name, data in contents.items() if name != omit}
        if kind == "wheel":
            with zipfile.ZipFile(artifact, "w") as archive:
                for name, data in members.items():
                    archive.writestr(name.removeprefix("src/"), data)
        else:
            runtime_archive(artifact, [("compag_annotator-1.0.0rc10/" + name, data, None)
                                      for name, data in members.items()])

    write_archive()
    assert len(scanner.scan_artifact(artifact, records)["members"]) == len(names)
    for omit in names:
        if "/help/" in omit or "/web/assets/" in omit or omit == "ADDENDUM_01_IMPLEMENTATION.md":
            write_archive(omit=omit)
            with pytest.raises(scanner.ScanError, match="missing_required_packaged_resource"):
                scanner.scan_artifact(artifact, records)
    write_archive(replace=b"changed editor")
    with pytest.raises(scanner.ScanError, match="identity_mismatch"):
        scanner.scan_artifact(artifact, records)
    write_archive(replace=("hf_" + "Q" * 32).encode())
    with pytest.raises(scanner.ScanError, match="access_token"):
        scanner.scan_artifact(artifact, records)


def test_require_real_artifacts_and_private_denylist(tmp_path):
    root = fixture_source(tmp_path)
    assert scanner.main(["--source", str(root), "--require-artifacts"]) == 1
    with pytest.raises(scanner.ScanError, match="denylist"):
        scanner.scan_content("README.md", b"sample private identifier", ["private identifier"])


def test_actual_backup_command(tmp_path):
    from compag_annotator.core.projects import Project
    project = Project.create(tmp_path / "project with spaces", "Backup fixture", [{"name":"One"}])
    output = tmp_path / "backup file.zip"
    result = invoke("backup", str(project.path), "--output", str(output), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(output) as archive:
        assert json.loads(archive.read("native_manifest.json"))["model_weights_included"] is False
    assert invoke("backup", str(project.path), "--output", str(output)).returncode == 1


def test_failed_first_install_is_retryable(tmp_path, monkeypatch):
    wheel = fixture_wheel(tmp_path)
    args = installer.make_parser().parse_args(["--wheel", str(wheel), "--prefix", str(tmp_path / "install"),
                                               "--bin-dir", str(tmp_path / "commands"), "--desktop-dir", str(tmp_path / "desktop")])
    monkeypatch.setattr(installer, "prepare_python", lambda *args: Path(sys.executable))
    calls = []
    def interrupted(command, **kwargs):
        calls.append(command)
        if "venv" in command:
            Path(command[-1]).mkdir(parents=True)
        else:
            raise subprocess.CalledProcessError(1, "fixture interrupted dependency install")
    monkeypatch.setattr(installer, "_run", interrupted)
    for _ in range(2):
        with pytest.raises(subprocess.CalledProcessError):
            installer.install(args)
        assert (args.prefix / "installation.json").is_file()
        assert not (args.prefix / "current").exists()
    assert len(calls) == 4


def test_numeric_json_arrays_and_encoded_payloads_are_blocked():
    with pytest.raises(scanner.ScanError, match="numeric"):
        scanner.scan_content("values.json", json.dumps({"coefficients":[0.5] * 70}).encode())
    with pytest.raises(scanner.ScanError, match="encoded"):
        scanner.scan_content("constants.py", ('value="' + 'Q' * 600 + '"').encode())


def test_staging_destination_parent_symlink_cannot_reenter_source(tmp_path):
    root = fixture_source(tmp_path)
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(scanner.ScanError, match="outside_source"):
        scanner.stage(root, link / "public", scanner.source_files(root))
    assert not (root / "public").exists()


def test_missing_private_denyfile_fails_without_disclosing_path(tmp_path, capsys):
    root = fixture_source(tmp_path)
    secret_name = tmp_path / "private-identity-list"
    assert scanner.main(["--source", str(root), "--deny-file", str(secret_name)]) == 1
    assert str(secret_name) not in capsys.readouterr().out


class PythonDownloadFixture(io.BytesIO):
    status = 200

    def __init__(self, payload, headers=None):
        super().__init__(payload)
        self.headers = headers if headers is not None else {"Content-Type":"application/octet-stream", "Content-Length":str(len(payload))}


def python_download_fixture():
    import gzip
    payload = gzip.compress(b"Synthetic Python archive fixture; not an executable runtime.", mtime=0)
    pin = installer.load_bootstrap_pin()
    pin.update(size_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    return pin, payload


def test_bootstrap_pin_binds_official_release_asset():
    pin = installer.load_bootstrap_pin()
    assert pin["python_version"].startswith("3.12.")
    assert pin["release_immutable"] is True
    assert pin["asset_api_url"].endswith(str(pin["asset_id"]))
    assert pin["release_commit"] and len(pin["release_commit"]) == 40
    assert pin["sha256"] and len(pin["sha256"]) == 64
    assert pin["asset_url"].startswith("https://github.com/astral-sh/python-build-standalone/releases/download/" + pin["release_tag"] + "/")


def test_download_requires_explicit_consent_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "_open_python_download", lambda *_: pytest.fail("Unexpected network access"))
    pin, _ = python_download_fixture()
    with pytest.raises(ValueError, match="explicit consent"):
        installer.download_python(tmp_path, pin)
    monkeypatch.setattr(installer.shutil, "which", lambda *_: None)
    with pytest.raises(ValueError, match="--download-python"):
        installer.prepare_python(installer.make_parser().parse_args([]), tmp_path)
    assert not (tmp_path / "runtimes").exists()


def test_download_verifies_and_reuses_cache_offline(tmp_path, monkeypatch):
    pin, payload = python_download_fixture()
    monkeypatch.setattr(installer, "_open_python_download", lambda *_: PythonDownloadFixture(payload))
    archive = installer.download_python(tmp_path, pin, consent=True)
    assert archive.read_bytes() == payload
    assert not list(archive.parent.glob("*.partial"))
    monkeypatch.setattr(installer, "_open_python_download", lambda *_: pytest.fail("Cached runtime should not use network"))
    assert installer.download_python(tmp_path, pin, consent=True) == archive
    archive.write_bytes(b"corrupted cache")
    with pytest.raises(ValueError, match="Cached Python archive"):
        installer.download_python(tmp_path, pin, consent=True)


@pytest.mark.parametrize("problem", ["offline", "interrupt", "digest", "short", "oversize", "html", "gzip", "length"])
def test_python_download_failure_never_promotes_archive(tmp_path, monkeypatch, problem):
    import urllib.error
    pin, payload = python_download_fixture()
    headers = {"Content-Type":"application/octet-stream"}
    if problem == "digest":
        pin["sha256"] = "0" * 64
    elif problem == "short":
        payload = payload[:-4]
    elif problem == "oversize":
        payload += b"extra"
    elif problem == "html":
        headers["Content-Type"] = "text/html; charset=utf-8"
        payload = b"<html>error</html>"
    elif problem == "gzip":
        payload = b"not a gzip archive"
    elif problem == "length":
        headers["Content-Length"] = str(pin["size_bytes"] + 1)
    class Interrupted(PythonDownloadFixture):
        def read(self, size=-1):
            raise KeyboardInterrupt()
    def open_fixture(_):
        if problem == "offline":
            raise urllib.error.URLError("Synthetic offline failure")
        return Interrupted(payload, headers) if problem == "interrupt" else PythonDownloadFixture(payload, headers)
    monkeypatch.setattr(installer, "_open_python_download", open_fixture)
    expected = KeyboardInterrupt if problem == "interrupt" else ValueError
    with pytest.raises(expected):
        installer.download_python(tmp_path, pin, consent=True)
    assert not (tmp_path / "runtimes/downloads" / pin["asset_name"]).exists()
    assert not list((tmp_path / "runtimes/downloads").glob("*.partial"))
    assert not list((tmp_path / "runtimes").glob(".extract-*"))


def test_download_rejects_other_platform_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(installer, "_open_python_download", lambda *_: pytest.fail("Unexpected network"))
    with pytest.raises(ValueError, match="Linux x86-64"):
        installer.download_python(tmp_path, installer.load_bootstrap_pin(), consent=True)


def test_download_pin_cannot_redirect_to_unrelated_source(tmp_path, monkeypatch):
    pin = installer.load_bootstrap_pin()
    pin["asset_url"] = "https://example.invalid/runtime.tar.gz"
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps(pin))
    monkeypatch.setattr(installer, "bootstrap_lock_path", lambda: lock)
    with pytest.raises(ValueError, match="official release"):
        installer.load_bootstrap_pin()


@pytest.mark.parametrize("url", ["http://github.com/runtime", "https://example.invalid/runtime", "https://github.com:444/runtime"])
def test_download_redirect_is_https_and_github_only(url):
    import urllib.request
    request = urllib.request.Request(installer.load_bootstrap_pin()["asset_url"])
    with pytest.raises(ValueError, match="permitted HTTPS"):
        installer._PythonDownloadRedirect().redirect_request(request, None, 302, "Found", {}, url)


def test_download_option_cannot_mix_runtime_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(installer, "download_python", lambda *_args, **_kwargs: pytest.fail("Unexpected download"))
    args = installer.make_parser().parse_args(["--download-python", "--python", "python3.12"])
    with pytest.raises(ValueError, match="by itself"):
        installer.prepare_python(args, tmp_path)


def test_source_allows_scanned_browser_test_documentation(tmp_path):
    root = fixture_source(tmp_path)
    doc = root / "tests/browser/README.md"
    doc.parent.mkdir(parents=True)
    doc.write_text("Synthetic browser test instructions.\n")
    assert any(record["path"] == "tests/browser/README.md" for record in scanner.source_files(root))


def test_installed_helper_pin_and_uninstall_cleanup(tmp_path, monkeypatch):
    prefix, _ = managed_install_fixture(tmp_path)
    pin = installer.load_bootstrap_pin()
    (prefix / "python-bootstrap.json").write_text(json.dumps(pin))
    monkeypatch.setattr(installer, "__file__", str(prefix / "install_ubuntu.py"))
    assert installer.load_bootstrap_pin() == pin
    uninstaller.uninstall(prefix)
    assert not prefix.exists()
