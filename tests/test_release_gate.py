"""Release contracts: new versions, scan completeness and frozen-tree restaging."""
import hashlib
import json
import subprocess
import pytest
from test_packaging import ROOT, scanner, script, fixture_source


def test_release_paths_tracks_changed_version(tmp_path):
    paths = script("release_paths")
    (tmp_path / "pyproject.toml").write_text('[project]\nname="compag-annotator"\nversion="2.3.4rc5"\n')
    result = paths.release_paths(tmp_path, "artifact folder")
    assert result["wheel"] == "artifact folder/compag_annotator-2.3.4rc5-py3-none-any.whl"
    assert result["sdist"] == "artifact folder/compag_annotator-2.3.4rc5.tar.gz"
    metadata = paths.release_paths(ROOT)
    from compag_annotator import __version__
    assert metadata["version"] == __version__


def test_release_report_is_reviewed_not_an_open_allowlist(tmp_path):
    root = fixture_source(tmp_path)
    (root / "RELEASE_REPORT_EN.md").write_text("Synthetic public summary.\n")
    assert any(r["path"] == "RELEASE_REPORT_EN.md" for r in scanner.source_files(root))
    (root / "PRIVATE_RELEASE_LOG.md").write_text("Do not stage.")
    with pytest.raises(scanner.ScanError, match="not_allowlisted"):
        scanner.source_files(root)


def test_restage_never_hashes_old_manifest_or_checksums(tmp_path):
    root = fixture_source(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    scanner.stage(root, first, scanner.source_files(root))
    (first / "SHA256SUMS").write_text("obsolete outer archive checksum\n")
    scanner.stage(first, second, scanner.source_files(first))
    assert not (second / "SHA256SUMS").exists()
    manifest = json.loads((second / "SOURCE_MANIFEST.json").read_text())
    assert manifest["publication_authorized"] is False
    assert not {"SOURCE_MANIFEST.json", "SHA256SUMS"} & {r["path"] for r in manifest["files"]}
    for row in manifest["files"]:
        assert hashlib.sha256((second / row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_unborn_history_is_not_a_passing_history_review(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    with pytest.raises(scanner.ScanError, match="history_has_no_commits"):
        scanner.scan_history(tmp_path)


def test_old_runtime_cannot_be_relabelled_with_new_security_pins(tmp_path, monkeypatch):
    from compag_annotator.models import runtime
    from compag_annotator.models.catalog import PROVIDERS
    root = tmp_path / "runtimes/sam2"
    python = root / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    receipt = {"source_revision":PROVIDERS["sam2"]["source_revision"],
        "lock_hash":"ca87879451ca170ac6df33b03885b7a21706a71e658ba1aeda6d45012bce99df"}
    path = root / "installed.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.delenv("COMPAG_SAM2_PYTHON", raising=False)
    monkeypatch.setattr(runtime, "run_command", lambda *a, **k: pytest.fail("Old runtime mutated"))
    monkeypatch.setattr(runtime, "_repair_adapters", lambda *a, **k: pytest.fail("Unsafe lock relabel"))
    with pytest.raises(RuntimeError, match="another lock/flavor"):
        runtime.install_runtime(tmp_path, "sam2", consent=True, flavor="cpu")
    assert json.loads(path.read_text()) == receipt


def test_provider_security_pins_match_packaged_locks():
    import re
    minimum_pins = {"pillow":"12.3.0", "setuptools":"84.0.0", "wheel":"0.48.0"}
    for lock in (ROOT / "locks").glob("provider-*.txt"):
        if lock.name == "provider-adapters.txt":
            continue
        assert lock.read_bytes() == (ROOT / "src/compag_annotator/models/requirements" / lock.name).read_bytes()
        pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", lock.read_text(), re.M))
        assert all(pins[k] == v for k, v in minimum_pins.items())
        if "sam2" in lock.name:
            assert pins["hydra-core"] == "1.3.4"
