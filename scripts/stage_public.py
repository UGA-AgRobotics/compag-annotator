#!/usr/bin/env python3
"""Fail-closed source allowlist, content scan and wheel/sdist inspection.

This is an engineering gate, not a determination of ownership or a guarantee
that arbitrary natural-language data is public. Review rights separately.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

TOP_FILES = {
    "pyproject.toml", "MANIFEST.in", ".gitignore", "LICENSE", "README.md",
    "INSTALL_UBUNTU.md", "USER_GUIDE.md", "TRAINING_AND_ROUNDS.md", "MODEL_SETUP.md",
    "EXPORT_FORMATS.md", "TROUBLESHOOTING.md", "ARCHITECTURE.md", "DEVELOPMENT.md",
    "KNOWN_LIMITATIONS.md", "CHANGELOG.md", "THIRD_PARTY_NOTICES.md", "LICENSE_REVIEW.md",
    "SECURITY.md", "CONTRIBUTING.md", "CITATION.cff", "SOURCE_REUSE_MAP.md", "MIGRATION_NOTES.md",
    "FEATURE_STATUS.json", "ACCEPTANCE_RESULTS.json", "PUBLICATION_CHECKLIST.md",
    "SOURCE_MANIFEST.json", "SHA256SUMS", "ADDENDUM_01_IMPLEMENTATION.md",
    "RELEASE_REPORT_EN.md",
}
SOURCE_SUFFIXES = {".py", ".js", ".css", ".html", ".svg"}
FORBIDDEN_PARTS = {"private", ".private", "preserved_source", "baseline_src", "historical", "inputs", "datasets", "checkpoints", "__pycache__", "node_modules"}
IGNORED_DIRS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist", "public_release", "test-results", "playwright-report"}
# Preserve prior local reports, but omit them from the English application release.
LOCAL_ONLY_FILES = {"docs/INTEGRATION_CONTRACT.md", "README_FA.md", "FINAL_REPORT_FA.md"}
MAX_FILE_BYTES = 8 * 1024**2
MAX_TOTAL_BYTES = 128 * 1024**2
MAX_MEMBERS = 20000
PATTERNS = [
    ("private_absolute_path", re.compile(r"(?:/(?:home|Users|mnt|scratch|work|gpfs|lustre)/[A-Za-z0-9_.-]+|[A-Za-z]:[\\/]Users[\\/][^\s'\"<>]+)", re.I)),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    ("access_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b")),
    ("credential_value", re.compile(r"(?i)[\"']?(?:api[_-]?key|password|access[_-]?token|secret[_-]?key)[\"']?\s*[:=]\s*[\"'][A-Za-z0-9_+=/.-]{12,}[\"']")),
    ("encoded_binary_literal", re.compile(r'"[A-Za-z0-9+/]{512,}={0,2}"')),
    ("embedded_binary", re.compile(r"data:[^\s;,]+;base64,[A-Za-z0-9+/=]{256,}")),
    ("serialized_model", re.compile(r'[\"\'](?:gradient_booster|learner_model_param|state_dict|model_state_dict)[\"\']\s*:\s*[{\[]')),
]


class ScanError(ValueError):
    pass


def safe_name(name):
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts
            or any(ord(c) < 32 for c in name) or any(p.casefold() in FORBIDDEN_PARTS for p in path.parts)):
        raise ScanError("unsafe_or_private_member_name")
    return path


def allowed_source(name):
    path = safe_name(name)
    if len(path.parts) == 1:
        return name in TOP_FILES
    if path.parts[:2] == ("src", "compag_annotator"):
        return path.suffix in SOURCE_SUFFIXES or (path.parts[2:4] == ("models", "requirements") and path.suffix == ".txt") or (path.parts[2:3] == ("help",) and path.suffix == ".md") or (path.parts[2:4] == ("web", "assets") and path.suffix == ".json")
    if path.parts[0] == "tests":
        return path.suffix in {".py", ".js", ".json", ".md"}
    if path.parts[0] == "docs":
        return name != "docs/INTEGRATION_CONTRACT.md" and path.suffix in {".md", ".json"}
    if path.parts[0] == "scripts":
        return path.suffix in {".py", ".sh"}
    if path.parts[0] == "locks":
        return path.suffix in {".txt", ".json", ".toml", ".lock", ".in"}
    return path.parts[0] == ".github" and path.suffix in {".md", ".yml", ".yaml"}


def scan_content(name, data, denied=()):
    if len(data) > MAX_FILE_BYTES:
        raise ScanError("oversized_member")
    if b"\0" in data:
        raise ScanError("binary_content")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScanError("non_utf8_content") from exc
    for label, pattern in PATTERNS:
        if pattern.search(text):
            raise ScanError(label)
    if any(value and value.casefold() in text.casefold() for value in denied):
        raise ScanError("private_denylist_match")
    if name.endswith(".svg") and re.search(r"<(?:script|foreignObject)\b|(?:href|src)\s*=\s*[\"'](?:https?:|data:)", text, re.I):
        raise ScanError("active_or_external_svg")
    if name.endswith(".json"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ScanError("invalid_json") from exc
        def inspect(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key.lower() in {"gradient_booster", "learner_model_param", "state_dict", "model_state_dict", "optimizer_state_dict"}:
                        raise ScanError("learned_parameters")
                    if key.lower() in {"annotations", "images", "trees", "weights", "training_data"} and isinstance(child, (list, dict)) and child:
                        raise ScanError("possible_dataset_or_learned_parameters")
                    inspect(child)
            elif isinstance(value, list):
                if len(value) > 64 and all(isinstance(child, (int, float)) for child in value):
                    raise ScanError("possible_numeric_model_or_data_array")
                for child in value:
                    inspect(child)
        inspect(obj)
    return hashlib.sha256(data).hexdigest()


def source_files(root, denied=()):
    """Inspect all candidates, never follow symlinks, report unexpected source files."""
    records = []
    total = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        parent = Path(folder)
        kept = []
        for name in sorted(dirs):
            path = parent / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ScanError(f"{rel}: symlink_directory")
            if name in IGNORED_DIRS or name.endswith(".egg-info"):
                continue
            safe_name(rel)
            kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            path = parent / name
            rel = path.relative_to(root).as_posix()
            if rel in LOCAL_ONLY_FILES:
                continue
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise ScanError(f"{rel}: not_a_regular_file")
            if not allowed_source(rel):
                raise ScanError(f"{rel}: not_allowlisted")
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ScanError(f"{rel}: oversized_member")
            data = path.read_bytes()
            try:
                sha = scan_content(rel, data, denied)
            except ScanError as exc:
                raise ScanError(f"{rel}: {exc}") from exc
            total += len(data)
            if total > MAX_TOTAL_BYTES or len(records) >= MAX_MEMBERS:
                raise ScanError("source_size_limit")
            records.append({"path": rel, "bytes": len(data), "sha256": sha})
    if not any(r["path"] == "pyproject.toml" for r in records):
        raise ScanError("missing_pyproject")
    return records


def _artifact_source_path(name, kind):
    path = safe_name(name)
    if kind == "wheel":
        if path.parts[0] == "compag_annotator":
            rel = "src/" + name
            if not allowed_source(rel):
                raise ScanError("unexpected_package_member")
            return rel
        if len(path.parts) >= 2 and re.fullmatch(r"compag_annotator-[\w.+]+\.dist-info", path.parts[0]):
            if len(path.parts) == 2 and path.name in {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt"}:
                return None
            if path.parts[1:2] == ("licenses",) and path.name in {"LICENSE", "LICENSE_REVIEW.md", "THIRD_PARTY_NOTICES.md"}:
                return path.name
        raise ScanError("unexpected_wheel_member")
    if not re.fullmatch(r"compag[_-]annotator-[\w.+]+", path.parts[0]):
        raise ScanError("unexpected_sdist_root")
    rel = PurePosixPath(*path.parts[1:]).as_posix()
    if rel in {"PKG-INFO", "setup.cfg"} or re.fullmatch(r"src/compag_annotator\.egg-info/(?:PKG-INFO|SOURCES.txt|dependency_links.txt|entry_points.txt|requires.txt|top_level.txt)", rel):
        return None
    if not allowed_source(rel):
        raise ScanError("unexpected_sdist_member")
    return rel


def scan_artifact(artifact, records=None, denied=()):
    expected = {r["path"]: r["sha256"] for r in records or []}
    seen = set()
    total = 0
    members = []
    def inspect(name, data, kind):
        nonlocal total
        if name in seen or len(seen) >= MAX_MEMBERS:
            raise ScanError("duplicate_or_excessive_archive_members")
        seen.add(name)
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ScanError("archive_size_limit")
        rel = _artifact_source_path(name, kind)
        identity = scan_content(name, data, denied)
        if rel and expected and expected.get(rel) != identity:
            raise ScanError(f"{name}: build_source_identity_mismatch")
        members.append({"path": name, "bytes": len(data), "sha256": identity})
    if artifact.stat().st_size > MAX_TOTAL_BYTES:
        raise ScanError("artifact_file_size_limit")
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as archive:
            if archive.comment:
                raise ScanError("unexpected_archive_comment")
            if len(archive.infolist()) > MAX_MEMBERS or sum(i.file_size for i in archive.infolist()) > MAX_TOTAL_BYTES:
                raise ScanError("archive_size_limit")
            for info in archive.infolist():
                if info.is_dir():
                    safe_name(info.filename)
                    continue
                if stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1 or info.file_size > MAX_FILE_BYTES or info.comment:
                    raise ScanError("unsafe_zip_member")
                inspect(info.filename, archive.read(info), "wheel")
    elif artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact, "r|gz") as archive:
            for member_number, info in enumerate(archive, 1):
                if member_number > MAX_MEMBERS:
                    raise ScanError("excessive_archive_members")
                safe_name(info.name)
                if info.isdir():
                    continue
                if not info.isfile() or info.size > MAX_FILE_BYTES:
                    raise ScanError("unsafe_tar_member")
                with archive.extractfile(info) as stream:
                    inspect(info.name, stream.read(MAX_FILE_BYTES + 1), "sdist")
    else:
        raise ScanError("expected_wheel_or_sdist")
    if not members:
        raise ScanError("empty_artifact")
    if artifact.suffix == ".whl":
        required = {"compag_annotator/cli.py", "compag_annotator/web/assets/index.html", "compag_annotator/assets/icon.svg", "compag_annotator/help/index.html"}
        # A source-bound release check must catch omitted new editor/help files,
        # not only stale bytes in files that happened to reach the archive.
        required.update(path.removeprefix("src/") for path in expected
                        if path.startswith(("src/compag_annotator/web/assets/", "src/compag_annotator/help/")))
        if not required <= seen:
            raise ScanError("missing_required_packaged_resource")
    elif expected:
        required = {path for path in expected
                    if path == "ADDENDUM_01_IMPLEMENTATION.md"
                    or path.startswith(("src/compag_annotator/web/assets/", "src/compag_annotator/help/"))}
        included = {_artifact_source_path(name, "sdist") for name in seen}
        if not required <= included:
            raise ScanError("missing_required_packaged_resource")
    return {"file": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(), "members": members}


def scan_history(root, denied=()):
    """Read reachable Git objects without checkout, changing refs or disclosing data."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout
    refs = git("rev-list", "--all", "--objects").splitlines()
    if not refs:
        raise ScanError("history_has_no_commits")
    count = 0
    for row in refs:
        oid, _, name = row.partition(b" ")
        kind = git("cat-file", "-t", oid.decode()).strip()
        size = int(git("cat-file", "-s", oid.decode()))
        if size > MAX_FILE_BYTES:
            raise ScanError("history_oversized_object")
        if kind == b"blob":
            rel = name.decode("utf-8", errors="strict")
            if rel != "docs/INTEGRATION_CONTRACT.md" and not allowed_source(rel):
                raise ScanError("history_non_allowlisted_file")
            scan_content(rel, git("cat-file", "blob", oid.decode()), denied)
            count += 1
        elif kind in {b"commit", b"tag"}:
            scan_content("git_metadata.txt", git("cat-file", kind.decode(), oid.decode()), denied)
    return {"status": "PASS", "blobs": count, "note": "Names and author identity still require owner review before any future push."}


def stage(root, destination, records, denied=()):
    if destination.is_symlink():
        raise ScanError("staging_destination_must_not_be_a_symlink")
    destination = destination.resolve()
    if destination.exists() or destination.is_symlink() or destination.is_relative_to(root):
        raise ScanError("staging_destination_must_be_new_and_outside_source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".compag-public-", dir=destination.parent))
    try:
        payload_records = [r for r in records if r["path"] not in {"SOURCE_MANIFEST.json", "SHA256SUMS"}]
        for record in records:
            source = root / record["path"]
            if source.is_symlink():
                raise ScanError("source_changed_during_staging")
            data = source.read_bytes()
            if scan_content(record["path"], data, denied) != record["sha256"]:
                raise ScanError("source_changed_during_staging")
            if record["path"] in {"SOURCE_MANIFEST.json", "SHA256SUMS"}:
                continue
            output = temporary / record["path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            output.chmod(0o755 if record["path"].startswith("scripts/") else 0o644)
        metadata = tomllib.loads((temporary / "pyproject.toml").read_text())
        manifest = {"schema_version": 2, "files": payload_records, "publication_authorized": False,
                    "declared_license": metadata.get("project", {}).get("license"),
                    "rights_record": "LICENSE_REVIEW.md" if (temporary / "LICENSE_REVIEW.md").is_file() else None,
                    "generated_files_excluded": ["SOURCE_MANIFEST.json", "SHA256SUMS"]}
        (temporary / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        source_files(temporary, denied)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Scan and optionally copy allowlisted public source; never publishes.")
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--destination", type=Path, help="New directory outside the source; omitted means scan only")
    parser.add_argument("--artifact", type=Path, action="append", default=[], help="Inspect an actual built wheel or sdist; repeatable")
    parser.add_argument("--require-artifacts", action="store_true", help="Fail unless both wheel and sdist were supplied")
    parser.add_argument("--history", action="store_true", help="Also scan reachable Git content and commit/tag metadata")
    parser.add_argument("--deny-file", type=Path, help="Private newline-separated identifiers to reject; never copied or printed")
    parser.add_argument("--report", type=Path, help="Write sanitized JSON report outside source/staging")
    args = parser.parse_args(argv)
    root = args.source.resolve()
    denied = ()
    report = {"schema_version": 1, "publication_authorized": False, "artifacts": [], "git_history": {"status": "NOT_RUN"}}
    try:
        denied = tuple(args.deny_file.read_text().splitlines()) if args.deny_file else ()
        if args.report and (args.report.resolve().is_relative_to(root) or (args.destination and args.report.resolve().is_relative_to(args.destination.resolve()))):
            raise ScanError("report_must_be_outside_source_and_staging")
        records = source_files(root, denied)
        if args.require_artifacts and (not any(p.suffix == ".whl" for p in args.artifact) or not any(p.name.endswith(".tar.gz") for p in args.artifact)):
            raise ScanError("both_actual_wheel_and_sdist_required")
        for artifact in args.artifact:
            report["artifacts"].append(scan_artifact(artifact, records, denied))
        if args.history:
            report["git_history"] = scan_history(root, denied)
        if args.destination:
            stage(root, args.destination, records, denied)
        report.update(status="PASS", source_files=len(records), artifact_gate="PASS" if args.require_artifacts else "NOT_REQUIRED_FOR_THIS_SCAN")
    except (ScanError, OSError, ValueError, zipfile.BadZipFile, tarfile.TarError, subprocess.CalledProcessError) as exc:
        # Print only rule descriptions, never matched source snippets, absolute paths or denied values.
        report.update(status="FAIL", error=str(exc) if isinstance(exc, ScanError) else type(exc).__name__)
    if args.report and not args.report.resolve().is_relative_to(root) and not (args.destination and args.report.resolve().is_relative_to(args.destination.resolve())):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "artifacts"}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
