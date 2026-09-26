"""Consented HTTPS downloads with byte progress, integrity checks and atomic promotion."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.parse
import urllib.request
import uuid
import zipfile

from ..providers.protocol import check_cancel, emit, sha256_file
from .files import file_lock, write_json

ALLOWED_HOSTS = {"github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com",
                 "dl.fbaipublicfiles.com", "codeload.github.com"}


def validate_checkpoint_file(path):
    """Only inspect a small header/ZIP directory, never pickle or Torch."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size < 16:
        raise ValueError("Checkpoint is missing, empty or truncated")
    with path.open("rb") as stream:
        header = stream.read(512).lstrip()
    if header.lower().startswith((b"<!doctype", b"<html", b"<?xml", b"<error", b"{", b"version https://git-lfs")):
        raise ValueError("The selected file is text/HTML, not a checkpoint")
    if header.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path) as archive:
                if not any(n.endswith("data.pkl") for n in archive.namelist()):
                    raise ValueError("ZIP contains no PyTorch checkpoint")
        except zipfile.BadZipFile as exc:
            raise ValueError("Truncated checkpoint ZIP") from exc
    elif not header.startswith(b"\x80"):
        raise ValueError("Unsupported checkpoint serialization; select a PyTorch .pt/.pth file")


class _OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _check_url(url):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username or parts.password:
        raise ValueError("Download must use a catalogued official HTTPS host")


def download_checkpoint(entry, destination, *, consent=False, progress=None, cancel=None):
    if entry.get("provider") == "sam3" or not entry.get("checkpoint_download_allowed"):
        raise ValueError("SAM3 weights are local-only; obtain authorized weights externally")
    if consent is not True:
        raise PermissionError("Explicit download consent is required")
    url = entry["url"]
    _check_url(url)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = destination.with_name(destination.name + ".download.json")
    with file_lock(destination.with_suffix(".download.lock"), cancel):
        check_cancel(cancel)
        if destination.exists():
            validate_checkpoint_file(destination)
            measured = sha256_file(destination, cancel)
            if entry.get("sha256") and measured != entry["sha256"]:
                raise ValueError("Existing checkpoint checksum mismatch; remove it explicitly before retrying")
            if not entry.get("sha256"):
                if not receipt_path.is_file():
                    raise ValueError("Existing file has no verified download receipt; register it explicitly as a local checkpoint")
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("sha256") != measured or receipt.get("source") != url:
                    raise ValueError("Existing checkpoint differs from its measured download identity")
            emit(progress, stage="download_verified", bytes=destination.stat().st_size,
                 total_bytes=destination.stat().st_size, percent=100, message="Existing checkpoint verified")
            return {"path": str(destination), "sha256": measured, "bytes": destination.stat().st_size, "cached": True}
        expected = entry.get("size_bytes")
        reserve = 64 * 1024 * 1024
        if shutil.disk_usage(destination.parent).free < (expected or reserve) + reserve:
            raise OSError("Insufficient free disk space for the checkpoint")
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")
        try:
            opener = urllib.request.build_opener(_OfficialRedirect())
            request = urllib.request.Request(url, headers={"User-Agent": "COMPAG-Annotator/1.0", "Accept-Encoding": "identity"})
            emit(progress, stage="download", source=url, bytes=0, total_bytes=expected)
            with opener.open(request, timeout=20) as response, temporary.open("xb") as output:
                if any(t in response.headers.get("Content-Type", "").lower() for t in ("text/html", "text/xml", "application/json")):
                    raise ValueError("Server returned an error/login page instead of weights")
                total = int(response.headers.get("Content-Length", 0)) or expected
                if expected and total != expected:
                    raise ValueError("Checkpoint size differs from the pinned release")
                if total and shutil.disk_usage(destination.parent).free < total + reserve:
                    raise OSError("Insufficient free disk space for the checkpoint")
                digest = hashlib.sha256()
                downloaded = 0
                while True:
                    check_cancel(cancel)
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if total and downloaded > total or downloaded > 16 * 1024**3:
                        raise ValueError("Download exceeds declared size")
                    output.write(chunk)
                    digest.update(chunk)
                    emit(progress, stage="download", bytes=downloaded, total_bytes=total)
                output.flush()
                os.fsync(output.fileno())
            check_cancel(cancel)
            if total and downloaded != total:
                raise ValueError("Checkpoint download interrupted or truncated")
            measured = digest.hexdigest()
            if entry.get("sha256") and measured != entry["sha256"]:
                raise ValueError("Checkpoint checksum does not match official release")
            validate_checkpoint_file(temporary)
            os.replace(temporary, destination)
            write_json(receipt_path, {"source": url, "sha256": measured, "bytes": downloaded,
                                      "digest_authority": entry.get("digest_authority")})
            emit(progress, stage="download_verified", bytes=downloaded, total_bytes=downloaded, percent=100, sha256=measured)
            return {"path": str(destination), "sha256": measured, "bytes": downloaded, "cached": False}
        finally:
            temporary.unlink(missing_ok=True)
