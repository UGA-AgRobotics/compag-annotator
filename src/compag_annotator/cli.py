"""Small local CLI. Importing this module never loads a model runtime."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import secrets
import socket
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser

from . import __version__

CORE_DISTRIBUTIONS = (
    "fastapi", "uvicorn", "Pillow", "numpy", "shapely", "pycocotools", "PyYAML",
    "defusedxml", "python-multipart", "platformdirs",
)


def _redact(value):
    """Diagnostics are shareable: drop paths, environment values and credentials."""
    if isinstance(value, dict):
        return {str(k): "[redacted]" if any(word in str(k).lower() for word in
                ("path", "directory", "token", "secret", "password", "hostname", "username", "environment"))
                else _redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(?:bearer\s+|(?:token|password|secret|api[_-]?key)[=: ]+)[^\s,;]+",
                       "[redacted]", value)
        value = re.sub(r"(?:/[A-Za-z0-9_.~-]+){2,}[^\s,;]*|[A-Za-z]:[\\/][^\s,;]+", "[path]", value)
        return value
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return type(value).__name__


def doctor(data_dir: Path | None = None) -> dict:
    versions = {}
    for name in CORE_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "version": __version__, "python": ".".join(map(str, sys.version_info[:3])),
        "core_ready": sys.version_info >= (3, 12) and all(versions.values()),
        "core_packages": versions, "provider_probe": "not_run",
        "note": "No weights are downloaded or loaded by this command.",
    }
    try:
        from platformdirs import user_data_path
        from .models.manager import ModelManager
        root = data_dir or user_data_path("compag-annotator", appauthor=False)
        report["providers"] = _redact(ModelManager(Path(root)).status())
        report["provider_probe"] = "metadata_only"
    except ImportError:
        report["providers"] = {"status": "unavailable", "reason": "Model manager is unavailable."}
    except Exception as exc:
        # A third-party error can contain credentials or a private pathname.
        report["providers"] = {"status": "failed", "error_type": type(exc).__name__}
    return report


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Port must be an integer from 0 to 65535.") from exc
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("Port must be from 0 to 65535; 0 selects an available port.")
    return port


def _serve_options(parser, inherit=False):
    default = argparse.SUPPRESS if inherit else None
    parser.add_argument("--data-dir", type=Path, default=default, help="Local application settings and model registry folder")
    parser.add_argument("--port", type=_port, default=argparse.SUPPRESS if inherit else 0, help="Loopback port; 0 chooses an available port (default)")
    parser.add_argument("--no-browser", action="store_true", default=argparse.SUPPRESS if inherit else False, help="Print the local address without opening a browser")
    parser.add_argument("--qa-mode", action="store_true", default=argparse.SUPPRESS if inherit else False, help="Record automated_qa actors; never human verification")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="compag-annotator", description="Annotate and review images locally. AI is optional.")
    root.add_argument("--version", action="version", version=f"COMPAG Annotator {__version__}")
    _serve_options(root)
    subs = root.add_subparsers(dest="command")
    serve_parser = subs.add_parser("serve", help="Start the local application")
    _serve_options(serve_parser, inherit=True)
    diagnostic = subs.add_parser("doctor", help="Check the core and optional provider setup without loading models")
    diagnostic.add_argument("--data-dir", type=Path)
    diagnostic.add_argument("--json", action="store_true", help="Print a redacted JSON report")
    backup = subs.add_parser("backup", help="Create a native project backup without changing the project")
    backup.add_argument("project", type=Path, help="Existing project folder")
    backup.add_argument("--output", required=True, type=Path, help="New archive path; existing files are never replaced")
    return root


def _open_when_ready(url: str, stopped: threading.Event) -> None:
    # Ignore proxy environment variables for the loopback readiness probe.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 20
    while not stopped.is_set() and time.monotonic() < deadline:
        try:
            with opener.open(url + "/api/session", timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (OSError, urllib.error.URLError):
            pass
        stopped.wait(0.1)


def serve(args) -> int:
    from .app import create_app
    import uvicorn

    token = secrets.token_urlsafe(32)
    app = create_app(data_dir=args.data_dir, token=token, qa_mode=args.qa_mode)
    stopped = threading.Event()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.port))
        listener.listen(128)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        print(f"COMPAG Annotator {__version__}: {url}", flush=True)
        if args.qa_mode:
            print("Automated QA mode: review events are not human verification.", flush=True)
        if not args.no_browser:
            threading.Thread(target=_open_when_ready, args=(url, stopped), daemon=True).start()
        config = uvicorn.Config(app, host="127.0.0.1", port=listener.getsockname()[1],
                                access_log=False, log_level="warning", proxy_headers=False)
        try:
            uvicorn.Server(config).run(sockets=[listener])
        finally:
            stopped.set()
    return 0


def backup_project(project: Path, output: Path) -> Path:
    if not project.is_dir():
        raise ValueError("Choose an existing project folder.")
    if output.exists() or output.is_symlink():
        raise ValueError("Backup destination already exists; choose a new filename.")
    if output.resolve().is_relative_to(project.resolve()):
        raise ValueError("Save the backup outside the project folder.")
    if not output.parent.is_dir():
        raise ValueError("The backup destination folder does not exist.")
    from .core.catalog import Catalog
    from .core.projects import Project
    # Delegate to the canonical backup writer in a private temporary workspace.
    # Publish via an exclusive hard link to avoid replacing a file created by
    # another process between preflight and completion.
    with tempfile.TemporaryDirectory(prefix=".compag-backup-", dir=output.parent) as temporary:
        scratch = Path(temporary)
        archive = scratch / "project.zip"
        Catalog(scratch / "catalog").backup(Project(project.resolve()), archive, include_images=True)
        os.link(archive, output.absolute())
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("The backup did not produce an archive.")
    return output


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    args = parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor(args.data_dir)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(f"COMPAG Annotator {__version__}\nCore: {'ready' if report['core_ready'] else 'needs repair'}")
                for name, version in report["core_packages"].items():
                    print(f"  {name}: {version or 'missing'}")
                print("Providers: " + json.dumps(report.get("providers", {}), ensure_ascii=False))
                print(report["note"])
            return 0 if report["core_ready"] else 1
        if args.command == "backup":
            backup_project(args.project, args.output)
            print("Project backup created. Model weights are not included.")
            return 0
        return serve(args)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"COMPAG Annotator: {_redact(str(exc))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
