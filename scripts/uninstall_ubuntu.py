#!/usr/bin/env python3
"""Remove only an installation created by the companion installer."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from install_ubuntu import digest, install_lock, location, read_manifest


def uninstall(prefix):
    prefix = location(prefix)
    with install_lock(prefix):
        record = read_manifest(prefix)
        if record is None:
            raise ValueError("No managed installation exists at the selected location.")
        for name, expected in record["owned_files"].items():
            path = Path(name)
            permitted = {Path(record["bin_dir"]) / "compag-annotator", Path(record["desktop_dir"]) / "compag-annotator.desktop"}
            if path not in permitted or path.is_symlink():
                raise ValueError("The installation launcher record is invalid.")
            if path.exists() and digest(path) != expected:
                raise ValueError("A launcher was modified; preserve it or restore it before uninstalling.")
        known = {"releases", "runtimes", "current", "icon.svg", "installation.json", "install_ubuntu.py", "uninstall_ubuntu.py", "python-bootstrap.json"}
        if any(path.name not in known for path in prefix.iterdir()):
            raise ValueError("The application folder contains extra files. Move them out before uninstalling.")
        for name in record["owned_files"]:
            Path(name).unlink(missing_ok=True)
        shutil.rmtree(prefix)
    print("Application removed. Projects, backups, settings and optional models were preserved.")


def main(argv=None):
    data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    parser = argparse.ArgumentParser(description="Uninstall application code while keeping project and model data.")
    parser.add_argument("--prefix", type=Path, default=data / "compag-annotator-app")
    args = parser.parse_args(argv)
    try:
        if os.geteuid() == 0:
            raise ValueError("Run as your ordinary desktop user, without sudo.")
        uninstall(args.prefix)
        return 0
    except (OSError, ValueError) as exc:
        print(f"Uninstall stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
