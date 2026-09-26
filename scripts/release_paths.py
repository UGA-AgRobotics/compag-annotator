#!/usr/bin/env python3
"""Resolve current artifact names from the same metadata used by the builder."""
import argparse
from pathlib import Path
import re
import tomllib


def release_paths(source, directory="dist"):
    project = tomllib.loads((Path(source) / "pyproject.toml").read_text())["project"]
    version = project["version"]
    if project["name"] != "compag-annotator" or not re.fullmatch(r"[0-9][A-Za-z0-9_.+]*", version):
        raise ValueError("Unsupported project metadata")
    prefix = "compag_annotator-" + version
    return {"version":version, "wheel":str(Path(directory) / (prefix + "-py3-none-any.whl")),
            "sdist":str(Path(directory) / (prefix + ".tar.gz"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--directory", default="dist")
    parser.add_argument("--field", required=True, choices=("version", "wheel", "sdist"))
    args = parser.parse_args()
    print(release_paths(args.source, args.directory)[args.field])


if __name__ == "__main__":
    main()
