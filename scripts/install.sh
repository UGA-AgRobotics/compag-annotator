#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v python3 >/dev/null 2>&1; then
    exec python3 "$script_dir/install_ubuntu.py" "$@"
fi
printf '%s\n' 'Python 3.10 or newer is required to run this installer. See INSTALL_UBUNTU.md.' >&2
exit 1
