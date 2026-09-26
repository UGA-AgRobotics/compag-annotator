"""Independent user-space provider installers; no dependency on an old environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from .catalog import PROVIDERS
from .files import file_lock, write_json
from ..providers.process import run_command, WorkerClient
from ..providers.protocol import check_cancel, emit, identity

# Dependency security pins require an explicit fresh provider runtime. An old
# adapter-only repair must never relabel an unpatched environment as current.


def _sam3_inventory(python):
    """Read distribution metadata only; never import Torch, SAM3 or checkpoints."""
    script = """
import importlib.metadata as m, json, sys
versions = {d.metadata['Name'].lower().replace('_', '-'): d.version for d in m.distributions()}
sam3 = m.distribution('sam3')
try:
    decord = m.distribution('decord')
    decoder = {'version': decord.version, 'wheel': decord.read_text('WHEEL') or ''}
except m.PackageNotFoundError:
    decoder = None
print(json.dumps({'versions': versions, 'python': list(sys.version_info[:2]), 'prefix': sys.prefix,
                  'source': json.loads(sam3.read_text('direct_url.json') or '{}'), 'decord': decoder}))
"""
    result = subprocess.run([str(python), "-I", "-c", script], capture_output=True, text=True, timeout=20)
    if result.returncode:
        return None
    return json.loads(result.stdout)


def _sam3_image_repair_eligible(inventory, python, lock):
    if not inventory or inventory.get("python") != [3, 12]:
        return False
    if Path(inventory["prefix"]).resolve() != Path(python).parent.parent.resolve():
        return False
    if inventory.get("source", {}).get("url") != PROVIDERS["sam3"]["source_url"]:
        return False
    expected = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", lock.read_text(), re.MULTILINE))
    versions = inventory["versions"]
    return (versions.get("sam3") == PROVIDERS["sam3"]["version"] and
            all(versions.get(name.lower().replace("_", "-")) == version for name, version in expected.items()))


def _finish_sam3_image_runtime(data_dir, python, inventory, lock_hash, adapter_hash, progress, cancel):
    """Recover this managed image runtime, leaving all source/ML versions intact.

    Decord is a notebooks/video extra, not a required dependency of the pinned
    SAM3 distribution. The checked release wheel has a CPython-3.6 ABI tag.
    This app does not offer video decoding, so it is removed, not retagged or
    substituted. Failed installs never receive an installed success receipt.
    """
    decoder = inventory.get("decord")
    if decoder and decoder["version"] != "0.6.0":
        raise RuntimeError("Managed SAM3 contains an unexpected Decord version; repair requires explicit runtime replacement")
    removed = []
    if decoder:
        emit(progress, stage="sam3_image_runtime_repair", message="Removing optional Decord video decoder; image APIs do not use it")
        run_command([python, "-m", "pip", "uninstall", "--yes", "decord"], progress=progress, cancel=cancel)
        removed = ["decord==0.6.0"]
    if isinstance(progress, _SetupProgress): progress.step("Checking installed dependencies", 5)
    run_command([python, "-m", "pip", "check"], progress=progress, cancel=cancel)
    if isinstance(progress, _SetupProgress): progress.step("Testing model runtime", 6)
    client = WorkerClient(python, data_dir)
    try:
        probe = client.request("probe", {"provider": "sam3"}, progress=progress, cancel=cancel, timeout=120)
    finally:
        client.close()
    if isinstance(progress, _SetupProgress): progress.step("Saving runtime receipt", 7)
    frozen = run_command([python, "-m", "pip", "freeze", "--all"], cancel=cancel)
    root = Path(data_dir) / "runtimes" / "sam3"
    (root / "resolved-packages.txt").write_text(frozen + "\n")
    receipt = {"provider": "sam3", "state": "installed", "python": str(python), "flavor": "cu128",
               "lock_hash": lock_hash, "adapter_lock_hash": adapter_hash,
               "source_revision": PROVIDERS["sam3"]["source_revision"], "created_at": time.time(),
               "probe": probe, "external": False, "real_checkpoint_tested": False,
               "runtime_scope": "still_image_interactivity", "omitted_optional_dependencies": ["decord"],
               "upstream_source_modified": False,
               "repair": {"kind": "remove_optional_video_decoder", "removed": removed,
                          "previous_decoder_metadata": decoder,
                          "reason": "Decord 0.6.0 release wheel declares CPython 3.6 ABI; video decoding is outside this image runtime"}}
    write_json(root / "installed.json", receipt)
    emit(progress, stage="runtime_installed", provider="sam3", scope="still_image_interactivity")
    return receipt


def _repair_adapters(data_dir, provider, python, receipt, adapter_lock, lock_hash, progress, cancel):
    """Stage a tiny immutable package overlay, then atomically switch its .pth.

    On failed import/probe, restore the prior .pth and leave the old runtime
    success receipt intact. Never uninstall or replace Torch/provider packages.
    Called only under a consented provider install lock.
    """
    adapter_hash = identity(adapter_lock.read_text())
    bundles = Path(data_dir) / "adapter-bundles"
    bundles.mkdir(parents=True, exist_ok=True)
    target = bundles / adapter_hash
    with file_lock(bundles / "install.lock", cancel):
        if not target.is_dir():
            staged = Path(tempfile.mkdtemp(prefix="staging-", dir=bundles))
            try:
                emit(progress, stage="adapter_repair_staging", provider=provider)
                run_command([python, "-m", "pip", "install", "--target", str(staged), "--no-deps",
                             "--require-hashes", "-r", str(adapter_lock)], progress=progress, cancel=cancel)
                check_cancel(cancel)
                write_json(staged / "compag-adapter-receipt.json", {"lock_hash": adapter_hash})
                os.replace(staged, target)
            finally:
                if staged.exists():
                    shutil.rmtree(staged)
    pth = Path(python).parent.parent / "lib" / "python3.12" / "site-packages" / "compag_adapter_deps.pth"
    previous = pth.read_text() if pth.exists() else None
    temporary = pth.with_name(pth.name + "." + uuid.uuid4().hex)
    try:
        temporary.write_text(str(target.resolve()) + "\n")
        os.replace(temporary, pth)
        client = WorkerClient(python, data_dir)
        try:
            probe = client.request("probe", {"provider": provider}, progress=progress, cancel=cancel, timeout=120)
        finally:
            client.close()
        run_command([python, "-m", "pip", "check"], progress=progress, cancel=cancel)
        result = {**receipt, "lock_hash": lock_hash, "adapter_lock_hash": adapter_hash, "probe": probe,
                  "adapter_repair": "atomic_overlay", "repaired_at": time.time()}
        write_json(Path(data_dir) / "runtimes" / provider / "installed.json", result)
        emit(progress, stage="adapter_repair_complete", provider=provider)
        return result
    except BaseException:
        if previous is None:
            pth.unlink(missing_ok=True)
        else:
            temporary.write_text(previous)
            os.replace(temporary, pth)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def runtime_python(data_dir, provider):
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider")
    external = os.environ.get(f"COMPAG_{provider.upper()}_PYTHON")
    if external:
        path = Path(external).expanduser().absolute()
        if not path.is_file():
            raise ValueError(f"Configured {provider} Python does not exist")
        return path
    root = Path(data_dir) / "runtimes" / provider
    receipt_path = root / "installed.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("source_revision") != PROVIDERS[provider]["source_revision"]:
        return None
    python = root / "venv" / "bin" / "python"
    return python if python.is_file() else None


def bootstrap_python():
    supplied = os.environ.get("COMPAG_RUNTIME_PYTHON")
    python = supplied or (sys.executable if sys.version_info[:2] == (3, 12) else shutil.which("python3.12"))
    if not python:
        raise RuntimeError("Python 3.12 is required; use the application installer or COMPAG_RUNTIME_PYTHON")
    result = subprocess.run([python, "-I", "-c", "import sys; print('.'.join(map(str,sys.version_info[:2])))"],
                            text=True, capture_output=True, timeout=15, check=True)
    if result.stdout.strip() != "3.12":
        raise RuntimeError("Provider bootstrap must use Python 3.12")
    return Path(python).absolute()


class _SetupProgress:
    """Completed setup steps, not an invented estimate of remaining time."""
    def __init__(self, callback):
        self.callback, self.completed, self.label = callback, 0, "Checking runtime"

    def step(self, label, completed):
        self.label, self.completed = label, completed
        self({"stage": "runtime_setup"})

    def __call__(self, event):
        emit(self.callback, **{**event, "setup_completed": self.completed, "setup_total": 8,
                              "setup_label": self.label, "setup_percent": self.completed * 100 / 8})


def install_runtime(data_dir, provider, *, consent=False, progress=None, cancel=None, flavor=None):
    progress = _SetupProgress(progress)
    result = _install_runtime(data_dir, provider, consent=consent, progress=progress, cancel=cancel, flavor=flavor)
    progress.step("Runtime ready", 8)
    return result


def _install_runtime(data_dir, provider, *, consent=False, progress=None, cancel=None, flavor=None):
    """Install code/dependencies only. flavor is cpu or cu128; weights are separate."""
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider")
    if consent is not True:
        raise PermissionError("Explicit runtime installation consent is required")
    flavor = flavor or os.environ.get("COMPAG_TORCH_FLAVOR", "cu128")
    if flavor not in {"cpu", "cu128"}:
        raise ValueError("COMPAG_TORCH_FLAVOR must be cpu or cu128")
    if provider == "sam3" and flavor != "cu128":
        raise ValueError("The pinned SAM3 runtime requires CUDA; CPU SAM3 is not supported")
    data_dir = Path(data_dir)
    root = data_dir / "runtimes" / provider
    root.mkdir(parents=True, exist_ok=True)
    spec = PROVIDERS[provider]
    lock = Path(__file__).with_name("requirements") / f"provider-{provider}-{flavor}.txt"
    adapter_lock = lock.with_name("provider-adapters.txt")
    adapter_hash = identity(adapter_lock.read_text())
    lock_hash = identity({"lock": lock.read_text(), "flavor": flavor, "revision": spec["source_revision"]})
    with file_lock(root / "install.lock", cancel):
        check_cancel(cancel)
        progress.step("Checking existing runtime", 0)
        python = runtime_python(data_dir, provider)
        if os.environ.get(f"COMPAG_{provider.upper()}_PYTHON"):
            emit(progress, stage="external_runtime_probe", provider=provider)
            client = WorkerClient(python, data_dir)
            try:
                probe = client.request("probe", {"provider": provider}, progress=progress, cancel=cancel, timeout=90)
            finally:
                client.close()
            return {"provider": provider, "python": str(python), "external": True, "probe": probe,
                    "state": "installed", "source_revision": "external_runtime_unverified"}
        receipt_path = root / "installed.json"
        # An initial SAM3 install may have installed all packages successfully
        # and then failed pip check on the optional Decord wheel. Verify every
        # current package pin and the exact source before performing a small
        # repair; do not recreate the environment or redownload large packages.
        candidate = root / "venv" / "bin" / "python"
        if provider == "sam3" and candidate.is_file():
            inventory = _sam3_inventory(candidate)
            if _sam3_image_repair_eligible(inventory, candidate, lock):
                old = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                if old.get("lock_hash") != lock_hash or inventory.get("decord"):
                    return _finish_sam3_image_runtime(data_dir, candidate, inventory, lock_hash, adapter_hash, progress, cancel)
        if python and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("lock_hash") == lock_hash:
                if receipt.get("adapter_lock_hash") != adapter_hash:
                    return _repair_adapters(data_dir, provider, python, receipt, adapter_lock, lock_hash, progress, cancel)
                return receipt
            raise RuntimeError("Existing runtime uses another lock/flavor. Remove its runtime directory after unloading before reinstalling")
        if shutil.disk_usage(root).free < (8 if flavor == "cu128" else 3) * 1024**3:
            raise OSError("Insufficient runtime installation space (8 GiB CUDA / 3 GiB CPU minimum)")
        base_python = bootstrap_python()
        env = os.environ.copy()
        # Keep pip's normal shared wheel cache; callers may explicitly set PIP_CACHE_DIR.
        env.update({"SAM2_BUILD_CUDA": "0", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PYTHONNOUSERSITE": "1"})
        emit(progress, stage="runtime_create", provider=provider, flavor=flavor)
        run_command([base_python, "-m", "venv", str(root / "venv")], env=env, progress=progress, cancel=cancel)
        progress.step("Installing package tools", 1)
        python = root / "venv" / "bin" / "python"
        run_command([python, "-m", "pip", "install", "pip==26.2.1", "setuptools==84.0.0", "wheel==0.48.0"],
                    env=env, progress=progress, cancel=cancel)
        env["PIP_PROGRESS_BAR"] = "raw"
        progress.step("Downloading and installing model dependencies", 2)
        run_command([python, "-m", "pip", "install", "--require-hashes", "-r", str(lock)],
                    env=env, progress=progress, cancel=cancel)
        progress.step("Installing image adapters", 3)
        run_command([python, "-m", "pip", "install", "--no-deps", "--require-hashes", "-r", str(adapter_lock)],
                    env=env, progress=progress, cancel=cancel)
        progress.step("Installing model provider", 4)
        if provider in {"sam2", "sam3"}:
            run_command([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", spec["source_url"]],
                        env=env, progress=progress, cancel=cancel)
        if provider == "sam3":
            inventory = _sam3_inventory(python)
            if not _sam3_image_repair_eligible(inventory, python, lock):
                raise RuntimeError("Installed SAM3 packages/source differ from the pinned image-runtime requirements")
            return _finish_sam3_image_runtime(data_dir, python, inventory, lock_hash, adapter_hash, progress, cancel)
        progress.step("Checking installed dependencies", 5)
        run_command([python, "-m", "pip", "check"], env=env, progress=progress, cancel=cancel)
        progress.step("Testing model runtime", 6)
        client = WorkerClient(python, data_dir)
        try:
            probe = client.request("probe", {"provider": provider}, progress=progress, cancel=cancel, timeout=120)
        finally:
            client.close()
        progress.step("Saving runtime receipt", 7)
        frozen = run_command([python, "-m", "pip", "freeze", "--all"], env=env, cancel=cancel)
        (root / "resolved-packages.txt").write_text(frozen + "\n")
        receipt = {"provider": provider, "state": "installed", "python": str(python), "flavor": flavor,
                   "lock_hash": lock_hash, "adapter_lock_hash": adapter_hash,
                   "source_revision": spec["source_revision"], "created_at": time.time(),
                   "probe": probe, "external": False, "real_checkpoint_tested": False,
                   "sam2_cuda_postprocessing": "disabled; no connected-component hole/sprinkle cleanup" if provider == "sam2" else None}
        write_json(receipt_path, receipt)
        emit(progress, stage="runtime_installed", provider=provider)
        return receipt
