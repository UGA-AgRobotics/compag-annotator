# Dependency review and scope — rc11

The version-specific PyPI advisory inventory was queried on 2026-09-25. Exact package metadata and query failures are recorded in the private release evidence; the public summary contains identifiers and outcomes, not raw advisory payloads or local paths. An empty advisory list is not proof of absence of vulnerabilities.

## Changes

New optional provider environments use Pillow 12.3.0 (was 11.3.0), Hydra 1.3.4 for SAM2 (was 1.3.2), setuptools 84.0.0 (was 80.9.0), wheel 0.48.0 (was 0.45.1), and pip 26.2.1 (was 25.2). Version-specific published hashes replace the old hashes in every source/packaged lock. Model source revisions, Torch/CUDA pins, checkpoint bindings, default SAM settings and training logic are unchanged.

Primary references: [Pillow 12.3.0](https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html), [Hydra advisory](https://github.com/hydra-ecosystem/hydra/security/advisories/GHSA-2cp2-2r3c-7p7r), and version-specific [PyPI metadata](https://pypi.org/pypi/pip/26.2.1/json). Hydra instantiation continues to require trusted application-owned configuration; its hardening is not a sandbox.

## Torch applicability and residual risk

PyPI base-version advisories were also checked for the CUDA/CPU local-version Torch pins; a missing PyPI entry for a +cu128/+cpu name is not a clean result. Torch 2.7.1 and 2.10.0 retain reported advisories. Major Torch upgrades require a separate compatibility/training validation budget and are not silently substituted here.

Reported local tensor-operation issues include CTC loss, LU, RNN unpacking/LSTM, JIT scripting and malformed tensor operations. The application exposes image annotation and pinned segmentation adapters, not a user-programmable tensor/JIT/RPC execution API. PT2 archive loading is not an application import format. These observations narrow exposure; they are not a proof about every upstream call or arbitrary third-party checkpoint.

The reported Torch 2.7.1 weights-only unpickler issue is particularly relevant to checkpoint handling. The application requires explicit checkpoint trust and recorded identity before model execution; official catalog downloads use their verification policies. A hash does not make an unknown model safe, and weights_only is not claimed as a security boundary. Use trusted models only. This residual dependency risk is disclosed, not marked patched or risk-accepted on the user's behalf.

## Existing installations

Installing rc11 does not modify an already-running application, user project or provider runtime. The installer refuses to label an old provider lock as the patched lock via adapter-only repair. For an existing installation, unload the provider, close active jobs, preserve its runtime folder as a backup outside the managed runtimes directory, and explicitly install that provider again through Models & AI. Keep registered checkpoint files and project data unchanged. Revert by restoring the old runtime directory if necessary, understanding that its old dependency risks also return.

Base package dependencies are unchanged and exclude Torch/SAM/YOLO. No model, private image or dependency wheelhouse is bundled in the public candidate.
