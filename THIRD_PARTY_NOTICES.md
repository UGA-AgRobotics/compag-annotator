# Third-party material inventory

This inventory is not a substitute for the full notices inside dependencies. The base application wheel contains application source, UI/help and a geometric SVG icon; it does not vendor ML providers or model weights. A redistributed dependency wheelhouse/runtime must retain the notices shipped in each actual artifact. The supplied author/rights approval and AGPL-3.0-only decision are recorded in LICENSE_REVIEW.md.

## Core runtime

Versions below were observed in installed metadata on 2026-09-24 and are pinned in `pyproject.toml`. License expressions are reported from that metadata; binary-wheel component notices need final review.

| Package | Version | Reported license / source |
|---|---|---|
| FastAPI | 0.141.1 | MIT — [source](https://github.com/fastapi/fastapi) |
| Uvicorn | 0.53.0 | BSD-3-Clause — [source](https://github.com/Kludex/uvicorn) |
| Pillow | 12.3.0 | MIT-CMU — [source](https://github.com/python-pillow/Pillow) |
| NumPy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0; additional bundled native components must retain their notices — [source](https://github.com/numpy/numpy) |
| Shapely | 2.1.2 | BSD-3-Clause; GEOS in binary wheels has separate terms — [source](https://github.com/shapely/shapely) |
| pycocotools | 2.0.11 | FreeBSD as reported in distribution metadata — [source](https://github.com/ppwwyyxx/cocoapi) |
| PyYAML | 6.0.3 | MIT — [source](https://github.com/yaml/pyyaml) |
| defusedxml | 0.7.1 | PSFL as reported — [source](https://github.com/tiran/defusedxml) |
| python-multipart | 0.0.32 | Apache-2.0 — [source](https://github.com/Kludex/python-multipart) |
| platformdirs | 4.11.12 | MIT — [source](https://github.com/tox-dev/platformdirs) |
| Starlette | 1.7.0 | BSD-3-Clause — [source](https://github.com/Kludex/starlette) |
| Pydantic / pydantic-core | 2.13.5 / 2.46.5 | MIT — [source](https://github.com/pydantic/pydantic) |
| annotated-types / annotated-doc | 0.8.0 / 0.0.5 | MIT — [types](https://github.com/annotated-types/annotated-types), [doc](https://github.com/fastapi/annotated-doc) |
| AnyIO | 4.15.1 | MIT — [source](https://github.com/agronholm/anyio) |
| Click | 8.5.0 | BSD-3-Clause — [source](https://github.com/pallets/click) |
| h11 | 0.16.0 | MIT — [source](https://github.com/python-hyper/h11) |
| idna | 3.20 | BSD-3-Clause — [source](https://github.com/kjd/idna) |
| typing-extensions / typing-inspection | 4.16.0 / 0.4.4 | PSF-2.0 / MIT — [extensions](https://github.com/python/typing_extensions), [inspection](https://github.com/pydantic/typing-inspection) |

Developer-only dependencies include pytest, HTTPX, Playwright, build, setuptools, wheel and Ruff. Their licenses and browser-distribution notices travel with any redistributed developer bundle; they are not required for ordinary manual annotation. No CDN, web font, third-party raster example or remote UI asset is required by the core.

## Optional model materials

- [SAM2](https://github.com/facebookresearch/sam2): principal code and checkpoints use Apache-2.0; the optional connected-components component has its own notice. Retain the exact pinned version's licenses.
- [SAM3](https://github.com/facebookresearch/sam3/blob/main/LICENSE): custom SAM License. No SAM3 checkpoint is included or automatically downloaded. Compatibility/redistribution remains an owner decision.
- [Ultralytics](https://www.ultralytics.com/license): AGPL-3.0 and Enterprise routes. This AGPL-3.0-only release must satisfy the applicable code/model obligations; no commercial license has been claimed acquired.
- Optional provider Torch/Torchvision, CUDA and native dependencies: inspect notices in the provider-specific installed wheels. CUDA availability is not a license grant or a claim that drivers may be redistributed.

The [source-reuse map](SOURCE_REUSE_MAP.md) records baseline relationships. The release coordinator confirmed authority to release the named contributors' application code. SAM3 materials remain separately licensed and excluded from the distribution. Existing notices must not be deleted or replaced by an invented attribution.

## Optional user-space Python bootstrap

`locks/python-bootstrap.json` pins CPython **3.12.14**, distributed by **astral-sh/python-build-standalone**, release **20260924**, asset **586643535**. The [immutable release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260924) and [official asset API](https://api.github.com/repos/astral-sh/python-build-standalone/releases/assets/586643535) supply its identity, size and SHA-256. The installer downloads this runtime only after explicit `--download-python` consent; the application source/wheel/sdist contains only the pin and installer, not the runtime archive or binaries.

CPython's [license page](https://docs.python.org/3.12/license.html) identifies PSF License Version 2 and explains that incorporated software has additional terms. The standalone distribution also supplies dependencies and pip's vendored components, which retain their own notices. Extraction preserves the archive's files without changing license text. The actual bootstrap test checked **42 existing LICENSE/COPYING/NOTICE files** against the original archive bytes, including `python/lib/python3.12/LICENSE.txt` and pip/vendor licenses. This count describes preserved files, not a claim that it is a complete license audit of every linked native component. Review the exact runtime distribution and upstream component terms before separately redistributing it.

This optional download is independent of SAM/YOLO setup. Its terms do not replace the application's AGPL-3.0-only license or authorize remote publication.
