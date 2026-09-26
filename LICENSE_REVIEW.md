# Application license and distribution scope

The application is licensed under **AGPL-3.0-only**. The complete, unmodified GNU Affero General Public License version 3 is in [LICENSE](LICENSE).

Copyright (C) 2026 Hasan Jahanifar, Hasan Mirzakhaninafchi, Wesley M. Porter, Denis O. Kiobia, and Glen C. Rains.

The release coordinator supplied these five names and explicitly confirmed authority to release their contributions and approval of AGPL-3.0-only on 2026-09-25. This records that supplied authorization; it is not an independently obtained institutional legal opinion. No institutional endorsement, contact address, DOI, or commercial license is implied.

The preserved research source remains unchanged. [SOURCE_REUSE_MAP.md](SOURCE_REUSE_MAP.md) records the application adaptations. Retain source and upstream notices when redistributing. If modified software is offered over a network, review the applicable corresponding-source requirements in section 13 of the license. The license does not authorize uploading private projects, photos or weights.

## Separately installed components

The base source, wheel and release archive contain application code, tests, documentation and installer scripts. They do not contain Python runtimes, third-party dependency wheels, SAM/YOLO provider source or checkpoints. Each separate component retains its own terms; application licensing does not replace them.

- SAM2 principal code/checkpoints: Apache-2.0, with applicable component notices retained upstream.
- Ultralytics: the AGPL open-source route is used for this application. No Enterprise license or exception has been claimed.
- SAM3: optional user-installed code and user-supplied local weights are governed by the custom [SAM License](https://github.com/facebookresearch/sam3/blob/2345a4ad109ac29c569da749c91d84f10dc08c40/LICENSE). SAM3 material is excluded from this distribution. This application grant does not grant rights to redistribute a combined SAM3 runtime or checkpoint; assess its separate terms before doing so. SAM3 weights are never automatically downloaded.
- Python bootstrap and dependencies: downloaded separately with their own license files; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Synthetic test fixtures are constructed by tests; no research photos, reviewed annotations or models are shipped. The geometric SVG icon and local UI are part of the application; no third-party font pack is bundled.

The author's license approval is separate from the action of publication. No remote push or upload is performed by the installer, build tools or this release preparation.
