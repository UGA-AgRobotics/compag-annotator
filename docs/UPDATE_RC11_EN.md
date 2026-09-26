# Release preparation — 1.0.0rc11

- Explicit light color scheme and dialog background preserve readable dialogs even when the operating system uses dark mode. Four browser theme combinations are tested.
- Current browser selectors cover exports, English help, annotation import and hidden rejected objects. Installed-package subprocess checks do not fall back to source imports.
- Release artifact names derive from project metadata. CI checks the installed wheel and JavaScript suite. Public staging accepts only the reviewed release report; restaging excludes stale generated manifest/checksum bytes, and empty Git history is not a PASS.
- Approved AGPL-3.0-only license and the five supplied authors replace the pending-rights notice. Model and runtime materials remain outside the bundle.
- Optional runtime security pins and hashes are updated; old environments cannot be falsely relabelled patched. See DEPENDENCY_SECURITY.md for remaining Torch advisories and tested scope.
- Current and inherited results are separated. Raw logs and personal paths are excluded from public artifacts. No live app, research source, reviewed project, weight or original rc10 artifact is overwritten.

See ACCEPTANCE_RESULTS.json and the outer final verification receipt for exact executed checks. This is a prerelease; native desktop and genuine human acceptance remain outstanding. No push/upload is performed.
