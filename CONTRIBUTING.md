# Contributing

Application contributions are governed by AGPL-3.0-only. A public repository has not yet been published. Work in an authorized copy, preserve historical sources read-only, and agree file/API ownership before parallel edits.

Use Python 3.12 and the commands in DEVELOPMENT.md. Add focused tests for changes that affect persisted data, geometry, conversion, permissions or job state. Do not use mocked model output as evidence of genuine provider operation or automated clicks as evidence of human review.

Keep UI language suitable for an ordinary image-annotation workflow. Preserve stable class identities, exact native masks and explicit review/activation choices. Heavy boundary recovery stays off except for explicitly enabled training preparation or automatic-mask annotation, scoped to the confirmed job.

Never commit private data, screenshots, checkpoints, access tokens, runtime caches, personal paths or research manifests. Fixtures must be synthetic or have confirmed redistribution rights. Preserve upstream notices and describe direct adaptations in SOURCE_REUSE_MAP.md. Do not invent authorship.

Run the allowlist/content/artifact scanner and inspect the staged result. Keep changes in your assigned files. Remote publication, new model downloads and GPU acceptance work require their respective explicit authorization; ordinary CI does not run models.
