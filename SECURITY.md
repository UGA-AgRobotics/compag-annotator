# Security and privacy

This application is intended for one user's local desktop. Remote/team hosting is outside v1. The CLI binds only to loopback and issues a random local session capability; the API must validate Host/Origin and require the token for mutations. Tokens do not belong in URLs, screenshots, logs or diagnostics.

Select only trusted local checkpoints. Some formats can execute code through upstream loaders. A file hash checks identity, not trust; an isolated provider process is not a security sandbox. SAM3 checkpoint access is the user's responsibility and is never bypassed.

Imported archives, image dimensions, paths, XML and YAML are validated. The default request limit is 256 MiB and image limit is 100 million pixels; large imports should use the approved local-folder flow. A valid archive still contains potentially personal project data. Preserve backups privately.

No cloud upload or telemetry is required for normal annotation. Provider downloads are explicit actions. The core continues to work offline after installation. Do not install GPU drivers or privileged packages through this app.

Before reporting a problem, remove images, project names, model paths, usernames, secrets and authentication tokens. Use the redacted diagnostic view and inspect it before sharing. There is no approved public security contact or repository yet. Report sensitive issues privately to the release owner through an already established private channel; no contact address is invented here.

`1.0.0rc24` is the current prerelease candidate. No security audit certification or production-support promise is made. Publication requires actual local-security tests, dependency review and review of source/history/artifacts.
