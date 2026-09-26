# Publish COMPAG Annotator 1.0.0rc24

This is a prepared **prerelease**. No remote repository, push, tag or release has
been created by the preparation tools. The application license is AGPL-3.0-only;
the five approved authors are recorded in CITATION.cff.

## What goes where

- Repository: upload the contents of the prepared `repository` folder, including
  `.github` and `.gitignore`. README.md, pyproject.toml and LICENSE belong at the
  repository root. Do not upload the surrounding delivery folder as source.
- Release assets: attach the Ubuntu bundle, wheel, source distribution, source ZIP
  and SHA256SUMS from `release_assets`. These belong on the release page, not in
  the repository's file tree.
- The source archive has no prebuilt wheel. End users should download the
  `compag-annotator-1.0.0rc24-ubuntu-x86_64.tar.gz` asset; it includes the installer
  and wheel. See INSTALL_UBUNTU.md.

No photos, annotations, projects, checkpoints, third-party runtimes, personal
logs or private development Git history are included in these public files.
The outer local `evidence` and `logs` folders are not release assets.

## Upload the repository

Create a new empty GitHub repository under your account, for example
`compag-annotator`. Do not initialize another README, license or .gitignore there.
Choose the visibility you want. From the prepared `repository` directory, use
Git (or GitHub Desktop, preserving hidden .github/.gitignore files).

Replace the public name, email and repository URL below with your own values.
Use your GitHub-provided no-reply email if you do not want a personal email in
public commit history. These commands are instructions for the publisher;
they have not been executed during preparation.

```sh
git init -b main
git config user.name "YOUR PUBLIC NAME"
git config user.email "YOUR GITHUB NOREPLY EMAIL"
git add .
git diff --cached --stat
git commit -m "Prepare COMPAG Annotator 1.0.0rc24"
python3.12 scripts/stage_public.py --source . --history --report ../history-scan.json
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

Stop if the history scan fails. Do not reuse a private development repository's
history. Configure the final repository URL and a private security-reporting
channel when available; none has been invented in this package.

GitHub Actions runs the supplied installed-release checks. Check its results on
the actual repository; local tests do not claim a successful remote CI run.

## Create the prerelease

Open **Releases**, then **Draft a new release**. Use tag `v1.0.0rc24`, target `main`,
and title `COMPAG Annotator 1.0.0rc24`. Copy the body from
`docs/RELEASE_NOTES_RC24.md`. Mark **This is a pre-release**, attach the prepared
files from `release_assets`, and save a draft to review before publishing.
The publisher chooses when to publish.

Keep it labelled prerelease: native Ubuntu desktop acceptance and genuine human
workflow acceptance are still outstanding. No general accuracy or security
certification is claimed. GPU/model weights remain separate optional installs.

Official instructions:
[Import local code](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github),
[Manage releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
