# 1.0.0rc4: multi-picture JPEG import and permanent project deletion

## Multi-picture JPEG / MPO

Some files ending in `.jpeg` contain an MPO (Multi Picture Object) container. The previous format allowlist rejected these as unsupported even though the primary picture was readable. MPO is now accepted, including the explicit `.mpo` extension. The primary picture (frame 0) becomes one annotation image, at its original pixel resolution and with EXIF orientation applied. Extra embedded pictures remain in the unchanged original file; they do not become additional dataset images. The import summary explains this choice. Metadata records source format, frame count and selected frame. HDR gain-map rendering is not applied; annotation uses the decoded primary RGB picture.

PNG display and thumbnail files are generated inside project storage. Original bytes are retained both for managed-copy and referenced storage. The existing exclusions for other animated/multipage formats and high-bit-depth images still apply. This is not an animation, stereo or HDR rendering feature.

Source: [Pillow's official MPO documentation](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#mpo), checked against the installed Pillow decoder.

## Delete a complete project

Open **Projects → Delete project** on the desired project. Read the displayed name, folder, image/annotation/round counts and project-job count. Type the exact project name, then choose **Permanently delete project**. Cancel or Close does nothing.

This permanently removes the project folder (including annotations, rounds and managed images) and its project-job folders, including in-app exports and training snapshots. The project is removed from the catalog and from the current workspace. This is different from removing an individual image, which preserves archived project history.

Original files and exported copies outside those folders are kept. Any other files you placed inside the project folder are deleted. Shared model registrations remain: registered checkpoints inside the deleted folders are first copied to shared model storage and checksum-verified, and the registry is updated before erasure. No model is trained or activated. Other projects are preserved; deletion is blocked if another project overlaps the folder or references an original within the deletion scope. Missing/changed registered checkpoints must be repaired before deletion.

Finish or cancel active jobs before deletion. A changed project revision, mismatched name/path, or absent confirmation prevents deletion. Staging failures roll the folder moves back. If the operating system prevents final erasure after catalog removal, the UI reports incomplete cleanup and its folder location; a small deletion receipt is retained under application storage. It does not claim that disk space was reclaimed in that case. This operation has no Undo; export a backup beforehand if you want one.

Automated deletion tests use only disposable synthetic projects and checkpoint byte fixtures; those fixtures are not real model execution. No existing user or research project is deleted during release testing.
