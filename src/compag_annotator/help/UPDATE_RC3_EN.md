# 1.0.0rc3: image import and project removal

The change is limited to importing images, managing their membership in the project, and related help/tests. Previous device choices, model settings, download progress and theme controls are retained.

## Add a folder

1. Open the project and choose **Images → Import images**.
2. Choose files, choose a folder, or drop files/a folder. Subfolders are included by default. Choosing a folder only selects files; it does not add them yet.
3. Check the selected count and **Skipped files**. Switch **Include subfolders** without choosing the folder again.
4. Click **Add N images to project**. Keep the window open until the summary appears. Larger selections are sent sequentially in bounded batches.
5. Read the counts: **added**, **already in project**, and **could not be added**. Expand failures to see filenames and reasons. Click **View images**; the Images page has already refreshed.

JPEG, PNG, BMP, WebP and supported TIFF files are accepted. Unsupported extensions are skipped before browser upload. Corrupt images and unsupported TIFF encodings are rejected during import; other valid files still import. Browser uploads are capped at 255 MiB per file, leaving room below the server's 256 MiB request limit. Batches target 64 MiB and at most 100 files; a larger permitted single file is sent alone. Byte-limit planning is unit-tested with synthetic size records; it is not a claim that a 255 MiB image was decoded in acceptance testing.

The approved local-path form reads a folder on the computer running COMPAG. Its completion also appears in the import window. Use the browser folder picker for folders on a different computer. If a request fails, previously completed batches remain imported. Inspect Images/Jobs; once the job finishes, selecting the files again safely skips duplicates.

## Remove an image

Use **Remove** on its image card, then confirm **Remove image**. Cancel leaves it unchanged. The image and its annotations leave the active dataset and new exports/training snapshots. Original files and archived project data remain; removal does not reclaim disk space. Reimporting the original creates a new image record and does not restore the old annotations automatically.

Images in a started review round are protected to keep round history intact. Planned allocations are adjusted and empty planned rounds are removed. Wait for the project's jobs to finish or cancel them in Jobs before removing an image. Existing immutable training snapshots and older exports remain unchanged.

Tests use generated images, actual browser uploads and the local API. They do not run models or additional training. Human usability review and a new native-Ubuntu installation are not claimed by this update.
