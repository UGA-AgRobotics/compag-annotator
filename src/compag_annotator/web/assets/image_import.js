import { el, input, field, check, select, form, dialog, button, projectPath } from "./ui.js";

const MiB = 1024 * 1024;
export const IMAGE_EXTENSIONS = /\.(jpe?g|png|bmp|webp|mpo|tiff?)$/i;
// Keep multipart overhead below the server's unchanged 256 MiB request limit.
export function selectImportFiles(files, recursive = true) {
  const selected = [], skipped = [];
  for (const item of files) {
    let reason;
    if (!recursive && item.name.split("/").length > 2) reason = "Inside a subfolder; enable Include subfolders";
    else if (!IMAGE_EXTENSIONS.test(item.name)) reason = "Unsupported file type";
    else if (item.file.size > 255 * MiB) reason = "Larger than the 255 MiB upload limit; use a smaller image";
    (reason ? skipped : selected).push(reason ? { name: item.name, reason } : item);
  }
  return { selected, skipped };
}
export function uploadBatches(files, byteLimit = 64 * MiB, countLimit = 100) {
  const batches = [];
  let current = [], bytes = 0;
  for (const item of files) {
    if (current.length && (bytes + item.file.size > byteLimit || current.length >= countLimit)) {
      batches.push(current); current = []; bytes = 0;
    }
    current.push(item); bytes += item.file.size;
  }
  if (current.length) batches.push(current);
  return batches;
}
export async function collectDropFiles(transfer, recursive = true) {
  const result = [];
  const walk = async (entry, prefix = "") => {
    if (entry.isFile) {
      const file = await new Promise((ok, no) => entry.file(ok, no));
      result.push({ file, name: prefix + file.name });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let items;
      do {
        items = await new Promise((ok, no) => reader.readEntries(ok, no));
        for (const child of items)
          if (!child.isDirectory || recursive) await walk(child, prefix + entry.name + "/");
      } while (items.length);
    }
  };
  const entries = [...(transfer.items || [])].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (entries.length) for (const entry of entries) await walk(entry);
  else for (const file of transfer.files) result.push({ file, name: file.name });
  return result;
}
function fileList(title, entries) {
  return entries.length && el("details", {}, el("summary", {}, `${title} (${entries.length})`),
    el("ul", {}, entries.slice(0, 100).map((item) => el("li", {}, item))),
    entries.length > 100 && el("p", {}, "Showing the first 100 entries."));
}
export function uploadDialog(app) {
  const pid = app.project.id, projectName = app.project.name;
  const recursive = check("upload-recursive", "Include subfolders", true);
  const preview = el("div", { "aria-live": "polite" });
  const result = el("section", { "aria-live": "polite" });
  const zone = el("div", { class: "dropzone" }, el("h3", {}, "Drop images or a folder here"),
    el("p", {}, "JPEG (including MPO), PNG, BMP, WebP and supported TIFF. Your original files are kept."));
  let staged = [], busy = false;
  const selection = () => selectImportFiles(staged, recursive.querySelector("input").checked);
  const updatePreview = () => {
    const { selected, skipped } = selection();
    add.disabled = busy || !selected.length;
    add.textContent = selected.length ? `Add ${selected.length} images to project` : "Add images to project";
    preview.replaceChildren(el("p", {}, staged.length
      ? `${selected.length} image files ready · ${(selected.reduce((n, i) => n + i.file.size, 0) / MiB).toFixed(1)} MiB · ${skipped.length} skipped`
      : "No files selected. Choose image files or a folder above."),
      ...[fileList("Selected files", selected.map((i) => i.name)),
        fileList("Skipped files", skipped.map((i) => `${i.name}: ${i.reason}`))].filter(Boolean));
    if (staged.length && !selected.length)
      preview.append(el("p", { class: "inline-warning" }, "No images ready to add. Check Include subfolders or choose supported image files."));
  };
  const stage = (items) => {
    staged = items; result.replaceChildren(); updatePreview();
  };
  const setBusy = (value) => {
    busy = value;
    d.querySelectorAll("input, select, button").forEach((c) => { c.disabled = value; });
    updatePreview();
  };
  const refreshImages = async () => {
    if (app.project?.id === pid) await app.navigate("images");
  };
  const summary = (totals, skipped, failure) => {
    result.replaceChildren(el("h3", {}, "Import summary"),
      el("p", {}, `${totals.imported.length} added · ${totals.duplicates.length} already in project · ${totals.errors.length} could not be added`),
      el("p", {}, `Project: ${projectName}`));
    if (totals.notes.length) result.append(el("p", {}, `${totals.notes.length} multi-picture JPEG/MPO files: primary image added; extra embedded pictures kept in the originals.`));
    if (skipped.length) result.append(el("p", {}, `${skipped.length} files skipped before upload.`));
    for (const detail of [
      fileList("Already in project", totals.duplicates),
      fileList("Import notes", totals.notes.map((n) => `${n.name}: ${n.message}`)),
      fileList("Could not be added", totals.errors.map((e) => `${e.name}: ${e.error}`)),
    ]) if (detail) result.append(detail);
    if (failure) result.append(el("p", { class: "inline-warning", role: "alert" },
      `Import stopped: ${failure}. Counts above cover completed batches. Some files may already have been added; open Images or Jobs to check. After the job finishes, selecting the files again safely skips duplicates.`));
    else if (totals.imported.length || totals.duplicates.length)
      result.append(el("p", {}, "Your images are now in the project. Choose View images to see them."));
    else result.append(el("p", {}, "No images were added. Check the file details above and choose supported image files."));
    result.append(button("View images", () => d.close(), "primary"));
    result.scrollIntoView({ block: "nearest" });
  };
  const runImport = async (batches, skipped, localBody) => {
    if (busy) return;
    setBusy(true);
    const totals = { imported: [], duplicates: [], errors: [], notes: [] };
    let failure, processed = 0;
    const count = batches.reduce((n, b) => n + b.length, 0);
    const progress = (text) => result.replaceChildren(el("p", { role: "status" }, text));
    try {
      const requests = localBody ? [null] : batches;
      for (let index = 0; index < requests.length; index++) {
        const batch = requests[index];
        progress(localBody ? "Reading the selected local folder…" : `Sending batch ${index + 1} of ${batches.length} · ${processed} of ${count} files processed…`);
        const body = localBody || new FormData();
        if (batch) for (const item of batch) body.append("files", item.file, item.name);
        const job = await app.api.post(projectPath(pid) + (localBody ? "/images/import" : "/images/upload"), body);
        const finished = await app.api.waitJob(job, { onProgress: (j) => {
          const done = processed + (j.progress?.completed || 0);
          const total = localBody ? j.progress?.total : count;
          progress(total ? `Processing images: ${done} of ${total} (${Math.floor(100 * done / total)}%)` : `Import ${j.status}…`);
        }});
        for (const key of Object.keys(totals)) totals[key].push(...(finished.result?.[key] || []));
        processed += batch?.length || 0;
      }
    } catch (e) { failure = e.message; }
    finally {
      try { await refreshImages(); }
      catch (e) { failure = `${failure ? failure + ". " : ""}Could not refresh the image list: ${e.message}`; }
      setBusy(false);
    }
    // Clear completed selections so the next action cannot accidentally resubmit them.
    if (!failure) { staged = []; files.value = ""; folder.value = ""; updatePreview(); }
    summary(totals, skipped, failure);
  };
  // Use a native handler: button() would re-enable an empty selection after completion.
  const add = el("button", { type: "button", class: "primary", onclick: () => {
    const { selected, skipped } = selection();
    if (selected.length && !busy) runImport(uploadBatches(selected), skipped);
  }}, "Add images to project");
  const files = input("image-files", "", "file", { multiple: true, accept: ".jpg,.jpeg,.png,.bmp,.webp,.mpo,.tif,.tiff" });
  files.addEventListener("change", () => { stage([...files.files].map((file) => ({ file, name: file.name }))); folder.value = ""; });
  const folder = input("image-folder", "", "file", { multiple: true, webkitdirectory: true, directory: true });
  folder.addEventListener("change", () => { stage([...folder.files].map((file) => ({ file, name: file.webkitRelativePath || file.name }))); files.value = ""; });
  recursive.addEventListener("change", updatePreview);
  zone.addEventListener("dragover", (e) => { e.preventDefault(); if (!busy) zone.classList.add("over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", async (e) => {
    e.preventDefault(); zone.classList.remove("over"); if (busy) return;
    setBusy(true);
    try { stage(await collectDropFiles(e.dataTransfer)); }
    catch (err) { result.replaceChildren(el("p", { role: "alert" }, `Could not read the selection: ${err.message}`)); }
    finally { setBusy(false); }
  });
  const pathForm = form([
    field("Local folder path", input("path", "", "text", { required: true }), "Folder on the computer running COMPAG. For folders on another computer, use Choose a folder above."),
    field("Storage", select("storage", [["copy", "Copy into project (portable)"], ["reference", "Reference originals (keep their original location)"]])),
    check("recursive", "Include subfolders in local path", true),
    check("approved", "I approve reading this folder"),
  ], async (v) => {
    if (!v.approved) throw Error("Approve the folder before importing.");
    await runImport([], [], v);
  }, "Add local folder to project");
  const d = dialog("Import images", el("div", {},
    el("p", {}, `Adding images to: ${projectName}`),
    el("p", {}, "1. Choose files or a folder → 2. Click Add images to project → 3. View images."),
    zone, field("Choose image files", files), field("Choose a folder", folder), recursive, preview, add,
    el("details", {}, el("summary", {}, "Import from an approved local path"), pathForm), result));
  d.addEventListener("cancel", (e) => { if (busy) e.preventDefault(); });
  updatePreview();
}
