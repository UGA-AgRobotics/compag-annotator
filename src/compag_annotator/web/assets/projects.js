import { uploadDialog } from "./image_import.js";
export { uploadDialog, collectDropFiles } from "./image_import.js";
import { generateDialog } from "./generation.js";
import { imagesGuide } from "./workflow-guide.js";
import { confirmImageReview, imagesTrainingPanel } from "./image-review.js";
import {
  el,
  button,
  input,
  select,
  field,
  check,
  card,
  row,
  heading,
  empty,
  badge,
  details,
  form,
  dialog,
  confirmAction,
  notify,
  projectPath,
} from "./ui.js";
export function newProject(app) {
  const classes = el("div", { class: "stack" });
  const add = () => {
    const line = row(
      input("class-name", "", "text", {
        placeholder: "Class name",
        "aria-label": "Class name",
      }),
      input("class-color", "#5e965b", "color", { "aria-label": "Class color" }),
      input("class-shortcut", "", "text", {
        placeholder: "Key",
        maxLength: 1,
        "aria-label": "Class shortcut",
      }),
    );
    line.append(button("Remove", () => line.remove(), "quiet"));
    classes.append(line);
  };
  const content = form(
    [
      field(
        "Project name",
        input("name", "", "text", {
          required: true,
          placeholder: "My annotation project",
        }),
      ),
      field(
        "Storage folder (optional)",
        input("path", "", "text", {
          placeholder: "Leave blank for the application default",
        }),
        "This path grants the application permission to create the project here.",
      ),
      el("h3", {}, "Object classes"),
      el(
        "p",
        {},
        "Classes are optional at project creation. Generate unassigned masks first, then create classes in the editor.",
      ),
      classes,
      button("Add class", add),
      el(
        "p",
        {},
        "Manual annotation is ready immediately. Optional AI setup is available in Models & AI.",
      ),
    ],
    async (v) => {
      const defs = [...classes.children]
        .map((c) => ({
          name: c.querySelector("[name=class-name]").value.trim(),
          color: c.querySelector("[name=class-color]").value,
          shortcut: c.querySelector("[name=class-shortcut]").value || "",
        }))
        .filter((c) => c.name);
      const p = await app.api.post("/api/projects", {
        name: v.name,
        path: v.path || undefined,
        classes: defs,
      });
      d.close();
      await app.openProject(p.id);
    },
    "Create project",
  );
  const d = dialog("New project", content);
}
export function openProject(app) {
  const d = dialog(
    "Open project",
    form(
      [
        field(
          "Project folder",
          input("path", "", "text", { required: true }),
          "Choose an existing COMPAG project folder on this computer.",
        ),
      ],
      async (v) => {
        const p = await app.api.post("/api/projects/open", v);
        d.close();
        await app.openProject(p.id);
      },
      "Open project",
    ),
  );
}
export function restoreProject(app) {
  const d = dialog(
    "Restore project backup",
    form(
      [
        field(
          "Native backup archive",
          input("archive", "", "file", { required: true, accept: ".zip" }),
        ),
        el(
          "p",
          {},
          "The backup is restored as a separate project in application-managed storage.",
        ),
      ],
      async (v, f) => {
        const data = new FormData();
        data.append("archive", f.elements.archive.files[0]);
        if (v.path) data.append("path", v.path);
        const result = await app.api.post("/api/projects/restore", data);
        if (result.id && result.status) {
          app.showJob(result);
        } else if (result.id) await app.openProject(result.id);
        d.close();
      },
      "Restore backup",
    ),
  );
}
async function deleteProjectDialog(app, pid) {
  const info = await app.api.get(projectPath(pid) + "/deletion-preview");
  const name = input("confirm_name", "", "text", { required: true, autocomplete: "off" });
  const status = el("p", { role: "alert" });
  const content = form([
    el("p", { class: "inline-warning" }, "Permanently delete this project, its images copied into the project, annotations, review rounds and project job outputs? This cannot be undone."),
    el("p", {}, `Project: ${info.name}`),
    el("p", { style: "overflow-wrap:anywhere" }, `Folder to delete: ${info.path}`),
    el("p", {}, `${info.image_count} images · ${info.annotation_count} annotations · ${info.round_count} rounds · ${info.job_count} project jobs`),
    el("p", {}, "Original files and exported copies outside these project/job folders are kept. Files you placed inside the project folder will be deleted. Shared model registrations and weights are kept."),
    info.shared_models_preserved > 0 && el("p", {}, `${info.shared_models_preserved} shared model files will be preserved in application model storage.`),
    info.blocked_reasons.map((reason) => el("p", { class: "inline-warning" }, reason)),
    field("Type the project name to confirm", name, info.name), status,
  ], async (v) => {
    if (v.confirm_name !== info.name) throw Error("Type the exact project name shown above.");
    try {
      const result = await app.api.delete(projectPath(pid), {
        confirm: true, confirm_name: v.confirm_name, expected_revision: info.revision, expected_path: info.path,
      });
      if (app.project?.id === pid) {
        app.project = null; app.projectLabel.textContent = "Choose a project";
        app.selectedImages.clear(); app.currentImageId = null;
      }
      d.close(); await app.navigate("projects");
      notify(result.cleanup_pending ? "Project removed, but some files could not be erased. " + result.cleanup_errors.map((e) => `${e.path}: ${e.error}`).join("; ") : `${info.name} permanently deleted. External original files and shared models kept.`, result.cleanup_pending);
    } catch (error) { status.textContent = error.message; }
  }, "Permanently delete project");
  const submit = content.querySelector("[type=submit]");
  submit.className = "danger";
  const validate = () => { submit.disabled = name.value !== info.name || info.blocked_reasons.length > 0; };
  name.addEventListener("input", validate);validate();
  const d = dialog("Delete project permanently?", el("div", {}, content, button("Cancel", () => d.close())));
}
export async function projectsPage(app) {
  const projects = await app.api.get("/api/projects");
  const list = Array.isArray(projects) ? projects : projects.projects || [];
  return el(
    "div",
    {},
    heading(
      "Projects",
      "Private, local annotation workspaces.",
      button("Open project", () => openProject(app)),
      button("New project", () => newProject(app), "primary"),
    ),
    !list.length
      ? empty(
          "emptyProjects",
          "manualFirst",
          button("Create your first project", () => newProject(app), "primary"),
        )
      : el(
          "div",
          { class: "data-list" },
          list.map((p) =>
            el(
              "article",
              { class: "data-row" },
              row(
                el("h2", {}, p.name),
                badge(`${p.images?.length ?? p.image_count ?? "—"} images`),
              ),
              row(
                button(
                  "Open workspace",
                  () => app.openProject(p.id),
                  "primary",
                ),
                button("Delete project", () => deleteProjectDialog(app, p.id), "danger", { "aria-label": `Delete project ${p.name}` }),
                button("Rename", () => {
                  const d = dialog(
                    "Rename project",
                    form(
                      field(
                        "Name",
                        input("name", p.name, "text", { required: true }),
                      ),
                      async (v) => {
                        const latest = await app.api.get(projectPath(p.id));
                        await app.api.patch(projectPath(p.id), {
                          name: v.name,
                          expected_revision: latest.revision,
                        });
                        d.close();
                        await app.navigate("projects");
                      },
                      "Rename",
                    ),
                  );
                }),
              ),
              p.path && el("small", {}, p.path),
            ),
          ),
        ),
    el(
      "div",
      { class: "row", style: "margin-top:22px" },
      button("Restore backup", () => restoreProject(app)),
      app.project &&
        button("Back up current project", () => app.navigate("export")),
    ),
  );
}
export async function imagesPage(app) {
  let p = app.project;
  const pid = p.id;
  let savingRole = false;
  const trainingPanel = el("div", { "data-testid": "images-training-panel" });
  const guidePanel = el("div");
  const stats = el("div", { class: "stats" });
  const refresh = async () => {
    if (app.project?.id !== pid || app.route !== "images") return;
    await app.refreshProject();
    if (app.project?.id !== pid || app.route !== "images") return;
    p = app.project;
    render();
  };
  let query = "",
    filter = "all";
  const grid = el("div", { class: "image-grid" }),
    search = input("search", "", "search", {
      placeholder: "Search images…",
      "aria-label": "Search images",
    }),
    role = select("role-filter", [
      ["all", "All images"],
      ["incomplete", "Needs review"],
      ["complete", "Reviewed"],
      ...["pool", "train", "validation", "test", "excluded", "missing"].map(
        (v) => [v, v],
      ),
    ]);
  const render = () => {
    trainingPanel.replaceChildren(...(p.images.length ? [imagesTrainingPanel(app, savingRole)] : []));
    guidePanel.replaceChildren(imagesGuide(app));
    stats.replaceChildren(...[
      [p.images.length, "Images"],
      [p.images.filter(i => i.complete).length, "Reviewed"],
      [p.classes.filter(c => !c.archived).length, "Object classes"],
      [p.rounds.length, "Review rounds"],
    ].map(([n, label]) => el("div", { class: "stat" }, el("strong", {}, String(n)), el("small", {}, label))));
    const shown = p.images.filter(
      (i) =>
        i.name.toLowerCase().includes(query.toLowerCase()) &&
        (filter === "all" ||
          (filter === "complete" && i.complete) ||
          (filter === "incomplete" && !i.complete) ||
          (filter === "missing" && i.missing) ||
          i.role === filter),
    );
    grid.replaceChildren(
      ...shown.map((i) => {
        const selected = input(`selected-${i.id}`, "", "checkbox", {
          "aria-label": `Select ${i.name}`,
          checked: app.selectedImages.has(i.id),
          onchange: (e) => {
            e.target.checked
              ? app.selectedImages.add(i.id)
              : app.selectedImages.delete(i.id);
          },
        });
        const roles = select(
          `role-${i.id}`,
          ["pool", "train", "validation", "test", "excluded"],
          i.role,
          {
            "aria-label": `Dataset role for ${i.name}`,
            onchange: async (e) => {
              if (savingRole) return;
              savingRole = true;
              grid.querySelectorAll("select,button").forEach(c => { c.disabled = true; });
              trainingPanel.replaceChildren(imagesTrainingPanel(app, true));
              try {
                await app.api.patch(projectPath(p.id) + `/images/${i.id}`, {
                  role: e.target.value,
                  expected_revision: app.project.revision,
                });
                await refresh();
              } catch (err) {
                e.target.value = i.role;
                notify(err.message, true);
              } finally {
                savingRole = false;
                if (app.project?.id === pid && app.route === "images") render();
              }
            },
          },
        );
        return el(
          "article",
          { class: "image-card" },
          el(
            "button",
            {
              type: "button",
              class: "thumb",
              "aria-label": `Annotate ${i.name}`,
              onclick: () => app.annotate(i.id),
            },
            el("img", {
              src:
                i.thumbnail_url ||
                projectPath(p.id) + `/images/${i.id}/thumbnail`,
              alt: i.name,
              loading: "lazy",
            }),
          ),
          el(
            "div",
            { class: "image-card-info" },
            el("strong", {}, i.name),
            el("small", {}, `${i.width} × ${i.height} px`),
            row(
              selected,
              badge(
                i.missing
                  ? "Missing source"
                  : i.complete
                    ? "Reviewed"
                    : "Needs review",
                i.missing ? "error" : i.complete ? "good" : "",
              ),
            ),
            row(
              field("Dataset role", roles),
              button("Details", () => imageDetails(app, i), "quiet"),
              button("Remove", () => removeImage(app, i), "danger", { "aria-label": `Remove ${i.name} from project` }),
            ),
            ["pool", "train", "validation"].includes(i.role) && (
              i.complete ? el("small", {}, `Review confirmed for ${i.role}. Continue to Train to check readiness.`)
                : button("Confirm review", () => confirmImageReview(app, i.id, refresh), "primary", {
                  "aria-label": `Confirm review for ${i.name}`, disabled: savingRole || i.missing,
                })
            ),
          ),
        );
      }),
    );
    if (!shown.length)
      grid.replaceChildren(
        empty("No matching images", "Import images or change the filter."),
      );
  };
  search.addEventListener("input", (e) => {
    query = e.target.value;
    render();
  });
  role.addEventListener("change", (e) => {
    filter = e.target.value;
    render();
  });
  render();
  return el(
    "div",
    {},
    heading(
      "Images",
      "Build your dataset, then review every object at full resolution.",
      button("Import images", () => uploadDialog(app), "primary"),
    ),
    guidePanel,
    stats,
    trainingPanel,
    el(
      "div",
      { class: "filterbar" },
      search,
      role,
      button("Export selected", () => app.navigate("export")),
      button("Generate masks", () => generateDialog(app)),
      button("Predict selected", () => app.predictDialog()),
    ),
    p.images.length
      ? grid
      : empty(
          "Add your first images",
          "Select files, drop a folder, or import from an approved local path.",
          button("Import images", () => uploadDialog(app), "primary"),
        ),
  );
}
async function removeImage(app, image) {
  const pid = app.project.id;
  if (!await confirmAction("Remove image?",
    `Remove ${image.name} from this project? Its annotations will no longer appear in the active dataset or new exports. The original file is kept. Images in started review rounds cannot be removed.`, "Remove image")) return false;
  await app.api.delete(projectPath(pid) + `/images/${image.id}`, {
    expected_revision: app.project.revision, confirm: true,
  });
  app.selectedImages.delete(image.id);
  if (app.currentImageId === image.id) app.currentImageId = null;
  await app.navigate("images");
  notify(`${image.name} removed from project. Original file kept.`);
  return true;
}
function imageDetails(app, i) {
  const path = projectPath(app.project.id) + `/images/${i.id}`;
  const d = dialog(
    i.name,
    el(
      "div",
      {},
      details("Image information", i),
      form(
        field(
          "Related image group",
          input("group_id", i.group_id || ""),
          "Keep related images in the same train/validation/test split.",
        ),
        async (v) => {
          await app.api.patch(path, {
            group_id: v.group_id || null,
            expected_revision: app.project.revision,
          });
          d.close();
          await app.refreshProject();
          await app.navigate("images");
        },
        "Save group",
      ),
      i.missing &&
        form(
          [
            field(
              "Replacement original path",
              input("path", "", "text", { required: true }),
            ),
            check("approved", "I approve reading this file"),
          ],
          async (v) => {
            if (!v.approved) throw Error("Approve the file first.");
            await app.api.post(path + "/relink", {
              ...v,
              expected_revision: app.project.revision,
            });
            d.close();
            await app.refreshProject();
            await app.navigate("images");
          },
          "Relink original",
        ),
      button("Remove from project", async () => { if (await removeImage(app, i)) d.close(); }, "danger"),
    ),
  );
}
export async function labelsPage(app) {
  const p = app.project,
    path = projectPath(p.id);
  const apply = async (body) => {
    const preview = await app.api.post(path + "/classes/preview", {
      ...body,
      expected_revision: app.project.revision,
    });
    const content = el(
      "div",
      {},
      details("Class change impact", preview, true),
    );
    const d = dialog(
      "Review class change",
      el(
        "div",
        {},
        content,
        button(
          "Apply class change",
          async () => {
            await app.api.post(path + "/classes", {
              ...body,
              expected_revision: app.project.revision,
              confirm: true,
            });
            d.close();
            await app.refreshProject();
            await app.navigate("labels");
          },
          "primary",
        ),
      ),
    );
  };
  const rows = p.classes.map((c, index) => {
    const name = input(`name-${c.id}`, c.name),
      color = input(`color-${c.id}`, c.color, "color"),
      shortcut = input(`shortcut-${c.id}`, c.shortcut || "", "text", {
        maxLength: 1,
      });
    return el(
      "div",
      { class: "class-row" },
      field("Color", color),
      field(c.archived ? "Archived class" : "Class name", name),
      field("Shortcut", shortcut),
      row(
        button("Save", () =>
          apply({
            action: "update",
            id: c.id,
            name: name.value,
            color: color.value,
            shortcut: shortcut.value || "",
          }),
        ),
        button(
          "↑",
          () => {
            const ids = p.classes.map((c) => c.id);
            [ids[index - 1], ids[index]] = [ids[index], ids[index - 1]];
            return apply({ action: "reorder", ids });
          },
          "quiet",
          { disabled: index === 0, "aria-label": `Move ${c.name} up` },
        ),
        button(
          "Archive",
          () => apply({ action: "archive", id: c.id }),
          "quiet",
          { disabled: c.archived },
        ),
        button(
          "Remap",
          () => {
            const target = select(
              "target_id",
              p.classes
                .filter((v) => v.id !== c.id && !v.archived)
                .map((v) => [v.id, v.name]),
            );
            const d = dialog(
              "Remap class",
              el(
                "div",
                {},
                el(
                  "p",
                  {},
                  `Move annotations from “${c.name}” into another class. Existing model mappings require compatibility review.`,
                ),
                field("Destination class", target),
                button(
                  "Preview remap",
                  async () => {
                    if (!target.value)
                      throw Error("Create another class first.");
                    d.close();
                    await apply({
                      action: "remap",
                      id: c.id,
                      target_id: target.value,
                    });
                  },
                  "primary",
                ),
              ),
            );
          },
          "quiet",
        ),
      ),
    );
  });
  return el(
    "div",
    {},
    heading(
      "Labels",
      "Stable class identities, with versioned changes and explicit model mappings.",
    ),
    card(
      "Project classes",
      rows.length
        ? rows
        : el(
            "p",
            {},
            "No classes yet. Add a class before accepting annotations.",
          ),
    ),
    card(
      "Add a class",
      form(
        [
          row(
            field("Name", input("name", "", "text", { required: true })),
            field("Color", input("color", "#6a9759", "color")),
            field("Shortcut", input("shortcut", "", "text", { maxLength: 1 })),
          ),
        ],
        async (v) => {
          await app.api.post(path + "/classes", {
            action: "create",
            ...v,
            shortcut: v.shortcut || "",
            expected_revision: p.revision,
          });
          await app.refreshProject();
          await app.navigate("labels");
        },
        "Create class",
      ),
    ),
  );
}
