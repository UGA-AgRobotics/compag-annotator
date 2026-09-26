import {
  el,
  button,
  input,
  select,
  field,
  check,
  row,
  details,
  form,
  notify,
  projectPath,
} from "./ui.js";
export function importFormFor(app, formats) {
  const p = app.project,
    rows = el("div", { class: "stack" }),
    imageRows = el("div", { class: "stack" }),
    info = el(
      "p",
      {},
      "Matching class names are linked automatically. Add a mapping to rename source labels during import.",
    );
  const mapping = [];
  const images = [];
  function addRow(source = "", label = source) {
    const matching = p.classes.find((c) => !c.archived && c.name === label),
      from = input("source-label", source, "text", {
        "aria-label": "Source label name or category number",
        placeholder: "Source label name / category number",
      }),
      to = select(
        "target-label",
        [
          ["", "Match name / create source class"],
          ...p.classes.filter((c) => !c.archived).map((c) => [c.id, c.name]),
        ],
        matching?.id || "",
        { "aria-label": `Project class for ${label || "source label"}` },
      );
    const record = { from, to };
    mapping.push(record);
    const line = el(
      "div",
      { class: "data-row" },
      label && el("strong", {}, label),
      row(
        from,
        el("span", {}, "→"),
        to,
        button(
          "Remove mapping",
          () => {
            mapping.splice(mapping.indexOf(record), 1);
            line.remove();
          },
          "quiet",
        ),
      ),
    );
    rows.append(line);
  }
  const file = input("file", "", "file", {
    required: true,
    accept: ".json,.zip,.xml,.txt,.yaml,.yml,.png",
  });
  const format = select("format", formats);
  const binary = el(
    "div",
    { hidden: true },
    el("h3", {}, "Standalone binary PNG target"),
    el(
      "p",
      {},
      "For a single binary mask, choose its image and class. Archives with mapping metadata use their own mappings.",
    ),
    field(
      "Mask image",
      select("image_id", [
        ["", "Use mapping metadata"],
        ...p.images.map((i) => [i.id, i.name]),
      ]),
    ),
    field(
      "Mask class",
      select("class_id", [
        ["", "Use mapping metadata"],
        ...p.classes.filter((c) => !c.archived).map((c) => [c.id, c.name]),
      ]),
    ),
  );
  format.addEventListener("change", () => {
    binary.hidden = format.value !== "png";
  });
  file.addEventListener("change", async () => {
    const chosen = file.files[0];
    rows.replaceChildren();
    imageRows.replaceChildren();
    mapping.length = 0;
    images.length = 0;
    if (!chosen) return;
    if (
      chosen.size > 20 * 1024 * 1024 ||
      !chosen.name.toLowerCase().endsWith(".json")
    ) {
      info.textContent =
        "Names are matched during import. For an archive, add explicit mappings below if source labels differ from your project.";
      return;
    }
    try {
      const doc = JSON.parse(await chosen.text());
      const labels =
        doc.categories?.map((c) => [String(c.id), c.name]) ||
        doc.classes?.map((c) => [String(c.index ?? c.id ?? c.name), c.name]) ||
        [...new Set((doc.shapes || []).map((s) => s.label))].map((name) => [
          name,
          name,
        ]);
      labels.forEach(([id, name]) => addRow(id, name));
      for (const image of doc.images || []) {
        const name = image.file_name || image.name,
          matching = p.images.filter(
            (i) => i.name === name || i.name === name?.split("/").at(-1),
          );
        if (matching.length !== 1) {
          const target = select(
            "target-image",
            [
              ["", "Choose a project image"],
              ...p.images.map((i) => [i.id, i.name]),
            ],
            "",
            { "aria-label": `Project image for ${name}` },
          );
          images.push({ name, id: String(image.id), target });
          imageRows.append(field(`Source image: ${name}`, target));
        }
      }
      info.textContent = labels.length
        ? `Found ${labels.length} source labels. Review the project class mapping before importing.`
        : "No label table detected. Names are matched during import, or add explicit mappings below.";
    } catch (e) {
      notify(`Could not preview annotation labels: ${e.message}`, true);
    }
  });
  return form(
    [
      field("Annotation format", format),
      field("Annotation file / archive", file),
      el(
        "p",
        {},
        "Import the source images first. Imported annotations begin unreviewed.",
      ),
      binary,
      el("h3", {}, "Class mapping"),
      info,
      rows,
      button("Add class mapping", () => addRow()),
      check(
        "create_classes",
        "Create a class for unmatched source names",
        true,
      ),
      imageRows,
      el(
        "details",
        {},
        el("summary", {}, "Advanced import options"),
        field(
          "YOLO class names (one per line, model index order)",
          el("textarea", {
            name: "class_names",
            placeholder: "Object\nSecond class",
          }),
          "Needed only when a YOLO archive has no dataset class table.",
        ),
        field(
          "Additional adapter options (JSON)",
          el("textarea", { name: "options", value: "{}" }),
        ),
      ),
      button("Restore native backup", () => app.restoreProject()),
    ],
    async (v, f) => {
      const options = JSON.parse(v.options || "{}");
      options.create_classes = v.create_classes;
      if (v.format === "png") {
        if (v.image_id) options.image_id = v.image_id;
        if (v.class_id) options.class_id = v.class_id;
      }
      options.class_mapping = {
        ...(options.class_mapping || {}),
        ...Object.fromEntries(
          mapping
            .filter((r) => r.from.value.trim() && r.to.value)
            .map((r) => [r.from.value.trim(), r.to.value]),
        ),
      };
      if (images.some((r) => !r.target.value))
        throw Error(
          "Choose the project image for each unmatched source image.",
        );
      options.image_mapping = {
        ...(options.image_mapping || {}),
        ...Object.fromEntries(
          images.flatMap((r) => [
            [r.name, r.target.value],
            [r.id, r.target.value],
          ]),
        ),
      };
      if (v.class_names.trim())
        options.class_names = v.class_names
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean);
      const data = new FormData();
      data.append("file", f.elements.file.files[0]);
      data.append("format", v.format);
      data.append("options", JSON.stringify(options));
      app.showJob(await app.api.post(projectPath(p.id) + "/imports", data));
    },
    "Import annotations",
  );
}
