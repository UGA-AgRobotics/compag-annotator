// English is the source locale. Additional locale dictionaries may override any
// English key without changing component code. All DOM text uses t().
export const en = Object.freeze({
  appName: "COMPAG Annotator",
  Projects: "Projects",
  Images: "Images",
  Labels: "Labels",
  Annotate: "Annotate",
  "Models & AI": "Models & AI",
  Rounds: "Rounds",
  Train: "Train",
  Export: "Export",
  Jobs: "Jobs",
  "Help & Settings": "Help & Settings",
  "Generate masks": "Generate masks",
  "Select (V)": "Select (V)",
  "Paint class (L)": "Paint class (L)",
  "Edit geometry (G)": "Edit geometry (G)",
  "Merge selected masks": "Merge selected masks",
  "Assign class": "Assign class",
  "Create class and assign": "Create class and assign",
  saved: "All changes saved",
  saving: "Saving…",
  conflict: "This image changed elsewhere. Reload before continuing.",
  emptyProjects: "Your images. Your labels. Your workspace.",
  manualFirst:
    "Create a local project and start annotating. AI assistance is optional.",
  editorHint:
    "Scroll to zoom · Space + drag to pan · V select · G edit geometry · L paint class",
  reviewAttestation:
    "I reviewed all objects and the entire image, including missing objects. An empty image is an intentional negative.",
  staleAssist:
    "This preview belongs to an earlier image, edit, model or prompt. Request a new preview.",
});
const locales = { en };
export const t = (key, vars = {}) =>
  Object.entries(vars).reduce(
    (s, [k, v]) => s.replaceAll(`{${k}}`, String(v)),
    locales.en[key] ?? String(key),
  );
