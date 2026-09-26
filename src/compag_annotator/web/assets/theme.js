import { el, select } from "./ui.js";
export function themeControl() {
  const control = select("theme", [["light", "Light"], ["dark", "Dark / night"]], document.documentElement.dataset.theme || "light", { "aria-label": "Theme" });
  control.addEventListener("change", () => {
    document.documentElement.dataset.theme = control.value;
    try { localStorage.setItem("compag-theme", control.value); } catch (_) { /* Optional local preference. */ }
  });
  return el("label", { class: "theme-control" }, "Theme", control);
}
