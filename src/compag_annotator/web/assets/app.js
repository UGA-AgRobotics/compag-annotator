import { API } from "./api.js";
import {
  el,
  button,
  row,
  badge,
  empty,
  notify,
  projectPath,
  confirmAction,
} from "./ui.js";
import {
  projectsPage,
  imagesPage,
  labelsPage,
  newProject,
  restoreProject,
} from "./projects.js";
import { modelsPage, predictDialog } from "./models.js";
import { roundsPage, trainingPage, exportPage } from "./workflows.js";
import { jobsPage } from "./jobs.js";
import { helpPage } from "./help.js";
import { Editor } from "./editor.js";
const routes = [
  ["projects", "▦", "Projects"],
  ["images", "▧", "Images"],
  ["labels", "◇", "Labels"],
  ["annotate", "⌁", "Annotate"],
  ["models", "◈", "Models & AI"],
  ["rounds", "▤", "Rounds"],
  ["train", "↗", "Train"],
  ["export", "⇧", "Export"],
  ["jobs", "≡", "Jobs"],
  ["help", "?", "Help & Settings"],
];
import { themeControl } from "./theme.js";
class App {
  constructor() {
    this.api = new API();
    this.project = null;
    this.selectedImages = new Set();
    this.device = localStorage.getItem("compag-device") || "cpu";
    this.route = "projects";
    this.navSeq = 0;
  }
  async start() {
    await this.api.init();
    this.main = el("main", { id: "main", tabIndex: -1 });
    this.projectLabel = el(
      "span",
      { class: "project-title" },
      "Choose a project",
    );
    this.navButtons = new Map();
    const nav = el(
      "nav",
      { "aria-label": "Main navigation" },
      routes.map(([key, symbol, name]) => {
        const b = button(
          [
            el("span", { class: "nav-symbol", "aria-hidden": "true" }, symbol),
            el("span", { class: "nav-label" }, name),
          ],
          () => this.navigate(key),
          "",
          { title: name, "aria-label": name },
        );
        this.navButtons.set(key, b);
        return b;
      }),
    );
    document.getElementById("app").replaceChildren(
      el(
        "div",
        { class: "shell" },
        el(
          "aside",
          { class: "sidebar" },
          el(
            "div",
            { class: "brand" },
            el("img", { src: "/assets/icon.svg", alt: "" }),
            el(
              "div",
              {},
              el("strong", {}, "COMPAG"),
              el("small", {}, "ANNOTATOR"),
            ),
          ),
          nav,
          el(
            "div",
            { class: "sidebar-foot" },
            el("span", { class: "local-dot" }),
            "Local & private",
            el("br"),
            `v${this.api.session.version}`,
          ),
        ),
        el(
          "div",
          { class: "workspace" },
          el(
            "header",
            { class: "topbar" },
            this.projectLabel,
            row(
              themeControl(),
              badge(
                this.api.session.qa_mode ? "Automated QA" : "Local workspace",
              ),
              el("span", { class: "version" }, "Manual tools ready"),
              button("New project", () => newProject(this), "quiet"),
            ),
          ),
          this.main,
        ),
      ),
    );
    window.addEventListener("hashchange", () => {
      const route = location.hash.slice(1).split("/")[0];
      this.navigate(routes.some((r) => r[0] === route) ? route : "projects");
    });
    await this.navigate("projects");
  }
  async refreshProject() {
    if (!this.project) return;
    this.project = await this.api.get(projectPath(this.project.id));
    this.projectLabel.textContent = this.project.name;
  }
  async openProject(id) {
    if (this.editor && !(await this.editor.canLeave())) return;
    this.project = await this.api.get(projectPath(id));
    this.projectLabel.textContent = this.project.name;
    this.selectedImages.clear();
    await this.navigate("images");
  }
  async annotate(id) {
    await this.navigate("annotate", id);
  }
  async navigate(route, imageId) {
    if (this.editor && !(await this.editor.canLeave())) return;
    const seq = ++this.navSeq;
    this.cleanup?.();
    this.cleanup = null;
    this.editor?.destroy();
    this.editor = null;
    this.route = route;
    for (const [key, b] of this.navButtons) {
      b.classList.toggle("active", key === route);
      b.setAttribute("aria-current", key === route ? "page" : "false");
    }
    if (location.hash !== `#${route}`) {
      history.replaceState(null, "", `#${route}`);
    }
    this.main.className = route === "annotate" ? "editor-main" : "";
    this.main.replaceChildren(
      el("p", { class: "startup", role: "status" }, "Loading workspace…"),
    );
    try {
      if (
        ["images", "labels", "annotate", "rounds", "train", "export"].includes(
          route,
        ) &&
        !this.project
      ) {
        this.main.className = "";
        this.main.replaceChildren(
          empty(
            "Open a project to continue",
            "Create a local project or open one from Projects.",
            button("Create project", () => newProject(this), "primary"),
            button("Choose project", () => this.navigate("projects")),
          ),
        );
        return;
      }
      if (this.project && route !== "projects") await this.refreshProject();
      if (seq !== this.navSeq) return;
      let page;
      if (route === "annotate") {
        if (!this.project.images.length) {
          this.main.className = "";
          page = empty(
            "Import an image to begin",
            "Manual tools are ready. Choose Images to add your full-resolution originals.",
            button("Open Images", () => this.navigate("images"), "primary"),
          );
        } else {
          this.editor = new Editor(this, imageId || this.currentImageId);
          this.currentImageId = this.editor.image.id;
          page = await this.editor.mount();
        }
      } else
        page = await (
          {
            projects: projectsPage,
            images: imagesPage,
            labels: labelsPage,
            models: modelsPage,
            rounds: roundsPage,
            train: trainingPage,
            export: exportPage,
            jobs: jobsPage,
            help: helpPage,
          }[route] || projectsPage
        )(this);
      if (seq === this.navSeq) {
        this.main.replaceChildren(page);
        this.main.scrollTop = 0;
      }
    } catch (e) {
      if (seq === this.navSeq) {
        this.main.className = "";
        this.main.replaceChildren(
          empty(
            "This view could not load",
            e.message,
            button("Retry", () => this.navigate(route), "primary"),
            button("Open Jobs", () => this.navigate("jobs")),
          ),
        );
        notify(e.message, true);
      }
    }
  }
  showJob(job) {
    notify(`Job ${job.kind || ""} ${job.status || "submitted"}.`);
    return this.navigate("jobs");
  }
  predictDialog() {
    return predictDialog(this);
  }
  restoreProject() {
    return restoreProject(this);
  }
}
const app = new App();
app.start().catch((error) => {
  document.getElementById("app").replaceChildren(
    empty(
      "Unable to connect to the local application",
      error.message,
      button("Retry connection", () => location.reload(), "primary"),
    ),
  );
});
