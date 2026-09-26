import { el, button, row, heading, badge, details, notify } from "./ui.js";
import { array } from "./models.js";
import { jobGuide, openJobContext } from "./workflow-guide.js";
export async function jobsPage(app) {
  const root = el(
    "div",
    {},
    heading(
      "Jobs",
      "Real progress, saved logs and recoverable failures.",
      button("Refresh", () => app.navigate("jobs")),
    ),
  );
  const list = el("div", { class: "data-list" });
  root.append(list);
  let stopped = false;
  app.cleanup = () => {
    stopped = true;
    clearTimeout(timer);
  };
  let timer;
  const refresh = async () => {
    try {
      const jobs = array(await app.api.get("/api/jobs"), "jobs");
      if (stopped) return;
      const opened = new Set(
        [...list.querySelectorAll("details[open]")].map(
          (d) =>
            `${d.closest("[data-job-id]").dataset.jobId}:${d.querySelector("summary").textContent}`,
        ),
      );
      list.replaceChildren(...jobs.map((j) => jobCard(app, j)));
      for (const d of list.querySelectorAll("details"))
        d.open = opened.has(
          `${d.closest("[data-job-id]").dataset.jobId}:${d.querySelector("summary").textContent}`,
        );
      if (!jobs.length)
        list.append(
          el(
            "p",
            {},
            "No jobs yet. Imports, exports and optional AI work appear here.",
          ),
        );
    } catch (e) {
      if (!stopped) notify(e.message, true);
    } finally {
      if (!stopped) timer = setTimeout(refresh, 2000);
    }
  };
  await refresh();
  return root;
}
export function jobCard(app, job) {
  const progress = job.progress || {},
    status = job.status;
  const ratio = status === "complete" ? 1 : progress.total_bytes > 0 ? progress.bytes / progress.total_bytes
    : Number.isFinite(progress.setup_percent) ? progress.setup_percent / 100
    : Number.isFinite(progress.percent)
    ? progress.percent / 100
    : Number.isFinite(progress.completed_tiles) && progress.total_tiles > 0
      ? progress.completed_tiles / progress.total_tiles
      : Number.isFinite(progress.completed) &&
          Number.isFinite(progress.total) &&
          progress.total > 0
        ? progress.completed / progress.total
        : null;
  return el(
    "article",
    { class: "data-row", dataset: { jobId: job.id } },
    row(
      el("h2", {}, job.kind.replaceAll("_", " ")),
      badge(
        status,
        status === "complete"
          ? "good"
          : ["failed", "cancelled", "interrupted"].includes(status)
            ? "error"
            : status === "awaiting_review"
              ? "warn"
              : "",
      ),
    ),
    el("small", {}, `${job.id} · ${job.created_at || ""}`),
    el("p", {}, progress.setup_label || progress.stage || ""),
    el("p", { class: "progress-percent", "aria-live": "polite" },
      ratio === null ? "Progress: total not yet known" : `${Math.max(0, Math.min(100, ratio * 100)).toFixed(1)}%${progress.setup_total ? ` · ${progress.setup_completed}/${progress.setup_total} setup steps (not elapsed time)` : ""}`),
    Number.isFinite(progress.bytes) && el("small", {}, `${formatBytes(progress.bytes)}${progress.total_bytes > 0 ? ` / ${formatBytes(progress.total_bytes)}` : " downloaded · total size unavailable"}`),
    Number.isFinite(progress.download_bytes) && el("div", { class: "dependency-progress" },
      el("p", {}, `${progress.download_label || "Dependency download"}: ${progress.download_total_bytes > 0 ? (100 * progress.download_bytes / progress.download_total_bytes).toFixed(1) + "%" : "total unknown"} · ${formatBytes(progress.download_bytes)}${progress.download_total_bytes > 0 ? " / " + formatBytes(progress.download_total_bytes) : ""}`),
      el("progress", { "aria-label": "Dependency download", ...(progress.download_total_bytes > 0 ? { value: progress.download_bytes, max: progress.download_total_bytes } : {}) })),
    Number.isFinite(progress.completed_tiles) &&
      el(
        "p",
        {},
        `${progress.completed_tiles} / ${progress.total_tiles} tiles saved`,
      ),
    ["running", "queued"].includes(status) &&
      el("progress", {
        "aria-label": "Job progress",
        ...(ratio === null
          ? {}
          : { value: Math.max(0, Math.min(1, ratio)), max: 1 }),
      }),
    job.error &&
      el(
        "p",
        { class: "inline-warning", role: "alert" },
        typeof job.error === "string" ? job.error : JSON.stringify(job.error),
      ),
    job.kind === "train" && job.result?.optimization?.passed &&
      el("p", { "data-testid": "training-update-check" },
        `Weight updates verified: ${job.result.optimization.positive_lr_steps} optimizer steps with a nonzero learning rate. Learnable weights changed. Check validation metrics to assess prediction quality.`),
    jobGuide(app, job),
    row(
      ["generate", "boundary_annotation", "predict"].includes(job.kind) &&
        button("Open annotation results", () => openJobContext(app, job, "annotate"), status === "complete" ? "primary" : ""),
      ["running", "queued"].includes(status) &&
        button("Cancel job", async () => {
          await app.api.post(`/api/jobs/${job.id}/cancel`);
          await app.navigate("jobs");
        }),
      ["failed", "cancelled", "interrupted"].includes(status) &&
        button(
          job.kind === "generate" ? "Resume unfinished tiles" : "Retry job",
          async () => {
            app.showJob(await app.api.post(`/api/jobs/${job.id}/retry`));
          },
        ),
      job.kind === "train" &&
        ["failed", "cancelled", "interrupted"].includes(status) &&
        button("Resume from checkpoint", () => {
          app.resumeJobId = job.id;
          app.navigate("train");
        }),
      status === "awaiting_review" &&
        button(
          ["boundary_annotation","predict"].includes(job.kind)
            ? "Review annotation changes"
            : "Review preparation changes",
          () =>
            ["boundary_annotation","predict"].includes(job.kind)
              ? openJobContext(app,job,"annotate")
              : app.navigate("train"),
          "primary",
        ),
      status === "complete" &&
        job.result?.artifact &&
        el(
          "a",
          {
            class: "button",
            href: `/api/jobs/${job.id}/artifact`,
            download: true,
          },
          "Download artifact",
        ),
      status === "complete" &&
        job.kind === "restore" &&
        job.result?.id &&
        button("Open restored project", () => app.openProject(job.result.id)),
      status === "complete" &&
        job.kind === "train" &&
        button("Inspect and activate model", () => app.navigate("models")),
    ),
    details("Progress details", progress),
    details("Job log", job.log || []),
    job.result && details("Result & validation report", job.result),
  );
}

function formatBytes(value) { return `${(value / 1048576).toFixed(1)} MiB`; }
