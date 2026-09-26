import { t } from "./i18n.js";
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else if (key === "text") node.textContent = t(value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key in node && !key.startsWith("aria-")) node[key] = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat(Infinity))
    if (child != null && child !== false)
      node.append(
        child instanceof Node ? child : document.createTextNode(t(child)),
      );
  return node;
}
export function notify(message, error = false) {
  const n = document.getElementById("notice");
  n.replaceChildren(
    el("span", {}, message),
    button("Dismiss", () => n.replaceChildren(), "quiet"),
  );
  n.className = error ? "notice error" : "notice";
  n.setAttribute("role", error ? "alert" : "status");
}
export function button(label, action, className = "", attrs = {}) {
  return el(
    "button",
    {
      type: "button",
      class: className,
      ...attrs,
      onclick: async (e) => {
        if (!action || e.currentTarget.disabled) return;
        const b = e.currentTarget;
        b.disabled = true;
        try {
          await action(e);
        } catch (err) {
          notify(err.message || String(err), true);
        } finally {
          b.disabled = false;
        }
      },
    },
    label,
  );
}
export function input(name, value = "", type = "text", attrs = {}) {
  return el("input", { name, id: name, type, value, ...attrs });
}
export function select(name, options, value = "", attrs = {}) {
  return el(
    "select",
    { name, id: name, ...attrs },
    options.map((o) =>
      el(
        "option",
        {
          value: typeof o === "string" ? o : o[0],
          selected: (typeof o === "string" ? o : o[0]) === value,
        },
        typeof o === "string" ? o : o[1],
      ),
    ),
  );
}
export const field = (label, control, hint = "") => {
  if (
    control instanceof HTMLElement &&
    control.matches("input,select,textarea") &&
    !control.hasAttribute("aria-label")
  )
    control.setAttribute("aria-label", t(label));
  return el(
    "label",
    { class: "field" },
    el("span", {}, label),
    control,
    hint && el("small", {}, hint),
  );
};
export const check = (name, label, checked = false) =>
  el(
    "label",
    { class: "check" },
    input(name, "", "checkbox", { checked }),
    el("span", {}, label),
  );
export const card = (title, ...content) =>
  el("section", { class: "card" }, title && el("h2", {}, title), ...content);
export const row = (...items) => el("div", { class: "row" }, ...items);
export const badge = (text, variant = "") =>
  el("span", { class: `badge ${variant}` }, text);
export const empty = (title, description, ...actions) =>
  el(
    "div",
    { class: "empty" },
    el("span", { class: "empty-symbol", "aria-hidden": "true" }, "◇"),
    el("h2", {}, title),
    el("p", {}, description),
    row(...actions),
  );
export const heading = (title, description, ...actions) =>
  el(
    "header",
    { class: "page-heading" },
    el(
      "div",
      {},
      el("p", { class: "eyebrow" }, "LOCAL WORKSPACE"),
      el("h1", {}, title),
      el("p", {}, description),
    ),
    row(...actions),
  );
export function details(title, data, open = false) {
  return el(
    "details",
    { open },
    el("summary", {}, title),
    el(
      "pre",
      {},
      typeof data === "string" ? data : JSON.stringify(data, null, 2),
    ),
  );
}
export function values(form) {
  const d = Object.fromEntries(new FormData(form));
  for (const c of form.querySelectorAll("input[type=checkbox]"))
    d[c.name] = c.checked;
  for (const c of form.querySelectorAll("input[type=number]"))
    d[c.name] = c.value === "" ? null : Number(c.value);
  return d;
}
export function form(content, onSubmit, label = "Save") {
  const f = el(
    "form",
    {},
    content,
    el("button", { type: "submit", class: "primary" }, label),
  );
  f.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (f.dataset.busy) return;
    f.dataset.busy = "true";
    const b = f.querySelector("[type=submit]");
    b.disabled = true;
    try {
      await onSubmit(values(f), f);
    } catch (err) {
      notify(err.message || String(err), true);
    } finally {
      delete f.dataset.busy;
      b.disabled = false;
    }
  });
  return f;
}
export function dialog(title, content) {
  const previous = document.activeElement;
  const d = el(
    "dialog",
    { "aria-label": title },
    el(
      "div",
      { class: "dialog-head" },
      el("h2", {}, title),
      button("Close", () => d.close(), "quiet"),
    ),
    content,
  );
  document.body.append(d);
  d.addEventListener("close", () => {
    d.remove();
    previous?.focus();
  });
  d.showModal();
  return d;
}
export function confirmAction(title, message, label = "Confirm") {
  return new Promise((resolve) => {
    let result = false;
    const d = dialog(
      title,
      el(
        "div",
        {},
        el("p", {}, message),
        row(
          button("Cancel", () => d.close()),
          button(
            label,
            () => {
              result = true;
              d.close();
            },
            "primary",
          ),
        ),
      ),
    );
    d.addEventListener("close", () => resolve(result));
  });
}
export const projectPath = (id) => `/api/projects/${encodeURIComponent(id)}`;
export function safeURL(url) {
  const u = new URL(url, location.href);
  if (u.origin !== location.origin)
    throw Error("Unexpected external artifact URL.");
  return u.href;
}
