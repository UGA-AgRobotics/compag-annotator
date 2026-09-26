export class APIError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}
export class API {
  async init() {
    this.session = await this.get("/api/session");
    return this.session;
  }
  async request(path, method = "GET", body, signal) {
    const headers = { Accept: "application/json" };
    if (method !== "GET") {
      headers["X-Compag-Token"] = this.session?.token || "";
      if (!(body instanceof FormData))
        headers["Content-Type"] = "application/json";
    }
    const res = await fetch(path, {
      method,
      headers,
      credentials: "same-origin",
      cache: "no-store",
      signal,
      body:
        body == null
          ? undefined
          : body instanceof FormData
            ? body
            : JSON.stringify(body),
    });
    const data = await res
      .json()
      .catch(() => ({ detail: `Request failed (${res.status})` }));
    if (!res.ok)
      throw new APIError(
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail ?? data),
        res.status,
        data,
      );
    return data;
  }
  get(path, signal) {
    return this.request(path, "GET", undefined, signal);
  }
  post(path, body = {}, signal) {
    return this.request(path, "POST", body, signal);
  }
  patch(path, body) {
    return this.request(path, "PATCH", body);
  }
  delete(path, body) {
    return this.request(path, "DELETE", body);
  }
  async waitJob(job, { signal, onProgress = () => {} } = {}) {
    let j = job;
    while (true) {
      if (signal?.aborted) throw new DOMException("Cancelled", "AbortError");
      onProgress(j);
      if (j.status === "complete") return j;
      if (
        ["failed", "cancelled", "interrupted", "awaiting_review"].includes(
          j.status,
        )
      )
        throw new APIError(
          j.error || `Job ${j.status}. Open Jobs for details.`,
          0,
          j,
        );
      await new Promise((resolve, reject) => {
        const done = () => {
          signal?.removeEventListener("abort", abort);
          resolve();
        };
        const timeout = setTimeout(done, 700);
        const abort = () => {
          clearTimeout(timeout);
          reject(new DOMException("Cancelled", "AbortError"));
        };
        signal?.addEventListener("abort", abort, { once: true });
      });
      j = await this.get(`/api/jobs/${encodeURIComponent(j.id)}`, signal);
    }
  }
}
