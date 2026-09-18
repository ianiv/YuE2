// HTTP + SSE client for docs/API.md.

export class ApiError extends Error {
  constructor(status, code, message) { super(message); this.status = status; this.code = code; }
}

async function request(method, path, body, opts = {}) {
  const init = { method, headers: {} };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(body); }
  let res;
  try { res = await fetch(path, init); }
  catch (e) { throw new ApiError(0, "network_error", `Cannot reach the server (${e.message})`); }
  if (res.status === 204) return null;
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (!res.ok) {
    const err = data && data.error ? data.error : { code: "http_" + res.status, message: text || res.statusText };
    throw new ApiError(res.status, err.code, err.message);
  }
  return opts.withStatus ? { status: res.status, data } : data;
}

export const api = {
  status: () => request("GET", "/api/status"),
  settings: () => request("GET", "/api/settings"),
  saveSettings: (patch) => request("PUT", "/api/settings", patch),
  submit: (body) => request("POST", "/api/jobs", body),
  jobs: (q = {}) => {
    const qs = Object.entries(q).filter(([, v]) => v !== undefined && v !== null && v !== "")
      .map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
    return request("GET", "/api/jobs" + (qs ? "?" + qs : ""));
  },
  job: (id) => request("GET", `/api/jobs/${id}`),
  remove: (id) => request("DELETE", `/api/jobs/${id}`),
  cancel: (id) => request("POST", `/api/jobs/${id}/cancel`, undefined, { withStatus: true }),
  upload: (file) => { const fd = new FormData(); fd.append("file", file, file.name); return request("POST", "/api/upload", fd); },
  listUploads: (unused = false) => request("GET", "/api/uploads" + (unused ? "?unused=true" : "")),
  deleteUpload: (id) => request("DELETE", `/api/uploads/${id}`),
  pruneUploads: (opts = {}) => request("POST", "/api/uploads/prune", { unused: true, older_than_days: null, ...opts }),
  text: async (path) => {
    const res = await fetch(path);
    if (!res.ok) {
      let code = { 404: "not_found", 503: "engine_unavailable", 409: "conflict", 400: "validation_error" }[res.status] || "internal_error", message = `${path}: ${res.status}`;
      try { const body = await res.json(); if (body && body.error) ({ code, message } = body.error); } catch { /* not JSON */ }
      throw new ApiError(res.status, code, message);
    }
    return res.text();
  },
};

export const songUrl = (id, name) => `/api/songs/${id}/${name}`;

/** Subscribe to a job's SSE stream. Returns a close() function. */
export function subscribe(jobId, { onProgress, onDone, onError }) {
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  es.addEventListener("progress", (e) => { try { onProgress && onProgress(JSON.parse(e.data)); } catch (err) { console.warn("bad progress frame", err); } });
  es.addEventListener("done", (e) => {
    es.close();
    try { onDone && onDone(JSON.parse(e.data).job); } catch (err) { console.warn("bad done frame", err); }
  });
  es.onerror = () => { if (es.readyState === EventSource.CLOSED) onError && onError(); };
  return () => es.close();
}
