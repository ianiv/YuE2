// Small DOM/format helpers shared by all views.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k in el && k !== "list" && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
/** Replace el's children (arrays/nulls handled like h()). */
export function fill(el, ...children) { clear(el); for (const c of children.flat(Infinity)) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : document.createTextNode(String(c))); return el; }

export function toast(message, kind = "info", { link, timeout = 6000 } = {}) {
  const box = document.getElementById("toasts");
  const t = h("div", { class: `toast ${kind}` },
    h("span", {}, message, link ? [" ", h("a", { href: link.href }, link.label)] : null),
    h("button", { class: "ghost sm", "aria-label": "Dismiss", onclick: () => t.remove() }, "✕"));
  box.append(t);
  while (box.children.length > 4) box.firstChild.remove();
  if (timeout) setTimeout(() => t.remove(), timeout);
  return t;
}
export const toastError = (e) => toast(e && e.message ? `${e.message}${e.code ? ` (${e.code})` : ""}` : String(e), "err", { timeout: 9000 });

export function applyTheme(theme, { persist = true } = {}) {
  const root = document.documentElement;
  if (theme === "light" || theme === "dark") root.dataset.theme = theme; else delete root.dataset.theme;
  if (persist) store.set("theme", theme || "system");
}

export const store = {
  get(k, fallback) { try { const v = localStorage.getItem("yue2." + k); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(k, v) { try { localStorage.setItem("yue2." + k, JSON.stringify(v)); } catch { /* ignore */ } },
};

export const fmt = {
  secs(s) { if (s === null || s === undefined || Number.isNaN(s)) return "—"; s = Math.round(s); return s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`; },
  dur(s) { if (s === null || s === undefined) return "—"; const m = Math.floor(s / 60), r = Math.round(s % 60); return `${m}:${String(r).padStart(2, "0")}`; },
  when(iso) { if (!iso) return "—"; const d = new Date(iso), diff = (Date.now() - d) / 1000; if (diff < 60) return "just now"; if (diff < 3600) return `${Math.floor(diff / 60)} min ago`; if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`; return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); },
  gib(x) { return x === null || x === undefined ? "—" : `${Number(x).toFixed(1)} GiB`; },
  tps(x) { return x ? `${x.toFixed(0)} tok/s` : "—"; },
  excerpt(s, n = 80) { s = (s || "").replace(/\s+/g, " ").trim(); return s.length > n ? s.slice(0, n - 1) + "…" : s; },
};

export const STAGE_ORDER = ["load", "transcribe", "plan", "semantic", "synthesize", "decode", "save"];
export const STAGE_NAMES = { load: "Load", transcribe: "Transcribe", plan: "Plan", semantic: "Semantic", synthesize: "Synthesize", decode: "Decode", save: "Save", e2e: "Total" };
// Typical share of wall time per stage (from PLAN timings: plan 16 s, semantic 36 s, synth 27 s, decode 3 s).
const STAGE_WEIGHT = { load: 0.04, transcribe: 0.06, plan: 0.18, semantic: 0.42, synthesize: 0.26, decode: 0.03, save: 0.01 };

/** Overall fraction + ETA from the latest stage event and the job's elapsed seconds. */
export function estimate(ev, elapsed, kind) {
  const stages = STAGE_ORDER.filter((s) => s !== "transcribe" || kind === "cover");
  const weights = stages.map((s) => STAGE_WEIGHT[s]); const sum = weights.reduce((a, b) => a + b, 0);
  if (!ev || !ev.stage) return { fraction: 0, eta: null };
  const idx = stages.indexOf(ev.stage);
  if (idx < 0) return { fraction: 0, eta: null };
  const done = stages.slice(0, Math.max(idx, 0)).reduce((a, s) => a + STAGE_WEIGHT[s], 0);
  const inStage = ev.total ? Math.min(1, (ev.completed || 0) / ev.total) : (ev.status === "complete" ? 1 : 0.3);
  const fraction = Math.min(0.99, (done + STAGE_WEIGHT[ev.stage] * inStage) / sum);
  let eta = null;
  if (ev.tps && ev.total && ev.unit === "tokens") {
    const restStage = (ev.total - (ev.completed || 0)) / ev.tps;
    const restWeight = stages.slice(idx + 1).reduce((a, s) => a + STAGE_WEIGHT[s], 0);
    const perWeight = elapsed > 2 && fraction > 0.05 ? elapsed / fraction / sum : (ev.seconds || 1) / Math.max(inStage, 0.05) / STAGE_WEIGHT[ev.stage];
    eta = restStage + restWeight * perWeight;
  } else if (elapsed > 3 && fraction > 0.05) eta = elapsed * (1 - fraction) / fraction;
  return { fraction, eta };
}

export const jobTitle = (job) => job.title || (job.params && (job.params.title || fmt.excerpt(job.params.style, 48))) || job.id.slice(0, 8);

/** abcjs score rendering, throttled to ≤2 Hz per element. */
export function renderScore(el, abc, opts = {}, { immediate = false } = {}) {
  if (!el._score) el._score = { pending: null, last: 0, timer: null };
  const s = el._score;
  const draw = () => {
    s.last = Date.now(); s.timer = null;
    if (!window.ABCJS) { el.textContent = "abcjs not loaded"; return; }
    try { el._visual = ABCJS.renderAbc(el, abc || "X:1\nK:C\n", { responsive: "resize", paddingtop: 4, paddingbottom: 4, add_classes: true, ...opts })[0]; }
    catch (e) { el.textContent = "Could not render score: " + e.message; }
  };
  clearTimeout(s.timer);
  const wait = immediate ? 0 : 500 - (Date.now() - s.last);
  if (wait <= 0) draw(); else s.timer = setTimeout(draw, wait);
}

/** Minimal MIDI preview via abcjs' synth. Returns a toggle button. */
export function scorePlayer(getVisual) {
  let synth = null, playing = false;
  const btn = h("button", { onclick: toggle }, "▶ Play score (MIDI)");
  async function toggle() {
    if (playing) { synth && synth.stop(); playing = false; btn.textContent = "▶ Play score (MIDI)"; return; }
    const visual = getVisual();
    if (!visual || !window.ABCJS || !ABCJS.synth || !ABCJS.synth.supportsAudio()) return toast("Audio synthesis is not supported in this browser", "err");
    btn.disabled = true; btn.textContent = "Loading sounds…";
    try {
      synth = new ABCJS.synth.CreateSynth();
      await synth.init({ visualObj: visual, millisecondsPerMeasure: visual.millisecondsPerMeasure() });
      await synth.prime();
      synth.start(); playing = true; btn.textContent = "■ Stop";
      setTimeout(() => { if (playing) { playing = false; btn.textContent = "▶ Play score (MIDI)"; } }, (synth.duration || 30) * 1000 + 300);
    } catch (e) { toast("Could not play score: " + e.message, "err"); }
    btn.disabled = false;
  }
  btn.stop = () => { if (playing) toggle(); };
  return btn;
}

/** Preset picker with custom precision/steps. value(): {preset, precision, ode_steps}. */
export function presetPicker(initial = {}, presets = null) {
  const list = presets || [{ name: "quality", label: "Quality", description: "BF16 AR, 32 ODE steps (reference quality)" }, { name: "fast", label: "Fast", description: "8-bit AR, 8 ODE steps (faster than realtime)" }, { name: "custom", label: "Custom", description: "Pick precision (bf16/8bit/4bit) and 4-64 ODE steps" }];
  const state = { preset: initial.preset || "quality", precision: initial.precision || "8bit", ode_steps: initial.ode_steps || 16 };
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Preset" });
  const prec = h("select", { id: "f-precision", "aria-label": "Precision", onchange: (e) => { state.precision = e.target.value; } },
    ["bf16", "8bit", "4bit"].map((p) => h("option", { value: p, selected: p === state.precision }, p)));
  const steps = h("input", { id: "f-ode", type: "number", min: 4, max: 64, step: 1, value: state.ode_steps, "aria-label": "ODE steps", onchange: (e) => { state.ode_steps = Math.max(4, Math.min(64, Number(e.target.value) || 16)); e.target.value = state.ode_steps; } });
  const custom = h("div", { class: "grid2", hidden: state.preset !== "custom" },
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Precision"), prec),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "ODE steps 4–64"), steps));
  const desc = h("div", { class: "hint" });
  const sync = () => {
    seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.p === state.preset)));
    custom.hidden = state.preset !== "custom";
    const p = list.find((x) => x.name === state.preset); desc.textContent = p ? p.description : "";
  };
  for (const p of list) seg.append(h("button", { type: "button", dataset: { p: p.name }, onclick: () => { state.preset = p.name; sync(); } }, p.label || p.name));
  sync();
  const el = h("div", { class: "stack" }, seg, desc, custom);
  el.value = () => state.preset === "custom" ? { ...state } : { preset: state.preset, precision: null, ode_steps: null };
  el.set = (v) => { Object.assign(state, v); prec.value = state.precision; steps.value = state.ode_steps; sync(); };
  return el;
}

/** Remember a variations group from the POST response: label + jobs in submit order. */
export function rememberGroup(group, jobs) {
  const groups = store.get("groups", {});
  groups[group.id] = { label: group.label, ids: (jobs || []).map((j) => j.id) };
  store.set("groups", groups);
}
export const groupLabel = (gid) => { const g = store.get("groups", {})[gid]; return g ? (typeof g === "string" ? g : g.label) : null; };

export const randomSeed = () => Math.floor(Math.random() * 2 ** 31);
export const confirmDialog = (msg) => window.confirm(msg);
