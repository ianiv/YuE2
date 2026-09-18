// Small DOM/format helpers shared by all views.
import { api } from "./api.js";

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

export const STAGE_ORDER = ["load", "transcribe", "hum", "plan", "semantic", "synthesize", "decode", "save"];
export const STAGE_NAMES = { load: "Load", transcribe: "Transcribe", hum: "Hum", plan: "Plan", semantic: "Semantic", synthesize: "Synthesize", decode: "Decode", save: "Save", e2e: "Total" };
// Typical share of wall time per stage (from PLAN timings: plan 16 s, semantic 36 s, synth 27 s, decode 3 s).
const STAGE_WEIGHT = { load: 0.04, transcribe: 0.06, hum: 0.03, plan: 0.18, semantic: 0.42, synthesize: 0.26, decode: 0.03, save: 0.01 };

/** The stages a job of this kind goes through: covers and (most) hums transcribe; hums with an adapter analyse the hum. */
export function stagesFor(kind, params = {}) {
  return STAGE_ORDER.filter((s) => {
    if (s === "transcribe") return kind === "cover" || (kind === "hum" && params.melody !== "ignore");
    if (s === "hum") return kind === "hum" && !!params.adapter;
    return true;
  });
}

/** Overall fraction + ETA from the latest stage event and the job's elapsed seconds. */
export function estimate(ev, elapsed, kind, params = {}) {
  const stages = stagesFor(kind, params);
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

/** Drop a rendered score (and any pending throttled draw) so the element holds no SVG. */
export function clearScore(el) {
  if (el._score) { clearTimeout(el._score.timer); el._score.timer = null; }
  el._visual = null;
  clear(el);
}

// One thing plays at a time. Any <audio> starting (native controls or play()) pauses every other
// <audio> and stops the MIDI preview; the MIDI preview starting pauses every <audio>. Paused media
// keeps its position so the user can resume it. Media events do not bubble, hence the capture listener.
let midiPlayer = null;
/** Pause every <audio>/<video> and stop the MIDI preview, except `except` (an element or the MIDI button). */
export function stopPlayback(except = null) {
  for (const m of document.querySelectorAll("audio, video")) if (m !== except && !m.paused) m.pause();
  if (midiPlayer && midiPlayer !== except) midiPlayer.stop();
}
document.addEventListener("play", (e) => stopPlayback(e.target), true);

/** Minimal MIDI preview via abcjs' synth. Returns a toggle button. */
export function scorePlayer(getVisual) {
  let synth = null, playing = false;
  const btn = h("button", { onclick: toggle }, "▶ Play score (MIDI)");
  function setPlaying(on) {
    playing = on; btn.textContent = on ? "■ Stop" : "▶ Play score (MIDI)";
    if (on) midiPlayer = btn; else if (midiPlayer === btn) midiPlayer = null;
  }
  async function toggle() {
    if (playing) { synth && synth.stop(); setPlaying(false); return; }
    const visual = getVisual();
    if (!visual || !window.ABCJS || !ABCJS.synth || !ABCJS.synth.supportsAudio()) return toast("Audio synthesis is not supported in this browser", "err");
    btn.disabled = true; btn.textContent = "Loading sounds…";
    try {
      synth = new ABCJS.synth.CreateSynth();
      // chordsOff: YuE2 scores carry chord symbols ("C", "Am", …) and abcjs would otherwise add its own
      // strummed piano accompaniment — more notes than the melody itself, and the same pattern for every song.
      await synth.init({ visualObj: visual, millisecondsPerMeasure: visual.millisecondsPerMeasure(), options: { chordsOff: true } });
      await synth.prime();
      stopPlayback(btn);
      synth.start(); setPlaying(true);
      const mine = synth;
      setTimeout(() => { if (playing && synth === mine) setPlaying(false); }, (synth.duration || 30) * 1000 + 300);
    } catch (e) { toast("Could not play score: " + e.message, "err"); setPlaying(false); }
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

/**
 * LoRA stack picker: rows of (adapter select, scale) with add/remove. `adapters` is the
 * `status.loras.adapters` list; unusable adapters are listed disabled with their error as title.
 * value() -> [{name, scale}] (empty when none). update(adapters) refreshes the options in place.
 */
export function loraPicker(initial = [], adapters = []) {
  let rows = (initial || []).filter((x) => x && x.name).map((x) => ({ name: x.name, scale: Number.isFinite(Number(x.scale)) ? Number(x.scale) : 1 }));
  let list = (adapters || []).filter((a) => a.kind !== "hum"); // hum adapters belong to the Hum page
  const body = h("div", { class: "stack", style: "gap:6px" });
  const hint = h("span", { class: "hint" });
  const add = h("button", { type: "button", class: "ghost sm", onclick: () => { const free = usable().find((a) => !rows.some((r) => r.name === a.name)); rows.push({ name: free ? free.name : (usable()[0] || {}).name || "", scale: 1 }); paint(); fire(); } }, "+ Add LoRA");
  const el = h("div", { class: "stack", style: "gap:6px" }, body, h("div", { class: "row" }, add, hint));
  const usable = () => list.filter((a) => a.valid);
  const describe = (a) => a ? `${a.parts.map((p) => p.toUpperCase()).join("+")}${a.rank ? ` · rank ${a.rank}` : ""}${a.metadata && a.metadata.intended_cot ? ` · cot ${a.metadata.intended_cot}` : ""}` : "";
  function fire() { el.dispatchEvent(new Event("input", { bubbles: true })); }
  function paint() {
    fill(body, rows.map((row, i) => {
      const sel = h("select", { "aria-label": "LoRA adapter", onchange: (e) => { row.name = e.target.value; note.textContent = describe(list.find((a) => a.name === row.name)); fire(); } },
        [...(row.name && !list.some((a) => a.name === row.name) ? [{ name: row.name, valid: false, error: "not found in models/loras" }] : []), ...list].map((a) =>
          h("option", { value: a.name, selected: a.name === row.name, disabled: !a.valid, title: a.valid ? describe(a) : a.error }, a.valid ? a.name : `${a.name} (unusable)`)));
      const scale = h("input", { type: "number", min: 0, max: 4, step: 0.05, value: row.scale, "aria-label": "LoRA scale", style: "width:84px", onchange: (e) => { row.scale = Math.max(0, Math.min(4, Number(e.target.value) || 0)); e.target.value = row.scale; fire(); } });
      const note = h("span", { class: "hint", style: "flex:1 1 100%" }, describe(list.find((a) => a.name === row.name)));
      return h("div", { class: "row nowrap", style: "flex-wrap:wrap" }, h("div", { style: "flex:1 1 160px;min-width:0" }, sel), h("span", { class: "hint" }, "×"), scale,
        h("button", { type: "button", class: "icon ghost", title: "Remove", "aria-label": "Remove LoRA", onclick: () => { rows.splice(i, 1); paint(); fire(); } }, "✕"), note);
    }));
    const n = usable().length;
    add.disabled = n === 0 || rows.length >= Math.min(8, n);
    hint.textContent = n === 0 ? "No adapters found — drop .safetensors files into models/loras." : rows.length ? "" : `${n} adapter${n === 1 ? "" : "s"} available; none selected.`;
  }
  paint();
  el.value = () => rows.filter((r) => r.name).map((r) => ({ name: r.name, scale: r.scale }));
  el.set = (v) => { rows = (v || []).filter((x) => x && x.name).map((x) => ({ name: x.name, scale: Number(x.scale) || 1 })); paint(); };
  el.update = (adapters) => { list = (adapters || []).filter((a) => a.kind !== "hum"); paint(); };
  return el;
}

/** What the "mode" chip shows: cot for creates, task for covers, "hum · continue" for hums. */
export const modeLabel = (job) => job.kind === "hum" ? `hum · ${(job.params.melody || "continue").replace("_", " ")}` : (job.params.cot || job.params.task || "—");

/** "inst ×0.7, realaudio" for cards and lists; empty string when no adapters. */
export const loraLabel = (loras) => (loras || []).map((l) => l.scale === 1 ? l.name : `${l.name} ×${l.scale}`).join(", ");

/** Remember a variations group from the POST response: label + jobs in submit order. */
export function rememberGroup(group, jobs) {
  const groups = store.get("groups", {});
  groups[group.id] = { label: group.label, ids: (jobs || []).map((j) => j.id) };
  store.set("groups", groups);
}
export const groupLabel = (gid) => { const g = store.get("groups", {})[gid]; return g ? (typeof g === "string" ? g : g.label) : null; };

/**
 * Seed input with a "Random" toggle. When random is on the input is disabled and value() is null
 * (server picks the seed); when off the field is editable and 🎲 rolls a value client-side.
 */
export function seedField({ id, seed = "", random = true, onChange = null, randomLabel = "Random" } = {}) {
  const input = h("input", { id, type: "number", min: 0, max: 2147483647, step: 1, value: seed === null ? "" : seed, placeholder: "random", "aria-label": "Seed" });
  const dice = h("button", { type: "button", class: "icon", title: "Roll a seed", "aria-label": "Roll a random seed", onclick: () => { input.value = randomSeed(); fire(); } }, "🎲");
  const toggle = h("input", { id: id + "-random", type: "checkbox", checked: !!random, onchange: () => { sync(); fire(); } });
  const el = h("div", { class: "stack", style: "gap:6px" },
    h("div", { class: "row nowrap" }, input, dice),
    h("label", { class: "check small", title: "Let the server pick a fresh seed for every submit" }, toggle, randomLabel));
  const sync = () => { input.disabled = toggle.checked; dice.disabled = toggle.checked; input.placeholder = toggle.checked ? "random" : "e.g. 1234"; el.classList.toggle("is-random", toggle.checked); };
  const fire = () => onChange && onChange(el.value(), toggle.checked);
  input.addEventListener("input", fire);
  el.input = input; el.toggle = toggle;
  el.isRandom = () => toggle.checked;
  el.value = () => (toggle.checked || input.value.trim() === "" ? null : Number(input.value));
  el.raw = () => (input.value.trim() === "" ? "" : Number(input.value)); // for persistence
  el.setRandom = (on) => { toggle.checked = !!on; sync(); };
  el.setSeed = (n) => { toggle.checked = false; input.value = n; sync(); };
  sync();
  return el;
}

/**
 * "Reuse a recent upload" select for the Cover and Hum pages. `onPick(upload | null)` fires when the
 * user picks an entry (or the placeholder). refresh() reloads `GET /api/uploads` (broken entries are
 * skipped) and hides the picker while there is nothing to pick; set(id) selects an id if it is still
 * listed and returns its entry (null otherwise); reset() clears the selection without firing.
 */
export function uploadPicker({ onPick, label = "Or reuse a recent upload" } = {}) {
  let list = [];
  const sel = h("select", { "aria-label": label, onchange: () => { const u = list.find((x) => x.upload_id === sel.value) || null; onPick && onPick(u); } });
  const el = h("label", { class: "field", hidden: true }, h("span", { class: "lbl" }, label), sel,
    h("span", { class: "hint" }, "Uploads stay in data/uploads until you delete them (Library → Uploads)."));
  const option = (u) => h("option", { value: u.upload_id }, `${u.filename} · ${fmt.dur(u.seconds)} · ${fmt.when(u.created_at)}`);
  function paint(keep) {
    fill(sel, h("option", { value: "" }, "— pick one —"), list.map(option));
    sel.value = keep && list.some((u) => u.upload_id === keep) ? keep : "";
    el.hidden = list.length === 0;
  }
  el.refresh = async () => {
    try { list = ((await api.listUploads()).uploads || []).filter((u) => !u.broken); } catch (e) { console.warn("uploads", e.message); list = []; }
    paint(sel.value);
    return list;
  };
  el.set = (id) => { paint(id); return list.find((u) => u.upload_id === sel.value) || null; };
  el.reset = () => paint("");
  el.value = () => list.find((u) => u.upload_id === sel.value) || null;
  el.select = sel;
  return el;
}

export const randomSeed = () => Math.floor(Math.random() * 2 ** 31);
export const confirmDialog = (msg) => window.confirm(msg);
