import { api } from "../api.js";
import { assistBox, fill, h, loraPicker, presetPicker, rememberGroup, seedField, store, toast, toastError, trackBanner } from "../ui.js";
import { liveCard, resultCard } from "./jobcard.js";

const RECENT_MAX = 20;

const GENRES = ["pop, female vocal, upbeat", "lo-fi hip hop", "orchestral cinematic", "indie rock, male vocal", "jazz trio", "edm, synth", "warm piano ballad", "city pop, groovy bass"];
const SECTIONS = ["[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Interlude]", "[Outro]"];
const DEFAULTS = { title: "", style: "", lyrics: "", cot: "full", seed: "", random_seed: true, cfg_scale: "", abc: "", count: 1, random_seeds: false, preset: "quality", precision: "8bit", ode_steps: 16, loras: [] };

export async function createView({ el, query, app }) {
  const saved = { ...DEFAULTS, ...store.get("create", {}) };
  if (app.settings && !store.get("create")) saved.preset = app.settings.default_preset || saved.preset;
  if (query.from) {
    try {
      const { job } = await api.job(query.from);
      Object.assign(saved, { title: job.params.title || "", style: job.params.style || "", lyrics: job.params.lyrics || "", cot: job.params.cot || "full",
        seed: job.seed, random_seed: false, cfg_scale: job.params.cfg_scale ?? "", abc: job.params.abc || "", preset: job.preset, precision: job.precision, ode_steps: job.ode_steps, loras: job.loras || [], count: 3 });
      toast(`Prefilled from ${job.title || job.id.slice(0, 8)} — set Variations and submit`, "info");
    } catch (e) { toastError(e); }
  }
  // ?track=<id>: the submission becomes a take of that track. Kept in this closure only — never in store("create").
  let trackId = null;
  const banner = await trackBanner(query.track, { onDismiss: () => { trackId = null; history.replaceState(null, "", "#/create"); } });
  if (banner) trackId = banner.track.id;

  const f = {
    title: h("input", { id: "f-title", type: "text", value: saved.title, placeholder: "Optional display title", autocomplete: "off" }),
    style: h("textarea", { id: "f-style", rows: 2, placeholder: "e.g. dreamy indie pop, female vocal, warm guitars, 96 BPM", required: true }, saved.style),
    lyrics: h("textarea", { id: "f-lyrics", class: "lyrics", placeholder: "[Verse]\nfirst line…\n\n[Chorus]\n…", required: true }, saved.lyrics),
    seed: seedField({ id: "f-seed", seed: saved.seed, random: saved.random_seed !== false, onChange: () => { collect(); updateCount(); } }),
    cfg: h("input", { id: "f-cfg", type: "number", min: 0, max: 20, step: 0.1, value: saved.cfg_scale, placeholder: "engine default" }),
    abc: h("textarea", { id: "f-abc", class: "mono", rows: 8, placeholder: "X:1\nT:\nM:4/4\nL:1/16\nK:C\n…" }, saved.abc),
    count: h("input", { id: "f-count", type: "number", min: 1, max: 16, step: 1, value: saved.count }),
    random: h("input", { id: "f-random", type: "checkbox", checked: !!saved.random_seeds }),
  };
  let cot = saved.cot;
  const cotSeg = h("div", { class: "seg", role: "group", "aria-label": "Mode" }, ["full", "melody", "off"].map((m) =>
    h("button", { type: "button", dataset: { m }, "aria-pressed": String(m === cot), onclick: () => { cot = m; cotSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === cot))); abcDetails.hidden = cot === "off"; } }, m)));
  const presets = presetPicker({ preset: saved.preset, precision: saved.precision, ode_steps: saved.ode_steps }, app.status && app.status.presets);
  let adapters = app.status && app.status.loras ? app.status.loras.adapters : [];
  const loras = loraPicker(saved.loras, adapters);
  // Instrumental AR adapters (name contains "inst") overrun to the length cap with timed section tags.
  const instHint = h("span", { class: "hint", hidden: true }, "Instrumental adapter: use bare section tags ([intro], [verse]…) or [instrumental] in the lyrics — timed tags tend to overrun to the length cap; songs land around 3–5 min regardless of the plan.");
  const isInst = (name) => { const a = adapters.find((x) => x.name === name); return /inst/i.test(name) && (!a || !a.parts || a.parts.includes("ar")); };
  const updateInstHint = () => { instHint.hidden = !loras.value().some((l) => isInst(l.name)); };
  loras.addEventListener("input", updateInstHint); updateInstHint();
  const onStatus = (s) => { if (s && s.loras) { adapters = s.loras.adapters; loras.update(adapters); updateInstHint(); } };
  app.listeners.add(onStatus);
  // Claude assist: fills title/style/lyrics/mode/CFG; applyFields returns the previous values so the box can undo.
  function applyFields(fields) {
    const prev = {};
    for (const k of ["title", "style", "lyrics"]) if (k in fields) { prev[k] = f[k].value; f[k].value = fields[k] ?? ""; }
    if ("cfg_scale" in fields) { prev.cfg_scale = f.cfg.value === "" ? null : Number(f.cfg.value); f.cfg.value = fields.cfg_scale === null || fields.cfg_scale === undefined ? "" : fields.cfg_scale; }
    if ("cot" in fields) { prev.cot = cot; const b = cotSeg.querySelector(`[data-m="${fields.cot}"]`); if (b) b.click(); }
    collect();
    return prev;
  }
  const getContext = () => { const c = { title: f.title.value.trim(), style: f.style.value.trim(), lyrics: f.lyrics.value.trim(), cot, cfg_scale: f.cfg.value === "" ? null : Number(f.cfg.value) }; for (const k in c) if (c[k] === "" || c[k] === null) delete c[k]; return c; };
  const assist = assistBox({ page: "create", app, getContext, apply: applyFields });
  app.listeners.add(assist.onStatus);
  const abcDetails = h("details", { hidden: cot === "off", open: !!saved.abc }, h("summary", {}, "Supply an ABC score (optional)"),
    h("p", { class: "hint" }, "Paste an ABC score (or load a .abc/.txt file) to skip planning; the engine follows it. Not allowed with mode “off”."),
    h("label", { class: "row small" }, "Load file", h("input", { id: "f-abc-file", type: "file", accept: ".abc,.txt,text/plain", style: "width:auto", onchange: async (e) => {
      const file = e.target.files[0]; if (!file) return;
      try { f.abc.value = await file.text(); collect(); toast(`Loaded ${file.name}`, "ok"); } catch (err) { toast(`Could not read ${file.name}: ${err.message}`, "err"); }
      e.target.value = "";
    } })), f.abc);
  const submit = h("button", { type: "submit", class: "primary", id: "f-submit" }, "Create song");
  const variationsHint = h("span", { class: "hint" });
  const updateCount = () => { const n = Number(f.count.value) || 1; submit.textContent = n > 1 ? `Create ${n} variations` : "Create song"; variationsHint.textContent = n > 1 ? "Submitted as one group; seeds " + (f.random.checked || f.seed.isRandom() ? "independent random" : "seed, seed+1, …") : ""; };
  f.count.addEventListener("input", updateCount); f.random.addEventListener("change", updateCount); updateCount();

  const insertTag = (tag) => { const t = f.lyrics, s = t.selectionStart, v = t.value; const pre = v.slice(0, s), post = v.slice(t.selectionEnd); const nl = !pre ? "" : pre.endsWith("\n\n") ? "" : pre.endsWith("\n") ? "\n" : "\n\n"; t.value = pre + nl + tag + "\n" + post; t.focus(); t.selectionStart = t.selectionEnd = (pre + nl + tag + "\n").length; };

  const form = h("form", { class: "stack", onsubmit: onSubmit },
    h("div", { class: "stack" }, assist.el,
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Title"), f.title),
      h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Style"), h("span", {}, "required")), f.style),
      h("div", { class: "chips", "aria-label": "Genre suggestions" }, GENRES.map((g) => h("button", { type: "button", class: "chip", onclick: () => { f.style.value = f.style.value.trim() ? f.style.value.replace(/,?\s*$/, ", ") + g : g; f.style.focus(); } }, g))),
      h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Lyrics"), h("span", {}, "section tags below")), f.lyrics),
      h("div", { class: "chips", "aria-label": "Insert section tag" }, SECTIONS.map((t) => h("button", { type: "button", class: "chip", onclick: () => insertTag(t) }, t))),
      abcDetails),
    h("div", { class: "panel stack" },
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Mode (chain of thought)"), cotSeg,
        h("span", { class: "hint" }, "full = plan score + arrangement, melody = plan melody only, off = no score")),
      h("div", { class: "field" }, h("span", { class: "lbl" }, "Preset"), presets),
      h("div", { class: "field" }, h("span", { class: "lbl" }, "LoRA adapters"), loras,
        h("span", { class: "hint" }, "Merged into the AR / acoustic weights for this job; stack an AR adapter with a NAR one. Scale 1 = as trained."), instHint),
      h("div", { class: "grid2" },
        h("div", { class: "field" }, h("span", { class: "lbl" }, "Seed"), f.seed),
        h("label", { class: "field" }, h("span", { class: "lbl" }, "CFG scale"), f.cfg)),
      h("div", { class: "grid2" },
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Variations"), f.count),
        h("label", { class: "check", style: "align-self:end;padding-bottom:6px" }, f.random, "random seeds")),
      variationsHint,
      submit));
  // Results: jobs submitted from this page (persisted ids), newest first; live cards while running.
  const results = h("div", { class: "stack" });
  const resultsEmpty = h("div", { class: "empty" }, "Songs you create here appear in this column — play them as soon as they finish, then tweak and submit another take.");
  const resultsHead = h("div", { class: "row between" }, h("h3", {}, "Results"), h("button", { type: "button", class: "ghost sm", onclick: clearResults }, "Clear list"));
  const live = new Map(); // id -> liveCard
  fill(el, h("div", { class: "view-head" }, h("h1", {}, "Create"), h("span", { class: "sub" }, "Describe the style, write lyrics with section tags, pick a preset — results play right here.")),
    banner ? banner.el : null,
    h("div", { class: "cols results-layout" }, form, h("div", { class: "stack results-col" }, resultsHead, results, resultsEmpty)));

  const recent = () => store.get("recent", []);
  const setRecent = (ids) => store.set("recent", ids.slice(0, RECENT_MAX));
  function clearResults() { live.forEach((c) => c.close()); live.clear(); setRecent([]); fill(results); resultsEmpty.hidden = false; resultsHead.querySelector("button").hidden = true; }
  function dismiss(job) { setRecent(recent().filter((id) => id !== job.id)); const el = results.querySelector(`[data-id="${job.id}"]`); el && el.remove(); const c = live.get(job.id); if (c) { c.close(); live.delete(job.id); } syncEmpty(); }
  function syncEmpty() { const n = results.children.length; resultsEmpty.hidden = n > 0; resultsHead.querySelector("button").hidden = n === 0; }
  function useSeed(job) { f.seed.setSeed(job.seed); collect(); f.seed.input.focus(); toast(`Seed ${job.seed} copied into the form (random seed off)`, "info", { timeout: 2500 }); }
  const doneCard = (job) => resultCard(job, { onUseSeed: useSeed, onDismiss: dismiss });
  /** Show a job in the results column (prepend unless `replace` gives an existing node). */
  function show(job, replace = null) {
    const prev = live.get(job.id); if (prev) { prev.close(); live.delete(job.id); }
    let node;
    if (["done", "failed", "cancelled"].includes(job.status)) node = doneCard(job);
    else {
      const c = liveCard(job, { scoreCollapsed: true, onFinish: (c2, j) => show(j, c2.el), onGone: async (c2) => { try { show((await api.job(c2.job.id)).job, c2.el); } catch { dismiss(c2.job); } } });
      live.set(job.id, c); node = c.el;
    }
    if (replace && replace.parentNode === results) replace.replaceWith(node); else results.prepend(node);
    syncEmpty();
  }
  async function restore() {
    const ids = recent(); if (!ids.length) { syncEmpty(); return; }
    const found = [];
    for (const id of ids.slice().reverse()) { // oldest first so prepend leaves newest on top
      try { const { job } = await api.job(id); found.unshift(id); show(job); }
      catch (e) { if (e.status !== 404) console.warn("recent job", id, e.message); }
    }
    setRecent(found);
  }
  await restore();
  const tick = setInterval(() => live.forEach((c) => c.paint()), 1000);

  function collect() {
    const v = { title: f.title.value.trim(), style: f.style.value.trim(), lyrics: f.lyrics.value, cot, seed: f.seed.raw(), random_seed: f.seed.isRandom(),
      cfg_scale: f.cfg.value === "" ? "" : Number(f.cfg.value), abc: f.abc.value, count: Math.max(1, Number(f.count.value) || 1), random_seeds: f.random.checked, loras: loras.value(), ...presets.value() };
    store.set("create", v);
    return v;
  }
  form.addEventListener("input", collect);

  async function onSubmit(e) {
    e.preventDefault();
    const v = collect();
    if (!v.style || !v.lyrics.trim()) return toast("Style and lyrics are required", "err");
    if (v.abc.trim() && v.cot === "off") return toast("An ABC score cannot be used with mode “off” — pick full or melody", "err");
    const base = { style: v.style, lyrics: v.lyrics, cot: v.cot, seed: f.seed.value(), cfg_scale: v.cfg_scale === "" ? null : v.cfg_scale, abc: v.abc.trim() || null, title: v.title || null };
    const common = { preset: v.preset, precision: v.precision, ode_steps: v.ode_steps, loras: v.loras, track_id: trackId };
    const link = banner && trackId ? { href: `#/project/${banner.track.project_id}`, label: "Open project" } : undefined;
    submit.disabled = true;
    try {
      let jobs;
      if (v.count > 1) {
        const r = await api.submit({ kind: "variations", params: { count: v.count, base, random_seeds: v.random_seeds || v.random_seed, label: null }, ...common });
        rememberGroup(r.group, r.jobs); jobs = r.jobs;
        toast(`Queued ${r.jobs.length} variations${trackId ? ` as takes of “${banner.track.name}”` : ""}`, "ok", { link });
      } else {
        const r = await api.submit({ kind: "create", params: base, ...common });
        jobs = [r.job];
        toast(`Queued “${r.job.title || r.job.id.slice(0, 8)}”${trackId ? ` as a take of “${banner.track.name}”` : ""}`, "ok", { link });
      }
      setRecent([...jobs.map((j) => j.id).reverse(), ...recent().filter((id) => !jobs.some((j) => j.id === id))]);
      for (const job of jobs) show(job);
      results.parentNode.scrollTop = 0;
    } catch (err) { toastError(err); }
    submit.disabled = false; // queue another take right away
  }
  return { unmount() { clearInterval(tick); app.listeners.delete(onStatus); app.listeners.delete(assist.onStatus); live.forEach((c) => c.close()); } };
}
