import { api } from "../api.js";
import { fill, h, presetPicker, randomSeed, store, toast, toastError } from "../ui.js";

const GENRES = ["pop, female vocal, upbeat", "lo-fi hip hop", "orchestral cinematic", "indie rock, male vocal", "jazz trio", "edm, synth", "warm piano ballad", "city pop, groovy bass"];
const SECTIONS = ["[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Bridge]", "[Interlude]", "[Outro]"];
const DEFAULTS = { title: "", style: "", lyrics: "", cot: "full", seed: "", cfg_scale: "", abc: "", count: 1, random_seeds: false, preset: "quality", precision: "8bit", ode_steps: 16 };

export async function createView({ el, query, app }) {
  const saved = { ...DEFAULTS, ...store.get("create", {}) };
  if (app.settings && !store.get("create")) saved.preset = app.settings.default_preset || saved.preset;
  if (query.from) {
    try {
      const { job } = await api.job(query.from);
      Object.assign(saved, { title: job.params.title || "", style: job.params.style || "", lyrics: job.params.lyrics || "", cot: job.params.cot || "full",
        seed: job.seed, cfg_scale: job.params.cfg_scale ?? "", abc: job.params.abc || "", preset: job.preset, precision: job.precision, ode_steps: job.ode_steps, count: 3 });
      toast(`Prefilled from ${job.title || job.id.slice(0, 8)} — set Variations and submit`, "info");
    } catch (e) { toastError(e); }
  }

  const f = {
    title: h("input", { id: "f-title", type: "text", value: saved.title, placeholder: "Optional display title", autocomplete: "off" }),
    style: h("textarea", { id: "f-style", rows: 2, placeholder: "e.g. dreamy indie pop, female vocal, warm guitars, 96 BPM", required: true }, saved.style),
    lyrics: h("textarea", { id: "f-lyrics", class: "lyrics", placeholder: "[Verse]\nfirst line…\n\n[Chorus]\n…", required: true }, saved.lyrics),
    seed: h("input", { id: "f-seed", type: "number", min: 0, max: 2147483647, step: 1, value: saved.seed, placeholder: "random" }),
    cfg: h("input", { id: "f-cfg", type: "number", min: 0, max: 20, step: 0.1, value: saved.cfg_scale, placeholder: "engine default" }),
    abc: h("textarea", { id: "f-abc", class: "mono", rows: 8, placeholder: "X:1\nT:\nM:4/4\nL:1/16\nK:C\n…" }, saved.abc),
    count: h("input", { id: "f-count", type: "number", min: 1, max: 16, step: 1, value: saved.count }),
    random: h("input", { id: "f-random", type: "checkbox", checked: !!saved.random_seeds }),
  };
  let cot = saved.cot;
  const cotSeg = h("div", { class: "seg", role: "group", "aria-label": "Mode" }, ["full", "melody", "off"].map((m) =>
    h("button", { type: "button", dataset: { m }, "aria-pressed": String(m === cot), onclick: () => { cot = m; cotSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.m === cot))); abcDetails.hidden = cot === "off"; } }, m)));
  const presets = presetPicker({ preset: saved.preset, precision: saved.precision, ode_steps: saved.ode_steps }, app.status && app.status.presets);
  const abcDetails = h("details", { hidden: cot === "off", open: !!saved.abc }, h("summary", {}, "Supply an ABC score (optional)"),
    h("p", { class: "hint" }, "Paste an ABC score to skip planning; the engine follows it. Not allowed with mode “off”."), f.abc);
  const submit = h("button", { type: "submit", class: "primary", id: "f-submit" }, "Create song");
  const variationsHint = h("span", { class: "hint" });
  const updateCount = () => { const n = Number(f.count.value) || 1; submit.textContent = n > 1 ? `Create ${n} variations` : "Create song"; variationsHint.textContent = n > 1 ? "Submitted as one group; seeds " + (f.random.checked ? "random" : "seed, seed+1, …") : ""; };
  f.count.addEventListener("input", updateCount); f.random.addEventListener("change", updateCount); updateCount();

  const insertTag = (tag) => { const t = f.lyrics, s = t.selectionStart, v = t.value; const pre = v.slice(0, s), post = v.slice(t.selectionEnd); const nl = !pre ? "" : pre.endsWith("\n\n") ? "" : pre.endsWith("\n") ? "\n" : "\n\n"; t.value = pre + nl + tag + "\n" + post; t.focus(); t.selectionStart = t.selectionEnd = (pre + nl + tag + "\n").length; };

  const form = h("form", { class: "cols", onsubmit: onSubmit },
    h("div", { class: "stack" },
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Title"), f.title),
      h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Style"), h("span", {}, "required")), f.style),
      h("div", { class: "chips", "aria-label": "Genre suggestions" }, GENRES.map((g) => h("button", { type: "button", class: "chip", onclick: () => { f.style.value = f.style.value.trim() ? f.style.value.replace(/,?\s*$/, ", ") + g : g; f.style.focus(); } }, g))),
      h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Lyrics"), h("span", {}, "section tags below")), f.lyrics),
      h("div", { class: "chips", "aria-label": "Insert section tag" }, SECTIONS.map((t) => h("button", { type: "button", class: "chip", onclick: () => insertTag(t) }, t))),
      abcDetails),
    h("div", { class: "panel sticky stack" },
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Mode (chain of thought)"), cotSeg,
        h("span", { class: "hint" }, "full = plan score + arrangement, melody = plan melody only, off = no score")),
      h("div", { class: "field" }, h("span", { class: "lbl" }, "Preset"), presets),
      h("div", { class: "grid2" },
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Seed"), h("div", { class: "row nowrap" }, f.seed, h("button", { type: "button", class: "icon", title: "Random seed", "aria-label": "Random seed", onclick: () => { f.seed.value = randomSeed(); } }, "🎲"))),
        h("label", { class: "field" }, h("span", { class: "lbl" }, "CFG scale"), f.cfg)),
      h("div", { class: "grid2" },
        h("label", { class: "field" }, h("span", { class: "lbl" }, "Variations"), f.count),
        h("label", { class: "check", style: "align-self:end;padding-bottom:6px" }, f.random, "random seeds")),
      variationsHint,
      submit));
  fill(el, h("div", { class: "view-head" }, h("h1", {}, "Create"), h("span", { class: "sub" }, "Describe the style, write lyrics with section tags, pick a preset.")), form);

  function collect() {
    const v = { title: f.title.value.trim(), style: f.style.value.trim(), lyrics: f.lyrics.value, cot, seed: f.seed.value === "" ? "" : Number(f.seed.value),
      cfg_scale: f.cfg.value === "" ? "" : Number(f.cfg.value), abc: f.abc.value, count: Math.max(1, Number(f.count.value) || 1), random_seeds: f.random.checked, ...presets.value() };
    store.set("create", v);
    return v;
  }
  form.addEventListener("input", collect);

  async function onSubmit(e) {
    e.preventDefault();
    const v = collect();
    if (!v.style || !v.lyrics.trim()) return toast("Style and lyrics are required", "err");
    if (v.abc.trim() && v.cot === "off") return toast("An ABC score cannot be used with mode “off” — pick full or melody", "err");
    const base = { style: v.style, lyrics: v.lyrics, cot: v.cot, seed: v.seed === "" ? null : v.seed, cfg_scale: v.cfg_scale === "" ? null : v.cfg_scale, abc: v.abc.trim() || null, title: v.title || null };
    const common = { preset: v.preset, precision: v.precision, ode_steps: v.ode_steps };
    submit.disabled = true;
    try {
      if (v.count > 1) {
        const r = await api.submit({ kind: "variations", params: { count: v.count, base, random_seeds: v.random_seeds, label: null }, ...common });
        const groups = store.get("groups", {}); groups[r.group.id] = r.group.label; store.set("groups", groups);
        toast(`Queued ${r.jobs.length} variations`, "ok");
      } else {
        const r = await api.submit({ kind: "create", params: base, ...common });
        toast(`Queued “${r.job.title || r.job.id.slice(0, 8)}”`, "ok");
      }
      location.hash = "#/queue";
    } catch (err) { toastError(err); }
    submit.disabled = false;
  }
  return {};
}
