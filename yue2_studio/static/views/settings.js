import { api } from "../api.js";
import { applyTheme, fill, fmt, h, loraLabel, store, toast, toastError } from "../ui.js";

export async function settingsView({ el, app }) {
  let s;
  try { s = await api.settings(); app.settings = s; }
  catch (e) { toastError(e); s = { default_preset: "quality", memory_budget_gib: 24, require_ac: false, theme: store.get("theme", "system"), prune_uploads_days: null }; }
  const f = {
    preset: h("select", { id: "st-preset" }, ["quality", "fast", "custom"].map((p) => h("option", { value: p, selected: p === s.default_preset }, p))),
    mem: h("input", { id: "st-mem", type: "number", min: 6, max: 44, step: 1, value: s.memory_budget_gib }),
    ac: h("input", { id: "st-ac", type: "checkbox", checked: !!s.require_ac }),
    prune: h("input", { id: "st-prune", type: "number", min: 1, max: 365, step: 1, value: s.prune_uploads_days === null || s.prune_uploads_days === undefined ? "" : s.prune_uploads_days, placeholder: "never", style: "width:120px" }),
    theme: h("select", { id: "st-theme", onchange: (e) => applyTheme(e.target.value) }, [["system", "Follow system"], ["light", "Light"], ["dark", "Dark"]].map(([v, l]) => h("option", { value: v, selected: v === (s.theme || "system") }, l))),
  };
  const save = h("button", { type: "submit", class: "primary" }, "Save settings");
  const form = h("form", { class: "panel stack", onsubmit: async (e) => {
    e.preventDefault();
    const mem = Number(f.mem.value);
    if (!(mem >= 4 && mem <= 44)) return toast("Memory budget must be between 4 and 44 GiB", "err");
    const prune = f.prune.value.trim() === "" ? null : Number(f.prune.value);
    if (prune !== null && !(Number.isInteger(prune) && prune >= 1 && prune <= 365)) return toast("Auto-delete uploads must be 1–365 days (or blank for never)", "err");
    save.disabled = true;
    try { s = await api.saveSettings({ default_preset: f.preset.value, memory_budget_gib: mem, require_ac: f.ac.checked, theme: f.theme.value, prune_uploads_days: prune }); app.settings = s; applyTheme(s.theme); toast("Settings saved", "ok"); }
    catch (err) { toastError(err); }
    save.disabled = false;
  } },
    h("h3", {}, "Generation"),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Default preset"), f.preset),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Memory budget (GiB, 6–44)"), f.mem, h("span", { class: "hint" }, "MLX watchdog limit; a change rebuilds the pipeline on the next job. Peak use is ≈11 GiB.")),
    h("label", { class: "check" }, f.ac, "Require AC power before running jobs"),
    h("h3", {}, "Storage"),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Auto-delete unused uploads after N days (blank = never)"), f.prune,
      h("span", { class: "hint" }, "Uploads no job references are removed from data/uploads at startup and after each job; uploads a queued or running job needs are never touched. Manage them under Library → Uploads.")),
    h("h3", {}, "Appearance"),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Theme"), f.theme),
    h("div", {}, save));

  const statusBox = h("dl", { class: "kv" });
  const modelsBox = h("dl", { class: "kv" });
  function paint(st) {
    if (!st) { fill(statusBox, h("dt", {}, "Server"), h("dd", { class: "muted" }, "offline")); return; }
    const e = st.engine, q = st.queue;
    fill(statusBox, 
      h("dt", {}, "Engine"), h("dd", {}, h("span", { class: `tag ${e.state === "ready" ? "ok" : e.state === "busy" ? "accent" : ""}` }, e.state)),
      h("dt", {}, "Precision"), h("dd", {}, e.precision || "—"), h("dt", {}, "Memory"), h("dd", {}, fmt.gib(e.memory_gib)),
      h("dt", {}, "LoRA"), h("dd", {}, e.loras && e.loras.length ? loraLabel(e.loras) : "—"),
      h("dt", {}, "Current job"), h("dd", {}, e.current_job_id ? h("a", { href: "#/queue", class: "mono" }, e.current_job_id.slice(0, 8)) : "—"),
      h("dt", {}, "Queued"), h("dd", {}, q.queued), h("dt", {}, "Version"), h("dd", {}, st.version));
    fill(modelsBox, 
      h("dt", {}, "Weights"), h("dd", {}, h("span", { class: `tag ${st.models.present ? "ok" : "err"}` }, st.models.present ? "present" : "missing")),
      h("dt", {}, "Converted"), h("dd", { class: "mono small" }, st.models.converted_dir), h("dt", {}, "VAE"), h("dd", { class: "mono small" }, st.models.vae_dir),
      h("dt", {}, "ffmpeg"), h("dd", {}, st.ffmpeg ? "found" : "missing"),
      h("dt", {}, "Cover"), h("dd", {}, st.cover.available ? "available" : ["unavailable", st.cover.reasons.length ? h("ul", { style: "margin:2px 0 0 16px" }, st.cover.reasons.map((r) => h("li", { class: "small" }, r))) : null]),
      h("dt", {}, "Presets"), h("dd", {}, h("ul", { style: "margin:0;padding-left:16px" }, st.presets.map((p) => h("li", { class: "small" }, h("b", {}, p.name), ` — ${p.description}`)))),
      h("dt", {}, "LoRA"), h("dd", {}, h("div", { class: "mono small" }, st.loras.dir),
        st.loras.adapters.length ? h("ul", { style: "margin:4px 0 0 16px" }, st.loras.adapters.map((a) => h("li", { class: "small", title: a.valid ? "" : a.error },
          h("b", {}, a.name), a.valid ? ` — ${a.parts.map((p) => p.toUpperCase()).join("+")}${a.rank ? `, rank ${a.rank}` : ""}${a.replaced.length ? `, replaces ${a.replaced.join("/")}` : ""}${a.metadata && a.metadata.intended_cot ? `, cot ${a.metadata.intended_cot}` : ""}` : [" ", h("span", { class: "tag err" }, "unusable"), ` ${a.error}`])))
          : h("div", { class: "hint" }, "No adapters — drop .safetensors files (or PEFT folders) into this directory; rescanned every 5 s.")));
  }
  paint(app.status);
  app.listeners.add(paint);
  fill(el, h("div", { class: "view-head" }, h("h1", {}, "Settings")),
    h("div", { class: "cols even" }, form, h("div", { class: "stack" },
      h("div", { class: "panel stack" }, h("h3", {}, "Engine status"), statusBox, h("p", { class: "hint" }, "Refreshes every 5 s.")),
      h("div", { class: "panel stack" }, h("h3", {}, "Models"), modelsBox))));
  return { unmount: () => app.listeners.delete(paint) };
}
