import { api } from "../api.js";
import { applyTheme, confirmDialog, fill, fmt, h, loraLabel, store, toast, toastError } from "../ui.js";

export async function settingsView({ el, app }) {
  let s;
  try { s = await api.settings(); app.settings = s; }
  catch (e) { toastError(e); s = { default_preset: "quality", memory_budget_gib: 24, require_ac: false, fast_numerics: true, low_memory: "auto", theme: store.get("theme", "system"), prune_uploads_days: null }; }
  // The budget range is the machine's: mlx-Yue refuses more than total RAM − 4 GiB (older servers: 6–44).
  const memMin = s.min_memory_budget_gib ?? 6, memMax = s.max_memory_budget_gib ?? 44;
  const f = {
    preset: h("select", { id: "st-preset" }, ["quality", "fast", "custom"].map((p) => h("option", { value: p, selected: p === s.default_preset }, p))),
    mem: h("input", { id: "st-mem", type: "number", min: memMin, max: memMax, step: 1, value: s.memory_budget_gib }),
    lowMem: h("select", { id: "st-lowmem" }, [["auto", autoLabel(s)], ["on", "On"], ["off", "Off"]].map(([v, l]) => h("option", { value: v, selected: v === (s.low_memory || "auto") }, l))),
    ac: h("input", { id: "st-ac", type: "checkbox", checked: !!s.require_ac }),
    fast: h("input", { id: "st-fast", type: "checkbox", checked: s.fast_numerics !== false }),
    prune: h("input", { id: "st-prune", type: "number", min: 1, max: 365, step: 1, value: s.prune_uploads_days === null || s.prune_uploads_days === undefined ? "" : s.prune_uploads_days, placeholder: "never", style: "width:120px" }),
    theme: h("select", { id: "st-theme", onchange: (e) => applyTheme(e.target.value) }, [["system", "Follow system"], ["light", "Light"], ["dark", "Dark"]].map(([v, l]) => h("option", { value: v, selected: v === (s.theme || "system") }, l))),
    assistProvider: h("select", { id: "st-assist-provider" }, [["auto", "Auto — CLI if installed, else API key"], ["cli", "Claude CLI"], ["api", "API key"], ["off", "Off — hide the Ask Claude box"]].map(([v, l]) => h("option", { value: v, selected: v === (s.assist_provider || "auto") }, l))),
    assistModel: h("input", { id: "st-assist-model", type: "text", maxlength: 80, value: s.assist_model || "", placeholder: "provider default — e.g. sonnet, claude-opus-5", autocomplete: "off" }),
    // The key never comes back from the server: the field is blank and only sent when the user typed one.
    apiKey: h("input", { id: "st-assist-key", type: "password", autocomplete: "new-password", placeholder: keyPlaceholder(s), "aria-label": "Anthropic API key" }),
  };
  function keyPlaceholder(st) { return st.has_api_key ? "•••••• key saved" : "sk-ant-…"; }
  /** "Auto" plus what it resolves to on this Mac with the saved budget (unknown on an older server). */
  function autoLabel(st) { return typeof st.low_memory_effective === "boolean" && (st.low_memory || "auto") === "auto" ? `Auto — ${st.low_memory_effective ? "on" : "off"} for this Mac` : "Auto"; }
  function memHint(st) {
    const ram = st.machine_ram_gib;
    const cap = ram ? ` This Mac has ${Math.round(ram)} GiB of RAM, so the most allowed is ${memMax} GiB (4 GiB stays free for macOS); a larger saved budget is lowered to that.` : "";
    return `MLX watchdog limit; a change rebuilds the pipeline on the next job. Peak use is ≈10 GiB, ≈5.5 GiB in low-memory mode.${cap}`;
  }
  /** The whole settings object from the form (validated; null + toast when a field is out of range). */
  function body() {
    const mem = Number(f.mem.value);
    if (!(mem >= memMin && mem <= memMax)) { toast(`Memory budget must be between ${memMin} and ${memMax} GiB on this Mac`, "err"); return null; }
    const prune = f.prune.value.trim() === "" ? null : Number(f.prune.value);
    if (prune !== null && !(Number.isInteger(prune) && prune >= 1 && prune <= 365)) { toast("Auto-delete uploads must be 1–365 days (or blank for never)", "err"); return null; }
    const b = { default_preset: f.preset.value, memory_budget_gib: mem, low_memory: f.lowMem.value, require_ac: f.ac.checked, fast_numerics: f.fast.checked, theme: f.theme.value, prune_uploads_days: prune, assist_provider: f.assistProvider.value, assist_model: f.assistModel.value.trim() };
    if (f.apiKey.value) b.anthropic_api_key = f.apiKey.value;
    return b;
  }
  async function put(b, msg) {
    s = await api.saveSettings(b); app.settings = s; applyTheme(s.theme);
    f.apiKey.value = ""; f.apiKey.placeholder = keyPlaceholder(s);
    f.lowMem.options[0].textContent = autoLabel(s); f.mem.value = s.memory_budget_gib;
    if (msg) toast(msg, "ok");
    if (app.refreshStatus) await app.refreshStatus(); // provider/key changes show up now, not on the next 5 s poll
  }
  const save = h("button", { type: "submit", class: "primary" }, "Save settings");
  const clearKey = h("button", { type: "button", class: "ghost sm", onclick: async () => {
    if (!s.has_api_key && !f.apiKey.value) return toast("No API key is saved", "info");
    if (!confirmDialog("Remove the saved Anthropic API key?")) return;
    const b = body(); if (!b) return;
    clearKey.disabled = true;
    try { await put({ ...b, anthropic_api_key: "" }, "API key removed"); } catch (err) { toastError(err); }
    clearKey.disabled = false;
  } }, "Clear key");
  const test = h("button", { type: "button", class: "sm", onclick: async () => {
    const b = body(); if (!b) return;
    test.disabled = true; test.textContent = "Testing…";
    try { await put(b, null); const r = await api.testAssist(); toast(`Claude answered via ${r.provider}${r.model ? ` (${r.model})` : ""} in ${Number(r.seconds).toFixed(1)}s`, "ok"); }
    catch (err) { toastError(err); }
    test.disabled = false; test.textContent = "Test";
  } }, "Test");
  const assistStatus = h("span", { class: "hint" });
  const form = h("form", { class: "panel stack", onsubmit: async (e) => {
    e.preventDefault();
    const b = body(); if (!b) return;
    save.disabled = true;
    try { await put(b, "Settings saved"); } catch (err) { toastError(err); }
    save.disabled = false;
  } },
    h("h3", {}, "Generation"),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Default preset"), f.preset),
    h("label", { class: "field" }, h("span", { class: "lbl" }, `Memory budget (GiB, ${memMin}–${memMax})`), f.mem, h("span", { class: "hint" }, memHint(s))),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Low-memory mode"), f.lowMem,
      h("span", { class: "hint" }, "Loads the models one at a time so a song fits in ~5.5 GiB; costs a second or two per Quality song. Auto turns it on for Macs with 24 GB or less.")),
    h("label", { class: "check" }, f.ac, "Require AC power before running jobs"),
    h("label", { class: "check" }, f.fast, "Fast numerics"),
    h("span", { class: "hint" }, "Runs both CFG branches in one pass and uses native BF16 attention for synthesis (about 2.5× faster synthesis on M5). Songs differ very slightly from exact mode, so a seed only reproduces a song made in the same mode."),
    h("h3", {}, "Storage"),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Auto-delete unused uploads after N days (blank = never)"), f.prune,
      h("span", { class: "hint" }, "Uploads no job references are removed from data/uploads at startup and after each job; uploads a queued or running job needs are never touched. Manage them under Library → Uploads.")),
    h("h3", {}, "Appearance"),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Theme"), f.theme),
    h("h3", {}, "Claude assist"),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Provider"), f.assistProvider,
      h("span", { class: "hint" }, "Fills the Create / Cover / Hum forms from a prompt. The CLI uses your Claude login; the API key is stored in data/app.db.")),
    h("div", { class: "grid2" },
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Model"), f.assistModel),
      h("label", { class: "field" }, h("span", { class: "lbl" }, "API key"), f.apiKey)),
    h("div", { class: "row" }, test, clearKey, assistStatus),
    h("div", {}, save));

  const statusBox = h("dl", { class: "kv" });
  const modelsBox = h("dl", { class: "kv" });
  function assistLine(st) {
    const a = st && st.assist;
    if (!a) return st ? "" : "offline";
    const bits = [a.cli ? "CLI found" : "no CLI", a.api_key ? (s.has_api_key ? "key set" : "key set (env)") : "no key"]; // has_api_key = stored key only
    if (a.provider) return [`via ${a.provider.toUpperCase()}${a.model ? ` (${a.model})` : ""}`, ...bits].join(" · ");
    return (a.reasons && a.reasons.length ? a.reasons : ["unavailable", ...bits]).join(" · ");
  }
  function paint(st) {
    assistStatus.textContent = assistLine(st);
    if (!st) { fill(statusBox, h("dt", {}, "Server"), h("dd", { class: "muted" }, "offline")); return; }
    const e = st.engine, q = st.queue;
    fill(statusBox, 
      h("dt", {}, "Engine"), h("dd", {}, h("span", { class: `tag ${e.state === "ready" ? "ok" : e.state === "busy" ? "accent" : ""}` }, e.state)),
      h("dt", {}, "Precision"), h("dd", {}, e.precision || "—"), h("dt", {}, "Memory"), h("dd", {}, fmt.gib(e.memory_gib)),
      h("dt", {}, "Low-memory"), h("dd", {}, e.low_memory === true ? h("span", { class: "tag accent" }, "on") : e.low_memory === false ? "off" : "—"),
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
