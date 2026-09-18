import { api } from "../api.js";
import { fill, fmt, h, loraPicker, presetPicker, seedField, store, toast, toastError, uploadPicker } from "../ui.js";

const TASKS = [["melody-full", "Melody → full", "Transcribe the melody, let YuE2 write the full arrangement. Recommended for covers."],
  ["melody-vocal", "Melody → vocal", "Transcribe the melody and follow it with the vocal line only."],
  ["full", "Full transcription", "Transcribe the whole arrangement and follow it closely (closest to the original)."]];
const ACCEPT = ".mp3,.wav,.flac,.m4a,.ogg";

export async function coverView({ el, app }) {
  const saved = store.get("cover", { style: "", lyrics: "", task: "melody-full", seed: "", random_seed: true, title: "", preset: (app.settings && app.settings.default_preset) || "quality", precision: "8bit", ode_steps: 16, loras: [], upload_id: "" });
  let upload = null, dirty = false, task = saved.task;
  const fileIn = h("input", { type: "file", accept: ACCEPT, id: "c-file", onchange: (e) => e.target.files[0] && doUpload(e.target.files[0]) });
  const dropIdle = () => [h("b", {}, "Drop an audio file"), " or click to choose", h("div", { class: "hint" }, "mp3, wav, flac, m4a, ogg · up to 200 MB")];
  const dropText = h("div", {}, dropIdle());
  // Reusing an earlier upload skips the round trip; a new drop/pick replaces it (and resets the picker).
  const picker = uploadPicker({ onPick: usePicked });
  function usePicked(u) {
    upload = u ? { upload_id: u.upload_id, filename: u.filename, seconds: u.seconds } : null;
    fill(dropText, u ? [h("b", {}, u.filename), h("div", { class: "hint" }, `${fmt.dur(u.seconds)} · recent upload · drop or click to use a different file`)] : dropIdle());
    if (u && !f.title.value) f.title.placeholder = u.filename.replace(/\.[^.]+$/, "");
    syncAvailability(app.status); collect();
  }
  const drop = h("div", { class: "drop", tabindex: 0, role: "button", "aria-label": "Choose audio file", onclick: () => fileIn.click(), onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } },
    ondragover: (e) => { e.preventDefault(); drop.classList.add("over"); }, ondragleave: () => drop.classList.remove("over"),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) doUpload(f); } }, dropText, fileIn);
  const f = {
    title: h("input", { id: "c-title", type: "text", value: saved.title, placeholder: "defaults to the file name" }),
    style: h("textarea", { id: "c-style", rows: 2, placeholder: "e.g. acoustic folk, male vocal, fingerpicked guitar", required: true }, saved.style),
    lyrics: h("textarea", { id: "c-lyrics", class: "lyrics", placeholder: "[Verse]\n…", required: true }, saved.lyrics),
    seed: seedField({ id: "c-seed", seed: saved.seed, random: saved.random_seed !== false, onChange: () => collect() }),
  };
  const taskHint = h("p", { class: "hint" });
  const taskSeg = h("div", { class: "seg", role: "group", "aria-label": "Task" }, TASKS.map(([v, l]) => h("button", { type: "button", dataset: { v }, onclick: () => setTask(v) }, l)));
  function setTask(v) { task = v; taskSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === task))); taskHint.textContent = TASKS.find((x) => x[0] === task)[2]; }
  setTask(task);
  const presets = presetPicker(saved, app.status && app.status.presets);
  const loras = loraPicker(saved.loras, app.status && app.status.loras ? app.status.loras.adapters : []);
  const submit = h("button", { type: "submit", class: "primary", id: "c-submit", disabled: true }, "Create cover");
  const unavailable = h("div", { class: "warnbox", hidden: true });

  function syncAvailability(status) {
    const cover = status && status.cover;
    const ok = !!(cover && cover.available);
    unavailable.hidden = ok;
    if (!ok) fill(unavailable, h("b", {}, "Covers are unavailable on this machine."), " ", (cover && cover.reasons && cover.reasons.length) ? h("ul", { style: "margin:4px 0 0 18px" }, cover.reasons.map((r) => h("li", {}, r))) : status ? "" : "Waiting for /api/status…", h("p", { class: "hint" }, "Run scripts/setup.py --with-cover and make sure ffmpeg is installed."));
    submit.disabled = !ok || !upload;
  }
  syncAvailability(app.status);
  const listener = (s) => { syncAvailability(s); if (s && s.loras) loras.update(s.loras.adapters); };
  app.listeners.add(listener);

  async function doUpload(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!ACCEPT.includes("." + ext)) return toast(`Unsupported file type .${ext}`, "err");
    dirty = true; // a fresh file wins over the restored pick even while still uploading
    fill(dropText, h("span", {}, "Uploading ", h("b", {}, file.name), "…"));
    picker.reset();
    try {
      upload = await api.upload(file);
      fill(dropText, h("b", {}, upload.filename), h("div", { class: "hint" }, upload.seconds !== null ? `${fmt.dur(upload.seconds)} · ` : "", `${(file.size / 1048576).toFixed(1)} MB · click to replace`));
      if (!f.title.value) f.title.placeholder = upload.filename.replace(/\.[^.]+$/, "");
      syncAvailability(app.status); collect();
      picker.refresh();
    } catch (e) { upload = null; toastError(e); fill(dropText, h("b", {}, "Upload failed"), " — click to try again"); collect(); }
  }

  function collect() { const v = { title: f.title.value.trim(), style: f.style.value.trim(), lyrics: f.lyrics.value, task, seed: f.seed.raw(), random_seed: f.seed.isRandom(), loras: loras.value(), upload_id: upload ? upload.upload_id : "", ...presets.value() }; store.set("cover", v); return v; }
  async function onSubmit(e) {
    e.preventDefault();
    const v = collect();
    if (!upload) return toast("Upload an audio file first", "err");
    if (!v.style || !v.lyrics.trim()) return toast("Style and lyrics are required", "err");
    submit.disabled = true;
    try {
      const r = await api.submit({ kind: "cover", preset: v.preset, precision: v.precision, ode_steps: v.ode_steps, loras: v.loras, params: { upload_id: upload.upload_id, task: v.task, style: v.style, lyrics: v.lyrics, seed: f.seed.value(), title: v.title || null } });
      toast(`Queued cover “${r.job.title || upload.filename}”`, "ok"); location.hash = "#/queue";
    } catch (err) { toastError(err); submit.disabled = false; }
  }

  const form = h("form", { class: "cols", onsubmit: onSubmit, oninput: collect },
    h("div", { class: "stack" }, unavailable, drop, picker,
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Task"), taskSeg, taskHint),
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Title"), f.title),
      h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Style"), h("span", {}, "required")), f.style),
      h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Lyrics"), h("span", {}, "required")), f.lyrics)),
    h("div", { class: "panel sticky stack" },
      h("div", { class: "field" }, h("span", { class: "lbl" }, "Preset"), presets),
      h("div", { class: "field" }, h("span", { class: "lbl" }, "LoRA adapters"), loras),
      h("div", { class: "field" }, h("span", { class: "lbl" }, "Seed"), f.seed),
      submit, h("p", { class: "hint" }, "The upload is transcribed first (SheetSage2 + MERT), then the song is generated from that score.")));
  fill(el, h("div", { class: "view-head" }, h("h1", {}, "Cover"), h("span", { class: "sub" }, "Transcribe an existing recording and re-imagine it in a new style.")), form);
  // Restore the last-used upload only if it still exists on the server — and only if the user has not
  // already dropped/recorded a fresh file while the list was loading.
  picker.refresh().then(() => { if (saved.upload_id && !upload && !dirty) { const u = picker.set(saved.upload_id); if (u) usePicked(u); else collect(); } });
  return { unmount: () => app.listeners.delete(listener) };
}
