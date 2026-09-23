import { api } from "../api.js";
import { assistBox, fill, fmt, h, loraPicker, presetPicker, seedField, store, toast, toastError, trackBanner, uploadPicker } from "../ui.js";

const MELODY = [["continue", "Continue my melody", "The hum's notes open the score; YuE2 writes the rest of the song around them (the hummed phrase tends to come back as the hook)."],
  ["hum_only", "Hum is the whole melody", "The transcribed hum is the complete vocal line, like a cover; the song is as long as the hum."],
  ["ignore", "Ignore the notes", "YuE2 plans its own melody; only the prosody adapter shapes phrasing from the hum (needs an adapter)."]];
const ACCEPT = ".mp3,.wav,.flac,.m4a,.ogg,.webm,.mp4";
const MIME_EXT = [["audio/mp4", "m4a"], ["audio/webm;codecs=opus", "webm"], ["audio/webm", "webm"], ["audio/ogg;codecs=opus", "ogg"], ["audio/ogg", "ogg"]];
const MAX_RECORD_S = 120;

export async function humView({ el, query, app }) {
  const saved = store.get("hum", { style: "", lyrics: "", melody: "continue", adapter: "", hum_influence: 1, offset_s: 0, cfg_scale: "", seed: "", random_seed: true, title: "", preset: (app.settings && app.settings.default_preset) || "quality", precision: "8bit", ode_steps: 16, loras: [], upload_id: "" });
  let upload = null, dirty = false, melody = saved.melody;
  let trackId = null; // ?track=<id>: closure only, never in store("hum")
  const banner = await trackBanner(query.track, { onDismiss: () => { trackId = null; history.replaceState(null, "", "#/hum"); } });
  if (banner) trackId = banner.track.id;

  // -- input: drop zone + in-browser recorder + recent-upload picker -----------------------------
  const fileIn = h("input", { type: "file", accept: ACCEPT, id: "h-file", onchange: (e) => e.target.files[0] && doUpload(e.target.files[0]) });
  const dropIdle = () => [h("b", {}, "Drop a recording of your hum"), " or click to choose", h("div", { class: "hint" }, "10–30 s is plenty · mp3, wav, flac, m4a, ogg, webm")];
  const dropText = h("div", {}, dropIdle());
  // A recording or drop is uploaded right away; the picker reuses an earlier upload instead (and is reset by a new one).
  const picker = uploadPicker({ onPick: usePicked, label: "Or reuse a recent upload or recording" });
  function usePicked(u) {
    upload = u ? { upload_id: u.upload_id, filename: u.filename, seconds: u.seconds } : null;
    fill(dropText, u ? [h("b", {}, u.filename), h("div", { class: "hint" }, `${fmt.dur(u.seconds)} · recent upload · drop or record again to replace`)] : dropIdle());
    if (u && !f.title.value) f.title.placeholder = u.filename.replace(/\.[^.]+$/, "");
    sync(app.status); collect();
  }
  const drop = h("div", { class: "drop", tabindex: 0, role: "button", "aria-label": "Choose a hum recording", onclick: () => fileIn.click(), onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } },
    ondragover: (e) => { e.preventDefault(); drop.classList.add("over"); }, ondragleave: () => drop.classList.remove("over"),
    ondrop: (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) doUpload(f); } }, dropText, fileIn);
  const rec = recorder({ onBlob: (file) => doUpload(file) });

  async function doUpload(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!ACCEPT.includes("." + ext)) return toast(`Unsupported file type .${ext}`, "err");
    dirty = true; // a fresh file wins over the restored pick even while still uploading
    fill(dropText, h("span", {}, "Uploading ", h("b", {}, file.name), "…"));
    picker.reset();
    try {
      upload = await api.upload(file);
      fill(dropText, h("b", {}, upload.filename), h("div", { class: "hint" }, upload.seconds !== null ? `${fmt.dur(upload.seconds)} · ` : "", `${(file.size / 1048576).toFixed(1)} MB · drop or record again to replace`));
      if (!f.title.value) f.title.placeholder = upload.filename.replace(/\.[^.]+$/, "");
      sync(app.status); collect();
      picker.refresh();
    } catch (e) { upload = null; toastError(e); fill(dropText, h("b", {}, "Upload failed"), " — try again"); collect(); }
  }

  // -- fields ----------------------------------------------------------------------------------
  const f = {
    title: h("input", { id: "h-title", type: "text", value: saved.title, placeholder: "defaults to the recording name" }),
    style: h("textarea", { id: "h-style", rows: 2, placeholder: "e.g. indie folk, male vocal, acoustic guitar, stomps and claps, 120 BPM", required: true }, saved.style),
    lyrics: h("textarea", { id: "h-lyrics", class: "lyrics", placeholder: "[Verse]\n…\n\n[Chorus]\n…", required: true }, saved.lyrics),
    seed: seedField({ id: "h-seed", seed: saved.seed, random: saved.random_seed !== false, onChange: () => collect() }),
    adapter: h("select", { id: "h-adapter", onchange: () => { syncAdapter(); collect(); } }),
    influence: h("input", { id: "h-influence", type: "range", min: 0, max: 3, step: 0.05, value: saved.hum_influence, oninput: () => { influenceOut.textContent = Number(f.influence.value).toFixed(2); } }),
    cfg: h("input", { id: "h-cfg", type: "number", min: 0, max: 20, step: 0.1, value: saved.cfg_scale ?? "", placeholder: "engine default" }),
    offset: h("input", { id: "h-offset", type: "number", min: 0, max: 600, step: 0.1, value: saved.offset_s, style: "width:110px" }),
  };
  // Claude assist: fills title/style/lyrics; applyFields returns the previous values so the box can undo.
  function applyFields(fields) {
    const prev = {};
    for (const k of ["title", "style", "lyrics"]) if (k in fields) { prev[k] = f[k].value; f[k].value = fields[k] ?? ""; }
    collect();
    return prev;
  }
  const getContext = () => { const c = { title: f.title.value.trim(), style: f.style.value.trim(), lyrics: f.lyrics.value.trim() }; for (const k in c) if (!c[k]) delete c[k]; return c; };
  const assist = assistBox({ page: "hum", app, getContext, apply: applyFields });
  app.listeners.add(assist.onStatus);
  const influenceOut = h("b", { class: "num" }, Number(saved.hum_influence).toFixed(2));
  const melodyHint = h("p", { class: "hint" });
  const melodySeg = h("div", { class: "seg", role: "group", "aria-label": "Melody" }, MELODY.map(([v, l]) => h("button", { type: "button", dataset: { v }, onclick: () => setMelody(v) }, l)));
  function setMelody(v) { melody = v; melodySeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === melody))); melodyHint.textContent = MELODY.find((x) => x[0] === melody)[2]; syncAdapter(); }
  const adapterHint = h("span", { class: "hint" });
  const presets = presetPicker(saved, app.status && app.status.presets);
  const loras = loraPicker(saved.loras, app.status && app.status.loras ? app.status.loras.adapters : []);
  const submit = h("button", { type: "submit", class: "primary", id: "h-submit", disabled: true }, "Create song from hum");
  const unavailable = h("div", { class: "warnbox", hidden: true });
  const prosodyBox = h("div", { class: "stack", style: "gap:8px" });

  function paintAdapters(status) {
    const names = (status && status.hum && status.hum.adapters) || [];
    const current = f.adapter.value || saved.adapter || "";
    fill(f.adapter, h("option", { value: "" }, "None — score continuation only"), names.map((n) => h("option", { value: n, selected: n === current }, n)));
    if (current && !names.includes(current)) f.adapter.value = "";
    adapterHint.textContent = names.length ? "" : "No hum adapter in models/loras — download hum_adapter_v1_combined.safetensors to shape phrasing from the hum.";
    syncAdapter();
  }
  function syncAdapter() {
    const has = !!f.adapter.value;
    f.influence.disabled = f.offset.disabled = !has;
    prosodyBox.hidden = !has;
    const ignoreBtn = melodySeg.querySelector('button[data-v="ignore"]');
    ignoreBtn.disabled = !has; ignoreBtn.title = has ? "" : "Needs a hum adapter";
    if (!has && melody === "ignore") setMelody("continue");
  }
  function sync(status) {
    const hum = status && status.hum;
    const ok = !!(hum && hum.available);
    unavailable.hidden = ok;
    if (!ok) fill(unavailable, h("b", {}, "Hum to song is unavailable on this machine."), " ", (hum && hum.reasons && hum.reasons.length) ? h("ul", { style: "margin:4px 0 0 18px" }, hum.reasons.map((r) => h("li", {}, r))) : status ? "" : "Waiting for /api/status…", h("p", { class: "hint" }, "Run scripts/setup.py --with-cover, install ffmpeg, and uv sync (librosa)."));
    submit.disabled = !ok || !upload;
    if (status && status.loras) loras.update(status.loras.adapters);
    if (status && status.hum && status.hum.adapters.join("|") !== [...f.adapter.options].slice(1).map((o) => o.value).join("|")) paintAdapters(status);
  }
  paintAdapters(app.status); setMelody(melody); sync(app.status);
  const listener = (s) => sync(s);
  app.listeners.add(listener);

  function collect() {
    const v = { title: f.title.value.trim(), style: f.style.value.trim(), lyrics: f.lyrics.value, melody, adapter: f.adapter.value, hum_influence: Number(f.influence.value), offset_s: Math.max(0, Number(f.offset.value) || 0), cfg_scale: f.cfg.value === "" ? "" : Number(f.cfg.value), seed: f.seed.raw(), random_seed: f.seed.isRandom(), loras: loras.value(), upload_id: upload ? upload.upload_id : "", ...presets.value() };
    store.set("hum", v); return v;
  }
  async function onSubmit(e) {
    e.preventDefault();
    const v = collect();
    if (!upload) return toast("Record or upload a hum first", "err");
    if (!v.style || !v.lyrics.trim()) return toast("Style and lyrics are required", "err");
    if (v.melody === "ignore" && !v.adapter) return toast("“Ignore the notes” needs a hum adapter", "err");
    if (v.cfg_scale !== "" && !(v.cfg_scale >= 0 && v.cfg_scale <= 20)) return toast("CFG scale must be between 0 and 20", "err");
    submit.disabled = true;
    try {
      const params = { upload_id: upload.upload_id, style: v.style, lyrics: v.lyrics, seed: f.seed.value(), title: v.title || null, melody: v.melody, adapter: v.adapter || null, hum_influence: v.hum_influence, offset_s: v.offset_s, cfg_scale: v.cfg_scale === "" ? null : v.cfg_scale };
      const r = await api.submit({ kind: "hum", preset: v.preset, precision: v.precision, ode_steps: v.ode_steps, loras: v.loras, track_id: trackId, params });
      toast(`Queued “${r.job.title || upload.filename}”`, "ok"); location.hash = trackId ? `#/project/${banner.track.project_id}` : "#/queue";
    } catch (err) { toastError(err); submit.disabled = false; }
  }

  fill(prosodyBox,
    h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Hum influence"), influenceOut), f.influence,
      h("span", { class: "hint" }, "Guidance on the hum channel of the decoder: 1 = as trained, 0 = ignore the hum's phrasing, above 1 exaggerates it (costs ~2× synthesis time when ≠ 1).")),
    h("label", { class: "field" }, h("span", { class: "lbl" }, "Hum starts at (seconds into the song)"), f.offset,
      h("span", { class: "hint" }, "Where the hum's timing is placed; 0 = the song opens with your hum.")));

  const form = h("form", { class: "cols", onsubmit: onSubmit, oninput: collect },
    h("div", { class: "stack" }, unavailable,
      h("div", { class: "grid2 stack-narrow" }, drop, rec.el), picker, assist.el,
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Melody"), melodySeg, melodyHint),
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Title"), f.title),
      h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Style"), h("span", {}, "required")), f.style),
      h("label", { class: "field" }, h("span", { class: "lbl" }, h("span", {}, "Lyrics"), h("span", {}, "required")), f.lyrics)),
    h("div", { class: "panel sticky stack" },
      h("label", { class: "field" }, h("span", { class: "lbl" }, "Prosody adapter"), f.adapter, adapterHint),
      prosodyBox,
      h("div", { class: "field" }, h("span", { class: "lbl" }, "Preset"), presets),
      h("div", { class: "field" }, h("span", { class: "lbl" }, "LoRA adapters"), loras),
      h("div", { class: "grid2" },
        h("div", { class: "field" }, h("span", { class: "lbl" }, "Seed"), f.seed),
        h("label", { class: "field" }, h("span", { class: "lbl" }, "CFG scale"), f.cfg)),
      submit, h("p", { class: "hint" }, "The hum is transcribed (SheetSage2), the score is continued by the planner, and — with an adapter — its pitch and timing shape the decoder.")));
  fill(el, h("div", { class: "view-head" }, h("h1", {}, "Hum to song"), h("span", { class: "sub" }, "Hum a melody for 10–30 seconds, add a style and lyrics, get a whole song built around it.")), banner ? banner.el : null, form);
  // Restore the last-used upload only if it still exists on the server — and only if the user has not
  // already dropped/recorded a fresh file while the list was loading.
  picker.refresh().then(() => { if (saved.upload_id && !upload && !dirty) { const u = picker.set(saved.upload_id); if (u) usePicked(u); else collect(); } });
  return { unmount: () => { app.listeners.delete(listener); app.listeners.delete(assist.onStatus); rec.stop(true); } };
}

/** MediaRecorder-based recorder with a level meter, timer and playback preview. */
function recorder({ onBlob }) {
  const supported = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
  const meter = h("div", { class: "meter" }, h("i"));
  const timer = h("span", { class: "num" }, "0:00");
  const btn = h("button", { type: "button", class: "primary", disabled: !supported, "aria-pressed": "false" }, "● Record");
  const preview = h("audio", { controls: true, hidden: true, style: "width:100%" });
  const hint = h("div", { class: "hint" }, supported ? "Uses your microphone (allow access when asked); stops automatically after 2 minutes." : "Recording needs a browser with MediaRecorder (Safari 14.1+, Chrome, Firefox) on localhost or HTTPS.");
  const el = h("div", { class: "panel stack", style: "gap:8px" }, h("div", { class: "row between" }, h("b", {}, "Record a hum"), timer), meter, btn, preview, hint);
  let stream = null, media = null, chunks = [], ctx = null, raf = 0, tick = 0, started = 0, url = null;

  function pickMime() { for (const [m] of MIME_EXT) if (MediaRecorder.isTypeSupported(m)) return m; return ""; }
  async function start() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    } catch (e) { toast(`Microphone unavailable: ${e.message}`, "err"); return; }
    const mime = pickMime();
    media = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    chunks = [];
    media.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    media.onstop = () => {
      const type = media.mimeType || mime || "audio/webm";
      const ext = (MIME_EXT.find(([m]) => type.startsWith(m.split(";")[0])) || [null, "webm"])[1];
      const blob = new Blob(chunks, { type });
      if (url) URL.revokeObjectURL(url);
      url = URL.createObjectURL(blob); preview.src = url; preview.hidden = false;
      const stamp = new Date().toISOString().replace(/[-:]/g, "").slice(0, 15).replace("T", "-");
      onBlob(new File([blob], `hum-${stamp}.${ext}`, { type }));
      teardown();
    };
    media.start(250);
    started = Date.now();
    btn.textContent = "■ Stop"; btn.setAttribute("aria-pressed", "true"); btn.classList.add("recording");
    tick = setInterval(() => { const s = Math.floor((Date.now() - started) / 1000); timer.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; if (s >= MAX_RECORD_S) stop(); }, 200);
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      const paint = () => { analyser.getByteTimeDomainData(buf); let sum = 0; for (const v of buf) { const x = (v - 128) / 128; sum += x * x; } const rms = Math.sqrt(sum / buf.length); meter.firstChild.style.width = `${Math.min(100, rms * 300)}%`; raf = requestAnimationFrame(paint); };
      paint();
    } catch { /* meter is optional */ }
  }
  function teardown() {
    clearInterval(tick); cancelAnimationFrame(raf); meter.firstChild.style.width = "0%";
    if (ctx) { ctx.close().catch(() => {}); ctx = null; }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    media = null;
    btn.textContent = "● Record"; btn.setAttribute("aria-pressed", "false"); btn.classList.remove("recording");
  }
  function stop(discard = false) {
    if (media && media.state !== "inactive") { if (discard) media.onstop = teardown; media.stop(); } else teardown();
    if (discard && url) { URL.revokeObjectURL(url); url = null; }
  }
  btn.addEventListener("click", () => (media && media.state === "recording" ? stop() : start()));
  return { el, stop };
}
