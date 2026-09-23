import { api, songUrl } from "../api.js";
import { confirmDialog, fill, fmt, h, inlineEdit, jobTitle, loraLabel, loraPicker, presetPicker, projectPicker, projectTag, randomSeed, rememberGroup, renderScore, scorePlayer, seedField, STAGE_NAMES, takeControls, toast, toastError } from "../ui.js";
import { playButton, player } from "../player.js";

export async function songView({ el, param, app }) {
  let job;
  try { ({ job } = await api.job(param)); }
  catch (e) { fill(el, h("div", { class: "empty" }, e.status === 404 ? "This song does not exist (it may have been deleted)." : e.message, " ", h("a", { href: "#/library" }, "Back to library"))); return {}; }
  if (job.status === "queued" || job.status === "running") { fill(el, h("div", { class: "empty" }, "This job is still ", job.status, ". ", h("a", { href: "#/queue" }, "Watch it in the queue"))); return {}; }
  const p = job.params, t = job.timing || {};
  const [abc, transcription, humAbc] = await Promise.all([
    job.artifacts.score ? api.text(songUrl(job.id, "score.abc")).catch(() => "") : "",
    job.kind === "cover" && job.artifacts.transcription ? api.text(songUrl(job.id, "transcription/score.abc")).catch(() => "") : "",
    job.kind === "hum" && job.artifacts.hum ? api.text(songUrl(job.id, "hum/hum.abc")).catch(() => "") : "",
  ]);

  // Score editor + regenerate
  const scoreEl = h("div", { class: "score" });
  const abcArea = h("textarea", { id: "s-abc", class: "mono", rows: 14, spellcheck: false }, abc);
  let redraw;
  abcArea.addEventListener("input", () => { clearTimeout(redraw); redraw = setTimeout(() => renderScore(scoreEl, abcArea.value), 400); });
  if (abc) renderScore(scoreEl, abc);
  const midi = scorePlayer(() => scoreEl._visual); // page-local MIDI preview; songs go through the global player
  const styleIn = h("textarea", { id: "s-style", rows: 2 }, p.style || "");
  // Regenerate inherits the parent's seed when blank; the toggle (off by default) opts into a fresh random one.
  const seedIn = seedField({ id: "s-seed", seed: job.seed, random: false, randomLabel: "Random seed (instead of inheriting)" });
  const presets = presetPicker({ preset: job.preset, precision: job.precision, ode_steps: job.ode_steps }, app.status && app.status.presets);
  const loras = loraPicker(job.loras || [], app.status && app.status.loras ? app.status.loras.adapters : []);
  const regenBtn = h("button", { class: "primary", onclick: regenerate }, "Regenerate from this score");
  const countIn = h("input", { id: "s-count", type: "number", min: 2, max: 16, value: 3, style: "width:70px" });
  const varBtn = h("button", { onclick: variations, disabled: job.kind === "cover" && !transcription, title: job.kind === "cover" ? "Variations of a cover reuse its transcription" : "" }, "More variations");

  async function regenerate() {
    const text = abcArea.value.trim();
    if (!text) return toast("The ABC score is empty", "err");
    if (p.cot === "off") toast("Parent used mode “off”; the server will regenerate with mode melody", "info");
    regenBtn.disabled = true;
    try {
      // RegenerateParams: seed null = inherit parent, so a random seed must be rolled client-side.
      const style = styleIn.value.trim(), seed = seedIn.isRandom() ? randomSeed() : seedIn.value();
      await api.submit({ kind: "regenerate", ...presets.value(), loras: loras.value(), track_id: trackId(), params: { parent_id: job.id, abc: text, style: style && style !== p.style ? style : null, lyrics: null, seed: seed !== null && seed !== job.seed ? seed : null, title: null } });
      toast(`Queued regeneration of “${jobTitle(job)}”`, "ok"); midi.stop(); location.hash = afterSubmit();
    } catch (e) { toastError(e); regenBtn.disabled = false; }
  }
  async function variations() {
    const count = Math.max(2, Math.min(16, Number(countIn.value) || 3));
    varBtn.disabled = true;
    try {
      // Covers: seed the variations from the transcription so the melody is kept (cot melody, as the cover flow does).
      const base = job.kind === "cover"
        ? { style: p.style, lyrics: p.lyrics, cot: p.task === "full" ? "full" : "melody", seed: randomSeed(), cfg_scale: null, abc: transcription, title: p.title || null }
        : job.kind === "hum"
          ? { style: p.style, lyrics: p.lyrics, cot: "melody", seed: randomSeed(), cfg_scale: null, abc: abc || null, title: p.title || null } // variations keep the continued score
          : { style: p.style, lyrics: p.lyrics, cot: p.cot || "full", seed: randomSeed(), cfg_scale: p.cfg_scale ?? null, abc: p.abc || null, title: p.title || null };
      const r = await api.submit({ kind: "variations", preset: job.preset, precision: job.precision, ode_steps: job.ode_steps, loras: loras.value(), track_id: trackId(), params: { count, base, random_seeds: false, label: null } });
      rememberGroup(r.group, r.jobs);
      toast(`Queued ${count} variations`, "ok"); location.hash = afterSubmit();
    } catch (e) { toastError(e); varBtn.disabled = false; }
  }
  async function remove() {
    if (!confirmDialog(`Delete “${jobTitle(job)}” and its files?`)) return;
    try { await api.remove(job.id); player.remove(job.id); toast("Deleted", "ok"); location.hash = "#/library"; } catch (e) { toastError(e); }
  }

  // -- project membership: new takes made from a take land in the same track; the panel attaches/rates/chooses/detaches.
  const trackId = () => (job.take ? job.take.track_id : null);
  const afterSubmit = () => (job.take ? `#/project/${job.take.project_id}` : "#/queue");
  const headTag = h("span", { class: "row", style: "gap:4px" });
  const projectPanel = h("div", { class: "panel stack", id: "s-project" });
  function paintProject() {
    fill(headTag, projectTag(job));
    const t = job.take;
    if (!t) {
      const picker = projectPicker({ onPick: () => { addBtn.disabled = !picker.value(); } });
      const addBtn = h("button", { class: "primary sm", disabled: true, onclick: () => attach(picker.value()) }, "Add as a take");
      fill(projectPanel, h("h3", {}, "Project"), h("p", { class: "hint" }, "Add this song to a project track as a take; rate it and choose it as the track's final take."), picker, h("div", {}, addBtn));
      return;
    }
    const canChoose = job.status === "done" && job.artifacts.audio;
    const chooseBtn = t.chosen
      ? h("button", { class: "sm choose on", "aria-pressed": "true", title: "Unchoose", onclick: () => choose(null) }, "★ Chosen take")
      : h("button", { class: "sm choose", "aria-pressed": "false", disabled: !canChoose, title: canChoose ? "Use this take on the album" : "Only a finished take with audio can be chosen", onclick: () => choose(job.id) }, "☆ Choose as final take");
    fill(projectPanel, h("h3", {}, "Project"),
      h("p", {}, h("a", { href: `#/project/${t.project_id}` }, t.project_name), h("span", { class: "muted" }, " › "), t.track_name, t.chosen ? h("span", { class: "tag ok", style: "margin-left:6px" }, "chosen") : null),
      takeControls(job, { onChange: (j) => { job = j; } }),
      h("div", { class: "row" }, chooseBtn, h("span", { class: "spacer" }), h("button", { class: "ghost sm", onclick: detach }, "Detach")));
  }
  async function attach(target) {
    if (!target) return;
    try {
      try { await api.attachTakes(target.id, [job.id]); }
      catch (e) { if (e.status !== 409 || !confirmDialog(`${e.message}. Move it to “${target.name}”?`)) throw e; await api.attachTakes(target.id, [job.id], true); }
      await reloadJob(); toast(`Added to ${target.project_name} › ${target.name}`, "ok");
    } catch (e) { toastError(e); }
  }
  async function choose(id) {
    try { await api.patchTrack(job.take.track_id, { chosen_job_id: id }); await reloadJob(); } catch (e) { toastError(e); }
  }
  async function detach() {
    if (!confirmDialog(`Remove “${jobTitle(job)}” from “${job.take.track_name}”? The song is kept; its rating is dropped.`)) return;
    try { await api.detachTake(job.id); await reloadJob(); } catch (e) { toastError(e); }
  }
  async function reloadJob() { ({ job } = await api.job(job.id)); paintProject(); }

  const stageKeys = ["load", "transcribe", "hum", "plan", "semantic", "synthesize", "decode", "save", "e2e"].filter((k) => t[k] !== undefined && t[k] !== null);
  const timingTable = stageKeys.length ? h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Stage"), h("th", { class: "num" }, "Seconds"), h("th", { class: "num" }, "Share"))),
    h("tbody", {}, stageKeys.map((k) => h("tr", { style: k === "e2e" ? "font-weight:600" : "" }, h("td", {}, STAGE_NAMES[k] || k), h("td", { class: "num" }, Number(t[k]).toFixed(1)), h("td", { class: "num" }, k === "e2e" || !t.e2e ? "" : `${Math.round(100 * t[k] / t.e2e)}%`)))))) : h("p", { class: "muted small" }, "No timing recorded.");

  // Click-to-rename heading; clearing the title falls back to the style excerpt (shown as the placeholder).
  const untitled = jobTitle({ ...job, title: null, params: { ...p, title: null } });
  const titleEl = inlineEdit(job.title || "", async (title) => {
    const { job: renamed } = await api.patchJob(job.id, { title: title || null });
    Object.assign(job, renamed); // in place: the play button and take controls hold this object
    player.retitle(job.id, jobTitle(job));
  }, { tag: "h1", placeholder: untitled, allowEmpty: true, title: "Click to rename" });

  const planBox = h("pre", { class: "block mono" }, "Loading…");
  const planDetails = h("details", { ontoggle: async () => { if (planDetails.open && !planDetails._loaded) { planDetails._loaded = true; try { planBox.textContent = JSON.stringify(JSON.parse(await api.text(songUrl(job.id, "plan.json"))), null, 2); } catch (e) { planBox.textContent = "plan.json not available: " + e.message; } } } },
    h("summary", {}, "plan.json"), planBox);

  fill(el,
    h("div", { class: "view-head" }, titleEl, h("span", { class: "tag" }, job.kind), h("span", { class: `tag ${job.status === "done" ? "ok" : "err"}` }, job.status),
      job.group_id ? h("a", { class: "tag accent", href: `#/library?group=${job.group_id}` }, "group") : null, job.parent_id ? h("a", { class: "tag", href: `#/song/${job.parent_id}` }, "parent") : null, headTag,
      h("span", { class: "spacer" }), h("a", { class: "btn ghost sm", href: "#/library" }, "← Library")),
    job.status === "failed" ? h("div", { class: "errbox mono", style: "margin-bottom:14px" }, job.error || "failed") : null,
    job.truncated ? h("div", { class: "warnbox", style: "margin-bottom:14px" }, h("b", {}, "Truncated"), ` during ${job.truncated.phase}: ${job.truncated.reason}. The song may end early — try a shorter lyric or mode melody.`) : null,
    h("div", { class: "cols" },
      h("div", { class: "stack" },
        job.artifacts.audio ? h("div", { class: "panel stack" },
          h("div", { class: "row" }, playButton(job, { label: true, size: "" }), h("span", { class: "hint" }, "Playback controls are in the bar at the bottom.")),
          h("div", { class: "row" }, h("a", { class: "btn sm", href: songUrl(job.id, "audio.flac"), download: true }, "Download FLAC"), h("a", { class: "btn sm", href: songUrl(job.id, "audio.mp3"), download: true }, "MP3"), h("a", { class: "btn sm", href: songUrl(job.id, "artifacts.zip") }, "artifacts.zip"), h("span", { class: "spacer" }), h("button", { class: "ghost sm danger", onclick: remove }, "Delete"))) : null,
        h("section", { class: "panel stack" }, h("h3", {}, "Request"),
          h("p", {}, h("span", { class: "muted small", style: "text-transform:uppercase;letter-spacing:.05em" }, "Style "), p.style || "—"),
          h("pre", { class: "block lyrics-block", "aria-label": "Lyrics" }, p.lyrics || "—")),
        h("section", { class: "stack" }, h("h3", {}, job.kind === "hum" ? (p.melody === "hum_only" ? "Score (from your hum)" : p.melody === "continue" ? "Continued score" : "Score") : "Score"),
          abc ? [scoreEl, h("div", { class: "row" }, midi, h("span", { class: "hint" }, "Edit the ABC below; the score re-renders as you type.")), abcArea]
            : h("p", { class: "muted" }, "No score for this song (mode off or planning did not finish)."),
          h("div", { class: "panel stack" },
            h("label", { class: "field" }, h("span", { class: "lbl" }, "Style (optional edit)"), styleIn),
            h("div", { class: "grid2 stack-narrow" }, h("div", { class: "field" }, h("span", { class: "lbl" }, "Seed"), seedIn), h("div", { class: "field" }, h("span", { class: "lbl" }, "Preset"), presets)),
            h("div", { class: "field" }, h("span", { class: "lbl" }, "LoRA adapters"), loras),
            h("div", { class: "row" }, regenBtn, h("span", { class: "spacer" }), h("label", { class: "row", style: "gap:6px" }, countIn, varBtn)),
            h("p", { class: "hint" }, "Regenerate keeps the lyrics and mode; variations start from a fresh random seed. ", h("a", { href: `#/create?from=${job.id}` }, "Open in Create")))),
        transcription ? h("section", { class: "stack" }, h("h3", {}, "Transcription (from the uploaded audio)"), (() => { const s = h("div", { class: "score" }); renderScore(s, transcription); return s; })(), h("details", {}, h("summary", {}, "transcription/score.abc"), h("pre", { class: "block mono" }, transcription))) : null,
        humAbc ? h("section", { class: "stack" }, h("h3", {}, p.melody === "continue" ? "Your hum (the open score the planner continued)" : "Your hum (transcribed)"), (() => { const s = h("div", { class: "score" }); renderScore(s, humAbc); return s; })(), h("details", {}, h("summary", {}, "hum/hum.abc"), h("pre", { class: "block mono" }, humAbc))) : null),
      h("div", { class: "stack" },
        projectPanel,
        h("div", { class: "panel stack" }, h("h3", {}, "Details"),
          h("dl", { class: "kv" },
            h("dt", {}, job.kind === "cover" ? "Task" : job.kind === "hum" ? "Melody" : "Mode"), h("dd", {}, job.kind === "hum" ? (p.melody || "continue").replace("_", " ") : (p.task || p.cot || "—")),
            job.kind === "hum" ? [h("dt", {}, "Hum adapter"), h("dd", {}, p.adapter ? `${p.adapter} · influence ${p.hum_influence} · from ${p.offset_s}s` : "none (score only)")] : null,
            h("dt", {}, "Preset"), h("dd", {}, `${job.preset} · ${job.precision} · ${job.ode_steps} steps`),
            job.loras && job.loras.length ? [h("dt", {}, "LoRA"), h("dd", {}, loraLabel(job.loras))] : null,
            h("dt", {}, "Seed"), h("dd", {}, job.seed), p.cfg_scale != null ? [h("dt", {}, "CFG"), h("dd", {}, p.cfg_scale)] : null,
            h("dt", {}, "Audio"), h("dd", {}, fmt.dur(t.audio_seconds)),
            h("dt", {}, "Created"), h("dd", { title: job.created_at }, fmt.when(job.created_at)), h("dt", {}, "Job"), h("dd", { class: "mono" }, job.id))),
        h("div", { class: "panel stack" }, h("h3", {}, "Timing"), timingTable,
          t.abc_tps || t.semantic_tps ? h("p", { class: "hint num" }, `ABC ${fmt.tps(t.abc_tps)} · semantic ${fmt.tps(t.semantic_tps)}`) : null),
        job.artifacts.plan ? h("div", { class: "panel" }, planDetails) : null)));
  paintProject();
  return { unmount: () => midi.stop() };
}
