import { api, songUrl } from "../api.js";
import { confirmDialog, fill, fmt, h, jobTitle, presetPicker, randomSeed, renderScore, scorePlayer, STAGE_NAMES, store, toast, toastError } from "../ui.js";

export async function songView({ el, param, app }) {
  let job;
  try { ({ job } = await api.job(param)); }
  catch (e) { fill(el, h("div", { class: "empty" }, e.status === 404 ? "This song does not exist (it may have been deleted)." : e.message, " ", h("a", { href: "#/library" }, "Back to library"))); return {}; }
  if (job.status === "queued" || job.status === "running") { fill(el, h("div", { class: "empty" }, "This job is still ", job.status, ". ", h("a", { href: "#/queue" }, "Watch it in the queue"))); return {}; }
  const p = job.params, t = job.timing || {};
  const [abc, transcription] = await Promise.all([
    job.artifacts.score ? api.text(songUrl(job.id, "score.abc")).catch(() => "") : "",
    job.kind === "cover" && job.artifacts.transcription ? api.text(songUrl(job.id, "transcription/score.abc")).catch(() => "") : "",
  ]);

  // Score editor + regenerate
  const scoreEl = h("div", { class: "score" });
  const abcArea = h("textarea", { id: "s-abc", class: "mono", rows: 14, spellcheck: false }, abc);
  let redraw;
  abcArea.addEventListener("input", () => { clearTimeout(redraw); redraw = setTimeout(() => renderScore(scoreEl, abcArea.value), 400); });
  if (abc) renderScore(scoreEl, abc);
  const player = scorePlayer(() => scoreEl._visual);
  const styleIn = h("textarea", { id: "s-style", rows: 2 }, p.style || "");
  const seedIn = h("input", { id: "s-seed", type: "number", min: 0, max: 2147483647, value: job.seed });
  const presets = presetPicker({ preset: job.preset, precision: job.precision, ode_steps: job.ode_steps }, app.status && app.status.presets);
  const regenBtn = h("button", { class: "primary", onclick: regenerate }, "Regenerate from this score");
  const countIn = h("input", { id: "s-count", type: "number", min: 2, max: 16, value: 3, style: "width:70px" });
  const varBtn = h("button", { onclick: variations }, "More variations");

  async function regenerate() {
    const text = abcArea.value.trim();
    if (!text) return toast("The ABC score is empty", "err");
    if (p.cot === "off") toast("Parent used mode “off”; the server will regenerate with mode melody", "info");
    regenBtn.disabled = true;
    try {
      const style = styleIn.value.trim(), seed = Number(seedIn.value);
      const r = await api.submit({ kind: "regenerate", ...presets.value(), params: { parent_id: job.id, abc: text, style: style && style !== p.style ? style : null, lyrics: null, seed: seed !== job.seed ? seed : null, title: null } });
      toast(`Queued regeneration of “${jobTitle(job)}”`, "ok"); player.stop(); location.hash = "#/queue";
      void r;
    } catch (e) { toastError(e); regenBtn.disabled = false; }
  }
  async function variations() {
    const count = Math.max(2, Math.min(16, Number(countIn.value) || 3));
    varBtn.disabled = true;
    try {
      const base = { style: p.style, lyrics: p.lyrics, cot: p.cot || "full", seed: randomSeed(), cfg_scale: p.cfg_scale ?? null, abc: p.abc || null, title: p.title || null };
      const r = await api.submit({ kind: "variations", preset: job.preset, precision: job.precision, ode_steps: job.ode_steps, params: { count, base, random_seeds: false, label: null } });
      const groups = store.get("groups", {}); groups[r.group.id] = r.group.label; store.set("groups", groups);
      toast(`Queued ${count} variations`, "ok"); location.hash = "#/queue";
    } catch (e) { toastError(e); varBtn.disabled = false; }
  }
  async function remove() {
    if (!confirmDialog(`Delete “${jobTitle(job)}” and its files?`)) return;
    try { await api.remove(job.id); toast("Deleted", "ok"); location.hash = "#/library"; } catch (e) { toastError(e); }
  }

  const stageKeys = ["load", "transcribe", "plan", "semantic", "synthesize", "decode", "save", "e2e"].filter((k) => t[k] !== undefined && t[k] !== null);
  const timingTable = stageKeys.length ? h("div", { class: "table-wrap" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "Stage"), h("th", { class: "num" }, "Seconds"), h("th", { class: "num" }, "Share"))),
    h("tbody", {}, stageKeys.map((k) => h("tr", { style: k === "e2e" ? "font-weight:600" : "" }, h("td", {}, STAGE_NAMES[k] || k), h("td", { class: "num" }, Number(t[k]).toFixed(1)), h("td", { class: "num" }, k === "e2e" || !t.e2e ? "" : `${Math.round(100 * t[k] / t.e2e)}%`)))))) : h("p", { class: "muted small" }, "No timing recorded.");

  const planBox = h("pre", { class: "block mono" }, "Loading…");
  const planDetails = h("details", { ontoggle: async () => { if (planDetails.open && !planDetails._loaded) { planDetails._loaded = true; try { planBox.textContent = JSON.stringify(JSON.parse(await api.text(songUrl(job.id, "plan.json"))), null, 2); } catch (e) { planBox.textContent = "plan.json not available: " + e.message; } } } },
    h("summary", {}, "plan.json"), planBox);

  fill(el,
    h("div", { class: "view-head" }, h("h1", {}, jobTitle(job)), h("span", { class: "tag" }, job.kind), h("span", { class: `tag ${job.status === "done" ? "ok" : "err"}` }, job.status),
      job.group_id ? h("a", { class: "tag accent", href: `#/library?group=${job.group_id}` }, "group") : null, job.parent_id ? h("a", { class: "tag", href: `#/song/${job.parent_id}` }, "parent") : null,
      h("span", { class: "spacer" }), h("a", { class: "btn ghost sm", href: "#/library" }, "← Library")),
    job.status === "failed" ? h("div", { class: "errbox mono", style: "margin-bottom:14px" }, job.error || "failed") : null,
    job.truncated ? h("div", { class: "warnbox", style: "margin-bottom:14px" }, h("b", {}, "Truncated"), ` during ${job.truncated.phase}: ${job.truncated.reason}. The song may end early — try a shorter lyric or mode melody.`) : null,
    h("div", { class: "cols" },
      h("div", { class: "stack" },
        job.artifacts.audio ? h("div", { class: "panel stack" },
          h("audio", { controls: true, preload: "metadata", src: songUrl(job.id, "audio.flac") }),
          h("div", { class: "row" }, h("a", { class: "btn sm", href: songUrl(job.id, "audio.flac"), download: true }, "Download FLAC"), h("a", { class: "btn sm", href: songUrl(job.id, "audio.mp3"), download: true }, "MP3"), h("a", { class: "btn sm", href: songUrl(job.id, "artifacts.zip") }, "artifacts.zip"), h("span", { class: "spacer" }), h("button", { class: "ghost sm danger", onclick: remove }, "Delete"))) : null,
        h("section", { class: "stack" }, h("h3", {}, "Score"),
          abc ? [scoreEl, h("div", { class: "row" }, player, h("span", { class: "hint" }, "Edit the ABC below; the score re-renders as you type.")), abcArea]
            : h("p", { class: "muted" }, "No score for this song (mode off or planning did not finish)."),
          h("div", { class: "panel stack" },
            h("label", { class: "field" }, h("span", { class: "lbl" }, "Style (optional edit)"), styleIn),
            h("div", { class: "grid2 stack-narrow" }, h("label", { class: "field" }, h("span", { class: "lbl" }, "Seed"), seedIn), h("div", { class: "field" }, h("span", { class: "lbl" }, "Preset"), presets)),
            h("div", { class: "row" }, regenBtn, h("span", { class: "spacer" }), h("label", { class: "row", style: "gap:6px" }, countIn, varBtn)),
            h("p", { class: "hint" }, "Regenerate keeps the lyrics and mode; variations start from a fresh random seed. ", h("a", { href: `#/create?from=${job.id}` }, "Open in Create")))),
        transcription ? h("section", { class: "stack" }, h("h3", {}, "Transcription (from the uploaded audio)"), (() => { const s = h("div", { class: "score" }); renderScore(s, transcription); return s; })(), h("details", {}, h("summary", {}, "transcription/score.abc"), h("pre", { class: "block mono" }, transcription))) : null),
      h("div", { class: "stack" },
        h("div", { class: "panel stack" }, h("h3", {}, "Request"),
          h("dl", { class: "kv" },
            h("dt", {}, "Style"), h("dd", {}, p.style || "—"),
            h("dt", {}, job.kind === "cover" ? "Task" : "Mode"), h("dd", {}, p.task || p.cot || "—"),
            h("dt", {}, "Preset"), h("dd", {}, `${job.preset} · ${job.precision} · ${job.ode_steps} steps`),
            h("dt", {}, "Seed"), h("dd", {}, job.seed), p.cfg_scale != null ? [h("dt", {}, "CFG"), h("dd", {}, p.cfg_scale)] : null,
            h("dt", {}, "Audio"), h("dd", {}, fmt.dur(t.audio_seconds)),
            h("dt", {}, "Created"), h("dd", { title: job.created_at }, fmt.when(job.created_at)), h("dt", {}, "Job"), h("dd", { class: "mono" }, job.id)),
          h("details", {}, h("summary", {}, "Lyrics"), h("pre", { class: "block" }, p.lyrics || "—"))),
        h("div", { class: "panel stack" }, h("h3", {}, "Timing"), timingTable,
          t.abc_tps || t.semantic_tps ? h("p", { class: "hint num" }, `ABC ${fmt.tps(t.abc_tps)} · semantic ${fmt.tps(t.semantic_tps)}`) : null),
        job.artifacts.plan ? h("div", { class: "panel" }, planDetails) : null)));
  return { unmount: () => player.stop() };
}
