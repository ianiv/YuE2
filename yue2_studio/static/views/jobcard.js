// Shared job cards: a live (queued/running) card driven by SSE, and result cards for terminal jobs.
import { api, songUrl, subscribe } from "../api.js";
import { clearScore, estimate, fill, fmt, groupLabel, h, jobTitle, renderScore, STAGE_NAMES, STAGE_ORDER, toast, toastError } from "../ui.js";

const groupTag = (job) => job.group_id ? h("span", { class: "tag accent", title: groupLabel(job.group_id) || "variation group" }, groupLabel(job.group_id) ? fmt.excerpt(groupLabel(job.group_id), 28) : "group") : null;

/**
 * Live card for a queued/running job. Subscribes to its SSE stream; calls onFinish(card, job) once when the
 * job reaches a terminal state (from SSE `done` or a cancel response). Returns {el, job, update, paint, close}.
 * The streaming score section is collapsible per card; `scoreCollapsed: true` starts it collapsed (no SVG is
 * rendered while collapsed — the latest ABC text is kept and drawn on expand).
 */
export function liveCard(job, { onFinish, onGone, scoreCollapsed = false } = {}) {
  const c = { job, last: job.progress || null, abc: null, abcFinal: false, scoreOpen: !scoreCollapsed, startedAt: job.started_at ? Date.parse(job.started_at) : null, finished: false };
  const stageRow = h("div", { class: "stages" });
  const bar = h("div", { class: "progress" }, h("i"));
  const stats = h("div", { class: "stats" });
  const abcBox = h("div", { class: "abc-live score", hidden: true });
  const scoreLabel = h("span", { class: "hint" });
  const scoreBtn = h("button", { type: "button", class: "ghost sm", onclick: () => setScoreOpen(!c.scoreOpen) });
  const scoreHead = h("div", { class: "row between score-head", hidden: true }, scoreLabel, scoreBtn);
  const cancelBtn = h("button", { class: "sm danger", onclick: () => cancel(cancelBtn) }, "Cancel");
  const status = h("span", { class: "tag" }, job.status);
  c.el = h("div", { class: "card", dataset: { id: job.id } },
    h("div", { class: "row between" },
      h("div", { class: "row" }, h("span", { class: "title" }, jobTitle(job)), status, h("span", { class: "tag" }, job.kind), groupTag(job)),
      cancelBtn),
    h("div", { class: "meta" }, h("span", {}, "preset ", h("b", {}, job.preset)), h("span", {}, "seed ", h("b", { class: "num" }, job.seed)), h("span", {}, "mode ", h("b", {}, job.params.cot || job.params.task || "—")),
      job.position !== null && job.status === "queued" ? h("span", { class: "pos" }, "position ", h("b", {}, job.position + 1)) : null),
    stageRow, bar, stats, scoreHead, abcBox);

  function onEvent(ev) {
    if (ev.type === "status" && ev.status === "running" && !c.startedAt) c.startedAt = Date.now();
    if (ev.type === "stage" || ev.type === "token") c.last = ev.type === "stage" ? ev : { ...(c.last || {}), tps: ev.tps ?? (c.last && c.last.tps), completed: ev.tokens ?? (c.last && c.last.completed) };
    if ((ev.type === "status" || ev.type === "log") && ev.message) c.message = ev.message;
    if (ev.type === "abc" && ev.text) {
      c.abc = ev.text; c.abcFinal = ev.partial === false;
      scoreHead.hidden = false; scoreLabel.textContent = c.abcFinal ? "Final score" : "Score streaming in…";
      if (c.scoreOpen) drawScore(c.abcFinal); // collapsed: keep buffering, draw on expand
    }
    paint();
  }

  function drawScore(immediate) {
    abcBox.hidden = false;
    abcBox.classList.toggle("final", c.abcFinal); abcBox.title = c.abcFinal ? "Final score" : "Score streaming in…";
    renderScore(abcBox, c.abc, { staffwidth: 600 }, { immediate });
  }
  /** Expand/collapse the score; collapsing removes the SVG (and cancels any pending throttled draw). */
  function setScoreOpen(open) {
    c.scoreOpen = open;
    scoreBtn.textContent = open ? "Hide score" : "Show score";
    scoreBtn.setAttribute("aria-expanded", String(open));
    if (open) { if (c.abc) drawScore(true); }
    else { abcBox.hidden = true; clearScore(abcBox); }
  }
  setScoreOpen(c.scoreOpen);

  function paint() {
    const { job, last } = c;
    const running = job.status === "running" || (last && last.stage);
    status.textContent = job.status === "queued" && !running ? "queued" : "running";
    status.className = "tag " + (running ? "accent" : "");
    const pos = c.el.querySelector(".pos"); if (pos) pos.hidden = !!running;
    const stages = STAGE_ORDER.filter((s) => s !== "transcribe" || job.kind === "cover");
    const idx = last && last.stage ? stages.indexOf(last.stage) : -1;
    fill(stageRow, stages.map((s, i) => h("span", { class: i < idx || (i === idx && last.status === "complete") ? "done" : i === idx ? "active" : "" }, STAGE_NAMES[s])));
    const elapsed = c.startedAt ? (Date.now() - c.startedAt) / 1000 : 0;
    const est = estimate(last, elapsed, job.kind);
    bar.classList.toggle("indeterminate", running && !(last && last.total));
    bar.firstChild.style.width = `${Math.round(est.fraction * 100)}%`;
    const parts = [];
    if (last && last.stage) parts.push(h("span", {}, h("b", {}, last.label || STAGE_NAMES[last.stage]), last.total ? ` ${last.completed ?? 0}/${last.total} ${last.unit || ""}` : last.completed ? ` ${last.completed} ${last.unit || ""}` : ""));
    else if (c.message) parts.push(h("span", {}, c.message));
    else parts.push(h("span", {}, job.status === "queued" ? "Waiting for the worker" : "Starting…"));
    if (last && last.tps) parts.push(h("span", {}, h("b", {}, fmt.tps(last.tps))));
    if (c.startedAt) parts.push(h("span", {}, "elapsed ", h("b", {}, fmt.secs(elapsed))));
    if (est.eta !== null && running) parts.push(h("span", {}, "ETA ", h("b", {}, "~" + fmt.secs(est.eta))));
    fill(stats, parts);
  }

  async function cancel(btn) {
    btn.disabled = true;
    try {
      const { status: code, data } = await api.cancel(c.job.id);
      if (code === 202) { btn.textContent = "Cancelling…"; toast("Cancel requested; the worker will stop at the next checkpoint", "info"); }
      else finish(data.job);
    } catch (e) { if (e.status === 409) { toast("Job already finished", "info"); onGone && onGone(c); } else toastError(e); btn.disabled = false; }
  }

  function finish(job) {
    if (c.finished) return; // cancel response and SSE `done` can both arrive
    c.finished = true; c.job = job; close();
    if (job.status === "done") toast(`Done: ${jobTitle(job)}`, "ok", { link: { href: `#/song/${job.id}`, label: "Open" } });
    else if (job.status === "failed") toast(`Failed: ${jobTitle(job)} — ${job.error || "unknown error"}`, "err", { timeout: 12000 });
    else toast(`Cancelled: ${jobTitle(job)}`, "info");
    onFinish && onFinish(c, job);
  }

  /** Merge a fresh Job from polling (position/started_at). */
  c.update = (job) => {
    c.job = job; if (job.started_at && !c.startedAt) c.startedAt = Date.parse(job.started_at);
    const pos = c.el.querySelector(".pos b"); if (pos && job.position !== null) pos.textContent = job.position + 1;
    if (["done", "failed", "cancelled"].includes(job.status)) finish(job);
  };
  const stop = subscribe(job.id, { onProgress: onEvent, onDone: finish, onError: () => { /* poll fallback keeps state fresh */ } });
  function close() { stop(); }
  c.paint = paint; c.close = close;
  paint();
  return c;
}

/** Result card for a terminal job (done → inline player; failed/cancelled → status + error). */
export function resultCard(job, { onUseSeed, onDismiss } = {}) {
  const t = job.timing || {};
  const head = h("div", { class: "row between" },
    h("div", { class: "row" }, h("a", { class: "title", href: `#/song/${job.id}` }, jobTitle(job)),
      h("span", { class: `tag ${job.status === "done" ? "ok" : job.status === "failed" ? "err" : ""}` }, job.status), h("span", { class: "tag" }, job.kind), groupTag(job)),
    onDismiss ? h("button", { class: "ghost sm", title: "Remove from this list", "aria-label": "Dismiss", onclick: () => onDismiss(job) }, "✕") : null);
  const meta = h("div", { class: "meta" },
    job.status === "done" ? h("span", {}, h("b", { class: "num" }, fmt.dur(t.audio_seconds))) : null,
    h("span", {}, "preset ", h("b", {}, job.preset)), h("span", {}, "seed ", h("b", { class: "num" }, job.seed)),
    h("span", {}, "mode ", h("b", {}, job.params.cot || job.params.task || "—")),
    h("span", { title: job.finished_at || job.created_at }, fmt.when(job.finished_at || job.created_at)),
    job.truncated ? h("span", { class: "tag warn", title: job.truncated.reason }, "truncated") : null);
  if (job.status !== "done") {
    return h("div", { class: "card", dataset: { id: job.id } }, head, job.error ? h("div", { class: "errbox mono" }, job.error) : null, meta);
  }
  return h("div", { class: "card", dataset: { id: job.id } }, head,
    h("audio", { controls: true, preload: "metadata", src: songUrl(job.id, "audio.flac") }),
    meta,
    h("div", { class: "row", style: "gap:4px" },
      h("a", { class: "btn sm", href: `#/song/${job.id}` }, "Open"),
      h("a", { class: "btn ghost sm", href: songUrl(job.id, "audio.flac"), download: true }, "FLAC"),
      h("a", { class: "btn ghost sm", href: songUrl(job.id, "audio.mp3"), download: true }, "MP3"),
      h("span", { class: "spacer" }),
      onUseSeed ? h("button", { class: "ghost sm", title: "Copy this seed into the form", onclick: () => onUseSeed(job) }, "Use this seed") : null,
      job.artifacts && job.artifacts.score ? h("a", { class: "btn ghost sm", href: `#/song/${job.id}`, title: "Open the song page to edit the score and regenerate" }, "Regenerate from its score") : null));
}
