import { api, subscribe } from "../api.js";
import { estimate, fill, fmt, groupLabel, h, jobTitle, renderScore, STAGE_NAMES, STAGE_ORDER, toast, toastError } from "../ui.js";

export async function queueView({ el }) {
  const cards = new Map(); // id -> {job, el, close, last, abc}
  const list = h("div", { class: "stack" });
  const empty = h("div", { class: "empty" }, "Nothing queued. ", h("a", { href: "#/create" }, "Create a song"), " or ", h("a", { href: "#/cover" }, "make a cover"), ".");
  fill(el, h("div", { class: "view-head" }, h("h1", {}, "Queue"), h("span", { class: "sub" }, "Jobs run one at a time on the GPU.")), list, empty);

  function card(job) {
    const c = { job, last: job.progress || null, abc: null, startedAt: job.started_at ? Date.parse(job.started_at) : null };
    const stageRow = h("div", { class: "stages" });
    const bar = h("div", { class: "progress" }, h("i"));
    const stats = h("div", { class: "stats" });
    const abcBox = h("div", { class: "abc-live score", hidden: true });
    const cancelBtn = h("button", { class: "sm danger", onclick: () => cancel(c, cancelBtn) }, "Cancel");
    const status = h("span", { class: "tag" }, job.status);
    c.el = h("div", { class: "card", dataset: { id: job.id } },
      h("div", { class: "row between" },
        h("div", { class: "row" }, h("span", { class: "title" }, jobTitle(job)), status, h("span", { class: "tag" }, job.kind),
          job.group_id ? h("span", { class: "tag accent", title: groupLabel(job.group_id) || "variation group" }, groupLabel(job.group_id) ? fmt.excerpt(groupLabel(job.group_id), 28) : "group") : null),
        cancelBtn),
      h("div", { class: "meta" }, h("span", {}, "preset ", h("b", {}, job.preset)), h("span", {}, "seed ", h("b", { class: "num" }, job.seed)), h("span", {}, "mode ", h("b", {}, job.params.cot || job.params.task || "—")),
        job.position !== null && job.status === "queued" ? h("span", { class: "pos" }, "position ", h("b", {}, job.position + 1)) : null),
      stageRow, bar, stats, abcBox);
    Object.assign(c, { stageRow, bar, stats, abcBox, status });
    paint(c);
    c.close = subscribe(job.id, {
      onProgress: (ev) => onEvent(c, ev),
      onDone: (j) => finish(c, j),
      onError: () => { /* poll fallback keeps state fresh */ },
    });
    return c;
  }

  function onEvent(c, ev) {
    if (ev.type === "status" && ev.status === "running" && !c.startedAt) c.startedAt = Date.now();
    if (ev.type === "stage" || ev.type === "token") c.last = ev.type === "stage" ? ev : { ...(c.last || {}), tps: ev.tps ?? (c.last && c.last.tps), completed: ev.tokens ?? (c.last && c.last.completed) };
    if ((ev.type === "status" || ev.type === "log") && ev.message) c.message = ev.message;
    if (ev.type === "abc" && ev.text) {
      c.abc = ev.text; c.abcBox.hidden = false;
      const final = ev.partial === false;
      renderScore(c.abcBox, ev.text, { staffwidth: 600 }, { immediate: final });
      c.abcBox.classList.toggle("final", final); c.abcBox.title = final ? "Final score" : "Score streaming in…";
    }
    paint(c);
  }

  function paint(c) {
    const { job, last } = c;
    const running = job.status === "running" || (last && last.stage);
    c.status.textContent = job.status === "queued" && !running ? "queued" : "running";
    c.status.className = "tag " + (running ? "accent" : "");
    const pos = c.el.querySelector(".pos"); if (pos) pos.hidden = !!running;
    const stages = STAGE_ORDER.filter((s) => s !== "transcribe" || job.kind === "cover");
    const idx = last && last.stage ? stages.indexOf(last.stage) : -1;
    fill(c.stageRow, stages.map((s, i) => h("span", { class: i < idx || (i === idx && last.status === "complete") ? "done" : i === idx ? "active" : "" }, STAGE_NAMES[s])));
    const elapsed = c.startedAt ? (Date.now() - c.startedAt) / 1000 : 0;
    const est = estimate(last, elapsed, job.kind);
    c.bar.classList.toggle("indeterminate", running && !(last && last.total));
    c.bar.firstChild.style.width = `${Math.round(est.fraction * 100)}%`;
    const parts = [];
    if (last && last.stage) parts.push(h("span", {}, h("b", {}, last.label || STAGE_NAMES[last.stage]), last.total ? ` ${last.completed ?? 0}/${last.total} ${last.unit || ""}` : last.completed ? ` ${last.completed} ${last.unit || ""}` : ""));
    else if (c.message) parts.push(h("span", {}, c.message));
    else parts.push(h("span", {}, job.status === "queued" ? "Waiting for the worker" : "Starting…"));
    if (last && last.tps) parts.push(h("span", {}, h("b", {}, fmt.tps(last.tps))));
    if (c.startedAt) parts.push(h("span", {}, "elapsed ", h("b", {}, fmt.secs(elapsed))));
    if (est.eta !== null && running) parts.push(h("span", {}, "ETA ", h("b", {}, "~" + fmt.secs(est.eta))));
    fill(c.stats, parts);
  }

  async function cancel(c, btn) {
    btn.disabled = true;
    try {
      const { status, data } = await api.cancel(c.job.id);
      if (status === 202) { btn.textContent = "Cancelling…"; toast("Cancel requested; the worker will stop at the next checkpoint", "info"); }
      else finish(c, data.job);
    } catch (e) { if (e.status === 409) { toast("Job already finished", "info"); refresh(); } else toastError(e); btn.disabled = false; }
  }

  function finish(c, job) {
    if (c.finished) return; // cancel response and SSE `done` can both arrive
    c.finished = true;
    c.close && c.close(); cards.delete(job.id); c.el.remove();
    if (job.status === "done") toast(`Done: ${jobTitle(job)}`, "ok", { link: { href: `#/song/${job.id}`, label: "Open" } });
    else if (job.status === "failed") toast(`Failed: ${jobTitle(job)} — ${job.error || "unknown error"}`, "err", { timeout: 12000 });
    else toast(`Cancelled: ${jobTitle(job)}`, "info");
    empty.hidden = cards.size > 0;
  }

  async function refresh() {
    try {
      const { jobs } = await api.jobs({ status: "queued,running", limit: 100 });
      const seen = new Set();
      for (const job of jobs.slice().reverse()) {
        seen.add(job.id);
        const c = cards.get(job.id);
        if (!c) { const n = card(job); cards.set(job.id, n); list.append(n.el); }
        else { c.job = job; if (job.started_at && !c.startedAt) c.startedAt = Date.parse(job.started_at); const pos = c.el.querySelector(".pos b"); if (pos && job.position !== null) pos.textContent = job.position + 1; }
      }
      for (const [id, c] of cards) if (!seen.has(id)) { c.close && c.close(); c.el.remove(); cards.delete(id); }
      empty.hidden = cards.size > 0;
    } catch (e) { if (e.code === "network_error") return; toastError(e); }
  }
  await refresh();
  const poll = setInterval(refresh, 5000), tick = setInterval(() => cards.forEach(paint), 1000);
  return { unmount() { clearInterval(poll); clearInterval(tick); cards.forEach((c) => c.close && c.close()); } };
}
