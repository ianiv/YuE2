import { api, songUrl } from "../api.js";
import { confirmDialog, fill, fmt, groupLabel, h, jobTitle, loraLabel, modeLabel, store, toast, toastError } from "../ui.js";

/** Index of each job inside its group (API has no group endpoint). Order: the submit response's
 * job order if we remembered it, else Job.seq (submit order), then seed, then created_at. */
export function groupIndex(jobs) {
  const byGroup = {}, known = store.get("groups", {});
  for (const j of jobs) if (j.group_id) (byGroup[j.group_id] ||= []).push(j);
  const idx = {};
  const num = (v) => (typeof v === "number" ? v : Number.POSITIVE_INFINITY);
  for (const [gid, members] of Object.entries(byGroup)) {
    const ids = known[gid] && Array.isArray(known[gid].ids) ? known[gid].ids : [];
    const pos = (j) => { const i = ids.indexOf(j.id); return i < 0 ? Number.POSITIVE_INFINITY : i; };
    members.sort((a, b) => (pos(a) - pos(b)) || (num(a.seq) - num(b.seq)) || (a.seed - b.seed) || a.created_at.localeCompare(b.created_at));
    members.forEach((j, i) => { idx[j.id] = { n: i + 1, total: members.length, gid }; });
  }
  return idx;
}

export async function libraryView({ el, query }) {
  const filters = { kind: "", group: query.group || "", preset: "", q: "" };
  const f = {
    kind: h("select", { id: "l-kind", onchange: (e) => { filters.kind = e.target.value; load(); } }, [["", "All kinds"], ["create", "Create"], ["regenerate", "Regenerate"], ["cover", "Cover"], ["hum", "Hum"]].map(([v, t]) => h("option", { value: v }, t))),
    preset: h("select", { id: "l-preset", onchange: (e) => { filters.preset = e.target.value; paint(); } }, [["", "All presets"], ["quality", "Quality"], ["fast", "Fast"], ["custom", "Custom"]].map(([v, t]) => h("option", { value: v }, t))),
    group: h("select", { id: "l-group", onchange: (e) => { filters.group = e.target.value; load(); } }),
    q: h("input", { id: "l-q", type: "search", placeholder: "Search title, style, lyrics…", oninput: (e) => { filters.q = e.target.value.toLowerCase(); paint(); } }),
  };
  const grid = h("div", { class: "grid-cards" });
  const count = h("span", { class: "sub" });
  const failedList = h("div", { class: "stack" });
  const clearAllBtn = h("button", { class: "sm danger", onclick: clearAllFailed }, "Clear all");
  const failedBar = h("div", { class: "row between", hidden: true }, h("span", { class: "hint" }), clearAllBtn);
  const failed = h("details", {}, h("summary", {}, "Failed and cancelled"), failedBar, failedList);
  const uploadsPanel = uploadsSection();
  const PAGE = 60;
  const more = h("button", { hidden: true, onclick: () => load(true) }, "Load more");

  // Multi-select
  let selectMode = false;
  const selected = new Set();
  const selectBtn = h("button", { id: "l-select", "aria-pressed": "false", onclick: () => setSelectMode(!selectMode) }, "Select");
  const selCount = h("b", {}, "0 selected");
  const deleteSelBtn = h("button", { class: "danger", disabled: true, onclick: deleteSelected }, "Delete selected");
  const selBar = h("div", { class: "selbar", hidden: true, role: "toolbar", "aria-label": "Selection" },
    selCount,
    h("button", { class: "sm", onclick: () => { visibleIds().forEach((id) => selected.add(id)); paintSelection(); } }, "Select all (visible)"),
    h("button", { class: "sm", onclick: () => { selected.clear(); paintSelection(); } }, "Clear selection"),
    h("span", { class: "spacer" }), deleteSelBtn, h("button", { class: "ghost sm", onclick: () => setSelectMode(false) }, "Done"));
  fill(el,
    h("div", { class: "view-head" }, h("h1", {}, "Library"), count),
    h("div", { class: "row", style: "margin-bottom:14px" }, h("div", { style: "flex:1 1 200px" }, f.q), f.kind, f.preset, f.group, selectBtn),
    selBar,
    grid, h("div", { class: "row", style: "justify-content:center;margin-top:14px" }, more), h("div", { style: "height:20px" }), failed,
    h("div", { style: "height:20px" }), uploadsPanel.el);

  let jobs = [], gidx = {}, total = 0, knownGroups = new Set(), badJobs = [];
  const visibleIds = () => [...el.querySelectorAll("input.sel")].map((i) => i.dataset.id);
  function setSelectMode(on) {
    selectMode = on; selectBtn.setAttribute("aria-pressed", String(on)); selectBtn.textContent = on ? "Selecting…" : "Select";
    selBar.hidden = !on; if (!on) selected.clear();
    if (on && badJobs.length) failed.open = true;
    paint(); paintBad();
  }
  function paintSelection() {
    el.querySelectorAll("input.sel").forEach((i) => { i.checked = selected.has(i.dataset.id); i.closest(".card").classList.toggle("selected", i.checked); });
    selCount.textContent = `${selected.size} selected`; deleteSelBtn.disabled = selected.size === 0;
  }
  const selBox = (j) => selectMode ? h("input", { type: "checkbox", class: "sel", dataset: { id: j.id }, "aria-label": `Select ${jobTitle(j)}`, checked: selected.has(j.id),
    onchange: (e) => { if (e.target.checked) selected.add(j.id); else selected.delete(j.id); paintSelection(); } }) : null;
  const onKey = (e) => { if (e.key === "Escape" && selectMode) { e.preventDefault(); setSelectMode(false); } };
  document.addEventListener("keydown", onKey);

  /** Delete jobs with a small concurrency limit; returns {ok, failed}. */
  async function deleteMany(ids, limit = 4) {
    const queue = ids.slice(); let ok = 0, bad = 0;
    await Promise.all(Array.from({ length: Math.min(limit, queue.length) }, async () => {
      while (queue.length) { const id = queue.shift(); try { await api.remove(id); ok++; } catch (e) { bad++; console.warn("delete", id, e.message); } }
    }));
    return { ok, failed: bad };
  }
  async function deleteSelected() {
    const ids = [...selected]; if (!ids.length) return;
    if (!confirmDialog(`Delete ${ids.length} job${ids.length === 1 ? "" : "s"} and their files? This cannot be undone.`)) return;
    deleteSelBtn.disabled = true; deleteSelBtn.textContent = "Deleting…";
    const r = await deleteMany(ids);
    deleteSelBtn.textContent = "Delete selected";
    toast(r.failed ? `${r.ok} deleted, ${r.failed} failed` : `${r.ok} deleted`, r.failed ? "err" : "ok");
    setSelectMode(false); load();
  }
  async function clearAllFailed() {
    const ids = badJobs.map((j) => j.id); if (!ids.length) return;
    if (!confirmDialog(`Delete all ${ids.length} failed/cancelled job${ids.length === 1 ? "" : "s"}? This cannot be undone.`)) return;
    clearAllBtn.disabled = true;
    const r = await deleteMany(ids);
    clearAllBtn.disabled = false;
    toast(r.failed ? `${r.ok} deleted, ${r.failed} failed` : `Cleared ${r.ok} failed/cancelled job${r.ok === 1 ? "" : "s"}`, r.failed ? "err" : "ok");
    load();
  }
  function paintBad() {
    fill(failedList, badJobs.length ? badJobs.map(badRow) : h("p", { class: "muted small" }, "None."));
    failed.querySelector("summary").textContent = `Failed and cancelled (${badJobs.length})`;
    failedBar.hidden = badJobs.length === 0;
    failedBar.firstChild.textContent = `${badJobs.length} job${badJobs.length === 1 ? "" : "s"}`;
    paintSelection();
  }
  async function load(append = false) {
    more.disabled = true;
    try {
      const offset = append ? jobs.length : 0;
      const [done, bad] = await Promise.all([api.jobs({ status: "done", kind: filters.kind, group: filters.group, limit: PAGE, offset }), append ? null : api.jobs({ status: "failed,cancelled", limit: 100 })]);
      jobs = append ? jobs.concat(done.jobs) : done.jobs; total = done.total; gidx = groupIndex(jobs);
      if (!append) selected.clear(); // filter change / reload: selection no longer meaningful
      more.hidden = jobs.length >= total; more.disabled = false;
      more.textContent = `Load more (${jobs.length} of ${total})`;
      jobs.forEach((j) => j.group_id && knownGroups.add(j.group_id));
      const groups = [...knownGroups];
      fill(f.group, h("option", { value: "" }, "All groups"), groups.map((g) => h("option", { value: g, selected: g === filters.group }, groupLabel(g) ? fmt.excerpt(groupLabel(g), 30) : `group ${g.slice(0, 6)}`)));
      f.group.hidden = groups.length === 0;
      paint();
      if (bad) { badJobs = bad.jobs; paintBad(); }
      paintSelection();
    } catch (e) { toastError(e); }
    if (!append) uploadsPanel.load();
  }

  function paint() {
    const shown = jobs.filter((j) => (!filters.preset || j.preset === filters.preset)
      && (!filters.q || [jobTitle(j), j.params.style, j.params.lyrics].join("\n").toLowerCase().includes(filters.q)));
    count.textContent = jobs.length < total ? `${shown.length} shown of ${jobs.length} loaded (${total} total)` : `${shown.length} of ${total} songs`;
    fill(grid, shown.length ? shown.map(songCard) : h("div", { class: "empty", style: "grid-column:1/-1" }, jobs.length || filters.kind || filters.group ? "No songs match these filters." : ["No finished songs yet. ", h("a", { href: "#/create" }, "Create one")]));
    paintSelection();
  }

  function songCard(j) {
    const g = gidx[j.id];
    return h("div", { class: "card" + (selected.has(j.id) ? " selected" : "") },
      h("div", { class: "row between" },
        h("div", { class: "row nowrap", style: "min-width:0" }, selBox(j), h("a", { class: "title", href: `#/song/${j.id}` }, jobTitle(j))),
        h("div", { class: "row", style: "gap:4px" }, h("span", { class: "tag" }, j.kind), g ? h("a", { class: "tag accent", href: `#/library?group=${g.gid}`, title: "Show this group", onclick: (e) => { e.preventDefault(); filters.group = g.gid; f.group.value = g.gid; load(); } }, `var ${g.n}/${g.total}`) : null)),
      h("p", { class: "small muted" }, fmt.excerpt(j.params.style, 90)),
      h("audio", { controls: true, preload: "none", src: songUrl(j.id, "audio.flac") }),
      h("div", { class: "meta" },
        h("span", {}, h("b", { class: "num" }, fmt.dur(j.timing && j.timing.audio_seconds))),
        h("span", {}, "preset ", h("b", {}, j.preset)), j.loras && j.loras.length ? h("span", {}, "lora ", h("b", {}, loraLabel(j.loras))) : null, h("span", {}, "seed ", h("b", { class: "num" }, j.seed)),
        h("span", {}, "mode ", h("b", {}, modeLabel(j))), h("span", { title: j.created_at }, fmt.when(j.created_at)),
        j.truncated ? h("span", { class: "tag warn", title: j.truncated.reason }, "truncated") : null),
      h("div", { class: "row between" },
        h("div", { class: "row", style: "gap:4px" }, h("a", { class: "btn ghost sm", href: songUrl(j.id, "audio.flac"), download: true }, "FLAC"), h("a", { class: "btn ghost sm", href: songUrl(j.id, "audio.mp3"), download: true }, "MP3"), h("a", { class: "btn ghost sm", href: songUrl(j.id, "artifacts.zip") }, "ZIP")),
        h("button", { class: "ghost sm danger", onclick: () => remove(j) }, "Delete")));
  }

  function badRow(j) {
    return h("div", { class: "card" + (selected.has(j.id) ? " selected" : "") },
      h("div", { class: "row between" }, h("div", { class: "row" }, selBox(j), h("span", { class: "title" }, jobTitle(j)), h("span", { class: `tag ${j.status === "failed" ? "err" : ""}` }, j.status), h("span", { class: "tag" }, j.kind)),
        h("button", { class: "ghost sm danger", onclick: () => remove(j) }, "Delete")),
      j.error ? h("div", { class: "errbox mono" }, j.error) : null,
      h("div", { class: "meta" }, h("span", {}, "preset ", h("b", {}, j.preset)), j.loras && j.loras.length ? h("span", {}, "lora ", h("b", {}, loraLabel(j.loras))) : null, h("span", {}, "seed ", h("b", { class: "num" }, j.seed)), h("span", {}, fmt.when(j.finished_at || j.created_at))));
  }

  async function remove(j) {
    if (!confirmDialog(`Delete “${jobTitle(j)}” and its files? This cannot be undone.`)) return;
    try { await api.remove(j.id); toast("Deleted", "ok"); load(); } catch (e) { toastError(e); }
  }
  await load();
  return { unmount: () => document.removeEventListener("keydown", onKey) };
}

/**
 * Collapsible "Uploads" table (data/uploads/): checkbox multi-select + "Delete selected", and
 * "Clear unused" (prune everything no job references). Uploads a queued/running job still needs
 * cannot be selected; finished jobs keep their song without the upload.
 */
function uploadsSection() {
  let uploads = [];
  const selected = new Set();
  const list = h("div", { class: "table-wrap" });
  const info = h("span", { class: "hint" });
  const deleteBtn = h("button", { class: "sm danger", disabled: true, onclick: deleteSelected }, "Delete selected");
  const pruneBtn = h("button", { class: "sm", onclick: clearUnused }, "Clear unused");
  const bar = h("div", { class: "row between", hidden: true }, info, h("div", { class: "row", style: "gap:6px" }, deleteBtn, pruneBtn));
  const summary = h("summary", {}, "Uploads");
  const el = h("details", { id: "l-uploads" }, summary, h("p", { class: "hint" }, "Source audio for covers and hums. A finished song keeps playing after its upload is deleted; only queued or running jobs still need theirs."), bar, list);
  const mb = (n) => (n === null || n === undefined ? "—" : n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);
  const inUse = (u) => u.jobs.active > 0;
  const result = (r, verb) => toast(r.failed ? `${r.ok} ${verb}, ${r.failed} failed` : `${r.ok} ${verb}`, r.failed ? "err" : "ok");

  function paintSelection() {
    for (const id of [...selected]) if (!uploads.some((u) => u.upload_id === id && !inUse(u))) selected.delete(id);
    list.querySelectorAll("input.usel").forEach((i) => { i.checked = selected.has(i.dataset.id); });
    deleteBtn.disabled = selected.size === 0;
    deleteBtn.textContent = selected.size ? `Delete selected (${selected.size})` : "Delete selected";
    const all = list.querySelector("input.usel-all");
    if (all) { const free = uploads.filter((u) => !inUse(u)); all.checked = free.length > 0 && free.every((u) => selected.has(u.upload_id)); all.disabled = free.length === 0; }
  }
  function row(u) {
    const busy = inUse(u);
    const box = h("input", { type: "checkbox", class: "usel", dataset: { id: u.upload_id }, disabled: busy, title: busy ? "In use by a queued or running job" : "", "aria-label": `Select ${u.filename}`, checked: selected.has(u.upload_id),
      onchange: (e) => { if (e.target.checked) selected.add(u.upload_id); else selected.delete(u.upload_id); paintSelection(); } });
    return h("tr", { title: busy ? "In use by a queued or running job" : "" },
      h("td", {}, box),
      h("td", { style: "overflow-wrap:anywhere" }, h("span", { class: "mono small", title: u.upload_id }, u.filename), u.broken ? [" ", h("span", { class: "tag err", title: "Media file or sidecar is missing" }, "broken")] : null),
      h("td", { class: "num" }, fmt.dur(u.seconds)), h("td", { class: "num" }, mb(u.size)),
      h("td", { title: u.created_at }, fmt.when(u.created_at)),
      h("td", { class: "num" }, u.jobs.total, busy ? [" ", h("span", { class: "tag accent" }, `${u.jobs.active} active`)] : null));
  }
  function paint() {
    const n = uploads.length, bytes = uploads.reduce((a, u) => a + (u.size || 0), 0), unused = uploads.filter((u) => u.jobs.total === 0).length;
    summary.textContent = `Uploads (${n})`;
    bar.hidden = n === 0;
    info.textContent = n ? `${n} upload${n === 1 ? "" : "s"} · ${mb(bytes)} · ${unused} unused` : "";
    pruneBtn.disabled = unused === 0;
    fill(list, n ? h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, h("input", { type: "checkbox", class: "usel-all", "aria-label": "Select all deletable uploads", onchange: (e) => { uploads.filter((u) => !inUse(u)).forEach((u) => (e.target.checked ? selected.add(u.upload_id) : selected.delete(u.upload_id))); paintSelection(); } })),
        h("th", {}, "File"), h("th", { class: "num" }, "Length"), h("th", { class: "num" }, "Size"), h("th", {}, "Uploaded"), h("th", { class: "num" }, "Jobs"))),
      h("tbody", {}, uploads.map(row))) : h("p", { class: "muted small" }, "None."));
    paintSelection();
  }
  async function load() {
    try { uploads = (await api.listUploads()).uploads; } catch (e) { toastError(e); uploads = []; }
    paint();
  }
  async function deleteSelected() {
    const ids = [...selected]; if (!ids.length) return;
    if (!confirmDialog(`Delete ${ids.length} upload${ids.length === 1 ? "" : "s"}? Finished songs are kept; this cannot be undone.`)) return;
    deleteBtn.disabled = true; deleteBtn.textContent = "Deleting…";
    let ok = 0, failed = 0;
    for (const id of ids) { try { await api.deleteUpload(id); ok++; } catch (e) { failed++; console.warn("delete upload", id, e.message); } }
    selected.clear();
    result({ ok, failed }, "deleted");
    load();
  }
  async function clearUnused() {
    const unused = uploads.filter((u) => u.jobs.total === 0).length; if (!unused) return;
    if (!confirmDialog(`Delete all ${unused} upload${unused === 1 ? "" : "s"} no job references? This cannot be undone.`)) return;
    pruneBtn.disabled = true;
    try { const r = await api.pruneUploads({ unused: true }); toast(`Cleared ${r.deleted} unused upload${r.deleted === 1 ? "" : "s"}${r.skipped ? ` (${r.skipped} still referenced)` : ""}`, "ok"); }
    catch (e) { toastError(e); }
    load();
  }
  return { el, load };
}
