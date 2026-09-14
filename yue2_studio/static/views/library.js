import { api, songUrl } from "../api.js";
import { confirmDialog, fill, fmt, groupLabel, h, jobTitle, store, toast, toastError } from "../ui.js";

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
    kind: h("select", { id: "l-kind", onchange: (e) => { filters.kind = e.target.value; paint(); } }, [["", "All kinds"], ["create", "Create"], ["regenerate", "Regenerate"], ["cover", "Cover"]].map(([v, t]) => h("option", { value: v }, t))),
    preset: h("select", { id: "l-preset", onchange: (e) => { filters.preset = e.target.value; paint(); } }, [["", "All presets"], ["quality", "Quality"], ["fast", "Fast"], ["custom", "Custom"]].map(([v, t]) => h("option", { value: v }, t))),
    group: h("select", { id: "l-group", onchange: (e) => { filters.group = e.target.value; paint(); } }),
    q: h("input", { id: "l-q", type: "search", placeholder: "Search title, style, lyrics…", oninput: (e) => { filters.q = e.target.value.toLowerCase(); paint(); } }),
  };
  const grid = h("div", { class: "grid-cards" });
  const count = h("span", { class: "sub" });
  const failedList = h("div", { class: "stack" });
  const failed = h("details", {}, h("summary", {}, "Failed and cancelled"), failedList);
  const PAGE = 60;
  const more = h("button", { hidden: true, onclick: () => load(true) }, "Load more");
  fill(el,
    h("div", { class: "view-head" }, h("h1", {}, "Library"), count),
    h("div", { class: "row", style: "margin-bottom:14px" }, h("div", { style: "flex:1 1 200px" }, f.q), f.kind, f.preset, f.group),
    grid, h("div", { class: "row", style: "justify-content:center;margin-top:14px" }, more), h("div", { style: "height:20px" }), failed);

  let jobs = [], gidx = {}, total = 0;
  async function load(append = false) {
    more.disabled = true;
    try {
      const offset = append ? jobs.length : 0;
      const [done, bad] = await Promise.all([api.jobs({ status: "done", limit: PAGE, offset }), append ? null : api.jobs({ status: "failed,cancelled", limit: 100 })]);
      jobs = append ? jobs.concat(done.jobs) : done.jobs; total = done.total; gidx = groupIndex(jobs);
      more.hidden = jobs.length >= total; more.disabled = false;
      more.textContent = `Load more (${jobs.length} of ${total})`;
      const groups = [...new Set(jobs.map((j) => j.group_id).filter(Boolean))];
      fill(f.group, h("option", { value: "" }, "All groups"), groups.map((g) => h("option", { value: g, selected: g === filters.group }, groupLabel(g) ? fmt.excerpt(groupLabel(g), 30) : `group ${g.slice(0, 6)}`)));
      f.group.hidden = groups.length === 0;
      paint();
      if (bad) {
        fill(failedList, bad.jobs.length ? bad.jobs.map(badRow) : h("p", { class: "muted small" }, "None."));
        failed.querySelector("summary").textContent = `Failed and cancelled (${bad.total})`;
      }
    } catch (e) { toastError(e); }
  }

  function paint() {
    const shown = jobs.filter((j) => (!filters.kind || j.kind === filters.kind) && (!filters.preset || j.preset === filters.preset) && (!filters.group || j.group_id === filters.group)
      && (!filters.q || [jobTitle(j), j.params.style, j.params.lyrics].join("\n").toLowerCase().includes(filters.q)));
    count.textContent = jobs.length < total ? `${shown.length} shown of ${jobs.length} loaded (${total} total)` : `${shown.length} of ${total} songs`;
    fill(grid, shown.length ? shown.map(songCard) : h("div", { class: "empty", style: "grid-column:1/-1" }, jobs.length ? "No songs match these filters." : ["No finished songs yet. ", h("a", { href: "#/create" }, "Create one")]));
  }

  function songCard(j) {
    const g = gidx[j.id];
    return h("div", { class: "card" },
      h("div", { class: "row between" },
        h("a", { class: "title", href: `#/song/${j.id}` }, jobTitle(j)),
        h("div", { class: "row", style: "gap:4px" }, h("span", { class: "tag" }, j.kind), g ? h("a", { class: "tag accent", href: `#/library?group=${g.gid}`, title: "Show this group", onclick: (e) => { e.preventDefault(); filters.group = g.gid; f.group.value = g.gid; paint(); } }, `var ${g.n}/${g.total}`) : null)),
      h("p", { class: "small muted" }, fmt.excerpt(j.params.style, 90)),
      h("audio", { controls: true, preload: "none", src: songUrl(j.id, "audio.flac") }),
      h("div", { class: "meta" },
        h("span", {}, h("b", { class: "num" }, fmt.dur(j.timing && j.timing.audio_seconds))),
        h("span", {}, "preset ", h("b", {}, j.preset)), h("span", {}, "seed ", h("b", { class: "num" }, j.seed)),
        h("span", {}, "mode ", h("b", {}, j.params.cot || j.params.task || "—")), h("span", { title: j.created_at }, fmt.when(j.created_at)),
        j.truncated ? h("span", { class: "tag warn", title: j.truncated.reason }, "truncated") : null),
      h("div", { class: "row between" },
        h("div", { class: "row", style: "gap:4px" }, h("a", { class: "btn ghost sm", href: songUrl(j.id, "audio.flac"), download: true }, "FLAC"), h("a", { class: "btn ghost sm", href: songUrl(j.id, "audio.mp3"), download: true }, "MP3"), h("a", { class: "btn ghost sm", href: songUrl(j.id, "artifacts.zip") }, "ZIP")),
        h("button", { class: "ghost sm danger", onclick: () => remove(j) }, "Delete")));
  }

  function badRow(j) {
    return h("div", { class: "card" },
      h("div", { class: "row between" }, h("div", { class: "row" }, h("span", { class: "title" }, jobTitle(j)), h("span", { class: `tag ${j.status === "failed" ? "err" : ""}` }, j.status), h("span", { class: "tag" }, j.kind)),
        h("button", { class: "ghost sm danger", onclick: () => remove(j) }, "Delete")),
      j.error ? h("div", { class: "errbox mono" }, j.error) : null,
      h("div", { class: "meta" }, h("span", {}, "preset ", h("b", {}, j.preset)), h("span", {}, "seed ", h("b", { class: "num" }, j.seed)), h("span", {}, fmt.when(j.finished_at || j.created_at))));
  }

  async function remove(j) {
    if (!confirmDialog(`Delete “${jobTitle(j)}” and its files? This cannot be undone.`)) return;
    try { await api.remove(j.id); toast("Deleted", "ok"); load(); } catch (e) { toastError(e); }
  }
  await load();
  return {};
}
