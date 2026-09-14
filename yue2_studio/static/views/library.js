import { api, songUrl } from "../api.js";
import { confirmDialog, fill, fmt, h, jobTitle, store, toast, toastError } from "../ui.js";

/** Index of each job inside its group, by created_at ascending (API has no group endpoint). */
export function groupIndex(jobs) {
  const byGroup = {};
  for (const j of jobs) if (j.group_id) (byGroup[j.group_id] ||= []).push(j);
  const idx = {};
  for (const [gid, members] of Object.entries(byGroup)) {
    members.sort((a, b) => a.created_at.localeCompare(b.created_at));
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
  fill(el,
    h("div", { class: "view-head" }, h("h1", {}, "Library"), count),
    h("div", { class: "row", style: "margin-bottom:14px" }, h("div", { style: "flex:1 1 200px" }, f.q), f.kind, f.preset, f.group),
    grid, h("div", { style: "height:20px" }), failed);

  let jobs = [], gidx = {};
  async function load() {
    try {
      const [done, bad] = await Promise.all([api.jobs({ status: "done", limit: 500 }), api.jobs({ status: "failed,cancelled", limit: 100 })]);
      jobs = done.jobs; gidx = groupIndex(jobs);
      const labels = store.get("groups", {});
      const groups = [...new Set(jobs.map((j) => j.group_id).filter(Boolean))];
      fill(f.group, h("option", { value: "" }, "All groups"), groups.map((g) => h("option", { value: g, selected: g === filters.group }, labels[g] ? fmt.excerpt(labels[g], 30) : `group ${g.slice(0, 6)}`)));
      f.group.hidden = groups.length === 0;
      paint();
      fill(failedList, bad.jobs.length ? bad.jobs.map(badRow) : h("p", { class: "muted small" }, "None."));
      failed.querySelector("summary").textContent = `Failed and cancelled (${bad.total})`;
    } catch (e) { toastError(e); }
  }

  function paint() {
    const shown = jobs.filter((j) => (!filters.kind || j.kind === filters.kind) && (!filters.preset || j.preset === filters.preset) && (!filters.group || j.group_id === filters.group)
      && (!filters.q || [jobTitle(j), j.params.style, j.params.lyrics].join("\n").toLowerCase().includes(filters.q)));
    count.textContent = `${shown.length} of ${jobs.length} songs`;
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
