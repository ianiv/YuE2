import { albumUrl, api } from "../api.js";
import { confirmDialog, fill, fmt, h, inlineEdit, jobTitle, loraLabel, lyricsButton, modeLabel, qualityButton, store, takeControls, toast, toastError } from "../ui.js";
import { playButton, player, trackOf } from "../player.js";

const SORTS = [["added", "Added"], ["stars", "Stars"], ["thumbs", "Thumbs"]];
const FILTERS = [["all", "All"], ["up", "👍"], ["unrated", "Unrated"]];
const isLive = (j) => j.status === "queued" || j.status === "running";
const playable = (t) => !!(t.chosen && t.chosen.artifacts && t.chosen.artifacts.audio);

/**
 * #/project/<id> — editable name/description, "Play album" (the chosen takes queued in the global player), ZIP export,
 * draggable tracklist (↑/↓ fallback) and per-track takes with thumbs/stars/note, Choose and Detach.
 */
export async function projectView({ el, param, app }) {
  let project = null, unmounted = false;
  const prefs = { sort: "added", filter: "all", ...store.get("project.takes", {}) };
  const savePrefs = () => store.set("project.takes", { sort: prefs.sort, filter: prefs.filter });

  // -- header ----------------------------------------------------------------------------------
  const nameEl = inlineEdit("", (name) => patchProject({ name }), { tag: "h1", title: "Click to rename" });
  const descEl = inlineEdit("", (description) => patchProject({ description }), { tag: "p", cls: "muted", placeholder: "Add a description…", multiline: true, allowEmpty: true, title: "Click to edit the description" });
  const counts = h("span", { class: "sub" });
  const exportable = () => !!project && project.tracks.some(playable);
  const exportLink = (label) => h("a", { class: "btn sm", href: "#", download: true, onclick: (e) => { if (!exportable()) { e.preventDefault(); toast("Choose a finished take on at least one track first", "err"); } } }, label);
  const zipFlac = exportLink("Export ZIP (FLAC)");
  const zipMp3 = exportLink("Export ZIP (MP3)");
  const syncStatus = (s) => { zipMp3.hidden = !(s && s.ffmpeg); };
  const deleteBtn = h("button", { class: "ghost sm danger", onclick: () => project && removeProject() }, "Delete project");

  // -- album: the chosen takes, in tracklist order, queued in the global player ----------------------
  const playBtn = h("button", { class: "primary sm", id: "album-play", onclick: () => project && toggleAlbum() }, "▶ Play album");
  const nowLabel = h("span", { class: "hint" });
  const albumPlayer = h("div", { class: "album-player panel stack", style: "gap:8px" }, h("div", { class: "row between" }, h("div", { class: "row" }, playBtn, nowLabel), h("div", { class: "row", style: "gap:6px" }, zipFlac, zipMp3)));
  const albumTracks = () => project.tracks.filter(playable).map((t) => trackOf(t.chosen, t.name));
  let bar = { queue: [], playing: false }; // last player state (player.on)
  /** True only when the bar's queue IS this album (same chosen takes, tracklist order) — a chosen take
   *  played on its own from the Library is not the album: no ⏮/⏭, no advance, so no "Pause album". */
  const inAlbum = () => { const ids = project ? project.tracks.filter(playable).map((t) => t.chosen.id) : []; return ids.length > 0 && bar.queue.length === ids.length && bar.queue.every((q, i) => q.id === ids[i]); };
  /** The track whose chosen take the player currently holds (however it got there), or null. */
  const currentTrack = () => { const c = player.current(); return c && project ? project.tracks.find((t) => playable(t) && t.chosen.id === c.id) || null : null; };
  const playAt = (i) => player.play(albumTracks(), i);
  function toggleAlbum() {
    const c = player.current(), t = currentTrack();
    if (inAlbum() && c) player.toggle(c.id);
    else playAt(t ? project.tracks.filter(playable).indexOf(t) : 0); // same id keeps the position: a lone chosen take becomes the album from its track
  }
  function syncAlbum(state = null) {
    if (state) bar = state;
    if (unmounted) return;
    const album = inAlbum(), t = currentTrack(), playing = album && bar.playing;
    playBtn.textContent = playing ? "⏸ Pause album" : album ? "▶ Resume album" : "▶ Play album";
    playBtn.disabled = !project || !project.tracks.some(playable);
    const tracks = project ? project.tracks.filter(playable) : [];
    nowLabel.textContent = album && t ? `${tracks.indexOf(t) + 1}/${tracks.length} · ${t.name} — ${jobTitle(t.chosen)}` : tracks.length ? `${tracks.length} chosen take${tracks.length === 1 ? "" : "s"} in tracklist order` : "Choose a take on a track to build the album";
    list.querySelectorAll("li.track").forEach((li) => li.classList.toggle("playing", !!t && li.dataset.id === t.id));
  }

  // -- tracklist -------------------------------------------------------------------------------
  const list = h("ol", { class: "tracklist" });
  const newTrackIn = h("input", { id: "t-name", type: "text", placeholder: "New track name (e.g. Main theme)", maxlength: 200, required: true, autocomplete: "off" });
  const newTrackBtn = h("button", { type: "submit", class: "primary", id: "t-add" }, "Add track");
  const newTrackForm = h("form", { class: "row nowrap", onsubmit: addTrack }, newTrackIn, newTrackBtn);
  const notFound = h("div", { class: "empty" }, "This project does not exist (it may have been deleted). ", h("a", { href: "#/projects" }, "Back to projects"));

  const view = h("div", { class: "stack", style: "gap:0" },
    h("div", { class: "view-head" }, nameEl, counts, h("span", { class: "spacer" }), h("a", { class: "btn ghost sm", href: "#/projects" }, "← Projects"), deleteBtn),
    descEl,
    h("div", { style: "height:14px" }),
    albumPlayer,
    h("div", { style: "height:14px" }),
    list,
    h("div", { class: "panel", style: "margin-top:14px" }, newTrackForm));
  fill(el, view);
  const offPlayer = player.on(syncAlbum); // after `list` exists: on() paints right away

  function paint() {
    if (unmounted || !project) return;
    nameEl.set(project.name); descEl.set(project.description);
    document.title = `${project.name} · YuE2 Studio`;
    const n = project.tracks.length, m = project.tracks.filter((t) => t.chosen_job_id).length;
    counts.textContent = `${n} track${n === 1 ? "" : "s"} · ${m} chosen`;
    zipFlac.href = albumUrl(project.id, "flac"); zipMp3.href = albumUrl(project.id, "mp3");
    zipFlac.classList.toggle("disabled", !exportable()); zipMp3.classList.toggle("disabled", !exportable());
    fill(list, n ? project.tracks.map(trackRow) : h("li", { class: "empty" }, "No tracks yet. Add one below — each track collects its takes; pick one as the final take."));
    syncAlbum();
  }

  function trackRow(t, i) {
    const li = h("li", { class: "track", dataset: { id: t.id }, draggable: false });
    const handle = h("span", { class: "drag-handle", title: "Drag to reorder", "aria-hidden": "true", onmousedown: () => { li.draggable = true; }, onmouseup: () => { li.draggable = false; } }, "⠿");
    const num = playable(t)
      ? h("button", { class: "track-n ghost", title: "Play the album from this track", "aria-label": `Play from track ${i + 1}`, onclick: () => playAt(project.tracks.filter(playable).indexOf(t)) }, String(i + 1).padStart(2, "0"))
      : h("span", { class: "track-n" }, String(i + 1).padStart(2, "0"));
    const name = inlineEdit(t.name, (v) => patchTrack(t, { name: v }), { cls: "track-name", title: "Click to rename" });
    const chosenBox = t.chosen
      ? h("div", { class: "chosen stack", style: "gap:4px" },
        h("div", { class: "row small" }, playable(t) ? playButton(t.chosen, { sub: t.name }) : null, h("span", { class: "tag ok" }, "chosen"), h("a", { href: `#/song/${t.chosen.id}` }, jobTitle(t.chosen)), lyricsButton(t.chosen), h("span", { class: "muted num" }, fmt.dur(t.chosen.timing && t.chosen.timing.audio_seconds)),
          playable(t) ? null : h("span", { class: "hint" }, "Audio missing")))
      : h("div", { class: "chosen hint" }, t.takes.length ? "No take chosen yet — pick one below." : "No takes yet — make a new take or add songs from the Library.");
    const menu = h("details", { class: "menu" }, h("summary", { class: "btn sm" }, "New take ▾"),
      h("div", { class: "menu-list" },
        h("a", { href: `#/create?track=${t.id}` }, "Create…"), h("a", { href: `#/cover?track=${t.id}` }, "Cover…"), h("a", { href: `#/hum?track=${t.id}` }, "Hum…"),
        h("a", { href: `#/library?attach=${t.id}` }, "Add from Library…")));
    const up = h("button", { class: "icon ghost sm", title: "Move up", "aria-label": "Move up", disabled: i === 0, onclick: () => act(() => patchTrack(t, { position: i - 1 })) }, "↑");
    const down = h("button", { class: "icon ghost sm", title: "Move down", "aria-label": "Move down", disabled: i === project.tracks.length - 1, onclick: () => act(() => patchTrack(t, { position: i + 1 })) }, "↓");
    const del = h("button", { class: "ghost sm danger", onclick: () => removeTrack(t) }, "Delete track");
    li.append(
      h("div", { class: "track-head row nowrap" }, handle, num, h("div", { class: "track-title" }, name), h("span", { class: "spacer" }), menu, up, down, del),
      chosenBox,
      takesPanel(t));
    // Native drag and drop: the handle arms `draggable` so text inside inputs can still be selected.
    li.addEventListener("dragstart", (e) => { if (e.target !== li || !li.draggable) return; dragId = t.id; li.classList.add("dragging"); e.dataTransfer.effectAllowed = "move"; try { e.dataTransfer.setData("text/plain", t.id); } catch { /* Safari */ } });
    li.addEventListener("dragend", () => { dragId = null; li.draggable = false; li.classList.remove("dragging"); clearDrop(); });
    li.addEventListener("dragover", (e) => { if (!dragId || dragId === t.id) return; e.preventDefault(); e.dataTransfer.dropEffect = "move"; const r = li.getBoundingClientRect(); const after = e.clientY > r.top + r.height / 2; li.classList.toggle("drop-before", !after); li.classList.toggle("drop-after", after); });
    li.addEventListener("dragleave", () => li.classList.remove("drop-before", "drop-after"));
    li.addEventListener("drop", (e) => { if (!dragId || dragId === t.id) return; e.preventDefault(); const after = li.classList.contains("drop-after"); clearDrop(); reorder(dragId, t.id, after); });
    return li;
  }
  let dragId = null;
  const clearDrop = () => list.querySelectorAll(".drop-before, .drop-after").forEach((x) => x.classList.remove("drop-before", "drop-after"));
  async function reorder(fromId, toId, after) {
    const ids = project.tracks.map((t) => t.id).filter((id) => id !== fromId);
    const at = ids.indexOf(toId) + (after ? 1 : 0);
    ids.splice(at, 0, fromId);
    if (ids.join() === project.tracks.map((t) => t.id).join()) return;
    try { ({ project } = await api.orderTracks(project.id, ids)); paint(); } catch (e) { toastError(e); load(); }
  }

  // -- takes -----------------------------------------------------------------------------------
  function sortTakes(takes) {
    const list = takes.slice();
    const added = (a, b) => (a.take.added_at || "").localeCompare(b.take.added_at || "") || (a.seq - b.seq);
    if (prefs.sort === "stars") list.sort((a, b) => ((b.take.stars || 0) - (a.take.stars || 0)) || added(a, b));
    else if (prefs.sort === "thumbs") list.sort((a, b) => ((b.take.thumb || 0) - (a.take.thumb || 0)) || added(a, b));
    else list.sort(added);
    return list;
  }
  const passes = (j) => prefs.filter === "up" ? j.take.thumb === 1 : prefs.filter === "unrated" ? !j.take.thumb && !j.take.stars : true;
  function takesPanel(t) {
    const body = h("div", { class: "stack", style: "gap:6px" });
    const sortSel = h("select", { class: "sm", "aria-label": "Sort takes", onchange: (e) => { prefs.sort = e.target.value; savePrefs(); paint(); } }, SORTS.map(([v, l]) => h("option", { value: v, selected: v === prefs.sort }, `Sort: ${l}`)));
    const chips = h("div", { class: "chips", role: "group", "aria-label": "Filter takes" }, FILTERS.map(([v, l]) => h("button", { type: "button", class: "chip", "aria-pressed": String(v === prefs.filter), onclick: () => { prefs.filter = v; savePrefs(); paint(); } }, l)));
    const shown = sortTakes(t.takes).filter(passes);
    fill(body, shown.length ? shown.map((j) => takeRow(t, j)) : h("p", { class: "muted small" }, t.takes.length ? "No takes match this filter." : "None yet."));
    return h("details", { class: "takes", open: true },
      h("summary", {}, `Takes (${t.takes.length})`),
      t.takes.length ? h("div", { class: "row between", style: "margin-bottom:6px" }, chips, sortSel) : null,
      body);
  }
  function takeRow(t, j) {
    const live = isLive(j);
    const stage = live && j.progress && j.progress.stage ? ` · ${j.progress.label || j.progress.stage}` : "";
    const canChoose = j.status === "done" && j.artifacts && j.artifacts.audio;
    const chosen = t.chosen_job_id === j.id;
    const chooseBtn = chosen
      ? h("button", { class: "sm choose on", title: "Unchoose", "aria-pressed": "true", onclick: () => act(() => patchTrack(t, { chosen_job_id: null })) }, "★ Chosen")
      : h("button", { class: "sm choose", disabled: !canChoose, title: canChoose ? "Use this take on the album" : "Only a finished take with audio can be chosen", "aria-pressed": "false", onclick: () => act(() => patchTrack(t, { chosen_job_id: j.id })) }, "☆ Choose");
    return h("div", { class: "take-row" + (chosen ? " chosen" : "") + (live ? " live" : ""), dataset: { id: j.id } },
      h("div", { class: "row between" },
        h("div", { class: "row", style: "gap:6px;min-width:0" },
          canChoose ? playButton(j, { sub: t.name }) : null,
          h("span", { class: `tag ${j.status === "done" ? "ok" : j.status === "failed" ? "err" : live ? "accent" : ""}` }, j.status + stage),
          live ? h("span", { class: "title" }, jobTitle(j)) : h("a", { class: "title", href: `#/song/${j.id}` }, jobTitle(j)),
          lyricsButton(j), h("span", { class: "tag" }, j.kind)),
        h("div", { class: "row", style: "gap:4px" }, qualityButton(j, { onQueued: () => load({ quiet: true }) }), chooseBtn, h("button", { class: "ghost sm", title: "Remove from this track (keeps the song)", onclick: () => detach(t, j) }, "Detach"))),
      h("div", { class: "meta" },
        j.status === "done" ? h("span", {}, h("b", { class: "num" }, fmt.dur(j.timing && j.timing.audio_seconds))) : null,
        h("span", {}, "preset ", h("b", {}, j.preset)), h("span", {}, "seed ", h("b", { class: "num" }, j.seed)), h("span", {}, "mode ", h("b", {}, modeLabel(j))),
        j.loras && j.loras.length ? h("span", {}, "lora ", h("b", {}, loraLabel(j.loras))) : null,
        h("span", { title: j.take.added_at }, "added ", fmt.when(j.take.added_at)),
        j.error ? h("span", { class: "tag err", title: j.error }, "error") : null),
      takeControls(j, { onChange: (nj) => { const k = t.takes.findIndex((x) => x.id === nj.id); if (k >= 0) t.takes[k] = nj; } }));
  }

  // -- actions ---------------------------------------------------------------------------------
  async function load({ quiet = false } = {}) {
    let next;
    try { ({ project: next } = await api.project(param)); }
    catch (e) {
      if (unmounted) return;
      if (e.status === 404) { fill(el, notFound); project = null; return; }
      if (!project) { fill(el, h("div", { class: "errbox row between" }, `Could not load this project: ${e.message}`, h("button", { class: "sm", onclick: () => { fill(el, view); load(); } }, "Retry"))); return; }
      if (!quiet) toastError(e);
      return;
    }
    if (unmounted) return;
    project = next; paint();
  }
  async function patchProject(patch) { ({ project } = await api.patchProject(project.id, patch)); paint(); }
  async function patchTrack(t, patch) { await api.patchTrack(t.id, patch); await load(); }
  const act = (fn) => fn().catch((e) => { toastError(e); if (e.status === 409) load({ quiet: true }); });
  async function addTrack(e) {
    e.preventDefault();
    const name = newTrackIn.value.trim();
    if (!name) return toast("Give the track a name", "err");
    newTrackBtn.disabled = true;
    try { await api.addTrack(project.id, name); newTrackIn.value = ""; await load(); toast(`Track “${name}” added`, "ok", { timeout: 2500 }); }
    catch (err) { toastError(err); }
    newTrackBtn.disabled = false;
    newTrackIn.focus();
  }
  async function removeTrack(t) {
    if (!confirmDialog(`Delete track “${t.name}”? Its ${t.takes.length} take${t.takes.length === 1 ? "" : "s"} are detached; the songs are kept.`)) return;
    try { await api.removeTrack(t.id); await load(); toast("Track deleted", "ok", { timeout: 2500 }); } catch (e) { toastError(e); }
  }
  async function detach(t, j) {
    if (!confirmDialog(`Remove “${jobTitle(j)}” from “${t.name}”? The song is kept; its rating is dropped.`)) return;
    try { await api.detachTake(j.id); await load(); } catch (e) { toastError(e); }
  }
  async function removeProject() {
    if (!confirmDialog(`Delete project “${project.name}”? Its tracks are removed; the songs themselves are kept.`)) return;
    try { await api.removeProject(project.id); toast("Project deleted", "ok"); location.hash = "#/projects"; } catch (e) { toastError(e); }
  }

  await load(); // 404 → notFound; other errors → errbox with Retry (the listener below guards on `project`)
  syncStatus(app.status);
  // While a take is still queued/running, refresh with the 5 s status poll (but never under the user's
  // cursor). Playback lives in the global bar, so a reload never interrupts it.
  const onStatus = (s) => {
    syncStatus(s);
    if (!project || !project.tracks.some((t) => t.takes.some(isLive))) return;
    const a = document.activeElement;
    if (a && el.contains(a) && (a.tagName === "INPUT" || a.tagName === "TEXTAREA")) return;
    load({ quiet: true });
  };
  app.listeners.add(onStatus);
  return { unmount() { unmounted = true; app.listeners.delete(onStatus); offPlayer(); document.title = "YuE2 Studio"; } };
}
