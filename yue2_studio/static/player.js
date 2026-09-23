// Global player: one <audio> in a fixed bottom bar outside #view, so no route change or re-render ever
// touches it. Views never own a player — they render playButton() widgets carrying data-play-id, and the
// bar repaints every such button in the document on each state change (re-created buttons pick the state
// up without subscriptions that would leak across re-renders). The queue is [track] for a single song and
// [tracks...] for an album; prev/next and auto-advance only matter when it holds more than one.
import { songUrl } from "./api.js";
import { h, jobTitle, store, toast } from "./ui.js";

let queue = [], idx = -1, restoring = false; // restoring: src set from storage, never played yet
const listeners = new Set();

const audio = h("audio", { controls: true, preload: "metadata", "aria-label": "Now playing" });
const prevBtn = h("button", { type: "button", class: "icon ghost sm", title: "Previous track", "aria-label": "Previous track", onclick: () => player.prev() }, "⏮");
const nextBtn = h("button", { type: "button", class: "icon ghost sm", title: "Next track", "aria-label": "Next track", onclick: () => player.next() }, "⏭");
const nowTitle = h("span", { class: "title" }), nowSub = h("span", { class: "sub" });
const now = h("a", { class: "now", href: "#", title: "Open the song page" }, nowTitle, nowSub);
const pos = h("span", { class: "pos num", hidden: true });
const closeBtn = h("button", { type: "button", class: "icon ghost sm", title: "Stop and hide the player", "aria-label": "Close player", onclick: () => player.stop() }, "✕");
const bar = h("footer", { class: "player-bar", "aria-label": "Player" }, h("div", { class: "player-inner" }, prevBtn, now, nextBtn, pos, audio, closeBtn));

const current = () => queue[idx] || null;
const state = () => ({ track: current(), idx, queue: queue.slice(), playing: !!current() && !audio.paused });

/** Track descriptor for a done job; `sub` is a second line (the project track name on album plays). */
export function trackOf(job, sub = null) {
  return { id: job.id, title: jobTitle(job), sub, href: `#/song/${job.id}`, src: songUrl(job.id, "audio.flac") };
}

/** Point the element at `t` unless it already is (same id keeps the position for toggle/resume; the
 *  sub line may still differ when the same song is replayed from another context, so re-announce). */
function setTrack(t) {
  if (audio.dataset.id === t.id && audio.getAttribute("src") && !audio.error) { announce(t); return; }
  audio.dataset.id = t.id; audio.src = t.src;
  announce(t); persist();
}
// AbortError: the src was swapped while a play() was pending (fast next/prev). NotSupportedError: a
// missing/broken file — the `error` listener owns that toast.
const fail = (e) => { if (e && (e.name === "AbortError" || e.name === "NotSupportedError")) return; const t = current(); toast(`Could not play ${t ? t.title : "this song"}${e && e.message ? `: ${e.message}` : ""}`, "err"); };
function start() {
  const t = current(); if (!t) return;
  restoring = false;
  setTrack(t);
  audio.play().catch(fail);
  document.body.classList.add("has-player");
  paint();
}

export const player = {
  /** Play `tracks[i]` with the rest queued after it. The same track keeps its position. */
  play(tracks, i = 0) {
    if (!tracks || !tracks.length) return;
    queue = tracks.slice(); idx = Math.max(0, Math.min(i, queue.length - 1));
    start();
  },
  /** Pause/resume when `id` is the current track; false otherwise (callers then play() it). */
  toggle(id) {
    const t = current(); if (!t || t.id !== id) return false;
    if (audio.paused) player.resume(); else player.pause();
    return true;
  },
  pause() { audio.pause(); },
  /** Resume the current track; after a load error (or a forgotten id) reload it through start() instead. */
  resume() {
    const t = current(); if (!t) return;
    if (audio.dataset.id !== t.id || audio.error) { start(); return; }
    restoring = false; audio.play().catch(fail);
  },
  /** Stop, forget the queue and hide the bar (also drops the remembered state). */
  stop() {
    queue = []; idx = -1; restoring = false;
    audio.pause(); audio.removeAttribute("src"); delete audio.dataset.id; audio.load();
    document.body.classList.remove("has-player");
    store.remove("player");
    paint();
  },
  /** Advance; at the end of the queue stay on the last track (paused). */
  next() { if (idx + 1 < queue.length) { idx++; start(); } else paint(); },
  /** Back one track when near the start of this one, else restart it (the usual ⏮ behaviour). */
  prev() { if (idx > 0 && audio.currentTime < 3) { idx--; start(); } else { audio.currentTime = 0; paint(); } },
  /** Drop a deleted song from the queue; stops when it is the one playing. */
  remove(id) {
    const i = queue.findIndex((t) => t.id === id); if (i < 0) return;
    if (i === idx) { player.stop(); return; }
    queue.splice(i, 1); if (i < idx) idx--;
    persist(); paint();
  },
  /** A renamed song: update its queue entries (bar, remembered state, media session) and play buttons. */
  retitle(id, title) {
    for (const b of document.querySelectorAll("[data-play-id]")) if (b.dataset.playId === id) b.dataset.playTitle = title;
    const hits = queue.filter((t) => t.id === id); if (!hits.length) { paint(); return; }
    for (const t of hits) t.title = title;
    const t = current(); if (t && t.id === id) announce(t);
    persist(); paint();
  },
  current,
  isPlaying: (id) => { const t = current(); return !!t && t.id === id && !audio.paused; },
  /** Subscribe to state changes ({track, idx, queue, playing}); called once right away. Returns off(). */
  on(fn) { listeners.add(fn); try { fn(state()); } catch (e) { console.error(e); } return () => listeners.delete(fn); },
};

/** ▶/⏸ button bound to a job (disabled without audio). Painted on creation and on every state change. */
export function playButton(job, { sub = null, size = "sm", label = false } = {}) {
  const ok = job.status === "done" && !!(job.artifacts && job.artifacts.audio);
  const btn = h("button", { type: "button", class: `play${size ? " " + size : ""}`, dataset: { playId: job.id, playTitle: jobTitle(job), playLabel: label ? "1" : "" }, disabled: !ok, title: ok ? null : "No audio for this job",
    onclick: () => { if (!player.toggle(job.id)) player.play([trackOf(job, sub)]); } });
  paintButton(btn);
  return btn;
}
function paintButton(btn) {
  const on = player.isPlaying(btn.dataset.playId), label = btn.dataset.playLabel === "1";
  btn.textContent = on ? (label ? "⏸ Pause" : "⏸") : (label ? "▶ Play" : "▶");
  btn.classList.toggle("playing", on);
  btn.setAttribute("aria-label", `${on ? "Pause" : "Play"} ${btn.dataset.playTitle}`);
}

function paint() {
  const t = current(), s = state();
  nowTitle.textContent = t ? t.title : ""; nowSub.textContent = t && t.sub ? t.sub : ""; nowSub.hidden = !(t && t.sub);
  now.href = t ? t.href : "#";
  const multi = queue.length > 1;
  prevBtn.hidden = !multi; nextBtn.hidden = !multi; pos.hidden = !multi;
  nextBtn.disabled = idx >= queue.length - 1;
  pos.textContent = multi ? `${idx + 1}/${queue.length}` : "";
  for (const b of document.querySelectorAll("[data-play-id]")) paintButton(b);
  if ("mediaSession" in navigator) { try { navigator.mediaSession.playbackState = t ? (s.playing ? "playing" : "paused") : "none"; } catch { /* not supported */ } }
  for (const fn of listeners) { try { fn(s); } catch (e) { console.error(e); } }
}

function persist() {
  const t = current();
  if (t) store.set("player", { queue, idx, time: audio.currentTime || 0 }); else store.remove("player");
}

function announce(t) {
  if (!("mediaSession" in navigator)) return;
  try { navigator.mediaSession.metadata = new MediaMetadata({ title: t.title, artist: t.sub || "", album: "YuE2 Studio" }); } catch { /* not supported */ }
}

audio.addEventListener("play", () => { restoring = false; paint(); });
audio.addEventListener("pause", () => { persist(); paint(); });
audio.addEventListener("emptied", paint);
audio.addEventListener("ended", () => player.next());
audio.addEventListener("error", () => {
  const t = current(); if (!t || !audio.getAttribute("src")) return; // stop() empties the element; not an error
  if (restoring) { player.stop(); return; } // the remembered song is gone: hide quietly
  toast(`Could not play ${t.title}`, "err");
  // `paused` stays false after a load error: pause so buttons do not stick at ⏸, and forget the id so a
  // retry of this track reloads it instead of hitting setTrack's same-id short-circuit.
  audio.pause(); delete audio.dataset.id;
  player.next();
});
window.addEventListener("pagehide", persist);

if ("mediaSession" in navigator) {
  try {
    for (const [action, fn] of [["play", () => player.resume()], ["pause", () => player.pause()], ["previoustrack", () => player.prev()], ["nexttrack", () => player.next()]]) navigator.mediaSession.setActionHandler(action, fn);
  } catch { /* not supported */ }
}

/** Bring back the last queue/position from a previous session, paused (autoplay is blocked anyway). */
function restore() {
  const s = store.get("player", null);
  if (!s || !Array.isArray(s.queue) || !s.queue.length || !s.queue[s.idx]) return;
  queue = s.queue; idx = s.idx; restoring = true;
  const t = current();
  audio.dataset.id = t.id; audio.src = t.src;
  // Seek once metadata is in, unless the user already moved on (other track) or started playing from 0.
  audio.addEventListener("loadedmetadata", () => { if (audio.dataset.id === t.id && audio.currentTime === 0 && s.time > 0 && s.time < (audio.duration || Infinity)) audio.currentTime = s.time; }, { once: true });
  announce(t);
  document.body.classList.add("has-player");
  paint();
}

document.body.append(bar);
restore();
