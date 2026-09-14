// Router, header status, theme.
import { api } from "./api.js";
import { applyTheme, clear, fill, h, store } from "./ui.js";
import { createView } from "./views/create.js";
import { queueView } from "./views/queue.js";
import { libraryView } from "./views/library.js";
import { songView } from "./views/song.js";
import { coverView } from "./views/cover.js";
import { settingsView } from "./views/settings.js";

export const app = { status: null, settings: null, listeners: new Set() };

async function refreshStatus() {
  const pill = document.getElementById("engine-pill"), count = document.getElementById("queue-count");
  try {
    app.status = await api.status();
    const e = app.status.engine, q = app.status.queue;
    pill.dataset.state = e.state;
    pill.querySelector(".txt").textContent = e.state + (e.precision ? ` · ${e.precision}` : "") + (e.memory_gib ? ` · ${e.memory_gib.toFixed(1)} GiB` : "");
    pill.title = `Engine ${e.state}` + (q.running ? `, running ${q.running.slice(0, 8)}` : "");
    const n = (q.queued || 0) + (q.running ? 1 : 0);
    count.textContent = String(n); count.hidden = n === 0;
  } catch (e) {
    pill.dataset.state = "offline"; pill.querySelector(".txt").textContent = "offline"; app.status = null;
  }
  for (const fn of app.listeners) { try { fn(app.status); } catch (err) { console.error(err); } }
}

const routes = {
  create: createView, queue: queueView, library: libraryView, song: songView, cover: coverView, settings: settingsView,
};
let current = null, routeGen = 0;

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "") || "create";
  const [path, qs] = raw.split("?");
  const [name, ...rest] = path.split("/");
  return { name: routes[name] ? name : "create", param: rest.join("/") || null, query: Object.fromEntries(new URLSearchParams(qs || "")) };
}

async function route() {
  const { name, param, query } = parseHash();
  const gen = ++routeGen;
  if (current && current.unmount) { try { current.unmount(); } catch (e) { console.error(e); } }
  current = null;
  document.querySelectorAll("nav.main a").forEach((a) => {
    const active = a.getAttribute("href") === `#/${name}` || (name === "song" && a.getAttribute("href") === "#/library");
    if (active) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
  const view = clear(document.getElementById("view"));
  view.append(h("p", { class: "muted" }, "Loading…"));
  try {
    const v = await routes[name]({ el: view, param, query, app });
    if (gen !== routeGen) { v && v.unmount && v.unmount(); return; } // hash moved on while loading
    current = v;
  } catch (e) { if (gen !== routeGen) return; console.error(e); fill(view, h("div", { class: "errbox" }, `This view failed to load: ${e.message}`)); }
  window.scrollTo(0, 0);
}

window.addEventListener("hashchange", route);
if (!location.hash) location.replace("#/create");
// Theme precedence: explicit local choice > server setting > system.
api.settings().then((s) => { app.settings = s; if (store.get("theme", null) === null && s.theme) applyTheme(s.theme, { persist: false }); }).catch(() => {});
refreshStatus();
setInterval(refreshStatus, 5000);
route();
