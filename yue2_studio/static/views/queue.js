import { api } from "../api.js";
import { fill, h, toastError } from "../ui.js";
import { liveCard } from "./jobcard.js";

export async function queueView({ el }) {
  const cards = new Map(); // id -> liveCard
  const list = h("div", { class: "stack" });
  const empty = h("div", { class: "empty" }, "Nothing queued. ", h("a", { href: "#/create" }, "Create a song"), " or ", h("a", { href: "#/cover" }, "make a cover"), ".");
  fill(el, h("div", { class: "view-head" }, h("h1", {}, "Queue"), h("span", { class: "sub" }, "Jobs run one at a time on the GPU.")), list, empty);

  function drop(c) { c.close(); cards.delete(c.job.id); c.el.remove(); empty.hidden = cards.size > 0; }

  async function refresh() {
    try {
      const { jobs } = await api.jobs({ status: "queued,running", limit: 100 });
      const seen = new Set();
      for (const job of jobs.slice().reverse()) {
        seen.add(job.id);
        const c = cards.get(job.id);
        if (!c) { const n = liveCard(job, { onFinish: drop, onGone: () => refresh() }); cards.set(job.id, n); list.append(n.el); }
        else c.update(job);
      }
      for (const [id, c] of cards) if (!seen.has(id)) drop(c);
      empty.hidden = cards.size > 0;
    } catch (e) { if (e.code === "network_error") return; toastError(e); }
  }
  await refresh();
  const poll = setInterval(refresh, 5000), tick = setInterval(() => cards.forEach((c) => c.paint()), 1000);
  return { unmount() { clearInterval(poll); clearInterval(tick); cards.forEach((c) => c.close()); } };
}
