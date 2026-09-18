import { api } from "../api.js";
import { confirmDialog, fill, fmt, h, toast, toastError } from "../ui.js";

/** #/projects — list of projects (name, tracks · chosen, updated) + a "New project" form. */
export async function projectsView({ el }) {
  const nameIn = h("input", { id: "p-name", type: "text", placeholder: "Project name (e.g. Soundtrack)", maxlength: 200, required: true, autocomplete: "off" });
  const createBtn = h("button", { type: "submit", class: "primary", id: "p-create" }, "New project");
  const form = h("form", { class: "row nowrap", onsubmit: create }, nameIn, createBtn);
  const grid = h("div", { class: "grid-cards" });
  fill(el,
    h("div", { class: "view-head" }, h("h1", {}, "Projects"), h("span", { class: "sub" }, "Albums and soundtracks: named tracks, several takes per track, one chosen take each.")),
    h("div", { class: "panel", style: "margin-bottom:14px" }, form),
    grid);

  async function load() {
    try { const { projects } = await api.projects(); paint(projects); } catch (e) { toastError(e); }
  }
  function paint(projects) {
    fill(grid, projects.length ? projects.map(card) : h("div", { class: "empty", style: "grid-column:1/-1" }, "No projects yet. Name one above, then add tracks and takes."));
  }
  function card(p) {
    const n = p.track_count || 0, m = p.chosen_count || 0;
    return h("div", { class: "card", dataset: { id: p.id } },
      h("div", { class: "row between" }, h("a", { class: "title", href: `#/project/${p.id}` }, p.name),
        h("button", { class: "ghost sm danger", onclick: () => remove(p) }, "Delete")),
      p.description ? h("p", { class: "small muted" }, fmt.excerpt(p.description, 120)) : null,
      h("div", { class: "meta" },
        h("span", {}, h("b", { class: "num" }, n), ` track${n === 1 ? "" : "s"}`),
        h("span", {}, h("b", { class: "num" }, m), " chosen"),
        h("span", { title: p.updated_at }, "updated ", fmt.when(p.updated_at))));
  }
  async function create(e) {
    e.preventDefault();
    const name = nameIn.value.trim();
    if (!name) return toast("Give the project a name", "err");
    createBtn.disabled = true;
    try { const { project } = await api.createProject({ name }); location.hash = `#/project/${project.id}`; }
    catch (err) { toastError(err); createBtn.disabled = false; }
  }
  async function remove(p) {
    if (!confirmDialog(`Delete project “${p.name}”? Its tracks are removed; the songs themselves are kept.`)) return;
    try { await api.removeProject(p.id); toast("Project deleted", "ok"); load(); } catch (e) { toastError(e); }
  }
  await load();
  return {};
}
