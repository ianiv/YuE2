# Workflow: yue2-studio-build
<!-- Drive docs/PLAN.md to a working YuE2 Studio app with a team of agents. -->

## Work items
- Source: `docs/PLAN.md` — section "Execution — orchestrated agent team" (items 1–4)
- Scoping: n/a — all items, in dependency order (1 → 2‖3 → 4)
- Item identity: one row of the work-item table + its "Produces" column as acceptance criteria

## Session-level stages (run once, in order)
1. plan-commit — role: controller; produces: docs/PLAN.md, .orchestrate/workflow.md, .gitignore on main; gate: auto
2. scaffold-engine — role: builder; produces: item 1 (env, weights, engine wrapper, real smoke generation); contract: files touched + `uv sync` result + smoke result (song dir path, result.json status, per-stage timings); gate: reviewer approve + verifier pass → merge to main
(per-item work for items 2–4 starts after stage 2 merges)

## Per-item stages (run per item)
| # | Stage | Role | Input | Output contract | Parallel across items? | Gate |
|---|-------|------|-------|-----------------|------------------------|------|
| 1 | build | builder | docs/PLAN.md sections for the item + builder-conduct | table of files touched, gate command + pass/fail, one-paragraph summary, surprises | yes (2‖3) | quality gate passes |
| 2 | review | reviewer | `git diff main...<branch>` + docs/PLAN.md | verdict approve/request-changes + numbered issues with file:line | yes | reviewer approve (binding) |
| 3 | verify | verifier | worktree path + acceptance criteria | pass/fail + evidence (command output summaries, artifact paths) | yes | verifier pass |
| 4 | merge | controller | branch | commit sha on main | no | — |

## Roles → agents
| Role | Bind to (subagent type) | Read-only? |
|------|-------------------------|-----------|
| builder | general-purpose | no |
| reviewer | general-purpose (instructed read-only) | yes |
| verifier | general-purpose (runs commands, no source edits) | yes |

## Quality gate (per-item, must pass before deliver)
- Command: `uv run ruff check . && uv run pytest`
- Who runs it: builder, before reporting complete; verifier re-runs it

## Deliver
- Action: merge branch into main with `git merge --no-ff`, one commit per item (plus the agent's own commits)
- Isolation: git worktree per item under `.worktrees/<item>` on branch `item/<item>`; default branch (never push): main (local-only repo, no remote)
- Evidence: gate output summary; for item 1 the smoke-generation artifacts; final gate = controller-driven Playwright UI run
- Link/transition source item: n/a

## Preconditions to verify in preflight
- docs/PLAN.md present; uv + ffmpeg on PATH; macOS ≥ 26.2; network for HF downloads
