# Changelog

All notable changes to YuE2 Studio. Dates are when the work was merged to `main`.

## 2026-09-22 — Quality button only queues, remembered Takes sections

### Changed
- **Takes sections stay the way you left them.** A collapsed *Takes* section on the project page used
  to reopen on every repaint and reload; its open/closed state is now kept per track in
  `localStorage` (`yue2.project.takesClosed.<project id>`, pruned to the project's current tracks).
- **⇧ Quality only queues.** It no longer turns into a "✓ Quality version" / "Quality queued" link
  (which took you to the song page or the Queue) once a Quality re-render exists: every click queues
  one, a toast confirms, and you stay where you are. `ui.qualityVersion` is gone and the song page no
  longer fetches its track's takes.

## 2026-09-22 — One-click Quality re-render

### Added
- **⇧ Quality** on finished Fast-preset takes (project page take rows and the song page). It submits a
  `regenerate` from the song's own `score.abc` on the Quality preset, inheriting seed, style, lyrics,
  title and LoRAs, with `track_id` set so the result is a take in the same track. Once a Quality
  re-render exists (queued, running or done) the button becomes a link to it, so a second click
  cannot queue a duplicate. Offered for create / regenerate / cover songs with a score; not for hum
  songs (a regenerate drops the hum carrier) or `cot=off` songs. Frontend only (`ui.qualityButton`,
  `ui.qualityVersion`).

## 2026-09-22 — Lyrics popup on the project page

### Added
- **Lyrics popup on the project page.** Each take, and each track's chosen take, gets a small lyrics
  icon; hovering or focusing it shows the song's title and lyrics in a popover (flips above the icon
  near the bottom of the window, scrolls when long, fits phone widths). Clicking pins it open for
  touch screens; Esc, a click elsewhere or scrolling the page closes it. Reusable as
  `ui.lyricsButton(job)`.

## 2026-09-22 — Library generation time, song renaming

### Added
- **Rename a song.** Click the title on a song's page to edit it (Enter or clicking away saves, Esc
  cancels; clearing it falls back to the style excerpt). New `PATCH /api/jobs/{id} {"title"}` rewrites
  `params.title` and the song folder's `job.json`; the player bar and play buttons pick up the new name.
- **Generation time on library cards.** Each finished song shows "made in …" (the job's end-to-end
  `timing.e2e`); hovering it shows the per-stage split (transcribe / plan / semantic / synthesize /
  decode) and the realtime factor. Frontend only: every finished job already records these timings.

## 2026-09-22 — mlx-Yue perf fork

### Changed
- **mlx-Yue now comes from the `perf` branch of the
  [ianiv/mlx-Yue](https://github.com/ianiv/mlx-Yue/tree/perf) fork** (commit `dd80c47`, upstream
  `9253ed1` plus speed work). Exact mode reproduces `9253ed1` bit for bit (same ABC tokens, semantic
  tokens and latents): the AR decode loop is software-pipelined (`async_eval` of the next token before
  reading the current one; repetition window kept on device), RMSNorm tail / RoPE / SwiGLU are fused
  with the same rounding points, RoPE uses shared full-context tables, and acoustic attention is no
  longer tiled into 256-query calls (`query_chunk_size=None`; tiling was memory-only). On an M5 Max this
  is 94 → 118 tok/s ABC, 97 → 123 tok/s semantic (bf16), 118 → 172 tok/s semantic (8-bit): the Quality
  preset goes 173.6 → 147.0 s and Fast 88.6 → 69.6 s for `examples/full-song.json`.

### Added
- **Fast numerics** (`settings.fast_numerics`, default on; Settings → Generation). Passes
  `fast_numerics=True` to mlx-Yue: native BF16 acoustic attention instead of FP32-promoted attention
  (~2.5× faster synthesis on M5) and both CFG branches in one weight pass per token (~1.25× faster
  CFG / `cot=off` semantic generation). Numerically equivalent, not bit-identical: without CFG a seed
  gives the same take (latent cosine 0.9995, audio SNR ≈ 26 dB), with CFG a different one. Quality
  preset 173.6 → 93.8 s, Fast 88.6 → 53.3 s, Quality with CFG 2.5 191.7 → 99.8 s. Applied per job
  without rebuilding the pipeline (hum synthesis included) and recorded as `fast_numerics` in
  `summary.json` and the song's `config.json`. `EngineOptions` / `resolve_preset` gain
  `fast_numerics`; `scripts/smoke.py` gains `--exact-numerics`.

## 2026-09-21 — mlx-Yue 9253ed1

### Changed
- **mlx-Yue pinned at `9253ed1`** (was `ab0f058`). Upstream's runtime gate now accepts
  **macOS ≥ 14.2 on M1–M4** and only requires **≥ 26.2 on M5** (detected via `mx.device_info()`),
  and rejects Intel / Rosetta Python outright; the README requirements table and
  the `scripts/setup.py` doctor report (which now reports mlx-Yue's `runtime` status instead of its own
  `macos_version >= 26.2` check) follow suit. Upstream also bumps `transformers` 5.0.0 → 5.10.4
  and `pretty-midi` → 0.2.11.post0 for the transcription (cover / hum) path.
- **PyPI refresh**: `uvicorn` 0.53, `watchfiles` 1.3, `huggingface-hub` 1.32, `filelock` 4.0 and
  other transitive minors.

### Removed
- **`compat/lyra-yue2/`**. Upstream fixed its `importlib.metadata.version("lyra-yue2")` lookups to
  read `mlx-yue`, so the metadata-only alias distribution is no longer needed.

## 2026-09-19 — Claude assist

### Added
- **Ask Claude** (backend). `POST /api/assist {prompt, page, context?}` fills the Create / Cover /
  Hum form from a description: Claude answers under a fixed system prompt
  (`yue2_studio/assist_prompt.md`, the `yue2-prompt` skill adapted for the app — style-tag order,
  section-tagged lyrics, mode/CFG guidance, per-page rules) and a JSON schema, so the reply is
  always `{fields, notes, provider, model, seconds}`; `fields` are stripped, length-checked and
  limited to `title`/`style`/`lyrics` on cover/hum. With `context` (the *Refine* toggle) the
  current form is sent and Claude returns only the fields that change. `POST /api/assist/test` is
  the Settings "Test" button. New module `yue2_studio/assist.py`.
- **Two providers.** The `claude` CLI (Claude Code's login; `claude -p --json-schema …` in a
  throwaway cwd, never `--bare`) or the Messages API with an API key (stdlib `urllib`, forced
  `tool_use`, default model `claude-sonnet-5`). `Settings` gains `assist_provider`
  (`auto|cli|api|off`; `auto` prefers the CLI), `assist_model` and a write-only
  `anthropic_api_key` (absent = unchanged, `""` = cleared; stored in plain text in `data/app.db`,
  never returned — responses carry `has_api_key` for the stored key). `GET /api/status` gains
  `assist {provider, cli, api_key, model, reasons}`, where `api_key` also honours the
  `ANTHROPIC_API_KEY` env fallback. The CLI runs with no MCP servers, skills or hooks and without
  `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` in its environment, so it bills the login.
- **Ask Claude UI.** An "Ask Claude" box at the top of the Create, Cover and Hum forms: describe
  the song, ⌘/Ctrl+Enter or *Ask Claude*, and the fields fill in (Create: title, style, lyrics,
  mode, CFG; Cover/Hum: title, style, lyrics) with a result line naming what changed, Claude's
  one-line tip and Undo/Redo. *Refine what's already in the form* (remembered) sends the current
  fields so follow-ups edit instead of rewrite. The box shows a `CLI`/`API` tag, is disabled with
  a hint when no provider can run, and is hidden entirely when the provider is `off`. Settings
  gains a *Claude assist* section (provider, model, API key with Clear, Test — which saves first —
  and a status line); status refreshes immediately after a save. `ui.js` gains `assistBox`.
- Logging: the `yue2_studio.assist` logger reports each step at INFO (request page/prompt
  length/context keys/resolved provider, "asking claude via cli|api …" right before the blocking
  call, CLI cost and token usage, "claude answered via … in Ns: fields=[…]") and failures at
  WARNING; the prompt text only at DEBUG, the API key never.
- Errors: 503 `assist_unavailable` (reasons joined with `; `) when no provider can run, 502
  `assist_failed` with a readable message (not logged in, key rejected, rate limited, timeout,
  unusable output). `scripts/mock_api.py` serves canned answers per page.

## 2026-09-18 — Projects, global player, header nav

### Added
- **Projects, tracks and takes** (backend). A project is an ordered tracklist of named tracks; jobs
  are attached to a track as *takes* (a job belongs to at most one track), rated with 👍/👎, 1–5
  stars and a note, and one finished take per track is *chosen*. Every `Job` now carries a
  nullable `take` (`{track_id, project_id, track_name, project_name, thumb, stars, note, added_at,
  chosen}`, one LEFT JOIN, no extra calls) and `GET /api/jobs` filters by `track=` / `project=`.
  `POST /api/jobs` accepts a top-level `track_id` that attaches every created job (all variations
  members) — 404 before any row is written when the track is unknown. New tables `projects`,
  `tracks`, `takes` are added to existing databases on open; deleting a job clears its take and
  any track that had chosen it, deleting a track/project only detaches (songs are untouched).
- **Projects API.** `GET/POST /api/projects`, `GET/PATCH/DELETE /api/projects/{id}`,
  `POST /api/projects/{id}/tracks`, `PUT /api/projects/{id}/order`, `GET /api/tracks/{id}`,
  `PATCH/DELETE /api/tracks/{id}` (choose = 409 unless a `done` take of that track),
  `POST /api/tracks/{id}/takes` (409 when the job is a take elsewhere unless `move: true`, which
  keeps its rating), `DELETE/PATCH /api/takes/{job_id}`. `jobs.Conflict` → 409.
- **Album export.** `GET /api/projects/{id}/album.zip?format=flac|mp3` builds
  `<name>/NN Track.flac|mp3` from the chosen takes plus `tracklist.json` / `tracklist.md` (missing
  tracks listed with `missing: true`), fresh per request into a temp file removed after sending;
  409 when nothing is exportable, 503 for MP3 without ffmpeg. New module `yue2_studio/projects.py`.
- **Projects UI.** New *Projects* nav entry: `#/projects` (create/delete, `N tracks · M chosen`)
  and `#/project/{id}` — inline-editable name/description, an album player that plays the chosen
  takes in order, Export ZIP (FLAC; MP3 when ffmpeg is present), a draggable tracklist (↑/↓
  fallback), per-track takes with 👍/👎, 1–5 stars, note, sort/filter (remembered), Choose and
  Detach, and a "New take" menu (Create / Cover / Hum with a "New take for Project › Track" banner
  and `track_id` on submit, or "Add from Library…"). Library multi-select gains "Add to project…"
  (a job already in another track asks before moving) and `?project=` / `?attach=`; song cards,
  job cards and the song page show a `Project › Track` tag, and the song page has a *Project*
  panel (attach, rate, choose, detach) — regenerate/variations from a take stay in its track.
  `api.js` gains the projects/tracks/takes client and `albumUrl()`; `ui.js` gains `projectTag`,
  `takeControls`, `projectPicker`, `inlineEdit` and `trackBanner`.
- `scripts/mock_api.py` mirrors the projects routes in memory; `docs/API.md` §2–§5 document the
  `Project` / `Track` / `Take` models, endpoints, `#/projects` + `#/project/{id}` routing and the
  "Album from takes" flow.

### Changed
- **Header nav reorganised.** `Generate ▾ · Projects · Library · Settings`, then `Queue` (with its
  badge) and the engine pill right-aligned. Generate is a `<details>` dropdown (Create / Cover /
  Hum) that works without JS, highlights when one of its pages is open, and closes on navigation,
  click outside or Escape (`installMenuAutoClose()` in `ui.js`, which the project page's "New take"
  menus now share).
- **Playback survives navigation and re-renders.** One `<audio>` lives in a player bar fixed to the
  bottom of the page (`yue2_studio/static/player.js`), outside the routed view, so switching tabs,
  typing in the Library search, a result card finishing or the project page's 5 s reload no longer
  stop the song. Cards, take rows and the song page render ▶/⏸ play buttons that reflect the bar's
  state; pressing play in a list plays just that song, **Play album** (and a track number) queues
  the chosen takes with ⏮/⏭ and auto-advance. The bar has the native seek/time/volume controls, a
  link to the song, Media Session metadata/keys, and ✕ to stop; the last song and position are
  remembered in `localStorage` and restored paused on reload. Routing no longer stops playback; the
  MIDI score preview and the hum recording preview stay page-local (and still pause the bar, and
  vice versa).

### Removed
- Per-card `<audio controls>` elements in the Create results, Library, song page and project page
  (chosen take, take rows and the album player).

## 2026-09-17 — LoRA adapters, hum to song, uploads management

### Added
- **LoRA adapters.** Published YuE2 LoRAs in `models/loras/` (single `.safetensors` in the
  Mothersuperior layout or PEFT directories) are listed by `GET /api/loras` and can be stacked
  per job (`loras: [{name, scale}]`) on Create, Cover, Hum and Regenerate. Adapters are merged
  into the resident AR/NAR weights (FP32 opmath, bf16 storage; 8-bit weights are dequantised,
  merged and requantised), full-weight `vae2llm`/`llm2vae` replacements are honoured, and the
  stack is recorded in every song's `config.json`/`result.json`. `scripts/smoke.py --lora`,
  `examples/instrumental-lora.json`.
- **Hum to song** (`#/hum`, job kind `hum`). Record in the browser or drop a file; the hum is
  transcribed (SheetSage2, melody-vocal) and the AR *continues* its open score into a full song
  (`melody: continue | hum_only | ignore`). With a hum adapter (`hum_adapter_v1_combined`,
  `scripts/setup.py --with-hum`) the acoustic decoder is additionally conditioned on the hum's
  prosody: `librosa.pyin` pitch track → sine carrier → MLX Oobleck VAE encoder → `hum_proj`
  injections in the NAR with classifier-free "hum influence" (0–3) and a start offset. Song pages
  show the hummed open score and the hum analysis. New modules `hum.py`, `hum_nar.py`,
  `vae_encoder.py`; uploads accept `.webm`/`.mp4` recordings.
- **Uploads management.** `GET /api/uploads` (with per-upload job counts and an `unused` filter),
  `DELETE /api/uploads/{id}` (409 while a queued/running job uses it) and
  `POST /api/uploads/prune`. Cover and Hum pages get a "Recent uploads" picker so a file is reused
  instead of re-uploaded; the Library gets an Uploads panel with multi-select delete and
  "Clear unused"; Settings gets "Auto-delete unused uploads after N days" (off by default; runs at
  startup and after each job).
- Create page hint when an instrumental AR adapter is in the stack.

### Changed
- `/static/*` is served with `Cache-Control: no-cache`, so browsers revalidate ES modules after a
  server restart (previously stale UI after upgrades).
- The instrumental LoRA example and README tip use untimed section tags / `[instrumental]`:
  timed tags (`[verse 0:15-0:45]`) tend to make `ar_lora_inst_v3abc` overrun to the 6-minute
  length cap; songs land around 3–5 minutes regardless of the plan.
- `POST /api/upload` rejects empty (0-byte) files and files ffprobe cannot read with a 400 instead
  of accepting them and failing later in the worker.

### Fixed
- Score player: abcjs no longer adds its own chord-symbol accompaniment, so the Play button plays
  the notated Vocal/Ins voices of *this* score rather than the same strummed preset for every song.
- Transcription failures caused by undecodable audio surface as
  `ffmpeg could not decode <file>: …` instead of a raw `CalledProcessError`.

## 2026-09-14 — UX round

- Only one audio source plays at a time across the app.
- Create results: collapsible live score per card, collapsed by default.
- Library: multi-select delete and Clear all for failed/cancelled jobs.
- Random-seed toggle on Create, Cover and Regenerate, on by default.
- Create keeps the form and shows inline results; song detail shows lyrics by default.
- MIT licence for the studio code.

## 2026-09-13 / 14 — Initial studio

- uv project, config, `StudioPipeline` wrapper around mlx-Yue, setup and smoke scripts.
- SQLite job store, worker thread + SSE event bus, FastAPI routes, FakeEngine, entrypoint.
- Vanilla-JS SPA (create, queue, library, song, cover, settings) with abcjs; dev mock API.
- Full test suite and README with measured M5 Max timings.
