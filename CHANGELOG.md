# Changelog

All notable changes to YuE2 Studio. Dates are when the work was merged to `main`.

## Unreleased — Global player

### Changed
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

## Unreleased — Projects

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
