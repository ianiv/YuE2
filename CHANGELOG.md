# Changelog

All notable changes to YuE2 Studio. Dates are when the work was merged to `main`.

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
