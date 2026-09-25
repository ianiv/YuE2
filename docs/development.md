# Development

```bash
uv run pytest                      # ~380 tests, under 15 s, no models or GPU needed
uv run ruff check .                # lint (E, F, W, I, UP, B; line length 110)
uv run yue2-studio --fake          # the real server with the fake engine
uv run python scripts/mock_api.py  # a standalone in-memory mock of docs/API.md on :8790, for UI work
```

The tests drive the real FastAPI app through `httpx` with the `FakeEngine` (`yue2_studio/fake.py`). It
implements the engine protocol the worker uses and emits the same raw event shapes as the MLX engine, so the
UI and API can be worked on without models or a GPU. `tests/test_imports.py` asserts that none of the HTTP-side
modules import `mlx`. `tests/test_engine_lifecycle.py` imports the real engine module (skipped where MLX is
unavailable) but injects a fake pipeline, so nothing touches the GPU. The VAE encoder tests use `models/vae`
when it is present.

## Architecture

```
browser (vanilla HTML/JS, abcjs for score rendering)
   │  REST + Server-Sent Events
FastAPI app (uvicorn, single process)
   │  in-process queue
Worker thread (owns the one resident YuE2Pipeline / GPU guard)
   │
mlx-Yue (lyra) → MLX on Metal          artifacts → data/songs/<job_id>/
SQLite (data/app.db) for jobs, groups, projects, settings
```

One uvicorn process. Request handlers only read and write the SQLite job store, enqueue job ids and stream
events. A single worker thread owns the resident `StudioPipeline` (a subclass of mlx-Yue's `YuE2Pipeline` whose
progress hook publishes events instead of printing) and runs jobs strictly one after another. Raw engine events
are normalised into the HTTP `ProgressEvent` shape and fanned out to SSE subscribers through an in-process event
bus; the last event per job is cached so late subscribers see the current state. mlx-Yue itself is never
modified.

Artifacts land in `data/songs/<job_id>/`: `plan/` right after planning, `song/` with `audio.flac` at the end,
`summary.json`, `transcription/` for covers and hums, `hum/` for hums.

## Code layout

| Path | Contents |
|---|---|
| `yue2_studio/config.py` | Paths, presets; sets `MLX_ENABLE_TF32=0` before anything imports MLX. |
| `yue2_studio/engine.py` | The pipeline wrapper. |
| `yue2_studio/lora.py` | Adapter discovery and weight merging. |
| `yue2_studio/hum.py` | Hum options, open-score trimming, carrier analysis. |
| `yue2_studio/hum_nar.py` | The hum-conditioned acoustic sampler (a `CachedNAR` subclass). |
| `yue2_studio/vae_encoder.py` | MLX port of the VAE encoder. |
| `yue2_studio/jobs.py` | Validation and the SQLite store (jobs, groups, projects, tracks, takes, settings). |
| `yue2_studio/worker.py` | Worker thread, cancellation, event bus, event normalisation. |
| `yue2_studio/projects.py` | Album export: the tracklist of chosen takes and its ZIP. |
| `yue2_studio/uploads.py` | Upload bookkeeping for `data/uploads/`: listing, deleting, pruning unused uploads. |
| `yue2_studio/api.py` | HTTP routes. |
| `yue2_studio/audio.py` | ffmpeg, uploads, ZIP files. |
| `yue2_studio/assist.py`, `assist_prompt.md` | Ask Claude: CLI and API providers, JSON schema, system prompt. |
| `yue2_studio/main.py` | App factory and the `yue2-studio` command. |
| `yue2_studio/fake.py` | The scripted fake engine. |
| `yue2_studio/static/` | The UI: `index.html`, `app.js` (router), `ui.js`, `player.js`, `api.js`, `views/`. |
| `scripts/setup.py` | Weight download and doctor report. |
| `scripts/smoke.py`, `scripts/hum_smoke.py` | One generation through the engine, with timings. |
| `docs/API.md` | The HTTP contract. |
| `docs/PLAN.md` | The original design. |

Only `engine.py`, `hum.py`, `hum_nar.py` and `vae_encoder.py` import `mlx`.

## Updating the screenshots

The screenshots in `docs/images/` were taken from a server running the fake engine on a separate
`YUE2_STUDIO_HOME` seeded with original demo songs, at a 1440 × 900 window (390 × 844 for the phone views),
then scaled to 1600 px wide and palette-compressed:

```bash
ffmpeg -i shot.png -vf "split[a][b];[a]palettegen=max_colors=256:reserve_transparent=0[p];[b][p]paletteuse=dither=sierra2_4a" -pred mixed out.png
```
