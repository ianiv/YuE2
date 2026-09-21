# YuE2 Studio — local music-generation app for Apple Silicon (M5 Max)

## Context

The user wants an application that generates music with [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) and makes full use of an M5 Max (48 GB). The repo `/Users/ianiv/Code/YuE2` is empty (git initialised, no commits).

Research findings that shape the design:

- YuE2-3B is a 4B-param AR–NAR Mixture-of-Transformers: (1) AR writes an ABC score, (2) AR writes semantic codec tokens (25 frames/s, 1920 samples/frame @ 48 kHz), (3) 32-step midpoint flow-matching (NAR) produces 64-dim acoustic latents, (4) an Oobleck VAE decodes to 48 kHz stereo. The official `yue2_infer` wheel targets PyTorch + CUDA graphs; it has MPS fallbacks but runs eager and slow on Mac.
- **A native MLX port already exists and is active**: [vanch007/mlx-Yue](https://github.com/vanch007/mlx-Yue) (Apache-2.0, last commit `ab0f058`, 2026-09-14). Torch-free, 8-bit AR quantisation (~83 tok/s on M3 Max), Metal Steel SDPA, 8-step "fast" mode (<1.0 RTF), all 5 generation modes, and an MLX SheetSage2/MERT transcription + cover pipeline. It vendors the upstream `yue2` protocol package. Requirements: macOS ≥ 26.2 (user has 27.0), Python 3.12 (`uv` will fetch it), `mlx==0.32.2`, `MLX_ENABLE_TF32=0`. Pre-converted weights: [vanch007/mlx-Yue2-3B](https://huggingface.co/vanch007/mlx-Yue2-3B) (ar-8bit 2.7 GB, ar-bf16 4.3 GB, nar-bf16 2.9 GB) + [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) (0.5 GB). Cover needs [m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2) (0.2 GB) + [m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) (2.5 GB) + ffmpeg (installed at /opt/homebrew/bin/ffmpeg).
- mlx-Yue is **CLI + Python API only**: no GUI, no server, no queue, no library. That is the gap this app fills.
- Engine constraints (from `src/lyra/measure.py` / `pipeline.py`): one GPU workload per process (file lock + `threading.RLock`, nested same-thread guards OK); `YuE2Pipeline` holds a `GPUExecution` guard for its lifetime; memory watchdog with a configurable budget (default 16 GiB, peak use ≈ 11 GiB); `transcribe()` needs no guard of its own. `precision="8bit"` still loads the BF16 AR for NAR conditioning and swaps models between stages. Progress is reported through `pipe._status(label, total=, unit=)` (a `yue2.progress._Stage` with `advance()/update()/finish()/set_total()`) and per-token `on_token(phase, token)` callbacks; every stage accepts `cancelled()`.

Decisions confirmed with the user: **local web app**, **built on mlx-Yue**, v1 scope = **Create + Edit-score-&-regenerate + Cover + Batch/variations**.

## Architecture

```
browser (vanilla HTML/JS, abcjs for score rendering)
   │  REST + Server-Sent Events
FastAPI app (uvicorn, single process)
   │  in-process queue
Worker thread (owns the one resident YuE2Pipeline / GPU guard)
   │
mlx-Yue (lyra) → MLX on Metal          artifacts → data/songs/<job_id>/
SQLite (data/app.db) for jobs, groups, metadata
```

Single uvicorn process, one dedicated worker thread that loads the pipeline once and processes jobs serially (mlx-Yue forbids concurrency). API handlers never touch MLX; they enqueue jobs and stream events.

## Repo layout

```
YuE2/
├── pyproject.toml              # uv project, python 3.12, deps below
├── README.md
├── .gitignore                  # data/, models/, .venv/
├── scripts/setup.py            # download weights (hf_hub snapshot_download), verify, print doctor report
├── yue2_studio/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app factory + `yue2-studio` entrypoint (uvicorn)
│   ├── config.py               # paths, presets, env (MLX_ENABLE_TF32=0 set before importing mlx)
│   ├── engine.py               # StudioPipeline(lyra.YuE2Pipeline) subclass + lifecycle
│   ├── worker.py               # queue, worker thread, cancellation, event bus
│   ├── jobs.py                 # job types (create / regenerate / cover / variations), SQLite models
│   ├── api.py                  # routes
│   ├── audio.py                # ffmpeg mp3/wav export, waveform peaks
│   └── static/                 # index.html, app.js, styles.css
├── examples/                   # 2–3 request JSONs (reuse mlx-Yue's quickstart/full-song)
└── tests/                      # unit tests for jobs/queue/api with a fake engine
```

## Dependencies (`pyproject.toml`)

- `mlx-yue @ git+https://github.com/vanch007/mlx-Yue@9253ed1` with `[transcription]` extra — pinned commit; it pulls `mlx==0.32.2`, `mlx-lm`, `transformers`, `tiktoken`, `soundfile`, `psutil`, and the vendored `yue2` package.
- `fastapi`, `uvicorn[standard]`, `sse-starlette`, `python-multipart` (uploads), `huggingface-hub`.
- `requires-python = ">=3.12,<3.13"` (mlx-Yue constraint). `uv sync` fetches CPython 3.12.
- Dev: `pytest`, `httpx`.

## Step 1 — Project scaffold + setup script

- `pyproject.toml`, `.gitignore`, `README.md`.
- `scripts/setup.py`: `snapshot_download("vanch007/mlx-Yue2-3B", local_dir="models/converted")`, `snapshot_download("m-a-p/YuE2-Vae", local_dir="models/vae")`; `--with-cover` adds `m-a-p/SheetSage2` and `m-a-p/MERT-v2-FullSong` into `models/hf-cache`. Then runs `lyra.commands.doctor`-equivalent (`verify_conversion(models/converted)`, `model_identity(models/vae)`) and prints a summary. Check ffmpeg on PATH.
- Verify: `uv run python scripts/setup.py` completes; `uv run mlx-yue doctor --model models/converted --vae models/vae` returns `"status": "pass"`.

## Step 2 — Engine wrapper (`engine.py`)

- `config.py` sets `os.environ.setdefault("MLX_ENABLE_TF32", "0")` before any `mlx` import (the guard raises otherwise).
- `class StudioPipeline(lyra.pipeline.YuE2Pipeline)`: override `_status(label, *, total=None, unit=None)` to yield a stage object with the same `advance/update/set_total/finish` interface that publishes `{"stage": label, "completed", "total", "unit", "status"}` events to a callback instead of stderr (construct with `progress=False` so the upstream stderr reporter stays quiet). Also pass an `on_token` that publishes token counts + live decoded ABC text during planning (`self.tokenizer.decode` of accumulated ids, throttled to ~4 Hz) so the score appears as it is written.
- **Presets** (map to real pipeline options):
  - `quality`: `precision="bf16"`, `GenerationConfig(ode_steps=32)`.
  - `fast`: `precision="8bit"`, `GenerationConfig(ode_steps=8)`.
  - `custom`: user picks precision ∈ {bf16, 8bit, 4bit} and ode_steps 4–64.
  - Note: `precision` is fixed at pipeline construction. The engine keeps one pipeline; when a job requests a different precision it calls `pipe.close()` and rebuilds (mlx-Yue's `_load_model` already swaps AR/NAR residency within a precision). `ode_steps` is per-job: assign `pipe.generation_config = GenerationConfig(ode_steps=n)` before the call (as `tools/run_fast_8step_benchmark.py` does).
  - `memory_budget_gib` default 24 on this 48 GB machine (guard requires ≤ total−4); `require_ac` off by default, exposed in settings.
- Stage functions used by jobs: `pipe.plan(request=...)`, `pipe.generate_semantic(plan)`, `lyra.pipeline.initial_noise(len(tokens), seed)`, `pipe.synthesize(semantic, noise=noise, cancelled=...)`, `pipe.decode(latents, cancelled=...)`, then build `lyra.pipeline.SongResult(...)` and `save_artifacts(dir)` — mirror `YuE2Pipeline.__call__` so timing/identity are correct and `result.json`/`plan.json`/`score.abc`/`audio.flac`/`latent.npy`/`noise.npy` land in the song directory. Running the stages ourselves lets us persist the plan after stage 1 and expose stage-level cancellation.
- Cover: call `lyra.transcription.pipeline.transcribe(audio_path, out_dir/"transcription", task=..., offline=True, cache_dir="models/hf-cache", cancelled=..., progress=cb)` from the same worker thread (RLock permits nesting; `transcribe` takes no guard), `gc.collect(); mx.clear_cache()`, then run the create flow with `abc=<score.abc>` and `cot="melody"` (or `"full"` when task=`full`) — this is exactly `lyra.commands.cover`.

## Step 3 — Jobs, queue, worker (`jobs.py`, `worker.py`)

- SQLite tables: `jobs(id, kind, status[queued|running|done|failed|cancelled], group_id, parent_id, params_json, preset, seed, created_at, started_at, finished_at, error, timing_json, audio_seconds, truncated_json)`, `groups(id, label, created_at)`, `settings(key, value)`.
- Job kinds and params:
  - `create`: style, lyrics, cot ∈ {full, melody, off}, seed, cfg_scale?, abc? (supplied score), preset.
  - `regenerate`: parent_id + edited `abc` + optional new style/lyrics/seed → `create` with `cot` inherited, `abc` supplied (upstream requires nonempty ABC with cot≠off).
  - `cover`: uploaded audio path, style, lyrics, task ∈ {melody-full, melody-vocal, full}, seed, preset.
  - `variations`: expands server-side into N `create` jobs sharing a `group_id` with seeds `seed, seed+1, …` (or random), interleaved nothing else — serial execution.
- Worker: `threading.Thread` consuming a `queue.Queue`; a per-job `threading.Event` backs `cancelled()`; cancel-while-queued just marks the row. Event bus: per-job `asyncio.Queue` fan-out via `loop.call_soon_threadsafe`; last event cached per job for late subscribers; stage history persisted to `jobs.timing_json` on completion.
- Startup: worker loads the pipeline lazily on the first job (so the server starts instantly); `/api/status` reports `engine: cold|loading|ready|busy`, precision, memory footprint (`psutil`), current job.
- Retention: songs live in `data/songs/<job_id>/`; delete removes row + dir.

## Step 4 — HTTP API (`api.py`)

- `GET  /api/status` — engine/queue state, presets, model paths, cover availability (transcription models present + ffmpeg).
- `POST /api/jobs` — body `{kind, params}`; returns job(s). `variations` returns the group.
- `GET  /api/jobs?status=&group=` / `GET /api/jobs/{id}` / `DELETE /api/jobs/{id}` / `POST /api/jobs/{id}/cancel`.
- `GET  /api/jobs/{id}/events` — SSE stream of progress events.
- `GET  /api/songs/{id}/audio.flac` (FileResponse, Range supported) and `/audio.mp3` (lazy ffmpeg transcode into the song dir), `/score.abc`, `/artifacts.zip`.
- `POST /api/upload` — multipart audio for covers → `data/uploads/`.
- `GET/PUT /api/settings` — default preset, memory budget, require_ac.
- Serve `static/` at `/`.

## Step 5 — Web UI (`static/`)

Vanilla HTML/CSS/JS, no build step. abcjs from cdnjs for sheet rendering + built-in MIDI preview of the plan. Views:

1. **Create** — style textarea (with a few genre chips), lyrics textarea (section tags `[verse]`/`[chorus]` helper), mode (full/melody/off), preset (Quality / Fast / Custom), seed (+ random), CFG, optional ABC paste/upload, "Variations: N" field → submits `variations` when N>1.
2. **Queue** — live cards for queued/running jobs: stage name, progress bar, tokens/s, ETA (from token rate and typical stage ratios), the ABC score streaming in during planning, cancel button.
3. **Library** — grid/list of finished songs with inline `<audio>` player (FLAC plays natively in Safari/Chrome), duration, preset, seed, mode, group badge; filters; delete; "download FLAC/MP3/artifacts".
4. **Song detail** — player, request (style/lyrics), timing breakdown per stage, rendered score (abcjs) with an editable ABC textarea and **"Regenerate from this score"** (keeps seed/lyrics; style editable) → `regenerate` job; "More variations" → `variations` with this request.
5. **Cover** — drop an audio file, pick task (melody-full default, as recommended for covers), style, lyrics, seed, preset → `cover` job; detail view shows the transcription artifacts (score/MIDI) alongside the result.
6. Settings drawer — preset defaults, memory budget, require-AC.

## Step 6 — Entrypoint, docs, tests

- `yue2-studio` console script → `uvicorn yue2_studio.main:app --host 127.0.0.1 --port 8765`; `--open` launches the browser.
- README: setup (`uv sync`, `uv run python scripts/setup.py --with-cover`, `uv run yue2-studio`), presets and expected timings, licence note (weights are CC BY-NC 4.0, non-commercial).
- Tests: `jobs`/`worker` with a `FakeEngine` that emits scripted stage events; API tests with `httpx.AsyncClient` (create job → SSE events → done → audio route). No GPU in tests.

## Verification

1. `uv sync` succeeds on Python 3.12; `uv run python -c "import lyra, mlx.core as mx; print(mx.metal.is_available())"` → `True`.
2. `uv run python scripts/setup.py --with-cover` downloads ~13 GB and the doctor check passes.
3. Smoke generation through the engine wrapper using mlx-Yue's `examples/quickstart.json` in `fast` preset → `data/songs/<id>/audio.flac` exists and plays; `result.json` has `"status": "complete"`.
4. Full song (`examples/full-song.json`) in both presets; record `timing` (AR tok/s, NAR seconds, VAE seconds, e2e) in the README as measured on this M5 Max.
5. UI end-to-end: start server, create a song, watch SSE progress and the streaming score, play from Library, edit the ABC and regenerate, run 3 variations, cancel a running job mid-NAR (worker returns to idle, job marked cancelled), upload an MP3 and run a cover.
6. `uv run pytest` green.

## Execution — orchestrated agent team (`/orchestrate`)

The controller (this session) only dispatches, validates output contracts, and gates; agents write all code. The workflow spec below is committed to `.orchestrate/workflow.md` as the first task.

**Work items** (4, sequential except 2‖3), each built in its own git worktree/branch and merged to `main` in order with one commit per item:

| # | Item | Depends on | Produces |
|---|------|-----------|----------|
| 0 | plan-commit | — | `docs/PLAN.md` (copy of this plan), `.orchestrate/workflow.md`, `.gitignore` — first commit |
| 1 | scaffold-engine | 0 | `pyproject.toml`, `scripts/setup.py`, `yue2_studio/{__init__,config,engine}.py`, `examples/`. Gate: `uv sync` OK, `setup.py --with-cover` OK, real smoke generation of quickstart in `fast` preset via `StudioPipeline` produces `audio.flac` + `result.json status=complete`; timings recorded in the report |
| 2 | backend | 1 | `yue2_studio/{jobs,worker,api,audio,main}.py`, `yue2-studio` entrypoint; unit-testable with a FakeEngine |
| 3 | frontend | 1 (API contract from Step 4; runs in parallel with 2) | `yue2_studio/static/{index.html,app.js,styles.css}`; 400 px + dark mode |
| 4 | tests-docs | 2, 3 | `tests/` (FakeEngine, httpx API tests), `README.md` with measured timings + CC BY-NC note |

**Per-item stages**: build (builder, `general-purpose`, in worktree) → review (reviewer, read-only diff against this plan; `request-changes` is binding) → verify (verifier runs the item's gate command in the worktree) → merge to `main` (controller, fast-forward/`--no-ff`, after both verdicts pass).

**Quality gate**: `uv run ruff check . && uv run pytest` (ruff in dev deps); builders run it before reporting. Item 1 additionally runs the smoke generation; item 4's verifier runs full pytest on merged `main`.

**Final gate** (controller, after item 4 merges): start `uv run yue2-studio`, drive the UI with Playwright MCP: create (fast), watch progress, play, edit ABC → regenerate, 3 variations, cancel a running job, upload MP3 → cover. Report per-stage timings on the M5 Max and anything incomplete.

**Rules for every builder**: never modify mlx-Yue (wrap/subclass instead and note it); `MLX_ENABLE_TF32=0` must be set before `mlx` is imported; only one GPU workload per process — the worker thread is the sole owner of the pipeline; Python `>=3.12,<3.13` via uv; the `data/` and `models/` dirs are gitignored and shared across worktrees via absolute path in config (`YUE2_STUDIO_HOME`, default = repo root) so agents in worktrees reuse the ~13 GB of downloaded weights.
