# YuE2 Studio

A local web studio for generating full songs with [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
on Apple Silicon. It wraps [mlx-Yue](https://github.com/vanch007/mlx-Yue) (the native MLX port of
the YuE2 pipeline: ABC score planning → semantic tokens → flow-matching acoustics → 48 kHz stereo
VAE decode) in a single-process FastAPI server with a job queue, live progress over Server-Sent
Events, a song library, score editing with regeneration, seed variations and audio-to-song covers.
Everything runs on your Mac; nothing leaves it except the optional soundfont download for the
score preview (see below).

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon Mac. Peak use is ~11 GiB of unified memory; the default memory budget is 24 GiB, so **48 GB is recommended** (a 24–32 GB machine works with the budget lowered in Settings, see [Troubleshooting](#troubleshooting)). Timings below are from an M5 Max / 48 GB. |
| macOS | **26.2 or newer** (mlx-Yue's guarded Metal runtime refuses older versions). |
| Python | 3.12 (fetched automatically by `uv`; the project is pinned to `>=3.12,<3.13`). |
| Tools | [`uv`](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH` (`brew install uv ffmpeg`). ffmpeg is needed for MP3 export, upload probing and covers. |
| Disk | ~13 GB of weights: generator (ar-8bit 2.7 GB, ar-bf16 4.3 GB, nar-bf16 2.9 GB) + VAE 0.5 GB; covers add SheetSage2 0.2 GB + MERT-v2-FullSong 2.5 GB. |

## Setup

```bash
uv sync                                          # creates .venv with Python 3.12 + mlx-Yue @ ab0f058
uv run python scripts/setup.py --with-cover      # downloads weights into models/ and prints a doctor report
uv run yue2-studio --open                        # http://127.0.0.1:8765
```

`scripts/setup.py` downloads [vanch007/mlx-Yue2-3B](https://huggingface.co/vanch007/mlx-Yue2-3B)
(pre-converted MLX weights) and [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) into
`models/converted` and `models/vae`; `--with-cover` also fetches
[m-a-p/SheetSage2](https://huggingface.co/m-a-p/SheetSage2) and
[m-a-p/MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong) into `models/hf-cache`.
It then verifies the weights (hashes; `--no-verify-hashes` to skip) and checks macOS, Metal,
`MLX_ENABLE_TF32` and ffmpeg. Re-run with `--skip-download` for just the doctor report.

The server starts instantly; the models are loaded lazily by the first job (a few tenths of a
second, they are memory-mapped) and stay resident until the precision changes or a job fails.

## Using the studio

The UI is a single page with hash routes; every view works at phone width and follows the system
light/dark theme (overridable in Settings).

**Create** (`#/create`) — describe the *style* (free text; genre chips append common tags), write
*lyrics* using section tags on their own lines (`[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`,
`[Bridge]`, `[Interlude]`, `[Outro]` — chips insert them), and choose a *mode*:

| Mode (`cot`) | What the model does |
|---|---|
| `full` (default) | Plans a complete ABC score (melody, chords, structure) before writing audio tokens. Best coherence. |
| `melody` | Plans only the vocal melody line, then generates audio. |
| `off` | No score planning; audio is generated straight from style + lyrics. Fastest, loosest. |

Pick a preset (below), a seed (blank = random; 🎲 rolls one), an optional CFG scale, and
optionally paste an ABC score to generate from (requires `full` or `melody`). Set **Variations**
to N > 1 to submit N jobs at once with seeds `seed, seed+1, …` (or independent random seeds);
they are grouped in the queue and library.

**Queue** (`#/queue`) — live cards for queued and running jobs: stage, progress, tokens/s, ETA,
the ABC score streaming in while it is being planned, and a cancel button. Jobs run one at a time
(the GPU allows one workload per process); cancelling a running job stops it at the next token /
ODE step / decode chunk and the worker moves on.

**Library** (`#/library`) — finished songs with inline players, duration, preset, seed, mode and
group badges; filters by kind, group and text; failed/cancelled jobs listed separately; delete; download FLAC / MP3 / `artifacts.zip`.

**Song detail** (`#/song/<id>`) — player, the request, per-stage timing, and the score rendered
with abcjs next to an editable ABC textarea. **Regenerate from this score** submits a `regenerate`
job that keeps the lyrics and seed (style editable) and generates audio from your edited score; the
result links back to its parent. **More variations** pre-fills Create with this song's request.
Covers additionally show the transcribed score. **▶ Play score (MIDI)** previews the score with
abcjs' synthesiser — this loads General MIDI soundfonts from the internet
(`paulrosen.github.io/midi-js-soundfonts`) the first time it is used.

**Cover** (`#/cover`) — drop an `mp3 / wav / flac / m4a / ogg` file (≤ 200 MB), pick a task,
style, lyrics, seed and preset. The audio is transcribed offline with SheetSage2 + MERT, and a
song is generated from the transcription:

| Task | Transcribes | Generates with |
|---|---|---|
| `melody-full` (default, recommended) | melody + chords | `cot=melody` |
| `melody-vocal` | vocal melody only | `cot=melody` |
| `full` | full score | `cot=full` |

Cover is disabled (and the API answers 409) until the transcription models and ffmpeg are present;
`#/settings` and `GET /api/status` say what is missing.

**Settings** (`#/settings`) — default preset, memory budget (4–44 GiB), require-AC-power, theme,
plus a live engine panel (state, precision, memory, current job, model paths).

### Presets

| Preset | Precision (AR) | ODE steps (NAR) | Notes |
|---|---|---|---|
| **Quality** | `bf16` | 32 | Reference quality; ~1× realtime on an M5 Max. |
| **Fast** | `8bit` | 8 | ~2× faster than realtime; the 8-bit AR still loads the BF16 AR for NAR conditioning. |
| **Custom** | `bf16` / `8bit` / `4bit` | 4–64 | Precision is fixed per resident pipeline (changing it rebuilds, ~0.1 s); steps are per job. |

### Measured timings (M5 Max, 48 GB, macOS 27, mlx-Yue `ab0f058`, memory budget 24 GiB)

Full song = `examples/full-song.json` (City Pop, ~3 min, `cot=full`, no supplied score). RTF is
generation time ÷ audio length (lower is faster; < 1 is faster than realtime).

| Preset | ABC planning | Semantic tokens | NAR (synthesis) | VAE (decode) | End-to-end | Audio | RTF |
|---|---|---|---|---|---|---|---|
| Fast (8bit, 8 steps) | 131 tok/s (2072 tok, 15.8 s) | 129 tok/s (35.9 s) | 27.2 s | 3.4 s | **82.4 s** | 184.7 s | 0.45 |
| Quality (bf16, 32 steps) | 92.6 tok/s (1966 tok, 21.2 s) | 98.6 tok/s (4309 tok, 43.7 s) | 95.8 s | 3.5 s | **164.5 s** | 172.3 s | 0.95 |

Quickstart clip (`examples/quickstart.json`, 16 s, score supplied): Fast 4.2 s, Quality 7.6 s
end-to-end. Model verification + first load adds ~3–5 s to the first job of a session; the
`Saving artifacts` stage is negligible. Reproduce with

```bash
uv run python scripts/smoke.py --preset quality --example examples/full-song.json
```

## Command line

```
uv run yue2-studio [--host 127.0.0.1] [--port 8765] [--open] [--fake] [--fake-delay 0.3]
                   [--reload] [--log-level info]
```

| Flag | Meaning |
|---|---|
| `--host`, `--port` | Bind address (default `127.0.0.1:8765`; the server has no authentication, keep it local). |
| `--open` | Open the UI in the default browser once the server is up. |
| `--fake` | Use the scripted `FakeEngine` (no models, no MLX): jobs emit realistic events and write a silent FLAC. For UI work and demos. |
| `--fake-delay` | Seconds between fake engine events (default 0.3). |
| `--reload` | Auto-reload on code changes (development). |

`YUE2_STUDIO_HOME` sets the directory that holds `data/` (SQLite `app.db`, `songs/<job_id>/`,
`uploads/`) and `models/`. It defaults to the main git checkout root — even when running from a
worktree under `.worktrees/`, so worktrees share the downloaded weights — or, without git, to the
parent of the `yue2_studio` package. Two servers sharing one home share a database and would steal
each other's jobs; give each its own home (with `models/` symlinked) if you must run two.

## HTTP API

The UI talks to a small JSON + SSE API documented in [`docs/API.md`](docs/API.md) (also browsable
at `/api/docs` while the server runs): `GET /api/status`, `POST /api/jobs` (`create`, `regenerate`,
`cover`, `variations`), `GET /api/jobs?status=&kind=&group=&limit=&offset=`, `GET|DELETE
/api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, `GET /api/jobs/{id}/events` (SSE), song artifacts
under `/api/songs/{id}/` (`audio.flac` with Range support, `audio.mp3`, `score.abc`, `plan.json`,
`artifacts.zip`, `transcription/score.abc`), `POST /api/upload`, `GET|PUT /api/settings`.

## Development

```bash
uv run pytest                      # ~225 tests, < 10 s, no models or GPU needed
uv run ruff check .                # lint (E, F, W, I, UP, B; line length 110)
uv run yue2-studio --fake          # the real server with the fake engine
uv run python scripts/mock_api.py  # standalone in-memory mock of docs/API.md on :8790 for UI work
```

The tests drive the real FastAPI app through `httpx` with the `FakeEngine`
(`yue2_studio/fake.py`), which implements the engine protocol the worker uses and emits the same
raw event shapes as the MLX engine; `tests/test_imports.py` asserts that none of the HTTP-side
modules ever import `mlx`. `tests/test_engine_lifecycle.py` imports the real engine module (skipped
where MLX is unavailable) but injects a fake pipeline, so nothing touches the GPU.

### Architecture

One uvicorn process. Request handlers only read/write the SQLite job store, enqueue job ids and
stream events; a single worker thread owns the resident `StudioPipeline` (a subclass of mlx-Yue's
`YuE2Pipeline` whose progress hook publishes events instead of printing) and runs jobs strictly
serially. Raw engine events are normalised into the HTTP `ProgressEvent` shape and fanned out to
SSE subscribers through an in-process event bus; the last event per job is cached so late
subscribers see the current state. Artifacts land in `data/songs/<job_id>/` (`plan/` right after
planning, `song/` with `audio.flac` at the end, `summary.json`, `transcription/` for covers).
mlx-Yue itself is never modified.

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

Layout: `yue2_studio/config.py` (paths, presets, sets `MLX_ENABLE_TF32=0` before anything imports
MLX), `engine.py` (pipeline wrapper; the only module importing `mlx`), `jobs.py` (validation,
SQLite store), `worker.py` (thread, cancellation, event bus, normalisation), `api.py` (routes),
`audio.py` (ffmpeg, uploads, zip), `main.py` (app factory + CLI), `static/` (UI),
`scripts/setup.py` (weights + doctor), `scripts/smoke.py` (one generation through the engine with
timings), `compat/lyra-yue2/` (see below), `docs/PLAN.md` (design), `docs/API.md` (contract).

## Troubleshooting

- **`MLX_ENABLE_TF32` error / guard refuses to run.** mlx-Yue requires `MLX_ENABLE_TF32=0` to be
  set before MLX initialises. `yue2_studio.config` sets it on import, and every entry point imports
  `config` first; if you embed the package elsewhere, import `yue2_studio.config` before `mlx`.
- **`MemoryError: Process footprint exceeds budget` / job fails then the engine shows `cold`.**
  The pipeline's memory watchdog tripped the configured budget (Settings → Memory budget, default
  24 GiB; the guard requires budget ≤ total RAM − 4 GiB, so use ≤ 20 on a 24 GB machine). Peak use
  is ~11 GiB for generation; covers release the song models before loading SheetSage2 + MERT
  (~3 GiB) and reload them lazily afterwards. After
  any non-cancellation failure the pipeline is discarded (the guard latches the error) and the next
  job rebuilds it automatically.
- **`AC power disconnected` failures.** With `require_ac` on, the guard aborts the job when the Mac
  leaves mains power. It is off by default; long batches on battery are simply slow.
- **`503 engine_unavailable` / models missing.** `GET /api/status` → `models.present=false`. Run
  `uv run python scripts/setup.py` (add `--with-cover` for covers) and check `models/converted`
  contains `conversion.json` and the `ar-*.safetensors` files. Cover shows its own reasons under
  `status.cover.reasons` (missing ffmpeg / SheetSage2 / MERT).
- **`audio.mp3` answers 503.** ffmpeg is not on `PATH`; FLAC download and playback still work.
- **Jobs left from a previous run.** Queued jobs are re-enqueued on start; a job that was `running`
  when the server died is marked `failed` ("server restarted…") — resubmit it.
- **Why `compat/lyra-yue2`?** mlx-Yue commit `ab0f058` renamed its distribution from `lyra-yue2` to
  `mlx-yue`, but `lyra.pipeline` / `lyra.commands` still call
  `importlib.metadata.version("lyra-yue2")`. The `compat/lyra-yue2` directory is an empty,
  metadata-only distribution with that name so the lookup succeeds without patching mlx-Yue. Keep it.
- **Two servers on one home.** They share `data/app.db` and steal each other's jobs; use
  `YUE2_STUDIO_HOME` per server.

## Licences and attribution

- The **YuE2-3B model weights** ([m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), and the
  MLX conversion [vanch007/mlx-Yue2-3B](https://huggingface.co/vanch007/mlx-Yue2-3B)) are released
  under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use
  only**. Music you generate with this studio is subject to that licence; check the model card
  before any commercial use. The VAE ([m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)) and
  the transcription models ([SheetSage2](https://huggingface.co/m-a-p/SheetSage2),
  [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong)) carry their own licences on
  their model cards.
- [mlx-Yue](https://github.com/vanch007/mlx-Yue) (the engine this studio wraps, pinned at
  `ab0f058`) is licensed under the
  [Apache License 2.0](https://github.com/vanch007/mlx-Yue/blob/main/LICENSE). The example
  requests in `examples/` are copied from it.
- [abcjs](https://github.com/paulrosen/abcjs) (MIT) renders and plays the scores in the browser;
  its MIDI preview downloads soundfonts from `paulrosen.github.io/midi-js-soundfonts` at runtime.
