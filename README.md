# YuE2 Studio

A local web studio for generating full songs with [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
on Apple Silicon. It wraps [mlx-Yue](https://github.com/vanch007/mlx-Yue) (the native MLX port of
the YuE2 pipeline: ABC score planning → semantic tokens → flow-matching acoustics → 48 kHz stereo
VAE decode) in a single-process FastAPI server with a job queue, live progress over Server-Sent
Events, a song library, score editing with regeneration, seed variations and audio-to-song covers.
Everything runs on your Mac; nothing leaves it except loading abcjs from cdnjs (the UI needs
internet for score rendering) and the optional soundfont download for the score preview (see below).

## Requirements

| | |
|---|---|
| Hardware | Apple Silicon Mac. Peak use is ~11 GiB of unified memory; the default memory budget is 24 GiB, so **48 GB is recommended** (a 24–32 GB machine works with the budget lowered in Settings, see [Troubleshooting](#troubleshooting)). Timings below are from an M5 Max / 48 GB. |
| macOS | **14.2 or newer** on M1–M4 chips; **26.2 or newer on M5** chips (mlx-Yue checks the chip via `mx.device_info()` and refuses older versions). Native arm64 Python only — not Intel or Rosetta. |
| Python | 3.12 (fetched automatically by `uv`; the project is pinned to `>=3.12,<3.13`). |
| Tools | [`uv`](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH` (`brew install uv ffmpeg`). ffmpeg is needed for MP3 export, upload probing and covers. |
| Disk | ~13 GB of weights: generator (ar-8bit 2.7 GB, ar-bf16 4.3 GB, nar-bf16 2.9 GB) + VAE 0.5 GB; covers add SheetSage2 0.2 GB + MERT-v2-FullSong 2.5 GB. |

## Setup

```bash
uv sync                                          # creates .venv with Python 3.12 + mlx-Yue (ianiv fork, perf branch)
uv run python scripts/setup.py --with-cover      # downloads weights into models/ and prints a doctor report
uv run yue2-studio --open                        # http://127.0.0.1:8765
```

mlx-Yue comes from the `perf` branch of the [ianiv/mlx-Yue](https://github.com/ianiv/mlx-Yue/tree/perf) fork,
pinned by commit in `pyproject.toml`: upstream `9253ed1` plus the speed work described under
[Measured timings](#measured-timings-m5-max-48-gb-macos-27-memory-budget-24-gib).

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
light/dark theme (overridable in Settings). The header reads **Generate ▾** (Create · Cover · Hum) ·
Projects · Library · Settings, with **Queue** (and its pending-jobs badge) and the engine pill at
the right.

**Create** (`#/create`) — describe the *style* (free text; genre chips append common tags), write
*lyrics* using section tags on their own lines (`[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`,
`[Bridge]`, `[Interlude]`, `[Outro]` — chips insert them), and choose a *mode*:

| Mode (`cot`) | What the model does |
|---|---|
| `full` (default) | Plans a complete ABC score (melody, chords, structure) before writing audio tokens. Best coherence. |
| `melody` | Plans only the vocal melody line, then generates audio. |
| `off` | No score planning; audio is generated straight from style + lyrics. Fastest, loosest. |

Pick a preset (below), optionally one or more **LoRA adapters** (see [LoRA adapters](#lora-adapters)), a seed (**Random seed** is on by default so the server picks a fresh one
per submit — untick it to type or 🎲-roll a fixed seed; *Use this seed* on a result does that for
you), an optional CFG scale, and
optionally paste an ABC score to generate from (requires `full` or `melody`). Set **Variations**
to N > 1 to submit N jobs at once with seeds `seed, seed+1, …` (or independent random seeds);
they are grouped in the queue and library.

Create is the iteration page: submitting keeps you on it with the form intact, and the job appears
in the **Results** column beside the form — live progress while it runs (the streaming score is
collapsed by default to keep several variations compact; *Show score* expands it per card), then a
play button the moment it finishes, with *Use this seed* and *Regenerate from its score*
shortcuts. Recent results persist across reloads (last 20); the Queue and Library still show
everything.

Every play button feeds one player bar at the bottom of the page (seek, time, volume, ⏮/⏭ for an
album), which keeps playing across pages and re-renders and remembers the last song across reloads.

**Queue** (`#/queue`) — live cards for queued and running jobs: stage, progress, tokens/s, ETA,
the ABC score streaming in while it is being planned, and a cancel button. Jobs run one at a time
(the GPU allows one workload per process); cancelling a running job stops it at the next token /
ODE step / decode chunk and the worker moves on.

**Library** (`#/library`) — finished songs with play buttons, duration, generation time (hover for the
per-stage split and realtime factor), preset, seed, mode and
group badges; filters by kind, group and text; failed/cancelled jobs listed separately; delete; download FLAC / MP3 / `artifacts.zip`.

**Song detail** (`#/song/<id>`) — click the title to rename the song; play button, the request (style and the full lyrics, shown expanded
under it), per-stage timing, and the score rendered with abcjs next to an editable ABC
textarea. **Regenerate from this score** submits a `regenerate`
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

**Settings** (`#/settings`) — default preset, memory budget (6–44 GiB; mlx-Yue's guard rejects ≤ 5 GiB), require-AC-power,
fast numerics (on by default, see [Measured timings](#measured-timings-m5-max-48-gb-macos-27-memory-budget-24-gib)), theme,
Claude assist (below), plus a live engine panel (state, precision, memory, merged LoRAs, current job,
model paths, the LoRA adapters found in `models/loras/` and why any of them is unusable).

### Claude assist

Every generate page (Create, Cover, Hum) has an **Ask Claude** box: describe the song ("a bittersweet
synth-pop duet about a long-distance call, 100 BPM") and Claude fills in the title, style tags,
section-tagged lyrics and — on Create — the mode and CFG, with a one-line "if it comes out X, change
Y" note. With **Refine** on, the current form fields are sent along and Claude edits them instead of
starting over. The system prompt is `yue2_studio/assist_prompt.md` (the `yue2-prompt` skill adapted
for the app); the reply is constrained to a JSON schema, so nothing needs parsing.

Two providers, chosen in Settings (`auto` by default):

- **`claude` CLI** — if [Claude Code](https://claude.com/claude-code) is installed and logged in,
  nothing else is needed; the studio runs `claude -p` with your login (no MCP servers, skills or
  hooks are loaded, and an exported `ANTHROPIC_API_KEY` is not passed through, so the call bills the
  login). The default model is the CLI's own; set *Model* (e.g. `haiku`, `sonnet`) to override.
- **Anthropic API** — paste an API key in Settings (or export `ANTHROPIC_API_KEY` before starting
  the server; Settings wins when both are set). Default model `claude-sonnet-5`. `POST /api/assist/test`
  (the *Test* button) checks the connection.

What is sent: the system prompt, the page name, your request and — with Refine on — the current
title/style/lyrics/mode/CFG. Nothing else (no audio, no job history). The key is stored **in plain
text** in `data/app.db` and never returned by the API (`GET /api/settings` only reports
`has_api_key`, which counts only the stored key); clear it with an empty value. `off` hides the box
entirely; a missing CLI/key leaves it in place but disabled, with a tag and a hint saying why;
`GET /api/status` → `assist` says which provider is active and why not.

### LoRA adapters

Drop adapter files into `models/loras/` (rescanned every 5 s; no restart needed) and pick them in
the **LoRA adapters** field on Create, Cover and Song (regenerate inherits the parent's stack).
Each row has a scale (0–4, 1 = as trained) and you can stack several — typically one adapter for
the **AR** planner and one for the **NAR** acoustic decoder. Two layouts are recognised:

| Layout | Files | Notes |
|---|---|---|
| single `.safetensors` (name = file stem) | `layers.N.<block>.<proj>.lora_A` `[r,in]`, `.lora_B` `[out,r]`; optional `vae2llm.*` / `llm2vae.*` full replacements | the layout of the YuE2 LoRAs published on Hugging Face, e.g. [`YuE2-instrumental-cot-full-loras`](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras) (`ar_lora_inst_v3abc.bf16.safetensors`, AR) and [`yue2-mothersuperior-realaudio-tokenizer-v4`](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4) (`nar_lora_joint_v4.bf16.safetensors`, NAR). The file's `lora_scale` metadata (default 1.0) is honoured. |
| PEFT directory (name = folder) | `adapter_config.json` + `adapter_model.safetensors` | `base_model.model.<module>.lora_{A,B}.weight`; scale `lora_alpha / r`. |

Supported targets are the AR linears (`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
`lm_head`) and the NAR linears (`nar_self_attn.*`, `nar_mlp.*`, `llm2vae`, `vae2llm`). Adapters
are *merged* into the weights when the models load (`W += scale · B @ A`, FP32 opmath; 8-/4-bit AR
weights are dequantised, merged and requantised), which takes well under a second per adapter and
costs nothing at generation time. A different stack drops the resident models and merges again on
the next job. Which adapters (name, sha256, scale) shaped a song is recorded in the job, in
`summary.json` and in the song's `result.json`.

Files with `hum_proj.*` tensors are [hum-to-song](https://huggingface.co/Mothersuperior/YuE2-hum-to-song)
adapters: they are listed with kind *hum* and chosen on the **Hum** page (below), not in the LoRA
stack. Files whose tensors the studio cannot run are listed as *unusable* with the reason.

Tip for the instrumental AR LoRA: use mode `full`, put only *untimed* bare section tags in the
lyrics field (`[intro]`, `[verse]`, `[chorus]`, … or just `[instrumental]`) and pair it with the NAR
LoRA; see `examples/instrumental-lora.json`. Timed tags (`[verse 0:15-0:45]`) tend to make this
adapter overrun to the length cap, and songs land around 3–5 min regardless of the plan; an
occasional overrun is a known trait of the adapter — re-roll the seed rather than lowering the
scale. Those weights are CC BY-NC 4.0 like YuE2 itself.

### Hum to song

**Hum** (`#/hum`) — hum a melody for 10–30 seconds (record in the browser — Safari, Chrome and
Firefox; the page must be on `localhost` or HTTPS for microphone access — or drop a recording),
add a style and lyrics, and get a whole song built around it. This is the two-stage mechanism from
[Mothersuperior/YuE2-hum-to-song](https://huggingface.co/Mothersuperior/YuE2-hum-to-song):

1. **Score continuation** (no extra weights). The hum is transcribed with SheetSage2 (melody only),
   its trailing rests are trimmed and the *open* score is placed in the planner prompt without an
   end token, so YuE2 keeps writing it: new sections, an ending, all in the hum's key and range —
   the hummed phrase tends to come back as the hook. **Melody** picks how the hum is used:
   *Continue my melody* (above), *Hum is the whole melody* (the transcription is the complete vocal
   line, like a cover) or *Ignore the notes* (the planner writes its own score; only the adapter
   below uses the hum).
2. **Prosody adapter** (optional, needs `hum_adapter_v1_combined.safetensors` in `models/loras/`).
   The hum is reduced to a carrier — its pitch track (`librosa.pyin`) drives a sine whose amplitude
   follows the hum's envelope — VAE-encoded to 25 Hz latents and added to the acoustic decoder's
   hidden state at four depths, so the sung line follows *how* you hummed it (timing, phrasing).
   **Hum influence** is classifier-free guidance on that channel (1 = as trained, 0 = off, above 1
   exaggerates; ≠ 1 costs about 2× synthesis time); **Hum starts at** places the carrier inside the
   song. `hum_adapter_v1.safetensors` (the non-combined file) must be stacked on
   `nar_lora_joint_v4.bf16` — add that one in the LoRA stack; the hum adapter is always merged last.

The transcriber needs something voice-like: a real hum works, a synthesised tone does not (the job
fails early with "no notes"). Artifacts land in `data/songs/<id>/hum/` (`hum.abc`, `carrier.flac`,
`carrier_latents.npy`, `prosody.json`) and the song page shows the hum's score next to the continued
one. Command line: `uv run python scripts/hum_smoke.py --audio my-hum.m4a --adapter
hum_adapter_v1_combined` (`--analyse-only` just pitch-tracks and encodes the hum). Requires the
cover prerequisites (`scripts/setup.py --with-cover`, ffmpeg) plus `librosa` (installed by `uv sync`).

### Projects

**Projects** (`#/projects`) — albums and soundtracks. A project is an ordered tracklist of named
*tracks*; jobs are attached to a track as *takes* (a job belongs to at most one track) and one
finished take per track is *chosen*. On the project page (`#/project/<id>`) each track has a
**New take** menu (Create / Cover / Hum open with a "New take for Project › Track" banner and the
result lands in that track; *Add from Library…* opens the Library in multi-select with "Add to
project…" preset to the track). Takes are rated with 👍/👎, 1–5 stars and a note, sorted and
filtered (one remembered preference shared by every project), and *Choose* marks the track's final
take (only finished takes with audio).
Every take (and the chosen take) has a lyrics icon: hover or focus it to see the song's lyrics in a
popup, click (tap) to keep it open, Esc or a click elsewhere to close.
Finished takes made with the **Fast** preset also get **⇧ Quality** (on the project page and the song
page): one click regenerates the song on the Quality preset from its own score with the same seed,
style, lyrics, title and LoRAs, as a new take in the same track. The button then links to that
version (✓ Quality version) instead of queueing another. Hum takes are left out (a regenerate would
drop the hum carrier) and so are `cot=off` songs, which have no score.
Tracks are renamed in place and reordered by dragging the handle (or ↑/↓). **Play album** queues
the chosen takes in tracklist order; **Export ZIP** downloads
`<name>/01 Track.flac …` plus `tracklist.json` / `tracklist.md` (tracks without a chosen take are
listed as missing) — the MP3 export needs ffmpeg. Song cards and the song page show a
`Project › Track` tag (★ when chosen); the song page's *Project* panel attaches, rates, chooses or
detaches the song, and *Regenerate* / *More variations* from a take stay in its track. Deleting a
track or project only detaches the songs; deleting a chosen song clears the track's choice.

### Presets

| Preset | Precision (AR) | ODE steps (NAR) | Notes |
|---|---|---|---|
| **Quality** | `bf16` | 32 | Reference quality; ~0.55× realtime on an M5 Max (fast numerics). |
| **Fast** | `8bit` | 8 | ~3.5× faster than realtime; the 8-bit AR still loads the BF16 AR for NAR conditioning. |
| **Custom** | `bf16` / `8bit` / `4bit` | 4–64 | Precision is fixed per resident pipeline (changing it rebuilds, well under a second); steps are per job. |

### Measured timings (M5 Max, 48 GB, macOS 27, memory budget 24 GiB)

Full song = `examples/full-song.json` (City Pop, ~3 min, `cot=full`, no supplied score, seed 12300);
"CFG 2.5" is the same request with `cfg_scale: 2.5` (two AR branches per semantic token; `cot=off` always
runs two branches too). RTF is generation time ÷ audio length (lower is faster; < 1 is faster than
realtime). *Stock* is upstream mlx-Yue `9253ed1`; *exact* and *fast numerics* are the fork's `perf` branch.

| Run | ABC planning | Semantic tokens | NAR (synthesis) | VAE | End-to-end | Audio | RTF |
|---|---|---|---|---|---|---|---|
| Fast preset — stock | 122 tok/s (17.0 s) | 118 tok/s (39.0 s) | 29.1 s | 3.4 s | 88.6 s | 184.7 s | 0.48 |
| Fast preset — exact | 175 tok/s (11.8 s) | 172 tok/s (26.9 s) | 27.4 s | 3.4 s | 69.6 s | 184.7 s | 0.38 |
| Fast preset — fast numerics | 176 tok/s (11.8 s) | 172 tok/s (26.9 s) | 11.1 s | 3.4 s | **53.3 s** | 184.7 s | 0.29 |
| Quality — stock | 94 tok/s (20.9 s) | 97 tok/s (44.4 s) | 104.2 s | 3.9 s | 173.6 s | 172.3 s | 1.01 |
| Quality — exact | 118 tok/s (16.7 s) | 123 tok/s (35.1 s) | 91.9 s | 3.1 s | 147.0 s | 172.3 s | 0.85 |
| Quality — fast numerics | 117 tok/s (16.7 s) | 123 tok/s (35.2 s) | 38.5 s | 3.2 s | **93.8 s** | 172.3 s | 0.54 |
| Quality, CFG 2.5 — stock | 94 tok/s (20.9 s) | 61 tok/s (69.7 s) | 97.9 s | 3.1 s | 191.7 s | 170.0 s | 1.13 |
| Quality, CFG 2.5 — exact | 117 tok/s (16.8 s) | 81 tok/s (52.4 s) | 89.8 s | 3.1 s | 162.2 s | 170.0 s | 0.95 |
| Quality, CFG 2.5 — fast numerics | 117 tok/s (16.8 s) | 102 tok/s (41.9 s) | 37.9 s | 3.1 s | **99.8 s** | 170.6 s | 0.58 |

**Exact** runs reproduce stock bit for bit (identical ABC tokens, semantic tokens and latents in all
three cases). The speed comes from a software-pipelined AR decode loop (the next token's forward pass
is queued before the current one is read back, so the GPU never waits on Python) and fused RMSNorm /
RoPE / SwiGLU kernels with the same rounding points; AR decode is now at the M5 Max's memory
bandwidth (weights + KV cache).

**Fast numerics** (Settings, on by default) adds two changes that are numerically equivalent but not
bit-identical: acoustic attention runs as native BF16 SDPA instead of FP32-promoted SDPA (on M5 the
promoted path cannot use the GPU's neural accelerators and was ~60 % of synthesis time; M1–M4 already
skip the promotion, so they gain less there), and the two CFG branches share one weight pass per token.
Without CFG the song is the same take: latents differ by 2.5–3 % RMS (cosine 0.9995, audio SNR ≈ 26 dB,
comparable to mlx-Yue's own port-vs-PyTorch difference). With CFG the sampled tokens diverge, so the
same seed gives a different take than exact mode. Untick *Fast numerics* when you need to reproduce a
song made before this change (or in exact mode) from its seed.

Quickstart clip (`examples/quickstart.json`, 16 s, score supplied, fast numerics): Fast 3.3 s, Quality 5.9 s
(stock: 4.2 s, 7.6 s)
end-to-end. Model verification + first load adds ~3–5 s to the first job of a session; the
`Saving artifacts` stage is negligible. Reproduce with

```bash
uv run python scripts/smoke.py --preset quality --example examples/full-song.json   # add --exact-numerics for exact mode
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
parent of the `yue2_studio` package. Don't point two servers at one home (see
[Troubleshooting](#troubleshooting)).

## HTTP API

The UI talks to a small JSON + SSE API documented in [`docs/API.md`](docs/API.md) (also browsable
at `/api/docs` while the server runs): `GET /api/status`, `POST /api/jobs` (`create`, `regenerate`,
`cover`, `variations`), `GET /api/jobs?status=&kind=&group=&limit=&offset=`, `GET|DELETE
/api/jobs/{id}`, `POST /api/jobs/{id}/cancel`, `GET /api/jobs/{id}/events` (SSE), song artifacts
under `/api/songs/{id}/` (`audio.flac` with Range support, `audio.mp3`, `score.abc`, `plan.json`,
`artifacts.zip`, `transcription/score.abc`), `POST /api/upload`, `GET|PUT /api/settings`,
`POST /api/assist` (Ask Claude) and `POST /api/assist/test`.

## Development

```bash
uv run pytest                      # ~380 tests, < 15 s, no models or GPU needed (VAE-encoder tests use models/vae when present)
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
MLX), `engine.py` (pipeline wrapper), `lora.py` (adapter discovery + weight merging), `hum.py`
(hum options, open-score trimming, carrier analysis), `hum_nar.py` (hum-conditioned acoustic
sampler, a `CachedNAR` subclass), `vae_encoder.py` (MLX port of the VAE encoder) — the last three
plus `engine.py` are the only modules importing `mlx` —, `jobs.py` (validation, SQLite store),
`worker.py` (thread, cancellation, event bus, normalisation), `api.py` (routes), `audio.py`
(ffmpeg, uploads, zip), `assist.py` + `assist_prompt.md` (Ask Claude: CLI / API providers, schema,
system prompt), `main.py` (app factory + CLI), `static/` (UI), `scripts/setup.py` (weights
+ doctor), `scripts/smoke.py` / `scripts/hum_smoke.py` (one generation through the engine with
timings), `docs/PLAN.md` (design), `docs/API.md` (contract).

## Troubleshooting

- **`MLX_ENABLE_TF32` error / guard refuses to run.** mlx-Yue requires `MLX_ENABLE_TF32=0` to be
  set before MLX initialises. `yue2_studio.config` sets it on import, and every entry point imports
  `config` first; if you embed the package elsewhere, import `yue2_studio.config` before `mlx`.
- **`MemoryError: Process footprint exceeds budget` / job fails then the engine shows `cold`.**
  The pipeline's memory watchdog tripped the configured budget (Settings → Memory budget, default
  24 GiB, settable 6–44; the guard requires 5 GiB < budget ≤ total RAM − 4 GiB, so use ≤ 20 on a
  24 GB machine). Peak use
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
- **"mlx-Yue requires macOS >=14.2" / "The MLX M5 runtime requires macOS >=26.2" / "Native Apple
  Silicon Python on macOS is required".** mlx-Yue's runtime gate (`lyra.runtime`): update macOS, or
  make sure `uv` created an arm64 (not Rosetta) Python. `uv run python scripts/setup.py
  --skip-download` prints the full `runtime` report.
- **Two servers on one home.** They share `data/app.db` and steal each other's jobs; give each its
  own `YUE2_STUDIO_HOME` (with `models/` symlinked to the shared weights).

## Licences and attribution

- **YuE2 Studio itself** (this repository's code) is released under the [MIT License](LICENSE).
  The model weights and other components below are separate works with their own licences.
- The **YuE2-3B model weights** ([m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), and the
  MLX conversion [vanch007/mlx-Yue2-3B](https://huggingface.co/vanch007/mlx-Yue2-3B)) are released
  under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use
  only**. Music you generate with this studio is subject to that licence; check the model card
  before any commercial use. The VAE ([m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)) is
  **also CC BY-NC 4.0** (`models/converted/LICENSE` covers both). The transcription models
  ([SheetSage2](https://huggingface.co/m-a-p/SheetSage2),
  [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong)) carry their own licences on
  their model cards.
- [mlx-Yue](https://github.com/vanch007/mlx-Yue) (the engine this studio wraps; `9253ed1` plus the
  speed work on the [ianiv/mlx-Yue](https://github.com/ianiv/mlx-Yue/tree/perf) `perf` branch) is licensed under the
  [Apache License 2.0](https://github.com/vanch007/mlx-Yue/blob/main/LICENSE). The example
  requests in `examples/` are copied from it.
- [abcjs](https://github.com/paulrosen/abcjs) (MIT) renders and plays the scores in the browser. It
  is loaded from cdnjs on every page load, and its MIDI preview downloads soundfonts from
  `paulrosen.github.io/midi-js-soundfonts` at runtime — the only network access the studio makes.
