# YuE2 Studio — HTTP + SSE API contract

Binding contract between `yue2_studio/api.py` (backend) and `yue2_studio/static/` (frontend).
Both sides implement this document independently; any change here must be made here first.

## 1. Conventions

- Base URL: `http://127.0.0.1:8765`. All API routes live under `/api/`. Everything else is static.
- Request/response bodies are JSON (`Content-Type: application/json`) unless stated (upload, SSE, files).
- Timestamps are ISO-8601 UTC with `Z` suffix and millisecond precision: `"2026-09-13T21:04:05.123Z"`.
- Optional fields (`?`) are present with value `null` when unset; never omitted. Unknown request fields are ignored.
- Errors: every non-2xx response has the body `{"error": {"code": "<snake_case>", "message": "<human text>"}}`.

| HTTP | `error.code`         | When |
|------|----------------------|------|
| 400  | `validation_error`   | bad/missing field, enum violation, `abc` with `cot=off`, bad upload type/size |
| 404  | `not_found`          | unknown job/group/upload id, missing artifact (e.g. `plan.json` for a failed job) |
| 409  | `conflict`           | cancel of done/failed/cancelled job; delete of running job; cover when `status.cover.available=false` |
| 413  | `too_large`          | upload > 200 MB |
| 503  | `engine_unavailable` | engine failed to load / models missing (`status.models.present=false`) |
| 500  | `internal_error`     | anything else |

## 2. Data models

Types: `str`, `int`, `float`, `bool`, `[T]` list, `T?` nullable, `enum(a|b)`.

### Job

| Field | Type | Notes |
|-------|------|-------|
| `id` | str | **uuid4 hex, 32 chars, no dashes** (`"3f9a…"`); also the song dir name `data/songs/<id>/` |
| `kind` | enum(create\|regenerate\|cover) | `variations` is expanded on submit; stored jobs are never `variations` |
| `status` | enum(queued\|running\|done\|failed\|cancelled) | terminal = done/failed/cancelled |
| `group_id` | str? | set for variations members |
| `parent_id` | str? | set for `regenerate` (source job) |
| `preset` | enum(quality\|fast\|custom) | |
| `precision` | enum(bf16\|8bit\|4bit) | resolved from preset |
| `ode_steps` | int | resolved from preset (quality=32, fast=8) |
| `seed` | int | the resolved seed (never null) |
| `params` | object | `CreateParams` / `RegenerateParams` / `CoverParams` per `kind`, with defaults filled in |
| `title` | str? | copied from `params.title`, else `null` |
| `created_at` | str | |
| `started_at` | str? | |
| `finished_at` | str? | |
| `error` | str? | set when `failed` |
| `timing` | object? | `{"<stage>": seconds, ..., "abc_tps": float?, "semantic_tps": float?, "audio_seconds": float?}`; set when done |
| `truncated` | object? | `{"phase": "abc"\|"semantic", "reason": str}` if generation hit a length limit; else null |
| `progress` | ProgressEvent? | last event emitted (also for terminal jobs); null if none yet |
| `artifacts` | object | `{"audio": bool, "score": bool, "plan": bool, "transcription": bool}`; all false until produced |
| `position` | int? | 0-based queue position while `queued`; null otherwise |

### CreateParams

| Field | Type | Default / rule |
|-------|------|----------------|
| `style` | str | required, non-empty |
| `lyrics` | str | required, non-empty |
| `cot` | enum(full\|melody\|off) | `full` |
| `seed` | int? | random 0..2^31-1 chosen by server; response echoes resolved value |
| `cfg_scale` | float? | `null` = engine default |
| `abc` | str? | supplied score; **400 if non-empty and `cot=off`** |
| `title` | str? | display only |

### RegenerateParams

| Field | Type | Rule |
|-------|------|------|
| `parent_id` | str | required; 404 if unknown |
| `abc` | str | required, non-empty |
| `style` | str? | `null` → inherit parent |
| `lyrics` | str? | `null` → inherit parent |
| `seed` | int? | `null` → inherit parent seed |
| `title` | str? | `null` → inherit |

`cot` is inherited from the parent; if the parent had `cot=off` the server uses `melody`. Preset/precision/ode_steps
come from the common submit fields (below), defaulting to the parent's. Stored `params` contains the resolved
values (`style`, `lyrics`, `cot`, `seed`, `abc`, `title`, `parent_id`).

### CoverParams

| Field | Type | Default / rule |
|-------|------|----------------|
| `upload_id` | str | required; from `POST /api/upload`; 404 if unknown |
| `task` | enum(melody-full\|melody-vocal\|full) | `melody-full` |
| `style` | str | required |
| `lyrics` | str | required |
| `seed` | int? | random |
| `title` | str? | defaults to upload filename stem |

### VariationsParams (submit only)

| Field | Type | Rule |
|-------|------|------|
| `count` | int | 2..16 |
| `base` | CreateParams | `base.seed` = starting seed (random if null) |
| `random_seeds` | bool | default `false`. `false`: seeds `seed, seed+1, …`; `true`: each job gets an independent random seed |
| `label` | str? | group label; default `"<title or first 40 chars of style> ×<count>"` |

### Common submit fields (top level of `POST /api/jobs`)

| Field | Type | Rule |
|-------|------|------|
| `kind` | enum(create\|regenerate\|cover\|variations) | required |
| `params` | object | required, per kind |
| `preset` | enum(quality\|fast\|custom)? | default `settings.default_preset` |
| `precision` | enum(bf16\|8bit\|4bit)? | only honoured when `preset=custom`; required then |
| `ode_steps` | int? | 4..64; only honoured when `preset=custom`; required then |

### Group

`{"id": str (uuid4 hex), "label": str, "created_at": str, "job_ids": [str]}` — `job_ids` in seed order.

### ProgressEvent (SSE `data`)

This is the **HTTP shape**. The engine emits a slightly different raw shape (see §6); the worker normalises it.

| Field | Type | Notes |
|-------|------|-------|
| `type` | enum(stage\|token\|abc\|log\|status) | |
| `job_id` | str | |
| `ts` | str | |
| `stage` | str? | short key: `load`, `transcribe`, `plan`, `semantic`, `synthesize`, `decode`, `save` |
| `label` | str? | `type=stage`: the engine's human label, e.g. `"Synthesizing audio"` |
| `completed` | int? | `type=stage`: units done |
| `total` | int? | `type=stage`: may be null when unknown |
| `unit` | str? | `type=stage`: e.g. `"tokens"`, `"steps"`, `"chunks"`, `"windows"` |
| `status` | enum(running\|complete\|failed\|cancelled\|truncated)? | `type=stage`: stage state; `type=status`: job state |
| `phase` | enum(abc\|semantic\|transcription)? | `type=token`/`abc` |
| `tokens` | int? | `type=token`: tokens generated so far in `phase`; `type=abc`: ABC tokens so far |
| `tps` | float? | `type=token`/`stage` (unit=tokens): tokens/s since the phase started |
| `seconds` | float? | `type=stage`/`token`: elapsed seconds in this stage/phase |
| `text` | str? | `type=abc`: full partial ABC decoded so far (replace, don't append); the last one has `partial=false` |
| `partial` | bool? | `type=abc`: `true` while planning streams, `false` for the final score |
| `message` | str? | `type=log`/`status`: human text, e.g. `"loading models (bf16)…"` |

Rules: fields not relevant to the `type` are `null`. `token` events are throttled to ≤4 Hz; `abc` events ≤4 Hz;
`stage` events ≤4 Hz plus one forced event at stage start (`completed=0`) and one at stage end.
A `status` event with `message` is emitted for engine load (`stage="load"`), on job start (`status=running`)
and immediately before `done` with the terminal status.

### Status (`GET /api/status`)

```json
{
  "engine":  {"state": "ready", "precision": "bf16", "memory_gib": 11.2, "current_job_id": null},
  "queue":   {"queued": 2, "running": null},
  "presets": [
    {"name": "quality", "precision": "bf16", "ode_steps": 32, "description": "BF16 AR, 32 ODE steps"},
    {"name": "fast",    "precision": "8bit", "ode_steps": 8,  "description": "8-bit AR, 8 ODE steps"},
    {"name": "custom",  "precision": null,   "ode_steps": null, "description": "Choose precision and steps"}
  ],
  "models":  {"converted_dir": "/abs/models/converted", "vae_dir": "/abs/models/vae", "present": true},
  "cover":   {"available": false, "reasons": ["MERT-v2-FullSong not downloaded"]},
  "ffmpeg":  true,
  "version": "0.1.0"
}
```

`engine.state ∈ cold|loading|ready|busy`; `precision`/`memory_gib` null when cold. `queue.running` = job id or null.

### Settings

`{"default_preset": "quality", "memory_budget_gib": 24, "require_ac": false, "theme": "system"}`
(`theme ∈ system|light|dark`, `memory_budget_gib` number 4..44). `PUT` accepts any subset and returns the full object.

## 3. Endpoints

### `GET /api/status` → 200 `Status` (above). Never 503; unavailability is reported in the body.

### `POST /api/jobs`

Body: `{"kind", "params", "preset"?, "precision"?, "ode_steps"?}`.

```json
{"kind": "create", "preset": "fast",
 "params": {"style": "dreamy indie pop, female vocal", "lyrics": "[verse]\n...\n[chorus]\n...", "cot": "full", "seed": 42}}
```
→ **201** `{"job": Job}` (status `queued`, `position` set).

`kind=variations` → **201** `{"group": Group, "jobs": [Job, …]}` (jobs in seed order, each `kind=create`, `group_id` set).

Errors: 400 validation; 404 unknown `parent_id`/`upload_id`; 409 cover unavailable; 503 models missing.

### `GET /api/jobs?status=&group=&kind=&limit=&offset=`

All query params optional. `status` and `kind` accept comma-separated lists (`status=queued,running`).
`limit` default 50, max 500; `offset` default 0. Ordered by `created_at` **descending**.
→ 200 `{"jobs": [Job, …], "total": int}` (`total` = count matching the filter, ignoring limit/offset).

### `GET /api/jobs/{id}` → 200 `{"job": Job}` | 404.

### `DELETE /api/jobs/{id}` → **204** (removes DB row and `data/songs/<id>/`) | 404 | 409 if `running`.
Deleting a `queued` job cancels it first. Deleting the last member of a group deletes the group too.

### `POST /api/jobs/{id}/cancel`

- `queued` → status becomes `cancelled` immediately → **200** `{"job": Job}`.
- `running` → cancel flag set; status stays `running` until the worker observes it → **202** `{"job": Job}`.
  The SSE stream then delivers `status=cancelled` and `done`.
- terminal → **409**.

### `GET /api/jobs/{id}/events` — Server-Sent Events

Response headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`.

Framing (exact):
```
event: progress
data: {"type":"stage","job_id":"…","stage":"semantic","completed":1200,"total":null,"unit":"tokens","status":"running",…}

: keepalive

event: done
data: {"job": <Job JSON>}
```
- Every `data:` line is a single-line JSON object (no embedded newlines; `\n` escaped inside strings).
- On connect the server first sends the last cached `progress` event for the job (if any) so late subscribers
  see current state, then live events. If the job is already terminal it sends the cached event then `done` and closes.
- Keepalive: a comment line `: keepalive` every **15 s** while idle.
- `event: done` is always the final frame; the server closes the stream afterwards. Client must not reconnect after `done`.
- For a `queued` job the stream stays open (keepalives only) until it starts. Clients use `EventSource` and
  `addEventListener("progress"|"done")`.
- 404 if the job id is unknown.

### Song artifacts — `GET /api/songs/{id}/…`

| Path | Content-Type | Notes |
|------|--------------|-------|
| `audio.flac` | `audio/flac` | supports `Range` (206) for `<audio>` scrubbing |
| `audio.mp3` | `audio/mpeg` | lazily transcoded with ffmpeg into the song dir on first request; 503 if ffmpeg missing |
| `score.abc` | `text/plain; charset=utf-8` | final (or supplied) ABC |
| `plan.json` | `application/json` | engine plan (stage 1 output) |
| `artifacts.zip` | `application/zip` | whole song dir, `Content-Disposition: attachment; filename="<id>.zip"` |
| `transcription/score.abc` | `text/plain; charset=utf-8` | cover jobs only; 404 otherwise |

404 when the job or the file does not exist (e.g. job not yet done). `{id}` is the job id.

### `POST /api/upload`

`multipart/form-data`, single field `file`. Accepted extensions: `mp3 wav flac m4a ogg`; max **200 MB**.
Stored at `data/uploads/<upload_id>.<ext>`.
→ **201** `{"upload_id": "<uuid4 hex>", "filename": "demo.mp3", "seconds": 187.4, "path_hint": "data/uploads/<id>.mp3"}`
(`seconds` null if ffprobe unavailable). 400 bad type, 413 too large.

### `GET /api/settings` → 200 `Settings`.  `PUT /api/settings` body = partial `Settings` → 200 full `Settings` | 400.

### Static

`GET /` → `static/index.html` (also for any non-`/api` path without extension, so hash routing needs nothing extra).
`GET /static/*` → files in `yue2_studio/static/`.

## 4. Frontend routing (hash-based)

Single `index.html`; the router reads `location.hash`:

| Route | View |
|-------|------|
| `#/create` (default) | Create form; `?from=<job_id>` prefills from an existing job ("More variations") |
| `#/queue` | queued + running jobs, live via one `EventSource` per visible job |
| `#/library` | `GET /api/jobs?status=done` (+ filters `kind`, `group`) |
| `#/song/{id}` | song detail (player, score, timing, regenerate) |
| `#/cover` | upload + cover form |
| `#/settings` | settings drawer/page |

The UI polls `GET /api/status` every 5 s and `GET /api/jobs?status=queued,running` every 5 s as a fallback to SSE.

## 5. Flows

Regenerate from score:
```
#/song/{id}  ──GET /api/jobs/{id}, GET /api/songs/{id}/score.abc──▶ render abcjs + editable textarea
      │ user edits ABC, optionally style; clicks "Regenerate from this score"
      ▼
POST /api/jobs {kind:"regenerate", preset:<parent or chosen>,
                params:{parent_id:id, abc:<edited>, style?:…}}   ──▶ 201 {job}  (job.parent_id = id)
      ▼
#/queue  ──GET /api/jobs/{new}/events──▶ progress… ──▶ done ──▶ #/song/{new}
```

Cover:
```
#/cover  ──POST /api/upload (multipart file)──▶ 201 {upload_id, seconds}
      │ user picks task/style/lyrics/seed/preset
      ▼
POST /api/jobs {kind:"cover", params:{upload_id, task, style, lyrics}} ──▶ 201 {job}
      ▼
events: stage=load → stage=transcribe → stage=plan (abc text streams) → semantic → synthesize → decode → save → done
      ▼
#/song/{id}: player + GET /api/songs/{id}/transcription/score.abc rendered beside the result score
```

Variations: `#/create` with N>1 → `POST /api/jobs {kind:"variations", params:{count:N, base:{…}, random_seeds}}`
→ `{group, jobs}`; the queue shows the group label on each card; library filters by `group=<group.id>`.

## 6. Test fixtures — fake engine

`yue2_studio.main.create_app(engine=None, *, home=None) -> FastAPI`. When `engine` is given the worker uses it
instead of constructing `StudioPipeline`; `home` overrides `YUE2_STUDIO_HOME` (tests pass a tmp dir).
`yue2-studio --fake` starts the server with the fake engine so the frontend can be developed without models.

Engine interface (the only methods the worker calls; `FakeEngine` in `tests/fake_engine.py` implements it).
This mirrors the **real** `yue2_studio/engine.py`:

```python
class Engine(Protocol):
    state: str                       # "cold" | "loading" | "ready" | "busy"
    precision: str | None            # current pipeline precision, None when cold
    pipeline: Any | None             # resident StudioPipeline (None when cold)

    def ensure(self, options: EngineOptions, on_event: Callable[[dict], None] | None = None) -> Any:
        """Build the pipeline lazily; rebuild (close + construct) when precision / memory_budget_gib /
        require_ac change (``EngineOptions.build_key``). ode_steps is per job. Cheap: weights load lazily
        inside the first job, surfacing as "Loading … model" stage events. Returns the pipeline."""

    def create_song(self, request: dict, out_dir: Path, *, options: EngineOptions,
                    on_event: Callable[[dict], None] | None = None,
                    cancelled: Callable[[], bool] | None = None) -> dict:
        """request = YuE2 request dict: style, lyrics, cot, seed, abc?, cfg_scale?, id?, plus optional
        abc_sampling / semantic_sampling / generation_config overrides (title etc. must be stripped
        by the worker). Raises InterruptedError when cancelled() is observed (any stage);
        any other exception -> job failed AND the engine unloads its pipeline (the GPU guard's
        latched error would otherwise poison every later job); the next ensure() rebuilds it."""

    def cover_song(self, audio_path: Path, out_dir: Path, *, task: str, request: dict,
                   options: EngineOptions, on_event=None, cancelled=None) -> dict:
        """task in {melody-full, melody-vocal, full}; request as above without abc (cot is forced to
        melody/full to match task). Transcribes first (stage "Transcribing audio", then releases and
        lazily reloads the song models), then runs create_song with the transcribed ABC. Same return
        plus "transcription" (below) and timing.transcription_seconds. State stays "busy" throughout."""

    def memory_footprint(self) -> dict:   # bytes: rss_bytes, system_available_bytes, mlx_active_bytes,
                                          # mlx_cache_bytes, mlx_peak_bytes (worker converts to GiB)
    def unload(self) -> None              # close pipeline, release GPU guard; state -> cold
```

`EngineOptions` (`yue2_studio.config`): frozen dataclass `{precision, ode_steps, memory_budget_gib, require_ac,
preset}` produced by `config.resolve_preset(name, precision=None, ode_steps=None, *, memory_budget_gib, require_ac)`.

**Artifact layout written by the engine** (`out_dir` = `data/songs/<job_id>/`):

| Path | Written | Contents |
|------|---------|----------|
| `plan/` | right after planning, before semantic generation | `plan.json`, `score.abc` (absent when `cot=off`), `abc_tokens.npy`, `prefix.npy`, `plan_manifest.json` |
| `song/` | at the end (`SongResult.save_artifacts`, needs an empty dir) | `audio.flac` (48 kHz stereo 24-bit), `result.json` (`status: "complete"`), `score.abc`, `plan.json`, `request.json`, `config.json`, `semantic.npy`, `latent.npy`, `noise.npy`, … |
| `summary.json` | at the end | the dict returned by `create_song` / `cover_song` |
| `transcription/` | covers only, before the song stages | `score.abc`, `result.json`, `melody.mid`, `*.lab`, `events.json`, … |
| `request.json`, `cover.json` | covers only | resolved request incl. transcribed `abc`; cover receipt |

The HTTP routes therefore map `audio.flac` → `song/audio.flac`, `score.abc` → `song/score.abc` (fall back to
`plan/score.abc` for a failed job), `plan.json` → `plan/plan.json`, `transcription/score.abc` → same path.

**`create_song` return dict** (also `summary.json`):

```json
{"status": "complete", "audio_path": "/abs/.../song/audio.flac", "score_path": "/abs/.../song/score.abc" | null,
 "song_dir": "...", "plan_dir": "...", "sample_rate": 48000, "seconds": 184.7,
 "timing": {"abc": {"seconds": 15.8, "output_tokens": 2072, "prefill_seconds": …, "ttft_seconds": …, "content_tokens": …},
            "semantic": {"seconds": 35.9, "output_tokens": 4618, …},
            "nar_seconds": 27.2, "vae_seconds": 3.4, "e2e_seconds": 82.4,
            "load": {"ar_load_seconds": 0.09, "conditioning_load_seconds": 0.11, "nar_load_seconds": 0.07, "vae_load_seconds": 0.11, …},
            "stages": {"Planning score": 15.8, "Generating song": 35.9, "Synthesizing audio": 27.0, "Decoding audio": 3.2, …},
            "transcription_seconds": 4.1},
 "truncated": {"abc": false, "semantic": true},
 "identity": "<sha256>", "preset": "fast", "precision": "8bit", "ode_steps": 8, "seed": 12300,
 "transcription": {"dir": "...", "task": "melody-full", "seconds": 4.1, "source_audio_sha256": "…", "duration_seconds": 16.0}}
```

`timing.abc` is `{"seconds": 0.0, "output_tokens": 0, "external_prefix_tokens": N}` when the ABC was supplied;
`transcription*` keys exist only for covers. `truncated` is always `{"abc": bool, "semantic": bool}`.
The worker derives the HTTP `Job.timing` (`{"plan": 15.8, "semantic": 35.9, "synthesize": 27.2, "decode": 3.4,
"transcribe": 4.1, "e2e": 82.4, "abc_tps": 131.1, "semantic_tps": 128.8, "audio_seconds": 184.7}`) and `Job.truncated`
(`{"phase": "semantic", "reason": "generation limit reached"}` for the first true flag, else null) from it.

**Raw engine events** (`on_event(dict)`; keys absent when null, no `job_id`):

| `type` | Keys | Notes |
|--------|------|-------|
| `stage` | `stage` (upstream human label), `completed`, `total?`, `unit?`, `status`, `tps?`, `seconds`, `ts` (unix float) | `status ∈ running\|completed\|failed\|cancelled\|truncated`. Labels: `Verifying model files`, `Loading {bf16\|8bit\|4bit} AR model`, `Loading BF16 acoustic conditioning`, `Loading acoustic model`, `Loading MLX audio decoder`, `Using provided score`, `Planning score` (unit tokens), `Generating song` (tokens), `Synthesizing audio` (steps), `Decoding audio` (chunks), `Transcribing audio` (windows) |
| `token` | `phase` (`abc`\|`semantic`\|`transcription`), `tokens`, `tps?`, `seconds`, `ts` | ≤4 Hz |
| `abc` | `phase="abc"`, `text`, `tokens`, `status` (`partial`\|`final`), `ts` | `partial` ≤4 Hz while planning; one `final` with the complete score after planning (also for supplied ABC) |
| `log` | `text`, `ts` | e.g. `"Completed 184.7s of audio in 82.4s"` |

**Engine → HTTP normalisation (the worker must implement):**

| Engine | HTTP |
|--------|------|
| `stage` label → `stage` key | `Verifying model files`, `Loading …` → `load`; `Transcribing audio` → `transcribe`; `Using provided score`, `Planning score` → `plan`; `Generating song` → `semantic`; `Synthesizing audio` → `synthesize`; `Decoding audio` → `decode`; the worker itself emits `save` around `save_artifacts`/DB write. Original label kept in `label`. |
| `status: completed` | `complete` (`running`/`failed`/`cancelled`/`truncated` pass through) |
| `abc.status: partial\|final` | `partial: true\|false` |
| `log.text` | `message` |
| `ts` (unix float) | ISO-8601 `Z` string; add `job_id` |
| absent keys | `null` (every ProgressEvent field present) |

`on_event` receives raw engine dicts; the worker adds `job_id`/`ts` after normalising. `FakeEngine` emits the same
raw engine shape (labels above) — load, plan (3 `abc` events with growing text, then `final`), semantic (token
events), synthesize, decode — with a configurable delay per event (default 0.05 s; `--fake` uses 0.3 s), honours
`cancelled()` between events (raising `InterruptedError`), writes a ~1 s silent stereo 48 kHz FLAC as
`song/audio.flac`, a minimal ABC (`X:1\nT:Fake\nK:C\nCDEF|`) as `plan/score.abc` and `song/score.abc`,
`{"fake": true}` as `plan/plan.json` and `song/result.json` (`status: "complete"`), `summary.json`, and for covers a
`transcription/score.abc`; returns the summary dict above. `FakeEngine(fail=True)` raises in `synthesize` to
exercise the `failed` path.
