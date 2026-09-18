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
| 404  | `not_found`          | unknown job/group/upload/project/track id, missing artifact (e.g. `plan.json` for a failed job), `PATCH/DELETE /api/takes/{id}` for a job that is not a take |
| 409  | `conflict`           | cancel of done/failed/cancelled job; delete of running job; cover when `status.cover.available=false`; attaching a job that is already a take of another track (without `move`); choosing a take that is not `done`; album export with nothing exportable |
| 413  | `too_large`          | upload > 200 MB |
| 503  | `engine_unavailable` | engine failed to load / models missing (`status.models.present=false`); `audio.mp3` or `album.zip?format=mp3` without ffmpeg |
| 500  | `internal_error`     | anything else |

## 2. Data models

Types: `str`, `int`, `float`, `bool`, `[T]` list, `T?` nullable, `enum(a|b)`.

### Job

| Field | Type | Notes |
|-------|------|-------|
| `id` | str | **uuid4 hex, 32 chars, no dashes** (`"3f9a…"`); also the song dir name `data/songs/<id>/` |
| `kind` | enum(create\|regenerate\|cover\|hum) | `variations` is expanded on submit; stored jobs are never `variations` |
| `status` | enum(queued\|running\|done\|failed\|cancelled) | terminal = done/failed/cancelled |
| `group_id` | str? | set for variations members |
| `parent_id` | str? | set for `regenerate` (source job) |
| `preset` | enum(quality\|fast\|custom) | |
| `precision` | enum(bf16\|8bit\|4bit) | resolved from preset |
| `ode_steps` | int | resolved from preset (quality=32, fast=8) |
| `loras` | [LoraRef] | LoRA adapters merged for this job, in order; `[]` when none |
| `seed` | int | the resolved seed (never null) |
| `params` | object | `CreateParams` / `RegenerateParams` / `CoverParams` per `kind`, with defaults filled in |
| `title` | str? | copied from `params.title`, else `null` |
| `created_at` | str | |
| `started_at` | str? | |
| `finished_at` | str? | |
| `error` | str? | set when `failed` |
| `timing` | object? | `{"plan", "semantic", "synthesize", "decode", "transcribe", "e2e": float?, "abc_tps": float?, "semantic_tps": float?, "audio_seconds": float?}`; set when done; every key present, `null` when not applicable (`transcribe` for non-covers, `abc_tps` for a supplied score) |
| `truncated` | object? | `{"phase": "abc"\|"semantic", "reason": str}` if generation hit a length limit; else null |
| `progress` | ProgressEvent? | last event emitted (also for terminal jobs); null if none yet |
| `artifacts` | object | `{"audio": bool, "score": bool, "plan": bool, "transcription": bool, "hum": bool}`; all false until produced |
| `position` | int? | 0-based queue position while `queued`; null otherwise |
| `seq` | int | server-wide insertion counter (strictly increasing); order by it when `created_at` ties (variations members share a millisecond) |
| `take` | Take? | project membership: set while the job is a take of a track (see `Take`), else null |

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
values (`style`, `lyrics`, `cot`, `seed`, `abc`, `title`, `parent_id`, plus `cfg_scale` inherited from the parent).

### CoverParams

| Field | Type | Default / rule |
|-------|------|----------------|
| `upload_id` | str | required; from `POST /api/upload`; 404 if unknown |
| `task` | enum(melody-full\|melody-vocal\|full) | `melody-full` |
| `style` | str | required |
| `lyrics` | str | required |
| `seed` | int? | random |
| `title` | str? | defaults to upload filename stem |

### HumParams

| Field | Type | Default / rule |
|-------|------|----------------|
| `upload_id` | str | required; the hum recording (any accepted upload, incl. `webm`/`m4a` from the in-browser recorder); 404 if unknown |
| `style` | str | required |
| `lyrics` | str | required |
| `seed` | int? | random |
| `title` | str? | defaults to the upload filename stem |
| `melody` | enum(continue\|hum_only\|ignore) | `continue`: the transcribed hum is the *open* start of the score and the planner continues it; `hum_only`: the hum is the whole melody (closed score, like a cover); `ignore`: the planner writes its own score (needs `adapter`) |
| `adapter` | str? | a `kind="hum"` adapter from `Status.hum.adapters` (400 if unknown, unusable, or a plain LoRA); without one only the score continuation runs |
| `hum_influence` | float | 1.0; 0..3; classifier-free guidance on the decoder's hum channel (1 = as trained, 0 = no hum, ≠1 costs ~2× synthesis) |
| `offset_s` | float | 0; 0..600; where the hum's carrier starts inside the song |

Hum jobs always run with `cot=melody`; the transcription is `melody-vocal`. `Job.artifacts.hum` is true once
`hum/hum.abc` (the open score) exists. Hum adapters are rejected in the `loras` stack (400) and plain LoRAs
are rejected as `adapter`.

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
| `kind` | enum(create\|regenerate\|cover\|hum\|variations) | required |
| `params` | object | required, per kind |
| `preset` | enum(quality\|fast\|custom)? | default `settings.default_preset` |
| `precision` | enum(bf16\|8bit\|4bit)? | only honoured when `preset=custom`; required then |
| `ode_steps` | int? | 4..64; only honoured when `preset=custom`; required then |
| `loras` | [LoraRef]? | adapters to merge, in order, at most 8, names unique; omitted = none (`regenerate`: inherited from the parent; send `[]` to clear) |
| `track_id` | str? | attach every created job (all `variations` members) to this project track as a take; 404 `not_found` for an unknown track **before any job is written**. Never inferred from `parent_id`: a regenerate is only a take when the client says so |

### LoraRef

`{"name": str, "scale": float = 1.0}` — `name` is an entry of `Status.loras.adapters` (letters, digits, `. _ -`);
`scale` in `[0, 4]` multiplies the adapter's own baked-in scale (1 = as trained). An unknown or unusable name is a
400 `validation_error` (`"unknown or unusable LoRA adapter '<name>'"`).

### LoraAdapter (`Status.loras.adapters[]`, `GET /api/loras`)

```json
{"name": "ar_lora_inst_v3abc.bf16", "path": "/abs/models/loras/ar_lora_inst_v3abc.bf16.safetensors",
 "format": "safetensors" | "peft", "valid": true, "error": null,
 "rank": 64, "scale": 1.0, "dtype": "BF16", "parts": ["ar"], "ar_modules": 196, "nar_modules": 0,
 "targets": ["mlp.down_proj", "…", "self_attn.q_proj"], "replaced": [],
 "size_bytes": 139502088, "metadata": {"intended_cot": "full", "rank": "64", "lora_scale": "1.0"}}
```

`parts` ⊆ `["ar", "nar"]` says which model the adapter touches (AR planner / acoustic decoder); `replaced` lists
NAR layers the file ships whole (`llm2vae`, `vae2llm`); `metadata` is the safetensors `__metadata__` filtered to
a few informative keys. `kind` is `"lora"` or `"hum"`: a hum-to-song adapter also carries `hum_proj` (count) and
`inject_layers` (NAR layer indices, from its metadata) and is selected as `HumParams.adapter`, never in `loras`.
`valid=false` entries carry the reason in `error` and cannot be submitted.

### Group

`{"id": str (uuid4 hex), "label": str, "created_at": str, "job_ids": [str]}` — `job_ids` in submit order
(= ascending `seq`; equals seed order unless `random_seeds`).

### Project

A project is an ordered tracklist; a **track is a named slot** whose candidate jobs are its **takes**, one of
which can be **chosen** as the final take. A job is a take of **at most one** track.

| Field | Type | Notes |
|-------|------|-------|
| `id` | str | uuid4 hex |
| `name` | str | 1..200 chars, whitespace collapsed |
| `description` | str | ≤ 2000 chars, `""` when none |
| `created_at` | str | |
| `updated_at` | str | bumped by every project/track/take write (including ratings) |
| `tracks` | [Track] | ordered by `position`; **only in `GET /api/projects/{id}` and the responses that return a full project** |
| `track_count` | int | **only in the `GET /api/projects` list** (which omits `tracks`) |
| `chosen_count` | int | list only: tracks with a `chosen_job_id` |

### Track

| Field | Type | Notes |
|-------|------|-------|
| `id` | str | uuid4 hex |
| `project_id` | str | |
| `project_name` | str | denormalised for banners/tags |
| `name` | str | 1..200 chars |
| `position` | int | 0-based, always packed `0..n-1` within the project |
| `chosen_job_id` | str? | the final take; always a `done` take of this track (cleared when that job is deleted, detached or moved) |
| `created_at` | str | |
| `takes` | [Job] | ordered by `take.added_at` then `seq`; every status (queued/running/failed takes are listed as such) |
| `chosen` | Job? | the `takes` entry whose id is `chosen_job_id` |

### Take (`Job.take`)

| Field | Type | Notes |
|-------|------|-------|
| `track_id` | str | |
| `project_id` | str | |
| `track_name` | str | |
| `project_name` | str | |
| `thumb` | int? | `1` 👍, `-1` 👎, `null` none |
| `stars` | int? | 1..5 or `null` |
| `note` | str | `""` when none, ≤ 4000 chars |
| `added_at` | str | when it was attached (reset on `move`) |
| `chosen` | bool | `track.chosen_job_id == job.id` |

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
  "engine":  {"state": "ready", "precision": "bf16", "memory_gib": 11.2, "current_job_id": null,
              "loras": [{"name": "ar_lora_inst_v3abc.bf16", "scale": 1.0}]},
  "queue":   {"queued": 2, "running": null},
  "presets": [
    {"name": "quality", "label": "Quality", "precision": "bf16", "ode_steps": 32, "description": "BF16 AR, 32 ODE steps"},
    {"name": "fast",    "label": "Fast",    "precision": "8bit", "ode_steps": 8,  "description": "8-bit AR, 8 ODE steps"},
    {"name": "custom",  "label": "Custom",  "precision": null,   "ode_steps": null, "description": "Choose precision and steps"}
  ],
  "models":  {"converted_dir": "/abs/models/converted", "vae_dir": "/abs/models/vae", "present": true,
              "precisions": ["bf16", "8bit"]},
  "cover":   {"available": false, "reasons": ["MERT-v2-FullSong not downloaded"]},
  "hum":     {"available": true, "reasons": [], "adapters": ["hum_adapter_v1_combined"]},
  "loras":   {"dir": "/abs/models/loras", "adapters": [LoraAdapter, …]},
  "ffmpeg":  true,
  "fake":    false,
  "version": "0.1.0"
}
```

`engine.state ∈ cold|loading|ready|busy`; `precision`/`memory_gib` null when cold (`memory_gib` is the MLX active
memory sampled by the worker at stage boundaries, so it lags slightly). `queue.running` = job id or null.
`models.precisions` lists the AR weight files present; `fake` is true under `--fake` (then `models.present` and
`cover.available` are reported true so jobs can be submitted). `engine.loras` is the stack merged into the resident
pipeline (`[]` when cold or none); `loras` rescans `models/loras/` on every call (header reads only).

### Settings

`{"default_preset": "quality", "memory_budget_gib": 24, "require_ac": false, "theme": "system", "prune_uploads_days": null}`
(`theme ∈ system|light|dark`, `memory_budget_gib` number 6..44 — mlx-Yue's guard rejects budgets ≤ 5 GiB and requires
total RAM − 4 GiB headroom — returned as a float, e.g. `24.0`; `prune_uploads_days` integer 1..365 or `null` = off:
uploads no job references and older than that are deleted at server startup and after every job finishes, exactly
as `POST /api/uploads/prune {"unused": true, "older_than_days": N}` would). `PUT` accepts any
subset, ignores unknown keys, and returns the full object; a rejected patch (400) changes nothing.

### Upload (`GET /api/uploads`)

| field | type | notes |
|-------|------|-------|
| `upload_id` | str | |
| `filename` | str | original name (the on-disk name for a broken entry without a sidecar) |
| `ext` | str? | stored extension |
| `seconds` | float? | duration from ffprobe at upload time |
| `size` | int? | bytes; `null` when `broken` |
| `created_at` | str | ISO-8601 UTC |
| `broken` | bool | media file or sidecar is missing/unreadable — cannot be submitted, can be deleted |
| `jobs` | `{"total": int, "active": int}` | jobs whose `params.upload_id` is this upload; `active` = queued or running |

## 3. Endpoints

### `GET /api/status` → 200 `Status` (above). Never 503; unavailability is reported in the body.

### `GET /api/loras` → 200 `{"dir": str, "adapters": [LoraAdapter, …]}` — same as `Status.loras`, rescanned.

`Status.hum` reports whether hum-to-song can run (the cover prerequisites plus `librosa`) and the usable hum
adapter names; `POST /api/jobs {kind:"hum"}` answers 409 while it is unavailable.

### `POST /api/jobs`

Body: `{"kind", "params", "preset"?, "precision"?, "ode_steps"?, "loras"?, "track_id"?}`.

```json
{"kind": "create", "preset": "fast",
 "params": {"style": "dreamy indie pop, female vocal", "lyrics": "[verse]\n...\n[chorus]\n...", "cot": "full", "seed": 42}}
```
→ **201** `{"job": Job}` (status `queued`, `position` set).

`kind=variations` → **201** `{"group": Group, "jobs": [Job, …]}` (jobs in submit/`seq` order, each `kind=create`, `group_id` set).

Errors: 400 validation; 404 unknown `parent_id`/`upload_id`/`track_id`; 409 cover unavailable; 503 models missing.
With `track_id` every returned job carries `take` (attached before the response, in submit order).

### `GET /api/jobs?status=&group=&kind=&track=&project=&limit=&offset=`

All query params optional. `status` and `kind` accept comma-separated lists (`status=queued,running`);
`track=<track_id>` / `project=<project_id>` keep only takes of that track / project.
`limit` default 50, max 500; `offset` default 0. Ordered by `created_at` **descending**.
→ 200 `{"jobs": [Job, …], "total": int}` (`total` = count matching the filter, ignoring limit/offset).

### `GET /api/jobs/{id}` → 200 `{"job": Job}` | 404.

### `DELETE /api/jobs/{id}` → **204** (removes DB row and `data/songs/<id>/`) | 404 | 409 if `running`.
Deleting a `queued` job cancels it first. Deleting the last member of a group deletes the group too.

### `POST /api/jobs/{id}/cancel`

- `queued` → status becomes `cancelled` immediately → **200** `{"job": Job}`. Open SSE streams for the job
  receive a `status` event (`status=cancelled`) and `done`.
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
| `artifacts.zip` | `application/zip` | whole song dir as written so far (a running or failed job yields `job.json`, `plan/`, …), `Content-Disposition: attachment; filename="<id>.zip"`; 404 only while the dir is empty (queued) |
| `transcription/score.abc` | `text/plain; charset=utf-8` | cover and hum jobs; 404 otherwise |
| `hum/hum.abc` | `text/plain; charset=utf-8` | hum jobs: the open score fed to the planner (404 for `melody=ignore`) |
| `hum.json` | `application/json` | hum jobs: receipt (melody, adapter, influence, offset, source/transcription/prosody pointers) |

404 when the job or the file does not exist (e.g. job not yet done). `{id}` is the job id. These routes also
answer `HEAD` (players probe with it before requesting ranges).

### Projects — `/api/projects`

Bodies are JSON objects; partial `PATCH` bodies only touch the keys present (`null` is a value, e.g.
`{"chosen_job_id": null}` clears the choice). Every Job inside a project/track response is the live `Job` shape
(running takes carry their last `progress` event). Deleting a project or track never touches jobs or song dirs.

- `GET /api/projects` → 200 `{"projects": [Project, …]}` — list shape (`track_count`, `chosen_count`, no `tracks`),
  most recently updated first.
- `POST /api/projects` body `{"name", "description"?}` → **201** `{"project": Project}` (with `tracks: []`) | 400.
- `GET /api/projects/{id}` → 200 `{"project": Project}` with `tracks[*].takes` / `chosen` | 404.
- `PATCH /api/projects/{id}` body `{"name"?, "description"?}` → 200 `{"project": Project}` | 400 | 404.
- `DELETE /api/projects/{id}` → **204** (tracks and take rows removed; jobs kept) | 404.
- `POST /api/projects/{id}/tracks` body `{"name"}` → **201** `{"track": Track}` (appended, `position` = n) | 400 | 404.
- `PUT /api/projects/{id}/order` body `{"track_ids": [str]}` → 200 `{"project": Project}`; 400 `validation_error`
  unless the list is an exact permutation of the project's track ids (missing, extra or duplicate ids) | 404.
- `GET /api/projects/{id}/album.zip?format=flac|mp3` (default `flac`) → 200 `application/zip`,
  `Content-Disposition: attachment; filename="<project name>.zip"` (RFC 5987 `filename*=` when the name needs it).
  Built fresh per request: `<name>/NN <track name>.<format>` for every track with a chosen take whose audio exists
  (`NN` = 1-based tracklist position, stored uncompressed), plus `<name>/tracklist.json` and `<name>/tracklist.md`.
  Tracks without an exportable take are skipped in the files but listed with `"missing": true`. Errors: 400 bad
  `format`; 404; **409** `conflict` when no track is exportable; **503** `engine_unavailable` for `mp3` without
  ffmpeg (checked before any work; MP3s are transcoded beside each FLAC and cached like `audio.mp3`).

  `tracklist.json`:
  ```json
  {"project": {"id": "…", "name": "Soundtrack", "description": ""}, "format": "flac", "generated_at": "…Z",
   "tracks": [{"n": 1, "track_id": "…", "name": "Main theme", "job_id": "…", "title": "Opening", "file": "01 Main theme.flac",
               "seconds": 187.4, "preset": "quality", "seed": 42, "kind": "create", "missing": false},
              {"n": 2, "track_id": "…", "name": "Credits", "job_id": null, "title": null, "file": null,
               "seconds": null, "preset": null, "seed": null, "kind": null, "missing": true}]}
  ```

### Tracks — `/api/tracks/{id}`

- `GET /api/tracks/{id}` → 200 `{"track": Track, "project": {"id", "name"}}` | 404 (for the `?track=` banners).
- `PATCH /api/tracks/{id}` body `{"name"?, "position"?, "chosen_job_id"?}` → 200 `{"track": Track}` | 400 (`position`
  < 0, blank name) | 404 | **409** `conflict` when `chosen_job_id` is not a take of this track or its job is not `done`.
  `position` moves the track and re-packs the project's positions (out of range = last).
- `DELETE /api/tracks/{id}` → **204** (take rows dropped, jobs kept, remaining positions re-packed) | 404.

### Takes

- `POST /api/tracks/{id}/takes` body `{"job_ids": [str] (≥ 1), "move"?: bool}` → 200 `{"track": Track}`. A job
  already in this track is left alone (rating kept); one in another track answers **409** `conflict`
  (`"job '<id>' is already a take of <track name>"`) unless `move: true`, which moves it keeping `thumb`/`stars`/`note`
  and clears the old track's `chosen_job_id` if it was that job. 404 unknown track or **any** unknown job (nothing is
  written); 400 empty list.
- `DELETE /api/takes/{job_id}` → **204** (detached: the rating row is dropped and the track's choice is cleared
  when it was chosen; the job and its song are kept) | 404 when the job is not a take.
- `PATCH /api/takes/{job_id}` body `{"thumb"?: -1|0|1|null, "stars"?: 1..5|null, "note"?: str}` → 200 `{"job": Job}`
  (`0` and `null` both clear `thumb`; `null` clears `stars`; `note` ≤ 4000 chars) | 400 | 404 not a take.
- `DELETE /api/jobs/{id}` also removes the job's take row and clears any track that had chosen it.

### `POST /api/upload`

`multipart/form-data`, single field `file`. Accepted extensions: `mp3 wav flac m4a ogg webm mp4` (the last two
are what `MediaRecorder` produces in Chrome / Safari); max **200 MB**.
Stored at `data/uploads/<upload_id>.<ext>`.
→ **201** `{"upload_id": "<uuid4 hex>", "filename": "demo.mp3", "seconds": 187.4, "path_hint": "data/uploads/<id>.mp3"}`
(`seconds` null if ffprobe unavailable). 400 bad type, 413 too large. The 200 MB cap is checked against
`Content-Length` before the body is read (browsers always send it for `FormData`); a chunked upload without it is
only rejected while being copied into `data/uploads/`, after the multipart parser has buffered it to a temp file.
- **400** `validation_error` — the file is empty (0 bytes), e.g. an iCloud/Dropbox placeholder that was never downloaded.
- **400** `validation_error` — ffprobe is installed but cannot read the file (unsupported or corrupt audio); nothing is stored.

### Uploads management

Uploads are kept until deleted; a finished cover/hum never reads its upload again (the song dir holds the
transcription), so only a **queued or running** job pins one.

- `GET /api/uploads?unused=` → 200 `{"uploads": [Upload, …]}`, newest first. `unused=true` keeps only uploads with
  `jobs.total == 0`.
- `DELETE /api/uploads/{id}` → **204** (removes the media file and its sidecar) | 404 unknown/malformed id |
  **409** `in_use` while `jobs.active > 0`.
- `POST /api/uploads/prune` body `{"unused": true, "older_than_days": 30}` (both optional; `unused` defaults to true,
  `older_than_days` integer ≥ 1 or `null`) → 200 `{"deleted": n, "skipped": m}`. With `unused` every upload no job
  references is deleted and the referenced ones are `skipped`; with `unused: false` everything without an active
  job goes and only active ones are `skipped`. `older_than_days` restricts either to uploads created before then.
  The `prune_uploads_days` setting runs this automatically (`unused: true`).

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
| `#/library` | `GET /api/jobs?status=done` (+ filters `kind`, `group`, `project`); "Uploads" panel over `GET /api/uploads` with delete / prune; `?attach=<track_id>` opens multi-select with "Add to project" preset to that track; cards show a `Project › Track` tag from `job.take` |
| `#/song/{id}` | song detail (player, score, timing, regenerate); "Project" panel: attach via picker, or rate (`PATCH /api/takes/{id}`), "Choose as final take", Detach; regenerate/variations from a take pass `track_id` |
| `#/cover` | upload + cover form |
| `#/projects` | `GET /api/projects` cards (name, `N tracks · M chosen`, updated) + "New project" form |
| `#/project/{id}` | `GET /api/projects/{id}`: editable name/description, album player over the chosen takes, Export ZIP (FLAC / MP3 when `status.ffmpeg`), draggable tracklist (`PUT …/order`), per-track takes with thumbs/stars/note, Choose, Detach, "New take" → `#/create?track=`, `#/cover?track=`, `#/hum?track=` |
| `#/settings` | settings drawer/page |

`#/create`, `#/cover` and `#/hum` read `?track=<track_id>` (`GET /api/tracks/{id}` for the banner "New take for
Project › Track") and send it as `track_id`; it lives only in that page's state, never in saved form state, so a
later plain submission is not attached.

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
         or pick a recent upload (GET /api/uploads) and skip the upload
      │ user picks task/style/lyrics/seed/preset
      ▼
POST /api/jobs {kind:"cover", params:{upload_id, task, style, lyrics}} ──▶ 201 {job}
      ▼
events: stage=load → stage=transcribe → stage=plan (abc text streams) → semantic → synthesize → decode → save → done
      ▼
#/song/{id}: player + GET /api/songs/{id}/transcription/score.abc rendered beside the result score
```

Hum to song:
```
#/hum    ──POST /api/upload (drop zone, or MediaRecorder blob named hum-<stamp>.m4a|webm|ogg)──▶ 201 {upload_id}
         or pick a recent upload (GET /api/uploads) and skip the upload
      │ user picks melody (continue | hum_only | ignore), optional adapter + influence + offset, style/lyrics/seed/preset
      ▼
POST /api/jobs {kind:"hum", params:{upload_id, style, lyrics, melody, adapter, hum_influence, offset_s}} ──▶ 201 {job}
      ▼
events: load → transcribe (unless ignore) → hum ("Analysing hum", "Encoding hum"; adapter only) → plan (the streamed
        abc text starts with the hum's open score) → semantic → synthesize (hum-conditioned with an adapter) → decode → save
      ▼
#/song/{id}: continued score + GET /api/songs/{id}/hum/hum.abc ("Your hum") + GET /api/songs/{id}/hum.json
```

Variations: `#/create` with N>1 → `POST /api/jobs {kind:"variations", params:{count:N, base:{…}, random_seeds}}`
→ `{group, jobs}`; the queue shows the group label on each card; library filters by `group=<group.id>`.

Album from takes:
```
#/projects ──POST /api/projects {name}──▶ #/project/{id} ──POST /api/projects/{id}/tracks {name}──▶ tracks
      │ "New take" on a track
      ▼
#/create?track=<track_id> ──GET /api/tracks/{id} (banner)──▶ POST /api/jobs {…, track_id} ──▶ 201 (job.take set)
      │ or Library multi-select "Add to project…" ──▶ POST /api/tracks/{id}/takes {job_ids} (409 → confirm → move:true)
      ▼
#/project/{id}: rate takes ──PATCH /api/takes/{job_id} {thumb, stars, note}──▶ pick one ──PATCH /api/tracks/{id}
                {chosen_job_id}──▶ reorder ──PUT /api/projects/{id}/order──▶ play chosen takes in order
      ▼
GET /api/projects/{id}/album.zip?format=flac|mp3 ──▶ <name>/01 Track.flac … + tracklist.json + tracklist.md
```

## 6. Test fixtures — fake engine

`yue2_studio.main.create_app(engine=None, *, home=None, fake=False, fake_delay=0.05, static_dir=None) -> FastAPI`.
When `engine` is given the worker uses it instead of constructing `StudioPipeline`; `fake=True` builds a
`FakeEngine(delay=fake_delay)`; `home` overrides `YUE2_STUDIO_HOME` (tests pass a tmp dir; it is applied through a
`config.Paths` object on `app.state.paths`, module constants are untouched); `static_dir` overrides
`yue2_studio/static`. The real engine module (which imports mlx) is only imported when neither `engine` nor
`fake` is given. `yue2-studio --fake` starts the server with the fake engine (0.3 s per event) so the frontend
can be developed without models.

Engine interface (the only methods the worker calls; `FakeEngine` in `yue2_studio/fake.py` implements it,
`FakeEngine(delay=0.05, fail=False)`). This mirrors the **real** `yue2_studio/engine.py`:

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
preset, loras}` produced by `config.resolve_preset(name, precision=None, ode_steps=None, *, memory_budget_gib,
require_ac, loras=None)`. `loras` is `((name, scale), …)`; it never changes `build_key`. The real engine resolves
each name in `models/loras/` (`yue2_studio.lora.find_adapter`) before touching the GPU, then
`StudioPipeline.set_loras` merges the stack into the AR / NAR weights as they load (a different stack drops the
resident models first; they reload from the memory-mapped files). Merges show up as `"Merging LoRA into … model"`
stage events (HTTP `stage="load"`) and the adapters (name, sha256, scale) are recorded in `pipe.weights["loras"]`,
hence in `result.json` and the request identity.

**Artifact layout written by the engine** (`out_dir` = `data/songs/<job_id>/`):

| Path | Written | Contents |
|------|---------|----------|
| `plan/` | right after planning, before semantic generation | `plan.json`, `score.abc` (absent when `cot=off`), `abc_tokens.npy`, `prefix.npy`, `plan_manifest.json` |
| `song/` | at the end (`SongResult.save_artifacts`, needs an empty dir) | `audio.flac` (48 kHz stereo 24-bit), `result.json` (`status: "complete"`), `score.abc`, `plan.json`, `request.json`, `config.json`, `semantic.npy`, `latent.npy`, `noise.npy`, … |
| `summary.json` | at the end | the dict returned by `create_song` / `cover_song` |
| `transcription/` | covers and hums, before the song stages | `score.abc`, `result.json`, `melody.mid`, `*.lab`, `events.json`, … |
| `hum/`, `hum.json` | hums only | `hum.abc` (open score fed to the planner), `carrier.flac` / `carrier_latents.npy` / `prosody.json` (adapter only); `hum.json` receipt (melody, adapter, influence, offset, hashes) |
| `request.json`, `cover.json` | covers only | resolved request incl. transcribed `abc`; cover receipt |
| `job.json` | by the worker, at job start | the stored job (id, kind, params, preset/precision/ode_steps, loras, seed, ids, created_at) |
| `audio.mp3` (in `song/`), `artifacts.zip` | lazily by the HTTP routes | cached MP3 transcode; zip of the song dir (excluded from itself) |

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
 "loras": [{"name": "ar_lora_inst_v3abc.bf16", "scale": 1.0}],
 "transcription": {"dir": "...", "task": "melody-full", "seconds": 4.1, "source_audio_sha256": "…", "duration_seconds": 16.0}}
```

`timing.abc` is `{"seconds": 0.0, "output_tokens": 0, "external_prefix_tokens": N}` when the ABC was supplied;
`transcription*` keys exist only for covers. `truncated` is always `{"abc": bool, "semantic": bool}`.
The worker derives the HTTP `Job.timing` (`{"plan": 15.8, "semantic": 35.9, "synthesize": 27.2, "decode": 3.4,
"transcribe": 4.1, "e2e": 82.4, "abc_tps": 131.1, "semantic_tps": 128.8, "audio_seconds": 184.7}`) and `Job.truncated`
(`{"phase": "semantic", "reason": "generation limit reached"}` for the first true flag, else null) from it.

Worker behaviour worth knowing: jobs run strictly serially in submit order; the worker emits the `status`
`stage="load"` event whenever the engine is cold or the requested precision differs from the resident one; a
non-cancellation engine error unloads the engine (state `cold`) and the next job rebuilds it. Shutdown (incl.
`--reload`) cancels only the running job and leaves queued rows `queued`; on server start any row still `queued`
is re-enqueued and any row left `running` is marked `failed` ("server restarted…").

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
