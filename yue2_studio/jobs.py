"""Job model, request validation and the SQLite job store.

Job kinds stored in the database are ``create`` / ``regenerate`` / ``cover``; ``variations`` is
expanded on submit into ``create`` jobs sharing a ``group_id``. ``Job.to_api`` produces the
exact ``Job`` JSON object from ``docs/API.md`` (every optional field present, ``null`` when unset).

The store keeps one SQLite connection (WAL) guarded by a lock so the worker thread and the
request handlers can share it.
"""

from __future__ import annotations

import json
import random
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from yue2_studio import config

STATUSES = ("queued", "running", "done", "failed", "cancelled")
TERMINAL = frozenset({"done", "failed", "cancelled"})
KINDS = ("create", "regenerate", "cover")
SUBMIT_KINDS = ("create", "regenerate", "cover", "variations")
COVER_TASKS = ("melody-full", "melody-vocal", "full")
COT_MODES = ("full", "melody", "off")
MAX_SEED = 2**31 - 1

Status = Literal["queued", "running", "done", "failed", "cancelled"]
Kind = Literal["create", "regenerate", "cover"]


def new_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    """ISO-8601 UTC with millisecond precision and a ``Z`` suffix."""
    return iso(datetime.now(UTC).timestamp())


def iso(unix: float) -> str:
    stamp = datetime.fromtimestamp(unix, UTC).strftime("%Y-%m-%dT%H:%M:%S.")
    return f"{stamp}{int(unix * 1000) % 1000:03d}Z"


def random_seed() -> int:
    return random.randint(0, MAX_SEED)


class ValidationFailure(ValueError):
    """Raised by request validation; the API maps it to 400 ``validation_error``."""


class NotFound(LookupError):
    """Raised when a referenced job / upload / group does not exist; API maps it to 404."""


# ---------------------------------------------------------------------------------------------
# Request models (pydantic). Unknown fields are ignored per API.md §1.
# ---------------------------------------------------------------------------------------------


class _Params(BaseModel):
    model_config = ConfigDict(extra="ignore")


def _non_empty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _check_seed(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return value


class CreateParams(_Params):
    style: str
    lyrics: str
    cot: Literal["full", "melody", "off"] = "full"
    seed: int | None = None
    cfg_scale: float | None = None
    abc: str | None = None
    title: str | None = None

    @field_validator("style")
    @classmethod
    def _style(cls, v):
        return _non_empty(v, "style")

    @field_validator("lyrics")
    @classmethod
    def _lyrics(cls, v):
        return _non_empty(v, "lyrics")

    @field_validator("seed")
    @classmethod
    def _seed(cls, v):
        return _check_seed(v)

    @field_validator("cfg_scale")
    @classmethod
    def _cfg(cls, v):
        if v is not None and not 0 <= v <= 20:
            raise ValueError("cfg_scale must be in [0, 20]")
        return v

    @field_validator("abc")
    @classmethod
    def _abc(cls, v):
        return v if v is None or v.strip() else None

    def check(self) -> None:
        if self.abc is not None and self.cot == "off":
            raise ValidationFailure("a supplied abc score requires cot=melody or cot=full")


class RegenerateParams(_Params):
    parent_id: str
    abc: str
    style: str | None = None
    lyrics: str | None = None
    seed: int | None = None
    title: str | None = None

    @field_validator("abc")
    @classmethod
    def _abc(cls, v):
        return _non_empty(v, "abc")

    @field_validator("seed")
    @classmethod
    def _seed(cls, v):
        return _check_seed(v)


class CoverParams(_Params):
    upload_id: str
    task: Literal["melody-full", "melody-vocal", "full"] = "melody-full"
    style: str
    lyrics: str
    seed: int | None = None
    title: str | None = None

    @field_validator("style")
    @classmethod
    def _style(cls, v):
        return _non_empty(v, "style")

    @field_validator("lyrics")
    @classmethod
    def _lyrics(cls, v):
        return _non_empty(v, "lyrics")

    @field_validator("seed")
    @classmethod
    def _seed(cls, v):
        return _check_seed(v)


class VariationsParams(_Params):
    count: int = Field(ge=2, le=16)
    base: CreateParams
    random_seeds: bool = False
    label: str | None = None


class SubmitRequest(_Params):
    kind: Literal["create", "regenerate", "cover", "variations"]
    params: dict[str, Any]
    preset: Literal["quality", "fast", "custom"] | None = None
    precision: Literal["bf16", "8bit", "4bit"] | None = None
    ode_steps: int | None = None

    @field_validator("ode_steps")
    @classmethod
    def _steps(cls, v):
        if v is not None and not config.MIN_ODE_STEPS <= v <= config.MAX_ODE_STEPS:
            raise ValueError(f"ode_steps must be in [{config.MIN_ODE_STEPS}, {config.MAX_ODE_STEPS}]")
        return v


class SettingsModel(_Params):
    default_preset: Literal["quality", "fast", "custom"] = "quality"
    memory_budget_gib: float = Field(default=config.DEFAULT_MEMORY_BUDGET_GIB, ge=4, le=44)
    require_ac: bool = config.DEFAULT_REQUIRE_AC
    theme: Literal["system", "light", "dark"] = "system"


DEFAULT_SETTINGS = SettingsModel().model_dump()


def format_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        loc = ".".join(str(x) for x in item.get("loc", ())) or "body"
        parts.append(f"{loc}: {item.get('msg')}")
    return "; ".join(parts)


def parse(model: type[BaseModel], data: Any):
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise ValidationFailure(format_validation_error(error)) from None


# ---------------------------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: str
    status: str
    preset: str
    precision: str
    ode_steps: int
    seed: int
    params: dict
    created_at: str
    group_id: str | None = None
    parent_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    timing: dict | None = None
    audio_seconds: float | None = None
    truncated: dict | None = None
    progress: dict | None = None
    seq: int = 0
    artifacts: dict = field(default_factory=lambda: {"audio": False, "score": False, "plan": False,
                                                     "transcription": False})
    position: int | None = None

    @property
    def title(self) -> str | None:
        title = self.params.get("title")
        return title if isinstance(title, str) and title.strip() else None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL

    def options(self, settings: dict | None = None) -> config.EngineOptions:
        settings = settings or DEFAULT_SETTINGS
        return config.resolve_preset(
            self.preset, self.precision, self.ode_steps,
            memory_budget_gib=settings.get("memory_budget_gib", config.DEFAULT_MEMORY_BUDGET_GIB),
            require_ac=settings.get("require_ac", config.DEFAULT_REQUIRE_AC),
        )

    def to_api(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "group_id": self.group_id,
            "parent_id": self.parent_id,
            "preset": self.preset,
            "precision": self.precision,
            "ode_steps": self.ode_steps,
            "seed": self.seed,
            "params": self.params,
            "title": self.title,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "timing": self.timing,
            "truncated": self.truncated,
            "progress": self.progress,
            "artifacts": dict(self.artifacts),
            "position": self.position,
            "seq": self.seq,
        }


def artifacts_for(song_dir: Path) -> dict:
    song_dir = Path(song_dir)
    return {
        "audio": (song_dir / "song" / "audio.flac").is_file(),
        "score": (song_dir / "song" / "score.abc").is_file() or (song_dir / "plan" / "score.abc").is_file(),
        "plan": (song_dir / "plan" / "plan.json").is_file(),
        "transcription": (song_dir / "transcription" / "score.abc").is_file(),
    }


# ---------------------------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    group_id TEXT,
    parent_id TEXT,
    params_json TEXT NOT NULL,
    preset TEXT NOT NULL,
    precision TEXT NOT NULL,
    ode_steps INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error TEXT,
    timing_json TEXT,
    audio_seconds REAL,
    truncated_json TEXT,
    progress_json TEXT,
    seq INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_group ON jobs(group_id);
CREATE INDEX IF NOT EXISTS jobs_created ON jobs(created_at DESC, seq DESC);
CREATE TABLE IF NOT EXISTS groups (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_JOB_COLUMNS = ("id", "kind", "status", "group_id", "parent_id", "params_json", "preset", "precision",
                "ode_steps", "seed", "created_at", "started_at", "finished_at", "error", "timing_json",
                "audio_seconds", "truncated_json", "progress_json")


def _loads(text: str | None):
    return None if text is None else json.loads(text)


def _dumps(value) -> str | None:
    return None if value is None else json.dumps(value, ensure_ascii=False)


class JobStore:
    def __init__(self, db_path: Path | str, songs_dir: Path | str | None = None):
        self.db_path = Path(db_path)
        self.songs_dir = None if songs_dir is None else Path(songs_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._seq = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM jobs").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- rows <-> Job --------------------------------------------------------------------------

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        job = Job(
            id=row["id"], kind=row["kind"], status=row["status"], group_id=row["group_id"],
            parent_id=row["parent_id"], params=json.loads(row["params_json"]), preset=row["preset"],
            precision=row["precision"], ode_steps=row["ode_steps"], seed=row["seed"],
            created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"],
            error=row["error"], timing=_loads(row["timing_json"]), audio_seconds=row["audio_seconds"],
            truncated=_loads(row["truncated_json"]), progress=_loads(row["progress_json"]), seq=row["seq"],
        )
        if self.songs_dir is not None:
            job.artifacts = artifacts_for(self.songs_dir / job.id)
        return job

    def _fill_positions(self, jobs: list[Job]) -> None:
        if any(j.status == "queued" for j in jobs):
            order = self.queued_ids()
            index = {job_id: i for i, job_id in enumerate(order)}
            for job in jobs:
                job.position = index.get(job.id) if job.status == "queued" else None

    # -- create --------------------------------------------------------------------------------

    def create(self, *, kind: str, params: dict, options: config.EngineOptions, seed: int,
               group_id: str | None = None, parent_id: str | None = None, job_id: str | None = None) -> Job:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        job = Job(
            id=job_id or new_id(), kind=kind, status="queued", preset=options.preset,
            precision=options.precision, ode_steps=options.ode_steps, seed=seed, params=params,
            created_at=now_iso(), group_id=group_id, parent_id=parent_id,
        )
        with self._lock:
            self._seq += 1
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, group_id, parent_id, params_json, preset, precision, "
                "ode_steps, seed, created_at, seq) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (job.id, job.kind, job.status, job.group_id, job.parent_id, _dumps(job.params), job.preset,
                 job.precision, job.ode_steps, job.seed, job.created_at, self._seq),
            )
        return self.get(job.id)

    def create_group(self, label: str, group_id: str | None = None) -> dict:
        group = {"id": group_id or new_id(), "label": label, "created_at": now_iso()}
        with self._lock:
            self._conn.execute("INSERT INTO groups (id, label, created_at) VALUES (?,?,?)",
                               (group["id"], group["label"], group["created_at"]))
        return self.get_group(group["id"])

    # -- read ----------------------------------------------------------------------------------

    def get(self, job_id: str) -> Job:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFound(f"job {job_id!r} not found")
            job = self._row_to_job(row)
            self._fill_positions([job])
            return job

    def find(self, job_id: str) -> Job | None:
        try:
            return self.get(job_id)
        except NotFound:
            return None

    def get_group(self, group_id: str) -> dict:
        with self._lock:
            row = self._conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
            if row is None:
                raise NotFound(f"group {group_id!r} not found")
            ids = [r["id"] for r in self._conn.execute(
                "SELECT id FROM jobs WHERE group_id = ? ORDER BY seq ASC", (group_id,))]
            return {"id": row["id"], "label": row["label"], "created_at": row["created_at"], "job_ids": ids}

    def list(self, *, status: list[str] | str | None = None, kind: list[str] | str | None = None,
             group: str | None = None, limit: int = 50, offset: int = 0) -> tuple[list[Job], int]:
        """Newest first. Returns ``(jobs, total)`` where ``total`` ignores limit/offset."""
        clauses, args = [], []
        if status:
            values = [status] if isinstance(status, str) else list(status)
            clauses.append(f"status IN ({','.join('?' * len(values))})")
            args.extend(values)
        if kind:
            values = [kind] if isinstance(kind, str) else list(kind)
            clauses.append(f"kind IN ({','.join('?' * len(values))})")
            args.extend(values)
        if group:
            clauses.append("group_id = ?")
            args.append(group)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM jobs{where}", args).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC, seq DESC LIMIT ? OFFSET ?",
                [*args, int(limit), int(offset)],
            ).fetchall()
            jobs = [self._row_to_job(r) for r in rows]
            self._fill_positions(jobs)
        return jobs, total

    def queued_ids(self) -> list[str]:
        with self._lock:
            return [r["id"] for r in self._conn.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY seq ASC")]

    def running_id(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE status = 'running' ORDER BY seq ASC").fetchone()
            return None if row is None else row["id"]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
            return {r["status"]: r["n"] for r in rows}

    # -- update --------------------------------------------------------------------------------

    def _set(self, job_id: str, where_status: str | None = None, **fields) -> bool:
        columns, args = [], []
        for key, value in fields.items():
            if key in ("timing", "truncated", "progress", "params"):
                key = "params_json" if key == "params" else f"{key}_json"
                value = _dumps(value)
            columns.append(f"{key} = ?")
            args.append(value)
        if not columns:
            return True
        sql = f"UPDATE jobs SET {', '.join(columns)} WHERE id = ?"
        args.append(job_id)
        if where_status is not None:
            sql += " AND status = ?"
            args.append(where_status)
        with self._lock:
            cursor = self._conn.execute(sql, args)
            if cursor.rowcount == 0 and where_status is None:
                raise NotFound(f"job {job_id!r} not found")
            return cursor.rowcount > 0

    def update_status(self, job_id: str, status: str, *, expected: str | None = None, **fields) -> bool:
        """Set ``status`` (and any extra columns). With ``expected`` the update is conditional and
        returns False if the job was not in that state (lets cancel/start race safely)."""
        if status not in STATUSES:
            raise ValueError(f"bad status {status!r}")
        if status == "running":
            fields.setdefault("started_at", now_iso())
        if status in TERMINAL:
            fields.setdefault("finished_at", now_iso())
        return self._set(job_id, expected, status=status, **fields)

    def set_progress(self, job_id: str, progress: dict | None) -> None:
        self._set(job_id, progress=progress)

    def set_timing(self, job_id: str, timing: dict | None, *, audio_seconds: float | None = None,
                   truncated: dict | None = None) -> None:
        self._set(job_id, timing=timing, audio_seconds=audio_seconds, truncated=truncated)

    def set_params(self, job_id: str, params: dict) -> None:
        self._set(job_id, params=params)

    # -- delete --------------------------------------------------------------------------------

    def delete(self, job_id: str) -> None:
        """Remove the row; drops the group when this was its last member."""
        with self._lock:
            row = self._conn.execute("SELECT group_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFound(f"job {job_id!r} not found")
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            group_id = row["group_id"]
            if group_id is not None:
                remaining = self._conn.execute("SELECT COUNT(*) FROM jobs WHERE group_id = ?",
                                               (group_id,)).fetchone()[0]
                if remaining == 0:
                    self._conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))

    # -- settings ------------------------------------------------------------------------------

    def get_settings(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        stored = {r["key"]: json.loads(r["value"]) for r in rows}
        merged = {**DEFAULT_SETTINGS, **{k: v for k, v in stored.items() if k in DEFAULT_SETTINGS}}
        try:
            return SettingsModel.model_validate(merged).model_dump()
        except ValidationError:
            return dict(DEFAULT_SETTINGS)

    def update_settings(self, patch: dict) -> dict:
        if not isinstance(patch, dict):
            raise ValidationFailure("settings body must be an object")
        merged = {**self.get_settings(), **{k: v for k, v in patch.items() if k in DEFAULT_SETTINGS}}
        settings = parse(SettingsModel, merged).model_dump()
        with self._lock:
            self._conn.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(k, json.dumps(v)) for k, v in settings.items()],
            )
        return settings


# ---------------------------------------------------------------------------------------------
# Submission: validation + expansion into stored jobs
# ---------------------------------------------------------------------------------------------


@dataclass
class Submission:
    """Result of ``prepare_submission``: jobs to insert (and a group for variations)."""

    jobs: list[Job]
    group: dict | None = None


def resolve_options(req: SubmitRequest, settings: dict, *, default_preset: str | None = None,
                    default_precision: str | None = None,
                    default_ode_steps: int | None = None) -> config.EngineOptions:
    preset = req.preset or default_preset or settings.get("default_preset", "quality")
    precision, ode_steps = None, None
    if preset == "custom":
        precision = req.precision or default_precision
        ode_steps = req.ode_steps or default_ode_steps
        if precision is None or ode_steps is None:
            raise ValidationFailure("preset=custom requires precision and ode_steps")
    try:
        return config.resolve_preset(
            preset, precision, ode_steps,
            memory_budget_gib=settings.get("memory_budget_gib", config.DEFAULT_MEMORY_BUDGET_GIB),
            require_ac=settings.get("require_ac", config.DEFAULT_REQUIRE_AC),
        )
    except ValueError as error:
        raise ValidationFailure(str(error)) from None


def _create_params_dict(p: CreateParams, seed: int) -> dict:
    return {"style": p.style, "lyrics": p.lyrics, "cot": p.cot, "seed": seed, "cfg_scale": p.cfg_scale,
            "abc": p.abc, "title": p.title}


def default_group_label(base: CreateParams, count: int) -> str:
    name = base.title.strip() if base.title and base.title.strip() else base.style.strip()[:40]
    return f"{name} ×{count}"


_PARAM_MODELS = {"create": CreateParams, "regenerate": RegenerateParams, "cover": CoverParams,
                 "variations": VariationsParams}


def validate_submit(body: Any) -> SubmitRequest:
    """Shape-check a ``POST /api/jobs`` body without touching the store (400 before any 503/409)."""
    req = parse(SubmitRequest, body)
    params = parse(_PARAM_MODELS[req.kind], req.params)
    if isinstance(params, CreateParams):
        params.check()
    elif isinstance(params, VariationsParams):
        params.base.check()
    if req.preset == "custom" and (req.precision is None or req.ode_steps is None):
        raise ValidationFailure("preset=custom requires precision and ode_steps")
    return req


def submit(store: JobStore, body: Any, *, upload_lookup=None) -> Submission:
    """Validate a ``POST /api/jobs`` body and insert the resulting job rows.

    ``upload_lookup(upload_id) -> dict | None`` resolves an upload to ``{"filename": str, ...}``;
    required for covers. Raises ``ValidationFailure`` (400) / ``NotFound`` (404).
    """
    req = validate_submit(body)
    settings = store.get_settings()

    if req.kind == "create":
        p = parse(CreateParams, req.params)
        p.check()
        options = resolve_options(req, settings)
        seed = p.seed if p.seed is not None else random_seed()
        job = store.create(kind="create", params=_create_params_dict(p, seed), options=options, seed=seed)
        return Submission([job])

    if req.kind == "variations":
        v = parse(VariationsParams, req.params)
        v.base.check()
        options = resolve_options(req, settings)
        start = v.base.seed if v.base.seed is not None else random_seed()
        seeds = ([random_seed() for _ in range(v.count)] if v.random_seeds
                 else [(start + i) % 2**63 for i in range(v.count)])
        label = v.label.strip() if v.label and v.label.strip() else default_group_label(v.base, v.count)
        group = store.create_group(label)
        jobs = [store.create(kind="create", params=_create_params_dict(v.base, seed), options=options,
                             seed=seed, group_id=group["id"]) for seed in seeds]
        return Submission(jobs, store.get_group(group["id"]))

    if req.kind == "regenerate":
        p = parse(RegenerateParams, req.params)
        parent = store.find(p.parent_id)
        if parent is None:
            raise NotFound(f"parent job {p.parent_id!r} not found")
        pp = parent.params
        cot = pp.get("cot", "full")
        if cot == "off":
            cot = "melody"
        options = resolve_options(req, settings, default_preset=parent.preset,
                                  default_precision=parent.precision, default_ode_steps=parent.ode_steps)
        seed = p.seed if p.seed is not None else parent.seed
        params = {
            "style": p.style if p.style is not None and p.style.strip() else pp.get("style", ""),
            "lyrics": p.lyrics if p.lyrics is not None and p.lyrics.strip() else pp.get("lyrics", ""),
            "cot": cot, "seed": seed, "abc": p.abc,
            "title": p.title if p.title is not None else pp.get("title"),
            "parent_id": parent.id, "cfg_scale": pp.get("cfg_scale"),
        }
        if not params["style"] or not params["lyrics"]:
            raise ValidationFailure("parent job has no style/lyrics to inherit")
        job = store.create(kind="regenerate", params=params, options=options, seed=seed, parent_id=parent.id)
        return Submission([job])

    if req.kind == "cover":
        p = parse(CoverParams, req.params)
        upload = upload_lookup(p.upload_id) if upload_lookup is not None else None
        if upload is None:
            raise NotFound(f"upload {p.upload_id!r} not found")
        options = resolve_options(req, settings)
        seed = p.seed if p.seed is not None else random_seed()
        title = p.title if p.title is not None and p.title.strip() else None
        if title is None:
            title = Path(upload.get("filename", "")).stem or None
        params = {"upload_id": p.upload_id, "task": p.task, "style": p.style, "lyrics": p.lyrics,
                  "seed": seed, "title": title}
        job = store.create(kind="cover", params=params, options=options, seed=seed)
        return Submission([job])

    raise ValidationFailure(f"unknown kind {req.kind!r}")  # pragma: no cover


def engine_request(job: Job) -> dict:
    """The ``SongRequest`` dict the engine accepts (style, lyrics, cot, seed, abc?, cfg_scale?, id)."""
    p = job.params
    request = {"style": p["style"], "lyrics": p["lyrics"], "cot": p.get("cot", "full"), "seed": job.seed,
               "id": job.id}
    if job.kind == "cover":
        request["cot"] = "full" if p.get("task") == "full" else "melody"
        return request
    if p.get("abc"):
        request["abc"] = p["abc"]
    if p.get("cfg_scale") is not None:
        request["cfg_scale"] = float(p["cfg_scale"])
    return request
