"""Job model, request validation and the SQLite job store.

Job kinds stored in the database are ``create`` / ``regenerate`` / ``cover``; ``variations`` is
expanded on submit into ``create`` jobs sharing a ``group_id``. ``Job.to_api`` produces the
exact ``Job`` JSON object from ``docs/API.md`` (every optional field present, ``null`` when unset).

The store keeps one SQLite connection (WAL) guarded by a lock so the worker thread and the
request handlers can share it.
"""

from __future__ import annotations

import contextlib
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

from yue2_studio import config, hum, lora

STATUSES = ("queued", "running", "done", "failed", "cancelled")
TERMINAL = frozenset({"done", "failed", "cancelled"})
KINDS = ("create", "regenerate", "cover", "hum")
SUBMIT_KINDS = ("create", "regenerate", "cover", "hum", "variations")
COVER_TASKS = ("melody-full", "melody-vocal", "full")
COT_MODES = ("full", "melody", "off")
MAX_SEED = 2**31 - 1

Status = Literal["queued", "running", "done", "failed", "cancelled"]
Kind = Literal["create", "regenerate", "cover", "hum"]


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


class Conflict(Exception):
    """Raised when a project/track/take write contradicts the current state; API maps it to 409."""


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


class HumParams(_Params):
    """Hum-to-song: an uploaded/recorded hum, style + lyrics, and the ``yue2_studio.hum.HumOptions``."""

    upload_id: str
    style: str
    lyrics: str
    seed: int | None = None
    title: str | None = None
    melody: Literal["continue", "hum_only", "ignore"] = "continue"
    adapter: str | None = None
    hum_influence: float = 1.0
    offset_s: float = 0.0

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

    @field_validator("adapter")
    @classmethod
    def _adapter(cls, v):
        if v is not None and not lora.valid_name(v):
            raise ValueError("must be an adapter name (letters, digits, . _ -)")
        return v

    @field_validator("hum_influence")
    @classmethod
    def _influence(cls, v):
        if not hum.MIN_INFLUENCE <= v <= hum.MAX_INFLUENCE:
            raise ValueError(f"must be in [{hum.MIN_INFLUENCE:g}, {hum.MAX_INFLUENCE:g}]")
        return v

    @field_validator("offset_s")
    @classmethod
    def _offset(cls, v):
        if not 0 <= v <= hum.MAX_OFFSET_S:
            raise ValueError(f"must be in [0, {hum.MAX_OFFSET_S:g}]")
        return v

    def options(self) -> hum.HumOptions:
        return hum.HumOptions(melody=self.melody, adapter=self.adapter, influence=self.hum_influence,
                              offset_s=self.offset_s)

    def check(self) -> None:
        try:
            self.options()
        except ValueError as error:
            raise ValidationFailure(str(error)) from None


class VariationsParams(_Params):
    count: int = Field(ge=2, le=16)
    base: CreateParams
    random_seeds: bool = False
    label: str | None = None


class LoraRef(_Params):
    name: str
    scale: float = 1.0

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        if not lora.valid_name(v):
            raise ValueError("must be an adapter name (letters, digits, . _ -)")
        return v

    @field_validator("scale")
    @classmethod
    def _scale(cls, v):
        return lora.check_scale(v)


class SubmitRequest(_Params):
    kind: Literal["create", "regenerate", "cover", "hum", "variations"]
    params: dict[str, Any]
    preset: Literal["quality", "fast", "custom"] | None = None
    precision: Literal["bf16", "8bit", "4bit"] | None = None
    ode_steps: int | None = None
    loras: list[LoraRef] | None = None  # omitted => none (create/cover) or inherited (regenerate)
    track_id: str | None = None  # attach every created job to this project track as a take

    @field_validator("track_id")
    @classmethod
    def _track(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("track_id must be a string")
        return v.strip() or None

    @field_validator("ode_steps")
    @classmethod
    def _steps(cls, v):
        if v is not None and not config.MIN_ODE_STEPS <= v <= config.MAX_ODE_STEPS:
            raise ValueError(f"ode_steps must be in [{config.MIN_ODE_STEPS}, {config.MAX_ODE_STEPS}]")
        return v

    @field_validator("loras")
    @classmethod
    def _loras(cls, v):
        if v is None:
            return None
        if len(v) > lora.MAX_STACK:
            raise ValueError(f"at most {lora.MAX_STACK} adapters")
        names = [item.name for item in v]
        if len(set(names)) != len(names):
            raise ValueError("an adapter is listed twice")
        return v

    @property
    def lora_stack(self) -> config.LoraStack | None:
        return None if self.loras is None else tuple((item.name, item.scale) for item in self.loras)


class SettingsModel(_Params):
    default_preset: Literal["quality", "fast", "custom"] = "quality"
    memory_budget_gib: float = Field(default=config.DEFAULT_MEMORY_BUDGET_GIB, ge=6, le=44)
    require_ac: bool = config.DEFAULT_REQUIRE_AC
    theme: Literal["system", "light", "dark"] = "system"
    # Auto-delete uploads no job references after this many days; None = never.
    prune_uploads_days: int | None = Field(default=None, ge=1, le=365)

    @field_validator("prune_uploads_days", mode="before")
    @classmethod
    def _days(cls, v):
        if isinstance(v, bool):
            raise ValueError("must be an integer number of days or null")
        return v


DEFAULT_SETTINGS = SettingsModel().model_dump()


# -- projects / tracks / takes request bodies (partial patches use ``model_fields_set``) --------


def _clean_name(value, name: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    value = " ".join(value.split())
    if len(value) > 200:
        raise ValueError(f"{name} must be at most 200 characters")
    return value


def _no_bool(value):
    if isinstance(value, bool):
        raise ValueError("must be an integer or null")
    return value


class ProjectBody(_Params):
    name: str
    description: str = Field(default="", max_length=2000)

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return _clean_name(v)


class ProjectPatch(_Params):
    name: str | None = None
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return None if v is None else _clean_name(v)


class TrackBody(_Params):
    name: str

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return _clean_name(v)


class TrackPatch(_Params):
    name: str | None = None
    chosen_job_id: str | None = None
    position: int | None = Field(default=None, ge=0)

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        return None if v is None else _clean_name(v)

    @field_validator("position", mode="before")
    @classmethod
    def _position(cls, v):
        return _no_bool(v)


class OrderBody(_Params):
    track_ids: list[str]


class AttachBody(_Params):
    job_ids: list[str] = Field(min_length=1)
    move: bool = False


class TakePatch(_Params):
    thumb: Literal[-1, 0, 1] | None = None  # 1 = thumbs up, -1 = thumbs down, 0/null = cleared
    stars: int | None = Field(default=None, ge=1, le=5)
    note: str | None = Field(default=None, max_length=4000)

    @field_validator("thumb", "stars", mode="before")
    @classmethod
    def _ints(cls, v):
        return _no_bool(v)


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
    loras: config.LoraStack = ()
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
                                                     "transcription": False, "hum": False})
    position: int | None = None
    take: dict | None = None  # project-track membership (see ``JobStore._take_of``), null when none

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
            loras=self.loras,
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
            "loras": config.loras_to_api(self.loras),
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
            "take": None if self.take is None else dict(self.take),
        }


def artifacts_for(song_dir: Path) -> dict:
    song_dir = Path(song_dir)
    return {
        "audio": (song_dir / "song" / "audio.flac").is_file(),
        "score": (song_dir / "song" / "score.abc").is_file() or (song_dir / "plan" / "score.abc").is_file(),
        "plan": (song_dir / "plan" / "plan.json").is_file(),
        "transcription": (song_dir / "transcription" / "score.abc").is_file(),
        "hum": (song_dir / "hum" / "hum.abc").is_file(),
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
    seq INTEGER NOT NULL,
    loras_json TEXT
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
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tracks (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    position INTEGER NOT NULL,
    chosen_job_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tracks_project ON tracks(project_id, position);
CREATE TABLE IF NOT EXISTS takes (
    job_id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL,
    thumb INTEGER,
    stars INTEGER,
    note TEXT NOT NULL DEFAULT '',
    added_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS takes_track ON takes(track_id);
"""

# Every job read goes through this join so ``Job.take`` costs no extra query (a job is a take of at
# most one track: ``takes.job_id`` is the primary key).
_JOB_FROM = ("FROM jobs LEFT JOIN takes ON takes.job_id = jobs.id "
             "LEFT JOIN tracks ON tracks.id = takes.track_id "
             "LEFT JOIN projects ON projects.id = tracks.project_id")
_JOB_SELECT = ("SELECT jobs.*, takes.track_id AS take_track_id, takes.thumb AS take_thumb, "
               "takes.stars AS take_stars, takes.note AS take_note, takes.added_at AS take_added_at, "
               "tracks.name AS take_track_name, tracks.project_id AS take_project_id, "
               "projects.name AS take_project_name, (tracks.chosen_job_id = jobs.id) AS take_chosen "
               + _JOB_FROM)
_UNSET = object()

_JOB_COLUMNS = ("id", "kind", "status", "group_id", "parent_id", "params_json", "preset", "precision",
                "ode_steps", "seed", "created_at", "started_at", "finished_at", "error", "timing_json",
                "audio_seconds", "truncated_json", "progress_json", "loras_json")

# Columns added after the first release: (name, SQL type). Older databases gain them on open.
_MIGRATIONS = (("loras_json", "TEXT"),)


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
        present = {row["name"] for row in self._conn.execute("PRAGMA table_info(jobs)")}
        for column, sql_type in _MIGRATIONS:
            if column not in present:
                self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {column} {sql_type}")
        self._seq = self._conn.execute("SELECT COALESCE(MAX(seq), 0) FROM jobs").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextlib.contextmanager
    def _tx(self):
        """One transaction for a multi-statement write (the connection is otherwise autocommit).

        Nested use joins the outer transaction; the lock is held for the whole block.
        """
        with self._lock:
            if self._conn.in_transaction:
                yield self._conn
                return
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    # -- rows <-> Job --------------------------------------------------------------------------

    def _row_to_job(self, row: sqlite3.Row) -> Job:
        job = Job(
            id=row["id"], kind=row["kind"], status=row["status"], group_id=row["group_id"],
            parent_id=row["parent_id"], params=json.loads(row["params_json"]), preset=row["preset"],
            precision=row["precision"], ode_steps=row["ode_steps"], seed=row["seed"],
            created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"],
            error=row["error"], timing=_loads(row["timing_json"]), audio_seconds=row["audio_seconds"],
            truncated=_loads(row["truncated_json"]), progress=_loads(row["progress_json"]), seq=row["seq"],
            loras=config.normalise_loras(_loads(row["loras_json"])),
        )
        if self.songs_dir is not None:
            job.artifacts = artifacts_for(self.songs_dir / job.id)
        if row["take_track_id"] is not None:
            job.take = {
                "track_id": row["take_track_id"], "project_id": row["take_project_id"],
                "track_name": row["take_track_name"], "project_name": row["take_project_name"],
                "thumb": row["take_thumb"], "stars": row["take_stars"], "note": row["take_note"],
                "added_at": row["take_added_at"], "chosen": bool(row["take_chosen"]),
            }
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
            created_at=now_iso(), group_id=group_id, parent_id=parent_id, loras=options.loras,
        )
        with self._lock:
            self._seq += 1
            self._conn.execute(
                "INSERT INTO jobs (id, kind, status, group_id, parent_id, params_json, preset, precision, "
                "ode_steps, seed, created_at, seq, loras_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job.id, job.kind, job.status, job.group_id, job.parent_id, _dumps(job.params), job.preset,
                 job.precision, job.ode_steps, job.seed, job.created_at, self._seq,
                 _dumps(config.loras_to_api(job.loras)) if job.loras else None),
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
            row = self._conn.execute(f"{_JOB_SELECT} WHERE jobs.id = ?", (job_id,)).fetchone()
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
             group: str | None = None, track: str | None = None, project: str | None = None,
             limit: int = 50, offset: int = 0) -> tuple[list[Job], int]:
        """Newest first. Returns ``(jobs, total)`` where ``total`` ignores limit/offset."""
        clauses, args = [], []
        if status:
            values = [status] if isinstance(status, str) else list(status)
            clauses.append(f"jobs.status IN ({','.join('?' * len(values))})")
            args.extend(values)
        if kind:
            values = [kind] if isinstance(kind, str) else list(kind)
            clauses.append(f"jobs.kind IN ({','.join('?' * len(values))})")
            args.extend(values)
        if group:
            clauses.append("jobs.group_id = ?")
            args.append(group)
        if track:
            clauses.append("takes.track_id = ?")
            args.append(track)
        if project:
            clauses.append("tracks.project_id = ?")
            args.append(project)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) {_JOB_FROM}{where}", args).fetchone()[0]
            rows = self._conn.execute(
                f"{_JOB_SELECT}{where} ORDER BY jobs.created_at DESC, jobs.seq DESC LIMIT ? OFFSET ?",
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

    def upload_counts(self) -> dict[str, dict[str, int]]:
        """``{upload_id: {"total": n, "active": m}}`` over every cover/hum job (one query); ``active``
        counts the queued/running ones."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT json_extract(params_json, '$.upload_id') AS upload_id, status, COUNT(*) AS n "
                "FROM jobs WHERE kind IN ('cover', 'hum') GROUP BY upload_id, status").fetchall()
        counts: dict[str, dict[str, int]] = {}
        for r in rows:
            if not r["upload_id"]:
                continue
            entry = counts.setdefault(r["upload_id"], {"total": 0, "active": 0})
            entry["total"] += r["n"]
            if r["status"] in ("queued", "running"):
                entry["active"] += r["n"]
        return counts

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
        """Remove the row; drops the group when this was its last member, detaches it from its track
        (clearing the track's choice when it was the chosen take)."""
        with self._tx():
            row = self._conn.execute("SELECT group_id FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFound(f"job {job_id!r} not found")
            self._detach(job_id)
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            group_id = row["group_id"]
            if group_id is not None:
                remaining = self._conn.execute("SELECT COUNT(*) FROM jobs WHERE group_id = ?",
                                               (group_id,)).fetchone()[0]
                if remaining == 0:
                    self._conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))

    # -- projects / tracks / takes -------------------------------------------------------------
    #
    # A project is an ordered list of named tracks; a job is a *take* of at most one track
    # (``takes.job_id`` is the primary key). All results are plain dicts; takes are ``Job`` objects.

    def _touch_project(self, project_id: str | None) -> None:
        if project_id is not None:
            self._conn.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now_iso(), project_id))

    def _project_row(self, project_id: str) -> sqlite3.Row:
        row = self._conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise NotFound(f"project {project_id!r} not found")
        return row

    def _track_row(self, track_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT tracks.*, projects.name AS project_name FROM tracks "
            "JOIN projects ON projects.id = tracks.project_id WHERE tracks.id = ?", (track_id,)).fetchone()
        if row is None:
            raise NotFound(f"track {track_id!r} not found")
        return row

    def _takes_where(self, where: str, args: tuple) -> list[Job]:
        rows = self._conn.execute(
            f"{_JOB_SELECT} WHERE {where} ORDER BY tracks.position ASC, takes.added_at ASC, jobs.seq ASC",
            args).fetchall()
        jobs = [self._row_to_job(r) for r in rows]
        self._fill_positions(jobs)
        return jobs

    @staticmethod
    def _track_dict(row: sqlite3.Row, takes: list[Job]) -> dict:
        chosen = next((j for j in takes if j.id == row["chosen_job_id"]), None)
        return {"id": row["id"], "project_id": row["project_id"], "project_name": row["project_name"],
                "name": row["name"], "position": row["position"], "chosen_job_id": row["chosen_job_id"],
                "created_at": row["created_at"], "takes": takes, "chosen": chosen}

    def _project_tracks(self, project_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT tracks.*, projects.name AS project_name FROM tracks "
            "JOIN projects ON projects.id = tracks.project_id WHERE tracks.project_id = ? "
            "ORDER BY tracks.position ASC, tracks.created_at ASC", (project_id,)).fetchall()
        by_track: dict[str, list[Job]] = {}
        for job in self._takes_where("tracks.project_id = ?", (project_id,)):
            by_track.setdefault(job.take["track_id"], []).append(job)
        return [self._track_dict(r, by_track.get(r["id"], [])) for r in rows]

    def _repack(self, project_id: str, ordered_ids: list[str]) -> None:
        self._conn.executemany("UPDATE tracks SET position = ? WHERE id = ?",
                               [(i, tid) for i, tid in enumerate(ordered_ids)])

    def _track_ids(self, project_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT id FROM tracks WHERE project_id = ? ORDER BY position ASC, created_at ASC", (project_id,))
        return [r["id"] for r in rows]

    def _detach(self, job_id: str) -> str | None:
        """Drop the take row (if any) and the track's choice of it; returns the track id or None."""
        row = self._conn.execute(
            "SELECT takes.track_id, tracks.project_id FROM takes "
            "LEFT JOIN tracks ON tracks.id = takes.track_id WHERE takes.job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        self._conn.execute("DELETE FROM takes WHERE job_id = ?", (job_id,))
        self._conn.execute("UPDATE tracks SET chosen_job_id = NULL WHERE chosen_job_id = ?", (job_id,))
        self._touch_project(row["project_id"])
        return row["track_id"]

    # projects

    def create_project(self, name: str, description: str = "") -> dict:
        stamp = now_iso()
        project_id = new_id()
        with self._tx():
            self._conn.execute(
                "INSERT INTO projects (id, name, description, created_at, updated_at) VALUES (?,?,?,?,?)",
                (project_id, name, description, stamp, stamp))
        return self.get_project(project_id)

    def get_project(self, project_id: str) -> dict:
        with self._lock:
            row = self._project_row(project_id)
            return {"id": row["id"], "name": row["name"], "description": row["description"],
                    "created_at": row["created_at"], "updated_at": row["updated_at"],
                    "tracks": self._project_tracks(project_id)}

    def list_projects(self) -> list[dict]:
        """Every project (no tracks) with ``track_count``/``chosen_count``, most recently updated first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT projects.*, "
                "(SELECT COUNT(*) FROM tracks WHERE tracks.project_id = projects.id) AS track_count, "
                "(SELECT COUNT(*) FROM tracks WHERE tracks.project_id = projects.id "
                " AND tracks.chosen_job_id IS NOT NULL) AS chosen_count "
                "FROM projects ORDER BY updated_at DESC, rowid DESC").fetchall()
        return [{"id": r["id"], "name": r["name"], "description": r["description"],
                 "created_at": r["created_at"], "updated_at": r["updated_at"],
                 "track_count": r["track_count"], "chosen_count": r["chosen_count"]} for r in rows]

    def update_project(self, project_id: str, *, name: str | None = None,
                       description: str | None = None) -> dict:
        with self._tx():
            self._project_row(project_id)
            if name is not None:
                self._conn.execute("UPDATE projects SET name = ? WHERE id = ?", (name, project_id))
            if description is not None:
                self._conn.execute("UPDATE projects SET description = ? WHERE id = ?",
                                   (description, project_id))
            self._touch_project(project_id)
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> None:
        """Remove the project, its tracks and their take rows; jobs and song dirs are untouched."""
        with self._tx():
            self._project_row(project_id)
            self._conn.execute(
                "DELETE FROM takes WHERE track_id IN (SELECT id FROM tracks WHERE project_id = ?)",
                (project_id,))
            self._conn.execute("DELETE FROM tracks WHERE project_id = ?", (project_id,))
            self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    # tracks

    def create_track(self, project_id: str, name: str) -> dict:
        track_id = new_id()
        with self._tx():
            self._project_row(project_id)
            position = self._conn.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM tracks WHERE project_id = ?",
                (project_id,)).fetchone()[0]
            self._conn.execute(
                "INSERT INTO tracks (id, project_id, name, position, chosen_job_id, created_at) "
                "VALUES (?,?,?,?,NULL,?)", (track_id, project_id, name, position, now_iso()))
            self._touch_project(project_id)
        return self.get_track(track_id)

    def get_track(self, track_id: str) -> dict:
        with self._lock:
            row = self._track_row(track_id)
            return self._track_dict(row, self._takes_where("takes.track_id = ?", (track_id,)))

    def update_track(self, track_id: str, *, name: str | None = None, chosen_job_id=_UNSET,
                     position: int | None = None) -> dict:
        """``chosen_job_id`` must be a ``done`` take of this track (``Conflict``) or ``None`` to clear;
        ``position`` moves the track and re-packs the project's positions to ``0..n-1``."""
        with self._tx():
            row = self._track_row(track_id)
            if name is not None:
                self._conn.execute("UPDATE tracks SET name = ? WHERE id = ?", (name, track_id))
            if chosen_job_id is not _UNSET:
                if chosen_job_id is not None:
                    take = self._conn.execute(
                        "SELECT jobs.status FROM takes JOIN jobs ON jobs.id = takes.job_id "
                        "WHERE takes.job_id = ? AND takes.track_id = ?", (chosen_job_id, track_id)).fetchone()
                    if take is None:
                        raise Conflict(f"job {chosen_job_id!r} is not a take of track {row['name']!r}")
                    if take["status"] != "done":
                        raise Conflict(f"only a finished take can be chosen (job is {take['status']})")
                self._conn.execute("UPDATE tracks SET chosen_job_id = ? WHERE id = ?",
                                   (chosen_job_id, track_id))
            if position is not None:
                ids = self._track_ids(row["project_id"])
                ids.remove(track_id)
                ids.insert(max(0, min(int(position), len(ids))), track_id)
                self._repack(row["project_id"], ids)
            self._touch_project(row["project_id"])
        return self.get_track(track_id)

    def delete_track(self, track_id: str) -> None:
        """Remove the track and its take rows (jobs are kept); remaining positions are re-packed."""
        with self._tx():
            row = self._track_row(track_id)
            self._conn.execute("DELETE FROM takes WHERE track_id = ?", (track_id,))
            self._conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
            self._repack(row["project_id"], self._track_ids(row["project_id"]))
            self._touch_project(row["project_id"])

    def order_tracks(self, project_id: str, track_ids: list[str]) -> dict:
        """Set the tracklist order; ``track_ids`` must be an exact permutation of the project's tracks."""
        with self._tx():
            self._project_row(project_id)
            current = self._track_ids(project_id)
            if len(track_ids) != len(set(track_ids)) or set(track_ids) != set(current):
                raise ValidationFailure("track_ids must list every track of the project exactly once")
            self._repack(project_id, list(track_ids))
            self._touch_project(project_id)
        return self.get_project(project_id)

    # takes

    def attach_takes(self, track_id: str, job_ids: list[str], *, move: bool = False) -> dict:
        """Make ``job_ids`` takes of the track. A job already in this track is left alone; one in
        another track raises ``Conflict`` unless ``move`` (its rating/note survive, the old track's
        choice of it is cleared). Nothing is written unless every job exists."""
        ordered = list(dict.fromkeys(job_ids))
        with self._tx():
            track = self._track_row(track_id)
            plan: list[tuple[str, sqlite3.Row | None]] = []
            for job_id in ordered:
                if self._conn.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is None:
                    raise NotFound(f"job {job_id!r} not found")
                current = self._conn.execute(
                    "SELECT takes.track_id, tracks.name, tracks.project_id FROM takes "
                    "LEFT JOIN tracks ON tracks.id = takes.track_id WHERE takes.job_id = ?",
                    (job_id,)).fetchone()
                if current is not None and current["track_id"] != track_id and not move:
                    where = current["name"] or "another track"
                    raise Conflict(f"job {job_id!r} is already a take of {where}")
                plan.append((job_id, current))
            stamp = now_iso()
            for job_id, current in plan:
                if current is None:
                    self._conn.execute(
                        "INSERT INTO takes (job_id, track_id, thumb, stars, note, added_at) "
                        "VALUES (?,?,NULL,NULL,'',?)", (job_id, track_id, stamp))
                elif current["track_id"] != track_id:
                    self._conn.execute("UPDATE takes SET track_id = ?, added_at = ? WHERE job_id = ?",
                                       (track_id, stamp, job_id))
                    self._conn.execute("UPDATE tracks SET chosen_job_id = NULL WHERE chosen_job_id = ?",
                                       (job_id,))
                    self._touch_project(current["project_id"])
            self._touch_project(track["project_id"])
        return self.get_track(track_id)

    def detach_take(self, job_id: str) -> None:
        with self._tx():
            if self._detach(job_id) is None:
                raise NotFound(f"job {job_id!r} is not a take")

    def update_take(self, job_id: str, *, thumb=_UNSET, stars=_UNSET, note=_UNSET) -> Job:
        """Rate/annotate a take (``thumb`` 1 / -1 / None, ``stars`` 1..5 / None, ``note`` str)."""
        columns, args = [], []
        if thumb is not _UNSET:
            columns.append("thumb = ?")
            args.append(None if not thumb else int(thumb))
        if stars is not _UNSET:
            columns.append("stars = ?")
            args.append(None if stars is None else int(stars))
        if note is not _UNSET:
            columns.append("note = ?")
            args.append("" if note is None else str(note))
        with self._tx():
            row = self._conn.execute(
                "SELECT tracks.project_id FROM takes LEFT JOIN tracks ON tracks.id = takes.track_id "
                "WHERE takes.job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise NotFound(f"job {job_id!r} is not a take")
            if columns:
                self._conn.execute(f"UPDATE takes SET {', '.join(columns)} WHERE job_id = ?", [*args, job_id])
            self._touch_project(row["project_id"])
        return self.get(job_id)

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
                    default_precision: str | None = None, default_ode_steps: int | None = None,
                    default_loras: config.LoraStack = (), lora_lookup=None) -> config.EngineOptions:
    """Engine options for a submit; ``lora_lookup(name) -> bool`` rejects unknown adapters (400)."""
    preset = req.preset or default_preset or settings.get("default_preset", "quality")
    precision, ode_steps = None, None
    if preset == "custom":
        precision = req.precision or default_precision
        ode_steps = req.ode_steps or default_ode_steps
        if precision is None or ode_steps is None:
            raise ValidationFailure("preset=custom requires precision and ode_steps")
    loras = req.lora_stack if req.lora_stack is not None else default_loras
    if lora_lookup is not None:
        unknown = [name for name, _ in loras if not lora_lookup(name)]
        if unknown:
            raise ValidationFailure(f"unknown or unusable LoRA adapter {unknown[0]!r}")
    try:
        return config.resolve_preset(
            preset, precision, ode_steps,
            memory_budget_gib=settings.get("memory_budget_gib", config.DEFAULT_MEMORY_BUDGET_GIB),
            require_ac=settings.get("require_ac", config.DEFAULT_REQUIRE_AC), loras=loras,
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
                 "hum": HumParams, "variations": VariationsParams}


def validate_submit(body: Any) -> SubmitRequest:
    """Shape-check a ``POST /api/jobs`` body without touching the store (400 before any 503/409)."""
    req = parse(SubmitRequest, body)
    params = parse(_PARAM_MODELS[req.kind], req.params)
    if isinstance(params, CreateParams | HumParams):
        params.check()
    elif isinstance(params, VariationsParams):
        params.base.check()
    if req.preset == "custom" and (req.precision is None or req.ode_steps is None):
        raise ValidationFailure("preset=custom requires precision and ode_steps")
    return req


def submit(store: JobStore, body: Any, *, upload_lookup=None, lora_lookup=None,
           hum_adapter_lookup=None) -> Submission:
    """Validate a ``POST /api/jobs`` body and insert the resulting job rows.

    ``upload_lookup(upload_id) -> dict | None`` resolves an upload to ``{"filename": str, ...}``;
    required for covers and hums. ``lora_lookup(name) -> bool`` says whether an adapter exists and is
    usable; ``hum_adapter_lookup(name) -> bool`` the same for hum-to-song adapters. A ``track_id``
    attaches every created job (all variations members) to that project track. Raises
    ``ValidationFailure`` (400) / ``NotFound`` (404).
    """
    req = validate_submit(body)
    if req.track_id is not None:
        store.get_track(req.track_id)  # 404 before any row is written
    submission = _submit(store, req, upload_lookup=upload_lookup, lora_lookup=lora_lookup,
                         hum_adapter_lookup=hum_adapter_lookup)
    if req.track_id is not None:
        store.attach_takes(req.track_id, [job.id for job in submission.jobs])
        submission.jobs = [store.get(job.id) for job in submission.jobs]  # now carrying ``take``
    return submission


def _submit(store: JobStore, req: SubmitRequest, *, upload_lookup, lora_lookup,
            hum_adapter_lookup) -> Submission:
    settings = store.get_settings()

    def options_for(**defaults) -> config.EngineOptions:
        return resolve_options(req, settings, lora_lookup=lora_lookup, **defaults)

    if req.kind == "create":
        p = parse(CreateParams, req.params)
        p.check()
        options = options_for()
        seed = p.seed if p.seed is not None else random_seed()
        job = store.create(kind="create", params=_create_params_dict(p, seed), options=options, seed=seed)
        return Submission([job])

    if req.kind == "variations":
        v = parse(VariationsParams, req.params)
        v.base.check()
        options = options_for()
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
        options = options_for(default_preset=parent.preset, default_precision=parent.precision,
                              default_ode_steps=parent.ode_steps, default_loras=parent.loras)
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
        options = options_for()
        seed = p.seed if p.seed is not None else random_seed()
        title = p.title if p.title is not None and p.title.strip() else None
        if title is None:
            title = Path(upload.get("filename", "")).stem or None
        params = {"upload_id": p.upload_id, "task": p.task, "style": p.style, "lyrics": p.lyrics,
                  "seed": seed, "title": title}
        job = store.create(kind="cover", params=params, options=options, seed=seed)
        return Submission([job])

    if req.kind == "hum":
        p = parse(HumParams, req.params)
        p.check()
        upload = upload_lookup(p.upload_id) if upload_lookup is not None else None
        if upload is None:
            raise NotFound(f"upload {p.upload_id!r} not found")
        if p.adapter is not None and not (hum_adapter_lookup is not None and hum_adapter_lookup(p.adapter)):
            raise ValidationFailure(f"unknown or unusable hum adapter {p.adapter!r}")
        options = options_for()
        seed = p.seed if p.seed is not None else random_seed()
        title = p.title if p.title is not None and p.title.strip() else None
        if title is None:
            title = Path(upload.get("filename", "")).stem or None
        params = {"upload_id": p.upload_id, "style": p.style, "lyrics": p.lyrics, "seed": seed,
                  "title": title, "melody": p.melody, "adapter": p.adapter,
                  "hum_influence": float(p.hum_influence), "offset_s": float(p.offset_s)}
        job = store.create(kind="hum", params=params, options=options, seed=seed)
        return Submission([job])

    raise ValidationFailure(f"unknown kind {req.kind!r}")  # pragma: no cover


def hum_options(job: Job) -> hum.HumOptions:
    """The stored hum parameters of a ``hum`` job as ``HumOptions``."""
    p = job.params
    return hum.HumOptions(melody=p.get("melody", "continue"), adapter=p.get("adapter"),
                          influence=float(p.get("hum_influence", 1.0)),
                          offset_s=float(p.get("offset_s", 0.0)))


def engine_request(job: Job) -> dict:
    """The ``SongRequest`` dict the engine accepts (style, lyrics, cot, seed, abc?, cfg_scale?, id)."""
    p = job.params
    request = {"style": p["style"], "lyrics": p["lyrics"], "cot": p.get("cot", "full"), "seed": job.seed,
               "id": job.id}
    if job.kind == "cover":
        request["cot"] = "full" if p.get("task") == "full" else "melody"
        return request
    if job.kind == "hum":
        request["cot"] = "melody"
        return request
    if p.get("abc"):
        request["abc"] = p["abc"]
    if p.get("cfg_scale") is not None:
        request["cfg_scale"] = float(p["cfg_scale"])
    return request
