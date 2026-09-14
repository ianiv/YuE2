"""Worker thread, cancellation and the SSE event bus.

* ``Worker`` owns the only thread that touches the engine. Jobs are handed over as ids through a
  ``queue.Queue``; each job has a ``threading.Event`` that backs the engine's ``cancelled()``.
* ``EventBus`` fans events out to per-job ``asyncio.Queue`` subscribers on the server's event loop
  (``loop.call_soon_threadsafe``) and caches the last progress event per job for late subscribers.
* ``normalise`` converts raw engine events into the HTTP ``ProgressEvent`` shape (API.md §6).
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import shutil
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from yue2_studio import config
from yue2_studio.jobs import Job, JobStore, engine_request, iso

log = logging.getLogger("yue2_studio.worker")


class Engine(Protocol):
    """The engine surface the worker relies on (``yue2_studio.engine.Engine`` and ``FakeEngine``)."""

    state: str
    precision: str | None
    pipeline: Any | None

    def ensure(self, options: config.EngineOptions,
               on_event: Callable[[dict], None] | None = None) -> Any: ...

    def create_song(self, request: dict, out_dir: Path, *, options: config.EngineOptions,
                    on_event: Callable[[dict], None] | None = None,
                    cancelled: Callable[[], bool] | None = None) -> dict: ...

    def cover_song(self, audio_path: Path, out_dir: Path, *, task: str, request: dict,
                   options: config.EngineOptions, on_event: Callable[[dict], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> dict: ...

    def memory_footprint(self) -> dict: ...

    def unload(self) -> None: ...


# ---------------------------------------------------------------------------------------------
# Normalisation: raw engine event -> HTTP ProgressEvent
# ---------------------------------------------------------------------------------------------

EVENT_FIELDS = ("type", "job_id", "ts", "stage", "label", "completed", "total", "unit", "status", "phase",
                "tokens", "tps", "seconds", "text", "partial", "message")

_STAGE_KEYS = {
    "verifying model files": "load",
    "transcribing audio": "transcribe",
    "using provided score": "plan",
    "planning score": "plan",
    "generating song": "semantic",
    "synthesizing audio": "synthesize",
    "decoding audio": "decode",
    "saving artifacts": "save",
}
_STATUS_MAP = {"completed": "complete"}


def stage_key(label: str | None) -> str | None:
    if label is None:
        return None
    text = label.strip().lower()
    if text in _STAGE_KEYS:
        return _STAGE_KEYS[text]
    if text.startswith("loading") or text.startswith("verifying"):
        return "load"
    if "transcri" in text:
        return "transcribe"
    return text.split()[0] if text else None


def _ts(value) -> str:
    if isinstance(value, int | float):
        return iso(float(value))
    if isinstance(value, str) and value:
        return value
    return iso(time.time())


def normalise(raw: dict, job_id: str) -> dict:
    """Raw engine event (API.md §6 table) -> HTTP ``ProgressEvent`` with every field present."""
    kind = raw.get("type")
    event = dict.fromkeys(EVENT_FIELDS)
    event.update(type=kind, job_id=job_id, ts=_ts(raw.get("ts")))
    if kind == "stage":
        label = raw.get("stage")
        event.update(stage=stage_key(label), label=label, completed=raw.get("completed"),
                     total=raw.get("total"), unit=raw.get("unit"),
                     status=_STATUS_MAP.get(raw.get("status"), raw.get("status")),
                     tps=raw.get("tps"), seconds=raw.get("seconds"))
    elif kind == "token":
        event.update(phase=raw.get("phase"), tokens=raw.get("tokens"), tps=raw.get("tps"),
                     seconds=raw.get("seconds"))
    elif kind == "abc":
        status = raw.get("status")
        event.update(phase=raw.get("phase", "abc"), tokens=raw.get("tokens"), text=raw.get("text"),
                     partial=None if status is None else status != "final")
    elif kind == "log":
        event.update(message=raw.get("text", raw.get("message")))
    elif kind == "status":
        event.update(stage=raw.get("stage"), status=raw.get("status"), message=raw.get("message"))
    else:  # unknown type: keep it visible rather than dropping it
        event.update(type="log", message=json.dumps(raw, default=str))
    return event


def status_event(job_id: str, status: str, message: str, *, stage: str | None = None) -> dict:
    return normalise({"type": "status", "status": status, "message": message, "stage": stage,
                      "ts": time.time()}, job_id)


def derive_timing(summary: dict) -> dict:
    """``Job.timing`` from the engine summary (API.md §6)."""
    t = summary.get("timing") or {}
    abc, semantic = t.get("abc") or {}, t.get("semantic") or {}

    def tps(section: dict) -> float | None:
        seconds, tokens = section.get("seconds"), section.get("output_tokens")
        if seconds and tokens:
            return round(tokens / seconds, 2)
        return None

    def secs(value) -> float | None:
        return None if value is None else round(float(value), 3)

    return {
        "plan": secs(abc.get("seconds")),
        "semantic": secs(semantic.get("seconds")),
        "synthesize": secs(t.get("nar_seconds")),
        "decode": secs(t.get("vae_seconds")),
        "transcribe": secs(t.get("transcription_seconds")),
        "e2e": secs(t.get("e2e_seconds")),
        "abc_tps": tps(abc),
        "semantic_tps": tps(semantic),
        "audio_seconds": secs(summary.get("seconds")),
    }


def derive_truncated(summary: dict) -> dict | None:
    truncated = summary.get("truncated") or {}
    for phase in ("abc", "semantic"):
        if truncated.get(phase):
            return {"phase": phase, "reason": "generation limit reached"}
    return None


# ---------------------------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------------------------


class EventBus:
    """Per-job fan-out to ``asyncio.Queue`` subscribers living on one event loop.

    ``publish``/``done`` may be called from any thread; ``subscribe``/``unsubscribe`` must run on
    the bound loop. Each subscriber receives ``("progress", event)`` items and finally
    ``("done", job_json)``.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None):
        self.loop = loop
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._last: dict[str, dict] = {}

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    # -- subscription (loop thread) -------------------------------------------------------------

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id)
            if subs and q in subs:
                subs.remove(q)
            if not subs:
                self._subscribers.pop(job_id, None)

    # -- publishing (any thread) ----------------------------------------------------------------

    def last(self, job_id: str) -> dict | None:
        with self._lock:
            return self._last.get(job_id)

    def publish(self, job_id: str, event: dict) -> None:
        with self._lock:
            self._last[job_id] = event
        self._dispatch(job_id, ("progress", event))

    def done(self, job_id: str, job_json: dict) -> None:
        """Deliver the terminal frame; subscribers and the cached event are dropped (the DB keeps the
        terminal progress, which the worker persists before calling this)."""
        with self._lock:
            self._last.pop(job_id, None)
        self._dispatch(job_id, ("done", job_json))

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._last.pop(job_id, None)
            self._subscribers.pop(job_id, None)

    def _dispatch(self, job_id: str, item: tuple) -> None:
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._deliver, job_id, item)
        except RuntimeError:  # loop shut down between the check and the call
            pass

    def _deliver(self, job_id: str, item: tuple) -> None:
        with self._lock:
            subs = list(self._subscribers.get(job_id, ()))
            if item[0] == "done":
                self._subscribers.pop(job_id, None)
        for q in subs:
            q.put_nowait(item)


# ---------------------------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------------------------


def _error_text(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".strip()
    return text[:2000]


class Worker:
    """Serial job runner; the only code path that touches the engine.

    Life cycle: ``start(loop)`` spawns the thread, ``submit(job_id)`` enqueues, ``cancel(job_id)``
    marks a queued job cancelled immediately (announcing ``done`` on the bus) or sets the running
    job's cancel flag, ``stop()`` drains and unloads the engine on the worker thread.
    """

    def __init__(self, store: JobStore, engine: Engine, paths: config.Paths, bus: EventBus | None = None, *,
                 fake: bool = False):
        self.store = store
        self.engine = engine
        self.paths = paths
        self.bus = bus or EventBus()
        self.fake = fake
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._cancels: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.current_job_id: str | None = None
        self.memory: dict | None = None
        self._last_memory_sample = 0.0
        self._stopping = False

    # -- lifecycle ----------------------------------------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        if loop is not None:
            self.bus.bind(loop)
        if self._thread is not None:
            return
        # Requeue anything left queued in the DB (e.g. after a restart); running rows are stale.
        for job_id in self.store.queued_ids():
            self.submit(job_id)
        for job in self.store.list(status="running", limit=500)[0]:
            self.store.update_status(job.id, "failed", error="server restarted while the job was running")
        self._thread = threading.Thread(target=self._run, name="yue2-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop after the current job: it is cancelled, queued rows stay ``queued`` (re-enqueued on start)."""
        thread = self._thread
        if thread is None:
            return
        self._stopping = True
        with self._lock:
            current = self.current_job_id
            event = self._cancels.get(current) if current is not None else None
        if event is not None:
            event.set()
        self._queue.put(None)
        thread.join(timeout)
        if not thread.is_alive():
            self._thread = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- submission / cancellation (any thread) -------------------------------------------------

    def submit(self, job_id: str) -> None:
        with self._lock:
            self._cancels.setdefault(job_id, threading.Event())
        self._queue.put(job_id)

    def cancel(self, job_id: str) -> str | None:
        """Returns ``"cancelled"`` (was queued), ``"requested"`` (running) or ``None`` (terminal)."""
        if self.store.update_status(job_id, "cancelled", expected="queued"):
            with self._lock:
                self._cancels.pop(job_id, None)
            event = status_event(job_id, "cancelled", "cancelled before it started")
            self.store.set_progress(job_id, event)
            self.bus.publish(job_id, event)
            self.bus.done(job_id, self.store.get(job_id).to_api())
            return "cancelled"
        job = self.store.get(job_id)
        if job.status == "running":
            with self._lock:
                self._cancels.setdefault(job_id, threading.Event()).set()
            return "requested"
        return None

    def cancelled(self, job_id: str) -> bool:
        with self._lock:
            event = self._cancels.get(job_id)
        return bool(event and event.is_set())

    # -- status -------------------------------------------------------------------------------

    def engine_status(self) -> dict:
        state = getattr(self.engine, "state", "cold")
        precision = getattr(self.engine, "precision", None)
        memory = self.memory
        memory_gib = None
        if memory and state != "cold":
            used = memory.get("mlx_active_bytes") or memory.get("rss_bytes") or 0
            memory_gib = round(used / 2**30, 2)
        return {"state": state, "precision": precision if state != "cold" else None,
                "memory_gib": memory_gib, "current_job_id": self.current_job_id}

    def _sample_memory(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_memory_sample < 1.0:
            return
        self._last_memory_sample = now
        try:
            self.memory = self.engine.memory_footprint()
        except Exception:  # never let a metrics failure hurt a job
            self.memory = None

    # -- worker thread ------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while True:
                job_id = self._queue.get()
                if job_id is None or self._stopping:
                    break
                job = self.store.find(job_id)
                if job is None or job.status != "queued":
                    with self._lock:
                        self._cancels.pop(job_id, None)
                    continue
                try:
                    self._run_job(job)
                except Exception:  # pragma: no cover - defensive: the loop must survive anything
                    log.exception("worker: unexpected error running job %s", job_id)
        finally:
            try:
                self.engine.unload()
            except Exception:  # pragma: no cover
                log.exception("worker: engine unload failed")

    def _publish(self, job_id: str, event: dict) -> None:
        self.bus.publish(job_id, event)

    def _on_engine_event(self, job_id: str, raw: dict) -> None:
        event = normalise(raw, job_id)
        self._publish(job_id, event)
        if event["type"] == "stage" and event["status"] != "running":
            self._sample_memory()

    def _upload_path(self, upload_id: str) -> Path:
        matches = sorted(p for p in self.paths.uploads_dir.glob(f"{upload_id}.*") if p.suffix != ".json")
        if not matches:
            raise FileNotFoundError(f"upload {upload_id!r} is missing from {self.paths.uploads_dir}")
        return matches[0]

    def _run_job(self, job: Job) -> None:
        with self._lock:
            cancel = self._cancels.setdefault(job.id, threading.Event())
        if not self.store.update_status(job.id, "running", expected="queued"):
            with self._lock:
                self._cancels.pop(job.id, None)
            return  # cancelled between dequeue and start
        self.current_job_id = job.id
        job = self.store.get(job.id)
        song_dir = self.paths.songs_dir / job.id
        song_dir.mkdir(parents=True, exist_ok=True)
        settings = self.store.get_settings()
        options = job.options(settings)
        (song_dir / "job.json").write_text(json.dumps({
            "id": job.id, "kind": job.kind, "params": job.params, "preset": options.preset,
            "precision": options.precision, "ode_steps": options.ode_steps, "seed": job.seed,
            "group_id": job.group_id, "parent_id": job.parent_id, "created_at": job.created_at,
        }, ensure_ascii=False, indent=2))
        self._publish(job.id, status_event(job.id, "running", f"{job.kind} job started"))

        def on_event(raw: dict) -> None:
            self._on_engine_event(job.id, raw)

        final_status, error, summary = "done", None, None
        try:
            if cancel.is_set():
                raise InterruptedError("Cancelled before start")
            engine = self.engine
            if engine.state == "cold" or engine.precision != options.precision:
                self._publish(job.id, status_event(job.id, "running",
                                                   f"loading models ({options.precision})…", stage="load"))
            engine.ensure(options, on_event)
            self._sample_memory(force=True)
            request = engine_request(job)
            if job.kind == "cover":
                audio_path = self._upload_path(job.params["upload_id"])
                summary = engine.cover_song(audio_path, song_dir, task=job.params.get("task", "melody-full"),
                                            request=request, options=options, on_event=on_event,
                                            cancelled=cancel.is_set)
            else:
                summary = engine.create_song(request, song_dir, options=options, on_event=on_event,
                                             cancelled=cancel.is_set)
        except (InterruptedError, asyncio.CancelledError):
            final_status = "cancelled"
        except Exception as exc:  # noqa: BLE001 - any engine error fails the job, never the worker
            final_status, error = "failed", _error_text(exc)
            log.warning("job %s failed: %s\n%s", job.id, error, traceback.format_exc())
        finally:
            self._sample_memory(force=True)

        if final_status == "done" and summary is not None:
            self._publish(job.id, normalise({"type": "stage", "stage": "Saving artifacts", "completed": 0,
                                             "total": None, "status": "running", "seconds": 0.0,
                                             "ts": time.time()}, job.id))
            started = time.perf_counter()
            self.store.set_timing(job.id, derive_timing(summary), audio_seconds=summary.get("seconds"),
                                  truncated=derive_truncated(summary))
            self._publish(job.id, normalise({"type": "stage", "stage": "Saving artifacts", "completed": 1,
                                             "total": 1, "status": "completed",
                                             "seconds": round(time.perf_counter() - started, 3),
                                             "ts": time.time()}, job.id))
            message = "done"
        elif final_status == "cancelled":
            message = "cancelled"
        else:
            message = f"failed: {error}"

        terminal = status_event(job.id, final_status, message)
        self.store.update_status(job.id, final_status, error=error, progress=terminal)
        self._publish(job.id, terminal)
        with self._lock:
            self._cancels.pop(job.id, None)
        self.current_job_id = None
        try:
            self.bus.done(job.id, self.store.get(job.id).to_api())
        except Exception:  # job deleted meanwhile
            log.debug("job %s vanished before done could be announced", job.id)

    # -- housekeeping used by the API ----------------------------------------------------------

    def delete_song_dir(self, job_id: str) -> None:
        shutil.rmtree(self.paths.songs_dir / job_id, ignore_errors=True)
        self.bus.forget(job_id)
