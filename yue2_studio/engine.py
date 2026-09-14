"""Engine wrapper around mlx-Yue's ``lyra.pipeline.YuE2Pipeline``.

Only the worker thread may import and use this module: it imports ``mlx`` and owns the single
resident pipeline (mlx-Yue allows one GPU workload per process). ``yue2_studio.config`` is
imported first so ``MLX_ENABLE_TF32=0`` is set before MLX initialises.

mlx-Yue is never modified; ``StudioPipeline`` subclasses ``YuE2Pipeline`` and replaces its
``_status`` progress hook so stage progress is published to a callback instead of stderr.
"""

from __future__ import annotations

from yue2_studio import config  # isort: skip  (sets MLX_ENABLE_TF32 before mlx import)

import gc  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import mlx.core as mx  # noqa: E402
import psutil  # noqa: E402
from lyra.pipeline import SongResult, YuE2Pipeline, initial_noise  # noqa: E402
from yue2.progress import Progress, _Stage  # noqa: E402
from yue2.protocol import GenerationConfig, SongRequest  # noqa: E402
from yue2.storage import identity, write_json  # noqa: E402

from yue2_studio.config import EngineOptions  # noqa: E402

EventCallback = Callable[[dict], None]
Cancelled = Callable[[], bool]

EVENT_INTERVAL = 0.25  # seconds between throttled events (<= 4 Hz)
SAMPLE_RATE = 48000


@dataclass
class ProgressEvent:
    """One progress event. ``type`` is ``stage`` | ``token`` | ``abc`` | ``log``.

    ``stage`` events carry ``completed``/``total``/``unit``/``status``/``seconds`` for a pipeline
    stage; ``token`` events carry ``phase``/``tokens``/``tps`` during AR generation; ``abc`` events
    carry the partial (then final) decoded score ``text``; ``log`` events carry free ``text``.
    """

    type: str
    stage: str | None = None
    completed: int | None = None
    total: int | None = None
    unit: str | None = None
    status: str | None = None
    phase: str | None = None
    tokens: int | None = None
    tps: float | None = None
    text: str | None = None
    seconds: float | None = None
    ts: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class _StudioStage(_Stage):
    """``yue2.progress._Stage`` whose updates publish events instead of rendering to stderr.

    Subclassing keeps the exact upstream interface and validation
    (``update``/``set_total``/``advance``/``finish``/``token``); the owner is a disabled
    ``Progress`` so nothing is written and no heartbeat thread starts.
    """

    def __init__(self, owner, label, *, total=None, unit=None, emit):
        super().__init__(owner, label, total=total, unit=unit)
        self._emit = emit
        self._last_emit = -1.0
        self._wall_start = None

    def __enter__(self):
        self._wall_start = time.perf_counter()
        super().__enter__()
        self._publish("running", force=True)
        return self

    def elapsed(self) -> float:
        return 0.0 if self._wall_start is None else time.perf_counter() - self._wall_start

    def _publish(self, status, *, force=False):
        now = time.perf_counter()
        if not force and now - self._last_emit < EVENT_INTERVAL:
            return
        self._last_emit = now
        elapsed = self.elapsed()
        tps = None
        if self.unit == "tokens" and elapsed > 0 and self.completed:
            tps = round(self.completed / elapsed, 2)
        self._emit(ProgressEvent(
            type="stage", stage=self.label, completed=self.completed, total=self.total, unit=self.unit,
            status=status, tps=tps, seconds=round(elapsed, 3), ts=time.time(),
        ))

    def update(self, completed, total=None):
        super().update(completed, total=total)
        if not self._finished:
            self._publish("running")

    def set_total(self, total):
        super().set_total(total)
        if not self._finished:
            self._publish("running")

    def finish(self, status="completed"):
        already = self._finished
        super().finish(status=status)
        if not already:
            self._publish(status, force=True)


class StudioPipeline(YuE2Pipeline):
    """``YuE2Pipeline`` that reports progress through ``on_event`` and records stage timings.

    Every upstream stage (model loads, planning, semantic generation, synthesis, decoding) goes
    through ``self._status(label, total=, unit=)``; ``lyra.pipeline`` has no other reporting
    path (``_load_model`` also uses ``_status``), so overriding it is sufficient.
    """

    def __init__(self, model_dir, vae_dir, *, on_event: EventCallback | None = None, **kwargs):
        self.on_event: EventCallback | None = on_event
        self.stage_timings: dict[str, float] = {}
        kwargs.setdefault("progress", False)
        super().__init__(model_dir, vae_dir, **kwargs)
        # Upstream only calls ``status.advance()`` per AR token and ``status.update()`` per NAR step
        # when ``self.progress`` is true. The stderr reporter it would otherwise construct lives
        # only in ``_status`` (overridden here) and ``__call__`` (not used), so enabling the flag
        # after construction routes those counts to our stage objects without any terminal output.
        self.progress = True

    # -- events -------------------------------------------------------------------------------

    def emit(self, event: ProgressEvent) -> None:
        callback = self.on_event
        if callback is None:
            return
        try:
            callback(event.to_dict())
        except Exception:  # a broken listener must never abort inference
            pass

    def log(self, text: str) -> None:
        self.emit(ProgressEvent(type="log", text=text, ts=time.time()))

    @contextmanager
    def _status(self, label, *, total=None, unit=None):
        stage = _StudioStage(Progress(enabled=False), label, total=total, unit=unit, emit=self.emit)
        start = time.perf_counter()
        try:
            with stage:
                yield stage
        finally:
            self.stage_timings[stage.label] = round(
                self.stage_timings.get(stage.label, 0.0) + time.perf_counter() - start, 3
            )

    def token_observer(self, user_callback: Callable[[str, int], None] | None = None):
        """Return an ``on_token(phase, token)`` callback publishing ``token``/``abc`` events (<= 4 Hz).

        During the ``abc`` phase the accumulated ids are decoded with the pipeline tokenizer so the
        score can be displayed while it is being written.
        """
        state: dict[str, Any] = {"phase": None, "ids": [], "count": 0, "start": 0.0, "last": -1.0}

        def on_token(phase, token):
            now = time.perf_counter()
            if state["phase"] != phase:
                state.update(phase=phase, ids=[], count=0, start=now, last=-1.0)
            state["count"] += 1
            if phase == "abc":
                state["ids"].append(int(token))
            if user_callback is not None:
                user_callback(phase, token)
            if now - state["last"] < EVENT_INTERVAL:
                return
            state["last"] = now
            elapsed = now - state["start"]
            tps = round(state["count"] / elapsed, 2) if elapsed > 0 else None
            self.emit(ProgressEvent(type="token", phase=phase, tokens=state["count"], tps=tps,
                                    seconds=round(elapsed, 3), ts=time.time()))
            if phase == "abc":
                self.emit(ProgressEvent(type="abc", phase=phase, tokens=state["count"],
                                        text=self.tokenizer.decode(state["ids"]), status="partial",
                                        ts=time.time()))

        return on_token

    # -- small public wrappers around upstream private helpers ----------------------------------

    def build_request(self, **fields) -> SongRequest:
        return self._request(**fields)

    def guarded_cancelled(self, cancelled: Cancelled | None) -> Cancelled:
        """``cancelled`` that also raises when the GPU guard latched a memory/power error."""
        return self._guarded_cancelled(cancelled)

    def check_execution(self) -> None:
        self._check_execution()

    def release_models(self) -> None:
        """Drop resident AR/NAR/VAE weights (upstream ``_release_models``); they reload lazily (mmap)."""
        self._release_models()


def _clear_gpu() -> None:
    gc.collect()
    mx.clear_cache()


class Engine:
    """Owns at most one resident ``StudioPipeline`` and runs song jobs with it.

    Not thread-safe for job execution: a single worker thread must call ``create_song`` /
    ``cover_song``. ``state`` and ``memory_footprint`` may be read from other threads.
    """

    STATES = ("cold", "loading", "ready", "busy")

    def __init__(self, *, converted_dir: Path | None = None, vae_dir: Path | None = None,
                 pipeline_factory: Callable[..., Any] | None = None):
        self.converted_dir = Path(converted_dir if converted_dir is not None else config.CONVERTED_DIR)
        self.vae_dir = Path(vae_dir if vae_dir is not None else config.VAE_DIR)
        # ``pipeline_factory`` lets tests inject a fake pipeline; it receives StudioPipeline's arguments.
        self._pipeline_factory = pipeline_factory or StudioPipeline
        self._pipe: StudioPipeline | None = None
        self._build_key: tuple | None = None
        self._state = "cold"
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def pipeline(self) -> StudioPipeline | None:
        return self._pipe

    @property
    def precision(self) -> str | None:
        return None if self._pipe is None else self._pipe.precision

    def ensure(self, options: EngineOptions, on_event: EventCallback | None = None) -> StudioPipeline:
        """Return a pipeline matching ``options``; build lazily, rebuild when precision/budget/AC change.

        ``precision`` is fixed at construction in mlx-Yue, so a change closes the resident pipeline
        (releasing its models and GPU guard) and constructs a new one. ``ode_steps`` is per job and
        is applied to ``generation_config`` by the callers.
        """
        with self._lock:
            if self._pipe is not None and self._build_key != options.build_key:
                self.unload()
            if self._pipe is None:
                self._state = "loading"
                try:
                    self._pipe = self._pipeline_factory(
                        self.converted_dir, self.vae_dir, precision=options.precision,
                        memory_budget_gib=options.memory_budget_gib, require_ac=options.require_ac,
                        progress=False, on_event=on_event,
                    )
                    self._build_key = options.build_key
                except BaseException:
                    self._state = "cold"
                    raise
            self._pipe.on_event = on_event
            self._state = "ready"
            return self._pipe

    def unload(self) -> None:
        with self._lock:
            pipe, self._pipe, self._build_key = self._pipe, None, None
            self._state = "cold"
            if pipe is not None:
                try:
                    pipe.close()  # releases models (``_release_models``) and exits the GPU guard
                finally:
                    _clear_gpu()

    close = unload

    def memory_footprint(self) -> dict:
        process = psutil.Process()
        info = {"rss_bytes": process.memory_info().rss,
                "system_available_bytes": psutil.virtual_memory().available}
        info.update(mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                    mlx_peak_bytes=mx.get_peak_memory())
        return info

    # -- request helpers ------------------------------------------------------------------------

    @staticmethod
    def _split_request(request: dict) -> tuple[dict, dict, Any, Any]:
        data = dict(request)
        generation = data.pop("generation_config", None) or {}
        abc_sampling = data.pop("abc_sampling", None)
        semantic_sampling = data.pop("semantic_sampling", None)
        if "tags" in data:
            if "style" in data and data["style"] != data["tags"]:
                raise ValueError("style and tags disagree")
            data["style"] = data.pop("tags")
        return data, dict(generation), abc_sampling, semantic_sampling

    # -- failure handling ----------------------------------------------------------------------

    def _after_failure(self, pipe, error: BaseException) -> None:
        """Discard the pipeline unless it is still healthy after a plain cancellation.

        ``GPUExecution.check()`` re-raises the resource monitor's latched error (MemoryError, AC
        disconnect) on every later call and ``lyra.measure`` never clears it, so a pipeline that
        failed once would fail every subsequent job. Any non-cancellation error therefore unloads
        the pipeline so the next ``ensure()`` rebuilds it; a cancellation keeps it only when the
        guard still passes.
        """
        _clear_gpu()
        if isinstance(error, InterruptedError):
            try:
                pipe.check_execution()
                return
            except Exception:
                pass
        self.unload()

    @contextmanager
    def _busy(self, pipe):
        with self._lock:
            self._state = "busy"
        try:
            yield
        except BaseException as error:
            self._after_failure(pipe, error)
            raise
        finally:
            with self._lock:
                self._state = "ready" if self._pipe is not None else "cold"

    # -- jobs ---------------------------------------------------------------------------------

    def create_song(self, request: dict, out_dir: Path, *, options: EngineOptions,
                    on_event: EventCallback | None = None, cancelled: Cancelled | None = None) -> dict:
        """Run plan -> semantic -> synthesize -> decode exactly like ``YuE2Pipeline.__call__``.

        Artifacts: ``out_dir/plan/`` (plan.json, score.abc, written right after planning) and
        ``out_dir/song/`` (audio.flac, result.json, latent.npy, noise.npy, ...) via
        ``SongResult.save_artifacts``, which requires an empty directory. ``out_dir/summary.json``
        holds the returned summary. ``InterruptedError`` (cancellation) propagates after the GPU
        cache is cleared; any other error also unloads the pipeline (see ``_after_failure``).
        """
        out_dir = Path(out_dir)
        fields, generation, abc_sampling, semantic_sampling = self._split_request(request)
        SongRequest(**fields)  # validate before touching the GPU
        if (out_dir / "song").exists() and any((out_dir / "song").iterdir()):
            raise FileExistsError(f"Song directory is not empty: {out_dir / 'song'}")
        pipe = self.ensure(options, on_event)
        with self._busy(pipe):
            return self._run_create(pipe, fields, generation, abc_sampling, semantic_sampling, out_dir,
                                    options=options, cancelled=cancelled)

    def _run_create(self, pipe, fields: dict, generation: dict, abc_sampling, semantic_sampling,
                    out_dir: Path, *, options: EngineOptions, cancelled: Cancelled | None,
                    extra_stages: dict[str, float] | None = None) -> dict:
        """Stage body shared by ``create_song`` and ``cover_song``; caller holds ``_busy``.

        ``_status`` accumulates seconds by label for the pipeline's lifetime, so the per-job stage
        timings are reset here; ``extra_stages`` (e.g. the cover's transcription stage) is merged
        into the reported ``timing["stages"]``.
        """
        song_dir, plan_dir = out_dir / "song", out_dir / "plan"
        pipe.stage_timings = {}
        pipe.generation_config = GenerationConfig.from_dict({**generation, "ode_steps": options.ode_steps})
        observer = pipe.token_observer()
        pipe.check_execution()
        native = pipe.build_request(**fields)
        effective = pipe.effective_config(native, abc_sampling, semantic_sampling)
        request_id = identity({"request": native.to_dict(), "config": effective, "weights": pipe.weights})
        start = time.perf_counter()
        plan = pipe.plan(request=native, abc_sampling=abc_sampling, cancelled=cancelled, on_token=observer)
        plan.save(plan_dir)
        if plan.abc is not None:
            pipe.emit(ProgressEvent(type="abc", phase="abc", text=plan.abc, status="final",
                                    tokens=len(plan.abc_ids), ts=time.time()))
        semantic = pipe.generate_semantic(plan, sampling=semantic_sampling, cancelled=cancelled,
                                          on_token=observer)
        noise = initial_noise(len(semantic.tokens), native.seed)
        nar_start = time.perf_counter()
        latents = pipe.synthesize(semantic, cancelled=cancelled, noise=noise)
        nar_seconds = time.perf_counter() - nar_start
        if pipe.guarded_cancelled(cancelled)():
            raise InterruptedError("Cancelled before VAE")
        vae_start = time.perf_counter()
        audio = pipe.decode(latents, cancelled=cancelled)
        timing = {
            "abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
            "vae_seconds": time.perf_counter() - vae_start, "load": dict(pipe.load_timing),
            "e2e_seconds": time.perf_counter() - start,
            "stages": {**(extra_stages or {}), **pipe.stage_timings},
        }
        result = SongResult(audio, SAMPLE_RATE, semantic, latents, effective, pipe.weights, timing,
                            request_id, noise)
        pipe.check_execution()
        receipt = result.save_artifacts(song_dir)
        summary = {
            "status": receipt["status"],
            "audio_path": str(song_dir / "audio.flac"),
            "score_path": str(song_dir / "score.abc") if plan.abc is not None else None,
            "song_dir": str(song_dir),
            "plan_dir": str(plan_dir),
            "sample_rate": SAMPLE_RATE,
            "seconds": len(audio) / SAMPLE_RATE,
            "timing": timing,
            "truncated": result.truncated,
            "identity": request_id,
            "preset": options.preset,
            "precision": options.precision,
            "ode_steps": options.ode_steps,
            "seed": native.seed,
        }
        write_json(out_dir / "summary.json", summary)
        pipe.log(f"Completed {summary['seconds']:.1f}s of audio in {timing['e2e_seconds']:.1f}s")
        return summary

    def cover_song(self, audio_path: Path, out_dir: Path, *, task: str = "melody-full", request: dict,
                   options: EngineOptions, on_event: EventCallback | None = None,
                   cancelled: Cancelled | None = None) -> dict:
        """Transcribe ``audio_path`` then create a song from the transcribed score (``lyra.commands.cover``).

        Transcription runs in the worker thread under the resident pipeline's GPU guard
        (``transcribe`` takes no guard of its own; the guard's RLock permits same-thread nesting, and
        the guard's memory watchdog keeps sampling), so no second ``GPUExecution`` is opened. The
        resident AR/NAR/VAE weights are released first so SheetSage2 + MERT (~3 GiB) do not stack on
        top of them; they reload lazily during the song stages.
        """
        from lyra.transcription.pipeline import transcribe

        if task not in {"full", "melody-full", "melody-vocal"}:
            raise ValueError("task must be full, melody-full or melody-vocal")
        out_dir, audio_path = Path(out_dir), Path(audio_path)
        fields, generation, abc_sampling, semantic_sampling = self._split_request(request)
        if fields.get("abc") is not None:
            raise ValueError("Cover takes source audio, not a supplied score")
        mode = "full" if task == "full" else "melody"
        if fields.get("cot", mode) != mode:
            raise ValueError("Cover mode must match the transcription task")
        fields["cot"] = mode
        SongRequest(**fields)  # validate text, seed and mode before spending time on transcription
        transcription_dir = out_dir / "transcription"
        for directory in (transcription_dir, out_dir / "song"):
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f"Directory is not empty: {directory}")
        pipe = self.ensure(options, on_event)
        pipe.stage_timings = {}
        with self._busy(pipe):
            started = time.perf_counter()
            guarded = pipe.guarded_cancelled(cancelled)
            pipe.release_models()
            with pipe._status("Transcribing audio", unit="windows") as stage:
                def progress(info: dict) -> None:
                    if info.get("stage") == "encoding":
                        stage.update(max(stage.completed, info["window"] - 1), total=info["windows"])
                    elif info.get("stage") == "decoding":
                        pipe.emit(ProgressEvent(type="token", phase="transcription", tokens=info["tokens"],
                                                seconds=round(stage.elapsed(), 3), ts=time.time()))

                transcription = transcribe(
                    audio_path, transcription_dir, task=task, offline=True,
                    cache_dir=str(config.HF_CACHE_DIR), cancelled=guarded, progress=progress,
                )
                if stage.total is not None:
                    stage.update(stage.total)
            pipe.check_execution()
            _clear_gpu()
            transcription_seconds = time.perf_counter() - started
            transcription_stages = dict(pipe.stage_timings)  # _run_create resets the per-job timings
            if transcription["status"] != "complete" or transcription["truncated"]:
                raise ValueError("Transcription is incomplete; inspect its artifacts before using the score")
            score = (transcription_dir / "score.abc").read_bytes().decode("utf-8")
            song_fields = {**fields, "abc": score}
            song_request = dict(song_fields)
            if generation:
                song_request["generation_config"] = generation
            if abc_sampling is not None:
                song_request["abc_sampling"] = abc_sampling
            if semantic_sampling is not None:
                song_request["semantic_sampling"] = semantic_sampling
            write_json(out_dir / "request.json", song_request)
            summary = self._run_create(pipe, song_fields, generation, abc_sampling, semantic_sampling,
                                       out_dir, options=options, cancelled=cancelled,
                                       extra_stages=transcription_stages)
        summary["transcription"] = {
            "dir": str(transcription_dir), "task": task, "seconds": transcription_seconds,
            "source_audio_sha256": transcription["source_audio_sha256"],
            "duration_seconds": transcription["duration_seconds"],
        }
        summary["timing"]["transcription_seconds"] = transcription_seconds
        write_json(out_dir / "cover.json", {
            "source_audio_sha256": transcription["source_audio_sha256"],
            "transcription": "transcription/result.json", "song": "song/result.json",
            "backend": "mlx", "truncated": summary["truncated"], "task": task,
        })
        write_json(out_dir / "summary.json", summary)
        return summary


def load_summary(out_dir: Path) -> dict | None:
    path = Path(out_dir) / "summary.json"
    return json.loads(path.read_text()) if path.is_file() else None
