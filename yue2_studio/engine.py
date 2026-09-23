"""Engine wrapper around mlx-Yue's ``lyra.pipeline.YuE2Pipeline``.

Only the worker thread may import and use this module: it imports ``mlx`` and owns the single
resident pipeline (mlx-Yue allows one GPU workload per process). ``yue2_studio.config`` is
imported first so ``MLX_ENABLE_TF32=0`` is set before MLX initialises.

mlx-Yue is never modified; ``StudioPipeline`` subclasses ``YuE2Pipeline`` and replaces its
``_status`` progress hook so stage progress is published to a callback instead of stderr, and wraps
``_load_model`` so LoRA adapters (``yue2_studio.lora``) are merged into whichever AR / NAR weights
upstream just loaded.
"""

from __future__ import annotations

from yue2_studio import config  # isort: skip  (sets MLX_ENABLE_TF32 before mlx import)

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from lyra.pipeline import SongResult, YuE2Pipeline, initial_noise  # noqa: E402
from yue2.pipeline import SymbolicPlan  # noqa: E402
from yue2.progress import Progress, _Stage  # noqa: E402
from yue2.protocol import GenerationConfig, SongRequest, resolve_sampling, token_prefixes  # noqa: E402
from yue2.storage import identity, sha256_file, write_json  # noqa: E402

from yue2_studio import audio as audio_mod  # noqa: E402
from yue2_studio import hum as hum_mod  # noqa: E402
from yue2_studio import hum_nar, vae_encoder  # noqa: E402
from yue2_studio import lora as lora_mod  # noqa: E402
from yue2_studio.config import EngineOptions, LoraStack, loras_to_api  # noqa: E402

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
        self.loras: list[tuple[lora_mod.AdapterInfo, float]] = []
        self.hum_adapter: lora_mod.AdapterInfo | None = None
        self._lora_key: tuple = ()
        kwargs.setdefault("progress", False)
        super().__init__(model_dir, vae_dir, **kwargs)
        self._base_weights = dict(self.weights)
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

    def token_observer(self, user_callback: Callable[[str, int], None] | None = None, *,
                       abc_prefix: str = ""):
        """Return an ``on_token(phase, token)`` callback publishing ``token``/``abc`` events (<= 4 Hz).

        During the ``abc`` phase the accumulated ids are decoded with the pipeline tokenizer so the
        score can be displayed while it is being written; ``abc_prefix`` (the hum's open score when
        the planner continues one) is prepended so the streamed text is the whole score.
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
                                        text=abc_prefix + self.tokenizer.decode(state["ids"]),
                                        status="partial", ts=time.time()))

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

    # -- hum-to-song: score continuation ----------------------------------------------------------

    def plan_continuation(self, request: SongRequest, open_abc: str, *, abc_sampling=None, cancelled=None,
                          on_token=None) -> SymbolicPlan:
        """Plan by *continuing* ``open_abc``: the prompt ends inside the score (no ``ABC_END``).

        Upstream ``plan()`` uses a supplied score verbatim; here the hum's open score is tokenised and
        appended to the ``[ABC_START]`` prefix so the AR keeps writing it. ``open_abc`` must end with a
        newline (``hum.trim_open_score``): the tokenizer splits exactly at line boundaries, so the
        finished score tokenises like one written in a single pass and ``generate_semantic``'s prefix
        check holds.
        """
        if request.cot == "off" or request.abc is not None:
            raise ValueError("Score continuation needs cot=melody|full and no supplied abc")
        if not open_abc.endswith("\n"):
            raise ValueError("The open score must end with a newline")
        partial = self.tokenizer.encode(open_abc)
        prefix = token_prefixes(request, self.tokenizer) + partial
        sampling = resolve_sampling(abc_sampling, self.generation_config.abc)
        ids, timing, truncated = self._generate(prefix, sampling, request.seed, "abc", cancelled=cancelled,
                                                on_token=on_token)
        full = partial + list(ids)
        timing = {**timing, "continuation_prefix_tokens": len(partial)}
        return SymbolicPlan(request, self.tokenizer.decode(full), full,
                            token_prefixes(request, self.tokenizer, full), timing, truncated)

    # -- LoRA ---------------------------------------------------------------------------------

    @property
    def lora_names(self) -> list[dict]:
        return [{"name": info.name, "scale": scale} for info, scale in self.loras]

    def set_loras(self, stack: list[tuple[lora_mod.AdapterInfo, float]],
                  hum_adapter: lora_mod.AdapterInfo | None = None) -> None:
        """Select the adapters merged into the models used from now on.

        Adapters are merged into the base weights, so a different stack drops the resident models
        (they reload from the memory-mapped files in ~0.1 s) and the merge happens again on the next
        ``_load_model``. ``weights`` gains ``loras`` / ``hum_adapter`` entries so request identities and
        saved ``result.json`` files record which adapters shaped the song. The hum adapter's NAR LoRA
        is merged last (``hum_adapter_v1`` was trained on top of ``nar_lora_joint_v4``).
        """
        key = tuple((info.identity()["sha256"], float(scale)) for info, scale in stack)
        if hum_adapter is not None:
            key += (("hum", hum_adapter.identity()["sha256"]),)
        if key == self._lora_key:
            self.loras, self.hum_adapter = list(stack), hum_adapter
            return
        self._release_models()
        self.loras, self.hum_adapter, self._lora_key = list(stack), hum_adapter, key
        weights = dict(self._base_weights)
        if stack:
            weights["loras"] = [{**info.identity(), "user_scale": scale} for info, scale in stack]
        if hum_adapter is not None:
            weights["hum_adapter"] = hum_adapter.identity()
        self.weights = weights

    def _load_model(self, for_nar=False):
        model = super()._load_model(for_nar=for_nar)
        if self._lora_key:
            for part, candidate, label in (("ar", self._ar, f"{self.precision} AR"),
                                           ("ar", self._bf16_ar, "BF16 conditioning"),
                                           ("nar", self._nar, "acoustic")):
                if candidate is not None and getattr(candidate, "_studio_loras", None) != self._lora_key:
                    self._merge_loras(candidate, part, label)
        return model

    def _merge_loras(self, model, part: str, label: str) -> None:
        wanted = [(info, scale) for info, scale in self.loras if part in info.parts]
        if part == "nar" and self.hum_adapter is not None:
            wanted.append((self.hum_adapter, 1.0))
        if wanted:
            title = f"Merging LoRA into {label} model"
            with self._status(title, total=len(wanted), unit="adapters") as stage:
                for info, scale in wanted:
                    count = lora_mod.apply_adapter(model, info, part=part, scale=scale)
                    self.log(f"Merged {info.name} into {count} {part.upper()} modules (scale {scale:g})")
                    stage.advance()
            mx.clear_cache()
            self._check_execution()
        object.__setattr__(model, "_studio_loras", self._lora_key)


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
                 loras_dir: Path | None = None, pipeline_factory: Callable[..., Any] | None = None):
        self.converted_dir = Path(converted_dir if converted_dir is not None else config.CONVERTED_DIR)
        self.vae_dir = Path(vae_dir if vae_dir is not None else config.VAE_DIR)
        self.loras_dir = Path(loras_dir if loras_dir is not None else config.LORAS_DIR)
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

    @property
    def loras(self) -> list[dict]:
        """``[{"name", "scale"}]`` merged into the resident pipeline (empty when none / cold)."""
        pipe = self._pipe
        return [] if pipe is None else list(getattr(pipe, "lora_names", []))

    def resolve_loras(self, stack: LoraStack) -> list[tuple[lora_mod.AdapterInfo, float]]:
        """Look every adapter of ``stack`` up in ``loras_dir`` (header-only; raises before any GPU work)."""
        return [(lora_mod.find_adapter(self.loras_dir, name), scale) for name, scale in stack]

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
        loras = self.resolve_loras(options.loras)
        pipe = self.ensure(options, on_event)
        with self._busy(pipe):
            pipe.set_loras(loras)
            return self._run_create(pipe, fields, generation, abc_sampling, semantic_sampling, out_dir,
                                    options=options, cancelled=cancelled)

    def _run_create(self, pipe, fields: dict, generation: dict, abc_sampling, semantic_sampling,
                    out_dir: Path, *, options: EngineOptions, cancelled: Cancelled | None,
                    extra_stages: dict[str, float] | None = None, planner: Callable[..., Any] | None = None,
                    synthesizer: Callable[..., Any] | None = None, config_extra: dict | None = None,
                    abc_prefix: str = "") -> dict:
        """Stage body shared by ``create_song``, ``cover_song`` and ``hum_song``; caller holds ``_busy``.

        ``_status`` accumulates seconds by label for the pipeline's lifetime, so the per-job stage
        timings are reset here; ``extra_stages`` (e.g. the cover's transcription stage) is merged
        into the reported ``timing["stages"]``. ``planner(pipe, request, abc_sampling, cancelled,
        on_token)`` and ``synthesizer(semantic, *, cancelled, noise)`` replace ``pipe.plan`` /
        ``pipe.synthesize`` (hum-to-song); ``config_extra`` is merged into the effective config before
        the request identity is computed (``SongResult.save_artifacts`` recomputes it from the same
        dict), so inputs that are not part of ``SongRequest`` still change the identity.
        """
        song_dir, plan_dir = out_dir / "song", out_dir / "plan"
        pipe.stage_timings = {}
        pipe.generation_config = GenerationConfig.from_dict({**generation, "ode_steps": options.ode_steps})
        pipe.fast_numerics = options.fast_numerics
        observer = pipe.token_observer(abc_prefix=abc_prefix)
        pipe.check_execution()
        native = pipe.build_request(**fields)
        effective = {**pipe.effective_config(native, abc_sampling, semantic_sampling), **(config_extra or {})}
        request_id = identity({"request": native.to_dict(), "config": effective, "weights": pipe.weights})
        start = time.perf_counter()
        if planner is not None:
            plan = planner(pipe, native, abc_sampling, cancelled, observer)
        else:
            plan = pipe.plan(request=native, abc_sampling=abc_sampling, cancelled=cancelled,
                             on_token=observer)
        plan.save(plan_dir)
        if plan.abc is not None:
            pipe.emit(ProgressEvent(type="abc", phase="abc", text=plan.abc, status="final",
                                    tokens=len(plan.abc_ids), ts=time.time()))
        semantic = pipe.generate_semantic(plan, sampling=semantic_sampling, cancelled=cancelled,
                                          on_token=observer)
        noise = initial_noise(len(semantic.tokens), native.seed)
        nar_start = time.perf_counter()
        if synthesizer is not None:
            latents = synthesizer(semantic, cancelled=cancelled, noise=noise)
        else:
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
            "fast_numerics": options.fast_numerics,
            "seed": native.seed,
            "loras": loras_to_api(options.loras),
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
        loras = self.resolve_loras(options.loras)
        pipe = self.ensure(options, on_event)
        pipe.stage_timings = {}
        with self._busy(pipe):
            pipe.set_loras(loras)
            pipe.release_models()
            transcription, transcription_seconds = self._transcribe(pipe, audio_path, transcription_dir, task,
                                                                    cancelled)
            transcription_stages = dict(pipe.stage_timings)  # _run_create resets the per-job timings
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

    def _transcribe(self, pipe, audio_path: Path, transcription_dir: Path, task: str,
                    cancelled: Cancelled | None) -> tuple[dict, float]:
        """SheetSage2 transcription as a ``Transcribing audio`` stage; caller holds ``_busy``.

        Returns ``(result, seconds)``; incomplete or truncated transcriptions are an error.
        """
        from lyra.transcription.pipeline import transcribe

        started = time.perf_counter()
        guarded = pipe.guarded_cancelled(cancelled)
        with pipe._status("Transcribing audio", unit="windows") as stage:
            def progress(info: dict) -> None:
                if info.get("stage") == "encoding":
                    stage.update(max(stage.completed, info["window"] - 1), total=info["windows"])
                elif info.get("stage") == "decoding":
                    pipe.emit(ProgressEvent(type="token", phase="transcription", tokens=info["tokens"],
                                            seconds=round(stage.elapsed(), 3), ts=time.time()))

            try:
                transcription = transcribe(
                    audio_path, transcription_dir, task=task, offline=True,
                    cache_dir=str(config.HF_CACHE_DIR), cancelled=guarded, progress=progress,
                )
            except subprocess.CalledProcessError as error:
                raise audio_mod.decode_error(error, audio_path) from None
            if stage.total is not None:
                stage.update(stage.total)
        pipe.check_execution()
        _clear_gpu()
        if transcription["status"] != "complete" or transcription["truncated"]:
            raise ValueError("Transcription is incomplete; inspect its artifacts before using the score")
        return transcription, time.perf_counter() - started

    # -- hum to song ----------------------------------------------------------------------------

    def hum_song(self, audio_path: Path, out_dir: Path, *, request: dict, hum: hum_mod.HumOptions,
                 options: EngineOptions, on_event: EventCallback | None = None,
                 cancelled: Cancelled | None = None) -> dict:
        """Hum -> song: transcribe the hum, continue its open score, optionally condition the decoder.

        Stages (all under ``_busy``): the resident models are released; the hum is transcribed
        (``melody-vocal``) unless ``hum.melody == "ignore"``; with an adapter the hum is decoded,
        pitch-tracked into a carrier (``hum.analyse_hum``) and VAE-encoded (``vae_encoder``); then the
        usual stage body runs with a continuation planner (``StudioPipeline.plan_continuation``) and/or
        the hum-conditioned synthesizer (``hum_nar.make_synthesizer``).

        Artifacts: ``transcription/`` (unless ignore), ``hum/hum.abc`` (the open score),
        ``hum/carrier.flac`` + ``hum/carrier_latents.npy`` + ``hum/prosody.json`` (adapter only),
        ``plan/``, ``song/``, ``request.json``, ``hum.json`` and ``summary.json``.
        """
        import soundfile as sf

        out_dir, audio_path = Path(out_dir), Path(audio_path)
        fields, generation, abc_sampling, semantic_sampling = self._split_request(request)
        if fields.get("abc") is not None:
            raise ValueError("Hum takes a recording, not a supplied score")
        fields["cot"] = "melody"
        SongRequest(**fields)
        transcription_dir, hum_dir = out_dir / "transcription", out_dir / "hum"
        for directory in (transcription_dir, hum_dir, out_dir / "song"):
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f"Directory is not empty: {directory}")
        loras = self.resolve_loras(options.loras)
        adapter = lora_mod.find_hum_adapter(self.loras_dir, hum.adapter) if hum.adapter else None
        pipe = self.ensure(options, on_event)
        pipe.stage_timings = {}
        started = time.perf_counter()
        transcription = None
        transcription_seconds = None
        hum_abc = None
        analysis = None
        latents = None
        with self._busy(pipe):
            guarded = pipe.guarded_cancelled(cancelled)
            pipe.set_loras(loras, hum_adapter=adapter)
            pipe.release_models()
            hum_dir.mkdir(parents=True, exist_ok=True)
            if hum.transcribes:
                transcription, transcription_seconds = self._transcribe(pipe, audio_path, transcription_dir,
                                                                        "melody-vocal", cancelled)
                score = (transcription_dir / "score.abc").read_bytes().decode("utf-8")
                hum_abc = hum_mod.trim_open_score(score) if hum.melody == "continue" else score
                if not hum_mod.open_score_has_notes(hum_abc):
                    raise ValueError("The hum transcription has no notes (only rests); sing louder and "
                                     "closer to the microphone, or use melody=ignore")
                (hum_dir / "hum.abc").write_text(hum_abc, encoding="utf-8")
                pipe.log(f"Hum score: {len(hum_mod.score_body_lines(hum_abc))} lines with notes "
                         f"({'open, to be continued' if hum.melody == 'continue' else 'used as the melody'})")
            if adapter is not None:
                with pipe._status("Analysing hum"):
                    samples = audio_mod.decode_pcm(audio_path, sample_rate=hum_mod.SAMPLE_RATE, channels=1,
                                                   cancelled=guarded)
                    analysis = hum_mod.analyse_hum(samples, cancelled=guarded)
                    stereo = hum_mod.carrier_stereo(analysis.carrier)
                    sf.write(hum_dir / "carrier.flac", stereo, hum_mod.SAMPLE_RATE, subtype="PCM_16")
                onset = "no sustained onset" if analysis.onset_s is None else f"onset {analysis.onset_s:.2f}s"
                pipe.log(f"Hum: {analysis.duration_s:.1f}s, voiced {analysis.voiced_fraction:.0%}, {onset}")
                with pipe._status("Encoding hum", unit="chunks") as stage:
                    encoder = vae_encoder.load_encoder(self.vae_dir)
                    latents = vae_encoder.encode(encoder, stereo, cancelled=guarded,
                                                 on_progress=lambda i, n: stage.update(i, total=n))
                    del encoder
                np.save(hum_dir / "carrier_latents.npy", latents)
                hum_mod.write_prosody(hum_dir / "prosody.json", analysis, hum, {
                    "adapter_identity": adapter.identity(), "latent_frames": int(len(latents)),
                    "source_audio_sha256": sha256_file(audio_path),
                })
                pipe.check_execution()
                _clear_gpu()
            hum_stages = dict(pipe.stage_timings)  # _run_create resets the per-job timings
            song_fields = dict(fields)
            if hum.melody == "hum_only":
                song_fields["abc"] = hum_abc
            song_request = dict(song_fields)
            if generation:
                song_request["generation_config"] = generation
            if abc_sampling is not None:
                song_request["abc_sampling"] = abc_sampling
            if semantic_sampling is not None:
                song_request["semantic_sampling"] = semantic_sampling
            write_json(out_dir / "request.json", song_request)
            config_extra = {"hum": {
                **hum.to_dict(),
                "adapter_identity": None if adapter is None else adapter.identity(),
                "source_audio_sha256": sha256_file(audio_path),
                "hum_abc_sha256": None if hum_abc is None else hashlib.sha256(hum_abc.encode()).hexdigest(),
                "carrier_latents_sha256": (None if latents is None
                                           else sha256_file(hum_dir / "carrier_latents.npy")),
            }}
            planner = None
            abc_prefix = ""
            if hum.melody == "continue":
                abc_prefix = hum_abc

                def planner(p, native, sampling, cancel, on_token):
                    return p.plan_continuation(native, hum_abc, abc_sampling=sampling, cancelled=cancel,
                                               on_token=on_token)

            synthesizer = None
            if adapter is not None:
                synthesizer = hum_nar.make_synthesizer(pipe, latents, adapter, hum.influence, hum.offset_s)
            summary = self._run_create(pipe, song_fields, generation, abc_sampling, semantic_sampling,
                                       out_dir, options=options, cancelled=cancelled, extra_stages=hum_stages,
                                       planner=planner, synthesizer=synthesizer, config_extra=config_extra,
                                       abc_prefix=abc_prefix)
        summary["hum"] = {
            "dir": str(hum_dir), **hum.to_dict(),
            "adapter_identity": None if adapter is None else adapter.identity(),
            "hum_abc": "hum/hum.abc" if hum_abc is not None else None,
            "prosody": None if analysis is None else analysis.prosody(),
            "latent_frames": None if latents is None else int(len(latents)),
            "seconds": time.perf_counter() - started - summary["timing"]["e2e_seconds"],
        }
        if transcription is not None:
            summary["transcription"] = {
                "dir": str(transcription_dir), "task": "melody-vocal", "seconds": transcription_seconds,
                "source_audio_sha256": transcription["source_audio_sha256"],
                "duration_seconds": transcription["duration_seconds"],
            }
            summary["timing"]["transcription_seconds"] = transcription_seconds
        write_json(out_dir / "hum.json", {
            "source_audio_sha256": sha256_file(audio_path),
            "transcription": "transcription/result.json" if transcription is not None else None,
            "hum": "hum/prosody.json" if analysis is not None else None, "song": "song/result.json",
            "backend": "mlx", "truncated": summary["truncated"], **hum.to_dict(),
        })
        write_json(out_dir / "summary.json", summary)
        return summary


def load_summary(out_dir: Path) -> dict | None:
    path = Path(out_dir) / "summary.json"
    return json.loads(path.read_text()) if path.is_file() else None
