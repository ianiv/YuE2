"""``FakeEngine``: a scripted stand-in for ``yue2_studio.engine.Engine`` (no mlx, no GPU).

It implements the same protocol the worker uses (``ensure`` / ``create_song`` / ``cover_song`` /
``memory_footprint`` / ``unload``), emits the **raw** engine event shapes from ``docs/API.md``
§6 so the worker's normalisation is exercised, writes the same artifact layout, honours
``cancelled()`` between steps (raising ``InterruptedError``) and, with ``fail=True``, raises
``RuntimeError`` in the synthesis stage. ``delay`` is the pause per emitted event (tests use 0;
``yue2-studio --fake`` uses 0.3 s).
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from yue2_studio.config import EngineOptions

FAKE_ABC = "X:1\nT:Fake\nK:C\nCDEF|"
SAMPLE_RATE = 48000


def write_silence_flac(path: Path, seconds: float = 1.0) -> None:
    """~1 s of stereo 48 kHz silence as FLAC (soundfile ships with mlx-Yue's dependencies)."""
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(SAMPLE_RATE * seconds)
    sf.write(str(path), np.zeros((frames, 2), dtype=np.float32), SAMPLE_RATE, format="FLAC", subtype="PCM_24")


class _Run:
    """Per-job emission helper: timing, throttling-free event dicts, cancellation checks."""

    def __init__(self, engine: FakeEngine, on_event, cancelled):
        self.engine = engine
        self.on_event = on_event or (lambda e: None)
        self.cancelled = cancelled or (lambda: False)
        self.start = time.perf_counter()
        self.stage_start = self.start
        self.stages: dict[str, float] = {}

    def step(self) -> None:
        if self.cancelled():
            raise InterruptedError("Cancelled by user")
        if self.engine.delay:
            time.sleep(self.engine.delay)
        if self.cancelled():
            raise InterruptedError("Cancelled by user")

    def emit(self, event: dict) -> None:
        event = {k: v for k, v in event.items() if v is not None}
        event.setdefault("ts", time.time())
        self.on_event(event)

    def stage(self, label: str, *, total: int | None, unit: str | None, steps: int = 3,
              fail: bool = False, tokens_phase: str | None = None) -> float:
        """Emit start/progress/end events for one stage; returns its duration."""
        self.stage_start = time.perf_counter()
        self.emit({"type": "stage", "stage": label, "completed": 0, "total": total, "unit": unit,
                   "status": "running", "seconds": 0.0})
        self.step()
        if total:
            for i in range(1, steps + 1):
                completed = round(total * i / steps)
                elapsed = time.perf_counter() - self.stage_start
                tps = round(completed / elapsed, 2) if unit == "tokens" and elapsed > 0 else None
                if fail and i == steps:
                    raise RuntimeError("fake synthesis failure")
                if i < steps:
                    self.emit({"type": "stage", "stage": label, "completed": completed, "total": total,
                               "unit": unit, "status": "running", "tps": tps, "seconds": round(elapsed, 3)})
                    if tokens_phase:
                        self.emit({"type": "token", "phase": tokens_phase, "tokens": completed, "tps": tps,
                                   "seconds": round(elapsed, 3)})
                    self.step()
        elif fail:
            raise RuntimeError("fake synthesis failure")
        elapsed = time.perf_counter() - self.stage_start
        tps = round(total / elapsed, 2) if total and unit == "tokens" and elapsed > 0 else None
        self.emit({"type": "stage", "stage": label, "completed": total or 0, "total": total, "unit": unit,
                   "status": "completed", "tps": tps, "seconds": round(elapsed, 3)})
        self.stages[label] = self.stages.get(label, 0.0) + round(elapsed, 3)
        return elapsed


class FakeEngine:
    STATES = ("cold", "loading", "ready", "busy")

    def __init__(self, *, delay: float = 0.05, fail: bool = False, audio_seconds: float = 1.0):
        self.delay = delay
        self.fail = fail
        self.audio_seconds = audio_seconds
        self.state = "cold"
        self.precision: str | None = None
        self.pipeline: Any | None = None
        self.calls: list[tuple[str, dict]] = []
        self.ensure_calls = 0
        self._lock = threading.RLock()

    # -- lifecycle ----------------------------------------------------------------------------

    def ensure(self, options: EngineOptions, on_event: Callable[[dict], None] | None = None) -> Any:
        with self._lock:
            self.ensure_calls += 1
            if self.pipeline is not None and self.precision != options.precision:
                self.unload()
            if self.pipeline is None:
                self.state = "loading"
                self.pipeline = {"precision": options.precision, "on_event": on_event}
                self.precision = options.precision
            self.pipeline["on_event"] = on_event
            self.state = "ready"
            return self.pipeline

    def unload(self) -> None:
        with self._lock:
            self.pipeline = None
            self.precision = None
            self.state = "cold"

    close = unload

    def memory_footprint(self) -> dict:
        return {"rss_bytes": 512 * 2**20, "system_available_bytes": 32 * 2**30,
                "mlx_active_bytes": 0 if self.pipeline is None else 3 * 2**30,
                "mlx_cache_bytes": 0, "mlx_peak_bytes": 0 if self.pipeline is None else 4 * 2**30}

    # -- jobs ---------------------------------------------------------------------------------

    def create_song(self, request: dict, out_dir: Path, *, options: EngineOptions,
                    on_event=None, cancelled=None) -> dict:
        self.calls.append(("create_song", dict(request)))
        out_dir = Path(out_dir)
        self._validate(request, allow_abc=True)
        song_dir = out_dir / "song"
        if song_dir.exists() and any(song_dir.iterdir()):
            raise FileExistsError(f"Song directory is not empty: {song_dir}")
        self.ensure(options, on_event)
        run = _Run(self, on_event, cancelled)
        with self._busy():
            return self._run_create(run, request, out_dir, options)

    def cover_song(self, audio_path: Path, out_dir: Path, *, task: str = "melody-full", request: dict,
                   options: EngineOptions, on_event=None, cancelled=None) -> dict:
        self.calls.append(("cover_song", dict(request)))
        if task not in {"full", "melody-full", "melody-vocal"}:
            raise ValueError("task must be full, melody-full or melody-vocal")
        if request.get("abc") is not None:
            raise ValueError("Cover takes source audio, not a supplied score")
        mode = "full" if task == "full" else "melody"
        if request.get("cot", mode) != mode:
            raise ValueError("Cover mode must match the transcription task")
        request = {**request, "cot": mode}
        self._validate(request, allow_abc=False)
        out_dir, audio_path = Path(out_dir), Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(str(audio_path))
        self.ensure(options, on_event)
        run = _Run(self, on_event, cancelled)
        with self._busy():
            started = time.perf_counter()
            transcription_dir = out_dir / "transcription"
            transcription_dir.mkdir(parents=True, exist_ok=True)
            run.stage("Transcribing audio", total=4, unit="windows")
            run.emit({"type": "token", "phase": "transcription", "tokens": 12, "seconds": 0.1})
            (transcription_dir / "score.abc").write_text(FAKE_ABC)
            (transcription_dir / "result.json").write_text(json.dumps(
                {"fake": True, "status": "complete", "truncated": False, "task": task}))
            transcription_seconds = time.perf_counter() - started
            song_request = {**request, "abc": FAKE_ABC}
            (out_dir / "request.json").write_text(json.dumps(song_request))
            summary = self._run_create(run, song_request, out_dir, options, provided_abc=True)
        summary["transcription"] = {"dir": str(transcription_dir), "task": task,
                                    "seconds": transcription_seconds, "source_audio_sha256": "fake",
                                    "duration_seconds": 16.0}
        summary["timing"]["transcription_seconds"] = transcription_seconds
        (out_dir / "cover.json").write_text(json.dumps({"fake": True, "task": task}))
        (out_dir / "summary.json").write_text(json.dumps(summary))
        return summary

    # -- internals ----------------------------------------------------------------------------

    @staticmethod
    def _validate(request: dict, *, allow_abc: bool) -> None:
        allowed = {"style", "lyrics", "cot", "seed", "abc", "cfg_scale", "id",
                   "abc_sampling", "semantic_sampling", "generation_config"}
        unknown = set(request) - allowed
        if unknown:
            raise TypeError(f"unexpected request fields: {sorted(unknown)}")
        for key in ("style", "lyrics"):
            if not isinstance(request.get(key), str):
                raise TypeError("style and lyrics must be strings")
        if request.get("cot", "full") not in ("off", "melody", "full"):
            raise ValueError("cot must be off, melody or full")
        seed = request.get("seed", 831001)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        abc = request.get("abc")
        if abc is not None and (request.get("cot", "full") == "off" or not str(abc).strip()):
            raise ValueError("External ABC requires nonempty text and cot=melody/full")

    class _Busy:
        def __init__(self, engine):
            self.engine = engine

        def __enter__(self):
            self.engine.state = "busy"

        def __exit__(self, exc_type, exc, tb):
            if exc is not None and not isinstance(exc, InterruptedError):
                self.engine.unload()  # mirror Engine._after_failure: a poisoned pipeline is discarded
            else:
                self.engine.state = "ready" if self.engine.pipeline is not None else "cold"
            return False

    def _busy(self):
        return FakeEngine._Busy(self)

    def _run_create(self, run: _Run, request: dict, out_dir: Path, options: EngineOptions, *,
                    provided_abc: bool | None = None) -> dict:
        song_dir, plan_dir = out_dir / "song", out_dir / "plan"
        cot = request.get("cot", "full")
        abc_in = request.get("abc")
        start = time.perf_counter()
        run.stage("Verifying model files", total=None, unit=None)
        run.stage(f"Loading {options.precision} AR model", total=None, unit=None)

        # -- plan ---------------------------------------------------------------------------
        abc_tokens = 0
        if cot == "off":
            abc_text = None
            run.stage("Planning score", total=None, unit="tokens")
            abc_timing = {"seconds": 0.0, "output_tokens": 0}
        elif abc_in:
            abc_text = abc_in
            run.stage("Using provided score", total=None, unit=None)
            abc_timing = {"seconds": 0.0, "output_tokens": 0, "external_prefix_tokens": len(abc_in) // 4}
        else:
            abc_text = FAKE_ABC
            pieces = ["X:1\n", "X:1\nT:Fake\n", "X:1\nT:Fake\nK:C\n"]
            run.stage_start = time.perf_counter()
            run.emit({"type": "stage", "stage": "Planning score", "completed": 0, "total": None,
                      "unit": "tokens", "status": "running", "seconds": 0.0})
            run.step()
            for i, text in enumerate(pieces, start=1):
                elapsed = round(time.perf_counter() - run.stage_start, 3)
                run.emit({"type": "stage", "stage": "Planning score", "completed": i * 4, "total": None,
                          "unit": "tokens", "status": "running", "tps": 4.0, "seconds": elapsed})
                run.emit({"type": "token", "phase": "abc", "tokens": i * 4, "tps": 4.0, "seconds": elapsed})
                run.emit({"type": "abc", "phase": "abc", "text": text, "tokens": i * 4, "status": "partial"})
                run.step()
            abc_tokens = 16
            elapsed = round(time.perf_counter() - run.stage_start, 3)
            run.emit({"type": "stage", "stage": "Planning score", "completed": abc_tokens, "total": None,
                      "unit": "tokens", "status": "completed", "tps": 4.0, "seconds": elapsed})
            run.stages["Planning score"] = elapsed
            abc_timing = {"seconds": elapsed, "output_tokens": abc_tokens}
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "plan.json").write_text(json.dumps({"fake": True, "cot": cot}))
        if abc_text is not None:
            (plan_dir / "score.abc").write_text(abc_text)
            run.emit({"type": "abc", "phase": "abc", "text": abc_text,
                      "tokens": abc_tokens or len(abc_text) // 4, "status": "final"})

        # -- semantic / synth / decode --------------------------------------------------------
        semantic_tokens = 25
        sem_seconds = run.stage("Generating song", total=semantic_tokens, unit="tokens",
                                tokens_phase="semantic")
        nar_seconds = run.stage("Synthesizing audio", total=options.ode_steps, unit="steps",
                                fail=self.fail)
        vae_seconds = run.stage("Decoding audio", total=2, unit="chunks")

        # -- artifacts ----------------------------------------------------------------------
        if song_dir.exists() and any(song_dir.iterdir()):
            raise FileExistsError("Use an empty artifact directory to avoid mixing recordings")
        song_dir.mkdir(parents=True, exist_ok=True)
        write_silence_flac(song_dir / "audio.flac", self.audio_seconds)
        if abc_text is not None:
            (song_dir / "score.abc").write_text(abc_text)
        (song_dir / "plan.json").write_text(json.dumps({"fake": True, "cot": cot}))
        (song_dir / "request.json").write_text(json.dumps(request))
        (song_dir / "result.json").write_text(json.dumps({"fake": True, "status": "complete"}))
        e2e = time.perf_counter() - start
        timing = {
            "abc": abc_timing,
            "semantic": {"seconds": sem_seconds, "output_tokens": semantic_tokens},
            "nar_seconds": nar_seconds, "vae_seconds": vae_seconds, "e2e_seconds": e2e,
            "load": {"ar_load_seconds": 0.01}, "stages": dict(run.stages),
        }
        summary = {
            "status": "complete",
            "audio_path": str(song_dir / "audio.flac"),
            "score_path": str(song_dir / "score.abc") if abc_text is not None else None,
            "song_dir": str(song_dir), "plan_dir": str(plan_dir),
            "sample_rate": SAMPLE_RATE, "seconds": self.audio_seconds,
            "timing": timing, "truncated": {"abc": False, "semantic": False},
            "identity": "fake", "preset": options.preset, "precision": options.precision,
            "ode_steps": options.ode_steps, "seed": request.get("seed", 831001),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary))
        run.emit({"type": "log", "text": f"Completed {self.audio_seconds:.1f}s of audio in {e2e:.1f}s"})
        return summary
