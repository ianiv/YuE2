"""Engine failure handling with a fake pipeline injected through ``pipeline_factory``.

Importing ``yue2_studio.engine`` imports ``mlx`` and ``lyra`` (no GPU work is done); the module
is skipped where they are unavailable.
"""

import contextlib

import pytest

from yue2_studio import config

pytest.importorskip("mlx.core")
pytest.importorskip("lyra.pipeline")

from yue2_studio.engine import Engine  # noqa: E402


class FakePipeline:
    """Just enough of StudioPipeline for Engine.create_song up to ``plan()``."""

    instances: list = []

    def __init__(self, model_dir, vae_dir, *, precision, memory_budget_gib, require_ac, progress, on_event):
        from yue2.protocol import SongRequest

        self.precision = precision
        self.on_event = on_event
        self.generation_config = None
        self.stage_timings = {}
        self.load_timing = {}
        self.weights = {"mot": "fake", "vae": "fake"}
        self.closed = False
        self.latched = None  # error the GPU guard would keep re-raising
        self.plan_error = None
        self.latch_on_plan = None  # error latched by the monitor while plan() was failing
        self._request_cls = SongRequest
        FakePipeline.instances.append(self)

    def close(self):
        self.closed = True

    def release_models(self):
        pass

    def log(self, message):
        pass

    @contextlib.contextmanager
    def _status(self, label, **kwargs):
        yield

    def set_loras(self, stack, hum_adapter=None):
        self.loras = list(stack)
        self.hum_adapter = hum_adapter

    def token_observer(self, user_callback=None, *, abc_prefix=""):
        return lambda phase, token: None

    def check_execution(self):
        if self.latched is not None:
            raise self.latched

    def build_request(self, **fields):
        return self._request_cls(**fields)

    def effective_config(self, request, abc_sampling=None, semantic_sampling=None):
        return {"generation": {"ode_steps": self.generation_config.ode_steps}}

    def plan(self, **kwargs):
        # Mirror StudioPipeline._status, which accumulates seconds by label for the pipeline's lifetime.
        self.stage_timings["Planning score"] = self.stage_timings.get("Planning score", 0.0) + 1.0
        if self.latch_on_plan is not None:
            self.latched = self.latch_on_plan
        if self.plan_error is not None:
            raise self.plan_error
        raise AssertionError("tests stop at plan()")


REQUEST = {"style": "test", "lyrics": "[verse]\nla", "cot": "off", "seed": 1}


@pytest.fixture
def engine():
    FakePipeline.instances.clear()
    return Engine(converted_dir="/nonexistent/converted", vae_dir="/nonexistent/vae",
                  pipeline_factory=FakePipeline)


def test_memory_error_discards_pipeline_and_next_job_rebuilds(engine, tmp_path):
    options = config.resolve_preset("fast")
    engine.ensure(options)
    pipe = engine.pipeline
    pipe.plan_error = MemoryError("Process footprint exceeds budget")
    pipe.latch_on_plan = pipe.plan_error  # the guard keeps re-raising after the monitor latches
    with pytest.raises(MemoryError):
        engine.create_song(REQUEST, tmp_path / "a", options=options)
    assert pipe.closed
    assert engine.pipeline is None
    assert engine.state == "cold"
    engine.ensure(options)
    assert engine.pipeline is not pipe
    assert engine.state == "ready"
    assert len(FakePipeline.instances) == 2


def test_cancellation_keeps_healthy_pipeline(engine, tmp_path):
    options = config.resolve_preset("fast")
    engine.ensure(options)
    pipe = engine.pipeline
    pipe.plan_error = InterruptedError("Cancelled during abc")
    with pytest.raises(InterruptedError):
        engine.create_song(REQUEST, tmp_path / "a", options=options)
    assert not pipe.closed
    assert engine.pipeline is pipe
    assert engine.state == "ready"


def test_cancellation_with_latched_guard_discards_pipeline(engine, tmp_path):
    options = config.resolve_preset("fast")
    engine.ensure(options)
    pipe = engine.pipeline
    pipe.plan_error = InterruptedError("Cancelled during abc")
    pipe.latch_on_plan = RuntimeError("AC power disconnected during acceptance execution")
    with pytest.raises(InterruptedError):
        engine.create_song(REQUEST, tmp_path / "a", options=options)
    assert pipe.closed
    assert engine.state == "cold"


def test_stage_timings_reset_between_jobs_on_resident_pipeline(engine, tmp_path):
    options = config.resolve_preset("fast")
    engine.ensure(options)
    pipe = engine.pipeline
    pipe.plan_error = InterruptedError("Cancelled during abc")  # keeps the pipeline resident
    for name in ("a", "b"):
        with pytest.raises(InterruptedError):
            engine.create_song(REQUEST, tmp_path / name, options=options)
    assert engine.pipeline is pipe
    assert pipe.stage_timings == {"Planning score": 1.0}  # not 2.0: reset at the top of each job


class Tok:
    """Word-level stand-in for the YuE2 tokenizer: one id per whitespace-separated token."""

    def encode(self, text):
        return [hash(w) % 1000 for w in text.replace("\n", " \n ").split(" ") if w]

    def decode(self, ids):
        return " ".join(f"<{i}>" for i in ids)


def test_plan_continuation_builds_an_open_prefix_and_a_closed_plan(monkeypatch):
    from yue2.protocol import ABC_END, ABC_START, MUSIC_START, GenerationConfig, SongRequest, token_prefixes

    from yue2_studio.engine import StudioPipeline

    pipe = StudioPipeline.__new__(StudioPipeline)  # no upstream __init__ (would need the models)
    pipe.tokenizer = Tok()
    pipe.generation_config = GenerationConfig()
    calls = []

    def fake_generate(prefix, sampling, seed, phase, **kwargs):
        calls.append((list(prefix), sampling, seed, phase, kwargs))
        return [7, 8, 9], {"seconds": 1.0, "output_tokens": 3}, False

    monkeypatch.setattr(pipe, "_generate", fake_generate)
    request = SongRequest(style="s", lyrics="l", cot="melody", seed=5)
    open_abc = "X:1\nK:C\nV: Vocal\nC D E |\n"
    plan = pipe.plan_continuation(request, open_abc, abc_sampling={"max_tokens": 50}, cancelled=None,
                                  on_token=None)
    partial = pipe.tokenizer.encode(open_abc)
    (prefix, sampling, seed, phase, kwargs), = calls
    assert prefix == token_prefixes(request, pipe.tokenizer) + partial  # ends inside the score
    assert prefix[-len(partial) - 1] == ABC_START and ABC_END not in prefix and MUSIC_START not in prefix
    assert sampling.max_tokens == 50 and seed == 5 and phase == "abc"
    assert plan.abc_ids == partial + [7, 8, 9] and plan.truncated is False
    assert plan.timing == {"seconds": 1.0, "output_tokens": 3, "continuation_prefix_tokens": len(partial)}
    assert plan.prefix == token_prefixes(request, pipe.tokenizer, plan.abc_ids)  # generate_semantic's check
    assert plan.prefix[-2:] == [ABC_END, MUSIC_START]
    assert plan.abc == pipe.tokenizer.decode(plan.abc_ids)
    with pytest.raises(ValueError, match="newline"):
        pipe.plan_continuation(request, "X:1\nK:C", cancelled=None, on_token=None)
    with pytest.raises(ValueError, match="cot=melody"):
        pipe.plan_continuation(SongRequest(style="s", lyrics="l", cot="off"), open_abc)


def test_run_create_uses_planner_synthesizer_and_config_extra(engine, tmp_path):
    """The hum hooks replace plan()/synthesize() and change the request identity via config_extra."""
    from yue2.storage import identity

    options = config.resolve_preset("fast")
    engine.ensure(options)
    pipe = engine.pipeline
    seen = {}

    def planner(p, native, sampling, cancelled, observer):
        seen["planner"] = (p is pipe, native.cot, sampling)
        raise InterruptedError("stop here")  # the rest of the stage body needs the real models

    with pytest.raises(InterruptedError), engine._busy(pipe):
        engine._run_create(pipe, {**REQUEST, "cot": "melody"}, {}, {"max_tokens": 5}, None, tmp_path / "a",
                           options=options, cancelled=None, planner=planner, config_extra={"hum": {"x": 1}})
    assert seen["planner"] == (True, "melody", {"max_tokens": 5})
    # config_extra is part of the identity: the same request without it hashes differently
    base = pipe.effective_config(pipe.build_request(**{**REQUEST, "cot": "melody"}), {"max_tokens": 5}, None)
    with_extra = {**base, "hum": {"x": 1}}
    assert identity({"request": {}, "config": base, "weights": {}}) != identity(
        {"request": {}, "config": with_extra, "weights": {}})


@pytest.mark.parametrize("mode", ["cover", "continue"])
def test_cover_song_clips_and_continues(engine, tmp_path, monkeypatch, mode):
    """The clip is cut before transcription; continue mode plans from the trimmed open score."""
    from yue2_studio import engine as engine_mod

    score = "X:1\nK:C\nV: Vocal\nC D E |\nV: Vocal\nZ4|\n"
    seen = {}

    def extract_clip(src, dst, *, start_s, end_s):
        seen["clip"] = (start_s, end_s)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"clip")
        return 20.0

    def transcribe(pipe, audio_path, transcription_dir, task, cancelled):
        seen["transcribed"] = (audio_path.name, task)
        transcription_dir.mkdir(parents=True)
        (transcription_dir / "score.abc").write_text(score)
        return {"source_audio_sha256": "abc123", "duration_seconds": 20.0}, 1.0

    def run_create(pipe, fields, *args, planner=None, config_extra=None, abc_prefix="", **kwargs):
        seen.update(fields=fields, planner=planner, config_extra=config_extra, abc_prefix=abc_prefix)
        return {"truncated": None, "timing": {"e2e_seconds": 1.0}}

    monkeypatch.setattr(engine_mod.audio_mod, "extract_clip", extract_clip)
    monkeypatch.setattr(engine, "_transcribe", transcribe)
    monkeypatch.setattr(engine, "_run_create", run_create)
    src = tmp_path / "song.mp3"
    src.write_bytes(b"audio")
    summary = engine.cover_song(src, tmp_path / "job", task="melody-vocal",
                                request={**REQUEST, "cot": "melody"}, options=config.resolve_preset("fast"),
                                mode=mode, clip_start_s=10, clip_end_s=30)
    assert seen["clip"] == (10.0, 30.0) and seen["transcribed"] == ("clip.flac", "melody-vocal")
    assert summary["cover"]["clip"] == {"start_s": 10.0, "end_s": 30.0, "seconds": 20.0,
                                        "path": "source/clip.flac"}
    if mode == "cover":
        assert seen["fields"]["abc"] == score and seen["planner"] is None and seen["config_extra"] is None
    else:
        opened = "X:1\nK:C\nV: Vocal\nC D E |\n"  # trailing rest bars trimmed
        assert "abc" not in seen["fields"] and seen["abc_prefix"] == opened and seen["planner"] is not None
        assert (tmp_path / "job" / "source" / "open.abc").read_text() == opened
        assert seen["config_extra"]["continuation"]["source_audio_sha256"] == "abc123"
        assert summary["cover"]["open_abc"] == "source/open.abc"


def test_cover_song_rejects_bad_mode_and_clip(engine, tmp_path):
    options = config.resolve_preset("fast")
    for kwargs, match in (({"mode": "continue", "task": "full"}, "melody task"), ({"mode": "remix"}, "mode"),
                          ({"clip_start_s": 5, "clip_end_s": 5.5}, "at least"),
                          ({"clip_start_s": -1}, "non-negative")):
        with pytest.raises(ValueError, match=match):
            engine.cover_song(tmp_path / "x.mp3", tmp_path / "job", request=REQUEST, options=options,
                              **{"task": "melody-full", **kwargs})


def test_precision_change_rebuilds_but_ode_steps_does_not(engine):
    engine.ensure(config.resolve_preset("fast"))
    first = engine.pipeline
    engine.ensure(config.resolve_preset("fast", ode_steps=16))
    assert engine.pipeline is first
    engine.ensure(config.resolve_preset("quality"))
    assert first.closed
    assert engine.pipeline is not first
    assert engine.precision == "bf16"
    engine.unload()
    assert engine.state == "cold"


@pytest.mark.parametrize("fast", [True, False])
def test_fast_numerics_is_applied_per_job_without_rebuild(engine, tmp_path, fast):
    engine.ensure(config.resolve_preset("quality", fast_numerics=not fast))
    pipe = engine.pipeline
    pipe.plan_error = InterruptedError("Cancelled during abc")  # stop right after the per-job setup
    with pytest.raises(InterruptedError):
        engine.create_song(REQUEST, tmp_path / "job",
                           options=config.resolve_preset("quality", fast_numerics=fast))
    assert engine.pipeline is pipe and pipe.fast_numerics is fast


@pytest.mark.skipif(not (config.CONVERTED_DIR / "qwen.tiktoken").is_file(), reason="models/converted missing")
def test_open_score_cut_is_an_exact_token_boundary_with_the_real_tokenizer():
    """The continuation tokenises exactly like a score written in one go when the cut ends a line."""
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    from yue2_studio import hum

    tok = YuE2TextTokenizer(config.CONVERTED_DIR / "qwen.tiktoken")
    head = hum.trim_open_score("X:1\nT:\nM:4/4\nL:1/16\nQ:1/4=100\nK:C\n% intro\nV: Vocal\n"
                               "C4D4E4F4|G8A8|\nV: Ins\nZ2|\nV: Vocal\nZ|\nV: Ins\nZ|\n")
    tail = "V: Ins\nZ2|\n% verse\nV: Vocal\nz4G2G2G4E2E2-|E2D2D4z8|\n"
    assert head.endswith("G8A8|\n")
    assert tok.encode(head + tail) == tok.encode(head) + tok.encode(tail)
    # "|\n" is one BPE token: cutting before the newline would split it and shift the continuation
    bare = head.rstrip("\n")
    assert tok.encode(bare + "\n" + tail) != tok.encode(bare) + tok.encode("\n" + tail)
