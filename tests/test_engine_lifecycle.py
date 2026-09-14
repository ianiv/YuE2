"""Engine failure handling with a fake pipeline injected through ``pipeline_factory``.

Importing ``yue2_studio.engine`` imports ``mlx`` and ``lyra`` (no GPU work is done); the module
is skipped where they are unavailable.
"""

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

    def token_observer(self):
        return lambda phase, token: None

    def check_execution(self):
        if self.latched is not None:
            raise self.latched

    def build_request(self, **fields):
        return self._request_cls(**fields)

    def effective_config(self, request, abc_sampling=None, semantic_sampling=None):
        return {"generation": {"ode_steps": self.generation_config.ode_steps}}

    def plan(self, **kwargs):
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
