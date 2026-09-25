"""``hum_nar.synthesize_hum`` model staging with a fake pipeline (no weights, no GPU work).

Low-memory mode (mlx-Yue ``low_memory``): the NAR conditioning of every chunk is precomputed with the
BF16 AR (``pipe.acoustic_conditioning``) *before* the NAR loads, and each ``HumNAR`` receives its chunk's
conditioning. Without low-memory support (``None`` / no method: the installed mlx-Yue) the calls are
exactly the pre-low-memory ones. Importing ``hum_nar`` imports ``mlx`` and ``lyra``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("lyra.nar")

from yue2_studio import hum_nar  # noqa: E402

FRAMES = 6
RANGES = [(0, 4), (4, 6)]  # two acoustic chunks


class FakePipe:
    """Just the surface ``synthesize_hum`` touches; ``conds`` = what ``acoustic_conditioning`` returns."""

    def __init__(self, conds):
        self.order: list = []
        self._conds = conds
        self.tokenizer = None
        self.generation_config = SimpleNamespace(ode_steps=2, context=8192)
        self.query_chunk_size = None
        self.nar_attention = "exact"

    def check_execution(self):
        pass

    def guarded_cancelled(self, cancelled):
        return lambda: False

    def _load_model(self, for_nar=False):
        self.order.append(("load", for_nar))
        return SimpleNamespace(config={"hidden_size": 16})

    def _status(self, label, **kwargs):
        import contextlib

        stage = SimpleNamespace(update=lambda *a, **k: None)
        return contextlib.nullcontext(stage)


class LowMemoryPipe(FakePipe):
    def acoustic_conditioning(self, chunks, *, cancelled=None):
        self.order.append(("conditioning", len(chunks), cancelled is not None))
        return self._conds


@pytest.fixture
def recorded(monkeypatch):
    engines = []

    class FakeHumNAR:
        def __init__(self, model, chunk, **kwargs):
            engines.append((chunk, kwargs))
            self.frames = len(kwargs["cond"])

        def solve(self, steps, cancelled, report, *, guidance):
            report(steps, steps)
            return np.zeros((self.frames, hum_nar._LATENT_DIM), dtype=np.float32)

        def close(self):
            pass

    monkeypatch.setattr(hum_nar, "HumNAR", FakeHumNAR)
    monkeypatch.setattr(hum_nar, "token_prefixes", lambda request, tokenizer, abc_ids: [1, 2])
    monkeypatch.setattr(hum_nar, "_chunks_with_supplied_noise",
                        lambda prefix, tokens, seed, context, noise: ["chunk0", "chunk1"])
    monkeypatch.setattr(hum_nar, "chunk_ranges", lambda frames, prefix, context: list(RANGES))
    monkeypatch.setattr(hum_nar, "load_hum_projections", lambda adapter, hidden: [("w", "b")])
    return engines


def _run(pipe):
    plan = SimpleNamespace(request=SimpleNamespace(seed=3), prefix=[1, 2], abc_ids=[])
    semantic = SimpleNamespace(plan=plan, tokens=list(range(FRAMES)))
    adapter = SimpleNamespace(inject_layers=[0])
    cond = np.zeros((FRAMES, hum_nar._LATENT_DIM), dtype=np.float32)
    noise = np.zeros((FRAMES, hum_nar._LATENT_DIM), dtype=np.float32)
    return hum_nar.synthesize_hum(pipe, semantic, cond=cond, adapter=adapter, influence=1.0, noise=noise)


def test_low_memory_precomputes_conditioning_before_loading_the_nar(recorded):
    pipe = LowMemoryPipe(["cond0", "cond1"])
    result = _run(pipe)
    assert result.shape == (FRAMES, hum_nar._LATENT_DIM)
    assert pipe.order == [("conditioning", 2, True), ("load", True)]
    assert [chunk for chunk, _ in recorded] == ["chunk0", "chunk1"]
    assert [kwargs["conditioning"] for _, kwargs in recorded] == ["cond0", "cond1"]


@pytest.mark.parametrize("pipe_cls", [LowMemoryPipe, FakePipe])  # returns None / method absent (old pin)
def test_without_low_memory_the_hum_nar_call_is_unchanged(recorded, pipe_cls):
    pipe = pipe_cls(None)
    _run(pipe)
    assert pipe.order[-1] == ("load", True)
    assert all("conditioning" not in kwargs for _, kwargs in recorded)
    assert set(recorded[0][1]) == {"cond", "projections", "inject_layers", "query_chunk_size", "cancelled",
                                   "attention"}


def test_conditioning_count_must_match_the_chunks(recorded):
    with pytest.raises(RuntimeError, match="one conditioning per chunk"):
        _run(LowMemoryPipe(["only one"]))


class _Stop(Exception):
    pass


@pytest.mark.parametrize("conditioning", [None, "precomputed"])
def test_hum_nar_forwards_conditioning_to_cached_nar_only_when_given(monkeypatch, conditioning):
    seen = {}

    def cached_init(self, model, chunk, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(hum_nar.CachedNAR, "__init__", cached_init)
    kwargs = {} if conditioning is None else {"conditioning": conditioning}
    with pytest.raises(_Stop):
        hum_nar.HumNAR(object(), "chunk", cond=np.zeros((1, 64)), projections=[], inject_layers=[],
                       query_chunk_size=None, cancelled=None, attention="native", **kwargs)
    expected = {"query_chunk_size": None, "cancelled": None, "attention": "native"}
    assert seen == (expected if conditioning is None else {**expected, "conditioning": conditioning})
