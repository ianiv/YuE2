"""LoRA adapter discovery, validation, merging and the HTTP surface.

Discovery/validation only read safetensors headers (pure Python). The merge tests need ``mlx`` and
are skipped where it is unavailable; they use tiny stand-in models with the YuE2 module names.
"""

import json
import struct

import numpy as np
import pytest
from conftest import BASE

from yue2_studio import config, jobs, lora
from yue2_studio.jobs import JobStore

# ---------------------------------------------------------------------------------------------
# helpers: write safetensors without the safetensors package
# ---------------------------------------------------------------------------------------------

_DTYPES = {np.dtype("float32"): "F32", np.dtype("float16"): "F16"}


def write_safetensors(path, tensors: dict, metadata: dict | None = None) -> None:
    header, blobs, offset = {}, [], 0
    if metadata:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        data = array.tobytes()
        header[name] = {"dtype": _DTYPES[array.dtype], "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for blob in blobs:
            handle.write(blob)


def ar_tensors(layers=2, rank=4, hidden=16, inter=24, kv=8, seed=0, prefix="layers."):
    """Mothersuperior-style AR LoRA tensors for a toy model."""
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(layers):
        for block, projections in (("self_attn", {"q_proj": (hidden, hidden), "k_proj": (kv, hidden),
                                                  "v_proj": (kv, hidden), "o_proj": (hidden, hidden)}),
                                   ("mlp", {"gate_proj": (inter, hidden), "up_proj": (inter, hidden),
                                            "down_proj": (hidden, inter)})):
            for name, (o, i_) in projections.items():
                out[f"{prefix}{i}.{block}.{name}.lora_A"] = rng.standard_normal((rank, i_), dtype=np.float32)
                out[f"{prefix}{i}.{block}.{name}.lora_B"] = rng.standard_normal((o, rank), dtype=np.float32)
    return out


def nar_tensors(layers=2, rank=3, hidden=16, inter=24, kv=8, latent=4, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(layers):
        for block, projections in (("nar_self_attn", {"q_proj": (hidden, hidden), "k_proj": (kv, hidden),
                                                      "v_proj": (kv, hidden), "o_proj": (hidden, hidden)}),
                                   ("nar_mlp", {"gate_proj": (inter, hidden), "up_proj": (inter, hidden),
                                                "down_proj": (hidden, inter)})):
            for name, (o, i_) in projections.items():
                out[f"layers.{i}.{block}.{name}.lora_A"] = rng.standard_normal((rank, i_), dtype=np.float32)
                out[f"layers.{i}.{block}.{name}.lora_B"] = rng.standard_normal((o, rank), dtype=np.float32)
    out["vae2llm.weight"] = rng.standard_normal((hidden, latent), dtype=np.float32)
    out["vae2llm.bias"] = rng.standard_normal((hidden,), dtype=np.float32)
    out["llm2vae.weight"] = rng.standard_normal((latent, hidden), dtype=np.float32)
    out["llm2vae.bias"] = rng.standard_normal((latent,), dtype=np.float32)
    return out


@pytest.fixture
def loras_dir(home):
    return config.paths_for(home).loras_dir


@pytest.fixture
def ar_adapter(loras_dir):
    write_safetensors(loras_dir / "inst.safetensors", ar_tensors(), {"lora_scale": "1.0", "rank": "4",
                                                                      "intended_cot": "full"})
    return "inst"


@pytest.fixture
def nar_adapter(loras_dir):
    write_safetensors(loras_dir / "realaudio.safetensors", nar_tensors(), {"lora_scale": "1.0"})
    return "realaudio"


@pytest.fixture
def peft_adapter(loras_dir):
    directory = loras_dir / "peft-style"
    tensors = {f"base_model.model.model.{k}.weight" if k.endswith(("lora_A", "lora_B")) else k: v
               for k, v in ar_tensors(layers=1, rank=2, seed=3).items()}
    write_safetensors(directory / lora.PEFT_WEIGHTS, tensors)
    (directory / lora.PEFT_CONFIG).write_text(json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 8}))
    return "peft-style"


# ---------------------------------------------------------------------------------------------
# discovery / validation
# ---------------------------------------------------------------------------------------------


def test_names_and_scales():
    assert lora.valid_name("ar_lora_inst_v3abc.bf16")
    assert not lora.valid_name("../x") and not lora.valid_name(".hidden") and not lora.valid_name("a/b")
    assert not lora.valid_name("") and not lora.valid_name(None)
    assert lora.check_scale(1) == 1.0 and lora.check_scale(0.5) == 0.5
    for bad in (-0.1, 4.1, float("nan"), True, "1"):
        with pytest.raises(ValueError):
            lora.check_scale(bad)


def test_part_of_module_paths():
    assert lora.part_of("model.layers.27.self_attn.q_proj") == "ar"
    assert lora.part_of("model.layers.0.mlp.down_proj") == "ar"
    assert lora.part_of("lm_head") == "ar"
    assert lora.part_of("model.layers.3.nar_self_attn.v_proj") == "nar"
    assert lora.part_of("model.layers.3.nar_mlp.gate_proj") == "nar"
    assert lora.part_of("vae2llm") == "nar" and lora.part_of("llm2vae") == "nar"
    assert lora.part_of("model.embed_tokens") is None
    assert lora.part_of("hum_proj.0") is None


def test_describe_single_file_ar_adapter(loras_dir, ar_adapter):
    info = lora.find_adapter(loras_dir, ar_adapter)
    assert info.valid and info.format == "safetensors" and info.rank == 4 and info.scale == 1.0
    assert info.parts == {"ar"} and info.ar_modules == 14 and info.nar_modules == 0
    assert info.targets == ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.k_proj",
                            "self_attn.o_proj", "self_attn.q_proj", "self_attn.v_proj"]
    assert info.dtype == "F32" and info.replaced == []
    assert info.loras[0].module == "model.layers.0.mlp.down_proj"  # sorted by layer then name
    d = info.to_dict()
    assert d["metadata"] == {"lora_scale": "1.0", "rank": "4", "intended_cot": "full"}
    assert d["valid"] and d["error"] is None and d["size_bytes"] > 0
    identity = info.identity()
    assert identity["name"] == "inst" and len(identity["sha256"]) == 64 and identity["modules"] == 14


def test_describe_nar_adapter_with_replacements(loras_dir, nar_adapter):
    info = lora.find_adapter(loras_dir, nar_adapter)
    assert info.parts == {"nar"} and info.nar_modules == 14 and info.replaced == ["llm2vae", "vae2llm"]
    assert info.modules("nar")[-2:] == ["llm2vae", "vae2llm"] and info.modules("ar") == []


def test_describe_peft_directory(loras_dir, peft_adapter):
    info = lora.find_adapter(loras_dir, peft_adapter)
    assert info.format == "peft" and info.rank == 2 and info.scale == 4.0  # alpha 8 / r 2
    assert info.ar_modules == 7 and info.loras[0].module.startswith("model.layers.0.")


def test_metadata_scale_is_honoured(loras_dir):
    write_safetensors(loras_dir / "half.safetensors", ar_tensors(layers=1), {"lora_scale": "0.5"})
    assert lora.find_adapter(loras_dir, "half").scale == 0.5
    write_safetensors(loras_dir / "bad.safetensors", ar_tensors(layers=1), {"lora_scale": "lots"})
    assert "lora_scale" in lora.describe_adapter(loras_dir / "bad.safetensors").error


def hum_tensors(layers=2, projections=2, hidden=16, latent=4, seed=5):
    """A hum-to-song adapter: NAR LoRA + io replacements + ``hum_proj`` injection layers."""
    rng = np.random.default_rng(seed)
    tensors = nar_tensors(layers=layers, hidden=hidden, latent=latent, seed=seed)
    for i in range(projections):
        tensors[f"hum_proj.{i}.weight"] = rng.standard_normal((hidden, latent), dtype=np.float32)
        tensors[f"hum_proj.{i}.bias"] = rng.standard_normal((hidden,), dtype=np.float32)
    return tensors


@pytest.fixture
def hum_adapter(loras_dir):
    write_safetensors(loras_dir / "hum.safetensors", hum_tensors(),
                      {"lora_scale": "1.0", "inject_layers": "[0, 1]"})
    return "hum"


def test_hum_adapter_is_classified_and_kept_out_of_the_stack(loras_dir, hum_adapter):
    info = lora.find_hum_adapter(loras_dir, hum_adapter)
    assert info.valid and info.kind == "hum" and info.inject_layers == [0, 1] and len(info.hum_proj) == 2
    assert info.parts == {"nar"} and info.nar_modules == 14 and info.replaced == ["llm2vae", "vae2llm"]
    assert info.hum_proj[0].hidden == 16 and info.hum_proj[0].latent == 4
    d = info.to_dict()
    assert d["kind"] == "hum" and d["inject_layers"] == [0, 1] and d["hum_proj"] == 2 and d["valid"]
    assert d["metadata"]["inject_layers"] == "[0, 1]"
    assert info.identity()["kind"] == "hum" and info.identity()["inject_layers"] == [0, 1]
    with pytest.raises(ValueError, match="hum-to-song adapter"):
        lora.find_adapter(loras_dir, hum_adapter)  # regular stack refuses it
    assert [a.name for a in lora.list_hum_adapters(loras_dir)] == ["hum"]
    write_safetensors(loras_dir / "plain.safetensors", nar_tensors(layers=1))
    with pytest.raises(ValueError, match="plain LoRA adapter"):
        lora.find_hum_adapter(loras_dir, "plain")
    with pytest.raises(FileNotFoundError):
        lora.find_hum_adapter(loras_dir, "nope")


@pytest.mark.parametrize("metadata, mutate, message", [
    ({}, None, "needs inject_layers"),
    ({"inject_layers": "[0]"}, None, "lists 1 layers"),
    ({"inject_layers": "[0, 0]"}, None, "distinct"),
    ({"inject_layers": "nope"}, None, "not a JSON list"),
    ({"inject_layers": "[0, 1]"}, lambda t: t.pop("hum_proj.1.bias"), "both weight and bias"),
    ({"inject_layers": "[0, 1]"}, lambda t: t.update({"hum_proj.1.weight": np.zeros((4, 4), np.float32)}),
     "matching bias"),
    ({"inject_layers": "[0]"}, lambda t: (t.pop("hum_proj.0.weight"), t.pop("hum_proj.0.bias")),
     "contiguous"),
    ({"inject_layers": "[0, 1]"},
     lambda t: t.update({"layers.0.self_attn.q_proj.lora_A": np.zeros((3, 16), np.float32),
                         "layers.0.self_attn.q_proj.lora_B": np.zeros((16, 3), np.float32)}),
     "only touch the acoustic"),
])
def test_malformed_hum_adapters(loras_dir, metadata, mutate, message):
    tensors = hum_tensors()
    if mutate is not None:
        mutate(tensors)
    write_safetensors(loras_dir / "bad-hum.safetensors", tensors, metadata)
    info = lora.describe_adapter(loras_dir / "bad-hum.safetensors")
    assert not info.valid and message in info.error, info.error


@pytest.mark.parametrize("mutate, message", [
    (lambda t: t.pop("layers.0.self_attn.q_proj.lora_B"), "missing its lora_A or lora_B"),
    (lambda t: t.update({"layers.0.self_attn.q_proj.lora_B": np.zeros((16, 5), np.float32)}), "disagree"),
    (lambda t: t.update({"model.embed_tokens.lora_A": np.zeros((4, 16), np.float32),
                         "model.embed_tokens.lora_B": np.zeros((16, 4), np.float32)}), "unsupported tensors"),
    (lambda t: t.update({"layers.1.mlp.up_proj.lora_A": np.zeros((2, 16), np.float32),
                         "layers.1.mlp.up_proj.lora_B": np.zeros((24, 2), np.float32)}), "per-module ranks"),
    (lambda t: t.clear(), "no LoRA tensors"),
])
def test_malformed_adapters_report_errors(loras_dir, mutate, message):
    tensors = ar_tensors(layers=1)
    mutate(tensors)
    write_safetensors(loras_dir / "broken.safetensors", tensors)
    info = lora.describe_adapter(loras_dir / "broken.safetensors")
    assert not info.valid and message in info.error


def test_list_adapters_and_lookup_errors(loras_dir, ar_adapter, nar_adapter, peft_adapter):
    (loras_dir / "notes.txt").write_text("ignored")
    (loras_dir / ".hidden.safetensors").write_bytes(b"")
    (loras_dir / "empty-dir").mkdir()
    (loras_dir / "bad").mkdir()
    (loras_dir / "bad" / lora.PEFT_CONFIG).write_text("{}")
    names = [(a.name, a.valid) for a in lora.list_adapters(loras_dir)]
    assert names == [("bad", False), ("empty-dir", False), ("inst", True), ("peft-style", True),
                     ("realaudio", True)]
    assert lora.list_adapters(loras_dir / "missing") == []
    with pytest.raises(FileNotFoundError):
        lora.find_adapter(loras_dir, "nope")
    with pytest.raises(ValueError, match="invalid"):
        lora.find_adapter(loras_dir, "../inst")
    with pytest.raises(ValueError, match="unusable"):
        lora.find_adapter(loras_dir, "bad")
    summary = lora.summary(loras_dir)
    assert summary["dir"] == str(loras_dir) and len(summary["adapters"]) == 5


def test_engine_options_normalise_loras():
    options = config.resolve_preset("fast", loras=[{"name": "a"}, ("b", 0.5), "c"])
    assert options.loras == (("a", 1.0), ("b", 0.5), ("c", 1.0))
    assert config.loras_to_api(options.loras)[1] == {"name": "b", "scale": 0.5}
    assert config.resolve_preset("fast").loras == ()
    assert options.build_key == config.resolve_preset("fast").build_key  # never forces a rebuild
    with pytest.raises(ValueError, match="twice"):
        config.resolve_preset("fast", loras=["a", "a"])
    with pytest.raises(ValueError, match="invalid"):
        config.resolve_preset("fast", loras=["../a"])
    with pytest.raises(ValueError, match="scale"):
        config.resolve_preset("fast", loras=[("a", 9)])
    with pytest.raises(ValueError, match="at most"):
        config.resolve_preset("fast", loras=[f"a{i}" for i in range(lora.MAX_STACK + 1)])


# ---------------------------------------------------------------------------------------------
# merging (mlx)
# ---------------------------------------------------------------------------------------------


@pytest.fixture
def mx():
    return pytest.importorskip("mlx.core")


def _toy_models(mx, hidden=16, inter=24, kv=8, latent=4, layers=2):
    import mlx.nn as nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, kv, bias=False)
            self.v_proj = nn.Linear(hidden, kv, bias=False)

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, inter, bias=False)
            self.up_proj = nn.Linear(hidden, inter, bias=False)
            self.down_proj = nn.Linear(inter, hidden, bias=False)

    class ArLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = Attn(), Mlp()

    class NarLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.nar_self_attn, self.nar_mlp = Attn(), Mlp()

    class Backbone(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layers = [layer() for _ in range(layers)]

    class Ar(nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.lm_head = Backbone(ArLayer), nn.Linear(hidden, 32, bias=False)

    class Nar(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone(NarLayer)
            self.vae2llm, self.llm2vae = nn.Linear(latent, hidden), nn.Linear(hidden, latent)

    ar, nar = Ar(), Nar()
    ar.set_dtype(mx.bfloat16)
    nar.set_dtype(mx.bfloat16)
    return ar, nar


def _expected(base, tensors, module, factor):
    a, b = tensors[f"{module}.lora_A"].astype(np.float64), tensors[f"{module}.lora_B"].astype(np.float64)
    # einsum sidesteps a spurious Accelerate matmul warning for these tiny shapes on Apple silicon
    return base.astype(np.float64) + factor * np.einsum("or,ri->oi", b, a)


def test_merge_ar_adapter_into_bf16_linear(mx, loras_dir, ar_adapter):
    ar, _ = _toy_models(mx)
    tensors = ar_tensors()
    info = lora.find_adapter(loras_dir, ar_adapter)
    before = np.array(ar.model.layers[1].mlp.down_proj.weight.astype(mx.float32))
    head_before = np.array(ar.lm_head.weight.astype(mx.float32))
    assert lora.apply_adapter(ar, info, part="ar", scale=0.5) == 14
    after = np.array(ar.model.layers[1].mlp.down_proj.weight.astype(mx.float32))
    expected = _expected(before, tensors, "layers.1.mlp.down_proj", 0.5)
    assert ar.model.layers[1].mlp.down_proj.weight.dtype == mx.bfloat16
    np.testing.assert_allclose(after, expected, rtol=2e-2, atol=2e-2)  # BF16 storage
    assert not np.allclose(after, before)
    np.testing.assert_array_equal(np.array(ar.lm_head.weight.astype(mx.float32)), head_before)  # untouched
    assert lora.apply_adapter(ar, info, part="nar", scale=1.0) == 0  # nothing for the other part


def test_merge_nar_adapter_replaces_io_layers(mx, loras_dir, nar_adapter):
    _, nar = _toy_models(mx)
    tensors = nar_tensors()
    info = lora.find_adapter(loras_dir, nar_adapter)
    before = np.array(nar.model.layers[0].nar_self_attn.k_proj.weight.astype(mx.float32))
    assert lora.apply_adapter(nar, info, part="nar", scale=1.0) == 16
    after = np.array(nar.model.layers[0].nar_self_attn.k_proj.weight.astype(mx.float32))
    np.testing.assert_allclose(after, _expected(before, tensors, "layers.0.nar_self_attn.k_proj", 1.0),
                               rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.array(nar.vae2llm.weight.astype(mx.float32)), tensors["vae2llm.weight"],
                               rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(np.array(nar.llm2vae.bias.astype(mx.float32)), tensors["llm2vae.bias"],
                               rtol=1e-2, atol=1e-2)
    assert nar.vae2llm.weight.dtype == mx.bfloat16


def test_merge_hum_adapter_lora_half_ignores_projections(mx, loras_dir, hum_adapter):
    _, nar = _toy_models(mx)
    info = lora.find_hum_adapter(loras_dir, hum_adapter)
    before = np.array(nar.model.layers[1].nar_mlp.up_proj.weight.astype(mx.float32))
    assert lora.apply_adapter(nar, info, part="nar", scale=1.0) == 16
    after = np.array(nar.model.layers[1].nar_mlp.up_proj.weight.astype(mx.float32))
    assert not np.allclose(after, before)
    assert "hum_proj" not in " ".join(n for n, _ in nar.parameters().items())


def test_merge_into_quantized_linear_requantises(mx, loras_dir):
    import mlx.nn as nn

    ar, _ = _toy_models(mx, hidden=64, inter=128, kv=64)
    tensors = ar_tensors(layers=2, hidden=64, inter=128, kv=64)
    write_safetensors(loras_dir / "wide.safetensors", tensors)
    info = lora.find_adapter(loras_dir, "wide")
    nn.quantize(ar, group_size=64, bits=8, mode="affine",
                class_predicate=lambda _p, m: isinstance(m, nn.Linear))
    q = ar.model.layers[0].self_attn.q_proj
    assert isinstance(q, nn.QuantizedLinear)
    before = np.array(mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=8,
                                    mode="affine").astype(mx.float32))
    assert lora.apply_adapter(ar, info, part="ar", scale=1.0) == 14
    q = ar.model.layers[0].self_attn.q_proj
    assert isinstance(q, nn.QuantizedLinear) and q.weight.dtype == mx.uint32 and q.scales.dtype == mx.bfloat16
    after = np.array(mx.dequantize(q.weight, q.scales, q.biases, group_size=64, bits=8,
                                   mode="affine").astype(mx.float32))
    expected = _expected(before, tensors, "layers.0.self_attn.q_proj", 1.0)
    scale = np.abs(expected).max()
    assert np.abs(after - expected).max() < 0.05 * scale  # 8-bit affine round trip


def test_merge_errors(mx, loras_dir, ar_adapter):
    ar, nar = _toy_models(mx)
    info = lora.find_adapter(loras_dir, ar_adapter)
    with pytest.raises(ValueError, match="missing from the AR model"):
        lora.apply_adapter(nar, info, part="ar")
    small, _ = _toy_models(mx, hidden=8, inter=8, kv=8)
    with pytest.raises(ValueError, match="shape"):
        lora.apply_adapter(small, info, part="ar")
    with pytest.raises(ValueError, match="scale"):
        lora.apply_adapter(ar, info, part="ar", scale=7)
    broken = lora.describe_adapter(loras_dir / "nope.safetensors")
    with pytest.raises(ValueError, match="unusable"):
        lora.apply_adapter(ar, broken, part="ar")


# ---------------------------------------------------------------------------------------------
# job store + HTTP
# ---------------------------------------------------------------------------------------------


def test_store_round_trips_loras_and_migrates_old_databases(tmp_path):
    import sqlite3

    db = tmp_path / "app.db"
    old_schema = jobs.SCHEMA.replace(",\n    loras_json TEXT", "")
    assert "loras_json" not in old_schema
    conn = sqlite3.connect(db)
    conn.executescript(old_schema)
    conn.close()
    store = JobStore(db)
    options = config.resolve_preset("fast", loras=[("inst", 0.8), ("realaudio", 1.0)])
    job = store.create(kind="create", params={"style": "s", "lyrics": "l"}, options=options, seed=1)
    assert job.loras == (("inst", 0.8), ("realaudio", 1.0))
    assert job.to_api()["loras"] == [{"name": "inst", "scale": 0.8}, {"name": "realaudio", "scale": 1.0}]
    assert job.options().loras == options.loras
    plain = store.create(kind="create", params={"style": "s", "lyrics": "l"},
                         options=config.resolve_preset("fast"), seed=2)
    assert plain.loras == () and plain.to_api()["loras"] == []
    store.close()


async def test_submit_with_loras_and_regenerate_inheritance(api, client, home, loras_dir, ar_adapter,
                                                            nar_adapter, engine):
    r = await client.get("/api/loras")
    assert r.status_code == 200
    names = [a["name"] for a in r.json()["adapters"]]
    assert names == ["inst", "realaudio"] and all(a["valid"] for a in r.json()["adapters"])
    status = (await client.get("/api/status")).json()
    assert [a["name"] for a in status["loras"]["adapters"]] == ["inst", "realaudio"]

    job = await api.create(preset="fast", loras=[{"name": "inst", "scale": 0.7}, {"name": "realaudio"}])
    assert job["loras"] == [{"name": "inst", "scale": 0.7}, {"name": "realaudio", "scale": 1.0}]
    progress, done = await api.events(job["id"])
    assert done["status"] == "done"
    merge = [e for e in progress if e.get("label") == "Merging LoRA into 8bit AR model"]
    assert merge and merge[0]["stage"] == "load" and merge[-1]["status"] == "complete"
    assert engine.loras == job["loras"]  # what the fake "merged" for the last job
    summary = json.loads((home / "data" / "songs" / job["id"] / "summary.json").read_text())
    assert summary["loras"] == job["loras"]
    job_json = json.loads((home / "data" / "songs" / job["id"] / "job.json").read_text())
    assert job_json["loras"] == job["loras"]
    assert (await client.get("/api/status")).json()["engine"]["loras"] == job["loras"]

    regen = {"parent_id": job["id"], "abc": "X:1\nK:C\n"}
    inherited = (await api.submit({"kind": "regenerate", "params": regen}))["job"]
    assert inherited["loras"] == job["loras"]
    cleared = (await api.submit({"kind": "regenerate", "loras": [],
                                 "params": regen}))["job"]
    assert cleared["loras"] == []
    swapped = (await api.submit({"kind": "regenerate", "loras": [{"name": "realaudio", "scale": 2}],
                                 "params": regen}))["job"]
    assert swapped["loras"] == [{"name": "realaudio", "scale": 2.0}]

    group = await api.submit({"kind": "variations", "loras": [{"name": "inst"}],
                              "params": {"count": 2, "base": BASE}})
    assert all(j["loras"] == [{"name": "inst", "scale": 1.0}] for j in group["jobs"])


async def test_submit_rejects_unknown_or_invalid_loras(api, client, loras_dir, ar_adapter):
    for body, fragment in [
        ({"loras": [{"name": "missing"}]}, "unknown or unusable LoRA adapter 'missing'"),
        ({"loras": [{"name": "../inst"}]}, "loras.0.name"),
        ({"loras": [{"name": "inst", "scale": 5}]}, "loras.0.scale"),
        ({"loras": [{"name": "inst"}, {"name": "inst"}]}, "listed twice"),
        ({"loras": "inst"}, "loras"),
    ]:
        r = await client.post("/api/jobs", json={"kind": "create", "params": BASE, **body})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "validation_error"
        assert fragment in r.json()["error"]["message"], r.text
    write_safetensors(loras_dir / "hum.safetensors", hum_tensors(), {"inject_layers": "[0, 1]"})
    r = await client.post("/api/jobs", json={"kind": "create", "params": BASE, "loras": [{"name": "hum"}]})
    assert r.status_code == 400 and "unusable" in r.json()["error"]["message"]
    listed = (await client.get("/api/loras")).json()["adapters"]
    hum = next(a for a in listed if a["name"] == "hum")
    assert hum["valid"] and hum["kind"] == "hum"  # listed (for the Hum page) but refused in the stack
