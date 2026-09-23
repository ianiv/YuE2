"""Tests for yue2_studio.config; must not import mlx or lyra."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from yue2_studio import config


def test_importing_config_sets_tf32_off_and_avoids_mlx():
    assert os.environ["MLX_ENABLE_TF32"] == "0"
    # Other test modules may import mlx in this process; check a fresh interpreter instead.
    code = (
        "import json, os, sys; os.environ.pop('MLX_ENABLE_TF32', None); from yue2_studio import config; "
        "print(json.dumps({'tf32': os.environ.get('MLX_ENABLE_TF32'), "
        "'mlx': any(n == 'mlx' or n.startswith('mlx.') for n in sys.modules), "
        "'lyra': 'lyra' in sys.modules}))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert json.loads(out) == {"tf32": "0", "mlx": False, "lyra": False}


def test_home_is_main_checkout_not_worktree():
    assert config.HOME.is_absolute()
    assert ".worktrees" not in config.HOME.parts
    assert config.DATA_DIR == config.HOME / "data"
    assert config.SONGS_DIR == config.DATA_DIR / "songs"
    assert config.UPLOADS_DIR == config.DATA_DIR / "uploads"
    assert config.DB_PATH == config.DATA_DIR / "app.db"
    assert config.MODELS_DIR == config.HOME / "models"
    assert config.CONVERTED_DIR == config.MODELS_DIR / "converted"
    assert config.VAE_DIR == config.MODELS_DIR / "vae"
    assert config.HF_CACHE_DIR == config.MODELS_DIR / "hf-cache"


def test_home_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("YUE2_STUDIO_HOME", str(tmp_path))
    import importlib

    module = importlib.reload(config)
    try:
        assert module.HOME == tmp_path.resolve()
        assert module.SONGS_DIR == tmp_path.resolve() / "data" / "songs"
    finally:
        monkeypatch.delenv("YUE2_STUDIO_HOME")
        importlib.reload(config)


@pytest.mark.parametrize(
    ("name", "precision", "steps"),
    [("quality", "bf16", 32), ("fast", "8bit", 8)],
)
def test_named_presets(name, precision, steps):
    options = config.resolve_preset(name)
    assert (options.preset, options.precision, options.ode_steps) == (name, precision, steps)
    assert options.memory_budget_gib == config.DEFAULT_MEMORY_BUDGET_GIB
    assert options.require_ac is config.DEFAULT_REQUIRE_AC
    assert options.fast_numerics is config.DEFAULT_FAST_NUMERICS


def test_named_preset_accepts_overrides():
    options = config.resolve_preset("fast", ode_steps=16, memory_budget_gib=20, require_ac=True,
                                    fast_numerics=False)
    assert (options.precision, options.ode_steps, options.memory_budget_gib, options.require_ac,
            options.fast_numerics) == ("8bit", 16, 20.0, True, False)


def test_fast_numerics_never_forces_a_pipeline_rebuild():
    exact = config.resolve_preset("quality", fast_numerics=False)
    assert exact.build_key == config.resolve_preset("quality", fast_numerics=True).build_key


def test_custom_preset_requires_and_validates_fields():
    with pytest.raises(ValueError):
        config.resolve_preset("custom")
    with pytest.raises(ValueError):
        config.resolve_preset("custom", precision="fp8", ode_steps=8)
    with pytest.raises(ValueError):
        config.resolve_preset("custom", precision="4bit", ode_steps=2)
    with pytest.raises(ValueError):
        config.resolve_preset("custom", precision="4bit", ode_steps=65)
    options = config.resolve_preset("custom", precision="4bit", ode_steps=64)
    assert (options.preset, options.precision, options.ode_steps) == ("custom", "4bit", 64)


def test_unknown_preset():
    with pytest.raises(ValueError):
        config.resolve_preset("turbo")


def test_build_key_ignores_ode_steps():
    a = config.resolve_preset("fast", ode_steps=8)
    b = config.resolve_preset("fast", ode_steps=32)
    c = config.resolve_preset("quality")
    assert a.build_key == b.build_key
    assert a.build_key != c.build_key


def test_hf_snapshot_dir_layout():
    path = config.hf_snapshot_dir("m-a-p/SheetSage2", "abc123", Path("/cache"))
    assert path == Path("/cache/models--m-a-p--SheetSage2/snapshots/abc123")


def test_cover_available_shape():
    info = config.cover_available()
    assert set(info) >= {"available", "ffmpeg", "sheetsage2", "mert", "ffmpeg_path"}
    assert info["available"] == (info["ffmpeg"] and info["sheetsage2"] and info["mert"])


def test_presets_summary_lists_all():
    assert [p["name"] for p in config.presets_summary()] == ["quality", "fast", "custom"]
