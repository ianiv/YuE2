"""Paths, presets and environment for YuE2 Studio.

Import this module before anything that may import ``mlx``: the guarded runtime in mlx-Yue
refuses to run unless ``MLX_ENABLE_TF32=0`` was set before MLX initialised. This module must
never import ``mlx`` or ``lyra`` itself so that API handlers and tests can use it freely.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import shutil  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

PRECISIONS = ("bf16", "8bit", "4bit")
MIN_ODE_STEPS, MAX_ODE_STEPS = 4, 64

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MEMORY_BUDGET_GIB = 24
DEFAULT_REQUIRE_AC = False

# Hugging Face repositories (revisions for the transcription models are the ones pinned by
# lyra.transcription.model; scripts/setup.py asserts they still agree).
MODEL_REPO = "vanch007/mlx-Yue2-3B"
VAE_REPO = "m-a-p/YuE2-Vae"
SHEETSAGE_REPO = "m-a-p/SheetSage2"
SHEETSAGE_REVISION = "eab522a8168e8b8b8c4856bf8609cd86198f01fe"
MERT_REPO = "m-a-p/MERT-v2-FullSong"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"


def _repo_root() -> Path:
    """Main checkout root, even when running from a worktree under ``.worktrees/``.

    ``git rev-parse --git-common-dir`` points at the primary ``.git`` directory for every
    worktree, so its parent is the shared repository root that owns ``data/`` and ``models/``.
    Without git (no binary, not a checkout, e.g. an installed wheel) the fallback is the parent of
    the ``yue2_studio`` package, i.e. whichever checkout the code runs from: a worktree when run
    from one. Set ``YUE2_STUDIO_HOME`` to override either result explicitly.
    """
    package_dir = Path(__file__).resolve().parent
    try:
        common = subprocess.run(
            ["git", "-C", str(package_dir), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if common:
            return Path(common).resolve().parent
    except (OSError, subprocess.SubprocessError):
        pass
    return package_dir.parent


HOME = Path(os.environ.get("YUE2_STUDIO_HOME") or _repo_root()).expanduser().resolve()
DATA_DIR = HOME / "data"
SONGS_DIR = DATA_DIR / "songs"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "app.db"
MODELS_DIR = HOME / "models"
CONVERTED_DIR = MODELS_DIR / "converted"
VAE_DIR = MODELS_DIR / "vae"
HF_CACHE_DIR = MODELS_DIR / "hf-cache"
FFMPEG = shutil.which("ffmpeg")


def ensure_dirs() -> None:
    for path in (SONGS_DIR, UPLOADS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EngineOptions:
    """Everything the engine needs to pick/build a pipeline and configure one job."""

    precision: str = "bf16"
    ode_steps: int = 32
    memory_budget_gib: float = DEFAULT_MEMORY_BUDGET_GIB
    require_ac: bool = DEFAULT_REQUIRE_AC
    preset: str = "quality"

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if type(self.ode_steps) is not int or not MIN_ODE_STEPS <= self.ode_steps <= MAX_ODE_STEPS:
            raise ValueError(f"ode_steps must be an integer in [{MIN_ODE_STEPS}, {MAX_ODE_STEPS}]")

    @property
    def build_key(self) -> tuple:
        """Fields that require a pipeline rebuild when they change."""
        return (self.precision, float(self.memory_budget_gib), bool(self.require_ac))


@dataclass(frozen=True)
class Preset:
    name: str
    label: str
    precision: str | None
    ode_steps: int | None
    description: str


PRESETS: dict[str, Preset] = {
    "quality": Preset("quality", "Quality", "bf16", 32, "BF16 AR, 32 ODE steps (reference quality)"),
    "fast": Preset("fast", "Fast", "8bit", 8, "8-bit AR, 8 ODE steps (faster than realtime)"),
    "custom": Preset("custom", "Custom", None, None, "Pick precision (bf16/8bit/4bit) and 4-64 ODE steps"),
}


def resolve_preset(
    name: str,
    precision: str | None = None,
    ode_steps: int | None = None,
    *,
    memory_budget_gib: float = DEFAULT_MEMORY_BUDGET_GIB,
    require_ac: bool = DEFAULT_REQUIRE_AC,
) -> EngineOptions:
    preset = PRESETS.get(name)
    if preset is None:
        raise ValueError(f"Unknown preset {name!r}; choose one of {sorted(PRESETS)}")
    if preset.name == "custom":
        if precision is None or ode_steps is None:
            raise ValueError("The custom preset requires precision and ode_steps")
    else:
        precision = preset.precision if precision is None else precision
        ode_steps = preset.ode_steps if ode_steps is None else ode_steps
    return EngineOptions(
        precision=precision,
        ode_steps=int(ode_steps),
        memory_budget_gib=float(memory_budget_gib),
        require_ac=bool(require_ac),
        preset=preset.name,
    )


def presets_summary() -> list[dict]:
    return [
        {"name": p.name, "label": p.label, "precision": p.precision, "ode_steps": p.ode_steps,
         "description": p.description}
        for p in PRESETS.values()
    ]


def hf_snapshot_dir(repo: str, revision: str, cache_dir: Path = HF_CACHE_DIR) -> Path:
    """Where ``huggingface_hub.snapshot_download(repo, revision=..., cache_dir=...)`` lands."""
    return cache_dir / f"models--{repo.replace('/', '--')}" / "snapshots" / revision


def models_available() -> dict:
    return {
        "converted": (CONVERTED_DIR / "conversion.json").is_file(),
        "vae": (VAE_DIR / "config.json").is_file() and (VAE_DIR / "model.safetensors").is_file(),
        "precisions": [p for p in PRECISIONS if (CONVERTED_DIR / f"ar-{p}.safetensors").is_file()],
    }


def cover_available() -> dict:
    """Whether the cover flow can run offline: ffmpeg plus both transcription snapshots."""
    sheetsage = hf_snapshot_dir(SHEETSAGE_REPO, SHEETSAGE_REVISION)
    mert = hf_snapshot_dir(MERT_REPO, MERT_REVISION)
    checks = {
        "ffmpeg": FFMPEG is not None,
        "sheetsage2": all((sheetsage / n).is_file() for n in ("config.json", "model.safetensors")),
        "mert": all((mert / n).is_file() for n in ("config.json", "model.safetensors")),
    }
    return {"available": all(checks.values()), **checks, "ffmpeg_path": FFMPEG}
