"""Paths, presets and environment for YuE2 Studio.

Import this module before anything that may import ``mlx``: the guarded runtime in mlx-Yue
refuses to run unless ``MLX_ENABLE_TF32=0`` was set before MLX initialised. This module must
never import ``mlx`` or ``lyra`` itself so that API handlers and tests can use it freely.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

import math  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

PRECISIONS = ("bf16", "8bit", "4bit")
MIN_ODE_STEPS, MAX_ODE_STEPS = 4, 64

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
# mlx-Yue's GPU guard rejects a budget <= 5 GiB or above total RAM - 4 GiB (OS headroom), so the studio's
# range is [6, total - 4] and the default is 24 GiB capped to the machine (12 GiB on a 16 GB Mac).
MIN_MEMORY_BUDGET_GIB = 6
PREFERRED_MEMORY_BUDGET_GIB = 24
OS_HEADROOM_GIB = 4
DEFAULT_REQUIRE_AC = False
# Low-memory mode (mlx-Yue ``low_memory``): synthesis stages the models so only one of the AR / NAR is
# resident (~7 GiB peak instead of ~11, bit-identical audio) and the guard's swap / available-memory
# thresholds are relaxed. It costs model reloads on every Quality job, so ``auto`` only turns it on for
# machines that need it: total RAM <= 24 GiB, or an effective budget below 14 GiB.
LOW_MEMORY_MODES = ("auto", "on", "off")
DEFAULT_LOW_MEMORY = "auto"
LOW_MEMORY_AUTO_MAX_RAM_GIB = 24
LOW_MEMORY_AUTO_MIN_BUDGET_GIB = 14
# mlx-Yue ``fast_numerics``: batched CFG branches + native BF16 acoustic attention. Numerically
# equivalent to the exact path but not bit-identical, so a seed made with one mode does not
# reproduce under the other.
DEFAULT_FAST_NUMERICS = True

# Cover source clips (seconds): the shortest clip worth transcribing and the latest bound accepted.
MIN_CLIP_S = 1.0
MAX_CLIP_S = 3600.0

# Hugging Face repositories (revisions for the transcription models are the ones pinned by
# lyra.transcription.model; scripts/setup.py asserts they still agree).
MODEL_REPO = "vanch007/mlx-Yue2-3B"
VAE_REPO = "m-a-p/YuE2-Vae"
SHEETSAGE_REPO = "m-a-p/SheetSage2"
SHEETSAGE_REVISION = "eab522a8168e8b8b8c4856bf8609cd86198f01fe"
MERT_REPO = "m-a-p/MERT-v2-FullSong"
MERT_REVISION = "d8ba1c745e733b3908ce6ad16ebeb17ac7600a42"


def machine_ram_gib() -> float:
    """Total physical memory in GiB (the figure mlx-Yue's guard checks the budget against)."""
    import psutil

    return psutil.virtual_memory().total / 2**30


def max_memory_budget_gib(total_gib: float | None = None) -> float:
    """Largest budget the guard accepts here: ``floor(total RAM - 4 GiB)``, never below the 6 GiB floor.

    Below 10 GiB of RAM the floor wins and the guard still refuses every job (such Macs are unsupported).
    """
    total = machine_ram_gib() if total_gib is None else float(total_gib)
    return float(max(MIN_MEMORY_BUDGET_GIB, math.floor(total - OS_HEADROOM_GIB)))


def default_memory_budget_gib(total_gib: float | None = None) -> float:
    """``min(24, total RAM - 4)``: 24 GiB on 32 GB+ Macs, 20 on 24 GB, 12 on 16 GB."""
    return float(min(PREFERRED_MEMORY_BUDGET_GIB, max_memory_budget_gib(total_gib)))


def clamp_memory_budget(value, total_gib: float | None = None) -> float:
    """``value`` limited to ``[6, max_memory_budget_gib()]``; a non-number becomes the machine default.

    Stored settings may predate the machine cap (the old default was 24 GiB everywhere) or come from a
    data directory moved between Macs, so every use of the setting goes through this.
    """
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return default_memory_budget_gib(total_gib)
    return float(min(max(float(value), MIN_MEMORY_BUDGET_GIB), max_memory_budget_gib(total_gib)))


def resolve_low_memory(mode: str, memory_budget_gib: float, total_gib: float | None = None) -> bool:
    """Effective low-memory flag: ``on`` / ``off`` as given; ``auto`` = RAM <= 24 GiB or budget < 14 GiB.

    ``memory_budget_gib`` should already be the clamped (effective) budget. Unknown modes count as auto.
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    total = machine_ram_gib() if total_gib is None else float(total_gib)
    return total <= LOW_MEMORY_AUTO_MAX_RAM_GIB or float(memory_budget_gib) < LOW_MEMORY_AUTO_MIN_BUDGET_GIB


DEFAULT_MEMORY_BUDGET_GIB = default_memory_budget_gib()


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
LORAS_DIR = MODELS_DIR / "loras"
HF_CACHE_DIR = MODELS_DIR / "hf-cache"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@dataclass(frozen=True)
class Paths:
    """Every filesystem location derived from one ``home`` directory.

    The module constants above are the process defaults (``YUE2_STUDIO_HOME`` / repo root);
    ``create_app(home=...)`` builds a ``Paths`` for a different root (tests use a tmp dir) and
    passes it explicitly to the store, worker and routes instead of mutating module globals.
    """

    home: Path

    @property
    def data_dir(self) -> Path:
        return self.home / "data"

    @property
    def songs_dir(self) -> Path:
        return self.data_dir / "songs"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def converted_dir(self) -> Path:
        return self.models_dir / "converted"

    @property
    def vae_dir(self) -> Path:
        return self.models_dir / "vae"

    @property
    def loras_dir(self) -> Path:
        return self.models_dir / "loras"

    @property
    def hf_cache_dir(self) -> Path:
        return self.models_dir / "hf-cache"

    def ensure_dirs(self) -> None:
        for path in (self.songs_dir, self.uploads_dir, self.models_dir, self.loras_dir):
            path.mkdir(parents=True, exist_ok=True)


def paths_for(home: str | os.PathLike | None = None) -> Paths:
    """``Paths`` rooted at ``home`` (default: the process ``HOME``)."""
    return Paths(Path(home).expanduser().resolve() if home is not None else HOME)


def ensure_dirs() -> None:
    paths_for().ensure_dirs()


LoraStack = tuple[tuple[str, float], ...]


def normalise_loras(value) -> LoraStack:
    """``[{"name", "scale"?}]`` / ``[(name, scale)]`` / ``["name"]`` -> validated ``((name, scale), ...)``."""
    from yue2_studio import lora as _lora

    if value is None:
        return ()
    if isinstance(value, str | dict):
        value = [value]
    stack: list[tuple[str, float]] = []
    for item in value:
        if isinstance(item, str):
            name, scale = item, 1.0
        elif isinstance(item, dict):
            name, scale = item.get("name"), item.get("scale", 1.0)
        else:
            name, scale = tuple(item)
        if not _lora.valid_name(name):
            raise ValueError(f"invalid LoRA adapter name {name!r}")
        if scale is None:
            scale = 1.0
        try:
            scale = _lora.check_scale(scale)
        except ValueError as error:
            raise ValueError(f"LoRA {name!r}: {error}") from None
        if any(n == name for n, _ in stack):
            raise ValueError(f"LoRA adapter {name!r} is listed twice")
        stack.append((name, scale))
    if len(stack) > _lora.MAX_STACK:
        raise ValueError(f"at most {_lora.MAX_STACK} LoRA adapters per job")
    return tuple(stack)


def loras_to_api(stack: LoraStack) -> list[dict]:
    return [{"name": name, "scale": scale} for name, scale in stack]


@dataclass(frozen=True)
class EngineOptions:
    """Everything the engine needs to pick/build a pipeline and configure one job.

    ``loras`` is the ordered stack of ``(adapter name, scale)`` merged into the resident weights for
    the job (see ``yue2_studio.lora``); it is per job and never forces a pipeline rebuild, and neither
    does ``fast_numerics`` (read by the pipeline on every call). ``low_memory`` is the *resolved* flag
    (``resolve_low_memory``); mlx-Yue fixes it at construction, so it is part of ``build_key``.
    """

    precision: str = "bf16"
    ode_steps: int = 32
    memory_budget_gib: float = DEFAULT_MEMORY_BUDGET_GIB
    require_ac: bool = DEFAULT_REQUIRE_AC
    preset: str = "quality"
    loras: LoraStack = ()
    fast_numerics: bool = DEFAULT_FAST_NUMERICS
    low_memory: bool = False

    def __post_init__(self):
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        if type(self.ode_steps) is not int or not MIN_ODE_STEPS <= self.ode_steps <= MAX_ODE_STEPS:
            raise ValueError(f"ode_steps must be an integer in [{MIN_ODE_STEPS}, {MAX_ODE_STEPS}]")
        object.__setattr__(self, "loras", normalise_loras(self.loras))

    @property
    def build_key(self) -> tuple:
        """Fields that require a pipeline rebuild when they change."""
        return (self.precision, float(self.memory_budget_gib), bool(self.require_ac), bool(self.low_memory))


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
    loras=None,
    fast_numerics: bool = DEFAULT_FAST_NUMERICS,
    low_memory: bool = False,
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
        loras=normalise_loras(loras),
        fast_numerics=bool(fast_numerics),
        low_memory=bool(low_memory),
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


def models_available(paths: Paths | None = None) -> dict:
    converted = CONVERTED_DIR if paths is None else paths.converted_dir
    vae = VAE_DIR if paths is None else paths.vae_dir
    return {
        "converted": (converted / "conversion.json").is_file(),
        "vae": (vae / "config.json").is_file() and (vae / "model.safetensors").is_file(),
        "precisions": [p for p in PRECISIONS if (converted / f"ar-{p}.safetensors").is_file()],
    }


def loras_dir_for(paths: Paths | None = None) -> Path:
    return LORAS_DIR if paths is None else paths.loras_dir


def hum_available(paths: Paths | None = None) -> dict:
    """Whether hum-to-song can run: the cover prerequisites plus librosa (pitch tracking).

    ``find_spec`` keeps librosa (and its numba JIT) out of the HTTP process; only the worker imports it.
    """
    import importlib.util

    info = cover_available(paths)
    info["librosa"] = importlib.util.find_spec("librosa") is not None
    info["available"] = bool(info["available"] and info["librosa"])
    return info


def cover_available(paths: Paths | None = None) -> dict:
    """Whether the cover flow can run offline: ffmpeg plus both transcription snapshots."""
    cache_dir = HF_CACHE_DIR if paths is None else paths.hf_cache_dir
    sheetsage = hf_snapshot_dir(SHEETSAGE_REPO, SHEETSAGE_REVISION, cache_dir)
    mert = hf_snapshot_dir(MERT_REPO, MERT_REVISION, cache_dir)
    checks = {
        "ffmpeg": FFMPEG is not None,
        "sheetsage2": all((sheetsage / n).is_file() for n in ("config.json", "model.safetensors")),
        "mert": all((mert / n).is_file() for n in ("config.json", "model.safetensors")),
    }
    return {"available": all(checks.values()), **checks, "ffmpeg_path": FFMPEG}
