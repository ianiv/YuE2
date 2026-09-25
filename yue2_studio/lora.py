"""LoRA adapters for the YuE2 generator (AR backbone and acoustic NAR layers).

Adapters live under ``models/loras/`` as either

* one ``<name>.safetensors`` file in the layout published for YuE2-3B (e.g. Mothersuperior's
  ``ar_lora_inst_v3abc`` / ``nar_lora_joint_v4``): ``layers.N.<block>.<proj>.lora_A`` ``[r, in]`` and
  ``.lora_B`` ``[out, r]`` with ``W' = W + lora_scale * lora_B @ lora_A`` (``lora_scale`` from the file
  metadata, default 1.0). ``vae2llm.{weight,bias}`` / ``llm2vae.{weight,bias}`` entries are full
  replacement weights for those NAR layers, or
* a ``<name>/`` directory holding a PEFT export: ``adapter_config.json`` (``r``, ``lora_alpha``) plus
  ``adapter_model.safetensors`` with ``base_model.model.<module>.lora_{A,B}.weight`` tensors.

Adapters are *merged* into the resident weights (FP32 opmath, BF16 storage; quantized linears are
dequantised, merged and requantised with their own group size / bits). Merging keeps mlx-Yue's
source-precision modules untouched, and the NAR's acoustic conditioning through the shared BF16 AR
layers sees adapted AR weights just like the reference merge scripts.

Target modules are named as in the model tree: ``model.layers.N.self_attn.q_proj`` ...
``lm_head`` (AR subset, ``part="ar"``); ``model.layers.N.nar_self_attn.*`` / ``nar_mlp.*`` /
``llm2vae`` / ``vae2llm`` (acoustic model, ``part="nar"``). A ``layers.N.`` key is the same as
``model.layers.N.``.

Only ``apply_adapter`` imports ``mlx``; discovery and validation are pure Python so the HTTP side
can list adapters.
"""

from __future__ import annotations

from yue2_studio import config as _config  # isort: skip  (sets MLX_ENABLE_TF32 before anything imports mlx)

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

del _config

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MIN_SCALE, MAX_SCALE = 0.0, 4.0
MAX_STACK = 8  # adapters per job
PEFT_WEIGHTS = "adapter_model.safetensors"
PEFT_CONFIG = "adapter_config.json"

_AR_LINEAR = re.compile(r"^(model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)|lm_head)$")
_NAR_LINEAR = re.compile(
    r"^(model\.layers\.\d+\.(nar_self_attn\.[qkvo]_proj|nar_mlp\.(gate|up|down)_proj)|llm2vae|vae2llm)$"
)
_REPLACEABLE = ("llm2vae", "vae2llm")  # NAR layers an adapter may ship whole
_HUM_PROJ = re.compile(r"^hum_proj\.(\d+)\.(weight|bias)$")  # hum-to-song conditioning projections
_DTYPES = {"F32", "F16", "BF16"}


def valid_name(name: Any) -> bool:
    return isinstance(name, str) and NAME_PATTERN.fullmatch(name) is not None and ".." not in name


def check_scale(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("scale must be a finite number")
    if not MIN_SCALE <= value <= MAX_SCALE:
        raise ValueError(f"scale must be in [{MIN_SCALE:g}, {MAX_SCALE:g}]")
    return float(value)


def part_of(module: str) -> str | None:
    """``"ar"`` / ``"nar"`` for a supported target module path, ``None`` otherwise."""
    if _AR_LINEAR.fullmatch(module):
        return "ar"
    if _NAR_LINEAR.fullmatch(module):
        return "nar"
    return None


def _kind(module: str) -> str:
    """``model.layers.3.self_attn.q_proj`` -> ``self_attn.q_proj`` (what the UI lists as targets)."""
    return re.sub(r"^model\.layers\.\d+\.", "", module)


def _sort_key(module: str) -> tuple:
    match = re.match(r"^model\.layers\.(\d+)\.(.*)$", module)
    return (0, int(match.group(1)), match.group(2)) if match else (1, 0, module)


# ---------------------------------------------------------------------------------------------
# safetensors header (no mlx / numpy needed)
# ---------------------------------------------------------------------------------------------


def read_safetensors_header(path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """``(tensors, metadata)``: name -> ``{"dtype", "shape", "data_offsets"}`` and ``__metadata__``."""
    with Path(path).open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors file is truncated")
        (size,) = struct.unpack("<Q", prefix)
        if size > 100 * 2**20:
            raise ValueError("safetensors header is implausibly large")
        header = json.loads(handle.read(size).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    metadata = header.pop("__metadata__", None) or {}
    for name, spec in header.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("shape"), list):
            raise ValueError(f"safetensors entry {name!r} is malformed")
    return header, {str(k): str(v) for k, v in metadata.items()} if isinstance(metadata, dict) else {}


# ---------------------------------------------------------------------------------------------
# discovery / description
# ---------------------------------------------------------------------------------------------


@dataclass
class LoraTarget:
    """One low-rank delta: tensor names (``a`` is ``[r, in]``, ``b`` is ``[out, r]``) for ``module``."""

    module: str
    a_name: str
    b_name: str
    rank: int
    in_features: int
    out_features: int


@dataclass
class Replacement:
    """A full weight (and optional bias) that replaces ``module``'s parameters."""

    module: str
    weight_name: str
    bias_name: str | None
    shape: tuple[int, ...]


@dataclass
class HumProjection:
    """One ``Linear(latent -> hidden)`` a hum adapter adds to the NAR hidden state at ``inject_layers[i]``."""

    index: int
    weight_name: str
    bias_name: str
    hidden: int
    latent: int


@dataclass
class AdapterInfo:
    """What the studio knows about one adapter without reading any tensor data.

    ``kind`` is ``"lora"`` for a plain weight-merge adapter or ``"hum"`` for a hum-to-song adapter,
    which carries a NAR LoRA plus ``hum_proj.*`` projections applied inside the decoder loop
    (``yue2_studio.hum_nar``); hum adapters are selected on the Hum page, never in the LoRA stack.
    """

    name: str
    path: Path
    format: str | None = None  # "safetensors" | "peft"
    weights_path: Path | None = None
    rank: int | None = None
    scale: float | None = None  # scale baked into the adapter (metadata lora_scale or alpha / r)
    dtype: str | None = None
    targets: list[str] = field(default_factory=list)  # distinct module kinds, e.g. "self_attn.q_proj"
    ar_modules: int = 0
    nar_modules: int = 0
    replaced: list[str] = field(default_factory=list)
    size_bytes: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    kind: str = "lora"  # "lora" | "hum"
    inject_layers: list[int] = field(default_factory=list)
    loras: list[LoraTarget] = field(default_factory=list, repr=False)
    replacements: list[Replacement] = field(default_factory=list, repr=False)
    hum_proj: list[HumProjection] = field(default_factory=list, repr=False)

    @property
    def valid(self) -> bool:
        return self.error is None

    @property
    def parts(self) -> set[str]:
        modules = [t.module for t in self.loras] + [r.module for r in self.replacements]
        return {p for p in (part_of(m) for m in modules) if p is not None}

    def modules(self, part: str) -> list[str]:
        return [m for m in [t.module for t in self.loras] + [r.module for r in self.replacements]
                if part_of(m) == part]

    def to_dict(self) -> dict:
        keep = ("recipe", "intended_cot", "base_model", "rank", "lora_scale", "delta", "requires",
                "inject_layers", "condition")
        return {
            "name": self.name, "path": str(self.path), "format": self.format, "valid": self.valid,
            "kind": self.kind, "inject_layers": list(self.inject_layers), "hum_proj": len(self.hum_proj),
            "rank": self.rank, "scale": self.scale, "dtype": self.dtype, "targets": list(self.targets),
            "parts": sorted(self.parts), "ar_modules": self.ar_modules, "nar_modules": self.nar_modules,
            "replaced": list(self.replaced), "size_bytes": self.size_bytes,
            "metadata": {k: v for k, v in self.metadata.items() if k in keep}, "error": self.error,
        }

    def identity(self) -> dict:
        """Stable description for ``weights["loras"]`` / request identities (hashes the weights file)."""
        return {"name": self.name, "format": self.format, "kind": self.kind, "rank": self.rank,
                "scale": self.scale, "sha256": sha256_file(self.weights_path),
                "modules": self.ar_modules + self.nar_modules, "replaced": list(self.replaced),
                "inject_layers": list(self.inject_layers)}


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]


def _module_path(key: str) -> str:
    """Normalise a checkpoint key prefix to the model-tree module path."""
    for prefix in ("base_model.model.", "base_model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    if key.startswith("layers."):
        key = "model." + key
    return key


def _classify(header: dict[str, dict], fmt: str) -> tuple[list[LoraTarget], list[Replacement],
                                                          list[HumProjection]]:
    """Group tensors into LoRA pairs, full replacements and hum projections; anything else is an error."""
    a_suffix, b_suffix = (".lora_A.weight", ".lora_B.weight") if fmt == "peft" else (".lora_A", ".lora_B")
    pairs: dict[str, dict[str, str]] = {}
    whole: dict[str, dict[str, str]] = {}
    proj: dict[int, dict[str, str]] = {}
    unknown: list[str] = []
    for name in header:
        hum = _HUM_PROJ.match(name) if fmt == "safetensors" else None
        if hum is not None:
            proj.setdefault(int(hum.group(1)), {})[hum.group(2)] = name
            continue
        if name.endswith(a_suffix):
            module, which = _module_path(name[: -len(a_suffix)]), "a"
        elif name.endswith(b_suffix):
            module, which = _module_path(name[: -len(b_suffix)]), "b"
        elif name.endswith((".weight", ".bias")) and fmt == "safetensors":
            module, which = _module_path(name.rsplit(".", 1)[0]), name.rsplit(".", 1)[1]
            if module in _REPLACEABLE:
                whole.setdefault(module, {})[which] = name
                continue
            unknown.append(name)
            continue
        else:
            unknown.append(name)
            continue
        if part_of(module) is None:
            unknown.append(name)
            continue
        pairs.setdefault(module, {})[which] = name
    if unknown:
        raise ValueError(f"unsupported tensors {', '.join(sorted(unknown)[:4])}")
    loras = []
    for module in sorted(pairs, key=_sort_key):
        names = pairs[module]
        if set(names) != {"a", "b"}:
            raise ValueError(f"module {module!r} is missing its lora_A or lora_B tensor")
        a_shape, b_shape = header[names["a"]]["shape"], header[names["b"]]["shape"]
        if len(a_shape) != 2 or len(b_shape) != 2:
            raise ValueError(f"module {module!r}: LoRA tensors must be 2-D")
        (rank, in_features), (out_features, rank_b) = a_shape, b_shape
        if rank != rank_b or rank < 1:
            raise ValueError(f"module {module!r}: lora_A [{rank}, in] and lora_B [out, {rank_b}] disagree")
        loras.append(LoraTarget(module, names["a"], names["b"], int(rank), int(in_features),
                                int(out_features)))
    replacements = []
    for module in sorted(whole):
        names = whole[module]
        if "weight" not in names:
            raise ValueError(f"module {module!r} ships a bias without its weight")
        shape = header[names["weight"]]["shape"]
        if len(shape) != 2:
            raise ValueError(f"module {module!r}: replacement weight must be 2-D")
        replacements.append(Replacement(module, names["weight"], names.get("bias"),
                                        tuple(int(x) for x in shape)))
    projections = []
    for index in sorted(proj):
        names = proj[index]
        if set(names) != {"weight", "bias"}:
            raise ValueError(f"hum_proj.{index} needs both weight and bias")
        w_shape, b_shape = header[names["weight"]]["shape"], header[names["bias"]]["shape"]
        if len(w_shape) != 2 or len(b_shape) != 1 or b_shape[0] != w_shape[0]:
            raise ValueError(f"hum_proj.{index}: weight must be [hidden, latent] with a matching bias")
        projections.append(HumProjection(index, names["weight"], names["bias"], int(w_shape[0]),
                                         int(w_shape[1])))
    if [p.index for p in projections] != list(range(len(projections))):
        raise ValueError("hum_proj indices must be contiguous from 0")
    if not loras and not replacements:
        raise ValueError("adapter contains no LoRA tensors")
    return loras, replacements, projections


def _inject_layers(metadata: dict[str, str], projections: list[HumProjection]) -> list[int]:
    raw = metadata.get("inject_layers")
    if raw is None:
        raise ValueError("hum adapter needs inject_layers metadata (one NAR layer index per hum_proj)")
    try:
        layers = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"metadata inject_layers {raw!r} is not a JSON list") from None
    if (not isinstance(layers, list) or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                                                 for x in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("metadata inject_layers must be a list of distinct non-negative layer indices")
    if len(layers) != len(projections):
        raise ValueError(f"inject_layers lists {len(layers)} layers but the file has "
                         f"{len(projections)} hum_proj")
    return [int(x) for x in layers]


def _peft_scale(config: dict, rank: int) -> float:
    if str(config.get("peft_type", "LORA")).upper() != "LORA":
        raise ValueError(f"peft_type must be LORA, got {config.get('peft_type')!r}")
    if config.get("alpha_pattern") or config.get("rank_pattern") or config.get("use_dora"):
        raise ValueError("alpha_pattern / rank_pattern / DoRA adapters are not supported")
    alpha = config.get("lora_alpha", rank)
    if isinstance(alpha, bool) or not isinstance(alpha, int | float) or alpha <= 0:
        raise ValueError("lora_alpha must be a positive number")
    return float(alpha) / (math.sqrt(rank) if config.get("use_rslora") else rank)


def _metadata_scale(metadata: dict[str, str]) -> float:
    raw = metadata.get("lora_scale", "1.0")
    try:
        scale = float(raw)
    except ValueError:
        raise ValueError(f"metadata lora_scale {raw!r} is not a number") from None
    if not math.isfinite(scale):
        raise ValueError("metadata lora_scale must be finite")
    return scale


def describe_adapter(path: Path, name: str | None = None) -> AdapterInfo:
    """Inspect one adapter (a ``.safetensors`` file or a PEFT directory). Never raises: see ``error``."""
    path = Path(path)
    info = AdapterInfo(name=name or (path.stem if path.suffix == ".safetensors" else path.name), path=path)
    try:
        config: dict | None = None
        if path.is_file():
            if path.suffix != ".safetensors":
                raise ValueError("adapter files must be .safetensors")
            info.format, info.weights_path = "safetensors", path
        elif path.is_dir():
            config_path, weights = path / PEFT_CONFIG, path / PEFT_WEIGHTS
            if not config_path.is_file() or not weights.is_file():
                raise FileNotFoundError(f"a PEFT adapter directory needs {PEFT_CONFIG} and {PEFT_WEIGHTS}")
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(f"unreadable {PEFT_CONFIG}: {error}") from None
            if not isinstance(config, dict):
                raise ValueError(f"{PEFT_CONFIG} must contain a JSON object")
            info.format, info.weights_path = "peft", weights
        else:
            raise FileNotFoundError("not found")
        header, info.metadata = read_safetensors_header(info.weights_path)
        info.loras, info.replacements, info.hum_proj = _classify(header, info.format)
        names = [t.a_name for t in info.loras] + [t.b_name for t in info.loras]
        names += [r.weight_name for r in info.replacements]
        names += [r.bias_name for r in info.replacements if r.bias_name]
        names += [p.weight_name for p in info.hum_proj] + [p.bias_name for p in info.hum_proj]
        dtypes = {header[n]["dtype"] for n in names}
        unsupported = dtypes - _DTYPES
        if unsupported:
            raise ValueError(f"unsupported tensor dtype {sorted(unsupported)[0]}")
        info.dtype = sorted(dtypes)[0] if len(dtypes) == 1 else "mixed"
        ranks = {t.rank for t in info.loras}
        if len(ranks) > 1:
            raise ValueError("adapters with per-module ranks are not supported")
        info.rank = next(iter(ranks)) if ranks else None
        if info.format == "peft":
            info.scale = _peft_scale(config or {}, info.rank or 1)
        else:
            info.scale = _metadata_scale(info.metadata)
        info.targets = sorted({_kind(t.module) for t in info.loras})
        info.ar_modules = sum(part_of(t.module) == "ar" for t in info.loras)
        info.nar_modules = sum(part_of(t.module) == "nar" for t in info.loras)
        info.replaced = [r.module for r in info.replacements]
        if info.hum_proj:
            info.kind = "hum"
            info.inject_layers = _inject_layers(info.metadata, info.hum_proj)
            if info.parts != {"nar"}:
                raise ValueError("a hum adapter may only touch the acoustic (NAR) model")
        info.size_bytes = info.weights_path.stat().st_size
    except (OSError, ValueError) as error:
        info.error = str(error) or error.__class__.__name__
        info.loras, info.replacements, info.hum_proj = [], [], []
    return info


def _candidate(loras_dir: Path, name: str) -> Path | None:
    file, directory = loras_dir / f"{name}.safetensors", loras_dir / name
    if file.is_file():
        return file
    if directory.is_dir():
        return directory
    return None


def list_adapters(loras_dir: Path) -> list[AdapterInfo]:
    """Every ``*.safetensors`` file and adapter directory under ``loras_dir``, valid or not."""
    loras_dir = Path(loras_dir)
    if not loras_dir.is_dir():
        return []
    adapters = []
    for child in sorted(loras_dir.iterdir(), key=lambda p: p.name.lower()):
        if child.name.startswith("."):
            continue
        name = child.stem if child.is_file() and child.suffix == ".safetensors" else child.name
        if child.is_file() and child.suffix != ".safetensors":
            continue
        if valid_name(name) and (child.is_dir() or child.is_file()):
            adapters.append(describe_adapter(child, name))
    return adapters


def find_adapter(loras_dir: Path, name: str) -> AdapterInfo:
    """A valid ``AdapterInfo`` for ``name`` or ``FileNotFoundError`` / ``ValueError``."""
    if not valid_name(name):
        raise ValueError(f"invalid LoRA adapter name {name!r}")
    path = _candidate(Path(loras_dir), name)
    if path is None:
        raise FileNotFoundError(f"LoRA adapter {name!r} not found in {loras_dir}")
    info = describe_adapter(path, name)
    if not info.valid:
        raise ValueError(f"LoRA adapter {name!r} is unusable: {info.error}")
    if info.kind == "hum":
        raise ValueError(f"{name!r} is a hum-to-song adapter; select it on the Hum page, "
                         "not in the LoRA stack")
    return info


def find_hum_adapter(loras_dir: Path, name: str) -> AdapterInfo:
    """A valid ``kind == "hum"`` adapter for ``name`` or ``FileNotFoundError`` / ``ValueError``."""
    if not valid_name(name):
        raise ValueError(f"invalid hum adapter name {name!r}")
    path = _candidate(Path(loras_dir), name)
    if path is None:
        raise FileNotFoundError(f"hum adapter {name!r} not found in {loras_dir}")
    info = describe_adapter(path, name)
    if not info.valid:
        raise ValueError(f"hum adapter {name!r} is unusable: {info.error}")
    if info.kind != "hum":
        raise ValueError(f"{name!r} is a plain LoRA adapter, not a hum-to-song adapter (no hum_proj tensors)")
    return info


def list_hum_adapters(loras_dir: Path) -> list[AdapterInfo]:
    return [a for a in list_adapters(loras_dir) if a.valid and a.kind == "hum"]


def summary(loras_dir: Path) -> dict:
    return {"dir": str(loras_dir), "adapters": [a.to_dict() for a in list_adapters(loras_dir)]}


# ---------------------------------------------------------------------------------------------
# merging (mlx)
# ---------------------------------------------------------------------------------------------


def apply_adapter(model: Any, info: AdapterInfo, *, part: str, scale: float = 1.0) -> int:
    """Merge every ``part`` ("ar" | "nar") target of ``info`` into ``model``; return the module count.

    LoRA deltas are added (``W += scale * info.scale * B @ A``); replacement weights overwrite the
    module's parameters (``scale`` does not apply to them). Raises ``ValueError`` when a target is
    missing from the model or its shape disagrees. Work is per module, so peak memory is one
    dequantised weight plus its delta; the tensor file is memory-mapped by ``mx.load``.

    An AR without ``lm_head`` (mlx-Yue low-memory mode loads the BF16 AR only to precompute the NAR
    conditioning, ``conditioning_only``) skips ``lm_head`` targets: the head never touches the
    conditioning, and the generating AR is a separate load that gets the full merge.
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    if not info.valid:
        raise ValueError(f"LoRA adapter {info.name!r} is unusable: {info.error}")
    factor = float(info.scale or 0.0) * check_scale(scale)
    targets = [t for t in info.loras if part_of(t.module) == part]
    if part == "ar" and getattr(model, "lm_head", None) is None:
        targets = [t for t in targets if t.module != "lm_head"]
    replacements = [r for r in info.replacements if part_of(r.module) == part]
    if not targets and not replacements:
        return 0
    modules = dict(model.named_modules())
    missing = [m for m in [t.module for t in targets] + [r.module for r in replacements] if m not in modules]
    if missing:
        raise ValueError(f"LoRA targets missing from the {part.upper()} model: {', '.join(missing[:5])}")
    tensors = mx.load(str(info.weights_path))
    merged = 0
    for target in targets:
        module = modules[target.module]
        a = tensors[target.a_name].astype(mx.float32)
        b = tensors[target.b_name].astype(mx.float32)
        delta = (b @ a) * factor  # [out, in]
        if isinstance(module, nn.QuantizedLinear):
            biases = module.get("biases")
            weight = mx.dequantize(module.weight, module.scales, biases, group_size=module.group_size,
                                   bits=module.bits, mode=module.mode)
            _check_shape(target.module, weight.shape, delta.shape)
            fused = (weight.astype(mx.float32) + delta).astype(mx.bfloat16)
            quantized = mx.quantize(fused, group_size=module.group_size, bits=module.bits, mode=module.mode)
            names = ("weight", "scales", "biases")[: len(quantized)]
            updates = [(f"{target.module}.{n}", v) for n, v in zip(names, quantized, strict=True)]
        elif isinstance(module, nn.Linear):
            weight = module.weight
            _check_shape(target.module, weight.shape, delta.shape)
            updates = [(f"{target.module}.weight", (weight.astype(mx.float32) + delta).astype(weight.dtype))]
        else:
            raise TypeError(f"LoRA target {target.module!r} is a {type(module).__name__}, not a linear layer")
        model.update(tree_unflatten(updates))
        mx.eval(*(v for _, v in updates))
        merged += 1
    for item in replacements:
        module = modules[item.module]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"replacement target {item.module!r} is a {type(module).__name__}, not nn.Linear")
        weight = tensors[item.weight_name]
        _check_shape(item.module, module.weight.shape, weight.shape)
        updates = [(f"{item.module}.weight", weight.astype(module.weight.dtype))]
        if item.bias_name is not None:
            if module.get("bias") is None:
                raise ValueError(f"replacement for {item.module!r} has a bias but the layer has none")
            bias = tensors[item.bias_name]
            _check_shape(item.module + ".bias", module.bias.shape, bias.shape)
            updates.append((f"{item.module}.bias", bias.astype(module.bias.dtype)))
        model.update(tree_unflatten(updates))
        mx.eval(*(v for _, v in updates))
        merged += 1
    del tensors
    return merged


def _check_shape(path: str, weight_shape, other_shape) -> None:
    if tuple(weight_shape) != tuple(other_shape):
        raise ValueError(f"adapter tensor for {path!r} has shape {tuple(other_shape)}, "
                         f"model has {tuple(weight_shape)}")
