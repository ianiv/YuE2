"""MLX port of the YuE2-Vae *encoder* (mlx-Yue only ships the decoder).

The hum-to-song prosody adapter conditions the acoustic decoder on VAE latents of a synthetic carrier,
so the studio needs ``encode``. The architecture mirrors ``yue2/modeling_vae.py::OobleckEncoder``:
``Conv1d(2->64, k7)`` then six ``EncoderBlock``s (three ``ResidualUnit``s + SnakeBeta + a strided
conv with ``kernel = 2*stride``, ``padding = ceil(stride/2)``) then SnakeBeta + ``Conv1d(2048->128, k3)``.
The 128 output channels are ``(mean, scale)``; ``encode`` returns the posterior mean (the first 64),
exactly like upstream ``YuE2VAE.encode(sample=False)``.

``SnakeBeta`` / ``ResidualUnit`` / ``activation`` are reused from ``lyra.vae``; the checkpoint's
weight-norm parametrisation is folded the same way as ``lyra.vae.decoder_weights`` (all encoder
convolutions are plain ``Conv1d``, so every kernel is transposed ``[out, in, k] -> [out, k, in]``).
"""

from __future__ import annotations

from yue2_studio import config as _config  # isort: skip  (sets MLX_ENABLE_TF32 before mlx import)

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402
from lyra.vae import ResidualUnit, activation  # noqa: E402
from safetensors import safe_open  # noqa: E402

del _config

SAMPLE_RATE = 48000


class EncoderBlock(nn.Module):
    def __init__(self, cin, cout, stride, use_snake):
        super().__init__()
        self.layers = [*[ResidualUnit(cin, d, use_snake) for d in (1, 3, 9)],
                       activation(cin, use_snake),
                       nn.Conv1d(cin, cout, 2 * stride, stride=stride, padding=math.ceil(stride / 2))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OobleckEncoder(nn.Module):
    """Channels-last MLX encoder: ``[B, S, 2]`` float32 -> ``[B, T, latent_dim]`` (mean + scale)."""

    def __init__(self, config):
        super().__init__()
        if config.get("antialias_activation", False):
            raise ValueError("Unsupported encoder architecture")
        c, mults = config["channels"], [1, *config["c_mults"]]
        strides, snake = config["strides"], config.get("use_snake", False)
        self.strides = tuple(strides)
        self.ratio = math.prod(strides)
        self.latent_dim = config["latent_dim"]
        self.mean_dim = self.latent_dim // 2
        self.layers = [nn.Conv1d(config["in_channels"], mults[0] * c, 7, padding=3)]
        for i in range(len(mults) - 1):
            self.layers.append(EncoderBlock(mults[i] * c, mults[i + 1] * c, strides[i], snake))
        self.layers += [activation(mults[-1] * c, snake),
                        nn.Conv1d(mults[-1] * c, self.latent_dim, 3, padding=1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def encoder_weights(weights) -> dict[str, mx.array]:
    """Fold official weight_norm (dim=0) and move kernels to MLX's ``[out, k, in]`` layout."""
    result = {}
    for full_key, value in weights.items():
        if not full_key.startswith("encoder.") or full_key.endswith("weight_g"):
            continue
        key = full_key.removeprefix("encoder.")
        value = np.asarray(value, dtype=np.float32)
        if key.endswith("weight_v"):
            gain = np.asarray(weights[full_key[:-1] + "g"], dtype=np.float32)
            norm = np.sqrt(np.sum(value * value, axis=(1, 2), keepdims=True))
            if np.any(norm == 0):
                raise ValueError("Zero weight-normalization denominator")
            value = (value * (gain / norm)).transpose(0, 2, 1)
            key = key.removesuffix("_v")
        result[key] = mx.array(value)
    return result


def load_encoder(directory) -> OobleckEncoder:
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    model = OobleckEncoder(config["encoder_config"])
    with safe_open(directory / "model.safetensors", framework="numpy") as reader:
        weights = {k: reader.get_tensor(k) for k in reader.keys() if k.startswith("encoder.")}
    model.load_weights(list(encoder_weights(weights).items()), strict=True)
    model.freeze()
    mx.eval(model.parameters())
    return model


def encode(model: OobleckEncoder, audio: np.ndarray, *, chunk_seconds: int = 30, cancelled=None,
           on_progress=None) -> np.ndarray:
    """FP32 stereo ``[S, 2]`` at 48 kHz -> posterior-mean latents ``[T, 64]`` (25 Hz), CPU float32.

    Encoded in ``chunk_seconds`` pieces to bound activations (~370 MB per 30 s); a trailing piece
    shorter than one latent frame (1920 samples) is dropped, as in the reference tooling.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < model.ratio:
        raise ValueError("Expected stereo audio [S, 2] with at least one latent frame")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite values")
    chunk = int(chunk_seconds * SAMPLE_RATE)
    starts = [s for s in range(0, len(audio), chunk) if len(audio) - s >= model.ratio]
    pieces = []
    for index, start in enumerate(starts, 1):
        if cancelled is not None and cancelled():
            raise InterruptedError("Cancelled during VAE encoding")
        latent = model(mx.array(audio[None, start:start + chunk]))[0, :, : model.mean_dim]
        mx.eval(latent)
        pieces.append(np.array(latent, dtype=np.float32))
        mx.clear_cache()
        if on_progress is not None:
            on_progress(index, len(starts))
    latents = np.concatenate(pieces)
    if latents.ndim != 2 or latents.shape[1] != model.mean_dim or not np.isfinite(latents).all():
        raise ValueError("VAE encoder produced non-finite or misshaped latents")
    return latents
