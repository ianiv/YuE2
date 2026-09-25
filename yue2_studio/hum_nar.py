"""Hum-conditioned acoustic synthesis: mlx-Yue's ``CachedNAR`` plus the prosody-adapter injections.

The adapter (``lora.AdapterInfo`` with ``kind == "hum"``) ships a NAR LoRA that ``StudioPipeline``
merges like any other, plus ``hum_proj.k`` linears (latent 64 -> hidden 2048). Their outputs for the
carrier latents are *added* to the decoder's hidden state: projection 0 at the input (next to
``vae2llm(x_t)``, the time embedding and the positional embedding) and the others before the layers
named in ``inject_layers``. Frames without a hum are zeros, which each projection maps to its learned
"no hum" bias.

Classifier-free guidance on the hum channel ("hum influence" ``g``) compares the conditioned velocity
with the velocity for an all-zero carrier: ``v = v_zero + g * (v_hum - v_zero)``; ``g == 1`` is the
plain conditioned pass (one NAR forward per evaluation), anything else costs two.

``HumNAR.velocity`` is ``lyra.nar.CachedNAR.velocity`` with the two additions; ``solve`` is the
reference midpoint solver calling the guided velocity. ``synthesize_hum`` re-implements the chunk loop
of ``lyra.nar.synthesize`` (which hardcodes ``CachedNAR``) with the same cuts and progress mapping, and
stages the models like ``YuE2Pipeline.synthesize`` in low-memory mode: the NAR conditioning of every chunk
is precomputed with the BF16 AR (``pipe.acoustic_conditioning``) before the NAR is loaded.
Only the worker imports this module (it imports ``mlx``).
"""

from __future__ import annotations

from yue2_studio import config as _config  # isort: skip  (sets MLX_ENABLE_TF32 before mlx import)

import math  # noqa: E402
from collections.abc import Callable  # noqa: E402
from numbers import Real  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from lyra.nar import _LATENT_DIM, CachedNAR, Chunk, _chunks_with_supplied_noise, _logit  # noqa: E402
from yue2.protocol import chunk_ranges, token_prefixes  # noqa: E402

from yue2_studio import hum as hum_mod  # noqa: E402
from yue2_studio import lora as lora_mod  # noqa: E402

del _config

Projection = tuple[mx.array, mx.array]  # (weight [hidden, latent] f32, bias [hidden] f32)


def load_hum_projections(info: lora_mod.AdapterInfo, *, hidden: int,
                         latent: int = _LATENT_DIM) -> list[Projection]:
    """``hum_proj.k`` weights (FP32) in ``inject_layers`` order, shape-checked against the model."""
    if not info.valid or info.kind != "hum":
        raise ValueError(f"{info.name!r} is not a usable hum-to-song adapter")
    tensors = mx.load(str(info.weights_path))
    projections = []
    for spec in info.hum_proj:
        weight = tensors[spec.weight_name].astype(mx.float32)
        bias = tensors[spec.bias_name].astype(mx.float32)
        if tuple(weight.shape) != (hidden, latent) or tuple(bias.shape) != (hidden,):
            raise ValueError(f"hum_proj.{spec.index} is {tuple(weight.shape)}, "
                             f"the model needs [{hidden}, {latent}]")
        projections.append((weight, bias))
    mx.eval(*(t for pair in projections for t in pair))
    return projections


class HumNAR(CachedNAR):
    """``CachedNAR`` whose hidden state receives the carrier projections; supports hum-channel CFG."""

    def __init__(self, model, chunk: Chunk, *, cond: np.ndarray, projections: list[Projection],
                 inject_layers: list[int], query_chunk_size=None, cancelled=None, attention="exact",
                 conditioning=None):
        # ``conditioning`` (low-memory mode) replaces CachedNAR's own AR prefill. It is forwarded only
        # when given, so the call stays valid for an mlx-Yue whose CachedNAR has no such keyword.
        extra = {} if conditioning is None else {"conditioning": conditioning}
        super().__init__(model, chunk, query_chunk_size=query_chunk_size, cancelled=cancelled,
                         attention=attention, **extra)
        cond = np.asarray(cond, dtype=np.float32)
        if cond.shape != (self.nar_length - 2, _LATENT_DIM):
            self.close()
            raise ValueError(f"condition must be {(self.nar_length - 2, _LATENT_DIM)}, got {cond.shape}")
        if len(projections) != len(inject_layers):
            self.close()
            raise ValueError("one hum projection per inject layer is required")
        depth = len(model.model.layers)
        if any(layer < 0 or layer >= depth for layer in inject_layers):
            self.close()
            raise ValueError(f"inject_layers must be in [0, {depth})")
        # The carrier gets the same boundary frame on each side as the ODE state (zeros -> bias).
        padded = mx.pad(mx.array(cond), ((1, 1), (0, 0)))
        self.stem: dict[int, mx.array] = {}
        self.zero: dict[int, mx.array] = {}
        for layer, (weight, bias) in zip(inject_layers, projections, strict=True):
            # [1, T+2, hidden]; the projection of a zero frame is just the bias, kept as a broadcast
            self.stem[layer] = ((padded @ weight.T) + bias).astype(mx.bfloat16)[None, :, :]
            self.zero[layer] = bias.astype(mx.bfloat16)[None, None, :]
        mx.eval(*self.stem.values(), *self.zero.values())

    # -- velocity -------------------------------------------------------------------------------

    def _velocity_with(self, state, raw_t, inject: dict[int, mx.array]) -> mx.array:
        """``CachedNAR.velocity`` (lyra/nar.py) plus the hum additions; returns BF16 ``[frames, 64]``."""
        if self._closed:
            raise RuntimeError("HumNAR is closed")
        if isinstance(raw_t, bool) or not isinstance(raw_t, Real) or not math.isfinite(raw_t):
            raise ValueError("raw_t must be a finite real number")
        state = self._state_array(state)
        boundary_state = mx.pad(state, ((1, 1), (0, 0)))
        shifted = self.model.shift_timestep(float(raw_t))
        x = self.model.vae2llm(boundary_state[None, :, :])
        if 0 in inject:
            x = x + inject[0]
        timesteps = mx.broadcast_to(shifted, (self.nar_length,))
        time_embedding = self.model.time_embedder(timesteps)[None, :, :]
        x = x + time_embedding
        x = x + self.pos_emb

        for index, (nar_layer, (ar_key, ar_value)) in enumerate(zip(self.model.model.layers, self.cache,
                                                                   strict=True)):
            if self._cancelled is not None and self._cancelled():
                raise InterruptedError("Cancelled during acoustic velocity")
            if index > 0 and index in inject:
                x = x + inject[index]
            query, key, value = self._project(
                nar_layer.nar_self_attn,
                nar_layer.nar_input_layernorm(x),
                self.cos,
                self.sin,
            )
            key = mx.concatenate([ar_key, key], axis=2)
            value = mx.concatenate([ar_value, value], axis=2)
            attention = self._attend(query, key, value)
            attention = attention.transpose(0, 2, 1, 3).reshape(1, self.nar_length, -1)
            x = x + nar_layer.nar_self_attn.o_proj(attention)
            x = x + nar_layer.nar_mlp(nar_layer.nar_pre_mlp_layernorm(x))

        x = self.model.ar_model.model.norm(x)
        return self.model.llm2vae(x)[0, 1:-1, :]

    def velocity(self, state, raw_t) -> mx.array:
        return self._velocity_with(state, raw_t, self.stem)

    def guided_velocity(self, state, raw_t, guidance: float) -> mx.array:
        """``v_zero + g * (v_hum - v_zero)``; ``g == 1`` skips the unconditioned pass."""
        if guidance == 1.0:
            return self.velocity(state, raw_t)
        conditioned = self._velocity_with(state, raw_t, self.stem).astype(mx.float32)
        mx.eval(conditioned)
        unconditioned = self._velocity_with(state, raw_t, self.zero).astype(mx.float32)
        return (unconditioned + guidance * (conditioned - unconditioned)).astype(mx.bfloat16)

    # -- solver -----------------------------------------------------------------------------------

    def solve(self, steps=32, cancelled: Callable[[], bool] | None = None,
              on_progress: Callable[[int, int], None] | None = None, *, guidance: float = 1.0) -> np.ndarray:
        """The reference midpoint solver (``CachedNAR.solve``) over the guided velocity."""
        if self._closed:
            raise RuntimeError("HumNAR is closed")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        guidance = hum_mod._finite_number(guidance, "hum_influence")
        state = self._initial_state
        dt = 1.0 / steps
        half_dt = mx.array(dt / 2.0, dtype=mx.float32)
        full_dt = mx.array(dt, dtype=mx.float32)
        mx.eval(half_dt, full_dt)
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            first = self.guided_velocity(state, _logit(t), guidance)
            mx.eval(first)
            midpoint = state - (first.astype(mx.float32) * half_dt).astype(mx.bfloat16)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            velocity = self.guided_velocity(midpoint, _logit(t - dt / 2.0), guidance)
            state = state - (velocity.astype(mx.float32) * full_dt).astype(mx.bfloat16)
            mx.eval(state)
            if on_progress is not None:
                on_progress(step + 1, steps)
        result = np.array(state.astype(mx.float32), dtype=np.float32, copy=True)
        if result.shape != (self.nar_length - 2, _LATENT_DIM):
            raise RuntimeError("Acoustic solver returned an invalid latent shape")
        if not np.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self) -> None:
        self.stem = {}
        self.zero = {}
        super().close()


# ---------------------------------------------------------------------------------------------
# pipeline-level entry points
# ---------------------------------------------------------------------------------------------


def synthesize_hum(pipe, semantic, *, cond: np.ndarray, adapter: lora_mod.AdapterInfo, influence: float,
                   cancelled=None, noise=None) -> np.ndarray:
    """``YuE2Pipeline.synthesize`` with ``HumNAR``: same cuts, noise, stage events and guard checks.

    ``cond`` is the full-song ``[T, 64]`` condition (``hum.place_condition``); it is sliced with the
    same ``chunk_ranges`` as the semantic tokens when a song spans several acoustic chunks.
    """
    pipe.check_execution()
    plan = semantic.plan
    if token_prefixes(plan.request, pipe.tokenizer, plan.abc_ids) != plan.prefix:
        raise ValueError("Semantic result does not retain the request's exact prefix")
    if not semantic.tokens:
        raise ValueError("Semantic result must contain at least one codec frame")
    cond = np.asarray(cond, dtype=np.float32)
    if cond.shape != (len(semantic.tokens), _LATENT_DIM):
        raise ValueError("condition must have one [64] frame per semantic token")
    guarded = pipe.guarded_cancelled(cancelled)
    if guarded():
        raise InterruptedError("Cancelled before acoustic prefill")
    steps, context = pipe.generation_config.ode_steps, pipe.generation_config.context
    seed = plan.request.seed
    if noise is None:
        from lyra.pipeline import initial_noise

        noise = initial_noise(len(semantic.tokens), seed)
    chunks = _chunks_with_supplied_noise(plan.prefix, semantic.tokens, seed, context, noise)
    ranges = chunk_ranges(len(semantic.tokens), len(plan.prefix), int(context))
    # Low-memory mode: the BF16 AR (with the user's AR LoRAs merged) precomputes each chunk's
    # conditioning and is released before the NAR loads; ``None`` otherwise (nothing is loaded).
    # TODO(mlx-Yue pin): call ``pipe.acoustic_conditioning`` directly once the pin provides it.
    precompute = getattr(pipe, "acoustic_conditioning", None)
    conds = None if precompute is None else precompute(chunks, cancelled=guarded)
    if conds is not None:
        conds = list(conds)
        if len(conds) != len(chunks):
            raise RuntimeError("acoustic_conditioning must return one conditioning per chunk")
    model = pipe._load_model(for_nar=True)  # merges the user LoRAs and the adapter's NAR half
    projections = load_hum_projections(adapter, hidden=model.config["hidden_size"])
    total_steps = steps * len(chunks)
    output: list[np.ndarray] = []
    with pipe._status("Synthesizing audio", unit="steps") as status:
        for index, (chunk, (start, end)) in enumerate(zip(chunks, ranges, strict=True)):
            if guarded():
                raise InterruptedError("Cancelled before acoustic prefill")
            extra = {} if conds is None else {"conditioning": conds[index]}
            engine = HumNAR(model, chunk, cond=cond[start:end], projections=projections,
                            inject_layers=adapter.inject_layers, query_chunk_size=pipe.query_chunk_size,
                            cancelled=guarded, attention=pipe.nar_attention, **extra)
            try:
                def report(completed, _total, *, _index=index):
                    pipe.check_execution()
                    status.update(_index * steps + completed, total=total_steps)

                output.append(engine.solve(steps, guarded, report, guidance=influence))
            finally:
                engine.close()
                if conds is not None:
                    conds[index] = None  # a chunk's AR K/V is not needed once it is solved
    del projections, conds
    mx.clear_cache()
    result = output[0] if len(output) == 1 else np.concatenate(output, axis=0)
    if result.shape != (len(semantic.tokens), _LATENT_DIM) or not np.isfinite(result).all():
        raise RuntimeError("Hum-conditioned synthesis returned invalid latents")
    pipe.check_execution()
    return result


def make_synthesizer(pipe, carrier_latents: np.ndarray, adapter: lora_mod.AdapterInfo, influence: float,
                     offset_s: float):
    """A ``synthesize(semantic, *, cancelled, noise)`` drop-in for ``Engine._run_create``."""

    def synthesize(semantic, *, cancelled=None, noise=None):
        cond = hum_mod.place_condition(carrier_latents, len(semantic.tokens), offset_s)
        return synthesize_hum(pipe, semantic, cond=cond, adapter=adapter, influence=influence,
                              cancelled=cancelled, noise=noise)

    return synthesize
