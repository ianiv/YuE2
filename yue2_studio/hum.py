"""Hum-to-song helpers that need no GPU: options, open-score trimming, prosody carrier analysis.

The mode has two stages (Mothersuperior/YuE2-hum-to-song):

* **Score continuation** - the hum is transcribed to ABC (SheetSage2, melody only); its trailing rest-only
  lines are dropped and the open score (no ``ABC_END``) is placed in the planner prompt so the AR keeps
  writing it. ``trim_open_score`` / ``open_score_has_notes`` live here; the prompt assembly is
  ``StudioPipeline.plan_continuation`` (``engine.py``).
* **Prosody adapter** - the hum is reduced to a *carrier*: its pYIN pitch track drives a sine whose
  amplitude follows the low-passed hum envelope (no words, no timbre). VAE-encoded to 25 Hz latents it
  conditions the acoustic decoder (``hum_nar.py``). ``analyse_hum`` builds that carrier and finds the vocal
  onset; ``place_condition`` positions the latents inside the song.

This module is imported by the HTTP side (job validation), so ``librosa`` / ``scipy`` / ``soundfile`` are
imported lazily inside the functions that need them.
"""

from __future__ import annotations

from yue2_studio import config as _config  # isort: skip  (sets MLX_ENABLE_TF32 before anything imports mlx)

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

del _config

MELODY_MODES = ("continue", "hum_only", "ignore")
MIN_INFLUENCE, MAX_INFLUENCE = 0.0, 3.0
MAX_OFFSET_S = 600.0

SAMPLE_RATE = 48000
HOP = 512
FRAME = 2048
FMIN, FMAX = 65.0, 1000.0
LATENT_HZ = 25  # 1920 samples per VAE latent frame at 48 kHz
ONSET_DB_BELOW_MAX = 30.0
ONSET_SUSTAIN_S = 0.4
CARRIER_PEAK = 0.9

# Reference trimming rule: trailing whole-bar rests (``Z|``, ``Z4|``), bare voice lines and blanks.
_REST_LINE = re.compile(r"(V: (Vocal|Ins))|(Z\d*\|)|")
_NOTE = re.compile(r"[A-Ga-g]")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class HumOptions:
    """Per-job hum settings.

    ``melody``: ``continue`` (AR continues the hum's open score), ``hum_only`` (the hum *is* the whole
    melody: closed score, like a cover) or ``ignore`` (planner writes its own score; needs the adapter so
    the hum still shapes phrasing). ``adapter`` names a ``kind="hum"`` adapter in ``models/loras``;
    ``influence`` is the classifier-free guidance weight on the hum channel (1 = as trained, 0 = no hum);
    ``offset_s`` is where the hum's carrier starts inside the song.
    """

    melody: str = "continue"
    adapter: str | None = None
    influence: float = 1.0
    offset_s: float = 0.0

    def __post_init__(self):
        from yue2_studio import lora

        if self.melody not in MELODY_MODES:
            raise ValueError(f"melody must be one of {MELODY_MODES}")
        if self.adapter is not None and not lora.valid_name(self.adapter):
            raise ValueError(f"invalid hum adapter name {self.adapter!r}")
        influence = _finite_number(self.influence, "hum_influence")
        if not MIN_INFLUENCE <= influence <= MAX_INFLUENCE:
            raise ValueError(f"hum_influence must be in [{MIN_INFLUENCE:g}, {MAX_INFLUENCE:g}]")
        offset = _finite_number(self.offset_s, "offset_s")
        if not 0.0 <= offset <= MAX_OFFSET_S:
            raise ValueError(f"offset_s must be in [0, {MAX_OFFSET_S:g}]")
        if self.melody == "ignore" and self.adapter is None:
            raise ValueError("melody=ignore needs a hum adapter (otherwise the hum has no effect)")
        object.__setattr__(self, "influence", influence)
        object.__setattr__(self, "offset_s", offset)

    @property
    def transcribes(self) -> bool:
        return self.melody != "ignore"

    def to_dict(self) -> dict:
        return {"melody": self.melody, "adapter": self.adapter, "hum_influence": self.influence,
                "offset_s": self.offset_s}


# ---------------------------------------------------------------------------------------------
# ABC helpers
# ---------------------------------------------------------------------------------------------


def trim_open_score(abc: str) -> str:
    """Drop trailing rest-only bars / bare voice lines so the score ends where the hum stopped singing.

    Keeps the header and ends with a newline: the tokenizer splits exactly at line boundaries, so the
    continuation the AR writes tokenises identically to a score written in one go.
    """
    lines = abc.strip("\n").split("\n")
    while lines and _REST_LINE.fullmatch(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines) + "\n"


def score_body_lines(abc: str) -> list[str]:
    """Music lines after the ``K:`` header that carry at least one note (not only rests)."""
    body, seen_key = [], False
    for line in abc.split("\n"):
        stripped = line.strip()
        if not seen_key:
            seen_key = stripped.startswith("K:")
            continue
        if not stripped or stripped.startswith(("%", "V:", "M:", "K:", "L:", "Q:")):
            continue
        if _NOTE.search(re.sub(r'"[^"]*"', "", stripped)):  # ignore chord symbols like "Am"
            body.append(stripped)
    return body


def open_score_has_notes(abc: str) -> bool:
    return bool(score_body_lines(abc))


# ---------------------------------------------------------------------------------------------
# carrier analysis
# ---------------------------------------------------------------------------------------------


@dataclass
class HumAnalysis:
    sample_rate: int
    samples: int
    f0_hz: np.ndarray  # [frames] float32, gaps filled
    voiced: np.ndarray  # [frames] bool
    voiced_fraction: float
    onset_s: float | None
    sing_s: float | None  # seconds of singing after the onset
    carrier: np.ndarray  # [samples] float32, peak CARRIER_PEAK

    @property
    def duration_s(self) -> float:
        return self.samples / self.sample_rate

    def prosody(self) -> dict:
        voiced_f0 = self.f0_hz[self.voiced] if self.voiced.any() else self.f0_hz
        return {
            "method": "librosa.pyin", "sample_rate": self.sample_rate, "hop": HOP, "frame": FRAME,
            "fmin": FMIN, "fmax": FMAX, "duration_s": round(self.duration_s, 3),
            "voiced_fraction": round(self.voiced_fraction, 4),
            "onset_s": None if self.onset_s is None else round(self.onset_s, 3),
            "sing_s": None if self.sing_s is None else round(self.sing_s, 3),
            "f0_median_hz": round(float(np.median(voiced_f0)), 2),
            "f0_min_hz": round(float(voiced_f0.min()), 2), "f0_max_hz": round(float(voiced_f0.max()), 2),
        }


def _fill_gaps(f0: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs; the leading unvoiced run takes the first voiced value."""
    filled = f0.astype(np.float32).copy()
    voiced = np.where(~np.isnan(filled))[0]
    filled[: voiced[0]] = filled[voiced[0]]
    for i in range(voiced[0] + 1, len(filled)):
        if np.isnan(filled[i]):
            filled[i] = filled[i - 1]
    return filled


def _onset(y: np.ndarray, voiced: np.ndarray, sr: int) -> tuple[float | None, float | None]:
    frames = y[: len(y) // HOP * HOP].reshape(-1, HOP)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    n = min(len(db), len(voiced))
    active = (db[:n] > db[:n].max() - ONSET_DB_BELOW_MAX) & voiced[:n].astype(bool)
    run = int(math.ceil(ONSET_SUSTAIN_S * sr / HOP))
    if n < run:
        return None, None
    sustained = np.convolve(active.astype(int), np.ones(run, dtype=int), "valid") >= run
    starts = np.where(sustained)[0]
    if not len(starts):
        return None, None
    onset = float(starts[0] * HOP / sr)
    sing = float(active[starts[0]:].sum() * HOP / sr)
    return onset, sing


def analyse_hum(y: np.ndarray, *, sr: int = SAMPLE_RATE, cancelled=None) -> HumAnalysis:
    """Pitch-track a mono hum and build its prosody carrier (see module docstring).

    ``cancelled`` is checked before and after pitch tracking (``pyin`` itself is not interruptible; the
    first call also pays numba's JIT warm-up). Raises ``ValueError`` when nothing voiced is found.
    """
    import librosa
    from scipy.signal import butter, sosfiltfilt

    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.ndim != 1 or len(y) < FRAME:
        raise ValueError("The hum is too short to analyse (need at least a few hundred milliseconds)")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before pitch tracking")
    f0, voiced_flag, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=sr, hop_length=HOP, frame_length=FRAME)
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled after pitch tracking")
    voiced = np.asarray(voiced_flag, dtype=bool) & ~np.isnan(f0)
    if not voiced.any():
        raise ValueError("No voiced pitch found in the hum; record a clearer, closer take")
    f0_filled = _fill_gaps(np.where(voiced, f0, np.nan))
    frame_samples = librosa.frames_to_samples(np.arange(len(f0_filled)), hop_length=HOP)
    freq = np.interp(np.arange(len(y)), frame_samples, f0_filled).astype(np.float64)
    env = sosfiltfilt(butter(4, 30.0, btype="low", fs=sr, output="sos"), np.abs(y).astype(np.float64))
    env = sosfiltfilt(butter(2, 80.0, btype="low", fs=sr, output="sos"), env)
    env = np.clip(env, 0.0, None)
    carrier = env * np.sin(2 * np.pi * np.cumsum(freq) / sr)
    peak = float(np.abs(carrier).max())
    if peak <= 0.0:
        raise ValueError("The hum is silent; record a clearer, closer take")
    carrier = (carrier / peak * CARRIER_PEAK).astype(np.float32)
    onset, sing = _onset(y, voiced, sr)
    return HumAnalysis(sample_rate=sr, samples=len(y), f0_hz=f0_filled, voiced=voiced,
                       voiced_fraction=float(voiced.mean()), onset_s=onset, sing_s=sing, carrier=carrier)


def carrier_stereo(carrier: np.ndarray) -> np.ndarray:
    """``[S]`` -> ``[S, 2]`` float32 (the VAE encoder takes stereo)."""
    return np.ascontiguousarray(np.stack([carrier, carrier], axis=1), dtype=np.float32)


def place_condition(latents: np.ndarray, total_frames: int, offset_s: float) -> np.ndarray:
    """Zero ``[total_frames, 64]`` condition with the carrier latents placed at ``offset_s``.

    Frames outside the hum stay zero, which the adapter's projections map to their learned "no hum"
    bias. The carrier is truncated at the end of the song; starting past the end is an error.
    """
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2 or latents.shape[1] != 64:
        raise ValueError("carrier latents must be [T, 64]")
    if type(total_frames) is not int or total_frames < 1:
        raise ValueError("total_frames must be a positive integer")
    offset = int(round(offset_s * LATENT_HZ))
    length = min(len(latents), total_frames - offset)
    if length <= 0:
        raise ValueError(f"the hum offset ({offset_s:g}s) starts after the song ends "
                         f"({total_frames / LATENT_HZ:.1f}s)")
    cond = np.zeros((total_frames, 64), dtype=np.float32)
    cond[offset:offset + length] = latents[:length]
    return cond


def write_prosody(path: Path, analysis: HumAnalysis, options: HumOptions, extra: dict | None = None) -> dict:
    payload = {**analysis.prosody(), **options.to_dict(), **(extra or {})}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def load_prosody(path: Path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None
