"""Hum-to-song helpers that need no GPU: options, open-score trimming, carrier/onset analysis."""

import json

import numpy as np
import pytest

from yue2_studio import hum

SR = hum.SAMPLE_RATE

# Tail of a real SheetSage2 melody transcription (data/songs/1811b92d…/transcription/score.abc)
SCORE = """X:1
T:
M:2/4
L:1/32
Q:1/4=84
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% intro
V: Vocal
Z4|
V: Ins
Z2|z8z2A2A4|G8c6G2|
% verse
V: Vocal
z4G2G2G4E2E2-|E2D2D4z8|z4C2C2C2C2D2E2-|E2G2E6D2C4|
V: Ins
G4z12|Z3|
V: Vocal
Z4|
V: Ins
Z4|
V: Vocal
Z|
V: Ins
Z|
"""


def synthetic_hum(seconds=3.0, silence=0.5, f_start=220.0, f_end=330.0, sr=SR, noise=0.0, seed=0):
    """Silence, then a glide from ``f_start`` to ``f_end`` with a slow amplitude swell (a clean hum)."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    freq = np.linspace(f_start, f_end, n)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = 0.6 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
    y = env * (np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase))
    y[: int(silence * sr)] = 0.0
    if noise:
        y = y + noise * np.random.default_rng(seed).standard_normal(n)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------------------------


def test_hum_options_defaults_and_validation():
    options = hum.HumOptions()
    assert options.melody == "continue" and options.adapter is None and options.influence == 1.0
    assert options.transcribes and options.to_dict() == {"melody": "continue", "adapter": None,
                                                         "hum_influence": 1.0, "offset_s": 0.0}
    assert hum.HumOptions(melody="ignore", adapter="hum_v1").transcribes is False
    assert hum.HumOptions(influence=2, offset_s=1).influence == 2.0
    for kwargs, message in [
        ({"melody": "loud"}, "melody"),
        ({"melody": "ignore"}, "needs a hum adapter"),
        ({"adapter": "../x"}, "invalid hum adapter"),
        ({"influence": 3.5}, "hum_influence"),
        ({"influence": float("nan")}, "finite"),
        ({"offset_s": -1}, "offset_s"),
        ({"offset_s": True}, "finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            hum.HumOptions(**kwargs)


# ---------------------------------------------------------------------------------------------
# ABC helpers
# ---------------------------------------------------------------------------------------------


def test_trim_open_score_drops_trailing_rests_and_keeps_a_newline():
    trimmed = hum.trim_open_score(SCORE)
    assert trimmed.endswith("G4z12|Z3|\n")
    assert "\nZ|\n" not in trimmed and trimmed.count("V: Vocal\n") == 2
    assert trimmed.startswith("X:1\nT:\nM:2/4")
    assert hum.trim_open_score(trimmed) == trimmed  # idempotent
    # a score that is only header + rests trims down to the header
    header_only = SCORE.split("% intro")[0] + "V: Vocal\nZ4|\nV: Ins\nZ4|\n"
    assert hum.trim_open_score(header_only).rstrip("\n").endswith("K:C")


def test_open_score_has_notes():
    assert hum.open_score_has_notes(SCORE)
    assert hum.score_body_lines(SCORE)[0] == "Z2|z8z2A2A4|G8c6G2|"
    header_only = SCORE.split("% intro")[0]
    assert not hum.open_score_has_notes(header_only)
    rests_only = header_only + '% verse\nV: Vocal\n"Am"Z4|\nV: Ins\nz8|\n'  # rests + a chord symbol
    assert not hum.open_score_has_notes(rests_only)


# ---------------------------------------------------------------------------------------------
# carrier analysis
# ---------------------------------------------------------------------------------------------


def _zero_crossing_rate(x, sr):
    return float(np.count_nonzero(np.diff(np.signbit(x))) / 2 / (len(x) / sr))


def test_analyse_hum_tracks_pitch_and_onset():
    y = synthetic_hum()
    analysis = hum.analyse_hum(y)
    assert analysis.sample_rate == SR and analysis.samples == len(y) and analysis.duration_s == 3.0
    assert analysis.voiced_fraction > 0.6
    assert analysis.onset_s is not None and 0.4 <= analysis.onset_s <= 0.7
    assert analysis.sing_s is not None and analysis.sing_s > 1.5
    carrier = analysis.carrier
    assert carrier.dtype == np.float32 and len(carrier) == len(y)
    assert abs(float(np.abs(carrier).max()) - hum.CARRIER_PEAK) < 1e-5
    assert np.abs(carrier[: int(0.45 * SR)]).max() < 0.02  # silent before the hum starts
    # the carrier's frequency follows the hum: ~245 Hz around 1 s, ~300 Hz around 2.5 s
    for centre, expected in ((1.0, 220 + (330 - 220) * 1.0 / 3), (2.5, 220 + (330 - 220) * 2.5 / 3)):
        window = carrier[int((centre - 0.1) * SR): int((centre + 0.1) * SR)]
        assert abs(_zero_crossing_rate(window, SR) - expected) < 12, (centre, expected)
    prosody = analysis.prosody()
    assert prosody["method"] == "librosa.pyin" and 200 < prosody["f0_median_hz"] < 350
    assert set(prosody) >= {"onset_s", "sing_s", "voiced_fraction", "duration_s", "f0_min_hz", "f0_max_hz"}


def test_analyse_hum_rejects_silence_and_short_input():
    with pytest.raises(ValueError, match="No voiced pitch"):
        hum.analyse_hum(np.zeros(SR * 2, dtype=np.float32))
    with pytest.raises(ValueError, match="too short"):
        hum.analyse_hum(np.zeros(100, dtype=np.float32))


def test_analyse_hum_honours_cancellation():
    with pytest.raises(InterruptedError):
        hum.analyse_hum(synthetic_hum(1.0), cancelled=lambda: True)


def test_carrier_stereo_and_place_condition():
    stereo = hum.carrier_stereo(np.arange(5, dtype=np.float32))
    assert stereo.shape == (5, 2) and stereo.dtype == np.float32 and (stereo[:, 0] == stereo[:, 1]).all()
    z = np.arange(10 * 64, dtype=np.float32).reshape(10, 64)
    cond = hum.place_condition(z, 20, 0.0)
    assert cond.shape == (20, 64) and (cond[:10] == z).all() and not cond[10:].any()
    shifted = hum.place_condition(z, 20, 0.2)  # 0.2 s = 5 frames
    assert (shifted[5:15] == z).all() and not shifted[:5].any() and not shifted[15:].any()
    truncated = hum.place_condition(z, 8, 0.0)
    assert truncated.shape == (8, 64) and (truncated == z[:8]).all()
    with pytest.raises(ValueError, match="after the song ends"):
        hum.place_condition(z, 4, 1.0)
    with pytest.raises(ValueError, match=r"\[T, 64\]"):
        hum.place_condition(np.zeros((3, 8), np.float32), 10, 0.0)


def test_prosody_round_trip(tmp_path):
    analysis = hum.analyse_hum(synthetic_hum(1.5))
    path = tmp_path / "prosody.json"
    options = hum.HumOptions(adapter="a", influence=1.5)
    payload = hum.write_prosody(path, analysis, options, {"latent_frames": 37})
    assert json.loads(path.read_text()) == payload == hum.load_prosody(path)
    assert payload["adapter"] == "a" and payload["hum_influence"] == 1.5 and payload["latent_frames"] == 37
    assert hum.load_prosody(tmp_path / "missing.json") is None
