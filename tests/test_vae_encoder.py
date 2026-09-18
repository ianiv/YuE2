"""MLX port of the YuE2-Vae encoder; skipped where mlx or the converted VAE weights are absent.

The roundtrip test encodes an existing song's decoded audio and compares with the latents that
produced it (the VAE is lossy, so the check is a loose per-frame cosine similarity).
"""

import glob

import numpy as np
import pytest

from yue2_studio import config

pytest.importorskip("mlx.core")
pytest.importorskip("lyra.vae")

from yue2_studio import vae_encoder  # noqa: E402

pytestmark = pytest.mark.skipif(not (config.VAE_DIR / "model.safetensors").is_file(),
                                reason="models/vae is not downloaded")


@pytest.fixture(scope="module")
def encoder():
    return vae_encoder.load_encoder(config.VAE_DIR)


def test_encoder_loads_strictly_and_has_the_published_geometry(encoder):
    assert encoder.ratio == 1920 and encoder.strides == (2, 2, 4, 4, 5, 6)
    assert encoder.latent_dim == 128 and encoder.mean_dim == 64
    assert len(encoder.layers) == 9  # in-conv, six blocks, snake, out-conv


def test_encode_frame_geometry(encoder):
    assert vae_encoder.encode(encoder, np.zeros((2 * 48000, 2), np.float32)).shape == (50, 64)
    # 30.5 s: a 30 s chunk plus a 0.5 s tail -> 750 + 12 frames
    calls = []
    latents = vae_encoder.encode(encoder, np.zeros((int(30.5 * 48000), 2), np.float32),
                                 on_progress=lambda i, n: calls.append((i, n)))
    assert latents.shape == (762, 64) and calls == [(1, 2), (2, 2)]
    with pytest.raises(ValueError, match="stereo"):
        vae_encoder.encode(encoder, np.zeros((48000,), np.float32))
    with pytest.raises(ValueError, match="at least one latent frame"):
        vae_encoder.encode(encoder, np.zeros((100, 2), np.float32))
    with pytest.raises(InterruptedError):
        vae_encoder.encode(encoder, np.zeros((48000, 2), np.float32), cancelled=lambda: True)


def test_roundtrip_against_a_generated_song(encoder):
    import soundfile as sf

    candidates = sorted(glob.glob(str(config.SONGS_DIR / "*" / "song" / "latent.npy")))
    picked = None
    for path in candidates:
        latents = np.load(path)
        if 100 <= len(latents) <= 3000:
            picked = path
            break
    if picked is None:
        pytest.skip("no finished song in data/songs to compare against")
    audio, sr = sf.read(picked.replace("latent.npy", "audio.flac"), dtype="float32")
    assert sr == 48000
    padded = np.concatenate([audio, np.zeros((64, 2), np.float32)])  # the decoder emits 1920*T - 64 samples
    encoded = vae_encoder.encode(encoder, padded)
    n = min(len(encoded), len(latents))
    assert n == len(latents)
    a, b = encoded[:n], latents[:n]
    cosine = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    assert cosine.mean() > 0.9
    assert np.sqrt(((a - b) ** 2).mean()) / np.sqrt((b**2).mean()) < 0.5
