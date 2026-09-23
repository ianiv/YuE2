"""Unit tests for ``yue2_studio.audio``: upload validation, the artifacts zip and the ffmpeg helpers.

ffmpeg/ffprobe are replaced by stub executables placed on ``PATH`` so the tests neither depend on
nor spend time in the real encoder.
"""

import io
import json
import os
import stat
import subprocess
import time
import zipfile
from pathlib import Path

import pytest

from yue2_studio import audio, config

STUB_FFMPEG = """#!/bin/sh
printf '%s\\n' "$*" >> "{log}"
{body}
"""


def _install(bin_dir, name, body, log):
    path = bin_dir / name
    path.write_text(STUB_FFMPEG.format(log=log, body=body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def stubs(tmp_path, monkeypatch):
    """Stub ``ffmpeg``/``ffprobe`` on PATH; returns the call log path (one line of args per call)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    # ffmpeg: write a few bytes to its last argument (the output file)
    _install(bin_dir, "ffmpeg", 'for last; do :; done; printf "ID3fake-mp3" > "$last"', log)
    _install(bin_dir, "ffprobe", 'echo \'{"format": {"duration": "12.3456"}}\'', log)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(config, "FFMPEG", None)
    monkeypatch.setattr(config, "FFPROBE", None)
    return log


@pytest.fixture
def no_ffmpeg(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.setattr(config, "FFMPEG", None)
    monkeypatch.setattr(config, "FFPROBE", None)


def calls(log):
    return log.read_text().splitlines() if log.exists() else []


# -- upload validation ---------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "ext"), [
    ("song.mp3", "mp3"), ("SONG.MP3", "mp3"), ("a.wav", "wav"), ("b.flac", "flac"), ("c.m4a", "m4a"),
    ("d.ogg", "ogg"), ("my song.final.Ogg", "ogg"), ("/tmp/../x.wav", "wav"), ("dir\\\\file.mp3", "mp3"),
])
def test_validate_upload_name_accepts(name, ext):
    assert audio.validate_upload_name(name) == ext


@pytest.mark.parametrize("name", ["notes.txt", "noext", "", None, "archive.zip", "x.mp3.exe", "x.aac"])
def test_validate_upload_name_rejects(name):
    with pytest.raises(audio.BadUploadType) as info:
        audio.validate_upload_name(name)
    assert "accepted: mp3, wav, flac, m4a, ogg" in str(info.value)


def test_save_upload_writes_atomically_and_reports_size(tmp_path):
    dest = tmp_path / "uploads" / "u1.wav"
    size = audio.save_upload(io.BytesIO(b"x" * 3000), dest, chunk=1000)
    assert size == 3000 and dest.read_bytes() == b"x" * 3000
    assert not (dest.parent / ".u1.wav.part").exists()
    assert audio.save_upload(io.BytesIO(b""), tmp_path / "empty.wav") == 0


def test_save_upload_enforces_limit_and_cleans_up(tmp_path):
    dest = tmp_path / "u2.wav"
    with pytest.raises(audio.UploadTooLarge) as info:
        audio.save_upload(io.BytesIO(b"x" * 5000), dest, max_bytes=4096, chunk=1000)
    assert "exceeds 0 MB" in str(info.value) and not dest.exists()
    assert not list(tmp_path.glob(".*.part"))
    assert audio.save_upload(io.BytesIO(b"x" * 4096), dest, max_bytes=4096) == 4096  # exactly at the cap


def test_upload_limit_is_200mb():
    assert audio.UPLOAD_MAX_BYTES == 200 * 1024 * 1024
    assert audio.UPLOAD_EXTENSIONS == ("mp3", "wav", "flac", "m4a", "ogg", "webm", "mp4")
    assert audio.validate_upload_name("hum-2026.webm") == "webm"  # MediaRecorder blobs (Chrome / Safari)
    assert audio.validate_upload_name("hum.MP4") == "mp4"


# -- ffmpeg discovery --------------------------------------------------------------------------


def test_ffmpeg_path_prefers_config_then_path(stubs, monkeypatch):
    found = audio.ffmpeg_path()
    assert found and found.endswith("/bin/ffmpeg")
    monkeypatch.setattr(config, "FFMPEG", "/custom/ffmpeg")
    assert audio.ffmpeg_path() == "/custom/ffmpeg"
    monkeypatch.setattr(config, "FFPROBE", "/custom/ffprobe")
    assert audio.ffprobe_path() == "/custom/ffprobe"


def test_ffmpeg_missing(no_ffmpeg, tmp_path):
    assert audio.ffmpeg_path() is None and audio.ffprobe_path() is None
    flac = tmp_path / "audio.flac"
    flac.write_bytes(b"fLaC")
    with pytest.raises(audio.FfmpegMissing):
        audio.transcode_mp3(flac)
    assert not (tmp_path / "audio.mp3").exists()
    assert audio.probe_duration(flac) is None
    with pytest.raises(audio.FfmpegMissing):
        audio.decode_pcm(flac, sample_rate=48000, channels=1)


def test_decode_pcm_streams_float32_from_ffmpeg(tmp_path, monkeypatch):
    """The stub emits 4 little-endian float32 samples (two stereo frames) whatever the input."""
    import numpy as np

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    samples = "".join(f"\\{b:03o}" for b in np.array([1.0, -2.0, 0.0, 1.0], np.float32).tobytes())
    _install(bin_dir, "ffmpeg", f"printf '{samples}'", log)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(config, "FFMPEG", None)
    source = tmp_path / "hum.webm"
    source.write_bytes(b"x")
    mono = audio.decode_pcm(source, sample_rate=48000, channels=1)
    assert mono.dtype == np.float32 and mono.tolist() == [1.0, -2.0, 0.0, 1.0]
    stereo = audio.decode_pcm(source, sample_rate=48000, channels=2)
    assert stereo.shape == (2, 2) and stereo[1].tolist() == [0.0, 1.0]
    args = calls(log)[-1]
    assert "-ac 2" in args and "-ar 48000" in args and "-f f32le" in args and args.endswith("pipe:1")
    with pytest.raises(InterruptedError):
        audio.decode_pcm(source, sample_rate=48000, channels=1, cancelled=lambda: True)


# -- transcode ---------------------------------------------------------------------------------


def test_transcode_mp3_invokes_ffmpeg_once_and_caches(stubs, tmp_path):
    flac = tmp_path / "song" / "audio.flac"
    flac.parent.mkdir()
    flac.write_bytes(b"fLaC")
    mp3 = audio.transcode_mp3(flac)
    assert mp3 == flac.with_suffix(".mp3") and mp3.read_bytes() == b"ID3fake-mp3"
    args = calls(stubs)
    assert len(args) == 1
    assert f"-i {flac}" in args[0] and "libmp3lame" in args[0] and "-b:a 192k" in args[0]
    assert args[0].endswith(".tmp.mp3")  # encoded to a temp name, then renamed
    assert not list(flac.parent.glob(".*.tmp.mp3"))
    # second call: cache hit, ffmpeg not run again
    assert audio.transcode_mp3(flac) == mp3 and len(calls(stubs)) == 1
    # explicit target + bitrate
    other = audio.transcode_mp3(flac, tmp_path / "out" / "x.mp3", bitrate="64k")
    assert other.is_file() and "-b:a 64k" in calls(stubs)[-1] and len(calls(stubs)) == 2


def test_transcode_mp3_re_encodes_when_flac_is_newer(stubs, tmp_path):
    flac = tmp_path / "audio.flac"
    flac.write_bytes(b"fLaC")
    mp3 = audio.transcode_mp3(flac)
    old = time.time() - 100
    os.utime(mp3, (old, old))
    audio.transcode_mp3(flac)
    assert len(calls(stubs)) == 2 and mp3.stat().st_mtime >= flac.stat().st_mtime
    audio.transcode_mp3(flac)
    assert len(calls(stubs)) == 2


def test_transcode_mp3_failure_raises_and_cleans_tmp(stubs, tmp_path):
    _install(tmp_path / "bin", "ffmpeg", 'echo "boom: bad input" >&2; exit 1', stubs)
    flac = tmp_path / "audio.flac"
    flac.write_bytes(b"fLaC")
    with pytest.raises(RuntimeError) as info:
        audio.transcode_mp3(flac)
    assert "ffmpeg failed: boom: bad input" in str(info.value)
    assert not (tmp_path / "audio.mp3").exists() and not list(tmp_path.glob(".*.tmp.mp3"))


def test_decode_error_is_readable():
    error = subprocess.CalledProcessError(183, ["ffmpeg", "-i", "x.mp3"], b"", b"x.mp3: Invalid data found\n")
    message = str(audio.decode_error(error, Path("/data/uploads/x.mp3")))
    assert message == "ffmpeg could not decode x.mp3: x.mp3: Invalid data found"
    assert str(audio.decode_error(subprocess.CalledProcessError(1, ["ffmpeg"]), Path("a.wav"))) \
        == "ffmpeg could not decode a.wav: exit status 1"


def test_probe_duration(stubs, tmp_path):
    f = tmp_path / "a.mp3"
    f.write_bytes(b"x")
    assert audio.probe_duration(f) == 12.346
    args = calls(stubs)[-1]
    assert "-show_entries format=duration" in args and args.endswith(str(f))
    _install(tmp_path / "bin", "ffprobe", "echo not-json", stubs)
    assert audio.probe_duration(f) is None
    _install(tmp_path / "bin", "ffprobe", 'echo \'{"format": {}}\'', stubs)
    assert audio.probe_duration(f) is None
    _install(tmp_path / "bin", "ffprobe", "exit 1", stubs)
    assert audio.probe_duration(f) is None


# -- artifacts zip -----------------------------------------------------------------------------


def _song_dir(tmp_path):
    d = tmp_path / "songs" / "abc123"
    (d / "song").mkdir(parents=True)
    (d / "plan").mkdir()
    (d / "song" / "audio.flac").write_bytes(b"fLaC" * 10)
    (d / "song" / "score.abc").write_text("X:1")
    (d / "plan" / "plan.json").write_text("{}")
    (d / "summary.json").write_text(json.dumps({"status": "complete"}))
    (d / ".audio.mp3.tmp.mp3").write_bytes(b"partial")
    (d / "song" / ".DS_Store").write_bytes(b"")
    return d


def test_build_zip_contents_and_exclusions(tmp_path):
    d = _song_dir(tmp_path)
    target = audio.build_zip(d)
    assert target == d / "artifacts.zip"
    with zipfile.ZipFile(target) as zf:
        assert sorted(zf.namelist()) == ["abc123/plan/plan.json", "abc123/song/audio.flac",
                                         "abc123/song/score.abc", "abc123/summary.json"]
        assert zf.read("abc123/song/audio.flac") == b"fLaC" * 10
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zf.infolist())
    assert not list(d.glob(".artifacts-*"))  # temp file renamed away
    # the zip never contains itself, even when rebuilt
    (d / "song" / "audio.mp3").write_bytes(b"ID3")
    future = time.time() + 5
    os.utime(d / "song" / "audio.mp3", (future, future))
    with zipfile.ZipFile(audio.build_zip(d)) as zf:
        names = zf.namelist()
    assert "abc123/song/audio.mp3" in names and "abc123/artifacts.zip" not in names
    with zipfile.ZipFile(audio.build_zip(d, exclude={"summary.json"})) as zf:
        assert "abc123/summary.json" not in zf.namelist()


def test_build_zip_is_cached_until_a_member_changes(tmp_path):
    d = _song_dir(tmp_path)
    first = audio.build_zip(d)
    stamp = first.stat().st_mtime_ns
    assert audio.build_zip(d).stat().st_mtime_ns == stamp  # no rebuild
    # a newer member forces a rebuild that picks it up
    time.sleep(0.02)
    (d / "song" / "extra.txt").write_text("new")
    future = time.time() + 5
    os.utime(d / "song" / "extra.txt", (future, future))
    rebuilt = audio.build_zip(d)
    assert rebuilt.stat().st_mtime_ns != stamp
    with zipfile.ZipFile(rebuilt) as zf:
        assert "abc123/song/extra.txt" in zf.namelist()


@pytest.mark.skipif(audio.ffmpeg_path() is None, reason="ffmpeg not installed")
def test_extract_clip_cuts_the_requested_range(tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 48000
    t = np.arange(4 * sr) / sr
    source = tmp_path / "src.wav"
    sf.write(source, np.stack([np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32) * 0.5, sr)
    seconds = audio.extract_clip(source, tmp_path / "out" / "clip.flac", start_s=1.0, end_s=2.5)
    assert seconds == pytest.approx(1.5, abs=0.01)
    info = sf.info(str(tmp_path / "out" / "clip.flac"))
    assert info.samplerate == sr and info.channels == 2
    assert audio.extract_clip(source, tmp_path / "tail.flac", start_s=3.0) == pytest.approx(1.0, abs=0.01)
    with pytest.raises(ValueError, match="after the end"):
        audio.extract_clip(source, tmp_path / "late.flac", start_s=10.0)
    assert not (tmp_path / "late.flac").exists()
