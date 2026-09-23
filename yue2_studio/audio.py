"""ffmpeg/ffprobe helpers, upload validation and the artifacts zip builder.

Everything here is blocking; the API calls it through ``asyncio.to_thread``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path

from yue2_studio import config

UPLOAD_EXTENSIONS = ("mp3", "wav", "flac", "m4a", "ogg", "webm", "mp4")  # webm/mp4: MediaRecorder blobs
UPLOAD_MAX_BYTES = 200 * 1024 * 1024
ZIP_EXCLUDE = {"artifacts.zip"}

_transcode_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class UploadTooLarge(ValueError):
    pass


class BadUploadType(ValueError):
    pass


class FfmpegMissing(RuntimeError):
    pass


def ffmpeg_path() -> str | None:
    return config.FFMPEG or shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return config.FFPROBE or shutil.which("ffprobe")


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _transcode_locks.setdefault(key, threading.Lock())


def transcode_mp3(flac_path: Path, mp3_path: Path | None = None, *, bitrate: str = "192k") -> Path:
    """Encode ``flac_path`` to MP3 next to it (cached: returns immediately if already present)."""
    flac_path = Path(flac_path)
    mp3_path = Path(mp3_path) if mp3_path is not None else flac_path.with_suffix(".mp3")
    if mp3_path.is_file() and mp3_path.stat().st_mtime >= flac_path.stat().st_mtime:
        return mp3_path
    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise FfmpegMissing("ffmpeg is not installed")
    with _lock_for(str(mp3_path)):
        if mp3_path.is_file() and mp3_path.stat().st_mtime >= flac_path.stat().st_mtime:
            return mp3_path
        mp3_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = mp3_path.with_name(f".{mp3_path.name}.tmp.mp3")
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(flac_path), "-vn",
               "-codec:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", str(tmp)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            os.replace(tmp, mp3_path)
        except subprocess.CalledProcessError as error:
            tmp.unlink(missing_ok=True)
            detail = error.stderr.decode(errors="replace").strip()[:500]
            raise RuntimeError(f"ffmpeg failed: {detail}") from None
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return mp3_path


def decode_pcm(path: Path, *, sample_rate: int, channels: int, cancelled=None, timeout: float = 600):
    """Decode any ffmpeg-readable file to float32 PCM: ``[S]`` (mono) or ``[S, channels]``.

    ``cancelled()`` is polled every 0.2 s; a cancellation kills ffmpeg and raises ``InterruptedError``.
    """
    import numpy as np

    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise FfmpegMissing("ffmpeg is not installed")
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", str(channels),
           "-ar", str(sample_rate), "-f", "f32le", "pipe:1"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks: list[bytes] = []
    try:
        import time

        deadline = time.monotonic() + timeout
        while True:
            if cancelled is not None and cancelled():
                process.kill()
                raise InterruptedError("Cancelled while decoding audio")
            if time.monotonic() > deadline:
                process.kill()
                raise RuntimeError("ffmpeg timed out while decoding audio")
            try:
                out, err = process.communicate(timeout=0.2)
            except subprocess.TimeoutExpired:
                continue
            chunks.append(out)
            break
    finally:
        if process.poll() is None:
            process.kill()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace').strip()[:500]}")
    data = np.frombuffer(b"".join(chunks), dtype=np.float32)
    return data if channels == 1 else data.reshape(-1, channels)


def extract_clip(source: Path, destination: Path, *, start_s: float = 0.0, end_s: float | None = None,
                 timeout: float = 600) -> float:
    """Cut ``[start_s, end_s)`` of ``source`` into a FLAC at ``destination``; returns the clip's seconds.

    Seeking before ``-i`` is sample-accurate because the audio is re-encoded. ``end_s`` ``None`` keeps
    everything after ``start_s``. A clip that starts at or after the end of the recording is an error.
    """
    import soundfile as sf

    ffmpeg = ffmpeg_path()
    if ffmpeg is None:
        raise FfmpegMissing("ffmpeg is not installed")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-v", "error", "-nostdin", "-ss", f"{start_s:.3f}", "-i", str(source), "-vn"]
    if end_s is not None:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    cmd += ["-c:a", "flac", "-f", "flac", str(destination)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    except subprocess.CalledProcessError as error:
        destination.unlink(missing_ok=True)
        raise decode_error(error, source) from None
    with sf.SoundFile(str(destination)) as clip:
        frames, rate = clip.frames, clip.samplerate
        if not 0 < frames < 2**62:  # an empty FLAC reports "unknown length" and fails to read
            try:
                frames = sum(len(block) for block in clip.blocks(1 << 16))
            except sf.LibsndfileError:
                frames = 0
    if frames == 0:
        destination.unlink(missing_ok=True)
        raise ValueError(f"The clip starts at {start_s:g}s, after the end of {Path(source).name}")
    return frames / rate


def decode_error(error: subprocess.CalledProcessError, path: Path) -> ValueError:
    """Turn ffmpeg's ``CalledProcessError`` (exit status only) into a readable ``ValueError``."""
    stderr = error.stderr or b""
    detail = (stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr).strip()[-500:]
    detail = detail or f"exit status {error.returncode}"
    return ValueError(f"ffmpeg could not decode {Path(path).name}: {detail}")


def probe_duration(path: Path) -> float | None:
    """Duration in seconds via ffprobe; ``None`` when ffprobe is missing or the file is unreadable."""
    ffprobe = ffprobe_path()
    if ffprobe is None:
        return None
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, timeout=60, text=True).stdout
        value = json.loads(out).get("format", {}).get("duration")
        return None if value is None else round(float(value), 3)
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def validate_upload_name(filename: str | None) -> str:
    """Return the lower-case extension (without dot) or raise ``BadUploadType``."""
    name = Path(filename or "").name
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in UPLOAD_EXTENSIONS:
        accepted = ", ".join(UPLOAD_EXTENSIONS)
        raise BadUploadType(f"unsupported file type {ext or '(none)'!r}; accepted: {accepted}")
    return ext


def save_upload(stream, destination: Path, *, max_bytes: int | None = None, chunk: int = 1 << 20) -> int:
    """Copy a binary stream to ``destination`` enforcing ``max_bytes``; returns the size."""
    max_bytes = UPLOAD_MAX_BYTES if max_bytes is None else max_bytes
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.part")
    size = 0
    try:
        with open(tmp, "wb") as out:
            while True:
                data = stream.read(chunk)
                if not data:
                    break
                size += len(data)
                if size > max_bytes:
                    raise UploadTooLarge(f"upload exceeds {max_bytes // (1024 * 1024)} MB")
                out.write(data)
        os.replace(tmp, destination)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return size


def build_zip(song_dir: Path, *, exclude: set[str] | None = None) -> Path:
    """Zip the whole song dir into ``<song_dir>/artifacts.zip`` (built via a temp file, then renamed).

    Rebuilt when any member is newer than the existing zip. Temporary/partial files and the zip
    itself are never included.
    """
    song_dir = Path(song_dir)
    exclude = set(ZIP_EXCLUDE) | (exclude or set())
    target = song_dir / "artifacts.zip"
    members = [p for p in sorted(song_dir.rglob("*"))
               if p.is_file() and p.name not in exclude and not p.name.startswith(".")]
    if target.is_file():
        newest = max((p.stat().st_mtime for p in members), default=0.0)
        if target.stat().st_mtime >= newest:
            return target
    with _lock_for(str(target)):
        fd, tmp_name = tempfile.mkstemp(prefix=".artifacts-", suffix=".zip.tmp", dir=song_dir)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in members:
                    if path.name.startswith(".") or path.name in exclude:
                        continue
                    zf.write(path, arcname=str(Path(song_dir.name) / path.relative_to(song_dir)))
            os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    return target
