"""Album export for projects: the tracklist of chosen takes and the zip that ships them.

Pure Python and blocking (the API calls ``build_album_zip`` through ``asyncio.to_thread``). The
``project`` dicts come from ``JobStore.get_project``; each track's ``chosen`` is a ``Job`` (or None).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

from yue2_studio import audio
from yue2_studio.jobs import Job, now_iso

FORMATS = ("flac", "mp3")
TEMP_PREFIX = ".album-"
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f\x7f]+')


def sweep_temp(dest_dir: Path) -> int:
    """Remove album zips left behind by interrupted downloads (their cleanup only runs after a full
    send); called at server startup. Returns the number removed."""
    removed = 0
    for path in Path(dest_dir).glob(f"{TEMP_PREFIX}*.zip"):
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def safe_name(text: str | None, fallback: str = "untitled", limit: int = 80) -> str:
    """A filesystem-safe single-line name: path separators and control characters become spaces,
    whitespace collapses, leading/trailing dots go; never empty."""
    text = _UNSAFE.sub(" ", str(text or ""))
    text = " ".join(text.split()).strip(". ")
    return text[:limit].rstrip(". ") or fallback


def tracklist(project: dict, fmt: str = "flac") -> dict:
    """The album manifest: one entry per track in order, ``missing`` when it has no chosen take or
    the take's audio is gone. ``file`` is the name inside the zip (``01 Name.flac``) or null."""
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    tracks = []
    for n, track in enumerate(project["tracks"], start=1):
        job: Job | None = track.get("chosen")
        missing = job is None or not job.artifacts.get("audio")
        tracks.append({
            "n": n,
            "track_id": track["id"],
            "name": track["name"],
            "job_id": None if job is None else job.id,
            "title": None if job is None else job.title,
            "file": None if missing else f"{n:02d} {safe_name(track['name'])}.{fmt}",
            "seconds": None if job is None else job.audio_seconds,
            "seed": None if job is None else job.seed,
            "preset": None if job is None else job.preset,
            "kind": None if job is None else job.kind,
            "missing": missing,
        })
    return {"project": {"id": project["id"], "name": project["name"],
                        "description": project.get("description", "")},
            "format": fmt, "generated_at": now_iso(), "tracks": tracks}


def _mmss(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _cell(value) -> str:
    """A Markdown table cell: pipes escaped, newlines flattened."""
    return " ".join(str("" if value is None else value).split()).replace("|", "\\|")


def tracklist_md(data: dict) -> str:
    project = data["project"]
    lines = [f"# {project['name']}", ""]
    if project.get("description"):
        lines += [project["description"], ""]
    lines += [f"Generated {data['generated_at']} · {data['format'].upper()}", "",
              "| # | Track | Length | File | Take | Seed |", "|---|---|---|---|---|---|"]
    for t in data["tracks"]:
        if t["missing"]:
            take = "*missing*" if t["job_id"] is None else f"{t['job_id']} *(audio missing)*"
            lines.append(f"| {t['n']} | {_cell(t['name'])} | – | – | {take} | – |")
        else:
            take = f"{_cell(t['title'])} ({t['job_id']})" if t["title"] else t["job_id"]
            lines.append(f"| {t['n']} | {_cell(t['name'])} | {_mmss(t['seconds'])} | {_cell(t['file'])} | "
                         f"{take} | {t['seed']} |")
    return "\n".join(lines) + "\n"


def build_album_zip(project: dict, songs_dir: Path, *, fmt: str = "flac", dest_dir: Path) -> Path:
    """Write ``<slug>/NN Name.<fmt>`` for every exportable track plus ``tracklist.json`` and
    ``tracklist.md`` into a fresh temp file under ``dest_dir`` and return its path (the caller
    deletes it). Audio is stored uncompressed; MP3s are transcoded beside each FLAC and cached.
    ``ValueError`` when no track can be exported."""
    data = tracklist(project, fmt)
    exportable = [t for t in data["tracks"] if not t["missing"]]
    if not exportable:
        raise ValueError("no track has a finished take with audio to export")
    songs_dir = Path(songs_dir)
    slug = safe_name(project["name"], fallback="album")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=TEMP_PREFIX, suffix=".zip", dir=dest_dir)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for t in exportable:
                flac = songs_dir / t["job_id"] / "song" / "audio.flac"
                source = audio.transcode_mp3(flac) if fmt == "mp3" else flac
                zf.write(source, arcname=f"{slug}/{t['file']}", compress_type=zipfile.ZIP_STORED)
            zf.writestr(f"{slug}/tracklist.json", json.dumps(data, indent=2, ensure_ascii=False))
            zf.writestr(f"{slug}/tracklist.md", tracklist_md(data))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return tmp
