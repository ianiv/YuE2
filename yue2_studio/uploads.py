"""Upload bookkeeping for ``data/uploads/`` (HTTP side; never imports mlx).

An upload is a media file ``<id>.<ext>`` plus a sidecar ``<id>.json`` holding
``{upload_id, filename, ext, seconds, size, created_at}``. Jobs of kind ``cover`` / ``hum`` reference
one via ``params.upload_id``; a finished song never needs it again, so an upload may be deleted
unless a queued/running job still points at it.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from yue2_studio import config
from yue2_studio.jobs import JobStore, iso

log = logging.getLogger("yue2_studio.uploads")


def valid_id(upload_id: str) -> bool:
    return bool(upload_id) and "/" not in upload_id and "\\" not in upload_id and "." not in upload_id


def sidecar(paths: config.Paths, upload_id: str) -> Path:
    return paths.uploads_dir / f"{upload_id}.json"


def _read_sidecar(path: Path) -> dict | None:
    try:
        info = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def files_for(paths: config.Paths, upload_id: str) -> list[Path]:
    """Every file ``<upload_id>.<suffix>`` in the uploads dir, by exact stem (no globbing: ``valid_id``
    admits ``*``/``?``/``[``, and a stray ``*.wav`` must never match the whole directory)."""
    if not valid_id(upload_id) or not paths.uploads_dir.is_dir():
        return []
    return sorted(p for p in paths.uploads_dir.iterdir() if p.stem == upload_id and p.suffix and p.is_file())


def _media(paths: config.Paths, upload_id: str, ext: str | None = None) -> Path | None:
    """The media file for ``upload_id``: ``<id>.<ext>`` when the sidecar names one, else any non-json one."""
    if ext:
        path = paths.uploads_dir / f"{upload_id}.{ext}"
        return path if path.is_file() else None
    matches = [p for p in files_for(paths, upload_id) if p.suffix != ".json"]
    return matches[0] if matches else None


def lookup(paths: config.Paths, upload_id: str) -> dict | None:
    """Sidecar info for a usable upload (both files present), else ``None``."""
    if not valid_id(upload_id):
        return None
    path = sidecar(paths, upload_id)
    if not path.is_file():
        return None
    info = _read_sidecar(path)
    if info is None or _media(paths, upload_id, info.get("ext", "")) is None:
        return None
    return info


def scan(paths: config.Paths, store: JobStore) -> list[dict]:
    """Every upload on disk, newest first, with its job counts.

    Entries whose sidecar or media file is missing (or unreadable) are ``broken`` with ``size: null``
    so they can still be listed and deleted.
    """
    counts = store.upload_counts()
    ids: set[str] = set()
    if paths.uploads_dir.is_dir():
        for path in paths.uploads_dir.iterdir():
            if path.is_file() and valid_id(path.stem) and path.suffix:
                ids.add(path.stem)
    items = []
    for upload_id in ids:
        side = sidecar(paths, upload_id)
        info = _read_sidecar(side) if side.is_file() else None
        media = _media(paths, upload_id, (info or {}).get("ext") or None)
        broken = info is None or media is None
        if info is None:  # no (readable) sidecar: describe what is on disk
            src = media or side
            try:
                stat = src.stat()
            except OSError:  # unlinked between iterdir and stat (e.g. the worker's auto_prune)
                continue
            info = {"filename": src.name, "ext": media.suffix.lstrip(".") if media else None, "seconds": None,
                    "size": stat.st_size, "created_at": iso(stat.st_mtime)}
        items.append({
            "upload_id": upload_id, "filename": info.get("filename") or f"{upload_id}.{info.get('ext', '')}",
            "ext": info.get("ext"), "seconds": info.get("seconds"),
            "size": None if broken else info.get("size"), "created_at": info.get("created_at"),
            "broken": broken,
            "jobs": counts.get(upload_id, {"total": 0, "active": 0}),
        })
    items.sort(key=lambda u: (u["created_at"] or "", u["upload_id"]), reverse=True)
    return items


def find(paths: config.Paths, store: JobStore, upload_id: str) -> dict | None:
    if not valid_id(upload_id):
        return None
    return next((u for u in scan(paths, store) if u["upload_id"] == upload_id), None)


def delete(paths: config.Paths, upload_id: str) -> None:
    """Remove the media file(s) and sidecar; missing pieces are ignored."""
    for path in files_for(paths, upload_id):
        path.unlink(missing_ok=True)


def _older_than(entry: dict, cutoff: datetime) -> bool:
    try:
        created = datetime.fromisoformat(entry["created_at"])
    except (KeyError, TypeError, ValueError):
        return True  # no usable timestamp: treat as old
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created < cutoff


def prune(paths: config.Paths, store: JobStore, *, unused: bool = True,
          older_than_days: int | None = None) -> dict:
    """Delete uploads no job needs any more.

    With ``unused`` only uploads no job references at all are deleted and the referenced ones are
    ``skipped``; without it every upload without a queued/running job goes. ``older_than_days``
    limits both to uploads created before that many days ago. Returns ``{"deleted", "skipped"}``.
    """
    cutoff = None if older_than_days is None else datetime.now(UTC) - timedelta(days=older_than_days)
    deleted = skipped = 0
    for entry in scan(paths, store):
        if cutoff is not None and not _older_than(entry, cutoff):
            continue
        busy = entry["jobs"]["total"] > 0 if unused else entry["jobs"]["active"] > 0
        if busy:
            skipped += 1
            continue
        delete(paths, entry["upload_id"])
        deleted += 1
    return {"deleted": deleted, "skipped": skipped}


def auto_prune(paths: config.Paths, store: JobStore) -> dict | None:
    """Apply the ``prune_uploads_days`` setting (``None`` = off). Never raises; runs at startup and
    after each job finishes."""
    try:
        days = store.get_settings().get("prune_uploads_days")
        if not days:
            return None
        result = prune(paths, store, unused=True, older_than_days=int(days))
    except Exception:  # noqa: BLE001 - housekeeping must never break the worker or startup
        log.exception("auto-prune of uploads failed")
        return None
    if result["deleted"]:
        log.info("auto-prune: deleted %d unused upload(s) older than %d day(s)", result["deleted"], days)
    return result
