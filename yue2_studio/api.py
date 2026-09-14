"""HTTP + SSE routes implementing ``docs/API.md``.

Handlers never touch the engine: they read/write the ``JobStore``, hand job ids to the ``Worker``
and stream events from the ``EventBus``. Shared objects live on ``app.state`` (``store``,
``worker``, ``bus``, ``paths``, ``fake``) and are installed by ``yue2_studio.main.create_app``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse, ServerSentEvent
from starlette.exceptions import HTTPException as StarletteHTTPException

from yue2_studio import __version__, audio, config, jobs
from yue2_studio.jobs import Job, JobStore, NotFound, ValidationFailure
from yue2_studio.worker import Worker

router = APIRouter(prefix="/api")

SSE_KEEPALIVE_SECONDS = 15
PLACEHOLDER_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>YuE2 Studio</title>
<meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: system-ui, sans-serif; margin: 2rem; color: #333">
<h1>YuE2 Studio</h1>
<p>The web UI has not been built yet (<code>yue2_studio/static/index.html</code> is missing).</p>
<p>The API is available: <a href="/api/status">/api/status</a>.</p>
</body></html>
"""


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status_code)


_STATUS_CODES = {400: "validation_error", 404: "not_found", 405: "method_not_allowed", 409: "conflict",
                 413: "too_large", 503: "engine_unavailable"}


def install_error_handlers(app) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(ValidationFailure)
    async def _validation(request: Request, exc: ValidationFailure):
        return error_response(400, "validation_error", str(exc))

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound):
        return error_response(404, "not_found", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _request_validation(request: Request, exc: RequestValidationError):
        return error_response(400, "validation_error", _fmt(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = _STATUS_CODES.get(exc.status_code, "internal_error" if exc.status_code >= 500 else "error")
        detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
        return error_response(exc.status_code, code, detail)

    @app.exception_handler(Exception)
    async def _any(request: Request, exc: Exception):
        return error_response(500, "internal_error", f"{type(exc).__name__}: {exc}")


def _fmt(exc: RequestValidationError) -> str:
    parts = []
    for item in exc.errors():
        loc = ".".join(str(x) for x in item.get("loc", ())) or "body"
        parts.append(f"{loc}: {item.get('msg')}")
    return "; ".join(parts) or "invalid request"


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _state(request: Request):
    return request.app.state


def _store(request: Request) -> JobStore:
    return request.app.state.store


def _worker(request: Request) -> Worker:
    return request.app.state.worker


def describe(request: Request, job: Job) -> dict:
    """Job JSON with the live last event overlaid for non-terminal jobs."""
    if job.progress is None:
        job.progress = _worker(request).bus.last(job.id)
    return job.to_api()


def _get_job(request: Request, job_id: str) -> Job:
    try:
        return _store(request).get(job_id)
    except NotFound:
        raise ApiError(404, "not_found", f"job {job_id!r} not found") from None


def _song_dir(request: Request, job_id: str) -> Path:
    return Path(_state(request).paths.songs_dir) / job_id


def _upload_sidecar(paths: config.Paths, upload_id: str) -> Path:
    return paths.uploads_dir / f"{upload_id}.json"


def upload_lookup(paths: config.Paths, upload_id: str) -> dict | None:
    if not upload_id or "/" in upload_id or "\\" in upload_id or "." in upload_id:
        return None
    sidecar = _upload_sidecar(paths, upload_id)
    if not sidecar.is_file():
        return None
    try:
        info = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    if not (paths.uploads_dir / f"{upload_id}.{info.get('ext', '')}").is_file():
        return None
    return info


async def _json_body(request: Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        raise ApiError(400, "validation_error", "request body must be valid JSON") from None


def _models_present(request: Request) -> bool:
    state = _state(request)
    if state.fake:
        return True
    models = config.models_available(state.paths)
    return bool(models["converted"] and models["vae"])


def _cover_status(request: Request) -> dict:
    state = _state(request)
    if state.fake:
        return {"available": True, "reasons": []}
    info = config.cover_available(state.paths)
    reasons = []
    if not info["ffmpeg"]:
        reasons.append("ffmpeg not found on PATH")
    if not info["sheetsage2"]:
        reasons.append("SheetSage2 not downloaded")
    if not info["mert"]:
        reasons.append("MERT-v2-FullSong not downloaded")
    return {"available": info["available"], "reasons": reasons}


# ---------------------------------------------------------------------------------------------
# status / settings
# ---------------------------------------------------------------------------------------------


@router.get("/status")
async def get_status(request: Request):
    state = _state(request)
    store, worker = state.store, state.worker
    counts = store.counts()
    models = config.models_available(state.paths)
    return {
        "engine": worker.engine_status(),
        "queue": {"queued": counts.get("queued", 0), "running": store.running_id()},
        "presets": config.presets_summary(),
        "models": {"converted_dir": str(state.paths.converted_dir), "vae_dir": str(state.paths.vae_dir),
                   "present": _models_present(request), "precisions": models["precisions"]},
        "cover": _cover_status(request),
        "ffmpeg": audio.ffmpeg_path() is not None,
        "fake": bool(state.fake),
        "version": __version__,
    }


@router.get("/settings")
async def get_settings(request: Request):
    return _store(request).get_settings()


@router.put("/settings")
async def put_settings(request: Request):
    body = await _json_body(request)
    if not isinstance(body, dict):
        raise ApiError(400, "validation_error", "settings body must be a JSON object")
    return _store(request).update_settings(body)


# ---------------------------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------------------------


@router.post("/jobs", status_code=201)
async def post_jobs(request: Request):
    body = await _json_body(request)
    if not isinstance(body, dict):
        raise ApiError(400, "validation_error", "body must be a JSON object")
    jobs.validate_submit(body)  # 400 for malformed bodies before any availability check
    if not _models_present(request):
        raise ApiError(503, "engine_unavailable", "models are missing; run scripts/setup.py")
    if body.get("kind") == "cover":
        cover = _cover_status(request)
        if not cover["available"]:
            raise ApiError(409, "conflict", "cover is unavailable: " + "; ".join(cover["reasons"]))
    state = _state(request)
    submission = jobs.submit(state.store, body, upload_lookup=lambda uid: upload_lookup(state.paths, uid))
    for job in submission.jobs:
        state.worker.submit(job.id)
    if submission.group is not None:
        # one query so queue positions are consistent across members; keep seed order
        rows, _ = state.store.list(group=submission.group["id"], limit=500)
        by_id = {j.id: j for j in rows}
        members = [describe(request, by_id[jid]) for jid in submission.group["job_ids"] if jid in by_id]
        return JSONResponse({"group": submission.group, "jobs": members}, status_code=201)
    return JSONResponse({"job": describe(request, state.store.get(submission.jobs[0].id))}, status_code=201)


def _csv(value: str | None, allowed: tuple[str, ...], name: str) -> list[str] | None:
    if not value:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in items if v not in allowed]
    if bad:
        raise ApiError(400, "validation_error", f"{name} must be one of {', '.join(allowed)}")
    return items or None


@router.get("/jobs")
async def list_jobs(request: Request, status: str | None = None, group: str | None = None,
                    kind: str | None = None, limit: int = 50, offset: int = 0):
    statuses = _csv(status, jobs.STATUSES, "status")
    kinds = _csv(kind, jobs.KINDS, "kind")
    if limit < 1 or limit > 500 or offset < 0:
        raise ApiError(400, "validation_error", "limit must be 1..500 and offset >= 0")
    items, total = _store(request).list(status=statuses, kind=kinds, group=group or None, limit=limit,
                                        offset=offset)
    return {"jobs": [describe(request, j) for j in items], "total": total}


@router.get("/jobs/{job_id}")
async def get_job(request: Request, job_id: str):
    return {"job": describe(request, _get_job(request, job_id))}


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(request: Request, job_id: str):
    job = _get_job(request, job_id)
    worker = _worker(request)
    if job.status == "running":
        raise ApiError(409, "conflict", "cannot delete a running job; cancel it first")
    if job.status == "queued":
        worker.cancel(job_id)
        job = _get_job(request, job_id)
        if job.status == "running":
            raise ApiError(409, "conflict", "job started before it could be deleted; cancel it first")
    _store(request).delete(job_id)
    await asyncio.to_thread(worker.delete_song_dir, job_id)
    return Response(status_code=204)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    job = _get_job(request, job_id)
    if job.terminal:
        raise ApiError(409, "conflict", f"job is already {job.status}")
    outcome = _worker(request).cancel(job_id)
    job = _get_job(request, job_id)
    if outcome is None:
        raise ApiError(409, "conflict", f"job is already {job.status}")
    code = 200 if outcome == "cancelled" else 202
    return JSONResponse({"job": describe(request, job)}, status_code=code)


def _sse_json(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _frame(event: str, payload: dict) -> ServerSentEvent:
    return ServerSentEvent(data=_sse_json(payload), event=event, sep="\n")


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str):
    store, worker = _store(request), _worker(request)
    _get_job(request, job_id)  # 404 before the stream starts
    bus = worker.bus

    async def stream():
        q = bus.subscribe(job_id)
        try:
            job = store.get(job_id)
            last = bus.last(job_id) or job.progress
            if last is not None:
                yield _frame("progress", last)
            if job.terminal:
                yield _frame("done", {"job": job.to_api()})
                return
            while True:
                kind, payload = await q.get()
                if kind == "progress":
                    yield _frame("progress", payload)
                else:
                    yield _frame("done", {"job": payload})
                    return
        except NotFound:
            return
        finally:
            bus.unsubscribe(job_id, q)

    return EventSourceResponse(
        stream(), ping=SSE_KEEPALIVE_SECONDS, sep="\n",
        ping_message_factory=lambda: ServerSentEvent(comment="keepalive", sep="\n"),
        headers={"Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------------------------
# song artifacts
# ---------------------------------------------------------------------------------------------


def _artifact(request: Request, job_id: str, *candidates: str) -> Path:
    _get_job(request, job_id)
    base = _song_dir(request, job_id)
    for rel in candidates:
        path = base / rel
        if path.is_file():
            return path
    raise ApiError(404, "not_found", f"{candidates[0]} is not available for job {job_id!r}")


@router.api_route("/songs/{job_id}/audio.flac", methods=["GET", "HEAD"])
async def song_audio_flac(request: Request, job_id: str):
    path = _artifact(request, job_id, "song/audio.flac")
    return FileResponse(path, media_type="audio/flac", filename=f"{job_id}.flac",
                        content_disposition_type="inline")


@router.api_route("/songs/{job_id}/audio.mp3", methods=["GET", "HEAD"])
async def song_audio_mp3(request: Request, job_id: str):
    flac = _artifact(request, job_id, "song/audio.flac")
    if audio.ffmpeg_path() is None:
        raise ApiError(503, "engine_unavailable", "ffmpeg is not installed; MP3 export unavailable")
    try:
        mp3 = await asyncio.to_thread(audio.transcode_mp3, flac)
    except audio.FfmpegMissing as error:
        raise ApiError(503, "engine_unavailable", str(error)) from None
    return FileResponse(mp3, media_type="audio/mpeg", filename=f"{job_id}.mp3",
                        content_disposition_type="inline")


@router.api_route("/songs/{job_id}/score.abc", methods=["GET", "HEAD"])
async def song_score(request: Request, job_id: str):
    path = _artifact(request, job_id, "song/score.abc", "plan/score.abc")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")


@router.api_route("/songs/{job_id}/plan.json", methods=["GET", "HEAD"])
async def song_plan(request: Request, job_id: str):
    path = _artifact(request, job_id, "plan/plan.json", "song/plan.json")
    return Response(path.read_bytes(), media_type="application/json")


@router.api_route("/songs/{job_id}/transcription/score.abc", methods=["GET", "HEAD"])
async def song_transcription_score(request: Request, job_id: str):
    path = _artifact(request, job_id, "transcription/score.abc")
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8")


@router.api_route("/songs/{job_id}/artifacts.zip", methods=["GET", "HEAD"])
async def song_artifacts_zip(request: Request, job_id: str):
    _get_job(request, job_id)
    song_dir = _song_dir(request, job_id)
    if not song_dir.is_dir() or not any(p.is_file() for p in song_dir.rglob("*")):
        raise ApiError(404, "not_found", f"no artifacts for job {job_id!r}")
    path = await asyncio.to_thread(audio.build_zip, song_dir)
    return FileResponse(path, media_type="application/zip", filename=f"{job_id}.zip",
                        content_disposition_type="attachment")


# ---------------------------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------------------------


@router.post("/upload", status_code=201)
async def post_upload(request: Request, file: UploadFile | None = None):
    if file is None:
        raise ApiError(400, "validation_error", "multipart field 'file' is required")
    try:
        ext = audio.validate_upload_name(file.filename)
    except audio.BadUploadType as error:
        raise ApiError(400, "validation_error", str(error)) from None
    # Browsers always send Content-Length for FormData bodies, so oversize uploads are rejected up
    # front; a chunked request without it is only capped while being copied to data/uploads/ (see
    # API.md), after the multipart parser has spooled it to a temp file.
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > audio.UPLOAD_MAX_BYTES + 4096:
        raise ApiError(413, "too_large", "upload exceeds 200 MB")
    paths: config.Paths = _state(request).paths
    upload_id = uuid.uuid4().hex
    destination = paths.uploads_dir / f"{upload_id}.{ext}"
    try:
        size = await asyncio.to_thread(audio.save_upload, file.file, destination)
    except audio.UploadTooLarge as error:
        raise ApiError(413, "too_large", str(error)) from None
    finally:
        await file.close()
    seconds = await asyncio.to_thread(audio.probe_duration, destination)
    info = {"upload_id": upload_id, "filename": Path(file.filename or f"upload.{ext}").name, "ext": ext,
            "seconds": seconds, "size": size, "created_at": jobs.now_iso()}
    _upload_sidecar(paths, upload_id).write_text(json.dumps(info))
    return JSONResponse({"upload_id": upload_id, "filename": info["filename"], "seconds": seconds,
                         "path_hint": f"data/uploads/{upload_id}.{ext}"}, status_code=201)


# ---------------------------------------------------------------------------------------------
# static / SPA fallback (registered last by create_app)
# ---------------------------------------------------------------------------------------------


def spa_response(static_dir: Path) -> HTMLResponse:
    index = Path(static_dir) / "index.html"
    if index.is_file():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse(PLACEHOLDER_HTML)


class _LenientStaticFiles:
    """``StaticFiles`` that answers 404 (instead of raising) while the directory does not exist yet."""

    def __new__(cls, directory: Path):
        from starlette.staticfiles import StaticFiles

        class Lenient(StaticFiles):
            async def check_config(self) -> None:
                if not Path(directory).is_dir():
                    return
                await super().check_config()

            async def get_response(self, path: str, scope):
                if not Path(directory).is_dir():
                    raise StarletteHTTPException(status_code=404)
                return await super().get_response(path, scope)

        return Lenient(directory=str(directory), check_dir=False)


def install_static(app, static_dir: Path) -> None:
    static_dir = Path(static_dir)
    app.mount("/static", _LenientStaticFiles(static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return spa_response(static_dir)

    @app.get("/{path:path}", include_in_schema=False)
    async def spa_fallback(path: str):
        if path == "api" or path.startswith("api/"):
            raise ApiError(404, "not_found", f"no such API route: /{path}")
        last = path.rsplit("/", 1)[-1]
        if "." in last:
            raise ApiError(404, "not_found", f"/{path} not found")
        return spa_response(static_dir)
