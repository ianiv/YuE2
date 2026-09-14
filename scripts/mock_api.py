"""Dev-only mock of docs/API.md so the static UI can be developed without models.

    uv run python scripts/mock_api.py --port 8790

In-memory jobs, scripted SSE progress (streaming ABC text), a silent FLAC (WAV fallback).
Not part of the product; the real server is yue2_studio.main.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import shutil
import subprocess
import uuid
import wave
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from yue2_studio.config import MAX_ODE_STEPS, MIN_ODE_STEPS, PRECISIONS, presets_summary

STATIC = Path(__file__).resolve().parents[1] / "yue2_studio" / "static"
ABC = (
    'X:1\nT:Mock song\nM:4/4\nL:1/8\nQ:1/4=96\nK:C\n"C"C2E2 G2E2|"F"F2A2 c2A2|"G"G2B2 d2B2|"C"c4 z4|\n'
    '"Am"A2c2 e2c2|"F"F2A2 c2A2|"G"G2B2 d2B2|"C"c8|]\n'
)
STAGES = [
    ("load", "Loading bf16 AR model", 3, None),
    ("plan", "Planning score", 8, "tokens"),
    ("semantic", "Generating song", 10, "tokens"),
    ("synthesize", "Synthesizing audio", 6, "steps"),
    ("decode", "Decoding audio", 3, "chunks"),
    ("save", "Saving artifacts", 1, None),
]
JOBS: dict[str, dict] = {}
GROUPS: dict[str, dict] = {}
UPLOADS: dict[str, dict] = {}
SUBS: dict[str, list[asyncio.Queue]] = {}
SETTINGS = {"default_preset": "quality", "memory_budget_gib": 24, "require_ac": False, "theme": "system"}
ENGINE = {"state": "cold", "precision": None, "memory_gib": None, "current_job_id": None}
DELAY = 0.2
SEQ = 0
COVER_OK = True


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(worker())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now(UTC).microsecond // 1000:03d}Z"


def err(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def make_audio() -> tuple[bytes, str]:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(48000), w.writeframes(b"\0" * 48000 * 4 * 2)
    if shutil.which("ffmpeg"):
        proc = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-f", "flac", "pipe:1"],
            input=buf.getvalue(),
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout, "audio/flac"
    return buf.getvalue(), "audio/wav"


AUDIO, AUDIO_TYPE = make_audio()


def event(job_id: str, **kw) -> dict:
    base = {
        k: None
        for k in (
            "type",
            "job_id",
            "ts",
            "stage",
            "label",
            "completed",
            "total",
            "unit",
            "status",
            "phase",
            "tokens",
            "tps",
            "seconds",
            "text",
            "partial",
            "message",
        )
    }
    base.update(kw, job_id=job_id, ts=now())
    return base


async def publish(job: dict, ev: dict | None, done: bool = False) -> None:
    if ev is not None:
        job["progress"] = ev
    for q in list(SUBS.get(job["id"], [])):
        await q.put(("done", {"job": job}) if done else ("progress", ev))


def new_job(kind: str, params: dict, preset: str, precision: str | None, ode_steps: int | None) -> dict:
    p = {"quality": ("bf16", 32), "fast": ("8bit", 8)}.get(preset, (precision, ode_steps))
    global SEQ
    SEQ += 1
    seed = params.get("seed") if params.get("seed") is not None else random.randint(0, 2**31 - 1)
    params = {**params, "seed": seed}
    return {
        "id": uuid.uuid4().hex,
        "seq": SEQ,
        "kind": kind,
        "status": "queued",
        "group_id": None,
        "parent_id": None,
        "preset": preset,
        "precision": p[0],
        "ode_steps": p[1],
        "seed": seed,
        "params": params,
        "title": params.get("title"),
        "created_at": now(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "timing": None,
        "truncated": None,
        "progress": None,
        "artifacts": {"audio": False, "score": False, "plan": False, "transcription": False},
        "position": None,
        "cancel": False,
    }


def public(job: dict) -> dict:
    out = {k: v for k, v in job.items() if k != "cancel"}
    queued = [
        j["id"] for j in sorted(JOBS.values(), key=lambda j: j["created_at"]) if j["status"] == "queued"
    ]
    out["position"] = queued.index(job["id"]) if job["status"] == "queued" else None
    return out


async def run_job(job: dict) -> None:
    ENGINE.update(state="busy", precision=job["precision"], memory_gib=11.2, current_job_id=job["id"])
    job.update(status="running", started_at=now())
    await publish(job, event(job["id"], type="status", status="running", message="job started"))
    stages = (
        STAGES
        if job["kind"] != "cover"
        else STAGES[:1] + [("transcribe", "Transcribing audio", 5, "windows")] + STAGES[1:]
    )
    t0, timing = asyncio.get_event_loop().time(), {}
    try:
        for key, label, steps, unit in stages:
            st = asyncio.get_event_loop().time()
            for i in range(steps + 1):
                if job["cancel"]:
                    raise InterruptedError
                if key == "load" and i == 0:
                    await publish(
                        job, event(job["id"], type="status", stage="load", message="loading models (bf16)…")
                    )
                secs = asyncio.get_event_loop().time() - st
                tps = (i * 40) / secs if unit == "tokens" and secs > 0 else None
                await publish(
                    job,
                    event(
                        job["id"],
                        type="stage",
                        stage=key,
                        label=label,
                        completed=i * (40 if unit == "tokens" else 1),
                        total=steps * (40 if unit == "tokens" else 1),
                        unit=unit,
                        status="running" if i < steps else "complete",
                        tps=tps,
                        seconds=secs,
                    ),
                )
                if key == "plan":
                    lines = ABC.splitlines(keepends=True)
                    await publish(
                        job,
                        event(
                            job["id"],
                            type="abc",
                            phase="abc",
                            tokens=i * 40,
                            partial=i < steps,
                            text="".join(lines[: max(1, round(len(lines) * i / steps))]),
                        ),
                    )
                await asyncio.sleep(DELAY if i < steps else DELAY / 2)
            timing[key] = round(asyncio.get_event_loop().time() - st, 2)
            if key == "plan":
                job["artifacts"].update(score=True, plan=True)
            if key == "transcribe":
                job["artifacts"]["transcription"] = True
        job["artifacts"]["audio"] = True
        job.update(
            status="done",
            timing={
                **timing,
                "e2e": round(asyncio.get_event_loop().time() - t0, 2),
                "abc_tps": 131.1,
                "semantic_tps": 128.8,
                "audio_seconds": 1.0,
            },
            truncated={"phase": "semantic", "reason": "generation limit reached"}
            if job["seed"] % 3 == 0
            else None,
        )
    except InterruptedError:
        job["status"] = "cancelled"
    except Exception as exc:  # pragma: no cover
        job.update(status="failed", error=str(exc))
    job["finished_at"] = now()
    await publish(job, event(job["id"], type="status", status=job["status"], message=f"job {job['status']}"))
    await publish(job, None, done=True)
    ENGINE.update(state="ready", current_job_id=None)


async def worker() -> None:
    while True:
        queued = [j for j in sorted(JOBS.values(), key=lambda j: j["created_at"]) if j["status"] == "queued"]
        if queued:
            await run_job(queued[0])
        await asyncio.sleep(0.2)


@app.get("/api/status")
def status() -> dict:
    running = ENGINE["current_job_id"]
    return {
        "engine": ENGINE,
        "queue": {"queued": sum(j["status"] == "queued" for j in JOBS.values()), "running": running},
        "presets": presets_summary(),
        "ffmpeg": AUDIO_TYPE == "audio/flac",
        "version": "0.1.0-mock",
        "models": {"converted_dir": "/abs/models/converted", "vae_dir": "/abs/models/vae", "present": True},
        "cover": {
            "available": COVER_OK,
            "reasons": [] if COVER_OK else ["MERT-v2-FullSong not downloaded", "ffmpeg missing"],
        },
    }


@app.post("/api/jobs")
async def post_job(req: Request):
    body = await req.json()
    kind, params, preset = (
        body.get("kind"),
        body.get("params") or {},
        body.get("preset") or SETTINGS["default_preset"],
    )
    precision, ode = body.get("precision"), body.get("ode_steps")
    if kind not in ("create", "regenerate", "cover", "variations"):
        return err(400, "validation_error", "kind must be create|regenerate|cover|variations")
    if preset == "custom" and (
        precision not in PRECISIONS or not isinstance(ode, int) or not MIN_ODE_STEPS <= ode <= MAX_ODE_STEPS
    ):
        return err(400, "validation_error", "custom preset needs precision and ode_steps 4..64")
    if kind == "variations":
        base, count = params.get("base") or {}, params.get("count")
        if (
            not isinstance(count, int)
            or not 2 <= count <= 16
            or not base.get("style")
            or not base.get("lyrics")
        ):
            return err(400, "validation_error", "variations need count 2..16 and base style/lyrics")
        seed0 = base.get("seed") if base.get("seed") is not None else random.randint(0, 2**31 - 1)
        gid = uuid.uuid4().hex
        label = params.get("label") or f"{base.get('title') or base['style'][:40]} ×{count}"
        jobs = []
        for i in range(count):
            seed = random.randint(0, 2**31 - 1) if params.get("random_seeds") else seed0 + i
            j = new_job(
                "create", {**base, "seed": seed, "cot": base.get("cot", "full")}, preset, precision, ode
            )
            j["group_id"], JOBS[j["id"]] = gid, j
            jobs.append(j)
            await asyncio.sleep(0.001)  # distinct created_at ordering
        GROUPS[gid] = {"id": gid, "label": label, "created_at": now(), "job_ids": [j["id"] for j in jobs]}
        return JSONResponse({"group": GROUPS[gid], "jobs": [public(j) for j in jobs]}, status_code=201)
    if kind == "create":
        if not params.get("style") or not params.get("lyrics"):
            return err(400, "validation_error", "style and lyrics are required")
        if params.get("abc") and params.get("cot") == "off":
            return err(400, "validation_error", "abc cannot be supplied with cot=off")
        params = {"cot": "full", "cfg_scale": None, "abc": None, "title": None, **params}
    elif kind == "regenerate":
        parent = JOBS.get(params.get("parent_id") or "")
        if parent is None:
            return err(404, "not_found", "unknown parent_id")
        if not params.get("abc"):
            return err(400, "validation_error", "abc is required")
        pp = parent["params"]
        params = {
            "parent_id": parent["id"],
            "abc": params["abc"],
            "style": params.get("style") or pp["style"],
            "lyrics": params.get("lyrics") or pp["lyrics"],
            "seed": params.get("seed") if params.get("seed") is not None else parent["seed"],
            "title": params.get("title") or pp.get("title"),
            "cot": "melody" if pp.get("cot") == "off" else pp.get("cot", "full"),
        }
        if body.get("preset") is None:
            preset, precision, ode = parent["preset"], parent["precision"], parent["ode_steps"]
    else:
        if not COVER_OK:
            return err(409, "conflict", "cover unavailable")
        up = UPLOADS.get(params.get("upload_id") or "")
        if up is None:
            return err(404, "not_found", "unknown upload_id")
        if not params.get("style") or not params.get("lyrics"):
            return err(400, "validation_error", "style and lyrics are required")
        params = {"task": "melody-full", **params, "title": params.get("title") or Path(up["filename"]).stem}
    job = new_job(kind, params, preset, precision, ode)
    job["parent_id"] = params.get("parent_id")
    JOBS[job["id"]] = job
    return JSONResponse({"job": public(job)}, status_code=201)


@app.get("/api/jobs")
def list_jobs(status: str = "", group: str = "", kind: str = "", limit: int = 50, offset: int = 0) -> dict:
    jobs = sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)
    if status:
        jobs = [j for j in jobs if j["status"] in status.split(",")]
    if kind:
        jobs = [j for j in jobs if j["kind"] in kind.split(",")]
    if group:
        jobs = [j for j in jobs if j["group_id"] == group]
    return {"jobs": [public(j) for j in jobs[offset : offset + min(limit, 500)]], "total": len(jobs)}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return {"job": public(JOBS[jid])} if jid in JOBS else err(404, "not_found", "unknown job")


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    if jid not in JOBS:
        return err(404, "not_found", "unknown job")
    if JOBS[jid]["status"] == "running":
        return err(409, "conflict", "job is running; cancel it first")
    JOBS.pop(jid)
    return Response(status_code=204)


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    job = JOBS.get(jid)
    if job is None:
        return err(404, "not_found", "unknown job")
    if job["status"] == "queued":
        job.update(status="cancelled", finished_at=now())
        return {"job": public(job)}
    if job["status"] == "running":
        job["cancel"] = True
        return JSONResponse({"job": public(job)}, status_code=202)
    return err(409, "conflict", f"job already {job['status']}")


@app.get("/api/jobs/{jid}/events")
async def events(jid: str):
    job = JOBS.get(jid)
    if job is None:
        return err(404, "not_found", "unknown job")

    async def gen():
        q: asyncio.Queue = asyncio.Queue()
        SUBS.setdefault(jid, []).append(q)
        try:
            if job["progress"]:
                yield f"event: progress\ndata: {json.dumps(job['progress'])}\n\n"
            if job["status"] in ("done", "failed", "cancelled"):
                yield f"event: done\ndata: {json.dumps({'job': public(job)})}\n\n"
                return
            while True:
                try:
                    name, data = await asyncio.wait_for(q.get(), 15)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                data = {"job": public(data["job"])} if name == "done" else data
                yield f"event: {name}\ndata: {json.dumps(data)}\n\n"
                if name == "done":
                    return
        finally:
            SUBS[jid].remove(q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/songs/{jid}/{path:path}")
def song_file(jid: str, path: str):
    job = JOBS.get(jid)
    if job is None or job["status"] != "done":
        return err(404, "not_found", "no such artifact")
    if path in ("audio.flac", "audio.mp3"):
        return Response(AUDIO, media_type=AUDIO_TYPE)
    if path == "score.abc" or (path == "transcription/score.abc" and job["kind"] == "cover"):
        return Response(job["params"].get("abc") or ABC, media_type="text/plain; charset=utf-8")
    if path == "plan.json":
        return {"fake": True, "seed": job["seed"], "abc_tokens": 2072}
    if path == "artifacts.zip":
        return Response(
            b"PK\x05\x06" + b"\0" * 18,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{jid}.zip"'},
        )
    return err(404, "not_found", "no such artifact")


@app.post("/api/upload")
async def upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in ("mp3", "wav", "flac", "m4a", "ogg"):
        return err(400, "validation_error", "unsupported audio type")
    data = await file.read()
    if len(data) > 200 * 2**20:
        return err(413, "too_large", "upload exceeds 200 MB")
    uid = uuid.uuid4().hex
    UPLOADS[uid] = {"filename": file.filename, "seconds": round(len(data) / 192000, 1)}
    return JSONResponse(
        {
            "upload_id": uid,
            "filename": file.filename,
            "seconds": UPLOADS[uid]["seconds"],
            "path_hint": f"data/uploads/{uid}.{ext}",
        },
        status_code=201,
    )


@app.get("/api/settings")
def get_settings() -> dict:
    return SETTINGS


@app.put("/api/settings")
async def put_settings(req: Request):
    body = await req.json()
    if "memory_budget_gib" in body and not 4 <= float(body["memory_budget_gib"]) <= 44:
        return err(400, "validation_error", "memory_budget_gib must be 4..44")
    SETTINGS.update({k: v for k, v in body.items() if k in SETTINGS})
    return SETTINGS


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/{path:path}")
def index(path: str):
    target = STATIC / path if path and "." in path.rsplit("/", 1)[-1] else STATIC / "index.html"
    return FileResponse(target) if target.is_file() else Response(status_code=404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--delay", type=float, default=DELAY)
    ap.add_argument("--no-cover", action="store_true")
    args = ap.parse_args()
    DELAY, COVER_OK = args.delay, not args.no_cover
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
