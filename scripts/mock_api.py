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
import zipfile
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
PROJECTS: dict[str, dict] = {}  # id -> {id, name, description, created_at, updated_at}
TRACKS: dict[str, dict] = {}  # id -> {id, project_id, name, position, chosen_job_id, created_at}
TAKES: dict[str, dict] = {}  # job_id -> {track_id, thumb, stars, note, added_at}
SUBS: dict[str, list[asyncio.Queue]] = {}
SETTINGS = {"default_preset": "quality", "memory_budget_gib": 24, "require_ac": False, "theme": "system",
            "prune_uploads_days": None, "assist_provider": "auto", "assist_model": "",
            "anthropic_api_key": ""}
ASSIST_FIELDS = {
    "create": {
        "title": "Kitchen Light",
        "style": "English, indie folk, acoustic guitar strumming, stomping kick drum and tambourine, "
                 "warm male lead vocal, group shout-along chorus, campfire feel, 118 BPM",
        "lyrics": "[Intro]\nHey! Ho!\n\n[Verse 1]\nLeft the kitchen light on for you\nCoffee's cold but the "
                  "kettle's new\nEvery creak of the floor's a song\nAbout the nights we got it wrong\n\n"
                  "[Chorus]\nCome home, come home, the porch is warm\nWe'll ride it out, whatever storm\n"
                  "Come home, come home, the light's still on\nI'll keep it burning till the dawn\n\n"
                  "[Verse 2]\nThere's a dent where your bike leaned in\nA ring of dust where the pictures "
                  "been\nI hum the tune you used to play\nA little flat but it's okay\n\n"
                  "[Chorus]\nCome home, come home, the porch is warm\nWe'll ride it out, whatever storm\n"
                  "Come home, come home, the light's still on\nI'll keep it burning till the dawn\n\n"
                  "[Outro: acoustic guitar]",
        "cot": "full",
    },
    "cover": {
        "title": "Neon Rain (cover)",
        "style": "Japanese city pop, smooth female vocal, slap bass, electric piano, brass stabs, glossy 80s "
                 "production, 108 BPM",
        "lyrics": "[Verse]\n街の灯り 雨に溶けて\n君の影を 追いかけて\n\n[Chorus]\nNeon rain, neon rain\n"
                  "Carry me back to you again",
    },
    "hum": {
        "title": "Hummed at Midnight",
        "style": "dreamy indie pop, breathy female vocal, soft synth pads, brushed drums, intimate, 92 BPM",
        "lyrics": "[Verse]\nHalf asleep I hum your name\nWindow fogged with morning rain\n\n[Chorus]\n"
                  "Stay a little, stay a while\nLet the quiet make us smile",
    },
}
ASSIST_NOTE = "if it comes out too polished, add `lo-fi, live room`; if the tempo drifts, put the BPM first"
ENGINE = {"state": "cold", "precision": None, "memory_gib": None, "current_job_id": None, "loras": []}


def _lora(name, **fields):
    base = {"name": name, "path": f"/abs/models/loras/{name}.safetensors", "format": "safetensors",
            "valid": True, "rank": None, "scale": 1.0, "dtype": "BF16", "parts": [], "targets": [],
            "ar_modules": 0, "nar_modules": 0, "replaced": [], "size_bytes": 0, "metadata": {}, "error": None}
    return {**base, **fields}


LORAS = [
    _lora("ar_lora_inst_v3abc.bf16", rank=64, parts=["ar"], ar_modules=196, size_bytes=139502088,
          targets=["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj", "self_attn.k_proj", "self_attn.o_proj",
                   "self_attn.q_proj", "self_attn.v_proj"],
          metadata={"intended_cot": "full", "rank": "64", "lora_scale": "1.0"}),
    _lora("nar_lora_joint_v4.bf16", rank=32, parts=["nar"], nar_modules=196, size_bytes=70301856,
          targets=["nar_mlp.down_proj", "nar_self_attn.q_proj"], replaced=["llm2vae", "vae2llm"],
          metadata={"lora_scale": "1.0"}),
    _lora("hum_adapter_v1", valid=False, scale=None, dtype=None,
          error="unsupported tensors hum_proj.0.bias, hum_proj.0.weight "
                "(hum-to-song conditioning projections need the hum carrier path, not supported)"),
]
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


def new_job(kind: str, params: dict, preset: str, precision: str | None, ode_steps: int | None,
            loras: list | None = None) -> dict:
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
        "loras": loras or [],
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
    out["take"] = _take_of(job["id"])
    return out


def _take_of(job_id: str) -> dict | None:
    take = TAKES.get(job_id)
    if take is None:
        return None
    track = TRACKS.get(take["track_id"]) or {}
    project = PROJECTS.get(track.get("project_id")) or {}
    return {"track_id": take["track_id"], "project_id": track.get("project_id"),
            "track_name": track.get("name"), "project_name": project.get("name"),
            "thumb": take["thumb"], "stars": take["stars"], "note": take["note"],
            "added_at": take["added_at"], "chosen": track.get("chosen_job_id") == job_id}


async def run_job(job: dict) -> None:
    ENGINE.update(state="busy", precision=job["precision"], memory_gib=11.2, current_job_id=job["id"],
                  loras=job["loras"])
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
        "loras": {"dir": "/abs/models/loras", "adapters": LORAS},
        "ffmpeg": AUDIO_TYPE == "audio/flac",
        "version": "0.1.0-mock",
        "models": {"converted_dir": "/abs/models/converted", "vae_dir": "/abs/models/vae", "present": True},
        "cover": {
            "available": COVER_OK,
            "reasons": [] if COVER_OK else ["MERT-v2-FullSong not downloaded", "ffmpeg missing"],
        },
        "assist": _assist_status(),
    }


def _assist_status() -> dict:
    """Mirrors ``assist.status``: auto = CLI (pretend installed) else API when a key is set."""
    mode, key = SETTINGS["assist_provider"], bool(SETTINGS["anthropic_api_key"])
    model = SETTINGS["assist_model"] or None
    base = {"provider": None, "cli": True, "api_key": key, "model": model, "reasons": []}
    if mode == "off":
        return {**base, "reasons": ["assist is turned off in Settings"]}
    if mode in ("auto", "cli"):
        return {**base, "provider": "cli"}
    if key:
        return {**base, "provider": "api", "model": model or "claude-sonnet-5"}
    return {**base, "model": None, "reasons": ["no API key: add one in Settings or set ANTHROPIC_API_KEY"]}


@app.get("/api/loras")
def get_loras() -> dict:
    return {"dir": "/abs/models/loras", "adapters": LORAS}


def _loras(body: dict):
    """Validated ``[{name, scale}]`` from the submit body, or an error response."""
    raw = body.get("loras")
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(x, dict) and x.get("name") for x in raw):
        return err(400, "validation_error", "loras must be a list of {name, scale}")
    out = []
    for item in raw:
        scale = item.get("scale", 1.0)
        if not isinstance(scale, int | float) or not 0 <= scale <= 4:
            return err(400, "validation_error", f"loras.{len(out)}.scale must be in [0, 4]")
        known = next((a for a in LORAS if a["name"] == item["name"]), None)
        if known is None or not known["valid"]:
            return err(400, "validation_error", f"unknown or unusable LoRA adapter {item['name']!r}")
        out.append({"name": item["name"], "scale": float(scale)})
    return out


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
    loras = _loras(body)
    if isinstance(loras, JSONResponse):
        return loras
    track_id = body.get("track_id") or None
    if track_id is not None and track_id not in TRACKS:
        return err(404, "not_found", "unknown track_id")
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
            j = new_job("create", {**base, "seed": seed, "cot": base.get("cot", "full")}, preset, precision,
                        ode, loras)
            j["group_id"], JOBS[j["id"]] = gid, j
            jobs.append(j)
            await asyncio.sleep(0.001)  # distinct created_at ordering
        GROUPS[gid] = {"id": gid, "label": label, "created_at": now(), "job_ids": [j["id"] for j in jobs]}
        if track_id:
            _attach(track_id, [j["id"] for j in jobs])
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
        if loras is None:
            loras = parent.get("loras") or []
    else:
        if not COVER_OK:
            return err(409, "conflict", "cover unavailable")
        up = UPLOADS.get(params.get("upload_id") or "")
        if up is None:
            return err(404, "not_found", "unknown upload_id")
        if not params.get("style") or not params.get("lyrics"):
            return err(400, "validation_error", "style and lyrics are required")
        params = {"task": "melody-full", **params, "title": params.get("title") or Path(up["filename"]).stem}
    job = new_job(kind, params, preset, precision, ode, loras)
    job["parent_id"] = params.get("parent_id")
    JOBS[job["id"]] = job
    if track_id:
        _attach(track_id, [job["id"]])
    return JSONResponse({"job": public(job)}, status_code=201)


@app.get("/api/jobs")
def list_jobs(status: str = "", group: str = "", kind: str = "", track: str = "", project: str = "",
              limit: int = 50, offset: int = 0) -> dict:
    jobs = sorted(JOBS.values(), key=lambda j: j["created_at"], reverse=True)
    if status:
        jobs = [j for j in jobs if j["status"] in status.split(",")]
    if kind:
        jobs = [j for j in jobs if j["kind"] in kind.split(",")]
    if group:
        jobs = [j for j in jobs if j["group_id"] == group]
    if track:
        jobs = [j for j in jobs if j["id"] in TAKES and TAKES[j["id"]]["track_id"] == track]
    if project:
        jobs = [j for j in jobs if j["id"] in TAKES
                and TRACKS.get(TAKES[j["id"]]["track_id"], {}).get("project_id") == project]
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
    _detach(jid)
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


# -- projects / tracks / takes ---------------------------------------------------------------------


def _project_tracks(pid: str) -> list[dict]:
    return sorted((t for t in TRACKS.values() if t["project_id"] == pid), key=lambda t: t["position"])


def _track_takes(tid: str) -> list[dict]:
    items = [(jid, t) for jid, t in TAKES.items() if t["track_id"] == tid and jid in JOBS]
    return [public(JOBS[jid]) for jid, _ in sorted(items, key=lambda item: item[1]["added_at"])]


def _track_json(tid: str) -> dict:
    t = TRACKS[tid]
    takes = _track_takes(tid)
    chosen = next((j for j in takes if j["id"] == t["chosen_job_id"]), None)
    return {**t, "project_name": PROJECTS[t["project_id"]]["name"], "takes": takes, "chosen": chosen}


def _project_json(pid: str) -> dict:
    return {**PROJECTS[pid], "tracks": [_track_json(t["id"]) for t in _project_tracks(pid)]}


def _touch(pid: str | None) -> None:
    if pid in PROJECTS:
        PROJECTS[pid]["updated_at"] = now()


def _repack(pid: str) -> None:
    for i, t in enumerate(_project_tracks(pid)):
        t["position"] = i


def _detach(jid: str) -> None:
    take = TAKES.pop(jid, None)
    if take is None:
        return
    for t in TRACKS.values():
        if t["chosen_job_id"] == jid:
            t["chosen_job_id"] = None
    _touch(TRACKS.get(take["track_id"], {}).get("project_id"))


def _attach(tid: str, job_ids: list[str], move: bool = False):
    for jid in job_ids:
        if jid not in JOBS:
            return err(404, "not_found", f"job {jid!r} not found")
        current = TAKES.get(jid)
        if current and current["track_id"] != tid and not move:
            where = TRACKS[current["track_id"]]["name"]
            return err(409, "conflict", f"job {jid!r} is already a take of {where}")
    for jid in job_ids:
        current = TAKES.get(jid)
        if current is None:
            TAKES[jid] = {"track_id": tid, "thumb": None, "stars": None, "note": "", "added_at": now()}
        elif current["track_id"] != tid:
            for t in TRACKS.values():
                if t["chosen_job_id"] == jid:
                    t["chosen_job_id"] = None
            current.update(track_id=tid, added_at=now())
    _touch(TRACKS[tid]["project_id"])
    return None


def _name(body: dict, required: bool = True):
    name = body.get("name")
    if name is None and not required:
        return None
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 200:
        return err(400, "validation_error", "name must be a non-empty string of at most 200 characters")
    return " ".join(name.split())


@app.get("/api/projects")
def list_projects() -> dict:
    items = []
    for p in sorted(PROJECTS.values(), key=lambda p: p["updated_at"], reverse=True):
        tracks = _project_tracks(p["id"])
        items.append({**p, "track_count": len(tracks),
                      "chosen_count": sum(t["chosen_job_id"] is not None for t in tracks)})
    return {"projects": items}


@app.post("/api/projects")
async def post_project(req: Request):
    body = await req.json()
    name = _name(body)
    if isinstance(name, JSONResponse):
        return name
    description = body.get("description") or ""
    if not isinstance(description, str) or len(description) > 2000:
        return err(400, "validation_error", "description must be a string of at most 2000 characters")
    pid = uuid.uuid4().hex
    PROJECTS[pid] = {"id": pid, "name": name, "description": description, "created_at": now(),
                     "updated_at": now()}
    return JSONResponse({"project": _project_json(pid)}, status_code=201)


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return {"project": _project_json(pid)} if pid in PROJECTS else err(404, "not_found", "unknown project")


@app.patch("/api/projects/{pid}")
async def patch_project(pid: str, req: Request):
    if pid not in PROJECTS:
        return err(404, "not_found", "unknown project")
    body = await req.json()
    name = _name(body, required=False)
    if isinstance(name, JSONResponse):
        return name
    if name is not None:
        PROJECTS[pid]["name"] = name
    if "description" in body:
        PROJECTS[pid]["description"] = body["description"] or ""
    _touch(pid)
    return {"project": _project_json(pid)}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    if pid not in PROJECTS:
        return err(404, "not_found", "unknown project")
    for t in _project_tracks(pid):
        for jid in [jid for jid, take in TAKES.items() if take["track_id"] == t["id"]]:
            del TAKES[jid]
        del TRACKS[t["id"]]
    del PROJECTS[pid]
    return Response(status_code=204)


@app.post("/api/projects/{pid}/tracks")
async def post_track(pid: str, req: Request):
    if pid not in PROJECTS:
        return err(404, "not_found", "unknown project")
    name = _name(await req.json())
    if isinstance(name, JSONResponse):
        return name
    tid = uuid.uuid4().hex
    TRACKS[tid] = {"id": tid, "project_id": pid, "name": name, "position": len(_project_tracks(pid)),
                   "chosen_job_id": None, "created_at": now()}
    _touch(pid)
    return JSONResponse({"track": _track_json(tid)}, status_code=201)


@app.put("/api/projects/{pid}/order")
async def put_order(pid: str, req: Request):
    if pid not in PROJECTS:
        return err(404, "not_found", "unknown project")
    ids = (await req.json()).get("track_ids")
    current = [t["id"] for t in _project_tracks(pid)]
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != set(current):
        return err(400, "validation_error", "track_ids must list every track of the project exactly once")
    for i, tid in enumerate(ids):
        TRACKS[tid]["position"] = i
    _touch(pid)
    return {"project": _project_json(pid)}


@app.get("/api/projects/{pid}/album.zip")
def album_zip(pid: str, format: str = "flac"):
    if pid not in PROJECTS:
        return err(404, "not_found", "unknown project")
    if format not in ("flac", "mp3"):
        return err(400, "validation_error", "format must be flac or mp3")
    if format == "mp3" and AUDIO_TYPE != "audio/flac":
        return err(503, "engine_unavailable", "ffmpeg is not installed; MP3 export unavailable")
    project = _project_json(pid)
    slug = "".join(c if c not in '\\/:*?"<>|' else " " for c in project["name"]).strip() or "album"
    tracks, exportable = [], 0
    for n, t in enumerate(project["tracks"], start=1):
        job = t["chosen"]
        missing = job is None or not job["artifacts"]["audio"]
        exportable += not missing
        tracks.append({"n": n, "track_id": t["id"], "name": t["name"], "job_id": job and job["id"],
                       "title": job and job["title"],
                       "file": None if missing else f"{n:02d} {t['name']}.{format}",
                       "seconds": job and (job["timing"] or {}).get("audio_seconds"),
                       "seed": job and job["seed"], "preset": job and job["preset"],
                       "kind": job and job["kind"], "missing": missing})
    if not exportable:
        return err(409, "conflict", "no track has a finished take with audio to export")
    data = {"project": {"id": pid, "name": project["name"], "description": project["description"]},
            "format": format, "generated_at": now(), "tracks": tracks}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for t in tracks:
            if not t["missing"]:
                zf.writestr(f"{slug}/{t['file']}", AUDIO)
        zf.writestr(f"{slug}/tracklist.json", json.dumps(data, indent=2))
        md = [f"# {project['name']}", ""]
        md += [f"{t['n']}. {t['name']} — {t['file'] or 'missing'}" for t in tracks]
        zf.writestr(f"{slug}/tracklist.md", "\n".join(md) + "\n")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{slug}.zip"'})


@app.get("/api/tracks/{tid}")
def get_track(tid: str):
    if tid not in TRACKS:
        return err(404, "not_found", "unknown track")
    p = PROJECTS[TRACKS[tid]["project_id"]]
    return {"track": _track_json(tid), "project": {"id": p["id"], "name": p["name"]}}


@app.patch("/api/tracks/{tid}")
async def patch_track(tid: str, req: Request):
    if tid not in TRACKS:
        return err(404, "not_found", "unknown track")
    body, t = await req.json(), TRACKS[tid]
    name = _name(body, required=False)
    if isinstance(name, JSONResponse):
        return name
    if name is not None:
        t["name"] = name
    if "chosen_job_id" in body:
        jid = body["chosen_job_id"]
        if jid is not None:
            take = TAKES.get(jid)
            if take is None or take["track_id"] != tid:
                return err(409, "conflict", "job is not a take of this track")
            if JOBS[jid]["status"] != "done":
                return err(409, "conflict", "only a finished take can be chosen")
        t["chosen_job_id"] = jid
    if "position" in body:
        pos = body["position"]
        if not isinstance(pos, int) or isinstance(pos, bool) or pos < 0:
            return err(400, "validation_error", "position must be an integer >= 0")
        others = [x for x in _project_tracks(t["project_id"]) if x["id"] != tid]
        others.insert(min(pos, len(others)), t)
        for i, x in enumerate(others):
            x["position"] = i
    _touch(t["project_id"])
    return {"track": _track_json(tid)}


@app.delete("/api/tracks/{tid}")
def delete_track(tid: str):
    if tid not in TRACKS:
        return err(404, "not_found", "unknown track")
    for jid in [jid for jid, take in TAKES.items() if take["track_id"] == tid]:
        del TAKES[jid]
    pid = TRACKS.pop(tid)["project_id"]
    _repack(pid)
    _touch(pid)
    return Response(status_code=204)


@app.post("/api/tracks/{tid}/takes")
async def post_takes(tid: str, req: Request):
    if tid not in TRACKS:
        return err(404, "not_found", "unknown track")
    body = await req.json()
    ids = body.get("job_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        return err(400, "validation_error", "job_ids must be a non-empty list of job ids")
    failure = _attach(tid, list(dict.fromkeys(ids)), move=bool(body.get("move")))
    return failure if failure is not None else {"track": _track_json(tid)}


@app.delete("/api/takes/{jid}")
def delete_take(jid: str):
    if jid not in TAKES:
        return err(404, "not_found", "job is not a take")
    _detach(jid)
    return Response(status_code=204)


@app.patch("/api/takes/{jid}")
async def patch_take(jid: str, req: Request):
    take = TAKES.get(jid)
    if take is None or jid not in JOBS:
        return err(404, "not_found", "job is not a take")
    body = await req.json()
    if "thumb" in body:
        if body["thumb"] not in (None, -1, 0, 1) or isinstance(body["thumb"], bool):
            return err(400, "validation_error", "thumb must be -1, 0, 1 or null")
        take["thumb"] = body["thumb"] or None
    if "stars" in body:
        stars = body["stars"]
        bad_stars = not isinstance(stars, int) or isinstance(stars, bool) or not 1 <= stars <= 5
        if stars is not None and bad_stars:
            return err(400, "validation_error", "stars must be 1..5 or null")
        take["stars"] = stars
    if "note" in body:
        note = body["note"] or ""
        if not isinstance(note, str) or len(note) > 4000:
            return err(400, "validation_error", "note must be a string of at most 4000 characters")
        take["note"] = note
    _touch(TRACKS[take["track_id"]]["project_id"])
    return {"job": public(JOBS[jid])}


@app.post("/api/upload")
async def upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in ("mp3", "wav", "flac", "m4a", "ogg"):
        return err(400, "validation_error", "unsupported audio type")
    data = await file.read()
    if len(data) > 200 * 2**20:
        return err(413, "too_large", "upload exceeds 200 MB")
    uid = uuid.uuid4().hex
    UPLOADS[uid] = {"upload_id": uid, "filename": file.filename, "ext": ext, "size": len(data),
                    "seconds": round(len(data) / 192000, 1), "created_at": now(), "broken": False}
    return JSONResponse(
        {
            "upload_id": uid,
            "filename": file.filename,
            "seconds": UPLOADS[uid]["seconds"],
            "path_hint": f"data/uploads/{uid}.{ext}",
        },
        status_code=201,
    )


def _upload_jobs(uid: str) -> dict:
    refs = [j for j in JOBS.values() if j["params"].get("upload_id") == uid]
    return {"total": len(refs), "active": sum(j["status"] in ("queued", "running") for j in refs)}


def _uploads() -> list[dict]:
    items = [{**u, "jobs": _upload_jobs(uid)} for uid, u in UPLOADS.items()]
    return sorted(items, key=lambda u: u["created_at"], reverse=True)


@app.get("/api/uploads")
def list_uploads(unused: bool = False) -> dict:
    items = _uploads()
    return {"uploads": [u for u in items if u["jobs"]["total"] == 0] if unused else items}


@app.delete("/api/uploads/{uid}")
def delete_upload(uid: str):
    if uid not in UPLOADS:
        return err(404, "not_found", "unknown upload")
    if _upload_jobs(uid)["active"]:
        return err(409, "in_use", "upload is used by a queued or running job")
    del UPLOADS[uid]
    return Response(status_code=204)


@app.post("/api/uploads/prune")
async def prune_uploads(req: Request) -> dict:
    body = await req.json()
    unused, deleted, skipped = body.get("unused", True), 0, 0
    for u in _uploads():
        if u["jobs"]["total"] > 0 if unused else u["jobs"]["active"] > 0:
            skipped += 1
        else:
            del UPLOADS[u["upload_id"]]
            deleted += 1
    return {"deleted": deleted, "skipped": skipped}


def _public_settings() -> dict:
    public = {k: v for k, v in SETTINGS.items() if k != "anthropic_api_key"}
    public["has_api_key"] = bool(SETTINGS["anthropic_api_key"])
    return public


@app.get("/api/settings")
def get_settings() -> dict:
    return _public_settings()


@app.put("/api/settings")
async def put_settings(req: Request):
    body = await req.json()
    if "memory_budget_gib" in body and not 4 <= float(body["memory_budget_gib"]) <= 44:
        return err(400, "validation_error", "memory_budget_gib must be 4..44")
    if body.get("assist_provider") not in (None, "auto", "cli", "api", "off"):
        return err(400, "validation_error", "assist_provider must be one of auto, cli, api, off")
    SETTINGS.update({k: v.strip() if isinstance(v, str) else v for k, v in body.items() if k in SETTINGS})
    return _public_settings()


@app.post("/api/assist")
async def post_assist(req: Request):
    body = await req.json()
    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        return err(400, "validation_error", "prompt: must be a non-empty string")
    page = body.get("page") or "create"
    if page not in ASSIST_FIELDS:
        return err(400, "validation_error", "page: must be one of create, cover, hum")
    status = _assist_status()
    if status["provider"] is None:
        return err(503, "assist_unavailable", "; ".join(status["reasons"]))
    await asyncio.sleep(0.6)
    fields = dict(ASSIST_FIELDS[page])
    context = body.get("context") or {}
    if isinstance(context, dict) and any(context.get(k) for k in ("title", "style", "lyrics")):
        # "refine": only the fields that change; keep the user's title so the edit is visible. On create an
        # explicit cfg_scale null is legitimate: it resets the CFG field to the engine default.
        fields = {"style": fields["style"], "lyrics": fields["lyrics"]}
        if page == "create" and context.get("cfg_scale") not in (None, ""):
            fields["cfg_scale"] = None
    return {"fields": fields, "notes": ASSIST_NOTE, "provider": status["provider"],
            "model": status["model"] or "claude-haiku-4-5-20251001", "seconds": 0.6}


@app.post("/api/assist/test")
async def post_assist_test():
    status = _assist_status()
    if status["provider"] is None:
        return err(503, "assist_unavailable", "; ".join(status["reasons"]))
    await asyncio.sleep(0.4)
    return {"fields": {"title": "ok"}, "notes": None, "provider": status["provider"],
            "model": status["model"] or "claude-haiku-4-5-20251001", "seconds": 0.4}


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
