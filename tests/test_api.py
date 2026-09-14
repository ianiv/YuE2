"""HTTP/SSE contract tests against ``create_app`` with the FakeEngine and a tmp home."""

import asyncio
import io
import json

import pytest
from httpx import ASGITransport, AsyncClient

from yue2_studio.fake import FakeEngine
from yue2_studio.main import create_app

BASE = {"style": "dreamy indie pop, female vocal", "lyrics": "[verse]\nla la\n[chorus]\nda da", "seed": 42}


@pytest.fixture
async def app(tmp_path):
    application = create_app(FakeEngine(delay=0), home=tmp_path / "home", static_dir=tmp_path / "static")
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def submit(client, body, expect=201):
    r = await client.post("/api/jobs", json=body)
    assert r.status_code == expect, r.text
    return r.json()


async def stream_events(client, job_id, timeout=10.0):
    """Parse the SSE stream into (progress_events, done_payload)."""
    progress, done = [], None
    async with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.headers["cache-control"] == "no-cache"
        assert r.headers["x-accel-buffering"] == "no"
        event, data = None, None

        async def consume():
            nonlocal event, data, done
            async for line in r.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    data = line[6:]
                    assert "\n" not in data
                elif line == "" and event is not None:
                    payload = json.loads(data)
                    if event == "progress":
                        progress.append(payload)
                    elif event == "done":
                        done = payload["job"]
                        return
                    event, data = None, None

        await asyncio.wait_for(consume(), timeout)
    return progress, done


async def wait_terminal(client, job_id, timeout=10.0):
    _, done = await stream_events(client, job_id, timeout)
    return done


# -- status / settings ------------------------------------------------------------------------


async def test_status_shape(client):
    r = await client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == {"state": "cold", "precision": None, "memory_gib": None, "current_job_id": None}
    assert body["queue"] == {"queued": 0, "running": None}
    assert [p["name"] for p in body["presets"]] == ["quality", "fast", "custom"]
    assert body["presets"][1]["precision"] == "8bit" and body["presets"][2]["ode_steps"] is None
    assert body["models"]["present"] is True and body["models"]["converted_dir"].endswith("models/converted")
    assert body["cover"] == {"available": True, "reasons": []}
    assert isinstance(body["ffmpeg"], bool) and body["version"]


async def test_settings_get_and_partial_put(client):
    r = await client.get("/api/settings")
    assert r.json() == {"default_preset": "quality", "memory_budget_gib": 24, "require_ac": False,
                        "theme": "system"}
    r = await client.put("/api/settings", json={"theme": "dark"})
    assert r.status_code == 200
    assert r.json() == {"default_preset": "quality", "memory_budget_gib": 24, "require_ac": False,
                        "theme": "dark"}
    r = await client.put("/api/settings", json={"memory_budget_gib": 100})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    r = await client.put("/api/settings", content=b"nope", headers={"content-type": "application/json"})
    assert r.status_code == 400


# -- jobs ---------------------------------------------------------------------------------------


async def test_submit_create_then_stream_to_done_and_fetch_artifacts(client, app):
    body = await submit(client, {"kind": "create", "preset": "fast", "params": {**BASE, "title": "Test"}})
    job = body["job"]
    assert job["status"] == "queued" and job["position"] == 0 and job["title"] == "Test"
    assert (job["preset"], job["precision"], job["ode_steps"], job["seed"]) == ("fast", "8bit", 8, 42)
    progress, done = await stream_events(client, job["id"])
    assert done["status"] == "done" and done["artifacts"]["audio"] is True
    assert done["timing"]["audio_seconds"] == 1.0 and done["progress"]["status"] == "done"
    types = {e["type"] for e in progress}
    assert {"status", "stage", "token", "abc", "log"} <= types
    assert all(e["job_id"] == job["id"] for e in progress)
    # GET job reflects terminal state and persisted progress
    r = await client.get(f"/api/jobs/{job['id']}")
    assert r.status_code == 200 and r.json()["job"]["status"] == "done"
    assert r.json()["job"]["progress"]["type"] == "status"
    # status now shows a warm engine
    status = (await client.get("/api/status")).json()
    assert status["engine"]["state"] == "ready" and status["engine"]["precision"] == "8bit"
    # already-terminal stream: cached event then done immediately
    progress2, done2 = await stream_events(client, job["id"])
    assert len(progress2) == 1 and progress2[0]["status"] == "done" and done2["id"] == job["id"]
    raw = (await client.get(f"/api/jobs/{job['id']}/events")).content  # terminal: completes on its own
    assert b"\r" not in raw and raw.startswith(b"event: progress\ndata: {") and raw.endswith(b"}\n\n")
    assert b"\n\nevent: done\ndata: {\"job\":" in raw

    # artifacts
    r = await client.get(f"/api/songs/{job['id']}/audio.flac")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/flac"
    assert r.headers.get("accept-ranges") == "bytes"
    full = r.content
    assert full[:4] == b"fLaC"
    r = await client.get(f"/api/songs/{job['id']}/audio.flac", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100 and r.content == full[:100]
    assert r.headers["content-range"] == f"bytes 0-99/{len(full)}"
    r = await client.get(f"/api/songs/{job['id']}/score.abc")
    assert r.status_code == 200 and r.headers["content-type"] == "text/plain; charset=utf-8"
    assert r.text.startswith("X:1")
    r = await client.get(f"/api/songs/{job['id']}/plan.json")
    assert r.status_code == 200 and r.json()["fake"] is True
    r = await client.get(f"/api/songs/{job['id']}/transcription/score.abc")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.get(f"/api/songs/{job['id']}/artifacts.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == f'attachment; filename="{job["id"]}.zip"'
    import zipfile

    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert f"{job['id']}/song/audio.flac" in names and not any(n.endswith("artifacts.zip") for n in names)
    r = await client.get("/api/songs/doesnotexist/audio.flac")
    assert r.status_code == 404


async def test_audio_mp3_transcode(client):
    import shutil

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    job = (await submit(client, {"kind": "create", "params": BASE}))["job"]
    await wait_terminal(client, job["id"])
    r = await client.get(f"/api/songs/{job['id']}/audio.mp3")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 1000
    r2 = await client.get(f"/api/songs/{job['id']}/audio.mp3")  # cached
    assert r2.content == r.content


async def test_list_jobs_filters(client):
    a = (await submit(client, {"kind": "create", "params": BASE}))["job"]
    b = (await submit(client, {"kind": "create", "params": BASE}))["job"]
    await wait_terminal(client, b["id"])
    r = await client.get("/api/jobs")
    assert r.status_code == 200 and r.json()["total"] == 2
    assert [j["id"] for j in r.json()["jobs"]] == [b["id"], a["id"]]  # newest first
    r = await client.get("/api/jobs", params={"status": "done", "kind": "create"})
    assert r.json()["total"] == 2 and {j["status"] for j in r.json()["jobs"]} == {"done"}
    r = await client.get("/api/jobs", params={"status": "queued,running"})
    assert r.json() == {"jobs": [], "total": 0}
    r = await client.get("/api/jobs", params={"limit": 1, "offset": 1})
    assert r.json()["total"] == 2 and [j["id"] for j in r.json()["jobs"]] == [a["id"]]
    r = await client.get("/api/jobs", params={"status": "bogus"})
    assert r.status_code == 400
    r = await client.get("/api/jobs", params={"limit": 0})
    assert r.status_code == 400
    r = await client.get("/api/jobs/nope")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "not_found", "message": "job 'nope' not found"}}


async def test_validation_errors_shape(client):
    r = await client.post("/api/jobs", json={"kind": "create", "params": {"style": "", "lyrics": "x"}})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "validation_error" and "style" in r.json()["error"]["message"]
    r = await client.post("/api/jobs", json={"kind": "create",
                                             "params": {**BASE, "cot": "off", "abc": "X:1"}})
    assert r.status_code == 400
    r = await client.post("/api/jobs", json={"kind": "regenerate",
                                             "params": {"parent_id": "zzz", "abc": "X:1"}})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.post("/api/jobs", json={"kind": "cover", "params": {"upload_id": "zzz", "style": "s",
                                                                          "lyrics": "l"}})
    assert r.status_code == 404
    r = await client.post("/api/jobs", content=b"{bad json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"


async def test_variations_returns_group_and_jobs(client):
    body = await submit(client, {"kind": "variations", "preset": "fast",
                                 "params": {"count": 3, "base": {**BASE, "seed": 10}}})
    group, members = body["group"], body["jobs"]
    assert set(group) == {"id", "label", "created_at", "job_ids"} and len(group["id"]) == 32
    assert [j["seed"] for j in members] == [10, 11, 12]
    assert group["job_ids"] == [j["id"] for j in members]
    assert all(j["group_id"] == group["id"] and j["kind"] == "create" for j in members)
    positions = [j["position"] for j in members if j["status"] == "queued"]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    for j in members:
        await wait_terminal(client, j["id"])
    r = await client.get("/api/jobs", params={"group": group["id"]})
    assert r.json()["total"] == 3 and {j["status"] for j in r.json()["jobs"]} == {"done"}


async def test_cancel_queued_running_and_terminal(client, app):
    engine = app.state.engine
    engine.delay = 0.03
    first = (await submit(client, {"kind": "create", "params": BASE}))["job"]
    second = (await submit(client, {"kind": "create", "params": BASE}))["job"]
    assert second["status"] == "queued" and second["position"] is not None
    # queued -> 200 cancelled immediately
    r = await client.post(f"/api/jobs/{second['id']}/cancel")
    assert r.status_code == 200 and r.json()["job"]["status"] == "cancelled"
    _, done = await stream_events(client, second["id"])
    assert done["status"] == "cancelled"
    # running -> 202, then the stream delivers cancelled + done
    for _ in range(200):
        if (await client.get(f"/api/jobs/{first['id']}")).json()["job"]["status"] == "running":
            break
        await asyncio.sleep(0.01)
    r = await client.post(f"/api/jobs/{first['id']}/cancel")
    assert r.status_code == 202 and r.json()["job"]["status"] == "running"
    progress, done = await stream_events(client, first["id"])
    assert done["status"] == "cancelled" and progress[-1]["status"] == "cancelled"
    # terminal -> 409
    r = await client.post(f"/api/jobs/{first['id']}/cancel")
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    r = await client.post("/api/jobs/nope/cancel")
    assert r.status_code == 404
    engine.delay = 0


async def test_delete_job(client, app):
    job = (await submit(client, {"kind": "create", "params": BASE}))["job"]
    await wait_terminal(client, job["id"])
    song_dir = app.state.paths.songs_dir / job["id"]
    assert song_dir.is_dir()
    r = await client.delete(f"/api/jobs/{job['id']}")
    assert r.status_code == 204 and not song_dir.exists()
    r = await client.delete(f"/api/jobs/{job['id']}")
    assert r.status_code == 404
    # deleting a queued job cancels it first; deleting the last group member drops the group
    app.state.engine.delay = 0.05
    blocker = (await submit(client, {"kind": "create", "params": BASE}))["job"]
    group = await submit(client, {"kind": "variations", "params": {"count": 2, "base": BASE}})
    for member in group["jobs"]:
        r = await client.delete(f"/api/jobs/{member['id']}")
        assert r.status_code == 204
    assert (await client.get("/api/jobs", params={"group": group["group"]["id"]})).json()["total"] == 0
    for _ in range(200):
        if (await client.get(f"/api/jobs/{blocker['id']}")).json()["job"]["status"] == "running":
            break
        await asyncio.sleep(0.01)
    r = await client.delete(f"/api/jobs/{blocker['id']}")
    assert r.status_code == 409
    await client.post(f"/api/jobs/{blocker['id']}/cancel")
    await wait_terminal(client, blocker["id"])
    app.state.engine.delay = 0


async def test_failed_job_reports_error(client, app):
    app.state.engine.fail = True
    try:
        job = (await submit(client, {"kind": "create", "params": BASE}))["job"]
        done = await wait_terminal(client, job["id"])
        assert done["status"] == "failed" and "fake synthesis failure" in done["error"]
        assert done["artifacts"]["plan"] is True and done["artifacts"]["audio"] is False
        r = await client.get(f"/api/songs/{job['id']}/score.abc")  # falls back to plan/score.abc
        assert r.status_code == 200
        r = await client.get(f"/api/songs/{job['id']}/audio.flac")
        assert r.status_code == 404
    finally:
        app.state.engine.fail = False


async def test_regenerate_flow(client):
    parent = (await submit(client, {"kind": "create", "preset": "fast", "params": BASE}))["job"]
    await wait_terminal(client, parent["id"])
    score = (await client.get(f"/api/songs/{parent['id']}/score.abc")).text
    body = await submit(client, {"kind": "regenerate",
                                 "params": {"parent_id": parent["id"], "abc": score + "G|"}})
    job = body["job"]
    assert job["kind"] == "regenerate" and job["parent_id"] == parent["id"] and job["preset"] == "fast"
    assert job["params"]["abc"] == score + "G|" and job["seed"] == parent["seed"]
    done = await wait_terminal(client, job["id"])
    assert done["status"] == "done"
    assert (await client.get(f"/api/songs/{job['id']}/score.abc")).text == score + "G|"


# -- upload + cover ---------------------------------------------------------------------------


async def test_upload_validation_and_cover(client, app):
    r = await client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    r = await client.post("/api/upload", files={"other": ("a.mp3", b"x", "audio/mpeg")})
    assert r.status_code == 400
    r = await client.post("/api/upload",
                          files={"file": ("demo song.MP3", b"\xff\xfb" + b"\0" * 100, "audio/mpeg")})
    assert r.status_code == 201, r.text
    up = r.json()
    assert set(up) == {"upload_id", "filename", "seconds", "path_hint"} and len(up["upload_id"]) == 32
    assert up["filename"] == "demo song.MP3" and up["path_hint"] == f"data/uploads/{up['upload_id']}.mp3"
    assert (app.state.paths.uploads_dir / f"{up['upload_id']}.mp3").is_file()

    body = await submit(client, {"kind": "cover", "params": {"upload_id": up["upload_id"], "style": "jazz",
                                                             "lyrics": "la"}})
    job = body["job"]
    assert job["kind"] == "cover" and job["title"] == "demo song" and job["params"]["task"] == "melody-full"
    progress, done = await stream_events(client, job["id"])
    assert done["status"] == "done" and done["artifacts"]["transcription"] is True
    assert any(e["type"] == "stage" and e["stage"] == "transcribe" for e in progress)
    r = await client.get(f"/api/songs/{job['id']}/transcription/score.abc")
    assert r.status_code == 200 and r.text.startswith("X:1")


async def test_upload_too_large(client, monkeypatch):
    from yue2_studio import audio

    monkeypatch.setattr(audio, "UPLOAD_MAX_BYTES", 1024)
    r = await client.post("/api/upload", files={"file": ("big.wav", b"\0" * 4096, "audio/wav")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"


# -- static / SPA -----------------------------------------------------------------------------


async def test_spa_fallback_and_static(client, app, tmp_path):
    r = await client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "YuE2 Studio" in r.text
    r = await client.get("/song/abc")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    r = await client.get("/missing.png")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.get("/static/app.js")
    assert r.status_code == 404
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>real ui</title>")
    (static / "app.js").write_text("console.log(1)")
    r = await client.get("/library")
    assert r.status_code == 200 and "real ui" in r.text
    r = await client.get("/static/app.js")
    assert r.status_code == 200 and "console.log" in r.text


async def test_models_missing_gives_503_with_real_paths(tmp_path):
    class Cold:  # minimal engine stand-in that is not a FakeEngine
        state, precision, pipeline = "cold", None, None

        def ensure(self, options, on_event=None):
            raise RuntimeError("never called")

        create_song = cover_song = ensure

        def memory_footprint(self):
            return {}

        def unload(self):
            pass

    application = create_app(Cold(), home=tmp_path / "h", static_dir=tmp_path / "s")
    async with application.router.lifespan_context(application):
        async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as c:
            status = (await c.get("/api/status")).json()
            assert status["models"]["present"] is False and status["cover"]["available"] is False
            assert "MERT-v2-FullSong not downloaded" in status["cover"]["reasons"]
            r = await c.post("/api/jobs", json={"kind": "create", "params": BASE})
            assert r.status_code == 503 and r.json()["error"]["code"] == "engine_unavailable"


def test_fake_app_never_imports_mlx(tmp_path):
    import subprocess
    import sys

    code = (
        "import json, sys, tempfile; from yue2_studio.main import create_app; "
        "create_app(fake=True, home=tempfile.mkdtemp()); "
        "print(json.dumps(sorted(n for n in sys.modules "
        "if n == 'mlx' or n.startswith('mlx.') or n == 'lyra')))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert json.loads(out) == []
