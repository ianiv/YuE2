"""HTTP/SSE contract tests (docs/API.md) against ``create_app`` with the FakeEngine and a tmp home.

Fixtures (``app``, ``client``, ``api``, ``make_app``) live in ``conftest.py``.
"""

import asyncio
import io
import json
import shutil
import zipfile
from datetime import UTC, datetime, timedelta

import pytest
from conftest import BASE, Api

from yue2_studio import api as api_module
from yue2_studio import audio
from yue2_studio.fake import FakeEngine

JOB_KEYS = {"id", "kind", "status", "group_id", "parent_id", "preset", "precision", "ode_steps", "loras",
            "seed", "params", "title", "created_at", "started_at", "finished_at", "error", "timing",
            "truncated", "progress", "artifacts", "position", "seq"}
EVENT_KEYS = {"type", "job_id", "ts", "stage", "label", "completed", "total", "unit", "status", "phase",
              "tokens", "tps", "seconds", "text", "partial", "message"}


class ColdEngine:
    """Minimal non-fake engine stand-in: makes ``status.fake`` false so real model paths are checked."""

    state, precision, pipeline = "cold", None, None

    def ensure(self, options, on_event=None):
        raise RuntimeError("no models in tests")

    create_song = cover_song = hum_song = ensure

    def memory_footprint(self):
        return {}

    def unload(self):
        pass


def fake_models(home):
    """Create the files ``config.models_available`` looks for (no real weights)."""
    converted, vae = home / "models" / "converted", home / "models" / "vae"
    converted.mkdir(parents=True, exist_ok=True)
    vae.mkdir(parents=True, exist_ok=True)
    (converted / "conversion.json").write_text("{}")
    (converted / "ar-8bit.safetensors").write_bytes(b"")
    (vae / "config.json").write_text("{}")
    (vae / "model.safetensors").write_bytes(b"")


# -- status / settings ------------------------------------------------------------------------


async def test_status_shape(client, home):
    r = await client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"engine", "queue", "presets", "models", "cover", "hum", "loras", "ffmpeg", "fake",
                         "version"}
    assert body["hum"] == {"available": True, "reasons": [], "adapters": []}
    assert body["engine"] == {"state": "cold", "precision": None, "memory_gib": None, "current_job_id": None,
                              "loras": []}
    assert body["loras"] == {"dir": str(home / "models" / "loras"), "adapters": []}
    assert body["queue"] == {"queued": 0, "running": None}
    assert [p["name"] for p in body["presets"]] == ["quality", "fast", "custom"]
    assert body["presets"][0]["precision"] == "bf16" and body["presets"][0]["ode_steps"] == 32
    assert body["presets"][1]["precision"] == "8bit" and body["presets"][1]["ode_steps"] == 8
    assert body["presets"][2]["precision"] is None and body["presets"][2]["ode_steps"] is None
    assert all(set(p) == {"name", "label", "precision", "ode_steps", "description"} for p in body["presets"])
    assert body["models"]["present"] is True and body["models"]["converted_dir"].endswith("models/converted")
    assert body["models"]["vae_dir"].endswith("models/vae") and body["models"]["precisions"] == []
    assert body["cover"] == {"available": True, "reasons": []}
    assert isinstance(body["ffmpeg"], bool) and body["fake"] is True and body["version"]


async def test_status_reflects_queue_and_engine(client, api, app):
    app.state.engine.delay = 0.02
    first = await api.create()
    second = await api.create()
    await api.wait_status(first["id"], "running")
    body = (await client.get("/api/status")).json()
    assert body["queue"] == {"queued": 1, "running": first["id"]}
    assert body["engine"]["current_job_id"] == first["id"]
    app.state.engine.delay = 0
    await api.wait(second["id"])
    body = (await client.get("/api/status")).json()
    assert body["queue"] == {"queued": 0, "running": None}
    assert body["engine"]["state"] == "ready" and body["engine"]["memory_gib"] > 0


async def test_settings_get_and_partial_put(client):
    r = await client.get("/api/settings")
    assert r.json() == {"default_preset": "quality", "memory_budget_gib": 24, "require_ac": False,
                        "theme": "system", "prune_uploads_days": None}
    r = await client.put("/api/settings", json={"theme": "dark"})
    assert r.status_code == 200
    assert r.json() == {"default_preset": "quality", "memory_budget_gib": 24, "require_ac": False,
                        "theme": "dark", "prune_uploads_days": None}
    assert (await client.get("/api/settings")).json()["theme"] == "dark"  # persisted
    r = await client.put("/api/settings", json={"bogus": 1})  # unknown keys ignored
    assert r.status_code == 200 and "bogus" not in r.json()
    r = await client.put("/api/settings", content=b"nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    r = await client.put("/api/settings", json=[1])
    assert r.status_code == 400


@pytest.mark.parametrize(("patch", "ok"), [
    ({"memory_budget_gib": 6}, True), ({"memory_budget_gib": 5.99}, False),
    ({"memory_budget_gib": 4}, False),
    ({"memory_budget_gib": 44}, True), ({"memory_budget_gib": 44.01}, False),
    ({"memory_budget_gib": 100}, False), ({"memory_budget_gib": "lots"}, False),
    ({"default_preset": "fast"}, True), ({"default_preset": "ultra"}, False),
    ({"require_ac": True}, True), ({"require_ac": 3}, False),
    ({"theme": "light"}, True), ({"theme": "neon"}, False),
    ({"prune_uploads_days": 1}, True), ({"prune_uploads_days": 365}, True),
    ({"prune_uploads_days": None}, True),
    ({"prune_uploads_days": 0}, False), ({"prune_uploads_days": 366}, False),
    ({"prune_uploads_days": "week"}, False),
    ({"prune_uploads_days": True}, False),
])
async def test_settings_validation_bounds(client, patch, ok):
    before = (await client.get("/api/settings")).json()
    r = await client.put("/api/settings", json=patch)
    if ok:
        assert r.status_code == 200, r.text
        assert r.json() == {**before, **patch}
    else:
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
        assert next(iter(patch)) in r.json()["error"]["message"]
        assert (await client.get("/api/settings")).json() == before  # rejected patch changes nothing


async def test_default_preset_setting_applies_to_submits(client, api):
    await client.put("/api/settings", json={"default_preset": "fast"})
    job = await api.create()
    assert (job["preset"], job["precision"], job["ode_steps"]) == ("fast", "8bit", 8)
    job = await api.create(preset="quality")
    assert (job["preset"], job["precision"], job["ode_steps"]) == ("quality", "bf16", 32)
    job = await api.create(preset="custom", precision="4bit", ode_steps=12)
    assert (job["preset"], job["precision"], job["ode_steps"]) == ("custom", "4bit", 12)


# -- jobs: create + stream + artifacts ------------------------------------------------------------


async def test_submit_create_then_stream_to_done(client, api):
    body = await api.submit({"kind": "create", "preset": "fast", "params": {**BASE, "title": "Test"}})
    job = body["job"]
    assert set(job) == JOB_KEYS
    assert job["status"] == "queued" and job["position"] == 0 and job["title"] == "Test"
    assert (job["preset"], job["precision"], job["ode_steps"], job["seed"]) == ("fast", "8bit", 8, 42)
    assert job["params"] == {**BASE, "cot": "full", "cfg_scale": None, "abc": None, "title": "Test"}
    assert len(job["id"]) == 32 and job["created_at"].endswith("Z")
    progress, done = await api.events(job["id"])
    assert done["status"] == "done" and done["artifacts"]["audio"] is True
    assert done["timing"]["audio_seconds"] == 1.0 and done["progress"]["status"] == "done"
    assert done["started_at"] and done["finished_at"] and done["position"] is None
    assert all(set(e) == EVENT_KEYS for e in progress)
    types = [e["type"] for e in progress]
    assert {"status", "stage", "token", "abc", "log"} <= set(types)
    assert types[0] == "status" and progress[0]["status"] == "running"
    assert types[-1] == "status" and progress[-1]["status"] == "done"
    assert all(e["job_id"] == job["id"] for e in progress)
    load = [e for e in progress if e["type"] == "status" and e["stage"] == "load"]
    assert len(load) == 1 and "8bit" in load[0]["message"]  # cold engine announced once
    stages = [e["stage"] for e in progress if e["type"] == "stage"]
    assert [s for i, s in enumerate(stages) if s not in stages[:i]] == \
        ["load", "plan", "semantic", "synthesize", "decode", "save"]
    abc = [e for e in progress if e["type"] == "abc"]
    assert [e["partial"] for e in abc] == [True, True, True, False] and abc[-1]["text"].startswith("X:1")
    tokens = [e for e in progress if e["type"] == "token"]
    assert {e["phase"] for e in tokens} == {"abc", "semantic"} and all(e["tokens"] > 0 for e in tokens)
    # GET job reflects terminal state and persisted progress
    r = await client.get(f"/api/jobs/{job['id']}")
    assert r.status_code == 200 and r.json()["job"]["status"] == "done"
    assert r.json()["job"]["progress"]["type"] == "status"
    status = (await client.get("/api/status")).json()
    assert status["engine"]["state"] == "ready" and status["engine"]["precision"] == "8bit"


async def test_song_artifact_routes(client, api):
    job = await api.create()
    await api.wait(job["id"])
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
    assert r.status_code == 200 and r.headers["content-type"] == "application/json"
    assert r.json()["fake"] is True
    r = await client.get(f"/api/songs/{job['id']}/transcription/score.abc")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.get(f"/api/songs/{job['id']}/artifacts.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == f'attachment; filename="{job["id"]}.zip"'
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert f"{job['id']}/song/audio.flac" in names and f"{job['id']}/summary.json" in names
    assert not any(n.endswith("artifacts.zip") for n in names)
    r = await client.get("/api/songs/doesnotexist/audio.flac")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.get(f"/api/songs/{job['id']}/nope.txt")
    assert r.status_code == 404


@pytest.mark.parametrize("path", ["audio.flac", "score.abc", "plan.json", "artifacts.zip"])
async def test_head_on_artifact_routes(client, api, path):
    job = await api.create()
    await api.wait(job["id"])
    get = await client.get(f"/api/songs/{job['id']}/{path}")
    head = await client.head(f"/api/songs/{job['id']}/{path}")
    assert head.status_code == 200 and head.content == b""
    assert head.headers["content-type"] == get.headers["content-type"]
    assert int(head.headers["content-length"]) == len(get.content)
    assert (await client.head(f"/api/songs/{job['id']}/transcription/score.abc")).status_code == 404
    assert (await client.head(f"/api/songs/nope/{path}")).status_code == 404


async def test_artifacts_404_before_done(client, api, app):
    app.state.engine.delay = 0.02
    running = await api.create()
    queued = await api.create()
    await api.wait_status(running["id"], "running")
    for path in ("audio.flac", "score.abc", "plan.json", "artifacts.zip", "audio.mp3"):
        r = await client.get(f"/api/songs/{queued['id']}/{path}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", path
    for path in ("audio.flac", "audio.mp3"):  # the running job has a dir (job.json) but no audio yet
        r = await client.get(f"/api/songs/{running['id']}/{path}")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", path
    app.state.engine.delay = 0
    await api.wait(queued["id"])
    assert (await client.get(f"/api/songs/{queued['id']}/audio.flac")).status_code == 200


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_audio_mp3_transcode(client, api, app):
    job = await api.create()
    await api.wait(job["id"])
    r = await client.get(f"/api/songs/{job['id']}/audio.mp3")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 1000
    mp3 = app.state.paths.songs_dir / job["id"] / "song" / "audio.mp3"
    assert mp3.is_file() and mp3.stat().st_size == len(r.content)
    mp3.write_bytes(b"cached")  # newer than the flac -> served as is, no re-encode
    r2 = await client.get(f"/api/songs/{job['id']}/audio.mp3")
    assert r2.content == b"cached"
    head = await client.head(f"/api/songs/{job['id']}/audio.mp3")
    assert head.status_code == 200 and head.content == b"" and head.headers["content-length"] == "6"
    names = zipfile.ZipFile(io.BytesIO((await client.get(f"/api/songs/{job['id']}/artifacts.zip")).content))
    assert f"{job['id']}/song/audio.mp3" in names.namelist()


async def test_audio_mp3_503_without_ffmpeg(client, api, monkeypatch):
    job = await api.create()
    await api.wait(job["id"])
    monkeypatch.setattr(audio, "ffmpeg_path", lambda: None)
    r = await client.get(f"/api/songs/{job['id']}/audio.mp3")
    assert r.status_code == 503 and r.json()["error"]["code"] == "engine_unavailable"
    assert (await client.get("/api/songs/nope/audio.mp3")).status_code == 404  # unknown job wins


# -- jobs: listing ------------------------------------------------------------------------------


async def test_list_jobs_filters_paging_and_total(client, api, app):
    a = await api.create()
    b = await api.create()
    await api.wait(b["id"])
    r = await client.get("/api/jobs")
    assert r.status_code == 200 and r.json()["total"] == 2
    assert [j["id"] for j in r.json()["jobs"]] == [b["id"], a["id"]]  # newest first
    assert all(set(j) == JOB_KEYS for j in r.json()["jobs"])
    r = await client.get("/api/jobs", params={"status": "done", "kind": "create"})
    assert r.json()["total"] == 2 and {j["status"] for j in r.json()["jobs"]} == {"done"}
    r = await client.get("/api/jobs", params={"status": "queued,running"})
    assert r.json() == {"jobs": [], "total": 0}
    r = await client.get("/api/jobs", params={"status": "done,failed,cancelled"})
    assert r.json()["total"] == 2
    r = await client.get("/api/jobs", params={"status": " done , cancelled "})  # whitespace tolerated
    assert r.json()["total"] == 2
    r = await client.get("/api/jobs", params={"limit": 1, "offset": 1})
    assert r.json()["total"] == 2 and [j["id"] for j in r.json()["jobs"]] == [a["id"]]
    r = await client.get("/api/jobs", params={"limit": 1})
    assert r.json()["total"] == 2 and [j["id"] for j in r.json()["jobs"]] == [b["id"]]
    r = await client.get("/api/jobs", params={"offset": 5})
    assert r.json() == {"jobs": [], "total": 2}
    r = await client.get("/api/jobs", params={"limit": 500})
    assert r.status_code == 200
    for bad in ({"status": "bogus"}, {"status": "done,bogus"}, {"kind": "variations"}, {"limit": 0},
                {"limit": 501}, {"limit": "x"}, {"offset": -1}):
        r = await client.get("/api/jobs", params=bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", bad
    r = await client.get("/api/jobs/nope")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "not_found", "message": "job 'nope' not found"}}


async def test_list_jobs_mixed_kinds_and_statuses(client, api, app):
    app.state.engine.delay = 0.02
    running = await api.create()
    up = (await client.post("/api/upload", files={"file": ("x.wav", b"RIFF", "audio/wav")})).json()
    cover = (await api.submit({"kind": "cover", "params": {"upload_id": up["upload_id"], "style": "s",
                                                           "lyrics": "l"}}))["job"]
    queued = await api.create()
    await api.wait_status(running["id"], "running")
    r = await client.get("/api/jobs", params={"status": "queued,running"})
    assert r.json()["total"] == 3 and {j["status"] for j in r.json()["jobs"]} == {"queued", "running"}
    positions = {j["id"]: j["position"] for j in r.json()["jobs"]}
    assert positions == {running["id"]: None, cover["id"]: 0, queued["id"]: 1}
    r = await client.get("/api/jobs", params={"kind": "cover,regenerate"})
    assert [j["id"] for j in r.json()["jobs"]] == [cover["id"]] and r.json()["total"] == 1
    r = await client.get("/api/jobs", params={"kind": "cover", "status": "done"})
    assert r.json()["total"] == 0
    app.state.engine.delay = 0
    await client.post(f"/api/jobs/{queued['id']}/cancel")
    await client.post(f"/api/jobs/{cover['id']}/cancel")
    await api.wait(running["id"])


# -- jobs: validation ---------------------------------------------------------------------------


async def test_validation_errors_shape(client):
    r = await client.post("/api/jobs", json={"kind": "create", "params": {"style": "", "lyrics": "x"}})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "validation_error" and "style" in r.json()["error"]["message"]
    r = await client.post("/api/jobs", json={"kind": "create",
                                             "params": {**BASE, "cot": "off", "abc": "X:1"}})
    assert r.status_code == 400 and "cot" in r.json()["error"]["message"]
    r = await client.post("/api/jobs", json={"kind": "regenerate",
                                             "params": {"parent_id": "zzz", "abc": "X:1"}})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = await client.post("/api/jobs", json={"kind": "cover", "params": {"upload_id": "zzz", "style": "s",
                                                                          "lyrics": "l"}})
    assert r.status_code == 404
    r = await client.post("/api/jobs", content=b"{bad json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    r = await client.post("/api/jobs", json=[1, 2])
    assert r.status_code == 400
    r = await client.post("/api/jobs", json={"kind": "create", "preset": "custom", "params": BASE})
    assert r.status_code == 400 and "custom" in r.json()["error"]["message"]
    r = await client.post("/api/jobs", json={"kind": "create", "preset": "custom", "precision": "8bit",
                                             "ode_steps": 3, "params": BASE})
    assert r.status_code == 400
    r = await client.post("/api/jobs", json={"kind": "create", "preset": "custom", "precision": "fp8",
                                             "ode_steps": 8, "params": BASE})
    assert r.status_code == 400
    r = await client.post("/api/jobs", json={"kind": "variations", "params": {"count": 17, "base": BASE}})
    assert r.status_code == 400 and "count" in r.json()["error"]["message"]
    r = await client.post("/api/jobs", json={"kind": "variations", "params": {"count": 2,
                                                                              "base": {"style": "x"}}})
    assert r.status_code == 400 and "lyrics" in r.json()["error"]["message"]
    r = await client.post("/api/jobs", json={"kind": "create", "params": {**BASE, "seed": 2**63}})
    assert r.status_code == 400 and "seed" in r.json()["error"]["message"]
    r = await client.post("/api/jobs", json={"kind": "create", "params": {**BASE, "cfg_scale": -1}})
    assert r.status_code == 400 and "cfg_scale" in r.json()["error"]["message"]
    assert (await client.get("/api/jobs")).json()["total"] == 0  # nothing was stored


async def test_unknown_fields_ignored_and_seed_randomised(client, api):
    body = await api.submit({"kind": "create", "bogus": 1,
                             "params": {"style": "s", "lyrics": "l", "extra": True, "cot": "melody",
                                        "abc": "X:1\nK:C\nC|"}})
    job = body["job"]
    assert "extra" not in job["params"] and job["params"]["abc"] == "X:1\nK:C\nC|"
    assert isinstance(job["seed"], int) and 0 <= job["seed"] < 2**31 and job["params"]["seed"] == job["seed"]
    blank = await api.create({**BASE, "abc": "   "})
    assert blank["params"]["abc"] is None  # whitespace-only abc means "not supplied"


# -- variations ---------------------------------------------------------------------------------


async def test_variations_returns_group_and_jobs(client, api):
    body = await api.submit({"kind": "variations", "preset": "fast",
                             "params": {"count": 3, "base": {**BASE, "seed": 10}}})
    group, members = body["group"], body["jobs"]
    assert set(body) == {"group", "jobs"}
    assert set(group) == {"id", "label", "created_at", "job_ids"} and len(group["id"]) == 32
    assert group["label"] == "dreamy indie pop, female vocal ×3"
    assert [j["seed"] for j in members] == [10, 11, 12]
    assert group["job_ids"] == [j["id"] for j in members]
    assert [j["seq"] for j in members] == sorted(j["seq"] for j in members)
    assert all(j["group_id"] == group["id"] and j["kind"] == "create" and j["preset"] == "fast"
               for j in members)
    positions = [j["position"] for j in members if j["status"] == "queued"]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    for j in members:
        await api.wait(j["id"])
    r = await client.get("/api/jobs", params={"group": group["id"]})
    assert r.json()["total"] == 3 and {j["status"] for j in r.json()["jobs"]} == {"done"}
    assert (await client.get("/api/jobs", params={"group": "nope"})).json() == {"jobs": [], "total": 0}


async def test_variations_random_seeds_and_label(api):
    body = await api.submit({"kind": "variations",
                             "params": {"count": 4, "base": {**BASE, "title": "Song"}, "random_seeds": True,
                                        "label": "my batch"}})
    assert body["group"]["label"] == "my batch"
    seeds = [j["seed"] for j in body["jobs"]]
    assert len(seeds) == 4 and all(0 <= s < 2**31 for s in seeds)
    assert all(j["title"] == "Song" for j in body["jobs"])
    body = await api.submit({"kind": "variations", "params": {"count": 2, "base": {**BASE, "title": "Song"}}})
    assert body["group"]["label"] == "Song ×2"


# -- cancel / delete ----------------------------------------------------------------------------


async def test_cancel_queued_running_and_terminal(client, api, app):
    app.state.engine.delay = 0.02
    first = await api.create()
    second = await api.create()
    assert second["status"] == "queued" and second["position"] is not None
    # queued -> 200 cancelled immediately
    r = await client.post(f"/api/jobs/{second['id']}/cancel")
    assert r.status_code == 200 and r.json()["job"]["status"] == "cancelled"
    assert r.json()["job"]["started_at"] is None and r.json()["job"]["finished_at"]
    progress, done = await api.events(second["id"])
    assert done["status"] == "cancelled" and len(progress) == 1 and progress[0]["status"] == "cancelled"
    # running -> 202, then the stream delivers cancelled + done
    await api.wait_status(first["id"], "running")
    r = await client.post(f"/api/jobs/{first['id']}/cancel")
    assert r.status_code == 202 and r.json()["job"]["status"] == "running"
    progress, done = await api.events(first["id"])
    assert done["status"] == "cancelled" and progress[-1]["status"] == "cancelled"
    assert done["artifacts"]["audio"] is False
    # terminal -> 409
    r = await client.post(f"/api/jobs/{first['id']}/cancel")
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    r = await client.post("/api/jobs/nope/cancel")
    assert r.status_code == 404
    # the worker is idle and healthy again
    app.state.engine.delay = 0
    assert (await api.wait((await api.create())["id"]))["status"] == "done"


async def test_delete_job(client, api, app):
    job = await api.create()
    await api.wait(job["id"])
    song_dir = app.state.paths.songs_dir / job["id"]
    assert song_dir.is_dir()
    r = await client.delete(f"/api/jobs/{job['id']}")
    assert r.status_code == 204 and r.content == b"" and not song_dir.exists()
    r = await client.delete(f"/api/jobs/{job['id']}")
    assert r.status_code == 404
    assert (await client.get(f"/api/jobs/{job['id']}/events")).status_code == 404
    # deleting a queued job cancels it first; deleting the last group member drops the group
    app.state.engine.delay = 0.02
    blocker = await api.create()
    group = await api.submit({"kind": "variations", "params": {"count": 2, "base": BASE}})
    for member in group["jobs"]:
        assert (await client.delete(f"/api/jobs/{member['id']}")).status_code == 204
    assert (await client.get("/api/jobs", params={"group": group["group"]["id"]})).json()["total"] == 0
    await api.wait_status(blocker["id"], "running")
    r = await client.delete(f"/api/jobs/{blocker['id']}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    await client.post(f"/api/jobs/{blocker['id']}/cancel")
    await api.wait(blocker["id"])
    assert (await client.delete(f"/api/jobs/{blocker['id']}")).status_code == 204
    assert (await client.get("/api/jobs")).json()["total"] == 0


async def test_failed_job_reports_error(client, api, app):
    app.state.engine.fail = True
    try:
        job = await api.create()
        progress, done = await api.events(job["id"])
        assert done["status"] == "failed" and "fake synthesis failure" in done["error"]
        assert progress[-1]["status"] == "failed" and "fake synthesis failure" in progress[-1]["message"]
        assert done["artifacts"]["plan"] is True and done["artifacts"]["audio"] is False
        assert done["timing"] is None
        r = await client.get(f"/api/songs/{job['id']}/score.abc")  # falls back to plan/score.abc
        assert r.status_code == 200 and r.text.startswith("X:1")
        r = await client.get(f"/api/songs/{job['id']}/audio.flac")
        assert r.status_code == 404
        assert (await client.get("/api/status")).json()["engine"]["state"] == "cold"  # pipeline discarded
    finally:
        app.state.engine.fail = False
    # the next job reloads the engine and succeeds
    job = await api.create()
    progress, done = await api.events(job["id"])
    assert done["status"] == "done" and any(e["stage"] == "load" and e["type"] == "status" for e in progress)


# -- SSE framing ----------------------------------------------------------------------------------


async def test_sse_terminal_job_replays_cached_event_then_done(client, api):
    job = await api.create()
    await api.wait(job["id"])
    progress, done = await api.events(job["id"])
    assert len(progress) == 1 and progress[0]["type"] == "status" and progress[0]["status"] == "done"
    assert done["id"] == job["id"] and done["status"] == "done"
    raw = (await client.get(f"/api/jobs/{job['id']}/events")).content  # terminal: completes on its own
    assert b"\r" not in raw
    assert raw.startswith(b"event: progress\ndata: {") and raw.endswith(b"}\n\n")
    frames = raw.split(b"\n\n")
    assert frames[-1] == b"" and len(frames) == 3  # exactly two frames
    assert frames[1].startswith(b'event: done\ndata: {"job":{"id":"' + job["id"].encode())
    for frame in frames[:2]:
        lines = frame.split(b"\n")
        assert len(lines) == 2 and lines[1].startswith(b"data: ")  # single-line JSON payloads
        json.loads(lines[1][6:])
    assert (await client.get("/api/jobs/nope/events")).status_code == 404


async def test_sse_keepalive_comment_frames(make_app, home, monkeypatch):
    monkeypatch.setattr(api_module, "SSE_KEEPALIVE_SECONDS", 0.01)
    # No lifespan: the worker never starts, so the job stays queued and the stream idles.
    async with make_app(FakeEngine(delay=0), home=home, lifespan=False) as (app, client):
        app.state.bus.bind(asyncio.get_running_loop())
        job = (await client.post("/api/jobs", json={"kind": "create", "params": BASE})).json()["job"]
        seen = asyncio.Event()
        original = api_module.ServerSentEvent

        class Counting(original):  # observe keepalive frames as the route builds them
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if kwargs.get("comment") == "keepalive":
                    Counting.count += 1
                    if Counting.count >= 2:
                        seen.set()

        Counting.count = 0
        monkeypatch.setattr(api_module, "ServerSentEvent", Counting)
        stream = asyncio.create_task(client.get(f"/api/jobs/{job['id']}/events"))
        await asyncio.wait_for(seen.wait(), 5)  # two keepalives sent, then cancel ends the stream
        r = await client.post(f"/api/jobs/{job['id']}/cancel")
        assert r.status_code == 200
        raw = (await asyncio.wait_for(stream, 5)).content
    frames = raw.split(b"\n\n")
    assert frames[-1] == b""
    keepalives = [f for f in frames if f == b": keepalive"]
    assert len(keepalives) >= 2 and frames[0] == b": keepalive"
    assert b"\r" not in raw
    tail = [f for f in frames[:-1] if f != b": keepalive"]
    assert len(tail) == 2 and tail[0].startswith(b"event: progress\ndata: ")
    assert json.loads(tail[0].split(b"\n", 1)[1][6:])["status"] == "cancelled"
    assert tail[1].startswith(b"event: done\ndata: ") and json.loads(tail[1].split(b"\n", 1)[1][6:])["job"][
        "status"] == "cancelled"


async def test_sse_late_subscriber_sees_last_event_first(client, api, app):
    app.state.engine.delay = 0.02
    job = await api.create()
    await api.wait_status(job["id"], "running")
    while ((await api.get(job["id"]))["progress"] or {}).get("type") != "stage":  # first stage event
        await asyncio.sleep(0.005)
    progress, done = await api.events(job["id"])
    assert done["status"] == "done"
    assert progress[0]["type"] == "stage"  # cached last event first, not the run-from-the-beginning status
    assert progress[0]["job_id"] == job["id"] and progress[-1]["status"] == "done"
    assert (await api.get(job["id"]))["progress"]["status"] == "done"


# -- regenerate ---------------------------------------------------------------------------------


async def test_regenerate_flow(client, api):
    parent = await api.create(preset="fast")
    await api.wait(parent["id"])
    score = (await client.get(f"/api/songs/{parent['id']}/score.abc")).text
    job = (await api.submit({"kind": "regenerate",
                             "params": {"parent_id": parent["id"], "abc": score + "G|"}}))["job"]
    assert job["kind"] == "regenerate" and job["parent_id"] == parent["id"] and job["preset"] == "fast"
    assert job["params"]["abc"] == score + "G|" and job["seed"] == parent["seed"]
    assert job["params"]["style"] == BASE["style"] and job["params"]["cot"] == "full"
    assert job["params"]["parent_id"] == parent["id"]
    done = await api.wait(job["id"])
    assert done["status"] == "done" and done["timing"]["abc_tps"] is None  # supplied score: no planning
    assert (await client.get(f"/api/songs/{job['id']}/score.abc")).text == score + "G|"


async def test_regenerate_validation_and_inheritance(client, api):
    parent = await api.create({**BASE, "cot": "off", "title": "Orig", "cfg_scale": 2.0}, preset="fast")
    for abc in ("", "   ", None):
        r = await client.post("/api/jobs", json={"kind": "regenerate", "params": {"parent_id": parent["id"],
                                                                                  "abc": abc}})
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", abc
        assert "abc" in r.json()["error"]["message"]
    r = await client.post("/api/jobs", json={"kind": "regenerate", "params": {"abc": "X:1"}})
    assert r.status_code == 400 and "parent_id" in r.json()["error"]["message"]
    job = (await api.submit({"kind": "regenerate", "preset": "quality",
                             "params": {"parent_id": parent["id"], "abc": "X:1\nK:C\nC|", "seed": 9,
                                        "style": "louder"}}))["job"]
    assert job["params"]["cot"] == "melody"  # parent cot=off -> melody (a score needs cot != off)
    assert job["params"]["style"] == "louder" and job["params"]["lyrics"] == BASE["lyrics"]
    assert job["seed"] == 9 and job["title"] == "Orig" and job["params"]["cfg_scale"] == 2.0
    assert (job["preset"], job["precision"], job["ode_steps"]) == ("quality", "bf16", 32)


# -- upload + cover -----------------------------------------------------------------------------


async def test_upload_validation_and_cover(client, api, app):
    r = await client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    assert "txt" in r.json()["error"]["message"]
    r = await client.post("/api/upload", files={"other": ("a.mp3", b"x", "audio/mpeg")})
    assert r.status_code == 400
    r = await client.post("/api/upload", files={"file": ("noext", b"x", "audio/mpeg")})
    assert r.status_code == 400
    r = await client.post("/api/upload",
                          files={"file": ("demo song.MP3", b"\xff\xfb" + b"\0" * 100, "audio/mpeg")})
    assert r.status_code == 201, r.text
    up = r.json()
    assert set(up) == {"upload_id", "filename", "seconds", "path_hint"} and len(up["upload_id"]) == 32
    assert up["filename"] == "demo song.MP3" and up["path_hint"] == f"data/uploads/{up['upload_id']}.mp3"
    assert (app.state.paths.uploads_dir / f"{up['upload_id']}.mp3").is_file()
    assert up["seconds"] is None or isinstance(up["seconds"], float)

    body = await api.submit({"kind": "cover", "params": {"upload_id": up["upload_id"], "style": "jazz",
                                                         "lyrics": "la"}})
    job = body["job"]
    assert job["kind"] == "cover" and job["title"] == "demo song" and job["params"]["task"] == "melody-full"
    assert set(job["params"]) == {"upload_id", "task", "style", "lyrics", "seed", "title"}
    progress, done = await api.events(job["id"])
    assert done["status"] == "done" and done["artifacts"]["transcription"] is True
    assert done["timing"]["transcribe"] is not None and done["timing"]["transcribe"] >= 0
    stages = [e["stage"] for e in progress if e["type"] == "stage"]
    assert stages.index("transcribe") < stages.index("plan")
    assert any(e["type"] == "token" and e["phase"] == "transcription" for e in progress)
    r = await client.get(f"/api/songs/{job['id']}/transcription/score.abc")
    assert r.status_code == 200 and r.text.startswith("X:1")
    names = zipfile.ZipFile(io.BytesIO((await client.get(f"/api/songs/{job['id']}/artifacts.zip")).content))
    assert f"{job['id']}/transcription/score.abc" in names.namelist()
    # the same upload can be reused; an explicit title and task win over the defaults
    job2 = (await api.submit({"kind": "cover",
                              "params": {"upload_id": up["upload_id"], "style": "jazz", "lyrics": "la",
                                         "task": "full", "title": "Mine"}}))["job"]
    assert job2["title"] == "Mine" and job2["params"]["task"] == "full"
    r = await client.post("/api/jobs", json={"kind": "cover", "params": {"upload_id": up["upload_id"],
                                                                          "style": "s", "lyrics": "l",
                                                                          "task": "karaoke"}})
    assert r.status_code == 400


@pytest.mark.parametrize("name", ["a.mp3", "b.WAV", "c.flac", "d.m4a", "e.ogg", "dir/../f.Mp3"])
async def test_upload_accepted_extensions(client, name):
    r = await client.post("/api/upload", files={"file": (name, b"\0" * 16, "application/octet-stream")})
    assert r.status_code == 201, r.text
    assert r.json()["path_hint"].endswith("." + name.rsplit(".", 1)[-1].lower())
    assert "/" not in r.json()["filename"]


async def test_upload_rejects_empty_file(client, app):
    r = await client.post("/api/upload", files={"file": ("cloud.mp3", b"", "audio/mpeg")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    assert "empty (0 bytes)" in r.json()["error"]["message"]
    assert not any(app.state.paths.uploads_dir.iterdir())  # no file, no sidecar


async def test_upload_rejects_unreadable_audio_when_ffprobe_present(client, app, monkeypatch):
    monkeypatch.setattr(audio, "ffprobe_path", lambda: "/stub/ffprobe")
    monkeypatch.setattr(audio, "probe_duration", lambda path: None)
    r = await client.post("/api/upload", files={"file": ("bad.wav", b"\0" * 16, "audio/wav")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    assert "could not read the uploaded audio" in r.json()["error"]["message"]
    assert not any(app.state.paths.uploads_dir.iterdir())


async def test_upload_without_ffprobe_keeps_unknown_duration(client, app, monkeypatch):
    monkeypatch.setattr(audio, "ffprobe_path", lambda: None)
    monkeypatch.setattr(audio, "probe_duration", lambda path: None)
    r = await client.post("/api/upload", files={"file": ("x.wav", b"\0" * 16, "audio/wav")})
    assert r.status_code == 201, r.text
    assert r.json()["seconds"] is None
    assert (app.state.paths.uploads_dir / f"{r.json()['upload_id']}.wav").is_file()


async def test_upload_too_large(client, monkeypatch):
    monkeypatch.setattr(audio, "UPLOAD_MAX_BYTES", 1024)
    r = await client.post("/api/upload", files={"file": ("big.wav", b"\0" * 4096, "audio/wav")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_large"
    r = await client.post("/api/upload", files={"file": ("big.wav", b"\0" * 8192, "audio/wav")})
    assert r.status_code == 413  # rejected from Content-Length before the body is read
    r = await client.post("/api/upload", files={"file": ("ok.wav", b"\0" * 1024, "audio/wav")})
    assert r.status_code == 201


async def test_upload_id_lookup_rejects_odd_ids(client, api):
    up = (await client.post("/api/upload", files={"file": ("x.wav", b"\0" * 8, "audio/wav")})).json()
    for bad in ("", "../x", up["upload_id"] + ".wav", "nope"):
        r = await client.post("/api/jobs", json={"kind": "cover", "params": {"upload_id": bad, "style": "s",
                                                                              "lyrics": "l"}})
        assert r.status_code in {400, 404}, bad
    job = (await api.submit({"kind": "cover", "params": {"upload_id": up["upload_id"], "style": "s",
                                                         "lyrics": "l"}}))["job"]
    assert job["title"] == "x"


# -- uploads management ---------------------------------------------------------------------------


async def _upload(client, name: str, payload: bytes = b"\0" * 32) -> dict:
    r = await client.post("/api/upload", files={"file": (name, payload, "application/octet-stream")})
    assert r.status_code == 201, r.text
    await asyncio.sleep(0.002)  # created_at has millisecond resolution: keep the order unambiguous
    return r.json()


def _backdate(app, upload_id: str, days: int) -> None:
    sidecar = app.state.paths.uploads_dir / f"{upload_id}.json"
    info = json.loads(sidecar.read_text())
    stamp = datetime.now(UTC) - timedelta(days=days)
    info["created_at"] = stamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    sidecar.write_text(json.dumps(info))


async def test_list_uploads_newest_first_with_counts_and_unused_filter(client, api, app):
    assert (await client.get("/api/uploads")).json() == {"uploads": []}
    a = await _upload(client, "first.wav", b"\0" * 100)
    b = await _upload(client, "second.mp3")
    c = await _upload(client, "third.m4a")
    cover = {"kind": "cover", "params": {"upload_id": b["upload_id"], "style": "s", "lyrics": "l"}}
    for _ in range(2):
        await api.wait((await api.submit(cover))["job"]["id"])
    app.state.engine.delay = 0.05
    blocker = await api.create()
    hum = (await api.submit({"kind": "hum", "params": {"upload_id": b["upload_id"], "style": "s",
                                                       "lyrics": "l"}}))["job"]
    ups = (await client.get("/api/uploads")).json()["uploads"]
    assert [u["upload_id"] for u in ups] == [c["upload_id"], b["upload_id"], a["upload_id"]]
    assert set(ups[0]) == {"upload_id", "filename", "ext", "seconds", "size", "created_at", "broken", "jobs"}
    assert ups[2] == {"upload_id": a["upload_id"], "filename": "first.wav", "ext": "wav", "seconds": 12.3,
                      "size": 100, "created_at": ups[2]["created_at"], "broken": False,
                      "jobs": {"total": 0, "active": 0}}
    assert ups[1]["jobs"] == {"total": 3, "active": 1} and ups[1]["ext"] == "mp3"
    unused = (await client.get("/api/uploads", params={"unused": "true"})).json()["uploads"]
    assert [u["upload_id"] for u in unused] == [c["upload_id"], a["upload_id"]]
    await client.post(f"/api/jobs/{blocker['id']}/cancel")
    await api.wait(hum["id"])
    assert (await client.get("/api/uploads")).json()["uploads"][1]["jobs"] == {"total": 3, "active": 0}


async def test_delete_upload_204_404_409(client, api, app):
    up = await _upload(client, "song.wav")
    paths = app.state.paths
    app.state.engine.delay = 0.05
    blocker = await api.create()
    job = (await api.submit({"kind": "cover", "params": {"upload_id": up["upload_id"], "style": "s",
                                                         "lyrics": "l"}}))["job"]
    r = await client.delete(f"/api/uploads/{up['upload_id']}")  # referenced by a queued job
    assert r.status_code == 409 and r.json()["error"]["code"] == "in_use"
    assert (paths.uploads_dir / f"{up['upload_id']}.wav").is_file()
    await client.post(f"/api/jobs/{blocker['id']}/cancel")
    assert (await api.wait(job["id"]))["status"] == "done"
    r = await client.delete(f"/api/uploads/{up['upload_id']}")  # finished jobs do not pin it
    assert r.status_code == 204 and r.content == b""
    assert not any(paths.uploads_dir.iterdir())
    assert (await client.delete(f"/api/uploads/{up['upload_id']}")).status_code == 404
    for bad in ("nope", "a..b", "a.b", "x%20y"):
        r = await client.delete(f"/api/uploads/{bad}")
        assert r.status_code == 404, bad
    # the job row survives and still serves its artifacts
    assert (await client.get(f"/api/songs/{job['id']}/audio.flac")).status_code == 200


async def test_prune_uploads_counts_and_skips_in_use(client, api, app):
    old_free = await _upload(client, "old-free.wav")
    old_used = await _upload(client, "old-used.wav")
    new_free = await _upload(client, "new-free.wav")
    active = await _upload(client, "active.wav")
    _backdate(app, old_free["upload_id"], 10)
    _backdate(app, old_used["upload_id"], 10)
    cover = {"kind": "cover", "params": {"upload_id": old_used["upload_id"], "style": "s", "lyrics": "l"}}
    await api.wait((await api.submit(cover))["job"]["id"])
    app.state.engine.delay = 0.05
    blocker = await api.create()
    hum = (await api.submit({"kind": "hum", "params": {"upload_id": active["upload_id"], "style": "s",
                                                       "lyrics": "l"}}))["job"]
    r = await client.post("/api/uploads/prune", json={"unused": True, "older_than_days": 7})
    assert r.status_code == 200 and r.json() == {"deleted": 1, "skipped": 1}
    left = {u["upload_id"] for u in (await client.get("/api/uploads")).json()["uploads"]}
    assert left == {old_used["upload_id"], new_free["upload_id"], active["upload_id"]}
    r = await client.post("/api/uploads/prune", json={"unused": True, "older_than_days": None})
    assert r.json() == {"deleted": 1, "skipped": 2}  # new-free went; the referenced two stay
    r = await client.post("/api/uploads/prune", json={"unused": False})  # everything without an active job
    assert r.json() == {"deleted": 1, "skipped": 1}
    left = {u["upload_id"] for u in (await client.get("/api/uploads")).json()["uploads"]}
    assert left == {active["upload_id"]}
    for bad in ({"unused": "yes"}, {"older_than_days": 0}, {"older_than_days": "7"},
                {"older_than_days": True}, [1]):
        r = await client.post("/api/uploads/prune", json=bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", bad
    await client.post(f"/api/jobs/{blocker['id']}/cancel")
    await api.wait(hum["id"])
    assert (await client.post("/api/uploads/prune", json={})).json() == {"deleted": 0, "skipped": 1}


async def test_broken_uploads_are_listed_and_deletable(client, app):
    paths = app.state.paths
    no_media = await _upload(client, "gone.wav")
    (paths.uploads_dir / f"{no_media['upload_id']}.wav").unlink()
    no_sidecar = await _upload(client, "orphan.mp3")
    (paths.uploads_dir / f"{no_sidecar['upload_id']}.json").unlink()
    bad_json = await _upload(client, "garbled.ogg")
    (paths.uploads_dir / f"{bad_json['upload_id']}.json").write_text("{not json")
    (paths.uploads_dir / ".DS_Store").write_bytes(b"junk")  # never an upload
    ok = await _upload(client, "fine.flac")
    by_id = {u["upload_id"]: u for u in (await client.get("/api/uploads")).json()["uploads"]}
    assert set(by_id) == {no_media["upload_id"], no_sidecar["upload_id"], bad_json["upload_id"],
                          ok["upload_id"]}
    assert by_id[ok["upload_id"]]["broken"] is False and by_id[ok["upload_id"]]["size"] == 32
    assert by_id[no_media["upload_id"]]["broken"] is True and by_id[no_media["upload_id"]]["size"] is None
    assert by_id[no_media["upload_id"]]["filename"] == "gone.wav"
    orphan = by_id[no_sidecar["upload_id"]]
    assert orphan["broken"] is True and orphan["size"] is None and orphan["ext"] == "mp3"
    assert orphan["filename"] == f"{no_sidecar['upload_id']}.mp3" and orphan["created_at"].endswith("Z")
    assert by_id[bad_json["upload_id"]]["broken"] is True
    # broken uploads cannot be submitted, but can be deleted (both pieces go)
    r = await client.post("/api/jobs", json={"kind": "cover", "params": {"upload_id": no_sidecar["upload_id"],
                                                                          "style": "s", "lyrics": "l"}})
    assert r.status_code == 404
    for uid in (no_media["upload_id"], no_sidecar["upload_id"], bad_json["upload_id"]):
        assert (await client.delete(f"/api/uploads/{uid}")).status_code == 204
    assert sorted(p.name for p in paths.uploads_dir.iterdir()) == sorted(
        [".DS_Store", f"{ok['upload_id']}.flac", f"{ok['upload_id']}.json"])
    r = await client.post("/api/uploads/prune", json={"unused": True})
    assert r.json() == {"deleted": 1, "skipped": 0}


async def test_glob_metacharacters_in_upload_names_never_match_other_files(client, app):
    """``valid_id`` admits ``*``: a stray ``*.wav`` is its own entry and deleting it touches nothing else."""
    paths = app.state.paths
    keep = await _upload(client, "keep.wav")
    (paths.uploads_dir / "*.wav").write_bytes(b"\0" * 8)
    (paths.uploads_dir / "[ab].mp3").write_bytes(b"\0" * 8)
    by_id = {u["upload_id"]: u for u in (await client.get("/api/uploads")).json()["uploads"]}
    assert set(by_id) == {keep["upload_id"], "*", "[ab]"}
    assert by_id["*"]["broken"] is True and by_id["*"]["filename"] == "*.wav"
    # the worker resolves the media file by exact stem too
    assert app.state.worker._upload_path("*").name == "*.wav"
    assert app.state.worker._upload_path(keep["upload_id"]).name == f"{keep['upload_id']}.wav"
    assert (await client.delete("/api/uploads/*")).status_code == 204
    assert sorted(p.name for p in paths.uploads_dir.iterdir()) == sorted(
        ["[ab].mp3", f"{keep['upload_id']}.wav", f"{keep['upload_id']}.json"])
    r = await client.post("/api/uploads/prune", json={"unused": True})  # prunes [ab] and keep, one file each
    assert r.json() == {"deleted": 2, "skipped": 0} and not any(paths.uploads_dir.iterdir())


async def test_auto_prune_setting_runs_after_jobs_and_at_startup(make_app, home, api):
    async with make_app(FakeEngine(delay=0), home=home) as (app, client):
        api = Api(client)
        await client.put("/api/settings", json={"prune_uploads_days": 2})
        stale = await _upload(client, "stale.wav")
        fresh = await _upload(client, "fresh.wav")
        pinned = await _upload(client, "pinned.wav")
        _backdate(app, stale["upload_id"], 3)
        _backdate(app, pinned["upload_id"], 3)
        app.state.engine.delay = 0.05
        blocker = await api.create()
        hum = (await api.submit({"kind": "hum", "params": {"upload_id": pinned["upload_id"], "style": "s",
                                                           "lyrics": "l"}}))["job"]
        assert len((await client.get("/api/uploads")).json()["uploads"]) == 3  # nothing finished yet
        await client.post(f"/api/jobs/{blocker['id']}/cancel")
        await api.wait(hum["id"])
        await asyncio.sleep(0.05)  # the hook runs on the worker thread right after ``done``
        left = {u["upload_id"] for u in (await client.get("/api/uploads")).json()["uploads"]}
        assert left == {fresh["upload_id"], pinned["upload_id"]}  # stale pruned; pinned is referenced
        # a job that finishes with the setting off prunes nothing
        await client.put("/api/settings", json={"prune_uploads_days": None})
        _backdate(app, fresh["upload_id"], 3)
        await api.wait((await api.create())["id"])
        await asyncio.sleep(0.05)
        assert len((await client.get("/api/uploads")).json()["uploads"]) == 2
        await client.put("/api/settings", json={"prune_uploads_days": 1})
    async with make_app(FakeEngine(delay=0), home=home) as (app, client):  # startup applies the setting
        left = {u["upload_id"] for u in (await client.get("/api/uploads")).json()["uploads"]}
        assert left == {pinned["upload_id"]}


# -- availability: real model paths, no models --------------------------------------------------


async def test_models_missing_gives_503_with_real_paths(make_app, home):
    async with make_app(ColdEngine(), home=home) as (app, c):
        status = (await c.get("/api/status")).json()
        assert status["fake"] is False and status["models"]["present"] is False
        assert status["models"]["precisions"] == [] and status["cover"]["available"] is False
        assert "MERT-v2-FullSong not downloaded" in status["cover"]["reasons"]
        assert "SheetSage2 not downloaded" in status["cover"]["reasons"]
        assert status["models"]["converted_dir"] == str(home.resolve() / "models" / "converted")
        r = await c.post("/api/jobs", json={"kind": "create", "params": BASE})
        assert r.status_code == 503 and r.json()["error"]["code"] == "engine_unavailable"
        r = await c.post("/api/jobs", json={"kind": "variations", "params": {"count": 2, "base": BASE}})
        assert r.status_code == 503
        r = await c.post("/api/jobs", json={"kind": "create", "params": {"style": ""}})
        assert r.status_code == 400  # malformed bodies are 400 even when models are missing
        assert (await c.get("/api/jobs")).json()["total"] == 0


async def test_hum_submit_streams_and_serves_artifacts(client, api, app, home):
    from test_lora import hum_tensors, nar_tensors, write_safetensors

    loras_dir = home / "models" / "loras"
    write_safetensors(loras_dir / "hum_v1.safetensors", hum_tensors(), {"inject_layers": "[0, 1]"})
    write_safetensors(loras_dir / "plain.safetensors", nar_tensors(layers=1))
    status = (await client.get("/api/status")).json()
    assert status["hum"] == {"available": True, "reasons": [], "adapters": ["hum_v1"]}
    r = await client.post("/api/upload", files={"file": ("hum-2026.webm", b"\x1aE\xdf\xa3" + b"\0" * 64,
                                                         "audio/webm")})
    assert r.status_code == 201, r.text
    up = r.json()
    assert up["path_hint"].endswith(".webm")

    job = (await api.submit({"kind": "hum", "params": {"upload_id": up["upload_id"], "style": "lo-fi",
                                                        "lyrics": "la", "adapter": "hum_v1",
                                                        "hum_influence": 1.5, "offset_s": 0.5}}))["job"]
    assert job["kind"] == "hum" and job["title"] == "hum-2026"
    assert set(job["params"]) == {"upload_id", "style", "lyrics", "seed", "title", "melody", "adapter",
                                  "hum_influence", "offset_s"}
    assert job["params"]["melody"] == "continue" and job["params"]["hum_influence"] == 1.5
    progress, done = await api.events(job["id"])
    assert done["status"] == "done" and done["artifacts"]["hum"] is True
    assert done["artifacts"]["transcription"] is True and done["timing"]["transcribe"] is not None
    stages = [e["stage"] for e in progress if e["type"] == "stage"]
    assert stages.index("transcribe") < stages.index("hum") < stages.index("plan")
    partial = [e for e in progress if e["type"] == "abc" and e["partial"]]
    assert partial and partial[0]["text"].startswith("X:1\nT:Hum\n")
    r = await client.get(f"/api/songs/{job['id']}/hum/hum.abc")
    assert r.status_code == 200 and r.text.startswith("X:1\nT:Hum")
    r = await client.get(f"/api/songs/{job['id']}/hum.json")
    assert r.status_code == 200 and r.json()["adapter"] == "hum_v1" and r.json()["melody"] == "continue"
    names = zipfile.ZipFile(io.BytesIO((await client.get(f"/api/songs/{job['id']}/artifacts.zip")).content))
    assert f"{job['id']}/hum/hum.abc" in names.namelist() and f"{job['id']}/hum.json" in names.namelist()
    listed = (await client.get("/api/jobs?kind=hum")).json()
    assert [j["id"] for j in listed["jobs"]] == [job["id"]]

    # no adapter, melody hum_only: no hum stage, score used verbatim
    plain = (await api.submit({"kind": "hum", "params": {"upload_id": up["upload_id"], "style": "s",
                                                          "lyrics": "l", "melody": "hum_only"}}))["job"]
    progress, done = await api.events(plain["id"])
    assert done["status"] == "done" and "hum" not in [e["stage"] for e in progress if e["type"] == "stage"]
    assert (await client.get(f"/api/songs/{plain['id']}/hum/hum.abc")).status_code == 200
    # melody ignore with an adapter: no transcription at all
    ignore = (await api.submit({"kind": "hum", "params": {"upload_id": up["upload_id"], "style": "s",
                                                           "lyrics": "l", "melody": "ignore",
                                                           "adapter": "hum_v1"}}))["job"]
    progress, done = await api.events(ignore["id"])
    assert done["status"] == "done" and done["artifacts"]["transcription"] is False
    assert done["artifacts"]["hum"] is False
    assert (await client.get(f"/api/songs/{ignore['id']}/hum/hum.abc")).status_code == 404

    # validation: unknown adapter, a plain LoRA as adapter, a hum adapter in the LoRA stack, bad ranges
    base = {"upload_id": up["upload_id"], "style": "s", "lyrics": "l"}
    for body, fragment in [
        ({"kind": "hum", "params": {**base, "adapter": "nope"}}, "unknown or unusable hum adapter"),
        ({"kind": "hum", "params": {**base, "adapter": "plain"}}, "unknown or unusable hum adapter"),
        ({"kind": "hum", "params": {**base, "melody": "ignore"}}, "needs a hum adapter"),
        ({"kind": "hum", "params": {**base, "hum_influence": 9}}, "hum_influence"),
        ({"kind": "hum", "params": {**base, "offset_s": -2}}, "offset_s"),
        ({"kind": "create", "params": BASE, "loras": [{"name": "hum_v1"}]}, "unknown or unusable LoRA"),
    ]:
        r = await client.post("/api/jobs", json=body)
        assert r.status_code == 400, r.text
        assert fragment in r.json()["error"]["message"], r.text
    r = await client.post("/api/jobs", json={"kind": "hum", "params": {**base, "upload_id": "missing"}})
    assert r.status_code == 404


async def test_hum_409_when_unavailable(make_app, home, monkeypatch):
    fake_models(home)
    async with make_app(ColdEngine(), home=home) as (app, c):
        status = (await c.get("/api/status")).json()
        assert status["hum"]["available"] is False and "not downloaded" in " ".join(status["hum"]["reasons"])
        up = (await c.post("/api/upload", files={"file": ("x.m4a", b"\0" * 8, "audio/mp4")})).json()
        r = await c.post("/api/jobs", json={"kind": "hum", "params": {"upload_id": up["upload_id"],
                                                                       "style": "s", "lyrics": "l"}})
        assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
        assert "hum is unavailable" in r.json()["error"]["message"]


async def test_cover_409_when_unavailable(make_app, home, monkeypatch):
    fake_models(home)
    async with make_app(ColdEngine(), home=home) as (app, c):
        status = (await c.get("/api/status")).json()
        assert status["models"]["present"] is True and status["models"]["precisions"] == ["8bit"]
        assert status["cover"]["available"] is False
        up = (await c.post("/api/upload", files={"file": ("x.wav", b"\0" * 8, "audio/wav")})).json()
        r = await c.post("/api/jobs", json={"kind": "cover", "params": {"upload_id": up["upload_id"],
                                                                         "style": "s", "lyrics": "l"}})
        assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
        assert "not downloaded" in r.json()["error"]["message"]
        r = await c.post("/api/jobs", json={"kind": "cover", "params": {"upload_id": up["upload_id"],
                                                                         "style": "", "lyrics": "l"}})
        assert r.status_code == 400  # validation still comes first
        assert (await c.get("/api/jobs")).json()["total"] == 0


# -- restart ------------------------------------------------------------------------------------


async def test_restart_requeues_queued_and_fails_stale_running(make_app, home):
    engine = FakeEngine(delay=0.02)
    async with make_app(engine, home=home) as (app, c):
        api = Api(c)
        running = await api.create()
        queued = [await api.create() for _ in range(2)]
        await api.wait_status(running["id"], "running")
    # lifespan exit = Worker.stop(): the running job is cancelled, queued rows stay queued
    engine.delay = 0
    async with make_app(FakeEngine(delay=0), home=home, lifespan=False) as (app, c):
        r = (await c.get("/api/jobs", params={"status": "queued"})).json()
        assert {j["id"] for j in r["jobs"]} == {j["id"] for j in queued} and r["total"] == 2
        assert (await c.get(f"/api/jobs/{running['id']}")).json()["job"]["status"] == "cancelled"
        app.state.store.update_status(queued[0]["id"], "running")  # simulate a crash mid-job
    async with make_app(FakeEngine(delay=0), home=home) as (app, c):
        api = Api(c)
        stale = await api.get(queued[0]["id"])
        assert stale["status"] == "failed" and "restarted" in stale["error"]
        done = await api.wait(queued[1]["id"])
        assert done["status"] == "done"
        assert (await c.get("/api/jobs", params={"status": "queued,running"})).json()["total"] == 0


# -- static / SPA -------------------------------------------------------------------------------


async def test_spa_fallback_and_static(client, static_dir):
    r = await client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "YuE2 Studio" in r.text
    for path in ("/song/abc", "/library", "/a/b/c", "/v1.2/route"):
        r = await client.get(path)
        assert r.status_code == 200 and "text/html" in r.headers["content-type"], path
    for path in ("/missing.png", "/js/app.js", "/static/app.js", "/static/missing.css"):
        r = await client.get(path)
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", path
    for path in ("/api", "/api/", "/api/nope", "/api/jobs/x/nope"):
        r = await client.get(path)
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", path
    r = await client.get("/static/../pyproject.toml")
    assert r.status_code == 404
    static_dir.mkdir()
    (static_dir / "index.html").write_text("<!doctype html><title>real ui</title>")
    (static_dir / "app.js").write_text("console.log(1)")
    r = await client.get("/library")
    assert r.status_code == 200 and "real ui" in r.text
    r = await client.get("/static/app.js")
    assert r.status_code == 200 and "console.log" in r.text
    assert r.headers["cache-control"] == "no-cache"  # modules revalidate (ETag) on every load
    r = await client.get("/static/missing.js")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


async def test_bundled_static_ui_is_served(make_app, home):
    from yue2_studio.main import STATIC_DIR

    assert (STATIC_DIR / "index.html").is_file() and (STATIC_DIR / "app.js").is_file()
    async with make_app(FakeEngine(delay=0), home=home, static_dir=STATIC_DIR, lifespan=False) as (app, c):
        r = await c.get("/")
        assert r.status_code == 200 and "<title>" in r.text and "/static/app.js" in r.text
        assert (await c.get("/#/library")).status_code == 200
        r = await c.get("/static/app.js")
        assert r.status_code == 200 and "javascript" in r.headers["content-type"]
        r = await c.get("/static/styles.css")
        assert r.status_code == 200 and "text/css" in r.headers["content-type"]
