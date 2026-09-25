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
from yue2_studio import assist, audio, config
from yue2_studio.fake import FakeEngine

JOB_KEYS = {"id", "kind", "status", "group_id", "parent_id", "preset", "precision", "ode_steps", "loras",
            "seed", "params", "title", "created_at", "started_at", "finished_at", "error", "timing",
            "truncated", "progress", "artifacts", "position", "seq", "take"}
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
    assert set(body) == {"engine", "memory", "queue", "presets", "models", "cover", "hum", "loras", "assist",
                         "ffmpeg", "fake", "version"}
    assert set(body["assist"]) == {"provider", "cli", "api_key", "model", "reasons"}
    assert body["assist"]["provider"] in ("cli", "api", None)
    assert isinstance(body["assist"]["cli"], bool) and isinstance(body["assist"]["api_key"], bool)
    assert body["hum"] == {"available": True, "reasons": [], "adapters": []}
    assert body["engine"] == {"state": "cold", "precision": None, "memory_gib": None, "current_job_id": None,
                              "loras": [], "low_memory": None}
    assert body["memory"] == {"machine_ram_gib": 48.0, "min_memory_budget_gib": 6.0,
                              "max_memory_budget_gib": 44.0,
                              "memory_budget_gib": config.DEFAULT_MEMORY_BUDGET_GIB, "low_memory": "auto",
                              "low_memory_effective": config.resolve_low_memory(
                                  "auto", config.DEFAULT_MEMORY_BUDGET_GIB)}
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


DEFAULT_PUBLIC_SETTINGS = {"default_preset": "quality",
                           "memory_budget_gib": config.DEFAULT_MEMORY_BUDGET_GIB, "require_ac": False,
                           "fast_numerics": True, "low_memory": "auto", "theme": "system",
                           "prune_uploads_days": None, "assist_provider": "auto", "assist_model": "",
                           "has_api_key": False,
                           # read-only machine fields (the tests pin a 48 GB Mac, see conftest.machine_ram)
                           "machine_ram_gib": 48.0, "min_memory_budget_gib": 6.0,
                           "max_memory_budget_gib": 44.0,
                           "low_memory_effective": config.resolve_low_memory(
                               "auto", config.DEFAULT_MEMORY_BUDGET_GIB, 48.0)}


async def test_settings_get_and_partial_put(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = await client.get("/api/settings")
    assert r.json() == DEFAULT_PUBLIC_SETTINGS
    r = await client.put("/api/settings", json={"theme": "dark"})
    assert r.status_code == 200
    assert r.json() == {**DEFAULT_PUBLIC_SETTINGS, "theme": "dark"}
    assert (await client.get("/api/settings")).json()["theme"] == "dark"  # persisted
    r = await client.put("/api/settings", json={"bogus": 1})  # unknown keys ignored
    assert r.status_code == 200 and "bogus" not in r.json()
    r = await client.put("/api/settings", content=b"nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
    r = await client.put("/api/settings", json=[1])
    assert r.status_code == 400


async def test_settings_api_key_is_write_only(client, app, monkeypatch):
    """The key is never echoed; absent = unchanged, "" = cleared, text = saved."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = await client.get("/api/settings")
    assert "anthropic_api_key" not in r.json() and r.json()["has_api_key"] is False
    r = await client.put("/api/settings", json={"anthropic_api_key": "  sk-ant-test  ",
                                                "assist_provider": "api", "assist_model": " claude-opus-5 "})
    assert r.status_code == 200, r.text
    assert "anthropic_api_key" not in r.json()
    assert r.json()["has_api_key"] is True and r.json()["assist_provider"] == "api"
    assert r.json()["assist_model"] == "claude-opus-5"
    assert app.state.store.get_settings()["anthropic_api_key"] == "sk-ant-test"  # stored stripped
    r = await client.put("/api/settings", json={"theme": "light"})  # key absent → unchanged
    assert r.json()["has_api_key"] is True
    assert app.state.store.get_settings()["anthropic_api_key"] == "sk-ant-test"
    r = await client.put("/api/settings", json={"anthropic_api_key": ""})  # "" → cleared
    assert r.json()["has_api_key"] is False
    assert app.state.store.get_settings()["anthropic_api_key"] == ""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")  # env fallback: usable, but not a *stored* key
    assert (await client.get("/api/settings")).json()["has_api_key"] is False
    assert (await client.get("/api/status")).json()["assist"]["api_key"] is True
    for bad in ({"assist_provider": "openai"}, {"assist_model": "x" * 81}, {"anthropic_api_key": "k" * 201},
                {"assist_model": 3}):
        r = await client.put("/api/settings", json=bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", bad


async def test_settings_and_status_on_a_16_gb_mac(client, api, app, machine_ram):
    """Budget capped to total RAM - 4, a stale 24 GiB clamped, low-memory auto -> on, recorded per job."""
    await client.put("/api/settings", json={"memory_budget_gib": 24})  # saved on the "48 GB" host
    machine_ram(16)
    body = (await client.get("/api/settings")).json()
    assert body["memory_budget_gib"] == 12.0 and body["low_memory"] == "auto"
    assert body["machine_ram_gib"] == 16.0 and body["min_memory_budget_gib"] == 6.0
    assert body["max_memory_budget_gib"] == 12.0
    assert body["low_memory_effective"] is True
    r = await client.put("/api/settings", json={"memory_budget_gib": 12.5})
    assert r.status_code == 400 and "at most 12 GiB" in r.json()["error"]["message"]
    status = (await client.get("/api/status")).json()
    assert status["memory"] == {"machine_ram_gib": 16.0, "min_memory_budget_gib": 6.0,
                                "max_memory_budget_gib": 12.0, "memory_budget_gib": 12.0,
                                "low_memory": "auto", "low_memory_effective": True}
    assert status["engine"]["low_memory"] is None  # cold
    job = await api.wait((await api.create())["id"])
    summary = json.loads((app.state.paths.songs_dir / job["id"] / "summary.json").read_text())
    assert summary["low_memory"] is True and summary["fast_numerics"] is True
    assert (await client.get("/api/status")).json()["engine"]["low_memory"] is True
    # turning it off rebuilds the (fake) pipeline and is recorded on the next job
    engine = app.state.engine
    pipeline = engine.pipeline
    r = await client.put("/api/settings", json={"low_memory": "off"})
    assert r.json()["low_memory"] == "off" and r.json()["low_memory_effective"] is False
    job = await api.wait((await api.create())["id"])
    summary = json.loads((app.state.paths.songs_dir / job["id"] / "summary.json").read_text())
    assert summary["low_memory"] is False and engine.pipeline is not pipeline
    assert (await client.get("/api/status")).json()["engine"]["low_memory"] is False


@pytest.mark.parametrize(("patch", "ok"), [
    ({"memory_budget_gib": 6}, True), ({"memory_budget_gib": 5.99}, False),
    ({"memory_budget_gib": 4}, False),
    ({"memory_budget_gib": 44}, True), ({"memory_budget_gib": 44.01}, False),
    ({"memory_budget_gib": 100}, False), ({"memory_budget_gib": "lots"}, False),
    ({"default_preset": "fast"}, True), ({"default_preset": "ultra"}, False),
    ({"require_ac": True}, True), ({"require_ac": 3}, False),
    ({"fast_numerics": False}, True), ({"fast_numerics": 3}, False),
    ({"low_memory": "on"}, True), ({"low_memory": "off"}, True), ({"low_memory": "maybe"}, False),
    ({"low_memory": True}, False),
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
        body = r.json()
        effective = body.pop("low_memory_effective")  # derived: auto follows the budget
        assert body == {k: v for k, v in {**before, **patch}.items() if k != "low_memory_effective"}
        assert effective == config.resolve_low_memory(body["low_memory"], body["memory_budget_gib"])
    else:
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error"
        assert next(iter(patch)) in r.json()["error"]["message"]
        assert (await client.get("/api/settings")).json() == before  # rejected patch changes nothing


# -- assist ------------------------------------------------------------------------------------


async def test_assist_returns_fields(client, monkeypatch):
    seen = {}

    def fake_assist(settings, *, prompt, page, context):
        seen.update(settings=settings, prompt=prompt, page=page, context=context)
        return assist.AssistResult(fields={"title": "Ok", "style": "pop", "lyrics": "[Verse]\nla"},
                                   notes="if it drags, raise the BPM", provider="cli", model="claude-x",
                                   seconds=1.5)

    monkeypatch.setattr(assist, "assist", fake_assist)
    r = await client.post("/api/assist", json={"prompt": "  a song  ", "page": "cover",
                                               "context": {"style": "old", "junk": 1}})
    assert r.status_code == 200, r.text
    assert r.json() == {"fields": {"title": "Ok", "style": "pop", "lyrics": "[Verse]\nla"},
                        "notes": "if it drags, raise the BPM", "provider": "cli", "model": "claude-x",
                        "seconds": 1.5}
    assert seen["prompt"] == "a song" and seen["page"] == "cover"
    assert seen["context"] == {"style": "old"}  # unknown keys dropped
    assert seen["settings"]["assist_provider"] == "auto"
    r = await client.post("/api/assist", json={"prompt": "x"})  # page defaults to create, context None
    assert r.status_code == 200 and seen["page"] == "create" and seen["context"] is None


async def test_assist_validation(client, monkeypatch):
    monkeypatch.setattr(assist, "assist", lambda *a, **k: pytest.fail("must not run"))
    for body in ({"prompt": ""}, {"prompt": "   "}, {}, {"prompt": "x", "page": "library"},
                 {"prompt": "x" * 4001}, {"prompt": "x", "context": [1]}, {"prompt": "x", "context": "s"},
                 {"prompt": "x", "context": {"title": True}}, {"prompt": "x", "context": {"lyrics": ["a"]}},
                 {"prompt": "x", "context": {"style": {"a": 1}}}, [1]):
        r = await client.post("/api/assist", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", body


async def test_assist_context_is_filtered_and_capped(client, monkeypatch):
    seen = {}

    def fake_assist(settings, *, prompt, page, context):
        seen["context"] = context
        return assist.AssistResult(fields={}, notes=None, provider="cli", model=None, seconds=0.0)

    monkeypatch.setattr(assist, "assist", fake_assist)
    r = await client.post("/api/assist", json={"prompt": "x", "context": {
        "title": "t" * 300, "style": "s", "lyrics": None, "cot": "full", "cfg_scale": 1.5, "seed": 4,
        "junk": "dropped"}})
    assert r.status_code == 200, r.text
    assert seen["context"] == {"title": "t" * 200, "style": "s", "lyrics": None, "cot": "full",
                               "cfg_scale": 1.5}
    r = await client.post("/api/assist", json={"prompt": "x", "context": {"seed": 4}})
    assert r.status_code == 200 and seen["context"] == {}


async def test_assist_unavailable_and_failed(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    await client.put("/api/settings", json={"assist_provider": "off"})
    r = await client.post("/api/assist", json={"prompt": "x"})  # real resolve(): off → 503
    assert r.status_code == 503
    assert r.json()["error"] == {"code": "assist_unavailable", "message": "assist is turned off in Settings"}
    r = await client.post("/api/assist/test")
    assert r.status_code == 503 and r.json()["error"]["code"] == "assist_unavailable"

    def boom(settings, **kw):
        raise assist.AssistError("claude CLI: Not logged in")

    monkeypatch.setattr(assist, "assist", boom)
    r = await client.post("/api/assist", json={"prompt": "x"})
    assert r.status_code == 502
    assert r.json()["error"] == {"code": "assist_failed", "message": "claude CLI: Not logged in"}

    def unavailable(settings, **kw):
        raise assist.AssistUnavailable(["claude CLI not found on PATH", "no API key"])

    monkeypatch.setattr(assist, "assist", unavailable)
    r = await client.post("/api/assist", json={"prompt": "x"})
    assert r.status_code == 503
    assert r.json()["error"]["message"] == "claude CLI not found on PATH; no API key"


async def test_assist_test_endpoint(client, monkeypatch):
    monkeypatch.setattr(assist, "test", lambda settings: assist.AssistResult(
        fields={"title": "ok"}, notes=None, provider="api", model="claude-sonnet-5", seconds=0.4))
    r = await client.post("/api/assist/test")
    assert r.status_code == 200
    assert r.json() == {"fields": {"title": "ok"}, "notes": None, "provider": "api",
                        "model": "claude-sonnet-5", "seconds": 0.4}


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
    assert set(job["params"]) == {"upload_id", "task", "mode", "style", "lyrics", "seed", "title",
                                  "cfg_scale"}
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
                                  "hum_influence", "offset_s", "cfg_scale"}
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


# -- projects / tracks / takes ------------------------------------------------------------------


PROJECT_KEYS = {"id", "name", "description", "created_at", "updated_at", "tracks"}
TRACK_KEYS = {"id", "project_id", "project_name", "name", "position", "chosen_job_id", "created_at", "takes",
              "chosen"}
TAKE_KEYS = {"track_id", "project_id", "track_name", "project_name", "thumb", "stars", "note", "added_at",
             "chosen"}


async def _project(client, name="Soundtrack", tracks=("Main theme", "Credits")) -> tuple[dict, list[dict]]:
    r = await client.post("/api/projects", json={"name": name})
    assert r.status_code == 201, r.text
    project = r.json()["project"]
    made = []
    for track_name in tracks:
        r = await client.post(f"/api/projects/{project['id']}/tracks", json={"name": track_name})
        assert r.status_code == 201, r.text
        made.append(r.json()["track"])
    return project, made


async def test_projects_crud(client):
    r = await client.get("/api/projects")
    assert r.status_code == 200 and r.json() == {"projects": []}
    r = await client.post("/api/projects", json={"name": " Soundtrack ", "description": "film"})
    assert r.status_code == 201
    project = r.json()["project"]
    assert set(project) == PROJECT_KEYS and project["name"] == "Soundtrack" and project["tracks"] == []
    for bad in ({}, {"name": ""}, {"name": "x", "description": "y" * 2001}, []):
        r = await client.post("/api/projects", json=bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", bad
    r = await client.post("/api/projects", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = await client.get("/api/projects")
    listed = r.json()["projects"]
    assert len(listed) == 1
    assert set(listed[0]) == (PROJECT_KEYS - {"tracks"}) | {"track_count", "chosen_count"}
    assert listed[0]["track_count"] == 0 and listed[0]["chosen_count"] == 0
    r = await client.patch(f"/api/projects/{project['id']}", json={"name": "Score"})
    assert r.status_code == 200 and r.json()["project"]["name"] == "Score"
    assert r.json()["project"]["description"] == "film"
    r = await client.patch(f"/api/projects/{project['id']}", json={"description": ""})
    assert r.json()["project"] == {**r.json()["project"], "name": "Score", "description": ""}
    assert (await client.patch(f"/api/projects/{project['id']}", json={"name": ""})).status_code == 400
    assert (await client.patch("/api/projects/nope", json={"name": "x"})).status_code == 404
    r = await client.get(f"/api/projects/{project['id']}")
    assert r.status_code == 200 and r.json()["project"]["name"] == "Score"
    assert (await client.get("/api/projects/nope")).status_code == 404
    r = await client.delete(f"/api/projects/{project['id']}")
    assert r.status_code == 204 and r.content == b""
    assert (await client.delete(f"/api/projects/{project['id']}")).status_code == 404
    assert (await client.get("/api/projects")).json()["projects"] == []


async def test_tracks_create_patch_order_and_delete(client, api):
    project, (a, b) = await _project(client)
    assert set(a) == TRACK_KEYS and a["project_id"] == project["id"] and a["project_name"] == "Soundtrack"
    assert (a["position"], b["position"]) == (0, 1) and a["takes"] == [] and a["chosen"] is None
    assert (await client.post(f"/api/projects/{project['id']}/tracks", json={"name": " "})).status_code == 400
    assert (await client.post("/api/projects/nope/tracks", json={"name": "x"})).status_code == 404
    r = await client.get(f"/api/tracks/{a['id']}")
    assert r.status_code == 200 and set(r.json()) == {"track", "project"}
    assert r.json()["project"] == {"id": project["id"], "name": "Soundtrack"}
    assert (await client.get("/api/tracks/nope")).status_code == 404
    r = await client.patch(f"/api/tracks/{a['id']}", json={"name": "Theme"})
    assert r.status_code == 200 and r.json()["track"]["name"] == "Theme"
    for bad in ({"name": ""}, {"position": -1}, {"position": "x"}):
        assert (await client.patch(f"/api/tracks/{a['id']}", json=bad)).status_code == 400, bad
    r = await client.put(f"/api/projects/{project['id']}/order", json={"track_ids": [b["id"], a["id"]]})
    assert r.status_code == 200
    assert [t["id"] for t in r.json()["project"]["tracks"]] == [b["id"], a["id"]]
    assert [t["position"] for t in r.json()["project"]["tracks"]] == [0, 1]
    for bad in ({"track_ids": [a["id"]]}, {"track_ids": [a["id"], b["id"], "x"]},
                {"track_ids": [a["id"]] * 2}, {}):
        r = await client.put(f"/api/projects/{project['id']}/order", json=bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", bad
    assert (await client.put("/api/projects/nope/order", json={"track_ids": []})).status_code == 404
    r = await client.patch(f"/api/tracks/{a['id']}", json={"position": 0})
    assert r.json()["track"]["position"] == 0
    detail = (await client.get(f"/api/projects/{project['id']}")).json()["project"]
    assert [t["id"] for t in detail["tracks"]] == [a["id"], b["id"]]
    listed = (await client.get("/api/projects")).json()["projects"][0]
    assert listed["track_count"] == 2
    # delete a track with a take: the job survives, untouched
    job = await api.create()
    r = await client.post(f"/api/tracks/{b['id']}/takes", json={"job_ids": [job["id"]]})
    assert r.status_code == 200
    r = await client.delete(f"/api/tracks/{b['id']}")
    assert r.status_code == 204
    assert (await client.delete(f"/api/tracks/{b['id']}")).status_code == 404
    assert (await api.get(job["id"]))["take"] is None
    assert (await client.get("/api/projects")).json()["projects"][0]["track_count"] == 1


async def test_attach_detach_rate_and_choose(client, api, app):
    project, (a, b) = await _project(client)
    app.state.engine.delay = 0.02  # keep j1/j2 unfinished until the "not done" checks below
    j1 = await api.create()
    j2 = await api.create()
    assert j1["take"] is None
    r = await client.post(f"/api/tracks/{a['id']}/takes", json={"job_ids": [j1["id"], "missing"]})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert (await client.post("/api/tracks/nope/takes", json={"job_ids": [j1["id"]]})).status_code == 404
    assert (await client.post(f"/api/tracks/{a['id']}/takes", json={"job_ids": []})).status_code == 400
    r = await client.post(f"/api/tracks/{a['id']}/takes", json={"job_ids": [j1["id"], j2["id"]]})
    assert r.status_code == 200
    track = r.json()["track"]
    assert [j["id"] for j in track["takes"]] == [j1["id"], j2["id"]] and track["chosen"] is None
    assert set(track["takes"][0]) == JOB_KEYS
    take = (await api.get(j1["id"]))["take"]
    assert set(take) == TAKE_KEYS
    assert take["track_id"] == a["id"] and take["project_id"] == project["id"]
    assert take["track_name"] == "Main theme" and take["project_name"] == "Soundtrack"
    assert (take["thumb"], take["stars"], take["note"], take["chosen"]) == (None, None, "", False)
    # library filters
    r = await client.get("/api/jobs", params={"track": a["id"]})
    assert r.json()["total"] == 2 and all(j["take"]["track_id"] == a["id"] for j in r.json()["jobs"])
    assert (await client.get("/api/jobs", params={"project": project["id"]})).json()["total"] == 2
    assert (await client.get("/api/jobs", params={"track": b["id"]})).json()["total"] == 0
    # rating
    r = await client.patch(f"/api/takes/{j1['id']}", json={"thumb": 1, "stars": 4, "note": "keeper"})
    assert r.status_code == 200 and set(r.json()) == {"job"}
    take = r.json()["job"]["take"]
    assert (take["thumb"], take["stars"], take["note"]) == (1, 4, "keeper")
    r = await client.patch(f"/api/takes/{j1['id']}", json={"thumb": -1})
    assert r.json()["job"]["take"]["thumb"] == -1 and r.json()["job"]["take"]["stars"] == 4
    r = await client.patch(f"/api/takes/{j1['id']}", json={"thumb": 0, "stars": None})
    assert r.json()["job"]["take"]["thumb"] is None and r.json()["job"]["take"]["stars"] is None
    for bad in ({"stars": 0}, {"stars": 6}, {"thumb": 2}, {"thumb": True}, {"note": "n" * 4001}, []):
        r = await client.patch(f"/api/takes/{j1['id']}", json=bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", bad
    r = await client.patch("/api/takes/nope", json={"stars": 3})
    assert r.status_code == 404
    unattached = await api.create()
    assert (await client.patch(f"/api/takes/{unattached['id']}", json={"stars": 3})).status_code == 404
    # choosing needs a done take of that track
    r = await client.patch(f"/api/tracks/{a['id']}", json={"chosen_job_id": j2["id"]})  # still queued
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    app.state.engine.delay = 0
    await api.wait(j1["id"])
    await api.wait(j2["id"])
    r = await client.patch(f"/api/tracks/{b['id']}", json={"chosen_job_id": j1["id"]})
    assert r.status_code == 409  # not a take of b
    r = await client.patch(f"/api/tracks/{a['id']}", json={"chosen_job_id": j1["id"]})
    assert r.status_code == 200
    track = r.json()["track"]
    assert track["chosen_job_id"] == j1["id"] and track["chosen"]["id"] == j1["id"]
    assert track["chosen"]["take"]["chosen"] is True
    assert (await api.get(j1["id"]))["take"]["chosen"] is True
    assert (await api.get(j2["id"]))["take"]["chosen"] is False
    listed = (await client.get("/api/projects")).json()["projects"][0]
    assert listed["chosen_count"] == 1
    detail = (await client.get(f"/api/projects/{project['id']}")).json()["project"]
    assert detail["tracks"][0]["chosen"]["id"] == j1["id"] and len(detail["tracks"][0]["takes"]) == 2
    r = await client.patch(f"/api/tracks/{a['id']}", json={"chosen_job_id": None})
    assert r.json()["track"]["chosen"] is None and r.json()["track"]["chosen_job_id"] is None
    await client.patch(f"/api/tracks/{a['id']}", json={"chosen_job_id": j1["id"]})
    # 409 for a job in another track, then move keeps its rating and clears the old choice
    await client.patch(f"/api/takes/{j1['id']}", json={"stars": 5})
    r = await client.post(f"/api/tracks/{b['id']}/takes", json={"job_ids": [j1["id"]]})
    assert r.status_code == 409 and "already a take of Main theme" in r.json()["error"]["message"]
    r = await client.post(f"/api/tracks/{b['id']}/takes", json={"job_ids": [j1["id"]], "move": True})
    assert r.status_code == 200 and [j["id"] for j in r.json()["track"]["takes"]] == [j1["id"]]
    take = (await api.get(j1["id"]))["take"]
    assert take["track_id"] == b["id"] and take["stars"] == 5 and take["chosen"] is False
    a_now = (await client.get(f"/api/tracks/{a['id']}")).json()["track"]
    assert a_now["chosen_job_id"] is None and [j["id"] for j in a_now["takes"]] == [j2["id"]]
    # detach
    r = await client.delete(f"/api/takes/{j1['id']}")
    assert r.status_code == 204
    assert (await client.delete(f"/api/takes/{j1['id']}")).status_code == 404
    assert (await api.get(j1["id"]))["take"] is None
    # deleting a chosen job clears the choice
    await client.patch(f"/api/tracks/{a['id']}", json={"chosen_job_id": j2["id"]})
    assert (await client.delete(f"/api/jobs/{j2['id']}")).status_code == 204
    a_now = (await client.get(f"/api/tracks/{a['id']}")).json()["track"]
    assert a_now["chosen_job_id"] is None and a_now["takes"] == []
    # deleting the project leaves jobs and song dirs alone
    assert (await client.delete(f"/api/projects/{project['id']}")).status_code == 204
    assert (await api.get(j1["id"]))["artifacts"]["audio"] is True


async def test_submit_with_track_id(client, api, app):
    project, (a, _) = await _project(client)
    r = await client.post("/api/jobs", json={"kind": "create", "params": BASE, "track_id": "nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    assert (await client.get("/api/jobs")).json()["total"] == 0
    r = await client.post("/api/jobs", json={"kind": "create", "params": BASE, "track_id": a["id"]})
    assert r.status_code == 201
    job = r.json()["job"]
    assert job["take"]["track_id"] == a["id"] and job["take"]["track_name"] == "Main theme"
    r = await client.post("/api/jobs", json={"kind": "variations", "track_id": a["id"],
                                             "params": {"count": 2, "base": BASE}})
    assert r.status_code == 201
    members = r.json()["jobs"]
    assert all(j["take"]["track_id"] == a["id"] for j in members)
    track = (await client.get(f"/api/tracks/{a['id']}")).json()["track"]
    assert [j["id"] for j in track["takes"]] == [job["id"], *(j["id"] for j in members)]
    # live progress overlays on takes while they run
    app.state.engine.delay = 0.02
    r = await client.post("/api/jobs", json={"kind": "create", "params": BASE, "track_id": a["id"]})
    live = r.json()["job"]
    await api.wait_status(live["id"], "running")
    track = (await client.get(f"/api/tracks/{a['id']}")).json()["track"]
    running = next(j for j in track["takes"] if j["id"] == live["id"])
    assert running["status"] == "running" and running["progress"] is not None
    app.state.engine.delay = 0
    await api.wait(live["id"])
    # regenerate can target the parent's track explicitly; a plain submit is never attached
    parent = await api.wait(job["id"])
    r = await client.post("/api/jobs", json={"kind": "regenerate", "track_id": a["id"],
                                             "params": {"parent_id": parent["id"], "abc": "X:1\nK:C\nC|"}})
    assert r.status_code == 201 and r.json()["job"]["take"]["track_id"] == a["id"]
    plain = await api.create()
    assert plain["take"] is None


async def test_album_zip_flac(client, api, app):
    project, (a, b) = await _project(client, name="My Album: Vol. 1/2")
    r = await client.get(f"/api/projects/{project['id']}/album.zip")
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    assert (await client.get("/api/projects/nope/album.zip")).status_code == 404
    r = await client.get(f"/api/projects/{project['id']}/album.zip", params={"format": "wav"})
    assert r.status_code == 400
    job = await api.create({**BASE, "title": "Opening"})
    await api.wait(job["id"])
    await client.post(f"/api/tracks/{a['id']}/takes", json={"job_ids": [job["id"]]})
    await client.patch(f"/api/tracks/{a['id']}", json={"chosen_job_id": job["id"]})
    r = await client.get(f"/api/projects/{project['id']}/album.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    disposition = r.headers["content-disposition"]  # starlette RFC 5987-encodes names with spaces
    assert disposition.startswith("attachment;") and "My%20Album%20Vol.%201%202.zip" in disposition
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = zf.namelist()
    slug = "My Album Vol. 1 2"
    assert names == [f"{slug}/01 Main theme.flac", f"{slug}/tracklist.json", f"{slug}/tracklist.md"]
    assert zf.read(f"{slug}/01 Main theme.flac")[:4] == b"fLaC"
    assert zf.getinfo(f"{slug}/01 Main theme.flac").compress_type == zipfile.ZIP_STORED
    data = json.loads(zf.read(f"{slug}/tracklist.json"))
    assert set(data) == {"project", "format", "generated_at", "tracks"} and data["format"] == "flac"
    assert data["project"] == {"id": project["id"], "name": "My Album: Vol. 1/2", "description": ""}
    first, second = data["tracks"]
    assert set(first) == {"n", "track_id", "name", "job_id", "title", "file", "seconds", "seed", "preset",
                          "kind", "missing"}
    assert first["n"] == 1 and first["job_id"] == job["id"] and first["title"] == "Opening"
    assert first["file"] == "01 Main theme.flac" and first["missing"] is False and first["kind"] == "create"
    assert first["seed"] == BASE["seed"] and first["preset"] == "quality"
    assert second == {"n": 2, "track_id": b["id"], "name": "Credits", "job_id": None, "title": None,
                      "file": None, "seconds": None, "seed": None, "preset": None, "kind": None,
                      "missing": True}
    md = zf.read(f"{slug}/tracklist.md").decode()
    assert md.startswith("# My Album: Vol. 1/2") and "01 Main theme.flac" in md and "Credits" in md
    assert not list(app.state.paths.data_dir.glob(".album-*"))  # temp file removed after sending
    # a chosen take whose audio is gone counts as missing (409 when it was the only one)
    (app.state.paths.songs_dir / job["id"] / "song" / "audio.flac").unlink()
    r = await client.get(f"/api/projects/{project['id']}/album.zip")
    assert r.status_code == 409


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
async def test_album_zip_mp3(client, api, app):
    project, (a, _) = await _project(client, name="Album")
    job = await api.create()
    await api.wait(job["id"])
    await client.post(f"/api/tracks/{a['id']}/takes", json={"job_ids": [job["id"]]})
    await client.patch(f"/api/tracks/{a['id']}", json={"chosen_job_id": job["id"]})
    r = await client.get(f"/api/projects/{project['id']}/album.zip", params={"format": "mp3"})
    assert r.status_code == 200 and r.headers["content-disposition"] == 'attachment; filename="Album.zip"'
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert "Album/01 Main theme.mp3" in zf.namelist() and len(zf.read("Album/01 Main theme.mp3")) > 1000
    assert json.loads(zf.read("Album/tracklist.json"))["format"] == "mp3"
    assert (app.state.paths.songs_dir / job["id"] / "song" / "audio.mp3").is_file()  # cached beside the flac


async def test_startup_sweeps_stale_album_zips(make_app, home):
    data_dir = home / "data"
    data_dir.mkdir(parents=True)
    stale = data_dir / ".album-abc123.zip"
    stale.write_bytes(b"PK")
    keep = data_dir / "app.db-keep.zip"
    keep.write_bytes(b"PK")
    async with make_app(FakeEngine(delay=0), home=home) as (app, client):
        assert not stale.exists() and keep.exists()
        assert (await client.get("/api/status")).status_code == 200


async def test_album_zip_mp3_503_without_ffmpeg(client, api, monkeypatch):
    project, (a, _) = await _project(client)
    job = await api.create()
    await api.wait(job["id"])
    await client.post(f"/api/tracks/{a['id']}/takes", json={"job_ids": [job["id"]]})
    await client.patch(f"/api/tracks/{a['id']}", json={"chosen_job_id": job["id"]})
    monkeypatch.setattr(audio, "ffmpeg_path", lambda: None)
    r = await client.get(f"/api/projects/{project['id']}/album.zip", params={"format": "mp3"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "engine_unavailable"
    assert (await client.get(f"/api/projects/{project['id']}/album.zip")).status_code == 200  # flac is fine


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


async def test_patch_job_renames_and_clears_the_title(app, client, api):
    job = await api.create({**BASE, "title": "Draft"})
    await api.wait(job["id"])
    job_json = app.state.paths.songs_dir / job["id"] / "job.json"
    r = await client.patch(f"/api/jobs/{job['id']}", json={"title": "  Final   Mix ", "style": "ignored"})
    assert r.status_code == 200 and r.json()["job"]["title"] == "Final Mix"
    assert (await api.get(job["id"]))["params"]["title"] == "Final Mix"
    assert (await api.get(job["id"]))["params"]["style"] == BASE["style"]
    assert json.loads(job_json.read_text())["params"]["title"] == "Final Mix"
    r = await client.patch(f"/api/jobs/{job['id']}", json={})
    assert r.status_code == 200 and r.json()["job"]["title"] == "Final Mix"
    for cleared in ("   ", None):
        await client.patch(f"/api/jobs/{job['id']}", json={"title": "x"})
        r = await client.patch(f"/api/jobs/{job['id']}", json={"title": cleared})
        assert r.status_code == 200 and r.json()["job"]["title"] is None
        assert json.loads(job_json.read_text())["params"]["title"] is None
    for bad in ({"title": "x" * 201}, {"title": 5}, []):
        r = await client.patch(f"/api/jobs/{job['id']}", json=bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "validation_error", bad
    assert (await client.patch("/api/jobs/nope", json={"title": "x"})).status_code == 404
