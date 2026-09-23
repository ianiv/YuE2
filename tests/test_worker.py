"""Worker + EventBus end-to-end with the FakeEngine (no app, no mlx)."""

import asyncio
import json

import pytest

from yue2_studio import config, jobs
from yue2_studio.fake import FakeEngine
from yue2_studio.jobs import JobStore
from yue2_studio.worker import (
    EVENT_FIELDS,
    EventBus,
    Worker,
    derive_timing,
    derive_truncated,
    normalise,
    stage_key,
    status_event,
)

BASE = {"style": "dreamy indie pop", "lyrics": "[verse]\nla la", "seed": 5}


class Harness:
    def __init__(self, tmp_path, engine):
        self.paths = config.paths_for(tmp_path)
        self.paths.ensure_dirs()
        self.store = JobStore(self.paths.db_path, self.paths.songs_dir)
        self.bus = EventBus()
        self.engine = engine
        self.worker = Worker(self.store, engine, self.paths, self.bus, fake=True)

    def submit(self, body):
        sub = jobs.submit(self.store, body, upload_lookup=lambda uid: {"filename": "demo.mp3"},
                          hum_adapter_lookup=lambda name: name == "hum_v1")
        for job in sub.jobs:
            self.worker.submit(job.id)
        return sub.jobs[0] if sub.group is None else sub.jobs

    async def collect(self, job_id, timeout=10.0):
        q = self.bus.subscribe(job_id)
        events = []
        try:
            while True:
                kind, payload = await asyncio.wait_for(q.get(), timeout)
                if kind == "done":
                    return events, payload
                events.append(payload)
        finally:
            self.bus.unsubscribe(job_id, q)


@pytest.fixture
async def harness(tmp_path):
    h = Harness(tmp_path, FakeEngine(delay=0))
    h.worker.start(asyncio.get_running_loop())
    yield h
    await asyncio.to_thread(h.worker.stop)
    h.store.close()


async def test_create_job_runs_to_done_with_normalised_events(harness):
    job = harness.submit({"kind": "create", "preset": "fast", "params": BASE})
    q = harness.bus.subscribe(job.id)  # subscribe before it runs
    events, done = await harness.collect(job.id)
    harness.bus.unsubscribe(job.id, q)
    assert done["status"] == "done" and done["id"] == job.id
    assert done["artifacts"] == {"audio": True, "score": True, "plan": True, "transcription": False,
                                 "hum": False}
    assert done["progress"]["type"] == "status" and done["progress"]["status"] == "done"
    assert done["started_at"] and done["finished_at"] and done["error"] is None and done["position"] is None
    timing = done["timing"]
    assert set(timing) == {"plan", "semantic", "synthesize", "decode", "transcribe", "e2e", "abc_tps",
                           "semantic_tps", "audio_seconds"}
    assert timing["audio_seconds"] == 1.0 and timing["transcribe"] is None and timing["semantic_tps"] > 0
    assert done["truncated"] is None

    types = [e["type"] for e in events]
    assert types[0] == "status" and events[0]["status"] == "running"
    assert any(e["type"] == "status" and e["stage"] == "load" for e in events)  # engine load announced
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    for key in ("load", "plan", "semantic", "synthesize", "decode", "save"):
        assert key in stages
    stage_events = [e for e in events if e["type"] == "stage"]
    assert all(e["status"] in {"running", "complete"} for e in stage_events)  # completed -> complete
    assert all(e["label"] for e in stage_events)
    abc = [e for e in events if e["type"] == "abc"]
    assert [e["partial"] for e in abc] == [True, True, True, False]
    assert abc[-1]["text"].startswith("X:1")
    assert all(e["job_id"] == job.id and e["ts"].endswith("Z") for e in events)
    assert all(set(e) == set(events[0]) for e in events)  # every ProgressEvent field present
    logs = [e for e in events if e["type"] == "log"]
    assert logs and logs[0]["message"].startswith("Completed")
    assert events[-1]["type"] == "status" and events[-1]["status"] == "done"
    # the engine got a clean SongRequest dict
    name, request = harness.engine.calls[-1]
    assert name == "create_song"
    assert set(request) == {"style", "lyrics", "cot", "seed", "id"} and request["id"] == job.id
    song_dir = harness.paths.songs_dir / job.id
    assert (song_dir / "song" / "audio.flac").is_file() and (song_dir / "job.json").is_file()
    assert json.loads((song_dir / "summary.json").read_text())["status"] == "complete"
    # progress was persisted for the terminal job
    assert harness.store.get(job.id).progress["status"] == "done"
    assert harness.worker.engine_status()["state"] == "ready"
    assert harness.worker.engine_status()["precision"] == "8bit"
    assert harness.worker.engine_status()["memory_gib"] > 0


async def test_cancel_queued_job_marks_row_without_running(harness):
    harness.engine.delay = 0.05
    first = harness.submit({"kind": "create", "params": BASE})
    second = harness.submit({"kind": "create", "params": BASE})
    assert harness.worker.cancel(second.id) == "cancelled"
    row = harness.store.get(second.id)
    assert row.status == "cancelled" and row.started_at is None and row.progress["status"] == "cancelled"
    _, done = await harness.collect(first.id)
    assert done["status"] == "done"
    await asyncio.sleep(0.05)
    assert all(req["id"] != second.id for _, req in harness.engine.calls)
    assert harness.worker.cancel(second.id) is None  # terminal


async def test_cancel_running_job(harness):
    harness.engine.delay = 0.02
    job = harness.submit({"kind": "create", "params": BASE})
    q = harness.bus.subscribe(job.id)
    try:
        while True:  # wait until the fake engine is mid-job
            kind, payload = await asyncio.wait_for(q.get(), 5)
            if kind == "progress" and payload["type"] == "stage" and payload["stage"] == "semantic":
                break
        assert harness.worker.cancel(job.id) == "requested"
        assert harness.store.get(job.id).status == "running"
        while True:
            kind, payload = await asyncio.wait_for(q.get(), 5)
            if kind == "done":
                break
            last = payload
    finally:
        harness.bus.unsubscribe(job.id, q)
    assert payload["status"] == "cancelled" and last["type"] == "status" and last["status"] == "cancelled"
    assert harness.store.get(job.id).status == "cancelled"
    assert not (harness.paths.songs_dir / job.id / "song" / "audio.flac").exists()
    assert harness.engine.state == "ready"  # cancellation keeps the pipeline
    # the worker is free again
    nxt = harness.submit({"kind": "create", "params": BASE})
    _, done = await harness.collect(nxt.id)
    assert done["status"] == "done"


async def test_failure_marks_failed_and_worker_survives(tmp_path):
    h = Harness(tmp_path, FakeEngine(delay=0, fail=True))
    h.worker.start(asyncio.get_running_loop())
    try:
        job = h.submit({"kind": "create", "params": BASE})
        events, done = await h.collect(job.id)
        assert done["status"] == "failed" and "RuntimeError" in done["error"]
        assert events[-1]["type"] == "status" and events[-1]["status"] == "failed"
        assert done["timing"] is None and done["artifacts"]["audio"] is False
        assert h.engine.state == "cold"  # a failing pipeline is discarded
        h.engine.fail = False
        again = h.submit({"kind": "create", "params": BASE})
        _, done2 = await h.collect(again.id)
        assert done2["status"] == "done" and h.engine.state == "ready"
    finally:
        h.worker.stop()
        h.store.close()


async def test_regenerate_uses_parent_abc(harness):
    parent = harness.submit({"kind": "create", "preset": "fast", "params": {**BASE, "cot": "off"}})
    _, done = await harness.collect(parent.id)
    assert done["status"] == "done"
    regen = harness.submit({"kind": "regenerate",
                            "params": {"parent_id": parent.id, "abc": "X:1\nT:Edited\nK:C\nGABc|"}})
    events, done = await harness.collect(regen.id)
    assert done["status"] == "done" and done["kind"] == "regenerate" and done["parent_id"] == parent.id
    name, request = harness.engine.calls[-1]
    assert name == "create_song" and request["cot"] == "melody"
    assert request["abc"] == "X:1\nT:Edited\nK:C\nGABc|"
    plan_stage = [e for e in events if e["type"] == "stage" and e["stage"] == "plan"]
    assert plan_stage and plan_stage[0]["label"] == "Using provided score"
    saved = (harness.paths.songs_dir / regen.id / "song" / "score.abc").read_text()
    assert saved == "X:1\nT:Edited\nK:C\nGABc|"
    assert done["timing"]["abc_tps"] is None


async def test_hum_job(harness):
    upload = harness.paths.uploads_dir / "u2.webm"
    upload.write_bytes(b"not really audio")
    job = harness.submit({"kind": "hum", "params": {"upload_id": "u2", "style": "lo-fi", "lyrics": "la",
                                                    "adapter": "hum_v1", "hum_influence": 1.2}})
    events, done = await harness.collect(job.id)
    assert done["status"] == "done"
    assert done["artifacts"] == {"audio": True, "score": True, "plan": True, "transcription": True,
                                 "hum": True}
    assert done["timing"]["transcribe"] is not None
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages.index("transcribe") < stages.index("hum") < stages.index("plan")
    labels = {e["label"] for e in events if e["type"] == "stage"}
    assert {"Analysing hum", "Encoding hum", "Transcribing audio", "Planning score"} <= labels
    name, request = harness.engine.calls[-1]
    assert name == "hum_song" and request["cot"] == "melody" and "abc" not in request
    partial = [e["text"] for e in events if e["type"] == "abc" and e["partial"]]
    assert partial and all(t.startswith("X:1\nT:Hum\n") for t in partial)  # the hum's open score leads
    summary = json.loads((harness.paths.songs_dir / job.id / "summary.json").read_text())
    assert summary["hum"]["melody"] == "continue" and summary["hum"]["adapter"] == "hum_v1"
    assert (harness.paths.songs_dir / job.id / "hum" / "hum.abc").is_file()
    assert (harness.paths.songs_dir / job.id / "hum.json").is_file()


async def test_cover_job(harness):
    upload = harness.paths.uploads_dir / "u1.mp3"
    upload.write_bytes(b"not really audio")
    job = harness.submit({"kind": "cover", "params": {"upload_id": "u1", "style": "jazz", "lyrics": "la"}})
    events, done = await harness.collect(job.id)
    assert done["status"] == "done" and done["artifacts"]["transcription"] is True
    assert done["timing"]["transcribe"] is not None
    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages.index("transcribe") < stages.index("plan")
    name, request = harness.engine.calls[-1]
    assert name == "cover_song" and request["cot"] == "melody" and "abc" not in request


async def test_cover_continue_job_streams_the_open_score_and_passes_the_clip(harness):
    (harness.paths.uploads_dir / "u3.mp3").write_bytes(b"not really audio")
    job = harness.submit({"kind": "cover", "params": {"upload_id": "u3", "style": "jazz", "lyrics": "la",
                                                      "mode": "continue", "task": "melody-vocal",
                                                      "clip_start_s": 12.5, "clip_end_s": 40}})
    events, done = await harness.collect(job.id)
    assert done["status"] == "done" and done["params"]["mode"] == "continue"
    name, request = harness.engine.calls[-1]
    assert name == "cover_song" and request["cot"] == "melody" and "abc" not in request
    partial = [e["text"] for e in events if e["type"] == "abc" and e["partial"]]
    assert partial and all(t.startswith("X:1\nT:Hum\n") for t in partial)  # the source's open score leads
    summary = json.loads((harness.paths.songs_dir / job.id / "summary.json").read_text())
    assert summary["cover"] == {"mode": "continue", "clip": {"start_s": 12.5, "end_s": 40.0},
                                "open_abc": "source/open.abc"}
    assert (harness.paths.songs_dir / job.id / "source" / "open.abc").is_file()


async def test_variations_run_serially_in_seed_order(harness):
    members = harness.submit({"kind": "variations", "params": {"count": 3, "base": BASE}})
    results = [await harness.collect(j.id) for j in members]
    assert [d["status"] for _, d in results] == ["done"] * 3
    order = [req["seed"] for name, req in harness.engine.calls]
    assert order == [5, 6, 7]


async def test_stop_cancels_only_running_job_and_restart_requeues(tmp_path):
    h = Harness(tmp_path, FakeEngine(delay=0.05))
    h.worker.start(asyncio.get_running_loop())
    running = h.submit({"kind": "create", "params": BASE})
    queued = [h.submit({"kind": "create", "params": BASE}) for _ in range(3)]
    q = h.bus.subscribe(running.id)
    await asyncio.wait_for(q.get(), 5)  # the first job is running
    h.bus.unsubscribe(running.id, q)
    await asyncio.to_thread(h.worker.stop, 5)
    assert not h.worker.alive
    assert h.store.get(running.id).status == "cancelled"
    for job in queued:  # untouched: still queued, never started, no song dir
        row = h.store.get(job.id)
        assert row.status == "queued" and row.started_at is None
        assert not (h.paths.songs_dir / job.id).exists()
    assert h.store.queued_ids() == [j.id for j in queued]
    assert h.engine.state == "cold"  # unloaded on the worker thread
    # a fresh worker (server restart) re-enqueues the queued rows and runs them
    h.engine.delay = 0
    h.worker = Worker(h.store, h.engine, h.paths, h.bus, fake=True)
    queues = {job.id: h.bus.subscribe(job.id) for job in queued}  # before start: no done can be missed
    h.worker.start(asyncio.get_running_loop())
    try:
        for job in queued:
            while True:
                kind, payload = await asyncio.wait_for(queues[job.id].get(), 10)
                if kind == "done":
                    break
            assert payload["status"] == "done"
    finally:
        for job in queued:
            h.bus.unsubscribe(job.id, queues[job.id])
        await asyncio.to_thread(h.worker.stop)
        h.store.close()


# -- pure functions ---------------------------------------------------------------------------


def test_normalise_raw_stage_event():
    raw = {"type": "stage", "stage": "Synthesizing audio", "completed": 3, "total": 8, "unit": "steps",
           "status": "completed", "seconds": 2.5, "ts": 1_757_800_000.123}
    event = normalise(raw, "abc123")
    assert event == {
        "type": "stage", "job_id": "abc123", "ts": "2025-09-13T21:46:40.123Z", "stage": "synthesize",
        "label": "Synthesizing audio", "completed": 3, "total": 8, "unit": "steps", "status": "complete",
        "phase": None, "tokens": None, "tps": None, "seconds": 2.5, "text": None, "partial": None,
        "message": None,
    }
    abc = normalise({"type": "abc", "phase": "abc", "text": "X:1", "tokens": 4, "status": "partial",
                     "ts": 1.0}, "j")
    assert abc["partial"] is True and abc["text"] == "X:1" and abc["stage"] is None
    final = normalise({"type": "abc", "phase": "abc", "text": "X:1\nK:C", "tokens": 9,
                       "status": "final"}, "j")
    assert final["partial"] is False
    log = normalise({"type": "log", "text": "Completed 1.0s", "ts": 1.0}, "j")
    assert log["message"] == "Completed 1.0s" and log["text"] is None
    token = normalise({"type": "token", "phase": "semantic", "tokens": 12, "tps": 100.5, "seconds": 0.1}, "j")
    assert token["tokens"] == 12 and token["tps"] == 100.5 and token["phase"] == "semantic"


# Every stage label the engine (mlx-Yue) emits, per docs/API.md §6, with its HTTP stage key and unit.
UPSTREAM_STAGES = [
    ("Analysing hum", "hum", None),
    ("Encoding hum", "hum", "chunks"),
    ("Verifying model files", "load", None),
    ("Loading bf16 AR model", "load", None),
    ("Loading 8bit AR model", "load", None),
    ("Loading 4bit AR model", "load", None),
    ("Loading BF16 acoustic conditioning", "load", None),
    ("Loading acoustic model", "load", None),
    ("Loading MLX audio decoder", "load", None),
    ("Using provided score", "plan", None),
    ("Planning score", "plan", "tokens"),
    ("Generating song", "semantic", "tokens"),
    ("Synthesizing audio", "synthesize", "steps"),
    ("Decoding audio", "decode", "chunks"),
    ("Transcribing audio", "transcribe", "windows"),
    ("Saving artifacts", "save", None),  # emitted by the worker itself
]


@pytest.mark.parametrize(("label", "key", "unit"), UPSTREAM_STAGES)
@pytest.mark.parametrize(("raw_status", "http_status"), [
    ("running", "running"), ("completed", "complete"), ("failed", "failed"), ("cancelled", "cancelled"),
    ("truncated", "truncated"),
])
def test_normalise_every_upstream_stage_label(label, key, unit, raw_status, http_status):
    raw = {"type": "stage", "stage": label, "completed": 3, "total": 8 if unit else None, "unit": unit,
           "status": raw_status, "seconds": 1.25, "ts": 1_757_800_000.5}
    if unit == "tokens":
        raw["tps"] = 99.5
    event = normalise(raw, "job1")
    assert stage_key(label) == key
    assert event["stage"] == key and event["label"] == label and event["unit"] == unit
    assert event["status"] == http_status and event["type"] == "stage" and event["job_id"] == "job1"
    assert event["completed"] == 3 and event["total"] == (8 if unit else None) and event["seconds"] == 1.25
    assert event["tps"] == (99.5 if unit == "tokens" else None)
    assert event["ts"] == "2025-09-13T21:46:40.500Z"
    assert set(event) == set(EVENT_FIELDS)
    assert all(event[k] is None for k in ("phase", "tokens", "text", "partial", "message"))


def test_stage_key_fallbacks():
    assert stage_key("Loading something new") == "load"
    assert stage_key("  planning score ") == "plan"
    assert stage_key("Transcription windows") == "transcribe"
    assert stage_key("Mystery step") == "mystery"
    assert stage_key(None) is None and stage_key("") is None


def test_normalise_edge_cases():
    unknown = normalise({"type": "weird", "x": 1, "ts": 2.0}, "j")
    assert unknown["type"] == "log" and json.loads(unknown["message"]) == {"type": "weird", "x": 1, "ts": 2.0}
    no_ts = normalise({"type": "log", "text": "hi"}, "j")
    assert no_ts["ts"].endswith("Z") and len(no_ts["ts"]) == 24
    iso_ts = normalise({"type": "log", "text": "hi", "ts": "2026-01-01T00:00:00.000Z"}, "j")
    assert iso_ts["ts"] == "2026-01-01T00:00:00.000Z"
    status = status_event("j", "running", "loading models (bf16)…", stage="load")
    assert (status["type"], status["stage"], status["status"], status["message"]) == \
        ("status", "load", "running", "loading models (bf16)…")
    assert normalise({"type": "abc", "text": "X:1", "tokens": 1}, "j")["partial"] is None  # no status
    assert normalise({"type": "log", "message": "m"}, "j")["message"] == "m"


def test_derive_timing_and_truncated():
    summary = {"seconds": 184.7, "truncated": {"abc": False, "semantic": True},
               "timing": {"abc": {"seconds": 15.8, "output_tokens": 2072},
                          "semantic": {"seconds": 35.9, "output_tokens": 4618},
                          "nar_seconds": 27.2, "vae_seconds": 3.4, "e2e_seconds": 82.4,
                          "transcription_seconds": 4.1}}
    assert derive_timing(summary) == {"plan": 15.8, "semantic": 35.9, "synthesize": 27.2, "decode": 3.4,
                                      "transcribe": 4.1, "e2e": 82.4, "abc_tps": 131.14,
                                      "semantic_tps": 128.64, "audio_seconds": 184.7}
    assert derive_truncated(summary) == {"phase": "semantic", "reason": "generation limit reached"}
    assert derive_truncated({"truncated": {"abc": False, "semantic": False}}) is None
    supplied = {"timing": {"abc": {"seconds": 0.0, "output_tokens": 0, "external_prefix_tokens": 300}}}
    assert derive_timing(supplied)["abc_tps"] is None and derive_timing(supplied)["plan"] == 0.0


def test_derive_timing_from_fake_engine_summaries(tmp_path):
    engine = FakeEngine(delay=0)
    options = config.resolve_preset("fast")
    plain = engine.create_song({"style": "s", "lyrics": "l", "cot": "full", "seed": 1}, tmp_path / "a",
                               options=options)
    timing = derive_timing(plain)
    assert timing["transcribe"] is None and timing["audio_seconds"] == 1.0
    assert timing["abc_tps"] is None or timing["abc_tps"] > 0  # None when the fake planned in ~0 s
    assert timing["plan"] >= 0 and timing["e2e"] >= timing["synthesize"]
    src = tmp_path / "src.mp3"
    src.write_bytes(b"x")
    cover = engine.cover_song(src, tmp_path / "b", task="melody-full", options=options,
                              request={"style": "s", "lyrics": "l", "cot": "melody", "seed": 1})
    assert "transcription_seconds" in cover["timing"] and cover["transcription"]["task"] == "melody-full"
    timing = derive_timing(cover)
    assert timing["transcribe"] is not None and timing["transcribe"] >= 0
    assert timing["abc_tps"] is None and timing["plan"] == 0.0  # transcribed score is a supplied score
    assert derive_truncated(cover) is None
    assert derive_timing({}) == dict.fromkeys(("plan", "semantic", "synthesize", "decode", "transcribe",
                                              "e2e", "abc_tps", "semantic_tps", "audio_seconds"))
    assert derive_truncated({"truncated": {"abc": True, "semantic": True}})["phase"] == "abc"


async def test_precision_switch_announces_load_and_rebuilds(harness):
    fast = harness.submit({"kind": "create", "preset": "fast", "params": BASE})
    events, _ = await harness.collect(fast.id)
    assert sum(1 for e in events if e["type"] == "status" and e["stage"] == "load") == 1
    assert harness.engine.precision == "8bit" and harness.engine.ensure_calls >= 1
    calls = harness.engine.ensure_calls
    again = harness.submit({"kind": "create", "preset": "fast", "params": BASE})
    events, _ = await harness.collect(again.id)
    assert not any(e["type"] == "status" and e["stage"] == "load" for e in events)  # warm, same precision
    assert harness.engine.ensure_calls > calls
    quality = harness.submit({"kind": "create", "preset": "quality", "params": BASE})
    events, done = await harness.collect(quality.id)
    load = [e for e in events if e["type"] == "status" and e["stage"] == "load"]
    assert len(load) == 1 and "bf16" in load[0]["message"]
    assert done["precision"] == "bf16" and harness.engine.precision == "bf16"
    assert harness.worker.engine_status()["precision"] == "bf16"
    custom = harness.submit({"kind": "create", "preset": "custom", "precision": "bf16", "ode_steps": 12,
                             "params": BASE})
    events, done = await harness.collect(custom.id)
    assert not any(e["type"] == "status" and e["stage"] == "load" for e in events)  # ode_steps is per job
    synth = [e for e in events if e["type"] == "stage" and e["stage"] == "synthesize"]
    assert synth[-1]["total"] == 12 and done["ode_steps"] == 12


async def test_start_marks_stale_running_rows_failed(tmp_path):
    h = Harness(tmp_path, FakeEngine(delay=0))
    stale = h.submit({"kind": "create", "params": BASE})  # worker not started: row stays queued
    fresh = h.submit({"kind": "create", "params": BASE})
    h.store.update_status(stale.id, "running")  # as if the server died mid-job
    q = h.bus.subscribe(fresh.id)
    h.worker.start(asyncio.get_running_loop())
    try:
        row = h.store.get(stale.id)
        assert row.status == "failed" and "server restarted" in row.error and row.finished_at
        while True:
            kind, payload = await asyncio.wait_for(q.get(), 10)
            if kind == "done":
                break
        assert payload["status"] == "done"
        assert all(req["id"] != stale.id for _, req in h.engine.calls)
    finally:
        h.bus.unsubscribe(fresh.id, q)
        await asyncio.to_thread(h.worker.stop)
        h.store.close()


async def test_event_bus_fanout_and_cache():
    bus = EventBus(asyncio.get_running_loop())
    a, b = bus.subscribe("j"), bus.subscribe("j")
    other = bus.subscribe("k")
    bus.publish("j", {"n": 1})
    assert bus.last("j") == {"n": 1} and bus.last("k") is None
    assert await asyncio.wait_for(a.get(), 1) == ("progress", {"n": 1})
    assert await asyncio.wait_for(b.get(), 1) == ("progress", {"n": 1})
    bus.unsubscribe("j", b)
    bus.publish("j", {"n": 2})
    assert bus.last("j") == {"n": 2}
    bus.done("j", {"id": "j"})
    assert await asyncio.wait_for(a.get(), 1) == ("progress", {"n": 2})
    assert await asyncio.wait_for(a.get(), 1) == ("done", {"id": "j"})
    assert bus.last("j") is None and b.empty() and other.empty()
    bus.publish("j", {"n": 3})  # no subscribers left: cached only
    assert bus.last("j") == {"n": 3} and a.empty()
    bus.forget("j")
    assert bus.last("j") is None
    bus.bind(None)
    bus.publish("k", {"n": 4})  # unbound bus drops fan-out but still caches
    assert other.empty() and bus.last("k") == {"n": 4}


def test_worker_cancel_unknown_job_raises(tmp_path):
    h = Harness(tmp_path, FakeEngine(delay=0))
    with pytest.raises(jobs.NotFound):
        h.worker.cancel("nope")
    assert h.worker.engine_status() == {"state": "cold", "precision": None, "memory_gib": None,
                                        "current_job_id": None, "loras": []}
    h.store.close()
