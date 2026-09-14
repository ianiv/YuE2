"""Worker + EventBus end-to-end with the FakeEngine (no app, no mlx)."""

import asyncio
import json

import pytest

from yue2_studio import config, jobs
from yue2_studio.fake import FakeEngine
from yue2_studio.jobs import JobStore
from yue2_studio.worker import EventBus, Worker, derive_timing, derive_truncated, normalise, stage_key

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
        sub = jobs.submit(self.store, body, upload_lookup=lambda uid: {"filename": "demo.mp3"})
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
    assert done["artifacts"] == {"audio": True, "score": True, "plan": True, "transcription": False}
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


async def test_variations_run_serially_in_seed_order(harness):
    members = harness.submit({"kind": "variations", "params": {"count": 3, "base": BASE}})
    results = [await harness.collect(j.id) for j in members]
    assert [d["status"] for _, d in results] == ["done"] * 3
    order = [req["seed"] for name, req in harness.engine.calls]
    assert order == [5, 6, 7]


async def test_stop_cancels_running_job(tmp_path):
    h = Harness(tmp_path, FakeEngine(delay=0.05))
    h.worker.start(asyncio.get_running_loop())
    job = h.submit({"kind": "create", "params": BASE})
    q = h.bus.subscribe(job.id)
    await asyncio.wait_for(q.get(), 5)
    h.bus.unsubscribe(job.id, q)
    h.worker.stop(timeout=5)
    assert not h.worker.alive
    assert h.store.get(job.id).status == "cancelled"
    assert h.engine.state == "cold"  # unloaded on the worker thread
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


@pytest.mark.parametrize(("label", "key"), [
    ("Verifying model files", "load"), ("Loading 8bit AR model", "load"),
    ("Loading BF16 acoustic conditioning", "load"), ("Loading acoustic model", "load"),
    ("Loading MLX audio decoder", "load"), ("Using provided score", "plan"),
    ("Planning score", "plan"), ("Generating song", "semantic"), ("Synthesizing audio", "synthesize"),
    ("Decoding audio", "decode"), ("Transcribing audio", "transcribe"), ("Saving artifacts", "save"),
])
def test_stage_keys(label, key):
    assert stage_key(label) == key


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
