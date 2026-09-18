"""JobStore, validation and submission expansion (no engine, no app)."""

import sqlite3
import time

import pytest

from yue2_studio import config, jobs
from yue2_studio.jobs import JobStore, NotFound, ValidationFailure

BASE = {"style": "dreamy indie pop", "lyrics": "[verse]\nla la\n[chorus]\nda da"}


@pytest.fixture
def store(tmp_path):
    s = JobStore(tmp_path / "data" / "app.db", tmp_path / "data" / "songs")
    yield s
    s.close()


def test_create_job_fills_defaults_and_api_shape(store):
    sub = jobs.submit(store, {"kind": "create", "params": BASE})
    job = sub.jobs[0]
    assert len(job.id) == 32 and sub.group is None
    assert job.kind == "create" and job.status == "queued"
    assert job.preset == "quality" and job.precision == "bf16" and job.ode_steps == 32
    assert 0 <= job.seed <= jobs.MAX_SEED
    assert job.params == {**BASE, "cot": "full", "seed": job.seed, "cfg_scale": None, "abc": None,
                          "title": None}
    api = job.to_api()
    expected_keys = {"id", "kind", "status", "group_id", "parent_id", "preset", "precision", "ode_steps",
                     "loras", "seed", "params", "title", "created_at", "started_at", "finished_at", "error",
                     "timing", "truncated", "progress", "artifacts", "position", "seq", "take"}
    assert set(api) == expected_keys
    assert isinstance(api["seq"], int) and api["seq"] >= 1
    assert api["position"] == 0 and api["title"] is None and api["timing"] is None
    assert api["artifacts"] == {"audio": False, "score": False, "plan": False, "transcription": False,
                                "hum": False}
    assert api["created_at"].endswith("Z") and len(api["created_at"]) == 24


def test_create_preset_and_custom(store):
    fast = jobs.submit(store, {"kind": "create", "preset": "fast", "params": BASE}).jobs[0]
    assert (fast.preset, fast.precision, fast.ode_steps) == ("fast", "8bit", 8)
    custom = jobs.submit(store, {"kind": "create", "preset": "custom", "precision": "4bit", "ode_steps": 12,
                                 "params": BASE}).jobs[0]
    assert (custom.preset, custom.precision, custom.ode_steps) == ("custom", "4bit", 12)
    with pytest.raises(ValidationFailure):
        jobs.submit(store, {"kind": "create", "preset": "custom", "params": BASE})
    with pytest.raises(ValidationFailure):
        jobs.submit(store, {"kind": "create", "preset": "custom", "precision": "8bit", "ode_steps": 999,
                            "params": BASE})


def test_default_preset_comes_from_settings(store):
    store.update_settings({"default_preset": "fast"})
    job = jobs.submit(store, {"kind": "create", "params": BASE}).jobs[0]
    assert job.preset == "fast"


@pytest.mark.parametrize("bad", [
    {"kind": "create", "params": {"style": "", "lyrics": "x"}},
    {"kind": "create", "params": {"style": "x"}},
    {"kind": "create", "params": {**BASE, "cot": "weird"}},
    {"kind": "create", "params": {**BASE, "cot": "off", "abc": "X:1\nK:C\nC|"}},
    {"kind": "create", "params": {**BASE, "seed": -1}},
    {"kind": "create", "params": {**BASE, "cfg_scale": 99}},
    {"kind": "nope", "params": BASE},
    {"kind": "create"},
    {"kind": "create", "preset": "ultra", "params": BASE},
    {"kind": "variations", "params": {"count": 1, "base": BASE}},
    {"kind": "variations", "params": {"count": 17, "base": BASE}},
    {"kind": "regenerate", "params": {"parent_id": "x", "abc": ""}},
])
def test_validation_errors(store, bad):
    with pytest.raises(ValidationFailure):
        jobs.submit(store, bad)


def test_unknown_fields_ignored_and_abc_with_melody_ok(store):
    body = {"kind": "create", "bogus": 1,
            "params": {**BASE, "cot": "melody", "abc": "X:1\nK:C\nC|", "extra": True}}
    job = jobs.submit(store, body).jobs[0]
    assert job.params["abc"] == "X:1\nK:C\nC|" and "extra" not in job.params


def test_variations_sequential_seeds_and_group(store):
    sub = jobs.submit(store, {"kind": "variations", "preset": "fast",
                              "params": {"count": 3, "base": {**BASE, "seed": 100}}})
    assert sub.group is not None
    assert [j.seed for j in sub.jobs] == [100, 101, 102]
    assert all(j.kind == "create" and j.group_id == sub.group["id"] for j in sub.jobs)
    assert sub.group["job_ids"] == [j.id for j in sub.jobs]
    assert sub.group["label"] == "dreamy indie pop ×3"
    assert [j.position for j in sub.jobs] == [0, 1, 2]


def test_variations_random_seeds_and_label(store):
    sub = jobs.submit(store, {"kind": "variations",
                              "params": {"count": 4, "base": {**BASE, "title": "Song"}, "random_seeds": True,
                                         "label": "my batch"}})
    seeds = [j.seed for j in sub.jobs]
    assert len(seeds) == 4 and all(0 <= s <= jobs.MAX_SEED for s in seeds)
    assert sub.group["label"] == "my batch"
    assert store.get_group(sub.group["id"])["job_ids"] == [j.id for j in sub.jobs]  # submit order, not seed
    assert [j.seq for j in sub.jobs] == sorted(j.seq for j in sub.jobs)


def test_regenerate_inherits_from_parent(store):
    parent_body = {"kind": "create", "preset": "fast",
                   "params": {**BASE, "cot": "off", "seed": 7, "title": "Orig", "cfg_scale": 1.5}}
    parent = jobs.submit(store, parent_body).jobs[0]
    with pytest.raises(NotFound):
        jobs.submit(store, {"kind": "regenerate", "params": {"parent_id": "nope", "abc": "X:1\nK:C\nC|"}})
    job = jobs.submit(store, {"kind": "regenerate", "params": {"parent_id": parent.id, "abc": "X:1\nK:C\nC|",
                                                               "style": "louder"}}).jobs[0]
    assert job.kind == "regenerate" and job.parent_id == parent.id
    assert job.preset == "fast" and job.precision == "8bit" and job.ode_steps == 8
    assert job.seed == 7
    assert job.params["cot"] == "melody"  # parent had cot=off -> melody
    assert job.params["style"] == "louder" and job.params["lyrics"] == BASE["lyrics"]
    assert job.params["title"] == "Orig" and job.params["abc"] == "X:1\nK:C\nC|"
    assert job.params["cfg_scale"] == 1.5
    request = jobs.engine_request(job)
    assert request == {"style": "louder", "lyrics": BASE["lyrics"], "cot": "melody", "seed": 7,
                       "id": job.id, "abc": "X:1\nK:C\nC|", "cfg_scale": 1.5}


def test_cover_requires_upload_and_defaults_title(store):
    body = {"kind": "cover", "params": {"upload_id": "u1", "style": "jazz", "lyrics": "la"}}
    with pytest.raises(NotFound):
        jobs.submit(store, body, upload_lookup=lambda uid: None)
    job = jobs.submit(store, body, upload_lookup=lambda uid: {"filename": "demo song.mp3"}).jobs[0]
    assert job.kind == "cover" and job.params["task"] == "melody-full" and job.params["title"] == "demo song"
    assert jobs.engine_request(job) == {"style": "jazz", "lyrics": "la", "cot": "melody", "seed": job.seed,
                                        "id": job.id}
    full = jobs.submit(store, {"kind": "cover", "params": {**body["params"], "task": "full"}},
                       upload_lookup=lambda uid: {"filename": "x.wav"}).jobs[0]
    assert jobs.engine_request(full)["cot"] == "full"


def test_hum_submit_defaults_validation_and_engine_request(store):
    params = {"upload_id": "u1", "style": "lo-fi", "lyrics": "[verse]\nla"}
    body = {"kind": "hum", "params": params}
    with pytest.raises(NotFound):
        jobs.submit(store, body, upload_lookup=lambda uid: None)
    job = jobs.submit(store, body, upload_lookup=lambda uid: {"filename": "my hum.webm"}).jobs[0]
    assert job.kind == "hum" and job.title == "my hum"
    assert set(job.params) == {"upload_id", "style", "lyrics", "seed", "title", "melody", "adapter",
                               "hum_influence", "offset_s"}
    assert (job.params["melody"], job.params["adapter"], job.params["hum_influence"],
            job.params["offset_s"]) == ("continue", None, 1.0, 0.0)
    assert jobs.engine_request(job) == {"style": "lo-fi", "lyrics": "[verse]\nla", "cot": "melody",
                                        "seed": job.seed, "id": job.id}
    options = jobs.hum_options(job)
    assert options.melody == "continue" and options.adapter is None and options.influence == 1.0
    # an adapter must be known to the lookup; ignore needs an adapter; ranges are enforced
    with_adapter = {**params, "adapter": "hum_v1", "melody": "ignore", "hum_influence": 1.5, "offset_s": 2}
    with pytest.raises(ValidationFailure, match="unknown or unusable hum adapter"):
        jobs.submit(store, {"kind": "hum", "params": with_adapter},
                    upload_lookup=lambda uid: {"filename": "a.m4a"})
    job2 = jobs.submit(store, {"kind": "hum", "params": with_adapter},
                       upload_lookup=lambda uid: {"filename": "a.m4a"},
                       hum_adapter_lookup=lambda name: name == "hum_v1").jobs[0]
    assert jobs.hum_options(job2).to_dict() == {"melody": "ignore", "adapter": "hum_v1", "hum_influence": 1.5,
                                                "offset_s": 2.0}
    for bad, message in [
        ({"melody": "ignore"}, "needs a hum adapter"),
        ({"melody": "loud"}, "melody"),
        ({"hum_influence": 5}, "hum_influence"),
        ({"offset_s": -1}, "offset_s"),
        ({"adapter": "../x"}, "adapter"),
    ]:
        with pytest.raises(ValidationFailure, match=message):
            jobs.validate_submit({"kind": "hum", "params": {**params, **bad}})


def test_list_filters_order_and_total(store):
    ids = [jobs.submit(store, {"kind": "create", "params": BASE}).jobs[0].id for _ in range(5)]
    group = jobs.submit(store, {"kind": "variations", "params": {"count": 2, "base": BASE}})
    store.update_status(ids[0], "done")
    store.update_status(ids[1], "failed", error="boom")
    all_jobs, total = store.list()
    assert total == 7 and len(all_jobs) == 7
    assert [j.id for j in all_jobs][-5:] == list(reversed(ids))  # newest first
    done, total = store.list(status=["done", "failed"])
    assert total == 2 and {j.id for j in done} == {ids[0], ids[1]}
    members, total = store.list(group=group.group["id"])
    assert total == 2 and {j.id for j in members} == set(group.group["job_ids"])
    page, total = store.list(limit=2, offset=1)
    assert total == 7 and len(page) == 2 and page[0].id == all_jobs[1].id
    queued, _ = store.list(status="queued")
    assert [j.position for j in sorted(queued, key=lambda j: j.position)] == [0, 1, 2, 3, 4]
    assert store.queued_ids() == [j.id for j in sorted(queued, key=lambda j: j.position)]


def test_status_transitions_and_conditional_update(store):
    job = jobs.submit(store, {"kind": "create", "params": BASE}).jobs[0]
    assert store.update_status(job.id, "cancelled", expected="queued")
    assert not store.update_status(job.id, "running", expected="queued")
    fetched = store.get(job.id)
    assert fetched.status == "cancelled" and fetched.finished_at is not None and fetched.position is None
    store.set_timing(job.id, {"plan": 1.0}, audio_seconds=3.5, truncated={"phase": "abc", "reason": "x"})
    store.set_progress(job.id, {"type": "status"})
    fetched = store.get(job.id)
    assert fetched.timing == {"plan": 1.0} and fetched.audio_seconds == 3.5
    assert fetched.truncated["phase"] == "abc" and fetched.progress == {"type": "status"}
    with pytest.raises(NotFound):
        store.update_status("missing", "done")


def test_delete_removes_group_when_last_member(store):
    sub = jobs.submit(store, {"kind": "variations", "params": {"count": 2, "base": BASE}})
    gid = sub.group["id"]
    store.delete(sub.jobs[0].id)
    assert store.get_group(gid)["job_ids"] == [sub.jobs[1].id]
    store.delete(sub.jobs[1].id)
    with pytest.raises(NotFound):
        store.get_group(gid)
    with pytest.raises(NotFound):
        store.delete(sub.jobs[1].id)


def test_settings_defaults_and_partial_update(store):
    assert store.get_settings() == {"default_preset": "quality",
                                    "memory_budget_gib": config.DEFAULT_MEMORY_BUDGET_GIB,
                                    "require_ac": False, "theme": "system", "prune_uploads_days": None}
    updated = store.update_settings({"theme": "dark", "memory_budget_gib": 20})
    assert updated["theme"] == "dark" and updated["memory_budget_gib"] == 20
    assert updated["default_preset"] == "quality"
    assert store.get_settings() == updated
    with pytest.raises(ValidationFailure):
        store.update_settings({"memory_budget_gib": 2})
    with pytest.raises(ValidationFailure):
        store.update_settings({"theme": "neon"})
    job = jobs.submit(store, {"kind": "create", "params": BASE}).jobs[0]
    assert job.options(store.get_settings()).memory_budget_gib == 20.0
    assert store.update_settings({"prune_uploads_days": 30})["prune_uploads_days"] == 30
    assert store.update_settings({"prune_uploads_days": None})["prune_uploads_days"] is None
    for bad in (0, 366, -1, 2.5, "7d", True):
        with pytest.raises(ValidationFailure):
            store.update_settings({"prune_uploads_days": bad})


def test_upload_counts_one_query_over_cover_and_hum_jobs(store):
    lookup = {"filename": "x.wav"}
    assert store.upload_counts() == {}
    jobs.submit(store, {"kind": "create", "params": BASE})  # no upload: never counted
    a = jobs.submit(store, {"kind": "cover", "params": {"upload_id": "u1", "style": "s", "lyrics": "l"}},
                    upload_lookup=lambda uid: lookup).jobs[0]
    b = jobs.submit(store, {"kind": "hum", "params": {"upload_id": "u1", "style": "s", "lyrics": "l"}},
                    upload_lookup=lambda uid: lookup).jobs[0]
    c = jobs.submit(store, {"kind": "cover", "params": {"upload_id": "u2", "style": "s", "lyrics": "l"}},
                    upload_lookup=lambda uid: lookup).jobs[0]
    assert store.upload_counts() == {"u1": {"total": 2, "active": 2}, "u2": {"total": 1, "active": 1}}
    store.update_status(a.id, "running")
    store.update_status(b.id, "done")
    store.update_status(c.id, "failed", error="x")
    assert store.upload_counts() == {"u1": {"total": 2, "active": 1}, "u2": {"total": 1, "active": 0}}
    store.update_status(a.id, "cancelled")
    assert store.upload_counts() == {"u1": {"total": 2, "active": 0}, "u2": {"total": 1, "active": 0}}
    store.delete(c.id)
    assert "u2" not in store.upload_counts()


def test_iso_timestamps_are_utc_millisecond_z():
    assert jobs.iso(1_757_800_000.123) == "2025-09-13T21:46:40.123Z"
    assert jobs.iso(0) == "1970-01-01T00:00:00.000Z"
    now = jobs.now_iso()
    assert len(now) == 24 and now.endswith("Z") and now[10] == "T"


def test_artifacts_for_reflects_song_dir(tmp_path):
    empty = {"audio": False, "score": False, "plan": False, "transcription": False, "hum": False}
    assert jobs.artifacts_for(tmp_path) == empty
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "score.abc").write_text("X:1")
    (tmp_path / "plan" / "plan.json").write_text("{}")
    assert jobs.artifacts_for(tmp_path) == {**empty, "score": True, "plan": True}
    (tmp_path / "song").mkdir()
    (tmp_path / "song" / "audio.flac").write_bytes(b"fLaC")
    (tmp_path / "transcription").mkdir()
    (tmp_path / "transcription" / "score.abc").write_text("X:1")
    assert jobs.artifacts_for(tmp_path) == {"audio": True, "score": True, "plan": True, "transcription": True,
                                            "hum": False}
    (tmp_path / "hum").mkdir()
    (tmp_path / "hum" / "hum.abc").write_text("X:1\n")
    assert jobs.artifacts_for(tmp_path)["hum"] is True


def test_engine_request_strips_display_fields(store):
    params = {**BASE, "title": "T", "cfg_scale": 1.25, "cot": "melody", "abc": "X:1\nK:C\nC|", "seed": 3}
    job = jobs.submit(store, {"kind": "create", "params": params}).jobs[0]
    assert jobs.engine_request(job) == {"style": BASE["style"], "lyrics": BASE["lyrics"], "cot": "melody",
                                        "seed": 3, "id": job.id, "abc": "X:1\nK:C\nC|", "cfg_scale": 1.25}
    plain = jobs.submit(store, {"kind": "create", "params": {**BASE, "seed": 4}}).jobs[0]
    assert set(jobs.engine_request(plain)) == {"style", "lyrics", "cot", "seed", "id"}


def test_title_and_blank_overrides(store):
    job = jobs.submit(store, {"kind": "create", "params": {**BASE, "title": "   "}}).jobs[0]
    assert job.title is None and job.to_api()["title"] is None
    parent = jobs.submit(store, {"kind": "create", "params": {**BASE, "title": "Orig"}}).jobs[0]
    regen = jobs.submit(store, {"kind": "regenerate",
                                "params": {"parent_id": parent.id, "abc": "X:1", "style": "  ", "lyrics": ""}}
                        ).jobs[0]
    assert regen.params["style"] == BASE["style"] and regen.params["lyrics"] == BASE["lyrics"]
    assert regen.title == "Orig" and regen.params["cot"] == "full"


def test_variations_seed_arithmetic_wraps_and_stays_valid(store):
    sub = jobs.submit(store, {"kind": "variations",
                              "params": {"count": 2, "base": {**BASE, "seed": 2**63 - 1}}})
    assert [j.seed for j in sub.jobs] == [2**63 - 1, 0]
    for job in sub.jobs:
        assert 0 <= jobs.engine_request(job)["seed"] < 2**63


def test_counts_and_running_id(store):
    a = jobs.submit(store, {"kind": "create", "params": BASE}).jobs[0]
    b = jobs.submit(store, {"kind": "create", "params": BASE}).jobs[0]
    assert store.counts() == {"queued": 2} and store.running_id() is None
    store.update_status(a.id, "running")
    assert store.counts() == {"queued": 1, "running": 1} and store.running_id() == a.id
    assert store.get(a.id).started_at is not None and store.get(b.id).position == 0
    store.update_status(a.id, "done")
    assert store.running_id() is None and store.get(a.id).finished_at is not None


def test_store_reopen_keeps_rows_and_seq(tmp_path):
    db = tmp_path / "app.db"
    s1 = JobStore(db)
    ids = [jobs.submit(s1, {"kind": "create", "params": BASE}).jobs[0].id for _ in range(3)]
    s1.update_settings({"theme": "dark"})
    s1.close()
    s2 = JobStore(db)
    try:
        rows, total = s2.list()
        assert total == 3 and [j.id for j in rows] == list(reversed(ids))
        assert s2.get_settings()["theme"] == "dark"
        new = jobs.submit(s2, {"kind": "create", "params": BASE}).jobs[0]
        assert new.seq == 4  # the counter continues where it left off
    finally:
        s2.close()


# -- projects / tracks / takes ---------------------------------------------------------------------

OLD_DDL = """
CREATE TABLE jobs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, group_id TEXT,
    parent_id TEXT, params_json TEXT NOT NULL, preset TEXT NOT NULL, precision TEXT NOT NULL,
    ode_steps INTEGER NOT NULL,
    seed INTEGER NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT,
    timing_json TEXT, audio_seconds REAL, truncated_json TEXT, progress_json TEXT, seq INTEGER NOT NULL);
CREATE TABLE groups (id TEXT PRIMARY KEY, label TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _finish(store, job_id, seconds=3.0):
    store.update_status(job_id, "done")
    song = store.songs_dir / job_id / "song"
    song.mkdir(parents=True, exist_ok=True)
    (song / "audio.flac").write_bytes(b"fLaC" + bytes(64))
    store.set_timing(job_id, {"e2e": 1.0}, audio_seconds=seconds)


def _create(store, **params):
    return jobs.submit(store, {"kind": "create", "params": {**BASE, **params}}).jobs[0]


def test_project_tables_are_added_to_an_old_database(tmp_path):
    db = tmp_path / "app.db"
    conn = sqlite3.connect(db)
    conn.executescript(OLD_DDL)
    conn.execute("INSERT INTO jobs VALUES ('j1','create','done',NULL,NULL,'{}','fast','8bit',8,1,'t',"
                 "NULL,NULL,NULL,NULL,NULL,NULL,NULL,1)")
    conn.commit()
    conn.close()
    store = JobStore(db, tmp_path / "songs")
    try:
        tables = {r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"projects", "tracks", "takes"} <= tables
        old = store.get("j1")
        assert old.take is None and old.to_api()["take"] is None and old.loras == ()
        project = store.create_project("Soundtrack")
        track = store.create_track(project["id"], "Main theme")
        store.attach_takes(track["id"], ["j1"])
        assert store.get("j1").take["track_name"] == "Main theme"
    finally:
        store.close()


def test_project_and_track_crud_and_listing(store):
    assert store.list_projects() == []
    project = store.create_project("Soundtrack", "for the short film")
    assert set(project) == {"id", "name", "description", "created_at", "updated_at", "tracks"}
    assert project["tracks"] == [] and len(project["id"]) == 32
    assert project["created_at"] == project["updated_at"]
    a = store.create_track(project["id"], "Main theme")
    b = store.create_track(project["id"], "Credits")
    assert set(a) == {"id", "project_id", "project_name", "name", "position", "chosen_job_id", "created_at",
                      "takes", "chosen"}
    assert (a["position"], b["position"]) == (0, 1) and a["project_name"] == "Soundtrack"
    assert a["takes"] == [] and a["chosen"] is None and a["chosen_job_id"] is None
    detail = store.get_project(project["id"])
    assert [t["id"] for t in detail["tracks"]] == [a["id"], b["id"]]
    assert detail["updated_at"] >= project["updated_at"]
    listed = store.list_projects()
    assert len(listed) == 1 and listed[0]["track_count"] == 2 and listed[0]["chosen_count"] == 0
    assert "tracks" not in listed[0]
    time.sleep(0.002)
    other = store.create_project("Other")
    assert [p["name"] for p in store.list_projects()] == ["Other", "Soundtrack"]  # updated_at desc
    time.sleep(0.002)
    renamed = store.update_project(project["id"], name="Score", description="")
    assert renamed["name"] == "Score" and renamed["description"] == ""
    assert [p["name"] for p in store.list_projects()] == ["Score", "Other"]
    assert store.update_track(a["id"], name="Theme")["name"] == "Theme"
    with pytest.raises(NotFound):
        store.get_project("nope")
    with pytest.raises(NotFound):
        store.get_track("nope")
    with pytest.raises(NotFound):
        store.create_track("nope", "x")
    with pytest.raises(NotFound):
        store.update_project("nope", name="x")
    store.delete_project(other["id"])
    with pytest.raises(NotFound):
        store.delete_project(other["id"])
    assert len(store.list_projects()) == 1


def test_track_order_repacks_and_rejects_bad_permutations(store):
    project = store.create_project("P")
    ids = [store.create_track(project["id"], name)["id"] for name in ("a", "b", "c", "d")]
    ordered = store.order_tracks(project["id"], [ids[2], ids[0], ids[3], ids[1]])
    assert [t["id"] for t in ordered["tracks"]] == [ids[2], ids[0], ids[3], ids[1]]
    assert [t["position"] for t in ordered["tracks"]] == [0, 1, 2, 3]
    for bad in ([ids[0]], ids + ["extra"], [ids[0], ids[0], ids[1], ids[2]], ids[:3] + ["nope"]):
        with pytest.raises(ValidationFailure):
            store.order_tracks(project["id"], bad)
    assert [t["id"] for t in store.get_project(project["id"])["tracks"]] == [ids[2], ids[0], ids[3], ids[1]]
    # position moves one track and re-packs the rest; out of range clamps to the end
    moved = store.update_track(ids[3], position=0)
    assert moved["position"] == 0
    assert [t["id"] for t in store.get_project(project["id"])["tracks"]] == [ids[3], ids[2], ids[0], ids[1]]
    store.update_track(ids[3], position=99)
    assert [t["id"] for t in store.get_project(project["id"])["tracks"]] == [ids[2], ids[0], ids[1], ids[3]]
    store.delete_track(ids[0])
    remaining = store.get_project(project["id"])["tracks"]
    assert [(t["id"], t["position"]) for t in remaining] == [(ids[2], 0), (ids[1], 1), (ids[3], 2)]
    with pytest.raises(NotFound):
        store.delete_track(ids[0])


def test_attach_detach_rate_and_choose_takes(store):
    project = store.create_project("P")
    track = store.create_track(project["id"], "T")
    j1, j2 = _create(store, title="one"), _create(store)
    assert store.get(j1.id).take is None
    with pytest.raises(NotFound):
        store.attach_takes(track["id"], [j1.id, "missing"])
    assert store.get_track(track["id"])["takes"] == []  # nothing written when one job is unknown
    with pytest.raises(NotFound):
        store.attach_takes("nope", [j1.id])
    result = store.attach_takes(track["id"], [j1.id, j2.id, j1.id])
    assert [j.id for j in result["takes"]] == [j1.id, j2.id]
    take = store.get(j1.id).take
    assert take == {"track_id": track["id"], "project_id": project["id"], "track_name": "T",
                    "project_name": "P", "thumb": None, "stars": None, "note": "",
                    "added_at": take["added_at"], "chosen": False}
    assert store.get(j1.id).to_api()["take"] == take
    # same track again is a no-op that keeps metadata
    store.update_take(j1.id, thumb=1, stars=4, note="keep")
    store.attach_takes(track["id"], [j1.id])
    take = store.get(j1.id).take
    assert (take["thumb"], take["stars"], take["note"]) == (1, 4, "keep")
    assert store.update_take(j1.id, thumb=0).take["thumb"] is None
    assert store.update_take(j1.id, thumb=-1, stars=None).take["stars"] is None
    assert store.update_take(j1.id, note=None).take["note"] == ""
    with pytest.raises(NotFound):
        store.update_take(_create(store).id, stars=3)
    # only a done take of this track can be chosen
    with pytest.raises(jobs.Conflict):
        store.update_track(track["id"], chosen_job_id=j1.id)
    with pytest.raises(jobs.Conflict):
        store.update_track(track["id"], chosen_job_id=_create(store).id)
    _finish(store, j1.id)
    chosen = store.update_track(track["id"], chosen_job_id=j1.id)
    assert chosen["chosen_job_id"] == j1.id and chosen["chosen"].id == j1.id
    assert store.get(j1.id).take["chosen"] is True and store.get(j2.id).take["chosen"] is False
    assert store.list_projects()[0]["chosen_count"] == 1
    assert store.update_track(track["id"], chosen_job_id=None)["chosen"] is None
    store.update_track(track["id"], chosen_job_id=j1.id)
    # detach clears the choice; jobs survive
    store.detach_take(j1.id)
    assert store.get(j1.id).take is None and store.get_track(track["id"])["chosen_job_id"] is None
    with pytest.raises(NotFound):
        store.detach_take(j1.id)
    # list filters
    both, total = store.list(track=track["id"])
    assert total == 1 and [j.id for j in both] == [j2.id]
    by_project, total = store.list(project=project["id"])
    assert total == 1 and by_project[0].id == j2.id
    assert store.list(project="nope")[1] == 0
    assert store.list()[1] == 4  # every job, joined or not


def test_attach_conflict_and_move_between_tracks(store):
    project = store.create_project("P")
    a = store.create_track(project["id"], "A")
    b = store.create_track(project["id"], "B")
    job = _create(store)
    _finish(store, job.id)
    store.attach_takes(a["id"], [job.id])
    store.update_take(job.id, stars=5, note="great")
    store.update_track(a["id"], chosen_job_id=job.id)
    with pytest.raises(jobs.Conflict, match="already a take of A"):
        store.attach_takes(b["id"], [job.id])
    assert store.get(job.id).take["track_id"] == a["id"]
    moved = store.attach_takes(b["id"], [job.id], move=True)
    assert [j.id for j in moved["takes"]] == [job.id]
    take = store.get(job.id).take
    assert take["track_id"] == b["id"] and take["stars"] == 5 and take["note"] == "great"
    assert take["chosen"] is False
    assert store.get_track(a["id"])["takes"] == [] and store.get_track(a["id"])["chosen_job_id"] is None
    # moving into a track of another project works the same way
    other = store.create_project("Q")
    c = store.create_track(other["id"], "C")
    store.attach_takes(c["id"], [job.id], move=True)
    assert store.get(job.id).take["project_name"] == "Q"


def test_deleting_jobs_tracks_and_projects_cascades_correctly(store):
    project = store.create_project("P")
    track = store.create_track(project["id"], "T")
    j1, j2 = _create(store), _create(store)
    _finish(store, j1.id)
    store.attach_takes(track["id"], [j1.id, j2.id])
    store.update_track(track["id"], chosen_job_id=j1.id)
    store.delete(j1.id)  # chosen job deleted -> take row gone, choice cleared
    detail = store.get_track(track["id"])
    assert detail["chosen_job_id"] is None and [j.id for j in detail["takes"]] == [j2.id]
    assert store._conn.execute("SELECT COUNT(*) FROM takes").fetchone()[0] == 1
    store.delete_track(track["id"])  # takes detached, job kept
    assert store.get(j2.id).take is None
    assert store._conn.execute("SELECT COUNT(*) FROM takes").fetchone()[0] == 0
    track = store.create_track(project["id"], "T2")
    store.attach_takes(track["id"], [j2.id])
    store.delete_project(project["id"])
    assert store.get(j2.id).take is None and store.list_projects() == []
    assert store._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0] == 0
    with pytest.raises(NotFound):
        store.get_track(track["id"])


def test_submit_with_track_id_attaches_every_job(store):
    project = store.create_project("P")
    track = store.create_track(project["id"], "T")
    with pytest.raises(NotFound):
        jobs.submit(store, {"kind": "create", "params": BASE, "track_id": "nope"})
    assert store.list()[1] == 0  # no rows written for an unknown track
    single = jobs.submit(store, {"kind": "create", "params": BASE, "track_id": track["id"]}).jobs[0]
    assert single.take["track_id"] == track["id"] and single.to_api()["take"]["track_name"] == "T"
    sub = jobs.submit(store, {"kind": "variations", "track_id": track["id"],
                              "params": {"count": 3, "base": BASE}})
    assert all(j.take["track_id"] == track["id"] for j in sub.jobs)
    assert [j.id for j in store.get_track(track["id"])["takes"]] == [single.id, *(j.id for j in sub.jobs)]
    plain = jobs.submit(store, {"kind": "create", "params": BASE, "track_id": "  "}).jobs[0]
    assert plain.take is None
    regen = jobs.submit(store, {"kind": "regenerate", "track_id": track["id"],
                                "params": {"parent_id": single.id, "abc": "X:1"}}).jobs[0]
    assert regen.take["track_id"] == track["id"]
    with pytest.raises(ValidationFailure):
        jobs.validate_submit({"kind": "create", "params": BASE, "track_id": 5})


def test_take_and_project_body_validation():
    assert jobs.parse(jobs.ProjectBody, {"name": "  My  album "}).name == "My album"
    assert jobs.parse(jobs.ProjectBody, {"name": "x"}).description == ""
    for bad in ({"name": ""}, {"name": "   "}, {}, {"name": "x" * 201},
                {"name": "x", "description": "y" * 2001}):
        with pytest.raises(ValidationFailure):
            jobs.parse(jobs.ProjectBody, bad)
    patch = jobs.parse(jobs.ProjectPatch, {"description": "d"})
    assert patch.model_fields_set == {"description"} and patch.name is None
    with pytest.raises(ValidationFailure):
        jobs.parse(jobs.ProjectPatch, {"name": ""})
    assert jobs.parse(jobs.TrackPatch, {"chosen_job_id": None}).model_fields_set == {"chosen_job_id"}
    for bad in ({"position": -1}, {"position": True}, {"position": 1.5}, {"name": " "}):
        with pytest.raises(ValidationFailure):
            jobs.parse(jobs.TrackPatch, bad)
    for good in ({"thumb": 1}, {"thumb": -1}, {"thumb": 0}, {"thumb": None}, {"stars": 5}, {"stars": None},
                 {"note": ""}, {}):
        jobs.parse(jobs.TakePatch, good)
    for bad in ({"thumb": 2}, {"thumb": True}, {"stars": 0}, {"stars": 6}, {"stars": True}, {"stars": 2.5},
                {"note": "n" * 4001}):
        with pytest.raises(ValidationFailure):
            jobs.parse(jobs.TakePatch, bad)
    for bad in ({"job_ids": []}, {}, {"job_ids": "x"}, {"job_ids": ["a"], "move": "maybe"}):
        with pytest.raises(ValidationFailure):
            jobs.parse(jobs.AttachBody, bad)
    assert jobs.parse(jobs.AttachBody, {"job_ids": ["a"]}).move is False
    with pytest.raises(ValidationFailure):
        jobs.parse(jobs.OrderBody, {"track_ids": "a"})


def test_tracklist_md_escapes_pipes_and_sweep_temp(tmp_path):
    from yue2_studio import projects

    class FakeJob:
        id, title, audio_seconds, seed, preset, kind = "j" * 32, "A | B", 61.0, 7, "fast", "create"
        artifacts = {"audio": True}

    project = {"id": "p", "name": "Vol | 1", "description": "",
               "tracks": [{"id": "t1", "name": "Main | theme", "chosen": FakeJob()},
                          {"id": "t2", "name": "Credits", "chosen": None}]}
    data = projects.tracklist(project, "flac")
    assert data["tracks"][0]["file"] == "01 Main theme.flac"  # safe_name drops the pipe from filenames
    md = projects.tracklist_md(data)
    assert "| Main \\| theme | 1:01 | 01 Main theme.flac | A \\| B (" in md
    assert md.count("\n") == 8 and md.startswith("# Vol | 1\n")
    (tmp_path / ".album-x.zip").write_bytes(b"PK")
    (tmp_path / ".album-y.zip").write_bytes(b"PK")
    (tmp_path / "other.zip").write_bytes(b"PK")
    assert projects.sweep_temp(tmp_path) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["other.zip"]
