"""JobStore, validation and submission expansion (no engine, no app)."""

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
                     "seed", "params", "title", "created_at", "started_at", "finished_at", "error", "timing",
                     "truncated", "progress", "artifacts", "position"}
    assert set(api) == expected_keys
    assert api["position"] == 0 and api["title"] is None and api["timing"] is None
    assert api["artifacts"] == {"audio": False, "score": False, "plan": False, "transcription": False}
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
    by_seed = [j.id for j in sorted(sub.jobs, key=lambda j: j.seed)]
    assert store.get_group(sub.group["id"])["job_ids"] == by_seed


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
                                    "require_ac": False, "theme": "system"}
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
