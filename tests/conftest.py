"""Shared fixtures for the HTTP/worker tests.

* ``home`` / ``static_dir`` - per-test temp directories (``YUE2_STUDIO_HOME`` is never touched).
* ``engine`` - ``FakeEngine(delay=0)``; tests mutate ``delay`` / ``fail`` on ``app.state.engine``.
* ``app`` - ``create_app`` with the fake engine, running inside its lifespan (worker started).
* ``client`` - ``httpx.AsyncClient`` over the ASGI app.
* ``api`` - helper bound to the client: ``submit``, ``events`` (parsed SSE), ``wait``, ``wait_status``.
* ``make_app`` - async context-manager factory for apps with a different engine / home.

Nothing here imports ``mlx``; the fake engine is pure Python.
"""

import asyncio
import contextlib
import json

import pytest
from httpx import ASGITransport, AsyncClient

from yue2_studio import audio
from yue2_studio.fake import FakeEngine
from yue2_studio.main import create_app

BASE = {"style": "dreamy indie pop, female vocal", "lyrics": "[verse]\nla la\n[chorus]\nda da", "seed": 42}


MACHINE_RAM_GIB = 48.0  # tests see a 48 GB Mac: budget range 6..44, low-memory auto -> off


@pytest.fixture(autouse=True)
def machine_ram(monkeypatch):
    """Pin ``config.machine_ram_gib`` so budget caps and low-memory ``auto`` do not depend on the host.

    Tests that need another machine call ``machine_ram(16)``. ``config.DEFAULT_MEMORY_BUDGET_GIB`` (and
    so ``jobs.DEFAULT_SETTINGS``) was computed from the real host at import: 24 on any Mac with 28 GiB+.
    """
    from yue2_studio import config

    def pin(total_gib: float) -> None:
        monkeypatch.setattr(config, "machine_ram_gib", lambda: float(total_gib))

    pin(MACHINE_RAM_GIB)
    return pin


@pytest.fixture
def fake_probe(monkeypatch):
    """Uploads here are a few junk bytes: pretend ffprobe read them so ``POST /api/upload`` accepts them.

    The real ``probe_duration`` is exercised in ``test_audio.py``; the unreadable-audio rejection in
    ``test_api.py`` overrides this stub.
    """
    monkeypatch.setattr(audio, "probe_duration", lambda path: 12.3)


@pytest.fixture
def base_params():
    return dict(BASE)


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


@pytest.fixture
def static_dir(tmp_path):
    return tmp_path / "static"


@pytest.fixture
def engine():
    return FakeEngine(delay=0)


@pytest.fixture
async def app(engine, home, static_dir, fake_probe):
    application = create_app(engine, home=home, static_dir=static_dir)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def make_app(tmp_path, fake_probe):
    """``async with make_app(engine, home=..., static_dir=...) as (app, client)``."""

    @contextlib.asynccontextmanager
    async def _make(engine=None, *, home=None, static_dir=None, lifespan=True, **kwargs):
        application = create_app(engine, home=home or tmp_path / "home",
                                 static_dir=static_dir or tmp_path / "s", **kwargs)
        span = application.router.lifespan_context(application) if lifespan else contextlib.nullcontext()
        async with span:
            async with AsyncClient(transport=ASGITransport(app=application), base_url="http://test") as c:
                yield application, c

    return _make


class Api:
    def __init__(self, client: AsyncClient):
        self.client = client

    async def submit(self, body: dict, expect: int = 201) -> dict:
        r = await self.client.post("/api/jobs", json=body)
        assert r.status_code == expect, r.text
        return r.json()

    async def create(self, params: dict | None = None, **top) -> dict:
        """Submit one create job and return its Job JSON."""
        return (await self.submit({"kind": "create", "params": params or BASE, **top}))["job"]

    async def events(self, job_id: str, timeout: float = 10.0) -> tuple[list[dict], dict]:
        """Consume ``GET /api/jobs/{id}/events`` into ``(progress_events, done_job)``."""
        progress, done = [], None
        async with self.client.stream("GET", f"/api/jobs/{job_id}/events") as r:
            assert r.status_code == 200, await r.aread()
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

    async def wait(self, job_id: str, timeout: float = 10.0) -> dict:
        """Block until the job is terminal (via SSE) and return its final Job JSON."""
        _, done = await self.events(job_id, timeout)
        return done

    async def get(self, job_id: str) -> dict:
        r = await self.client.get(f"/api/jobs/{job_id}")
        assert r.status_code == 200, r.text
        return r.json()["job"]

    async def wait_status(self, job_id: str, status: str, timeout: float = 5.0) -> dict:
        """Poll until the job reports ``status`` (used to catch a job while it is running)."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            job = await self.get(job_id)
            if job["status"] == status:
                return job
            assert asyncio.get_running_loop().time() < deadline, f"{job_id} never became {status}"
            await asyncio.sleep(0.01)


@pytest.fixture
def api(client):
    return Api(client)
