"""FastAPI app factory and the ``yue2-studio`` console entry point.

``create_app(engine=None, *, home=None, fake=False)`` wires the job store, worker thread, event
bus and routes. The real engine (``yue2_studio.engine``, which imports mlx) is imported lazily
and only when neither ``engine`` nor ``fake`` is given, so tests and ``--fake`` never touch MLX.
"""

from __future__ import annotations

from yue2_studio import config  # isort: skip  (sets MLX_ENABLE_TF32 before anything imports mlx)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import threading  # noqa: E402
import webbrowser  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI  # noqa: E402

from yue2_studio import __version__, api  # noqa: E402
from yue2_studio.jobs import JobStore  # noqa: E402
from yue2_studio.worker import EventBus, Worker  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("yue2_studio")


def _real_engine(paths: config.Paths):
    from yue2_studio.engine import Engine  # imports mlx; deliberately lazy

    return Engine(converted_dir=paths.converted_dir, vae_dir=paths.vae_dir)


def create_app(engine=None, *, home: str | os.PathLike | None = None, fake: bool = False,
               fake_delay: float = 0.05, static_dir: Path | None = None) -> FastAPI:
    paths = config.paths_for(home)
    paths.ensure_dirs()
    if engine is None and fake:
        from yue2_studio.fake import FakeEngine

        engine = FakeEngine(delay=fake_delay)
    is_fake = fake or (engine is not None and type(engine).__name__ == "FakeEngine")
    if engine is None:
        engine = _real_engine(paths)

    store = JobStore(paths.db_path, paths.songs_dir)
    bus = EventBus()
    worker = Worker(store, engine, paths, bus, fake=is_fake)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start(asyncio.get_running_loop())
        try:
            yield
        finally:
            await asyncio.to_thread(worker.stop)
            store.close()

    app = FastAPI(title="YuE2 Studio", version=__version__, lifespan=lifespan, docs_url="/api/docs",
                  openapi_url="/api/openapi.json", redoc_url=None)
    app.state.paths = paths
    app.state.store = store
    app.state.worker = worker
    app.state.bus = bus
    app.state.engine = engine
    app.state.fake = is_fake
    api.install_error_handlers(app)
    app.include_router(api.router)
    api.install_static(app, static_dir or STATIC_DIR)
    return app


def app_factory() -> FastAPI:
    """Factory for ``uvicorn --factory yue2_studio.main:app_factory`` (used by ``--reload``)."""
    return create_app(
        fake=os.environ.get("YUE2_STUDIO_FAKE") == "1",
        fake_delay=float(os.environ.get("YUE2_STUDIO_FAKE_DELAY", "0.3")),
        home=os.environ.get("YUE2_STUDIO_HOME") or None,
    )


def _open_browser_later(url: str, delay: float = 1.0) -> None:
    threading.Timer(delay, lambda: webbrowser.open(url)).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yue2-studio", description="Local YuE2 music-generation studio")
    parser.add_argument("--host", default=config.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=config.DEFAULT_PORT)
    parser.add_argument("--open", action="store_true", help="open the UI in the default browser")
    parser.add_argument("--fake", action="store_true", help="use the scripted fake engine (no models needed)")
    parser.add_argument("--fake-delay", type=float, default=0.3, help="seconds between fake engine events")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    import uvicorn

    logging.basicConfig(level=args.log_level.upper())
    url = f"http://{args.host}:{args.port}/"
    if args.open:
        _open_browser_later(url)
    if args.reload:
        os.environ["YUE2_STUDIO_FAKE"] = "1" if args.fake else "0"
        os.environ["YUE2_STUDIO_FAKE_DELAY"] = str(args.fake_delay)
        uvicorn.run("yue2_studio.main:app_factory", factory=True, host=args.host, port=args.port,
                    reload=True, log_level=args.log_level)
        return 0
    app = create_app(fake=args.fake, fake_delay=args.fake_delay)
    log.info("YuE2 Studio %s (%s engine) on %s", __version__, "fake" if args.fake else "mlx", url)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
