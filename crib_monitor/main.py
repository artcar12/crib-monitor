"""Service entry point: wire everything together and run the web page and monitor loop."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from .arming import Arming
from .capture import Capture, ffmpeg_command
from .classifier import build_classifiers
from .config import Config, ConfigError, Secrets, load_config, load_secrets
from .health import Health, Heartbeat
from .monitor import Monitor
from .notifier import Alerter, Pushover
from .schedule import Schedule
from .storage import Storage
from .web import Lifespan, create_app

log = logging.getLogger("crib_monitor")


def loop_done(task: asyncio.Task[None], on_exit: Callable[[], None]) -> None:
    """Done-callback for the monitor loop task: anything but cancellation is fatal."""
    if task.cancelled():
        return
    exc = task.exception()
    # Type name only: an exception message could carry a URL or token.
    log.critical("monitor loop stopped (%s); exiting", type(exc).__name__ if exc else "returned")
    on_exit()


def monitor_lifespan(monitor: Monitor, on_exit: Callable[[], None]) -> Lifespan:
    """Run the monitor loop for as long as the web server runs.

    uvicorn runs lifespan shutdown on SIGTERM before re-raising the signal, so this is where
    the capture process gets stopped. shutdown() never cancels an open emergency receipt.
    """

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(monitor.run_forever(), name="monitor-loop")
        task.add_done_callback(lambda t: loop_done(t, on_exit))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):  # a crash was already logged
                await task
            monitor.shutdown()

    return lifespan


def build(
    cfg: Config,
    secrets: Secrets,
    env: Mapping[str, str],
    client: httpx.AsyncClient,
    on_loop_exit: Callable[[], None] = lambda: None,
) -> tuple[Monitor, FastAPI]:
    tz = ZoneInfo(cfg.schedule.timezone)
    storage = Storage(cfg.storage.data_dir, cfg.storage.retention_days, tz)
    arming = Arming(Schedule(cfg.schedule), cfg.arming, Path(cfg.storage.data_dir) / "state.json")
    crop = cfg.camera.crop

    def capture_factory() -> Capture:
        return Capture(
            ffmpeg_command(secrets.tapo_rtsp_url, crop), crop.w, crop.h,
            stall_s=cfg.camera.stream_timeout_s, max_backoff_s=cfg.camera.max_backoff_s,
        )

    monitor = Monitor(
        cfg=cfg,
        arming=arming,
        capture_factory=capture_factory,
        classifiers=build_classifiers(cfg.classifier, env),
        alerter=Alerter(Pushover(secrets.pushover_token, secrets.pushover_user, client), cfg.alerts),
        health=Health(cfg.health, [m.name for m in cfg.classifier.models], cfg.alerts.health_repeat_s),
        heartbeat=Heartbeat(secrets.healthchecks_url, client),
        storage=storage,
        now=lambda: datetime.now(UTC),
    )
    app = create_app(monitor, storage, secrets.control_token, tz, lifespan=monitor_lifespan(monitor, on_loop_exit))
    return monitor, app


async def amain(cfg: Config, secrets: Secrets, env: Mapping[str, str]) -> int:
    """Serve until stopped. Returns 1 if the monitor loop died, so systemd restarts us."""
    exit_code = 0
    server: uvicorn.Server | None = None

    def loop_died() -> None:
        nonlocal exit_code
        exit_code = 1
        if server is not None:
            server.should_exit = True

    async with httpx.AsyncClient() as client:
        _, app = build(cfg, secrets, env, client, on_loop_exit=loop_died)
        server = uvicorn.Server(
            uvicorn.Config(app, host=cfg.web.host, port=cfg.web.port, access_log=False, log_level="info")
        )
        await server.serve()
    return exit_code


def _import_litellm() -> None:
    # Import at startup rather than on the first check: the import is slow and would block the
    # event loop mid-session, and without this flag it fetches the model cost map over the network.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    litellm.suppress_debug_info = True


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crib-monitor")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    if Path(args.env_file).exists():
        load_dotenv(args.env_file)
    try:
        cfg = load_config(Path(args.config))
        secrets = load_secrets(os.environ, cfg)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    _import_litellm()
    code = asyncio.run(amain(cfg, secrets, os.environ))
    if code:
        raise SystemExit(code)
