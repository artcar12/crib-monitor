"""Service entry point: wire everything together and run the web page and monitor loop."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Mapping
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
from .web import create_app

log = logging.getLogger("crib_monitor")


def build(cfg: Config, secrets: Secrets, env: Mapping[str, str], client: httpx.AsyncClient) -> tuple[Monitor, FastAPI]:
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
    return monitor, create_app(monitor, storage, secrets.control_token, tz)


async def amain(cfg: Config, secrets: Secrets, env: Mapping[str, str]) -> None:
    async with httpx.AsyncClient() as client:
        monitor, app = build(cfg, secrets, env, client)
        server = uvicorn.Server(
            uvicorn.Config(app, host=cfg.web.host, port=cfg.web.port, access_log=False, log_level="info")
        )
        loop_task = asyncio.create_task(monitor.run_forever())
        try:
            await server.serve()
        finally:
            loop_task.cancel()
            monitor.shutdown()


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
    asyncio.run(amain(cfg, secrets, os.environ))
