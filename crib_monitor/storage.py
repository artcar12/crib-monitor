"""Checked frames, the per-check log, retention, and the labeled eval set."""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

LABELS = ("back", "stomach", "side", "not_visible", "skip")


class Storage:
    def __init__(self, data_dir: Path, retention_days: int, tz: ZoneInfo) -> None:
        self.root = Path(data_dir)
        self.frames = self.root / "frames"
        self.logs = self.root / "log"
        self.eval = self.root / "eval"
        self._labels_file = self.eval / "labels.jsonl"
        self._retention = retention_days
        self._tz = tz

    def save_check(self, now: datetime, jpeg: bytes, record: dict[str, Any]) -> str:
        local = now.astimezone(self._tz)
        day = f"{local:%Y-%m-%d}"
        rel = f"{day}/{local:%H%M%S}_{local.microsecond // 1000:03d}.jpg"
        path = self.frames / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        self.logs.mkdir(parents=True, exist_ok=True)
        with open(self.logs / f"{day}.jsonl", "a") as f:
            f.write(json.dumps({"ts": now.isoformat(), "frame": rel, **record}) + "\n")
        return rel

    def cleanup(self, now: datetime) -> None:
        cutoff = now.astimezone(self._tz).date() - timedelta(days=self._retention)
        if self.frames.exists():
            for day_dir in self.frames.iterdir():
                if _day(day_dir.name) is not None and _day(day_dir.name) < cutoff:
                    shutil.rmtree(day_dir)
        if self.logs.exists():
            for log_file in self.logs.glob("*.jsonl"):
                if _day(log_file.stem) is not None and _day(log_file.stem) < cutoff:
                    log_file.unlink()

    def frame_path(self, rel: str) -> Path:
        base = self.frames.resolve()
        path = (self.frames / rel).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"no such frame: {rel}")
        return path

    def unlabeled(self) -> list[str]:
        if not self.frames.exists():
            return []
        done = {entry["source"] for entry in self._entries()}
        all_frames = sorted(p.relative_to(self.frames).as_posix() for p in self.frames.glob("*/*.jpg"))
        return [rel for rel in all_frames if rel not in done]

    def add_label(self, rel: str, label: str) -> None:
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        source = self.frame_path(rel)
        entry: dict[str, str] = {"source": rel, "label": label}
        if label != "skip":
            dest = self.eval / "images" / rel.replace("/", "_")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            entry["image"] = f"images/{dest.name}"
        self.eval.mkdir(parents=True, exist_ok=True)
        with open(self._labels_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def eval_items(self) -> list[tuple[Path, str]]:
        latest: dict[str, dict[str, str]] = {}
        for entry in self._entries():
            latest[entry["source"]] = entry
        return [(self.eval / e["image"], e["label"]) for e in latest.values() if "image" in e]

    def _entries(self) -> list[dict[str, str]]:
        if not self._labels_file.exists():
            return []
        return [json.loads(line) for line in self._labels_file.read_text().splitlines() if line.strip()]


def _day(name: str) -> date | None:
    try:
        return date.fromisoformat(name)
    except ValueError:
        return None
