"""How well does each model (and the either-says-stomach combination) do on labeled crib frames?"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from .classifier import ClassifyResult, build_classifiers, classify_all
from .config import load_config
from .labels import Position
from .storage import Storage


@dataclass
class Stats:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    unavailable: int = 0
    latencies: list[float] = field(default_factory=list)

    def add(self, truth_stomach: bool, predicted_stomach: bool, available: bool) -> None:
        if not available:
            self.unavailable += 1
        if truth_stomach:
            if predicted_stomach:
                self.tp += 1
            else:
                self.fn += 1
        elif predicted_stomach:
            self.fp += 1
        else:
            self.tn += 1

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def fp_rate(self) -> float | None:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else None


def score(items: Iterable[tuple[str, list[ClassifyResult]]]) -> dict[str, Stats]:
    stats: dict[str, Stats] = {}
    for truth, results in items:
        is_stomach = truth == "stomach"
        for r in results:
            s = stats.setdefault(r.model_name, Stats())
            s.add(is_stomach, r.position is Position.STOMACH, r.position is not None)
            s.latencies.append(r.latency_s)
        stats.setdefault("combined", Stats()).add(
            is_stomach,
            any(r.position is Position.STOMACH for r in results),
            any(r.position is not None for r in results),
        )
    # keep "combined" last in the report
    combined = stats.pop("combined", None)
    if combined is not None:
        stats["combined"] = combined
    return stats


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def format_report(stats: dict[str, Stats], checks_per_night: int) -> str:
    lines = []
    for name, s in stats.items():
        latency = f"{statistics.median(s.latencies):.1f}s" if s.latencies else "-"
        estimate = "n/a" if s.fp_rate is None else f"{checks_per_night * s.fp_rate ** 2:.2f}"
        lines.append(
            f"{name}: stomach caught {s.tp}/{s.tp + s.fn} ({_pct(s.recall)}), "
            f"false stomach {s.fp}/{s.fp + s.tn} ({_pct(s.fp_rate)}), unavailable {s.unavailable}, "
            f"median latency {latency}, est. false alerts/night {estimate}"
        )
    lines.append(
        f"Estimate assumes {checks_per_night} checks per night and that a false alert needs two independent "
        "false stomach readings in a row. Real errors are correlated, so treat it as a lower bound."
    )
    return "\n".join(lines)


async def evaluate(classifiers: Sequence[Any], items: list[tuple[Path, str]]) -> list[tuple[str, list[ClassifyResult]]]:
    out = []
    for path, truth in items:
        out.append((truth, await classify_all(classifiers, path.read_bytes())))
    return out


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crib-monitor-eval", description=__doc__)
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--checks-per-night", type=int, default=200)
    args = parser.parse_args(argv)
    if Path(args.env_file).exists():
        load_dotenv(args.env_file)
    cfg = load_config(Path(args.config))
    storage = Storage(cfg.storage.data_dir, cfg.storage.retention_days, ZoneInfo(cfg.schedule.timezone))
    items = storage.eval_items()
    if not items:
        print("No labeled frames yet. Label some on the control page's /label view first.")
        raise SystemExit(1)
    results = asyncio.run(evaluate(build_classifiers(cfg.classifier, os.environ), items))
    stats = score(results)
    print(format_report(stats, args.checks_per_night))
    passed = stats["combined"].recall == 1.0
    print("PASS: every labeled stomach frame was caught." if passed else "FAIL: at least one stomach frame was missed.")
    raise SystemExit(0 if passed else 1)
