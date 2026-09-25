"""Per-model positions and the combined reading the decision logic consumes."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class Position(StrEnum):
    BACK = "back"
    STOMACH = "stomach"
    SIDE = "side"
    UNCLEAR = "unclear"
    NOT_VISIBLE = "not_visible"


class Combined(StrEnum):
    STOMACH = "stomach"
    SIDE = "side"
    BACK = "back"
    NO_VIEW = "no_view"
    FAILED = "failed"


IN_CRIB = frozenset({Combined.BACK, Combined.SIDE, Combined.STOMACH})


def combine(labels: Iterable[Position | None]) -> Combined:
    """Most alarming answer wins. `None` means that model was unavailable."""
    available = [label for label in labels if label is not None]
    if not available:
        return Combined.FAILED
    if Position.STOMACH in available:
        return Combined.STOMACH
    if Position.SIDE in available:
        return Combined.SIDE
    if Position.BACK in available:
        return Combined.BACK
    return Combined.NO_VIEW
