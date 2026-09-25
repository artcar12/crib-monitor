import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from crib_monitor.storage import Storage

NY = ZoneInfo("America/New_York")
T0 = datetime(2026, 9, 26, 3, 4, 5, 678000, tzinfo=NY)


@pytest.fixture
def storage(tmp_path) -> Storage:
    return Storage(tmp_path, retention_days=30, tz=NY)


def test_save_check_writes_frame_and_log(storage, tmp_path):
    rel = storage.save_check(T0, b"jpeg", {"combined": "back"})
    assert rel == "2026-09-26/030405_678.jpg"
    assert (tmp_path / "frames" / rel).read_bytes() == b"jpeg"
    line = json.loads((tmp_path / "log" / "2026-09-26.jsonl").read_text().strip())
    assert line["frame"] == rel and line["combined"] == "back" and line["ts"] == T0.isoformat()


def test_cleanup_removes_old_days_only(storage, tmp_path):
    old = storage.save_check(T0 - timedelta(days=31), b"old", {})
    new = storage.save_check(T0, b"new", {})
    storage.add_label(old, "stomach")
    storage.cleanup(T0)
    assert not (tmp_path / "frames" / old).exists()
    assert not (tmp_path / "log" / f"{(T0 - timedelta(days=31)):%Y-%m-%d}.jsonl").exists()
    assert (tmp_path / "frames" / new).exists()
    assert len(storage.eval_items()) == 1          # labeled copy survives


def test_labeling_flow(storage):
    a = storage.save_check(T0, b"a", {})
    b = storage.save_check(T0 + timedelta(seconds=10), b"b", {})
    c = storage.save_check(T0 + timedelta(seconds=20), b"c", {})
    assert storage.unlabeled() == [a, b, c]
    storage.add_label(a, "stomach")
    storage.add_label(b, "skip")
    assert storage.unlabeled() == [c]
    items = storage.eval_items()
    assert len(items) == 1
    path, label = items[0]
    assert label == "stomach" and path.read_bytes() == b"a"


def test_invalid_label_rejected(storage):
    rel = storage.save_check(T0, b"a", {})
    with pytest.raises(ValueError):
        storage.add_label(rel, "upside_down")


@pytest.mark.parametrize("rel", ["../state.json", "/etc/passwd", "2026-09-26/missing.jpg"])
def test_frame_path_rejects_outside_or_missing(storage, rel):
    with pytest.raises(ValueError):
        storage.frame_path(rel)
