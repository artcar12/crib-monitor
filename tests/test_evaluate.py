from crib_monitor.classifier import ClassifyResult
from crib_monitor.evaluate import format_report, score
from crib_monitor.labels import Position

B, S, U = Position.BACK, Position.STOMACH, Position.UNCLEAR


def item(truth, local, cloud):
    return truth, [ClassifyResult("local", local, 2.0), ClassifyResult("cloud", cloud, 1.0)]


def test_score_per_model_and_combined():
    stats = score([
        item("stomach", S, B),       # local catches, cloud misses
        item("stomach", None, S),    # local unavailable, cloud catches
        item("back", B, S),          # cloud false alarm
        item("side", U, B),
        item("not_visible", None, None),
    ])
    assert (stats["local"].tp, stats["local"].fn, stats["local"].fp, stats["local"].unavailable) == (1, 1, 0, 2)
    assert (stats["cloud"].tp, stats["cloud"].fn, stats["cloud"].fp) == (1, 1, 1)
    combined = stats["combined"]
    assert combined.recall == 1.0
    assert combined.fp == 1 and combined.tn == 2
    assert combined.unavailable == 1


def test_score_empty_still_has_combined():
    stats = score([])
    assert stats["combined"].tp == 0 and stats["combined"].fn == 0


def test_report_mentions_every_model():
    report = format_report(score([item("stomach", S, B), item("back", B, B)]), checks_per_night=200)
    assert "local: stomach caught 1/1 (100%)" in report
    assert "cloud: stomach caught 0/1 (0%)" in report
    assert "combined:" in report
    assert "200 checks per night" in report
