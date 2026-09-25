import pytest

from crib_monitor.labels import Combined, Position, combine

B, S, SD, U, NV = Position.BACK, Position.STOMACH, Position.SIDE, Position.UNCLEAR, Position.NOT_VISIBLE


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([B, S], Combined.STOMACH),
        ([S, None], Combined.STOMACH),
        ([B, SD], Combined.SIDE),
        ([SD, S], Combined.STOMACH),
        ([B, B], Combined.BACK),
        ([B, U], Combined.BACK),
        ([B, None], Combined.BACK),
        ([U, NV], Combined.NO_VIEW),
        ([NV, None], Combined.NO_VIEW),
        ([None, None], Combined.FAILED),
        ([], Combined.FAILED),
    ],
)
def test_combine(labels, expected):
    assert combine(labels) is expected
