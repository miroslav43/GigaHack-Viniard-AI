from __future__ import annotations

import math
from dataclasses import replace

import pytest

from tests.waste.synth_waste import make_candidate
from vineyard.perception.waste.types import (
    RECORD_COLUMNS,
    BoxPx,
    RejectReason,
    candidate_from_record,
    candidate_to_record,
)


def test_box_validation_and_geometry() -> None:
    b = BoxPx(10, 20, 30, 60)
    assert (b.width, b.height, b.area, b.centre) == (20, 40, 800, (20.0, 40.0))
    assert b.polygon().area == 800
    assert b.iou(BoxPx(10, 20, 30, 60)) == 1.0
    assert b.iou(BoxPx(100, 100, 110, 110)) == 0.0
    for bad in ((5, 0, 5, 1), (0, 5, 1, 4), (-1, 0, 1, 1), (0, 0, 2049, 1), (0, 0, math.nan, 1)):
        with pytest.raises(ValueError):
            BoxPx(*bad)


def test_record_round_trip() -> None:
    c = make_candidate(is_white=True)
    rec = candidate_to_record(c)
    assert tuple(rec) == RECORD_COLUMNS
    assert candidate_from_record(rec) == c
    r = replace(c, reject_reason=RejectReason.HOSE)
    assert candidate_from_record(candidate_to_record(r)) == r
    assert candidate_from_record({**rec, "reject_reason": float("nan")}).reject_reason is None
    assert c.rejected is False and r.rejected is True
