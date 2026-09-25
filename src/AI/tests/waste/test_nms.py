from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.waste.synth_waste import make_candidate
from vineyard.perception.waste.nms import box_iou_matrix, mark_suppressed, nms, nms_keep
from vineyard.perception.waste.types import BoxPx, RejectReason


def boxed(key: str, b: tuple[float, float, float, float], tile: str = "siret3_r021_c012"):
    return replace(make_candidate(tile_id=tile, key=key), box=BoxPx(*b))


def overlap_pair(iou: float):
    """Two 100x100 boxes sharing a vertical strip so that IoU == iou."""
    shift = 100 * (1 - iou) / (1 + iou)
    return boxed("a", (0, 0, 100, 100)), boxed("b", (shift, 0, 100 + shift, 100))


def test_iou_035_keeps_only_higher_ranked() -> None:
    a, b = overlap_pair(0.35)
    assert a.box.iou(b.box) == pytest.approx(0.35)
    assert nms([a, b], {"a": 0.4, "b": 0.9}, 0.3) == (b,)


def test_iou_029_keeps_both() -> None:
    a, b = overlap_pair(0.29)
    assert nms([a, b], {"a": 0.4, "b": 0.9}, 0.3) == (a, b)


def test_other_tile_never_suppresses() -> None:
    a = boxed("a", (0, 0, 100, 100))
    b = boxed("b", (0, 0, 100, 100), tile="siret3_r006_c004")
    assert nms_keep([a, b], {"a": 1.0, "b": 0.5}, 0.3) == (True, True)


def test_ties_broken_by_key() -> None:
    a, b = boxed("b", (0, 0, 10, 10)), boxed("a", (0, 0, 10, 10))
    assert nms([a, b], {"a": 0.5, "b": 0.5}, 0.3) == (b,)


def test_missing_rank_raises() -> None:
    with pytest.raises(KeyError, match="no rank"):
        nms([boxed("a", (0, 0, 10, 10))], {}, 0.3)


def test_iou_matrix() -> None:
    m = box_iou_matrix(np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]))
    assert m[0, 1] == pytest.approx(50 / 150)
    assert m[0, 2] == 0.0
    assert np.allclose(np.diag(m), 1.0)


def test_mark_suppressed_skips_rejected_and_keeps_order() -> None:
    a, b = overlap_pair(0.5)
    c = replace(boxed("c", (0, 0, 100, 100)), reject_reason=RejectReason.NEAR_AXIS)
    out = mark_suppressed([c, a, b], {"a": 0.9, "b": 0.1, "c": 1.0}, 0.3)
    assert [x.reject_reason for x in out] == [RejectReason.NEAR_AXIS, None, RejectReason.NMS]
