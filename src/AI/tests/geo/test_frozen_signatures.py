"""Signatures that P0 froze for P1 implementers (01 §2.2 geo.ops, 02 §2.2 perception.corridor).

These stay valid once the stubs get real bodies; only a signature change breaks them.
"""

from __future__ import annotations

import inspect

import pytest

from vineyard.geo import ops
from vineyard.perception import corridor

FROZEN = {
    (ops, "clip_polygonal"): ["geom", "clip"],
    (ops, "clip_line"): ["line", "clip"],
    (ops, "clip_box"): ["xyxy", "bounds"],
    (ops, "notch_holes"): ["poly", "notch_width"],
    (corridor, "clip_rows_to_tile"): ["rows", "tile", "clip", "margin_m"],
    (corridor, "corridor_polygon"): ["axis", "half_m"],
    (corridor, "corridor_labels"): ["pieces", "tile", "half_m"],
}


@pytest.mark.parametrize(("module", "name"), list(FROZEN))
def test_frozen_signature(module: object, name: str) -> None:
    params = list(inspect.signature(getattr(module, name)).parameters)
    assert params == FROZEN[(module, name)]


def test_clip_rows_margin_default() -> None:
    assert inspect.signature(corridor.clip_rows_to_tile).parameters["margin_m"].default == 0.0
