"""cvat.rle: CVAT <mask> RLE -> polygon rings in continuous px (import only)."""

import numpy as np
import pytest
from shapely.geometry import Polygon

from vineyard.cvat.rle import parse_rle, rle_to_mask, rle_to_rings
from vineyard.errors import CvatFormatError


def test_parse_rle_accepts_spaces() -> None:
    assert parse_rle("2, 8,2") == (2, 8, 2)


@pytest.mark.parametrize("text", ["", "a,b", "2,-1", " , "])
def test_parse_rle_rejects_garbage(text: str) -> None:
    with pytest.raises(CvatFormatError):
        parse_rle(text)


def test_rle_to_mask_starts_with_background_row_major() -> None:
    mask = rle_to_mask((2, 8, 2), width=4, height=3)
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask.astype(int), [[0, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0]])


def test_rle_to_mask_rejects_wrong_total() -> None:
    with pytest.raises(CvatFormatError):
        rle_to_mask((2, 8), width=4, height=3)


def test_rle_to_mask_rejects_bad_size() -> None:
    with pytest.raises(CvatFormatError):
        rle_to_mask((0,), width=0, height=3)


def test_rle_ring_has_pixel_exact_area_and_offset() -> None:
    rings = rle_to_rings("2, 8, 2", left=10, top=20, width=4, height=3)
    assert len(rings) == 1
    poly = Polygon(rings[0])
    assert poly.is_valid
    assert poly.area == pytest.approx(8.0)
    assert poly.bounds == (10.0, 20.0, 14.0, 23.0)
    assert rings[0][0] != rings[0][-1]  # no closing vertex


def test_rle_two_components_sorted_top_left_first() -> None:
    # 3x2 box: row 0 = [1,0,1], row 1 = [1,0,1] -> 2 separate columns
    rings = rle_to_rings("0,1,1,2,1,1", left=0, top=0, width=3, height=2)
    assert len(rings) == 2
    assert [Polygon(r).bounds[0] for r in rings] == [0.0, 2.0]


def test_rle_empty_mask_gives_no_rings() -> None:
    assert rle_to_rings("6", left=0, top=0, width=3, height=2) == ()
