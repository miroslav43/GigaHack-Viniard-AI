from __future__ import annotations

import numpy as np
import pytest

from fte.convert.targets import (
    IGNORE,
    GroundVegSpec,
    contact_band,
    excess_green,
    fg_from_instances,
    ground_veg_labels,
    stack_labels,
)
from fte.convert.yolo_seg import parse_yolo_seg, rasterize_instances


def test_parse_scales_normalised_coords_to_pixels() -> None:
    polys = parse_yolo_seg("0 0.1 0.2 0.5 0.2 0.5 0.6\n\n0 0 0 1 0 1 1 0 1\n", width=100, height=50)
    assert len(polys) == 2
    np.testing.assert_allclose(polys[0].xy, [[10, 10], [50, 10], [50, 30]])
    assert polys[1].cls == 0 and polys[1].xy.shape == (4, 2)


@pytest.mark.parametrize("line", ["0 0.1 0.2 0.3 0.4", "0 0.1 0.2 0.3 0.4 0.5", "0 a b c d e f"])
def test_parse_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(ValueError, match="line 1"):
        parse_yolo_seg(line, 10, 10)


def test_rasterize_instances_ids_in_file_order() -> None:
    polys = parse_yolo_seg("0 0 0 0.5 0 0.5 1 0 1\n0 0.5 0 1 0 1 1 0.5 1\n", 20, 10)
    inst = rasterize_instances(polys, (10, 20))
    assert inst.dtype == np.uint16
    assert set(np.unique(inst)) <= {0, 1, 2}
    assert inst[5, 2] == 1 and inst[5, 17] == 2
    assert (inst > 0).mean() > 0.8


def test_contact_band_marks_only_between_touching_instances() -> None:
    inst = np.zeros((20, 40), np.uint16)
    inst[5:15, 5:19] = 1
    inst[5:15, 21:35] = 2  # 2 px gap at columns 19-20
    band = contact_band(inst, radius_px=2)
    assert band[10, 19] and band[10, 20]
    assert not band[10, 18] and not band[10, 21]  # 3 px from the other plant
    assert not band[10, 10] and not band[10, 30]
    inst2 = inst.copy()
    inst2[5:15, 20] = 2  # 1 px gap: the plant border pixels are inside the band too
    band2 = contact_band(inst2, radius_px=2)
    assert band2[10, 18] and band2[10, 20] and not band2[10, 16]
    assert not band[0, 0]


def test_contact_band_empty_for_single_instance() -> None:
    inst = np.zeros((20, 20), np.uint16)
    inst[5:15, 5:15] = 3
    assert not contact_band(inst, 2).any()


def test_contact_band_requires_uint16() -> None:
    with pytest.raises(ValueError, match="uint16"):
        contact_band(np.zeros((4, 4), np.int32))


def test_excess_green_and_ground_veg_labels() -> None:
    rgb = np.zeros((10, 30, 3), np.uint8)
    rgb[:, :10] = (60, 140, 60)  # ExG 160 (green)
    rgb[:, 10:20] = (120, 120, 120)  # ExG 0 (grey soil)
    rgb[:, 20:] = (100, 125, 100)  # ExG 50 (green, inside the vine region below)
    np.testing.assert_allclose(excess_green(rgb, 0)[0, [0, 15, 25]], [160, 0, 50])
    vine = np.zeros((10, 30), bool)
    vine[:, 20:] = True
    c2 = ground_veg_labels(rgb, vine, GroundVegSpec(blur_sigma_px=0))
    assert (c2[:, :10] == 1).all() and (c2[:, 10:20] == 0).all() and (c2[:, 20:] == IGNORE).all()


def test_stack_labels_ignore_sets_all_channels() -> None:
    inst = np.zeros((4, 4), np.uint16)
    inst[1:3, 1:3] = 1
    fg = fg_from_instances(inst)
    ign = np.zeros((4, 4), bool)
    ign[0, 0] = True
    lbl = stack_labels(fg, np.zeros_like(fg), np.ones_like(fg), ign)
    assert lbl.shape == (4, 4, 3) and lbl.dtype == np.uint8
    assert (lbl[0, 0] == IGNORE).all() and lbl[1, 1, 0] == 1 and lbl[3, 3, 2] == 1
