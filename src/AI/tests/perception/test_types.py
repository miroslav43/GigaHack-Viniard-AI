"""Cross-package perception types: TileStats (tile_prep cache) and TileMasks."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from vineyard.contracts.enums import TileStatus
from vineyard.perception.types import (
    VIS_NODATA,
    VIS_OK,
    VIS_OVEREXPOSED,
    VIS_SHADOW,
    TileMasks,
    TileStats,
    readonly,
)


def _stats(**kw: object) -> TileStats:
    base: dict[str, object] = {
        "tile_id": "siret3_r021_c012", "valid_frac": 0.9, "nodata_frac": 0.1, "veg_frac": 0.12,
        "status": TileStatus.OK, "veg_threshold": 4.0, "veg_method": "lab_a",
    }
    return TileStats(**{**base, **kw})  # type: ignore[arg-type]


def test_vis_codes_constants() -> None:
    assert (VIS_OK, VIS_SHADOW, VIS_OVEREXPOSED, VIS_NODATA) == (0, 1, 2, 3)


def test_tile_stats_roundtrip_dict() -> None:
    s = _stats()
    d = s.to_dict()
    assert d["status"] == "ok" and isinstance(d["status"], str)
    assert TileStats.from_dict(d) == s
    assert TileStats.from_dict({**d, "status": "empty_nodata"}).status is TileStatus.EMPTY_NODATA


def test_tile_stats_frozen() -> None:
    s = _stats()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.veg_frac = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        {"valid_frac": 1.2},
        {"nodata_frac": -0.1},
        {"veg_frac": float("nan")},
        {"status": "weird"},
        {"tile_id": ""},
        {"veg_threshold": float("inf")},
    ],
)
def test_tile_stats_validation(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _stats(**bad)


def test_tile_stats_from_dict_requires_exact_keys() -> None:
    d = _stats().to_dict()
    with pytest.raises(ValueError, match="missing"):
        TileStats.from_dict({k: v for k, v in d.items() if k != "veg_frac"})
    with pytest.raises(ValueError, match="unexpected"):
        TileStats.from_dict({**d, "extra": 1})


def test_readonly_is_a_view_and_leaves_input_writable() -> None:
    a = np.zeros((3, 3), np.uint8)
    r = readonly(a)
    assert not r.flags.writeable
    assert a.flags.writeable
    with pytest.raises(ValueError):
        r[0, 0] = 1


def test_tile_masks_validates_and_freezes_arrays() -> None:
    veg = np.zeros((8, 8), bool)
    vis = np.zeros((8, 8), np.uint8)
    m = TileMasks(veg=veg, vis=vis, method="lab_a", threshold=4.0, veg_frac=0.0)
    assert not m.veg.flags.writeable and not m.vis.flags.writeable
    assert veg.flags.writeable
    with pytest.raises(ValueError):
        TileMasks(veg=veg.astype(np.uint8), vis=vis, method="lab_a", threshold=4.0, veg_frac=0.0)
    with pytest.raises(ValueError):
        TileMasks(veg=veg, vis=np.zeros((4, 4), np.uint8), method="lab_a", threshold=4.0, veg_frac=0.0)
    with pytest.raises(ValueError):
        TileMasks(veg=veg, vis=vis.astype(np.int32), method="lab_a", threshold=4.0, veg_frac=0.0)
    with pytest.raises(ValueError):
        TileMasks(veg=veg, vis=vis, method="lab_a", threshold=4.0, veg_frac=2.0)
