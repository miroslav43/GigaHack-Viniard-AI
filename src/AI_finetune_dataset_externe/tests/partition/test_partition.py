"""Unit tests for the union-preserving canopy partition (synthetic polygons on a real tile frame)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Polygon, box

from fte.canopy.partition import PartitionParams, partition_canopies, partition_polygon
from fte.canopy.partition_profile import local_extrema, smooth

TILE = "siret3_r006_c004"
X0, Y0 = 629196.8, 5220915.2  # tile origin (upper-left) of TILE
CX, CY = X0 + 20.0, Y0 - 20.0
CANOPY_RE = r"^siret3_r\d{3}_c\d{3}:C\d{4}$"


def _dumbbell(length: float = 3.0, width: float = 0.5, neck: float = 0.1, neck_len: float = 0.2) -> Polygon:
    """Two boxes joined by a thin neck at the middle, along x."""
    half = length / 2
    left = box(CX - half, CY - width / 2, CX - neck_len / 2, CY + width / 2)
    right = box(CX + neck_len / 2, CY - width / 2, CX + half, CY + width / 2)
    bridge = box(CX - neck_len / 2 - 0.01, CY - neck / 2, CX + neck_len / 2 + 0.01, CY + neck / 2)
    return shapely.union_all([left, right, bridge]).normalize()


def _axis() -> LineString:
    return LineString([(CX - 10, CY), (CX + 10, CY)])


def _frames(polys: list[Polygon]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    n = len(polys)
    can = gpd.GeoDataFrame({
        "canopy_id": [f"{TILE}:C{k + 1:04d}" for k in range(n)], "tile_id": [TILE] * n,
        "vineyard_id": ["V01"] * n, "row_id": ["V01-R001"] * n, "area_m2": [p.area for p in polys],
        "n_vertices": np.array([len(p.exterior.coords) - 1 for p in polys], dtype="int16"),
        "along_m": np.array([p.bounds[2] - p.bounds[0] for p in polys], dtype="float32"),
        "is_clump": [False] * n, "touches_edge": [False] * n, "confidence": np.ones(n, dtype="float32"),
        "qa_flags": [""] * n,
    }, geometry=polys, crs="EPSG:32635")
    rows = gpd.GeoDataFrame({"piece_id": [f"V01-R001@{TILE}"], "row_id": ["V01-R001"], "tile_id": [TILE]},
                            geometry=[_axis()], crs="EPSG:32635")
    return can, rows


def _tile_ctx(contact: np.ndarray | None = None):
    from fte.canopy.partition import TileContext

    return TileContext(TILE, X0, Y0, contact)


def test_smooth_and_extrema() -> None:
    v = np.array([3.0, 2.0, 1.0, 2.0, 3.0, 3.0])
    assert list(local_extrema(v, minima=True)) == [2]
    assert list(local_extrema(-v, minima=False)) == [2]
    assert np.allclose(smooth(np.ones(5), 3), 1.0)


def test_dumbbell_cut_at_neck_and_union_preserved() -> None:
    poly = _dumbbell()
    pieces = partition_polygon(poly, _axis(), PartitionParams(alpha=0.55), _tile_ctx())
    assert len(pieces) == 2
    cut_x = max(p.bounds[0] for p in pieces)
    assert abs(cut_x - CX) < 0.15
    assert abs(sum(p.area for p in pieces) - poly.area) < 1e-6
    assert abs(shapely.union_all(pieces).area - poly.area) < 1e-6
    assert shapely.union_all(pieces).symmetric_difference(poly).area < 1e-6


def test_short_or_convex_polygons_are_not_cut() -> None:
    params = PartitionParams(alpha=0.55)
    short = _dumbbell(length=1.4)
    assert partition_polygon(short, _axis(), params, _tile_ctx()) == [short]
    convex = box(CX - 2, CY - 0.25, CX + 2, CY + 0.25)
    assert partition_polygon(convex, _axis(), params, _tile_ctx()) == [convex]


def test_min_piece_length_and_area_constraints() -> None:
    # neck close to one end: the short piece would be < min_piece_len_m -> no cut
    half = 1.0
    left = box(CX - half, CY - 0.25, CX - half + 0.4, CY + 0.25)
    right = box(CX - half + 0.5, CY - 0.25, CX + half, CY + 0.25)
    bridge = box(CX - half + 0.39, CY - 0.03, CX - half + 0.51, CY + 0.03)
    poly = shapely.union_all([left, right, bridge])
    assert len(partition_polygon(poly, _axis(), PartitionParams(alpha=0.55), _tile_ctx())) == 1
    # min area too large for the halves -> no cut
    big = PartitionParams(alpha=0.55, min_area_m2=0.8)  # halves are 0.7 m2
    assert len(partition_polygon(_dumbbell(), _axis(), big, _tile_ctx())) == 1


def test_nn_mode_uses_contact_map() -> None:
    poly = box(CX - 1.5, CY - 0.25, CX + 1.5, CY + 0.25)  # no narrowing at all
    contact = np.zeros((2048, 2048), dtype=np.uint8)
    col = int((CX - X0) / 0.025)
    contact[:, col - 2: col + 3] = 230
    pieces = partition_polygon(poly, _axis(), PartitionParams(mode="nn", t_c=0.5), _tile_ctx(contact))
    assert len(pieces) == 2
    assert abs(max(p.bounds[0] for p in pieces) - CX) < 0.1
    none = partition_polygon(poly, _axis(), PartitionParams(mode="nn", t_c=0.5), _tile_ctx(None))
    assert none == [poly]


def test_frame_ids_valid_unique_and_input_untouched() -> None:
    polys = [_dumbbell(), box(CX - 0.3, CY + 3 - 0.2, CX + 0.3, CY + 3 + 0.2)]
    can, rows = _frames(polys)
    before = can.copy(deep=True)
    out = partition_canopies(can, rows, PartitionParams(alpha=0.55))
    assert len(out) == 3
    assert list(out.columns) == list(can.columns)
    assert out.dtypes.to_dict() == can.dtypes.to_dict()
    assert out["canopy_id"].is_unique
    assert out["canopy_id"].str.match(CANOPY_RE).all()
    assert set(out["row_id"]) == {"V01-R001"}
    assert np.allclose(out["area_m2"], out.geometry.area)
    union_delta = abs(shapely.union_all(out.geometry.values).area - shapely.union_all(can.geometry.values).area)
    assert union_delta < 1e-6
    assert can.equals(before)


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        PartitionParams(mode="bogus")
    with pytest.raises(ValueError):
        PartitionParams.from_mapping({"alpha": 0.5, "nope": 1})
    assert PartitionParams.from_json('{"mode":"width","alpha":0.35}').alpha == 0.35
