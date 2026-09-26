"""rows_seams: crossing row families, end-to-end joins, duplicate chains (label audit 2026-09-26)."""

from __future__ import annotations

import math

import geopandas as gpd
from shapely.geometry import LineString

from vineyard.geo.tiling import CRS_EPSG
from vineyard.perception.rows_seams import (
    REASON_ORIENTATION_CONFLICT,
    duplicate_pairs,
    groups_of,
    join_plan,
    reject_orientation_conflicts,
)


def _rows(angle_deg: float, n: int, spacing: float, length: float = 50.0) -> list[LineString]:
    a = math.radians(angle_deg)
    u, v = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    out = []
    for k in range(n):
        cx, cy = 25.0 + v[0] * spacing * (k - n / 2), 25.0 + v[1] * spacing * (k - n / 2)
        out.append(LineString([(cx - u[0] * length / 2, cy - u[1] * length / 2),
                               (cx + u[0] * length / 2, cy + u[1] * length / 2)]))
    return out


def _cands(groups: list[tuple[float, int, float]]) -> gpd.GeoDataFrame:
    recs = []
    for g, (angle, n, snr) in enumerate(groups):
        for k, ln in enumerate(_rows(angle, n, 2.5)):
            recs.append({"cand_id": f"t:K{g}{k:02d}", "tile_id": "t", "angle_deg": angle, "snr": snr,
                         "rejected_reason": None, "geometry": ln})
    return gpd.GeoDataFrame(recs, geometry="geometry", crs=CRS_EPSG)


def test_weaker_crossing_family_rejected() -> None:
    out = reject_orientation_conflicts(_cands([(123.0, 8, 250.0), (112.0, 8, 55.0)]), 3.0, 3)
    rejected = out[out.rejected_reason == REASON_ORIENTATION_CONFLICT]
    assert len(rejected) == 8 and set(rejected.cand_id.str[:4]) == {"t:K1"}
    assert out[out.cand_id.str.startswith("t:K0")].rejected_reason.isna().all()


def test_single_family_untouched() -> None:
    cands = _cands([(123.0, 8, 250.0), (124.0, 3, 20.0)])
    assert reject_orientation_conflicts(cands, 3.0, 3).rejected_reason.isna().all()


def test_join_plan_orders_path_and_aligns_seam() -> None:
    a = LineString([(0, 0), (10, 0)])
    b = LineString([(20, 0.2), (10.1, 0.2)])  # reversed, facing a's last end
    c = LineString([(20.1, 0.0), (30, 0.0)])
    plan = join_plan([a, b, c], 0.4, 3.0)
    assert len(plan) == 1
    path, line = plan[0]
    assert set(path) == {0, 1, 2}
    xs = [x for x, _ in line.coords]
    assert xs == sorted(xs) or xs == sorted(xs, reverse=True)
    assert line.length > 29.9


def test_join_refuses_side_by_side_and_far_ends() -> None:
    a = LineString([(0, 0), (10, 0)])
    assert join_plan([a, LineString([(0, 0.3), (10, 0.3)])], 0.4, 3.0) == []  # parallel, not head to tail
    assert join_plan([a, LineString([(10.5, 0), (20, 0)])], 0.4, 3.0) == []


def test_duplicate_pairs_and_groups() -> None:
    lines = [LineString([(0, 0), (40, 0)]), LineString([(2, 0.3), (38, 0.3)]), LineString([(0, 2.5), (40, 2.5)])]
    pairs = duplicate_pairs(lines, 0.4, 0.6, 3.0)
    assert pairs == [(0, 1)]
    assert groups_of(pairs, 3) == [(0, 1)]
