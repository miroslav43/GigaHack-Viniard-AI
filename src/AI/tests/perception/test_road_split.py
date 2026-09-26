from __future__ import annotations

import pytest
from shapely.geometry import LineString

from vineyard.perception.road_split import RoadSplitOptions, gap_interval, road_cuts, road_lines

X0, Y0 = 629000.0, 5220000.0
OPTS = RoadSplitOptions(enabled=True, min_rows=3, min_gap_frac=0.6, search_m=4.0, window_half_m=1.0,
                     gap_share_max=0.12, gap_rel_max=0.35, max_cut_m=12.0, min_line_m=10.0,
                        min_side_m=10.0)
TRACK_X = (X0 + 48.0, X0 + 52.0)  # the real, vine-free track (the OSM line below is drawn 2 m off)


def rows(n: int = 5) -> list[LineString]:
    return [LineString([(X0, Y0 + 2.5 * k), (X0 + 100, Y0 + 2.5 * k)]) for k in range(n)]


def evidence_with_track(track: tuple[float, float] | None):
    """Veg share of a row band: 0 on the track, 0.6 elsewhere (a window straddling the edge gets the mix)."""
    def ev(line: LineString, _offset: float) -> float | None:
        x0, x1 = sorted((line.coords[0][0], line.coords[-1][0]))
        if track is None or x1 <= track[0] or x0 >= track[1]:
            return 0.6
        inside = min(x1, track[1]) - max(x0, track[0])
        return 0.6 * (1 - inside / max(x1 - x0, 1e-9))
    return ev


def across(x: float) -> LineString:
    return LineString([(x, Y0 - 20), (x, Y0 + 40)])


def test_gap_interval_finds_the_track_near_an_offset_crossing() -> None:
    a, b = gap_interval(rows(1)[0], 46.0, evidence_with_track(TRACK_X), OPTS)
    assert 46.0 <= a <= 49.0 and 51.0 <= b <= 54.0


def test_no_gap_without_a_vine_free_stretch() -> None:
    assert gap_interval(rows(1)[0], 50.0, evidence_with_track(None), OPTS) is None


def test_a_road_over_a_vine_free_track_cuts_every_crossing_row() -> None:
    lines = rows(5)
    (cut,) = road_cuts(lines, range(5), [across(X0 + 50.0)], evidence_with_track(TRACK_X), OPTS)
    assert cut.rows == (0, 1, 2, 3, 4) and cut.n_crossing == 5
    for line in lines:
        rest = line.difference(cut.polygon)
        assert len(rest.geoms) == 2 and all(p.length > 40 for p in rest.geoms)
    assert cut.barrier.intersects(LineString([(X0 + 50, Y0), (X0 + 50, Y0 + 10)]))


def test_a_road_over_vines_cuts_nothing() -> None:
    assert road_cuts(rows(5), range(5), [across(X0 + 50.0)], evidence_with_track(None), OPTS) == ()


def test_too_few_crossing_rows_or_not_crossing() -> None:
    assert road_cuts(rows(2), range(2), [across(X0 + 50.0)], evidence_with_track(TRACK_X), OPTS) == ()
    beside = LineString([(X0 - 10, Y0 - 20), (X0 - 10, Y0 + 40)])
    assert road_cuts(rows(5), range(5), [beside], evidence_with_track(TRACK_X), OPTS) == ()


def test_road_lines_explode_filter_and_sort() -> None:
    from shapely.geometry import MultiLineString
    multi = MultiLineString([[(0, 0), (20, 0)], [(0, 5), (3, 5)]])
    out = road_lines([multi, LineString([(-5, 0), (-5, 30)])], 10.0)
    assert [round(ln.length) for ln in out] == [30, 20]


@pytest.mark.parametrize("enabled", [True, False])
def test_options_from_config(enabled: bool) -> None:
    from vineyard.config import load_config
    cfg = load_config(overrides=[f"blocks.road_split.enabled={str(enabled).lower()}"])
    assert RoadSplitOptions.from_config(cfg.blocks.road_split).enabled is enabled


def test_road_stubs_are_dropped_next_to_the_cut() -> None:
    from vineyard.perception.blocks import drop_road_stubs
    from vineyard.perception.blocks_graph import RowUnit
    lines = rows(3)
    (cut,) = road_cuts(lines, range(3), [across(X0 + 50.0)], evidence_with_track(TRACK_X), OPTS)
    stub = RowUnit(0, LineString([(X0 + 53, Y0), (X0 + 58, Y0)]), (), True)
    far = RowUnit(1, LineString([(X0 + 80, Y0), (X0 + 85, Y0)]), (), True)
    long = RowUnit(2, LineString([(X0 + 53, Y0 + 2.5), (X0 + 100, Y0 + 2.5)]), (), True)
    uncut = RowUnit(3, LineString([(X0 + 53, Y0 + 5), (X0 + 58, Y0 + 5)]), (), False)
    kept = drop_road_stubs([stub, far, long, uncut], [cut], 10.0)
    assert [u.src for u in kept] == [1, 2, 3]
    assert drop_road_stubs([stub], [], 10.0) == (stub,)
