"""Cross-paths (tracks across the rows): detection, gap subtraction, targets, walk graph, web bundle, config."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from shapely.geometry import LineString, Point, box

from tests.post.passable_factories import mini_vineyard
from vineyard.config import load_config
from vineyard.contracts.enums import EdgeKind, TargetKind
from vineyard.errors import ConfigError
from vineyard.route.cross_paths import (
    CrossPathParams,
    block_frame,
    detect_cross_paths,
    paths_frame,
    subtract_paths,
)
from vineyard.route.target_gaps import RowGap
from vineyard.route.target_rules import RowContext, gap_drafts, missing_plant_drafts
from vineyard.route.targets import COUNT_CROSS_PATH, _without_cross_paths
from vineyard.route.walk_build import PassableInputs, PassableParams, build_passable, cross_path_drafts
from vineyard.web.bundle import cross_path_features

SPACING = 2.6
LENGTH = 80.0


def _params(**kw) -> CrossPathParams:
    base = dict(min_gap_m=1.5, max_gap_m=15.0, max_width_m=8.0, min_rows=6, min_segment_rows=4, link_min_m=1.2,
                link_max_m=4.5, overlap_tol_m=0.3, max_residual_m=1.5, end_extend_spacing=0.75)
    return CrossPathParams(**{**base, **kw})


def _row(k: int, gaps: list[tuple[float, float]], *, vid: str = "V01", y0: float = 0.0,
         length: float = LENGTH) -> RowContext:
    """Horizontal row k (y = y0 + k * SPACING) with interior gaps at the given along-axis spans."""
    axis = LineString([(0.0, y0 + k * SPACING), (length, y0 + k * SPACING)])
    return RowContext(row_id=f"{vid}-R{k:03d}", vineyard_id=vid, row_index=k, axis=axis,
                      gaps=tuple(RowGap(a, b, "interior") for a, b in gaps), unknown=(),
                      head_on_boundary=False, tail_on_boundary=False)


def _track(n: int, centre: float = 40.0, width: float = 4.0, slope: float = 0.0) -> list[RowContext]:
    rows = []
    for k in range(1, n + 1):
        c = centre + slope * k * SPACING
        rows.append(_row(k, [(c - width / 2, c + width / 2)]))
    return rows


# ------------------------------------------------------------------ detection


def test_straight_track_across_ten_rows_is_one_path() -> None:
    paths = detect_cross_paths(_track(10), _params())
    assert len(paths) == 1
    p = paths[0]
    assert (p.path_id, p.vineyard_id, p.n_rows) == ("XP001", "V01", 10)
    assert p.width_m == pytest.approx(4.0, abs=1e-6)
    assert p.residual_m == pytest.approx(0.0, abs=1e-6)
    # 9 spacings between the outer crossings + 0.75 spacing past each of them
    assert p.length_m == pytest.approx(10.5 * SPACING, abs=1e-6)
    for k in range(1, 11):
        assert p.polygon.buffer(1e-6).covers(Point(40.0, k * SPACING))
    assert p.row_ids == tuple(f"V01-R{k:03d}" for k in range(1, 11))


def test_oblique_track_width_is_measured_across_it() -> None:
    (p,) = detect_cross_paths(_track(8, slope=1.0), _params())
    assert p.n_rows == 8
    assert p.width_m == pytest.approx(4.0 / math.sqrt(2.0), abs=1e-6)


def test_too_few_rows_scattered_gaps_and_wide_patches_are_not_paths() -> None:
    assert detect_cross_paths(_track(5), _params()) == ()
    rng = np.random.default_rng(7)
    scattered = [_row(k, [(float(c), float(c) + 3.0)]) for k, c in enumerate(rng.uniform(5, 70, 12), start=1)]
    assert detect_cross_paths(scattered, _params()) == ()
    assert detect_cross_paths(_track(8, width=10.0), _params()) == ()
    assert len(detect_cross_paths(_track(8, width=10.0), _params(max_width_m=12.0))) == 1


def test_short_gaps_below_min_gap_do_not_chain() -> None:
    assert detect_cross_paths(_track(10, width=1.0), _params()) == ()


def test_bent_track_is_one_polyline() -> None:
    rows = []
    for k in range(1, 15):
        c = 40.0 + (0.0 if k <= 7 else (k - 7) * SPACING * 0.8)
        rows.append(_row(k, [(c - 2.0, c + 2.0)]))
    (p,) = detect_cross_paths(rows, _params())
    assert p.n_rows == 14 and len(p.centerline.coords) == 5   # 2 extended ends + first, bend, last
    assert p.residual_m <= 1.5


def test_wiggle_splits_a_chain_into_straight_pieces() -> None:
    rows = _track(14)
    rows[6] = _row(7, [(39.0, 43.0)])   # one crossing 1 m off the line
    assert len(detect_cross_paths(rows, _params(max_residual_m=0.5))) == 2  # 6 + 7 rows around the wiggle
    assert len(detect_cross_paths(rows, _params(max_residual_m=1.5))) == 1


def test_fragmented_rows_link_by_distance_not_row_index() -> None:
    rows = _track(8)
    # a short collinear fragment of row 4 (0.3 m off) with its own id and index, far from the track
    frag = RowContext("V01-R900", "V01", 100, LineString([(60.0, 4 * SPACING + 0.3), (75.0, 4 * SPACING + 0.3)]),
                      (), (), False, False)
    # a row missing from the detection between rows 5 and 6: spacing 1.6x
    shifted = [r if r.row_index <= 5 else _row(r.row_index, [(38.0, 42.0)], y0=0.6 * SPACING) for r in rows]
    (p,) = detect_cross_paths([*shifted, frag], _params())
    assert p.n_rows == 8


def test_two_blocks_and_ids_are_deterministic() -> None:
    a = _track(7)
    b = [_row(k, [(20.0, 24.0)], vid="V02", y0=100.0) for k in range(1, 8)]
    paths = detect_cross_paths([*b, *a], _params())
    assert [(p.path_id, p.vineyard_id) for p in paths] == [("XP001", "V01"), ("XP002", "V02")]
    again = detect_cross_paths([*reversed(a), *reversed(b)], _params())
    assert [p.polygon.wkt for p in again] == [p.polygon.wkt for p in paths]


def test_block_frame_aligns_opposite_directions() -> None:
    f = block_frame([LineString([(0, 0), (10, 0)]), LineString([(10, 3), (0, 3)])])
    assert np.allclose(np.abs(f.u), [1.0, 0.0]) and np.allclose(np.abs(f.v), [0.0, 1.0])


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        _params(min_gap_m=5.0, max_gap_m=4.0)
    with pytest.raises(ValueError):
        _params(min_segment_rows=8)
    with pytest.raises(ValueError):
        _params(link_min_m=5.0)


# ------------------------------------------------------------------ gaps minus paths / targets


def test_subtract_paths_keeps_the_rest_of_a_longer_gap_and_row_ends() -> None:
    ctx = _row(1, [(30.0, 42.0)])
    ctx = RowContext(ctx.row_id, ctx.vineyard_id, 1, ctx.axis, (RowGap(0.0, 3.0, "head"), *ctx.gaps), (), False,
                     False)
    strip = box(38.0, -5.0, 42.0, 10.0)
    out, removed = subtract_paths(ctx, strip)
    assert removed == pytest.approx(4.0)
    assert [(g.start_m, g.end_m, g.kind) for g in out.gaps] == [(0.0, 3.0, "head"), (30.0, 38.0, "interior")]
    assert subtract_paths(ctx, box(0, 100, 1, 101)) == (ctx, 0.0)
    assert subtract_paths(ctx, None) == (ctx, 0.0)


def test_targets_inside_a_track_are_dropped_and_counted() -> None:
    cfg = load_config(environ={}).targets
    rows = _track(8, width=6.0)
    rows[2] = _row(3, [(37.0, 43.0), (60.0, 63.0)])  # a real missing-plant gap elsewhere in the row
    (p,) = detect_cross_paths(rows, _params())
    before = [d for r in rows for d in (*gap_drafts(r, cfg), *missing_plant_drafts(r, cfg))]
    assert sum(d.kind is TargetKind.ROW_GAP for d in before) == 8
    trimmed, counts = _without_cross_paths(rows, cfg, p.polygon, box(-10, -10, 100, 100))
    after = [d for r in trimmed for d in (*gap_drafts(r, cfg), *missing_plant_drafts(r, cfg))]
    assert [(d.kind, d.row_id) for d in after] == [(TargetKind.MISSING_PLANT, "V01-R003")]
    assert counts[f"{COUNT_CROSS_PATH}_row_gap"] == 8 and counts[f"{COUNT_CROSS_PATH}_rows"] == 8
    assert _without_cross_paths(rows, cfg, None, box(0, 0, 1, 1)) == (tuple(rows), {})


# ------------------------------------------------------------------ walk graph


def test_cross_path_joins_every_interrow_centerline_it_crosses() -> None:
    mini = mini_vineyard(n_rows=6, length_m=60.0, spacing_m=SPACING)
    ys = sorted(float(g.coords[0][1]) for g in mini.rows.geometry)
    line = LineString([(1030.0, ys[0] - 1.5), (1030.0, ys[-1] + 1.5)])
    inputs = PassableInputs(rows=mini.rows, pieces=mini.interrow_pieces, canopies=(), passages=mini.passages,
                            forbidden=None, start_xy=mini.start_xy, cross_lines=(("XP001", line),))
    res = build_passable(inputs, PassableParams.from_config(load_config(environ={}).route))
    g = res.graph
    cross = [i for i, k in enumerate(g.edge_kind) if k == EdgeKind.CROSS_PATH]
    assert len(res.lines.crossings) == 2 * 5     # 5 interrows crossed, one attachment on each line
    nodes = {int(n) for i in cross for n in g.edge_uv[i]}
    ir_nodes = {int(n) for i, k in enumerate(g.edge_kind) if k == EdgeKind.INTERROW_CENTERLINE for n in g.edge_uv[i]}
    assert len(nodes & ir_nodes) == 5
    assert g.edge_len_m[cross].sum() == pytest.approx(line.length, abs=1e-6)


def test_cross_path_drafts_without_crossings() -> None:
    drafts, attach = cross_path_drafts((), (("XP001", LineString([(0, 0), (1, 1)])),))
    assert [d.kind for d in drafts] == [EdgeKind.CROSS_PATH] and attach == []


def test_passable_option_adds_the_strip_to_the_domain() -> None:
    mini = mini_vineyard(n_rows=4, length_m=60.0, spacing_m=SPACING)
    strip = box(1028.0, 1995.0, 1032.0, 2015.0)
    params = PassableParams.from_config(load_config(environ={}).route)
    base = PassableInputs(rows=mini.rows, pieces=mini.interrow_pieces, canopies=(), passages=mini.passages,
                          forbidden=None, start_xy=mini.start_xy)
    with_strip = PassableInputs(rows=mini.rows, pieces=mini.interrow_pieces, canopies=(), passages=mini.passages,
                                forbidden=None, start_xy=mini.start_xy, cross_domain=strip)
    a = build_passable(base, params).lines.domain.raw.area
    b = build_passable(with_strip, params).lines.domain.raw.area
    assert b > a


# ------------------------------------------------------------------ layer / web bundle / config


def test_layer_and_web_features() -> None:
    paths = detect_cross_paths(_track(7), _params())
    frame = paths_frame(paths, {"source": "model", "run_id": "r", "model_version": "m", "confidence": 1.0,
                                "qa_flags": ""})
    assert list(frame.columns[:7]) == ["path_id", "vineyard_id", "n_rows", "width_m", "length_m", "residual_m",
                                       "row_ids"]
    assert frame.crs.to_epsg() == 32635
    web = cross_path_features(frame)
    assert list(web.columns) == ["id", "vineyard_id", "n_rows", "width_m", "length_m", "geometry"]
    assert web.iloc[0]["id"] == "XP001" and web.iloc[0]["n_rows"] == 7
    assert cross_path_features(None) is None
    assert shapely.equals(web.geometry.iloc[0], paths[0].polygon)


def test_config_section_and_validators() -> None:
    cfg = load_config(environ={}).cross_paths
    assert cfg.enabled and cfg.drop_targets and cfg.route and not cfg.passable
    assert CrossPathParams.from_config(cfg).min_rows == cfg.min_rows
    with pytest.raises(ConfigError):
        load_config(overrides=("cross_paths.link_min_m=9.0",), environ={})
