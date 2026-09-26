"""row_regularize: headland trims, lattice removals, strays and flags in the row frame (RC8)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString

from vineyard.config import load_config
from vineyard.geo.tiling import GRID_ORIGIN_X, GRID_ORIGIN_Y, TILE_PX
from vineyard.perception.row_evidence import VegEvidence
from vineyard.perception.row_regularize import (
    ACTION_EXTEND,
    ACTION_OFF_LATTICE,
    ACTION_STRAY,
    ACTION_TRIM,
    ACTION_UNDERSHOOT,
    FLAG_OFF_LATTICE,
    FLAG_OVERSHOOT,
    RegularizeOptions,
    apply_span,
    fit_headland,
    huber_line,
    lattice_lines,
    lattice_residuals,
    lattice_spacing,
    regularize_block,
    row_contrast,
    row_frame,
    span_on_line,
)

S = 2.5
OPTS = RegularizeOptions.from_config(load_config().blocks.regularize)


def _rows(n: int = 12, length: float = 60.0, spacing: float = S) -> list[LineString]:
    return [LineString([(0.0, k * spacing), (length, k * spacing)]) for k in range(n)]


def _vines(x0: float, x1: float, spacing: float = S, rows: int = 40) -> object:
    """Evidence: vines on the lattice rows for x in [x0, x1], bare soil elsewhere."""

    def ev(line: LineString, offset_m: float) -> float:
        c = np.asarray(line.coords)
        y = float(c[:, 1].mean()) + offset_m * (1.0 if c[-1, 0] >= c[0, 0] else -1.0)
        on_row = abs(y / spacing - round(y / spacing)) * spacing < 0.3 and -0.5 <= round(y / spacing) < rows
        xs = np.linspace(c[:, 0].min(), c[:, 0].max(), 50)
        inside = (xs >= x0) & (xs <= x1)
        return float(inside.mean() * 0.8) if on_row else 0.0

    return ev


def _by_action(changes: tuple) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for c in changes:
        out.setdefault(c.action, []).append(c.index)
    return out


def test_config_defaults_load() -> None:
    assert OPTS.enabled
    assert OPTS.max_segments == 3
    assert not OPTS.extend_enabled


def test_regular_block_is_left_unchanged() -> None:
    assert regularize_block(_rows(), OPTS, _vines(0.0, 60.0)) == ()


def test_disabled_or_small_block_returns_nothing() -> None:
    off = dataclasses.replace(OPTS, enabled=False)
    assert regularize_block(_rows(), off, _vines(0.0, 60.0)) == ()
    assert regularize_block(_rows(3), OPTS, _vines(0.0, 60.0)) == ()


def test_overshoot_on_bare_soil_is_trimmed_to_headland() -> None:
    rows = _rows()
    rows[5] = LineString([(0.0, 5 * S), (70.0, 5 * S)])  # 10 m past the x=60 headland, bare soil there
    changes = regularize_block(rows, OPTS, _vines(0.0, 60.0))
    trims = [c for c in changes if c.action == ACTION_TRIM]
    assert [c.index for c in trims] == [5]
    assert trims[0].hi == pytest.approx(60.0 + OPTS.trim_margin_m, abs=0.3)
    assert apply_span(rows[5], trims[0].lo, trims[0].hi).length == pytest.approx(60.5, abs=0.3)


def test_overshoot_with_vines_is_only_flagged() -> None:
    rows = _rows()
    rows[5] = LineString([(0.0, 5 * S), (70.0, 5 * S)])
    changes = regularize_block(rows, OPTS, _vines(0.0, 80.0))
    assert _by_action(changes) == {FLAG_OVERSHOOT: [5]}


def test_reversed_row_start_overshoot_is_trimmed_at_the_right_end() -> None:
    rows = _rows()
    rows[4] = LineString([(60.0, 4 * S), (-8.0, 4 * S)])  # reversed, 8 m past the x=0 headland
    changes = regularize_block(rows, OPTS, _vines(0.0, 60.0))
    trim = next(c for c in changes if c.action == ACTION_TRIM)
    kept = apply_span(rows[4], trim.lo, trim.hi)
    assert min(x for x, _ in kept.coords) == pytest.approx(-OPTS.trim_margin_m, abs=0.3)


def test_unknown_evidence_flags_instead_of_trimming() -> None:
    rows = _rows()
    rows[5] = LineString([(0.0, 5 * S), (70.0, 5 * S)])
    changes = regularize_block(rows, OPTS, lambda line, off: None)
    assert _by_action(changes) == {FLAG_OVERSHOOT: [5]}


def test_undershoot_is_flagged_and_extended_only_when_enabled() -> None:
    rows = _rows()
    rows[6] = LineString([(0.0, 6 * S), (50.0, 6 * S)])  # 10 m short of the headland
    assert _by_action(regularize_block(rows, OPTS, _vines(0.0, 60.0))) == {ACTION_UNDERSHOOT: [6]}
    ext = dataclasses.replace(OPTS, extend_enabled=True)
    changes = regularize_block(rows, ext, _vines(0.0, 60.0))
    assert _by_action(changes) == {ACTION_EXTEND: [6]}
    c = changes[0]
    assert apply_span(rows[6], c.lo, c.hi).length == pytest.approx(60.0, abs=0.3)
    weak = regularize_block(rows, ext, _vines(0.0, 50.0))
    assert _by_action(weak) == {ACTION_UNDERSHOOT: [6]}


def test_short_off_lattice_row_is_removed_and_long_supported_one_flagged() -> None:
    rows = _rows()
    rows.append(LineString([(10.0, 5.5 * S), (20.0, 5.5 * S)]))  # tillage furrow between rows 5 and 6
    assert _by_action(regularize_block(rows, OPTS, _vines(0.0, 60.0))).get(ACTION_OFF_LATTICE) == [12]
    rows[-1] = LineString([(0.0, 5.5 * S), (60.0, 5.5 * S)])

    def green(line: LineString, off: float) -> float:
        return 0.8 if abs(off) < 1e-9 else 0.1

    assert _by_action(regularize_block(rows, OPTS, green)).get(FLAG_OFF_LATTICE) == [12]


def test_short_edge_row_without_overlap_is_removed_as_stray() -> None:
    rows = _rows()
    rows.append(LineString([(70.0, 12 * S), (75.0, 12 * S)]))
    changes = regularize_block(rows, OPTS, _vines(0.0, 80.0))
    assert _by_action(changes).get(ACTION_STRAY) == [12]


def test_step_in_the_headland_is_a_segment_not_an_overshoot() -> None:
    rows = [LineString([(0.0, k * S), (60.0 if k < 8 else 80.0, k * S)]) for k in range(16)]
    assert regularize_block(rows, OPTS, _vines(0.0, 80.0)) == ()
    head = fit_headland(np.arange(16) * S, np.array([60.0] * 8 + [80.0] * 8), OPTS)
    assert head.breaks == (8,)
    assert np.allclose(head.pred, [60.0] * 8 + [80.0] * 8, atol=0.1)


def test_huber_line_ignores_an_outlier_and_handles_degenerate_input() -> None:
    x = np.arange(10.0)
    y = 2.0 * x + 1.0
    y[4] += 30.0
    b0, b1 = huber_line(x, y, 1.0)
    assert b1 == pytest.approx(2.0, abs=0.1)
    assert b0 == pytest.approx(1.0, abs=0.5)
    assert huber_line(np.array([3.0]), np.array([5.0]), 1.0) == (5.0, 0.0)


def test_lattice_helpers() -> None:
    v = np.array([0.0, 2.5, 5.0, 10.0, 12.5, 15.0])  # one skip-one gap
    assert lattice_spacing(v) == pytest.approx(2.5)
    assert np.isnan(lattice_spacing(np.array([1.0])))
    res, loc = lattice_residuals(np.array([0.0, 2.5, 5.0, 6.3, 7.5, 10.0, 12.5]), 2.5, 3)
    assert abs(res[3]) > 1.0
    assert np.all(np.abs(np.delete(res, 3)) < 0.3)
    assert loc[3] == pytest.approx(2.5)
    r0, _ = lattice_residuals(np.array([0.0, 2.5]), 2.5, 3)
    assert np.all(r0 == 0.0)


def test_lattice_lines_group_collinear_pieces() -> None:
    rows = [LineString([(0, 0), (20, 0)]), LineString([(25, 0.1), (60, 0.1)]), LineString([(0, 2.5), (60, 2.5)])]
    groups = lattice_lines(rows, row_frame(rows), 0.6)
    assert len(groups) == 2
    assert sorted(len(g.members) for g in groups) == [1, 2]


def test_span_on_line_and_apply_span_extrapolate() -> None:
    line = LineString([(10.0, 0.0), (0.0, 0.0)])
    fr = row_frame([LineString([(0.0, 0.0), (10.0, 0.0)])])
    lo, hi = span_on_line(line, fr, 2.0, 4.0)  # frame origin at x=5: u in [2, 4] is x in [7, 9]
    assert (lo, hi) == pytest.approx((1.0, 3.0))
    longer = apply_span(LineString([(0, 0), (1, 0), (10, 0)]), -2.0, 12.0)
    assert longer.length == pytest.approx(14.0)
    assert apply_span(line, 2.0, 5.0).length == pytest.approx(3.0)


def test_row_contrast_none_when_unknown() -> None:
    line = LineString([(0, 0), (10, 0)])
    assert row_contrast(lambda ln, off: None, line, S) is None
    assert row_contrast(lambda ln, off: 0.5 if off == 0 else None, line, S) is None
    assert row_contrast(lambda ln, off: 0.6 if off == 0 else 0.2, line, S) == pytest.approx(0.4)


def test_veg_evidence_reads_masks_per_tile(tmp_path: Path) -> None:
    veg = np.zeros((TILE_PX, TILE_PX), dtype=bool)
    veg[:, :400] = True  # the first 10 m of the tile are green
    valid = np.ones_like(veg)
    loads: list[str] = []

    def load_veg(_: Path, tile_id: str) -> np.ndarray:
        loads.append(tile_id)
        if tile_id != "siret3_r000_c000":
            raise OSError("missing")
        return veg

    ev = VegEvidence(tmp_path, 0.3, 0.1, load_veg, lambda _p, _t: valid)
    y = GRID_ORIGIN_Y - 20.0
    green = LineString([(GRID_ORIGIN_X + 1.0, y), (GRID_ORIGIN_X + 8.0, y)])
    bare = LineString([(GRID_ORIGIN_X + 20.0, y), (GRID_ORIGIN_X + 30.0, y)])
    assert ev(green, 0.0) == pytest.approx(1.0)
    assert ev(bare, 1.0) == pytest.approx(0.0)
    outside = LineString([(GRID_ORIGIN_X + 60.0, y), (GRID_ORIGIN_X + 70.0, y)])
    assert ev(outside, 0.0) is None
    assert ev(LineString([(GRID_ORIGIN_X + 1.0, y), (GRID_ORIGIN_X + 1.0 + 1e-12, y)]), 0.0) is None
    assert loads.count("siret3_r000_c000") == 1
