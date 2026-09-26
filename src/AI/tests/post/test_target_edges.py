"""build_targets: must/optional roles from config, edge-margin artefacts, outermost rows, fragmented rows."""

from __future__ import annotations

import geopandas as gpd
import shapely
from shapely.geometry import LineString, box

from tests.post.factories import BlockSpec, make_annset
from tests.post.tgt_helpers import reference_gap_fn, rows_from_pieces, tile_coverage
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.geo.tiling import CRS_EPSG
from vineyard.route.target_rules import end_samples
from vineyard.route.targets import TargetInputs, TargetProvenance, TargetSettings, build_targets

PROV = TargetProvenance(Source.REFERENCE, "20260926T1200-post-abcdef", "reference-examples@test")
TILE = "siret3_r018_c011"


def _settings(*overrides: str) -> TargetSettings:
    return TargetSettings.from_config(load_config(overrides=overrides, environ={}))


def _run(ann, *overrides: str, rows: gpd.GeoDataFrame | None = None, coverage=None):
    inputs = TargetInputs(rows=rows_from_pieces(ann) if rows is None else rows, canopies=ann.canopies,
                          waste=ann.waste, coverage=tile_coverage(ann) if coverage is None else coverage)
    return build_targets(inputs, _settings(*overrides), PROV, reference_gap_fn)


def _roles(result) -> dict[str, set[str]]:
    t = result.targets
    return {k: set(t.loc[t["kind"] == k, "route_role"]) for k in sorted(set(t["kind"]))}


# ------------------------------------------------------------------ roles


def _mixed_block():
    spec = BlockSpec(n_rows=3, length_m=(60.0, 40.0, 60.0), gaps=((1, 20.0, 26.0), (3, 30.0, 33.0)))
    waste = [(TILE, tuple(spec.point(1, 10.0, -1.25)), 0.5, "V01")]
    return make_annset(spec, waste=waste)


def test_route_role_follows_the_configured_must_kinds():
    result = _run(_mixed_block())
    assert _roles(result) == {"missing_plant": {"optional"}, "row_end_short": {"optional"},
                              "row_gap": {"must"}, "waste": {"must"}}


def test_must_kinds_override_changes_the_roles():
    result = _run(_mixed_block(), "targets.must_kinds=[row_gap]")
    assert _roles(result)["waste"] == {"optional"} and _roles(result)["row_gap"] == {"must"}


# ------------------------------------------------------------------ edge margin


def _hole_case():
    spec = BlockSpec(gaps=((2, 20.0, 26.0),))
    gap_xy = spec.point(2, 23.0)
    waste = [(TILE, (float(gap_xy[0]) + 3.0, float(gap_xy[1]) + 1.25), 0.3, "V01")]
    ann = make_annset(spec, waste=waste)
    hole = box(gap_xy[0] - 1.0, gap_xy[1] + 1.9, gap_xy[0] + 1.0, gap_xy[1] + 2.9)  # 1.9 m across the row
    return ann, tile_coverage(ann).difference(hole)


def test_gap_target_within_the_edge_margin_of_nodata_is_dropped():
    ann, coverage = _hole_case()
    result = _run(ann, coverage=coverage)
    assert set(result.targets["kind"]) == {"waste"}
    assert result.counts["edge_dropped"] == 1


def test_edge_margin_is_configurable_and_never_drops_waste():
    ann, coverage = _hole_case()
    result = _run(ann, "targets.edge_margin_m=1.5", coverage=coverage)
    assert sorted(result.targets["kind"]) == ["row_gap", "waste"]
    assert result.counts["edge_dropped"] == 0


def test_targets_far_from_the_boundary_are_kept():
    ann, _ = _hole_case()
    result = _run(ann)
    assert sorted(result.targets["kind"]) == ["row_gap", "waste"]
    pts = shapely.points(result.targets["x"], result.targets["y"])
    assert (shapely.distance(pts, tile_coverage(ann).boundary) > 3.0).all()


# ------------------------------------------------------------------ END on outer rows / fragments


def test_no_row_end_short_on_the_outermost_rows():
    outer = make_annset(BlockSpec(n_rows=3, gaps=((1, 50.0, 60.0),)))
    skipped = _run(outer)
    assert "row_end_short" not in set(skipped.targets["kind"])
    assert skipped.counts["end_outer_row"] == 1
    kept = _run(outer, "targets.end_skip_outer_rows=false")
    assert list(kept.targets["kind"]) == ["row_end_short"] and kept.counts["end_outer_row"] == 0


def test_row_end_short_on_an_inner_row_is_kept():
    inner = _run(make_annset(BlockSpec(n_rows=3, gaps=((2, 50.0, 60.0),))))
    assert list(inner.targets["kind"]) == ["row_end_short"]
    assert set(inner.targets["route_role"]) == {"optional"}


def _fragmented_rows(spec: BlockSpec) -> gpd.GeoDataFrame:
    """Row 2 split in two collinear pieces (a failed tile-seam join) that got consecutive row_index."""
    def line(k: int, a: float, b: float) -> LineString:
        return LineString([spec.point(k, a), spec.point(k, b)])

    geoms = [line(1, 0.0, 60.0), line(2, 0.0, 20.0), line(2, 40.0, 60.0), line(3, 0.0, 60.0), line(4, 0.0, 60.0)]
    return gpd.GeoDataFrame({"row_id": ["V01-R001", "V01-R002", "V01-R092", "V01-R003", "V01-R004"],
                             "vineyard_id": ["V01"] * 5, "row_index": [1, 2, 3, 4, 5]}, geometry=geoms, crs=CRS_EPSG)


def test_collinear_row_fragments_give_no_row_end_short():
    spec = BlockSpec(n_rows=4)
    ann = make_annset(spec)
    result = _run(ann, rows=_fragmented_rows(spec))
    assert "row_end_short" not in set(result.targets["kind"])
    cfg = load_config(environ={}).targets
    assert result.counts["end_ill_defined"] == 2 * end_samples(40.0, cfg)
