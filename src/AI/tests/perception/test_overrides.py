"""Overrides: model validation, digest, candidate overrides (exclude, force_empty), row overrides."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pytest
import shapely
from shapely.geometry import LineString

from vineyard.config import load_config
from vineyard.contracts.enums import Severity
from vineyard.errors import ConfigError
from vineyard.geo.tiling import CRS_EPSG, tile_ref
from vineyard.perception.overrides import (
    CODE_APPLIED,
    CODE_UNMATCHED,
    EMPTY_OVERRIDES,
    Overrides,
    apply_candidate_overrides,
    apply_row_overrides,
    load_overrides,
    overrides_digest,
)

TILE = "siret3_r021_c012"
TOL = load_config().blocks.override_match_tol_m
T = tile_ref(TILE)
X0, Y0 = T.x0, T.y0 - 25.0


def cands() -> gpd.GeoDataFrame:
    lines = [LineString([(X0 + 1, Y0 + 2.5 * k), (X0 + 41, Y0 + 2.5 * k)]) for k in range(3)]
    return gpd.GeoDataFrame({"cand_id": [f"{TILE}:K{k + 1:02d}" for k in range(3)], "tile_id": [TILE] * 3},
                            geometry=lines, crs=CRS_EPSG)


def chains() -> gpd.GeoDataFrame:
    lines = [LineString([(X0, Y0 + 2.5 * k), (X0 + 40, Y0 + 2.5 * k)]) for k in range(3)]
    return gpd.GeoDataFrame({"chain_id": ["L00001", "L00002", "L00007"], "qa_flags": ["", "row_interpolated", ""],
                             "extent_m": [40.0] * 3}, geometry=lines, crs=CRS_EPSG)


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "overrides.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_template_file_loads(project_root: Path) -> None:
    ov = load_overrides(project_root / "configs" / "overrides.yaml")
    assert ov == EMPTY_OVERRIDES


def test_empty_yaml_and_missing_file(tmp_path: Path) -> None:
    assert load_overrides(write_yaml(tmp_path, "")) == EMPTY_OVERRIDES
    with pytest.raises(ConfigError):
        load_overrides(tmp_path / "nope.yaml")


@pytest.mark.parametrize("text", [
    "version: 1\nbogus: []\n",
    "version: 2\n",
    "version: 1\nforce_empty_tiles: [not_a_tile]\n",
    "version: 1\nadd_rows: [{id: A001, wkt: 'POINT (1 2)'}]\n",
    "version: 1\ndelete_rows: [{id: D001, wkt: 'garbage'}]\n",
    "version: 1\ndelete_rows: [{id: D001, wkt: 'POINT (1 2)'}]\nadd_rows: [{id: D001, wkt: 'LINESTRING (0 0, 5 0)'}]\n",
    "version: 1\nexclude_areas: [{id: X001, wkt: 'POINT (1 2)'}]\n",
    "version: 1\nextend_rows: [{id: E001, row_hint_wkt: 'POINT (1 2)', to_wkt: 'LINESTRING (0 0, 1 1)'}]\n",
    "version: 1\nadd_rows: [{id: A001, wkt: 'LINESTRING (0 0, 0.01 0)'}]\n",
    ": : :\n",
])
def test_invalid_files_raise(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_overrides(write_yaml(tmp_path, text))


def test_digest_is_stable_and_content_sensitive() -> None:
    a = Overrides(version=1, force_empty_tiles=(TILE,))
    b = Overrides(version=1, force_empty_tiles=(TILE,))
    assert overrides_digest(a) == overrides_digest(b)
    assert overrides_digest(a) != overrides_digest(EMPTY_OVERRIDES)


def test_exclude_area_cuts_and_drops() -> None:
    area = shapely.box(X0 + 20, Y0 - 1, X0 + 50, Y0 + 1).wkt
    unmatched = shapely.box(X0 - 100, Y0 - 100, X0 - 90, Y0 - 90).wkt
    ov = Overrides.model_validate({"version": 1, "exclude_areas": [{"id": "X001", "wkt": area},
                                                                    {"id": "X002", "wkt": unmatched}]})
    src = cands()
    res = apply_candidate_overrides(src, ov)
    assert len(res.frame) == 3
    assert res.frame.geometry.iloc[0].length == pytest.approx(19.0)
    assert src.geometry.iloc[0].length == pytest.approx(40.0)  # input untouched
    assert [i.code for i in res.issues] == [CODE_APPLIED, CODE_UNMATCHED]
    assert res.issues[1].severity == Severity.WARNING
    full = shapely.box(X0, Y0 - 1, X0 + 50, Y0 + 1).wkt
    res2 = apply_candidate_overrides(src, Overrides.model_validate({"version": 1,
                                                                    "exclude_areas": [{"id": "X003", "wkt": full}]}))
    assert len(res2.frame) == 2 and list(res2.removed.reason) == ["override:X003"]


def test_force_empty_drops_tile_candidates() -> None:
    other = "siret3_r021_c013"
    ov = Overrides(version=1, force_empty_tiles=(TILE, other))
    res = apply_candidate_overrides(cands(), ov)
    assert len(res.frame) == 0 and len(res.removed) == 3
    assert [i.tile_id for i in res.issues] == [TILE, other]
    assert all(i.code == CODE_APPLIED for i in res.issues)


def test_delete_row_removes_exactly_one_chain() -> None:
    ov = Overrides.model_validate({"version": 1, "delete_rows": [
        {"id": "D001", "wkt": f"POINT ({X0 + 10} {Y0 + 2.5 + 0.6})"},
        {"id": "D002", "wkt": f"POINT ({X0 + 10} {Y0 + 200})"},
    ]})
    res = apply_row_overrides(chains(), ov, default_tol_m=TOL)
    assert list(res.frame.chain_id) == ["L00001", "L00007"]
    assert list(res.removed.chain_id) == ["L00002"]
    assert [i.code for i in res.issues] == [CODE_APPLIED, CODE_UNMATCHED]


def test_extend_row_moves_end_to_projection() -> None:
    ov = Overrides.model_validate({"version": 1, "extend_rows": [
        {"id": "E001", "row_hint_wkt": f"POINT ({X0 + 39} {Y0 + 0.2})", "to_wkt": f"POINT ({X0 + 45} {Y0 + 0.7})"},
        {"id": "E002", "row_hint_wkt": f"POINT ({X0 - 50} {Y0})", "to_wkt": f"POINT ({X0} {Y0})"},
    ]})
    src = chains()
    res = apply_row_overrides(src, ov, default_tol_m=TOL)
    coords = list(res.frame.geometry.iloc[0].coords)
    assert coords[0] == pytest.approx((X0, Y0))
    assert coords[-1] == pytest.approx((X0 + 45, Y0))
    assert res.frame.extent_m.iloc[0] == pytest.approx(45.0)
    assert res.frame.qa_flags.iloc[0] == "override:E001"
    assert src.geometry.iloc[0].length == pytest.approx(40.0)
    assert [i.code for i in res.issues] == [CODE_APPLIED, CODE_UNMATCHED]


def test_extend_row_start_end() -> None:
    ov = Overrides.model_validate({"version": 1, "extend_rows": [
        {"id": "E001", "row_hint_wkt": f"POINT ({X0 + 1} {Y0 + 5})", "to_wkt": f"POINT ({X0 - 3} {Y0 + 5})"}]})
    res = apply_row_overrides(chains(), ov, default_tol_m=TOL)
    assert list(res.frame.geometry.iloc[2].coords)[0] == pytest.approx((X0 - 3, Y0 + 5))


def test_add_row_gets_next_chain_id_and_flag() -> None:
    wkt = f"LINESTRING ({X0} {Y0 + 10}, {X0 + 30} {Y0 + 10})"
    ov = Overrides.model_validate({"version": 1, "add_rows": [{"id": "A001", "wkt": wkt, "note": "missed"}]})
    res = apply_row_overrides(chains(), ov, default_tol_m=TOL)
    added = res.frame.iloc[-1]
    assert added.chain_id == "L00008"
    assert added.source == "model"
    assert "override:A001" in added.qa_flags
    assert added.tile_ids == TILE and added.n_tiles == 1
    assert added.extent_m == pytest.approx(30.0)
    assert res.issues[0].code == CODE_APPLIED


def test_no_overrides_is_identity() -> None:
    res = apply_row_overrides(chains(), EMPTY_OVERRIDES, default_tol_m=TOL)
    assert list(res.frame.chain_id) == ["L00001", "L00002", "L00007"]
    assert len(res.removed) == 0 and res.issues == ()
    cres = apply_candidate_overrides(cands(), EMPTY_OVERRIDES)
    assert len(cres.frame) == 3 and len(cres.removed) == 0
