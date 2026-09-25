"""The single gap engine (S2): pixel-set gaps on canopy rasters, 1-px bins, ends included."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Polygon

from tests.helpers.examples import canopy_polygons_utm, load_examples, row_lines_utm
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import RowStructure
from vineyard.contracts.schemas import validate_layer
from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_ref
from vineyard.perception import attrs
from vineyard.perception.attrs import Gap, GapKind
from vineyard.perception.types import VIS_NODATA, VIS_OK, VIS_SHADOW
from vineyard.perception.vegmask import vis_codes

TILE = tile_ref("siret3_r021_c012")
U0, V0 = 100.0, 1000.0
HALF_M = 0.30
TOL = 0.05  # 2 px: the index rasterization includes both polygon boundaries


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config(environ={})


def _axis(length_m: float) -> LineString:
    return LineString(px_to_utm(TILE, np.array([[U0, V0], [U0 + length_m / GSD_M, V0]])))


def _rect(a0_m: float, a1_m: float, half_m: float = 0.2) -> Polygon:
    u0, u1 = U0 + a0_m / GSD_M, U0 + a1_m / GSD_M
    v0, v1 = V0 - half_m / GSD_M, V0 + half_m / GSD_M
    return Polygon(px_to_utm(TILE, np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])))


def _vis_band(a0_m: float, a1_m: float, code: int) -> np.ndarray:
    vis = np.full((TILE_PX, TILE_PX), VIS_OK, dtype=np.uint8)
    u0, u1 = int(U0 + a0_m / GSD_M), int(U0 + a1_m / GSD_M)
    vis[int(V0) - 40 : int(V0) + 40, u0 : u1 + 1] = code
    return vis


def _measure(cfg: AppConfig, length_m: float, spans: list[tuple[float, float]], vis=None, rs=None):
    canopy_ij = attrs.canopy_pixels([_rect(a, b) for a, b in spans], TILE)
    return attrs.measure_piece(_axis(length_m), canopy_ij, TILE, rs or cfg.row_structure, half_m=HALF_M, vis=vis)


# ------------------------------------------------------------------ gaps


def test_interior_and_end_gaps(cfg: AppConfig) -> None:
    res = _measure(cfg, 15.0, [(0, 2), (3, 4), (10, 12)])
    lengths = sorted(g.length_m for g in res.gaps)
    assert lengths == pytest.approx([1.0, 3.0, 6.0], abs=TOL)
    assert res.max_gap_m == pytest.approx(6.0, abs=TOL)
    assert res.row_structure == RowStructure.DISRUPTED
    tail = max(res.gaps, key=lambda g: g.start_m)
    assert tail.kind == GapKind.TAIL and not tail.censored
    assert tail.end_m == pytest.approx(15.0, abs=1e-6)
    assert [g.kind for g in res.gaps if g.kind == GapKind.INTERIOR] == [GapKind.INTERIOR] * 2


def test_head_gap_counts(cfg: AppConfig) -> None:
    res = _measure(cfg, 15.0, [(7, 15)])
    assert len(res.gaps) == 1
    gap = res.gaps[0]
    assert gap.kind == GapKind.HEAD and gap.start_m == 0.0
    assert gap.length_m == pytest.approx(7.0, abs=TOL)
    assert res.row_structure == RowStructure.DISRUPTED


def test_ends_excluded_when_configured(cfg: AppConfig) -> None:
    rs = cfg.row_structure.model_copy(update={"include_end_gaps": False})
    res = _measure(cfg, 15.0, [(7, 15)], rs=rs)
    assert res.max_gap_m == 0.0
    assert res.row_structure == RowStructure.REGULAR


def test_empty_piece_is_one_full_gap(cfg: AppConfig) -> None:
    res = _measure(cfg, 4.0, [])
    assert [g.kind for g in res.gaps] == [GapKind.FULL]
    assert res.max_gap_m == pytest.approx(4.0, abs=TOL)
    assert res.row_structure == RowStructure.REGULAR


def test_nodata_span_is_unknown_not_gap(cfg: AppConfig) -> None:
    vis = _vis_band(4.5, 9.5, VIS_NODATA)
    res = _measure(cfg, 15.0, [(0, 4), (10, 15)], vis=vis)
    assert res.max_gap_m < 1.0
    assert res.row_structure == RowStructure.REGULAR
    assert len(res.gaps) == 2 and all(g.censored for g in res.gaps)
    assert all(g.kind == GapKind.INTERIOR for g in res.gaps)
    assert res.visible_frac == pytest.approx(10.0 / 15.0, abs=0.01)


def test_low_visibility_is_unassessable(cfg: AppConfig) -> None:
    vis = _vis_band(0.0, 9.0, VIS_SHADOW)
    res = _measure(cfg, 15.0, [(12, 15)], vis=vis)
    assert res.visible_frac == pytest.approx(0.4, abs=0.01)
    assert res.row_structure == RowStructure.UNASSESSABLE


def test_borderline_flag(cfg: AppConfig) -> None:
    res = _measure(cfg, 15.0, [(0, 5), (10.5, 15)])
    assert res.max_gap_m == pytest.approx(5.5, abs=TOL)
    assert res.row_structure == RowStructure.DISRUPTED
    assert res.flags == (attrs.BORDERLINE_CODE,)
    assert _measure(cfg, 15.0, [(0, 15)]).flags == ()


def test_smoothing_closes_small_holes_only_when_enabled(cfg: AppConfig) -> None:
    spans = [(0, 5), (5.3, 10), (10.3, 15)]
    raw = _measure(cfg, 15.0, spans)
    assert raw.max_gap_m == pytest.approx(0.3, abs=TOL)
    rs = cfg.row_structure.model_copy(update={"occ_smoothing_enabled": True})
    smooth = _measure(cfg, 15.0, spans, rs=rs)
    assert smooth.max_gap_m == 0.0
    big = _measure(cfg, 15.0, [(0, 4), (10, 15)], rs=rs)
    assert big.max_gap_m == pytest.approx(6.0, abs=0.3)


def test_polyline_axis_along_both_segments(cfg: AppConfig) -> None:
    pts_px = np.array([[U0, V0], [U0 + 300, V0], [U0 + 300 + 260, V0 + 150]])
    axis = LineString(px_to_utm(TILE, pts_px))
    squares = []
    for along in [1, 2, 3, 4, 5, 12, 13, 14]:
        p = np.asarray(axis.interpolate(along).coords[0])
        squares.append(Polygon(p + np.array([[-0.15, -0.15], [0.15, -0.15], [0.15, 0.15], [-0.15, 0.15]])))
    res = attrs.measure_piece(axis, attrs.canopy_pixels(squares, TILE), TILE, cfg.row_structure, half_m=HALF_M)
    interior = [g for g in res.gaps if g.kind == GapKind.INTERIOR]
    assert max(g.length_m for g in interior) == pytest.approx(6.645, abs=TOL)  # 2nd-segment square is rotated


def test_gap_geometry_helpers() -> None:
    axis = _axis(15.0)
    gap = Gap(start_m=4.0, end_m=10.0, kind=GapKind.INTERIOR, censored=False)
    centre = attrs.gap_centre(axis, gap)
    assert centre.distance(axis.interpolate(7.0)) < 1e-9
    assert attrs.gap_line(axis, gap).length == pytest.approx(6.0)


def test_gap_rejects_bad_interval() -> None:
    with pytest.raises(ValueError):
        Gap(start_m=5.0, end_m=4.0, kind=GapKind.INTERIOR, censored=False)


def test_measure_piece_rejects_degenerate_axis(cfg: AppConfig) -> None:
    with pytest.raises(ValueError):
        attrs.measure_piece(LineString([(0, 0), (0, 0)]), np.zeros((0, 2)), TILE, cfg.row_structure, half_m=HALF_M)


def test_profile_arrays_are_readonly(cfg: AppConfig) -> None:
    axis_px = attrs.axis_to_px(_axis(3.0), TILE)
    prof = attrs.occupancy_profile(axis_px, np.zeros((0, 2)), half_px=12.0)
    with pytest.raises(ValueError):
        prof.occupied[0] = True


# ------------------------------------------------------------------ per-tile frames


def _pieces(rows: list[tuple[str, LineString]], tile_id: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"row_id": [r for r, _ in rows], "vineyard_id": [r.split("-")[0] for r, _ in rows],
         "tile_id": [tile_id] * len(rows)},
        geometry=[g for _, g in rows], crs=CRS_EPSG,
    )


def _canopies(polys: list[Polygon], tile_id: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"tile_id": [tile_id] * len(polys)}, geometry=polys, crs=CRS_EPSG)


def test_row_pieces_attributes_frame(cfg: AppConfig) -> None:
    pieces = _pieces([("V01-R001", _axis(15.0)), ("V01-R001", _axis(0.02))], TILE.tile_id)
    canopies = _canopies([_rect(0, 4), _rect(10, 15)], TILE.tile_id)
    out = attrs.row_pieces_attributes(
        pieces, canopies, TILE, cfg.row_structure, half_m=HALF_M, min_piece_m=0.5,
        run_id="r1", model_version="mv",
    )
    validate_layer(out.row_pieces, "row_pieces")
    assert list(out.row_pieces["piece_id"]) == [f"V01-R001@{TILE.tile_id}"]
    row = out.row_pieces.iloc[0]
    assert row["row_structure"] == "disrupted" and row["n_vertices"] == 2
    assert row["max_gap_m"] == pytest.approx(6.0, abs=TOL)
    assert "structure_borderline" in row["qa_flags"]
    assert set(out.gaps["kind"]) == {"interior"}
    assert out.gaps.iloc[0]["piece_id"] == row["piece_id"]


def test_row_pieces_attributes_dup_ids_and_empty(cfg: AppConfig) -> None:
    far = LineString(px_to_utm(TILE, np.array([[U0, V0 + 400], [U0 + 400, V0 + 400]])))
    pieces = _pieces([("V01-R002", far), ("V01-R002", _axis(15.0))], TILE.tile_id)
    out = attrs.row_pieces_attributes(pieces, _canopies([], TILE.tile_id), TILE, cfg.row_structure,
                                      half_m=HALF_M, min_piece_m=0.5)
    assert sorted(out.row_pieces["piece_id"]) == [f"V01-R002@{TILE.tile_id}", f"V01-R002@{TILE.tile_id}#2"]
    empty = attrs.row_pieces_attributes(pieces.iloc[:0], _canopies([], TILE.tile_id), TILE, cfg.row_structure,
                                        half_m=HALF_M, min_piece_m=0.5)
    assert len(empty.row_pieces) == 0 and len(empty.gaps) == 0
    validate_layer(empty.row_pieces, "row_pieces")


def test_row_pieces_attributes_requires_columns(cfg: AppConfig) -> None:
    bad = gpd.GeoDataFrame({"x": [1]}, geometry=[_axis(3.0)], crs=CRS_EPSG)
    with pytest.raises(ValueError, match="row_id"):
        attrs.row_pieces_attributes(bad, _canopies([], TILE.tile_id), TILE, cfg.row_structure,
                                    half_m=HALF_M, min_piece_m=0.5)


# ------------------------------------------------------------------ reference acceptance (S2)

EXPECTED_MAX_GAPS = {  # pixel-set gaps (1-px bins), R15 is the 5.009 m vector edge case
    ("siret3_r006_c004", "V02-R06"): 5.025,
    ("siret3_r006_c004", "V02-R07"): 11.31,
    ("siret3_r006_c004", "V02-R08"): 9.225,
    ("siret3_r006_c004", "V02-R09"): 12.875,
    ("siret3_r006_c004", "V02-R15"): 4.975,
    ("siret3_r006_c004", "V02-R23"): 6.75,
}


@pytest.fixture(scope="module")
def reference_results(examples_dir: Path, cfg: AppConfig) -> dict[tuple[str, str], tuple[str, float, float]]:
    images = load_examples((examples_dir / "annotations.xml").read_bytes())
    out: dict[tuple[str, str], tuple[str, float, float]] = {}
    for tile_id, img in images.items():
        tile = tile_ref(tile_id)
        rgb = read_tile(examples_dir / "images" / f"{tile_id}.tif")
        vis = vis_codes(rgb, np.ones(rgb.shape[:2], bool), shadow_v_max=cfg.veg.shadow_v_max,
                        overexp_v_min=cfg.veg.overexposed_v_min)
        canopy_ij = attrs.canopy_pixels(canopy_polygons_utm(img), tile)
        for shape, line in zip(img.by_label("row"), row_lines_utm(img), strict=True):
            res = attrs.measure_piece(line, canopy_ij, tile, cfg.row_structure, half_m=HALF_M, vis=vis)
            out[(tile_id, shape.attributes["row_id"])] = (shape.attributes["row_structure"], res.max_gap_m,
                                                          res.visible_frac)
            assert res.row_structure.value in {"regular", "disrupted"}
            out[(tile_id, shape.attributes["row_id"] + "#pred")] = (res.row_structure.value, 0.0, 0.0)
    return out


@pytest.mark.examples
def test_reference_row_structure_51_of_51(reference_results) -> None:
    labels = {k: v for k, v in reference_results.items() if not k[1].endswith("#pred")}
    assert len(labels) == 51
    wrong = [k for k, (label, _, _) in labels.items() if reference_results[(k[0], k[1] + "#pred")][0] != label]
    assert wrong == []


@pytest.mark.examples
def test_imported_reference_row_structure_51_of_51(examples_dir: Path, examples_xml: bytes, cfg: AppConfig) -> None:
    from vineyard.pipeline.stages.import_reference import build_reference_annset

    ref, _ = build_reference_annset(cfg, examples_xml, run_id="ref", source_name="annotations.xml")
    wrong, n = [], 0
    for tile_id in ref.meta.tile_ids:
        tile = tile_ref(tile_id)
        rgb = read_tile(examples_dir / "images" / f"{tile_id}.tif")
        vis = vis_codes(rgb, np.ones(rgb.shape[:2], bool), shadow_v_max=cfg.veg.shadow_v_max,
                        overexp_v_min=cfg.veg.overexposed_v_min)
        can = ref.canopies[(ref.canopies["tile_id"] == tile_id).to_numpy()]
        rows = ref.row_pieces[(ref.row_pieces["tile_id"] == tile_id).to_numpy()]
        out = attrs.row_pieces_attributes(rows, can, tile, cfg.row_structure,
                                          half_m=HALF_M, min_piece_m=0.0, vis=vis)
        n += len(out.row_pieces)
        wrong += [r for r, want, got in zip(rows["row_id"], rows["row_structure"], out.row_pieces["row_structure"],
                                            strict=True) if want != got]
    assert (n, wrong) == (51, [])


@pytest.mark.examples
def test_reference_edge_gaps_and_visibility(reference_results) -> None:
    for key, expected in EXPECTED_MAX_GAPS.items():
        assert reference_results[key][1] == pytest.approx(expected, abs=1e-3), key
    labels = {k: v for k, v in reference_results.items() if not k[1].endswith("#pred")}
    assert min(vf for _, _, vf in labels.values()) >= 0.5
