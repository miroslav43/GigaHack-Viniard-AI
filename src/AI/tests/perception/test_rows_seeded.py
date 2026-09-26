"""perception.row_seeds (seed file) + perception.rows_seeded (lattice fit inside a reviewed seed region)."""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from shapely.geometry import LineString, Polygon, box

from tests.helpers.synth import striped_mask
from vineyard.config import AppConfig, load_config
from vineyard.config.sections_seeded import RowsSeededConfig
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TILE_PX, px_to_utm, tile_ref
from vineyard.perception.row_seeds import (
    SEED_COLUMNS,
    RowSeed,
    SeedFileError,
    load_row_seeds,
    parse_polygon,
    px_angle_to_seed,
    seed_angle_to_px,
    usable_seeds,
)
from vineyard.perception.rows_detect import DetectParams
from vineyard.perception.rows_seeded import (
    FLAG_SEEDED,
    REASON_FEW_ROWS,
    SeededInputs,
    region_mask,
    seeded_detect,
    spacing_window,
    trim_to_evidence,
)
from vineyard.pipeline.stages._rows_seeded_io import mark_seeded_chains

TILE_ID = "siret3_r021_c012"
TILE = tile_ref(TILE_ID)
SHAPE = (TILE_PX, TILE_PX)
FRAME = box(0.0, 0.0, float(TILE_PX), float(TILE_PX))
SEED_ANGLE = 30.0                       # image angle, counter-clockwise towards the top
ANGLE_PX = seed_angle_to_px(SEED_ANGLE)  # 150: profile/striped_mask convention (towards +v)
SPACING_PX = 100.0                      # 2.5 m
HEADER = ",".join(SEED_COLUMNS)


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture(scope="module")
def s(cfg: AppConfig) -> RowsSeededConfig:
    return cfg.rows_seeded


@pytest.fixture(scope="module")
def params(cfg: AppConfig) -> DetectParams:
    return DetectParams.from_config(cfg)


def _seed(polygon: Polygon, angle: float = SEED_ANGLE, spacing: float | None = None) -> RowSeed:
    return RowSeed(tile_id=TILE_ID, vines="yes", confidence="high", polygon_px=polygon, angle_deg=angle,
                   spacing_m=spacing, kind="young", note="", line_no=2)


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "row_seeds.csv"
    path.write_text("\n".join([HEADER, *lines]) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ seed file


def test_header_only_file_has_no_seeds(tmp_path: Path) -> None:
    assert load_row_seeds(_write(tmp_path)) == ()


def test_committed_seed_file_parses() -> None:
    path = Path(__file__).resolve().parents[2] / "configs" / "row_seeds.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0] == HEADER
    load_row_seeds(path)


def test_committed_seeds_are_usable_and_avoid_force_empty_tiles() -> None:
    """Every committed seed was accepted in visual review: usable, and never on a force_empty tile."""
    root = Path(__file__).resolve().parents[2]
    seeds = load_row_seeds(root / "configs" / "row_seeds.csv")
    overrides = yaml.safe_load((root / "configs" / "overrides.yaml").read_text(encoding="utf-8"))
    forced = set(overrides.get("force_empty_tiles") or ())
    assert usable_seeds(seeds, ("high", "medium")) == seeds
    assert not {s.tile_id for s in seeds} & forced


def test_parses_a_seed_line(tmp_path: Path) -> None:
    path = _write(tmp_path, f'{TILE_ID},yes,high,"10 10;500 10;500 400",30,2.4,young,"stakes, top-right"')
    (seed,) = load_row_seeds(path, {TILE_ID})
    assert (seed.tile_id, seed.vines, seed.confidence, seed.angle_deg, seed.spacing_m) == (
        TILE_ID, "yes", "high", 30.0, 2.4)
    assert seed.polygon_px.area == pytest.approx(0.5 * 490 * 390)
    assert (seed.note, seed.line_no) == ("stakes, top-right", 2)


def test_optional_spacing_is_none(tmp_path: Path) -> None:
    (seed,) = load_row_seeds(_write(tmp_path, f"{TILE_ID},maybe,low,0 0;10 0;10 10,0,,other,"))
    assert seed.spacing_m is None


@pytest.mark.parametrize(("line", "match"), [
    ("siret3_r999_c999,yes,high,0 0;10 0;10 10,30,,k,n", "unknown tile"),
    ("bogus,yes,high,0 0;10 0;10 10,30,,k,n", "invalid tile id"),
    (f"{TILE_ID},yes,high,0 0;10 0,30,,k,n", "vertices"),
    (f"{TILE_ID},yes,high,0 0;10 0;0 10;10 10,30,,k,n", "valid polygon"),
    (f"{TILE_ID},yes,high,0 0;3000 0;10 10,30,,k,n", "outside the tile"),
    (f"{TILE_ID},yes,high,0 0;10 0;10 10,180,,k,n", r"\[0, 180\)"),
    (f"{TILE_ID},yes,high,0 0;10 0;10 10,-1,,k,n", r"\[0, 180\)"),
    (f"{TILE_ID},yes,high,0 0;10 0;10 10,30,-2,k,n", "spacing_m"),
    (f"{TILE_ID},sure,high,0 0;10 0;10 10,30,,k,n", "vines"),
    (f"{TILE_ID},yes,top,0 0;10 0;10 10,30,,k,n", "confidence"),
    (f"{TILE_ID},yes,high,0 0;10 0;10 10,30,,k", "fields"),
])
def test_bad_lines_name_file_and_line(tmp_path: Path, line: str, match: str) -> None:
    path = _write(tmp_path, f"{TILE_ID},yes,high,0 0;10 0;10 10,30,,k,n", line)
    with pytest.raises(SeedFileError, match=match) as exc:
        load_row_seeds(path, {TILE_ID, "siret3_r021_c013"})
    assert f"{path}:3:" in str(exc.value)


def test_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("tile_id,angle\n", encoding="utf-8")
    with pytest.raises(SeedFileError, match=":1: header"):
        load_row_seeds(path)


def test_usable_seeds_filters_vines_and_confidence(tmp_path: Path) -> None:
    lines = [f"{TILE_ID},{v},{c},0 0;10 0;10 10,30,,k,n" for v, c in
             (("yes", "high"), ("yes", "medium"), ("yes", "low"), ("maybe", "high"), ("no", "high"))]
    seeds = load_row_seeds(_write(tmp_path, *lines))
    assert [(x.vines, x.confidence) for x in usable_seeds(seeds, ("high", "medium"))] == [
        ("yes", "high"), ("yes", "medium")]


def test_parse_polygon_rejects_nan() -> None:
    with pytest.raises(ValueError):
        parse_polygon("0 0;nan 0;10 10")


# ------------------------------------------------------------------ angle convention


@pytest.mark.parametrize("angle", [0.0, 15.0, 30.0, 90.0, 135.5, 179.9])
def test_angle_round_trip(angle: float) -> None:
    assert px_angle_to_seed(seed_angle_to_px(angle)) == pytest.approx(angle)


def test_seed_angle_points_to_the_image_top() -> None:
    # 30 deg counter-clockwise from +x towards the top: direction (cos, -sin) in (u, v) with v down
    t = math.radians(seed_angle_to_px(30.0))
    d = np.array([math.cos(t), math.sin(t)])
    d = d if d[0] > 0 else -d
    assert d == pytest.approx([math.cos(math.radians(30.0)), -math.sin(math.radians(30.0))])


def test_seed_angle_equals_the_utm_row_angle(params: DetectParams, s: RowsSeededConfig) -> None:
    veg = striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=24.0)
    res = seeded_detect(SeededInputs(veg=veg, clip_px=FRAME, tile=TILE), (_seed(FRAME),), params, s)
    assert res.orientations[0].angle_utm_deg == pytest.approx(SEED_ANGLE, abs=0.5)


# ------------------------------------------------------------------ lattice fit


def test_region_mask_fills_polygon() -> None:
    mask = region_mask(box(100, 200, 300, 250), SHAPE)
    assert mask[210:240, 110:290].all()
    assert not mask[:190].any() and not mask[260:].any()


def test_spacing_window(s: RowsSeededConfig) -> None:
    assert spacing_window(None, s) == tuple(s.spacing_range_m)
    assert spacing_window(2.0, s) == pytest.approx((2.0 * (1 - s.spacing_rel_tol), 2.0 * (1 + s.spacing_rel_tol)))


def test_lattice_fit_on_striped_region(params: DetectParams, s: RowsSeededConfig) -> None:
    """Rows only inside the polygon, at the stripe centres, with the right spacing and flag."""
    veg = striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=24.0, offset_px=37.0)
    region = Polygon([(200, 200), (1600, 300), (1500, 1500), (300, 1400)])
    res = seeded_detect(SeededInputs(veg=veg, clip_px=FRAME, tile=TILE), (_seed(region, angle=SEED_ANGLE + 2.0),),
                        params, s)
    assert res.outcomes[0].reason is None
    assert res.orientations[0].spacing.spacing_m == pytest.approx(SPACING_PX * GSD_M, rel=0.03)
    assert len(res.candidates) >= 10
    n = np.array([-math.sin(math.radians(ANGLE_PX)), math.cos(math.radians(ANGLE_PX))])
    for c in res.candidates:
        assert region.buffer(2.0).contains(c.line_px)
        mid = np.asarray(c.line_px.interpolate(0.5, normalized=True).coords[0])
        frac = (float(mid @ n) - 37.0) / SPACING_PX
        assert abs(frac - round(frac)) < 0.05
        assert c.soft_flags == (FLAG_SEEDED,) and c.rejected_reason is None


def test_seed_spacing_constrains_the_estimate(params: DetectParams, s: RowsSeededConfig) -> None:
    veg = striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=24.0)
    res = seeded_detect(SeededInputs(veg=veg, clip_px=FRAME, tile=TILE), (_seed(FRAME, spacing=2.6),), params, s)
    assert res.orientations[0].spacing.spacing_m == pytest.approx(2.5, abs=0.1)


def test_rows_stop_at_the_last_vine(params: DetectParams, s: RowsSeededConfig) -> None:
    """A headland/road without vegetation inside the polygon: rows end at the first/last vine."""
    veg = striped_mask(SHAPE, angle_deg=0.0, spacing_px=SPACING_PX, width_px=24.0)
    veg[:, :400] = False
    veg[:, 1500:] = False
    res = seeded_detect(SeededInputs(veg=veg, clip_px=FRAME, tile=TILE), (_seed(FRAME, angle=0.0),), params, s)
    assert len(res.candidates) >= 10
    for c in res.candidates:
        xs = [x for x, _ in c.line_px.coords]
        assert min(xs) >= 400 - 1.0 / GSD_M * params.occ_bin_m and max(xs) <= 1500 + params.occ_bin_m / GSD_M


def test_trim_splits_at_long_gaps(params: DetectParams) -> None:
    veg = np.zeros(SHAPE, dtype=bool)
    veg[1000:1010, 100:600] = True
    veg[1000:1010, 1000:1800] = True  # 400 px = 10 m gap
    pieces = trim_to_evidence(LineString([(0, 1005), (2048, 1005)]), veg, None, params.corridor_half_m,
                              params.occ_bin_m, cut_gap_m=4.0)
    assert len(pieces) == 2
    (a0, _), (a1, _) = pieces[0].coords[0], pieces[0].coords[-1]
    assert a0 == pytest.approx(100, abs=15) and a1 == pytest.approx(600, abs=15)


def test_too_few_rows_rejects_the_region(params: DetectParams, s: RowsSeededConfig) -> None:
    veg = striped_mask(SHAPE, angle_deg=0.0, spacing_px=SPACING_PX, width_px=24.0)
    region = box(200, 180, 1800, 330)  # two stripe centres (200, 300) inside
    res = seeded_detect(SeededInputs(veg=veg, clip_px=FRAME, tile=TILE), (_seed(region, angle=0.0),), params, s)
    assert res.candidates == ()
    assert res.outcomes[0].reason == REASON_FEW_ROWS


def test_low_occupancy_rows_dropped(params: DetectParams, s: RowsSeededConfig) -> None:
    veg = np.zeros(SHAPE, dtype=bool)
    res = seeded_detect(SeededInputs(veg=veg, clip_px=FRAME, tile=TILE), (_seed(FRAME, angle=0.0),), params, s)
    assert res.candidates == ()


def test_existing_rows_are_not_duplicated(params: DetectParams, s: RowsSeededConfig) -> None:
    veg = striped_mask(SHAPE, angle_deg=0.0, spacing_px=SPACING_PX, width_px=24.0)
    first = seeded_detect(SeededInputs(veg=veg, clip_px=FRAME, tile=TILE), (_seed(FRAME, angle=0.0),), params, s)
    existing = tuple(LineString(px_to_utm(TILE, np.asarray(c.line_px.coords))) for c in first.candidates[:5])
    again = seeded_detect(SeededInputs(veg=veg, clip_px=FRAME, tile=TILE, existing_utm=existing),
                          (_seed(FRAME, angle=0.0),), params, s)
    assert len(again.candidates) == len(first.candidates) - 5


def test_deterministic(params: DetectParams, s: RowsSeededConfig) -> None:
    veg = striped_mask(SHAPE, angle_deg=ANGLE_PX, spacing_px=SPACING_PX, width_px=24.0)
    inp = SeededInputs(veg=veg, clip_px=FRAME, tile=TILE)
    a = seeded_detect(inp, (_seed(FRAME),), params, s)
    b = seeded_detect(inp, (_seed(FRAME),), params, s)
    assert [c.line_px.wkt for c in a.candidates] == [c.line_px.wkt for c in b.candidates]


# ------------------------------------------------------------------ chain flag


def test_mark_seeded_chains() -> None:
    line = LineString([(0, 0), (1, 1)])
    rows = gpd.GeoDataFrame({"chain_id": ["C1", "C2"], "member_cand_ids": ["a,b", "c"],
                             "qa_flags": ["curved_row", ""]}, geometry=[line, line], crs=CRS_EPSG)
    cands = gpd.GeoDataFrame({"cand_id": ["a", "b", "c"], "method": ["veg", "seeded", "veg"]},
                             geometry=[line] * 3, crs=CRS_EPSG)
    out = mark_seeded_chains(rows, cands)
    assert list(out.qa_flags) == ["curved_row;seeded", ""]
    assert list(rows.qa_flags) == ["curved_row", ""]  # input untouched
