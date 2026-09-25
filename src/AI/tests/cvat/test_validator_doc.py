"""cvat.validator_doc: label, attribute, enum, geometry and overlap checks on a CvatDocument."""

from __future__ import annotations

import pytest

from vineyard.config import load_config
from vineyard.config.sections_io import CvatExportConfig
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.validator_doc import validate_document, validate_image

TILE = "siret3_r021_c012"
NAME = f"{TILE}.tif"
PX = 2048


@pytest.fixture(scope="module")
def cvat_cfg() -> CvatExportConfig:
    return load_config().export.cvat


def _square(u0: float, v0: float, size: float) -> tuple[tuple[float, float], ...]:
    """Negative shoelace in px (CCW in UTM): (u0,v0) -> (u0,v0+s) -> (u0+s,v0+s) -> (u0+s,v0)."""
    return ((u0, v0), (u0, v0 + size), (u0 + size, v0 + size), (u0 + size, v0))


def canopy(points=None, vid: str = "V01", **kw) -> CvatShape:
    return CvatShape("polygon", "vineyard", points or _square(10, 10, 20), (("vineyard_id", vid),), **kw)


def interrow(points=None, cover: str = "bare_soil") -> CvatShape:
    pts = points or _square(100, 100, 50)
    return CvatShape("polygon", "interrow_area", pts, (("vineyard_id", "V01"), ("interrow_cover", cover)))


def row(points=((0.0, 5.0), (300.0, 5.0)), structure: str = "regular", rid: str = "V01-R001") -> CvatShape:
    attrs = (("vineyard_id", "V01"), ("row_id", rid), ("row_structure", structure))
    return CvatShape("polyline", "row", points, attrs)


def waste(box=((50.0, 50.0), (60.0, 58.0)), vid: str = "V01", ref: str | None = "W0001") -> CvatShape:
    return CvatShape("box", "waste", box, (("vineyard_id", vid),), ref_id=ref)


def _doc(*shapes: CvatShape, name: str = NAME, width: int = PX) -> CvatDocument:
    return CvatDocument((CvatImage(0, name, width, width, tuple(shapes)),))


def _codes(doc: CvatDocument, cfg: CvatExportConfig, **kw) -> list[str]:
    rep = validate_document(doc, cfg=cfg, tile_px=PX, **kw)
    return sorted(i.code for i in rep.errors)


def test_clean_document_has_no_errors(cvat_cfg: CvatExportConfig) -> None:
    doc = _doc(row(), interrow(), canopy(), waste())
    rep = validate_document(doc, cfg=cvat_cfg, tile_px=PX)
    assert rep.ok, rep.summary_lines()
    assert rep.issues == ()


def test_empty_image_is_valid(cvat_cfg: CvatExportConfig) -> None:
    assert validate_document(_doc(), cfg=cvat_cfg, tile_px=PX).ok


@pytest.mark.parametrize(
    ("shape", "code"),
    [
        (CvatShape("polygon", "vineyard", _square(10, 10, 20), (("vineyard_idd", "V01"),)), "missing_attr"),
        (CvatShape("polygon", "vineyard", _square(10, 10, 20), (("vineyard_id", "V01"), ("x", "1"))), "extra_attr"),
        (CvatShape("polygon", "vineyard", _square(10, 10, 20), (("vineyard_id", "V01"), ("vineyard_id", "V02"))),
         "duplicate_attr"),
        (row(structure="Regular"), "bad_enum"),
        (row(structure=" regular"), "bad_enum"),
        (interrow(cover="grass"), "bad_enum"),
        (canopy(vid=""), "empty_id"),
        (canopy(vid=" V01"), "id_whitespace"),
        (row(rid=""), "empty_id"),
        (CvatShape("polygon", "vine", _square(10, 10, 20), (("vineyard_id", "V01"),)), "unknown_label"),
        (CvatShape("polygon", "row", _square(10, 10, 20),
                   (("vineyard_id", "V01"), ("row_id", "V01-R001"), ("row_structure", "regular"))), "wrong_shape"),
    ],
)
def test_attribute_and_label_errors(cvat_cfg: CvatExportConfig, shape: CvatShape, code: str) -> None:
    assert code in _codes(_doc(shape), cvat_cfg)


@pytest.mark.parametrize(
    ("points", "code"),
    [
        (((0, 0), (10, 10), (10, 0), (0, 10)), "invalid_geometry"),  # bowtie
        (_square(2040, 10, 8.5), "coord_out_of_range"),  # u = 2048.5
        (((-0.1, 10.0), (-0.1, 20.0), (10.0, 20.0), (10.0, 10.0)), "coord_out_of_range"),
        (((10, 10), (20, 20), (10, 10)), "too_few_vertices"),  # 2 distinct vertices
        (_square(10, 10, 3), "area_too_small"),  # 9 px2 < 16
        ((*_square(10, 10, 20), (10, 10)), "closing_vertex"),
        (tuple(reversed(_square(10, 10, 20))), "wrong_orientation"),  # positive shoelace in px
    ],
)
def test_polygon_geometry_errors(cvat_cfg: CvatExportConfig, points, code: str) -> None:
    assert code in _codes(_doc(canopy(points)), cvat_cfg)


def test_polyline_and_box_errors(cvat_cfg: CvatExportConfig) -> None:
    assert "polyline_too_short" in _codes(_doc(row(((10.0, 10.0), (13.0, 10.0)))), cvat_cfg)
    assert "too_few_vertices" in _codes(_doc(row(((10.0, 10.0), (10.0, 10.0)))), cvat_cfg)
    assert "box_degenerate" in _codes(_doc(waste(((60.0, 50.0), (60.0, 58.0)))), cvat_cfg)
    assert "coord_out_of_range" in _codes(_doc(waste(((60.0, 50.0), (2049.0, 58.0)))), cvat_cfg)


def test_duplicate_vertex_is_a_warning(cvat_cfg: CvatExportConfig) -> None:
    pts = ((10.0, 10.0), (10.0, 30.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0))
    rep = validate_document(_doc(canopy(pts)), cfg=cvat_cfg, tile_px=PX)
    assert rep.ok and rep.has_code("duplicate_vertex")


def test_image_level_errors(cvat_cfg: CvatExportConfig) -> None:
    assert "image_size" in _codes(_doc(width=1024), cvat_cfg)
    assert "image_name_invalid" in _codes(_doc(name="images/siret3_r021_c012.tif"), cvat_cfg)
    assert "image_name_invalid" in _codes(_doc(name="siret3_r021_c012.TIF"), cvat_cfg)
    img = CvatImage(0, NAME, PX, PX, ())
    dup = CvatDocument((img, img.with_id(1)))
    assert "duplicate_image" in _codes(dup, cvat_cfg)


def test_canopy_interrow_overlap_threshold(cvat_cfg: CvatExportConfig) -> None:
    # interrow square (100..150); canopy overlapping 8 x 12 = 96 px2 = 0.06 m2 -> error
    over = canopy(_square(142, 100, 12))
    assert "canopy_interrow_overlap" in _codes(_doc(interrow(), over), cvat_cfg)
    # 8 x 8 = 64 px2 = 0.04 m2 -> ok
    ok = canopy(_square(142, 100, 8))
    assert validate_document(_doc(interrow(), ok), cfg=cvat_cfg, tile_px=PX).ok


def test_waste_empty_vineyard_id_rules(cvat_cfg: CvatExportConfig) -> None:
    doc = _doc(waste(vid=""))
    far = validate_document(doc, cfg=cvat_cfg, tile_px=PX, waste_block_dist_m={"W0001": 12.0})
    assert far.ok and not far.warnings
    near = validate_document(doc, cfg=cvat_cfg, tile_px=PX, waste_block_dist_m={"W0001": 5.0})
    assert [i.code for i in near.errors] == ["waste_vid_empty_near_block"]
    unknown = validate_document(doc, cfg=cvat_cfg, tile_px=PX)
    assert unknown.ok and unknown.has_code("waste_block_unknown")
    custom = validate_document(doc, cfg=cvat_cfg, tile_px=PX, waste_block_dist_m={"W0001": 12.0},
                               waste_min_dist_m=20.0)
    assert not custom.ok


def test_shape_source_mismatch_is_a_warning(cvat_cfg: CvatExportConfig) -> None:
    rep = validate_document(_doc(canopy(source="auto")), cfg=cvat_cfg, tile_px=PX)
    assert rep.ok and rep.has_code("shape_source")


def test_issues_carry_location(cvat_cfg: CvatExportConfig) -> None:
    rep = validate_image(CvatImage(0, NAME, PX, PX, (row(structure="Regular"),)), cfg=cvat_cfg, tile_px=PX,
                         zip_name="p.zip")
    issue = rep.errors[0]
    assert issue.tile_id == TILE and issue.zip_name == "p.zip" and issue.object_ref.startswith("row#0")
