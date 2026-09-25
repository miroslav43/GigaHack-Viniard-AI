"""cvat.to_annset: CvatDocument -> AnnSet + qa_issues (synthetic cases)."""

import math

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

from vineyard.config import load_config
from vineyard.contracts.enums import Severity, Source
from vineyard.contracts.schemas import validate_layer
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.to_annset import assign_canopy_rows, document_to_annset
from vineyard.errors import CvatFormatError
from vineyard.geo.tiling import GSD_M, px_to_utm, tile_ref

R021, R006, R005 = "siret3_r021_c012", "siret3_r006_c004", "siret3_r005_c004"
CFG = load_config()
REFS = {t: tile_ref(t) for t in (R021, R006, R005)}


def _row(pts, rid="V01-R001", vid="V01", st="regular") -> CvatShape:
    return CvatShape("polyline", "row", pts, (("vineyard_id", vid), ("row_id", rid), ("row_structure", st)))


def _poly(label, pts, **attrs) -> CvatShape:
    return CvatShape("polygon", label, pts, tuple(attrs.items()))


def _sq(u0, v0, s=40.0):
    return ((u0, v0), (u0, v0 + s), (u0 + s, v0 + s), (u0 + s, v0))


def _convert(images, source=Source.REFERENCE, **kw):
    return document_to_annset(CvatDocument(tuple(images)), tile_refs=REFS, source=source, run_id="run-1",
                              model_version="reference-examples@deadbeef", cfg=CFG.import_,
                              canopy_cfg=CFG.canopy, **kw)


def _issues(qa, code):
    return qa[qa["code"] == code]


@pytest.fixture(scope="module")
def converted():
    shapes = (
        _row(((0.0, 100.0), (2048.0, 100.0))),
        _row(((0.0, 200.0), (2048.0, 200.0)), st=" Regular "),
        _row(((0.0, 300.0), (1000.0, 300.0)), rid="V01-R001"),  # duplicate row_id in the tile -> #2
        CvatShape("polyline", "row", ((0.0, 400.0), (50.0, 400.0)), (("vineyard_id", "V01"),)),  # no row_id
        _poly("interrow_area", ((0.0, 110.0), (0.0, 190.0), (2048.0, 190.0), (2048.0, 110.0)),
              vineyard_id="V01", interrow_cover="bare soil"),
        _poly("vineyard", _sq(100.0, 90.0), vineyard_id="V01"),                      # on R001 -> assigned
        _poly("vineyard", _sq(600.0, 500.0), vineyard_id="V01"),                     # far -> no row
        _poly("vineyard", ((0.0, 0.0), (40.0, 40.0), (40.0, 0.0), (0.0, 40.0)), vineyard_id="V01"),  # bowtie
        _poly("vineyard", _sq(1500.0, 1500.0, 80.0), vineyard_id=" V03 "),
        _poly("vineyard", _sq(1700.0, 1500.0), vineyard_id="v03"),
        _poly("vine", _sq(10.0, 10.0), vineyard_id="V01"),                           # unknown label
        CvatShape("box", "vineyard", ((1.0, 1.0), (5.0, 5.0)), (("vineyard_id", "V01"),)),  # wrong shape
        _poly("vineyard", ((1.0, 1.0), (2.0, 2.0), (1.0, 1.0)), vineyard_id="V01"),  # degenerate
        CvatShape("box", "waste", ((540.0, 630.0), (500.0, 600.0)), (("vineyard_id", ""),)),  # swapped
        CvatShape("box", "waste", ((100.0, 50.0), (140.0, 60.0)), (("vineyard_id", "V01"),)),
    )
    r006_box = CvatShape("box", "waste", ((10.0, 10.0), (20.0, 20.0)), (("vineyard_id", ""),))
    images = (CvatImage(0, R021 + ".tif", 2048, 2048, shapes),
              CvatImage(1, "images/" + R006 + ".tif", 2048, 2048, (r006_box,)),
              CvatImage(2, R005, 2048, 2048, ()))
    return _convert(images, inputs=("a.xml",))


def test_meta_lists_every_image(converted) -> None:
    annset, _ = converted
    assert annset.meta.tile_ids == (R005, R006, R021)
    assert annset.meta.source == Source.REFERENCE
    assert annset.meta.inputs == ("a.xml",)
    assert annset.meta.counts == {"canopies": 6, "row_pieces": 3, "interrow_pieces": 1, "waste": 3}


def test_layers_are_valid(converted) -> None:
    annset, qa = converted
    for name in ("canopies", "row_pieces", "interrow_pieces", "waste"):
        validate_layer(annset.layer(name), name)
    validate_layer(qa, "qa_issues")
    assert set(annset.canopies["confidence"]) == {1.0}
    assert set(annset.canopies["source"]) == {"reference"}


def test_row_pieces(converted) -> None:
    rows = converted[0].row_pieces
    assert sorted(rows["piece_id"]) == [f"V01-R001@{R021}", f"V01-R001@{R021}#2", f"V01-R001@{R021}#3"]
    assert set(rows["row_structure"]) == {"regular"}
    assert rows["length_m"].iloc[0] == pytest.approx(2048 * GSD_M)
    assert rows["max_gap_m"].isna().all()
    assert set(rows["n_vertices"]) == {2}


def test_canopy_ids_rows_and_attributes(converted) -> None:
    can = converted[0].canopies.set_index("canopy_id")
    assert list(can.index) == [f"{R021}:C{k:04d}" for k in range(1, 7)]
    assert can.loc[f"{R021}:C0001", "row_id"] == "V01-R001"
    assert can.loc[f"{R021}:C0001", "along_m"] == pytest.approx(40 * GSD_M)
    assert can.loc[f"{R021}:C0001", "area_m2"] == pytest.approx(1600 * GSD_M**2)
    assert can.loc[f"{R021}:C0001", "n_vertices"] == 4
    assert bool(can.loc[f"{R021}:C0001", "touches_edge"]) is False
    assert pd.isna(can.loc[f"{R021}:C0002", "row_id"])
    assert math.isnan(can.loc[f"{R021}:C0002", "along_m"])
    assert bool(can.loc[f"{R021}:C0003", "touches_edge"]) is True  # bowtie part at u=0
    assert can.loc[f"{R021}:C0005", "vineyard_id"] == "V03"
    assert bool(can.loc[f"{R021}:C0005", "is_clump"]) is True  # 80 px square = 4 m2 > 2 m2
    assert bool(can.loc[f"{R021}:C0001", "is_clump"]) is False


def test_polygons_ccw_valid(converted) -> None:
    annset = converted[0]
    for name in ("canopies", "interrow_pieces", "waste"):
        for geom in annset.layer(name).geometry:
            assert geom.is_valid and geom.exterior.is_ccw


def test_interrow_piece(converted) -> None:
    ir = converted[0].interrow_pieces
    assert list(ir["piece_id"]) == [f"{R021}:I001"]
    assert ir["interrow_cover"].iloc[0] == "bare_soil"
    assert ir["interrow_id"].isna().all()
    assert ir["area_m2"].iloc[0] == pytest.approx(80 * 2048 * GSD_M**2)
    assert int(ir["n_notches"].iloc[0]) == 0


def test_waste_ids_sorted_by_tile_ytl_xtl(converted) -> None:
    w = converted[0].waste
    assert list(w["waste_id"]) == ["W0001", "W0002", "W0003"]
    assert list(w["tile_id"]) == [R006, R021, R021]
    assert list(w["px_ytl"]) == [10.0, 50.0, 600.0]
    swapped = w[w["px_ytl"] == 600.0].iloc[0]
    assert (swapped["px_xtl"], swapped["px_xbr"], swapped["vineyard_id"]) == (500.0, 540.0, "")
    assert swapped["area_m2"] == pytest.approx(40 * 30 * GSD_M**2)
    assert set(w["detector"]) == {"manual"} and w["exported"].all()


def test_qa_issues(converted) -> None:
    qa = converted[1]
    assert len(_issues(qa, "schema_violation")) == 2           # label vine + vineyard as box
    assert len(_issues(qa, "geom_dropped")) == 1               # degenerate polygon
    assert len(_issues(qa, "missing_attr")) == 2               # row without row_id and row_structure
    assert len(_issues(qa, "id_case_collision")) == 1          # V03 / v03
    assert len(_issues(qa, "id_whitespace")) == 1
    assert len(_issues(qa, "enum_normalized")) == 2            # " Regular ", "bare soil"
    repaired = _issues(qa, "geom_repaired")
    assert len(repaired) == 2                                  # bowtie split, swapped box
    assert set(qa[qa["severity"] == "error"]["code"]) == {
        "schema_violation", "geom_dropped", "missing_attr", "id_case_collision"}
    assert qa["issue_id"].iloc[0] == "Q00001"
    assert (qa["tile_id"] == R021).sum() >= 8


def test_extra_issues_are_merged() -> None:
    from vineyard.contracts.qa import QaIssue
    extra = QaIssue(Severity.INFO, "geom_dropped", R021, "tag[3]", "tag ignored")
    _, qa = _convert((CvatImage(0, R021, 2048, 2048, ()),), extra_issues=(extra,))
    assert list(qa["object_id"]) == ["tag[3]"]


@pytest.mark.parametrize(
    "image",
    [
        CvatImage(0, "siret3_r099_c099.tif", 2048, 2048, ()),   # not in tile_refs
        CvatImage(0, "SIRET3_R021_C012.tif", 2048, 2048, ()),   # case mismatch
        CvatImage(0, R021, 1024, 1024, ()),                      # wrong size
    ],
)
def test_structural_errors(image: CvatImage) -> None:
    with pytest.raises(CvatFormatError):
        _convert((image,))


def test_duplicate_image_in_document_is_structural() -> None:
    img = CvatImage(0, R021, 2048, 2048, ())
    with pytest.raises(CvatFormatError):
        _convert((img, img.with_id(1)))


def test_model_source_passes_strict_validation() -> None:
    shapes = (_row(((0.0, 100.0), (2048.0, 100.0))), _poly("vineyard", _sq(100.0, 90.0), vineyard_id="V01"))
    annset, qa = _convert((CvatImage(0, R021, 2048, 2048, shapes),), source=Source.MODEL)
    validate_layer(annset.row_pieces, "row_pieces")
    validate_layer(annset.canopies, "canopies")
    assert len(qa) == 0


def test_utm_mapping_of_first_row_point() -> None:
    annset, _ = _convert((CvatImage(0, R021, 2048, 2048, (_row(((2048.0, 32.1), (2017.8, 0.0))),)),))
    x, y = annset.row_pieces.geometry.iloc[0].coords[0]
    assert (x, y) == pytest.approx((629657.6, 5220146.3975), abs=1e-6)


def test_assign_ties_break_by_perpendicular_then_piece_id() -> None:
    t = REFS[R021]
    line_a = LineString(px_to_utm(t, np.array([[0.0, 100.0], [2048.0, 100.0]])))
    line_b = LineString(px_to_utm(t, np.array([[0.0, 110.0], [2048.0, 110.0]])))
    # canopy crosses both lines (distance 0 to both); centroid is nearer to B
    canopy = Polygon(px_to_utm(t, np.array(_sq(500.0, 97.0, 20.0))))
    pieces = (("V01-R001", "p_a", line_a), ("V01-R002", "p_b", line_b))
    assert assign_canopy_rows([canopy], pieces, max_m=0.5)[0][0] == "V01-R002"
    # symmetric canopy: equal perpendicular distances -> lowest piece_id wins
    sym = Polygon(px_to_utm(t, np.array(_sq(500.0, 95.0, 20.0))))
    pieces_rev = (("V01-R009", "p_z", line_b), ("V01-R001", "p_a", line_a))
    assert assign_canopy_rows([sym], pieces_rev, max_m=0.5)[0][0] == "V01-R001"
    assert assign_canopy_rows([canopy], (), max_m=0.5) == [(None, None)]


# ---------------------------------------------------------------- examples (acceptance M1)

EXPECTED = {  # tile: (canopies, Σ canopy m2, rows, Σ row m, interrows, Σ interrow m2)
    R021: (399, 237.119, 25, 910.104, 24, 2068.031),
    R006: (251, 299.056, 26, 1031.451, 25, 1996.361),
}


@pytest.fixture(scope="module")
def reference(examples_dir):
    from vineyard.cvat.reader import read_cvat_file

    doc, issues = read_cvat_file(examples_dir / "annotations.xml")
    refs = {img.tile_id: tile_ref(img.tile_id) for img in doc.images}
    annset, qa = document_to_annset(doc, tile_refs=refs, source=Source.REFERENCE, run_id="ref",
                                    model_version="reference-examples@x", cfg=CFG.import_, canopy_cfg=CFG.canopy,
                                    extra_issues=issues)
    return doc, annset, qa


@pytest.mark.examples
def test_reference_counts_and_sums(reference) -> None:
    _, annset, qa = reference
    assert len(qa) == 0
    assert annset.meta.tile_ids == (R006, R021)
    assert len(annset.waste) == 0
    for tile, (n_can, a_can, n_row, l_row, n_ir, a_ir) in EXPECTED.items():
        can = annset.canopies[annset.canopies["tile_id"] == tile]
        row = annset.row_pieces[annset.row_pieces["tile_id"] == tile]
        ir = annset.interrow_pieces[annset.interrow_pieces["tile_id"] == tile]
        assert (len(can), len(row), len(ir)) == (n_can, n_row, n_ir)
        assert can["area_m2"].sum() == pytest.approx(a_can, abs=0.01)
        assert row["length_m"].sum() == pytest.approx(l_row, abs=0.01)
        assert ir["area_m2"].sum() == pytest.approx(a_ir, abs=0.01)


@pytest.mark.examples
def test_reference_attributes_and_ids(reference) -> None:
    _, annset, _ = reference
    rows = annset.row_pieces
    assert sorted(rows[rows["row_structure"] == "disrupted"]["row_id"]) == [
        "V02-R06", "V02-R07", "V02-R08", "V02-R09", "V02-R23"]
    ir = annset.interrow_pieces
    assert ir[ir["tile_id"] == R006]["interrow_cover"].value_counts().to_dict() == {"bare_soil": 21, "mixed": 4}
    assert set(ir[ir["tile_id"] == R021]["interrow_cover"]) == {"bare_soil"}
    can = annset.canopies
    assert set(can[can["tile_id"] == R021]["vineyard_id"]) == {"V01"}
    assert set(can[can["tile_id"] == R006]["vineyard_id"]) == {"V02"}
    assert f"{R021}:C0399" in set(can["canopy_id"]) and f"{R021}:I024" in set(ir["piece_id"])
    assert f"V01-R01@{R021}" in set(rows["piece_id"])
    assert can["row_id"].notna().mean() >= 0.99
    per_row = can.groupby("row_id").size()
    assert [per_row[r] for r in ("V02-R17", "V02-R24", "V02-R25", "V02-R26", "V02-R01")] == [1, 1, 1, 1, 2]


@pytest.mark.examples
def test_reference_polygons_ccw_no_closing_vertex(reference) -> None:
    _, annset, _ = reference
    for name in ("canopies", "interrow_pieces"):
        for geom in annset.layer(name).geometry:
            assert geom.is_valid and geom.exterior.is_ccw
    assert (annset.canopies["n_vertices"] >= 3).all()


def _inverse_points(geom, tref) -> np.ndarray:
    """Minimal AnnSet -> CVAT inverse for the test: UTM -> px, drop the closing vertex, 1 decimal."""
    from vineyard.geo.tiling import utm_to_px

    coords = np.asarray(geom.exterior.coords if isinstance(geom, Polygon) else geom.coords)
    closed = isinstance(geom, Polygon)
    return np.round(utm_to_px(tref, coords[:-1] if closed else coords), 1)


@pytest.mark.examples
def test_reference_inverse_within_006_px(reference) -> None:
    doc, annset, _ = reference
    layers = {"vineyard": (annset.canopies, "canopy_id"), "row": (annset.row_pieces, None),
              "interrow_area": (annset.interrow_pieces, "piece_id")}
    worst = 0.0
    for img in doc.images:
        tref = tile_ref(img.tile_id)
        for label, (gdf, id_col) in layers.items():
            sub = gdf[gdf["tile_id"] == img.tile_id]
            if id_col is not None:
                sub = sub.sort_values(id_col)
            originals = [np.asarray(s.points) for s in img.shapes if s.label == label]
            assert len(originals) == len(sub)
            for orig, geom in zip(originals, sub.geometry, strict=True):
                back = _inverse_points(geom, tref)
                assert back.shape == orig.shape
                worst = max(worst, float(np.abs(back - orig).max()))
    assert worst <= 0.06


@pytest.mark.examples
def test_reference_annset_writes_and_reads(reference, tmp_path) -> None:
    from vineyard.annset.io import read_annset, write_annset

    _, annset, _ = reference
    back = read_annset(write_annset(annset, tmp_path / "annset"))
    assert back.meta.counts == annset.meta.counts
    assert back.canopies["area_m2"].sum() == pytest.approx(annset.canopies["area_m2"].sum())
