from __future__ import annotations

import csv
import io
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box

from vineyard.contracts.ids import tile_grid_ids
from vineyard.contracts.schemas import validate_layer
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.perception.waste.confirm import (
    CONFIRM_HEADER,
    ConfirmationFileError,
    MergeParams,
    confirm_line,
    merge_confirmed,
    read_confirmations,
)
from vineyard.perception.waste.types import DECISIONS, BoxPx, Category

T1 = "siret3_r021_c012"
T2 = "siret3_r006_c004"
KNOWN = frozenset({T1, T2})
HEADER = ",".join(CONFIRM_HEADER)
CRS = "EPSG:32635"
P = MergeParams(
    match_iou=0.5,
    block_assign_max_m=10.0,
    edge_tol_px=1.0,
    run_id="20260926T0100-model-abcdef",
    model_version="pipe@0.1.0+abcdef0;nn=none;waste=rule@1",
)
TILES = {T1: tile_ref(T1), T2: tile_ref(T2)}


def write_csv(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "waste_confirmed.csv"
    path.write_text("\n".join([HEADER, *lines]) + "\n", encoding="utf-8")
    return path


def cand_frame(rows: list[dict]) -> gpd.GeoDataFrame:
    base = {
        "tile_id": T1,
        "category": "unknown",
        "detector": "rule",
        "confidence": 0.4,
        "auto": False,
        "reject_reason": None,
    }
    recs = [{**base, **r} for r in rows]
    return gpd.GeoDataFrame(recs, geometry=[box(0, 0, 1, 1)] * len(recs), crs=CRS)


def blocks_over(tile_id: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"vineyard_id": ["V01"]}, geometry=[tile_box(tile_ref(tile_id))], crs=CRS)


NO_BLOCKS = gpd.GeoDataFrame({"vineyard_id": []}, geometry=[], crs=CRS)


def test_header_only_file_and_missing_file(tmp_path: Path) -> None:
    assert read_confirmations(write_csv(tmp_path), KNOWN) == ()
    assert read_confirmations(tmp_path / "absent.csv", KNOWN) == ()


def test_committed_file_parses(project_root: Path) -> None:
    confs = read_confirmations(project_root / "configs" / "waste_confirmed.csv", frozenset(tile_grid_ids()))
    assert all(c.decision in DECISIONS for c in confs)


def test_valid_rows_comments_and_blank_lines(tmp_path: Path) -> None:
    path = write_csv(
        tmp_path,
        f"{T1},100,200,140,230,accept,bag,ana,blue bag",
        "",
        "# comment",
        f"{T2},10,10,20,20,ADD,,ion,",
    )
    a, b = read_confirmations(path, KNOWN)
    assert (a.line_no, a.decision, a.category, a.box) == (
        2,
        "accept",
        Category.BAG,
        BoxPx(100, 200, 140, 230),
    )
    assert (b.line_no, b.decision, b.category) == (5, "add", Category.UNKNOWN)


@pytest.mark.parametrize(
    ("line", "field"),
    [
        (f"{T1},1,2,3,4,maybe,bag,,", "decision"),
        ("siret3_r999_c999,1,2,3,4,accept,bag,,", "tile_id"),
        (f"{T1},x,2,3,4,accept,bag,,", "xtl"),
        (f"{T1},5,2,3,4,accept,bag,,", "box"),
        (f"{T1},1,2,3,4,accept,sofa,,", "category"),
        (f"{T1},1,2,3,4,accept", "columns"),
    ],
)
def test_line_level_errors(tmp_path: Path, line: str, field: str) -> None:
    path = write_csv(tmp_path, f"{T1},1,2,3,4,accept,bag,,", line)
    with pytest.raises(ConfirmationFileError) as err:
        read_confirmations(path, KNOWN)
    assert err.value.line_no == 3 and err.value.field == field
    assert "line 3" in str(err.value)


def test_bad_header(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    path.write_text("tile,x\n", encoding="utf-8")
    with pytest.raises(ConfirmationFileError) as err:
        read_confirmations(path, KNOWN)
    assert err.value.field == "header"


def test_confirm_line_round_trips(tmp_path: Path) -> None:
    line = confirm_line(T1, BoxPx(10.04, 20, 30, 40.25), note='bag, "blue"')
    assert next(csv.reader(io.StringIO(line)))[:6] == [T1, "10.0", "20.0", "30.0", "40.2", "accept"]
    (c,) = read_confirmations(write_csv(tmp_path, line), KNOWN)
    assert c.note == 'bag, "blue"' and c.decision == "accept"


def test_accept_of_review_candidate_is_exported(tmp_path: Path) -> None:
    cands = cand_frame([{"px_xtl": 100.0, "px_ytl": 100.0, "px_xbr": 140.0, "px_ybr": 140.0}])
    confs = read_confirmations(write_csv(tmp_path, f"{T1},102,101,141,139,accept,bottle,ana,"), KNOWN)
    out = merge_confirmed(cands, confs, blocks_over(T1), TILES, P)
    validate_layer(out, "waste")
    assert list(out["waste_id"]) == ["W0001"]
    row = out.iloc[0]
    assert (row["px_xtl"], row["px_xbr"], row["category"], row["detector"]) == (
        100.0,
        140.0,
        "bottle",
        "rule",
    )
    assert (row["vineyard_id"], row["dist_block_m"], row["confidence"], bool(row["exported"])) == (
        "V01",
        0.0,
        1.0,
        True,
    )
    assert row["area_m2"] == pytest.approx(1600 * 0.000625)
    assert out.geometry.iloc[0].within(tile_box(tile_ref(T1)).buffer(1e-6))


def test_reject_overrides_auto(tmp_path: Path) -> None:
    cands = cand_frame(
        [
            {"px_xtl": 100.0, "px_ytl": 100.0, "px_xbr": 140.0, "px_ybr": 140.0, "auto": True},
            {"px_xtl": 500.0, "px_ytl": 500.0, "px_xbr": 520.0, "px_ybr": 520.0, "auto": True},
        ]
    )
    confs = read_confirmations(write_csv(tmp_path, f"{T1},104,104,144,144,reject,,ana,"), KNOWN)
    out = merge_confirmed(cands, confs, NO_BLOCKS, TILES, P)
    assert list(out["px_xtl"]) == [500.0]
    assert out.iloc[0]["confidence"] == pytest.approx(0.4)


def test_add_is_manual_with_vineyard(tmp_path: Path) -> None:
    confs = read_confirmations(write_csv(tmp_path, f"{T2},10,10,30,30,add,tyre,ion,"), KNOWN)
    out = merge_confirmed(cand_frame([]).iloc[0:0], confs, blocks_over(T2), TILES, P)
    validate_layer(out, "waste")
    row = out.iloc[0]
    assert (row["detector"], row["confidence"], row["vineyard_id"], row["category"]) == (
        "manual",
        1.0,
        "V01",
        "tyre",
    )


def test_unmatched_accept_keeps_own_box_and_flag(tmp_path: Path) -> None:
    cands = cand_frame([{"px_xtl": 100.0, "px_ytl": 100.0, "px_xbr": 140.0, "px_ybr": 140.0}])
    confs = read_confirmations(write_csv(tmp_path, f"{T1},900,900,950,950,accept,,ana,"), KNOWN)
    out = merge_confirmed(cands, confs, NO_BLOCKS, TILES, P)
    validate_layer(out, "waste")
    row = out.iloc[0]
    assert (row["px_xtl"], row["qa_flags"], row["detector"], row["vineyard_id"]) == (
        900.0,
        "confirm_unmatched",
        "manual",
        "",
    )


def test_rejected_candidates_are_never_auto_and_nothing_to_export() -> None:
    cands = cand_frame(
        [
            {
                "px_xtl": 1.0,
                "px_ytl": 1.0,
                "px_xbr": 9.0,
                "px_ybr": 9.0,
                "auto": True,
                "reject_reason": "near_axis",
            }
        ]
    )
    out = merge_confirmed(cands, (), NO_BLOCKS, TILES, P)
    validate_layer(out, "waste")
    assert out.empty


def test_unknown_tile_in_confirmation_raises(tmp_path: Path) -> None:
    confs = read_confirmations(write_csv(tmp_path, f"{T1},10,10,30,30,add,,,"), KNOWN)
    with pytest.raises(Exception, match="tile index"):
        merge_confirmed(cand_frame([]).iloc[0:0], confs, NO_BLOCKS, {}, P)
