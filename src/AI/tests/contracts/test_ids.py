import pytest

from vineyard.contracts.enums import TargetKind
from vineyard.contracts.ids import (
    ID_PATTERNS,
    TARGET_CODES,
    IdKind,
    file_name_from_tile_id,
    format_canopy_id,
    format_chain_id,
    format_interrow_id,
    format_interrow_piece_id,
    format_issue_id,
    format_marcaj_interrow_piece_id,
    format_row_candidate_id,
    format_row_id,
    format_row_piece_id,
    format_target_id,
    format_tile_id,
    format_vineyard_id,
    format_waste_candidate_id,
    format_waste_id,
    interrow_index_of,
    is_valid_id,
    parse_canopy_id,
    parse_interrow_piece_id,
    parse_row_candidate_id,
    parse_row_piece_id,
    parse_target_id,
    parse_tile_id,
    parse_waste_id,
    row_index_of,
    tile_grid_ids,
    tile_id_from_file_name,
)
from vineyard.errors import SchemaError

TILE = "siret3_r021_c012"

STRICT_OK = [
    (IdKind.TILE, TILE),
    (IdKind.BLOCK, "V01"),
    (IdKind.BLOCK, "V123"),
    (IdKind.ROW, "V01-R001"),
    (IdKind.ROW, "V123-R900"),
    (IdKind.INTERROW, "V01-I001"),
    (IdKind.ROW_PIECE, f"V01-R001@{TILE}"),
    (IdKind.ROW_PIECE, f"V01-R001@{TILE}#2"),
    (IdKind.INTERROW_PIECE, f"V01-I001@{TILE}"),
    (IdKind.INTERROW_PIECE, f"V01-I001@{TILE}#3"),
    (IdKind.INTERROW_PIECE, f"{TILE}:I001"),
    (IdKind.CANOPY, f"{TILE}:C0001"),
    (IdKind.ROW_CANDIDATE, f"{TILE}:K07"),
    (IdKind.ROW_CANDIDATE, f"{TILE}:K123"),
    (IdKind.WASTE, "W0001"),
    (IdKind.WASTE, "W0001a"),
    (IdKind.WASTE, "W0001b"),
    (IdKind.WASTE_CANDIDATE, f"{TILE}:W0001"),
    (IdKind.TARGET, "T-GAP-0001"),
    (IdKind.TARGET, "T-MSP-0001"),
    (IdKind.TARGET, "T-SPR-0012"),
    (IdKind.ISSUE, "Q00001"),
    (IdKind.CHAIN, "L00001"),
]

STRICT_BAD = [
    (IdKind.TILE, "siret3_r21_c012"),
    (IdKind.BLOCK, "v01"),
    (IdKind.BLOCK, "V1"),
    (IdKind.ROW, "V01-R01"),
    (IdKind.ROW, "V01-R0001"),
    (IdKind.ROW_PIECE, f"V01-R01@{TILE}"),
    (IdKind.ROW_PIECE, f"V01-R001@{TILE}#"),
    (IdKind.CANOPY, f"{TILE}:C001"),
    (IdKind.ROW_CANDIDATE, f"{TILE}:K7"),
    (IdKind.ROW_CANDIDATE, f"{TILE}:K1234"),
    (IdKind.WASTE, "W0001c"),
    (IdKind.WASTE, "W001"),
    (IdKind.TARGET, "T-MIS-0001"),
    (IdKind.TARGET, "T-GAP-001"),
    (IdKind.ISSUE, "Q0001"),
]


@pytest.mark.parametrize(("kind", "value"), STRICT_OK)
def test_strict_accepts(kind: IdKind, value: str) -> None:
    assert is_valid_id(kind, value)
    assert ID_PATTERNS[kind].fullmatch(value)


@pytest.mark.parametrize(("kind", "value"), STRICT_BAD)
def test_strict_rejects(kind: IdKind, value: str) -> None:
    assert not is_valid_id(kind, value)


def test_relaxed_accepts_example_style_ids() -> None:
    assert is_valid_id(IdKind.ROW, "V01-R01", strict=False)
    assert is_valid_id(IdKind.BLOCK, "v03", strict=False)
    assert not is_valid_id(IdKind.ROW, "   ", strict=False)
    assert not is_valid_id(IdKind.ROW, "", strict=False)


def test_is_valid_id_rejects_non_str() -> None:
    assert not is_valid_id(IdKind.BLOCK, None)  # type: ignore[arg-type]
    assert not is_valid_id(IdKind.BLOCK, 1, strict=False)  # type: ignore[arg-type]


def test_every_kind_has_a_pattern() -> None:
    assert set(ID_PATTERNS) == set(IdKind)


def test_tile_ids_round_trip() -> None:
    assert format_tile_id(21, 12) == TILE
    assert parse_tile_id(TILE) == (21, 12)
    assert tile_id_from_file_name("siret3_r021_c012.tif") == TILE
    assert tile_id_from_file_name("images/siret3_r021_c012.tif") == TILE
    assert file_name_from_tile_id(TILE) == "siret3_r021_c012.tif"


@pytest.mark.parametrize("bad", ["foo.tif", "siret3_r021_c012.png", "siret3_r021_c012.TIF"])
def test_tile_id_from_bad_file_name(bad: str) -> None:
    with pytest.raises(SchemaError):
        tile_id_from_file_name(bad)


def test_parse_tile_id_rejects_garbage() -> None:
    with pytest.raises(SchemaError):
        parse_tile_id("siret3_r21_c012")


def test_format_vineyard_id_width() -> None:
    assert format_vineyard_id(5, 12) == "V05"
    assert format_vineyard_id(5, 120) == "V005"
    assert format_vineyard_id(99, 99) == "V99"
    with pytest.raises(SchemaError):
        format_vineyard_id(0, 12)
    with pytest.raises(SchemaError):
        format_vineyard_id(13, 12)


def test_row_and_interrow_formatters() -> None:
    assert format_row_id("V03", 17) == "V03-R017"
    assert format_interrow_id("V03", 17) == "V03-I017"
    assert row_index_of("V03-R017") == 17
    assert row_index_of("V01-R01") == 1
    assert row_index_of("V01") is None
    assert row_index_of("V01-R900 ") is None
    assert interrow_index_of("V03-I017") == 17
    assert interrow_index_of("x") is None
    with pytest.raises(SchemaError):
        format_row_id("V03", 1000)
    with pytest.raises(SchemaError):
        format_row_id("v03", 1)


def test_piece_ids_round_trip() -> None:
    assert format_row_piece_id("V01-R001", TILE) == f"V01-R001@{TILE}"
    assert format_row_piece_id("V01-R001", TILE, dup=2) == f"V01-R001@{TILE}#2"
    assert parse_row_piece_id(f"V01-R001@{TILE}#2") == ("V01-R001", TILE, 2)
    assert parse_row_piece_id(f"V01-R01@{TILE}") == ("V01-R01", TILE, 1)
    assert format_interrow_piece_id("V01-I004", TILE, dup=3) == f"V01-I004@{TILE}#3"
    assert parse_interrow_piece_id(f"V01-I004@{TILE}#3") == ("V01-I004", TILE, 3)
    assert format_marcaj_interrow_piece_id(TILE, 1) == f"{TILE}:I001"
    with pytest.raises(SchemaError):
        format_row_piece_id("V01-R001", TILE, dup=0)
    with pytest.raises(SchemaError):
        parse_row_piece_id("nonsense")


def test_tile_scoped_ids_round_trip() -> None:
    assert format_canopy_id(TILE, 1) == f"{TILE}:C0001"
    assert parse_canopy_id(f"{TILE}:C0042") == (TILE, 42)
    assert format_row_candidate_id(TILE, 7) == f"{TILE}:K07"
    assert format_row_candidate_id(TILE, 123) == f"{TILE}:K123"
    assert parse_row_candidate_id(f"{TILE}:K123") == (TILE, 123)
    assert format_waste_candidate_id(TILE, 1) == f"{TILE}:W0001"
    with pytest.raises(SchemaError):
        format_row_candidate_id(TILE, 1000)
    with pytest.raises(SchemaError):
        format_canopy_id(TILE, 10000)
    with pytest.raises(SchemaError):
        parse_canopy_id(f"{TILE}:K07")


def test_waste_and_target_ids_round_trip() -> None:
    assert format_waste_id(1) == "W0001"
    assert format_waste_id(12, "a") == "W0012a"
    assert parse_waste_id("W0012b") == (12, "b")
    assert parse_waste_id("W0012") == (12, "")
    with pytest.raises(SchemaError):
        format_waste_id(1, "c")  # type: ignore[arg-type]
    for kind in TargetKind:
        tid = format_target_id(kind, 3)
        assert is_valid_id(IdKind.TARGET, tid)
        assert parse_target_id(tid) == (kind, 3)
    assert format_target_id(TargetKind.MISSING_PLANT, 1) == "T-MSP-0001"
    assert set(TARGET_CODES.values()) == {"GAP", "END", "MRW", "MSP", "SPR", "WST", "OTH"}
    with pytest.raises(SchemaError):
        parse_target_id("T-XXX-0001")


def test_misc_formatters() -> None:
    assert format_issue_id(1) == "Q00001"
    assert format_chain_id(12) == "L00012"


def test_all_formatters_match_patterns() -> None:
    samples = {
        IdKind.TILE: format_tile_id(5, 0),
        IdKind.BLOCK: format_vineyard_id(1, 1),
        IdKind.ROW: format_row_id("V01", 1),
        IdKind.INTERROW: format_interrow_id("V01", 1),
        IdKind.ROW_PIECE: format_row_piece_id("V01-R001", TILE, 4),
        IdKind.INTERROW_PIECE: format_interrow_piece_id("V01-I001", TILE),
        IdKind.CANOPY: format_canopy_id(TILE, 9999),
        IdKind.ROW_CANDIDATE: format_row_candidate_id(TILE, 1),
        IdKind.WASTE: format_waste_id(9999, "b"),
        IdKind.WASTE_CANDIDATE: format_waste_candidate_id(TILE, 5),
        IdKind.TARGET: format_target_id(TargetKind.OTHER, 1),
        IdKind.ISSUE: format_issue_id(99999),
        IdKind.CHAIN: format_chain_id(1),
    }
    for kind, value in samples.items():
        assert is_valid_id(kind, value), (kind, value)


def test_tile_grid_ids() -> None:
    ids = tile_grid_ids()
    assert len(ids) == 311
    assert len(set(ids)) == 311
    assert list(ids) == sorted(ids)
    assert all(is_valid_id(IdKind.TILE, t) for t in ids)
    rows = {parse_tile_id(t)[0] for t in ids}
    cols = {parse_tile_id(t)[1] for t in ids}
    assert (min(rows), max(rows)) == (5, 39)
    assert (min(cols), max(cols)) == (0, 33)
    assert "siret3_r006_c004" in ids
    assert TILE in ids
    assert "siret3_r018_c010" in ids
