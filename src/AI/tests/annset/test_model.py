import dataclasses

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from vineyard.annset.model import ANNSET_LAYERS, AnnSetMeta, empty_annset, make_meta
from vineyard.contracts import CONTRACT_VERSION
from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import coerce_layer
from vineyard.errors import SchemaError

T1 = "siret3_r006_c004"
T2 = "siret3_r021_c012"
T3 = "siret3_r018_c010"

PROV = {"source": "reference", "run_id": "r1", "model_version": "reference-examples@x", "confidence": 1.0,
        "qa_flags": ""}


def _meta(tile_ids: tuple[str, ...] = (T1, T2, T3)) -> AnnSetMeta:
    return AnnSetMeta(
        contract_version=CONTRACT_VERSION, source=Source.REFERENCE, run_id="r1", model_version="m",
        created_at="2026-09-26T00:00:00+03:00", n_tiles=len(tile_ids), counts={}, inputs=("a.xml",),
        tile_ids=tile_ids,
    )


def _canopies() -> gpd.GeoDataFrame:
    rows = [
        {"canopy_id": f"{t}:C0001", "tile_id": t, "vineyard_id": "V01", "row_id": None, "area_m2": 1.0,
         "n_vertices": 4, "along_m": 1.0, "is_clump": False, "touches_edge": False, **PROV}
        for t in (T1, T2)
    ]
    return coerce_layer(gpd.GeoDataFrame(rows, geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)], crs=32635),
                        "canopies")


def _row_pieces() -> gpd.GeoDataFrame:
    rows = [{"piece_id": f"V01-R01@{T2}", "row_id": "V01-R01", "vineyard_id": "V01", "tile_id": T2,
             "row_structure": "regular", "length_m": 10.0, "max_gap_m": 1.0, "n_vertices": 2, **PROV}]
    return coerce_layer(gpd.GeoDataFrame(rows, geometry=[LineString([(0, 0), (10, 0)])], crs=32635), "row_pieces")


def test_layers_constant() -> None:
    assert ANNSET_LAYERS == ("canopies", "row_pieces", "interrow_pieces", "waste")


def test_meta_normalises_and_is_frozen() -> None:
    meta = AnnSetMeta(
        contract_version="1.1", source="marcaj", run_id="r", model_version="m", created_at="t", n_tiles=2,  # type: ignore[arg-type]
        counts={"canopies": 3}, inputs=["b", "a"], tile_ids=[T2, T1],  # type: ignore[arg-type]
    )
    assert meta.source is Source.MARCAJ
    assert meta.tile_ids == (T1, T2)
    assert meta.inputs == ("b", "a")
    assert dict(meta.counts) == {"canopies": 3}
    with pytest.raises(TypeError):
        meta.counts["x"] = 1  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.run_id = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_tiles": 5},
        {"tile_ids": ("bad",), "n_tiles": 1},
        {"tile_ids": (T1, T1), "n_tiles": 2},
        {"source": "somewhere"},
        {"counts": {"canopies": -1}},
    ],
)
def test_meta_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(SchemaError):
        dataclasses.replace(_meta(), **kwargs)


def test_meta_json_round_trip() -> None:
    meta = _meta()
    assert AnnSetMeta.from_json_dict(meta.to_json_dict()) == meta
    doc = meta.to_json_dict()
    assert doc["source"] == "reference" and doc["tile_ids"] == sorted([T1, T2, T3])
    with pytest.raises(SchemaError):
        AnnSetMeta.from_json_dict({"source": "model"})


def test_make_meta_counts_tiles() -> None:
    meta = make_meta(Source.MODEL, "run", "pipe@1", (T2, T1), inputs=("x",), created_at="now")
    assert meta.contract_version == CONTRACT_VERSION
    assert meta.n_tiles == 2 and meta.tile_ids == (T1, T2)
    assert meta.created_at == "now"
    assert make_meta(Source.MODEL, "run", "pipe@1", ()).created_at  # defaults to a timestamp


def test_empty_annset() -> None:
    ann = empty_annset(_meta())
    for name in ANNSET_LAYERS:
        assert len(ann.layer(name)) == 0
    assert dict(ann.meta.counts) == dict.fromkeys(ANNSET_LAYERS, 0)
    assert ann.tile_ids() == frozenset({T1, T2, T3})
    with pytest.raises(SchemaError):
        ann.layer("rows")


def test_with_layer_returns_new_object() -> None:
    ann = empty_annset(_meta())
    canopies = _canopies()
    ann2 = ann.with_layer("canopies", canopies)
    assert ann2 is not ann
    assert len(ann.canopies) == 0 and len(ann2.canopies) == 2
    assert ann2.meta.counts["canopies"] == 2
    assert ann.meta.counts["canopies"] == 0
    with pytest.raises(SchemaError):
        ann.with_layer("rows", canopies)
    with pytest.raises(SchemaError):
        ann.with_layer("canopies", canopies.drop(columns=["tile_id"]))


def test_for_tiles_filters_every_layer() -> None:
    ann = empty_annset(_meta()).with_layer("canopies", _canopies()).with_layer("row_pieces", _row_pieces())
    sub = ann.for_tiles([T2, "siret3_r099_c099"])
    assert sub.canopies["tile_id"].tolist() == [T2]
    assert len(sub.row_pieces) == 1
    assert sub.meta.tile_ids == (T2,)
    assert sub.meta.n_tiles == 1
    assert dict(sub.meta.counts) == {"canopies": 1, "row_pieces": 1, "interrow_pieces": 0, "waste": 0}
    assert len(ann.canopies) == 2  # original untouched
    assert sub.canopies.index.tolist() == [0]


def test_tile_ids_include_object_tiles() -> None:
    ann = empty_annset(_meta((T3,))).with_layer("canopies", _canopies())
    assert ann.tile_ids() == frozenset({T1, T2, T3})


def test_annset_rejects_non_frames() -> None:
    ann = empty_annset(_meta())
    with pytest.raises(SchemaError):
        dataclasses.replace(ann, waste="nope")
