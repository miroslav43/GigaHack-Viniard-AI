from __future__ import annotations

import json
from pathlib import Path

import pytest

from vineyard.errors import SchemaError
from vineyard.farms.cadastre import (
    COLUMNS,
    parcels_collection,
    parse_area_ha,
    read_parcels,
    wfs_query,
    write_snapshot,
)

BBOX = (28.70, 47.11, 28.72, 47.13)
RING = [[28.700, 47.120], [28.701, 47.120], [28.701, 47.121], [28.700, 47.121], [28.700, 47.120]]


def parcel(code: str | None, landuse: str = "Teren pentru grădini", aria: str = "0.12 ha") -> dict:
    return {"type": "Feature", "id": f"terenuri.{code}",
            "properties": {"codcadastral": code, "cod_parcel": " 0327", "landuse": landuse, "typeproperty": "NEDETERMINAT",
                           "aria": aria, "description": "<table><tr><td>owner?</td></tr></table>"},
            "geometry": {"type": "Polygon", "coordinates": [RING]}}


@pytest.mark.parametrize(("text", "expected"), [("0.12 ha", 0.12), (" 1,80 HA ", 1.8), ("12 ha", 12.0),
                                                 ("n/a", None), (None, None), ("0.12 m2", None)])
def test_parse_area_ha(text: str | None, expected: float | None) -> None:
    assert parse_area_ha(text) == expected


def test_collection_dedupes_sorts_and_drops_the_html_description() -> None:
    doc = parcels_collection([parcel("80372"), parcel("80371"), parcel("80372"), parcel(None),
                              {**parcel("1"), "geometry": None}], BBOX, "t")
    assert [f["properties"]["codcadastral"] for f in doc["features"]] == ["80371", "80372"]
    props = doc["features"][0]["properties"]
    assert "description" not in props and props["area_ha"] == 0.12 and props["cod_parcel"] == "0327"


def test_wfs_query_axis_order_and_paging() -> None:
    q = wfs_query(BBOX, 2000, 1000)
    assert q["bbox"] == "47.11,28.7,47.13,28.72,urn:ogc:def:crs:EPSG::4326"
    assert (q["startIndex"], q["count"], q["srsName"]) == ("2000", "1000", "EPSG:4326")


def test_snapshot_round_trip_is_utm(tmp_path: Path) -> None:
    path = write_snapshot(parcels_collection([parcel("80371")], BBOX, "t"), tmp_path / "p.geojson")
    frame = read_parcels(path)
    assert frame.crs.to_epsg() == 32635 and tuple(c for c in frame.columns if c != "geometry") == COLUMNS
    assert frame.area_ha.iloc[0] == pytest.approx(0.12)
    assert frame.geometry.iloc[0].area == pytest.approx(76 * 111, rel=0.1)  # ~0.001° lon x 0.001° lat


def test_snapshot_without_landuse_is_rejected(tmp_path: Path) -> None:
    doc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"codcadastral": "1"},
                                                       "geometry": {"type": "Polygon", "coordinates": [RING]}}]}
    path = tmp_path / "bad.geojson"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SchemaError, match="codcadastral and landuse"):
        read_parcels(path)
