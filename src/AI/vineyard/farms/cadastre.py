"""Cadastral parcels of the AGCC (Agency for Geodesy, Cartography and Cadastre, Moldova): the local snapshot.

`fetch_parcels` is the only network access (`vineyard cadastre-fetch`): the public WFS of geodata.gov.md,
reprojected by the server to EPSG:4326. The snapshot stays local (data/cadastre is gitignored) until the
AGCC confirms the reuse terms; the pipeline reads it when present and works without it. Only the parcel
number, land use, property type and area are kept: no owner data is exposed or stored.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import pandas as pd

from vineyard.errors import SchemaError
from vineyard.farms.osm import CRS_4326, Bbox, https_context
from vineyard.geo.tiling import CRS_EPSG
from vineyard.pipeline.atomic import atomic_write_text

WFS_URL: Final = "https://geodata.gov.md/geoserver/cadastru_data/ows"
LAYER: Final = "cadastru_data:terenuri"
ATTRIBUTION: Final = "Cadastru © AGCC (geodata.gov.md), informativ"
USER_AGENT: Final = "solemtrix-vineyard/1.0 (GigaHack 2026)"
KEEP: Final = ("codcadastral", "cod_parcel", "landuse", "typeproperty")
COLUMNS: Final = (*KEEP, "area_ha")
PAGE: Final = 1000
TIMEOUT_S: Final = 90
_AREA_RE: Final = re.compile(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s*ha\s*$", re.IGNORECASE)


def parse_area_ha(text: Any) -> float | None:
    """'0.12 ha' -> 0.12 (None when the text is not an area in hectares)."""
    match = _AREA_RE.match(str(text)) if text is not None else None
    return float(match.group(1).replace(",", ".")) if match else None


def wfs_query(bbox: Bbox, start: int, count: int) -> dict[str, str]:
    west, south, east, north = bbox
    return {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": LAYER,
            "srsName": "EPSG:4326", "outputFormat": "application/json", "count": str(count),
            "startIndex": str(start), "bbox": f"{south},{west},{north},{east},urn:ogc:def:crs:EPSG::4326"}


def _feature(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    props = raw.get("properties") or {}
    geom = raw.get("geometry")
    if not geom or not props.get("codcadastral"):
        return None
    kept = {k: (str(props[k]).strip() if props.get(k) is not None else None) for k in KEEP}
    return {"type": "Feature", "properties": {**kept, "area_ha": parse_area_ha(props.get("aria"))}, "geometry": geom}


def parcels_collection(features: Iterable[Mapping[str, Any]], bbox: Bbox, fetched_at: str) -> dict[str, Any]:
    """RFC 7946 FeatureCollection of the parcels (one per cadastral number, sorted), with provenance."""
    unique: dict[str, dict[str, Any]] = {}
    for feature in filter(None, map(_feature, features)):
        unique.setdefault(feature["properties"]["codcadastral"], feature)
    return {"type": "FeatureCollection", "name": "cadastre_parcels", "attribution": ATTRIBUTION,
            "source": f"{WFS_URL} {LAYER}", "bbox_4326": list(bbox), "fetched_at": fetched_at,
            "features": [unique[k] for k in sorted(unique)]}


def _get(params: Mapping[str, str], timeout_s: int) -> dict[str, Any]:
    url = f"{WFS_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s, context=https_context()) as response:  # noqa: S310 - fixed https URL
        return json.load(response)


def fetch_parcels(bbox: Bbox, fetched_at: str, *, page: int = PAGE, timeout_s: int = TIMEOUT_S,
                  max_pages: int = 100) -> dict[str, Any]:
    """Every parcel intersecting `bbox`, page by page (network)."""
    features: list[Mapping[str, Any]] = []
    for k in range(max_pages):
        batch = _get(wfs_query(bbox, k * page, page), timeout_s).get("features") or []
        features.extend(batch)
        if len(batch) < page:
            return parcels_collection(features, bbox, fetched_at)
    raise SchemaError("cadastre WFS paging did not end", pages=max_pages, page=page)


def write_snapshot(collection: Mapping[str, Any], path: Path) -> Path:
    return atomic_write_text(Path(path), json.dumps(collection, ensure_ascii=False) + "\n")


def read_parcels(path: Path) -> gpd.GeoDataFrame:
    """The snapshot as an EPSG:32635 frame with COLUMNS."""
    source = Path(path)
    try:
        frame = gpd.read_file(source)
    except Exception as exc:  # pyogrio raises its own error types for unreadable files
        raise SchemaError("cadastre snapshot is unreadable", path=str(source), error=str(exc)) from exc
    missing = [c for c in ("codcadastral", "landuse") if c not in frame.columns]
    if missing or frame.crs is None or frame.crs.to_epsg() != 4326:
        raise SchemaError("cadastre snapshot must be EPSG:4326 with codcadastral and landuse", path=str(source),
                          missing=missing, crs=str(frame.crs))
    data = {c: [_text(v) for v in frame[c]] if c in frame.columns else [None] * len(frame) for c in KEEP}
    area = pd.to_numeric(frame["area_ha"], errors="coerce") if "area_ha" in frame.columns else None
    out = gpd.GeoDataFrame({**data, "area_ha": area if area is not None else [None] * len(frame)},
                           geometry=frame.geometry.values, crs=CRS_4326)
    return out.to_crs(CRS_EPSG)


def _text(value: Any) -> str | None:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
