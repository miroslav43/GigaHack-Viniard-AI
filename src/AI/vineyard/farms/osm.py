"""OpenStreetMap highways: the committed snapshot (read offline by the `farms` stage) and its refresh.

`fetch_highways` is the only network access of the package (`vineyard osm-fetch`); the pipeline itself only
reads the snapshot, so `docker run --network none` still works. OSM data: ODbL 1.0, attribution required.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import pandas as pd
from pyproj import Transformer

from vineyard.errors import SchemaError
from vineyard.geo.tiling import CRS_EPSG
from vineyard.pipeline.atomic import atomic_write_text

OVERPASS_URL: Final = "https://overpass-api.de/api/interpreter"
USER_AGENT: Final = "solemtrix-vineyard/1.0 (GigaHack 2026)"
ATTRIBUTION: Final = "© OpenStreetMap contributors, ODbL 1.0"
KEEP_TAGS: Final = ("highway", "name", "surface", "tracktype")
COLUMNS: Final = ("osm_id", *KEEP_TAGS)
CRS_4326: Final = "EPSG:4326"
TIMEOUT_S: Final = 90

Bbox = tuple[float, float, float, float]  # (west, south, east, north), degrees


def bbox_4326(bounds_utm: Bbox, pad_m: float) -> Bbox:
    """(west, south, east, north) of UTM bounds (minx, miny, maxx, maxy) grown by `pad_m`, rounded out."""
    minx, miny, maxx, maxy = bounds_utm
    to_4326 = Transformer.from_crs(CRS_EPSG, CRS_4326, always_xy=True)
    xs, ys = to_4326.transform([minx - pad_m, maxx + pad_m, minx - pad_m, maxx + pad_m],
                               [miny - pad_m, miny - pad_m, maxy + pad_m, maxy + pad_m])
    return (round(min(xs), 5), round(min(ys), 5), round(max(xs), 5), round(max(ys), 5))


def overpass_query(bbox: Bbox, timeout_s: int = TIMEOUT_S) -> str:
    west, south, east, north = bbox
    return f'[out:json][timeout:{timeout_s}];way["highway"]({south},{west},{north},{east});out tags geom;'


def _feature(element: Mapping[str, Any]) -> dict[str, Any] | None:
    coords = [[p["lon"], p["lat"]] for p in element.get("geometry") or ()]
    if element.get("type") != "way" or len(coords) < 2:
        return None
    tags = element.get("tags") or {}
    props = {"osm_id": int(element["id"]), **{k: tags.get(k) for k in KEEP_TAGS}}
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def highways_collection(elements: Iterable[Mapping[str, Any]], bbox: Bbox, fetched_at: str) -> dict[str, Any]:
    """RFC 7946 FeatureCollection of the highway ways (sorted by osm_id), with the snapshot's provenance."""
    features = sorted((f for f in map(_feature, elements) if f is not None),
                      key=lambda f: f["properties"]["osm_id"])
    return {"type": "FeatureCollection", "name": "osm_highways", "attribution": ATTRIBUTION,
            "source": OVERPASS_URL, "bbox_4326": list(bbox), "fetched_at": fetched_at, "features": features}


def fetch_highways(bbox: Bbox, fetched_at: str, *, timeout_s: int = TIMEOUT_S) -> dict[str, Any]:
    """Every OSM way tagged highway=* in `bbox`, from the Overpass API (network)."""
    body = urllib.parse.urlencode({"data": overpass_query(bbox, timeout_s)}).encode()
    request = urllib.request.Request(OVERPASS_URL, data=body, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s + 30) as response:  # noqa: S310 - fixed https URL
        payload = json.load(response)
    return highways_collection(payload.get("elements") or (), bbox, fetched_at)


def write_snapshot(collection: Mapping[str, Any], path: Path) -> Path:
    return atomic_write_text(Path(path), json.dumps(collection, ensure_ascii=False, indent=None) + "\n")


def read_highways(path: Path) -> gpd.GeoDataFrame:
    """The snapshot as an EPSG:32635 frame with COLUMNS (missing tags are None)."""
    source = Path(path)
    try:
        frame = gpd.read_file(source)
    except Exception as exc:  # pyogrio raises its own error types for unreadable files
        raise SchemaError("OSM highway snapshot is unreadable", path=str(source), error=str(exc)) from exc
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise SchemaError("OSM highway snapshot must be EPSG:4326 (RFC 7946)", path=str(source), crs=str(frame.crs))
    missing = [c for c in ("osm_id", "highway") if c not in frame.columns]
    if missing:
        raise SchemaError("OSM highway snapshot lacks columns", path=str(source), missing=missing)
    data = {c: [_text(v) for v in frame[c]] if c in frame.columns else [None] * len(frame) for c in KEEP_TAGS}
    out = gpd.GeoDataFrame({"osm_id": frame["osm_id"].astype("int64"), **data}, geometry=frame.geometry.values,
                           crs=CRS_4326)
    return out.to_crs(CRS_EPSG)


def _text(value: Any) -> str | None:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
