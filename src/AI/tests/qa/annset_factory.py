"""Test builders for AnnSet layers: small synthetic frames and the reference-shaped AnnSet of the examples."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, box

from tests.helpers.examples import load_examples, to_utm
from vineyard.annset.model import AnnSet, make_meta
from vineyard.contracts.enums import Source
from vineyard.contracts.ids import format_canopy_id, format_marcaj_interrow_piece_id
from vineyard.contracts.schemas import coerce_layer, empty_layer
from vineyard.geo.tiling import CRS_EPSG, tile_ref

TILE_A = "siret3_r021_c012"
TILE_B = "siret3_r006_c004"


def _prov(n: int, source: Source) -> dict[str, list]:
    return {"source": [source.value] * n, "run_id": ["t"] * n, "model_version": ["mv"] * n,
            "confidence": [1.0] * n, "qa_flags": [""] * n}


def local(tile_id: str, x: float, y: float) -> tuple[float, float]:
    """Metres from the tile's lower-left corner -> UTM."""
    t = tile_ref(tile_id)
    return (t.x0 + x, t.y0 - 51.2 + y)


def rect(tile_id: str, x0: float, y0: float, x1: float, y1: float) -> Polygon:
    a, b = local(tile_id, x0, y0), local(tile_id, x1, y1)
    return box(a[0], a[1], b[0], b[1])


def hline(tile_id: str, y: float, x0: float = 0.0, x1: float = 51.2) -> LineString:
    return LineString([local(tile_id, x0, y), local(tile_id, x1, y)])


def canopies(items: Sequence[tuple[str, str, Polygon]], source: Source = Source.MODEL) -> gpd.GeoDataFrame:
    """items: (tile_id, vineyard_id, polygon)."""
    if not items:
        return empty_layer("canopies")
    counters: dict[str, int] = {}
    ids = []
    for tile_id, _, _ in items:
        counters[tile_id] = counters.get(tile_id, 0) + 1
        ids.append(format_canopy_id(tile_id, counters[tile_id]))
    n = len(items)
    data = {"canopy_id": ids, "tile_id": [t for t, _, _ in items], "vineyard_id": [v for _, v, _ in items],
            "row_id": [None] * n, "area_m2": [p.area for _, _, p in items],
            "n_vertices": [len(p.exterior.coords) - 1 for _, _, p in items], "along_m": [np.nan] * n,
            "is_clump": [p.area > 2.0 for _, _, p in items], "touches_edge": [False] * n, **_prov(n, source)}
    return coerce_layer(gpd.GeoDataFrame(data, geometry=[p for _, _, p in items], crs=CRS_EPSG), "canopies")


def row_pieces(items: Sequence[tuple[str, str, LineString]], source: Source = Source.MODEL,
               structure: str = "regular") -> gpd.GeoDataFrame:
    """items: (tile_id, row_id, line); vineyard_id is the row_id prefix."""
    if not items:
        return empty_layer("row_pieces")
    n = len(items)
    data = {"piece_id": [f"{r}@{t}" for t, r, _ in items], "row_id": [r for _, r, _ in items],
            "vineyard_id": [r.split("-")[0] for _, r, _ in items], "tile_id": [t for t, _, _ in items],
            "row_structure": [structure] * n, "length_m": [g.length for _, _, g in items],
            "max_gap_m": [np.nan] * n, "n_vertices": [len(g.coords) for _, _, g in items], **_prov(n, source)}
    return coerce_layer(gpd.GeoDataFrame(data, geometry=[g for _, _, g in items], crs=CRS_EPSG), "row_pieces")


def interrow_pieces(items: Sequence[tuple[str, str, Polygon]], source: Source = Source.MODEL,
                    cover: str = "bare_soil") -> gpd.GeoDataFrame:
    """items: (tile_id, interrow_id, polygon); vineyard_id is the interrow_id prefix."""
    if not items:
        return empty_layer("interrow_pieces")
    n = len(items)
    data = {"piece_id": [f"{i}@{t}" for t, i, _ in items], "interrow_id": [i for _, i, _ in items],
            "vineyard_id": [i.split("-")[0] for _, i, _ in items], "tile_id": [t for t, _, _ in items],
            "row_left_id": [None] * n, "row_right_id": [None] * n, "interrow_cover": [cover] * n,
            "veg_frac": [0.1] * n, "shadow_frac": [0.0] * n, "area_m2": [p.area for _, _, p in items],
            "width_mean_m": [np.nan] * n, "n_notches": [0] * n, **_prov(n, source)}
    return coerce_layer(gpd.GeoDataFrame(data, geometry=[p for _, _, p in items], crs=CRS_EPSG), "interrow_pieces")


def waste(items: Sequence[tuple[str, str, Polygon]]) -> gpd.GeoDataFrame:
    """items: (tile_id, vineyard_id or "", box polygon)."""
    if not items:
        return empty_layer("waste")
    n = len(items)
    nan = [np.nan] * n
    data = {"waste_id": [f"W{k:04d}" for k in range(1, n + 1)], "tile_id": [t for t, _, _ in items],
            "vineyard_id": [v for _, v, _ in items], "dist_block_m": [0.0] * n, "px_xtl": nan, "px_ytl": nan,
            "px_xbr": nan, "px_ybr": nan, "area_m2": [p.area for _, _, p in items], "category": ["debris"] * n,
            "detector": ["manual"] * n, "exported": [True] * n, **_prov(n, Source.MODEL)}
    return coerce_layer(gpd.GeoDataFrame(data, geometry=[p for _, _, p in items], crs=CRS_EPSG), "waste")


def annset(tile_ids: Sequence[str], *, can=None, rows=None, irs=None, wst=None,
           source: Source = Source.MODEL) -> AnnSet:
    meta = make_meta(source, "t", "mv", tile_ids, created_at="2026-09-26T00:00:00+03:00")
    return AnnSet(meta=meta, canopies=can if can is not None else empty_layer("canopies"),
                  row_pieces=rows if rows is not None else empty_layer("row_pieces"),
                  interrow_pieces=irs if irs is not None else empty_layer("interrow_pieces"),
                  waste=wst if wst is not None else empty_layer("waste"))


# ------------------------------------------------------------------ reference (examples) -> AnnSet layers


def reference_layers(xml_path: Path) -> dict[str, gpd.GeoDataFrame]:
    """AnnSet(reference)-shaped layers built with the test oracle reader (until import_reference exists)."""
    images = load_examples(Path(xml_path).read_bytes())
    can, rows, irs = [], [], []
    for tile_id, img in images.items():
        for shape in img.by_label("vineyard"):
            can.append((tile_id, shape.attributes["vineyard_id"], Polygon(to_utm(tile_id, shape.points))))
        for shape in img.by_label("row"):
            rows.append((tile_id, shape.attributes, LineString(to_utm(tile_id, shape.points))))
        for k, shape in enumerate(img.by_label("interrow_area"), start=1):
            irs.append((tile_id, k, shape.attributes, Polygon(to_utm(tile_id, shape.points))))
    return {"canopies": canopies(can, Source.REFERENCE), "row_pieces": _reference_rows(rows),
            "interrow_pieces": _reference_interrows(irs), "waste": empty_layer("waste")}


def _reference_rows(rows: list) -> gpd.GeoDataFrame:
    frame = row_pieces([(t, a["row_id"], g) for t, a, g in rows], Source.REFERENCE)
    return frame.assign(row_structure=[a["row_structure"] for _, a, _ in rows])


def _reference_interrows(irs: list) -> gpd.GeoDataFrame:
    frame = interrow_pieces([(t, f"{a['vineyard_id']}-I{k:03d}", g) for t, k, a, g in irs], Source.REFERENCE)
    return frame.assign(piece_id=[format_marcaj_interrow_piece_id(t, k) for t, k, _, _ in irs],
                        interrow_id=[None] * len(irs), interrow_cover=[a["interrow_cover"] for _, _, a, _ in irs])
