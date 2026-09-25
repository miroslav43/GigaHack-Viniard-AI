"""Fixtures for eval tests: AnnSet factories (synthetic + organizers' examples) and the eval config."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from tests.helpers.examples import ExampleImage, load_examples, to_utm
from vineyard.annset.model import ANNSET_LAYERS, AnnSet, make_meta
from vineyard.config import EvalConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.contracts.schemas import coerce_layer, empty_layer, get_schema
from vineyard.geo.tiling import CRS_EPSG, tile_ref

EXAMPLE_TILES = ("siret3_r006_c004", "siret3_r021_c012")
Record = Mapping[str, Any]


def _fill_value(name: str, kind: str, blank_ok: bool, source: Source, run_id: str) -> Any:
    if name == "source":
        return source.value
    if name == "run_id":
        return run_id
    if kind == "str":
        return "" if blank_ok else "x"
    if kind.startswith("float"):
        return math.nan
    return False if kind == "bool" else 0


def layer_frame(name: str, records: Sequence[Record], *, source: Source = Source.REFERENCE,
                run_id: str = "ref-test") -> gpd.GeoDataFrame:
    """Schema-complete layer from partial records (missing required columns get neutral values)."""
    if not records:
        return empty_layer(name)
    gdf = gpd.GeoDataFrame([dict(r) for r in records], geometry="geometry", crs=CRS_EPSG)
    for col in get_schema(name).all_columns:
        if col.name not in gdf.columns and not col.nullable:
            gdf[col.name] = _fill_value(col.name, col.kind, col.blank_ok, source, run_id)
    return coerce_layer(gdf, name)


def make_annset(tile_ids: Sequence[str], *, source: Source = Source.REFERENCE, run_id: str = "ref-test",
                **layers: Sequence[Record]) -> AnnSet:
    meta = make_meta(source, run_id, "test", tile_ids, created_at="2026-09-26T00:00:00+03:00")
    frames = {name: layer_frame(name, layers.get(name, ()), source=source, run_id=run_id)
              for name in ANNSET_LAYERS}
    return AnnSet(meta=meta, **frames).with_layer("waste", frames["waste"])


def _example_records(img: ExampleImage) -> dict[str, list[dict[str, Any]]]:
    t = img.tile_id
    canopies = [{"canopy_id": f"{t}:C{k:04d}", "tile_id": t, "vineyard_id": s.attributes["vineyard_id"],
                 "geometry": Polygon(to_utm(t, s.points))} for k, s in enumerate(img.by_label("vineyard"), 1)]
    rows = [{"piece_id": f"{s.attributes['row_id']}@{t}", "row_id": s.attributes["row_id"], "tile_id": t,
             "vineyard_id": s.attributes["vineyard_id"], "row_structure": s.attributes["row_structure"],
             "geometry": LineString(to_utm(t, s.points))} for s in img.by_label("row")]
    interrows = [{"piece_id": f"{t}:I{k:03d}", "tile_id": t, "vineyard_id": s.attributes["vineyard_id"],
                  "interrow_cover": s.attributes["interrow_cover"], "geometry": Polygon(to_utm(t, s.points))}
                 for k, s in enumerate(img.by_label("interrow_area"), 1)]
    return {"canopies": canopies, "row_pieces": rows, "interrow_pieces": interrows}


def examples_annset(xml: bytes, *, source: Source = Source.REFERENCE, run_id: str = "ref-test") -> AnnSet:
    images = load_examples(xml)
    merged: dict[str, list[dict[str, Any]]] = {"canopies": [], "row_pieces": [], "interrow_pieces": []}
    for tile_id in sorted(images):
        for name, recs in _example_records(images[tile_id]).items():
            merged[name].extend(recs)
    return make_annset(sorted(images), source=source, run_id=run_id, **merged)


@pytest.fixture(scope="session")
def eval_cfg() -> EvalConfig:
    return load_config(environ={}).eval


@pytest.fixture(scope="session")
def reference_annset(examples_xml: bytes) -> AnnSet:
    return examples_annset(examples_xml)


@pytest.fixture(scope="session")
def example_images(examples_xml: bytes) -> dict[str, ExampleImage]:
    return load_examples(examples_xml)


def tile_origin(tile_id: str) -> tuple[float, float]:
    """(x0, y_bottom) of a tile, for building synthetic geometry inside it."""
    t = tile_ref(tile_id)
    return t.bounds[0], t.bounds[1]
