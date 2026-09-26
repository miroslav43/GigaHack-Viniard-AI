"""tiles.geojson of the web bundle (web contract §6.3): one footprint Polygon per tile of the grid.

Counts and status come from the AnnSet (so a Marcaj-corrected set is described by its own objects); veg_frac,
nodata_frac and review_priority come from the model run's tile_status layer and/or tile_prep's stats cache and
stay null when neither exists; review_status / review_note come from the human tile review list.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from vineyard.annset.model import TILE_COLUMN, AnnSet
from vineyard.geo.ops import orient_ccw
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.pipeline.tile_cache import load_tile_stats, stats_path
from vineyard.web.rows_export import features_frame
from vineyard.web.tile_review import TileReview

STATUS_VINEYARD: Final = "vineyard"
STATUS_EMPTY: Final = "no_vineyard"
# tiles.geojson count property -> AnnSet layer
COUNT_LAYERS: Final[Mapping[str, str]] = MappingProxyType({
    "n_rows": "row_pieces", "n_canopies": "canopies", "n_interrows": "interrow_pieces", "n_waste": "waste",
})
VINEYARD_COUNTS: Final = ("n_rows", "n_canopies")
TILE_COLUMNS: Final = ("tile", "status", *COUNT_LAYERS, "veg_frac", "nodata_frac", "review_priority",
                       "review_note", "review_status", "has_mask")


@dataclass(frozen=True)
class TileFacts:
    """Per-tile image facts the AnnSet does not hold; None = unknown."""

    veg_frac: float | None = None
    nodata_frac: float | None = None
    review_priority: int | None = None


def tile_footprint(tile_id: str) -> Polygon:
    """The tile's 51.2 m grid square in EPSG:32635, CCW exterior."""
    return orient_ccw(tile_box(tile_ref(tile_id)))


def tile_counts(annset: AnnSet) -> dict[str, dict[str, int]]:
    """tile -> {n_rows, n_canopies, n_interrows, n_waste} for every tile holding an AnnSet object."""
    per_layer = {key: annset.layer(layer)[TILE_COLUMN].astype(str).value_counts().to_dict()
                 for key, layer in COUNT_LAYERS.items()}
    tiles = sorted({t for counts in per_layer.values() for t in counts})
    return {t: {key: int(per_layer[key].get(t, 0)) for key in COUNT_LAYERS} for t in tiles}


def _finite(value: Any) -> float | None:
    if value is None or value is pd.NA:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    number = _finite(value)
    return None if number is None else int(number)


def facts_from_status(status: pd.DataFrame | None) -> dict[str, TileFacts]:
    """veg_frac / review_priority from the model run's tile_status layer ({} without one)."""
    if status is None:
        return {}
    veg = status["veg_frac"] if "veg_frac" in status else [None] * len(status)
    prio = status["review_priority"] if "review_priority" in status else [None] * len(status)
    return {str(t): TileFacts(veg_frac=_finite(v), review_priority=_int_or_none(p))
            for t, v, p in zip(status["tile_id"], veg, prio, strict=True)}


def facts_from_stats(cache_dir: Path | None, tile_ids: Iterable[str]) -> dict[str, TileFacts]:
    """veg_frac / nodata_frac from tile_prep's cache/stats/<tile>.json where it exists."""
    if cache_dir is None:
        return {}
    found = (load_tile_stats(cache_dir, t) for t in tile_ids if stats_path(cache_dir, t).is_file())
    return {s.tile_id: TileFacts(veg_frac=float(s.veg_frac), nodata_frac=float(s.nodata_frac)) for s in found}


def merge_facts(primary: Mapping[str, TileFacts], fallback: Mapping[str, TileFacts]) -> dict[str, TileFacts]:
    """Field by field: `primary` where known, else `fallback`."""
    def pick(a: TileFacts, b: TileFacts, name: str) -> Any:
        value = getattr(a, name)
        return getattr(b, name) if value is None else value

    empty = TileFacts()
    return {t: TileFacts(**{name: pick(primary.get(t, empty), fallback.get(t, empty), name)
                            for name in ("veg_frac", "nodata_frac", "review_priority")})
            for t in sorted({*primary, *fallback})}


def _record(tile_id: str, counts: Mapping[str, int], facts: TileFacts, review: TileReview | None,
            has_mask: bool) -> dict[str, Any]:
    status = STATUS_VINEYARD if any(counts[k] for k in VINEYARD_COUNTS) else STATUS_EMPTY
    return {"tile": tile_id, "status": status, **counts, "veg_frac": facts.veg_frac,
            "nodata_frac": facts.nodata_frac, "review_priority": facts.review_priority,
            "review_note": None if review is None else review.note,
            "review_status": None if review is None else review.status, "has_mask": has_mask}


def tile_features(tile_ids: Iterable[str], counts: Mapping[str, Mapping[str, int]],
                  facts: Mapping[str, TileFacts], review: Mapping[str, TileReview],
                  with_mask: frozenset[str]) -> gpd.GeoDataFrame:
    """One feature per tile id (in the given order) with the TILE_COLUMNS properties."""
    ids = list(tile_ids)
    zero = dict.fromkeys(COUNT_LAYERS, 0)
    records = [_record(t, counts.get(t, zero), facts.get(t, TileFacts()), review.get(t), t in with_mask)
               for t in ids]
    return features_frame(records, [tile_footprint(t) for t in ids], TILE_COLUMNS)
