"""AnnSet: the 4-layer annotation set exchanged between perception, Marcaj and post (contract §2.6).

Everything is immutable: `with_layer` and `for_tiles` return new objects, frames are never modified.
`AnnSetMeta.tile_ids` lists every image of the set, including tiles without objects.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final

import geopandas as gpd

from vineyard.contracts import CONTRACT_VERSION
from vineyard.contracts.enums import Source
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.contracts.schemas import empty_layer
from vineyard.errors import SchemaError

ANNSET_LAYERS: Final = ("canopies", "row_pieces", "interrow_pieces", "waste")
TILE_COLUMN: Final = "tile_id"


def _check_tile_ids(tile_ids: tuple[str, ...], run_id: str) -> None:
    bad = [t for t in tile_ids if not is_valid_id(IdKind.TILE, t)]
    if bad:
        raise SchemaError("invalid tile ids in AnnSet meta", run_id=run_id, examples=bad[:5])
    if len(set(tile_ids)) != len(tile_ids):
        raise SchemaError("duplicate tile ids in AnnSet meta", run_id=run_id)


@dataclass(frozen=True)
class AnnSetMeta:
    """Content of annset.json."""

    contract_version: str
    source: Source
    run_id: str
    model_version: str
    created_at: str
    n_tiles: int
    counts: Mapping[str, int]
    inputs: tuple[str, ...]
    tile_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            source = Source(self.source)
        except ValueError:
            raise SchemaError("invalid AnnSet source", source=self.source, run_id=self.run_id) from None
        tile_ids = tuple(self.tile_ids)
        _check_tile_ids(tile_ids, self.run_id)
        if self.n_tiles != len(tile_ids):
            raise SchemaError("n_tiles != len(tile_ids)", run_id=self.run_id, n_tiles=self.n_tiles,
                              n_tile_ids=len(tile_ids))
        counts = {str(k): int(v) for k, v in dict(self.counts).items()}
        if any(v < 0 for v in counts.values()):
            raise SchemaError("negative AnnSet count", run_id=self.run_id, counts=counts)
        # Frozen dataclass: canonicalise containers once, at construction.
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "tile_ids", tuple(sorted(tile_ids)))
        object.__setattr__(self, "inputs", tuple(str(i) for i in self.inputs))
        object.__setattr__(self, "counts", MappingProxyType(counts))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "source": self.source.value,
            "run_id": self.run_id,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "n_tiles": self.n_tiles,
            "counts": dict(self.counts),
            "inputs": list(self.inputs),
            "tile_ids": list(self.tile_ids),
        }

    @classmethod
    def from_json_dict(cls, doc: Mapping[str, Any]) -> "AnnSetMeta":
        try:
            return cls(**{name: doc[name] for name in cls.__dataclass_fields__})
        except KeyError as exc:
            raise SchemaError("annset.json is missing a field", field=str(exc)) from None


def make_meta(
    source: Source,
    run_id: str,
    model_version: str,
    tile_ids: Iterable[str],
    *,
    inputs: Iterable[str] = (),
    created_at: str | None = None,
) -> AnnSetMeta:
    """Meta with zero counts (AnnSet methods fill them) and an ISO timestamp with UTC offset."""
    ids = tuple(tile_ids)
    stamp = created_at or datetime.now().astimezone().isoformat(timespec="seconds")
    return AnnSetMeta(CONTRACT_VERSION, source, run_id, model_version, stamp, len(ids), {}, tuple(inputs), ids)


def _require_layer_name(name: str) -> None:
    if name not in ANNSET_LAYERS:
        raise SchemaError("not an AnnSet layer", layer=name, allowed=ANNSET_LAYERS)


def _require_frame(name: str, gdf: object) -> None:
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise SchemaError("AnnSet layer must be a GeoDataFrame", layer=name, got=type(gdf).__name__)
    if TILE_COLUMN not in gdf.columns:
        raise SchemaError("AnnSet layer has no tile_id column", layer=name)


@dataclass(frozen=True)
class AnnSet:
    meta: AnnSetMeta
    canopies: gpd.GeoDataFrame
    row_pieces: gpd.GeoDataFrame
    interrow_pieces: gpd.GeoDataFrame
    waste: gpd.GeoDataFrame

    def __post_init__(self) -> None:
        for name in ANNSET_LAYERS:
            _require_frame(name, getattr(self, name))

    def layer(self, name: str) -> gpd.GeoDataFrame:
        _require_layer_name(name)
        return getattr(self, name)

    def counts(self) -> dict[str, int]:
        return {name: len(self.layer(name)) for name in ANNSET_LAYERS}

    def with_layer(self, name: str, gdf: gpd.GeoDataFrame) -> "AnnSet":
        _require_layer_name(name)
        _require_frame(name, gdf)
        updated = replace(self, **{name: gdf})
        return replace(updated, meta=replace(self.meta, counts=updated.counts()))

    def for_tiles(self, tile_ids: Iterable[str]) -> "AnnSet":
        keep = frozenset(tile_ids)
        layers = {
            name: self.layer(name)[self.layer(name)[TILE_COLUMN].isin(keep)].reset_index(drop=True)
            for name in ANNSET_LAYERS
        }
        ids = tuple(sorted(self.tile_ids() & keep))
        filtered = replace(self, **layers)
        return replace(filtered, meta=replace(self.meta, tile_ids=ids, n_tiles=len(ids), counts=filtered.counts()))

    def tile_ids(self) -> frozenset[str]:
        """Tiles covered by the set: meta.tile_ids plus any tile that carries an object."""
        present = {t for name in ANNSET_LAYERS for t in self.layer(name)[TILE_COLUMN].dropna().unique()}
        return frozenset(self.meta.tile_ids) | frozenset(present)


def empty_annset(meta: AnnSetMeta) -> AnnSet:
    layers = {name: empty_layer(name) for name in ANNSET_LAYERS}
    return AnnSet(meta=replace(meta, counts=dict.fromkeys(ANNSET_LAYERS, 0)), **layers)
