"""AnnSet edits on a copied run: canopy partition, interrow cover override, waste from a confirmation CSV.

read_annset(<run>/annset) -> edits -> write_annset (validated by vineyard) + tile_status counts refresh.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import pandas as pd

from fte.canopy.merge_fix import MergeParams, merge_fragments
from fte.canopy.partition import PartitionParams, partition_canopies, png_contact_loader

ANNSET_DIR: Final = "annset"
TILE_STATUS_FILE: Final = Path("layers") / "tile_status.parquet"
WASTE_LAYER_FILE: Final = Path("layers") / "waste.parquet"
COVER_COLUMNS: Final = ("piece_id", "interrow_cover")
STATUS_COUNTS: Final = {"n_canopies": "canopies", "n_row_pieces": "row_pieces",
                        "n_interrow_pieces": "interrow_pieces", "n_waste": "waste"}

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EditSpec:
    partition: PartitionParams | None = None
    merge: MergeParams | None = None
    contact_dir: Path | None = None
    cover_csv: Path | None = None
    waste_csv: Path | None = None


@dataclass(frozen=True)
class EditSummary:
    n_canopies_before: int
    n_canopies_after: int
    n_cover_changed: int
    n_waste_before: int
    n_waste_after: int

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def apply_cover_csv(pieces: gpd.GeoDataFrame, csv_path: Path) -> tuple[gpd.GeoDataFrame, int]:
    """New interrow_pieces with interrow_cover taken from the CSV (joined on piece_id)."""
    from vineyard.contracts.enums import InterrowCover

    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in COVER_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"cover CSV {csv_path} lacks columns {missing}")
    allowed = {c.value for c in InterrowCover}
    bad = sorted(set(table["interrow_cover"]) - allowed)
    if bad:
        raise ValueError(f"cover CSV {csv_path}: values {bad} not in {sorted(allowed)}")
    if table["piece_id"].duplicated().any():
        raise ValueError(f"cover CSV {csv_path}: duplicate piece_id")
    unknown = sorted(set(table["piece_id"]) - set(pieces["piece_id"]))
    if unknown:
        raise ValueError(f"cover CSV {csv_path}: unknown piece ids {unknown[:5]}")
    mapping = dict(zip(table["piece_id"], table["interrow_cover"], strict=True))
    new_cover = pieces["piece_id"].map(mapping).fillna(pieces["interrow_cover"])
    changed = int((new_cover != pieces["interrow_cover"]).sum())
    out = pieces.copy()
    out["interrow_cover"] = new_cover.astype(pieces["interrow_cover"].dtype)
    return out, changed


def _with_run_id(annset: Any, run_id: str) -> Any:
    """AnnSet whose meta and provenance columns carry the new run id."""
    out = annset
    for name in ("canopies", "row_pieces", "interrow_pieces", "waste"):
        layer = out.layer(name).copy()
        if "run_id" in layer.columns:
            layer["run_id"] = pd.Series([run_id] * len(layer), dtype=layer["run_id"].dtype, index=layer.index)
        out = out.with_layer(name, layer)
    return replace(out, meta=replace(out.meta, run_id=run_id))


def _waste_model_version(waste: gpd.GeoDataFrame, fallback: str) -> str:
    return str(waste["model_version"].iloc[0]) if len(waste) else fallback


def _edit_waste(annset: Any, run_dir: Path, csv_path: Path, run_id: str) -> Any:
    from vineyard.config.loader import load_config

    from fte.integrate.waste_direct import rebuild_waste

    mv = _waste_model_version(annset.waste, annset.meta.model_version)
    waste = rebuild_waste(run_dir, csv_path, annset.meta.tile_ids, load_config(), run_id=run_id, model_version=mv)
    return annset.with_layer("waste", waste)


def _write_waste_layer(run_dir: Path, waste: gpd.GeoDataFrame) -> None:
    from vineyard.geo.vector_io import write_layer

    write_layer(waste, "waste", Path(run_dir) / WASTE_LAYER_FILE)


def refresh_tile_status(run_dir: Path, annset: Any) -> None:
    """Rewrite layers/tile_status.parquet with per-tile object counts of the edited AnnSet."""
    from vineyard.geo.vector_io import read_layer, write_layer

    path = Path(run_dir) / TILE_STATUS_FILE
    if not path.is_file():
        _log.warning("no %s: tile_status not refreshed", path)
        return
    status = read_layer(path, "tile_status")
    out = status.copy()
    for column, layer in STATUS_COUNTS.items():
        counts = annset.layer(layer)["tile_id"].value_counts()
        out[column] = status["tile_id"].map(counts).fillna(0).astype(status[column].dtype)
    write_layer(out, "tile_status", path)


def apply_edits(run_dir: Path, spec: EditSpec) -> EditSummary:
    """Edit <run_dir>/annset in place (the run is a copy made by fte.integrate.run_copy)."""
    from vineyard.annset.io import read_annset, write_annset

    run_dir = Path(run_dir)
    annset = read_annset(run_dir / ANNSET_DIR)
    n_can, n_waste = len(annset.canopies), len(annset.waste)
    annset = _with_run_id(annset, run_dir.name)
    contact = png_contact_loader(spec.contact_dir) if spec.contact_dir is not None else None
    if spec.partition is not None:
        canopies = partition_canopies(annset.canopies, annset.row_pieces, spec.partition, contact)
        annset = annset.with_layer("canopies", canopies)
    if spec.merge is not None:
        canopies = merge_fragments(annset.canopies, annset.row_pieces, spec.merge, contact)
        annset = annset.with_layer("canopies", canopies)
    n_changed = 0
    if spec.cover_csv is not None:
        pieces, n_changed = apply_cover_csv(annset.interrow_pieces, spec.cover_csv)
        annset = annset.with_layer("interrow_pieces", pieces)
    if spec.waste_csv is not None:
        annset = _edit_waste(annset, run_dir, spec.waste_csv, run_dir.name)
        _write_waste_layer(run_dir, annset.waste)
    write_annset(annset, run_dir / ANNSET_DIR)
    refresh_tile_status(run_dir, annset)
    summary = EditSummary(n_can, len(annset.canopies), n_changed, n_waste, len(annset.waste))
    _log.info("annset edited: %s", summary)
    return summary
