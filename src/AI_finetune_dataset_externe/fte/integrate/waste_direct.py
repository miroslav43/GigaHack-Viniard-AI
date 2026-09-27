"""Final waste layer rebuilt from a confirmation CSV, exactly as the `waste` stage finalizes it.

Uses the run's own layers/waste_candidates.parquet and layers/blocks.parquet with
vineyard.perception.waste.confirm.merge_confirmed, so no pipeline re-run (and no shared tile cache) is
involved. With the CSV the run was made with, the result equals the run's waste layer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import geopandas as gpd

CANDIDATES_FILE: Final = Path("layers") / "waste_candidates.parquet"
BLOCKS_FILE: Final = Path("layers") / "blocks.parquet"

_log = logging.getLogger(__name__)


def _layer_or_empty(path: Path, name: str) -> gpd.GeoDataFrame:
    from vineyard.contracts.schemas import empty_layer
    from vineyard.geo.vector_io import read_layer

    if path.is_file():
        return read_layer(path, name)
    _log.warning("%s missing (%s): using an empty layer", name, path)
    return empty_layer(name)


def rebuild_waste(run_dir: Path, csv_path: Path, tile_ids: Sequence[str], cfg: Any, *, run_id: str,
                  model_version: str) -> gpd.GeoDataFrame:
    """AnnSet waste layer = auto candidates (none by config) + rows confirmed in `csv_path`."""
    from vineyard.contracts.enums import Source
    from vineyard.geo.tiling import tile_ref
    from vineyard.perception.waste.confirm import MergeParams, merge_confirmed, read_confirmations

    source = Path(csv_path)
    if not source.is_file():
        raise FileNotFoundError(f"waste confirmation CSV not found: {source}")
    refs = {t: tile_ref(t) for t in sorted(tile_ids)}
    confs = read_confirmations(source, frozenset(refs))
    cands = _layer_or_empty(Path(run_dir) / CANDIDATES_FILE, "waste_candidates")
    blocks = _layer_or_empty(Path(run_dir) / BLOCKS_FILE, "blocks")
    params = MergeParams(match_iou=cfg.waste.review.confirm_match_iou,
                         block_assign_max_m=cfg.waste.block_assign_max_m,
                         edge_tol_px=cfg.waste.review.edge_tol_px, run_id=run_id, model_version=model_version,
                         source=Source.MODEL.value)
    waste = merge_confirmed(cands, list(confs), blocks, refs, params)
    _log.info("waste rebuilt from %s: %d confirmations -> %d boxes", source, len(confs), len(waste))
    return waste
