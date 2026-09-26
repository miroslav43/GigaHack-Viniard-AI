"""derive: AnnSet -> rows, blocks, interrows, interrow_pieces_linked, qa issues (contract §4.5, design 04 §3.3-3.4).

Pure orchestration over merge_rows / row_order / blocks / link_interrows / import_checks. Row gap statistics
(max_gap_m, n_gaps_ge5) come from the single gap engine through an injected `GapFn`; without one they stay
NaN / null. Layers carry the AnnSet's source and model_version and the deriving run's run_id.

Every AnnSet (model or Marcaj) passes here before targets / measure, so the interrow pieces lose their
cross-block overlap first (perception.block_overlap): blocks, links and `interrow_pieces_linked` all use
the overlap-free pieces, and the block interrow areas add up to the survey union. Import checks still see
the AnnSet as given.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import geopandas as gpd
import numpy as np
import shapely
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from vineyard.annset.blocks import BlockParams, block_record, build_blocks
from vineyard.annset.import_checks import ImportCheckParams, check_annset
from vineyard.annset.link_interrows import LinkParams, link_interrows
from vineyard.annset.merge_rows import Coverage, MergeParams, merge_rows
from vineyard.annset.model import AnnSet
from vineyard.annset.row_order import OrderedRow, block_axes, order_rows, row_record
from vineyard.config import AppConfig
from vineyard.contracts.enums import Severity, Source
from vineyard.contracts.qa import QaIssue, issues_to_gdf
from vineyard.contracts.schemas import coerce_layer, empty_layer, validate_layer
from vineyard.errors import SchemaError
from vineyard.geo.tiling import CRS_EPSG, tile_box, tile_ref
from vineyard.perception.block_overlap import BlockOverlapParams, BlockOverlapResult, remove_block_overlap
from vineyard.route.target_gaps import GapFn, row_gaps

# CONFIG-REQUEST: derive.join_support_min_m = 2.0
JOIN_SUPPORT_MIN_M: Final = 2.0  # shorter pieces do not define the junction support line (tile-corner stubs)
# CONFIG-REQUEST: derive.interrow_link_along_tol_m = 1.0
LINK_ALONG_TOL_M: Final = 1.0
INTERIOR: Final = "interior"
DERIVED_LAYERS: Final = ("rows", "blocks", "interrows", "interrow_pieces_linked")


@dataclass(frozen=True)
class DeriveParams:
    merge: MergeParams
    blocks: BlockParams
    link: LinkParams
    checks: ImportCheckParams
    gap_ge_m: float  # n_gaps_ge5 threshold (row_structure.gap_disrupted_m)
    corridor_half_m: float  # canopies within this of a row feed its gap statistics
    overlap: BlockOverlapParams

    @classmethod
    def from_config(cls, cfg: AppConfig) -> DeriveParams:
        d = cfg.derive
        return cls(
            merge=MergeParams(d.join_lateral_max_m, JOIN_SUPPORT_MIN_M, d.row_missing_min_len_m, d.vertex_dedupe_m),
            blocks=BlockParams(cfg.blocks.outline_buffer_m, d.blocks_merge_warn_m, d.garden_forbidden_dist_m,
                               d.garden_max_rows, cfg.blocks.min_rows_per_block, cfg.canopy.corridor_half_m),
            link=LinkParams(d.interrow_link_max_m, LINK_ALONG_TOL_M, cfg.route.domain.seam_close_m),
            checks=ImportCheckParams.from_config(cfg),
            gap_ge_m=cfg.row_structure.gap_disrupted_m,
            corridor_half_m=cfg.canopy.corridor_half_m,
            overlap=BlockOverlapParams.from_config(cfg),
        )


@dataclass(frozen=True)
class DeriveInputs:
    annset: AnnSet
    coverage: Coverage | None = None  # tile_id -> imaged area; None skips row_missing_in_tile / unknown
    passages: BaseGeometry | None = None
    forbidden: BaseGeometry | None = None


@dataclass(frozen=True)
class DeriveResult:
    rows: gpd.GeoDataFrame
    blocks: gpd.GeoDataFrame
    interrows: gpd.GeoDataFrame
    interrow_pieces_linked: gpd.GeoDataFrame
    issues: tuple[QaIssue, ...]
    overlap: BlockOverlapResult  # the cross-block interrow overlap removal (for its metrics)

    def layer(self, name: str) -> gpd.GeoDataFrame:
        if name not in DERIVED_LAYERS:
            raise SchemaError("not a derived layer", layer=name, allowed=DERIVED_LAYERS)
        return getattr(self, name)

    def metrics(self) -> dict[str, float]:
        linked = self.interrow_pieces_linked["interrow_id"].notna()
        severities = [i.severity for i in self.issues]
        return {
            "n_rows": float(len(self.rows)), "n_blocks": float(len(self.blocks)),
            "n_interrow_pieces": float(len(linked)), "n_interrow_pieces_linked": float(linked.sum()),
            "n_interrows": float(len(self.interrows)), "row_length_m": float(self.rows["length_m"].sum()),
            "n_issues_error": float(severities.count(Severity.ERROR)),
            "n_issues_warning": float(severities.count(Severity.WARNING)),
        } | self.overlap.metrics()


@dataclass(frozen=True)
class LayerProv:
    source: Source
    run_id: str
    model_version: str

    @classmethod
    def of(cls, annset: AnnSet, run_id: str | None = None) -> LayerProv:
        return cls(annset.meta.source, run_id or annset.meta.run_id, annset.meta.model_version)

    def columns(self) -> dict[str, Any]:
        return {"source": self.source.value, "run_id": self.run_id, "model_version": self.model_version,
                "confidence": 1.0, "qa_flags": ""}

    def qa_layer(self, issues: Iterable[QaIssue]) -> gpd.GeoDataFrame:
        return issues_to_gdf(issues, source=self.source, run_id=self.run_id, model_version=self.model_version)


def build_coverage(tile_ids: Iterable[str], tile_valid: gpd.GeoDataFrame | None = None) -> Coverage:
    """tile_id -> tile square, intersected with its `tile_valid` geometry when that layer is given."""
    ids = sorted(set(tile_ids))
    if tile_valid is None:
        return MappingProxyType({t: tile_box(tile_ref(t)) for t in ids})
    valid = dict(zip(tile_valid["tile_id"].astype(str), tile_valid.geometry, strict=True))
    missing = [t for t in ids if t not in valid]
    if missing:
        raise SchemaError("tiles missing from tile_valid", n_missing=len(missing), examples=missing[:5])
    return MappingProxyType({t: tile_box(tile_ref(t)).intersection(valid[t]) for t in ids})


def records_layer(records: Sequence[Mapping[str, Any]], name: str, prov: LayerProv) -> gpd.GeoDataFrame:
    """Contract layer from dict records (each with `geometry`), provenance from `prov`, validated."""
    if not records:
        return empty_layer(name)
    rows = [prov.columns() | {k: v for k, v in r.items() if k != "geometry"} for r in records]
    geoms = gpd.GeoSeries([r["geometry"] for r in records], crs=CRS_EPSG)
    out = coerce_layer(gpd.GeoDataFrame(rows, geometry=geoms, crs=CRS_EPSG), name)
    validate_layer(out, name)
    return out


def _unknown(line: BaseGeometry, coverage: Coverage | None, index: tuple[STRtree, list] | None,
             margin_m: float) -> BaseGeometry | None:
    if coverage is None or index is None:
        return None
    window = line.envelope.buffer(margin_m)
    tree, geoms = index
    near = [geoms[int(i)] for i in tree.query(window)]
    return window.difference(shapely.union_all(near)) if near else window


def gap_stats(ordered: Sequence[OrderedRow], canopies: gpd.GeoDataFrame, coverage: Coverage | None,
              gap_fn: GapFn, params: DeriveParams) -> dict[str, tuple[float, int]]:
    """row_id -> (max interior gap m, number of interior gaps >= gap_ge_m) on the merged row."""
    geoms = canopies.geometry.to_numpy()
    tree = STRtree(geoms)
    cov_geoms = [coverage[t] for t in sorted(coverage)] if coverage else []
    cov_index = (STRtree(cov_geoms), cov_geoms) if cov_geoms else None
    out: dict[str, tuple[float, int]] = {}
    for o in ordered:
        near = [geoms[int(i)] for i in sorted(tree.query(o.row.line, predicate="dwithin",
                                                           distance=params.corridor_half_m))]
        unknown = _unknown(o.row.line, coverage, cov_index, params.corridor_half_m)
        gaps = [g for g in row_gaps(gap_fn, o.row.line, near, unknown, row_id=o.row.row_id) if g.kind == INTERIOR]
        lengths = [g.length_m for g in gaps]
        out[o.row.row_id] = (max(lengths) if lengths else 0.0, sum(1 for x in lengths if x >= params.gap_ge_m))
    return out


def _plant_counts(canopies: gpd.GeoDataFrame) -> Mapping[str, int]:
    ids = canopies["row_id"].dropna().astype(str)
    return ids.value_counts().to_dict()


def _row_records(ordered: Sequence[OrderedRow], canopies: gpd.GeoDataFrame,
                 gaps: Mapping[str, tuple[float, int]] | None) -> list[dict[str, Any]]:
    counts = _plant_counts(canopies)
    records = []
    for o in ordered:
        rec = row_record(o, plant_count=counts.get(o.row.row_id, 0))
        if gaps is not None:
            rec = rec | {"max_gap_m": gaps[o.row.row_id][0], "n_gaps_ge5": gaps[o.row.row_id][1]}
        records.append(rec)
    return sorted(records, key=lambda r: r["row_id"])


def _linked_layer(linked: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """The AnnSet pieces keep their own provenance; only the link columns are new."""
    if not len(linked):
        return empty_layer("interrow_pieces_linked")
    out = coerce_layer(linked.sort_values("piece_id", kind="stable"), "interrow_pieces_linked")
    validate_layer(out, "interrow_pieces_linked")
    return out


def derive(inputs: DeriveInputs, params: DeriveParams, *, run_id: str | None = None,
           gap_fn: GapFn | None = None) -> DeriveResult:
    """All derived layers of one AnnSet plus import, merge, block and link issues."""
    annset = inputs.annset
    prov = LayerProv.of(annset, run_id)
    rows, merge_issues = merge_rows(annset.row_pieces, params.merge, inputs.coverage)
    ordered = order_rows(rows)
    axes = block_axes(rows)
    overlap = remove_block_overlap(annset.interrow_pieces, annset.canopies, params.overlap)
    resolved = annset.with_layer("interrow_pieces", overlap.pieces)
    blocks, block_issues = build_blocks(resolved, ordered, axes, params.blocks, passages=inputs.passages,
                                        forbidden=inputs.forbidden)
    link = link_interrows(resolved.interrow_pieces, annset.row_pieces, ordered, axes, params.link)
    gaps = None if gap_fn is None else gap_stats(ordered, annset.canopies, inputs.coverage, gap_fn, params)
    issues = check_annset(annset, params.checks) + merge_issues + block_issues + link.issues + overlap.issues
    return DeriveResult(
        rows=records_layer(_row_records(ordered, annset.canopies, gaps), "rows", prov),
        blocks=records_layer([block_record(b) for b in blocks], "blocks", prov),
        interrows=records_layer(link.interrows, "interrows", prov),
        interrow_pieces_linked=_linked_layer(link.pieces),
        issues=tuple(sorted(issues, key=lambda i: i.sort_key)),
        overlap=overlap,
    )


def total_length_by_block(rows: gpd.GeoDataFrame) -> dict[str, float]:
    """vineyard_id -> Σ row length_m (the acceptance numbers of design 04 §7)."""
    return {str(v): float(np.sum(g["length_m"])) for v, g in rows.groupby("vineyard_id", sort=True)}
