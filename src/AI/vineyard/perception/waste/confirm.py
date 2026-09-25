"""Human waste confirmations (A§4.9 step 6, plan S8): configs/waste_confirmed.csv -> AnnSet `waste` layer.

CSV header: tile_id,xtl,ytl,xbr,ybr,decision,category,reviewer,note  (boxes in CVAT tile px).
decision: accept (a candidate, matched by IoU, or the own box when unmatched -> qa flag confirm_unmatched),
reject (drops a matched auto candidate), add (a manual box). Only these rows (plus auto candidates, which
are disabled before Publish) are exported; every error names the file line and field.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import pandas as pd

from vineyard.contracts.enums import Source
from vineyard.contracts.ids import IdKind, is_valid_id
from vineyard.contracts.schemas import LAYER_SCHEMAS, coerce_layer, empty_layer
from vineyard.errors import ConfigError
from vineyard.geo.tiling import CRS_EPSG, GSD_M, TileRef
from vineyard.perception.waste.assign import assign_vineyard_id, assign_waste_ids, utm_box
from vineyard.perception.waste.types import DECISIONS, BoxPx, Category, Confirmation, Detector

CONFIRM_HEADER: Final = ("tile_id", "xtl", "ytl", "xbr", "ybr", "decision", "category", "reviewer", "note")
COORD_FIELDS: Final = ("xtl", "ytl", "xbr", "ybr")
COMMENT_PREFIX: Final = "#"
QA_CONFIRM_UNMATCHED: Final = "confirm_unmatched"
HUMAN_CONFIDENCE: Final = 1.0
LAYER: Final = "waste"


class ConfirmationFileError(ConfigError):
    """A line of the confirmations CSV is invalid (context: path, line_no, field, value)."""

    @property
    def line_no(self) -> int:
        return int(self.context["line_no"])

    @property
    def field(self) -> str:
        return str(self.context["field"])


def _fail(path: Path, line_no: int, field: str, value: object, why: str) -> ConfirmationFileError:
    return ConfirmationFileError(
        f"{path.name} line {line_no}: {why}", path=str(path), line_no=line_no, field=field, value=value
    )


def _coord(path: Path, line_no: int, field: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise _fail(path, line_no, field, text, f"{field} is not a number") from None
    if not math.isfinite(value):
        raise _fail(path, line_no, field, text, f"{field} is not finite")
    return value


def _parse_row(path: Path, line_no: int, row: Mapping[str, str], known: frozenset[str]) -> Confirmation:
    tile_id = row["tile_id"].strip()
    if not is_valid_id(IdKind.TILE, tile_id) or (known and tile_id not in known):
        raise _fail(path, line_no, "tile_id", tile_id, "unknown tile_id")
    xyxy = [_coord(path, line_no, f, row[f].strip()) for f in COORD_FIELDS]
    try:
        box = BoxPx(*xyxy)
    except ValueError as exc:
        raise _fail(path, line_no, "box", xyxy, str(exc)) from None
    decision = row["decision"].strip().lower()
    if decision not in DECISIONS:
        raise _fail(path, line_no, "decision", row["decision"], f"decision must be one of {DECISIONS}")
    category = row["category"].strip().lower() or Category.UNKNOWN.value
    if category not in {c.value for c in Category}:
        raise _fail(path, line_no, "category", row["category"], "unknown category")
    return Confirmation(
        line_no=line_no,
        tile_id=tile_id,
        box=box,
        decision=decision,
        category=Category(category),
        reviewer=row["reviewer"].strip(),
        note=row["note"].strip(),
    )


def _rows(path: Path) -> list[tuple[int, list[str]]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        return [
            (reader.line_num, r)
            for r in reader
            if r and not r[0].strip().startswith(COMMENT_PREFIX) and any(c.strip() for c in r)
        ]


def read_confirmations(path: Path, known_tiles: frozenset[str]) -> tuple[Confirmation, ...]:
    """Parse the CSV; a missing file means no confirmations. known_tiles empty = any valid tile id."""
    source = Path(path)
    if not source.is_file():
        return ()
    rows = _rows(source)
    if not rows or tuple(c.strip() for c in rows[0][1]) != CONFIRM_HEADER:
        got = rows[0][1] if rows else []
        raise _fail(
            source, rows[0][0] if rows else 1, "header", got, f"header must be {','.join(CONFIRM_HEADER)}"
        )
    out = []
    for line_no, cells in rows[1:]:
        if len(cells) != len(CONFIRM_HEADER):
            raise _fail(source, line_no, "columns", len(cells), f"expected {len(CONFIRM_HEADER)} columns")
        out.append(_parse_row(source, line_no, dict(zip(CONFIRM_HEADER, cells, strict=True)), known_tiles))
    return tuple(out)


def confirm_line(
    tile_id: str,
    box: BoxPx,
    *,
    decision: str = "accept",
    category: str = Category.UNKNOWN.value,
    reviewer: str = "",
    note: str = "",
) -> str:
    """The exact CSV line (no newline) that confirms `box`; quoting follows the csv module."""
    fields = [tile_id, *(f"{v:.1f}" for v in box.as_tuple()), decision, category, reviewer, note]
    return ",".join(_csv_cell(f) for f in fields)


def _csv_cell(text: str) -> str:
    needs = any(ch in text for ch in ',"\n\r')
    return '"' + text.replace('"', '""') + '"' if needs else text


# ------------------------------------------------------------------ merge


@dataclass(frozen=True)
class MergeParams:
    match_iou: float
    block_assign_max_m: float
    edge_tol_px: float
    run_id: str
    model_version: str
    source: str = Source.MODEL.value


@dataclass(frozen=True)
class _Obj:
    tile_id: str
    box: BoxPx
    category: str
    detector: str
    confidence: float
    qa_flags: str


def _cand_box(row: Mapping[str, Any]) -> BoxPx:
    return BoxPx(*(float(row[c]) for c in ("px_xtl", "px_ytl", "px_xbr", "px_ybr")))


def _best_match(cands: Sequence[Mapping[str, Any]], conf: Confirmation, min_iou: float) -> int | None:
    scored = [(conf.box.iou(_cand_box(c)), i) for i, c in enumerate(cands) if c["tile_id"] == conf.tile_id]
    good = [(iou, i) for iou, i in scored if iou >= min_iou]
    return max(good, key=lambda t: (t[0], -t[1]))[1] if good else None


def _from_candidate(c: Mapping[str, Any], conf: Confirmation | None) -> _Obj:
    human = conf is not None
    category = conf.category.value if human and conf.category != Category.UNKNOWN else str(c["category"])
    return _Obj(
        str(c["tile_id"]),
        _cand_box(c),
        category,
        str(c["detector"]),
        HUMAN_CONFIDENCE if human else float(c["confidence"]),
        "",
    )


def _manual(conf: Confirmation, flag: str) -> _Obj:
    return _Obj(conf.tile_id, conf.box, conf.category.value, Detector.MANUAL.value, HUMAN_CONFIDENCE, flag)


def _is_auto(c: Mapping[str, Any]) -> bool:
    auto = c.get("auto", False)
    return bool(auto) and not pd.isna(auto) and pd.isna(c.get("reject_reason"))


def _objects(
    cands: list[dict[str, Any]], confirmations: Sequence[Confirmation], p: MergeParams
) -> list[_Obj]:
    chosen: dict[int, Confirmation | None] = {i: None for i, c in enumerate(cands) if _is_auto(c)}
    manual: list[_Obj] = []
    for conf in sorted(confirmations, key=lambda c: c.line_no):
        if conf.decision == "add":
            manual.append(_manual(conf, ""))
            continue
        match = _best_match(cands, conf, p.match_iou)
        if conf.decision == "reject":
            if match is not None:
                chosen.pop(match, None)
        elif match is not None:
            chosen[match] = conf
        else:
            manual.append(_manual(conf, QA_CONFIRM_UNMATCHED))
    return [_from_candidate(cands[i], chosen[i]) for i in sorted(chosen)] + manual


def _tile(tiles: Mapping[str, TileRef], tile_id: str) -> TileRef:
    if tile_id not in tiles:
        raise ConfigError("confirmed waste on a tile missing from the tile index", tile_id=tile_id)
    return tiles[tile_id]


def merge_confirmed(
    candidates: gpd.GeoDataFrame,
    confirmations: Sequence[Confirmation],
    blocks: gpd.GeoDataFrame,
    tiles: Mapping[str, TileRef],
    p: MergeParams,
) -> gpd.GeoDataFrame:
    """AnnSet `waste` layer (contract §2.5.10): auto candidates minus rejects, plus accepted/added boxes."""
    objs = _objects(candidates.drop(columns=candidates.geometry.name).to_dict("records"), confirmations, p)
    if not objs:
        return empty_layer(LAYER)
    geoms = [utm_box(_tile(tiles, o.tile_id), o.box) for o in objs]
    assigned = [assign_vineyard_id(g, blocks, p.block_assign_max_m) for g in geoms]
    frame = gpd.GeoDataFrame(
        {
            "tile_id": [o.tile_id for o in objs],
            "vineyard_id": [a.vineyard_id for a in assigned],
            "dist_block_m": [a.dist_block_m for a in assigned],
            **{f"px_{k}": [getattr(o.box, k) for o in objs] for k in ("xtl", "ytl", "xbr", "ybr")},
            "area_m2": [o.box.area * GSD_M * GSD_M for o in objs],
            "category": [o.category for o in objs],
            "detector": [o.detector for o in objs],
            "exported": [True] * len(objs),
            "source": [p.source] * len(objs),
            "run_id": [p.run_id] * len(objs),
            "model_version": [p.model_version] * len(objs),
            "confidence": [o.confidence for o in objs],
            "qa_flags": [o.qa_flags for o in objs],
        },
        geometry=geoms,
        crs=f"EPSG:{CRS_EPSG}",
    )
    with_ids = assign_waste_ids(frame, p.edge_tol_px)
    ordered = [c for c in LAYER_SCHEMAS[LAYER].column_names if c in with_ids.columns]
    return coerce_layer(with_ids[[*ordered, "geometry"]], LAYER)
