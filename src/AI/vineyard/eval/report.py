"""evaluate_annsets -> EvalReport (per tile, mean of tiles, pooled), gates, baseline regression, JSON/Markdown.

Metric names are dotted ("canopy.score", "rows.f1", ...); None marks a metric that does not apply.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import pandas as pd
from shapely.geometry import LineString, MultiLineString
from shapely.geometry.base import BaseGeometry

from vineyard.annset.model import TILE_COLUMN, AnnSet
from vineyard.config import EvalConfig, EvalGatesConfig
from vineyard.errors import StageError
from vineyard.eval.matching import clean_polygonal
from vineyard.eval.metrics import (
    INTERROW_COVER_LABELS,
    ROW_STRUCTURE_LABELS,
    Counts,
    EvalParams,
    LayerObjects,
    TileObjects,
    TileStats,
    attribute_scores,
    evaluate_tile,
    grouping_consistency,
    merge_stats,
    relative_error,
    value_score,
    weighted_canopy_score,
)
from vineyard.geo.tiling import TILE_M, tile_box, tile_ref
from vineyard.pipeline.atomic import atomic_write_json, atomic_write_text

__all__ = ["EvalParams", "EvalReport", "GateResult", "compare_to_baseline", "evaluate_annsets", "load_baseline",
           "render_markdown", "report_to_dict", "tile_objects", "write_report_json", "write_report_markdown"]

Flat = dict[str, float | int | None]
GateScope = Literal["mean", "min_tile"]

REPORT_SCHEMA: Final = "vineyard.eval/1"
REPORT_DECIMALS: Final = 6
MD_DECIMALS: Final = 4
TILE_AREA_M2: Final = TILE_M * TILE_M
VINEYARD_COLUMN: Final = "vineyard_id"
# layer -> (polygonal, object-id column, attribute column)
LAYER_SPEC: Final[Mapping[str, tuple[bool, str | None, str | None]]] = {
    "canopies": (True, None, None),
    "row_pieces": (False, "row_id", "row_structure"),
    "interrow_pieces": (True, None, "interrow_cover"),
    "waste": (True, None, None),
}
# Counts and raw values are summed in `pooled`, never averaged across tiles.
MEAN_EXCLUDED_SUFFIXES: Final = (".tp", ".n_pred", ".n_ref", ".n_objects", ".pred", ".ref")
HEADLINE_KEYS: Final = ("canopy.score", "canopy.iou", "canopy.f1", "canopy.fp_penalty", "rows.f1", "interrow.iou",
                        "interrow.f1", "attributes.score", "grouping.f1", "counts.score", "waste.f1")
BASELINE_KEYS: Final = ("canopy.score", "canopy.iou", "canopy.f1", "rows.f1", "interrow.iou", "attributes.score",
                        "grouping.f1", "counts.score", "waste.f1")
_DROP_EPS: Final = 1e-9  # absorbs float noise from the rounding of stored reports


@dataclass(frozen=True)
class GateSpec:
    name: str  # field of EvalGatesConfig
    key: str
    scope: GateScope


GATE_SPECS: Final = (
    GateSpec("canopy_score", "canopy.score", "mean"),
    GateSpec("row_f1", "rows.f1", "min_tile"),
    GateSpec("interrow_iou", "interrow.iou", "mean"),
    GateSpec("attributes", "attributes.score", "mean"),
)


@dataclass(frozen=True)
class GateResult:
    name: str
    key: str
    scope: GateScope
    value: float | None
    threshold: float
    passed: bool


@dataclass(frozen=True)
class EvalReport:
    tiles: tuple[str, ...]
    per_tile: Mapping[str, Flat]
    mean: Flat
    pooled: Flat
    gates: tuple[GateResult, ...]
    params: Mapping[str, float]
    pred_run: str = ""
    ref_run: str = ""
    regressions: tuple[str, ...] = ()

    @property
    def gates_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def with_regressions(self, regressions: Sequence[str]) -> EvalReport:
        return replace(self, regressions=tuple(regressions))


# ------------------------------------------------------------------ AnnSet -> per-tile objects


def _text(value: Any) -> str | None:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _column(frame: pd.DataFrame, column: str | None) -> tuple[str | None, ...]:
    if column is None or column not in frame.columns:
        return (None,) * len(frame)
    return tuple(_text(v) for v in frame[column].tolist())


def _clean_line(geom: BaseGeometry | None) -> BaseGeometry:
    if geom is None or geom.is_empty:
        return LineString()
    parts = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, LineString) and g.length > 0]
    if not parts:
        return LineString()
    return parts[0] if len(parts) == 1 else MultiLineString(parts)


def _prepare(geom: BaseGeometry | None, clip: BaseGeometry | None, polygonal: bool) -> BaseGeometry:
    clean = clean_polygonal if polygonal else _clean_line
    g = clean(geom)
    if clip is not None and not g.is_empty and not clip.covers(g):
        g = clean(g.intersection(clip))
    return g


def _layer_objects(annset: AnnSet, name: str, tile_id: str, clip: BaseGeometry | None) -> LayerObjects:
    polygonal, id_col, attr_col = LAYER_SPEC[name]
    layer = annset.layer(name)
    frame = layer[layer[TILE_COLUMN] == tile_id]
    geoms = [_prepare(g, clip, polygonal) for g in frame.geometry]
    keep = [k for k, g in enumerate(geoms) if not g.is_empty]
    vids, ids, attrs = (_column(frame, c) for c in (VINEYARD_COLUMN, id_col, attr_col))
    return LayerObjects(tuple(geoms[k] for k in keep), tuple(vids[k] for k in keep),
                        tuple(ids[k] for k in keep), tuple(attrs[k] for k in keep))


def tile_objects(annset: AnnSet, tile_id: str, *, clip: bool = True) -> TileObjects:
    """Objects of one tile, cleaned (valid polygons, linear rows) and, by default, clipped to the tile."""
    box = tile_box(tile_ref(tile_id)) if clip else None
    layers = {name: _layer_objects(annset, name, tile_id, box) for name in LAYER_SPEC}
    return TileObjects(tile_id, layers["canopies"], layers["row_pieces"], layers["interrow_pieces"],
                       layers["waste"])


# ------------------------------------------------------------------ statistics -> flat metrics


def _match_flat(prefix: str, c: Counts) -> Flat:
    return {f"{prefix}.f1": c.f1, f"{prefix}.tp": c.tp, f"{prefix}.n_pred": c.n_pred, f"{prefix}.n_ref": c.n_ref}


def _canopy_flat(s: TileStats, p: EvalParams) -> Flat:
    scored = s.canopy_tiles > 0
    iou = s.canopy_overlap.iou if scored else None
    f1 = s.canopy.f1 if scored else None
    score = weighted_canopy_score(iou, f1, w_iou=p.canopy_w_iou, w_f1=p.canopy_w_f1) if scored else None
    return {"canopy.iou": iou, "canopy.f1": f1, "canopy.score": score, "canopy.tp": s.canopy.tp,
            "canopy.n_pred": s.canopy.n_pred, "canopy.n_ref": s.canopy.n_ref,
            "canopy.fp_penalty": s.fp_penalty if s.fp_tiles else None}


def _attr_flat(s: TileStats) -> Flat:
    out: Flat = {}
    scores = []
    for name, pairs, labels in (("row_structure", s.row_structure, ROW_STRUCTURE_LABELS),
                                ("interrow_cover", s.interrow_cover, INTERROW_COVER_LABELS)):
        a = attribute_scores([r for r, _ in pairs], [q for _, q in pairs], labels)
        out |= {f"attributes.{name}.accuracy": a.accuracy, f"attributes.{name}.macro_f1": a.macro_f1,
                f"attributes.{name}.score": a.score, f"attributes.{name}.n_ref": a.n_ref}
        if a.n_ref:
            scores.append(a.score)
    out["attributes.score"] = sum(scores) / len(scores) if scores else None
    return out


def _counts_flat(s: TileStats, p: EvalParams) -> Flat:
    values = (("blocks", len(s.blocks.pred), len(s.blocks.ref), p.count_tol),
              ("rows", len(s.row_ids.pred), len(s.row_ids.ref), p.count_tol),
              ("canopy_area_m2", s.canopy_area.pred, s.canopy_area.ref, p.count_tol),
              ("interrow_area_m2", s.interrow_area.pred, s.interrow_area.ref, p.count_tol),
              ("row_length_m", s.row_length.pred, s.row_length.ref, p.length_tol))
    out: Flat = {}
    for name, pred, ref, tol in values:
        out |= {f"counts.{name}.pred": pred, f"counts.{name}.ref": ref,
                f"counts.{name}.rel_err": relative_error(pred, ref), f"counts.{name}.score": value_score(pred, ref, tol)}
    out["counts.score"] = sum(value_score(pr, rf, tol) for _, pr, rf, tol in values) / len(values)
    return out


def stats_to_flat(s: TileStats, p: EvalParams) -> Flat:
    """Dotted metric name -> value (None = not applicable on these tiles)."""
    return {
        **_canopy_flat(s, p),
        **_match_flat("rows", s.rows),
        "interrow.iou": s.interrow_overlap.iou,
        **_match_flat("interrow", s.interrows),
        **_attr_flat(s),
        "grouping.f1": grouping_consistency([r for r, _ in s.groups], [q for _, q in s.groups]),
        "grouping.n_objects": len(s.groups),
        **_counts_flat(s, p),
        **_match_flat("waste", s.waste),
    }


def mean_of_tiles(per_tile: Mapping[str, Flat]) -> Flat:
    """Mean over tiles of every rate metric (None values skipped; None when no tile has one)."""
    keys = sorted({k for flat in per_tile.values() for k in flat if not k.endswith(MEAN_EXCLUDED_SUFFIXES)})
    out: Flat = {}
    for key in keys:
        vals = [v for flat in per_tile.values() if (v := flat.get(key)) is not None and math.isfinite(v)]
        out[key] = sum(vals) / len(vals) if vals else None
    return out


# ------------------------------------------------------------------ evaluation and gates


def _check_tiles(tiles: Sequence[str], ref: AnnSet) -> tuple[str, ...]:
    tile_ids = tuple(sorted(set(tiles)))
    if not tile_ids:
        raise StageError("no tiles to evaluate", ref_run=ref.meta.run_id)
    missing = [t for t in tile_ids if t not in ref.tile_ids()]
    if missing:
        raise StageError("tiles not covered by the reference AnnSet", ref_run=ref.meta.run_id, tiles=missing)
    return tile_ids


def _gate_value(spec: GateSpec, per_tile: Mapping[str, Flat], mean: Flat) -> float | None:
    if spec.scope == "mean":
        return mean.get(spec.key)
    vals = [flat.get(spec.key) for flat in per_tile.values()]
    return None if not vals or any(v is None for v in vals) else min(vals)


def evaluate_gates(per_tile: Mapping[str, Flat], mean: Flat, gates: EvalGatesConfig) -> tuple[GateResult, ...]:
    out = []
    for spec in GATE_SPECS:
        value = _gate_value(spec, per_tile, mean)
        threshold = float(getattr(gates, spec.name))
        out.append(GateResult(spec.name, spec.key, spec.scope, value, threshold,
                              value is not None and value >= threshold))
    return tuple(out)


def evaluate_annsets(pred: AnnSet, ref: AnnSet, tiles: Sequence[str], cfg: EvalConfig, *,
                     clip: bool = True) -> EvalReport:
    """Official metrics of `pred` against `ref` on `tiles` (a tile missing from `pred` is unannotated)."""
    tile_ids = _check_tiles(tiles, ref)
    params = EvalParams.from_config(cfg)
    stats = {t: evaluate_tile(tile_objects(pred, t, clip=clip), tile_objects(ref, t, clip=clip), params,
                              tile_area_m2=TILE_AREA_M2) for t in tile_ids}
    per_tile = {t: stats_to_flat(s, params) for t, s in stats.items()}
    mean = mean_of_tiles(per_tile)
    pooled = stats_to_flat(merge_stats(list(stats.values())), params)
    return EvalReport(tile_ids, per_tile, mean, pooled, evaluate_gates(per_tile, mean, cfg.gates), asdict(params),
                      pred.meta.run_id, ref.meta.run_id)


def _drop_message(scope: str, key: str, base: Any, value: float | None, max_drop: float) -> str | None:
    if not isinstance(base, int | float) or not math.isfinite(base):
        return None
    if value is None or base - value > max_drop + _DROP_EPS:
        shown = "n/a" if value is None else f"{value:.4f}"
        return f"{scope} {key}: {base:.4f} -> {shown} (max drop {max_drop})"
    return None


def compare_to_baseline(report: EvalReport, baseline: Mapping[str, Any], *, max_drop: float) -> tuple[str, ...]:
    """Messages for every headline metric that dropped by more than max_drop (mean and per tile)."""
    scopes: list[tuple[str, Mapping[str, Any], Flat]] = [("mean", baseline.get("mean", {}), report.mean)]
    base_tiles = baseline.get("per_tile", {})
    scopes += [(t, base_tiles.get(t, {}), report.per_tile[t]) for t in report.tiles]
    msgs = (_drop_message(scope, key, base.get(key), current.get(key), max_drop)
            for scope, base, current in scopes for key in BASELINE_KEYS)
    return tuple(m for m in msgs if m is not None)


# ------------------------------------------------------------------ JSON / Markdown


def _json_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    number = float(value)
    return round(number, REPORT_DECIMALS) if math.isfinite(number) else None


def _json_flat(flat: Flat) -> dict[str, Any]:
    return {k: _json_value(flat[k]) for k in sorted(flat)}


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA, "pred_run": report.pred_run, "ref_run": report.ref_run,
        "tiles": list(report.tiles), "params": dict(report.params),
        "per_tile": {t: _json_flat(report.per_tile[t]) for t in report.tiles},
        "mean": _json_flat(report.mean), "pooled": _json_flat(report.pooled),
        "gates": [asdict(g) | {"value": _json_value(g.value)} for g in report.gates],
        "gates_passed": report.gates_passed, "regressions": list(report.regressions),
    }


def write_report_json(report: EvalReport, path: Path) -> Path:
    return atomic_write_json(Path(path), report_to_dict(report))


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return str(value) if isinstance(value, int) else f"{value:.{MD_DECIMALS}f}"


def _metric_rows(report: EvalReport) -> list[str]:
    keys = sorted({k for flat in (*report.per_tile.values(), report.pooled) for k in flat})
    ordered = [k for k in HEADLINE_KEYS if k in keys] + [k for k in keys if k not in HEADLINE_KEYS]
    cells = lambda k: [report.per_tile[t].get(k) for t in report.tiles] + [report.mean.get(k), report.pooled.get(k)]  # noqa: E731
    return [f"| {k} | " + " | ".join(_fmt(v) for v in cells(k)) + " |" for k in ordered]


def render_markdown(report: EvalReport) -> str:
    head = ["metric", *report.tiles, "mean", "pooled"]
    lines = [f"# Eval: {report.pred_run} vs {report.ref_run}", "",
             "| " + " | ".join(head) + " |", "|---" + "|---:" * (len(head) - 1) + "|", *_metric_rows(report),
             "", "## Gates", "", "| gate | metric | scope | value | threshold | status |", "|---|---|---|---:|---:|---|"]
    lines += [f"| {g.name} | {g.key} | {g.scope} | {_fmt(g.value)} | {g.threshold} | {'PASS' if g.passed else 'FAIL'} |"
              for g in report.gates]
    lines += ["", "## Regressions", ""] + ([f"- {m}" for m in report.regressions] or ["- none"])
    return "\n".join(lines) + "\n"


def write_report_markdown(report: EvalReport, path: Path) -> Path:
    return atomic_write_text(Path(path), render_markdown(report))


def load_baseline(path: Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise StageError("eval baseline not found", path=str(source))
    try:
        doc = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError("eval baseline is not valid JSON", path=str(source), error=str(exc)) from exc
    if not isinstance(doc, dict):
        raise StageError("eval baseline must be a JSON object", path=str(source))
    return doc
