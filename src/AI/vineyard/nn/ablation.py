"""Canopy ablation A-F on the reference tiles (A§4.6, design 03 N7), torch-free.

A = a* ∩ corridor (baseline), B = NN ∩ corridor, C = a* ∧ NN, D = a* ∨ NN, E = NN without corridor,
F = variant B with weights trained without ignore band and label smoothing. Every variant is scored with
the official vector metric per tile, for two corridor sources: the model rows (primary: what is
exported) and the reference axes (mask quality alone). Promotion: B, C or D only when it beats A on
EVERY tile by more than nn.ablation.min_gain; several winners -> highest mean. Nothing is applied
automatically: the report prints the `--set nn.fusion=X` to commit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from vineyard.contracts.enums import FusionVariant
from vineyard.errors import StageError
from vineyard.nn.validate import ValTile, score_tile
from vineyard.pipeline.atomic import atomic_write_json, atomic_write_text

if TYPE_CHECKING:
    from vineyard.config import AppConfig

AXES_MODEL: Final = "model"
AXES_REFERENCE: Final = "reference"
BASELINE: Final = FusionVariant.A.value
PROMOTION_ALLOWED: Final = frozenset({"B", "C", "D"})  # E and F are controls, never exported
MAIN_WEIGHTS: Final = "main"
F_WEIGHTS: Final = "F"
F_FUSION: Final = FusionVariant.B  # F = the B fusion with the F weights
SCORE_DECIMALS: Final = 4

__all__ = ["AXES_MODEL", "AXES_REFERENCE", "AblationReport", "VariantScore", "decide_promotion",
           "render_markdown", "report_to_dict", "run_ablation", "write_report"]


@dataclass(frozen=True)
class VariantScore:
    variant: str
    tile_id: str
    iou: float
    f1: float
    score: float
    axes: str
    n_pred: int
    n_ref: int


@dataclass(frozen=True)
class AblationReport:
    scores: tuple[VariantScore, ...]
    means: Mapping[str, Mapping[str, float]]  # axes -> variant -> mean score over tiles
    promoted: FusionVariant
    reason: str
    primary_axes: str
    tiles: tuple[str, ...]
    skipped: tuple[str, ...] = ()
    infer_s_per_tile: Mapping[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------ promotion rule


def _table(scores: Sequence[VariantScore]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for s in scores:
        table.setdefault(s.variant, {})[s.tile_id] = s.score
    return table


def _judge(cand: str, table: Mapping[str, Mapping[str, float]], baseline: str, tiles: Sequence[str],
           min_gain: float) -> str | None:
    """None when `cand` beats the baseline on every tile, else why not."""
    mine = table.get(cand, {})
    missing = [t for t in tiles if t not in mine]
    if missing:
        return f"{cand}: scores missing for {', '.join(missing)}"
    for t in tiles:
        if not mine[t] > table[baseline][t] + min_gain:
            return f"{cand} loses on {t} ({mine[t]:.4f} <= {table[baseline][t]:.4f} + {min_gain})"
    return None


def decide_promotion(scores: Sequence[VariantScore], baseline: str, candidates: Sequence[str],
                     tiles: Sequence[str], min_gain: float) -> tuple[FusionVariant, str]:
    """(variant to export, reason). Only B/C/D can win; they must beat the baseline on every tile."""
    table = _table(scores)
    if any(t not in table.get(baseline, {}) for t in tiles):
        raise StageError("ablation baseline scores missing", baseline=baseline, tiles=", ".join(tiles))
    allowed = [c for c in candidates if c in PROMOTION_ALLOWED]
    verdicts = {c: _judge(c, table, baseline, tiles, min_gain) for c in allowed}
    winners = [c for c in allowed if verdicts[c] is None]
    if not winners:
        why = "; ".join(v for v in verdicts.values() if v) or "no promotable candidate"
        return FusionVariant(baseline), f"keep {baseline}: no NN variant beats it on every tile ({why})"
    best = max(winners, key=lambda c: (float(np.mean([table[c][t] for t in tiles])), -allowed.index(c)))
    mean_best = float(np.mean([table[best][t] for t in tiles]))
    mean_base = float(np.mean([table[baseline][t] for t in tiles]))
    return FusionVariant(best), (f"{best} beats {baseline} on every tile (mean {mean_best:.4f} vs {mean_base:.4f};"
                                 f" winners: {', '.join(winners)})")


# ------------------------------------------------------------------ scoring


def _fusion_and_weights(variant: str) -> tuple[FusionVariant, str | None]:
    if variant == F_WEIGHTS:
        return F_FUSION, F_WEIGHTS
    fusion = FusionVariant(variant)
    return fusion, (None if fusion is FusionVariant.A else MAIN_WEIGHTS)


def _score_variant(tiles: Sequence[ValTile], variant: str, probs: Mapping[str, Mapping[str, np.ndarray]],
                   axes: str, cfg: AppConfig) -> list[VariantScore]:
    fusion, weights = _fusion_and_weights(variant)
    out = []
    for tile in tiles:
        prob = None if weights is None else probs[weights][tile.tile_id]
        s = score_tile(tile, prob, fusion, cfg)
        out.append(VariantScore(variant, tile.tile_id, s.iou, s.f1, s.score, axes, s.n_pred, s.n_ref))
    return out


def _means(scores: Sequence[VariantScore]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, list[float]]] = {}
    for s in scores:
        out.setdefault(s.axes, {}).setdefault(s.variant, []).append(s.score)
    return {a: {v: float(np.mean(vals)) for v, vals in per.items()} for a, per in out.items()}


def _check_probs(tiles_by_axes: Mapping[str, Sequence[ValTile]], probs: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    if MAIN_WEIGHTS not in probs:
        raise StageError("ablation needs the main NN probabilities", have=", ".join(sorted(probs)) or "none")
    needed = {t.tile_id for tiles in tiles_by_axes.values() for t in tiles}
    for key, per_tile in probs.items():
        missing = sorted(needed - set(per_tile))
        if missing:
            raise StageError("ablation probabilities missing for tiles", weights=key, tiles=", ".join(missing))


def run_ablation(tiles_by_axes: Mapping[str, Sequence[ValTile]], probs: Mapping[str, Mapping[str, np.ndarray]],
                 cfg: AppConfig, *, infer_s_per_tile: Mapping[str, float] | None = None) -> AblationReport:
    """Score cfg.nn.ablation.variants for each axes source; promotion is decided on the first axes source
    of cfg.nn.ablation.axes_sources. F is skipped (and listed) when its probabilities are absent."""
    _check_probs(tiles_by_axes, probs)
    ab = cfg.nn.ablation
    axes_order = [a for a in ab.axes_sources if a in tiles_by_axes]
    if not axes_order:
        raise StageError("no tiles for the configured axes sources", configured=", ".join(ab.axes_sources))
    variants = [v for v in ab.variants if v != F_WEIGHTS or F_WEIGHTS in probs]
    skipped = tuple(v for v in ab.variants if v not in variants)
    scores = tuple(s for axes in axes_order for v in variants
                   for s in _score_variant(tiles_by_axes[axes], v, probs, axes, cfg))
    primary = axes_order[0]
    tiles = tuple(t.tile_id for t in tiles_by_axes[primary])
    promoted, reason = decide_promotion([s for s in scores if s.axes == primary], BASELINE, ab.promotable, tiles,
                                        ab.min_gain)
    return AblationReport(scores, MappingProxyType(_means(scores)), promoted, reason, primary, tiles, skipped,
                          MappingProxyType(dict(infer_s_per_tile or {})))


# ------------------------------------------------------------------ report


def _r(x: float) -> float:
    return round(float(x), SCORE_DECIMALS)


def report_to_dict(report: AblationReport) -> dict[str, Any]:
    tiles: dict[str, dict[str, dict[str, Any]]] = {}
    for s in report.scores:
        tiles.setdefault(s.axes, {}).setdefault(s.variant, {})[s.tile_id] = {
            "iou": _r(s.iou), "f1": _r(s.f1), "score": _r(s.score), "n_pred": s.n_pred, "n_ref": s.n_ref}
    return {"promoted": report.promoted.value, "reason": report.reason, "primary_axes": report.primary_axes,
            "tile_ids": list(report.tiles), "skipped_variants": list(report.skipped),
            "means": {a: {v: _r(m) for v, m in per.items()} for a, per in report.means.items()},
            "tiles": tiles, "infer_s_per_tile": {k: _r(v) for k, v in report.infer_s_per_tile.items()},
            "set_option": f"--set nn.fusion={report.promoted.value}"}


def _axes_table(report: AblationReport, axes: str) -> list[str]:
    head = "| variant | " + " | ".join(f"{t} (IoU / F1 / score)" for t in report.tiles) + " | mean |"
    lines = [f"### Axes: {axes}", "", head, "|---" * (len(report.tiles) + 2) + "|"]
    by = {(s.variant, s.tile_id): s for s in report.scores if s.axes == axes}
    for v in report.means.get(axes, {}):
        cells = [f"{by[(v, t)].iou:.4f} / {by[(v, t)].f1:.4f} / **{by[(v, t)].score:.4f}**" for t in report.tiles]
        lines.append(f"| {v} | " + " | ".join(cells) + f" | {report.means[axes][v]:.4f} |")
    return [*lines, ""]


def render_markdown(report: AblationReport) -> str:
    lines = ["# NN canopy ablation (A-F)", "",
             f"Promoted: **{report.promoted.value}** ({report.reason}).", "",
             f"Apply with `--set nn.fusion={report.promoted.value}` (promotion is decided on axes "
             f"`{report.primary_axes}`; B/C/D must beat A on every tile).", ""]
    for axes in report.means:
        lines.extend(_axes_table(report, axes))
    if report.skipped:
        lines.extend([f"Skipped variants (no weights): {', '.join(report.skipped)}", ""])
    if report.infer_s_per_tile:
        timing = ", ".join(f"{k}: {v:.3f} s/tile" for k, v in report.infer_s_per_tile.items())
        lines.extend([f"Inference time: {timing}", ""])
    return "\n".join(lines)


def write_report(report: AblationReport, json_path: Path, md_path: Path) -> tuple[Path, Path]:
    return atomic_write_json(json_path, report_to_dict(report)), atomic_write_text(md_path, render_markdown(report))
