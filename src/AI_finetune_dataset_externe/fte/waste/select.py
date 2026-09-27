"""Threshold choice for the recall-oriented import policy (humans delete false positives in Marcaj).

Start at the DroneWaste-val F1-optimal threshold t0 (model card), lower it along the grid while the 2
example tiles (no waste) keep 0 boxes after the vetoes, and stop as soon as the synthetic Sireț3 recall
(model card, same grid) reaches ``min_syn_recall``. Then cap: <= 5 boxes per tile and <= 400 in total
(highest scores kept).

    python -m fte.waste.select --pred-dir work/preds/waste/v1 --card models/fte-waste/v1/model_card.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from fte.paths import EXAMPLE_TILES, FTE_ROOT
from fte.waste.evaluate import THRESHOLDS
from fte.waste.filters import apply_vetoes
from fte.waste.infer import boxes_frame, heat_path, read_heat

log = logging.getLogger("fte.waste.select")

SELECTED: Final = "selected.parquet"
REPORT: Final = "select_report.json"


@dataclass(frozen=True)
class SelectParams:
    min_syn_recall: float = 0.8
    max_per_tile: int = 5
    max_total: int = 400
    example_max_boxes: int = 0
    probe_device: str | None = None  # "cpu" / "mps": re-rank with the CLIP probe (0.7 p + 0.3 probe)


def candidate_thresholds(t0: float, grid: Sequence[float] = THRESHOLDS) -> list[float]:
    """Grid values <= t0 in descending order (t0 first even when it is off-grid)."""
    below = sorted({float(t) for t in grid if t < t0 - 1e-9}, reverse=True)
    return [float(t0), *below]


def choose_threshold(t0: float, syn_recall: Mapping[float, float], example_boxes: Mapping[float, int],
                     p: SelectParams) -> tuple[float, str]:
    """Lowest admissible threshold on the way down from t0; returns (t, reason).

    If t0 itself leaves boxes on the example tiles, the threshold is raised instead (first grid value
    above t0 with <= ``example_max_boxes``).
    """
    grid = sorted(set(syn_recall) | set(example_boxes) | {float(t0)})
    if example_boxes.get(t0, 0) > p.example_max_boxes:
        for t in (g for g in grid if g > t0):
            if example_boxes.get(t, 0) <= p.example_max_boxes:
                return t, f"raised: {example_boxes[t0]} example boxes at t0"
        return grid[-1], "raised to the grid top (example boxes remain)"
    chosen = t0
    for t in candidate_thresholds(t0, grid):
        if example_boxes.get(t, 0) > p.example_max_boxes:
            return chosen, f"stopped: {example_boxes[t]} example boxes at {t:.2f}"
        chosen = t
        if syn_recall.get(t, 0.0) >= p.min_syn_recall:
            return chosen, f"synthetic recall {syn_recall.get(t, 0.0):.3f} >= {p.min_syn_recall}"
    return chosen, "grid exhausted"


def apply_caps(frame: pd.DataFrame, max_per_tile: int, max_total: int) -> pd.DataFrame:
    """Highest-score boxes: <= max_per_tile per tile, then <= max_total overall (new frame)."""
    ranked = frame.sort_values(["tile_id", "score"], ascending=[True, False])
    per_tile = ranked.groupby("tile_id", sort=False).head(max_per_tile)
    return per_tile.sort_values("score", ascending=False).head(max_total).reset_index(drop=True)


def rank_and_cap(kept: pd.DataFrame, p: SelectParams) -> pd.DataFrame:
    """Caps on the model score, or (probe) pre-cap at 2x, re-rank by 0.7 p + 0.3 probe, cap."""
    if p.probe_device is None or kept.empty:
        return apply_caps(kept, p.max_per_tile, p.max_total)
    from fte.waste.probe_rescore import rescore

    pre = apply_caps(kept, 2 * p.max_per_tile, 2 * p.max_total)
    return apply_caps(rescore(pre, p.probe_device), p.max_per_tile, p.max_total)


def boxes_at(pred_dir: Path, tiles: Sequence[str], t: float) -> pd.DataFrame:
    frames = [boxes_frame(tile, read_heat(heat_path(pred_dir, tile)), threshold=t) for tile in tiles]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return apply_vetoes(frame) if len(frame) else frame.assign(veto=pd.Series(dtype=str))


def _card_curves(card: Mapping) -> tuple[float, dict[float, float]]:
    best = card.get("best_metrics") or {}
    if "val_threshold" not in best:
        raise ValueError("model card has no best_metrics.val_threshold")
    syn = {float(k): float(v["recall"]) for k, v in best.get("synthetic", {}).items()}
    return float(best["val_threshold"]), syn


def run(pred_dir: Path, card_path: Path, p: SelectParams = SelectParams()) -> dict:
    t0, syn = _card_curves(json.loads(card_path.read_text(encoding="utf-8")))
    tiles = sorted(h.stem for h in (pred_dir / "heat").glob("*.png"))
    examples = [t for t in tiles if t in EXAMPLE_TILES]
    if len(examples) < len(EXAMPLE_TILES):
        log.warning("example heatmaps missing: %s", sorted(set(EXAMPLE_TILES) - set(examples)))
    ex_counts = {}
    for t in sorted(set(syn) | set(THRESHOLDS) | {t0}):
        ex = boxes_at(pred_dir, examples, t)
        ex_counts[t] = int((ex["veto"] == "").sum()) if len(ex) else 0
    t, why = choose_threshold(t0, syn, ex_counts, p)
    all_boxes = boxes_at(pred_dir, tiles, t)
    kept = all_boxes[all_boxes["veto"] == ""]
    selected = rank_and_cap(kept[~kept["tile_id"].isin(EXAMPLE_TILES)], p)
    selected.to_parquet(pred_dir / SELECTED, index=False)
    report = {"t0": t0, "threshold": t, "reason": why, "params": asdict(p),
              "synthetic_recall_at_t": syn.get(t), "example_boxes_by_t": {f"{k:.2f}": v for k, v in ex_counts.items()},
              "n_raw": int(len(all_boxes)), "n_vetoed": all_boxes["veto"].value_counts().to_dict(),
              "n_selected": int(len(selected)), "n_tiles_with_boxes": int(selected["tile_id"].nunique())}
    (pred_dir / REPORT).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="choose the waste threshold and cap the boxes")
    ap.add_argument("--pred-dir", type=Path, default=Path("work/preds/waste/v1"))
    ap.add_argument("--card", type=Path, default=Path("models/fte-waste/v1/model_card.json"))
    ap.add_argument("--min-syn-recall", type=float, default=0.8)
    ap.add_argument("--max-per-tile", type=int, default=5)
    ap.add_argument("--max-total", type=int, default=400)
    ap.add_argument("--probe-device", default=None, help="cpu|mps: re-rank with the src/AI CLIP probe")
    a = ap.parse_args(argv)
    pred = a.pred_dir if a.pred_dir.is_absolute() else FTE_ROOT / a.pred_dir
    card = a.card if a.card.is_absolute() else FTE_ROOT / a.card
    return run(pred, card, SelectParams(a.min_syn_recall, a.max_per_tile, a.max_total,
                                          probe_device=a.probe_device))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(json.dumps(main(), indent=2, default=str))
