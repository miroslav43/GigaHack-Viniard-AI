"""Router-style AI + CV mix per tile: keep the classical canopy, add model-found vines where CV has none.

hybrid = ExG (blur 2 px, > 25) AND dilate(model c0 >= t, d px)   -- the model says where, ExG draws edges
added  = hybrid AND NOT dilate(CV canopy, keep-out)               -- only vines CV did not map
mix    = CV OR added

Panels: CV (green) | model (magenta) | mix (CV green + added amber). Also scores the mix against the
reference on the 2 example tiles (pixel IoU / precision / recall) to show it does not hurt where CV is good.

CLI: ``python -m fte.viz.mix_review --pred-dir work/preds/canopy/v1b_review --out work/reports/mix_review``
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from fte.paths import AI_ROOT, AI_RUNS, AI_TILES, BASE_RUN, EXAMPLE_TILES
from fte.viz.nn_review import GREEN, MAGENTA, PX_AREA_M2, caption, cv_mask, overlay, read_review_list

log = logging.getLogger(__name__)

AMBER = (255, 176, 0)
PANEL = 640


@dataclass(frozen=True)
class MixParams:
    threshold: float = 0.5
    dilate_px: int = 3
    keep_out_px: int = 20  # 0.5 m around existing CV canopy: CV already decided there
    exg_threshold: float = 25.0
    exg_sigma_px: float = 2.0


@dataclass(frozen=True)
class MixStats:
    tile_id: str
    status: str
    note: str
    cv_m2: float
    nn_m2: float
    mix_m2: float
    added_m2: float


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def exg_mask(rgb: np.ndarray, p: MixParams) -> np.ndarray:
    x = rgb.astype(np.float32)
    e = cv2.GaussianBlur(2 * x[..., 1] - x[..., 0] - x[..., 2], (0, 0), p.exg_sigma_px)
    return (e > p.exg_threshold) & (rgb.max(axis=2) > 0)


def mix_masks(rgb: np.ndarray, prob: np.ndarray, cv: np.ndarray, p: MixParams) -> tuple[np.ndarray, np.ndarray]:
    """(added, mix) boolean masks; inputs are left untouched."""
    model = (prob >= int(round(p.threshold * 255))).astype(np.uint8)
    near_model = cv2.dilate(model, _disk(p.dilate_px)).astype(bool) if p.dilate_px else model.astype(bool)
    hybrid = exg_mask(rgb, p) & near_model
    keep_out = cv2.dilate(cv.astype(np.uint8), _disk(p.keep_out_px)).astype(bool)
    added = hybrid & ~keep_out
    return added, cv | added


def pixel_scores(mask: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    tp = float((mask & ref).sum())
    return {"iou": tp / max(float((mask | ref).sum()), 1.0), "precision": tp / max(float(mask.sum()), 1.0),
            "recall": tp / max(float(ref.sum()), 1.0)}


def three_panel(rgb: np.ndarray, cv: np.ndarray, model: np.ndarray, added: np.ndarray, s: MixStats) -> np.ndarray:
    small = lambda a: cv2.resize(a, (PANEL, PANEL), interpolation=cv2.INTER_AREA)  # noqa: E731
    left = caption(small(overlay(rgb, cv, GREEN)), f"CV actual  {s.cv_m2:.0f} m2")
    mid = caption(small(overlay(rgb, model, MAGENTA)), f"model  {s.nn_m2:.0f} m2")
    right = caption(small(overlay(overlay(rgb, cv, GREEN), added, AMBER, 0.75)),
                    f"mix  {s.mix_m2:.0f} m2  (+{s.added_m2:.0f} adaugat)")
    gap = np.full((PANEL, 6, 3), 255, np.uint8)
    return np.hstack([left, gap, mid, gap, right])


def review_tile(tile_id: str, status: str, note: str, canopies, pred_dir: Path, p: MixParams,
                out_dir: Path) -> tuple[MixStats, dict[str, np.ndarray]]:
    from vineyard.geo.raster import read_tile

    rgb = read_tile(AI_TILES / f"{tile_id}.tif")
    prob = cv2.imread(str(pred_dir / "c0" / f"{tile_id}.png"), cv2.IMREAD_GRAYSCALE)
    if prob is None:
        raise FileNotFoundError(f"no c0 prediction for {tile_id} in {pred_dir}")
    cv = cv_mask(canopies, tile_id)
    model = prob >= int(round(p.threshold * 255))
    added, mix = mix_masks(rgb, prob, cv, p)
    s = MixStats(tile_id, status, note, float(cv.sum() * PX_AREA_M2), float(model.sum() * PX_AREA_M2),
                 float(mix.sum() * PX_AREA_M2), float(added.sum() * PX_AREA_M2))
    panel = three_panel(rgb, cv, model, added, s)
    cv2.imwrite(str(out_dir / f"{tile_id}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return s, {"cv": cv, "model": model, "mix": mix}


def example_scores(masks: dict[str, dict[str, np.ndarray]], ref_canopies) -> dict[str, dict[str, dict[str, float]]]:
    out = {}
    for tile_id, m in masks.items():
        ref = cv_mask(ref_canopies, tile_id)
        out[tile_id] = {k: pixel_scores(v, ref) for k, v in m.items()}
    return out


def main(argv: Sequence[str] | None = None) -> None:
    import geopandas as gpd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-run", default=BASE_RUN)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    p = MixParams()
    review = read_review_list(AI_ROOT / "configs" / "tile_review.csv")
    wanted = {t: v for t, v in review.items() if v[0] in ("missed", "partial")}
    wanted.update({t: ("example", "reference tile (CV is good here)") for t in EXAMPLE_TILES})
    canopies = gpd.read_parquet(AI_RUNS / a.base_run / "annset" / "canopies.parquet")
    stats, ex_masks = [], {}
    for tile_id, (status, note) in wanted.items():
        s, masks = review_tile(tile_id, status, note, canopies, a.pred_dir, p, a.out)
        stats.append(s)
        if status == "example":
            ex_masks[tile_id] = masks
    ref = gpd.read_parquet(AI_RUNS / "LATEST_REFERENCE" / "annset" / "canopies.parquet")
    summary = {"params": asdict(p), "examples": example_scores(ex_masks, ref), "tiles": [asdict(s) for s in stats]}
    (a.out / "mix_summary.json").write_text(json.dumps(summary, indent=1))
    with (a.out / "mix_areas.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(stats[0])))
        w.writeheader()
        w.writerows(asdict(s) for s in sorted(stats, key=lambda s: s.added_m2, reverse=True))
    log.info("%d mix panels -> %s", len(stats), a.out)


if __name__ == "__main__":
    main()
