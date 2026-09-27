"""Waste inference on the Sireț3 tiles: sliding windows (1024 / 128 overlap), hflip TTA, nodata zeroed.

Per tile a uint8 heatmap PNG (``<out>/heat/<tile_id>.png``, p * 255); all boxes with score >= 0.1 go to
``<out>/boxes.parquet`` (tile_id, xtl, ytl, xbr, ybr, score; tile px 0..2048). Tiles whose heatmap
already exists are not re-predicted (resume), their boxes are re-extracted from the PNG.

    python -m fte.waste.infer --weights models/fte-waste/v1/best.pt --tiles all --out work/preds/waste/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pandas as pd

from fte.paths import AI_CACHE, AI_TILES, AI_WORK, EXAMPLE_TILES, FTE_ROOT
from fte.waste.boxes import BoxParams, heatmap_to_boxes

log = logging.getLogger("fte.waste.infer")

BOX_COLUMNS: Final = ["tile_id", "xtl", "ytl", "xbr", "ybr", "score"]
MIN_SCORE: Final = 0.1
HEAT_FLOOR_U8: Final = 13  # p < 0.05 stored as 0: keeps the PNGs small, boxes start at 0.1
ENCODER: Final = "resnet34"


def all_tile_ids() -> list[str]:
    index = AI_WORK / "tile_index.parquet"
    if index.is_file():
        return sorted(pd.read_parquet(index, columns=["tile_id"])["tile_id"].astype(str))
    return sorted(p.stem for p in AI_TILES.glob("siret3_*.tif"))


def resolve_tiles(spec: str) -> list[str]:
    """'all' | 'examples' | comma list of tile ids."""
    if spec == "all":
        return all_tile_ids()
    if spec == "examples":
        return list(EXAMPLE_TILES)
    ids = [t.strip() for t in spec.split(",") if t.strip()]
    known = set(all_tile_ids())
    unknown = [t for t in ids if t not in known]
    if unknown:
        raise ValueError(f"unknown tile ids: {unknown}")
    return ids


def heat_path(out: Path, tile_id: str) -> Path:
    return out / "heat" / f"{tile_id}.png"


def read_heat(path: Path) -> np.ndarray:
    heat = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if heat is None:
        raise FileNotFoundError(f"cannot read heatmap {path}")
    return heat


def boxes_frame(tile_id: str, heat: np.ndarray, threshold: float = MIN_SCORE) -> pd.DataFrame:
    boxes = heatmap_to_boxes(heat, BoxParams(threshold=threshold))
    frame = pd.DataFrame(boxes, columns=BOX_COLUMNS[1:])
    frame.insert(0, "tile_id", tile_id)
    return frame


def predict_tile(model: object, device: object, tile_id: str, tta: bool, window: int, overlap: int) -> np.ndarray:
    from vineyard.geo.raster import read_tile
    from vineyard.pipeline.tile_cache import load_valid_mask

    from fte.waste.evaluate import to_u8
    from fte.waste.predict import PredictSpec, predict_heat

    rgb = read_tile(AI_TILES / f"{tile_id}.tif")
    try:
        valid = load_valid_mask(AI_CACHE, tile_id)
    except Exception as exc:  # noqa: BLE001 - cache miss: derive from pixels
        log.warning("%s: no valid mask (%s)", tile_id, exc)
        valid = rgb.max(axis=2) > 0
    heat = to_u8(np.where(valid, predict_heat(model, rgb, device, PredictSpec(window, overlap, tta)), 0.0))
    return np.where(heat < HEAT_FLOOR_U8, 0, heat).astype(np.uint8)


def run(weights: Path, tiles: Sequence[str], out: Path, device_pref: str = "auto", tta: bool = True,
        window: int = 1024, overlap: int = 128) -> pd.DataFrame:
    from fte.device import pick_device
    from fte.nn.model import load_model

    device = pick_device(device_pref)
    model = load_model(weights, ENCODER, 1, device)
    frames, t0 = [], time.time()
    for i, tile_id in enumerate(tiles):
        path = heat_path(out, tile_id)
        if path.is_file():
            heat = read_heat(path)
        else:
            heat = predict_tile(model, device, tile_id, tta, window, overlap)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), heat):
                raise OSError(f"cannot write {path}")
        frames.append(boxes_frame(tile_id, heat))
        if (i + 1) % 10 == 0 or i + 1 == len(tiles):
            log.info("infer: %d / %d tiles (%.1f s/tile)", i + 1, len(tiles), (time.time() - t0) / (i + 1))
    boxes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=BOX_COLUMNS)
    boxes.to_parquet(out / "boxes.parquet", index=False)
    meta = {"weights": str(weights), "n_tiles": len(tiles), "tta_hflip": tta, "window": window,
            "overlap": overlap, "min_score": MIN_SCORE, "n_boxes": int(len(boxes))}
    (out / "infer.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return boxes


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="fte waste inference on Sireț3 tiles")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--tiles", default="all", help="all | examples | comma-separated tile ids")
    ap.add_argument("--out", type=Path, default=Path("work/preds/waste/v1"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=128)
    a = ap.parse_args(argv)
    out = a.out if a.out.is_absolute() else FTE_ROOT / a.out
    weights = a.weights if a.weights.is_absolute() else FTE_ROOT / a.weights
    boxes = run(weights, resolve_tiles(a.tiles), out, a.device, not a.no_tta, a.window, a.overlap)
    return {"n_boxes": int(len(boxes)), "out": str(out)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(json.dumps(main(), indent=2))
