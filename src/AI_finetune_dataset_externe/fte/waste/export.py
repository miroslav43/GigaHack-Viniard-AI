"""Export selected waste boxes as a Marcaj-import CSV + a review list / contact sheets of the misses.

CSV = the format of ``src/AI/configs/waste_confirmed.csv``
(``tile_id,xtl,ytl,xbr,ybr,decision,category,reviewer,note``, tile px 0..2048): the existing confirmed rows
first, unchanged, then one ``add`` row per selected box (category ``unknown``, reviewer
``fte-waste@v1 (auto)``, note ``score=0.83``); a box duplicating a confirmed row (IoU >= 0.3) is skipped.
Review: below-threshold boxes (score >= ``--review-min``, not vetoed, sorted desc) as CSV and JPEG contact
sheets (30 crops each, box drawn, tile id + score) in ``<pred-dir>/review/``.

    python -m fte.waste.export --pred-dir work/preds/waste/v1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pandas as pd

from fte.paths import AI_TILES, AI_WORK, EXAMPLE_TILES, FTE_ROOT
from fte.waste.boxes import box_iou

log = logging.getLogger("fte.waste.export")

CONFIRMED_CSV: Final = AI_WORK.parent / "configs" / "waste_confirmed.csv"
HEADER: Final = ("tile_id", "xtl", "ytl", "xbr", "ybr", "decision", "category", "reviewer", "note")
CATEGORIES: Final = frozenset({"bag", "bottle", "tyre", "debris", "heap", "unknown"})
REVIEWER: Final = "fte-waste@v1 (auto)"
TILE_PX: Final = 2048.0
CROP_PX: Final = 160
SHEET_COLS: Final = 6
SHEET_ROWS: Final = 5
CAPTION_PX: Final = 18


def confirmed_text(path: Path = CONFIRMED_CSV) -> str:
    """The confirmed CSV verbatim (header checked)."""
    text = path.read_text(encoding="utf-8")
    header = text.splitlines()[0].strip() if text else ""
    if tuple(header.split(",")) != HEADER:
        raise ValueError(f"unexpected header in {path}: {header!r}")
    return text if text.endswith("\n") else text + "\n"


def _confirmed_frame(text: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text))


def drop_duplicates_of(frame: pd.DataFrame, confirmed: pd.DataFrame, iou: float = 0.3) -> pd.DataFrame:
    """Boxes of ``frame`` that do not overlap (IoU >= iou) a confirmed box of the same tile."""
    keep = np.ones(len(frame), dtype=bool)
    for i, row in enumerate(frame.itertuples(index=False)):
        same = confirmed[confirmed["tile_id"] == row.tile_id][["xtl", "ytl", "xbr", "ybr"]].to_numpy(float)
        if len(same):
            b = np.array([[row.xtl, row.ytl, row.xbr, row.ybr]], dtype=float)
            keep[i] = not (box_iou(b, same) >= iou).any()
    return frame[keep].reset_index(drop=True)


def csv_rows(frame: pd.DataFrame, category: str = "unknown", reviewer: str = REVIEWER) -> list[list[str]]:
    """One ``add`` row per box, coordinates clipped to the tile and rounded to 0.1 px."""
    if category not in CATEGORIES:
        raise ValueError(f"category must be one of {sorted(CATEGORIES)}")
    rows = []
    for r in frame.sort_values(["tile_id", "score"], ascending=[True, False]).itertuples(index=False):
        xs = [float(np.clip(v, 0.0, TILE_PX)) for v in (r.xtl, r.ytl, r.xbr, r.ybr)]
        if xs[2] <= xs[0] or xs[3] <= xs[1]:
            continue
        rows.append([r.tile_id, *(f"{v:.1f}" for v in xs), "add", category, reviewer, f"score={r.score:.2f}"])
    return rows


def write_csv(path: Path, confirmed: str, rows: Sequence[Sequence[str]]) -> Path:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(confirmed + buf.getvalue(), encoding="utf-8")
    return path


def crop_with_box(rgb: np.ndarray, box: Sequence[float], out_px: int = CROP_PX) -> np.ndarray:
    """Square context crop around the box (>= 96 px, box + 60 %), resized, box drawn in red."""
    x0, y0, x1, y1 = box
    side = max(96.0, 1.6 * max(x1 - x0, y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h, w = rgb.shape[:2]
    sx = int(np.clip(round(cx - side / 2), 0, max(0, w - side)))
    sy = int(np.clip(round(cy - side / 2), 0, max(0, h - side)))
    s = int(side)
    crop = rgb[sy:sy + s, sx:sx + s]
    scale = out_px / max(1, max(crop.shape[:2]))
    img = cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_LINEAR)
    p0 = (int((x0 - sx) * scale), int((y0 - sy) * scale))
    p1 = (int((x1 - sx) * scale), int((y1 - sy) * scale))
    cv2.rectangle(img, p0, p1, (255, 0, 0), 1)
    return img


def _cell(crop: np.ndarray, caption: str) -> np.ndarray:
    cell = np.full((CROP_PX + CAPTION_PX, CROP_PX, 3), 255, np.uint8)
    cell[:CROP_PX] = crop
    cv2.putText(cell, caption, (2, CROP_PX + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
    return cell


def contact_sheets(frame: pd.DataFrame, out_dir: Path, per_sheet: int = SHEET_COLS * SHEET_ROWS) -> list[Path]:
    """JPEG sheets of the boxes of ``frame`` in its order (tiles read once each)."""
    from vineyard.geo.raster import read_tile

    cells: dict[int, np.ndarray] = {}
    for tile_id, sub in frame.groupby("tile_id", sort=False):
        rgb = read_tile(AI_TILES / f"{tile_id}.tif")
        for idx, r in sub.iterrows():
            cap = f"{str(tile_id).removeprefix('siret3_')} {r.score:.2f}"
            cells[idx] = _cell(crop_with_box(rgb, (r.xtl, r.ytl, r.xbr, r.ybr)), cap)
    ordered = [cells[i] for i in frame.index if i in cells]
    paths = []
    for k in range(0, len(ordered), per_sheet):
        chunk = ordered[k:k + per_sheet]
        blank = np.full_like(chunk[0], 255)
        chunk = chunk + [blank] * (per_sheet - len(chunk))
        rows = [np.concatenate(chunk[r * SHEET_COLS:(r + 1) * SHEET_COLS], axis=1) for r in range(SHEET_ROWS)]
        path = out_dir / f"sheet_{k // per_sheet + 1:03d}.jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.concatenate(rows, axis=0)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    return paths


def review_frame(pred_dir: Path, threshold: float, review_min: float, selected: pd.DataFrame,
                 limit: int) -> pd.DataFrame:
    """Boxes at ``review_min`` not vetoed, not overlapping a selected box, scored below ``threshold``."""
    from fte.waste.select import boxes_at

    tiles = sorted(h.stem for h in (pred_dir / "heat").glob("*.png") if h.stem not in EXAMPLE_TILES)
    low = boxes_at(pred_dir, tiles, review_min)
    low = low[(low["veto"] == "") & (low["score"] < threshold)]
    low = drop_duplicates_of(low, selected, iou=0.1) if len(selected) else low
    return low.sort_values("score", ascending=False).head(limit).reset_index(drop=True)


def run(pred_dir: Path, review_min: float, review_limit: int, sheets: bool = True) -> dict:
    selected = pd.read_parquet(pred_dir / "selected.parquet")
    report = json.loads((pred_dir / "select_report.json").read_text(encoding="utf-8"))
    confirmed = confirmed_text()
    new = drop_duplicates_of(selected, _confirmed_frame(confirmed))
    out_csv = write_csv(pred_dir / "waste_fte.csv", confirmed, csv_rows(new))
    review = review_frame(pred_dir, float(report["threshold"]), review_min, selected, review_limit)
    review_dir = pred_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(review_dir / "review_boxes.csv", index=False, float_format="%.3f")
    sheet_paths = contact_sheets(review, review_dir) if sheets and len(review) else []
    return {"csv": str(out_csv), "n_added": int(len(new)), "n_confirmed_dupes": int(len(selected) - len(new)),
            "n_review": int(len(review)), "n_sheets": len(sheet_paths), "threshold": report["threshold"]}


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="export waste boxes for Marcaj + review sheets")
    ap.add_argument("--pred-dir", type=Path, default=Path("work/preds/waste/v1"))
    ap.add_argument("--review-min", type=float, default=0.15)
    ap.add_argument("--review-limit", type=int, default=600)
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args(argv)
    pred = a.pred_dir if a.pred_dir.is_absolute() else FTE_ROOT / a.pred_dir
    return run(pred, a.review_min, a.review_limit, not a.no_sheets)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(json.dumps(main(), indent=2))
