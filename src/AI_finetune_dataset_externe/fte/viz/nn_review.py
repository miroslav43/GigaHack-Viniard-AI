"""Side-by-side review panels: classical canopy (green) vs the canopy model's c0 mask (magenta).

For tiles where the classical pipeline misses vines (``src/AI/configs/tile_review.csv``: missed /
partial) this shows whether the network finds the vine rows. Writes one JPEG per tile, contact
sheets and a CSV with areas (m²) found by each method.

CLI: ``python -m fte.viz.nn_review --pred-dir work/preds/canopy/v1b_review --out work/reports/nn_review``
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fte.paths import AI_ROOT, AI_RUNS, AI_TILES, BASE_RUN, EXAMPLE_TILES

log = logging.getLogger(__name__)

PANEL = 768
PX_AREA_M2 = 0.025 * 0.025
GREEN = (60, 220, 60)
MAGENTA = (255, 0, 200)


@dataclass(frozen=True)
class TileReview:
    tile_id: str
    status: str
    note: str
    cv_m2: float
    nn_m2: float


def read_review_list(path: Path) -> dict[str, tuple[str, str]]:
    with path.open() as fh:
        return {r["tile_id"]: (r["status"], r.get("note", "")) for r in csv.DictReader(fh)}


def cv_mask(canopies, tile_id: str) -> np.ndarray:
    """Rasterised classical canopy polygons of one tile (bool 2048²)."""
    from vineyard.geo.tiling import tile_ref, utm_to_px

    mask = np.zeros((2048, 2048), np.uint8)
    t = tile_ref(tile_id)
    for geom in canopies.loc[canopies["tile_id"] == tile_id, "geometry"]:
        for poly in getattr(geom, "geoms", [geom]):
            ring = utm_to_px(t, np.asarray(poly.exterior.coords)[:, :2])
            cv2.fillPoly(mask, [np.round(ring).astype(np.int32)], 1)
    return mask.astype(bool)


def overlay(rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(colour, np.float32)
    return out.astype(np.uint8)


def caption(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def tile_panel(rgb: np.ndarray, cv: np.ndarray, nn: np.ndarray, r: TileReview) -> np.ndarray:
    small = lambda a: cv2.resize(a, (PANEL, PANEL), interpolation=cv2.INTER_AREA)  # noqa: E731
    left = caption(small(overlay(rgb, cv, GREEN)), f"{r.tile_id} [{r.status}] CV canopy {r.cv_m2:.0f} m2")
    right = caption(small(overlay(rgb, nn, MAGENTA)), f"model c0>=0.5  {r.nn_m2:.0f} m2  | {r.note[:48]}")
    return np.hstack([left, np.full((PANEL, 6, 3), 255, np.uint8), right])


def contact_sheets(panels: Sequence[np.ndarray], out_dir: Path, per_sheet: int = 6) -> list[Path]:
    paths = []
    for k in range(0, len(panels), per_sheet):
        chunk = [cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2), interpolation=cv2.INTER_AREA)
                 for p in panels[k:k + per_sheet]]
        sheet = np.vstack(chunk)
        path = out_dir / f"sheet_{k // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        paths.append(path)
    return paths


def review_tile(tile_id: str, status: str, note: str, canopies, pred_dir: Path, thr: float,
                out_dir: Path) -> tuple[TileReview, np.ndarray]:
    from vineyard.geo.raster import read_tile

    rgb = read_tile(AI_TILES / f"{tile_id}.tif")
    prob = cv2.imread(str(pred_dir / "c0" / f"{tile_id}.png"), cv2.IMREAD_GRAYSCALE)
    if prob is None:
        raise FileNotFoundError(f"no c0 prediction for {tile_id} in {pred_dir}")
    nn = prob >= int(round(thr * 255))
    cv = cv_mask(canopies, tile_id)
    r = TileReview(tile_id, status, note, float(cv.sum() * PX_AREA_M2), float(nn.sum() * PX_AREA_M2))
    panel = tile_panel(rgb, cv, nn, r)
    cv2.imwrite(str(out_dir / f"{tile_id}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    return r, panel


def main(argv: Sequence[str] | None = None) -> None:
    import geopandas as gpd

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-run", default=BASE_RUN)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a.out.mkdir(parents=True, exist_ok=True)
    review = read_review_list(AI_ROOT / "configs" / "tile_review.csv")
    wanted = {t: v for t, v in review.items() if v[0] in ("missed", "partial")}
    wanted.update({t: ("example", "reference tile (CV is good here)") for t in EXAMPLE_TILES})
    canopies = gpd.read_parquet(AI_RUNS / a.base_run / "annset" / "canopies.parquet")
    rows, panels = [], []
    for tile_id, (status, note) in wanted.items():
        r, panel = review_tile(tile_id, status, note, canopies, a.pred_dir, a.threshold, a.out)
        rows.append(r)
        panels.append(panel)
    order = sorted(range(len(rows)), key=lambda i: rows[i].nn_m2 - rows[i].cv_m2, reverse=True)
    sheets = contact_sheets([panels[i] for i in order], a.out)
    with (a.out / "areas.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_id", "status", "cv_m2", "nn_m2", "nn_minus_cv_m2", "note"])
        for i in order:
            r = rows[i]
            w.writerow([r.tile_id, r.status, round(r.cv_m2, 1), round(r.nn_m2, 1), round(r.nn_m2 - r.cv_m2, 1), r.note])
    log.info("%d tile panels, %d sheets -> %s", len(rows), len(sheets), a.out)


if __name__ == "__main__":
    main()
