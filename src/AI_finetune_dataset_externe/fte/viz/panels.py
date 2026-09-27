"""QA panels: per tile a 1024 px JPEG with base canopies (green), new partition pieces (red), waste boxes
(magenta, with score when a scores CSV is given) and interrow cover changes (cyan label).

    python -m fte.viz.panels --run <out-run> --base complete-v4 [--tiles t1 t2 ...] [--scores scores.csv]
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import pandas as pd

from fte.paths import AI_RUNS, AI_TILES, WORK

OUT_DIR: Final = WORK / "reports" / "panels"
PANEL_PX: Final = 1024
TILE_PX: Final = 2048
GSD_M: Final = 0.025
JPEG_QUALITY: Final = 88
GREEN, RED, MAGENTA, CYAN = (0, 200, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0)  # BGR
HARD_TILES: Final = ("siret3_r024_c016", "siret3_r006_c003", "siret3_r036_c025", "siret3_r010_c001",
                     "siret3_r032_c021", "siret3_r035_c025", "siret3_r009_c002", "siret3_r019_c012",
                     "siret3_r038_c023")

_log = logging.getLogger(__name__)


def _to_panel_px(xy: np.ndarray, x0: float, y0: float) -> np.ndarray:
    scale = PANEL_PX / TILE_PX
    return np.column_stack(((xy[:, 0] - x0) / GSD_M * scale, (y0 - xy[:, 1]) / GSD_M * scale))


def _rings(geom: Any) -> list[np.ndarray]:
    polys = list(getattr(geom, "geoms", [geom]))
    return [np.asarray(p.exterior.coords) for p in polys if not p.is_empty]


def _draw_polys(img: np.ndarray, geoms: Sequence[Any], x0: float, y0: float, color: tuple[int, int, int],
                thickness: int) -> np.ndarray:
    out = img.copy()
    for geom in geoms:
        for ring in _rings(geom):
            pts = np.round(_to_panel_px(ring, x0, y0)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, color, thickness, lineType=cv2.LINE_AA)
    return out


def _label(img: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def _background(tile_id: str) -> np.ndarray:
    from vineyard.geo.raster import read_tile

    path = AI_TILES / f"{tile_id}.tif"
    if not path.is_file():
        raise FileNotFoundError(f"tile image missing: {path}")
    rgb = read_tile(path)
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (PANEL_PX, PANEL_PX), interpolation=cv2.INTER_AREA)


def _new_geoms(base: Any, run: Any) -> list[Any]:
    known = {g.wkb for g in base.geometry}
    return [g for g in run.geometry if g.wkb not in known]


def _waste_boxes(img: np.ndarray, waste: Any, scores: pd.DataFrame | None) -> np.ndarray:
    out = img.copy()
    scale = PANEL_PX / TILE_PX
    for rec in waste.itertuples(index=False):
        p0 = (int(rec.px_xtl * scale), int(rec.px_ytl * scale))
        p1 = (int(rec.px_xbr * scale), int(rec.px_ybr * scale))
        cv2.rectangle(out, p0, p1, MAGENTA, 2)
        text = str(rec.waste_id)
        if scores is not None:
            hit = scores[(scores["tile_id"] == rec.tile_id) & ((scores["xtl"] - rec.px_xtl).abs() < 2)
                         & ((scores["ytl"] - rec.px_ytl).abs() < 2)]
            if len(hit):
                text += f" {float(hit['score'].iloc[0]):.2f}"
        out = _label(out, text, (p0[0], max(p0[1] - 4, 10)), MAGENTA)
    return out


def _cover_changes(img: np.ndarray, base: Any, run: Any, x0: float, y0: float) -> tuple[np.ndarray, int]:
    merged = base[["piece_id", "interrow_cover"]].merge(run[["piece_id", "interrow_cover", "geometry"]],
                                                        on="piece_id", suffixes=("_base", ""))
    changed = merged[merged["interrow_cover_base"] != merged["interrow_cover"]]
    out = img
    for rec in changed.itertuples(index=False):
        c = rec.geometry.representative_point()
        u, v = _to_panel_px(np.array([[c.x, c.y]]), x0, y0)[0]
        out = _label(out, f"{rec.interrow_cover_base}->{rec.interrow_cover}", (int(u), int(v)), CYAN)
    return out, len(changed)


def render_panel(tile_id: str, base: Any, run: Any, scores: pd.DataFrame | None = None) -> np.ndarray:
    """BGR panel of one tile (base / run are AnnSets)."""
    from vineyard.geo.tiling import tile_ref

    ref = tile_ref(tile_id)
    pick = lambda a, n: a.layer(n)[a.layer(n)["tile_id"] == tile_id]  # noqa: E731
    base_can, run_can = pick(base, "canopies"), pick(run, "canopies")
    img = _draw_polys(_background(tile_id), list(base_can.geometry), ref.x0, ref.y0, GREEN, 1)
    new = _new_geoms(base_can, run_can)
    img = _draw_polys(img, new, ref.x0, ref.y0, RED, 1)
    img = _waste_boxes(img, pick(run, "waste"), scores)
    img, n_cover = _cover_changes(img, pick(base, "interrow_pieces"), pick(run, "interrow_pieces"), ref.x0, ref.y0)
    title = (f"{tile_id}  canopies {len(base_can)}->{len(run_can)} (new pieces {len(new)})  "
             f"waste {len(pick(run, 'waste'))}  cover changes {n_cover}")
    return _label(img, title, (8, 16), (255, 255, 255))


def write_panels(run_id: str, base_id: str, tiles: Sequence[str], scores_csv: Path | None = None,
                 out_dir: Path = OUT_DIR) -> list[Path]:
    from vineyard.annset.io import read_annset

    base, run = read_annset(AI_RUNS / base_id / "annset"), read_annset(AI_RUNS / run_id / "annset")
    scores = pd.read_csv(scores_csv) if scores_csv is not None else None
    if scores is not None and not {"tile_id", "xtl", "ytl", "score"} <= set(scores.columns):
        raise ValueError(f"scores CSV {scores_csv} needs tile_id,xtl,ytl,score")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for tile_id in tiles:
        if not (AI_TILES / f"{tile_id}.tif").is_file() or tile_id not in run.meta.tile_ids:
            _log.warning("skipping %s (no tile image or not in the run)", tile_id)
            continue
        path = out_dir / f"{run_id}_{tile_id}.jpg"
        cv2.imwrite(str(path), render_panel(tile_id, base, run, scores), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="complete-v4")
    ap.add_argument("--tiles", nargs="*", default=list(HARD_TILES))
    ap.add_argument("--scores", type=Path, default=None, help="CSV tile_id,xtl,ytl,...,score for waste labels")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for path in write_panels(args.run, args.base, args.tiles, args.scores, args.out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
