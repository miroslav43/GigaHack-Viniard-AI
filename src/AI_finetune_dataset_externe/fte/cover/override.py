"""Interrow cover override for borderline pieces, from the ground-vegetation (c2) probability map.

    python -m fte.cover.override --c2-dir work/preds/canopy/v1/c2 --annset <run-id|run dir|annset dir>
        [--out work/reports/cover_override.csv] [--check-examples]

Pieces the rule flagged `cover_borderline` get veg_frac2 = mean(c2 >= 0.5) over the visible pixels of
the piece; new_frac = mean(veg_frac, veg_frac2) is classified with the rule's thresholds
(bare < cover_bare_max <= mixed <= cover_veg_min < vegetation). Unassessable pieces are kept.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import numpy as np
import pandas as pd

from fte.paths import AI_WORK, PREDS, WORK

BORDERLINE_FLAG: Final = "cover_borderline"
UNASSESSABLE: Final = "unassessable"
C2_THRESHOLD: Final = 0.5
DEFAULT_C2_DIR: Final = PREDS / "canopy" / "v1" / "c2"
DEFAULT_OUT: Final = WORK / "reports" / "cover_override.csv"
CSV_COLUMNS: Final = ("piece_id", "interrow_cover", "veg_frac", "veg_frac2", "changed")
CHANGED_MAX_FRAC: Final = 0.15
MapFn = Callable[[str], np.ndarray | None]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoverThresholds:
    bare_max: float = 0.25
    veg_min: float = 0.75

    @classmethod
    def from_config(cls, interrow_cfg: Any) -> CoverThresholds:
        return cls(float(interrow_cfg.cover_bare_max), float(interrow_cfg.cover_veg_min))


def classify_frac(frac: float, th: CoverThresholds) -> str:
    """Same rule as vineyard.perception.interrow.classify_cover for an assessable piece."""
    if frac < th.bare_max:
        return "bare_soil"
    if frac > th.veg_min:
        return "vegetation"
    return "mixed"


def is_borderline(flags: object) -> bool:
    return isinstance(flags, str) and BORDERLINE_FLAG in flags.replace(",", ";").split(";")


def piece_mask(poly: Any, x0: float, y0: float, shape: tuple[int, int]) -> tuple[np.ndarray, int, int]:
    """Bool mask of the pixels whose centre is inside `poly` (UTM), cropped to its bounds; (mask, row0, col0)."""
    import rasterio.features
    import shapely
    from affine import Affine

    gsd = 0.025
    px = shapely.transform(poly, lambda xy: np.column_stack(((xy[:, 0] - x0) / gsd, (y0 - xy[:, 1]) / gsd)))
    minx, miny, maxx, maxy = px.bounds
    c0, r0 = max(int(math.floor(minx)), 0), max(int(math.floor(miny)), 0)
    c1, r1 = min(int(math.ceil(maxx)), shape[1]), min(int(math.ceil(maxy)), shape[0])
    if c1 <= c0 or r1 <= r0:
        return np.zeros((0, 0), dtype=bool), r0, c0
    local = shapely.affinity.translate(px, -c0, -r0)
    m = rasterio.features.rasterize([(local, 1)], out_shape=(r1 - r0, c1 - c0), transform=Affine.identity(),
                                    fill=0, dtype=np.uint8, all_touched=False)
    return m > 0, r0, c0


def veg_frac2(poly: Any, c2: np.ndarray, valid: np.ndarray | None, x0: float, y0: float) -> float:
    """Share of the piece's valid pixels with c2 >= 0.5 (NaN when none)."""
    m, r0, c0 = piece_mask(poly, x0, y0, c2.shape)
    if m.size == 0:
        return math.nan
    h, w = m.shape
    probs = c2[r0: r0 + h, c0: c0 + w].astype(float)
    probs = probs / 255.0 if c2.dtype == np.uint8 else probs
    inside = m if valid is None else m & valid[r0: r0 + h, c0: c0 + w]
    return float((probs[inside] >= C2_THRESHOLD).mean()) if inside.any() else math.nan


def _tile_rows(group: gpd.GeoDataFrame, c2: np.ndarray, valid: np.ndarray | None, th: CoverThresholds,
               x0: float, y0: float) -> list[dict[str, Any]]:
    rows = []
    for rec in group.itertuples(index=False):
        if rec.interrow_cover == UNASSESSABLE or not is_borderline(rec.qa_flags):
            continue
        f2 = veg_frac2(rec.geometry, c2, valid, x0, y0)
        if math.isnan(f2) or math.isnan(float(rec.veg_frac)):
            continue
        cover = classify_frac((float(rec.veg_frac) + f2) / 2.0, th)
        rows.append({"piece_id": rec.piece_id, "interrow_cover": cover, "veg_frac": float(rec.veg_frac),
                     "veg_frac2": f2, "changed": cover != rec.interrow_cover})
    return rows


def override_cover(pieces: gpd.GeoDataFrame, c2_map: MapFn, th: CoverThresholds,
                   valid_map: MapFn | None = None) -> pd.DataFrame:
    """One row per borderline, assessable piece with a c2 map: the proposed cover and the fractions."""
    from vineyard.geo.tiling import tile_ref

    rows: list[dict[str, Any]] = []
    for tile_id, group in pieces.groupby("tile_id", sort=True):
        c2 = c2_map(str(tile_id))
        if c2 is None:
            continue
        ref = tile_ref(str(tile_id))
        valid = valid_map(str(tile_id)) if valid_map is not None else None
        rows.extend(_tile_rows(group, c2, valid, th, float(ref.x0), float(ref.y0)))
    return pd.DataFrame(rows, columns=list(CSV_COLUMNS))


def png_map_loader(directory: Path) -> MapFn:
    import cv2

    root = Path(directory)

    def load(tile_id: str) -> np.ndarray | None:
        path = root / f"{tile_id}.png"
        if not path.is_file():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cannot read map {path}")
        return img if img.ndim == 2 else img[..., 0]

    return load


def vis_valid_loader(cache_dir: Path = AI_WORK / "cache") -> MapFn:
    """Visible-data mask from the tile-prep vis codes (as the rule), else the valid mask, else None."""
    from vineyard.perception.types import VIS_NODATA
    from vineyard.pipeline.tile_cache import load_valid_mask, load_vis, valid_mask_path, vis_path

    def load(tile_id: str) -> np.ndarray | None:
        if vis_path(cache_dir, tile_id).is_file():
            return load_vis(cache_dir, tile_id) != VIS_NODATA
        if valid_mask_path(cache_dir, tile_id).is_file():
            return load_valid_mask(cache_dir, tile_id)
        return None

    return load


def _annset_dir(ref: str) -> Path:
    from vineyard.annset.io import resolve_run_dir

    path = Path(ref)
    if (path / "annset.json").is_file():
        return path
    return resolve_run_dir(AI_WORK, ref) / "annset"


def check_examples(annset_dir: Path, csv_path: Path) -> dict[str, Any]:
    """Official interrow-cover attribute score on the example tiles with the override applied."""
    from vineyard.annset.io import read_annset, resolve_run_dir
    from vineyard.config.loader import load_config
    from vineyard.eval.report import evaluate_annsets

    from fte.integrate.annset_edit import apply_cover_csv
    from fte.paths import EXAMPLE_TILES

    base = read_annset(annset_dir).for_tiles(EXAMPLE_TILES)
    table = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    local = table[table["piece_id"].isin(set(base.interrow_pieces["piece_id"]))]
    tmp = csv_path.with_suffix(".examples.csv")
    local.to_csv(tmp, index=False)
    pieces, n_changed = apply_cover_csv(base.interrow_pieces, tmp)
    ref = read_annset(resolve_run_dir(AI_WORK, "LATEST_REFERENCE") / "annset")
    report = evaluate_annsets(base.with_layer("interrow_pieces", pieces), ref, EXAMPLE_TILES, load_config().eval)
    keys = ("attributes.interrow_cover.accuracy", "attributes.interrow_cover.score", "attributes.score")
    return {"n_changed_examples": n_changed, **{k: report.mean.get(k) for k in keys}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c2-dir", type=Path, default=DEFAULT_C2_DIR)
    ap.add_argument("--annset", default="complete-v4")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check-examples", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from vineyard.annset.io import read_annset
    from vineyard.config.loader import load_config

    if not args.c2_dir.is_dir():
        raise SystemExit(f"c2 directory not found: {args.c2_dir}")
    annset_dir = _annset_dir(args.annset)
    pieces = read_annset(annset_dir).interrow_pieces
    th = CoverThresholds.from_config(load_config().interrow)
    table = override_cover(pieces, png_map_loader(args.c2_dir), th, vis_valid_loader())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    frac = float(table["changed"].sum()) / max(len(pieces), 1)
    summary = {"annset": str(annset_dir), "n_pieces": len(pieces), "n_borderline_scored": len(table),
               "n_changed": int(table["changed"].sum()), "changed_frac_all": round(frac, 4),
               "gate_changed_frac_ok": frac <= CHANGED_MAX_FRAC}
    if args.check_examples:
        summary |= check_examples(annset_dir, args.out)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
