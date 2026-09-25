"""Human review files (A§4.9 step 6): qa/waste_candidates.csv, qa/waste_crops/<cand_key>.jpg and
qa/waste_review.html (crops by rank, each with the exact configs/waste_confirmed.csv line to paste).
"""

from __future__ import annotations

import csv
import html
import io
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
import pandas as pd

from vineyard.perception.waste.confirm import CONFIRM_HEADER, confirm_line
from vineyard.perception.waste.crops import box_to_crop, crop_window, extract_crop
from vineyard.perception.waste.types import BoxPx
from vineyard.pipeline.atomic import atomic_write_bytes, atomic_write_text

CSV_NAME: Final = "waste_candidates.csv"
CROPS_DIRNAME: Final = "waste_crops"
HTML_NAME: Final = "waste_review.html"
CSV_COLUMNS: Final = (
    "rank",
    "waste_id",
    "cand_key",
    "tile_id",
    "xtl",
    "ytl",
    "xbr",
    "ybr",
    "area_m2",
    "colour_class",
    "rank_score",
    "vineyard_id",
    "dist_block_m",
    "crop",
    "confirm_line",
)
BOX_RGB: Final = (255, 0, 255)
BOX_THICKNESS: Final = 1
JPEG_QUALITY: Final = 90
_SAFE: Final = re.compile(r"[^A-Za-z0-9_@.-]")


@dataclass(frozen=True)
class ReviewItem:
    rank: int
    waste_id: str
    cand_key: str
    tile_id: str
    box: BoxPx
    area_m2: float
    colour_class: str
    rank_score: float
    vineyard_id: str
    dist_block_m: float
    crop: str  # path relative to the qa dir
    line: str


def crop_name(cand_key: str) -> str:
    return f"{CROPS_DIRNAME}/{_SAFE.sub('_', cand_key)}.jpg"


def review_items(frame: pd.DataFrame) -> tuple[ReviewItem, ...]:
    """Items of the rows with review == True, best rank_score first (ties by cand_key)."""
    rows = frame[frame["review"].astype(bool)] if len(frame) else frame
    ordered = sorted(rows.to_dict("records"), key=lambda r: (-float(r["rank_score"]), str(r["cand_key"])))
    return tuple(_item(k + 1, r) for k, r in enumerate(ordered))


def _item(rank: int, r: Mapping[str, Any]) -> ReviewItem:
    box = BoxPx(*(float(r[c]) for c in ("px_xtl", "px_ytl", "px_xbr", "px_ybr")))
    return ReviewItem(
        rank=rank,
        waste_id=str(r["waste_id"]),
        cand_key=str(r["cand_key"]),
        tile_id=str(r["tile_id"]),
        box=box,
        area_m2=float(r["area_m2"]),
        colour_class=str(r["colour_class"]),
        rank_score=float(r["rank_score"]),
        vineyard_id=str(r["vineyard_id"]),
        dist_block_m=float(r["dist_block_m"]),
        crop=crop_name(str(r["cand_key"])),
        line=confirm_line(str(r["tile_id"]), box, note=str(r["cand_key"])),
    )


def write_review_csv(items: Sequence[ReviewItem], path: Path) -> Path:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for it in items:
        writer.writerow(
            [
                it.rank,
                it.waste_id,
                it.cand_key,
                it.tile_id,
                *(f"{v:.1f}" for v in it.box.as_tuple()),
                f"{it.area_m2:.4f}",
                it.colour_class,
                f"{it.rank_score:.4f}",
                it.vineyard_id,
                f"{it.dist_block_m:.2f}",
                it.crop,
                it.line,
            ]
        )
    return atomic_write_text(path, buf.getvalue())


def _draw_crop(rgb: np.ndarray, box: BoxPx, context: float, min_px: int, out_px: int) -> np.ndarray:
    extent = min(rgb.shape[:2])
    window = crop_window(box, context, min_px, extent)
    crop = extract_crop(rgb, window, out_px)
    x0, y0, x1, y1 = (round(v) for v in box_to_crop(box.as_tuple(), window, out_px))
    return cv2.rectangle(crop.copy(), (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), BOX_RGB, BOX_THICKNESS)


def write_review_crops(
    items: Sequence[ReviewItem],
    read_rgb: Callable[[str], np.ndarray],
    qa_dir: Path,
    *,
    context: float,
    min_px: int,
    out_px: int,
) -> tuple[Path, ...]:
    """One JPEG per item (box drawn in magenta); each tile is read once."""
    written = []
    for tile_id in sorted({it.tile_id for it in items}):
        rgb = read_rgb(tile_id)
        for it in (x for x in items if x.tile_id == tile_id):
            crop = _draw_crop(rgb, it.box, context, min_px, out_px)
            ok, data = cv2.imencode(
                ".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if not ok:
                raise ValueError(f"JPEG encoding failed for crop {it.cand_key}")
            written.append(atomic_write_bytes(qa_dir / it.crop, data.tobytes()))
    return tuple(written)


_CSS: Final = """body{font-family:system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px}
.card img{width:256px;height:256px;image-rendering:pixelated;display:block}
.meta{font-size:12px;margin:4px 0}
input{width:100%;font-family:monospace;font-size:11px}
code{background:#eee;padding:2px 4px}"""
_JS: Final = (
    "function cp(id){const e=document.getElementById(id);e.select();"
    "navigator.clipboard&&navigator.clipboard.writeText(e.value);}"
)


def _card(it: ReviewItem) -> str:
    e = html.escape
    meta = (
        f"#{it.rank} · {e(it.waste_id)} · {e(it.colour_class)} · score {it.rank_score:.3f} · "
        f"{it.area_m2:.3f} m² · vineyard {e(it.vineyard_id) or '—'}"
    )
    return (
        f'<div class="card"><img src="{e(it.crop)}" alt="{e(it.cand_key)}" loading="lazy">'
        f'<div class="meta">{meta}</div><div class="meta">{e(it.tile_id)} '
        f"[{it.box.xtl:.1f}, {it.box.ytl:.1f}, {it.box.xbr:.1f}, {it.box.ybr:.1f}]</div>"
        f'<input id="l{it.rank}" readonly value="{e(it.line)}" onclick="cp(\'l{it.rank}\')"></div>'
    )


def render_review_html(items: Sequence[ReviewItem], summary: Mapping[str, object]) -> str:
    e = html.escape
    facts = " · ".join(f"{e(str(k))}: {e(str(v))}" for k, v in sorted(summary.items()))
    cards = "\n".join(_card(it) for it in items) or "<p>Niciun candidat de revizuit.</p>"
    return (
        f'<!doctype html><html lang="ro"><head><meta charset="utf-8"><title>Waste review</title>'
        f"<style>{_CSS}</style><script>{_JS}</script></head><body><h1>Revizie deșeuri ({len(items)})</h1>"
        f"<p>{facts}</p><p>Pentru fiecare deșeu real, click pe linie (se copiază) și lipește-o în "
        f"<code>configs/waste_confirmed.csv</code> sub antetul <code>{e(','.join(CONFIRM_HEADER))}</code>; "
        f"schimbă <code>unknown</code> cu categoria (bag, bottle, tyre, debris, heap). Tuburile, țărușii, "
        f'pietrele, solul și furtunurile NU sunt deșeuri.</p><div class="grid">{cards}</div></body></html>\n'
    )


def write_review_html(items: Sequence[ReviewItem], path: Path, summary: Mapping[str, object]) -> Path:
    return atomic_write_text(path, render_review_html(items, summary))
