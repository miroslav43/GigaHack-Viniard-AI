"""Farm route figure for the Route slide: the farm's orthophoto with its per-farm route drawn on top.

Needs rasterio and pyproj, so run it in the pipeline environment, from the repo root:
    cd src/AI && uv run --no-sync python ../../prezentare/scripts/make_farm_map.py
Inputs: the web bundle (farms, targets), the survey tiles in src/AI/work/tiles and the GPX exported by the
web app's farm route tool (prezentare/data/). Writes prezentare/img/farm_ortho.jpg and prezentare/fig/map_farm.tex.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "src/Web/frontend/public/data/siret3"
TILES = ROOT / "src/AI/work/tiles"
FARM_ID = "F09"
FARM_GPX = ROOT / "prezentare/data/traseu_F09.gpx"
OUT_IMG = ROOT / "prezentare/img/farm_ortho.jpg"
OUT_FIG = ROOT / "prezentare/fig/map_farm.tex"

UTM = "EPSG:32635"
RES_M = 0.35
PAD_M = 6.0
MAP_H_CM = 6.3
WHITE_BLEND = 0.12
SCALE_BAR_M = 50.0
JPEG_QUALITY = 86


def _fail(msg: str) -> None:
    sys.exit(f"make_farm_map: {msg}")


def _load(name: str) -> list[dict]:
    path = BUNDLE / name
    if not path.exists():
        _fail(f"missing {path}")
    return json.loads(path.read_text())["features"]


def _rings(geometry: dict) -> list[list[list[float]]]:
    polys = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    return [poly[0] for poly in polys]


def _read_gpx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        _fail(f"missing {path} (export the farm route from the web app)")
    text = path.read_text()
    track = np.array([(float(lon), float(lat)) for lat, lon in re.findall(r'<trkpt lat="([-\d.]+)" lon="([-\d.]+)"', text)])
    wpts = np.array([(float(lon), float(lat)) for lat, lon in re.findall(r'<wpt lat="([-\d.]+)" lon="([-\d.]+)"', text)])
    if len(track) < 2:
        _fail(f"{path}: no track")
    return track, wpts


def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker; a closed loop's zero-length base falls back to the distance from its end point."""
    keep = np.zeros(len(pts), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[b] - pts[a]
        rel = pts[a + 1:b] - pts[a]
        norm = np.hypot(*seg)
        d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / norm if norm > 1e-6 else np.hypot(rel[:, 0], rel[:, 1])
        i = int(np.argmax(d))
        if d[i] > tol:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


def _ortho(bbox: tuple[float, float, float, float]) -> np.ndarray:
    """RGB mosaic of the tiles over bbox at RES_M, nodata painted white, lightly blended to white."""
    x0, y0, x1, y1 = bbox
    paths = []
    for tif in sorted(TILES.glob("*.tif")):
        with rasterio.open(tif) as src:
            b = src.bounds
            if b.right > x0 and b.left < x1 and b.top > y0 and b.bottom < y1:
                paths.append(tif)
    if not paths:
        _fail(f"no tile in {TILES} covers the farm")
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, _ = merge(srcs, bounds=bbox, res=RES_M, indexes=[1, 2, 3], nodata=0)
    finally:
        for s in srcs:
            s.close()
    rgb = np.moveaxis(arr, 0, -1).astype(np.float32)
    empty = rgb.sum(axis=-1) == 0
    rgb = rgb * (1 - WHITE_BLEND) + 255 * WHITE_BLEND
    rgb[empty] = 255
    print(f"{len(paths)} tiles -> {rgb.shape[1]}x{rgb.shape[0]} px")
    return rgb.astype(np.uint8)


def _path(pts_cm: np.ndarray) -> str:
    return " -- ".join(f"({x:.3f},{y:.3f})" for x, y in pts_cm)


def main() -> None:
    farm = next((f for f in _load("farms.geojson") if f["properties"]["farm_id"] == FARM_ID), None)
    if farm is None:
        _fail(f"farm {FARM_ID} not in farms.geojson")
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    utm = lambda lonlat: np.column_stack(to_utm.transform(lonlat[:, 0], lonlat[:, 1]))  # noqa: E731

    rings = [utm(np.array(r)) for r in _rings(farm["geometry"])]
    track_ll, wpts_ll = _read_gpx(FARM_GPX)
    track, wpts = utm(track_ll), utm(wpts_ll)
    allpts = np.vstack(rings + [track])
    x0, y0 = allpts.min(axis=0) - PAD_M
    x1, y1 = allpts.max(axis=0) + PAD_M
    x1 = x0 + np.ceil((x1 - x0) / RES_M) * RES_M
    y1 = y0 + np.ceil((y1 - y0) / RES_M) * RES_M

    OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_ortho((x0, y0, x1, y1))).save(OUT_IMG, quality=JPEG_QUALITY)

    scale = MAP_H_CM / (y1 - y0)
    w_cm, h_cm = (x1 - x0) * scale, MAP_H_CM
    cm = lambda p: (p - [x0, y0]) * scale  # noqa: E731
    route = cm(_simplify(track, 0.2))
    sx, sy = cm(track[:1])[0]
    bar = SCALE_BAR_M * scale
    lines = [
        "% generated by prezentare/scripts/make_farm_map.py; do not edit",
        "\\begin{tikzpicture}[line cap=round, line join=round]",
        "\\begin{scope}",
        f"\\clip[rounded corners=4pt] (0,0) rectangle ({w_cm:.3f},{h_cm:.3f});",
        f"\\node[anchor=south west, inner sep=0pt] at (0,0) {{\\includegraphics[width={w_cm:.3f}cm,height={h_cm:.3f}cm]{{farm_ortho.jpg}}}};",
        *(f"\\draw[white, line width=0.6pt, dash pattern=on 2pt off 1.5pt] {_path(cm(r))} -- cycle;" for r in rings),
        f"\\draw[white, opacity=0.85, line width=1.5pt] {_path(route)};",
        f"\\draw[sxViolet, line width=0.8pt] {_path(route)};",
        *(f"\\filldraw[fill=sxAmber, draw=white, line width=0.2pt] ({x:.3f},{y:.3f}) circle (0.8pt);" for x, y in cm(wpts)),
        f"\\filldraw[fill=sxNight, draw=white, line width=0.6pt] ({sx:.3f},{sy:.3f}) circle (2.2pt);",
        f"\\node[anchor=west, font=\\tiny\\bfseries, text=sxNight, fill=white, inner sep=1pt, rounded corners=1pt]"
        f" at ({sx + 0.12:.3f},{sy:.3f}) {{START}};",
        f"\\fill[white, opacity=0.85, rounded corners=1pt] (0.15,0.15) rectangle ({0.35 + bar:.3f},0.62);",
        f"\\draw[sxNight, line width=0.8pt] (0.25,0.28) -- ({0.25 + bar:.3f},0.28);",
        f"\\node[anchor=south, font=\\tiny, text=sxNight, inner sep=0.5pt] at ({0.25 + bar / 2:.3f},0.3) {{{SCALE_BAR_M:.0f} m}};",
        "\\end{scope}",
        "\\end{tikzpicture}",
    ]
    OUT_FIG.write_text("\n".join(lines) + "\n")
    print(f"{FARM_ID}: {len(track)} -> {len(route)} track points, {len(wpts)} targets, map {w_cm:.2f} x {h_cm:.2f} cm")


if __name__ == "__main__":
    main()
