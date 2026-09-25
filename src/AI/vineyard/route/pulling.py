"""String pulling between anchors (arch §4.11.5, design 04 §3.9).

From vertex i, the window i+1 .. min(next anchor, i + lookahead) is tested at once with
`covered_by(segment, eroded)` on the prepared domain, and the farthest vertex whose straight segment
is covered becomes the next vertex (i + 1 is always allowed: it is an original edge). Anchors are never
skipped, so every stop, and therefore every must target, keeps its distance to the route.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.route.unroll import dedupe_xy


def _check(n: int, anchors: Sequence[int], lookahead: int) -> None:
    if lookahead < 1:
        raise ValueError(f"lookahead must be >= 1, got {lookahead}")
    a = list(anchors)
    if not a or a[0] != 0 or a[-1] != n - 1 or any(y < x for x, y in zip(a[:-1], a[1:], strict=True)):
        raise ValueError(f"anchors must start at 0, end at {n - 1} and be non-decreasing, got {a[:10]}")


def _covered(pi: np.ndarray, pj: np.ndarray, eroded: BaseGeometry) -> np.ndarray:
    """covered_by for the segments pi -> pj[k]; a degenerate segment is tested as its point."""
    same = np.all(pj == pi, axis=1)
    out = np.zeros(len(pj), dtype=bool)
    if np.any(~same):
        segs = shapely.linestrings(np.stack([np.broadcast_to(pi, pj[~same].shape), pj[~same]], axis=1))
        out[~same] = shapely.covered_by(segs, eroded)
    if np.any(same):
        out[same] = shapely.covered_by(shapely.points(pi), eroded)
    return out


def _next_vertex(xy: np.ndarray, i: int, hi: int, eroded: BaseGeometry) -> int:
    if hi <= i + 1:
        return hi
    ok = _covered(xy[i], xy[i + 2:hi + 1], eroded)
    hits = np.flatnonzero(ok)
    return int(i + 2 + hits[-1]) if len(hits) else i + 1


def string_pull(xy: np.ndarray, anchors: Sequence[int], eroded: BaseGeometry,
                lookahead: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Pulled vertices and the new anchor indices (consecutive duplicates removed)."""
    pts = np.asarray(xy, dtype=np.float64)
    _check(len(pts), anchors, lookahead)
    shapely.prepare(eroded)
    keep, new_anchors = [0], [0]
    for a, b in zip(anchors[:-1], anchors[1:], strict=True):
        i = a
        while i < b:
            i = _next_vertex(pts, i, min(b, i + lookahead), eroded)
            keep.append(i)
        new_anchors.append(len(keep) - 1)
    out, final = dedupe_xy(pts[keep], new_anchors)
    return out, final


__all__ = ["string_pull"]
