"""Row label-correctness fixes around linking (label audit 2026-09-26, run complete-v4).

* `orientation_conflicts`: in one tile, two accepted row families more than `angle_deg` apart whose lines
  cross each other cannot both be real (rows of one planting never cross); the weaker family (lower
  median SNR, then shorter total length) is rejected (r009_c002: false 112° rows between 123° rows).
* `join_plan`: chains whose facing ends meet (<= max_m, <= angle_deg, head to tail) are one physical row
  cut at a tile edge (or a split row): the ends are joined so the row keeps one row_id across the seam.
* `duplicate_pairs`: two chains that run within max_m of each other over most of the shorter one are one
  row drawn twice; they are merged by the caller.

All functions are pure and deterministic (sorted inputs, ties broken by index).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import LineString

from vineyard.perception.blocks_graph import axial_diff_deg, line_coords

REASON_ORIENTATION_CONFLICT: Final = "orientation_conflict"
EPS_M: Final = 1e-9


# ---------------------------------------------------------------- orientation conflicts (candidates)


def _families(angles: np.ndarray, angle_deg: float) -> list[list[int]]:
    """Greedy axial clusters: indices sorted by angle, a new family when the gap to the family's first
    member exceeds angle_deg (wrap-around merged)."""
    order = np.argsort(angles % 180.0, kind="stable")
    fams: list[list[int]] = []
    for i in order:
        if fams and axial_diff_deg(float(angles[fams[-1][0]]), float(angles[i])) <= angle_deg:
            fams[-1].append(int(i))
        else:
            fams.append([int(i)])
    if len(fams) > 1 and axial_diff_deg(float(angles[fams[0][0]]), float(angles[fams[-1][-1]])) <= angle_deg:
        fams[0] = fams.pop() + fams[0]
    return fams


def _family_strength(frame: gpd.GeoDataFrame, idx: Sequence[int]) -> tuple[float, float]:
    snr = frame["snr"].to_numpy(dtype=np.float64)[list(idx)] if "snr" in frame.columns else np.array([np.nan])
    med = float(np.nanmedian(snr)) if np.isfinite(snr).any() else 0.0
    return med, float(sum(frame.geometry.iloc[i].length for i in idx))


def orientation_conflicts(cands: gpd.GeoDataFrame, angle_deg: float, min_crossings: int) -> list[str]:
    """cand_ids of the accepted candidates to reject: per tile, of two crossing families the weaker one."""
    if cands.empty:
        return []
    accepted = cands[cands.rejected_reason.isna()]
    out: list[str] = []
    for _, grp in accepted.groupby("tile_id", sort=True):
        if len(grp) < 2:
            continue
        g = grp.reset_index(drop=True)
        fams = _families(g["angle_deg"].to_numpy(dtype=np.float64), angle_deg)
        if len(fams) < 2:
            continue
        lines = list(g.geometry)
        dropped: set[int] = set()
        for a in range(len(fams)):
            for b in range(a + 1, len(fams)):
                fa, fb = [i for i in fams[a] if i not in dropped], [i for i in fams[b] if i not in dropped]
                crossings = sum(1 for i in fa for j in fb if lines[i].crosses(lines[j]))
                if crossings < min_crossings:
                    continue
                weak = fa if _family_strength(g, fa) < _family_strength(g, fb) else fb
                dropped.update(weak)
        out.extend(str(g.cand_id.iloc[i]) for i in sorted(dropped))
    return sorted(out)


def reject_orientation_conflicts(cands: gpd.GeoDataFrame, angle_deg: float, min_crossings: int
                                 ) -> gpd.GeoDataFrame:
    """New frame with rejected_reason = orientation_conflict on the weaker crossing family of each tile."""
    ids = set(orientation_conflicts(cands, angle_deg, min_crossings))
    if not ids:
        return cands
    reason = [REASON_ORIENTATION_CONFLICT if str(c) in ids else r
              for c, r in zip(cands.cand_id, cands.rejected_reason, strict=True)]
    return cands.assign(rejected_reason=reason)


# ---------------------------------------------------------------- end joins (chains)


@dataclass(frozen=True)
class _End:
    chain: int
    first: bool  # True: the line's first vertex
    xy: np.ndarray
    out: np.ndarray  # unit vector pointing out of the line at this end


def _ends(lines: Sequence[LineString]) -> list[_End]:
    out = []
    for k, ln in enumerate(lines):
        xy = line_coords(ln)
        for first, end, inner in ((True, xy[0], xy[1]), (False, xy[-1], xy[-2])):
            d = end - inner
            n = float(np.hypot(*d))
            if n > EPS_M:
                out.append(_End(k, first, end, d / n))
    return out


def join_pairs(lines: Sequence[LineString], max_m: float, angle_deg: float) -> list[tuple[_End, _End]]:
    """Greedy (nearest first) end pairs of different chains that continue each other head to tail."""
    ends = _ends(lines)
    if len(ends) < 2:
        return []
    tree = shapely.STRtree([shapely.Point(e.xy) for e in ends])
    cand = []
    for i, j in zip(*tree.query([shapely.Point(e.xy) for e in ends], predicate="dwithin", distance=max_m),
                    strict=True):
        a, b = ends[int(i)], ends[int(j)]
        if int(i) >= int(j) or a.chain == b.chain:
            continue
        if float(a.out @ b.out) > -math.cos(math.radians(angle_deg)):
            continue  # not facing each other along one line
        if axial_diff_deg(_chord_angle(lines[a.chain]), _chord_angle(lines[b.chain])) > angle_deg:
            continue
        cand.append((float(np.hypot(*(a.xy - b.xy))), int(i), int(j)))
    used: set[int] = set()
    root = list(range(len(lines)))

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    out = []
    for _, i, j in sorted(cand):
        a, b = ends[i], ends[j]
        if i in used or j in used or find(a.chain) == find(b.chain):
            continue
        used.update((i, j))
        root[find(b.chain)] = find(a.chain)
        out.append((a, b))
    return out


def _chord_angle(line: LineString) -> float:
    xy = line_coords(line)
    d = xy[-1] - xy[0]
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


def join_plan(lines: Sequence[LineString], max_m: float, angle_deg: float, align_min_m: float = 0.0
              ) -> list[tuple[tuple[int, ...], LineString]]:
    """(chain indices in path order, joined line) of every group of >= 2 chains joined end to end. Where the
    two facing ends are more than align_min_m apart they are replaced by their midpoint (the seam is
    aligned, no lateral jump); closer ends are both kept, so each tile keeps its own row geometry."""
    pairs = join_pairs(lines, max_m, angle_deg)
    if not pairs:
        return []
    link: dict[tuple[int, bool], tuple[int, bool]] = {}
    for a, b in pairs:
        link[(a.chain, a.first)] = (b.chain, b.first)
        link[(b.chain, b.first)] = (a.chain, a.first)
    seen: set[int] = set()
    out = []
    for k in sorted({e.chain for p in pairs for e in p}):
        if k in seen:
            continue
        # walk to a path end: a chain with a free end
        start, start_free_first = _path_start(k, link)
        path, xy_parts = _walk(start, start_free_first, link, lines, align_min_m)
        seen.update(path)
        out.append((tuple(path), LineString(np.vstack(xy_parts))))
    return out


def _path_start(k: int, link: dict[tuple[int, bool], tuple[int, bool]]) -> tuple[int, bool]:
    cur, free_first = k, True
    visited = {k}
    while (cur, free_first) in link:
        nxt, nxt_first = link[(cur, free_first)]
        if nxt in visited:
            break
        visited.add(nxt)
        cur, free_first = nxt, not nxt_first
    return cur, free_first


def _walk(start: int, free_first: bool, link: dict[tuple[int, bool], tuple[int, bool]],
          lines: Sequence[LineString], align_min_m: float = 0.0) -> tuple[list[int], list[np.ndarray]]:
    """Chains from `start` (entered at its free end) along the links; coordinates oriented along the path,
    each seam vertex the midpoint of the two facing ends."""
    path: list[int] = []
    parts: list[np.ndarray] = []
    cur, entry_first = start, free_first
    head: np.ndarray | None = None  # seam vertex that replaces the entry vertex of `cur`
    while cur not in path:
        xy = line_coords(lines[cur])
        xy = xy if entry_first else xy[::-1]
        if head is not None:
            xy = np.vstack([head[None, :], xy[1:]])
        path.append(cur)
        exit_key = (cur, not entry_first)
        if exit_key not in link or link[exit_key][0] in path:
            parts.append(xy)
            break
        nxt, nxt_first = link[exit_key]
        nxy = line_coords(lines[nxt])
        facing = nxy[0] if nxt_first else nxy[-1]
        if float(np.hypot(*(xy[-1] - facing))) > align_min_m:
            head = 0.5 * (xy[-1] + facing)
            parts.append(np.vstack([xy[:-1], head[None, :]]))
        else:  # both ends kept: the part of each tile is unchanged
            head = None
            parts.append(np.vstack([xy, np.full((1, 2), np.nan)]))
        cur, entry_first = nxt, nxt_first
    return path, _concat(parts)


def _concat(parts: list[np.ndarray]) -> list[np.ndarray]:
    """Path parts -> coordinate blocks; a part ending in a NaN marker keeps its last vertex and the next part
    keeps its first one, otherwise the shared seam vertex is written once."""
    out = [parts[0][~np.isnan(parts[0]).any(axis=1)]]
    for prev, part in zip(parts, parts[1:], strict=False):
        kept = np.isnan(prev[-1]).any()
        clean = part[~np.isnan(part).any(axis=1)]
        out.append(clean if kept else clean[1:])
    return out


# ---------------------------------------------------------------- duplicates (chains)


def duplicate_pairs(lines: Sequence[LineString], max_m: float, min_frac: float, angle_deg: float
                    ) -> list[tuple[int, int]]:
    """(i, j), i < j, of chains where >= min_frac of the shorter lies within max_m of the longer."""
    if len(lines) < 2:
        return []
    tree = shapely.STRtree(list(lines))
    out = []
    for i, j in zip(*tree.query(list(lines), predicate="dwithin", distance=max_m), strict=True):
        i, j = int(i), int(j)
        if i >= j or axial_diff_deg(_chord_angle(lines[i]), _chord_angle(lines[j])) > angle_deg:
            continue
        short, long_ = (lines[i], lines[j]) if lines[i].length <= lines[j].length else (lines[j], lines[i])
        if short.length <= EPS_M:
            continue
        if short.intersection(long_.buffer(max_m)).length / short.length >= min_frac:
            out.append((i, j))
    return sorted(out)


def groups_of(pairs: Sequence[tuple[int, int]], n: int) -> list[tuple[int, ...]]:
    """Connected components (size >= 2) of the pair graph over 0..n-1, sorted."""
    root = list(range(n))

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            root[max(ra, rb)] = min(ra, rb)
    comps: dict[int, list[int]] = {}
    for k in range(n):
        comps.setdefault(find(k), []).append(k)
    return sorted(tuple(v) for v in comps.values() if len(v) > 1)
