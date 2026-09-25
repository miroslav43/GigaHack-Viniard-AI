"""Route validation (arch §4.11.6, design 04 §3.9, contract §5.1 and §10).

Two levels: a fast `covered_by(line, inner)` on the prepared domain, then the exact outside length summed
over individual segments (`difference(seg, inner, grid_size)`), so a dead end walked out and back counts
its outside part twice. There is deliberately NO `is_simple` check: corridor tours revisit dead ends.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.config import RouteConfig


@dataclass(frozen=True)
class ValidateParams:
    max_outside_frac: float
    closure_max_m: float
    zero_len_eps_m: float
    visit_radius_m: float
    grid_size_m: float

    @classmethod
    def from_route_cfg(cls, cfg: RouteConfig) -> ValidateParams:
        return cls(max_outside_frac=cfg.max_outside_frac_publish, closure_max_m=cfg.validate_.closure_max_m,
                   zero_len_eps_m=cfg.validate_.zero_len_eps_m, visit_radius_m=cfg.visit_radius_m,
                   grid_size_m=cfg.domain.grid_size_m)


@dataclass(frozen=True)
class RouteValidation:
    length_m: float
    closure_m: float
    outside_len_m: float
    outside_frac: float
    n_targets: int
    n_reachable: int
    n_visited_est: int
    coverage_est: float
    coverage_required: float
    legs_outside: tuple[tuple[int, float], ...]
    fast_covered: bool
    single_linestring: bool
    is_valid: bool
    zero_length_segments: int
    passed: bool
    failures: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        doc = asdict(self)
        doc["legs_outside"] = [{"seq": seq, "len_m": round(length, 3)} for seq, length in self.legs_outside]
        doc["failures"] = list(self.failures)
        return doc


def segments(line: LineString) -> np.ndarray:
    """Individual 2-point segments of `line` as a shapely array."""
    xy = shapely.get_coordinates(line)
    if len(xy) < 2:
        return np.empty(0, dtype=object)
    return shapely.linestrings(np.stack([xy[:-1], xy[1:]], axis=1))


def outside_lengths(line: LineString, inner: BaseGeometry, grid_size_m: float) -> np.ndarray:
    """Outside length of every segment of `line` against `inner` (exact, grid-snapped overlay)."""
    segs = segments(line)
    out = np.zeros(len(segs), dtype=np.float64)
    if not len(segs):
        return out
    shapely.prepare(inner)
    todo = np.flatnonzero(~shapely.covered_by(segs, inner))
    if len(todo):
        out[todo] = shapely.length(shapely.difference(segs[todo], inner, grid_size=grid_size_m))
    return out


def _legs_outside(per_seg: np.ndarray, anchor_idx: Sequence[int] | None) -> tuple[tuple[int, float], ...]:
    if anchor_idx is None or len(anchor_idx) < 2:
        return ()
    legs = []
    for k, (lo, hi) in enumerate(zip(anchor_idx[:-1], anchor_idx[1:], strict=True)):
        length = float(per_seg[lo:hi].sum())
        if length > 0.0:
            legs.append((k, length))
    return tuple(legs)


def covered_mask(line: BaseGeometry, xy: np.ndarray, radius_m: float) -> np.ndarray:
    """Which points of `xy` lie within `radius_m` of `line` (vectorised on the segment STRtree)."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if not len(pts) or line.is_empty:
        return np.zeros(len(pts), dtype=bool)
    parts = shapely.get_parts(line)
    segs = np.concatenate([segments(p) for p in parts if isinstance(p, LineString)] or [np.empty(0, object)])
    if not len(segs):
        return np.zeros(len(pts), dtype=bool)
    hits = shapely.STRtree(segs).query(shapely.points(pts), predicate="dwithin", distance=radius_m)
    mask = np.zeros(len(pts), dtype=bool)
    mask[np.unique(hits[0])] = True
    return mask


def _closure(line: BaseGeometry, start_xy: tuple[float, float]) -> float:
    xy = shapely.get_coordinates(line)
    if not len(xy):
        return float("inf")
    start = np.asarray(start_xy, dtype=np.float64)
    return float(max(np.hypot(*(xy[0] - start)), np.hypot(*(xy[-1] - start))))


def _coverage(line: BaseGeometry, required_xy: np.ndarray, all_xy: np.ndarray,
              radius_m: float) -> tuple[int, int, int, float, float]:
    req = covered_mask(line, required_xy, radius_m)
    every = covered_mask(line, all_xy, radius_m)
    cov_req = float(req.mean()) if len(req) else 1.0
    cov_all = float(every.mean()) if len(every) else 1.0
    return len(every), len(req), int(every.sum()), cov_all, cov_req


def _failures(v: dict[str, Any], params: ValidateParams) -> tuple[str, ...]:
    checks = (
        ("single_linestring", v["single_linestring"]),
        ("is_valid", v["is_valid"]),
        ("closure", v["closure_m"] <= params.closure_max_m),
        ("zero_length_segments", v["zero_length_segments"] == 0),
        ("outside_frac", v["outside_frac"] <= params.max_outside_frac),
        ("coverage", v["coverage_required"] >= 1.0),
    )
    return tuple(name for name, ok in checks if not ok)


def validate_route(geom: BaseGeometry, inner: BaseGeometry, *, start_xy: tuple[float, float],
                   required_xy: np.ndarray, all_xy: np.ndarray, params: ValidateParams,
                   anchor_idx: Sequence[int] | None = None) -> RouteValidation:
    """Every check of the route; `passed` only when no blocking check fails."""
    single = isinstance(geom, LineString) and not geom.is_empty
    per_seg = outside_lengths(geom, inner, params.grid_size_m) if single else np.zeros(0)
    seg_len = shapely.length(segments(geom)) if single else np.zeros(0)
    length = float(geom.length) if not geom.is_empty else 0.0
    outside = float(per_seg.sum())
    n_targets, n_req, n_visited, cov_all, cov_req = _coverage(geom, required_xy, all_xy, params.visit_radius_m)
    values: dict[str, Any] = {
        "length_m": length, "closure_m": _closure(geom, start_xy), "outside_len_m": outside,
        "outside_frac": outside / length if length > 0 else 1.0, "n_targets": n_targets,
        "n_reachable": n_req, "n_visited_est": n_visited, "coverage_est": cov_all, "coverage_required": cov_req,
        "legs_outside": _legs_outside(per_seg, anchor_idx),
        "fast_covered": bool(single and shapely.covered_by(geom, inner)), "single_linestring": single,
        "is_valid": bool(geom.is_valid), "zero_length_segments": int(np.sum(seg_len < params.zero_len_eps_m)),
    }
    failures = _failures(values, params)
    return RouteValidation(**values, passed=not failures, failures=failures)


__all__ = ["RouteValidation", "ValidateParams", "covered_mask", "outside_lengths", "segments", "validate_route"]
