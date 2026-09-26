"""Probe negatives from Sireț3 tiles (A§4.9 step 3, design 03 W4).

- Example tiles (0 waste by definition): EVERY rule candidate (white tubes, stakes, stones, soil highlights,
  filter survivors included) plus `random_per_example_tile` random candidate-sized boxes.
- Other tiles of a run (waste unknown): only WHITE candidates rejected by the rule filters for a reason in
  `other_reasons` (tubes near the axis, too-small white blobs, pale soil, tube shapes), capped per tile and
  stratified by reason, plus a few random boxes whose crop windows avoid every kept candidate. Vivid
  candidates of these tiles are never used: a real bottle can be rejected as `tube_shape`.
Every crop uses the pipeline's probe crop policy (crops.crop_window + extract_crop), so negatives look
exactly like the candidates the probe scores at run time. CV group = tile id.
`background_sampler` hands free windows of non-example tiles to the pasted positives.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd

from vineyard.errors import VineyardError
from vineyard.perception.waste.crop_store import CropRecord, write_crop_store
from vineyard.perception.waste.crops import crop_window, extract_crop
from vineyard.perception.waste.types import TILE_EXTENT_PX, BoxPx, Candidate, RejectReason

if TYPE_CHECKING:
    from vineyard.config import AppConfig

SOURCE_EXAMPLE: Final = "example_candidate"
SOURCE_TILE: Final = "tile_candidate"
SOURCE_RANDOM: Final = "random"
KEPT: Final = "kept"
NEGATIVE: Final = 0
TILE_STATUS_FILE: Final = "layers/tile_status.parquet"
CANDIDATE_CACHE: Final = Path("cache") / "waste"  # pipeline.stages.waste tile cache
TILES_DIR: Final = "tiles"
CACHE_DIR: Final = "cache"
TILE_EXT: Final = ".tif"
_BACKGROUND_STREAM: Final = 7919  # RNG stream id: paste backgrounds never replay the negatives' boxes


class NegativeError(VineyardError):
    """Negative crop planning or extraction failure."""


@dataclass(frozen=True)
class NegativeParams:
    example_tiles: tuple[str, ...]
    other_reasons: frozenset[str]
    max_per_other_tile: int
    random_per_example_tile: int
    random_per_other_tile: int
    random_box_px: tuple[float, float]
    max_random_attempts: int
    context: float
    min_px: int
    out_px: int
    extent_px: int
    seed: int
    background_tiles: int  # non-example tiles supplying paste backgrounds (waste.probe.positives)

    @classmethod
    def from_config(cls, cfg: AppConfig) -> NegativeParams:
        probe = cfg.waste.probe
        neg = probe.negatives
        lo, hi = neg.random_box_px
        return cls(
            example_tiles=tuple(cfg.eval.example_tiles),
            other_reasons=_reject_reasons(neg.other_reasons),
            max_per_other_tile=neg.max_per_other_tile,
            random_per_example_tile=neg.random_per_tile,
            random_per_other_tile=neg.random_per_other_tile,
            random_box_px=(float(lo), float(hi)),
            max_random_attempts=neg.max_random_attempts,
            context=probe.crop_context,
            min_px=probe.crop_min_px,
            out_px=probe.crop_resize_px,
            extent_px=int(TILE_EXTENT_PX),
            seed=cfg.runtime.seed,
            background_tiles=probe.positives.background_tiles,
        )


def _reject_reasons(names: Sequence[str]) -> frozenset[str]:
    """Validated reject-reason names (waste.probe.negatives.other_reasons)."""
    known = {r.value for r in RejectReason}
    unknown = sorted(set(names) - known)
    if unknown:
        raise NegativeError("unknown reject reasons in waste.probe.negatives.other_reasons", unknown=unknown,
                            known=sorted(known))
    return frozenset(names)


@dataclass(frozen=True)
class NegativeSpec:
    tile_id: str
    key: str
    source: str
    reason: str
    survives: bool
    is_example: bool
    box: BoxPx

    def record(self) -> CropRecord:
        meta = {
            "tile_id": self.tile_id,
            "reason": self.reason,
            "survives": self.survives,
            "is_example": self.is_example,
            "box": list(self.box.as_tuple()),
        }
        return CropRecord(key=self.key, source=self.source, group=self.tile_id, label=NEGATIVE, meta=meta)


def tile_rng(seed: int, tile_id: str) -> np.random.Generator:
    """Deterministic per-tile generator (independent of the tile order)."""
    return np.random.default_rng((seed, zlib.crc32(tile_id.encode("utf-8"))))


# ------------------------------------------------------------------ planning


def _spec(c: Candidate, source: str, is_example: bool) -> NegativeSpec:
    reason = KEPT if c.reject_reason is None else c.reject_reason.value
    return NegativeSpec(
        c.tile_id, f"neg:{c.cand_key}", source, reason, is_example and not c.rejected, is_example, c.box
    )


def _stratified(
    by_reason: Mapping[str, list[Candidate]], cap: int, rng: np.random.Generator
) -> list[Candidate]:
    """Round-robin over reasons (sorted), random order inside each reason, at most `cap` in total."""
    queues = {r: [cs[i] for i in rng.permutation(len(cs))] for r, cs in sorted(by_reason.items())}
    picked: list[Candidate] = []
    while len(picked) < cap and any(queues.values()):
        for r in sorted(queues):
            if queues[r] and len(picked) < cap:
                picked.append(queues[r].pop(0))
    return picked


def candidate_negatives(
    cands: Sequence[Candidate], tile_id: str, p: NegativeParams, rng: np.random.Generator
) -> tuple[NegativeSpec, ...]:
    foreign = sorted({c.tile_id for c in cands} - {tile_id})
    if foreign:
        raise NegativeError("candidates from another tile", tile_id=tile_id, foreign=foreign)
    if tile_id in p.example_tiles:
        return tuple(_spec(c, SOURCE_EXAMPLE, True) for c in cands)
    by_reason: dict[str, list[Candidate]] = {}
    for c in cands:
        if c.is_white and c.reject_reason is not None and c.reject_reason.value in p.other_reasons:
            by_reason.setdefault(c.reject_reason.value, []).append(c)
    picked = _stratified(by_reason, p.max_per_other_tile, rng)
    return tuple(_spec(c, SOURCE_TILE, False) for c in picked)


def _random_box(rng: np.random.Generator, p: NegativeParams) -> BoxPx:
    lo, hi = np.log(p.random_box_px[0]), np.log(p.random_box_px[1])
    w, h = np.exp(rng.uniform(lo, hi, size=2))
    x0 = rng.uniform(0.0, p.extent_px - w)
    y0 = rng.uniform(0.0, p.extent_px - h)
    return BoxPx(float(x0), float(y0), float(x0 + w), float(y0 + h))


def _free(valid: np.ndarray, avoid: Sequence[BoxPx], x0: int, y0: int, side: int) -> bool:
    """Window [x0, x0 + side)^2 fully valid and disjoint from every `avoid` box."""
    if not valid[y0 : y0 + side, x0 : x0 + side].all():
        return False
    x1, y1 = x0 + side, y0 + side
    return not any(a.xtl < x1 and x0 < a.xbr and a.ytl < y1 and y0 < a.ybr for a in avoid)


def _window_ok(box: BoxPx, valid: np.ndarray, avoid: Sequence[BoxPx], p: NegativeParams) -> bool:
    w = crop_window(box, p.context, p.min_px, p.extent_px)
    return _free(valid, avoid, w.x0, w.y0, w.side)


def random_negatives(
    valid: np.ndarray,
    tile_id: str,
    n: int,
    avoid: Sequence[BoxPx],
    p: NegativeParams,
    rng: np.random.Generator,
) -> tuple[NegativeSpec, ...]:
    """Up to n random candidate-sized boxes whose crop window is fully valid and avoids `avoid`."""
    if valid.shape != (p.extent_px, p.extent_px):
        raise NegativeError(
            "valid mask shape does not match the tile extent", tile_id=tile_id, shape=valid.shape
        )
    boxes: list[BoxPx] = []
    for _ in range(p.max_random_attempts):
        if len(boxes) >= n:
            break
        box = _random_box(rng, p)
        if _window_ok(box, valid, avoid, p):
            boxes.append(box)
    is_example = tile_id in p.example_tiles
    return tuple(
        NegativeSpec(tile_id, f"rand:{tile_id}@{i:04d}", SOURCE_RANDOM, SOURCE_RANDOM, False, is_example, b)
        for i, b in enumerate(boxes)
    )


def plan_tile(
    tile_id: str, cands: Sequence[Candidate], valid: np.ndarray, p: NegativeParams
) -> tuple[NegativeSpec, ...]:
    rng = tile_rng(p.seed, tile_id)
    is_example = tile_id in p.example_tiles
    n_random = p.random_per_example_tile if is_example else p.random_per_other_tile
    avoid = () if is_example else tuple(c.box for c in cands if not c.rejected)
    return (
        *candidate_negatives(cands, tile_id, p, rng),
        *random_negatives(valid, tile_id, n_random, avoid, p, rng),
    )


# ------------------------------------------------------------------ extraction and store


@dataclass(frozen=True)
class TileSources:
    load_candidates: Callable[[str], Sequence[Candidate]]
    load_rgb: Callable[[str], np.ndarray]
    load_valid: Callable[[str], np.ndarray]


def work_sources(work_dir: Path) -> TileSources:
    """Readers over a work dir: cache/waste/<t>.parquet, tiles/<t>.tif, cache/valid/<t>.png."""
    from vineyard.geo.raster import read_tile
    from vineyard.perception.waste.types import candidate_from_record
    from vineyard.pipeline.tile_cache import load_valid_mask

    work = Path(work_dir)

    def candidates(tile_id: str) -> tuple[Candidate, ...]:
        path = work / CANDIDATE_CACHE / f"{tile_id}.parquet"
        if not path.is_file():
            raise NegativeError(
                "waste candidate cache missing (run the waste stage)", tile_id=tile_id, path=str(path)
            )
        return tuple(candidate_from_record(r) for r in pd.read_parquet(path).to_dict("records"))

    return TileSources(
        load_candidates=candidates,
        load_rgb=lambda t: read_tile(work / TILES_DIR / f"{t}{TILE_EXT}"),
        load_valid=lambda t: load_valid_mask(work / CACHE_DIR, t),
    )


def run_tile_ids(run_dir: Path) -> tuple[str, ...]:
    """Sorted unique tile ids of a run (layers/tile_status.parquet)."""
    path = Path(run_dir) / TILE_STATUS_FILE
    if not path.is_file():
        raise NegativeError("run has no tile_status layer", path=str(path))
    return tuple(sorted(set(pd.read_parquet(path, columns=["tile_id"])["tile_id"].astype(str))))


def _crops(
    specs_by_tile: Mapping[str, Sequence[NegativeSpec]], sources: TileSources, p: NegativeParams
) -> Iterator[np.ndarray]:
    for tile_id, specs in specs_by_tile.items():
        if not specs:
            continue
        rgb = sources.load_rgb(tile_id)
        for s in specs:
            yield extract_crop(rgb, crop_window(s.box, p.context, p.min_px, p.extent_px), p.out_px)


def build_negative_store(
    tile_ids: Sequence[str], sources: TileSources, p: NegativeParams, out_dir: Path
) -> Path:
    """Plan every tile, then stream the crops tile by tile into a crop store."""
    if not tile_ids:
        raise NegativeError("no tiles to take negatives from", out_dir=str(out_dir))
    plan = {t: plan_tile(t, sources.load_candidates(t), sources.load_valid(t), p) for t in tile_ids}
    specs = [s for t in tile_ids for s in plan[t]]
    counts: dict[str, int] = {}
    for s in specs:
        counts[s.source] = counts.get(s.source, 0) + 1
    meta = {
        "kind": "negatives",
        "tiles": list(tile_ids),
        "example_tiles": list(p.example_tiles),
        "counts": counts,
        "params": {
            "max_per_other_tile": p.max_per_other_tile,
            "random_per_example_tile": p.random_per_example_tile,
            "random_per_other_tile": p.random_per_other_tile,
            "other_reasons": sorted(p.other_reasons),
            "context": p.context,
            "min_px": p.min_px,
            "seed": p.seed,
        },
        "licence": "Sireț3 orthophoto tiles, CC BY 4.0 (challenge data)",
    }
    return write_crop_store(out_dir, _crops(plan, sources, p), [s.record() for s in specs], p.out_px, meta)


def _spread(items: Sequence[str], n: int) -> list[str]:
    """n items evenly spaced over the sequence (all when n >= len)."""
    step = max(1, len(items) // max(n, 1))
    return list(items[::step][:n])


def background_sampler(
    tile_ids: Sequence[str], sources: TileSources, p: NegativeParams, n_tiles: int | None = None
) -> Callable[[int], np.ndarray]:
    """side -> random fully-valid (side, side, 3) window of a non-example tile, away from kept candidates
    (they may be real waste); backgrounds for the pasted positives (positives.BackgroundFn)."""
    others = _spread([t for t in tile_ids if t not in p.example_tiles], p.background_tiles if n_tiles is None else n_tiles)
    if not others:
        raise NegativeError("no non-example tiles for paste backgrounds", tiles=list(tile_ids))
    loaded = tuple(
        (
            sources.load_rgb(t),
            sources.load_valid(t),
            tuple(c.box for c in sources.load_candidates(t) if not c.rejected),
        )
        for t in others
    )
    rng = np.random.default_rng((p.seed, _BACKGROUND_STREAM))

    def sample(side: int) -> np.ndarray:
        for _ in range(p.max_random_attempts):
            rgb, valid, avoid = loaded[int(rng.integers(len(loaded)))]
            x0, y0 = (int(v) for v in rng.integers(0, p.extent_px - side + 1, size=2))
            if _free(valid, avoid, x0, y0, side):
                return rgb[y0 : y0 + side, x0 : x0 + side].copy()
        raise NegativeError("no free background window found", side=side, tiles=others)

    return sample
