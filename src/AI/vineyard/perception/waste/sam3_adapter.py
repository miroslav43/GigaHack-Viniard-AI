"""SAM 3 verifier adapter (A§4.9 step 3, design 03 W6). Torch-free at import: the transformers backend is
imported lazily by the default loader, so the pipeline workers never load torch.

- sources = config `waste.sam3.model_ids`, each preceded by its local mirror `<models_dir>/hf/<name>` when
  that directory exists; a source without transformers weights (facebook/sam3.1: native .pt only) is
  skipped with reason `no_transformers_weights`; access/IO/import errors are recorded per source.
- verify = every text prompt run separately on one crop (plus 0..max negative boxes, label 0, on the white
  tubes of the same crop); an instance counts only if its mask covers >= min_overlap_frac of the candidate
  box and is not a blob covering most of the crop; score = the best counted instance over all prompts.
- verify_with_budget = keep going while elapsed + last per-job cost <= budget; the rest stay None.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np

from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.waste.crops import CropWindow, box_to_crop
from vineyard.perception.waste.types import Candidate, Category, RejectReason

if TYPE_CHECKING:
    from vineyard.config import Sam3Config

XYXY = tuple[float, float, float, float]
WEIGHT_FILES: Final = frozenset({"model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"})
LOCAL_MIRROR_DIR: Final = "hf"
NEGATIVE_LABEL: Final = 0
NEGATIVE_REASONS: Final = frozenset(
    {
        RejectReason.NEAR_AXIS,
        RejectReason.PERIODIC_TUBE,
        RejectReason.TUBE_SHAPE,
        RejectReason.TOO_SMALL_WHITE,
    }
)
RGB_BANDS: Final = 3

_log = get_logger("perception.waste.sam3")


class Sam3Reason(StrEnum):
    OK = "ok"
    DISABLED = "disabled"
    NO_IDS = "no_model_ids"
    NO_WEIGHTS = "no_transformers_weights"
    LIST_FAILED = "list_failed"
    GATED = "gated"
    NOT_FOUND = "not_found"
    IMPORT_FAILED = "import_failed"
    LOAD_FAILED = "load_failed"


@dataclass(frozen=True)
class Sam3Params:
    enabled: bool
    sources: tuple[str, ...]
    device: str
    crop_px: int
    text_prompts: tuple[str, ...]
    prompt_categories: Mapping[str, Category]
    max_negative_boxes: int
    threshold: float
    mask_threshold: float
    min_overlap_frac: float
    max_mask_crop_frac: float
    allow_download: bool
    torch_threads: int  # 0 = all cores

    def __post_init__(self) -> None:
        missing = [t for t in self.text_prompts if t not in self.prompt_categories]
        if missing:
            raise ValueError(f"waste.sam3.prompt_categories has no category for prompts {missing}")
        if not 0.0 < self.max_mask_crop_frac <= 1.0:
            raise ValueError(f"max_mask_crop_frac must be in (0, 1], got {self.max_mask_crop_frac}")

    @classmethod
    def from_config(cls, cfg: Sam3Config, models_dir: Path) -> Sam3Params:
        return cls(
            enabled=cfg.enabled,
            sources=model_sources(cfg.model_ids, models_dir),
            device=cfg.device,
            crop_px=cfg.crop_px,
            text_prompts=tuple(cfg.text_prompts),
            prompt_categories=MappingProxyType({k: Category(v) for k, v in cfg.prompt_categories.items()}),
            max_negative_boxes=cfg.max_negative_boxes,
            threshold=cfg.threshold,
            mask_threshold=cfg.mask_threshold,
            min_overlap_frac=cfg.min_overlap_frac,
            max_mask_crop_frac=cfg.max_mask_crop_frac,
            allow_download=cfg.allow_download,
            torch_threads=cfg.torch_threads,
        )


@dataclass(frozen=True)
class Sam3Status:
    available: bool
    model_id: str | None
    reason: str
    tried: tuple[tuple[str, str], ...]
    device: str
    load_s: float


@dataclass(frozen=True)
class Instance:
    score: float
    mask: np.ndarray  # bool (H, W) in crop px


@dataclass(frozen=True)
class BackendOutput:
    per_prompt: tuple[tuple[Instance, ...], ...]  # same order as the prompts
    vision_s: float
    prompt_s: tuple[float, ...]


class Sam3Backend(Protocol):
    def detect(
        self, crop_rgb: np.ndarray, prompts: Sequence[str], negatives: Sequence[XYXY]
    ) -> BackendOutput: ...


FileLister = Callable[[str], Sequence[str]]
Sam3Loader = Callable[[str, Sam3Params], Sam3Backend]


@dataclass(frozen=True)
class PromptScore:
    prompt: str
    score: float  # best counted instance, 0.0 when none
    overlap: float  # of that instance with the candidate box
    n_instances: int
    n_counted: int
    seconds: float


@dataclass(frozen=True)
class SamResult:
    score: float
    prompt: str | None
    category: Category
    overlap: float
    per_prompt: tuple[PromptScore, ...]
    seconds: float
    n_negatives: int


@dataclass(frozen=True)
class SamJob:
    cand_key: str
    tile_id: str
    window: CropWindow
    cand_box: XYXY  # crop px
    negatives: tuple[XYXY, ...]  # crop px


# ------------------------------------------------------------------ sources and preflight


def model_sources(model_ids: Sequence[str], models_dir: Path) -> tuple[str, ...]:
    """Each id preceded by its local mirror models_dir/hf/<basename> when present; local dirs as given."""
    out: list[str] = []
    for model_id in model_ids:
        if Path(model_id).is_dir():
            out.append(str(Path(model_id)))
            continue
        mirror = Path(models_dir) / LOCAL_MIRROR_DIR / model_id.rsplit("/", 1)[-1]
        out.extend([str(mirror), model_id] if mirror.is_dir() else [model_id])
    return tuple(out)


def list_files(source: str) -> tuple[str, ...]:
    """File names of a local directory (no hidden entries) or of a hub repo (network)."""
    path = Path(source)
    if path.is_dir():
        return tuple(sorted(p.name for p in path.iterdir() if p.is_file() and not p.name.startswith(".")))
    from huggingface_hub import list_repo_files

    return tuple(list_repo_files(source))


def has_transformers_weights(files: Sequence[str]) -> bool:
    return any(Path(f).name in WEIGHT_FILES for f in files)


def _error_reason(exc: BaseException) -> Sam3Reason:
    name = type(exc).__name__
    if name == "GatedRepoError":
        return Sam3Reason.GATED
    if name in {"RepositoryNotFoundError", "RevisionNotFoundError"}:
        return Sam3Reason.NOT_FOUND
    return Sam3Reason.IMPORT_FAILED if isinstance(exc, ImportError) else Sam3Reason.LOAD_FAILED


def _describe(reason: Sam3Reason, exc: BaseException) -> str:
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{reason}: {type(exc).__name__}: {first_line}"


# Expected failure modes of a source (hub errors are OSError subclasses); programming errors propagate.
_BASE_ERRORS: Final[tuple[type[Exception], ...]] = (OSError, ImportError, ValueError, KeyError)


def _source_errors() -> tuple[type[Exception], ...]:
    """Base errors plus httpx transport errors (an offline hub listing raises httpx.ConnectError)."""
    try:
        import httpx
    except ImportError:
        return _BASE_ERRORS
    return (*_BASE_ERRORS, httpx.HTTPError)


def _try_source(
    source: str, p: Sam3Params, loader: Sam3Loader, lister: FileLister
) -> tuple[Sam3Backend | None, str]:
    errors = _source_errors()
    try:
        files = lister(source)
    except errors as exc:
        return None, _describe(Sam3Reason.LIST_FAILED, exc)
    if not has_transformers_weights(files):
        return None, f"{Sam3Reason.NO_WEIGHTS}: {len(files)} files, none of {sorted(WEIGHT_FILES)}"
    try:
        return loader(source, p), Sam3Reason.OK.value
    except errors as exc:
        return None, _describe(_error_reason(exc), exc)


def _default_loader(source: str, p: Sam3Params) -> Sam3Backend:
    from vineyard.perception.waste.sam3_backend import load_transformers_backend

    return load_transformers_backend(source, p)


def load_sam3(
    p: Sam3Params,
    loader: Sam3Loader | None = None,
    lister: FileLister | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Sam3Verifier | None, Sam3Status]:
    """First source that loads -> (verifier, status); none -> (None, status with every attempt)."""
    if not p.enabled:
        return None, Sam3Status(False, None, Sam3Reason.DISABLED, (), p.device, 0.0)
    if not p.sources:
        return None, Sam3Status(False, None, Sam3Reason.NO_IDS, (), p.device, 0.0)
    start, tried = clock(), []
    for source in p.sources:
        backend, outcome = _try_source(source, p, loader or _default_loader, lister or list_files)
        tried.append((source, outcome))
        if backend is not None:
            status = Sam3Status(True, source, Sam3Reason.OK, tuple(tried), p.device, clock() - start)
            log_event(
                _log, "waste.sam3.loaded", model_id=source, device=p.device, load_s=round(status.load_s, 2)
            )
            return Sam3Verifier(backend, p), status
    reason = "no SAM 3 model loaded: " + "; ".join(f"{s} -> {o}" for s, o in tried)
    log_event(_log, "waste.sam3.unavailable", level=logging.WARNING, reason=reason)
    return None, Sam3Status(False, None, reason, tuple(tried), p.device, 0.0)


# ------------------------------------------------------------------ scoring


def _box_slices(box: XYXY, h: int, w: int) -> tuple[slice, slice]:
    x0, y0 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x1, y1 = min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"candidate box {box} is outside the {w}x{h} crop")
    return slice(y0, y1), slice(x0, x1)


def mask_overlap(mask: np.ndarray, box: XYXY) -> float:
    """Fraction of the candidate box pixels covered by the mask."""
    rows, cols = _box_slices(box, mask.shape[0], mask.shape[1])
    return float(np.asarray(mask[rows, cols], dtype=bool).mean())


def box_prompt(
    negatives: Sequence[XYXY], max_n: int
) -> tuple[list[list[list[float]]], list[list[int]]] | None:
    """Processor `input_boxes` / `input_boxes_labels` (one image, label 0 = exclude), or None when empty."""
    boxes = [list(map(float, b)) for b in negatives[: max(0, max_n)]]
    return ([boxes], [[NEGATIVE_LABEL] * len(boxes)]) if boxes else None


def _prompt_score(
    prompt: str, instances: Sequence[Instance], box: XYXY, p: Sam3Params, secs: float
) -> PromptScore:
    best, best_overlap, counted = 0.0, 0.0, 0
    for inst in instances:
        mask = np.asarray(inst.mask, dtype=bool)
        overlap = mask_overlap(mask, box)
        if overlap < p.min_overlap_frac or mask.mean() > p.max_mask_crop_frac:
            continue
        counted += 1
        if inst.score > best:
            best, best_overlap = float(inst.score), overlap
    return PromptScore(prompt, best, best_overlap, len(instances), counted, secs)


def _check_crop(crop_rgb: np.ndarray) -> None:
    if crop_rgb.dtype != np.uint8 or crop_rgb.ndim != RGB_BANDS or crop_rgb.shape[2] != RGB_BANDS:
        raise ValueError(f"SAM crop must be uint8 (H, W, 3), got {crop_rgb.dtype} {crop_rgb.shape}")


@dataclass(frozen=True)
class Sam3Verifier:
    backend: Sam3Backend
    params: Sam3Params

    def verify(self, crop_rgb: np.ndarray, cand_box: XYXY, negatives: Sequence[XYXY]) -> SamResult:
        """Best counted instance over the text prompts (each run separately) for one candidate crop."""
        _check_crop(crop_rgb)
        _box_slices(cand_box, crop_rgb.shape[0], crop_rgb.shape[1])
        p = self.params
        negs = tuple(negatives[: p.max_negative_boxes])
        out = self.backend.detect(crop_rgb, p.text_prompts, negs)
        if len(out.per_prompt) != len(p.text_prompts) or len(out.prompt_s) != len(p.text_prompts):
            raise RuntimeError(
                f"SAM 3 backend returned {len(out.per_prompt)} results for {len(p.text_prompts)} prompts"
            )
        per = tuple(
            _prompt_score(t, inst, cand_box, p, s)
            for t, inst, s in zip(p.text_prompts, out.per_prompt, out.prompt_s, strict=True)
        )
        best = max(per, key=lambda ps: ps.score)  # first prompt wins ties
        hit = best.score > 0.0
        return SamResult(
            score=best.score,
            prompt=best.prompt if hit else None,
            category=p.prompt_categories[best.prompt] if hit else Category.UNKNOWN,
            overlap=best.overlap,
            per_prompt=per,
            seconds=out.vision_s + sum(out.prompt_s),
            n_negatives=len(negs),
        )


# ------------------------------------------------------------------ jobs and budget


def select_negative_boxes(
    cand: Candidate, pool: Sequence[Candidate], window: CropWindow, max_n: int
) -> tuple[XYXY, ...]:
    """Nearest rejected white tubes of the same tile whose centre is in the window, clipped, in crop px.
    Boxes touching the candidate box are left out (they would suppress the candidate itself)."""
    if max_n <= 0:
        return ()
    x0, y0, x1, y1 = window.as_xyxy()
    cx, cy = cand.centroid_px
    picked = sorted(
        (
            (math.hypot(t.centroid_px[0] - cx, t.centroid_px[1] - cy), t.cand_key, t)
            for t in pool
            if t.tile_id == cand.tile_id
            and t.is_white
            and t.reject_reason in NEGATIVE_REASONS
            and x0 <= t.box.centre[0] < x1
            and y0 <= t.box.centre[1] < y1
            and not t.box.polygon().intersects(cand.box.polygon())
        ),
        key=lambda item: (item[0], item[1]),
    )[:max_n]
    clipped = (
        (max(t.box.xtl, x0), max(t.box.ytl, y0), min(t.box.xbr, x1), min(t.box.ybr, y1)) for _, _, t in picked
    )
    return tuple(box_to_crop(b, window) for b in clipped)


def verify_with_budget(
    v: Sam3Verifier,
    jobs: Sequence[SamJob],
    crop_fn: Callable[[SamJob], np.ndarray],
    budget_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[SamResult | None, ...]:
    """Run jobs in order while elapsed + last per-job cost <= budget_s; the rest are None."""
    results: list[SamResult | None] = [None] * len(jobs)
    if budget_s <= 0.0:
        return tuple(results)
    start, last_cost = clock(), 0.0
    for i, job in enumerate(jobs):
        t0 = clock()
        if (t0 - start) + last_cost > budget_s:
            log_event(_log, "waste.sam3.budget_exhausted", done=i, left=len(jobs) - i, budget_s=budget_s)
            break
        results[i] = v.verify(crop_fn(job), job.cand_box, job.negatives)
        last_cost = clock() - t0
    return tuple(results)
