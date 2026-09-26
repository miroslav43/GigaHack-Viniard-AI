"""Assemble the waste verifier of a run (design 03 W3, arch §4.9): main process only, after the tile pool
has closed. The trained probe (models/<waste.probe.name>/<version>), the OpenCLIP embedder and SAM 3 are
each used only when enabled and available; whatever is missing lowers the degradation level (L0..L3) and
is named in the status reason. A probe that exists but fails its sha256 check, or was trained on another
CLIP model, is an error: silently ranking with the wrong weights would be worse than rule-only.
Torch-free at import: OpenCLIP / SAM 3 are imported by their loaders only when called.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

from vineyard.logging_setup import get_logger, log_event
from vineyard.perception.waste.clip_embed import ClipEmbedError, ClipParams, load_openclip
from vineyard.perception.waste.probe import (
    PROBE_FILE,
    ProbeError,
    ProbeModel,
    load_probe,
    probe_version_string,
    score_probe,
)
from vineyard.perception.waste.sam3_adapter import FileLister, Sam3Loader
from vineyard.perception.waste.verify import Verifier, VerifyStatus, load_verifier
from vineyard.perception.waste.verify_ml import ClipEmbedder, ProbeHandle, TileReader, build_verifier

if TYPE_CHECKING:
    from vineyard.config import AppConfig, ProbeConfig

ClipLoader = Callable[["ProbeConfig"], ClipEmbedder]
ProbeLoader = Callable[[Path, str, str], ProbeModel]
REASON_SEP: Final = "; "

_log = get_logger("perception.waste.verify_setup")


def _default_clip_loader(cfg: ProbeConfig) -> ClipEmbedder:
    return load_openclip(ClipParams.from_config(cfg))


def probe_handle(m: ProbeModel) -> ProbeHandle:
    """The verify_ml view of a trained probe (auto threshold = max(auto_probe_min, tau*) from its card)."""
    return ProbeHandle(
        score=partial(score_probe, m),
        auto_threshold=m.auto_threshold,
        auto_allowed=m.auto_enabled,
        version=probe_version_string(m),
    )


def find_probe(cfg: ProbeConfig, models_dir: Path, loader: ProbeLoader = load_probe) -> ProbeModel | None:
    """The configured probe, None when its weights file does not exist (sha256 errors propagate)."""
    path = Path(models_dir) / cfg.name / cfg.version / PROBE_FILE
    if not path.is_file():
        log_event(_log, "waste.probe_missing", level=logging.WARNING, path=str(path),
                  hint="vineyard waste probe-train")
        return None
    return loader(Path(models_dir), cfg.name, cfg.version)


def load_clip(cfg: ProbeConfig, loader: ClipLoader = _default_clip_loader) -> tuple[ClipEmbedder | None, str]:
    """(embedder, "") or (None, reason) when OpenCLIP cannot be loaded (offline without cached weights)."""
    try:
        return loader(cfg), ""
    except ClipEmbedError as exc:
        log_event(_log, "waste.clip_unavailable", level=logging.WARNING, error=str(exc))
        return None, f"clip unavailable: {exc}"


def _check_embed_model(probe: ProbeModel, embedder: ClipEmbedder) -> None:
    model_id = getattr(embedder, "model_id", None)
    if model_id is not None and model_id != probe.embed_model:
        raise ProbeError("probe was trained on another CLIP model", probe=probe_version_string(probe),
                         probe_embed_model=probe.embed_model, clip_model=model_id)


def setup_verifier(
    cfg: AppConfig,
    read_tile: TileReader,
    *,
    clip_loader: ClipLoader = _default_clip_loader,
    probe_loader: ProbeLoader = load_probe,
    sam_loader: Sam3Loader | None = None,
    sam_lister: FileLister | None = None,
) -> tuple[Verifier, VerifyStatus]:
    """The verifier and its status: rule-only when probe and SAM 3 are both disabled by config."""
    w = cfg.waste
    if not w.probe.enabled and not w.sam3.enabled:
        return load_verifier(w, cfg.grid.gsd_m)
    models_dir = cfg.paths.models_dir
    notes: list[str] = []
    probe: ProbeModel | None = None
    embedder: ClipEmbedder | None = None
    if w.probe.enabled:
        probe = find_probe(w.probe, models_dir, probe_loader)
        if probe is None:
            notes.append(f"probe {w.probe.name}@{w.probe.version} missing")
        embedder, clip_note = load_clip(w.probe, clip_loader)
        notes.extend([clip_note] if clip_note else [])
        if probe is not None and embedder is None:
            notes.append("probe unusable without CLIP")
            probe = None
        if probe is not None and embedder is not None:
            _check_embed_model(probe, embedder)
    handle = None if probe is None else probe_handle(probe)
    verifier, status, _ = build_verifier(
        w, cfg.grid.gsd_m, models_dir, read_tile, embedder=embedder, probe=handle,
        sam_loader=sam_loader, sam_lister=sam_lister,
    )
    reason = REASON_SEP.join([status.reason, *notes]) if notes else status.reason
    return verifier, replace(status, reason=reason)
