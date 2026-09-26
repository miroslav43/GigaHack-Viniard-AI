"""Main-process batched NN inference (contract §6, design 03 N6). Imports torch: never import it from a
pipeline worker; the nn_infer stage loads this module lazily, only when nn.enabled and weights exist.

Tile (2048², nodata -> 0) --INTER_AREA--> 1024² uint8 -> U-Net -> sigmoid -> p · valid₁₀₂₄ -> uint8 PNG.
The whole 1024² tile goes through the network in one pass (divisible by 32), so there are no seams.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import torch

from vineyard.geo.raster import read_mask_png, read_tile
from vineyard.logging_setup import get_logger, log_event, log_failure
from vineyard.nn.model import build_unet_from, images_to_tensor
from vineyard.nn.probs import write_prob_png
from vineyard.nn.validate import network_input
from vineyard.nn.weights import LoadedWeights
from vineyard.pipeline.cache import write_key

VALID_MIN_FRAC: Final = 0.5  # an NN pixel is data when most of its 2x2 tile pixels are
READ_THREADS: Final = 4  # GDAL JPEG decoding releases the GIL; tiles are read ahead of the GPU
NOT_IMPLEMENTED_MARKERS: Final = ("not implemented", "not currently implemented")
DEVICES: Final = ("mps", "cuda", "cpu")
EVENT_CPU_RETRY: Final = "nn.infer.cpu_retry"
EVENT_TILE_FAILED: Final = "nn.infer.tile_failed"
EVENT_DEVICE: Final = "nn.device"

ModelFactory = Callable[[str, int, str | None], torch.nn.Module]
ReadRgb = Callable[[Path], np.ndarray]

_log = get_logger("nn.infer")

__all__ = ["InferJob", "InferOutcome", "infer_tiles", "load_model", "predict_batch", "predict_tiles",
           "select_device", "state_dict_bytes", "sync", "to_input", "valid_to_net"]


@dataclass(frozen=True)
class InferJob:
    tile_id: str
    tif_path: Path
    valid_path: Path
    out_path: Path
    key: str


@dataclass(frozen=True)
class InferOutcome:
    tile_id: str
    ok: bool
    path: Path | None
    mean_prob: float
    error: str | None = None


@dataclass(frozen=True)
class _Loaded:
    job: InferJob
    net: np.ndarray | None
    valid: np.ndarray | None
    error: str | None


# ------------------------------------------------------------------ device and model


def select_device(preferred: str, fallback: str) -> torch.device:
    """`preferred` when available, else `fallback` (logged)."""
    for name in (preferred, fallback):
        if name not in DEVICES:
            raise ValueError(f"unknown torch device {name!r}; expected one of {DEVICES}")
    available = {"mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(), "cpu": True}
    chosen = preferred if available[preferred] else fallback
    log_event(_log, EVENT_DEVICE, preferred=preferred, chosen=chosen)
    return torch.device(chosen)


def sync(device: torch.device) -> None:
    """Wait for queued GPU work, so wall-clock timings are honest."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def default_model_factory(encoder: str, out_channels: int, encoder_weights: str | None) -> torch.nn.Module:
    """nn.model.VineUnet: ImageNet normalisation inside, input RGB in [0, 1]."""
    return build_unet_from(encoder, out_channels, encoder_weights)


def _network(model: torch.nn.Module) -> torch.nn.Module:
    """The segmentation network inside nn.model.VineUnet (its state_dict is the released one)."""
    inner = getattr(model, "net", None)
    return inner if isinstance(inner, torch.nn.Module) else model


def state_dict_bytes(model: torch.nn.Module) -> bytes:
    """Serialised plain smp.Unet state_dict (CPU tensors, no wrapper prefix) for weights.save_weights."""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in _network(model).state_dict().items()}, buf)
    return buf.getvalue()


def load_model(weights: LoadedWeights, device: torch.device, *, factory: ModelFactory | None = None) -> torch.nn.Module:
    """U-Net with `encoder_weights=None` (offline) and our verified state_dict (strict), eval mode on `device`."""
    make = factory or default_model_factory
    model = make(weights.card.encoder, len(weights.card.classes), None)
    state = torch.load(weights.weights_path, map_location="cpu", weights_only=True)
    _network(model).load_state_dict(state, strict=True)
    return model.eval().to(device)


# ------------------------------------------------------------------ prediction


def to_input(batch_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 in [0, 1] on `device` (the model normalises itself)."""
    return images_to_tensor(batch_u8).to(device)


def _forward(model: torch.nn.Module, batch_u8: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        logits = model(to_input(batch_u8, device))
        return torch.sigmoid(logits[:, 0]).float().cpu().numpy()


def _is_not_implemented(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in NOT_IMPLEMENTED_MARKERS)


def predict_batch(model: torch.nn.Module, batch_u8: np.ndarray, device: torch.device
                  ) -> tuple[np.ndarray, torch.device]:
    """float32 (N, H, W) probabilities and the device that produced them. An operator missing on the GPU
    moves the model to the CPU once (logged with the error) and the batch is retried there."""
    model.eval()
    try:
        return _forward(model, batch_u8, device), device
    except RuntimeError as exc:
        if device.type == "cpu" or not _is_not_implemented(exc):
            raise
        log_failure(_log, EVENT_CPU_RETRY, exc, device=device.type)
        cpu = torch.device("cpu")
        model.to(cpu)
        return _forward(model, batch_u8, cpu), cpu


def predict_tiles(model: torch.nn.Module, inputs: Sequence[np.ndarray], device: torch.device,
                  batch_size: int) -> list[np.ndarray]:
    """Probabilities of equally-shaped uint8 inputs, in order, `batch_size` at a time."""
    out: list[np.ndarray] = []
    for start in range(0, len(inputs), batch_size):
        probs, device = predict_batch(model, np.stack(inputs[start : start + batch_size]), device)
        out.extend(probs)
    return out


def valid_to_net(valid: np.ndarray, factor: int) -> np.ndarray:
    """Bool NN-resolution mask: at least VALID_MIN_FRAC of the covered tile pixels are data."""
    size = (valid.shape[1] // factor, valid.shape[0] // factor)
    return cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= VALID_MIN_FRAC


# ------------------------------------------------------------------ tiles -> PNG


def _load(job: InferJob, factor: int, read_rgb: ReadRgb) -> _Loaded:
    try:
        valid = read_mask_png(job.valid_path)
        net = network_input(read_rgb(job.tif_path), valid, factor)
        return _Loaded(job, net, valid_to_net(valid, factor), None)
    except Exception as exc:  # noqa: BLE001  (one unreadable tile fails alone; reported in its outcome)
        log_failure(_log, EVENT_TILE_FAILED, exc, tile_id=job.tile_id, path=str(job.tif_path))
        return _Loaded(job, None, None, f"{type(exc).__name__}: {exc}")


def _prefetched(jobs: Sequence[InferJob], batch_size: int, factor: int, read_rgb: ReadRgb
                ) -> Iterator[list[_Loaded]]:
    """Batches of loaded tiles; the next batch is read while the current one is on the GPU."""
    chunks = [jobs[i : i + batch_size] for i in range(0, len(jobs), batch_size)]
    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        pending: list[Future[_Loaded]] = [pool.submit(_load, j, factor, read_rgb) for j in chunks[0]] if chunks else []
        for k in range(len(chunks)):
            current = [f.result() for f in pending]
            nxt = chunks[k + 1] if k + 1 < len(chunks) else []
            pending = [pool.submit(_load, j, factor, read_rgb) for j in nxt]
            yield current


def _write(item: _Loaded, prob: np.ndarray) -> InferOutcome:
    final = np.where(item.valid, prob, np.float32(0.0)).astype(np.float32)
    write_prob_png(item.job.out_path, final)
    write_key(item.job.out_path, item.job.key)
    return InferOutcome(item.job.tile_id, True, item.job.out_path, float(final.mean()))


def infer_tiles(jobs: Sequence[InferJob], model: torch.nn.Module, device: torch.device, *, batch_size: int,
                factor: int, read_rgb: ReadRgb = read_tile) -> tuple[InferOutcome, ...]:
    """Write each job's canopy_prob PNG then its .key; outcomes keep the job order."""
    outcomes: dict[str, InferOutcome] = {}
    for batch in _prefetched(jobs, batch_size, factor, read_rgb):
        good = [b for b in batch if b.error is None]
        outcomes.update({b.job.tile_id: InferOutcome(b.job.tile_id, False, None, 0.0, b.error)
                         for b in batch if b.error is not None})
        if not good:
            continue
        probs, device = predict_batch(model, np.stack([b.net for b in good]), device)
        outcomes.update({b.job.tile_id: _write(b, p) for b, p in zip(good, probs, strict=True)})
    return tuple(outcomes[j.tile_id] for j in jobs)


def timed_infer(jobs: Sequence[InferJob], model: torch.nn.Module, device: torch.device, *, batch_size: int,
                factor: int, read_rgb: ReadRgb = read_tile) -> tuple[tuple[InferOutcome, ...], float]:
    """infer_tiles plus the synced wall time in seconds."""
    sync(device)
    t0 = time.perf_counter()
    outcomes = infer_tiles(jobs, model, device, batch_size=batch_size, factor=factor, read_rgb=read_rgb)
    sync(device)
    return outcomes, time.perf_counter() - t0
