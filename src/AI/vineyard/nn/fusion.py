"""Canopy mask fusion variants A-E (classic veg mask × NN probability × row corridor), torch-free.

A = veg ∩ corridor (classic), B = (p ≥ t) ∩ corridor, C = veg ∧ (p ≥ t) ∩ corridor,
D = (veg ∨ p ≥ t) ∩ corridor, E = p ≥ t without corridor (ablation only).
Without a probability raster every variant falls back to A; the caller logs `nn.fallback_classic`.
A `None` corridor means no corridor restriction.
"""

from dataclasses import dataclass

import numpy as np

from vineyard.contracts.enums import FusionVariant
from vineyard.errors import SchemaError

__all__ = ["FusedMask", "fuse_masks"]


@dataclass(frozen=True)
class FusedMask:
    mask: np.ndarray  # bool, read-only
    variant_used: FusionVariant
    fell_back: bool


def _check_inputs(veg: np.ndarray, prob: np.ndarray | None, corridor: np.ndarray | None, threshold: float) -> None:
    if veg.dtype != np.bool_ or veg.ndim != 2:
        raise SchemaError("veg must be a 2-D bool mask", dtype=str(veg.dtype), shape=veg.shape)
    if prob is not None and prob.shape != veg.shape:
        raise SchemaError("prob shape differs from veg", prob=prob.shape, veg=veg.shape)
    if corridor is not None and corridor.shape != veg.shape:
        raise SchemaError("corridor shape differs from veg", corridor=corridor.shape, veg=veg.shape)
    if not 0.0 <= threshold <= 1.0:
        raise SchemaError("fusion threshold must be in [0, 1]", threshold=threshold)


def _combine(veg: np.ndarray, nn: np.ndarray | None, variant: FusionVariant) -> np.ndarray:
    if variant is FusionVariant.A or nn is None:
        return veg
    if variant in (FusionVariant.B, FusionVariant.E):
        return nn
    if variant is FusionVariant.C:
        return veg & nn
    return veg | nn


def fuse_masks(
    veg: np.ndarray,
    prob: np.ndarray | None,
    corridor: np.ndarray | None,
    variant: FusionVariant,
    threshold: float,
) -> FusedMask:
    """Fused bool mask for `variant`; inputs are never modified. Corridor may be bool or int labels (>0)."""
    _check_inputs(veg, prob, corridor, threshold)
    requested = FusionVariant(variant)
    used = requested if prob is not None else FusionVariant.A
    nn = None if prob is None else np.asarray(prob) >= threshold
    mask = _combine(veg, nn, used)
    if used is not FusionVariant.E and corridor is not None:
        mask = mask & (corridor if corridor.dtype == np.bool_ else corridor > 0)
    out = np.array(mask, dtype=bool, copy=True)
    out.setflags(write=False)
    return FusedMask(mask=out, variant_used=used, fell_back=used is not requested)
