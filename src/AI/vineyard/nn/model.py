"""U-Net (smp, resnet18 encoder, 1 output channel = canopy_prob logits) and state_dict helpers.

The network takes RGB scaled to [0, 1] (N, 3, H, W) and applies the ImageNet normalisation itself, so
training (PatchDataset) and inference (full 1024² tiles) feed the same tensors. The saved state_dict is
the plain smp.Unet one (no wrapper prefix): `smp.Unet("resnet18", encoder_weights=None, classes=1)`
plus ImageNet normalisation reproduces the model outside this package. Imports torch: never import
this module from the CPU pipeline workers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn

from vineyard.errors import VineyardError
from vineyard.pipeline.atomic import atomic_path

if TYPE_CHECKING:
    from vineyard.config import NnConfig

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)
U8_SCALE: Final = 255.0
RGB_CHANNELS: Final = 3


class ModelLoadError(VineyardError):
    """Weights file missing, unreadable or not matching the configured architecture."""


class VineUnet(nn.Module):
    """ImageNet normalisation + the segmentation network; forward returns logits."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, RGB_CHANNELS, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, RGB_CHANNELS, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def build_unet_from(encoder: str, out_channels: int, encoder_weights: str | None) -> VineUnet:
    """smp.Unet with raw logits; encoder_weights=None never touches the network."""
    net = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=RGB_CHANNELS,
                   classes=out_channels, activation=None)
    return VineUnet(net)


def build_unet(cfg: NnConfig, *, pretrained: bool) -> VineUnet:
    """nn.arch/encoder/out_channels; ImageNet encoder weights (nn.train.encoder_weights) only when pretrained."""
    weights = cfg.train.encoder_weights if pretrained else None
    return build_unet_from(cfg.encoder, cfg.out_channels, weights)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def save_state_dict(model: VineUnet, path: Path) -> Path:
    """Atomically write the network state_dict (CPU tensors) to `path`."""
    target = Path(path)
    state = {k: v.detach().to("cpu") for k, v in model.net.state_dict().items()}
    with atomic_path(target) as tmp:
        torch.save(state, tmp)
    return target


def load_state_dict_file(path: Path) -> dict[str, torch.Tensor]:
    """Tensors-only load (weights_only=True, map_location=cpu)."""
    source = Path(path)
    if not source.is_file():
        raise ModelLoadError("weights file missing", path=str(source))
    try:
        state = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for a corrupt file
        raise ModelLoadError(f"unreadable weights file: {type(exc).__name__}: {exc}", path=str(source)) from exc
    if not isinstance(state, dict):
        raise ModelLoadError("weights file does not hold a state_dict", path=str(source), type=type(state).__name__)
    return state


def load_model(cfg: NnConfig, path: Path, device: str | torch.device = "cpu") -> VineUnet:
    """Build the configured U-Net without downloading anything, load `path` strictly, eval mode on `device`."""
    state = load_state_dict_file(path)
    model = build_unet(cfg, pretrained=False)
    try:
        model.net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"state_dict does not match {cfg.arch}/{cfg.encoder}: {exc}", path=str(path)) from exc
    return model.to(device).eval()


def images_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) or (N, H, W, 3) -> float32 (N, 3, H, W) in [0, 1]."""
    if rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB, got {rgb.dtype}")
    batch = rgb[None] if rgb.ndim == RGB_CHANNELS else rgb
    if batch.ndim != 4 or batch.shape[-1] != RGB_CHANNELS:
        raise ValueError(f"expected RGB (H, W, 3) or (N, H, W, 3), got {rgb.shape}")
    chw = np.ascontiguousarray(batch.transpose(0, 3, 1, 2))
    return torch.from_numpy(chw).float().div_(U8_SCALE)


def predict_proba(model: VineUnet, x: torch.Tensor) -> torch.Tensor:
    """sigmoid(logits) in inference mode; the caller puts the model in eval() and on x's device."""
    with torch.inference_mode():
        return torch.sigmoid(model(x))
