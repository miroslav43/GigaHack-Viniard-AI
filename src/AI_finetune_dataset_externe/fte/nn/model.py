"""Multi-head smp U-Net: RGB in [0, 1] -> K logit channels, ImageNet normalisation inside the module.

The saved state_dict is the plain smp one, so `smp.Unet(encoder, encoder_weights=None, classes=K)`
plus ImageNet normalisation reproduces the model outside this package.
"""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiHeadUnet(nn.Module):
    def __init__(self, encoder: str, n_heads: int, encoder_weights: str | None = "imagenet",
                 decoder_attention: str | None = "scse") -> None:
        super().__init__()
        self.encoder_name = encoder
        self.n_heads = n_heads
        self.net = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3,
                            classes=n_heads, activation=None, decoder_attention_type=decoder_attention)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def save_weights(model: MultiHeadUnet, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.net.state_dict(), tmp)
    tmp.replace(path)
    return path


def load_model(path: Path, encoder: str, n_heads: int, device: torch.device,
               decoder_attention: str | None = "scse") -> MultiHeadUnet:
    model = MultiHeadUnet(encoder, n_heads, encoder_weights=None, decoder_attention=decoder_attention)
    model.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()
