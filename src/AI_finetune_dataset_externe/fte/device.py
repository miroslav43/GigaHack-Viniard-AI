"""Torch device selection: MPS (fp32 only) on Apple silicon, CUDA elsewhere, CPU fallback."""

from __future__ import annotations

import os

import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
