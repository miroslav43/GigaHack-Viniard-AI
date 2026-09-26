"""nn.panels: RGB | a* | NN | diff JPEG panels and the automatic pick of hard tiles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from vineyard.nn.panels import (
    COLOUR_BOTH,
    COLOUR_NN_ONLY,
    COLOUR_VEG_ONLY,
    diff_image,
    disagreement,
    pick_hard_tiles,
    render_panel,
    write_jpeg,
)


def test_import_is_torch_free() -> None:
    code = "import sys, vineyard.nn.panels; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_disagreement() -> None:
    veg = np.array([[True, True, False, False]])
    prob = np.array([[0.9, 0.1, 0.9, 0.1]], np.float32)
    valid = np.array([[True, True, True, False]])
    assert disagreement(prob, veg, valid, 0.5) == pytest.approx(2 / 3)
    assert disagreement(prob, veg, np.zeros_like(valid), 0.5) == 0.0


def test_pick_hard_tiles() -> None:
    scores = {"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.5}
    assert pick_hard_tiles(scores, 2) == ("b", "d")
    assert pick_hard_tiles(scores, 2, exclude=("b",)) == ("d", "c")
    assert pick_hard_tiles(scores, 0) == ()


def test_diff_colours() -> None:
    veg = np.array([[True, True, False, False]])
    nn = np.array([[True, False, True, False]])
    img = diff_image(nn, veg)
    assert img[0, 0].tolist() == list(COLOUR_BOTH)
    assert img[0, 1].tolist() == list(COLOUR_VEG_ONLY)
    assert img[0, 2].tolist() == list(COLOUR_NN_ONLY)
    assert img[0, 3].tolist() == [0, 0, 0]


def test_render_and_write(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    na = rng.normal(size=(64, 64)).astype(np.float32)
    prob = rng.random((64, 64)).astype(np.float32)
    veg = prob > 0.3
    panel = render_panel(rgb, na, prob, veg, 0.5, panel_px=32)
    assert panel.shape[0] > 32 and panel.shape[1] == 4 * 32 and panel.dtype == np.uint8
    path = write_jpeg(tmp_path / "qa" / "nn_panels" / "t.jpg", panel)
    back = cv2.imread(str(path))
    assert back is not None and back.shape == panel.shape
    with pytest.raises(ValueError, match="shape"):
        render_panel(rgb, na[:10], prob, veg, 0.5, panel_px=32)
