"""Per-tile QA preview (JPEG overlay) and the block-coloured overview mosaic."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.qa import annset_factory as af
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.geo.raster import read_tile
from vineyard.geo.tiling import tile_ref
from vineyard.qa import overview, render

A = af.TILE_A


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config(environ={})


def _objects() -> render.TileObjects:
    rows = af.row_pieces([(A, "V01-R001", af.hline(A, 40.0)), (A, "V01-R002", af.hline(A, 20.0))])
    rows = rows.assign(row_structure=["regular", "disrupted"])
    ann = af.annset(
        [A], rows=rows,
        can=af.canopies([(A, "V01", af.rect(A, 2, 39.8, 4, 40.2))]),
        irs=af.interrow_pieces([(A, "V01-I001", af.rect(A, 0, 20.3, 51.2, 39.7))]),
        wst=af.waste([(A, "V01", af.rect(A, 30, 5, 31, 6))]),
    )
    return render.TileObjects.from_annset(ann, A)


def _header() -> render.PreviewHeader:
    return render.PreviewHeader(tile_id=A, status="ok", counts={"C": 1, "R": 2, "I": 1, "W": 1},
                                flags=("structure_borderline",), priority=2)


def _px(cfg: AppConfig, x_m: float, y_m: float) -> tuple[int, int]:
    scale = cfg.qa.preview_px / 51.2
    return int(round((51.2 - y_m) * scale)), int(round(x_m * scale))  # (row, col)


def test_preview_colours_and_size(cfg: AppConfig) -> None:
    rgb = np.full((2048, 2048, 3), 120, dtype=np.uint8)
    img = render.render_preview(rgb, tile_ref(A), _objects(), _header(), cfg.qa)
    assert img.shape == (cfg.qa.preview_px, cfg.qa.preview_px, 3) and img.dtype == np.uint8
    colors = cfg.qa.colors_bgr
    r, c = _px(cfg, 25.6, 40.0)
    assert tuple(img[r, c]) == tuple(colors["row"])
    r, c = _px(cfg, 25.6, 20.0)
    assert tuple(img[r, c]) == tuple(colors["disrupted"])
    r, c = _px(cfg, 3.0, 39.9)  # canopy fill is a blend towards green
    assert img[r, c][1] > 150 and img[r, c][2] < 100
    assert img[2, 2].sum() < 3 * 120  # header bar darkens the top


def test_preview_is_deterministic_and_encodes(cfg: AppConfig, tmp_path: Path) -> None:
    rgb = np.full((2048, 2048, 3), 90, dtype=np.uint8)
    a = render.render_preview(rgb, tile_ref(A), _objects(), _header(), cfg.qa)
    b = render.render_preview(rgb, tile_ref(A), _objects(), _header(), cfg.qa)
    assert np.array_equal(a, b)
    path = render.write_preview(tmp_path / "p" / f"{A}.jpg", a, cfg.qa.preview_jpeg_quality)
    back = cv2.imread(str(path))
    assert back.shape == a.shape


def test_empty_tile_preview(cfg: AppConfig) -> None:
    rgb = np.zeros((2048, 2048, 3), dtype=np.uint8)
    objs = render.TileObjects.from_annset(af.annset([A]), A)
    img = render.render_preview(rgb, tile_ref(A), objs, render.PreviewHeader(tile_id=A), cfg.qa)
    assert img.shape[:2] == (cfg.qa.preview_px, cfg.qa.preview_px)


def test_render_rejects_bad_rgb(cfg: AppConfig) -> None:
    with pytest.raises(ValueError):
        render.render_preview(np.zeros((10, 10), np.uint8), tile_ref(A), _objects(), _header(), cfg.qa)


@pytest.mark.examples
def test_reference_preview_under_one_second(examples_dir: Path, cfg: AppConfig) -> None:
    layers = af.reference_layers(examples_dir / "annotations.xml")
    ann = af.annset([A, af.TILE_B], can=layers["canopies"], rows=layers["row_pieces"], irs=layers["interrow_pieces"],
                    source=Source.REFERENCE)
    t0 = time.perf_counter()
    rgb = read_tile(examples_dir / "images" / f"{A}.tif")
    img = render.render_preview(rgb, tile_ref(A), render.TileObjects.from_annset(ann, A), _header(), cfg.qa)
    assert time.perf_counter() - t0 < 1.0
    assert img.shape == (cfg.qa.preview_px, cfg.qa.preview_px, 3)


# ------------------------------------------------------------------ overview


def test_overview_mosaic(cfg: AppConfig) -> None:
    t1, t2, t3 = "siret3_r002_c003", "siret3_r002_c004", "siret3_r004_c003"
    thumb = np.full((64, 64, 3), 200, dtype=np.uint8)
    img = overview.build_overview({t1: thumb, t2: thumb}, {t1: "V01", t2: "V02"}, cfg.qa, tile_ids=(t1, t2, t3))
    px = cfg.qa.overview_px_per_tile
    assert img.shape == (3 * px, 2 * px, 3)
    assert tuple(img[1, 1]) == overview.block_colour("V01")
    assert tuple(img[1, px + 1]) == overview.block_colour("V02")
    assert tuple(img[2 * px + px // 2, px // 2]) == overview.MISSING_BGR  # r004_c003 has no thumbnail
    assert overview.block_colour("V01") == overview.block_colour("V01")
    assert overview.block_colour("V01") != overview.block_colour("V02")


def test_overview_needs_tiles(cfg: AppConfig) -> None:
    with pytest.raises(ValueError):
        overview.build_overview({}, {}, cfg.qa)
