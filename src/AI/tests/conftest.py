"""Shared fixtures and marker policy.

Markers: `needs_tiles` tests skip when the 5 challenge tile ZIPs are absent, `examples` tests skip
when the organizers' examples are absent. With VINEYARD_REQUIRE_TILES=1 both FAIL instead of
skipping, so gates never go green on a machine without data.
"""

from __future__ import annotations

import os
import zipfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from vineyard.geo.tiling import CRS_EPSG, GSD_M, tile_ref

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / ".." / ".." / "data & info"
ENV_DATA_ROOT = "VINEYARD_DATA_ROOT"
ENV_WORK_DIR = "VINEYARD_WORK_DIR"
ENV_REQUIRE_TILES = "VINEYARD_REQUIRE_TILES"
EXAMPLES_SUBDIR = Path("05_examples") / "siret3_examples_cvat"
TILES_SUBDIR = Path("01_tiles")
N_TILE_ZIPS = 5
TILE_ZIP_NAME = "siret3_challenge_tiles_part{k}of5.zip"
DEFAULT_FILL_RGB = (120, 110, 90)

MakeGeotiff = Callable[..., Path]
MakeZip = Callable[..., Path]


def resolve_data_root() -> Path:
    return Path(os.environ.get(ENV_DATA_ROOT, str(DEFAULT_DATA_ROOT))).resolve()


def tile_zip_paths(data_root: Path) -> list[Path]:
    return [data_root / TILES_SUBDIR / TILE_ZIP_NAME.format(k=k) for k in range(1, N_TILE_ZIPS + 1)]


def _require_data() -> bool:
    return os.environ.get(ENV_REQUIRE_TILES, "").strip() == "1"


def _skip_or_fail(reason: str) -> None:
    if _require_data():
        pytest.fail(f"{reason} ({ENV_REQUIRE_TILES}=1)", pytrace=False)
    pytest.skip(reason)


def pytest_runtest_setup(item: pytest.Item) -> None:
    root = resolve_data_root()
    if item.get_closest_marker("needs_tiles") is not None:
        missing = [p.name for p in tile_zip_paths(root) if not p.is_file()]
        if missing:
            _skip_or_fail(f"tile ZIPs missing under {root / TILES_SUBDIR}: {missing}")
    if item.get_closest_marker("examples") is not None and not (root / EXAMPLES_SUBDIR / "annotations.xml").is_file():
        _skip_or_fail(f"examples missing under {root / EXAMPLES_SUBDIR}")


# ------------------------------------------------------------------ paths


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_root() -> Path:
    return resolve_data_root()


@pytest.fixture(scope="session")
def examples_dir(data_root: Path) -> Path:
    path = data_root / EXAMPLES_SUBDIR
    if not path.is_dir():
        _skip_or_fail(f"examples missing under {path}")
    return path


@pytest.fixture(scope="session")
def examples_xml(examples_dir: Path) -> bytes:
    return (examples_dir / "annotations.xml").read_bytes()


@pytest.fixture(scope="session")
def example_tif(examples_dir: Path) -> Callable[[str], Path]:
    def _path(name: str) -> Path:
        file_name = name if name.endswith(".tif") else f"{name}.tif"
        path = examples_dir / "images" / file_name
        if not path.is_file():
            _skip_or_fail(f"example tile missing: {path}")
        return path

    return _path


@pytest.fixture
def tmp_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Empty work dir; VINEYARD_WORK_DIR points at it for the duration of the test."""
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv(ENV_WORK_DIR, str(work))
    yield work


# ------------------------------------------------------------------ factories


def _fill_image(size: int, bands: int) -> np.ndarray:
    img = np.empty((size, size, bands), np.uint8)
    img[...] = np.asarray(DEFAULT_FILL_RGB[:bands], np.uint8) if bands <= 3 else DEFAULT_FILL_RGB[0]
    return img


def write_geotiff(
    path: Path,
    *,
    tile_id: str | None = None,
    size: int = 64,
    rgb: np.ndarray | None = None,
    bands: int = 3,
    dx: float = 0.0,
    dy: float = 0.0,
    epsg: int = CRS_EPSG,
    gsd: float = GSD_M,
    compress: str = "JPEG",
) -> Path:
    """Small JPEG GeoTIFF georeferenced like the challenge tile `tile_id` (default: the file stem)."""
    t = tile_ref(tile_id or Path(path).stem)
    img = _fill_image(size, bands) if rgb is None else np.asarray(rgb, np.uint8)
    height, width, n_bands = img.shape
    profile = {
        "driver": "GTiff", "width": width, "height": height, "count": n_bands, "dtype": "uint8",
        "crs": f"EPSG:{epsg}", "transform": Affine(gsd, 0.0, t.x0 + dx, 0.0, -gsd, t.y0 + dy),
        "compress": compress, "tiled": True, "blockxsize": 16, "blockysize": 16,
    }
    if compress.upper() == "JPEG" and n_bands == 3:
        profile["photometric"] = "YCBCR"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(np.moveaxis(img, -1, 0))
    return Path(path)


@pytest.fixture
def make_geotiff() -> MakeGeotiff:
    return write_geotiff


def write_zip(
    path: Path, members: Mapping[str, Path | bytes], *, compression: int = zipfile.ZIP_STORED
) -> Path:
    """ZIP with `members` (arcname -> file path or raw bytes) in the given order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        for arcname, src in members.items():
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            zf.writestr(arcname, data)
    return Path(path)


@pytest.fixture
def make_zip() -> MakeZip:
    return write_zip
