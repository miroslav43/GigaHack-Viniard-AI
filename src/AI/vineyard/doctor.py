"""`vineyard doctor`: environment checks. torch is probed in a subprocess so this process stays torch-free."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final

from vineyard import _env
from vineyard.config import AppConfig

EXPECTED_PYTHON: Final = (3, 12)
REQUIRED_MODULES: Final = (
    "rasterio", "pyogrio", "pyproj", "geopandas", "shapely", "cv2", "skimage", "scipy", "lxml", "pyarrow",
)
EXPECTED_TILE_ZIPS: Final = 5
EXPECTED_EXAMPLE_TIFS: Final = 2
ROUTE_FILES: Final = ("start.geojson", "passages.geojson", "forbidden.geojson", "study_area.geojson")
MIN_FREE_DISK_GB: Final = 5.0
TORCH_PROBE_TIMEOUT_S: Final = 180.0
BYTES_PER_GB: Final = 1024**3
TORCH_PROBE_CODE: Final = "import torch; print(torch.__version__, torch.backends.mps.is_available())"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    required: bool
    detail: str


def check_python(version: Sequence[int] = tuple(sys.version_info[:3])) -> Check:
    ok = tuple(version[:2]) == EXPECTED_PYTHON
    return Check("python", ok, True, ".".join(str(v) for v in version))


def check_imports(
    modules: Sequence[str] = REQUIRED_MODULES, importer: Callable[[str], ModuleType] = importlib.import_module
) -> tuple[Check, ...]:
    checks = []
    for name in modules:
        try:
            module = importer(name)
        except ImportError as exc:
            checks.append(Check(f"import {name}", False, True, f"{type(exc).__name__}: {exc}"))
            continue
        checks.append(Check(f"import {name}", True, True, str(getattr(module, "__version__", "?"))))
    return tuple(checks)


def check_geo_versions() -> Check:
    try:
        import pyproj
        import rasterio
    except ImportError as exc:
        return Check("GDAL/PROJ", False, True, f"{type(exc).__name__}: {exc}")
    return Check("GDAL/PROJ", True, True, f"GDAL {rasterio.__gdal_version__}, PROJ {pyproj.proj_version_str}")


def probe_torch(timeout_s: float = TORCH_PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """(ok, detail) from a child interpreter importing torch."""
    try:
        result = subprocess.run([sys.executable, "-c", TORCH_PROBE_CODE], capture_output=True, text=True,
                                timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_s:.0f}s"
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        return False, lines[-1] if lines else f"exit {result.returncode}"
    return True, result.stdout.strip()


def check_torch(probe: Callable[[], tuple[bool, str]] = probe_torch) -> tuple[Check, ...]:
    ok, detail = probe()
    torch_check = Check("torch (extra nn)", ok, False, detail)
    if not ok:
        return (torch_check,)
    parts = detail.split()
    mps = len(parts) >= 2 and parts[1] == "True"
    return torch_check, Check("MPS", mps, False, "available" if mps else "not available (CPU fallback)")


def check_conda_env(environ: Mapping[str, str] = os.environ, plan: _env.EnvPlan = _env.STARTUP_PLAN) -> Check:
    leftover = _env.plan_env(environ).drop
    notes = [f"dropped at startup: {', '.join(plan.drop)}"] if plan.drop else []
    if environ.get("CONDA_PREFIX"):
        notes.append(f"CONDA_PREFIX={environ['CONDA_PREFIX']} (conda active)")
    if leftover:
        return Check("conda PROJ/GDAL vars", False, True, f"still set: {', '.join(leftover)}")
    return Check("conda PROJ/GDAL vars", True, False, "; ".join(notes) or "clean")


def _exists(name: str, path: Path, *, required: bool = True) -> Check:
    return Check(name, path.exists(), required, str(path))


def check_data(cfg: AppConfig) -> tuple[Check, ...]:
    paths = cfg.paths
    zips = sorted(paths.data_root.glob(paths.tiles_zip_glob)) if paths.data_root.is_dir() else []
    tifs = sorted((paths.examples_dir / "images").glob("*.tif")) if paths.examples_dir.is_dir() else []
    missing_route = [f for f in ROUTE_FILES if not (paths.route_dir / f).is_file()]
    return (
        _exists("data_root", paths.data_root),
        Check("tile ZIPs", len(zips) == EXPECTED_TILE_ZIPS, True, f"{len(zips)}/{EXPECTED_TILE_ZIPS} found"),
        _exists("examples annotations.xml", paths.examples_dir / "annotations.xml"),
        Check("example tifs", len(tifs) == EXPECTED_EXAMPLE_TIFS, True, f"{len(tifs)}/{EXPECTED_EXAMPLE_TIFS}"),
        Check("route inputs", not missing_route, True,
              f"missing: {', '.join(missing_route)}" if missing_route else str(paths.route_dir)),
    )


def _round_trip(frame: object, path: Path) -> object:
    import geopandas as gpd

    if path.suffix == ".parquet":
        frame.to_parquet(path)  # type: ignore[attr-defined]
        return gpd.read_parquet(path)
    frame.to_file(path, driver="GPKG")  # type: ignore[attr-defined]
    return gpd.read_file(path)


def check_vector_io() -> tuple[Check, ...]:
    import geopandas as gpd
    from shapely.geometry import Point

    frame = gpd.GeoDataFrame({"k": [1]}, geometry=[Point(629504.7, 5220250.75)], crs="EPSG:32635")
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, suffix, required in (("parquet round-trip", ".parquet", True), ("GPKG round-trip", ".gpkg", False)):
            try:
                back = _round_trip(frame, Path(tmp) / f"probe{suffix}")
            except (OSError, ValueError, RuntimeError) as exc:
                results.append(Check(name, False, required, f"{type(exc).__name__}: {exc}"))
                continue
            ok = back.crs == frame.crs and back.geometry.iloc[0].equals(frame.geometry.iloc[0])  # type: ignore[attr-defined]
            results.append(Check(name, bool(ok), required, "ok" if ok else "mismatch"))
    return tuple(results)


def check_disk(path: Path, min_free_gb: float = MIN_FREE_DISK_GB) -> Check:
    probe = next((p for p in (path, *path.parents) if p.exists()), Path("/"))
    free_gb = shutil.disk_usage(probe).free / BYTES_PER_GB
    return Check("free disk", free_gb >= min_free_gb, True, f"{free_gb:.1f} GB free at {probe} (min {min_free_gb})")


def run_checks(
    cfg: AppConfig, *, torch_probe: Callable[[], tuple[bool, str]] = probe_torch, min_free_gb: float = MIN_FREE_DISK_GB
) -> tuple[Check, ...]:
    return (
        check_python(),
        *check_imports(),
        check_geo_versions(),
        *check_torch(torch_probe),
        check_conda_env(),
        *check_data(cfg),
        *check_vector_io(),
        check_disk(cfg.paths.work_dir, min_free_gb),
    )


def exit_code(checks: Sequence[Check]) -> int:
    """1 if any required check failed, else 0 (optional failures are warnings)."""
    return 1 if any(c.required and not c.ok for c in checks) else 0


def render(checks: Sequence[Check]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="vineyard doctor")
    for column in ("verificare", "stare", "obligatoriu", "detalii"):
        table.add_column(column)
    for c in checks:
        state = "[green]OK[/green]" if c.ok else ("[red]EȘUAT[/red]" if c.required else "[yellow]AVERTISMENT[/yellow]")
        table.add_row(c.name, state, "da" if c.required else "nu", c.detail)
    Console().print(table)
