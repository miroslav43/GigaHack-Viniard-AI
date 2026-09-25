"""Process environment set before numpy/cv2/GDAL load (arch §4.1). Imported first by `vineyard`."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

THREAD_ENV_DEFAULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "GDAL_NUM_THREADS": "1",
        "GDAL_CACHEMAX": "256",
    }
)
CONDA_SCRUB_VARS: Final[tuple[str, ...]] = ("GDAL_DATA", "PROJ_DATA", "PROJ_LIB")
CONDA_MARKERS: Final[tuple[str, ...]] = ("conda", "miniforge", "mambaforge")


@dataclass(frozen=True)
class EnvPlan:
    set_defaults: Mapping[str, str] = field(default_factory=dict)
    drop: tuple[str, ...] = ()


def _points_into_conda(value: str, conda_prefix: str | None) -> bool:
    lowered = value.lower()
    if conda_prefix and value.startswith(conda_prefix):
        return True
    return any(marker in lowered for marker in CONDA_MARKERS)


def plan_env(environ: Mapping[str, str]) -> EnvPlan:
    """Pure: which defaults to set and which conda-owned PROJ/GDAL variables to drop."""
    defaults = {k: v for k, v in THREAD_ENV_DEFAULTS.items() if k not in environ}
    prefix = environ.get("CONDA_PREFIX")
    drop = tuple(
        sorted(name for name in CONDA_SCRUB_VARS if name in environ and _points_into_conda(environ[name], prefix))
    )
    return EnvPlan(set_defaults=MappingProxyType(defaults), drop=drop)


def apply_env(environ: MutableMapping[str, str]) -> EnvPlan:
    """Apply `plan_env` to `environ` (the process environment is inherently mutable)."""
    plan = plan_env(environ)
    for name, value in plan.set_defaults.items():
        environ[name] = value
    for name in plan.drop:
        warnings.warn(
            f"{name}={environ[name]} points into conda and breaks the rasterio/pyproj wheels; ignored",
            RuntimeWarning,
            stacklevel=2,
        )
        del environ[name]
    return plan


STARTUP_PLAN: Final[EnvPlan] = apply_env(os.environ)
