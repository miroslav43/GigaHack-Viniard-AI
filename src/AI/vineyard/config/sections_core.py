"""Core sections: project, paths, grid, runtime, nodata, veg, logging (+ shared field types)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


class Section(BaseModel):
    """Base of every config model: frozen, unknown keys rejected, no defaults (YAML is the source)."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


def _ordered(pair: tuple[float, float]) -> tuple[float, float]:
    if pair[0] > pair[1]:
        raise ValueError(f"range must be ordered (low <= high), got {pair}")
    return pair


Frac = Annotated[float, Field(ge=0.0, le=1.0)]
PosFloat = Annotated[float, Field(gt=0.0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
PosInt = Annotated[int, Field(gt=0)]
NonNegInt = Annotated[int, Field(ge=0)]
Range = Annotated[tuple[float, float], AfterValidator(_ordered)]
IntRange = Annotated[tuple[int, int], AfterValidator(_ordered)]
FracRange = Annotated[tuple[Frac, Frac], AfterValidator(_ordered)]
TileIdTuple = tuple[str, ...]


class ProjectConfig(Section):
    name: str
    crs: Literal["EPSG:32635"]


class PathsConfig(Section):
    project_root: Path
    data_root: Path
    tiles_zip_glob: str
    route_dir: Path
    examples_dir: Path
    work_dir: Path
    models_dir: Path
    publish_dir: Path
    overrides: Path
    waste_confirmed: Path


class GridConfig(Section):
    gsd_m: PosFloat
    tile_px: PosInt
    tile_m: PosFloat
    origin_x: float
    origin_y: float
    expected_tiles: PosInt
    tiepoint_tol_m: PosFloat

    @model_validator(mode="after")
    def _tile_size_consistent(self) -> GridConfig:
        if not math.isclose(self.tile_m, self.gsd_m * self.tile_px, rel_tol=0.0, abs_tol=self.tiepoint_tol_m):
            raise ValueError(f"grid.tile_m={self.tile_m} != gsd_m*tile_px={self.gsd_m * self.tile_px}")
        return self


class RuntimeConfig(Section):
    n_workers: PosInt | Literal["auto"]
    workers_auto_reserve: NonNegInt
    workers_sweep: tuple[PosInt, ...]
    chunksize: PosInt
    maxtasksperchild: PosInt
    seed: int
    allow_failures: bool
    thread_env: dict[str, str]


class NodataConfig(Section):
    max_rgb: Annotated[int, Field(ge=0, le=255)]
    min_area_m2: NonNegFloat
    close_px: NonNegInt
    dilate_px: NonNegInt
    veg_erode_px: NonNegInt
    min_valid_frac: Frac
    approx_eps_px: NonNegFloat


class TextureFallbackConfig(Section):
    enabled: bool
    window_m: PosFloat
    top_frac: Frac
    trigger_min_veg_frac: Frac


class VegConfig(Section):
    method: Literal["lab_a"]
    blur_sigma_px: NonNegFloat
    threshold: float
    morph_close_px: NonNegInt
    otsu_fallback_enabled: bool
    otsu_fallback_veg_frac: FracRange
    shadow_v_max: Annotated[int, Field(ge=0, le=255)]
    overexposed_v_min: Annotated[int, Field(ge=0, le=255)]
    texture_fallback: TextureFallbackConfig


class LoggingConfig(Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    jsonl: bool
    console: Literal["rich", "plain"]
    tz: str
