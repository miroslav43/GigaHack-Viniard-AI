"""One demo job: the real perception stages (ingest ... assemble) on a single uploaded tile, fully isolated.

The job folder gets its own data root (a one-tile ZIP plus a link to the organizers' 02_route) and its own work dir,
so neither the main runs' cache nor their `LATEST_*` links are touched. Model only: the human/agent QA inputs
(overrides.yaml, row_seeds.csv, waste_confirmed.csv) are replaced by empty ones.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from vineyard.cli_options import CommonOptions, load_cli_config, run_pipeline, yaml_value
from vineyard.contracts.enums import Source
from vineyard.demo.status import StageProgress
from vineyard.demo.tile_input import TileInput
from vineyard.pipeline.context import RunContext
from vineyard.pipeline.registry import PRE_STAGES, select_stages

RUN_ID: Final = "demo"
LAST_STAGE: Final = "assemble"
DEMO_STAGES: Final = select_stages(PRE_STAGES, until=LAST_STAGE)
TILES_DIR: Final = "01_tiles"
TILES_ZIP: Final = "demo.zip"
EMPTY_OVERRIDES: Final = "version: 1\n"
EMPTY_CONFIRMATIONS: Final = "tile_id,xtl,ytl,xbr,ybr,decision,category,reviewer,note\n"
WORKERS: Final = 1


@dataclass(frozen=True)
class JobPaths:
    out_dir: Path
    data_root: Path
    work_dir: Path
    overrides: Path
    waste_confirmed: Path


def prepare_job(out_dir: Path, tile: TileInput, route_dir: Path) -> JobPaths:
    """Create the isolated data root and the empty QA inputs under `out_dir`."""
    job = JobPaths(out_dir, out_dir / "data", out_dir / "work", out_dir / "overrides.yaml",
                   out_dir / "waste_confirmed.csv")
    tiles = job.data_root / TILES_DIR
    tiles.mkdir(parents=True, exist_ok=True)
    job.work_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tiles / TILES_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.write(tile.path, arcname=f"{tile.tile_id}.tif")
    link = job.data_root / route_dir.name
    if not link.exists():
        link.symlink_to(route_dir.resolve(), target_is_directory=True)
    job.overrides.write_text(EMPTY_OVERRIDES, encoding="utf-8")
    job.waste_confirmed.write_text(EMPTY_CONFIRMATIONS, encoding="utf-8")
    return job


def isolation_sets(job: JobPaths) -> tuple[str, ...]:
    """--set values that point the pipeline at the job folder and switch the QA inputs off."""
    return (
        f"paths.data_root={yaml_value(str(job.data_root))}",
        f"paths.tiles_zip_glob={yaml_value(f'{TILES_DIR}/*.zip')}",
        "grid.expected_tiles=1",
        f"paths.work_dir={yaml_value(str(job.work_dir))}",
        f"paths.overrides={yaml_value(str(job.overrides))}",
        f"paths.waste_confirmed={yaml_value(str(job.waste_confirmed))}",
        "rows_seeded.enabled=false",
    )


def base_route_dir(base_sets: tuple[str, ...] = ()) -> Path:
    """The organizers' route inputs of the normal config (the job links to them)."""
    return Path(load_cli_config(CommonOptions(sets=base_sets)).paths.route_dir)


def run_job(job: JobPaths, base_sets: tuple[str, ...] = ()) -> tuple[RunContext, int]:
    """Run DEMO_STAGES in the job folder; status.json follows every stage.start. Returns (context, exit code)."""
    opts = CommonOptions(sets=(*base_sets, *isolation_sets(job)), run_id=RUN_ID, workers=WORKERS)
    with StageProgress(job.out_dir, DEMO_STAGES):
        return run_pipeline(opts, DEMO_STAGES, source=Source.MODEL)
