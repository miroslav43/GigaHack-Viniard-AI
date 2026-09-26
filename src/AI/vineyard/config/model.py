"""Root `AppConfig`: every section, in the plan's order (X3). `import` is aliased to `import_`."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from vineyard.config.sections_core import (
    GridConfig,
    LoggingConfig,
    NodataConfig,
    PathsConfig,
    ProjectConfig,
    RuntimeConfig,
    Section,
    VegConfig,
)
from vineyard.config.sections_io import EvalConfig, ExportConfig, ImportConfig
from vineyard.config.sections_nn import NnConfig
from vineyard.config.sections_perception import (
    BlocksConfig,
    CanopyConfig,
    InterrowConfig,
    OrchardConfig,
    QaConfig,
    RowsConfig,
    RowsGuidedConfig,
    RowStructureConfig,
)
from vineyard.config.sections_post import (
    CrossPathsConfig,
    DeriveConfig,
    FarmsConfig,
    MeasureConfig,
    PublishConfig,
    RouteConfig,
    TargetsConfig,
    WebConfig,
)
from vineyard.config.sections_seeded import RowsSeededConfig
from vineyard.config.sections_waste import WasteConfig


class AppConfig(Section):
    contract_version: Literal["1.1"]
    project: ProjectConfig
    paths: PathsConfig
    grid: GridConfig
    runtime: RuntimeConfig
    nodata: NodataConfig
    veg: VegConfig
    nn: NnConfig
    rows: RowsConfig
    orchard: OrchardConfig
    rows_guided: RowsGuidedConfig
    rows_seeded: RowsSeededConfig          # seeded rows from reviewed tile seeds (rows_link)
    blocks: BlocksConfig
    canopy: CanopyConfig
    interrow: InterrowConfig
    row_structure: RowStructureConfig
    waste: WasteConfig
    qa: QaConfig
    derive: DeriveConfig
    targets: TargetsConfig
    cross_paths: CrossPathsConfig
    route: RouteConfig
    export: ExportConfig
    import_: ImportConfig = Field(alias="import")
    measure: MeasureConfig
    farms: FarmsConfig
    publish: PublishConfig
    web: WebConfig
    eval: EvalConfig
    logging: LoggingConfig

    @model_validator(mode="after")
    def _web_mask_px_divides_tile_px(self) -> AppConfig:
        if self.grid.tile_px % self.web.mask_px:
            raise ValueError(f"web.mask_px={self.web.mask_px} must divide grid.tile_px={self.grid.tile_px}")
        return self
