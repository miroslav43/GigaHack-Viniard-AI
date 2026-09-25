"""Root `AppConfig`: every section, in the plan's order (X3). `import` is aliased to `import_`."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

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
    RowStructureConfig,
)
from vineyard.config.sections_post import (
    DeriveConfig,
    MeasureConfig,
    PublishConfig,
    RouteConfig,
    TargetsConfig,
    WebConfig,
)
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
    blocks: BlocksConfig
    canopy: CanopyConfig
    interrow: InterrowConfig
    row_structure: RowStructureConfig
    waste: WasteConfig
    qa: QaConfig
    derive: DeriveConfig
    targets: TargetsConfig
    route: RouteConfig
    export: ExportConfig
    import_: ImportConfig = Field(alias="import")
    measure: MeasureConfig
    publish: PublishConfig
    web: WebConfig
    eval: EvalConfig
    logging: LoggingConfig
