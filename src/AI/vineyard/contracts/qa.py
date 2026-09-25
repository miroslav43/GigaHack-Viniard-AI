"""The single QA issue type shared by every stage, and its `qa_issues` layer (contract §2.5.14)."""

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import geopandas as gpd
from shapely.geometry import Point

from vineyard.contracts.enums import QA_CODES, Severity, Source
from vineyard.contracts.ids import format_issue_id
from vineyard.contracts.schemas import coerce_layer, empty_layer
from vineyard.errors import SchemaError
from vineyard.geo.tiling import CRS_EPSG

QA_LAYER: Final = "qa_issues"
SEVERITY_RANK: Final[Mapping[Severity, int]] = MappingProxyType(
    {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
)
_CODE_RE: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_ISSUE_CONFIDENCE: Final = 1.0

__all__ = ["QA_CODES", "QA_LAYER", "SEVERITY_RANK", "QaIssue", "issues_to_gdf"]


@dataclass(frozen=True)
class QaIssue:
    """One point to check. `x`, `y` are UTM metres; both None when the issue has no location."""

    severity: Severity
    code: str
    tile_id: str
    object_id: str
    message: str
    x: float | None = None
    y: float | None = None

    def __post_init__(self) -> None:
        try:
            severity = Severity(self.severity)
        except ValueError:
            raise SchemaError("invalid qa severity", severity=self.severity, code=self.code) from None
        # Frozen dataclass: normalise a plain-string severity once, at construction.
        object.__setattr__(self, "severity", severity)
        if not isinstance(self.code, str) or not _CODE_RE.fullmatch(self.code):
            raise SchemaError("qa code must be snake_case", code=self.code, tile_id=self.tile_id)
        if (self.x is None) != (self.y is None):
            raise SchemaError("qa issue needs both x and y or neither", code=self.code, tile_id=self.tile_id)
        if self.x is not None and not (math.isfinite(self.x) and math.isfinite(self.y)):
            raise SchemaError("qa issue coordinates must be finite", code=self.code, tile_id=self.tile_id)

    @property
    def sort_key(self) -> tuple[int, str, str, str, str]:
        return (SEVERITY_RANK[self.severity], self.code, self.tile_id, self.object_id, self.message)


def issues_to_gdf(
    issues: Iterable[QaIssue],
    *,
    source: Source = Source.MODEL,
    run_id: str = "",
    model_version: str = "",
) -> gpd.GeoDataFrame:
    """`qa_issues` layer, ids Q00001.. assigned after sorting by (severity, code, tile_id, object_id)."""
    ordered = sorted(issues, key=lambda issue: issue.sort_key)
    if not ordered:
        return empty_layer(QA_LAYER)
    n = len(ordered)
    data = {
        "issue_id": [format_issue_id(k) for k in range(1, n + 1)],
        "severity": [i.severity.value for i in ordered],
        "code": [i.code for i in ordered],
        "tile_id": [i.tile_id for i in ordered],
        "object_id": [i.object_id for i in ordered],
        "message": [i.message for i in ordered],
        "source": [Source(source).value] * n,
        "run_id": [run_id] * n,
        "model_version": [model_version] * n,
        "confidence": [_ISSUE_CONFIDENCE] * n,
        "qa_flags": [""] * n,
    }
    geoms = [Point() if i.x is None else Point(i.x, i.y) for i in ordered]
    return coerce_layer(gpd.GeoDataFrame(data, geometry=geoms, crs=CRS_EPSG), QA_LAYER)
