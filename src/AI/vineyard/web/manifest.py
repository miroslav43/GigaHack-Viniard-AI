"""manifest.json of the web data bundle (src/Web/CLAUDE.md §6.3): survey identity, stage, provenance."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo

from vineyard.contracts.enums import Source
from vineyard.errors import SchemaError

if TYPE_CHECKING:
    from vineyard.config import AppConfig

CAPTURED_AT: Final = "2025-05-20"  # CONFIG-REQUEST: web.captured_at = "2025-05-20"
IMAGERY_SOURCE: Final = "3DATA COLLECT / OpenAerialMap"  # CONFIG-REQUEST: web.imagery_source = "3DATA COLLECT / OpenAerialMap"
IMAGERY_LICENSE: Final = "CC BY 4.0"  # CONFIG-REQUEST: web.imagery_license = "CC BY 4.0"

STAGE_MODEL: Final = "model"
STAGE_CORRECTED: Final = "marcaj_corrected"
MANIFEST_STAGES: Final = (STAGE_MODEL, STAGE_CORRECTED)
# The web enum has no "reference": the organizers' reference annotations are human-made, like Marcaj corrections.
STAGE_OF_SOURCE: Final[Mapping[Source, str]] = MappingProxyType(
    {Source.MODEL: STAGE_MODEL, Source.MARCAJ: STAGE_CORRECTED, Source.REFERENCE: STAGE_CORRECTED}
)
_SURVEY_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_RUN_STAMP_RE: Final = re.compile(r"^(\d{8}T\d{4})-")
_RUN_STAMP_FORMAT: Final = "%Y%m%dT%H%M"


@dataclass(frozen=True)
class SurveyInfo:
    survey_id: str
    name: str
    captured_at: str
    gsd_m: float
    crs: str
    source: str
    license: str
    tiles: int

    def __post_init__(self) -> None:
        if not _SURVEY_ID_RE.match(self.survey_id):
            raise SchemaError("survey_id must be lowercase letters, digits, '_' or '-'", survey_id=self.survey_id)
        try:
            date.fromisoformat(self.captured_at)
        except ValueError:
            raise SchemaError("captured_at is not an ISO date", captured_at=self.captured_at) from None
        if self.gsd_m <= 0 or self.tiles < 0:
            raise SchemaError("gsd_m must be > 0 and tiles >= 0", gsd_m=self.gsd_m, tiles=self.tiles)


def survey_info(cfg: AppConfig) -> SurveyInfo:
    """Survey identity from `web.survey_id` / `web.survey_name`, imagery facts from the grid and constants."""
    return SurveyInfo(
        survey_id=cfg.web.survey_id, name=cfg.web.survey_name, captured_at=CAPTURED_AT, gsd_m=cfg.grid.gsd_m,
        crs=cfg.project.crs, source=IMAGERY_SOURCE, license=IMAGERY_LICENSE, tiles=cfg.grid.expected_tiles,
    )


def manifest_stage(source: Source | str) -> str:
    """`model` for model output, `marcaj_corrected` for human-corrected sets (Marcaj, reference)."""
    try:
        return STAGE_OF_SOURCE[Source(source)]
    except ValueError:
        raise SchemaError("unknown AnnSet source for the web manifest", source=str(source)) from None


def generated_at_from_run_id(run_id: str, tz: str) -> str | None:
    """ISO 8601 time from a `YYYYMMDDTHHMM-...` run id (local time in `tz`); None when the id has no stamp.

    Using the run's own stamp keeps the manifest byte-identical when the same run is rebuilt.
    """
    match = _RUN_STAMP_RE.match(run_id)
    if match is None:
        return None
    try:
        stamp = datetime.strptime(match.group(1), _RUN_STAMP_FORMAT)
    except ValueError:
        return None  # digits that are not a calendar time: treated like a custom run id
    return stamp.replace(tzinfo=ZoneInfo(tz)).isoformat(timespec="seconds")


def build_manifest(
    info: SurveyInfo,
    *,
    stage: str,
    generated_at: str,
    pipeline_version: str,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Contract fields of §6.3 plus informative `extras` (which may not redefine a contract field)."""
    if stage not in MANIFEST_STAGES:
        raise SchemaError("manifest stage must be model or marcaj_corrected", stage=stage)
    try:
        datetime.fromisoformat(generated_at)
    except ValueError:
        raise SchemaError("generated_at is not ISO 8601", generated_at=generated_at) from None
    doc = {
        "survey_id": info.survey_id, "name": info.name, "captured_at": info.captured_at, "gsd_m": info.gsd_m,
        "crs": info.crs, "source": info.source, "license": info.license, "stage": stage,
        "generated_at": generated_at, "pipeline_version": pipeline_version, "tiles": info.tiles,
    }
    extra = dict(extras or {})
    clash = sorted(set(extra) & set(doc))
    if clash:
        raise SchemaError("manifest extras redefine contract fields", keys=clash)
    return doc | extra
