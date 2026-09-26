"""Stage `publish`: validated copy of exports/route.geojson + exports/measurements.csv to the repo root.

Everything is recomputed from the files, never trusted from their properties (contract §5.1, design 04
§3.10): one LineString Feature, the EPSG:32635 `crs` member, <= 2 decimals, first/last vertex = START
(endpoints within closure_max_m are snapped to START exactly and `length_m` recomputed), declared
`length_m` = recomputed length (±length_tol_m), outside share against this run's `passable_domain`
<= route.max_outside_frac_publish, the CSV in the exact web format with consistent sums, and an
optional `publish.require_source`. A refusal raises `PublishRefused` and leaves the root untouched.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry

from vineyard.errors import StageError
from vineyard.logging_setup import get_logger, log_event
from vineyard.measure.csv_format import check_measurements_bytes
from vineyard.pipeline.atomic import atomic_write_bytes, atomic_write_json
from vineyard.pipeline.registry import StageSpec
from vineyard.pipeline.stages._post_io import (
    find_post_run,
    load_annset_meta,
    stage_result,
    start_xy,
    walking_domain,
)
from vineyard.route.geojson import LENGTH_PROP, RouteFileLimits, check_route_file

if TYPE_CHECKING:
    from vineyard.pipeline.context import RunContext

STAGE_NAME: Final = "publish"
STAGE_VERSION: Final = "2"  # 2: block interrow_area_m2 sum check (publish.sum_check_tol_m2)
CFG_KEYS: Final = ("publish", "paths.publish_dir", "route.max_outside_frac_publish", "route.validate",
                   "route.domain")
ROUTE_NAME: Final = "route.geojson"
CSV_NAME: Final = "measurements.csv"
STAGING_DIR: Final = "publish"
REPORT_NAME: Final = "publish_report.json"
EVENT_PUBLISHED: Final = "publish.written"
EVENT_REFUSED: Final = "publish.refused"
# 1 cm between 2-decimal UTM coordinates computes as 0.01000000001: absorb that float noise when snapping.
FLOAT_EPS_M: Final = 1e-6

_log = get_logger("pipeline.stages.publish")


class PublishRefused(StageError):
    """The files failed a publish check; nothing was written to the publish directory."""

    exit_code: ClassVar[int] = 2


@dataclass(frozen=True)
class PublishLimits:
    route: RouteFileLimits
    sum_tol_m: float  # block / row row_length_m sums vs the survey line
    sum_tol_m2: float  # block interrow_area_m2 sum vs the survey line
    require_source: str | None


@dataclass(frozen=True)
class PublishSources:
    route: Path
    csv: Path


@dataclass(frozen=True)
class Verdict:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class PublishReport:
    staged: PublishSources
    checks: tuple[Verdict, ...]
    source: str

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failed(self) -> tuple[Verdict, ...]:
        return tuple(c for c in self.checks if not c.ok)

    def to_json(self) -> dict[str, Any]:
        return {"passed": self.passed, "source": self.source, "staged_route": str(self.staged.route),
                "staged_csv": str(self.staged.csv), "failed": [c.name for c in self.failed],
                "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks]}


# ------------------------------------------------------------------ route normalisation


def _single_line_coords(doc: Any) -> list[list[float]] | None:
    try:
        (feature,) = doc["features"]
        geometry = feature["geometry"]
    except (KeyError, TypeError, ValueError):
        return None
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if geometry.get("type") != "LineString" or not isinstance(coords, list) or len(coords) < 2:
        return None
    return coords


def _snap_end(xy: Sequence[float], start: list[float], tol_m: float) -> list[float] | None:
    dist = math.hypot(float(xy[0]) - start[0], float(xy[1]) - start[1])
    return start if 0.0 < dist <= tol_m + FLOAT_EPS_M else None


def prepared_route_bytes(raw: bytes, start: tuple[float, float], limits: RouteFileLimits) -> bytes:
    """`raw` unchanged, or a copy whose endpoints within closure_max_m are START exactly (length_m redone)."""
    try:
        doc = json.loads(raw)
    except ValueError:
        return raw
    coords = _single_line_coords(doc)
    if coords is None:
        return raw
    exact = [round(float(start[0]), limits.decimals), round(float(start[1]), limits.decimals)]
    head, tail = _snap_end(coords[0], exact, limits.closure_max_m), _snap_end(coords[-1], exact, limits.closure_max_m)
    if head is None and tail is None:
        return raw
    snapped = [head or coords[0], *coords[1:-1], tail or coords[-1]]
    feature = doc["features"][0]
    props = dict(feature.get("properties") or {})
    props[LENGTH_PROP] = round(LineString(snapped).length, limits.decimals)
    new_feature = feature | {"properties": props, "geometry": {"type": "LineString", "coordinates": snapped}}
    text = json.dumps(doc | {"features": [new_feature]}, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


# ------------------------------------------------------------------ evaluate / write


def _read(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PublishRefused(f"cannot read {what}", path=str(path), error=str(exc)) from exc


def _stage_files(sources: PublishSources, staging: Path, start: tuple[float, float],
                 limits: PublishLimits) -> tuple[PublishSources, bytes]:
    route = prepared_route_bytes(_read(sources.route, ROUTE_NAME), start, limits.route)
    table = _read(sources.csv, CSV_NAME)
    staged = PublishSources(route=atomic_write_bytes(staging / ROUTE_NAME, route),
                            csv=atomic_write_bytes(staging / CSV_NAME, table))
    return staged, table


def evaluate_publish(sources: PublishSources, staging: Path, *, inner: BaseGeometry | None,
                     start_xy: tuple[float, float], limits: PublishLimits, source: str) -> PublishReport:
    """Stage both files under `staging` and run every check on the staged copies."""
    staged, table = _stage_files(sources, Path(staging), start_xy, limits)
    checks: list[Verdict] = []
    if inner is None or inner.is_empty:
        checks.append(Verdict("passable_domain", False, "no passable_domain layer to measure the outside share"))
    else:
        route_checks = check_route_file(staged.route, start_xy=start_xy, inner=inner, limits=limits.route)
        checks += [Verdict(c.name, c.ok, c.detail) for c in route_checks]
    failed_csv = check_measurements_bytes(table, sum_tol_m=limits.sum_tol_m, sum_tol_m2=limits.sum_tol_m2)
    checks += [Verdict(c.name, c.ok, c.detail) for c in failed_csv] or [Verdict(CSV_NAME, True)]
    required = limits.require_source
    checks.append(Verdict("require_source", required is None or required == source,
                          f"required={required} source={source}"))
    return PublishReport(staged=staged, checks=tuple(checks), source=source)


def write_published(report: PublishReport, dest_dir: Path) -> tuple[Path, ...]:
    """Atomically copy the staged files to `dest_dir`; refuses when any check failed."""
    if not report.passed:
        raise PublishRefused("publish refused", failed="; ".join(f"{c.name}: {c.detail}" for c in report.failed))
    dest = Path(dest_dir)
    return (atomic_write_bytes(dest / ROUTE_NAME, report.staged.route.read_bytes()),
            atomic_write_bytes(dest / CSV_NAME, report.staged.csv.read_bytes()))


# ------------------------------------------------------------------ stage


def _limits(ctx: RunContext) -> PublishLimits:
    cfg = ctx.cfg
    return PublishLimits(route=RouteFileLimits.from_route_cfg(cfg.route), sum_tol_m=cfg.publish.sum_check_tol_m,
                         sum_tol_m2=cfg.publish.sum_check_tol_m2, require_source=cfg.publish.require_source)


def run(ctx: RunContext) -> Any:
    meta = load_annset_meta(ctx, STAGE_NAME)
    paths = find_post_run(ctx, (f"exports/{ROUTE_NAME}", f"exports/{CSV_NAME}"), STAGE_NAME)
    domain = walking_domain(paths, ctx)
    sources = PublishSources(route=paths.exports_dir / ROUTE_NAME, csv=paths.exports_dir / CSV_NAME)
    report = evaluate_publish(sources, ctx.paths.run_dir / STAGING_DIR, inner=None if domain is None else domain.inner,
                              start_xy=start_xy(ctx, STAGE_NAME), limits=_limits(ctx), source=meta.source.value)
    doc = report.to_json() | {"route_src": str(sources.route), "csv_src": str(sources.csv),
                              "publish_dir": str(ctx.cfg.paths.publish_dir)}
    report_path = atomic_write_json(ctx.paths.metrics_dir / REPORT_NAME, doc)
    if not report.passed:
        log_event(_log, EVENT_REFUSED, stage=STAGE_NAME, failed=doc["failed"])
    outputs = write_published(report, ctx.cfg.paths.publish_dir)
    log_event(_log, EVENT_PUBLISHED, stage=STAGE_NAME, outputs=[str(p) for p in outputs])
    return stage_result(STAGE_NAME, n_items=len(outputs), outputs=(*outputs, report_path),
                        metrics={"passed": 1.0, "n_checks": float(len(report.checks))})


STAGE: Final = StageSpec(
    name=STAGE_NAME, version=STAGE_VERSION, scope="global", cfg_keys=CFG_KEYS, requires=("route", "measure"),
    run=run, description="validated copy of route.geojson + measurements.csv to the repo root",
)
