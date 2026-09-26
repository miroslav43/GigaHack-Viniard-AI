"""End to end on the reference examples: derive -> passable -> targets -> route -> measure -> publish -> web_bundle.

AnnSet(reference) (the organizers' 2 example tiles, tests/post/factories.py), the real passages / forbidden /
START (02_route), the real nodata of the example GeoTIFFs as `tile_valid`, and the reference gap function
(stand-in for perception.attrs). Every stage runs through its StageSpec in one post run directory, so the
test covers the file hand-offs between the stages as well. The chain runs once per module.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import math
import sys
import time
import types
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import geopandas as gpd
import pandas as pd
import pytest
import shapely
from shapely.geometry import LineString

from tests.conftest import EXAMPLES_SUBDIR
from tests.post.factories import reference_annset
from tests.post.tgt_helpers import reference_gap_fn
from tests.post.web_oracle import CSV_HEADER, check_bundle
from vineyard.annset.io import write_annset
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.geo.raster import read_tile, valid_mask, valid_polygon
from vineyard.geo.tiling import GSD_M, tile_ref
from vineyard.geo.vector_io import read_layer, write_layer
from vineyard.pipeline.context import RunContext, ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages import derive as derive_stage
from vineyard.pipeline.stages import targets as targets_stage
from vineyard.pipeline.stages._post_io import walking_domain
from vineyard.web.bundle import bundle_dir

pytestmark = [pytest.mark.slow, pytest.mark.examples]

REF_RUN: Final = "20260926T0100-reference-e2e000"
POST_RUN: Final = "20260926T0310-post-e2e000"
CHAIN: Final = ("derive", "passable", "targets", "route", "measure", "publish", "web_bundle")
TIME_LIMIT_S: Final = 3
START: Final = (629504.70, 5220250.75)
MAX_OUTSIDE_SHARE: Final = 0.015  # route.max_outside_frac_publish (official elimination: 0.02)
VISIT_RADIUS_M: Final = 2.0
LEN_TOL_M: Final = 0.01
OUTSIDE_PIECE_MIN_M: Final = 0.01
# Web-team control facts for the examples (src/Web/CLAUDE.md): rows, length, canopy m², interrow m², plants.
BLOCK_FACTS: Final = {"V01": (25, 910.10, 237.12, 2068.03, 399), "V02": (26, 1031.45, 299.06, 1996.36, 251)}
FACT_TOL: Final = 0.06
RUNNER_MODULE: Final = "vineyard.pipeline.runner"


@dataclass(frozen=True)
class _StageResult:
    """Stand-in for vineyard.pipeline.runner.StageResult while the runner is not merged in this worktree."""

    stage: str
    n_items: int
    n_cached: int
    n_failed: int
    failed: tuple[str, ...] = ()
    outputs: tuple[Path, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ChainRun:
    ctx: RunContext
    results: Mapping[str, Any]
    seconds: Mapping[str, float]
    report: Mapping[str, Any]


# ------------------------------------------------------------------ setup


def _stub_runner(mp: pytest.MonkeyPatch) -> None:
    try:
        importlib.import_module(RUNNER_MODULE)
    except ModuleNotFoundError:
        stub = types.ModuleType(RUNNER_MODULE)
        stub.StageResult = _StageResult  # type: ignore[attr-defined]
        mp.setitem(sys.modules, RUNNER_MODULE, stub)


def _tile_valid(cfg: AppConfig, examples_dir: Path, tile_ids: list[str]) -> gpd.GeoDataFrame:
    """The example tiles' real imaged area (contract §1.7 nodata rule, config `nodata`)."""
    nd = cfg.nodata
    geoms, fracs = [], []
    for tile_id in tile_ids:
        rgb = read_tile(examples_dir / "images" / f"{tile_id}.tif")
        mask = valid_mask(rgb, max_rgb=nd.max_rgb, min_area_px=nd.min_area_m2 / GSD_M**2, close_px=nd.close_px,
                          dilate_px=nd.dilate_px)
        geoms.append(valid_polygon(mask, tile_ref(tile_id), approx_eps_px=nd.approx_eps_px))
        fracs.append(float(mask.mean()))
    return gpd.GeoDataFrame({"tile_id": tile_ids, "valid_frac": pd.Series(fracs, dtype="float32")},
                            geometry=geoms, crs="EPSG:32635")


def _context(root: Path, data_root: Path, examples_dir: Path, examples_xml: bytes) -> RunContext:
    cfg = load_config(overrides=(f"route.solver.time_limit_s={TIME_LIMIT_S}", f"paths.publish_dir={root / 'publish'}",
                                 f"web.out_dir={root / 'web'}"),
                      environ={"VINEYARD_WORK_DIR": str(root / "work"), "VINEYARD_DATA_ROOT": str(data_root)})
    ann = reference_annset(examples_xml)
    write_annset(ann, cfg.paths.work_dir / "runs" / REF_RUN / "annset")
    ctx = new_run_context(cfg, source=Source.REFERENCE, kind="post", annset_ref=REF_RUN, run_id=POST_RUN)
    ensure_run_dirs(ctx.paths)
    write_layer(_tile_valid(cfg, examples_dir, sorted(ann.tile_ids())), "tile_valid", ctx.paths.tile_valid)
    return ctx


# ------------------------------------------------------------------ report


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _outside_pieces(line: LineString, inner: Any) -> list[float]:
    """Walked length outside `inner` per excursion (consecutive segments that leave the domain), longest first;
    an out-and-back through the same gap counts twice, like the validator's outside length."""
    xy = shapely.get_coordinates(line)
    segments = shapely.linestrings(list(zip(xy[:-1], xy[1:], strict=True)))
    outside = shapely.length(shapely.difference(segments, inner))
    runs, current = [], 0.0
    for length in outside:
        if length > OUTSIDE_PIECE_MIN_M:
            current += float(length)
        elif current:
            runs, current = [*runs, current], 0.0
    runs = [*runs, current] if current else runs
    return sorted(runs, reverse=True)


def _route_report(ctx: RunContext) -> dict[str, Any]:
    doc = _read_json(ctx.paths.metrics_dir / "route_validation.json")
    line = read_layer(ctx.paths.layers_dir / "route.parquet", "route").geometry.iloc[0]
    domain = walking_domain(ctx.paths, ctx)
    assert domain is not None
    pieces = _outside_pieces(line, domain.inner)
    keys = ("passed", "policy", "policy_accepted", "plan_outside_limit", "solver", "length_m", "outside_len_m",
            "outside_frac", "coverage_est", "coverage_required", "n_targets", "n_unreachable", "unreachable_notes",
            "solve_time_s", "headland", "fallback_reason", "optional_delta_m", "dropped_outside_budget")
    return {k: doc.get(k) for k in keys} | {
        "policies_tried": [(p.get("policy"), p.get("acceptable"), p.get("outside_frac")) for p in doc["policies"]],
        "route_outside_pieces": len(pieces), "route_outside_pieces_m": [round(p, 2) for p in pieces[:10]],
        "route_outside_total_m": round(sum(pieces), 3),
        "headland_ok": bool(doc["outside_frac"] < MAX_OUTSIDE_SHARE),
        "baseline": _read_json(ctx.paths.metrics_dir / "route_baseline.json")}


def _report(ctx: RunContext, seconds: Mapping[str, float]) -> dict[str, Any]:
    passable = _read_json(ctx.paths.metrics_dir / "passable_report.json")
    targets = _read_json(ctx.paths.metrics_dir / "targets.json")
    return {"seconds": {k: round(v, 2) for k, v in seconds.items()}, "targets": targets["counts"],
            "passable": {k: passable[k] for k in ("strategy", "recommended_strategy", "graph", "domain")},
            "passable_strategies": passable["strategies"], "route": _route_report(ctx)}


def run_chain(root: Path, data_root: Path, examples_dir: Path, examples_xml: bytes) -> ChainRun:
    """Every post stage in order in one post run under `root`; the caller patches runner / gap engine."""
    ctx = _context(root, data_root, examples_dir, examples_xml)
    results: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    for name in CHAIN:
        t0 = time.perf_counter()
        results[name] = load_stage(name).run(ctx)
        seconds[name] = time.perf_counter() - t0
    report = _report(ctx, seconds)
    (root / "e2e_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return ChainRun(ctx=ctx, results=results, seconds=seconds, report=report)


@pytest.fixture(scope="module")
def chain(tmp_path_factory: pytest.TempPathFactory, data_root: Path, examples_xml: bytes) -> Iterator[ChainRun]:
    with pytest.MonkeyPatch.context() as mp:
        _stub_runner(mp)
        mp.setattr(targets_stage, "make_gap_fn", lambda ctx: reference_gap_fn)
        mp.setattr(derive_stage, "gap_function", lambda ctx: reference_gap_fn)
        yield run_chain(tmp_path_factory.mktemp("e2e"), data_root, data_root / EXAMPLES_SUBDIR, examples_xml)


# ------------------------------------------------------------------ tests


def test_every_stage_succeeds(chain: ChainRun) -> None:
    assert {name: (r.stage, r.n_failed) for name, r in chain.results.items()} == {n: (n, 0) for n in CHAIN}


def test_route_is_published_closed_at_start_and_inside_the_domain(chain: ChainRun) -> None:
    doc = _read_json(chain.ctx.cfg.paths.publish_dir / "route.geojson")
    (feature,) = doc["features"]
    coords = feature["geometry"]["coordinates"]
    assert feature["geometry"]["type"] == "LineString" and coords[0] == coords[-1] == list(START)
    assert doc["crs"]["properties"]["name"] == "urn:ogc:def:crs:EPSG::32635"
    props = feature["properties"]
    assert props["length_m"] == pytest.approx(LineString(coords).length, abs=LEN_TOL_M)
    assert props["outside_share"] < MAX_OUTSIDE_SHARE and props["baseline_length_m"] > props["length_m"]
    assert props["duration_min"] == pytest.approx(props["length_m"] / 1000.0 / 4.0 * 60.0, abs=0.1)
    route = chain.report["route"]
    assert route["passed"] and route["outside_frac"] < MAX_OUTSIDE_SHARE
    assert route["policy_accepted"] and route["outside_frac"] <= route["plan_outside_limit"]


def test_every_reachable_target_is_within_two_metres(chain: ChainRun) -> None:
    visits = read_layer(chain.ctx.paths.layers_dir / "target_visits.parquet", "target_visits")
    targets = read_layer(chain.ctx.paths.layers_dir / "targets.parquet", "targets")
    assert set(visits.target_id) == set(targets.target_id)
    reachable = visits[visits.reachable_final.astype(bool)]
    assert len(reachable) and reachable.covered.astype(bool).all()
    assert (reachable.visit_dist_m <= VISIT_RADIUS_M).all()
    gaps = set(targets.target_id[targets.kind == "row_gap"])
    assert gaps and gaps <= set(reachable.target_id), "every row-gap target must be visited"


def _csv_rows(text: str) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["level"], r["vineyard_id"] or r["row_id"]): r for r in csv.DictReader(io.StringIO(text))}


def test_one_measurements_csv_everywhere_matching_the_web_facts(chain: ChainRun) -> None:
    paths = chain.ctx.paths
    exported = (paths.exports_dir / "measurements.csv").read_bytes()
    web = bundle_dir(chain.ctx.cfg.web.out_dir, "siret3") / "measurements.csv"
    assert (chain.ctx.cfg.paths.publish_dir / "measurements.csv").read_bytes() == exported == web.read_bytes()
    table = _csv_rows(exported.decode("utf-8"))
    assert exported.decode("utf-8").splitlines()[0] == CSV_HEADER
    for vid, (n_rows, length, canopy, interrow, plants) in BLOCK_FACTS.items():
        row = table[("block", vid)]
        assert (int(row["row_count"]), int(row["plant_count"])) == (n_rows, plants)
        for col, fact in (("row_length_m", length), ("canopy_area_m2", canopy), ("interrow_area_m2", interrow)):
            assert float(row[col]) == pytest.approx(fact, abs=FACT_TOL), (vid, col)
    survey = table[("survey", "")]
    assert (survey["block_count"], survey["row_count"], survey["plant_count"]) == ("2", "51", "650")


def test_web_bundle_passes_the_contract_oracle(chain: ChainRun) -> None:
    out = bundle_dir(chain.ctx.cfg.web.out_dir, "siret3")
    found = check_bundle(out)
    counts = {name: len(found[f"{name}.geojson"]) for name in ("blocks", "rows", "interrows")}
    assert counts == {"blocks": 2, "rows": 51, "interrows": 49} and len(found["canopies.geojsonl"]) == 650
    (route,) = found["route.geojson"]
    published = _read_json(chain.ctx.cfg.paths.publish_dir / "route.geojson")["features"][0]["properties"]
    assert route["length_m"] == pytest.approx(published["length_m"], abs=LEN_TOL_M)
    assert route["outside_share"] == pytest.approx(published["outside_share"])
    targets = found["targets.geojson"]
    assert len(targets) == chain.report["targets"]["total"] and any(t["route_order"] == 1 for t in targets)


def test_headland_metrics_are_reported(chain: ChainRun) -> None:
    headland = chain.report["route"]["headland"]
    assert headland["n_interrow_ends"] > 0 and 0.0 <= headland["share_interrow_ends_near_passage"] <= 1.0
    assert headland["n_connectors"] >= headland["n_connectors_outside"] > 0
    strategies = {s["strategy"] for s in chain.report["passable_strategies"]}
    assert strategies == {"penalty", "limit", "inside_only"}
    route = chain.report["route"]
    assert math.isfinite(route["solve_time_s"]) and route["headland_ok"]
    assert route["route_outside_total_m"] == pytest.approx(route["outside_len_m"], abs=0.05)
