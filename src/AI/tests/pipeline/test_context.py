"""RunPaths / RunContext / make_run_id / new_run_context."""

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vineyard import __version__
from vineyard.config import AppConfig, load_config
from vineyard.contracts.enums import Source
from vineyard.errors import ConfigError
from vineyard.pipeline.context import (
    RunContext,
    RunPaths,
    ensure_run_dirs,
    make_run_id,
    make_run_paths,
    new_run_context,
    source_from_annset_ref,
)
from vineyard.pipeline.registry import StageSpec

RUN_ID_RE = re.compile(r"^\d{8}T\d{4}-(model|marcaj|reference|post)-[0-9a-f]{6}$")


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return load_config(environ={"VINEYARD_WORK_DIR": str(tmp_path / "work")})


def _ctx(cfg: AppConfig, **kwargs: object) -> RunContext:
    return new_run_context(cfg, source=Source.MODEL, **kwargs)  # type: ignore[arg-type]


def test_make_run_id_format() -> None:
    now = datetime(2026, 9, 26, 3, 10, 59)
    assert make_run_id(now, Source.MODEL, "a1b2c3d4e5") == "20260926T0310-model-a1b2c3"
    assert make_run_id(now, "post", "ffffff") == "20260926T0310-post-ffffff"
    assert make_run_id(now, "marcaj", "0123456789abcdef") == "20260926T0310-marcaj-012345"


@pytest.mark.parametrize(("source", "digest"), [("bogus", "a1b2c3"), (Source.MODEL, "xyz123"), (Source.MODEL, "a1b")])
def test_make_run_id_rejects_bad_input(source: str, digest: str) -> None:
    with pytest.raises(ConfigError):
        make_run_id(datetime(2026, 9, 26), source, digest)


def test_run_paths_layout(cfg: AppConfig, tmp_path: Path) -> None:
    paths = make_run_paths(cfg, "r1")
    work = tmp_path / "work"
    assert paths.work_dir == work
    assert paths.tiles_dir == work / "tiles"
    assert paths.cache_dir == work / "cache"
    assert paths.static_layers_dir == work / "layers"
    assert paths.runs_dir == work / "runs"
    assert paths.run_dir == work / "runs" / "r1"
    assert paths.layers_dir == work / "runs" / "r1" / "layers"
    assert paths.annset_dir == work / "runs" / "r1" / "annset"
    assert paths.exports_dir == work / "runs" / "r1" / "exports"
    assert paths.qa_dir == work / "runs" / "r1" / "qa"
    assert paths.metrics_dir == work / "runs" / "r1" / "metrics"
    assert paths.logs_dir == work / "runs" / "r1" / "logs"
    assert paths.project_root == cfg.paths.project_root
    assert paths.tile_index == work / "tile_index.parquet"
    assert paths.tile_valid == work / "layers" / "tile_valid.parquet"
    assert paths.run_json == work / "runs" / "r1" / "run.json"
    assert paths.pipeline_log == work / "runs" / "r1" / "logs" / "pipeline.jsonl"
    assert paths.tile_cache("canopy", "siret3_r021_c012", "parquet") == work / "cache" / "canopy" / "siret3_r021_c012.parquet"
    assert paths.tile_cache("veg", "t", ".png") == work / "cache" / "veg" / "t.png"


def test_ensure_run_dirs_creates_tree(cfg: AppConfig) -> None:
    paths = make_run_paths(cfg, "r2")
    ensure_run_dirs(paths)
    for directory in (paths.cache_dir, paths.static_layers_dir, paths.layers_dir, paths.annset_dir,
                      paths.exports_dir, paths.qa_dir, paths.metrics_dir, paths.logs_dir, paths.tiles_dir):
        assert directory.is_dir()


def test_new_run_context_defaults(cfg: AppConfig) -> None:
    ctx = _ctx(cfg)
    assert RUN_ID_RE.match(ctx.run_id) and "-model-" in ctx.run_id
    assert ctx.source is Source.MODEL
    assert ctx.workers == 8
    assert ctx.allow_failures is False
    assert ctx.force == frozenset() and ctx.force_all is False
    assert ctx.tile_filter == ()
    assert ctx.package_version == __version__
    assert ctx.annset_ref is None
    assert ctx.paths.run_dir == cfg.paths.work_dir / "runs" / ctx.run_id
    assert not ctx.paths.run_dir.exists()


def test_new_run_context_explicit(cfg: AppConfig) -> None:
    ctx = new_run_context(cfg, source=Source.MARCAJ, run_id="i1-ex", tiles=["siret3_r0*"], force=["canopy"],
                          force_all=False, workers=2, allow_failures=True, annset_ref="LATEST_MODEL")
    assert ctx.run_id == "i1-ex"
    assert ctx.tile_filter == ("siret3_r0*",)
    assert ctx.force == frozenset({"canopy"})
    assert (ctx.workers, ctx.allow_failures, ctx.annset_ref) == (2, True, "LATEST_MODEL")


def test_kind_post_run_id(cfg: AppConfig) -> None:
    ctx = new_run_context(cfg, source=Source.MARCAJ, kind="post", annset_ref="LATEST_MARCAJ")
    assert "-post-" in ctx.run_id and ctx.source is Source.MARCAJ


def test_generated_run_id_depends_on_inputs(cfg: AppConfig) -> None:
    now = datetime(2026, 9, 26, 3, 10, tzinfo=UTC)
    a = new_run_context(cfg, source=Source.MODEL, now=now)
    b = new_run_context(cfg, source=Source.MODEL, now=now)
    c = new_run_context(cfg, source=Source.MODEL, now=now, tiles=["siret3_r021_c012"])
    assert a.run_id == b.run_id != c.run_id


@pytest.mark.parametrize(
    "kwargs",
    [{"run_id": "../escape"}, {"run_id": "a b"}, {"run_id": ""}, {"force": ["canopyy"]}, {"workers": 0},
     {"tiles": [""]}, {"kind": "weird"}],
)
def test_new_run_context_rejects_bad_input(cfg: AppConfig, kwargs: dict) -> None:
    with pytest.raises(ConfigError):
        _ctx(cfg, **kwargs)


def test_workers_auto(tmp_path: Path) -> None:
    cfg = load_config(overrides=("runtime.n_workers=auto", "runtime.workers_auto_reserve=1000"),
                      environ={"VINEYARD_WORK_DIR": str(tmp_path)})
    assert _ctx(cfg).workers == 1


def test_selected_tiles_globs_sorted(cfg: AppConfig) -> None:
    ids = ["siret3_r021_c012", "siret3_r006_c004", "siret3_r006_c005", "siret3_r010_c001"]
    assert _ctx(cfg).selected_tiles(ids) == tuple(sorted(ids))
    ctx = _ctx(cfg, tiles=["siret3_r006_*", "siret3_r021_c012"])
    assert ctx.selected_tiles(ids) == ("siret3_r006_c004", "siret3_r006_c005", "siret3_r021_c012")
    assert ctx.selected_tiles(iter(["siret3_r006_c004", "siret3_r006_c004"])) == ("siret3_r006_c004",)


def test_should_force(cfg: AppConfig) -> None:
    assert _ctx(cfg, force=["canopy"]).should_force("canopy")
    assert not _ctx(cfg, force=["canopy"]).should_force("interrow")
    assert _ctx(cfg, force_all=True).should_force("interrow")


def _run(ctx: object) -> object:
    return ctx


def test_stage_cfg_digest_tracks_cfg_keys(cfg: AppConfig) -> None:
    spec = StageSpec("canopy", "1", "tile", ("canopy", "veg"), ("blocks",), _run)
    ctx = _ctx(cfg)
    other = _ctx(load_config(overrides=("canopy.min_area_m2=0.3",), environ={}))
    same = _ctx(load_config(overrides=("route.solver.time_limit_s=3",), environ={}))
    assert ctx.stage_cfg_digest(spec) != other.stage_cfg_digest(spec)
    assert ctx.stage_cfg_digest(spec) == same.stage_cfg_digest(spec)


def test_model_version_format(cfg: AppConfig) -> None:
    ctx = _ctx(cfg)
    assert re.fullmatch(rf"pipe@{re.escape(__version__)}\+[0-9a-z]{{7}};nn=none;waste=none", ctx.model_version())
    assert ctx.model_version(nn="vine-unet@v1+1a2b3c4d", waste="rule@1").endswith(
        ";nn=vine-unet@v1+1a2b3c4d;waste=rule@1"
    )


def test_run_paths_is_frozen(cfg: AppConfig) -> None:
    paths: RunPaths = make_run_paths(cfg, "r3")
    with pytest.raises(AttributeError):
        paths.run_dir = Path("/tmp")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("LATEST_MODEL", Source.MODEL),
        ("LATEST_MARCAJ", Source.MARCAJ),
        ("LATEST_REFERENCE", Source.REFERENCE),
        ("20260926T0310-marcaj-a1b2c3", Source.MARCAJ),
        ("/abs/work/runs/20260926T0310-reference-a1b2c3", Source.REFERENCE),
        ("i1-ex", Source.MODEL),
        (None, Source.MODEL),
    ],
)
def test_source_from_annset_ref(ref: str | None, expected: Source) -> None:
    assert source_from_annset_ref(ref) is expected
