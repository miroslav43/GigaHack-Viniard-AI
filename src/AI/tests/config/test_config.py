"""load_config: default.yaml, deep merge, --set, env overrides, path resolution, validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vineyard.config import DEFAULT_CONFIG, PROJECT_ROOT, AppConfig, load_config
from vineyard.config.loader import apply_override, deep_merge
from vineyard.errors import ConfigError

REPO_ROOT = PROJECT_ROOT.parents[1]
DATA_ROOT = REPO_ROOT / "data & info"
NO_ENV: dict[str, str] = {}


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config(environ=NO_ENV)


def _overlay(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "overlay.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_project_root_and_default_path() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert DEFAULT_CONFIG == PROJECT_ROOT / "configs" / "default.yaml"


def test_root_sections_in_plan_order() -> None:
    assert list(AppConfig.model_fields) == [
        "contract_version", "project", "paths", "grid", "runtime", "nodata", "veg", "nn", "rows",
        "orchard", "rows_guided", "blocks", "canopy", "interrow", "row_structure", "waste", "qa", "derive", "targets",
        "route", "export", "import_", "measure", "publish", "web", "eval", "logging",
    ]


def test_measured_defaults(cfg: AppConfig) -> None:
    assert cfg.contract_version == "1.1"
    assert cfg.canopy.corridor_half_m == 0.30
    assert (cfg.rows.detect.spacing_min_m, cfg.rows.detect.spacing_max_m) == (1.8, 3.8)
    assert cfg.export.cvat.max_zip_bytes == 85_000_000
    assert cfg.export.cvat.shape_source == "manual"
    assert cfg.export.cvat.max_canopy_interrow_overlap_m2 == 0.05
    assert cfg.export.cvat.notch_width_px == 1.0
    assert (cfg.canopy.vector_offset_px, cfg.canopy.vector_outset_px) == (0.0, 0.0)
    assert cfg.nn.fusion == "A" and cfg.nn.enabled is False
    assert (cfg.nn.use_in_rows, cfg.nn.use_in_interrow) == (False, False)
    assert cfg.veg.shadow_v_max == 50
    assert cfg.runtime.n_workers == 8
    assert cfg.waste.decide.auto_accept_enabled is False
    solver = cfg.route.solver
    assert (solver.method, solver.time_limit_s, solver.time_limit_final_s, solver.final) == ("auto", 30, 60, False)
    assert cfg.waste.sam3.model_ids == ("facebook/sam3",)
    assert cfg.grid.expected_tiles == 311 and cfg.grid.tile_px == 2048
    assert cfg.logging.tz == "Europe/Chisinau"


def test_paths_resolve_against_project_root(cfg: AppConfig) -> None:
    assert cfg.paths.project_root == PROJECT_ROOT
    assert cfg.paths.data_root == DATA_ROOT
    assert cfg.paths.work_dir == PROJECT_ROOT / "work"
    assert cfg.paths.publish_dir == REPO_ROOT
    assert cfg.paths.models_dir == PROJECT_ROOT / "models"
    assert cfg.paths.overrides == PROJECT_ROOT / "configs" / "overrides.yaml"
    assert cfg.paths.waste_confirmed == PROJECT_ROOT / "configs" / "waste_confirmed.csv"
    assert cfg.web.out_dir == REPO_ROOT / "src" / "Web" / "data"
    assert cfg.waste.sam3.local_checkpoint == REPO_ROOT / "models" / "sam3.1" / "sam3.1_multiplex.pt"
    assert cfg.paths.route_dir == DATA_ROOT / "02_route"
    assert cfg.paths.examples_dir == DATA_ROOT / "05_examples" / "siret3_examples_cvat"
    assert cfg.route.start_file == DATA_ROOT / "02_route" / "start.geojson"
    assert cfg.paths.tiles_zip_glob == "01_tiles/siret3_challenge_tiles_part*of5.zip"


def test_env_overrides_data_root_and_work_dir(tmp_path: Path) -> None:
    env = {"VINEYARD_DATA_ROOT": str(tmp_path / "data"), "VINEYARD_WORK_DIR": "scratch/work"}
    cfg = load_config(environ=env)
    assert cfg.paths.data_root == tmp_path / "data"
    assert cfg.paths.route_dir == tmp_path / "data" / "02_route"
    assert cfg.paths.work_dir == PROJECT_ROOT / "scratch" / "work"


def test_empty_env_value_is_ignored() -> None:
    assert load_config(environ={"VINEYARD_WORK_DIR": ""}).paths.work_dir == PROJECT_ROOT / "work"


def test_custom_project_root(tmp_path: Path) -> None:
    cfg = load_config(project_root=tmp_path, environ=NO_ENV)
    assert cfg.paths.work_dir == tmp_path / "work"
    assert cfg.paths.project_root == tmp_path


def test_set_overrides_are_yaml_typed() -> None:
    cfg = load_config(
        overrides=(
            "canopy.min_area_m2=0.25",
            "export.cvat.shape_source=auto",
            "import.accept_masks=false",
            "nn.fusion=C",
            "eval.example_tiles=[siret3_r006_c004]",
        ),
        environ=NO_ENV,
    )
    assert isinstance(cfg.canopy.min_area_m2, float) and cfg.canopy.min_area_m2 == 0.25
    assert cfg.export.cvat.shape_source == "auto"
    assert cfg.import_.accept_masks is False
    assert cfg.nn.fusion == "C"
    assert cfg.eval.example_tiles == ("siret3_r006_c004",)


def test_set_beats_env_and_env_beats_yaml(tmp_path: Path) -> None:
    env = {"VINEYARD_WORK_DIR": str(tmp_path / "env")}
    cfg = load_config(overrides=(f"paths.work_dir={tmp_path / 'cli'}",), environ=env)
    assert cfg.paths.work_dir == tmp_path / "cli"


@pytest.mark.parametrize(
    "override",
    [
        "canopy.min_area_mm=0.25",       # typo in leaf
        "canopyy.min_area_m2=0.25",      # typo in section
        "canopy.min_area_m2.x=1",        # descends into a scalar
        "canopy.min_area_m2",            # no '='
        "=3",                            # empty key
        "canopy..x=3",                   # empty key part
        "nn.fusion=Z",                   # not an A-E variant
        "export.cvat.shape_source=semi", # not a CVAT source
        "veg.otsu_fallback_veg_frac=[0.9, 0.1]",  # unordered range
        "grid.tile_px=1024",             # tile_m != gsd_m * tile_px
        "runtime.n_workers=0",
        "web.mask_px=1000",              # does not divide grid.tile_px
    ],
)
def test_bad_overrides_fail(override: str) -> None:
    with pytest.raises(ConfigError):
        load_config(overrides=(override,), environ=NO_ENV)


def test_unknown_yaml_key_fails(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path, "canopy:\n  bogus_key: 1\n")
    with pytest.raises(ConfigError, match="bogus_key"):
        load_config((DEFAULT_CONFIG, overlay), environ=NO_ENV)


def test_overlay_deep_merges(tmp_path: Path) -> None:
    overlay = _overlay(tmp_path, "route:\n  solver:\n    time_limit_s: 5\n")
    cfg = load_config((DEFAULT_CONFIG, overlay), environ=NO_ENV)
    assert cfg.route.solver.time_limit_s == 5
    assert cfg.route.solver.time_limit_final_s == 60


def test_partial_yaml_alone_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config((_overlay(tmp_path, "project:\n  name: x\n"),), environ=NO_ENV)


@pytest.mark.parametrize("text", ["a: [1, 2\n", "- 1\n- 2\n"])
def test_unreadable_yaml_fails(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError):
        load_config((DEFAULT_CONFIG, _overlay(tmp_path, text)), environ=NO_ENV)


def test_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="missing.yaml"):
        load_config((tmp_path / "missing.yaml",), environ=NO_ENV)


def test_empty_overlay_is_noop(tmp_path: Path, cfg: AppConfig) -> None:
    assert load_config((DEFAULT_CONFIG, _overlay(tmp_path, "")), environ=NO_ENV) == cfg


def test_config_is_frozen(cfg: AppConfig) -> None:
    with pytest.raises(ValidationError):
        cfg.canopy.min_area_m2 = 1.0  # type: ignore[misc]
    assert isinstance(cfg.rows.detect.end_percentiles, tuple)


def test_local_example_overlay_loads(tmp_path: Path) -> None:
    cfg = load_config((DEFAULT_CONFIG, PROJECT_ROOT / "configs" / "local.example.yaml"), environ=NO_ENV)
    assert cfg.runtime.n_workers >= 1


def test_deep_merge_and_apply_override_are_pure() -> None:
    base = {"a": {"b": 1, "c": [1]}, "d": 2}
    merged = deep_merge(base, {"a": {"b": 5}})
    assert merged == {"a": {"b": 5, "c": [1]}, "d": 2}
    assert base == {"a": {"b": 1, "c": [1]}, "d": 2}
    changed = apply_override(base, "a.e=[1, 2]")
    assert changed["a"]["e"] == [1, 2]
    assert "e" not in base["a"]
