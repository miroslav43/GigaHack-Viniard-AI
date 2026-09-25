"""cfg_hash / cfg_subtree / resolved_config_dict."""

import json

import pytest

from vineyard.config import (
    AppConfig,
    canonical_json,
    cfg_hash,
    cfg_subtree,
    load_config,
    resolved_config_dict,
)
from vineyard.errors import ConfigError

NO_ENV: dict[str, str] = {}


@pytest.fixture(scope="module")
def cfg() -> AppConfig:
    return load_config(environ=NO_ENV)


def test_hash_is_stable_across_loads(cfg: AppConfig) -> None:
    again = load_config(environ=NO_ENV)
    assert cfg_hash(cfg, ["canopy"]) == cfg_hash(again, ["canopy"])
    assert len(cfg_hash(cfg, ["canopy"])) == 40


def test_hash_is_sensitive_to_its_subtree_only(cfg: AppConfig) -> None:
    canopy_changed = load_config(overrides=("canopy.min_area_m2=0.2",), environ=NO_ENV)
    route_changed = load_config(overrides=("route.solver.time_limit_s=5",), environ=NO_ENV)
    assert cfg_hash(canopy_changed, ["canopy"]) != cfg_hash(cfg, ["canopy"])
    assert cfg_hash(route_changed, ["canopy"]) == cfg_hash(cfg, ["canopy"])
    assert cfg_hash(route_changed, ["route.solver"]) != cfg_hash(cfg, ["route.solver"])


def test_hash_ignores_key_order_duplicates_and_aliases(cfg: AppConfig) -> None:
    assert cfg_hash(cfg, ["veg", "nodata"]) == cfg_hash(cfg, ["nodata", "veg", "veg"])
    assert cfg_hash(cfg, ["import_"]) == cfg_hash(cfg, ["import"])
    assert cfg_hash(cfg, ["route.validate_"]) == cfg_hash(cfg, ["route.validate"])


def test_hash_unknown_key_fails(cfg: AppConfig) -> None:
    with pytest.raises(ConfigError, match="canopyy"):
        cfg_hash(cfg, ["canopyy"])
    with pytest.raises(ConfigError):
        cfg_hash(cfg, ["canopy..x"])
    with pytest.raises(ConfigError):
        cfg_subtree(cfg, "canopy.min_area_m2.x")


def test_subtree_is_a_json_copy(cfg: AppConfig) -> None:
    sub = cfg_subtree(cfg, "rows.detect")
    assert sub["spacing_min_m"] == 1.8
    assert sub["end_percentiles"] == [0.5, 99.5]
    sub["spacing_min_m"] = 99.0
    assert cfg.rows.detect.spacing_min_m == 1.8
    assert cfg_subtree(cfg, "import")["duplicate_policy"] == "last_wins"
    assert cfg_subtree(cfg, "canopy.corridor_half_m") == 0.30


def test_resolved_dict_uses_yaml_names_and_is_json(cfg: AppConfig) -> None:
    tree = resolved_config_dict(cfg)
    assert "import" in tree and "import_" not in tree
    assert "validate" in tree["route"]
    assert isinstance(tree["paths"]["data_root"], str)
    json.loads(canonical_json(tree))


def test_canonical_json_rejects_nan() -> None:
    assert canonical_json({"b": 1, "a": [1.5]}) == '{"a":[1.5],"b":1}'
    with pytest.raises(ValueError):
        canonical_json({"a": float("nan")})
