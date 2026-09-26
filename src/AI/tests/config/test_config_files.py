"""The committed config templates: overrides.yaml, waste_confirmed.csv, default.yaml completeness."""

import csv

import yaml

from vineyard import _env
from vineyard.config import DEFAULT_CONFIG, PROJECT_ROOT, AppConfig, load_config, resolved_config_dict
from vineyard.config.loader import read_yaml

CONFIGS = PROJECT_ROOT / "configs"


def _leaf_keys(tree: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value and not path.endswith(("enum_synonyms", "prompt_categories",
                                                                   "colors_bgr", "thread_env")):
            keys |= _leaf_keys(value, path + ".")
        else:
            keys.add(path)
    return keys


def test_overrides_template_is_empty() -> None:
    data = yaml.safe_load((CONFIGS / "overrides.yaml").read_text(encoding="utf-8"))
    assert data == {
        "version": 1,
        "force_empty_tiles": [],
        "exclude_areas": [],
        "delete_rows": [],
        "extend_rows": [],
        "add_rows": [],
    }


def test_waste_confirmed_header_and_row_width() -> None:
    with (CONFIGS / "waste_confirmed.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header = ["tile_id", "xtl", "ytl", "xbr", "ybr", "decision", "category", "reviewer", "note"]
    assert rows[0] == header
    assert all(len(r) == len(header) for r in rows[1:])


def test_default_yaml_lists_every_model_key() -> None:
    yaml_keys = _leaf_keys(read_yaml(DEFAULT_CONFIG))
    model_keys = _leaf_keys(resolved_config_dict(load_config(environ={})))
    assert model_keys - {"paths.project_root"} == yaml_keys


def test_thread_env_matches_env_module() -> None:
    cfg: AppConfig = load_config(environ={})
    assert cfg.runtime.thread_env == dict(_env.THREAD_ENV_DEFAULTS)
