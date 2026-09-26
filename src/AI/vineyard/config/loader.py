"""YAML deep-merge + env overrides + `--set a.b=<yaml>` overrides + path resolution -> AppConfig."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import yaml
from pydantic import ValidationError

from vineyard.config.model import AppConfig
from vineyard.config.sections_core import PROJECT_ROOT
from vineyard.errors import ConfigError

DEFAULT_CONFIG: Final[Path] = PROJECT_ROOT / "configs" / "default.yaml"
ENV_DATA_ROOT: Final = "VINEYARD_DATA_ROOT"
ENV_WORK_DIR: Final = "VINEYARD_WORK_DIR"
ENV_PATH_OVERRIDES: Final[Mapping[str, str]] = {ENV_DATA_ROOT: "paths.data_root", ENV_WORK_DIR: "paths.work_dir"}
PROJECT_RELATIVE_KEYS: Final = (
    "paths.data_root",
    "paths.work_dir",
    "paths.models_dir",
    "paths.publish_dir",
    "paths.overrides",
    "paths.waste_confirmed",
    "web.out_dir",
    "web.tile_review",
    "farms.osm_highways",
    "farms.cadastre_parcels",
    "waste.sam3.local_checkpoint",
    "nn.pseudolabels.train_tiles_file",
    "eval.baseline_file",
)
PROJECT_RELATIVE_LIST_KEYS: Final = ("import.files",)
DATA_RELATIVE_KEYS: Final = ("paths.route_dir", "paths.examples_dir", "route.start_file")
MAX_REPORTED_ERRORS: Final = 8
_MISSING: Final = object()


def read_yaml(path: Path) -> dict[str, Any]:
    """Top-level mapping of a YAML file; an empty file is {}."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError("config file not found", path=str(path)) from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}", path=str(path)) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("config file must hold a mapping at top level", path=str(path))
    return data


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """New dict: nested mappings merge recursively, everything else is replaced by `overlay`."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _split_key(key: str) -> list[str]:
    parts = key.split(".")
    if not key or any(not part for part in parts):
        raise ConfigError("empty config key part", key=key)
    return parts


def get_dotted(tree: Mapping[str, Any], key: str) -> Any:
    node: Any = tree
    for part in _split_key(key):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def set_dotted(tree: Mapping[str, Any], key: str, value: Any) -> dict[str, Any]:
    """New tree with `key` set; copies each mapping along the path, creating missing ones."""
    head, *rest = _split_key(key)
    updated = dict(tree)
    if not rest:
        updated[head] = value
        return updated
    child = tree.get(head, {})
    if not isinstance(child, Mapping):
        raise ConfigError("cannot set a key below a scalar value", key=key, parent=head)
    updated[head] = set_dotted(child, ".".join(rest), value)
    return updated


def apply_override(tree: Mapping[str, Any], text: str) -> dict[str, Any]:
    """Apply one `a.b.c=<yaml value>` override (the value is parsed as YAML: 0.25, true, [1, 2], null)."""
    key, sep, raw = text.partition("=")
    if not sep:
        raise ConfigError("override must look like key=value", override=text)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"override value is not valid YAML: {exc}", override=text) from exc
    return set_dotted(tree, key.strip(), value)


def apply_env_overrides(tree: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    updated = dict(tree)
    for env_name, key in ENV_PATH_OVERRIDES.items():
        value = environ.get(env_name, "").strip()
        if value:
            updated = set_dotted(updated, key, value)
    return updated


def _resolve(value: Any, base: Path, key: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise ConfigError("path value must be a string", key=key, value=repr(value))
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else Path(os.path.normpath(base / path)))


def _resolve_keys(tree: dict[str, Any], keys: Sequence[str], base: Path) -> dict[str, Any]:
    for key in keys:
        value = get_dotted(tree, key)
        if value is not _MISSING:
            tree = set_dotted(tree, key, _resolve(value, base, key))
    return tree


def resolve_paths(tree: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Project-relative paths against `project_root`; data paths against the resolved data_root."""
    resolved = _resolve_keys(dict(tree), PROJECT_RELATIVE_KEYS, project_root)
    for key in PROJECT_RELATIVE_LIST_KEYS:
        values = get_dotted(resolved, key)
        if isinstance(values, list):
            resolved = set_dotted(resolved, key, [_resolve(v, project_root, key) for v in values])
    data_root = get_dotted(resolved, "paths.data_root")
    if data_root is not _MISSING and data_root is not None:
        resolved = _resolve_keys(resolved, DATA_RELATIVE_KEYS, Path(data_root))
    return set_dotted(resolved, "paths.project_root", str(project_root))


def _validation_message(exc: ValidationError) -> str:
    items = [
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:MAX_REPORTED_ERRORS]
    ]
    more = exc.error_count() - len(items)
    return "invalid config: " + "; ".join(items) + (f"; (+{more} more)" if more > 0 else "")


def load_config(
    paths: Sequence[Path] = (DEFAULT_CONFIG,),
    overrides: Sequence[str] = (),
    *,
    project_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Merge YAML files in order, then env overrides, then `--set` overrides; validate strictly."""
    root = (project_root or PROJECT_ROOT).resolve()
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, read_yaml(Path(path)))
    merged = apply_env_overrides(merged, os.environ if environ is None else environ)
    for override in overrides:
        merged = apply_override(merged, override)
    merged = resolve_paths(merged, root)
    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(_validation_message(exc), sources=[str(p) for p in paths]) from exc
