"""Canonical JSON of config subtrees and their sha1 (contract §3.3 cache keys)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from vineyard.errors import ConfigError

if TYPE_CHECKING:
    from vineyard.config.model import AppConfig

FIELD_ALIASES: Final = {"import_": "import", "validate_": "validate"}


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NaN rejected."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def resolved_config_dict(cfg: AppConfig) -> dict[str, Any]:
    """The whole config as plain JSON-able data, keyed by YAML names (aliases applied)."""
    return cfg.model_dump(mode="json", by_alias=True)


def canonical_key(key: str) -> str:
    """Dotted key with Python field names mapped to YAML names ("import_" -> "import")."""
    parts = key.split(".")
    if not key or any(not part for part in parts):
        raise ConfigError("empty config key part", key=key)
    return ".".join(FIELD_ALIASES.get(part, part) for part in parts)


def _walk(tree: dict[str, Any], key: str) -> Any:
    node: Any = tree
    for part in canonical_key(key).split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError("unknown config key", key=key)
        node = node[part]
    return node


def cfg_subtree(cfg: AppConfig, key: str) -> Any:
    """JSON-able copy of the dotted subtree `key` (e.g. "rows.detect", "import")."""
    return copy.deepcopy(_walk(resolved_config_dict(cfg), key))


def cfg_hash(cfg: AppConfig, keys: Sequence[str]) -> str:
    """sha1 of the canonical JSON of the given dotted subtrees (order- and alias-independent)."""
    tree = resolved_config_dict(cfg)
    names = sorted({canonical_key(key) for key in keys})
    payload = {key: _walk(tree, key) for key in names}
    return hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()
