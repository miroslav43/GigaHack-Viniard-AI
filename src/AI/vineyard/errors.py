"""Exception hierarchy. Every error carries keyword context (tile_id, layer, key, ...)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar


class VineyardError(Exception):
    """Base error; `context` is a read-only mapping rendered into str()."""

    exit_code: ClassVar[int] = 1

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: Mapping[str, Any] = MappingProxyType(dict(context))

    def __str__(self) -> str:
        if not self.context:
            return self.message
        parts = ", ".join(f"{key}={self.context[key]}" for key in sorted(self.context))
        return f"{self.message} [{parts}]"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild_error, (type(self), self.message, dict(self.context)))


def _rebuild_error(cls: type[VineyardError], message: str, context: dict[str, Any]) -> VineyardError:
    return cls(message, **context)


class ConfigError(VineyardError):
    """Invalid or unreadable configuration (YAML, --set, env override)."""


class SchemaError(VineyardError):
    """A layer violates its contract schema."""


class IngestError(VineyardError):
    """Input tiles or route inputs do not match the contract."""


class StageError(VineyardError):
    """A pipeline stage failed or is misconfigured."""


class StageNotImplemented(StageError):
    """The stage module does not exist yet."""

    exit_code: ClassVar[int] = 3


class ExportBlocked(VineyardError):
    """The CVAT export refused to publish its ZIPs."""

    exit_code: ClassVar[int] = 2


class CvatFormatError(VineyardError):
    """A CVAT XML/ZIP is structurally invalid."""


class RouteValidationError(VineyardError):
    """The walking route failed validation."""
