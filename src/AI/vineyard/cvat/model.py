"""In-memory CVAT 1.1 document (immutable). Coordinates are continuous CVAT px (contract §1.2)."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final, Literal

from vineyard.contracts.ids import tile_id_from_file_name
from vineyard.errors import CvatFormatError

ShapeTagName = Literal["polygon", "polyline", "box"]
SHAPE_TAGS: Final[frozenset[str]] = frozenset({"polygon", "polyline", "box"})
BOX_POINTS: Final = 2
DEFAULT_SHAPE_SOURCE: Final = "manual"
CVAT_VERSION: Final = "1.1"


def _float_pair(p: object, label: str) -> tuple[float, float]:
    try:
        u, v = p  # type: ignore[misc]
        pair = (float(u), float(v))
    except (TypeError, ValueError) as exc:
        raise CvatFormatError("point must be a (u, v) pair of numbers", label=label, point=repr(p)) from exc
    if not (math.isfinite(pair[0]) and math.isfinite(pair[1])):
        raise CvatFormatError("non-finite point", label=label, point=pair)
    return pair


def _str_pair(a: object, label: str) -> tuple[str, str]:
    if not (isinstance(a, tuple | list) and len(a) == 2 and all(isinstance(x, str) for x in a)):
        raise CvatFormatError("attribute must be a (name, value) pair of str", label=label, attribute=repr(a))
    return (a[0], a[1])


@dataclass(frozen=True)
class CvatShape:
    """One shape. A box stores ((xtl, ytl), (xbr, ybr)); `ref_id` links back to the AnnSet, not serialized."""

    tag: ShapeTagName
    label: str
    points: tuple[tuple[float, float], ...]
    attributes: tuple[tuple[str, str], ...]
    source: str = DEFAULT_SHAPE_SOURCE
    occluded: int = 0
    z_order: int = 0
    ref_id: str | None = None

    def __post_init__(self) -> None:
        if self.tag not in SHAPE_TAGS:
            raise CvatFormatError("unsupported shape tag", tag=self.tag, label=self.label)
        points = tuple(_float_pair(p, self.label) for p in self.points)
        if self.tag == "box" and len(points) != BOX_POINTS:
            raise CvatFormatError("box needs exactly 2 points", label=self.label, n_points=len(points))
        if self.occluded not in (0, 1):
            raise CvatFormatError("occluded must be 0 or 1", label=self.label, occluded=self.occluded)
        attributes = tuple(_str_pair(a, self.label) for a in self.attributes)
        # Frozen dataclass: canonicalise sequences to tuples of floats/str once, at construction.
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "attributes", attributes)

    @property
    def attribute_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.attributes)

    @property
    def attribute_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.attributes))

    def attr(self, name: str) -> str | None:
        for key, value in self.attributes:
            if key == name:
                return value
        return None

    @property
    def box_xyxy(self) -> tuple[float, float, float, float]:
        if self.tag != "box":
            raise CvatFormatError("not a box", tag=self.tag, label=self.label)
        (xtl, ytl), (xbr, ybr) = self.points
        return (xtl, ytl, xbr, ybr)


@dataclass(frozen=True)
class CvatImage:
    id: int
    name: str
    width: int
    height: int
    shapes: tuple[CvatShape, ...]

    def __post_init__(self) -> None:
        if self.id < 0:
            raise CvatFormatError("image id must be >= 0", name=self.name, id=self.id)
        if self.width <= 0 or self.height <= 0:
            raise CvatFormatError("image size must be positive", name=self.name, width=self.width,
                                  height=self.height)
        if not isinstance(self.shapes, tuple):
            raise CvatFormatError("shapes must be a tuple", name=self.name)

    @property
    def tile_id(self) -> str:
        return tile_id_from_file_name(self.name)

    def count(self, label: str) -> int:
        return sum(1 for s in self.shapes if s.label == label)

    def with_shapes(self, shapes: tuple[CvatShape, ...]) -> "CvatImage":
        return replace(self, shapes=shapes)

    def with_id(self, image_id: int) -> "CvatImage":
        return replace(self, id=image_id)


@dataclass(frozen=True)
class CvatDocument:
    images: tuple[CvatImage, ...]
    meta_xml: str | None = None
    version: str = CVAT_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.images, tuple):
            raise CvatFormatError("images must be a tuple")

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(img.name for img in self.images)

    @property
    def n_shapes(self) -> int:
        return sum(len(img.shapes) for img in self.images)

    def image(self, name: str) -> CvatImage:
        for img in self.images:
            if img.name == name:
                return img
        raise KeyError(f"no image named {name!r}")
