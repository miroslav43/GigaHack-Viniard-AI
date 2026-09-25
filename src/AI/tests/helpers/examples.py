"""Minimal lxml reader for the organizers' example annotations.xml (test oracle, independent of cvat/).

Points are CVAT continuous px (u, v); `to_utm` applies the contract §1.2 formula via geo.tiling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from lxml import etree
from shapely.geometry import LineString, Polygon

from vineyard.contracts.ids import tile_id_from_file_name
from vineyard.geo.tiling import px_to_utm, tile_ref

SHAPE_TAGS = ("polygon", "polyline", "box")


@dataclass(frozen=True, eq=False)
class ExampleShape:
    tag: str
    label: str
    points: np.ndarray  # (N, 2) px; box = [[xtl, ytl], [xbr, ybr]]
    attributes: Mapping[str, str]


@dataclass(frozen=True, eq=False)
class ExampleImage:
    image_id: int
    name: str
    tile_id: str
    width: int
    height: int
    shapes: tuple[ExampleShape, ...]

    def by_label(self, label: str) -> tuple[ExampleShape, ...]:
        return tuple(s for s in self.shapes if s.label == label)

    def points(self, label: str) -> list[np.ndarray]:
        return [s.points for s in self.by_label(label)]


def parse_points(text: str) -> np.ndarray:
    """'x1,y1;x2,y2;...' -> (N, 2) float64."""
    return np.array([[float(v) for v in pair.split(",")] for pair in text.split(";")], dtype=np.float64)


def _shape_points(el: etree._Element) -> np.ndarray:
    if el.tag == "box":
        keys = ("xtl", "ytl", "xbr", "ybr")
        xtl, ytl, xbr, ybr = (float(el.get(k)) for k in keys)
        return np.array([[xtl, ytl], [xbr, ybr]], dtype=np.float64)
    return parse_points(el.get("points"))


def _parse_image(el: etree._Element) -> ExampleImage:
    shapes = tuple(
        ExampleShape(
            tag=child.tag,
            label=child.get("label"),
            points=_shape_points(child),
            attributes=MappingProxyType({a.get("name"): (a.text or "") for a in child.findall("attribute")}),
        )
        for child in el
        if child.tag in SHAPE_TAGS
    )
    name = el.get("name")
    return ExampleImage(
        image_id=int(el.get("id")), name=name, tile_id=tile_id_from_file_name(name),
        width=int(el.get("width")), height=int(el.get("height")), shapes=shapes,
    )


def load_examples(xml: bytes) -> dict[str, ExampleImage]:
    """Images keyed by tile_id, in document order."""
    root = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False, huge_tree=True))
    images = [_parse_image(el) for el in root.iter("image")]
    return {img.tile_id: img for img in images}


def to_utm(tile_id: str, points_px: np.ndarray) -> np.ndarray:
    return px_to_utm(tile_ref(tile_id), points_px)


def canopy_polygons_utm(img: ExampleImage) -> list[Polygon]:
    return [Polygon(to_utm(img.tile_id, p)) for p in img.points("vineyard")]


def interrow_polygons_utm(img: ExampleImage) -> list[Polygon]:
    return [Polygon(to_utm(img.tile_id, p)) for p in img.points("interrow_area")]


def row_lines_utm(img: ExampleImage) -> list[LineString]:
    return [LineString(to_utm(img.tile_id, p)) for p in img.points("row")]
