"""Waste targets (arch §4.10): one target at the centre of each waste object.

A box cut by a tile edge is stored as two boxes `W0001a` / `W0001b` (contract §1.6); they are merged
back into one object `W0001` whose centre is the centre of the union of both boxes.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Final

import geopandas as gpd
import shapely
from shapely.geometry.base import BaseGeometry

from vineyard.contracts.enums import TargetKind
from vineyard.contracts.ids import format_waste_id, parse_waste_id
from vineyard.errors import SchemaError
from vineyard.route.target_rules import PRIORITY_HIGH, TargetDraft

WASTE_COLUMNS: Final = ("waste_id", "vineyard_id")


def _object_id(waste_id: str) -> str:
    """W0001a / W0001b → W0001; ids outside the contract pattern are kept as they are."""
    try:
        number, _ = parse_waste_id(waste_id)
    except SchemaError:
        return waste_id
    return format_waste_id(number)


def _centre(geoms: list[BaseGeometry]) -> tuple[float, float]:
    minx, miny, maxx, maxy = shapely.union_all(geoms).bounds
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0


def _vineyard_of(values: list[object]) -> str:
    named = sorted(str(v) for v in values if isinstance(v, str) and v.strip())
    return named[0] if named else ""


def waste_drafts(waste: gpd.GeoDataFrame) -> tuple[TargetDraft, ...]:
    """One WST draft per waste object (split boxes merged), ordered by object id."""
    missing = [c for c in WASTE_COLUMNS if c not in waste.columns]
    if missing:
        raise SchemaError("waste layer lacks columns needed for targets", columns=missing)
    groups: dict[str, list[tuple[str, object, BaseGeometry]]] = defaultdict(list)
    for wid, vid, geom in zip(waste["waste_id"], waste["vineyard_id"], waste.geometry, strict=True):
        groups[_object_id(str(wid))].append((str(wid), vid, geom))
    drafts = []
    for object_id in sorted(groups):
        parts = sorted(groups[object_id], key=lambda p: p[0])
        x, y = _centre([p[2] for p in parts])
        drafts.append(TargetDraft(
            kind=TargetKind.WASTE, x=x, y=y, vineyard_id=_vineyard_of([p[1] for p in parts]),
            priority=PRIORITY_HIGH, reason="waste " + "+".join(p[0] for p in parts), along_m=math.nan,
            waste_id=object_id, sort_ref=object_id))
    return tuple(drafts)
