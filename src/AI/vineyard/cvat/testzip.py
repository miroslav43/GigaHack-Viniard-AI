"""Marcaj format-test ZIP (arch §4.13.5, milestone M1): the 2 example tiles with their reference
annotations, 1 synthetic waste box on r021 and 1 EMPTY tile, in the upload layout
(annotations.xml DEFLATED first, then images/<tile>.tif STORED, image ids 0..n-1 by file name).
"""

import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from vineyard.config import AppConfig
from vineyard.contracts.ids import file_name_from_tile_id
from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.cvat.packer import write_upload_zip
from vineyard.cvat.reader import image_key, read_cvat_file
from vineyard.cvat.writer import serialize_document
from vineyard.errors import CvatFormatError
from vineyard.geo.tiling import TILE_PX

EMPTY_TILE: Final = "siret3_r005_c004"
WASTE_TILE: Final = "siret3_r021_c012"
# 1.0 x 0.75 m inside the widest bare interrow of r021, >= 34 px from every reference canopy.
WASTE_BOX_PX: Final = (1304.0, 934.0, 1344.0, 964.0)
WASTE_LABEL: Final = "waste"
VINEYARD_ID: Final = "vineyard_id"
TEST_ZIP_NAME: Final = "siret3_format_test.zip"
TEST_ZIP_SUBDIR: Final = ("exports", "marcaj_test")
EXAMPLES_XML: Final = "annotations.xml"
EXAMPLES_IMAGES: Final = "images"
TILES_SUBDIR: Final = "tiles"
MIN_RING_POINTS: Final = 3


def default_test_zip_path(cfg: AppConfig) -> Path:
    return Path(cfg.paths.work_dir).joinpath(*TEST_ZIP_SUBDIR, TEST_ZIP_NAME)


def _waste_box(shape_source: str, vineyard_id: str) -> CvatShape:
    (xtl, ytl, xbr, ybr) = WASTE_BOX_PX
    return CvatShape("box", WASTE_LABEL, ((xtl, ytl), (xbr, ybr)), ((VINEYARD_ID, vineyard_id),),
                     source=shape_source)


def _shape_geometry_px(shape: CvatShape) -> BaseGeometry:
    if shape.tag == "box":
        return box(*shape.box_xyxy)
    if shape.tag == "polygon" and len(shape.points) >= MIN_RING_POINTS:
        return Polygon(shape.points)
    return LineString(shape.points) if len(shape.points) > 1 else Point(shape.points[0])


def nearest_vineyard_id(image: CvatImage, box_px: tuple[float, float, float, float]) -> str:
    """vineyard_id of the annotated shape nearest to the box (px distance, ties -> XML order); '' if none."""
    target = box(*box_px)
    candidates = [(_shape_geometry_px(s).distance(target), k, (s.attr(VINEYARD_ID) or "").strip())
                  for k, s in enumerate(image.shapes) if s.points]
    named = [c for c in candidates if c[2]]
    return min(named)[2] if named else ""


def build_test_document(examples: CvatDocument, *, shape_source: str,
                        waste_vineyard_id: str | None = None) -> CvatDocument:
    """Examples + waste box appended on WASTE_TILE (shape order row, interrow, vineyard, waste) + empty
    EMPTY_TILE; images sorted by file name and numbered 0..n-1; the verbatim example <meta> block.
    `waste_vineyard_id=None` takes the block the box lies in (nearest annotated shape of that tile)."""
    by_name = {image_key(img.name): img for img in examples.images}
    waste_name = file_name_from_tile_id(WASTE_TILE)
    if waste_name not in by_name:
        raise CvatFormatError("examples document has no image for the waste tile", tile_id=WASTE_TILE)
    host = by_name[waste_name]
    vid = nearest_vineyard_id(host, WASTE_BOX_PX) if waste_vineyard_id is None else waste_vineyard_id
    with_box = host.with_shapes((*host.shapes, _waste_box(shape_source, vid)))
    empty_name = file_name_from_tile_id(EMPTY_TILE)
    images = {**by_name, waste_name: with_box, empty_name: CvatImage(0, empty_name, TILE_PX, TILE_PX, ())}
    return CvatDocument(tuple(_as_upload(images[name], name, i) for i, name in enumerate(sorted(images))))


def _as_upload(img: CvatImage, name: str, image_id: int) -> CvatImage:
    """Image under its plain `<tile>.tif` name (no `images/` prefix) with its position as id."""
    return CvatImage(image_id, name, img.width, img.height, img.shapes)


def locate_tile(tile_id: str, *, search_dirs: Sequence[Path], zip_paths: Sequence[Path], scratch: Path) -> Path:
    """First `<tile>.tif` found in search_dirs, else extracted from the first tile ZIP holding it."""
    file_name = file_name_from_tile_id(tile_id)
    for directory in search_dirs:
        candidate = Path(directory) / file_name
        if candidate.is_file():
            return candidate
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if Path(n).name == file_name]
            if members:
                target = Path(scratch) / file_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(members[0]))
                return target
    raise CvatFormatError("tile not found in the examples, work/tiles or the tile ZIPs", tile_id=tile_id)


def _tile_zips(cfg: AppConfig) -> tuple[Path, ...]:
    return tuple(sorted(Path(cfg.paths.data_root).glob(cfg.paths.tiles_zip_glob)))


def make_test_zip(out: Path, cfg: AppConfig, *, waste_vineyard_id: str | None = None) -> Path:
    """Write the format-test ZIP to `out` (atomic, deterministic) and return `out`."""
    examples_dir = Path(cfg.paths.examples_dir)
    examples, _ = read_cvat_file(examples_dir / EXAMPLES_XML)
    doc = build_test_document(examples, shape_source=cfg.export.cvat.shape_source,
                              waste_vineyard_id=waste_vineyard_id)
    xml = serialize_document(doc, decimals=cfg.export.cvat.coord_decimals)
    search = (examples_dir / EXAMPLES_IMAGES, Path(cfg.paths.work_dir) / TILES_SUBDIR)
    with tempfile.TemporaryDirectory(prefix="vineyard-testzip-") as scratch:
        images = [(img.name, locate_tile(img.tile_id, search_dirs=search, zip_paths=_tile_zips(cfg),
                                         scratch=Path(scratch))) for img in doc.images]
        write_upload_zip(Path(out), xml, images, deflate_level=cfg.export.cvat.xml_deflate_level)
    return Path(out)
