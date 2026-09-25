import dataclasses

import pytest

from vineyard.cvat.model import CvatDocument, CvatImage, CvatShape
from vineyard.errors import CvatFormatError

TILE_NAME = "siret3_r021_c012.tif"


def _row() -> CvatShape:
    return CvatShape(
        tag="polyline",
        label="row",
        points=((2048.0, 32.1), (2017.8, 0.0)),
        attributes=(("vineyard_id", "V01"), ("row_id", "V01-R01"), ("row_structure", "regular")),
    )


def _box() -> CvatShape:
    return CvatShape(tag="box", label="waste", points=((1.0, 2.0), (3.0, 4.0)), attributes=(("vineyard_id", ""),))


def test_shape_defaults_and_attrs() -> None:
    shape = _row()
    assert (shape.source, shape.occluded, shape.z_order, shape.ref_id) == ("manual", 0, 0, None)
    assert shape.attr("row_id") == "V01-R01"
    assert shape.attr("missing") is None
    assert shape.attribute_names == ("vineyard_id", "row_id", "row_structure")
    assert dict(shape.attribute_map) == {"vineyard_id": "V01", "row_id": "V01-R01", "row_structure": "regular"}


def test_shape_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _row().label = "vineyard"  # type: ignore[misc]


def test_shape_coerces_points_to_float_tuples() -> None:
    shape = CvatShape(tag="polygon", label="vineyard", points=[[0, 0], [1, 0], [1, 1]],  # type: ignore[arg-type]
                      attributes=[["vineyard_id", "V01"]])  # type: ignore[arg-type]
    assert shape.points == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    assert isinstance(shape.points[0][0], float)
    assert shape.attributes == (("vineyard_id", "V01"),)


def test_box_bounds() -> None:
    assert _box().box_xyxy == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(CvatFormatError):
        _row().box_xyxy  # noqa: B018


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tag": "mask"},
        {"tag": "box", "points": ((1.0, 2.0),)},
        {"points": ((1.0, 2.0, 3.0),)},
        {"points": (("a", 2.0),)},
        {"points": ((float("nan"), 2.0), (1.0, 1.0))},
        {"attributes": (("only_name",),)},
        {"occluded": 2},
    ],
)
def test_shape_validation(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "tag": "polyline", "label": "row", "points": ((0.0, 0.0), (1.0, 1.0)), "attributes": (),
    }
    base.update(kwargs)
    with pytest.raises(CvatFormatError):
        CvatShape(**base)  # type: ignore[arg-type]


def test_image_and_document() -> None:
    image = CvatImage(id=0, name=TILE_NAME, width=2048, height=2048, shapes=(_row(), _box()))
    assert image.tile_id == "siret3_r021_c012"
    assert image.count("row") == 1
    assert image.count("vineyard") == 0
    empty = image.with_shapes(())
    assert empty.shapes == () and image.shapes != ()
    doc = CvatDocument(images=(image, empty.with_id(1)))
    assert doc.version == "1.1" and doc.meta_xml is None
    assert doc.image_names == (TILE_NAME, TILE_NAME)
    assert doc.image(TILE_NAME) is image
    with pytest.raises(KeyError):
        doc.image("nope.tif")
    assert doc.n_shapes == 2


def test_image_validation() -> None:
    with pytest.raises(CvatFormatError):
        CvatImage(id=-1, name=TILE_NAME, width=2048, height=2048, shapes=())
    with pytest.raises(CvatFormatError):
        CvatImage(id=0, name=TILE_NAME, width=0, height=2048, shapes=())
    with pytest.raises(CvatFormatError):
        CvatImage(id=0, name=TILE_NAME, width=2048, height=2048, shapes=[_row()])  # type: ignore[arg-type]
    with pytest.raises(CvatFormatError):
        CvatDocument(images=[])  # type: ignore[arg-type]
