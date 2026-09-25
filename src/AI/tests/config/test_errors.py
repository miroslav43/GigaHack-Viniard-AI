"""Error hierarchy: context kwargs, readable str, exit codes, pickling."""

import pickle

import pytest

from vineyard import errors


def test_context_is_kept_and_rendered_sorted() -> None:
    err = errors.SchemaError("coloana lipsește", layer="canopies", tile_id="siret3_r021_c012")
    assert err.message == "coloana lipsește"
    assert dict(err.context) == {"layer": "canopies", "tile_id": "siret3_r021_c012"}
    assert str(err) == "coloana lipsește [layer=canopies, tile_id=siret3_r021_c012]"


def test_message_without_context() -> None:
    assert str(errors.ConfigError("x")) == "x"


def test_context_is_read_only() -> None:
    err = errors.IngestError("bad", tile_id="t")
    with pytest.raises(TypeError):
        err.context["tile_id"] = "u"  # type: ignore[index]


@pytest.mark.parametrize(
    ("cls", "base"),
    [
        (errors.ConfigError, errors.VineyardError),
        (errors.SchemaError, errors.VineyardError),
        (errors.IngestError, errors.VineyardError),
        (errors.StageError, errors.VineyardError),
        (errors.StageNotImplemented, errors.StageError),
        (errors.ExportBlocked, errors.VineyardError),
        (errors.CvatFormatError, errors.VineyardError),
        (errors.RouteValidationError, errors.VineyardError),
    ],
)
def test_hierarchy(cls: type, base: type) -> None:
    assert issubclass(cls, base)


def test_exit_codes() -> None:
    assert errors.VineyardError("x").exit_code == 1
    assert errors.ExportBlocked("x").exit_code == 2
    assert errors.StageNotImplemented("x", stage="derive").exit_code == 3


def test_pickle_round_trip_keeps_context() -> None:
    err = errors.StageError("eșec", stage="canopy", tile_id="t1")
    back = pickle.loads(pickle.dumps(err))
    assert isinstance(back, errors.StageError)
    assert back.message == "eșec"
    assert dict(back.context) == {"stage": "canopy", "tile_id": "t1"}
