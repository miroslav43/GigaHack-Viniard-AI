"""`vineyard cvat ...` sub-app: validate, make-test-zip, roundtrip-examples."""

import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vineyard.config import load_config
from vineyard.cvat.cli import app, expected_sha256, roundtrip_checks, zip_tile_ids

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("validate", "make-test-zip", "roundtrip-examples"):
        assert name in result.output


def test_zip_tile_ids(tmp_path: Path, make_zip) -> None:
    z = make_zip(tmp_path / "u.zip", {"annotations.xml": b"<a/>", "images/siret3_r021_c012.tif": b"x",
                                      "images/readme.txt": b"y"})
    assert zip_tile_ids([z]) == ("siret3_r021_c012",)


def test_expected_sha256_from_tile_zips(tmp_path: Path, make_zip, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    make_zip(data / "01_tiles" / "siret3_challenge_tiles_part1of5.zip", {"siret3_r021_c012.tif": b"abc"})
    monkeypatch.setenv("VINEYARD_DATA_ROOT", str(data))
    monkeypatch.setenv("VINEYARD_WORK_DIR", str(tmp_path / "work"))
    shas = expected_sha256(load_config(), ("siret3_r021_c012", "siret3_r039_c033"))
    assert shas == {"siret3_r021_c012": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}


@pytest.mark.examples
def test_roundtrip_checks_pass() -> None:
    checks = roundtrip_checks(load_config())
    failed = [c for c in checks if not c.ok]
    assert not failed, failed
    names = [c.name for c in checks]
    assert names[0] == "xml_byte_identical"
    assert "counts_siret3_r021_c012" in names and "sums_siret3_r006_c004" in names


@pytest.mark.examples
def test_roundtrip_command() -> None:
    result = runner.invoke(app, ["roundtrip-examples"])
    assert result.exit_code == 0, result.output
    assert "xml_byte_identical" in result.output


@pytest.mark.examples
@pytest.mark.needs_tiles
def test_make_test_zip_then_validate(tmp_path: Path) -> None:
    out = tmp_path / "t.zip"
    made = runner.invoke(app, ["make-test-zip", "--out", str(out)])
    assert made.exit_code == 0, made.output
    assert out.is_file() and str(out) in made.output
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist()[0] == "annotations.xml"
    checked = runner.invoke(app, ["validate", str(out)])
    assert checked.exit_code == 0, checked.output
    assert "0 erori, 0 avertismente" in checked.output


def test_validate_reports_errors(tmp_path: Path, make_zip) -> None:
    bad = make_zip(tmp_path / "bad.zip", {".DS_Store": b"x", "annotations.xml": b"<annotations/>"})
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "erori" in result.output


def test_validate_missing_file_fails(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "none.zip")])
    assert result.exit_code != 0
