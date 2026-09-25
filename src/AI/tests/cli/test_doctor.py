"""doctor checks with fakes (torch is never imported in this process)."""

import subprocess
import sys
from pathlib import Path

import pytest

from vineyard import _env, doctor
from vineyard.config import AppConfig, load_config
from vineyard.doctor import Check


def _synthetic_cfg(tmp_path: Path, *, complete: bool) -> AppConfig:
    data = tmp_path / "data & info"
    if complete:
        (data / "01_tiles").mkdir(parents=True)
        for i in range(1, 6):
            (data / "01_tiles" / f"siret3_challenge_tiles_part{i}of5.zip").write_bytes(b"")
        images = data / "05_examples" / "siret3_examples_cvat" / "images"
        images.mkdir(parents=True)
        (images.parent / "annotations.xml").write_text("<annotations/>", encoding="utf-8")
        for name in ("siret3_r006_c004.tif", "siret3_r021_c012.tif"):
            (images / name).write_bytes(b"")
        (data / "02_route").mkdir()
        for name in doctor.ROUTE_FILES:
            (data / "02_route" / name).write_text("{}", encoding="utf-8")
    return load_config(environ={"VINEYARD_DATA_ROOT": str(data), "VINEYARD_WORK_DIR": str(tmp_path / "work")})


def test_check_python() -> None:
    assert doctor.check_python((3, 12, 13)).ok
    assert not doctor.check_python((3, 14, 0)).ok
    assert doctor.check_python().required


def test_check_imports_real_and_failing() -> None:
    assert all(c.ok for c in doctor.check_imports())

    def importer(name: str) -> object:
        if name == "cv2":
            raise ImportError("no cv2")
        return sys.modules[__name__]

    checks = doctor.check_imports(("numpy", "cv2"), importer=importer)  # type: ignore[arg-type]
    assert [c.ok for c in checks] == [True, False]
    assert "no cv2" in checks[1].detail


def test_check_geo_versions() -> None:
    check = doctor.check_geo_versions()
    assert check.ok and "GDAL" in check.detail and "PROJ" in check.detail


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        ((True, "2.14.0 True"), [("torch (extra nn)", True), ("MPS", True)]),
        ((True, "2.14.0 False"), [("torch (extra nn)", True), ("MPS", False)]),
        ((False, "ModuleNotFoundError: torch"), [("torch (extra nn)", False)]),
    ],
)
def test_check_torch_is_optional(probe: tuple[bool, str], expected: list[tuple[str, bool]]) -> None:
    checks = doctor.check_torch(lambda: probe)
    assert [(c.name, c.ok) for c in checks] == expected
    assert not any(c.required for c in checks)


def test_probe_torch_outcomes(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter([
        subprocess.CompletedProcess([], 0, stdout="2.14.0 True\n", stderr=""),
        subprocess.CompletedProcess([], 1, stdout="", stderr="Traceback\nModuleNotFoundError: torch\n"),
        subprocess.CompletedProcess([], 1, stdout="", stderr=""),
    ])

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return next(outcomes)

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    assert doctor.probe_torch() == (True, "2.14.0 True")
    assert doctor.probe_torch() == (False, "ModuleNotFoundError: torch")
    assert doctor.probe_torch() == (False, "exit 1")

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="python", timeout=1)

    monkeypatch.setattr(doctor.subprocess, "run", timeout)
    ok, detail = doctor.probe_torch(timeout_s=1)
    assert not ok and "timeout" in detail


def test_check_conda_env() -> None:
    bad = doctor.check_conda_env({"PROJ_DATA": "/opt/anaconda3/share/proj"}, _env.EnvPlan())
    assert not bad.ok and bad.required and "PROJ_DATA" in bad.detail
    noted = doctor.check_conda_env({"CONDA_PREFIX": "/opt/anaconda3"}, _env.EnvPlan(drop=("GDAL_DATA",)))
    assert noted.ok and "GDAL_DATA" in noted.detail and "conda active" in noted.detail
    assert doctor.check_conda_env({}, _env.EnvPlan()).detail == "clean"


def test_check_data_complete_and_missing(tmp_path: Path) -> None:
    assert all(c.ok for c in doctor.check_data(_synthetic_cfg(tmp_path / "a", complete=True)))
    missing = doctor.check_data(_synthetic_cfg(tmp_path / "b", complete=False))
    assert not any(c.ok for c in missing)
    assert all(c.required for c in missing)


def test_check_vector_io() -> None:
    checks = doctor.check_vector_io()
    assert [c.name for c in checks] == ["parquet round-trip", "GPKG round-trip"]
    assert all(c.ok for c in checks)


def test_check_disk(tmp_path: Path) -> None:
    assert doctor.check_disk(tmp_path, 0.0).ok
    assert not doctor.check_disk(tmp_path, 1e12).ok
    assert doctor.check_disk(tmp_path / "not" / "yet", 0.0).ok


def test_run_checks_and_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _synthetic_cfg(tmp_path, complete=True)
    checks = doctor.run_checks(cfg, torch_probe=lambda: (False, "absent"), min_free_gb=0.0)
    names = [c.name for c in checks]
    assert names[0] == "python" and "free disk" in names and "tile ZIPs" in names
    assert doctor.exit_code(checks) == 0
    assert doctor.exit_code((*checks, Check("x", False, True, "boom"))) == 1
    doctor.render(checks)
    assert "vineyard doctor" in capsys.readouterr().out


@pytest.mark.needs_tiles
def test_doctor_on_real_data_passes() -> None:
    cfg = load_config()
    checks = doctor.check_data(cfg)
    assert all(c.ok for c in checks), [c for c in checks if not c.ok]
