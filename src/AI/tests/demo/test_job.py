"""vineyard.demo.job: the job folder isolates the run (own data root, work dir, empty QA inputs)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from vineyard.demo.job import DEMO_STAGES, isolation_sets, prepare_job
from vineyard.demo.tile_input import TileInput


def test_prepare_job(tmp_path: Path) -> None:
    upload = tmp_path / "upload (1).tif"
    upload.write_bytes(b"tif bytes")
    route = tmp_path / "pkg" / "02_route"
    route.mkdir(parents=True)
    job = prepare_job(tmp_path / "job", TileInput("siret3_r021_c012", upload), route)
    with zipfile.ZipFile(job.data_root / "01_tiles" / "demo.zip") as zf:
        assert zf.namelist() == ["siret3_r021_c012.tif"]
        assert zf.read("siret3_r021_c012.tif") == b"tif bytes"
    assert (job.data_root / "02_route").resolve() == route.resolve()
    assert job.overrides.read_text() == "version: 1\n"
    assert job.waste_confirmed.read_text().startswith("tile_id,")
    assert job.work_dir.is_dir()


def test_isolation_sets_point_at_the_job(tmp_path: Path) -> None:
    job = prepare_job(tmp_path / "job & co", TileInput("siret3_r021_c012", _tif(tmp_path)), _route(tmp_path))
    sets = isolation_sets(job)
    assert "grid.expected_tiles=1" in sets and "rows_seeded.enabled=false" in sets
    assert any(s.startswith("paths.work_dir=") and "job & co" in s for s in sets)


def test_demo_stages_end_at_assemble() -> None:
    assert DEMO_STAGES[0] == "ingest" and DEMO_STAGES[-1] == "assemble" and len(DEMO_STAGES) == 11


def _tif(tmp_path: Path) -> Path:
    path = tmp_path / "t.tif"
    path.write_bytes(b"x")
    return path


def _route(tmp_path: Path) -> Path:
    path = tmp_path / "02_route"
    path.mkdir(exist_ok=True)
    return path
