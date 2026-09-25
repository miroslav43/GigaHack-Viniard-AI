"""Stage import_reference: examples annotations.xml -> AnnSet(reference) + LATEST_REFERENCE."""

import json
from pathlib import Path

import pytest

from vineyard.annset.io import read_annset, resolve_run_dir
from vineyard.config import load_config
from vineyard.contracts.enums import Source
from vineyard.errors import StageError
from vineyard.geo.vector_io import read_layer
from vineyard.pipeline.context import ensure_run_dirs, new_run_context
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.stages.import_reference import (
    STAGE,
    build_reference_annset,
    examples_xml_path,
    find_reusable_run,
)

RUN_A = "20260926T0100-reference-aaaaaa"
RUN_B = "20260926T0105-reference-bbbbbb"


def _ctx(run_id: str, *overrides: str, force: tuple[str, ...] = ()):
    cfg = load_config(overrides=overrides)
    ctx = new_run_context(cfg, source=Source.REFERENCE, run_id=run_id, force=force, workers=1)
    ensure_run_dirs(ctx.paths)
    return ctx


def test_stage_spec() -> None:
    assert load_stage("import_reference") is STAGE
    assert (STAGE.name, STAGE.scope) == ("import_reference", "global")
    assert "import" in STAGE.cfg_keys and "canopy" in STAGE.cfg_keys


@pytest.mark.examples
def test_build_reference_annset(examples_xml: bytes) -> None:
    annset, qa = build_reference_annset(load_config(), examples_xml, run_id=RUN_A, source_name="annotations.xml")
    assert annset.meta.source == Source.REFERENCE
    assert annset.meta.counts == {"canopies": 650, "row_pieces": 51, "interrow_pieces": 49, "waste": 0}
    assert annset.meta.model_version.startswith("reference-examples@") and len(annset.meta.model_version) == 27
    assert len(qa) == 0


@pytest.mark.examples
def test_run_writes_annset_qa_and_latest(tmp_work: Path) -> None:
    ctx = _ctx(RUN_A)
    result = STAGE.run(ctx)
    assert (result.n_items, result.n_cached, result.n_failed) == (2, 0, 0)
    assert result.metrics["canopies"] == 650.0
    annset = read_annset(ctx.paths.annset_dir)
    assert annset.meta.tile_ids == ("siret3_r006_c004", "siret3_r021_c012")
    assert len(read_layer(ctx.paths.qa_dir / "qa_issues.parquet", "qa_issues")) == 0
    assert resolve_run_dir(tmp_work, "LATEST_REFERENCE") == ctx.paths.run_dir.resolve()
    assert find_reusable_run(ctx.paths.runs_dir, _key_of(ctx)) == ctx.paths.run_dir


def _key_of(ctx) -> str:
    return (ctx.paths.annset_dir / "annset.json.key").read_text().strip()


@pytest.mark.examples
def test_rerun_reuses_the_same_run(tmp_work: Path) -> None:
    first = _ctx(RUN_A)
    STAGE.run(first)
    second = _ctx(RUN_B)
    result = STAGE.run(second)
    assert (result.n_cached, result.outputs) == (1, (first.paths.annset_dir,))
    assert not (second.paths.annset_dir / "annset.json").exists()
    assert resolve_run_dir(tmp_work, "LATEST_REFERENCE") == first.paths.run_dir.resolve()


@pytest.mark.examples
def test_force_rebuilds(tmp_work: Path) -> None:
    STAGE.run(_ctx(RUN_A))
    forced = _ctx(RUN_B, force=("import_reference",))
    assert STAGE.run(forced).n_cached == 0
    assert (forced.paths.annset_dir / "annset.json").is_file()
    assert resolve_run_dir(tmp_work, "LATEST_REFERENCE") == forced.paths.run_dir.resolve()


@pytest.mark.examples
def test_config_change_invalidates(tmp_work: Path) -> None:
    STAGE.run(_ctx(RUN_A))
    other = _ctx(RUN_B, "canopy.clump_area_m2=3.0")
    assert STAGE.run(other).n_cached == 0


def test_missing_examples_raise(tmp_work: Path, tmp_path: Path) -> None:
    ctx = _ctx(RUN_A, f"paths.examples_dir={json.dumps(str(tmp_path / 'nope'))}")
    assert examples_xml_path(ctx.cfg) == tmp_path / "nope" / "annotations.xml"
    with pytest.raises(StageError):
        STAGE.run(ctx)


def test_find_reusable_run_without_runs(tmp_path: Path) -> None:
    assert find_reusable_run(tmp_path / "runs", "k") is None


@pytest.mark.examples
def test_run_stages_end_to_end(tmp_work: Path) -> None:
    from vineyard.pipeline.runner import run_stages

    ctx = _ctx(RUN_A)
    report = run_stages(ctx, ["import_reference"])
    assert report.exit_code == 0
    assert resolve_run_dir(tmp_work, "LATEST_REFERENCE") == ctx.paths.run_dir.resolve()
    run_doc = json.loads(ctx.paths.run_json.read_text())
    assert run_doc["stages"]["import_reference"]["n_items"] == 2
