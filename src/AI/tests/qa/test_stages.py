"""Stages row_attrs -> assemble -> qa_previews on the two-tile synthetic work dir (inline, workers=1)."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import cv2
import geopandas as gpd
import pandas as pd
import pytest

from tests.qa import stage_env as env
from vineyard.annset.io import read_annset
from vineyard.config import load_config
from vineyard.contracts.schemas import validate_layer
from vineyard.errors import StageError
from vineyard.geo.vector_io import read_layer
from vineyard.pipeline.cache import key_path, read_key
from vineyard.pipeline.context import RunContext, make_run_paths
from vineyard.pipeline.registry import load_stage
from vineyard.pipeline.runner import TileFailure, load_tile_failures, record_tile_failures
from vineyard.pipeline.stages import assemble as assemble_stage
from vineyard.pipeline.stages import qa_previews as qa_stage
from vineyard.pipeline.stages import row_attrs as row_attrs_stage
from vineyard.qa import review

A, B = env.A, env.B


@pytest.fixture
def ctx(tmp_work: Path) -> RunContext:
    return env.make_env()


def _assembled(ctx: RunContext) -> RunContext:
    row_attrs_stage.run(ctx)
    assemble_stage.run(ctx)
    return ctx


def _status(ctx: RunContext) -> pd.DataFrame:
    return read_layer(ctx.paths.layers_dir / "tile_status.parquet", "tile_status").set_index("tile_id")


def test_registry_loads_my_stages() -> None:
    assert load_stage("row_attrs") is row_attrs_stage.STAGE
    assert load_stage("assemble") is assemble_stage.STAGE
    assert load_stage("qa_previews") is qa_stage.STAGE


# ------------------------------------------------------------------ row_attrs


def test_row_attrs_structure_gaps_and_cache(ctx: RunContext) -> None:
    res = row_attrs_stage.run(ctx)
    assert (res.n_items, res.n_failed, res.n_cached) == (2, 0, 0)
    pieces = read_layer(row_attrs_stage.row_pieces_path(ctx.paths, A), "row_pieces")
    assert list(pieces["piece_id"]) == [f"V01-R00{k}@{A}" for k in (1, 2, 3)]
    assert list(pieces["row_structure"]) == ["regular", "disrupted", "regular"]
    assert pieces["max_gap_m"].iloc[1] == pytest.approx(env.GAP[1] - env.GAP[0], abs=0.05)
    assert pieces["length_m"].iloc[0] == pytest.approx(51.2)
    gaps = gpd.read_parquet(row_attrs_stage.gaps_path(ctx.paths, A))
    assert list(gaps["kind"]) == ["interior"] and gaps["piece_id"].iloc[0] == f"V01-R002@{A}"
    assert len(read_layer(row_attrs_stage.row_pieces_path(ctx.paths, B), "row_pieces")) == 0
    key = read_key(row_attrs_stage.row_pieces_path(ctx.paths, A))
    again = row_attrs_stage.run(ctx)
    assert again.n_cached == 2 and read_key(row_attrs_stage.row_pieces_path(ctx.paths, A)) == key


def test_row_attrs_needs_canopy_cache_and_rows(ctx: RunContext) -> None:
    key_path(row_attrs_stage.canopy_cache_path(ctx.paths, A)).unlink()
    res = row_attrs_stage.run(ctx)
    assert res.failed == (A,)
    (ctx.paths.layers_dir / "rows.parquet").unlink()
    with pytest.raises(StageError, match="rows layer missing"):
        row_attrs_stage.run(ctx)


def test_task_cfgs_pickle(ctx: RunContext) -> None:
    task = row_attrs_stage._make_task(ctx, A)
    # Round-trip of our own task object: spawn workers receive tasks pickled.
    assert pickle.loads(pickle.dumps(task)).cfg.run_id == env.RUN_ID
    preview = qa_stage._make_task(ctx, ctx.paths, {A: _preview_cfg(ctx)}, A)
    assert pickle.loads(pickle.dumps(preview)).cfg.header.tile_id == A
    with pytest.raises(StageError, match="RowAttrsTaskCfg"):
        row_attrs_stage.row_attrs_tile(preview)


def _preview_cfg(ctx: RunContext) -> qa_stage.PreviewTaskCfg:
    return qa_stage.PreviewTaskCfg(qa=ctx.cfg.qa, objects=qa_stage.TileObjects(),
                                   header=qa_stage.PreviewHeader(tile_id=A))


# ------------------------------------------------------------------ assemble


def test_assemble_writes_annset_status_and_queue(ctx: RunContext) -> None:
    row_attrs_stage.run(ctx)
    res = assemble_stage.run(ctx)
    assert res.n_failed == 0 and res.metrics["n_errors"] == 0.0
    ann = read_annset(ctx.paths.annset_dir)
    assert ann.counts() == {"canopies": 4, "row_pieces": 3, "interrow_pieces": 2, "waste": 0}
    assert ann.meta.tile_ids == tuple(sorted((A, B)))
    for name in ("canopies", "row_pieces", "interrow_pieces"):
        assert set(ann.layer(name)["run_id"]) == {env.RUN_ID}
    status = _status(ctx)
    assert (status.loc[A, "status"], status.loc[A, "review_priority"]) == ("ok", 2)
    assert status.loc[A, "issues"] == "structure_borderline"
    assert (status.loc[B, "status"], status.loc[B, "review_priority"]) == ("no_vineyard", 1)
    queue = pd.read_csv(ctx.paths.qa_dir / "review_queue.csv")
    assert list(queue["tile_id"]) == [B, A]
    validate_layer(read_layer(ctx.paths.qa_dir / "qa_issues.parquet"), "qa_issues")


def test_assemble_is_deterministic(ctx: RunContext) -> None:
    _assembled(ctx)
    first = (ctx.paths.qa_dir / "review_queue.csv").read_bytes(), (ctx.paths.qa_dir / "tile_status.csv").read_bytes()
    assemble_stage.run(ctx)
    second = (ctx.paths.qa_dir / "review_queue.csv").read_bytes(), (ctx.paths.qa_dir / "tile_status.csv").read_bytes()
    assert first == second


def test_assemble_skips_failed_stage_cache(ctx: RunContext) -> None:
    row_attrs_stage.run(ctx)
    record_tile_failures(ctx.paths, "canopy", [A], [TileFailure("canopy", A, "boom")])
    res = assemble_stage.run(ctx)
    assert res.n_failed == 0 and res.metrics["n_canopies"] == 0.0
    status = _status(ctx)
    assert (status.loc[A, "status"], status.loc[A, "review_priority"]) == ("failed", 1)
    assert "tile_failed" in status.loc[A, "issues"]


def test_assemble_missing_cache_is_a_failure(ctx: RunContext) -> None:
    row_attrs_stage.run(ctx)
    ctx.paths.tile_cache("interrow", B, "parquet").unlink()
    res = assemble_stage.run(ctx)
    assert res.failed == (B,)
    assert [(f.stage, f.tile_id) for f in load_tile_failures(ctx.paths)] == [("assemble", B)]
    assert _status(ctx).loc[B, "status"] == "failed"


def test_assemble_force_empty_from_overrides(tmp_work: Path, tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text(f"version: 1\nforce_empty_tiles: [{A}]\n")
    cfg = load_config()
    ctx = env.make_env(cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"overrides": overrides})}))
    _assembled(ctx)
    status = _status(ctx)
    assert status.loc[A, "n_canopies"] == 0 and "override_applied" in status.loc[A, "issues"]


def test_assemble_waste_layer_and_evidence(ctx: RunContext) -> None:
    from tests.qa import annset_factory as af

    wst = af.waste([(A, "V01", af.rect(A, 45.0, 45.0, 46.0, 46.0)), (B, "", af.rect(B, 5.0, 5.0, 6.0, 6.0))])
    wst.to_parquet(ctx.paths.layers_dir / "waste.parquet")
    env.write_detect_summary(ctx, A, 1.5)
    _assembled(ctx)
    status = _status(ctx)
    assert (status.loc[A, "n_waste"], status.loc[B, "n_waste"]) == (1, 1)
    assert "canopy_on_non_vineyard_tile" in status.loc[A, "issues"]


def test_assemble_merges_stage_issue_files(ctx: RunContext) -> None:
    from vineyard.contracts.enums import Severity
    from vineyard.contracts.qa import QaIssue, issues_to_gdf
    from vineyard.geo.vector_io import write_layer

    issues = [QaIssue(Severity.WARNING, "missing_row_suspect", A, "V01-R002", "skip-one edge", 1.0, 2.0),
              QaIssue(Severity.WARNING, "curved_row", "siret3_r000_c000", "V09-R001", "not selected"),
              QaIssue(Severity.INFO, "override_unmatched", "", "D001", "no tile")]
    write_layer(issues_to_gdf(issues), "qa_issues", ctx.paths.qa_dir / "qa_rows_link.parquet")
    _assembled(ctx)
    qa = read_layer(ctx.paths.qa_dir / "qa_issues.parquet", "qa_issues")
    assert "missing_row_suspect" in set(qa["code"]) and "curved_row" not in set(qa["code"])
    assert "override_unmatched" in set(qa["code"])
    assert _status(ctx).loc[A, "review_priority"] == 1
    located = qa[qa["code"] == "missing_row_suspect"].geometry.iloc[0]
    assert (located.x, located.y) == (1.0, 2.0)


def test_evidence_helpers(ctx: RunContext) -> None:
    assert assemble_stage.tile_snr({"orientations": [{"snr": 2.0}, {"snr": "x"}, {"snr": 7.5}]}) == 7.5
    ev = assemble_stage.tile_evidence(ctx.paths, A)
    assert (ev.snr, ev.veg_frac, ev.empty_nodata) == (12.0, 0.25, False)
    bad = ctx.paths.tile_cache("canopy", A, "json")
    bad.write_text("{not json")
    with pytest.raises(StageError, match="unreadable"):
        assemble_stage.tile_evidence(ctx.paths, A)


def test_assemble_requires_tile_valid(ctx: RunContext) -> None:
    row_attrs_stage.run(ctx)
    ctx.paths.tile_valid.unlink()
    with pytest.raises(StageError, match="tile_valid"):
        assemble_stage.run(ctx)


# ------------------------------------------------------------------ qa_previews


def test_qa_previews_overview_and_queue(ctx: RunContext) -> None:
    _assembled(ctx)
    res = qa_stage.run(ctx)
    assert (res.n_items, res.n_failed) == (2, 0)
    for tile_id in (A, B):
        img = cv2.imread(str(qa_stage.preview_path(ctx.paths, tile_id)))
        assert img.shape == (ctx.cfg.qa.preview_px, ctx.cfg.qa.preview_px, 3)
    overview = cv2.imread(str(ctx.paths.qa_dir / "overview.jpg"))
    assert overview is not None and overview.shape[2] == 3
    queue = pd.read_csv(ctx.paths.qa_dir / "review_queue.csv")
    assert list(queue.columns) == list(review.QUEUE_COLUMNS) and list(queue["tile_id"]) == [B, A]
    assert queue.loc[queue["tile_id"] == A, "n_warnings"].iloc[0] == 1
    assert qa_stage.run(ctx).n_cached == 2


def test_qa_previews_on_another_runs_annset(ctx: RunContext) -> None:
    # `vineyard qa --annset <run>` runs in a new post run but must read and write the AnnSet's run
    _assembled(ctx)
    post = replace(ctx, run_id="post-run", paths=make_run_paths(ctx.cfg, "post-run"), annset_ref=env.RUN_ID)
    assert qa_stage.annset_paths(post).qa_dir == ctx.paths.qa_dir
    res = qa_stage.run(post)
    assert (res.n_items, res.n_failed) == (2, 0)
    assert qa_stage.preview_path(ctx.paths, A).is_file() and (ctx.paths.qa_dir / "overview.jpg").is_file()
    assert not post.paths.qa_dir.exists()


def test_rejected_candidates_drawn_in_grey(ctx: RunContext) -> None:
    _assembled(ctx)
    rejected = qa_stage.rejected_candidates(ctx.paths.tile_cache("rows_detect", A, "parquet"))
    assert len(rejected) == 1 and qa_stage.rejected_candidates(None) == ()
    qa_stage.run(ctx)
    img = cv2.imread(str(qa_stage.preview_path(ctx.paths, A)))
    scale = ctx.cfg.qa.preview_px / 51.2
    r, c = int(round((51.2 - 10.0) * scale)), int(round(25.6 * scale))
    grey = ctx.cfg.qa.colors_bgr["rejected"]
    assert max(abs(int(a) - int(b)) for a, b in zip(img[r, c], grey, strict=True)) < 40


def test_qa_preview_header_and_blocks(ctx: RunContext) -> None:
    _assembled(ctx)
    inputs = qa_stage.load_qa_inputs(ctx.paths)
    header = qa_stage.header_for(A, inputs)
    assert (header.status, header.priority, header.counts["R"]) == ("ok", 2, 3)
    assert header.flags == ("structure_borderline",)
    assert qa_stage.dominant_blocks(inputs.annset) == {A: "V01"}
    orphan = qa_stage.header_for("siret3_r000_c000", inputs)
    assert orphan.priority is None


def test_qa_previews_requires_assemble(ctx: RunContext) -> None:
    with pytest.raises(StageError, match="run assemble first"):
        qa_stage.run(ctx)


def test_unreadable_preview_fails_overview(ctx: RunContext) -> None:
    _assembled(ctx)
    qa_stage.run(ctx)
    qa_stage.preview_path(ctx.paths, A).write_bytes(b"not a jpeg")
    with pytest.raises(StageError, match="unreadable preview"):
        qa_stage.write_overview(ctx, read_annset(ctx.paths.annset_dir), (A, B))
