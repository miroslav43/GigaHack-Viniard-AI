from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from typer.testing import CliRunner

from tests.waste.synth_waste import make_candidate
from tests.waste.test_waste_stage import TILE_ID, ctx  # noqa: F401  (fixture)
from vineyard.perception.waste.cli import app, calibration_markdown
from vineyard.perception.waste.types import RejectReason
from vineyard.pipeline.context import RunContext
from vineyard.pipeline.stages import waste as waste_stage

runner = CliRunner()


def test_calibration_markdown_table() -> None:
    kept = make_candidate()
    rej = replace(make_candidate(centroid=(10.0, 10.0)), reject_reason=RejectReason.NEAR_AXIS)
    text = calibration_markdown({"siret3_r021_c012": [kept, rej]}, "LATEST_REFERENCE")
    assert "| tile | candidates | near_axis | kept |" in text
    assert "| siret3_r021_c012 | 2 | 1 | 1 |" in text
    assert kept.cand_key in text and rej.cand_key not in text


def test_review_html_rebuilds_from_a_run(ctx: RunContext, tmp_work: Path) -> None:  # noqa: F811
    waste_stage.run(ctx)
    html = ctx.paths.qa_dir / "waste_review.html"
    html.unlink()
    res = runner.invoke(
        app,
        [
            "review-html",
            "--run",
            ctx.run_id,
            "--set",
            f'paths.waste_confirmed="{tmp_work / "waste_confirmed.csv"}"',
        ],
    )
    assert res.exit_code == 0, res.output
    assert html.is_file() and "2 crop" in res.output


def test_review_html_unknown_run_fails(tmp_work: Path) -> None:
    res = runner.invoke(app, ["review-html", "--run", "nope"])
    assert res.exit_code == 1 and "run not found" in res.output


def test_calibrate_report_on_a_run(ctx: RunContext, tmp_path: Path) -> None:  # noqa: F811
    out = tmp_path / "cal.md"
    res = runner.invoke(app, ["calibrate-report", "--tile", TILE_ID, "--axes", ctx.run_id, "--out", str(out)])
    assert res.exit_code == 0, res.output
    text = out.read_text(encoding="utf-8")
    assert f"| {TILE_ID} |" in text and "near_axis" in text


def test_calibrate_report_missing_cache_fails(tmp_work: Path) -> None:
    res = runner.invoke(app, ["calibrate-report", "--tile", TILE_ID, "--axes", "nope"])
    assert res.exit_code != 0
