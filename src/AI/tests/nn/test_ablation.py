"""nn.ablation: variants A-F per tile and axis source, promotion rule (A§4.6), JSON + Markdown report."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.nn.n2_synth import OTHER_TILE, TILE_ID, synth_tile
from vineyard.config import load_config
from vineyard.contracts.enums import FusionVariant
from vineyard.errors import StageError
from vineyard.geo.tiling import tile_box, tile_ref
from vineyard.nn.ablation import (
    AXES_MODEL,
    AXES_REFERENCE,
    AblationReport,
    VariantScore,
    decide_promotion,
    render_markdown,
    run_ablation,
    write_report,
)
from vineyard.nn.probs import PROB_PX
from vineyard.nn.validate import ValTile, input_factor, make_val_tile, predict_canopies

CFG = load_config()
R021, R006 = "siret3_r021_c012", "siret3_r006_c004"
TILES = (R021, R006)


def _scores(table: dict[str, tuple[float, float]], axes: str = AXES_MODEL) -> list[VariantScore]:
    return [VariantScore(v, t, 0.0, 0.0, s, axes, 0, 0) for v, vals in table.items() for t, s in zip(TILES, vals,
                                                                                                    strict=True)]


def test_import_is_torch_free() -> None:
    code = "import sys, vineyard.nn.ablation; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_b_loses_on_one_tile_keeps_a() -> None:
    promoted, reason = decide_promotion(_scores({"A": (0.782, 0.861), "B": (0.80, 0.85)}), "A", ("B", "C", "D"),
                                        TILES, 0.0)
    assert promoted is FusionVariant.A
    assert "B" in reason and R006 in reason


def test_c_wins_on_both_tiles() -> None:
    promoted, reason = decide_promotion(_scores({"A": (0.782, 0.861), "C": (0.79, 0.87)}), "A", ("B", "C", "D"),
                                        TILES, 0.0)
    assert promoted is FusionVariant.C
    assert "C" in reason


def test_highest_mean_among_winners() -> None:
    table = {"A": (0.782, 0.861), "B": (0.79, 0.865), "C": (0.80, 0.87)}
    promoted, _ = decide_promotion(_scores(table), "A", ("B", "C", "D"), TILES, 0.0)
    assert promoted is FusionVariant.C


def test_e_and_f_are_never_promoted() -> None:
    table = {"A": (0.5, 0.5), "E": (0.9, 0.9), "F": (0.95, 0.95)}
    promoted, _ = decide_promotion(_scores(table), "A", ("B", "C", "D", "E", "F"), TILES, 0.0)
    assert promoted is FusionVariant.A


def test_min_gain_and_ties() -> None:
    tie = {"A": (0.8, 0.8), "B": (0.8, 0.9)}
    assert decide_promotion(_scores(tie), "A", ("B",), TILES, 0.0)[0] is FusionVariant.A
    small = {"A": (0.8, 0.8), "B": (0.803, 0.9)}
    assert decide_promotion(_scores(small), "A", ("B",), TILES, 0.005)[0] is FusionVariant.A
    assert decide_promotion(_scores(small), "A", ("B",), TILES, 0.0)[0] is FusionVariant.B


def test_missing_scores() -> None:
    promoted, reason = decide_promotion(_scores({"A": (0.8, 0.8)}), "A", ("B",), TILES, 0.0)
    assert promoted is FusionVariant.A and "missing" in reason
    with pytest.raises(StageError, match="baseline"):
        decide_promotion(_scores({"B": (0.8, 0.8)}), "A", ("B",), TILES, 0.0)


# ------------------------------------------------------------------ run_ablation on synthetic tiles


def _half(mask: np.ndarray) -> np.ndarray:
    return mask.reshape(PROB_PX, 2, PROB_PX, 2).mean(axis=(1, 3)).astype(np.float32)


@pytest.fixture(scope="module")
def tiles() -> dict[str, tuple[ValTile, ...]]:
    out = []
    for tile_id in (TILE_ID, OTHER_TILE):
        s = synth_tile(tile_id)
        base = make_val_tile(tile_id, s.rgb, s.valid, s.veg, s.pieces, tile_box(tile_ref(tile_id)), (),
                             input_factor(CFG))
        out.append(replace(base, ref=tuple(predict_canopies(base, None, FusionVariant.A, CFG))))
    return {AXES_MODEL: tuple(out), AXES_REFERENCE: tuple(out)}


def test_run_ablation_prob_equal_to_veg_keeps_a(tiles: dict[str, tuple[ValTile, ...]]) -> None:
    """Fake probabilities equal to the a* mask: B/C/D tie A, so A stays (WP-N3 acceptance)."""
    probs = {t: _half(synth_tile(t).veg) for t in (TILE_ID, OTHER_TILE)}
    report = run_ablation(tiles, {"main": probs, "F": probs}, CFG, infer_s_per_tile={"main": 0.1})
    assert isinstance(report, AblationReport)
    assert report.promoted is FusionVariant.A
    by = {(s.axes, s.variant, s.tile_id): s.score for s in report.scores}
    for v in ("A", "B", "C", "D", "F"):
        assert by[(AXES_MODEL, v, TILE_ID)] == pytest.approx(1.0)
    assert by[(AXES_MODEL, "E", TILE_ID)] < 1.0  # grass outside the corridors becomes canopy without them
    assert report.means[AXES_REFERENCE]["A"] == pytest.approx(1.0)
    assert report.primary_axes == AXES_MODEL


def test_run_ablation_without_f_weights(tiles: dict[str, tuple[ValTile, ...]]) -> None:
    probs = {t: _half(synth_tile(t).canopy) for t in (TILE_ID, OTHER_TILE)}
    report = run_ablation(tiles, {"main": probs}, CFG)
    assert "F" not in {s.variant for s in report.scores}
    assert "F" in report.skipped


def test_run_ablation_needs_main_probs(tiles: dict[str, tuple[ValTile, ...]]) -> None:
    with pytest.raises(StageError, match="main"):
        run_ablation(tiles, {}, CFG)


def test_report_files(tmp_path: Path, tiles: dict[str, tuple[ValTile, ...]]) -> None:
    probs = {t: _half(synth_tile(t).veg) for t in (TILE_ID, OTHER_TILE)}
    report = run_ablation(tiles, {"main": probs}, CFG, infer_s_per_tile={"main": 0.25})
    json_path, md_path = write_report(report, tmp_path / "metrics" / "nn_ablation.json",
                                      tmp_path / "reports" / "nn_ablation.md")
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    assert doc["promoted"] == "A" and doc["primary_axes"] == AXES_MODEL
    assert set(doc["means"]) == {AXES_MODEL, AXES_REFERENCE}
    assert doc["tiles"][AXES_MODEL]["B"][TILE_ID]["score"] == pytest.approx(1.0)
    md = md_path.read_text(encoding="utf-8")
    assert "nn.fusion=A" in md and TILE_ID in md and "| B |" in md
    assert render_markdown(report) == md
