"""Training-tile selection and the QA list file (design 03 §3 N2, §5): holdout never selected."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vineyard.config import load_config
from vineyard.nn.pseudolabels import (
    PseudoLabelError,
    SelectionPolicy,
    read_train_tiles_file,
    select_training_tiles,
)

HOLDOUT = ("siret3_r006_c004", "siret3_r021_c012")


def _status(n_clean: int, n_urgent: int, n_empty: int, extra: list[dict[str, object]] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for k in range(n_clean):
        rows.append(dict(tile_id=f"siret3_r030_c{k:03d}", status="ok", has_vineyard=True, n_row_pieces=10,
                         review_priority=2, issues="curved_row", veg_frac=0.2))
    for k in range(n_urgent):
        rows.append(dict(tile_id=f"siret3_r031_c{k:03d}", status="ok", has_vineyard=True, n_row_pieces=10,
                         review_priority=1, issues="missing_row_suspect", veg_frac=0.2))
    for k in range(n_empty):
        rows.append(dict(tile_id=f"siret3_r032_c{k:03d}", status="no_vineyard", has_vineyard=False,
                         n_row_pieces=0, review_priority=1, issues="empty_tile_confirm", veg_frac=k / 100))
    return pd.DataFrame(rows + (extra or []))


def _policy(**kw: object) -> SelectionPolicy:
    base = dict(min_row_pieces=3, blocking_codes=frozenset({"low_snr", "tile_failed"}), fill_from_urgent=True,
                include_empty_tiles=True, max_empty_tile_frac=0.2, min_train_tiles=3)
    return SelectionPolicy(**{**base, **kw})  # type: ignore[arg-type]


def _holdout_rows() -> list[dict[str, object]]:
    return [dict(tile_id=t, status="ok", has_vineyard=True, n_row_pieces=20, review_priority=3, issues="",
                 veg_frac=0.2) for t in HOLDOUT]


def test_holdout_never_selected_auto() -> None:
    sel = select_training_tiles(_status(5, 0, 0, _holdout_rows()), None, HOLDOUT, _policy())
    assert not set(HOLDOUT) & set(sel.positives)
    assert set(sel.holdout_removed) == set(HOLDOUT)
    assert sel.source == "auto" and len(sel.positives) == 5


def test_holdout_never_selected_from_qa_list() -> None:
    status = _status(5, 0, 0, _holdout_rows())
    approved = [*HOLDOUT, "siret3_r030_c000", "siret3_r030_c001", "siret3_r030_c002"]
    sel = select_training_tiles(status, approved, HOLDOUT, _policy())
    assert sel.source == "qa_file"
    assert sel.positives == ("siret3_r030_c000", "siret3_r030_c001", "siret3_r030_c002")
    assert set(sel.holdout_removed) == set(HOLDOUT)


def test_qa_list_drops_tiles_without_rows() -> None:
    status = _status(3, 0, 1)
    approved = ["siret3_r030_c000", "siret3_r030_c001", "siret3_r030_c002", "siret3_r032_c000", "siret3_r040_c040"]
    sel = select_training_tiles(status, approved, HOLDOUT, _policy())
    assert sel.positives == ("siret3_r030_c000", "siret3_r030_c001", "siret3_r030_c002")
    assert dict(sel.excluded)["siret3_r032_c000"] == "no_rows"
    assert dict(sel.excluded)["siret3_r040_c040"] == "not_in_run"


def test_too_few_tiles_raises() -> None:
    with pytest.raises(PseudoLabelError, match="min_train_tiles"):
        select_training_tiles(_status(2, 0, 0), None, HOLDOUT, _policy(fill_from_urgent=False))


def test_urgent_tiles_fill_only_when_clean_ones_are_too_few() -> None:
    enough = select_training_tiles(_status(3, 4, 0), None, HOLDOUT, _policy())
    assert len(enough.positives) == 3 and enough.n_clean == 3 and enough.n_urgent_added == 0
    short = select_training_tiles(_status(2, 4, 0), None, HOLDOUT, _policy())
    assert len(short.positives) == 6 and short.n_urgent_added == 4


def test_blocking_codes_errors_and_few_rows_are_excluded() -> None:
    extra = [
        dict(tile_id="siret3_r033_c001", status="ok", has_vineyard=True, n_row_pieces=10, review_priority=1,
             issues="low_snr;curved_row", veg_frac=0.1),
        dict(tile_id="siret3_r033_c002", status="ok", has_vineyard=True, n_row_pieces=2, review_priority=2,
             issues="", veg_frac=0.1),
        dict(tile_id="siret3_r033_c003", status="ok", has_vineyard=True, n_row_pieces=10, review_priority=2,
             issues="", veg_frac=0.1),
        dict(tile_id="siret3_r033_c004", status="failed", has_vineyard=False, n_row_pieces=0, review_priority=1,
             issues="tile_failed", veg_frac=0.1),
    ]
    sel = select_training_tiles(_status(3, 0, 0, extra), None, HOLDOUT, _policy(),
                                error_tiles=frozenset({"siret3_r033_c003"}))
    reasons = dict(sel.excluded)
    assert reasons["siret3_r033_c001"] == "blocked:low_snr"
    assert reasons["siret3_r033_c002"] == "few_rows"
    assert reasons["siret3_r033_c003"] == "qa_error"
    assert reasons["siret3_r033_c004"] == "status:failed"
    assert set(sel.positives) == {f"siret3_r030_c{k:03d}" for k in range(3)}


def test_empties_are_capped_at_the_fraction() -> None:
    sel = select_training_tiles(_status(40, 0, 50), None, HOLDOUT, _policy())
    assert len(sel.empties) == 10  # 10 / (40 + 10) = 20 %
    # the greenest empty tiles first (orchards, grass: the hard negatives)
    assert sel.empties[0] == "siret3_r032_c049"


def test_empties_disabled() -> None:
    sel = select_training_tiles(_status(4, 0, 5), None, HOLDOUT, _policy(include_empty_tiles=False))
    assert sel.empties == ()


def test_selection_rejects_missing_columns() -> None:
    with pytest.raises(PseudoLabelError, match="columns"):
        select_training_tiles(pd.DataFrame({"tile_id": ["x"]}), None, HOLDOUT, _policy())


def test_selection_summary_counts() -> None:
    sel = select_training_tiles(_status(3, 0, 2, _holdout_rows()), None, HOLDOUT, _policy())
    summary = sel.summary()
    assert summary["n_positives"] == 3 and summary["n_empties"] == 0  # cap: floor(0.25 * 3) = 0
    assert summary["holdout_removed"] == sorted(HOLDOUT)


def test_policy_from_config() -> None:
    policy = SelectionPolicy.from_config(load_config())
    assert policy.min_train_tiles == 30 and policy.max_empty_tile_frac == pytest.approx(0.2)
    assert "low_snr" in policy.blocking_codes and policy.min_row_pieces == 3


def test_read_train_tiles_file(tmp_path: Path) -> None:
    path = tmp_path / "train_tiles.txt"
    path.write_text("# QA approved\nsiret3_r030_c001\n\nsiret3_r030_c002  # ok\n", encoding="utf-8")
    assert read_train_tiles_file(path) == ("siret3_r030_c001", "siret3_r030_c002")
    path.write_text("not-a-tile\n", encoding="utf-8")
    with pytest.raises(PseudoLabelError, match="line 1"):
        read_train_tiles_file(path)
    with pytest.raises(PseudoLabelError, match="missing"):
        read_train_tiles_file(tmp_path / "absent.txt")
