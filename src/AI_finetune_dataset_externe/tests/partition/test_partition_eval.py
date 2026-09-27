"""Golden test: the identity partition reproduces `vineyard eval-examples` of complete-v4 exactly."""

from __future__ import annotations

import pytest

from fte.paths import AI_RUNS

pytestmark = pytest.mark.examples

BASE_SCORES = {"siret3_r006_c004": 0.8941, "siret3_r021_c012": 0.8455}


@pytest.fixture(scope="module")
def data():
    if not (AI_RUNS / "complete-v4" / "annset" / "annset.json").is_file():
        pytest.skip("complete-v4 run not available")
    from fte.canopy.partition_eval import load_examples

    return load_examples("complete-v4")


def test_identity_matches_eval_examples(data) -> None:
    from fte.canopy.partition_eval import score_canopies

    row = score_canopies(data, data.base.canopies, "identity")
    for tile, expected in BASE_SCORES.items():
        assert round(row["per_tile"][tile]["canopy.score"], 4) == expected
    assert max(row["union_delta_m2"].values()) == 0.0


def test_width_partition_preserves_union(data) -> None:
    from fte.canopy.partition import PartitionParams, partition_canopies
    from fte.canopy.partition_eval import score_canopies

    cut = partition_canopies(data.base.canopies, data.base.row_pieces, PartitionParams(alpha=0.35, min_len_m=2.0))
    row = score_canopies(data, cut, "width")
    ident = score_canopies(data, data.base.canopies, "identity")
    assert max(row["union_delta_m2"].values()) < 1e-6
    for tile in BASE_SCORES:
        assert row["per_tile"][tile]["canopy.iou"] == pytest.approx(ident["per_tile"][tile]["canopy.iou"], abs=1e-6)
    assert row["mean_score"] >= 0.875
