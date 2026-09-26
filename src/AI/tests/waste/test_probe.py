"""Linear probe on crop embeddings (design 03 W5): grouped CV, 0-FP calibration, npz round-trip."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vineyard.perception.waste.probe import (
    CARD_FILE,
    PROBE_FILE,
    ProbeData,
    ProbeError,
    ProbeParams,
    calibrate_threshold,
    decide_threshold,
    grouped_oof_scores,
    load_probe,
    probe_version_string,
    save_probe,
    score_probe,
    train_probe,
)

DIM = 16
EMBED_MODEL = "open_clip:ViT-B-32/laion2b_s34b_b79k"
EXAMPLES = ("siret3_r006_c004", "siret3_r021_c012")


def params(**over: object) -> ProbeParams:
    base = ProbeParams(
        lr_c=1.0,
        calibration_margin=0.02,
        auto_probe_min=0.9,
        min_recall=0.2,
        cv_folds=4,
        max_iter=2000,
        seed=0,
        calibrate_on="all",
    )
    return replace(base, **over)


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def make_data(sep: float, n_pos: int = 80, n_neg_tile: int = 30, seed: int = 0) -> ProbeData:
    """Positives around +mu, negatives around -mu; negatives spread over 2 example + 4 other tiles."""
    rng = np.random.default_rng(seed)
    mu = np.zeros(DIM, np.float32)
    mu[0] = sep
    tiles = (*EXAMPLES, "siret3_r010_c010", "siret3_r011_c010", "siret3_r012_c010", "siret3_r013_c010")
    pos = rng.normal(size=(n_pos, DIM)) + mu
    neg = rng.normal(size=(n_neg_tile * len(tiles), DIM)) - mu
    neg_groups = np.repeat(np.array(tiles), n_neg_tile)
    pos_groups = np.array([f"uav:{i // 4}" for i in range(n_pos)])
    is_example = np.concatenate([np.zeros(n_pos, bool), np.isin(neg_groups, EXAMPLES)])
    survives = np.concatenate([np.zeros(n_pos, bool), rng.random(len(neg)) < 0.1]) & is_example
    return ProbeData(
        emb=_unit(np.concatenate([pos, neg])),
        label=np.concatenate([np.ones(n_pos, np.int8), np.zeros(len(neg), np.int8)]),
        group=np.concatenate([pos_groups, neg_groups]),
        is_example=is_example,
        survives=survives,
    )


# ------------------------------------------------------------------ data validation


def test_probe_data_rejects_bad_shapes_and_labels() -> None:
    good = make_data(3.0)
    with pytest.raises(ProbeError, match="label"):
        replace(good, label=good.label[:-1])
    with pytest.raises(ProbeError, match="0 or 1"):
        replace(good, label=np.full_like(good.label, 2))
    with pytest.raises(ProbeError, match="finite"):
        bad = good.emb.copy()
        bad[0, 0] = np.nan
        replace(good, emb=bad)
    with pytest.raises(ProbeError, match="example"):
        flags = good.is_example.copy()
        flags[0] = True  # a positive flagged as an example negative
        replace(good, is_example=flags)


def test_probe_params_reject_bad_values() -> None:
    with pytest.raises(ProbeError, match="cv_folds"):
        params(cv_folds=1)
    with pytest.raises(ProbeError, match="calibrate_on"):
        params(calibrate_on="nope")


# ------------------------------------------------------------------ calibration


def test_calibration_is_max_example_negative_plus_margin() -> None:
    oof = np.array([0.99, 0.2, 0.5, 0.7, 0.1])
    label = np.array([1, 0, 0, 0, 0])
    is_example = np.array([False, True, True, False, True])
    survives = np.array([False, False, True, False, False])
    cal = calibrate_threshold(oof, label, is_example, survives, params())
    assert cal.tau_all == pytest.approx(0.52)
    assert cal.tau_survivors == pytest.approx(0.52)
    assert cal.tau_star == pytest.approx(0.52)
    assert cal.auto_threshold == pytest.approx(0.9)  # max(0.9, tau*)
    survivors_only = calibrate_threshold(oof, label, is_example, ~survives & is_example, params())
    assert survivors_only.tau_survivors == pytest.approx(0.22)


def test_calibration_on_survivors_falls_back_to_all_when_none_survive() -> None:
    oof = np.array([0.99, 0.95, 0.5])
    label = np.array([1, 0, 0])
    is_example = np.array([False, True, True])
    none = np.zeros(3, bool)
    cal = calibrate_threshold(oof, label, is_example, none, params(calibrate_on="survivors"))
    assert cal.tau_star == pytest.approx(0.97)
    assert cal.auto_threshold == pytest.approx(0.97)


def test_calibration_needs_example_negatives() -> None:
    with pytest.raises(ProbeError, match="example negatives"):
        calibrate_threshold(np.array([0.9]), np.array([1]), np.array([False]), np.array([False]), params())


# ------------------------------------------------------------------ grouped CV


def test_oof_scores_never_train_on_the_scored_group() -> None:
    data = make_data(3.0)
    oof, folds = grouped_oof_scores(data, params())
    assert oof.shape == data.label.shape
    assert np.all((oof >= 0) & (oof <= 1))
    for g in np.unique(data.group):
        assert len(np.unique(folds[data.group == g])) == 1


def test_oof_needs_enough_groups() -> None:
    data = make_data(3.0)
    with pytest.raises(ProbeError, match="groups"):
        grouped_oof_scores(data, params(cv_folds=50))


# ------------------------------------------------------------------ training


def test_separable_training_zero_example_fp_and_high_recall() -> None:
    data = make_data(4.0)
    result = train_probe(data, params(), EMBED_MODEL, "v1")
    cv = result.cv
    ex_neg = data.is_example & (data.label == 0)
    assert cv.fp_example_at_threshold == 0
    assert np.all(result.oof[ex_neg] < result.model.auto_threshold)
    assert cv.tau_star == pytest.approx(float(result.oof[ex_neg].max()) + 0.02)
    assert result.model.auto_threshold == pytest.approx(max(0.9, cv.tau_star))
    assert cv.recall_at_threshold > 0.8
    assert cv.auc > 0.95
    assert result.model.auto_enabled
    assert cv.n_pos == 80 and cv.n_example_neg == 60
    assert set(cv.recall_by_source) == {"positive"} and cv.fp_by_source == {
        "negative": cv.fp_all_at_threshold
    }


def test_per_source_metrics_use_the_source_column() -> None:
    data = make_data(4.0)
    src = np.where(data.label == 1, np.where(np.arange(len(data.label)) % 2 == 0, "plain", "paste"), "neg")
    cv = train_probe(replace(data, source=src), params(), EMBED_MODEL, "v1").cv
    assert set(cv.recall_by_source) == {"plain", "paste"} and set(cv.fp_by_source) == {"neg"}
    with pytest.raises(ProbeError, match="source"):
        replace(data, source=src[:-1])


def test_inseparable_data_disables_auto() -> None:
    result = train_probe(make_data(0.0), params(), EMBED_MODEL, "v1")
    assert result.cv.recall_at_threshold < 0.2
    assert not result.model.auto_enabled
    assert "recall" in result.cv.auto_reason


def test_score_probe_matches_sigmoid_and_checks_dim() -> None:
    result = train_probe(make_data(4.0), params(), EMBED_MODEL, "v1")
    m = result.model
    emb = make_data(4.0, seed=3).emb[:5]
    z = emb.astype(np.float64) @ m.coef + m.intercept
    np.testing.assert_allclose(score_probe(m, emb), 1.0 / (1.0 + np.exp(-z)), rtol=1e-12)
    with pytest.raises(ProbeError, match="dimension"):
        score_probe(m, emb[:, :4])


# ------------------------------------------------------------------ persistence


def test_npz_round_trip_no_pickle_and_card(tmp_path: Path) -> None:
    result = train_probe(make_data(4.0), params(), EMBED_MODEL, "v1")
    saved = save_probe(result, tmp_path, "waste-probe", train_data=(), train_cmd="pytest")
    folder = tmp_path / "waste-probe" / "v1"
    assert (folder / PROBE_FILE).is_file() and (folder / CARD_FILE).is_file()
    with np.load(folder / PROBE_FILE, allow_pickle=False) as z:
        assert set(z.files) >= {"coef", "intercept", "auto_threshold", "tau_star", "embed_model"}
    with zipfile.ZipFile(folder / PROBE_FILE) as zf:
        assert not any(n.endswith(".pkl") for n in zf.namelist())
    card = json.loads((folder / CARD_FILE).read_text())
    assert card["sha256"] == saved.sha256 and card["embed_model"] == EMBED_MODEL
    assert card["metrics"]["fp_example_at_threshold"] == 0
    loaded = load_probe(tmp_path, "waste-probe", "v1", expected_sha256=saved.sha256)
    emb = make_data(4.0, seed=5).emb
    np.testing.assert_array_equal(score_probe(loaded, emb), score_probe(result.model, emb))
    assert loaded.auto_threshold == result.model.auto_threshold
    assert probe_version_string(loaded) == f"waste-probe@v1+{saved.sha256[:8]}"
    assert probe_version_string(None) == "none"


def test_load_detects_tampering(tmp_path: Path) -> None:
    result = train_probe(make_data(4.0), params(), EMBED_MODEL, "v1")
    save_probe(result, tmp_path, "waste-probe", train_data=(), train_cmd="pytest")
    path = tmp_path / "waste-probe" / "v1" / PROBE_FILE
    raw = bytearray(path.read_bytes())
    raw[-10] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(ProbeError, match="sha256"):
        load_probe(tmp_path, "waste-probe", "v1")


def test_load_missing_probe_names_path(tmp_path: Path) -> None:
    with pytest.raises(ProbeError, match="missing"):
        load_probe(tmp_path, "waste-probe", "v9")


def test_probe_data_modules_import_no_torch_or_sklearn() -> None:
    import subprocess
    import sys

    mods = ("uavvaste", "positives", "negatives", "clip_embed", "probe")
    imports = "; ".join(f"import vineyard.perception.waste.{m}" for m in mods)
    code = f"import sys; {imports}; print(sorted(m for m in ('torch', 'sklearn', 'open_clip') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_decide_threshold_blocks_auto_when_disabled() -> None:
    import math

    on = train_probe(make_data(4.0), params(), EMBED_MODEL, "v1").model
    off = train_probe(make_data(0.0), params(), EMBED_MODEL, "v1").model
    assert decide_threshold(None) is None
    assert decide_threshold(on) == on.auto_threshold
    assert decide_threshold(off) == math.inf
