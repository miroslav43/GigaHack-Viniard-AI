"""Linear waste probe on CLIP crop embeddings (A§4.9 step 3, design 03 W5).

- Logistic regression (class_weight balanced) on L2-normalised embeddings; scoring is pure numpy.
- Out-of-fold (OOF) scores from a grouped, stratified K-fold: negatives are grouped by tile, positives by
  UAVVaste image, so no score comes from a model that saw the same tile / image.
- Auto threshold: tau* = max OOF score of the example-tile negatives (+ margin), so the examples give
  0 out-of-fold false positives; the effective threshold is max(auto_probe_min, tau*). Recall of the OOF
  positives at that threshold below min_recall (or tau* > 1) disables auto-accept (ranking only).
- Persistence: models/<name>/<version>/{probe.npz, model_card.json}; npz without pickle, card with sha256.
sklearn is imported lazily (training only).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from vineyard.errors import VineyardError
from vineyard.pipeline.atomic import atomic_path, atomic_write_json

if TYPE_CHECKING:
    from vineyard.config import WasteConfig

PROBE_FILE: Final = "probe.npz"
CARD_FILE: Final = "model_card.json"
ARCH: Final = "logistic-regression-on-clip-embeddings"
CLASSES: Final = ("not_waste", "waste")
NO_PROBE: Final = "none"
SHA_PREFIX_LEN: Final = 8
MIN_CV_FOLDS: Final = 2
CALIBRATE_MODES: Final = ("all", "survivors")
_HASH_CHUNK: Final = 1 << 20


class ProbeError(VineyardError):
    """Probe data, training or persistence failure."""


@dataclass(frozen=True)
class ProbeParams:
    lr_c: float
    calibration_margin: float
    auto_probe_min: float
    min_recall: float
    cv_folds: int
    max_iter: int
    seed: int
    calibrate_on: str

    def __post_init__(self) -> None:
        if self.cv_folds < MIN_CV_FOLDS:
            raise ProbeError("cv_folds must be >= 2", cv_folds=self.cv_folds)
        if self.calibrate_on not in CALIBRATE_MODES:
            raise ProbeError("unknown calibrate_on", calibrate_on=self.calibrate_on, allowed=CALIBRATE_MODES)
        if self.lr_c <= 0 or self.max_iter <= 0:
            raise ProbeError("lr_c and max_iter must be positive", lr_c=self.lr_c, max_iter=self.max_iter)

    @classmethod
    def from_config(cls, cfg: WasteConfig, seed: int) -> ProbeParams:
        return cls(
            lr_c=cfg.probe.lr_c,
            calibration_margin=cfg.probe.calibration_margin,
            auto_probe_min=cfg.decide.auto_probe_min,
            min_recall=cfg.probe.min_recall,
            cv_folds=cfg.probe.cv_folds,
            max_iter=cfg.probe.max_iter,
            seed=seed,
            calibrate_on=cfg.probe.calibrate_on,
        )


@dataclass(frozen=True)
class ProbeData:
    """emb (N, D) float32 L2-normalised; label 1 = waste; group = CV group (tile id / "uav:<image>");
    is_example = negative from an example tile (0 waste there); survives = example negative that passes
    the rule filters (the only ones the pipeline would score); source = crop source (per-source metrics)."""

    emb: np.ndarray
    label: np.ndarray
    group: np.ndarray
    is_example: np.ndarray
    survives: np.ndarray
    source: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.emb.ndim != 2 or len(self.emb) == 0:
            raise ProbeError("emb must be a non-empty (N, D) array", shape=self.emb.shape)
        n = len(self.emb)
        for name in ("label", "group", "is_example", "survives", "source"):
            if getattr(self, name) is not None and getattr(self, name).shape != (n,):
                raise ProbeError(f"{name} must have shape (N,)", n=n, shape=getattr(self, name).shape)
        if not np.all(np.isfinite(self.emb)):
            raise ProbeError("emb must be finite", n_bad=int((~np.isfinite(self.emb)).sum()))
        if not np.all(np.isin(self.label, (0, 1))):
            raise ProbeError("label must be 0 or 1", values=sorted(set(np.unique(self.label).tolist())))
        if np.any((self.is_example | self.survives) & (self.label == 1)):
            raise ProbeError("is_example / survives flag a positive; they describe example negatives only")


@dataclass(frozen=True)
class Calibration:
    tau_all: float
    tau_survivors: float
    tau_star: float
    auto_threshold: float


@dataclass(frozen=True)
class CvReport:
    n_pos: int
    n_neg: int
    n_example_neg: int
    n_folds: int
    auc: float
    average_precision: float
    tau_all: float
    tau_survivors: float
    tau_star: float
    auto_threshold: float
    recall_at_threshold: float
    fp_example_at_threshold: int
    fp_all_at_threshold: int
    recall_by_source: Mapping[str, float]
    fp_by_source: Mapping[str, int]
    recall_at_half: float
    fpr_at_half: float
    auto_enabled: bool
    auto_reason: str


@dataclass(frozen=True)
class ProbeModel:
    coef: np.ndarray
    intercept: float
    auto_threshold: float
    tau_star: float
    recall_at_threshold: float
    auto_enabled: bool
    embed_model: str
    name: str
    version: str
    sha256: str = ""


@dataclass(frozen=True)
class ProbeResult:
    model: ProbeModel
    cv: CvReport
    oof: np.ndarray
    folds: np.ndarray


# ------------------------------------------------------------------ training


def _fit(emb: np.ndarray, label: np.ndarray, p: ProbeParams) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(label)) != len(CLASSES):
        raise ProbeError("training fold needs both classes", n=len(label), n_pos=int(label.sum()))
    clf = LogisticRegression(C=p.lr_c, class_weight="balanced", max_iter=p.max_iter, random_state=p.seed)
    clf.fit(emb.astype(np.float64), label)
    return clf.coef_[0].astype(np.float64), float(clf.intercept_[0])


def _logistic(emb: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    """Numerically stable sigmoid of the linear score (no overflow for large |z|)."""
    z = emb.astype(np.float64) @ coef + intercept
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def grouped_oof_scores(data: ProbeData, p: ProbeParams) -> tuple[np.ndarray, np.ndarray]:
    """OOF probabilities and the fold index of every sample (a group sits in exactly one fold)."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_groups = {int(c): len(np.unique(data.group[data.label == c])) for c in (0, 1)}
    if min(n_groups.values()) < p.cv_folds:
        raise ProbeError("too few groups per class for grouped CV", groups=n_groups, cv_folds=p.cv_folds)
    splitter = StratifiedGroupKFold(n_splits=p.cv_folds, shuffle=True, random_state=p.seed)
    oof = np.full(len(data.label), np.nan)
    folds = np.full(len(data.label), -1, np.int16)
    for k, (tr, te) in enumerate(splitter.split(data.emb, data.label, data.group)):
        coef, intercept = _fit(data.emb[tr], data.label[tr], p)
        oof[te] = _logistic(data.emb[te], coef, intercept)
        folds[te] = k
    return oof, folds


def calibrate_threshold(
    oof: np.ndarray, label: np.ndarray, is_example: np.ndarray, survives: np.ndarray, p: ProbeParams
) -> Calibration:
    """tau = max OOF score of example negatives + margin (survivor-only variant falls back to all)."""
    ex_neg = is_example & (label == 0)
    if not ex_neg.any():
        raise ProbeError("calibration needs example negatives (0-FP rule on the example tiles)")
    tau_all = float(oof[ex_neg].max()) + p.calibration_margin
    surv = ex_neg & survives
    tau_surv = float(oof[surv].max()) + p.calibration_margin if surv.any() else tau_all
    tau_star = tau_all if p.calibrate_on == "all" else tau_surv
    return Calibration(tau_all, tau_surv, tau_star, max(p.auto_probe_min, tau_star))


def _auc_ap(label: np.ndarray, oof: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    return float(roc_auc_score(label, oof)), float(average_precision_score(label, oof))


def _auto_state(cal: Calibration, recall: float, p: ProbeParams) -> tuple[bool, str]:
    if cal.auto_threshold > 1.0:
        return False, f"threshold {cal.auto_threshold:.3f} > 1: an example negative scores too high"
    if recall < p.min_recall:
        return False, f"recall {recall:.3f} < min_recall {p.min_recall}: probe used for ranking only"
    return True, "ok"


def cv_report(data: ProbeData, oof: np.ndarray, cal: Calibration, p: ProbeParams) -> CvReport:
    pos, neg = data.label == 1, data.label == 0
    ex_neg = data.is_example & neg
    thr = cal.auto_threshold
    recall = float((oof[pos] >= thr).mean())
    auc, ap = _auc_ap(data.label, oof)
    enabled, reason = _auto_state(cal, recall, p)
    src = data.source if data.source is not None else np.where(pos, "positive", "negative")
    return CvReport(
        n_pos=int(pos.sum()),
        n_neg=int(neg.sum()),
        n_example_neg=int(ex_neg.sum()),
        n_folds=p.cv_folds,
        auc=auc,
        average_precision=ap,
        tau_all=cal.tau_all,
        tau_survivors=cal.tau_survivors,
        tau_star=cal.tau_star,
        auto_threshold=thr,
        recall_at_threshold=recall,
        fp_example_at_threshold=int((oof[ex_neg] >= thr).sum()),
        fp_all_at_threshold=int((oof[neg] >= thr).sum()),
        recall_by_source={str(k): float((oof[pos & (src == k)] >= thr).mean()) for k in np.unique(src[pos])},
        fp_by_source={str(k): int((oof[neg & (src == k)] >= thr).sum()) for k in np.unique(src[neg])},
        recall_at_half=float((oof[pos] >= 0.5).mean()),
        fpr_at_half=float((oof[neg] >= 0.5).mean()),
        auto_enabled=enabled,
        auto_reason=reason,
    )


def train_probe(
    data: ProbeData, p: ProbeParams, embed_model: str, version: str, name: str = ""
) -> ProbeResult:
    """Grouped OOF calibration, then a final fit on all samples."""
    oof, folds = grouped_oof_scores(data, p)
    cal = calibrate_threshold(oof, data.label, data.is_example, data.survives, p)
    cv = cv_report(data, oof, cal, p)
    coef, intercept = _fit(data.emb, data.label, p)
    model = ProbeModel(
        coef=coef,
        intercept=intercept,
        auto_threshold=cal.auto_threshold,
        tau_star=cal.tau_star,
        recall_at_threshold=cv.recall_at_threshold,
        auto_enabled=cv.auto_enabled,
        embed_model=embed_model,
        name=name,
        version=version,
    )
    return ProbeResult(model=model, cv=cv, oof=oof, folds=folds)


def score_probe(m: ProbeModel, emb: np.ndarray) -> np.ndarray:
    """P(waste) for (N, D) embeddings (pure numpy)."""
    if emb.ndim != 2 or emb.shape[1] != m.coef.shape[0]:
        raise ProbeError("embedding dimension does not match the probe", emb=emb.shape, coef=m.coef.shape)
    return _logistic(emb, m.coef, m.intercept)


# ------------------------------------------------------------------ persistence


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _npz_payload(m: ProbeModel) -> dict[str, np.ndarray]:
    return {
        "coef": np.asarray(m.coef, np.float64),
        "intercept": np.asarray([m.intercept], np.float64),
        "auto_threshold": np.asarray([m.auto_threshold], np.float64),
        "tau_star": np.asarray([m.tau_star], np.float64),
        "recall_at_threshold": np.asarray([m.recall_at_threshold], np.float64),
        "auto_enabled": np.asarray([m.auto_enabled], np.bool_),
        "embed_model": np.asarray(m.embed_model),
    }


def _card(m: ProbeModel, cv: CvReport, train_data: Sequence[Mapping[str, Any]], train_cmd: str) -> dict:
    return {
        "name": m.name,
        "version": m.version,
        "arch": ARCH,
        "embed_model": m.embed_model,
        "classes": list(CLASSES),
        "dim": int(m.coef.shape[0]),
        "train_data": [dict(d) for d in train_data],
        "metrics": asdict(cv),
        "sha256": m.sha256,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "train_cmd": train_cmd,
    }


def save_probe(
    result: ProbeResult,
    models_dir: Path,
    name: str,
    *,
    train_data: Sequence[Mapping[str, Any]],
    train_cmd: str,
) -> ProbeModel:
    """Write models/<name>/<version>/{probe.npz, model_card.json}; returns the model with its sha256."""
    folder = Path(models_dir) / name / result.model.version
    path = folder / PROBE_FILE
    with atomic_path(path) as tmp, tmp.open("wb") as fh:
        np.savez(fh, **_npz_payload(result.model))
    saved = replace(result.model, name=name, sha256=sha256_file(path))
    atomic_write_json(folder / CARD_FILE, _card(saved, result.cv, train_data, train_cmd))
    return saved


def _read_card(folder: Path) -> dict[str, Any]:
    path = folder / CARD_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError("unreadable probe model card", path=str(path), error=str(exc)) from exc


def load_probe(models_dir: Path, name: str, version: str, expected_sha256: str | None = None) -> ProbeModel:
    """Load and verify (sha256 against the card and, when given, the expected value)."""
    folder = Path(models_dir) / name / version
    path = folder / PROBE_FILE
    if not path.is_file():
        raise ProbeError("probe weights missing", path=str(path))
    card = _read_card(folder)
    actual = sha256_file(path)
    for label, want in (("card", card.get("sha256")), ("expected", expected_sha256)):
        if want is not None and want != actual:
            raise ProbeError("probe sha256 mismatch", path=str(path), source=label, want=want, got=actual)
    with np.load(path, allow_pickle=False) as z:
        return ProbeModel(
            coef=z["coef"].copy(),
            intercept=float(z["intercept"][0]),
            auto_threshold=float(z["auto_threshold"][0]),
            tau_star=float(z["tau_star"][0]),
            recall_at_threshold=float(z["recall_at_threshold"][0]),
            auto_enabled=bool(z["auto_enabled"][0]),
            embed_model=str(z["embed_model"]),
            name=name,
            version=version,
            sha256=actual,
        )


def probe_version_string(m: ProbeModel | None) -> str:
    """`<name>@<version>+<sha8>` for model_version, "none" without a probe."""
    return NO_PROBE if m is None else f"{m.name}@{m.version}+{m.sha256[:SHA_PREFIX_LEN]}"


def decide_threshold(m: ProbeModel | None) -> float | None:
    """`probe_auto_min` for decide.decide: None without a probe, +inf when calibration disabled auto."""
    if m is None:
        return None
    return m.auto_threshold if m.auto_enabled else math.inf
