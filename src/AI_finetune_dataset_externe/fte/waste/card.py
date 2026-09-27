"""model_card.json of the waste detector: datasets + licences, config, metrics, weights sha256."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final

DATASETS: Final = (
    {"name": "DroneWaste v1.0", "licence": "CC-BY-4.0", "doi": "10.5281/zenodo.17045559",
     "use": "waste items (positives), rubble/soil/wood/vehicle piles (negatives), val sites"},
    {"name": "UAVVaste v1", "licence": "CC-BY-4.0", "doi": "10.5281/zenodo.8214061",
     "attribution": "M. Kraft, M. Piechocki, B. Ptak, K. Walas (Poznan University of Technology)",
     "use": "rubbish instances (positives), scale from a 0.25 m median object"},
    {"name": "Sireț3 UAV orthomosaic (GigaHack 2026)", "licence": "CC-BY-4.0",
     "use": "hard-negative / background windows and copy-paste backgrounds; no hand labels"},
)
CARD: Final = "model_card.json"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def read_card(out: Path) -> dict[str, Any]:
    path = out / CARD
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_card(out: Path, cfg: Any, src: Any, history: list[dict], best_epoch: int,
               best_record: dict | None) -> Path:
    """(Re)write the card; ``best_record`` None keeps the previously stored best metrics."""
    old = read_card(out)
    best = best_record if best_record is not None else old.get("best_metrics", {})
    card = {
        "model": "fte-waste", "architecture": "smp.Unet(resnet34, classes=1, decoder_attention_type='scse')",
        "input": "RGB uint8 / 255, ImageNet normalisation inside the module, 2.5 cm/px",
        "output": "1 logit channel (waste); boxes = tight bboxes of connected components of p >= t",
        "weights": {"best": "best.pt", "best_sha256": sha256_file(out / "best.pt"),
                    "last": "last.pt", "last_sha256": sha256_file(out / "last.pt")},
        "datasets": list(DATASETS), "config": asdict(cfg), "counts": src.counts(),
        "val_sites": list(src.val_sites), "best_epoch": best_epoch,
        "best_metrics": best, "history": history,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / CARD
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(card, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path
