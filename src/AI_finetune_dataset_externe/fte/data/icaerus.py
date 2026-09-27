"""ICAERUS AI4Leafhopper vine segmentation data (Zenodo 14605849, CC BY-NC 4.0).

Only the hand-labelled YOLO-seg subset (85 RGBA crops at 1.4 cm/px, 1,093 per-vine polygons,
~28.5 MB) is pulled out of the 3.1 GB ``AI_model.zip`` via HTTP Range requests.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from fte.data import zip_range
from fte.data.http_range import remote_size

log = logging.getLogger(__name__)

RECORD = "14605849"
URL = f"https://zenodo.org/api/records/{RECORD}/files/AI_model.zip/content"
LICENCE = "CC-BY-NC-4.0"
SRC_GSD_M = 0.014
PATTERNS = (
    "AI_model/INPUT/YOLODataset/images/train/*.png",
    "AI_model/INPUT/YOLODataset/images/val/*.png",
    "AI_model/INPUT/YOLODataset/labels/train/*.txt",
    "AI_model/INPUT/YOLODataset/labels/val/*.txt",
    "AI_model/INPUT/YOLODataset/dataset.yaml",
    "AI_model/INPUT/YOLODataset/README.dataset.txt",
)
PREFIX = "AI_model/INPUT/YOLODataset/"


def prepare(out_dir: Path) -> dict:
    total = remote_size(URL)
    members = zip_range.select(zip_range.list_members(URL, total=total), PATTERNS)
    log.info("icaerus: %d members selected (%.1f MB compressed)", len(members),
             sum(m.comp_size for m in members) / 1e6)
    files = zip_range.fetch_members(URL, members, total=total)
    sha = {}
    for name, data in files.items():
        rel = name[len(PREFIX):]
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        sha[rel] = hashlib.sha256(data).hexdigest()
    manifest = {
        "dataset": "ICAERUS AI4Leafhopper - input dataset for the vine segmentation model",
        "doi": "10.5281/zenodo.14605849",
        "licence": LICENCE,
        "attribution": "Marengo I., Sirsat M. (2025), ICAERUS project, EU grant 101060643",
        "source": URL,
        "src_gsd_m": SRC_GSD_M,
        "n_files": len(files),
        "sha256": sha,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[2] / "work"
    m = prepare(root / "raw/icaerus")
    print(m["n_files"], "files")
