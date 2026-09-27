"""DroneWaste (Zenodo 17045559, CC BY 4.0): select, stream and resample to 2.5 cm/px.

The 3.9 GB ``images.tar.gz`` is one gzip stream, so it is read sequentially through parallel
Range requests and only the selected members are decoded, degraded to Sireț3 GSD and written as
JPEG. The archive itself never lands on disk.
"""

from __future__ import annotations

import io
import json
import logging
import random
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from fte.convert.gsd import TARGET_GSD_M, degrade_to_target
from fte.data.http_range import ParallelRangeReader, remote_size

log = logging.getLogger(__name__)

RECORD = "17045559"
IMAGES_URL = f"https://zenodo.org/api/records/{RECORD}/files/images.tar.gz/content"
LICENCE = "CC-BY-4.0"

# Category ids from dronewaste_v1.0.json -> role in our single-class `waste` task.
KEEP_IDS = frozenset({5, 6, 7, 8, 9, 13, 15, 16, 19})  # appliances..textile: litter-like items
CAP_IDS = frozenset({11})  # pallets: kept, but images capped (few sites dominate)
MASK_IDS = frozenset({12, 20})  # scrap, mixed piles: ambiguous -> ignore region
DROP_IDS = frozenset({1, 2, 3, 4, 10, 14, 17, 18})  # soils, rubble, wood, vehicles, asbestos -> 0

SITE_GSD_M = {"site6": 0.028}  # preprint: ~2 cm/px everywhere except site 6 (2.8 cm/px)
DEFAULT_SITE_GSD_M = 0.020


def category_role(cat_id: int) -> str:
    if cat_id in KEEP_IDS or cat_id in CAP_IDS:
        return "waste"
    if cat_id in MASK_IDS:
        return "ignore"
    if cat_id in DROP_IDS:
        return "background"
    raise ValueError(f"unknown DroneWaste category id {cat_id}")


@dataclass(frozen=True)
class Selection:
    roles: dict[str, str]  # file_name -> pos | neg_pile | neg_empty

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for role in self.roles.values():
            out[role] += 1
        return dict(out)


def select_images(coco: dict, n_empty: int = 600, pallet_cap: int = 60, seed: int = 0) -> Selection:
    """Pick positive images (any litter-like item), pile-only negatives and a sample of empties."""
    cats_by_img: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        cats_by_img[ann["image_id"]].add(ann["category_id"])
    rng = random.Random(seed)
    roles: dict[str, str] = {}
    pallet_only: list[str] = []
    empties: list[str] = []
    for img in coco["images"]:
        cats = cats_by_img.get(img["id"], set())
        name = img["file_name"]
        if cats & KEEP_IDS:
            roles[name] = "pos"
        elif cats & CAP_IDS:
            pallet_only.append(name)
        elif cats and cats <= DROP_IDS:
            roles[name] = "neg_pile"
        elif not cats:
            empties.append(name)
    rng.shuffle(pallet_only)
    roles.update({n: "pos" for n in pallet_only[:pallet_cap]})
    rng.shuffle(empties)
    roles.update({n: "neg_empty" for n in empties[:n_empty]})
    return Selection(roles=roles)


def site_gsd(site: str) -> float:
    return SITE_GSD_M.get(site, DEFAULT_SITE_GSD_M)


def scale_annotations(coco: dict, selection: Selection) -> dict:
    """Annotations of the selected images with bbox/polygons rescaled to 2.5 cm/px."""
    images = {i["id"]: i for i in coco["images"] if i["file_name"] in selection.roles}
    out_images, out_anns = [], []
    for img in images.values():
        scale = site_gsd(img["site"]) / TARGET_GSD_M
        out_images.append(
            {
                "id": img["id"],
                "file_name": Path(img["file_name"]).with_suffix(".jpg").name,
                "site": img["site"],
                "role": selection.roles[img["file_name"]],
                "src_gsd_m": site_gsd(img["site"]),
                "width": int(round(img["width"] * scale)),
                "height": int(round(img["height"] * scale)),
            }
        )
    for ann in coco["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            continue
        scale = site_gsd(img["site"]) / TARGET_GSD_M
        polys = [[round(v * scale, 2) for v in poly] for poly in ann.get("segmentation", []) if poly]
        out_anns.append(
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "role": category_role(ann["category_id"]),
                "bbox": [round(v * scale, 2) for v in ann["bbox"]],
                "segmentation": polys,
            }
        )
    return {"images": out_images, "annotations": out_anns, "categories": coco["categories"]}


def process_member(data: bytes, site: str, out_path: Path, jpeg_quality: int = 95) -> tuple[int, int]:
    """Decode a PNG tile, degrade it to 2.5 cm/px and write it as JPEG; returns (h, w)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode {out_path.name}")
    small = degrade_to_target(img, site_gsd(site))
    if not cv2.imwrite(str(out_path), small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise OSError(f"cannot write {out_path}")
    return small.shape[:2]


def stream_selected(coco: dict, selection: Selection, out_dir: Path, workers: int = 8) -> dict:
    """Stream the archive once and write every selected member; returns a small report."""
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    site_of = {i["file_name"]: i["site"] for i in coco["images"]}
    total = remote_size(IMAGES_URL)
    reader = ParallelRangeReader(IMAGES_URL, stop=total, workers=workers)
    written, errors = 0, []
    with tarfile.open(fileobj=io.BufferedReader(reader, 1 << 20), mode="r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in selection.roles:
                continue
            out = img_dir / Path(name).with_suffix(".jpg").name
            if out.exists():
                written += 1
                continue
            try:
                process_member(tar.extractfile(member).read(), site_of[name], out)
                written += 1
            except (ValueError, OSError) as exc:
                errors.append(f"{name}: {exc}")
            if written % 100 == 0:
                log.info("dronewaste: %d written, %.0f/%.0f MB read", written,
                         reader.bytes_read / 1e6, total / 1e6)
    return {"archive_bytes": total, "bytes_read": reader.bytes_read, "written": written, "errors": errors}


def prepare(json_path: Path, out_dir: Path, workers: int = 8) -> dict:
    """Full DroneWaste preparation: selection -> streamed images -> scaled annotations + manifest."""
    coco = json.loads(json_path.read_text())
    selection = select_images(coco)
    log.info("dronewaste selection: %s", selection.counts())
    report = stream_selected(coco, selection, out_dir, workers=workers)
    (out_dir / "annotations.json").write_text(json.dumps(scale_annotations(coco, selection)))
    manifest = {
        "dataset": "DroneWaste v1.0",
        "doi": "10.5281/zenodo.17045559",
        "licence": LICENCE,
        "source": IMAGES_URL,
        "target_gsd_m": TARGET_GSD_M,
        "site_gsd_m": {"default": DEFAULT_SITE_GSD_M, **SITE_GSD_M},
        "selection": selection.counts(),
        **report,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(__file__).resolve().parents[2] / "work"
    result = prepare(root / "raw/dronewaste/dronewaste_v1.0.json", root / "data/dronewaste",
                     workers=int(sys.argv[1]) if len(sys.argv) > 1 else 12)
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, indent=2))
    print("errors:", len(result["errors"]))
