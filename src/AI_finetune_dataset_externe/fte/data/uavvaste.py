"""UAVVaste (CC BY 4.0, doi:10.5281/zenodo.8214061) -> scaled waste store at 2.5 cm/px.

UAVVaste publishes no GSD; the src/AI heuristic is reused: the median object long side of an image (or of
its flight batch / the whole dataset when it has < 3 objects) measures 0.25 m. Each image is degraded to
2.5 cm/px through the Sireț3 capture GSD (``fte.convert.gsd.degrade_to_target``); the downscaled images
are small (~300 x 170 px), so they are kept whole. The official train/val/test file decides the split
(val + test -> ``val``).

Output: ``work/data/uavvaste/{images/*.jpg, annotations.json, manifest.json}``, same layout as DroneWaste.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

from fte.convert.coco import POS_IMAGE, WASTE, read_rgb, write_jpeg
from fte.convert.gsd import TARGET_GSD_M, degrade_to_target
from fte.paths import AI_WORK, DATA

log = logging.getLogger(__name__)

SRC_ROOT: Final = AI_WORK / "external" / "uavvaste" / "dataset"
OUT_DIR: Final = DATA / "uavvaste"
DISTRIBUTION: Final = Path("annotations") / "train_val_test_distribution_file.json"
ASSUMED_OBJECT_M: Final = 0.25
MIN_OBJECTS_FOR_SCALE: Final = 3
CATEGORY: Final = {"id": 0, "name": "rubbish"}


def _positive_params() -> Any:
    from vineyard.perception.waste.positives import PositiveParams

    return PositiveParams(
        target_gsd_m=TARGET_GSD_M, assumed_object_m=ASSUMED_OBJECT_M, long_side_px=(8.0, 40.0),
        min_objects_for_scale=MIN_OBJECTS_FOR_SCALE, max_per_image=4, max_n=1500, context=0.5,
        min_px=64, out_px=224, seed=0,
    )


def split_of_files(root: Path) -> dict[str, str]:
    """file_name -> train | val (official val and test both become val)."""
    doc = json.loads((root / DISTRIBUTION).read_text(encoding="utf-8"))
    out = {name: "train" for name in doc.get("train", [])}
    out.update({name: "val" for key in ("val", "test") for name in doc.get(key, [])})
    return out


def _scale_polys(polys: tuple[tuple[float, ...], ...], s: float) -> list[list[float]]:
    return [[round(v * s, 2) for v in p] for p in polys]


def _image_record(img: Any, name: str, gsd: float, shape: tuple[int, ...], split: str, n_obj: int) -> dict:
    return {"id": img.image_id, "file_name": name, "site": img.batch,
            "role": POS_IMAGE if n_obj else "neg_empty", "src_gsd_m": round(gsd, 6),
            "width": int(shape[1]), "height": int(shape[0]), "split": split}


def convert(src_root: Path = SRC_ROOT, out_dir: Path = OUT_DIR, quality: int = 95) -> dict:
    """Write the scaled store; returns the manifest."""
    from vineyard.perception.waste.positives import gsd_estimate, image_scales
    from vineyard.perception.waste.uavvaste import load_dataset

    ds = load_dataset(src_root)
    params = _positive_params()
    scales = image_scales(ds, params)
    fallback = gsd_estimate([o.long_side for o in ds.objects], ASSUMED_OBJECT_M)
    splits = split_of_files(src_root)
    by_image = ds.objects_by_image()
    images, anns = [], []
    for image_id, img in sorted(ds.images.items()):
        gsd = scales[image_id].gsd_m if image_id in scales else fallback
        small = degrade_to_target(read_rgb(ds.image_path(img)), gsd)
        s_x, s_y = small.shape[1] / img.width, small.shape[0] / img.height
        s = 0.5 * (s_x + s_y)
        name = Path(img.file_name).with_suffix(".jpg").name
        write_jpeg(out_dir / "images" / name, small, quality)
        objs = by_image.get(image_id, ())
        images.append(_image_record(img, name, gsd, small.shape, splits.get(img.file_name, "train"), len(objs)))
        anns.extend({"image_id": image_id, "category_id": 0, "role": WASTE,
                     "bbox": [round(v * s, 2) for v in o.bbox], "segmentation": _scale_polys(o.polygons, s)}
                    for o in objs)
        if len(images) % 100 == 0:
            log.info("uavvaste: %d / %d images", len(images), len(ds.images))
    doc = {"images": images, "annotations": anns, "categories": [CATEGORY]}
    (out_dir / "annotations.json").write_text(json.dumps(doc), encoding="utf-8")
    manifest = _manifest(images, anns, src_root)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _manifest(images: list[dict], anns: list[dict], src_root: Path) -> dict:
    source = json.loads((src_root / "SOURCE.json").read_text(encoding="utf-8"))
    sides = [max(a["bbox"][2], a["bbox"][3]) for a in anns]
    return {
        "dataset": "UAVVaste", "doi": source.get("doi"), "licence": source.get("licence"),
        "attribution": source.get("attribution"), "target_gsd_m": TARGET_GSD_M,
        "assumed_object_m": ASSUMED_OBJECT_M, "n_images": len(images), "n_objects": len(anns),
        "n_val_images": sum(i["split"] == "val" for i in images),
        "median_long_side_px": float(np.median(sides)) if sides else None,
        "median_image_px": [float(np.median([i["width"] for i in images])),
                            float(np.median([i["height"] for i in images]))],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(convert(), indent=2))
