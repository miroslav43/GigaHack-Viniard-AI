"""VINEPICs (Zenodo 7866442, CC BY 4.0, Piacenza, Italy: Red Globe, Cabernet Sauvignon, Ortrugo; 238 photos, 2,403
bunches) -> YOLO boxes, as a bunch test on a vineyard none of our models has seen.

Download (346 MB) into data/vinepics and unzip, then (from the model folder):
  python -m tools.prepare_vinepics
  python -m tools.evaluate_boxes --images data/vinepics/yolo/images --classes 0=bunch --output results/boxes_vinepics
"""
import json
import shutil
from pathlib import Path

SRC = Path("data/vinepics/data")
DST = Path("data/vinepics/yolo")


def yolo_lines(annotations, width, height):
    """COCO boxes [x, y, w, h] in pixels -> YOLO lines "0 cx cy w h" (0-1)."""
    return [f"0 {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}"
            for x, y, w, h in (a["bbox"] for a in annotations)]


def main():
    coco = json.loads((SRC / "annotations" / "VINEPICs_annotations.json").read_text(encoding="utf-8"))
    per_image = {}
    for a in coco["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    for sub in ("images", "labels"):
        (DST / sub).mkdir(parents=True, exist_ok=True)
    for im in coco["images"]:
        session, name = im["file_name"].split("/")
        stem = f"{session}_{Path(name).stem}"  # names repeat between sessions (rgb1.png ...)
        shutil.copy(SRC / "images" / im["file_name"], DST / "images" / f"{stem}{Path(name).suffix}")
        (DST / "labels" / f"{stem}.txt").write_text("\n".join(yolo_lines(per_image.get(im["id"], []), im["width"], im["height"])))
    print(f"{len(coco['images'])} photos, {len(coco['annotations'])} bunches -> {DST}")


if __name__ == "__main__":
    main()
