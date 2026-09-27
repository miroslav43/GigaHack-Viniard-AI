"""Box-level test of the whole detector (all models + merging) on photos that have YOLO labels.

Unlike EscaYard (only counts per vine), here every box found is matched to a real box, so we get real precision
(how many boxes shown are right) and recall (how many real objects are found), for several confidence thresholds:
this is how the thresholds in vineyard/config.py should be chosen.

Test sets with labels on the laptop:
  Vairão (Portugal, Zenodo 18152298): 494 frames, trunk outlines, never used for training -> unseen-vineyard trunk test
    python -m tools.evaluate_boxes --images data/vairao --classes 0=trunk --output results/boxes_vairao
  label map test = round 1 (Aveleda, robot): bunch + trunk boxes, but the old trunk model was trained on this vineyard
    python -m tools.evaluate_boxes --images data/labelmap/test/images --classes 0=bunch 1=bunch 2=trunk 3=trunk

Boxes of different datasets are drawn differently (a trunk may be the whole woody part or only its lowest piece), so
the report gives the scores at IoU 0.5 (strict) and 0.3 (lenient).
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import cv2

from tools.evaluate_escayard import share
from vineyard.config import MODELS, TILE_IOU
from vineyard.detector import VineyardDetector, iou
from vineyard.drawing import draw

THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MATCH_IOUS = (0.5, 0.3)
LOWEST_CONF = min(THRESHOLDS)
IMAGE_TYPES = {".jpg", ".jpeg", ".png"}


def read_labels(path, width, height, classes):
    """YOLO label file -> list of (label, box in pixels). Boxes ("c x y w h") and outlines ("c x1 y1 x2 y2 ...") both work."""
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().split("\n"):
        values = line.split()
        if len(values) < 5 or int(values[0]) not in classes:
            continue
        nums = [float(v) for v in values[1:]]
        if len(nums) == 4:
            x, y, w, h = nums
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        else:  # outline: the box around all its points
            xs, ys = nums[0::2], nums[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append((classes[int(values[0])], (x1 * width, y1 * height, x2 * width, y2 * height)))
    return boxes


def match(found, real, min_iou):
    """Greedy matching, most confident box first. Returns (conf, is_right) per found box and the number of real boxes."""
    free = list(real)
    marked = []
    for d in sorted(found, key=lambda d: d.conf, reverse=True):
        best = max(free, key=lambda b: iou(b, d.box), default=None)
        if best is not None and iou(best, d.box) >= min_iou:
            free.remove(best)
            marked.append((d.conf, True))
        else:
            marked.append((d.conf, False))
    return marked


def scores(per_image, label, threshold, min_iou):
    """per_image: list of (found detections, real boxes). -> (right, found, real)"""
    right = found = real = 0
    for dets, gt in per_image:
        mine = [d for d in dets if d.label == label and d.conf >= threshold]
        truth = [b for l, b in gt if l == label]
        marked = match(mine, truth, min_iou)
        right += sum(ok for _, ok in marked)
        found += len(marked)
        real += len(truth)
    return right, found, real


def table(per_image, real_pipeline, labels):
    """per_image: detections at the lowest threshold (for the sweep); real_pipeline: detections of the detector exactly
    as configured in vineyard/config.py (each model with its own threshold)."""
    lines = []
    for label in labels:
        for min_iou in MATCH_IOUS:
            lines.append(f"== {label} | a found box is right when IoU >= {min_iou} ==")
            lines.append(f"{'conf':>5} {'precision':>10} {'recall':>8} {'F1':>6}   right/found, right/real")
            for t in THRESHOLDS:
                right, found, real = scores(per_image, label, t, min_iou)
                p = right / found if found else 0.0
                r = right / real if real else 0.0
                f1 = 2 * p * r / (p + r) if p + r else 0.0
                lines.append(f"{t:>5.1f} {p:>10.2f} {r:>8.2f} {f1:>6.2f}   {right}/{found}, {right}/{real}")
            right, found, real = scores(real_pipeline, label, 0.0, min_iou)
            lines.append(f"As configured now: precision {share(right, found)}, recall {share(right, real)}")
            lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder with photos; labels in the matching 'labels' folder")
    p.add_argument("--classes", nargs="+", required=True, help="label class id -> our label, e.g. 0=bunch 3=trunk")
    p.add_argument("--output", default="results/boxes")
    p.add_argument("--tiles", type=int, default=1)
    p.add_argument("--tile-iou", type=float, default=TILE_IOU)
    p.add_argument("--save-every", type=int, default=20, help="save every N-th drawn photo")
    args = p.parse_args()

    classes = {int(k): v for k, v in (c.split("=") for c in args.classes)}
    labels = sorted(set(classes.values()))
    photos = sorted(f for f in Path(args.images).rglob("*") if f.suffix.lower() in IMAGE_TYPES)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # only the models that find these labels; once exactly as configured, once at a low threshold for the sweep
    models = [cfg for cfg in MODELS if set(cfg.classes.values()) & set(labels)]
    detector = VineyardDetector(models=models, tiles=args.tiles, tile_iou=args.tile_iou)
    sweep = VineyardDetector(models=[replace(cfg, conf=LOWEST_CONF) for cfg in models], tiles=args.tiles,
                             tile_iou=args.tile_iou)

    per_image, real_pipeline, rows = [], [], []
    for i, photo in enumerate(photos, 1):
        img = cv2.imread(str(photo))
        if img is None:
            print(f"WARNING: cannot read {photo}")
            continue
        h, w = img.shape[:2]
        gt = read_labels(photo.parent.parent / "labels" / f"{photo.stem}.txt", w, h, classes)
        dets = [d for d in detector.detect([img])[0] if d.label in labels]
        real_pipeline.append((dets, gt))
        per_image.append(([d for d in sweep.detect([img])[0] if d.label in labels], gt))
        rows.append({"image": str(photo), **{f"real {l}": sum(g == l for g, _ in gt) for l in labels},
                     **{f"found {l}": sum(d.label == l for d in dets) for l in labels}})
        if i % args.save_every == 0:
            drawn = draw(img, dets, title=photo.name)
            for _, (x1, y1, x2, y2) in gt:  # real boxes in white
                cv2.rectangle(drawn, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
            cv2.imwrite(str(out / f"{photo.stem}.jpg"), drawn)
        if i % 50 == 0 or i == len(photos):
            print(f"Processed {i}/{len(photos)} ({i / len(photos):.0%})", flush=True)

    report = [f"{args.images}: {len(per_image)} photos, tiles={args.tiles}, real boxes: " +
              ", ".join(f"{l} {sum(g == l for _, gt in per_image for g, _ in gt)}" for l in labels), ""]
    report += table(per_image, real_pipeline, labels)
    (out / "boxes_report.txt").write_text("\n".join(report), encoding="utf-8")
    with open(out / "boxes_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n" + "\n".join(report))
    print(f"Details: {out}")


if __name__ == "__main__":
    main()
