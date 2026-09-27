"""Evaluates the 3 models on the downy mildew dataset (Mendeley yxf2yv5ymt, CC BY 4.0): 99 photos of whole Merlot vines
taken from a tractor at 70 cm height and 50 cm from the vine (close to the robot's view), Bordeaux, July 2018.
72 photos come with a mask of the downy mildew pixels (black = mildew), drawn by experts.

  - box on mildew:   share of "diseased leaf" boxes that contain real mildew pixels (1.0 = every box is on a symptom)
  - spots found:     share of mildew spots (connected mildew areas, >= MIN_SPOT_PX pixels) touched by at least one box
  - vine level:      mildew area vs number of boxes per photo

Note: the diseased-leaf model was trained on Flavescence doree / Esca leaves, not on downy mildew.

Run (from the model folder):
  python -m tools.evaluate_mildew --tiles 3
"""
import argparse
import csv
import statistics
from pathlib import Path

import cv2
import numpy as np

from vineyard.detector import VineyardDetector, count
from vineyard.drawing import draw

MIN_SPOT_PX = 400  # smaller mildew specks are ignored (at 4 px/mm this is about 5 x 5 mm)


def mildew_mask(path):
    """Mildew_annotation masks are RGBA: black = mildew, white = everything else."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    rgb = m[..., :3] if m.ndim == 3 else m[..., None]
    return rgb.max(axis=-1) < 128


def score_photo(mask, boxes):
    """Returns (boxes on mildew, total boxes, spots touched, total spots, mildew pixels)."""
    on = 0
    for x1, y1, x2, y2 in boxes:
        if mask[int(y1):int(y2), int(x1):int(x2)].any():
            on += 1
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    spots = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= MIN_SPOT_PX]
    covered = np.zeros_like(mask)
    for x1, y1, x2, y2 in boxes:
        covered[int(y1):int(y2), int(x1):int(x2)] = True
    touched = sum(covered[labels == i].any() for i in spots)
    return on, len(boxes), touched, len(spots), int(mask.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/mildew/downy_mildew_images")
    p.add_argument("--output", default="results/mildew")
    p.add_argument("--tiles", type=int, default=3)
    args = p.parse_args()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # photo im_00162.jpg <-> mask calc_00162.png
    masks = {m.stem.replace("calc_", "im_"): m for m in (root / "Mildew_annotation").glob("*.png")}

    detector = VineyardDetector(tiles=args.tiles)
    rows = []
    photos = sorted(root.glob("*.jpg"))
    for i, photo in enumerate(photos, 1):
        img = cv2.imread(str(photo))
        det = detector.detect([img])[0]
        n = count(det)
        row = {"file": photo.name, "bunch": n["bunch"], "trunk": n["trunk"], "diseased leaf": n["diseased leaf"]}
        drawn = draw(img, det, title=photo.name)
        if photo.stem in masks:
            mask = mildew_mask(masks[photo.stem])
            leaf_boxes = [d.box for d in det if d.label == "diseased leaf"]
            on, total, touched, spots, px = score_photo(mask, leaf_boxes)
            row.update({"boxes_on_mildew": on, "spots_touched": touched, "spots": spots, "mildew_px": px})
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(drawn, contours, -1, (0, 0, 255), 3)  # real mildew in red
        rows.append(row)
        cv2.imwrite(str(out / photo.name), drawn)
        if i % 20 == 0 or i == len(photos):
            print(f"Processed {i}/{len(photos)} ({i / len(photos):.0%})", flush=True)

    fields = ["file", "bunch", "trunk", "diseased leaf", "boxes_on_mildew", "spots_touched", "spots", "mildew_px"]
    with open(out / "mildew_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if "spots" in r]
    boxes = sum(r["diseased leaf"] for r in scored)
    on = sum(r["boxes_on_mildew"] for r in scored)
    spots = sum(r["spots"] for r in scored)
    touched = sum(r["spots_touched"] for r in scored)
    report = [f"Downy mildew dataset: {len(rows)} photos, {len(scored)} with a mildew mask", "",
              "== Diseased leaves (on the 72 photos with a mildew mask) ==",
              f"Boxes found: {boxes} | on real mildew: {on} ({on / boxes:.0%})" if boxes else "Boxes found: 0",
              f"Mildew spots (>= {MIN_SPOT_PX} px): {spots} | touched by a box: {touched} ({touched / spots:.0%})" if spots else "",
              f"Photos with mildew: {sum(r['mildew_px'] > 0 for r in scored)} | with at least one box: "
              f"{sum(r['diseased leaf'] > 0 for r in scored)}"]
    if len(scored) > 2 and len({r['diseased leaf'] for r in scored}) > 1:
        report.append("Correlation mildew area vs boxes per photo: "
                      f"{statistics.correlation([r['mildew_px'] for r in scored], [r['diseased leaf'] for r in scored]):.2f}")
    report += ["", "== Bunches / trunks (all photos, no ground truth: check the drawn photos) ==",
               f"Bunches: {sum(r['bunch'] for r in rows)} (avg {statistics.mean(r['bunch'] for r in rows):.1f}/photo) | "
               f"photos with a trunk: {sum(r['trunk'] > 0 for r in rows)}/{len(rows)}"]
    (out / "mildew_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"\nDrawn photos (real mildew outlined in red): {out}")


if __name__ == "__main__":
    main()
