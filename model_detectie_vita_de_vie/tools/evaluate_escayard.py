"""Evaluates the 3 models on EscaYard (Zenodo 10362567, CC BY 4.0): 253 smartphone photos of whole vines,
Northern Spain, July 2022, each with its real Esca status (YES/NO) and, for some vines, the real number of bunches.

None of these photos was used in training, so this is an independent test:
  - disease: a vine is "suspect" when the model finds at least N diseased leaves -> compared with the real Esca status
  - bunches: bunches found by the model vs bunches counted by people (only 42 vines have a count: 24 in B7, 18 in B9).
    There are no boxes in the dataset, so this compares totals per vine: it is NOT recall (false bunches are counted too)
  - trunks: in how many photos at least one trunk is found (no boxes in the dataset: checked visually)

Run (from the model folder):
  python -m tools.evaluate_escayard                      # uses data/escayard/photos + data/escayard/datasheet.csv
  python -m tools.evaluate_escayard --save-every 1       # save every drawn photo (default: every 5th)
  python -m tools.evaluate_escayard --vineyards B9       # only vineyard B9 (the final test: B7 is used for auto-labels)
"""
import argparse
import csv
import math
import statistics
from pathlib import Path

import cv2

from vineyard.config import TILE_IOU
from vineyard.detector import VineyardDetector, count
from vineyard.drawing import draw

SUSPECT_THRESHOLDS = (1, 2, 3, 5, 8)


def load_datasheet(path):
    rows = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            n = r["Number of grape clusters"].strip()
            status = r["Esca"].strip().upper()
            # "NO*" (uncertain) and empty statuses are left out of the disease scores
            esca = True if status == "YES" else False if status == "NO" else None
            rows[r["File"].strip()] = {"esca": esca, "bunches_true": int(n) if n else None,
                                       "time": r["Time"].strip(), "vineyard": r["Vineyard"].strip()}
    return rows


def find_photos(folder):
    return {p.name: p for p in Path(folder).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg"}}


def shrink(img, max_side=2048):
    """The phone photos are up to 48 MP: shrink them once, the models resize again anyway."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def wilson(k, n, z=1.96):
    """95% interval for a share k/n (Wilson): with few photos, 96/120 = 80% really means "somewhere in 72-86%"."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def share(k, n):
    """'96/120 = 80% (95% interval 72-86%)'"""
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%} (95% interval {lo:.0%}-{hi:.0%})" if n else "0/0"


def disease_table(results):
    lines = ["Suspect vine = at least N diseased leaves found",
             f"{'N':>3} {'sick found':>11} {'healthy flagged':>16} {'precision':>10} {'recall':>7}"]
    sick = [r for r in results if r["esca"] is True]
    healthy = [r for r in results if r["esca"] is False]
    for n in SUSPECT_THRESHOLDS:
        tp = sum(r["diseased leaf"] >= n for r in sick)
        fp = sum(r["diseased leaf"] >= n for r in healthy)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(sick) if sick else 0.0
        lines.append(f"{n:>3} {f'{tp}/{len(sick)}':>11} {f'{fp}/{len(healthy)}':>16} {precision:>10.2f} {recall:>7.2f}")
    tp3 = sum(r["diseased leaf"] >= 3 for r in sick)
    fp3 = sum(r["diseased leaf"] >= 3 for r in healthy)
    lines.append(f"At N=3: sick found {share(tp3, len(sick))}, healthy flagged {share(fp3, len(healthy))}")
    if sick and healthy:
        lines.append(f"Average diseased leaves per photo: sick vines {statistics.mean(r['diseased leaf'] for r in sick):.1f}, "
                     f"healthy vines {statistics.mean(r['diseased leaf'] for r in healthy):.1f}")
    return lines


def bunch_table(results):
    counted = [r for r in results if r["bunches_true"] is not None]
    if not counted:
        return ["No vines with counted bunches."]
    errors = [r["bunch"] - r["bunches_true"] for r in counted]
    lines = [f"Vines with bunches counted by people: {len(counted)}",
             f"Real bunches: {sum(r['bunches_true'] for r in counted)} | found by the model: {sum(r['bunch'] for r in counted)} "
             f"(totals, not recall: no boxes to match)",
             f"Mean absolute error per vine: {statistics.mean(abs(e) for e in errors):.1f} bunches "
             f"(real average: {statistics.mean(r['bunches_true'] for r in counted):.1f} per vine)"]
    if len(counted) > 2 and len({r['bunch'] for r in counted}) > 1:
        lines.append(f"Correlation real vs found: {statistics.correlation([r['bunches_true'] for r in counted], [r['bunch'] for r in counted]):.2f}")
    return lines


def trunk_line(results):
    return f"Photos with at least one trunk: {share(sum(r['trunk'] > 0 for r in results), len(results))}"


def build_report(results):
    """The whole report, and again per vineyard (B7 / B9) when both are present."""
    report = [f"EscaYard: {len(results)} photos ({sum(r['esca'] is True for r in results)} vines with Esca, "
              f"{sum(r['esca'] is False for r in results)} healthy, {sum(r['esca'] is None for r in results)} uncertain)", "",
              "== Diseased leaves (vine level) =="] + disease_table(results) + ["", "== Bunches =="] + bunch_table(results) + [
              "", "== Trunks ==", trunk_line(results)]
    vineyards = sorted({r["vineyard"] for r in results if r["vineyard"]})
    if len(vineyards) > 1:
        for v in vineyards:
            part = [r for r in results if r["vineyard"] == v]
            report += ["", f"======== Vineyard {v} only ({len(part)} photos) ========",
                       "== Diseased leaves =="] + disease_table(part) + ["== Bunches =="] + bunch_table(part) + [
                       "== Trunks ==", trunk_line(part)]
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--photos", default="data/escayard/photos")
    p.add_argument("--datasheet", default="data/escayard/datasheet.csv")
    p.add_argument("--output", default="results/escayard")
    p.add_argument("--save-every", type=int, default=5, help="save every N-th drawn photo (1 = all)")
    p.add_argument("--tiles", type=int, default=3, help="N x N pieces for the bunch model (1 = whole image only)")
    p.add_argument("--tile-iou", type=float, default=TILE_IOU, help="overlap from which boxes of neighbouring tiles are merged")
    p.add_argument("--vineyards", nargs="+", help="only these vineyards, e.g. B9 (default: all)")
    args = p.parse_args()

    sheet = load_datasheet(args.datasheet)
    photos = find_photos(args.photos)
    missing = sorted(set(sheet) - set(photos))
    if missing:
        print(f"WARNING: {len(missing)} photos from the datasheet were not found, e.g. {missing[:3]}")
    names = sorted(n for n in set(sheet) & set(photos) if not args.vineyards or sheet[n]["vineyard"] in args.vineyards)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    detector = VineyardDetector(tiles=args.tiles, tile_iou=args.tile_iou)
    results = []
    for i, name in enumerate(names, 1):
        img = cv2.imread(str(photos[name]))
        if img is None:
            print(f"WARNING: cannot read {name}")
            continue
        img = shrink(img)
        det = detector.detect([img])[0]
        n = count(det)
        results.append({"file": name, **sheet[name], "bunch": n["bunch"], "trunk": n["trunk"], "diseased leaf": n["diseased leaf"]})
        if i % args.save_every == 0:
            tag = {True: "ESCA", False: "healthy", None: "uncertain"}[sheet[name]["esca"]]
            cv2.imwrite(str(out / f"{Path(name).stem}_{tag}.jpg"), draw(img, det, title=f"{name} | real: {tag}"))
        if i % 25 == 0 or i == len(names):
            print(f"Processed {i}/{len(names)} ({i / len(names):.0%})", flush=True)

    with open(out / "escayard_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    report = build_report(results)
    (out / "escayard_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n" + "\n".join(report))
    print(f"\nDetails: {out / 'escayard_results.csv'} | drawn photos: {out}")


if __name__ == "__main__":
    main()
