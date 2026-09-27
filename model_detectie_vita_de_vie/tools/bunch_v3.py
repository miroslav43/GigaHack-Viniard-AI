"""Bunch model v3 for unseen vineyards: model_ciorchine_v2 fine-tuned on ViViD-5K (13 varieties) + WGISD.

Measured on 27 Sept on VINEPICs (Italy, 238 photos, 2,403 bunches, never used by any of our models):
the current bunch models find only 22% of the bunches (precision 87%); on the 2022-09-15 sessions (dark Cabernet,
low camera, bright sky) v1 finds 10/714 and v2 13/714. The old v1 model is just as bad, so the red-leaf negatives of v2
are not the cause: the models only know WGISD. ViViD-5K adds 4,000 training photos from 13 varieties.

Data:
  ViViD-5K (Hugging Face XZhi/ViViD-5k, CC BY 4.0): the 1024 px previews in parquet/ (the full photos are ~40 GB) +
    data/anns/instances_updated.json. Its boxes are [[x1, y1, x2, y2]] in full-photo pixels (checked: all 18,561 fit
    the photo as corners, 14,446 would not as x, y, w, h); 14 boxes are empty and skipped.
    train -> train, val -> valid (chooses best.pt), test -> test_vivid
  WGISD: train.txt -> train, test.txt -> test_wgisd (as for the first bunch model)
  VINEPICs: all 238 photos -> test_vinepics (the unseen vineyard)

Steps (the same file runs in Colab: BUNCH_ROOT = folder with data/ and models/, BUNCH_RUNS = where runs go):
  python bunch_v3.py build    # needs data/vivid (parquet + instances_updated.json), data/wgisd, data/vinepics
  python bunch_v3.py train
  python bunch_v3.py report --weights <best.pt>
"""
import argparse
import io
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(os.environ.get("BUNCH_ROOT", Path(__file__).resolve().parent.parent))
RUNS = Path(os.environ.get("BUNCH_RUNS", ROOT / "runs" / "bunch_v3"))
DS = ROOT / "data" / "bunch_v3"
VIVID, WGISD, VINEPICS = ROOT / "data" / "vivid", ROOT / "data" / "wgisd", ROOT / "data" / "vinepics" / "data"
OLD = ROOT / "models" / "model_ciorchine_v2.pt"
IMGSZ, CONF = 640, 0.5  # as in vineyard/config.py
TEST_SETS = ("test_vinepics", "test_wgisd", "test_vivid")


def vivid_boxes(raw_bbox, width, height, preview_w, preview_h):
    """ViViD box [[x1, y1, x2, y2]] in full-photo pixels -> YOLO line for the 1024 px preview, or None if empty."""
    b = raw_bbox[0] if raw_bbox and isinstance(raw_bbox[0], list) else raw_bbox
    if not b or len(b) != 4:
        return None
    x1, y1, x2, y2 = (min(max(v, 0), lim) for v, lim in zip(b, (width, height, width, height)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    sx, sy = preview_w / width, preview_h / height
    x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
    return (f"0 {(x1 + x2) / 2 / preview_w:.6f} {(y1 + y2) / 2 / preview_h:.6f} "
            f"{(x2 - x1) / preview_w:.6f} {(y2 - y1) / preview_h:.6f}")


def write(split, name, image_bytes_or_path, lines):
    for sub in ("images", "labels"):
        (DS / split / sub).mkdir(parents=True, exist_ok=True)
    src = image_bytes_or_path
    ext = ".jpg" if isinstance(src, bytes) else Path(src).suffix.lower()
    target = DS / split / "images" / f"{name}{ext}"
    if isinstance(src, bytes):
        target.write_bytes(src)
    else:
        shutil.copy(src, target)
    (DS / split / "labels" / f"{name}.txt").write_text("\n".join(lines))


def build_vivid():
    import pandas as pd
    from PIL import Image

    coco = json.loads((VIVID / "instances_updated.json").read_text(encoding="utf-8"))
    images = {im["file_name"]: im for im in coco["images"]}
    boxes = {}
    for a in coco["annotations"]:
        if a["category_id"] == 1:
            boxes.setdefault(a["image_id"], []).append(a["bbox"])
    counts, skipped, mismatch = {}, 0, 0
    for split, target in (("train", "train"), ("val", "valid"), ("test", "test_vivid")):
        for shard in sorted(VIVID.glob(f"{split}-*.parquet")):
            for row in pd.read_parquet(shard).itertuples():
                data, name = row.image["bytes"], row.image["path"]
                im = images[name]
                pw, ph = Image.open(io.BytesIO(data)).size
                if abs(pw / im["width"] - ph / im["height"]) > 0.02:  # rotated preview: the boxes would not fit
                    mismatch += 1
                    continue
                lines = [line for line in (vivid_boxes(b, im["width"], im["height"], pw, ph)
                                           for b in boxes.get(im["id"], [])) if line]
                skipped += len(boxes.get(im["id"], [])) - len(lines)
                write(target, f"vv_{Path(name).stem}", data, lines)
                counts[target] = counts.get(target, 0) + 1
            print(f"ViViD {shard.name} done", flush=True)
    print(f"ViViD photos per split: {counts} | empty boxes skipped: {skipped} | rotated previews left out: {mismatch}")


def build_wgisd():
    for listing, target in (("train.txt", "train"), ("test.txt", "test_wgisd")):
        stems = [s.strip() for s in (WGISD / listing).read_text().splitlines() if s.strip()]
        for s in stems:
            write(target, f"wg_{s}", WGISD / "data" / f"{s}.jpg", (WGISD / "data" / f"{s}.txt").read_text().splitlines())
        print(f"WGISD {listing}: {len(stems)} photos", flush=True)


def build_vinepics():
    coco = json.loads((VINEPICS / "annotations" / "VINEPICs_annotations.json").read_text(encoding="utf-8"))
    per = {}
    for a in coco["annotations"]:
        per.setdefault(a["image_id"], []).append(a["bbox"])
    for im in coco["images"]:
        session, name = im["file_name"].split("/")
        W, H = im["width"], im["height"]
        lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}" for x, y, w, h in per.get(im["id"], [])]
        write("test_vinepics", f"vp_{session}_{Path(name).stem}", VINEPICS / "images" / im["file_name"], lines)
    print(f"VINEPICs: {len(coco['images'])} photos", flush=True)


def build(args):
    if DS.exists():
        shutil.rmtree(DS)
    build_vivid()
    build_wgisd()
    build_vinepics()
    (DS / "data.yaml").write_text(f"path: {DS.as_posix()}\ntrain: train/images\nval: valid/images\n"
                                  "names:\n  0: ciorchine\n", encoding="utf-8")
    print({s: len(list((DS / s / "images").iterdir())) for s in ("train", "valid") + TEST_SETS})


def train(args):
    from ultralytics import YOLO

    t0 = time.time()

    def print_epoch(trainer):
        ep, total = trainer.epoch + 1, trainer.epochs
        if ep > total:
            return
        elapsed = time.time() - t0
        print(f"==> Epoch {ep}/{total} ({ep / total:.0%}) | ViViD validation mAP50: "
              f"{trainer.metrics.get('metrics/mAP50(B)', 0.0):.3f} | {elapsed / 60:.0f} min | "
              f"left: max ~{elapsed / ep * (total - ep) / 60:.0f} min", flush=True)

    m = YOLO(args.init)
    m.add_callback("on_fit_epoch_end", print_epoch)
    # AdamW with an explicit learning rate: optimizer="auto" picks its own and ignores lr0 on short fine-tunes
    m.train(data=str(DS / "data.yaml"), imgsz=IMGSZ, epochs=args.epochs, patience=args.epochs, batch=args.batch,
            optimizer="AdamW", lr0=args.lr0, workers=2, project=str(RUNS), name=args.name, exist_ok=True,
            plots=False, hsv_v=0.5, fraction=args.fraction, amp=not args.no_amp)


# ---------- evaluation ----------

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score_set(model, folder):
    """[(found [(box, conf)], real boxes)] per photo, boxes normalized xyxy."""
    import cv2
    out = []
    for p in sorted((folder / "images").iterdir()):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        r = model.predict(img, imgsz=IMGSZ, conf=0.1, verbose=False)[0]
        found = [((b[0] / w, b[1] / h, b[2] / w, b[3] / h), c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        real = []
        for line in (folder / "labels" / f"{p.stem}.txt").read_text().splitlines():
            v = [float(x) for x in line.split()[1:5]]
            if len(v) == 4:
                real.append((v[0] - v[2] / 2, v[1] - v[3] / 2, v[0] + v[2] / 2, v[1] + v[3] / 2))
        out.append((found, real))
    return out


def prf(per_photo, conf, min_iou=0.5):
    right = found = real = 0
    for dets, gt in per_photo:
        free = list(gt)
        for box, c in sorted(dets, key=lambda d: -d[1]):
            if c < conf:
                continue
            found += 1
            best = max(free, key=lambda g: iou(g, box), default=None)
            if best is not None and iou(best, box) >= min_iou:
                free.remove(best)
                right += 1
        real += len(gt)
    p, r = (right / found if found else 0.0), (right / real if real else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def report(args):
    from ultralytics import YOLO
    rows = []
    for name, weights in (("v2 (now)", OLD), ("v3", Path(args.weights))):
        m = YOLO(str(weights))
        r = {"model": name}
        for s in TEST_SETS:
            per = score_set(m, DS / s)
            p, rec, f1 = prf(per, CONF)
            r[f"{s} P/R/F1 at conf {CONF}"] = f"{p:.2f}/{rec:.2f}/{f1:.2f}"
            f1_best, t = max((prf(per, t)[2], t) for t in (0.2, 0.3, 0.4, 0.5, 0.6))
            r[f"{s} best F1 (conf)"] = f"{f1_best:.2f} ({t})"
        rows.append(r)
        print(f"{name} done", flush=True)
    keys = [k for k in rows[0] if k != "model"]
    print(f"\n{'':40s}{rows[0]['model']:>16s}{rows[1]['model']:>16s}")
    for k in keys:
        print(f"{k:40s}{rows[0][k]:>16s}{rows[1][k]:>16s}")
    (RUNS / "bunch_v3_report.json").write_text(json.dumps(rows, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    t = sub.add_parser("train")
    t.add_argument("--name", default="bunch_v3")
    t.add_argument("--init", default=str(OLD))
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr0", type=float, default=0.0005)
    t.add_argument("--fraction", type=float, default=1.0)
    t.add_argument("--no-amp", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--weights", required=True)
    args = p.parse_args()
    {"build": build, "train": train, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
