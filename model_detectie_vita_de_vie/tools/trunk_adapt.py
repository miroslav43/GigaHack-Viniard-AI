"""Trunk model for new vineyards (model_trunchi_v2): multi-vineyard training + a "post" class + self-training on the
new vineyard's own unlabeled photos. Round 1 (robot view, Aveleda) keeps using the original model_trunchi.pt.

Data (classes: 0 = trunchi, 1 = stalp):
  - label map v8 (Aveleda, Portugal, robot camera)  data/labelmap/{train,valid,test}   trunk + truk -> trunchi
  - HUMAIN-Lab vine-trunk (3 vineyards, Greece)      data/humain/git/...                only the 629 original photos
  - posts: auto-labeled by YOLOE ("wooden post", "metal pole"), dropped where they overlap a labeled trunk
  - EscaYard block B7 (Spain): photos without labels -> trunks auto-labeled by the previous model (self-training)
Test: EscaYard block B9 is never used for training (different vines than B7).

Steps (from the project folder):
  python -m tools.trunk_adapt build                  # label map + HUMAIN + posts -> data/trunk_v2
  python -m tools.trunk_adapt train --name r0        # fine-tune from models/model_trunchi.pt
  python -m tools.trunk_adapt pseudo --weights runs/trunk/r0/weights/best.pt   # auto-label B7, add to the dataset
  python -m tools.trunk_adapt train --name r1 --init runs/trunk/r0/weights/best.pt
  python -m tools.trunk_adapt eval --weights runs/trunk/r1/weights/best.pt     # B9: photos with a trunk + drawn photos
"""
import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import cv2

# Self-contained so the same file runs in Colab: TRUNK_ROOT = folder with data/ and models/, TRUNK_RUNS = where runs go
ROOT = Path(os.environ.get("TRUNK_ROOT", Path(__file__).resolve().parent.parent))
RUNS = Path(os.environ.get("TRUNK_RUNS", ROOT / "runs" / "trunk"))
DS = ROOT / "data" / "trunk_v2"
LM, HUM = ROOT / "data" / "labelmap", ROOT / "data" / "humain" / "git"
ESCA = ROOT / "data" / "escayard"
POST_PROMPTS = ["wooden post", "metal pole", "grapevine trunk", "tree trunk"]
MAX_SIDE = 1280  # EscaYard phone photos are shrunk to this before training / testing


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(a, b):
    """Which share of the smaller box lies inside the other one."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def read_yolo(path):
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append((int(p[0]), *map(float, p[1:])))
    return boxes


def to_xyxy(b, w, h):
    _, cx, cy, bw, bh = b
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


def to_line(c, box, w, h):
    x1, y1, x2, y2 = box
    return f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def escayard_blocks():
    blocks = {"B7": [], "B9": []}
    with open(ESCA / "datasheet.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["Vineyard"].strip() in blocks and r["File"].strip():
                blocks[r["Vineyard"].strip()].append(r["File"].strip())
    photos = {p.name: p for p in (ESCA / "photos").rglob("*.jpg")}
    return {b: [photos[n] for n in sorted(names) if n in photos] for b, names in blocks.items()}


def shrink(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def is_trunk_shape(box, w, h):
    """Trunks stand in the lower part of the photo and are taller than wide; posts reach the top wires."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return bh >= 0.8 * bw and y2 >= 0.55 * h and bh <= 0.75 * h


def is_post_shape(box, w, h):
    x1, y1, x2, y2 = box
    return (y2 - y1) >= 2.5 * (x2 - x1)


class PostLabeler:
    def __init__(self):
        from ultralytics import YOLOE
        local = ROOT / "models" / "yoloe-11l-seg.pt"
        self.m = YOLOE(str(local) if local.exists() else "yoloe-11l-seg.pt")  # Ultralytics downloads it if missing
        self.m.set_classes(POST_PROMPTS, self.m.get_text_pe(POST_PROMPTS))

    def posts(self, img, trunks, conf=0.3):
        """Post boxes (xyxy) in img that do not overlap any trunk box."""
        h, w = img.shape[:2]
        r = self.m.predict(img, imgsz=1024, conf=conf, verbose=False)[0]
        out = []
        for b, k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
            if POST_PROMPTS[int(k)] in ("wooden post", "metal pole") and is_post_shape(b, w, h):
                if all(iou(b, t) < 0.1 and inside(b, t) < 0.3 for t in trunks):
                    out.append(b)
        return out


def write_item(split, name, img, lines):
    for sub in ("images", "labels"):
        (DS / split / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DS / split / "images" / f"{name}.jpg"), img)
    (DS / split / "labels" / f"{name}.txt").write_text("\n".join(lines))


def write_yaml():
    (DS / "data.yaml").write_text(
        f"path: {DS.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        "names:\n  0: trunchi\n  1: stalp\n", encoding="utf-8")


def build(args):
    if DS.exists():
        shutil.rmtree(DS)
    posts = PostLabeler()
    sources = [("lm", LM / s / "images", LM / s / "labels", s, {2: 0, 3: 0}) for s in ("train", "valid", "test")]
    sources += [("hu", HUM / "train" / "Original_photos", HUM / "train" / "Original_photos", "train", {0: 0}),
                ("hu", HUM / "valid", HUM / "valid", "valid", {0: 0}),
                ("hu", HUM / "test", HUM / "test", "test", {0: 0})]
    n_posts = 0
    for prefix, img_dir, lbl_dir, split, cmap in sources:
        files = sorted(img_dir.glob("*.jpg"))
        for i, p in enumerate(files, 1):
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            trunks = [to_xyxy(b, w, h) for b in read_yolo(lbl_dir / f"{p.stem}.txt") if b[0] in cmap]
            lines = [to_line(0, t, w, h) for t in trunks]
            if split != "test":  # the test sets keep only their real labels
                found = posts.posts(img, trunks)
                n_posts += len(found)
                lines += [to_line(1, b, w, h) for b in found]
            write_item(split, f"{prefix}_{p.stem}", img, lines)
            if i % 200 == 0 or i == len(files):
                print(f"{prefix} {split}: {i}/{len(files)} ({i / len(files):.0%})", flush=True)
    write_yaml()
    print("Posts auto-labeled:", n_posts)


def pseudo(args):
    from ultralytics import YOLO
    m = YOLO(args.weights)
    posts = PostLabeler()
    b7 = escayard_blocks()["B7"]
    random.Random(0).shuffle(b7)
    n_valid = len(b7) // 6
    kept = skipped = 0
    for split in ("train", "valid"):  # replace the previous round's pseudo-labels
        for old in (DS / split / "images").glob("es_*"):
            old.unlink()
            (DS / split / "labels" / f"{old.stem}.txt").unlink(missing_ok=True)
    for i, p in enumerate(b7):
        img = shrink(cv2.imread(str(p)))
        h, w = img.shape[:2]
        r = m.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        trunks = [b for b, k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())
                  if int(k) == 0 and is_trunk_shape(b, w, h)]
        if not trunks:
            # the trunk is almost always there: a photo without a pseudo-label would teach "trunk = background"
            skipped += 1
            continue
        found_posts = posts.posts(img, trunks)
        lines = [to_line(0, t, w, h) for t in trunks] + [to_line(1, b, w, h) for b in found_posts]
        kept += len(trunks)
        write_item("valid" if i < n_valid else "train", f"es_{p.stem}", img, lines)
    print(f"EscaYard B7: {len(b7)} photos, {kept} pseudo-labeled trunks (conf >= {args.conf}, imgsz {args.imgsz}), "
          f"{skipped} photos without a trunk left out")


def train(args):
    from ultralytics import YOLO
    m = YOLO(args.init)
    m.train(data=str(DS / "data.yaml"), imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
            workers=2, project=str(RUNS), name=args.name, exist_ok=True, plots=False,
            hsv_h=0.03, hsv_s=0.8, hsv_v=0.5, degrees=5, scale=0.6, amp=args.amp)
    best = RUNS / args.name / "weights" / "best.pt"
    res = YOLO(str(best)).val(data=str(DS / "data.yaml"), split="test", imgsz=args.imgsz, verbose=False, plots=False)
    print(f"Test (label map + HUMAIN) trunk mAP50: {res.box.ap50[0]:.3f}")


def evaluate(args):
    from ultralytics import YOLO
    m = YOLO(args.weights)
    w = Path(args.weights)
    out = RUNS / f"eval_B9_{w.parent.parent.name if w.parent.name == 'weights' else w.stem}"
    out.mkdir(parents=True, exist_ok=True)
    b9 = escayard_blocks()["B9"]
    found = 0
    for p in b9:
        img = shrink(cv2.imread(str(p)))
        r = m.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        trunks = [(b, c) for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()) if int(k) == 0]
        found += bool(trunks)
        for b, c in trunks:
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 140, 255), 4)
            cv2.putText(img, f"trunk {c:.2f}", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 140, 255), 3)
        cv2.imwrite(str(out / p.name), img)
    print(f"EscaYard B9 (never trained on): photos with a trunk {found}/{len(b9)} ({found / len(b9):.0%}) -> {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--init", default=str(ROOT / "models" / "model_trunchi.pt"))
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=8)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--amp", action="store_true", help="mixed precision (use on Colab T4; GTX 16xx cards misbehave with it)")
    for name in ("pseudo", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--weights", required=True)
        s.add_argument("--imgsz", type=int, default=640)
        s.add_argument("--conf", type=float, default=0.5 if name == "pseudo" else 0.4)
    args = p.parse_args()
    {"build": build, "train": train, "pseudo": pseudo, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
