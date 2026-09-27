"""Builds the combined 2-class dataset (0 = ciorchine / bunch, 1 = trunchi / trunk) for one bunch+trunk model.

Runs in Colab (paths below). Every source is labeled for only some classes; the missing class is filled in by the
current models (teachers) where they are very sure, so the new model is not taught that a real trunk/bunch is background.

Train sources                        bunch labels              trunk labels
  label map v8 (robot, Portugal)     real (tiny + medium)      real
  WGISD (Brazil)                     real                      teacher (trunk v2)
  Pinheiro (Porto) - train part      real                      teacher
  wGrapeUNIPD (Italy)                real                      teacher
  CERTH 300 photos (Greece)          real                      teacher
  HUMAIN-Lab (Greece)                teacher (bunch v2)        real
  EscaYard B7 (Spain)                teacher                   teacher (trunk v2, same as the trunk round)
  mildew even photos                 teacher                   none (negatives for trunk)
Held-out tests (never trained): label map test, WGISD test, Pinheiro test part, Vairão (all), EscaYard B9, mildew odd.
"""
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2

BT = Path("/content/bt")
TV = Path("/content/tv/data")
OUT = Path("/content/bt/ds")
DRIVE = Path("/content/drive/MyDrive/vie_hackathon")
T_BUNCH, T_TRUNK = 0.6, 0.6  # teacher confidence for filling in a missing class
MAX_SIDE = 1280


def shrink(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def yolo_boxes(path, keep):
    """Reads a YOLO label file; keep maps source class -> new class. Polygons are turned into boxes."""
    out = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().split("\n"):
        p = line.split()
        if len(p) < 5 or int(p[0]) not in keep:
            continue
        v = list(map(float, p[1:]))
        if len(v) == 4:
            cx, cy, w, h = v
        else:  # segmentation polygon
            xs, ys = v[0::2], v[1::2]
            cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        out.append((keep[int(p[0])], cx, cy, w, h))
    return out


class Teachers:
    def __init__(self):
        from ultralytics import YOLO
        self.bunch = YOLO(str(DRIVE / "model_ciorchine_v2.pt"))
        self.trunk = YOLO(str(DRIVE / "model_trunchi_v2.pt"))

    def _run(self, m, img, imgsz, conf, cls_name, new_cls):
        r = m.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
        h, w = img.shape[:2]
        return [(new_cls, (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h, (b[2] - b[0]) / w, (b[3] - b[1]) / h)
                for b, k in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()) if r.names[int(k)] == cls_name]

    def bunches(self, img):
        return self._run(self.bunch, img, 960, T_BUNCH, "ciorchine", 0)

    def trunks(self, img):
        return self._run(self.trunk, img, 1280, T_TRUNK, "trunchi", 1)


def write(split, name, img, boxes, copies=1):
    for sub in ("images", "labels"):
        (OUT / split / sub).mkdir(parents=True, exist_ok=True)
    lines = "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes)
    for k in range(copies):
        n = name if copies == 1 else f"{name}_{k}"
        cv2.imwrite(str(OUT / split / "images" / f"{n}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / split / "labels" / f"{n}.txt").write_text(lines)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    T = Teachers()
    stats = {}

    def add(split, prefix, files, real, fill_bunch, fill_trunk, copies=1):
        n = 0
        files = list(files)
        for k, (img_path, lbl_path) in enumerate(files, 1):
            if k % 200 == 0:
                print(f"  {prefix} {split}: {k}/{len(files)} ({k / len(files):.0%})", flush=True)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = shrink(img)
            boxes = real(lbl_path) if real else []
            if fill_bunch:
                boxes += T.bunches(img)
            if fill_trunk:
                boxes += T.trunks(img)
            write(split, f"{prefix}_{Path(img_path).stem}", img, boxes, copies)
            n += 1
        stats[f"{prefix} {split}"] = n
        print(f"{prefix} {split}: {n}", flush=True)

    # label map v8: 0 medium, 1 tiny -> bunch ; 2 truk, 3 trunk -> trunk
    lm_keep = {0: 0, 1: 0, 2: 1, 3: 1}
    for split in ("train", "valid"):
        files = [(p, TV / "labelmap" / split / "labels" / f"{p.stem}.txt") for p in sorted((TV / "labelmap" / split / "images").glob("*.jpg"))]
        add(split, "lm", files, lambda l: yolo_boxes(l, lm_keep), False, False)

    # WGISD: train.txt (minus a validation part, same split as before) ; test.txt stays out
    wg = Path("/content/wgisd")
    ids = [l.strip() for l in open(wg / "train.txt") if l.strip()]
    random.Random(0).shuffle(ids)
    nv = round(len(ids) * 0.15)
    for split, part in (("train", ids[nv:]), ("valid", ids[:nv])):
        add(split, "wg", [(wg / "data" / f"{i}.jpg", wg / "data" / f"{i}.txt") for i in part],
            lambda l: yolo_boxes(l, {0: 0}), False, True)

    # Pinheiro, wGrapeUNIPD, CERTH: real bunch labels (paths found by glob, YOLO format)
    def yolo_pairs(root):
        """(photo, label) pairs: label in the sibling labels/ folder, else any .txt with the same name in the tree."""
        files = sorted(Path(root).rglob("*"))
        by_stem = {}
        for f in files:
            if f.suffix == ".txt" and f.stem != "classes":
                by_stem.setdefault(f.stem, f)
        pairs = []
        for img in files:
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
                if not lbl.exists():
                    lbl = img.with_suffix(".txt") if img.with_suffix(".txt").exists() else by_stem.get(img.stem)
                if lbl is not None and lbl.exists():
                    pairs.append((img, lbl))
        return pairs

    # Pinheiro: every photo exists 11 times (original + 10 augmented copies, e.g. "flip_IMG_...") and a few photos sit in
    # two splits. Keep only originals; any photo that appears in test/ is held out completely (bunch test set).
    import re
    pin_all = [(i, l) for i, l in yolo_pairs(BT / "pinheiro") if re.search(r"IMG_\d+_\d+", i.name)]
    base = lambda p: re.search(r"IMG_\d+_\d+", p.name).group()
    test_ids = {base(img) for img, _ in pin_all if "/test/" in str(img)}
    pin = [(img, lbl) for img, lbl in pin_all if img.name.startswith("IMG_") and base(img) not in test_ids]
    pin = list({base(img): (img, lbl) for img, lbl in pin}.values())  # one copy per photo
    random.Random(1).shuffle(pin)
    add("train", "pin", pin[: int(len(pin) * 0.85)], lambda l: yolo_boxes(l, {0: 0}), False, True)
    add("valid", "pin", pin[int(len(pin) * 0.85):], lambda l: yolo_boxes(l, {0: 0}), False, True)
    pin_test = list({base(img): (img, lbl) for img, lbl in pin_all if base(img) in test_ids and img.name.startswith("IMG_")}.values())
    tdir = BT / "test_pinheiro"
    for sub in ("images", "labels"):
        (tdir / sub).mkdir(parents=True, exist_ok=True)
    for img, lbl in pin_test:
        shutil.copy(img, tdir / "images" / img.name)
        shutil.copy(lbl, tdir / "labels" / lbl.name)
    print("pinheiro test (held out):", len(pin_test), flush=True)
    # wGrapeUNIPD has one class (white grape bunch), whatever its id
    add("train", "wgu", yolo_pairs(BT / "wgrape"), lambda l: yolo_boxes(l, {k: 0 for k in range(10)}), False, True)
    add("train", "certh", [(p, BT / "certh" / "labels" / f"{p.stem}.txt") for p in sorted((BT / "certh" / "images").glob("*.jpg"))],
        lambda l: yolo_boxes(l, {0: 0}), False, True)

    # HUMAIN (Greece): real trunks, teacher bunches
    hu = TV / "humain" / "git"
    for split, d in (("train", hu / "train" / "Original_photos"), ("valid", hu / "valid")):
        add(split, "hu", [(p, d / f"{p.stem}.txt") for p in sorted(d.glob("*.jpg"))], lambda l: yolo_boxes(l, {0: 1}), True, False)

    # EscaYard B7 (not B9!): both classes from teachers ; the 20 hand-labeled photos get their real trunks
    sys.path.insert(0, "/content")
    os.environ.setdefault("TRUNK_ROOT", "/content/tv")
    import trunk_adapt as t
    b7 = [str(p) for p in t.escayard_blocks()["B7"]]
    manual = json.load(open("/content/manual.json")) if Path("/content/manual.json").exists() else {}
    for p in b7:
        img = shrink(cv2.imread(p))
        boxes = T.bunches(img)
        if Path(p).name in manual:
            boxes += [(1, (a + c) / 2, (b + d) / 2, c - a, d - b) for a, b, c, d in manual[Path(p).name]]
            write("train", f"es_{Path(p).stem}", img, boxes, copies=2)
        else:
            boxes += T.trunks(img)
            write("train", f"es_{Path(p).stem}", img, boxes)
    print("es train:", len(b7), flush=True)

    # mildew even photos: canopy without trunks (bunch teacher only)
    mil = sorted(glob.glob("/content/tv/data/mildew/downy_mildew_images/*.jpg"))[0::2]
    add("train", "mil", [(Path(p), None) for p in mil], None, True, False)

    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: train/images\nval: valid/images\nnames:\n  0: ciorchine\n  1: trunchi\n")
    n_tr = len(list((OUT / "train" / "images").glob("*")))
    n_va = len(list((OUT / "valid" / "images").glob("*")))
    print(f"DONE train {n_tr} valid {n_va}", json.dumps(stats))


if __name__ == "__main__":
    main()
