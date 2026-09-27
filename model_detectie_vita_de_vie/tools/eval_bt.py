"""Scores bunch/trunk models on held-out sets only (Colab paths). Usage:
  python eval_bt.py prepare                     # builds the test folders once
  python eval_bt.py new  <weights.pt>           # the new 2-class model (0 = ciorchine, 1 = trunchi)
  python eval_bt.py old                         # current models: model_ciorchine_v2 (bunch), model_trunchi_v2 (trunk)

Tests (never trained on): label map test (robot view: bunch + trunk), WGISD test (bunch), Pinheiro test (bunch),
Vairão (trunk, new vineyard), EscaYard B9 (trunk found per photo, bunch count vs people's count).
"""
import csv
import glob
import os
import shutil
import sys
from pathlib import Path

import cv2

BT, TV = Path("/content/bt"), Path("/content/tv/data")
TD = BT / "tests"
DRIVE = Path("/content/drive/MyDrive/vie_hackathon")
sys.path.insert(0, "/content")
os.environ.setdefault("TRUNK_ROOT", "/content/tv")


def remap(src_lbl, dst_lbl, keep):
    lines = []
    if Path(src_lbl).exists():
        for line in Path(src_lbl).read_text().split("\n"):
            p = line.split()
            if len(p) >= 5 and int(p[0]) in keep:
                v = list(map(float, p[1:]))
                if len(v) > 4:
                    xs, ys = v[0::2], v[1::2]
                    v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
                lines.append(f"{keep[int(p[0])]} " + " ".join(f"{x:.6f}" for x in v))
    Path(dst_lbl).write_text("\n".join(lines))


def make(name, pairs, keep):
    d = TD / name
    for sub in ("images", "labels"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for i, (img, lbl) in enumerate(pairs):
        stem = f"{i:05d}_{Path(img).stem}"
        shutil.copy(img, d / "images" / f"{stem}{Path(img).suffix}")
        remap(lbl, d / "labels" / f"{stem}.txt", keep)
    (d / "data.yaml").write_text(f"path: {d}\ntrain: images\nval: images\nnames:\n  0: ciorchine\n  1: trunchi\n")
    print(name, len(pairs))


def prepare():
    shutil.rmtree(TD, ignore_errors=True)
    lm = TV / "labelmap" / "test"
    make("labelmap_robot", [(p, lm / "labels" / f"{p.stem}.txt") for p in sorted((lm / "images").glob("*.jpg"))], {0: 0, 1: 0, 2: 1, 3: 1})
    wg = Path("/content/wgisd")
    make("wgisd", [(wg / "data" / f"{i.strip()}.jpg", wg / "data" / f"{i.strip()}.txt") for i in open(wg / "test.txt") if i.strip()], {0: 0})
    pt = BT / "test_pinheiro"
    make("pinheiro", [(p, pt / "labels" / f"{p.stem}.txt") for p in sorted((pt / "images").glob("*.jpg"))], {0: 0})
    va = [Path(p) for p in glob.glob("/content/vairao/**/images/*.png", recursive=True)]
    make("vairao", [(p, p.parent.parent / "labels" / f"{p.stem}.txt") for p in sorted(va)], {0: 1})


TEST_IDS = {"ciorchine": 0, "trunchi": 1}  # class ids in the test labels


def ap50(weights, test, cls, imgsz):
    """AP50 of one class. Models number their classes differently (old trunk model: 0 = trunchi), so the test labels
    are rewritten into the model's own numbering, keeping only the boxes of that class."""
    from ultralytics import YOLO
    model = YOLO(weights)
    names = dict(model.names)
    idx = next(k for k, v in names.items() if v == cls)
    src, d = TD / test, TD / f"_{test}_{cls}_{idx}_{len(names)}"
    if not d.exists():
        (d / "labels").mkdir(parents=True)
        try:
            os.symlink(src / "images", d / "images")
        except OSError:  # Windows without symlink rights
            shutil.copytree(src / "images", d / "images")
        for lbl in (src / "labels").glob("*.txt"):
            keep = [f"{idx} " + " ".join(l.split()[1:]) for l in lbl.read_text().split("\n")
                    if l.split() and int(l.split()[0]) == TEST_IDS[cls]]
            (d / "labels" / lbl.name).write_text("\n".join(keep))
        (d / "data.yaml").write_text(f"path: {d}\ntrain: images\nval: images\nnames:\n"
                                     + "".join(f"  {k}: {v}\n" for k, v in sorted(names.items())))
    r = model.val(data=str(d / "data.yaml"), imgsz=imgsz, batch=8, verbose=False, plots=False)
    return float(r.box.ap50[list(r.box.ap_class_index).index(idx)]) if idx in list(r.box.ap_class_index) else 0.0


def escayard(bunch_w, trunk_w, bunch_cls, trunk_cls, imgsz_b=1280, imgsz_t=1280):
    import trunk_adapt as t
    from ultralytics import YOLO
    rows = {r["File"].strip(): r for r in csv.DictReader(open(t.ESCA / "datasheet.csv", encoding="utf-8-sig"))}
    b9 = t.escayard_blocks()["B9"]
    mb, mt = YOLO(bunch_w), YOLO(trunk_w)
    found, errs = 0, []
    for p in b9:
        img = t.shrink(cv2.imread(str(p)))
        rt = mt.predict(img, imgsz=imgsz_t, conf=0.3, verbose=False)[0]
        found += any(rt.names[int(k)] == trunk_cls for k in rt.boxes.cls.tolist())
        n_true = rows[p.name]["Number of grape clusters"].strip()
        if n_true:
            rb = mb.predict(img, imgsz=imgsz_b, conf=0.4, verbose=False)[0]
            n = sum(rb.names[int(k)] == bunch_cls for k in rb.boxes.cls.tolist())
            errs.append(abs(n - int(n_true)))
    return found, len(b9), (sum(errs) / len(errs) if errs else float("nan")), len(errs)


def report(bunch_w, trunk_w, bunch_cls="ciorchine", trunk_cls="trunchi", imgsz_b=960, imgsz_t=1280, imgsz_es=None):
    """imgsz_es: size for counting bunches on EscaYard (the old bunch model was measured at 1280 there)."""
    out = {
        "robot bunch AP50": ap50(bunch_w, "labelmap_robot", bunch_cls, imgsz_b),
        "robot trunk AP50": ap50(trunk_w, "labelmap_robot", trunk_cls, imgsz_t),
        "WGISD bunch AP50": ap50(bunch_w, "wgisd", bunch_cls, imgsz_b),
        "Pinheiro bunch AP50": ap50(bunch_w, "pinheiro", bunch_cls, imgsz_b),
        "Vairao trunk AP50": ap50(trunk_w, "vairao", trunk_cls, imgsz_t),
    }
    f, n, err, nc = escayard(bunch_w, trunk_w, bunch_cls, trunk_cls, imgsz_es or imgsz_b, imgsz_t)
    out["EscaYard B9 trunk found %"] = f / n if n else 0.0
    out[f"EscaYard B9 bunch count error ({nc} vines)"] = round(err, 2)
    for k, v in out.items():
        print(f"{k:40s} {v:.3f}" if isinstance(v, float) else f"{k:40s} {v}", flush=True)
    return out


def score(out):
    """One number to pick the best checkpoint: average of the AP50s and the share of B9 photos with a trunk."""
    return sum(v for k, v in out.items() if "AP50" in k or "found" in k) / 6


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare()
    elif cmd == "old":
        report(str(DRIVE / "model_ciorchine_v2.pt"), str(DRIVE / "model_trunchi_v2.pt"), imgsz_es=1280)
    elif cmd == "new":
        size = int(sys.argv[3]) if len(sys.argv) > 3 else 960
        report(sys.argv[2], sys.argv[2], imgsz_b=size, imgsz_t=size)
