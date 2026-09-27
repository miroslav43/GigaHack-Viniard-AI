"""Diseased-leaf model for daylight (model_frunze_bolnave_v3): fine-tuned from model_frunze_bolnave.pt with strong light
augmentation, because the model was trained on flash photos (Bordeaux FD dataset) and loses the symptoms in harsh sun.

Measured on 27 Sept (tools/tune_vine_disease.py): EscaYard B7 was photographed in the evening, B9 at midday. Diseased
leaves found per sick vine: 6.1 in B7, 1.9 in B9 -> B9 only 3/12 sick vines found. No rule or tiling fixed it.
Auto-labels on EscaYard are not usable (almost every photo has leaves at an unsure 0.25-0.5), so the only new
ingredient is the augmentation: brighter / higher-contrast / gamma / shadows (midday sun) + JPEG and noise (ESP32).

Data: the same FD train / valid / test split as the first leaf model (colab/fd_frunze_bolnave.ipynb, seed 0).
Epoch chosen on EscaYard B7 at vine level (today's rule: at least 3 leaves at conf >= 0.5); B9 is only the final test.

Steps (the same file runs in Colab: LEAF_ROOT = folder with data/ and models/, LEAF_RUNS = where runs go):
  python leaf_daylight.py train --data /content/fd-vie/data.yaml   # ~15-20 min on a T4
  python leaf_daylight.py select --run leaf_v3                     # every 2nd epoch on B7 vines -> best epoch
  python leaf_daylight.py report --weights <best.pt>               # old vs new: B9 vines + FD test
"""
import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(os.environ.get("LEAF_ROOT", Path(__file__).resolve().parent.parent))
RUNS = Path(os.environ.get("LEAF_RUNS", ROOT / "runs" / "leaf_v3"))
ESCA = ROOT / "data" / "escayard"
OLD = ROOT / "models" / "model_frunze_bolnave.pt"
IMGSZ, CONF, MIN_LEAVES = 960, 0.5, 3  # as in vineyard/config.py and the suspect-vine rule
MAX_AREA = 0.08  # vineyard/config.py MAX_BOX_AREA: a "leaf" bigger than 8% of the photo is soil or stones
MAX_SIDE = 2048  # EscaYard photos are shrunk like in tools/evaluate_escayard.py


def light_augmentations():
    """Midday sun + ESP32 camera. Each transform is skipped if this albumentations version does not know it."""
    try:
        import albumentations as A
    except ImportError:
        print("albumentations is not installed: training with the colour (HSV) augmentation only")
        return None
    wanted = [
        lambda: A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.4), contrast_limit=(-0.2, 0.4), p=0.6),
        lambda: A.RandomGamma(gamma_limit=(60, 140), p=0.4),
        lambda: A.RandomShadow(p=0.2),
        lambda: A.CLAHE(p=0.1),
        lambda: A.ImageCompression(quality_range=(40, 90), p=0.3),
        lambda: A.GaussNoise(p=0.2),
        lambda: A.MotionBlur(p=0.1),
    ]
    out = []
    for make in wanted:
        try:
            t = make()
            A.to_dict(t)  # Ultralytics saves the transforms this way: one that cannot be saved would stop the training
            out.append(t)
        except Exception as e:  # older / newer albumentations with other argument names
            print("skipping an augmentation:", e)
    return out


# ---------- vine level (EscaYard) ----------

def vine_rows():
    rows = []
    with open(ESCA / "datasheet.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            status = r["Esca"].strip().upper()
            if r["File"].strip() and status in ("YES", "NO"):
                rows.append((r["File"].strip(), r["Vineyard"].strip(), status == "YES"))
    return rows


def leaves_per_photo(model, block):
    """{photo: [conf of each diseased leaf >= 0.2]} for the vines of one EscaYard block."""
    import cv2
    photos = {p.name: p for p in (ESCA / "photos").rglob("*.jpg")}
    out = {}
    for name, b, _ in vine_rows():
        if b != block or name not in photos:
            continue
        img = cv2.imread(str(photos[name]))
        h, w = img.shape[:2]
        s = MAX_SIDE / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
        r = model.predict(img, imgsz=IMGSZ, conf=0.2, verbose=False)[0]
        out[name] = [c for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())
                     if (x2 - x1) * (y2 - y1) <= MAX_AREA * w * h]
    return out


def vine_scores(leaves, block, conf=CONF, min_leaves=MIN_LEAVES):
    """(sick found, sick, healthy flagged, healthy) with the rule "at least min_leaves leaves at conf >= conf"."""
    tp = n_sick = fp = n_healthy = 0
    for name, b, sick in vine_rows():
        if b != block or name not in leaves:
            continue
        flagged = sum(c >= conf for c in leaves[name]) >= min_leaves
        if sick:
            n_sick, tp = n_sick + 1, tp + flagged
        else:
            n_healthy, fp = n_healthy + 1, fp + flagged
    return tp, n_sick, fp, n_healthy


def youden(s):
    """sick found share - healthy flagged share (1 = perfect, 0 = guessing)."""
    tp, n_sick, fp, n_healthy = s
    return (tp / n_sick if n_sick else 0) - (fp / n_healthy if n_healthy else 0)


# ---------- commands ----------

def train(args):
    from ultralytics import YOLO

    t0 = time.time()

    def print_epoch(trainer):
        ep, total = trainer.epoch + 1, trainer.epochs
        if ep > total:
            return
        elapsed = time.time() - t0
        print(f"==> Epoch {ep}/{total} ({ep / total:.0%}) | FD validation mAP50: "
              f"{trainer.metrics.get('metrics/mAP50(B)', 0.0):.3f} | {elapsed / 60:.0f} min | "
              f"left: max ~{elapsed / ep * (total - ep) / 60:.0f} min", flush=True)

    m = YOLO(args.init)
    m.add_callback("on_fit_epoch_end", print_epoch)
    extra = {}
    aug = light_augmentations()
    if aug:
        extra["augmentations"] = aug
    # AdamW with an explicit learning rate: optimizer="auto" picks its own and ignores lr0 on short fine-tunes
    m.train(data=args.data, imgsz=IMGSZ, epochs=args.epochs, patience=args.epochs, batch=args.batch,
            optimizer="AdamW", lr0=args.lr0, save_period=2, workers=2, cache="ram", project=str(RUNS),
            name=args.name, exist_ok=True, plots=False, hsv_s=0.7, hsv_v=0.7, fraction=args.fraction,
            amp=not args.no_amp, **extra)


def select(args):
    from ultralytics import YOLO
    w = RUNS / args.run / "weights"
    cands = sorted(w.glob("epoch*.pt"), key=lambda p: int(p.stem[5:])) + [w / "best.pt", w / "last.pt"]
    cands = [c for c in cands if c.exists()] + [OLD]
    table = []
    for i, c in enumerate(cands, 1):
        s = vine_scores(leaves_per_photo(YOLO(str(c)), "B7"), "B7")
        table.append((youden(s), c, s))
        print(f"{i}/{len(cands)} ({i / len(cands):.0%}) {c.name:26s} B7: sick found {s[0]}/{s[1]}, "
              f"healthy flagged {s[2]}/{s[3]}", flush=True)
    j, best, s = max(t for t in table if t[1] != OLD)
    old = next(t for t in table if t[1] == OLD)
    print(f"\nBest: {best.name} (B7 sick found {s[0]}/{s[1]}, healthy flagged {s[2]}/{s[3]}) | "
          f"old model: {old[2][0]}/{old[2][1]}, {old[2][2]}/{old[2][3]}")
    shutil.copy(best, RUNS / "model_frunze_bolnave_v3.pt")
    (RUNS / "leaf_v3_choice.json").write_text(json.dumps({"epoch": best.name, "b7": s, "b7_old": old[2]}))
    print("Saved:", RUNS / "model_frunze_bolnave_v3.pt")


def report(args):
    from ultralytics import YOLO
    rows = []
    for name, weights in (("old", OLD), ("new (v3)", Path(args.weights))):
        m = YOLO(str(weights))
        r = {"model": name}
        if args.data:
            res = m.val(data=args.data, split="test", imgsz=IMGSZ, verbose=False, plots=False)
            r["FD test mAP50 (flash photos)"] = f"{res.box.map50:.3f}"
        leaves = leaves_per_photo(m, "B9")
        s = vine_scores(leaves, "B9")
        r["B9 sick vines found (>=3 leaves)"] = f"{s[0]}/{s[1]}"
        r["B9 healthy vines flagged"] = f"{s[2]}/{s[3]}"
        sick = [len([c for c in leaves[n] if c >= CONF]) for n, b, e in vine_rows() if b == "B9" and e and n in leaves]
        r["B9 leaves per sick vine"] = f"{sum(sick) / max(1, len(sick)):.1f}"
        rows.append(r)
        print(f"{name} done", flush=True)
    keys = [k for k in rows[0] if k != "model"]
    print(f"\n{'':36s}{rows[0]['model']:>12s}{rows[1]['model']:>12s}")
    for k in keys:
        print(f"{k:36s}{rows[0][k]:>12s}{rows[1][k]:>12s}")
    (RUNS / "leaf_v3_report.json").write_text(json.dumps(rows, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--name", default="leaf_v3")
    t.add_argument("--init", default=str(OLD))
    t.add_argument("--epochs", type=int, default=24)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr0", type=float, default=0.0005)
    t.add_argument("--fraction", type=float, default=1.0)
    t.add_argument("--no-amp", action="store_true")
    s = sub.add_parser("select")
    s.add_argument("--run", default="leaf_v3")
    r = sub.add_parser("report")
    r.add_argument("--weights", required=True)
    r.add_argument("--data", help="FD data.yaml, to also score the FD test photos")
    args = p.parse_args()
    {"train": train, "select": select, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
