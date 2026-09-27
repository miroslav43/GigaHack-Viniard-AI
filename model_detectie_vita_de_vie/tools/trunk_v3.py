"""Trunk model v3 for unseen vineyards: the data mix is rebalanced so the model stops learning "trunk = the stub at the
bottom edge of the photo" (label map, 65% of v2's training photos, 98% of its trunk boxes touch the bottom edge).

What changed from v2 (research: reports/Îmbunătățire modele ciorchini și trunchi.md, measured 27 Sept):
  - label map cut to at most 30% of the training photos
  - Vairão (Portugal, whole trunks seen from low, autumn) added: 6 recordings train, 1 validation, 2 test.
    v1+v2 find a trunk in only 74/460 Vairão photos, so this is the vineyard the current models do not know.
  - EscaYard B7 auto-labels are stricter: a box is kept only if v2 is sure (conf >= 0.4) and finds it again on the
    mirrored photo; a photo with an unsure box (0.25-0.4) is left out, because a missing box teaches "trunk = background"
  - posts (class 1 "stalp") auto-labeled by YOLOE with the prompt "concrete post" (tested: finds real posts,
    confidence 0.15-0.35), so posts become explicit negatives for the trunk class
  - checkpoint chosen on photos from the target vineyards (20 hand-marked B7 photos + Vairão recording 170105), not on
    label map; every epoch is saved and compared
Not used: the Spanish pruning set (Voxel51/vineyard-pruning): its "trunk" outline also covers the horizontal arms
(median box width 0.80 of the photo), a third trunk convention.

Final tests (never trained on): Vairão recordings 170318 + 170442, label map test (robot), EscaYard B9 (photos with a
trunk), mildew odd photos (foliage without trunks: false alarms).

Steps (the same file runs in Colab: TRUNK_ROOT = folder with data/ and models/, TRUNK_RUNS = where runs go):
  python -m tools.trunk_v3 build             # ~20-30 min on a T4 (auto-labels B7 + posts)
  python -m tools.trunk_v3 train --name v3   # fine-tune from models/model_trunchi_v2.pt, every epoch saved
  python -m tools.trunk_v3 select --run v3   # every epoch on the validation photos -> best epoch + threshold
  python -m tools.trunk_v3 report --weights <best.pt> --conf 0.3   # v2 vs v3 on the final tests
"""
import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

ROOT = Path(os.environ.get("TRUNK_ROOT", Path(__file__).resolve().parent.parent))
RUNS = Path(os.environ.get("TRUNK_RUNS", ROOT / "runs" / "trunk_v3"))
DS = ROOT / "data" / "trunk_v3"
LM, HUM, VAI = ROOT / "data" / "labelmap", ROOT / "data" / "humain" / "git", ROOT / "data" / "vairao"
ESCA, MILDEW = ROOT / "data" / "escayard", ROOT / "data" / "mildew" / "downy_mildew_images"
MANUAL = ROOT / "data" / "manual_trunks_b7.json"
V2 = ROOT / "models" / "model_trunchi_v2.pt"

MAX_SIDE = 1280  # every photo is stored at most this big: phone photos are 4000 px and would slow training down
LM_SHARE = 0.30  # label map photos: at most this share of the training photos
VAI_VALID, VAI_TEST = {"170105"}, {"170318", "170442"}  # Vairão recordings kept out of training
POST_PROMPTS = ["wooden post", "metal pole", "concrete post", "vine trunk", ""]  # "" = background, absorbs doubtful boxes
POST_CONF = 0.15
# Auto-label thresholds, chosen on the 20 hand-marked B7 photos (27 Sept): v2 is rarely very sure on EscaYard
# (median box conf 0.43). sure 0.5 kept 17/113 B7 photos; sure 0.4 + unsure 0.25 keeps 50/113, and on the hand-marked
# photos 7 of its 8 auto-boxes were real trunks.
SURE, UNSURE, MIRROR = 0.4, 0.25, 0.25
TEST_SETS = ("test_vairao", "test_lm", "test_foliage")


# ---------- geometry (boxes are (x1, y1, x2, y2), normalized 0-1 unless said otherwise) ----------

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def read_labels(path, classes):
    """YOLO boxes ("c x y w h") or outlines ("c x1 y1 x2 y2 ...") -> normalized xyxy boxes of the wanted classes."""
    boxes = []
    if path.exists():
        for line in path.read_text().splitlines():
            v = line.split()
            if len(v) < 5 or int(v[0]) not in classes:
                continue
            n = [float(x) for x in v[1:]]
            if len(n) == 4:
                boxes.append((n[0] - n[2] / 2, n[1] - n[3] / 2, n[0] + n[2] / 2, n[1] + n[3] / 2))
            else:
                boxes.append((min(n[0::2]), min(n[1::2]), max(n[0::2]), max(n[1::2])))
    return boxes


def yolo_line(c, b):
    return f"{c} {(b[0] + b[2]) / 2:.6f} {(b[1] + b[3]) / 2:.6f} {b[2] - b[0]:.6f} {b[3] - b[1]:.6f}"


# ---------- which photo goes where ----------

def recording(path):
    """Vairão frame '163311_frame_0004.png' -> recording '163311'."""
    return Path(path).stem.split("_frame")[0]


def vairao_split(path):
    r = recording(path)
    return "valid" if r in VAI_VALID else "test_vairao" if r in VAI_TEST else "train"


def lm_quota(n_other, share=LM_SHARE):
    """How many label map photos keep label map at `share` of the training photos."""
    return round(share / (1 - share) * n_other)


def escayard_blocks():
    blocks = {"B7": [], "B9": []}
    with open(ESCA / "datasheet.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["Vineyard"].strip() in blocks and r["File"].strip():
                blocks[r["Vineyard"].strip()].append(r["File"].strip())
    photos = {p.name: p for p in (ESCA / "photos").rglob("*.jpg")}
    return {b: [photos[n] for n in sorted(names) if n in photos] for b, names in blocks.items()}


def keep_pseudo(found, found_mirror):
    """Auto-labels for one photo, or None when the photo must be left out.
    found / found_mirror: [(box, conf)] from the photo and from its mirror image (boxes already mirrored back).
    Kept: boxes with conf >= SURE that are found again on the mirror (IoU >= 0.6, conf >= MIRROR) and at least 8% tall.
    Left out: photos with any unsure box (between UNSURE and SURE, or sure but not confirmed), or with no trunk."""
    kept = []
    for box, conf in found:
        if conf < UNSURE:
            continue
        confirmed = any(iou(box, m) >= 0.6 and c >= MIRROR for m, c in found_mirror)
        if conf < SURE or not confirmed:
            return None
        if box[3] - box[1] >= 0.08:
            kept.append(box)
    return kept or None


def posts_for(candidates, trunks):
    """YOLOE "post" boxes [(box, conf, prompt)] -> post boxes that are tall and do not overlap a trunk."""
    out = []
    for box, conf, prompt in candidates:
        tall = (box[3] - box[1]) >= 4 * (box[2] - box[0])
        if prompt in ("wooden post", "metal pole", "concrete post") and conf >= POST_CONF and tall and \
                all(iou(box, t) < 0.1 and inside(box, t) < 0.3 for t in trunks):
            out.append(box)
    return out


# ---------- models (imported only when needed, so the maths above runs without ultralytics) ----------

def load_image(path):
    import cv2
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def predict(model, img, imgsz, conf, cls=0):
    """[(normalized box, conf)] of one class."""
    h, w = img.shape[:2]
    r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False)[0]
    return [((b[0] / w, b[1] / h, b[2] / w, b[3] / h), c)
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()) if int(k) == cls]


class PostLabeler:
    def __init__(self):
        from ultralytics import YOLOE
        local = ROOT / "models" / "yoloe-11l-seg.pt"
        self.m = YOLOE(str(local) if local.exists() else "yoloe-11l-seg.pt")  # Ultralytics downloads it if missing
        self.m.set_classes(POST_PROMPTS, self.m.get_text_pe(POST_PROMPTS))

    def __call__(self, img, trunks):
        h, w = img.shape[:2]
        r = self.m.predict(img, imgsz=1024, conf=POST_CONF, verbose=False)[0]
        cands = [((b[0] / w, b[1] / h, b[2] / w, b[3] / h), c, POST_PROMPTS[int(k)])
                 for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        return posts_for(cands, trunks)


# ---------- build ----------

def write(split, name, img, trunks, posts=()):
    import cv2
    for sub in ("images", "labels"):
        (DS / split / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DS / split / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    lines = [yolo_line(0, b) for b in trunks] + [yolo_line(1, b) for b in posts]
    (DS / split / "labels" / f"{name}.txt").write_text("\n".join(lines))


def build(args):
    import cv2
    from ultralytics import YOLO

    if DS.exists():
        shutil.rmtree(DS)
    manual = json.loads(MANUAL.read_text())
    items = []  # (split, name, photo path, trunk boxes or None = auto-label with v2)
    lm_train = sorted((LM / "train" / "images").glob("*.jpg"))
    items += [("test_lm", f"lm_{p.stem}", p, read_labels(LM / "test" / "labels" / f"{p.stem}.txt", {2, 3}))
              for p in sorted((LM / "test" / "images").glob("*.jpg"))]
    for folder in (HUM / "train" / "Original_photos", HUM / "valid"):
        items += [("train", f"hu_{p.stem}", p, read_labels(folder / f"{p.stem}.txt", {0})) for p in sorted(folder.glob("*.jpg"))]
    for p in sorted(VAI.rglob("*.png")):
        items.append((vairao_split(p), f"va_{p.stem}", p, read_labels(p.parent.parent / "labels" / f"{p.stem}.txt", {0})))
    b7 = escayard_blocks()["B7"]
    items += [("valid", f"esm_{p.stem}", p, [tuple(b) for b in manual[p.name]]) for p in b7 if p.name in manual]
    items += [("train", f"es_{p.stem}", p, None) for p in b7 if p.name not in manual]
    mildew = sorted(MILDEW.glob("*.jpg"))
    items += [("train", f"neg_{p.stem}", p, []) for p in mildew[0::2]]  # foliage without trunks: negatives
    items += [("test_foliage", f"neg_{p.stem}", p, []) for p in mildew[1::2]]

    n_other = sum(s == "train" for s, *_ in items)
    quota = lm_quota(n_other)
    random.Random(0).shuffle(lm_train)
    items += [("train", f"lm_{p.stem}", p, read_labels(LM / "train" / "labels" / f"{p.stem}.txt", {2, 3}))
              for p in lm_train[:quota]]
    print(f"label map: {min(quota, len(lm_train))} of {len(lm_train)} training photos kept "
          f"({LM_SHARE:.0%} of the training photos)", flush=True)

    teacher = YOLO(str(V2))
    posts = PostLabeler()
    stats = {"posts": 0, "pseudo_kept": 0, "pseudo_left_out": 0}
    t0 = time.time()
    for i, (split, name, path, trunks) in enumerate(items, 1):
        img = load_image(path)
        if trunks is None:  # EscaYard B7 without hand marks: auto-label with v2 + mirror check
            found = predict(teacher, img, 1280, UNSURE)
            mirror = [((1 - b[2], b[1], 1 - b[0], b[3]), c) for b, c in predict(teacher, cv2.flip(img, 1), 1280, MIRROR)]
            trunks = keep_pseudo(found, mirror)
            if trunks is None:
                stats["pseudo_left_out"] += 1
                continue
            stats["pseudo_kept"] += 1
        found_posts = posts(img, trunks) if split in ("train", "valid") else []
        stats["posts"] += len(found_posts)
        write(split, name, img, trunks, found_posts)
        if i % 100 == 0 or i == len(items):
            left = (time.time() - t0) / i * (len(items) - i)
            print(f"{i}/{len(items)} photos ({i / len(items):.0%}) | about {left / 60:.0f} min left", flush=True)

    (DS / "data.yaml").write_text(f"path: {DS.as_posix()}\ntrain: train/images\nval: valid/images\n"
                                  "names:\n  0: trunchi\n  1: stalp\n", encoding="utf-8")
    counts = {}
    for f in (DS / "train" / "images").glob("*.jpg"):
        counts[f.stem.split("_")[0]] = counts.get(f.stem.split("_")[0], 0) + 1
    print("Training photos per source:", counts, "| label map share:",
          f"{counts.get('lm', 0) / max(1, sum(counts.values())):.0%}")
    print({s: len(list((DS / s / "images").glob("*.jpg"))) for s in ("train", "valid") + TEST_SETS})
    print(f"EscaYard B7 auto-labels: {stats['pseudo_kept']} photos kept, {stats['pseudo_left_out']} left out (unsure) | "
          f"posts auto-labeled: {stats['posts']}")


# ---------- train ----------

def train(args):
    from ultralytics import YOLO

    t0 = time.time()

    def print_epoch(trainer):
        ep, total = trainer.epoch + 1, trainer.epochs
        if ep > total:  # the final validation after training calls this once more
            return
        elapsed = time.time() - t0
        best = getattr(getattr(trainer, "stopper", None), "best_epoch", ep)  # already counts from 1
        print(f"==> Epoch {ep}/{total} ({ep / total:.0%}) | validation mAP50: "
              f"{trainer.metrics.get('metrics/mAP50(B)', 0.0):.3f} | best so far: epoch {best} | "
              f"{elapsed / 60:.0f} min | left: max ~{elapsed / ep * (total - ep) / 60:.0f} min", flush=True)

    m = YOLO(args.init)
    m.add_callback("on_fit_epoch_end", print_epoch)
    # AdamW with an explicit learning rate: optimizer="auto" picks its own and ignores lr0 on short fine-tunes
    m.train(data=str(DS / "data.yaml"), imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
            optimizer="AdamW", lr0=args.lr0, save_period=1, workers=2, project=str(RUNS), name=args.name,
            exist_ok=True, plots=False, hsv_h=0.03, hsv_s=0.8, hsv_v=0.6, degrees=5, scale=0.6, fraction=args.fraction,
            amp=not args.no_amp)


# ---------- evaluation ----------

def score_photos(model, folder, imgsz, conf):
    """[(found boxes with conf, real trunk boxes)] for every photo of a dataset folder."""
    out = []
    for p in sorted((folder / "images").glob("*.jpg")):
        import cv2
        img = cv2.imread(str(p))
        out.append((predict(model, img, imgsz, conf), read_labels(folder / "labels" / f"{p.stem}.txt", {0})))
    return out


def prf(per_photo, conf, min_iou):
    """Precision, recall, F1 at a threshold: each real trunk matched at most once, most confident box first."""
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
    p = right / found if found else 0.0
    r = right / real if real else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), right, found, real


def best_threshold(per_photo, min_iou=0.3, thresholds=(0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6)):
    """(F1, threshold) with the highest F1."""
    return max((prf(per_photo, t, min_iou)[2], t) for t in thresholds)


def select(args):
    from ultralytics import YOLO
    w = RUNS / args.run / "weights"
    cands = sorted(w.glob("epoch*.pt"), key=lambda p: int(p.stem[5:])) + [w / "best.pt", w / "last.pt"]
    cands = [c for c in cands if c.exists()] + [V2]
    table = []
    for i, c in enumerate(cands, 1):
        f1, t = best_threshold(score_photos(YOLO(str(c)), DS / "valid", args.imgsz, 0.2))
        table.append((f1, t, c))
        print(f"{i}/{len(cands)} ({i / len(cands):.0%}) {c.name:14s} F1 {f1:.3f} at conf {t}", flush=True)
    f1, t, best = max(r for r in table if r[2] != V2)
    v2 = next(r for r in table if r[2] == V2)
    print(f"\nBest: {best.name} (F1 {f1:.3f} at conf {t}) | v2 on the same photos: F1 {v2[0]:.3f} at conf {v2[1]}")
    shutil.copy(best, RUNS / "model_trunchi_v3.pt")
    (RUNS / "v3_choice.json").write_text(json.dumps({"epoch": best.name, "conf": t, "f1": f1, "v2_f1": v2[0]}))
    print("Saved:", RUNS / "model_trunchi_v3.pt")


def b9_photos_with_trunk(model, imgsz, conf):
    b9 = escayard_blocks()["B9"]
    return sum(bool(predict(model, load_image(p), imgsz, conf)) for p in b9), len(b9)


def report(args):
    from ultralytics import YOLO
    rows = []
    for name, weights, conf in (("v2 (now)", V2, 0.3), ("v3", Path(args.weights), args.conf)):
        m = YOLO(str(weights))
        r = {"model": name, "conf": conf}
        for s in ("test_vairao", "test_lm"):
            per = score_photos(m, DS / s, args.imgsz, conf)
            for min_iou in (0.3, 0.5):
                p, rec, f1, right, found, real = prf(per, conf, min_iou)
                r[f"{s} IoU{min_iou} P/R/F1"] = f"{p:.2f}/{rec:.2f}/{f1:.2f}"
            r[f"{s} photos with trunk found"] = f"{sum(bool(d) for d, g in per if g)}/{sum(bool(g) for _, g in per)}"
        fol = score_photos(m, DS / "test_foliage", args.imgsz, conf)
        r["foliage false alarms (photos)"] = f"{sum(bool(d) for d, _ in fol)}/{len(fol)}"
        k, n = b9_photos_with_trunk(m, args.imgsz, conf)
        r["EscaYard B9 photos with trunk"] = f"{k}/{n}"
        rows.append(r)
        print(f"{name} done", flush=True)
    keys = [k for k in rows[0] if k != "model"]
    print(f"\n{'':40s}{rows[0]['model']:>18s}{rows[1]['model']:>18s}")
    for k in keys:
        print(f"{k:40s}{str(rows[0][k]):>18s}{str(rows[1][k]):>18s}")
    (RUNS / "v3_report.json").write_text(json.dumps(rows, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    t = sub.add_parser("train")
    t.add_argument("--name", default="v3")
    t.add_argument("--init", default=str(V2))
    t.add_argument("--imgsz", type=int, default=960)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr0", type=float, default=0.0005)
    t.add_argument("--fraction", type=float, default=1.0, help="share of the training photos (e.g. 0.05 for a quick check)")
    t.add_argument("--no-amp", action="store_true", help="no mixed precision (GTX 16xx cards misbehave with it)")
    s = sub.add_parser("select")
    s.add_argument("--run", default="v3")
    s.add_argument("--imgsz", type=int, default=1280)
    r = sub.add_parser("report")
    r.add_argument("--weights", required=True)
    r.add_argument("--conf", type=float, default=0.3)
    r.add_argument("--imgsz", type=int, default=1280)
    args = p.parse_args()
    {"build": build, "train": train, "select": select, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
