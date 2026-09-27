"""Generates colab/retrain_negatives.ipynb: fine-tunes the diseased-leaf and bunch models again,
this time with "negative" images (nothing to find), and compares them with the old models."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "retrain_negatives.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Reantrenare cu exemple negative: frunze bolnave + ciorchini (v2)

**Problema:** modelul de frunze bolnave vede „frunze bolnave” în iarbă (vie sănătoasă, tura 1), iar modelul de ciorchini
vede „ciorchini” în frunzele roșii de boală (tura 2). Niciun model n-a văzut la antrenare pozele celuilalt.

**Soluția:** le arătăm poze **fără nimic de găsit** (exemple negative):
- modelul de **frunze**: + ~450 poze de vie sănătoasă (label map de pe robot, cu iarbă, și WGISD);
- modelul de **ciorchini**: + ~180 decupaje de frunze bolnave din dataset-ul Bordeaux (fără ciorchini).

Pornim de la modelele deja antrenate (nu de la zero), deci durează mai puțin: **~1 oră** în total pe T4.

## Ce faceți voi (totul e la început, apoi puteți pleca de la calculator)
1. **Runtime → Change runtime type → T4 GPU → Save.**
2. **Runtime → Run all.**
3. Celula 3: „Connect to Google Drive” și permiteți accesul.
4. Celula 4: dacă cele 2 modele vechi nu sunt în Drive, apare un buton „Choose files”: alegeți de pe laptop
   `models/model_frunze_bolnave.pt` și `models/model_ciorchine.pt`.
5. Celula 5: lipiți cheia API Roboflow (nu se afișează și nu se salvează).

Rezultat: `MyDrive/vie_hackathon/model_frunze_bolnave_v2.pt` și `model_ciorchine_v2.pt` + un tabel „vechi vs nou” la final.
Modelele vechi **nu** se suprascriu.
"""),
    md("## 1. Verificare placă video (trebuie să apară Tesla T4)"),
    code("!nvidia-smi"),
    md("## 2. Instalare (aceeași versiune ca pe laptop)"),
    code("""
!pip install -q ultralytics==8.4.163 roboflow
import ultralytics
print("ultralytics", ultralytics.__version__)
"""),
    md("## 3. Google Drive (aici se salvează modelele)"),
    code("""
import os
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/vie_hackathon'
os.makedirs(OUT, exist_ok=True)
print("Results are saved in", OUT)
"""),
    md("## 4. Modelele vechi (din Drive; dacă lipsesc, le încărcați de pe laptop)"),
    code("""
from google.colab import files

OLD_LEAVES, OLD_BUNCHES = f"{OUT}/model_frunze_bolnave.pt", f"{OUT}/model_ciorchine.pt"
missing = [p for p in (OLD_LEAVES, OLD_BUNCHES) if not os.path.exists(p)]
while missing:
    print("Missing in Drive:", ", ".join(os.path.basename(p) for p in missing),
          "-> choose them from the laptop (folder models/)")
    for name, data in files.upload().items():
        for p in missing:
            if os.path.basename(p) in name:
                open(p, "wb").write(data)
    missing = [p for p in (OLD_LEAVES, OLD_BUNCHES) if not os.path.exists(p)]
print("Old models found:", OLD_LEAVES, OLD_BUNCHES)
"""),
    md("## 5. Descărcare label map v8 (Roboflow) — lipiți cheia API"),
    code("""
from getpass import getpass
from roboflow import Roboflow

rf = Roboflow(api_key=getpass("Roboflow API key: "))
rf.workspace("grape-detection").project("label-map").version(8).download("yolov8", location="/content/label-map")
"""),
    md("## 6. Descărcare WGISD (~1,5 GB, cam un minut)"),
    code("""
import os
if not os.path.exists("/content/wgisd/train.txt"):
    !git clone --depth 1 https://github.com/thsant/wgisd.git /content/wgisd
else:
    print("WGISD already downloaded")
"""),
    md("## 7. Setări (căi și câte exemple negative)"),
    code("""
from pathlib import Path

ROOT = Path("/content")
FD_ALL = ROOT / "fd_all"          # all Flavescence doree photos + YOLO labels + metadata
LABEL_MAP = ROOT / "label-map"    # Roboflow label map v8 (robot photos, healthy vineyard, grass)
WGISD = ROOT / "wgisd"
LEAVES = ROOT / "leaves-v2"       # dataset for the diseased-leaf model
BUNCHES = ROOT / "bunches-v2"     # dataset for the bunch model
BUNCH_NEG_TEST = ROOT / "bunch-negatives-test"  # leaf crops from the FD test split, only for measuring

# healthy-vineyard photos added to the diseased-leaf model (nothing to find in them)
N_BG_LEAVES = {"labelmap_train": 350, "labelmap_valid": 60, "wgisd_train": 100}
# crops of diseased leaves added to the bunch model (no bunches in them)
N_CROPS = {"train": 150, "valid": 30, "test": 60}
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
"""),
    md("""
## 8. Descărcare Flavescence dorée + conversie (~1 GB, câteva minute)

La fel ca la prima antrenare. În plus, păstrăm pentru fiecare poză și unde sunt ciorchinii bolnavi, ca să nu-i prindem în decupaje.
Dacă se întrerupe, rulați celula din nou: sare peste ce e deja descărcat.
"""),
    code("""
import base64, collections, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://data.mendeley.com/public-api/datasets/3dr9r3w3jn"
FOLDERS = {
    "complete_CS20": "cc48eedb-c553-40cc-8d10-409cd5115763",
    "complete_UB20": "d805e05d-e696-4fc5-9256-6eef0f22b529",
    "soft_CF21": "9883150a-0a0a-4e18-a37d-d9dd1b33cd5a",
    "soft_CS21": "82f6b951-8daa-4fd8-85ef-0cc9e24fdc6d",
    "soft_M21": "fbdd115d-f7ed-4494-a14e-71d58d246165",
    "soft_SB21": "8c069f41-c13b-464a-a1c4-c1b0a962fcaa",
    "soft_UB20": "74f8f93f-762d-4f23-b28b-b0c24a9bd66d",
    "soft_UB21": "90153e4b-e3f1-43a4-99b6-0119eff52a0f",
}
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
LEAF = {"FD leaf", "ESCA leaf", "confounding leaf", "facteur confondant +"}
BUNCH = {"symptomatic bunch"}

for sub in ("images", "labels", "meta"):
    (FD_ALL / sub).mkdir(parents=True, exist_ok=True)

def get(url, as_json=True):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            return json.loads(data) if as_json else data
        except Exception as e:
            err = e
    raise err

jobs, jpg_urls = [], {}
for folder, fid in FOLDERS.items():
    for f in get(f"{API}/files?folder_id={fid}&version=2"):
        url = f["content_details"]["download_url"]
        if f["filename"].endswith(".json"):
            stem = re.sub(r"\\s*\\((\\d+)\\)", r"_dup\\1", Path(f["filename"]).stem)  # "im_00031 (2)" -> "im_00031_dup2"
            stem = re.sub(r"[^A-Za-z0-9_]", "", f"{folder}_{stem}")
            jobs.append((folder, stem, url))
        else:
            jpg_urls[(folder, f["filename"])] = url
print(len(jobs), "annotation files")

def clip_box(points, W, H):
    xs = [min(max(p[0], 0), W) for p in points]
    ys = [min(max(p[1], 0), H) for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def convert(job):
    folder, stem, url = job
    img_path, lbl_path, meta_path = (FD_ALL / "images" / f"{stem}.jpg", FD_ALL / "labels" / f"{stem}.txt",
                                     FD_ALL / "meta" / f"{stem}.json")
    if img_path.exists() and lbl_path.exists() and meta_path.exists():
        return 0
    a = get(url)
    W, H = a["imageWidth"], a["imageHeight"]
    lines, leaves, bunches = [], [], []
    for s in a["shapes"]:
        if s["label"] in BUNCH:  # rectangle or line: we only need where it is
            bunches.append(clip_box(s["points"], W, H))
        if s["shape_type"] == "rectangle" and s["label"] in LEAF:
            x1, y1, x2, y2 = clip_box(s["points"], W, H)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            leaves.append((x1, y1, x2, y2))
    img = base64.b64decode(a["imageData"]) if a.get("imageData") else get(jpg_urls[(folder, a["imagePath"])], as_json=False)
    img_path.write_bytes(img)
    lbl_path.write_text("\\n".join(lines))
    meta_path.write_text(json.dumps({"W": W, "H": H, "leaves": leaves, "bunches": bunches}))
    return len(leaves)

total = 0
with ThreadPoolExecutor(16) as ex:
    for i, n in enumerate(ex.map(convert, jobs), 1):
        total += n
        if i % 25 == 0 or i == len(jobs):
            print(f"Downloaded {i}/{len(jobs)} ({i / len(jobs):.0%})", flush=True)
print("Photos:", len(list((FD_ALL / "images").glob("*.jpg"))), "| new diseased-leaf boxes:", total)
"""),
    md("""
## 9. Împărțire FD train / valid / test — **identică** cu prima antrenare

Aceeași regulă și aceeași „sămânță” aleatoare (0), deci pozele de test sunt aceleași ca data trecută și putem compara corect vechi vs nou.
"""),
    code("""
import random

groups = collections.defaultdict(list)
for img in sorted((FD_ALL / "images").glob("*.jpg")):
    groups[re.sub(r"_dup\\d+$", "", img.stem)].append(img)

keys = sorted(groups)
random.seed(0)
random.shuffle(keys)
n = len(keys)
group_split = {k: ("train" if i < 0.70 * n else "valid" if i < 0.85 * n else "test") for i, k in enumerate(keys)}
FD_SPLIT = {img.stem: group_split[k] for k, imgs in groups.items() for img in imgs}
print(collections.Counter(FD_SPLIT.values()))
"""),
    md("""
## 10. Dataset pentru frunze bolnave: FD + poze de vie sănătoasă (negative)

Pozele negative intră doar în train și valid. Setul de test rămâne exact cel FD de data trecută.
"""),
    code("""
import shutil, yaml

def images_in(folder):
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMAGE_EXT)

def add(root, split, img, label_text, name=None):
    name = name or img.stem
    for sub in ("images", "labels"):
        (root / split / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(img, root / split / "images" / f"{name}{img.suffix.lower()}")
    (root / split / "labels" / f"{name}.txt").write_text(label_text)

def sample(items, k, seed):
    items = list(items)
    return random.Random(seed).sample(items, min(k, len(items)))

if LEAVES.exists():
    shutil.rmtree(LEAVES)
for stem, split in FD_SPLIT.items():
    add(LEAVES, split, FD_ALL / "images" / f"{stem}.jpg", (FD_ALL / "labels" / f"{stem}.txt").read_text())

wgisd_train_ids = [l.strip() for l in open(WGISD / "train.txt") if l.strip()]
backgrounds = {
    "train": [("lm", p) for p in sample(images_in(LABEL_MAP / "train" / "images"), N_BG_LEAVES["labelmap_train"], 1)]
             + [("wg", WGISD / "data" / f"{i}.jpg") for i in sample(wgisd_train_ids, N_BG_LEAVES["wgisd_train"], 2)],
    "valid": [("lm", p) for p in sample(images_in(LABEL_MAP / "valid" / "images"), N_BG_LEAVES["labelmap_valid"], 3)],
}
for split, items in backgrounds.items():
    for src, img in items:
        add(LEAVES, split, img, "", name=f"bg_{src}_{img.stem}")

yaml.safe_dump({"path": str(LEAVES), "train": "train/images", "val": "valid/images", "test": "test/images",
                "names": {0: "frunza_bolnava"}}, open(LEAVES / "data.yaml", "w"))
for split in ("train", "valid", "test"):
    imgs = images_in(LEAVES / split / "images")
    print(f"{split}: {len(imgs)} photos, of which {sum(p.name.startswith('bg_') for p in imgs)} negative")
"""),
    md("""
## 11. Dataset pentru ciorchini: WGISD + decupaje de frunze bolnave (negative)

Decupăm în jurul frunzelor bolnave din FD (câte cel mult 2 decupaje pe poză) și sărim peste orice decupaj care atinge un ciorchine
marcat de experți. Decupajele din FD-train merg la antrenare, cele din FD-valid la validare, iar cele din FD-test doar la măsurat.
"""),
    code("""
from PIL import Image

def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

def leaf_crops(split, n, seed, out_dir, prefix):
    rng = random.Random(seed)
    stems = [s for s, sp in FD_SPLIT.items() if sp == split and "_dup" not in s]
    candidates = []
    for s in stems:
        meta = json.loads((FD_ALL / "meta" / f"{s}.json").read_text())
        candidates += [(s, meta, box) for box in meta["leaves"]]
    rng.shuffle(candidates)
    out_dir.mkdir(parents=True, exist_ok=True)
    made, per_photo = [], collections.Counter()
    for s, meta, (x1, y1, x2, y2) in candidates:
        if len(made) >= n:
            break
        if per_photo[s] >= 2:
            continue
        W, H = meta["W"], meta["H"]
        side = min(max(max(x2 - x1, y2 - y1) * rng.uniform(1.3, 2.0), 256), W, H)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x0, y0 = min(max(cx - side / 2, 0), W - side), min(max(cy - side / 2, 0), H - side)
        window = (x0, y0, x0 + side, y0 + side)
        if any(overlaps(window, b) for b in meta["bunches"]):
            continue
        path = out_dir / f"{prefix}_{s}_{per_photo[s]}.jpg"
        Image.open(FD_ALL / "images" / f"{s}.jpg").convert("RGB").crop(tuple(round(v) for v in window)).save(path, quality=92)
        made.append(path)
        per_photo[s] += 1
    return made

if BUNCHES.exists():
    shutil.rmtree(BUNCHES)
random.seed(0)
random.shuffle(wgisd_train_ids)  # same WGISD split as the first training
n_val = round(len(wgisd_train_ids) * 0.15)
wgisd_test_ids = [l.strip() for l in open(WGISD / "test.txt") if l.strip()]
for split, ids in {"train": wgisd_train_ids[n_val:], "valid": wgisd_train_ids[:n_val], "test": wgisd_test_ids}.items():
    for i in ids:
        lines = [l for l in (WGISD / "data" / f"{i}.txt").read_text().splitlines() if l.strip()]
        add(BUNCHES, split, WGISD / "data" / f"{i}.jpg", "\\n".join(lines))

tmp_crops = ROOT / "crops_tmp"
for split, seed in (("train", 11), ("valid", 12)):
    for crop in leaf_crops(split, N_CROPS[split], seed, tmp_crops / split, "bg_leafcrop"):
        add(BUNCHES, split, crop, "")
if BUNCH_NEG_TEST.exists():
    shutil.rmtree(BUNCH_NEG_TEST)
test_crops = leaf_crops("test", N_CROPS["test"], 13, BUNCH_NEG_TEST, "leafcrop")

yaml.safe_dump({"path": str(BUNCHES), "train": "train/images", "val": "valid/images", "test": "test/images",
                "names": {0: "ciorchine"}}, open(BUNCHES / "data.yaml", "w"))
for split in ("train", "valid", "test"):
    imgs = images_in(BUNCHES / split / "images")
    print(f"{split}: {len(imgs)} photos, of which {sum(p.name.startswith('bg_') for p in imgs)} negative")
print("Leaf crops kept aside for measuring:", len(test_crops))
"""),
    md("""
## 12. Antrenare model frunze bolnave v2 (~35–45 min)

Pornește de la `model_frunze_bolnave.pt` (cel vechi), la 960 px. După fiecare epocă apare o linie cu procentul și timpul rămas.
"""),
    code('''
import time
from ultralytics import YOLO

T0 = time.time()

def print_epoch(trainer):
    """After every epoch: one clear line with progress, score and time left."""
    ep, total = trainer.epoch + 1, trainer.epochs
    if ep > total:  # the final validation after training calls this once more
        return
    elapsed = time.time() - (getattr(trainer, "train_time_start", None) or T0)
    left = elapsed / ep * (total - ep)
    map50 = trainer.metrics.get("metrics/mAP50(B)", 0.0)
    best = getattr(getattr(trainer, "stopper", None), "best_epoch", ep)  # Ultralytics already counts from 1
    print(f"==> Epoch {ep}/{total} ({ep / total:.0%}) | validation mAP50: {map50:.3f} | best: epoch {best} | "
          f"elapsed: {elapsed / 60:.1f} min | left: max ~{left / 60:.0f} min", flush=True)

leaf_model = YOLO(OLD_LEAVES)
leaf_model.add_callback("on_fit_epoch_end", print_epoch)
leaf_model.train(data=str(LEAVES / "data.yaml"), imgsz=960, epochs=40, patience=10, batch=8,
                 cache="ram", project=f"{OUT}/runs", name="model_frunze_bolnave_v2", exist_ok=True)
'''),
    md("## 13. Antrenare model ciorchini v2 (~10–15 min)"),
    code("""
T0 = time.time()
bunch_model = YOLO(OLD_BUNCHES)
bunch_model.add_callback("on_fit_epoch_end", print_epoch)
bunch_model.train(data=str(BUNCHES / "data.yaml"), imgsz=640, epochs=60, patience=15, batch=16,
                  cache="ram", project=f"{OUT}/runs", name="model_ciorchine_v2", exist_ok=True)
"""),
    md("""
### Dacă sesiunea s-a închis în timpul antrenării
Rulați din nou celulele 1–11, apoi decomentați (scoateți `#`) linia potrivită de mai jos: antrenarea continuă de unde a rămas.
"""),
    code("""
# m = YOLO(f"{OUT}/runs/model_frunze_bolnave_v2/weights/last.pt"); m.add_callback("on_fit_epoch_end", print_epoch); m.train(resume=True)
# m = YOLO(f"{OUT}/runs/model_ciorchine_v2/weights/last.pt"); m.add_callback("on_fit_epoch_end", print_epoch); m.train(resume=True)
"""),
    md("""
## 14. Vechi vs nou + salvare

- **mAP50 test**: cât de bine găsește ce trebuie (mai mare = mai bine; ideal să nu scadă).
- **Alarme false pe poză**: la pragul 0,5 folosit pe laptop (mai mic = mai bine). Pentru frunze: pe cele 194 de poze de test label map
  (vie sănătoasă, exact „tura 1”). Pentru ciorchini: pe decupajele de frunze bolnave din FD-test.
"""),
    code("""
NEW_LEAVES = f"{OUT}/runs/model_frunze_bolnave_v2/weights/best.pt"
NEW_BUNCHES = f"{OUT}/runs/model_ciorchine_v2/weights/best.pt"

# save the new models first, so they are in Drive even if the comparison below fails
shutil.copy(NEW_LEAVES, f"{OUT}/model_frunze_bolnave_v2.pt")
shutil.copy(NEW_BUNCHES, f"{OUT}/model_ciorchine_v2.pt")

# free the video card memory left over from training
import gc, torch
for name in ("leaf_model", "bunch_model", "m"):
    globals().pop(name, None)
gc.collect()
torch.cuda.empty_cache()

def test_map50(weights, data, imgsz):
    return YOLO(weights).val(data=str(data), split="test", imgsz=imgsz, verbose=False, plots=False).box.map50

def false_alarms_per_photo(weights, images, imgsz, conf=0.5):
    # one photo at a time: a list passed to predict() becomes ONE batch of all photos (out of memory at 960 px)
    model = YOLO(weights)
    boxes = sum(len(model.predict(str(p), imgsz=imgsz, conf=conf, verbose=False)[0].boxes) for p in images)
    return boxes / len(images)

healthy = images_in(LABEL_MAP / "test" / "images")
rows = [
    ("Frunze: mAP50 test (FD)", test_map50(OLD_LEAVES, LEAVES / "data.yaml", 960), test_map50(NEW_LEAVES, LEAVES / "data.yaml", 960)),
    ("Frunze: alarme false/poza (vie sanatoasa)", false_alarms_per_photo(OLD_LEAVES, healthy, 960),
     false_alarms_per_photo(NEW_LEAVES, healthy, 960)),
    ("Ciorchini: mAP50 test (WGISD)", test_map50(OLD_BUNCHES, BUNCHES / "data.yaml", 640), test_map50(NEW_BUNCHES, BUNCHES / "data.yaml", 640)),
    ("Ciorchini: alarme false/decupaj (frunze bolnave)", false_alarms_per_photo(OLD_BUNCHES, test_crops, 640),
     false_alarms_per_photo(NEW_BUNCHES, test_crops, 640)),
]
print(f"{'':50s} {'vechi':>8s} {'nou':>8s}")
for name, old, new in rows:
    print(f"{name:50s} {old:8.3f} {new:8.3f}")

print("\\nSaved:", f"{OUT}/model_frunze_bolnave_v2.pt", "and", f"{OUT}/model_ciorchine_v2.pt")
print("Copy them to the laptop, in the models/ folder.")
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Written:", OUT)
