import json
from pathlib import Path

OUT = Path(str(Path(__file__).resolve().parent.parent / "fd_frunze_bolnave.ipynb"))


EPOCH_PRINTER = '''
import time
from ultralytics import YOLO

T0 = time.time()

def afiseaza_epoca(trainer):
    """După fiecare epocă: o linie clară cu progresul, scorul și timpul rămas."""
    ep, total = trainer.epoch + 1, trainer.epochs
    scurs = time.time() - (getattr(trainer, "train_time_start", None) or T0)
    ramas = scurs / ep * (total - ep)
    map50 = trainer.metrics.get("metrics/mAP50(B)", 0.0)
    best = getattr(getattr(trainer, "stopper", None), "best_epoch", ep)  # Ultralytics numără deja de la 1
    print(f"==> Epoca {ep}/{total} ({ep / total:.0%}) | mAP50 validare: {map50:.3f} | cea mai bună: epoca {best} | "
          f"trecut: {scurs / 60:.1f} min | rămas: max ~{ramas / 60:.0f} min", flush=True)
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Fine-tuning YOLO pe frunze bolnave — dataset Flavescence dorée (Franța)

Dataset: [An expertized grapevine disease image database focused on Flavescence dorée and its confounding diseases](https://data.mendeley.com/datasets/3dr9r3w3jn/2) (Mendeley Data, v2, CC BY 4.0).
Butuci întregi pozați de la 1–2 m, regiunea Bordeaux, 5 soiuri (Cabernet Sauvignon, Cabernet Franc, Merlot, Ugni Blanc, Sauvignon Blanc).

Folosim partea `symptom_scale_box`: **744 poze** cu boxuri pe frunze, format LabelMe (JSON). Toate clasele de frunze devin **frunza_bolnava**:
- `FD leaf`, `ESCA leaf`, `confounding leaf`, `facteur confondant +` → **frunza_bolnava**
- `symptomatic bunch` și `symptomatic shoot` (linii) → ignorate

**Descărcare: ~1 GB.** Fiecare JSON conține și poza (codată în base64), așa că descărcăm doar JSON-urile, nu și JPG-urile separat.

Frunzele sunt mici în poză (mediana ~80–120 px dintr-o poză de 2448×2048), așa că antrenăm la **960 px** în loc de 640.

Atenție: pe o parte din poze (folderul `soft_annotation`) experții au marcat doar o parte din frunzele cu simptome, deci modelul poate rata unele frunze.

**Ce trebuie să faceți voi:** la celula *Google Drive*, apăsați „Connect to Google Drive” și permiteți accesul. Nu e nevoie de cheie API.

Rezultatul final: `MyDrive/vie_hackathon/model_frunze_bolnave.pt`.
"""),
    md("## 1. Verificare placă video (trebuie să apară Tesla T4 sau alta)"),
    code("!nvidia-smi"),
    md("## 2. Instalare"),
    code("""
!pip install -q -U ultralytics
import ultralytics
print("ultralytics", ultralytics.__version__, "— instalați aceeași versiune și pe laptop")
"""),
    md("## 3. Google Drive (aici se salvează modelul)"),
    code("""
import os
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/vie_hackathon'
os.makedirs(OUT, exist_ok=True)
print("Rezultatele se salvează în", OUT)
"""),
    md("""
## 4. Descărcare + conversie în format YOLO (~1 GB, câteva minute)

Descarcă în paralel cele 744 de JSON-uri, scoate poza din fiecare și scrie etichetele YOLO. Dacă se întrerupe, rulați celula din nou: sare peste ce e deja descărcat.
"""),
    code("""
import base64, collections, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

ALL = Path("/content/fd_all")
(ALL / "images").mkdir(parents=True, exist_ok=True)
(ALL / "labels").mkdir(parents=True, exist_ok=True)

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
            stem = re.sub(r"\\s*\\((\\d+)\\)", r"_dup\\1", Path(f["filename"]).stem)  # „im_00031 (2)” → „im_00031_dup2”
            stem = re.sub(r"[^A-Za-z0-9_]", "", f"{folder}_{stem}")
            jobs.append((folder, stem, url))
        else:
            jpg_urls[(folder, f["filename"])] = url
print(len(jobs), "fișiere de adnotare")

def convert(job):
    folder, stem, url = job
    img_path, lbl_path = ALL / "images" / f"{stem}.jpg", ALL / "labels" / f"{stem}.txt"
    if img_path.exists() and lbl_path.exists():
        return None
    a = get(url)
    W, H = a["imageWidth"], a["imageHeight"]
    lines, kept, skipped = [], collections.Counter(), collections.Counter()
    for s in a["shapes"]:
        if s["shape_type"] == "rectangle" and s["label"] in LEAF:
            (x1, y1), (x2, y2) = s["points"]
            x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))
            y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            kept[s["label"]] += 1
        else:
            skipped[s["label"]] += 1
    if a.get("imageData"):
        img = base64.b64decode(a["imageData"])
    else:
        img = get(jpg_urls[(folder, a["imagePath"])], as_json=False)
    img_path.write_bytes(img)
    lbl_path.write_text("\\n".join(lines))
    return kept, skipped

kept, skipped = collections.Counter(), collections.Counter()
with ThreadPoolExecutor(16) as ex:
    for i, res in enumerate(ex.map(convert, jobs), 1):
        if res:
            kept.update(res[0])
            skipped.update(res[1])
        if i % 25 == 0 or i == len(jobs):
            print(f"Descărcat {i}/{len(jobs)} ({i / len(jobs):.0%})", flush=True)

print("Poze:", len(list((ALL / "images").glob("*.jpg"))))
print("Frunze bolnave păstrate:", dict(kept), "— total", sum(kept.values()))
print("Ignorate:", dict(skipped))
"""),
    md("""
## 5. Împărțire train / valid / test (70 / 15 / 15)

Unele poze apar de două ori, cu sufixul „(2)”. Le ținem în același set, ca să nu ajungă aceeași poză și la antrenare, și la test.
"""),
    code("""
import random, shutil, yaml

DST = Path("/content/fd-vie")
if DST.exists():
    shutil.rmtree(DST)

groups = collections.defaultdict(list)
for img in sorted((ALL / "images").glob("*.jpg")):
    groups[re.sub(r"_dup\\d+$", "", img.stem)].append(img)

keys = sorted(groups)
random.seed(0)
random.shuffle(keys)
n = len(keys)
split_of = {k: ("train" if i < 0.70 * n else "valid" if i < 0.85 * n else "test") for i, k in enumerate(keys)}

counts = collections.Counter()
for k, imgs in groups.items():
    split = split_of[k]
    for img in imgs:
        for sub in ("images", "labels"):
            (DST / split / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, DST / split / "images" / img.name)
        shutil.copy(ALL / "labels" / f"{img.stem}.txt", DST / split / "labels" / f"{img.stem}.txt")
        counts[split] += 1
print(dict(counts))

yaml.safe_dump({"path": str(DST), "train": "train/images", "val": "valid/images", "test": "test/images",
                "names": {0: "frunza_bolnava"}},
               open(DST / "data.yaml", "w"), allow_unicode=True)
print(open(DST / "data.yaml").read())
"""),
    md("""
## 6. Fine-tuning (pornește de la modelul pre-antrenat pe COCO)

Antrenăm la 960 px, pentru că frunzele sunt mici în poză; din cauza asta punem 8 poze pe lot în loc de 16.
"""),
    code(EPOCH_PRINTER + """
try:
    model = YOLO("yolo26s.pt")
except Exception as e:
    print("yolo26s.pt nu e disponibil, folosesc yolo11s.pt:", e)
    model = YOLO("yolo11s.pt")

model.add_callback("on_fit_epoch_end", afiseaza_epoca)
model.train(data=str(DST / "data.yaml"), imgsz=960, epochs=100, patience=20, batch=8,
            cache="ram", project=f"{OUT}/runs", name="model_frunze_bolnave", exist_ok=True)
"""),
    md("""
### Dacă sesiunea s-a închis în timpul antrenării
Rulați din nou celulele 1–5 (fără 6), apoi celula de mai jos: antrenarea continuă de unde a rămas.
"""),
    code("""
# model = YOLO(f"{OUT}/runs/model_frunze_bolnave/weights/last.pt")
# model.add_callback("on_fit_epoch_end", afiseaza_epoca)
# model.train(resume=True)
"""),
    md("## 7. Evaluare pe setul de test + salvare model final"),
    code("""
best = YOLO(f"{OUT}/runs/model_frunze_bolnave/weights/best.pt")
metrics = best.val(data=str(DST / "data.yaml"), split="test", imgsz=960,
                   project=f"{OUT}/runs", name="test_frunze_bolnave", exist_ok=True)
print(f"mAP50 test: {metrics.box.map50:.3f}   mAP50-95 test: {metrics.box.map:.3f}")

shutil.copy(f"{OUT}/runs/model_frunze_bolnave/weights/best.pt", f"{OUT}/model_frunze_bolnave.pt")
print("Model salvat:", f"{OUT}/model_frunze_bolnave.pt")
"""),
    md("## 8. Verificare vizuală pe câteva poze de test"),
    code("""
import glob
from IPython.display import Image, display

best.predict(source=str(DST / "test" / "images"), imgsz=960, conf=0.3, save=True,
             project=f"{OUT}/runs", name="predictii_frunze", exist_ok=True)
for f in sorted(glob.glob(f"{OUT}/runs/predictii_frunze/*.jpg"))[:6]:
    display(Image(filename=f, width=640))
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

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Scris:", OUT)
