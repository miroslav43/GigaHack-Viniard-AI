import json
from pathlib import Path

OUT = Path(str(Path(__file__).resolve().parent.parent / "label_map_trunchi.ipynb"))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Fine-tuning YOLO pe „label map” — trunchi + ciorchini tineri

Dataset: [label map (Roboflow Universe)](https://universe.roboflow.com/grape-detection/label-map), versiunea **v8** (1.929 poze, fără augmentări, CC BY 4.0).

Atenție: doar v8 (și v9, augmentată) au trunchiuri. v10–v14 au doar ciorchini (`grappoli_medi`, `grappoli_piccoli`), v15 doar `grappoli_medi`.

Clasele originale sunt unificate astfel:
- `trunk`, `truk` → **trunchi**
- `medium_grape_bunch`, `tiny_grape_bunch` → **ciorchine**

**Ce trebuie să faceți voi când rulați:**
1. La celula *Google Drive*: apăsați „Connect to Google Drive” și permiteți accesul.
2. La celula *Descărcare dataset*: lipiți cheia API Roboflow (roboflow.com → Settings → API Keys). Cheia nu se afișează și nu se salvează în notebook.

Rezultatul final: `MyDrive/vie_hackathon/model_trunchi.pt`.
"""),
    md("## 1. Verificare placă video (trebuie să apară Tesla T4 sau alta)"),
    code("!nvidia-smi"),
    md("## 2. Instalare"),
    code("""
!pip install -q -U ultralytics roboflow
import ultralytics
print("ultralytics", ultralytics.__version__, "— instalați aceeași versiune și pe laptop")
"""),
    md("## 3. Google Drive (aici se salvează modelul, ca să nu se piardă dacă se închide sesiunea)"),
    code("""
import os
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/vie_hackathon'
os.makedirs(OUT, exist_ok=True)
print("Rezultatele se salvează în", OUT)
"""),
    md("## 4. Descărcare dataset (v8, format YOLOv8)"),
    code("""
from getpass import getpass
from roboflow import Roboflow

rf = Roboflow(api_key=getpass("Cheia API Roboflow: "))
dataset = rf.workspace("grape-detection").project("label-map").version(8).download(
    "yolov8", location="/content/label-map")
"""),
    md("""
## 5. Unificare clase + împărțire train / valid / test

Versiunea v8 vine deja împărțită: 1.347 train / 388 valid / 194 test. Păstrăm împărțirea originală.

Atenție: pozele sunt cadre dintr-un video filmat de un robot, deci cadre vecine pot ajunge în seturi diferite și scorul de test poate ieși puțin mai bun decât în realitate.
"""),
    code("""
import shutil, yaml
from pathlib import Path

SRC = Path("/content/label-map")
DST = Path("/content/label-map-vie")
if DST.exists():
    shutil.rmtree(DST)

names = yaml.safe_load(open(SRC / "data.yaml"))["names"]
if isinstance(names, dict):
    names = [names[k] for k in sorted(names)]
print("Clase originale:", names)

MAP = {"trunk": 0, "truk": 0, "medium_grape_bunch": 1, "tiny_grape_bunch": 1}
NEW_NAMES = ["trunchi", "ciorchine"]
old2new = {}
for i, n in enumerate(names):
    if n in MAP:
        old2new[i] = MAP[n]
    else:
        print("ATENȚIE: clasă necunoscută, o ignor:", n)

def images(split):
    return sorted(p for p in (SRC / split / "images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

plan = {"train": images("train"), "valid": images("valid"), "test": images("test")}

counts = {0: 0, 1: 0}
for split, imgs in plan.items():
    (DST / split / "images").mkdir(parents=True)
    (DST / split / "labels").mkdir(parents=True)
    for img in imgs:
        shutil.copy(img, DST / split / "images" / img.name)
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        lines = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts and int(parts[0]) in old2new:
                    c = old2new[int(parts[0])]
                    lines.append(" ".join([str(c)] + parts[1:]))
                    counts[c] += 1
        (DST / split / "labels" / (img.stem + ".txt")).write_text("\\n".join(lines))
    print(f"{split}: {len(imgs)} poze")

print("Boxuri:", {NEW_NAMES[k]: v for k, v in counts.items()})

yaml.safe_dump({"path": str(DST), "train": "train/images", "val": "valid/images", "test": "test/images",
                "names": dict(enumerate(NEW_NAMES))},
               open(DST / "data.yaml", "w"), allow_unicode=True)
print(open(DST / "data.yaml").read())
"""),
    md("## 6. Fine-tuning (pornește de la modelul pre-antrenat pe COCO)"),
    code("""
from ultralytics import YOLO

try:
    model = YOLO("yolo26s.pt")
except Exception as e:
    print("yolo26s.pt nu e disponibil, folosesc yolo11s.pt:", e)
    model = YOLO("yolo11s.pt")

model.train(data=str(DST / "data.yaml"), imgsz=640, epochs=100, patience=20, batch=16,
            cache="ram", project=f"{OUT}/runs", name="model_trunchi", exist_ok=True)
"""),
    md("""
### Dacă sesiunea s-a închis în timpul antrenării
Rulați din nou celulele 1–5 (fără 6), apoi celula de mai jos: antrenarea continuă de unde a rămas.
"""),
    code("""
# model = YOLO(f"{OUT}/runs/model_trunchi/weights/last.pt")
# model.train(resume=True)
"""),
    md("## 7. Evaluare pe setul de test + salvare model final"),
    code("""
best = YOLO(f"{OUT}/runs/model_trunchi/weights/best.pt")
metrics = best.val(data=str(DST / "data.yaml"), split="test", project=f"{OUT}/runs", name="test", exist_ok=True)
print(f"mAP50 test: {metrics.box.map50:.3f}   mAP50-95 test: {metrics.box.map:.3f}")

shutil.copy(f"{OUT}/runs/model_trunchi/weights/best.pt", f"{OUT}/model_trunchi.pt")
print("Model salvat:", f"{OUT}/model_trunchi.pt")
"""),
    md("## 8. Verificare vizuală pe câteva poze de test"),
    code("""
import glob
from IPython.display import Image, display

best.predict(source=str(DST / "test" / "images"), conf=0.4, save=True,
             project=f"{OUT}/runs", name="predictii_test", exist_ok=True)
for f in sorted(glob.glob(f"{OUT}/runs/predictii_test/*.jpg"))[:6]:
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
