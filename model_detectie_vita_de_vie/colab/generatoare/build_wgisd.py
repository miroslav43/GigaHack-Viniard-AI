import json
from pathlib import Path

OUT = Path(str(Path(__file__).resolve().parent.parent / "wgisd_ciorchine.ipynb"))


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
# Fine-tuning YOLO pe WGISD — ciorchini dezvoltați

Dataset: [Embrapa WGISD](https://github.com/thsant/wgisd) — 300 poze, 5 soiuri (Chardonnay, Cabernet Franc, Cabernet Sauvignon, Sauvignon Blanc, Syrah), vie pe spalier.
Licență: CC BY-NC 4.0 (necomercial — OK pentru hackathon).

Etichetele sunt deja în format YOLO, o singură clasă → **ciorchine**.
Împărțire: `test.txt` oficial (58 poze) rămâne test; din `train.txt` (242 poze) scoatem 15% pentru validare.

**Ce trebuie să faceți voi:** la celula *Google Drive*, apăsați „Connect to Google Drive” și permiteți accesul. Nu e nevoie de cheie API.

Rezultatul final: `MyDrive/vie_hackathon/model_ciorchine.pt`.
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
    md("## 4. Descărcare WGISD (~1,5 GB, durează cam un minut)"),
    code("""
import os
if not os.path.exists("/content/wgisd"):
    !git clone --depth 1 https://github.com/thsant/wgisd.git /content/wgisd
else:
    print("WGISD e deja descărcat")
"""),
    md("## 5. Pregătire dataset: train / valid / test"),
    code("""
import random, shutil, yaml
from pathlib import Path

SRC = Path("/content/wgisd")
DST = Path("/content/wgisd-vie")
if DST.exists():
    shutil.rmtree(DST)

def ids(fname):
    return [l.strip() for l in open(SRC / fname) if l.strip()]

train_ids, test_ids = ids("train.txt"), ids("test.txt")
random.seed(0)
random.shuffle(train_ids)
n_val = round(len(train_ids) * 0.15)
plan = {"train": train_ids[n_val:], "valid": train_ids[:n_val], "test": test_ids}

boxes = 0
for split, names in plan.items():
    (DST / split / "images").mkdir(parents=True)
    (DST / split / "labels").mkdir(parents=True)
    for n in names:
        shutil.copy(SRC / "data" / f"{n}.jpg", DST / split / "images" / f"{n}.jpg")
        lines = [l for l in (SRC / "data" / f"{n}.txt").read_text().splitlines() if l.strip()]
        assert all(l.split()[0] == "0" for l in lines), f"clasă neașteptată în {n}"
        (DST / split / "labels" / f"{n}.txt").write_text("\\n".join(lines))
        boxes += len(lines)
    print(f"{split}: {len(names)} poze")
print("Boxuri ciorchine:", boxes)

yaml.safe_dump({"path": str(DST), "train": "train/images", "val": "valid/images", "test": "test/images",
                "names": {0: "ciorchine"}},
               open(DST / "data.yaml", "w"), allow_unicode=True)
print(open(DST / "data.yaml").read())
"""),
    md("""
## 6. Fine-tuning (pornește de la modelul pre-antrenat pe COCO)

Dataset-ul e mic (~200 poze de antrenare), așa că lăsăm mai multe epoci (150) și mai multă răbdare (30). O epocă durează câteva secunde.
"""),
    code(EPOCH_PRINTER + """
try:
    model = YOLO("yolo26s.pt")
except Exception as e:
    print("yolo26s.pt nu e disponibil, folosesc yolo11s.pt:", e)
    model = YOLO("yolo11s.pt")

model.add_callback("on_fit_epoch_end", afiseaza_epoca)
model.train(data=str(DST / "data.yaml"), imgsz=640, epochs=150, patience=30, batch=16,
            cache="ram", project=f"{OUT}/runs", name="model_ciorchine", exist_ok=True)
"""),
    md("""
### Dacă sesiunea s-a închis în timpul antrenării
Rulați din nou celulele 1–5 (fără 6), apoi celula de mai jos: antrenarea continuă de unde a rămas.
"""),
    code("""
# model = YOLO(f"{OUT}/runs/model_ciorchine/weights/last.pt")
# model.add_callback("on_fit_epoch_end", afiseaza_epoca)
# model.train(resume=True)
"""),
    md("## 7. Evaluare pe setul de test + salvare model final"),
    code("""
best = YOLO(f"{OUT}/runs/model_ciorchine/weights/best.pt")
metrics = best.val(data=str(DST / "data.yaml"), split="test", project=f"{OUT}/runs", name="test_ciorchine", exist_ok=True)
print(f"mAP50 test: {metrics.box.map50:.3f}   mAP50-95 test: {metrics.box.map:.3f}")

shutil.copy(f"{OUT}/runs/model_ciorchine/weights/best.pt", f"{OUT}/model_ciorchine.pt")
print("Model salvat:", f"{OUT}/model_ciorchine.pt")
"""),
    md("## 8. Verificare vizuală pe câteva poze de test"),
    code("""
import glob
from IPython.display import Image, display

best.predict(source=str(DST / "test" / "images"), conf=0.4, save=True,
             project=f"{OUT}/runs", name="predictii_ciorchine", exist_ok=True)
for f in sorted(glob.glob(f"{OUT}/runs/predictii_ciorchine/*.jpg"))[:6]:
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
