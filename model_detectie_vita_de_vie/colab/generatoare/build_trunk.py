"""Generates colab/trunk_new_vineyard.ipynb: trunk model for new vineyards (label map + HUMAIN-Lab + auto-labeled posts,
then self-training on EscaYard block B7, tested on block B9). Embeds tools/trunk_adapt.py as-is."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
OUT = HERE.parent / "trunk_new_vineyard.ipynb"
SCRIPT = (PROJ / "tools" / "trunk_adapt.py").read_text(encoding="utf-8")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Trunchiuri într-o vie nouă (model_trunchi_v2)

**Scop:** modelul de trunchi să găsească trunchiul în **peste 80%** din pozele unei vii pe care n-a văzut-o
(EscaYard, Spania). Tura 1 (robotul, Portugalia) rămâne pe modelul vechi `model_trunchi.pt`.

**Cum:**
1. Antrenăm pe **mai multe vii**: label map (Portugalia) + HUMAIN-Lab (3 vii din Grecia), plus o clasă „stâlp”
   etichetată automat, ca modelul să nu mai confunde stâlpii cu trunchiurile.
2. **Auto-adaptare la via nouă**: modelul etichetează singur trunchiurile din parcela **B7** (fără desen manual),
   ne reantrenăm pe ele, de 2 ori.
3. **Test cinstit** pe parcela **B9**, pe care modelul n-o vede niciodată la antrenare.

## Ce faceți voi (totul la început)
1. **Runtime → Change runtime type → T4 GPU → Save**, apoi **Runtime → Run all**.
2. Celula 3: „Connect to Google Drive”.
3. Celula 4: dacă `model_trunchi.pt` nu e în Drive, alegeți-l de pe laptop (`models/`).
4. Celula 5: lipiți cheia API Roboflow.

Durată: **~1–1,5 ore**. Rezultat în `MyDrive/vie_hackathon/trunk_runs/` (modele + poze desenate din B9).
"""),
    md("## 1. Placa video"),
    code("!nvidia-smi"),
    md("## 2. Instalare"),
    code("""
!pip install -q ultralytics==8.4.163 py7zr
import ultralytics
print("ultralytics", ultralytics.__version__)
"""),
    md("## 3. Google Drive"),
    code("""
import os
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/vie_hackathon'
os.makedirs(f"{OUT}/trunk_runs", exist_ok=True)
os.environ["TRUNK_ROOT"] = "/content/tv"
os.environ["TRUNK_RUNS"] = f"{OUT}/trunk_runs"
os.makedirs("/content/tv/data", exist_ok=True)
os.makedirs("/content/tv/models", exist_ok=True)
"""),
    md("## 4. Modelul vechi de trunchi (punctul de plecare)"),
    code("""
import shutil
from google.colab import files

if os.path.exists(f"{OUT}/model_trunchi.pt"):
    shutil.copy(f"{OUT}/model_trunchi.pt", "/content/tv/models/model_trunchi.pt")
while not os.path.exists("/content/tv/models/model_trunchi.pt"):
    print("model_trunchi.pt is not in Drive -> choose it from the laptop (folder models/)")
    for name, data in files.upload().items():
        if "model_trunchi" in name:
            open("/content/tv/models/model_trunchi.pt", "wb").write(data)
print("OK: model_trunchi.pt")
"""),
    md("## 5. label map v8 (Roboflow) — lipiți cheia API"),
    code("""
import json, urllib.request, zipfile
from getpass import getpass

key = getpass("Roboflow API key: ")
info = json.load(urllib.request.urlopen(f"https://api.roboflow.com/grape-detection/label-map/8/yolov8?api_key={key}"))
urllib.request.urlretrieve(info["export"]["link"], "/content/labelmap.zip")
zipfile.ZipFile("/content/labelmap.zip").extractall("/content/tv/data/labelmap")
print({s: len(os.listdir(f"/content/tv/data/labelmap/{s}/images")) for s in ("train", "valid", "test")})
"""),
    md("## 6. HUMAIN-Lab vine-trunk (Grecia, ~100 MB)"),
    code("""
urllib.request.urlretrieve("https://github.com/humain-lab/vine-trunk/raw/master/Humain-Lab-vine-trunk-dataset.zip",
                           "/content/humain.zip")
zipfile.ZipFile("/content/humain.zip").extractall("/content/tv/data/humain")
print("original train photos:", len(os.listdir("/content/tv/data/humain/git/train/Original_photos")) // 2)
"""),
    md("## 7. EscaYard (Spania, 1,4 GB): B7 = adaptare, B9 = test"),
    code("""
import py7zr
base = "https://zenodo.org/api/records/10362568/files"
os.makedirs("/content/tv/data/escayard/photos", exist_ok=True)
urllib.request.urlretrieve(f"{base}/datasheet.csv/content", "/content/tv/data/escayard/datasheet.csv")
urllib.request.urlretrieve(f"{base}/PLANTs_SmartPhonePhotos_JPG.7z/content", "/content/escayard.7z")
with py7zr.SevenZipFile("/content/escayard.7z") as a:
    a.extractall("/content/tv/data/escayard/photos")
print("photos:", sum(len(f) for _, _, f in os.walk("/content/tv/data/escayard/photos")))
"""),
    md("## 8. Scriptul (același ca în proiect: `tools/trunk_adapt.py`)"),
    code("%%writefile /content/trunk_adapt.py\n" + SCRIPT),
    md("## 9. Setul combinat + stâlpi etichetați automat (~5–10 min)"),
    code("!cd /content && python trunk_adapt.py build"),
    md("## 10. Runda 0: antrenare pe mai multe vii (~20 min) + test pe B9"),
    code("""
R = f"{OUT}/trunk_runs"
!cd /content && python trunk_adapt.py train --name r0 --epochs 30 --patience 8 --batch 16 --amp
!cd /content && python trunk_adapt.py eval --weights "{R}/r0/weights/best.pt"
"""),
    md("## 11. Runda 1: etichetare automată pe B7 + reantrenare (~20 min) + test pe B9"),
    code("""
!cd /content && python trunk_adapt.py pseudo --weights "{R}/r0/weights/best.pt" --conf 0.4
!cd /content && python trunk_adapt.py train --name r1 --init "{R}/r0/weights/best.pt" --epochs 25 --patience 8 --batch 16 --amp
!cd /content && python trunk_adapt.py eval --weights "{R}/r1/weights/best.pt"
"""),
    md("## 12. Runda 2: încă o dată, cu modelul mai bun (~20 min) + test pe B9"),
    code("""
!cd /content && python trunk_adapt.py pseudo --weights "{R}/r1/weights/best.pt" --conf 0.5
!cd /content && python trunk_adapt.py train --name r2 --init "{R}/r1/weights/best.pt" --epochs 25 --patience 8 --batch 16 --amp
!cd /content && python trunk_adapt.py eval --weights "{R}/r2/weights/best.pt"
"""),
    md("""
## 13. Pentru laptop
Arhivăm pozele desenate din B9 și cele 3 modele. Descărcați din Drive `trunk_runs/trunk_results.zip`
și puneți-l în `Downloads` — Claude verifică pe poze dacă trunchiurile găsite sunt reale.
"""),
    code("""
import glob
for r in ("r0", "r1", "r2"):
    if os.path.exists(f"{R}/{r}/weights/best.pt"):
        shutil.copy(f"{R}/{r}/weights/best.pt", f"{R}/model_trunchi_v2_{r}.pt")
with zipfile.ZipFile(f"{R}/trunk_results.zip", "w") as z:
    for f in glob.glob(f"{R}/model_trunchi_v2_*.pt") + glob.glob(f"{R}/eval_B9_*/*.jpg"):
        z.write(f, os.path.relpath(f, R))
print("Saved:", f"{R}/trunk_results.zip", round(os.path.getsize(f"{R}/trunk_results.zip") / 1e6), "MB")
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Written:", OUT)
