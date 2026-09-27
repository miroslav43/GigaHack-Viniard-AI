"""Generates colab/leaf_daylight.ipynb: diseased-leaf model for daylight (fine-tune of model_frunze_bolnave.pt with strong
light augmentation), epoch chosen on EscaYard B7 vines, old vs new on B9. Budget: under 1 hour on a T4.
Embeds tools/leaf_daylight.py; the FD download and split cells are copied from colab/fd_frunze_bolnave.ipynb, so the
train / valid / test split is exactly the one of the first leaf model."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
OUT = HERE.parent / "leaf_daylight.ipynb"
SCRIPT = (PROJ / "tools" / "leaf_daylight.py").read_text(encoding="utf-8")
FD = json.loads((HERE.parent / "fd_frunze_bolnave.ipynb").read_text(encoding="utf-8"))["cells"]
FD_DOWNLOAD = next(c for c in FD if "".join(c["source"]).startswith("import base64"))
FD_SPLIT = next(c for c in FD if "".join(c["source"]).startswith("import random, shutil, yaml"))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Frunze bolnave pentru lumina de zi (model_frunze_bolnave_v3) — maxim ~1 oră

**De ce:** pe EscaYard, parcela B7 (pozată **seara**) modelul găsește ~6 frunze bolnave pe un butuc bolnav, dar pe B9
(pozată **la amiază**) doar ~1,9, așa că pe B9 găsește doar 3 din 12 butuci bolnavi. Modelul a fost antrenat pe poze
cu **blitz**; în soare puternic simptomele „dispar”.

**Ce face:** pornește de la modelul actual și îl reantrenează pe **aceleași** poze FD (aceeași împărțire), cu augmentare
puternică de lumină: mai luminos, mai mult contrast, gamma, umbre (soare de amiază) + compresie JPEG și zgomot (ESP32).
Epoca se alege pe butucii **B7**; **B9** e doar testul final (vechi vs nou).

## Ce faceți voi (totul la început)
1. **Runtime → Change runtime type → T4 GPU → Save**, apoi **Runtime → Run all**.
2. Celula 3: „Connect to Google Drive”.
3. Celula 4: dacă `model_frunze_bolnave.pt` nu e în Drive, alegeți-l de pe laptop (`models/`).

Durată estimată: **~40–55 min** (descărcări ~10 min, antrenare ~20 min, alegere + test ~10 min).
Rezultatul: `MyDrive/vie_hackathon/leaf_v3_runs/model_frunze_bolnave_v3.pt` + tabelul vechi vs nou.
"""),
    md("## 1. Placa video"),
    code("!nvidia-smi"),
    md("## 2. Instalare"),
    code("""
!pip install -q ultralytics==8.4.163 py7zr
import ultralytics, albumentations
print("ultralytics", ultralytics.__version__, "| albumentations", albumentations.__version__)
"""),
    md("## 3. Google Drive"),
    code("""
import os
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/vie_hackathon'
R = f"{OUT}/leaf_v3_runs"
os.makedirs(R, exist_ok=True)
os.environ["LEAF_ROOT"] = "/content/lv"
os.environ["LEAF_RUNS"] = R
for d in ("data/escayard/photos", "models"):
    os.makedirs(f"/content/lv/{d}", exist_ok=True)
"""),
    md("## 4. Modelul actual de frunze (punctul de plecare)"),
    code("""
import shutil
from google.colab import files

if os.path.exists(f"{OUT}/model_frunze_bolnave.pt"):
    shutil.copy(f"{OUT}/model_frunze_bolnave.pt", "/content/lv/models/model_frunze_bolnave.pt")
while not os.path.exists("/content/lv/models/model_frunze_bolnave.pt"):
    print("model_frunze_bolnave.pt is not in Drive -> choose it from the laptop (folder models/)")
    for name, data in files.upload().items():
        if "model_frunze_bolnave" in name and "v2" not in name:
            open("/content/lv/models/model_frunze_bolnave.pt", "wb").write(data)
print("OK: model_frunze_bolnave.pt")
"""),
    md("## 5. EscaYard (1,4 GB) se descarcă în fundal cât timp luăm pozele FD"),
    code("""
import subprocess
escayard = subprocess.Popen(["bash", "-c", '''
cd /content
curl -sSfL --retry 3 -o /content/lv/data/escayard/datasheet.csv "https://zenodo.org/api/records/10362568/files/datasheet.csv/content" &&
curl -sSfL --retry 3 -o escayard.7z "https://zenodo.org/api/records/10362568/files/PLANTs_SmartPhonePhotos_JPG.7z/content" &&
python3 -c "import py7zr; py7zr.SevenZipFile('escayard.7z').extractall('/content/lv/data/escayard/photos')" &&
echo ESCAYARD_OK'''], stdout=open("/content/dl_escayard.log", "w"), stderr=subprocess.STDOUT)
print("EscaYard download started")
"""),
    md("## 6. Pozele FD (Bordeaux) — aceeași descărcare ca la primul model de frunze (~1 GB)"),
    FD_DOWNLOAD,
    md("## 7. Aceeași împărțire train / valid / test ca la primul model (seed 0)"),
    FD_SPLIT,
    md("## 8. Așteptăm EscaYard"),
    code("""
import glob, time
while escayard.poll() is None:
    time.sleep(10)
print(open("/content/dl_escayard.log").read()[-300:])
print("EscaYard photos:", len(glob.glob("/content/lv/data/escayard/photos/**/*.jpg", recursive=True)))
"""),
    md("## 9. Scriptul (același ca în proiect: `tools/leaf_daylight.py`)"),
    code("%%writefile /content/leaf_daylight.py\n" + SCRIPT),
    md("## 10. Antrenare (~20 min, 24 de epoci, progres pe epoci)"),
    code("!cd /content && python leaf_daylight.py train --data /content/fd-vie/data.yaml --epochs 24"),
    md("## 11. Alegem epoca pe butucii B7 (~5 min)"),
    code("!cd /content && python leaf_daylight.py select --run leaf_v3"),
    md("## 12. Testul final: vechi vs nou pe B9 (pozat la amiază) + testul FD (~5 min)"),
    code('!cd /content && python leaf_daylight.py report --weights "{R}/model_frunze_bolnave_v3.pt" --data /content/fd-vie/data.yaml'),
    md("""
## 13. Pe laptop
Trimiteți-i lui Claude tabelul de mai sus. Dacă modelul nou e mai bun pe B9 fără să marcheze mai mulți butuci sănătoși,
descărcați `leaf_v3_runs/model_frunze_bolnave_v3.pt` în `models/`: îl verificăm pe laptop și pe tura 1 (alarme false
pe via sănătoasă de primăvară) înainte să-l punem în `vineyard/config.py`.
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Written:", OUT)
