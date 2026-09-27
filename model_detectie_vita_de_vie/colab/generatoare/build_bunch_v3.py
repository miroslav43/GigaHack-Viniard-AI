"""Generates colab/bunch_v3.ipynb: bunch model v3 (model_ciorchine_v2 fine-tuned on ViViD-5K + WGISD), tested on
VINEPICs (unseen vineyard), WGISD test and ViViD test, old vs new. Budget: about 1 hour on a T4.
Embeds tools/bunch_v3.py as-is."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
OUT = HERE.parent / "bunch_v3.ipynb"
SCRIPT = (PROJ / "tools" / "bunch_v3.py").read_text(encoding="utf-8")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Ciorchini v3: 13 soiuri noi (ViViD-5K) — ~1 oră

**De ce:** pe **VINEPICs** (Italia, 238 de poze, vie pe care n-a văzut-o niciun model de-al nostru) ciorchinii de acum
găsesc doar **22%** din ciorchini (precizie 87%). Pe sesiunile cu Cabernet închis la culoare, cameră jos și cer luminos
găsesc 13 din 714. Și modelul vechi (v1) e la fel de slab: modelele cunosc doar WGISD.

**Ce face:** pornește de la `model_ciorchine_v2.pt` și îl antrenează pe **ViViD-5K** (4.000 de poze, 13 soiuri, SUA,
CC BY 4.0; folosim variantele de 1024 px, ~0,7 GB) + **WGISD** (ca să nu uite ce știe). **VINEPICs** rămâne în întregime
test. La final: tabel vechi vs nou pe VINEPICs, WGISD test și ViViD test.

## Ce faceți voi (totul la început)
1. **Runtime → Change runtime type → T4 GPU → Save**, apoi **Runtime → Run all**.
2. Celula 3: „Connect to Google Drive”.
3. Celula 4: dacă `model_ciorchine_v2.pt` nu e în Drive, alegeți-l de pe laptop (`models/`).

Durată estimată (nemăsurată): descărcări ~10 min, set ~5 min, antrenare ~25–30 min (12 epoci), test ~10 min.
Rezultatul: `MyDrive/vie_hackathon/bunch_v3_runs/bunch_v3/weights/best.pt` + tabelul vechi vs nou.
"""),
    md("## 1. Placa video"),
    code("!nvidia-smi"),
    md("## 2. Instalare"),
    code("""
!pip install -q ultralytics==8.4.163
import ultralytics, pandas, pyarrow
print("ultralytics", ultralytics.__version__, "| pandas", pandas.__version__, "| pyarrow", pyarrow.__version__)
"""),
    md("## 3. Google Drive"),
    code("""
import os
from google.colab import drive
drive.mount('/content/drive')
OUT = '/content/drive/MyDrive/vie_hackathon'
R = f"{OUT}/bunch_v3_runs"
os.makedirs(R, exist_ok=True)
os.environ["BUNCH_ROOT"] = "/content/bv"
os.environ["BUNCH_RUNS"] = R
for d in ("data/vivid", "data/vinepics", "models"):
    os.makedirs(f"/content/bv/{d}", exist_ok=True)
"""),
    md("## 4. Modelul actual de ciorchini (punctul de plecare)"),
    code("""
import shutil
from google.colab import files

if os.path.exists(f"{OUT}/model_ciorchine_v2.pt"):
    shutil.copy(f"{OUT}/model_ciorchine_v2.pt", "/content/bv/models/model_ciorchine_v2.pt")
while not os.path.exists("/content/bv/models/model_ciorchine_v2.pt"):
    print("model_ciorchine_v2.pt is not in Drive -> choose it from the laptop (folder models/)")
    for name, data in files.upload().items():
        if "model_ciorchine_v2" in name:
            open("/content/bv/models/model_ciorchine_v2.pt", "wb").write(data)
            shutil.copy("/content/bv/models/model_ciorchine_v2.pt", f"{OUT}/model_ciorchine_v2.pt")
print("OK: model_ciorchine_v2.pt")
"""),
    md("## 5. Descărcări, toate deodată (~10 min): ViViD-5K (~1,3 GB), WGISD (git), VINEPICs (346 MB)"),
    code("""
%%bash
cd /content
dl() {  # dl <name> <command>: runs in the background, writes dl_<name>.log ending with OK or FAILED
  ( bash -c "$2" > dl_$1.log 2>&1 && echo OK >> dl_$1.log || echo FAILED >> dl_$1.log ) &
}
HF=https://huggingface.co/datasets/XZhi/ViViD-5k/resolve/main
dl vivid "cd /content/bv/data/vivid && for f in train-00000-of-00004 train-00001-of-00004 train-00002-of-00004 train-00003-of-00004 val-00000-of-00001 test-00000-of-00001; do curl -sSfL --retry 3 -o \\$f.parquet $HF/parquet/\\$f.parquet || exit 1; done && curl -sSfL --retry 3 -o instances_updated.json $HF/data/anns/instances_updated.json"
dl wgisd 'rm -rf /content/bv/data/wgisd && git clone -q --depth 1 https://github.com/thsant/wgisd.git /content/bv/data/wgisd'
dl vinepics 'curl -sSfL --retry 3 -o vinepics.zip "https://zenodo.org/api/records/7866442/files/VINEPICs.zip/content" && unzip -q -o vinepics.zip -d /content/bv/data/vinepics'
wait
for n in vivid wgisd vinepics; do
  echo "$n: $(tail -n 1 dl_$n.log)"
  if [ "$(tail -n 1 dl_$n.log)" != OK ]; then tail -n 4 dl_$n.log | head -n 3 | sed 's/^/   /'; fi
done
"""),
    code("""
import glob
counts = {
    "ViViD parquet files": len(glob.glob("/content/bv/data/vivid/*.parquet")),
    "ViViD boxes file (MB)": round(os.path.getsize("/content/bv/data/vivid/instances_updated.json") / 1e6)
        if os.path.exists("/content/bv/data/vivid/instances_updated.json") else 0,
    "WGISD photos": len(glob.glob("/content/bv/data/wgisd/data/*.jpg")),
    "VINEPICs photos": len(glob.glob("/content/bv/data/vinepics/data/images/*/*.png")),
}
print(counts)
assert counts["ViViD parquet files"] == 6 and all(counts.values()), "a download failed: run the cell above again"
"""),
    md("## 6. Scriptul (același ca în proiect: `tools/bunch_v3.py`)"),
    code("%%writefile /content/bunch_v3.py\n" + SCRIPT),
    md("## 7. Setul de date (~5 min)"),
    code("!cd /content && python bunch_v3.py build"),
    md("## 8. Antrenare (~25–30 min, 12 epoci, progres pe epoci)"),
    code("!cd /content && python bunch_v3.py train --epochs 12 --batch 16"),
    md("""
### Continuare (doar dacă sesiunea s-a închis în timpul antrenării)
Rulați din nou celulele 1–7, apoi scoateți `#` din linia de mai jos.
"""),
    code("""
# !cd /content && python -c "from ultralytics import YOLO; YOLO('{R}/bunch_v3/weights/last.pt').train(resume=True)"
"""),
    md("## 9. Testul final: vechi vs nou (~10 min)"),
    code('!cd /content && python bunch_v3.py report --weights "{R}/bunch_v3/weights/best.pt"'),
    md("""
## 10. Pe laptop
Trimiteți-i lui Claude tabelul. Dacă v3 e clar mai bun pe VINEPICs fără să piardă mult pe WGISD, descărcați
`bunch_v3_runs/bunch_v3/weights/best.pt` în `models/` cu numele `model_ciorchine_v3.pt`: îl verificăm pe laptop și pe
tura 1 / tura 2 (ciorchini falși pe frunze roșii) înainte să-l punem în `vineyard/config.py`.
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Written:", OUT)
