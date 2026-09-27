"""Generates colab/trunk_v3.ipynb: trunk model v3 (rebalanced data mix + Vairão + stricter EscaYard auto-labels +
posts), checkpoint chosen on target-vineyard photos, compared with v2 on tests never trained on.
Embeds tools/trunk_v3.py and data/manual_trunks_b7.json as-is."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
OUT = HERE.parent / "trunk_v3.ipynb"
SCRIPT = (PROJ / "tools" / "trunk_v3.py").read_text(encoding="utf-8")
MANUAL = (PROJ / "data" / "manual_trunks_b7.json").read_text(encoding="utf-8")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# Trunchi v3: mai multe vii, convenție reparată

**De ce:** pe Vairão (Portugalia, vie nevăzută) modelele de acum găsesc trunchiul în doar **74 din 460** de poze.
65% din pozele de antrenare ale lui v2 vin din label map, unde trunchiul e un „ciot” tăiat de marginea de jos a pozei.

**Ce schimbă v3:**
1. label map redus la **30%** din pozele de antrenare;
2. **Vairão** adăugat (6 înregistrări la antrenare, 1 pentru alegerea epocii, 2 păstrate ca test);
3. etichete automate pe EscaYard B7 **mai stricte** (sigure + regăsite în oglindă; pozele nesigure sunt lăsate afară);
4. clasa „stâlp” etichetată automat cu YOLOE, ca modelul să nu mai confunde stâlpii cu trunchiurile;
5. epoca e aleasă pe **via-țintă** (cele 20 de poze B7 marcate de voi + o înregistrare Vairão), nu pe label map.

**Teste finale (niciodată la antrenare):** Vairão 170318 + 170442, testul label map (robot), EscaYard **B9**,
frunziș fără trunchi (alarme false). La final: tabel **v2 vs v3**.

## Ce faceți voi (totul la început)
1. **Runtime → Change runtime type → T4 GPU → Save**, apoi **Runtime → Run all**.
2. Celula 3: „Connect to Google Drive”.
3. Celula 4: dacă `model_trunchi_v2.pt` nu e în Drive, alegeți-l de pe laptop (`models/`).
4. Celula 5: lipiți cheia API Roboflow.

Durată estimată (nemăsurată pe acest set): **~2–3 ore**. Rezultatul: `MyDrive/vie_hackathon/trunk_v3_runs/model_trunchi_v3.pt`.
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
R = f"{OUT}/trunk_v3_runs"
os.makedirs(R, exist_ok=True)
os.environ["TRUNK_ROOT"] = "/content/tv"
os.environ["TRUNK_RUNS"] = R
for d in ("data/escayard/photos", "data/mildew", "data/vairao", "models"):
    os.makedirs(f"/content/tv/{d}", exist_ok=True)
"""),
    md("## 4. Modelul v2 (punctul de plecare și profesorul pentru etichetele automate)"),
    code("""
import shutil
from google.colab import files

if os.path.exists(f"{OUT}/model_trunchi_v2.pt"):
    shutil.copy(f"{OUT}/model_trunchi_v2.pt", "/content/tv/models/model_trunchi_v2.pt")
while not os.path.exists("/content/tv/models/model_trunchi_v2.pt"):
    print("model_trunchi_v2.pt is not in Drive -> choose it from the laptop (folder models/)")
    for name, data in files.upload().items():
        if "model_trunchi_v2" in name:
            open("/content/tv/models/model_trunchi_v2.pt", "wb").write(data)
            shutil.copy("/content/tv/models/model_trunchi_v2.pt", f"{OUT}/model_trunchi_v2.pt")
print("OK: model_trunchi_v2.pt")
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
    md("## 6. Celelalte dataseturi, toate deodată (~10–20 min): HUMAIN, Vairão, EscaYard, mană"),
    code("""
%%bash
cd /content
dl() {  # dl <name> <command>: runs in the background, writes dl_<name>.log ending with OK or FAILED
  ( bash -c "$2" > dl_$1.log 2>&1 && echo OK >> dl_$1.log || echo FAILED >> dl_$1.log ) &
}
# -o = overwrite without asking (a second run would otherwise wait for an answer and fail); --retry = retry on network errors
dl humain 'curl -sSfL --retry 3 -o humain.zip "https://github.com/humain-lab/vine-trunk/raw/master/Humain-Lab-vine-trunk-dataset.zip" && unzip -q -o humain.zip -d /content/tv/data/humain'
dl vairao 'curl -sSfL --retry 3 -o vairao.zip "https://zenodo.org/api/records/18152298/files/Dataset_Vairao_Grapevine_Trunks.zip/content" && unzip -q -o vairao.zip -d /content/tv/data/vairao'
dl escayard 'curl -sSfL --retry 3 -o /content/tv/data/escayard/datasheet.csv "https://zenodo.org/api/records/10362568/files/datasheet.csv/content" && curl -sSfL --retry 3 -o escayard.7z "https://zenodo.org/api/records/10362568/files/PLANTs_SmartPhonePhotos_JPG.7z/content" && python3 -c "import py7zr; py7zr.SevenZipFile(\\"escayard.7z\\").extractall(\\"/content/tv/data/escayard/photos\\")"'
dl mildew 'cd /content/tv/data/mildew && curl -sSfL --retry 3 -A "Mozilla/5.0" -o m.zip "https://data.mendeley.com/public-files/datasets/yxf2yv5ymt/files/baa9a436-bd95-425a-8d6f-aca7152cb1b0/file_downloaded" && unzip -q -o m.zip'
wait
for n in humain vairao escayard mildew; do
  echo "$n: $(tail -1 dl_$n.log)"
  if [ "$(tail -1 dl_$n.log)" != OK ]; then echo "   error:"; tail -4 dl_$n.log | head -3 | sed 's/^/   /'; fi
done
"""),
    code("""
import glob
counts = {
    "HUMAIN train photos": len(glob.glob("/content/tv/data/humain/git/train/Original_photos/*.jpg")),
    "Vairão frames": len(glob.glob("/content/tv/data/vairao/**/*.png", recursive=True)),
    "EscaYard photos": len(glob.glob("/content/tv/data/escayard/photos/**/*.jpg", recursive=True)),
    "mildew photos": len(glob.glob("/content/tv/data/mildew/downy_mildew_images/*.jpg")),
}
print(counts)
assert all(counts.values()), "a download failed: run the cell above again"
"""),
    md("## 7. Scriptul (același ca în proiect: `tools/trunk_v3.py`) + cele 20 de poze B7 marcate de voi"),
    code("%%writefile /content/trunk_v3.py\n" + SCRIPT),
    code("%%writefile /content/tv/data/manual_trunks_b7.json\n" + MANUAL),
    md("## 8. Setul de date (~20–30 min: etichete automate B7 + stâlpi)"),
    code("!cd /content && python trunk_v3.py build"),
    md("## 9. Antrenare (~1–2 ore, progres pe epoci; fiecare epocă se salvează în Drive)"),
    code("!cd /content && python trunk_v3.py train --name v3 --epochs 30 --batch 8"),
    md("""
### Continuare (doar dacă sesiunea s-a închis în timpul antrenării)
Rulați din nou celulele 1–8, apoi scoateți `#` din linia de mai jos: antrenarea continuă de unde a rămas.
"""),
    code("""
# !cd /content && python -c "from ultralytics import YOLO; YOLO('{R}/v3/weights/last.pt').train(resume=True)"
"""),
    md("## 10. Alegem cea mai bună epocă pe via-țintă (~10 min)"),
    code("!cd /content && python trunk_v3.py select --run v3"),
    md("## 11. Testul final: v2 vs v3 pe poze nefolosite niciodată la antrenare (~10 min)"),
    code("""
choice = json.load(open(f"{R}/v3_choice.json"))
print(choice)
conf = choice["conf"]
!cd /content && python trunk_v3.py report --weights "{R}/model_trunchi_v3.pt" --conf {conf}
"""),
    md("""
## 12. Pe laptop
1. Descărcați din Drive `trunk_v3_runs/model_trunchi_v3.pt` în `models/`.
2. Spuneți-i lui Claude pragul ales (`v3_choice.json`) și tabelul de mai sus: îl verificăm și pe laptop
   (`python -m tools.evaluate_boxes` pe Vairão și `python -m tools.evaluate_escayard --vineyards B9`) și, dacă e mai bun,
   îl punem în `vineyard/config.py` în locul lui v2.
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Written:", OUT)
