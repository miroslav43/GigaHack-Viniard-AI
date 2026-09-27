"""Generates colab/bunch_trunk.ipynb: ONE model for bunches + trunks (YOLO26m, 960), trained on 8 vineyard datasets,
chosen on held-out tests and compared with the current models (model_ciorchine_v2 + model_trunchi_v2).
Embeds tools/trunk_adapt.py, build_bt.py, eval_bt.py, certh_sub.py, colab_setup_bt.sh and data/manual_trunks_b7.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
OUT = HERE.parent / "bunch_trunk.ipynb"


def read(rel):
    return (PROJ / rel).read_text(encoding="utf-8").rstrip("\n") + "\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def writefile(path, rel):
    return code(f"%%writefile {path}\n" + read(rel))


cells = [
    md("""
# Un singur model pentru ciorchini + trunchiuri (YOLO26m)

**Scop:** robotul să găsească mai mulți ciorchini și trunchiuri, și în vii pe care nu le-a văzut.

**Cum:** antrenăm un model cu 2 clase (`ciorchine`, `trunchi`) pe **8 dataseturi** din 6 țări:
label map (robot, Portugalia), WGISD (Brazilia), Pinheiro (Portugalia), wGrapeUNIPD (Italia), CERTH (Grecia),
HUMAIN-Lab (Grecia), EscaYard B7 (Spania), mană (frunziș fără trunchi).
Fiecare dataset are etichete doar pentru o clasă; clasa lipsă o completează modelele actuale, doar unde sunt foarte sigure (≥ 60%).

**Test cinstit** (poze nefolosite la antrenare): label map test (robot), WGISD test, Pinheiro test, Vairão (vie nouă),
EscaYard B9 (vie nouă). La final: tabel **vechi vs nou** și cel mai bun model salvat în Drive.

## Ce faceți voi (totul la început)
1. **Runtime → Change runtime type → T4 GPU → Save**, apoi **Runtime → Run all**.
2. Celula 2: **Connect to Google Drive**.
3. Celula 3: dacă modelele actuale nu sunt în Drive (`MyDrive/vie_hackathon/`), alegeți-le de pe laptop din `models/`:
   `model_ciorchine_v2.pt` și `model_trunchi_v2.pt`.
4. Celula 4: lipiți cheia API Roboflow și apăsați Enter.

**Durată:** descărcări ~15–25 min, set de date ~20 min, antrenare ~2–3 ore (progres pe epoci, cu procente), alegere ~20 min.
Totul se salvează în `MyDrive/vie_hackathon/bt_runs/`. Dacă sesiunea se închide, vedeți celula „Continuare”.
"""),
    md("## 1. Placa video"),
    code("!nvidia-smi -L"),
    md("## 2. Google Drive"),
    code("""
import os
from google.colab import drive

drive.mount("/content/drive")
OUT = "/content/drive/MyDrive/vie_hackathon"
os.makedirs(OUT, exist_ok=True)
"""),
    md("## 3. Modelele actuale (profesori pentru clasa lipsă + comparație)"),
    code("""
from google.colab import files

for name in ("model_ciorchine_v2.pt", "model_trunchi_v2.pt"):
    while not os.path.exists(f"{OUT}/{name}"):
        print(f"{name} is not in Drive -> choose it from the laptop (folder models/)")
        for up, data in files.upload().items():
            for want in ("model_ciorchine_v2", "model_trunchi_v2"):
                if want in up:
                    open(f"{OUT}/{want}.pt", "wb").write(data)
    print("OK:", name, round(os.path.getsize(f"{OUT}/{name}") / 1e6), "MB")
"""),
    md("## 4. label map v8 (Roboflow) — lipiți cheia API"),
    code("""
import json, urllib.request, zipfile
from getpass import getpass

key = getpass("Roboflow API key: ")
info = json.load(urllib.request.urlopen(f"https://api.roboflow.com/grape-detection/label-map/8/yolov8?api_key={key}"))
urllib.request.urlretrieve(info["export"]["link"], "/content/labelmap.zip")
zipfile.ZipFile("/content/labelmap.zip").extractall("/content/tv/data/labelmap")
print({s: len(os.listdir(f"/content/tv/data/labelmap/{s}/images")) for s in ("train", "valid", "test")})
"""),
    md("## 5. Scripturile (aceleași ca în proiect, folderul `tools/`)"),
    code("!mkdir -p /content/bt"),
    writefile("/content/trunk_adapt.py", "tools/trunk_adapt.py"),
    writefile("/content/bt/certh_sub.py", "tools/certh_sub.py"),
    writefile("/content/bt/build_bt.py", "tools/build_bt.py"),
    writefile("/content/bt/eval_bt.py", "tools/eval_bt.py"),
    writefile("/content/bt/setup.sh", "tools/colab_setup_bt.sh"),
    writefile("/content/manual.json", "data/manual_trunks_b7.json"),
    md("## 6. Descărcarea dataseturilor (toate deodată, ~15–25 min)"),
    code("""
import glob, subprocess, time

subprocess.run(["bash", "/content/bt/setup.sh"], check=True)
names = ["pip", "wgisd", "pinheiro", "wgrape", "vairao", "humain", "escayard", "mildew", "certh"]
t0 = time.time()
while True:
    logs = {n: open(f"/content/bt/dl_{n}.log").read() if os.path.exists(f"/content/bt/dl_{n}.log") else "" for n in names}
    done = [n for n in names if "END" in logs[n]]
    print(f"{time.time() - t0:5.0f}s  finished {len(done)}/{len(names)} ({len(done) / len(names):.0%})  waiting: "
          + ", ".join(n for n in names if n not in done), flush=True)
    if len(done) == len(names):
        break
    time.sleep(30)
failed = [n for n in names if "_OK" not in logs[n]]
print("FAILED:", failed if failed else "none")
for n in failed:
    print(f"--- {n}\\n" + logs[n][-800:])
"""),
    md("## 7. Setul combinat (~20 min: modelele actuale completează clasa lipsă)"),
    code("!cd /content/bt && python build_bt.py"),
    md("## 8. Testele cinstite + scorul modelelor actuale (~10 min)"),
    code("""
import sys
sys.path.insert(0, "/content/bt")
import eval_bt as E

E.prepare()
old = E.report(f"{OUT}/model_ciorchine_v2.pt", f"{OUT}/model_trunchi_v2.pt", imgsz_es=1280)
json.dump(old, open(f"{OUT}/bt_old_scores.json", "w"), indent=1)
"""),
    md("## 9. Antrenare YOLO26m la 960 (~2–3 ore, progres pe epoci)"),
    code('''
import time
from ultralytics import YOLO

T0 = time.time()
RUN = f"{OUT}/bt_runs/bt_m960"

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

model = YOLO("yolo26m.pt")  # pre-trained on COCO, only fine-tuning
model.add_callback("on_fit_epoch_end", print_epoch)
model.train(data="/content/bt/ds/data.yaml", imgsz=960, epochs=30, patience=10, batch=8, workers=4,
            save_period=2, plots=False, project=f"{OUT}/bt_runs", name="bt_m960", exist_ok=True)
'''),
    md("""
### Continuare (doar dacă sesiunea s-a închis în timpul antrenării)
Rulați din nou celulele 1–8 (ca să existe datele), apoi scoateți `#` din linia de mai jos: antrenarea continuă de unde a rămas.
"""),
    code("""
# m = YOLO(f"{OUT}/bt_runs/bt_m960/weights/last.pt"); m.add_callback("on_fit_epoch_end", print_epoch); m.train(resume=True)
"""),
    md("""
## 10. Alegem cea mai bună epocă pe testele cinstite (~20 min)
Scor = media dintre AP50 (robot ciorchini, robot trunchiuri, WGISD, Pinheiro, Vairão) și procentul de poze B9 cu trunchi găsit.
Cel mai bun model se salvează ca `MyDrive/vie_hackathon/model_bunch_trunk.pt`.
"""),
    code("""
import shutil

W = f"{OUT}/bt_runs/bt_m960/weights"
epochs = sorted(glob.glob(f"{W}/epoch*.pt"), key=lambda p: int(p.split("epoch")[-1][:-3]))
candidates = [f"{W}/best.pt", f"{W}/last.pt"] + epochs[len(epochs) // 2:]
results = {}
for i, c in enumerate(candidates, 1):
    if os.path.exists(c):
        print(f"\\n=== {i}/{len(candidates)} ({i / len(candidates):.0%}): {os.path.basename(c)}", flush=True)
        results[os.path.basename(c)] = E.report(c, c, imgsz_b=960, imgsz_t=960)
best = max(results, key=lambda k: E.score(results[k]))

keys = list(old)
print(f"\\n{'':42s}{'old models':>12s}{'new ' + best:>18s}")
for k in keys:
    print(f"{k:42s}{old[k]:12.3f}{results[best][k]:18.3f}")
print(f"{'score (higher = better)':42s}{E.score(old):12.3f}{E.score(results[best]):18.3f}")
shutil.copy(f"{W}/{best}", f"{OUT}/model_bunch_trunk.pt")
json.dump(results, open(f"{OUT}/bt_new_scores.json", "w"), indent=1)
print("\\nSaved:", f"{OUT}/model_bunch_trunk.pt", "(from", best + ")")
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Written:", OUT)
