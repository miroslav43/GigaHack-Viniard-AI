# RUNBOOK Linux + CUDA: antrenare și inferență `fte` (Sireț3, GigaHack 2026)

> Pentru cine rulează pe laptopul Linux (om sau Claude Code). Execută pașii **în ordine**. Fiecare pas are un
> time-box. Dacă un pas se blochează mai mult de 15 min, **treci la următorul** și notează problema în
> `work/reports/linux_notes.md`.
>
> **Termen: rezultatele (`fte_results.tar`) trebuie să fie înapoi pe Mac până la 17:00** (sâmbătă, ora Chișinăului).
> Pe Mac urmează integrarea în pre-adnotările Marcaj și gate-ul GO/NO-GO de la 18:00.

## 0. Context în 60 de secunde

- **Sarcina:** proiectul adnotează ortomozaicul Sireț3: 311 tile-uri GeoTIFF de 2048×2048 px, la 2.5 cm/px, în EPSG:32635.
- **Etichetele:**
  - `vineyard`: un poligon per viță;
  - `waste`: bbox;
  - `row`: axele rândurilor;
  - `interrow_area` cu atributul `interrow_cover`.
- **Pipeline-ul clasic** e în `src/AI` (pachetul `vineyard`). **Nu îl modifica.**
- **Aici** (`src/AI_finetune_dataset_externe`, pachetul `fte`) antrenăm rețele pe dataset-uri externe plus pseudo-etichete Sireț3. Licențele sunt în `DATASETS.md`.
- **Ce produce laptopul Linux, în ordinea priorității:**
  1. **Waste (P1):** detector de deșeuri, `smp.Unet(resnet34)` cu 1 cap, apoi cutii din componente conexe. Aduce cel mai mult scor: waste F1 plus acoperirea traseului.
  2. **Modelul ICAERUS YOLOv9 zero-shot (P2):** singurul model public antrenat pe un poligon per viță (CC BY 4.0). Îl evaluăm pe cele 2 tile-uri exemplu și vedem dacă ajută la separarea plantelor.
  3. **Canopy v2 (P3):** U-Net-ul nostru cu 3 capete (c0 viță, c1 contact între plante, c2 vegetație de sol), continuat de la checkpoint-ul `v1b` antrenat pe Mac.
  4. **Întoarcerea rezultatelor pe Mac (P4).**

## 1. Transfer și dezarhivare

Fișierele vin de pe Mac, din `src/AI_finetune_dataset_externe/work/bundle/`:
- `fte_bundle.tar` (~1.4 GB): cod `vineyard` + `fte`, tile-uri, cache, run-urile `complete-v4` și de referință, date externe preprocesate, checkpoint-ul `v1b`, encoder-ul resnet34 ImageNet. Are `.sha256`.
- `data_info.tar` (769 MB, opțional): folderul `data & info` complet.

```bash
cd ~
sha256sum fte_bundle.tar && cat fte_bundle.tar.sha256     # cele două hash-uri trebuie să fie identice
tar -xf fte_bundle.tar -C ~/                              # creează ~/Vin Gigahack/...
tar -xf data_info.tar -C ~/                               # opțional
cd ~/"Vin Gigahack"/src/AI_finetune_dataset_externe
```

Căile conțin spații („Vin Gigahack”, „data & info”), deci pune-le **mereu între ghilimele**.

## 2. Setup (time-box 20 min)

```bash
nvidia-smi                                  # notează: GPU, VRAM, driver (intră în README)
python3 --version                           # vineyard cere >= 3.12 (folosește sintaxa PEP 695)
bash scripts/setup_linux_cuda.sh
```

**Ce face scriptul:**
- Dacă `python3` e ≥ 3.12 **și** are torch cu CUDA: creează `.venv-cuda` cu `--system-site-packages` (refolosește torch-ul) și instalează `scripts/requirements-cuda.txt`.
- Altfel: instalează `uv`, creează un env 3.12 și instalează torch CUDA (`CUDA_INDEX=cu126` implicit; pentru drivere noi: `CUDA_INDEX=cu128 bash scripts/setup_linux_cuda.sh`).
- La final trebuie să afișeze `torch … cuda True <nume GPU>` și `resnet34 imagenet OK; vineyard + fte import OK`.

**Probleme cunoscute:**
- **torch < 2.3 cu numpy 2:** apar erori `_ARRAY_API`. Rulează `.venv-cuda/bin/pip install "numpy<2"`, sau refă env-ul cu `uv` (`PYTHON=/nu/exista bash scripts/setup_linux_cuda.sh` forțează ramura `uv`).
- **rasterio/pyogrio fără wheel:** actualizează pip (`.venv-cuda/bin/pip install -U pip`) și reîncearcă; wheel-urile includ GDAL.
- **Env pentru toate comenzile manuale de mai jos:**
  ```bash
  export PYTHONPATH="$HOME/Vin Gigahack/src/AI:$HOME/Vin Gigahack/src/AI_finetune_dataset_externe"
  PY=.venv-cuda/bin/python
  ```

**Verificare rapidă** (30 s; ambele comenzi trebuie să treacă):
```bash
$PY -m pytest tests/data tests/nn -q
$PY -c "from fte.canopy.validate import ExampleValidator; print(ExampleValidator().baseline()['mean'])"
```

## 3. P1: Waste (time-box total ~55 min), pornește-l PRIMUL

```bash
bash scripts/run_laptop.sh waste 2>&1 | tee work/logs/p1_waste.log
```

**Ce face:**
1. **Antrenare:** `fte.waste.train` pe CUDA, până la 40 de epoci × 4.000 de patch-uri 384², batch 12, time-box 40 min, patience 6.
   - Date: DroneWaste (clase „item” → waste; grămezi de sol/moloz/lemn → fundal), UAVVaste, copy-paste pe ferestre Sireț3, negative dificile Sireț3 (tuburi și pari de viță, sol deschis).
   - Validare la fiecare epocă:
     - box F1@IoU0.3 pe site-urile de validare DroneWaste (site4, site7, site17);
     - recall pe validarea sintetică Sireț3;
     - numărul de cutii pe cele 2 exemple, care trebuie să ajungă la 0.
   - Log: `work/logs/waste_train.jsonl`. Greutăți: `models/fte-waste/v1/{best.pt,last.pt,model_card.json}`.
2. **Inferență pe toate cele 311 tile-uri:** `work/preds/waste/v1/{heat/*.png,boxes.parquet,infer.json}`.
3. **`select`:** alege pragul (recall sintetic ≥ 0.8, 0 cutii pe exemple, ≤ 5 cutii per tile, ≤ 400 în total) și scrie `selected.parquet` și `select_report.json`.
4. **`export`:** scrie `waste_fte.csv` (formatul `src/AI/configs/waste_confirmed.csv`), iar pentru review manual în Marcaj `review/review_boxes.csv` și contact sheet-urile `review/sheet_*.jpg`.

**Reglaje:**
- **CUDA OOM:** `BATCH=8 bash scripts/run_laptop.sh waste`.
- **Loader lent:** `WORKERS=4 …`.
- **Mai mult timp de antrenare:** `TBOX=50 …`, dar numai dacă a pornit înainte de 15:45.

**Criteriu de reușită:** F1 pe validarea DroneWaste raportat și crescător, iar pe exemple 0 cutii la pragul ales. Notează în `work/reports/linux_notes.md` best F1, epoca și numărul de cutii selectate.

## 4. P2: Modelul ICAERUS YOLOv9 zero-shot (time-box ~45 min, poate rula în paralel cu P1)

Modelul e „Vine segmentation model”, doi:10.5281/zenodo.14610756, licență **CC BY 4.0**, un singur fișier `best.pt` de 55.7 MB. E un YOLOv9 **gelan-c-seg** (WongKinYiu/yolov9), antrenat cu `--img 640` pe crop-uri de ~502 px la 1.4 cm/px (vițe tinere, Portugalia).

Codul yolov9 e **GPL-3.0**: clonează-l **în afara repo-ului** (`~/yolov9`) și nu copia nimic din el în repo.

### 4.1 Setup
```bash
cd ~ && git clone --depth 1 https://github.com/WongKinYiu/yolov9.git
wget -O ~/yolov9/icaerus_vine_best.pt "https://zenodo.org/api/records/14610756/files/best.pt/content"
sha256sum ~/yolov9/icaerus_vine_best.pt                 # notează hash-ul în linux_notes.md
# NU instala requirements.txt din yolov9 (ar aduce opencv-python și ar putea schimba torch/numpy)
~/"Vin Gigahack"/src/AI_finetune_dataset_externe/.venv-cuda/bin/pip install tqdm seaborn thop gitpython pillow psutil requests
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1                # torch >= 2.6: checkpoint-ul e un pickle complet
```
- **Eroare `np.int` / `np.float`:** înlocuiește în `~/yolov9` cu `int` / `float`.
- **Eroare la `torch.load`:** adaugă `weights_only=False` în `models/experimental.py` → `attempt_load`.

### 4.2 Inferență pe cele 2 tile-uri exemplu (`siret3_r006_c004`, `siret3_r021_c012`)

Scrie modulul `fte/icaerus_yolo/` (fișiere mici, < 200 linii, fără cod copiat din yolov9):
- **`windows.py`:**
  - citește tile-ul RGB cu `vineyard.geo.raster.read_tile(<src/AI/work/tiles>/<tile>.tif)` (vezi `fte/paths.py` pentru `AI_TILES`);
  - îl mărește cu factorul `f` (INTER_CUBIC):
    - `f = 0.025/0.014 = 1.786`: GSD-ul de captură al datelor de antrenare;
    - `f = 2.27`: GSD-ul efectiv după letterbox-ul 502→640 din antrenare;
    - încearcă **ambele**.
  - taie ferestre de 640×640 cu stride 512 și le salvează PNG (RGB → BGR pentru cv2), cu offset-urile într-un JSON.
- **Rulează predicția** oficială yolov9 pe folderul de ferestre:
  ```bash
  cd ~/yolov9 && $PY segment/predict.py --weights icaerus_vine_best.pt --source <dir_ferestre> --img 640 \
      --conf 0.25 --iou 0.5 --device 0 --save-txt --nosave --project <out> --name f1786 --exist-ok
  ```
  Etichetele ies în `<out>/f1786/labels/<fereastra>.txt`, format YOLO-seg normalizat `cls x1 y1 x2 y2 …`. Pentru parsare refolosește `fte/convert/yolo_seg.py`.
- **`merge.py`:**
  - duce poligoanele în px de tile: `(x_win + x0) / f`;
  - păstrează un poligon doar dacă centroidul lui cade în zona centrală a ferestrei (fără banda de overlap de 64 px; excepție la marginile imaginii);
  - convertește în UTM cu `vineyard.geo.tiling.tile_ref(tile_id)` + `px_to_utm(t, uv)`;
  - taie la tile, repară cu `make_valid`, aruncă tot ce are < 0.05 m²;
  - scrie un GeoDataFrame în EPSG:32635 cu `tile_id`.
- **`evaluate.py`:** refolosește `fte/canopy/partition_eval.py`:
  - `load_examples()` și `score_canopies(data, canopies, label)`, care dau metrica oficială;
  - identitatea pe `complete-v4` reproduce 0.8941 / 0.8455, adică media 0.8698.

**Variante de evaluat pentru fiecare `f`** (raportează per tile: IoU, F1, score și n_pred):
- **(A) YOLO brut:** poligoanele YOLO ca `canopies` (păstrează coloanele și tipurile din `data.base.canopies`; `vineyard_id` și `row_id` vin din canopy-ul bazei cu cea mai mare suprapunere, altfel din primul rând).
- **(B) YOLO ∩ bază:** fiecare poligon YOLO intersectat cu uniunea canopy-urilor bazei. Păstrează precizia clasică la margini.
- **(C) Împărțire după YOLO (uniunea neschimbată):**
  - fiecare poligon al bazei care se suprapune cu ≥ 2 instanțe YOLO (fiecare suprapunere ≥ 0.1 m²) se împarte între ele;
  - împărțirea: rasterizezi poligonul la 2.5 cm, atribui fiecare pixel celei mai apropiate instanțe (distance transform / etichetare pe cea mai apropiată) și vectorizezi;
  - uniunea trebuie să rămână identică: Δarie < 0.02 m² per tile;
  - piesele < 0.19 m² se lipesc la vecinul cel mai mare.

Scrie rezultatele în `work/reports/icaerus_yolo_eval.{json,md}` și un tabel în `linux_notes.md`.

**Gate:** o variantă e utilă doar dacă media trece de **0.8826** (cel mai bun rezultat clasic partition+merge pe Mac) și niciun tile nu scade sub baza lui − 0.002. Dacă e utilă, rulează aceeași variantă pe toate tile-urile cu vie: lista e în `fte/canopy/infer.py` → `--tiles vineyard` (~108 tile-uri). Scrie în `work/preds/icaerus_yolo/<f>/<tile>.parquet`.

## 5. P3: Canopy v2 (time-box ~35 min), după P1 (GPU liber)

```bash
bash scripts/run_laptop.sh canopy 2>&1 | tee work/logs/p3_canopy.log
```

**Ce face:**
1. **Antrenare:** `fte.canopy.train --device cuda --amp --init-weights models/fte-canopy/v1b/last.pt`: până la 60 de epoci × 1.000 de patch-uri 512², batch 8, lr 2e-4 (encoder 7e-5), time-box 25 min, patience 8.
   - Validare la fiecare epocă:
     - metrica oficială pe cele 2 exemple, cu fuziunile B și C;
     - F1 pe instanțele din validarea ICAERUS.
   - Log: `work/logs/canopy_train.jsonl`. Greutăți: `models/fte-canopy/v2/{best_mask,best_contact,last}.pt` + `model_card.json`.
   - Referință `v1b` (Mac, ~11 epoci): B 0.739, F1 ICAERUS 0.46.
2. **Inferență TTA×4** cu `best_contact` și `best_mask` pe tile-urile cu vie: `work/preds/canopy/v2_{best_contact,best_mask}/{c0,c1,c2}/<tile>.png`.
3. **`partition_eval`** cu hărțile c1: modurile `nn` și `width+nn` față de bază → `work/reports/partition_grid.{json,md}`.

**Reglaje:**
- **OOM:** `BATCH=6 …`.
- **Pornire de la zero, fără warm start:** `INIT=none …`.
- **Mai puțin timp:** `TBOX=15 …`.

## 6. P4: Pachetul de întoarcere (termen 17:00)

```bash
cd ~/"Vin Gigahack"/src/AI_finetune_dataset_externe
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > work/reports/linux_gpu.csv
bash scripts/run_laptop.sh pack          # -> work/bundle/fte_results.tar
sha256sum work/bundle/fte_results.tar > work/bundle/fte_results.tar.sha256
```

Dacă nu termini toate prioritățile, **împachetează ce ai la 16:45** și trimite. P1 este cel mai important.

Pachetul conține:
- `models/fte-waste/v1`, `models/fte-canopy/v2`: greutăți și model cards;
- `work/preds/waste/v1`: heatmaps, `boxes.parquet`, `selected.parquet`, `waste_fte.csv`, `review/`;
- `work/preds/canopy/v2_*`: hărțile c0/c1/c2;
- `work/preds/icaerus_yolo`, dacă există;
- `work/reports`: rapoarte, `linux_notes.md`, `linux_gpu.csv`;
- `work/logs`.

Pe Mac se dezarhivează în `src/AI_finetune_dataset_externe/` cu `tar -xf fte_results.tar` și urmează `fte.integrate`.

## 7. Reguli

- **Nu modifica `src/AI`.** Nu scrie în `src/AI/work/runs/complete-v4`. Nu face commit sau push.
- **Cod nou doar în `src/AI_finetune_dataset_externe/fte/…`**, cu fișiere mici, type hints, `logging`, fără mutarea input-urilor și teste pytest scurte pentru logica pură. yolov9 rămâne în `~/yolov9`.
- **Pentru README:** notează timpii măsurați (train s/epocă, infer s/tile) și hardware-ul în `work/reports/linux_notes.md`.
- **Nu șterge date.** Dacă se umple discul, șterge doar `work/preds/*/heat` vechi sau checkpoint-uri `smoke`.
- **Licențe:** ICAERUS data CC BY-NC 4.0 (canopy v2 e deci NC); greutățile YOLOv9 ICAERUS CC BY 4.0; DroneWaste și UAVVaste CC BY 4.0. Detalii în `DATASETS.md`.

## 8. Cronologie orientativă

| Ora | GPU | În paralel |
|---|---|---|
| până la 15:40 | transfer + setup (§1–§2) | clone yolov9, `best.pt` (§4.1) |
| 15:40–16:35 | **P1 waste** (train 40 min + infer ~5 min + select/export) | P2: ferestre + predicție YOLO pe 2 tile-uri (GPU mic), evaluare |
| 16:35–16:50 | **P3 canopy v2** (`TBOX=10` dacă e deja târziu) + inferență | P2: dacă trece gate-ul, YOLO pe tile-urile cu vie |
| 16:50–17:00 | **P4 pack**, transfer pe Mac | — |

Dacă setup-ul se termină după 15:50, sari peste P3 (canopy v2): P1 și P2 contează mai mult.
