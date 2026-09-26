# Sireț3 Vineyard AI: arhitectura completă și planul de lucru

Versiunea 1.1 · vineri 25.09.2026, seara · bazat pe regulile v1.0, criteriile de scor și analiza măsurată a celor 2 tile-uri exemplu (`analiza_exemple/RAPORT_exemple_si_reteta.md`). Actualizată după cercetarea din `research/CERCETARE_Siret3.md`; schimbările sunt rezumate în §11.

Ore în fusul Chișinău. **Gate-ul critic: Publish în Marcaj sâmbătă ~12:00.** Deadline final: duminică 15:00.

---

## 0. Rezumat executiv

### Arhitectura într-o imagine

```
                ┌──────────────────────── ETAPA A: PRE-ADNOTARE (până sâmbătă 12:00) ────────────────────────┐
311 tile-uri ─► [1 Index tile-uri + georef] ─► [2 Mască vegetație a* (+ NN opțional)] ─► [3 Axe rânduri per tile]
 (GeoTIFF)                                                                                   │
                                                                                            ▼
            [6 Canopy = mască ∩ coridor ±0,30 m] ◄── [5 Blocuri + vineyard_id/row_id] ◄── [4 Legare rânduri între tile-uri (UTM)]
                  │                                         │
                  ▼                                         ▼
            [7 Inter-rânduri + cover]   [8 row_structure]   [9 Waste (candidați + filtru NN)]
                  └──────────────┬──────────────────────────┘
                                 ▼
             [10 QA vizual 311 tile-uri + fișier de override-uri] ─► [11 Export CVAT 1.1 → 7 ZIP-uri] ─► Marcaj: upload, 311 fișiere, PUBLISH
                └──────────────────────────────────────────────────────────────────────────────────────────────────┘

                ┌──────────────────────── ETAPA B: DUPĂ CORECTURĂ (duminică) ────────────────────────┐
Marcaj export (CVAT 1.1) ─► [12 Import → UTM, unire per row_id] ─► [13 Măsurători] ─► measurements.csv
                                         │                     └─► [14 Ținte de inspecție] ─► [15 Graf de mers + TSP] ─► route.geojson
                                         └─────────────────────────────────────────────────────► [16 Web UI (hartă, tabele, traseu)]
                └──────────────────────────────────────────────────────────────────────────────────────┘
```

### Cele 7 decizii care contează cel mai mult

1. **Totul se calculează în UTM (EPSG:32635) la nivel de bloc**, nu per tile. Tile-ul e doar unitatea de citire și de export. Așa ID-urile rămân aceleași peste margini din construcție.
2. **Canopy = mască vegetație (Lab a\*) ∩ coridor de ±0,30 m în jurul axei**, componente ≥ 0,19 m². E exact cum pare construită referința. Pe exemple, fără nicio intervenție umană, scorul e ~0,82 din 1 (0,855 cu axe perfecte).
3. **Axele sunt piesa centrală.** Din ele ies canopy-urile, inter-rândurile, row_structure, blocurile, țintele și traseul. Calitatea axelor se verifică vizual pe toate tile-urile **înainte** de Publish, pentru că după Publish nu mai putem regenera canopy-urile, ci doar le corectăm manual.
4. **Aceleași module de măsurare, ținte și traseu rulează pe exportul corectat din Marcaj.** Organizatorii calculează măsurătorile din Marcaj, deci și noi le calculăm de acolo.
5. **Traseul merge pe liniile mediane ale inter-rândurilor și pe scheletul pasajelor**, departe de marginile poligoanelor, ca să rămână sub 0,5% în afara zonei permise (pragul de eliminare e 2%).
6. **Rețeaua neuronală (obligatorie ca livrabil)** e un U-Net mic antrenat pe pseudo-etichete din pipeline-ul clasic, **pe tot tile-ul**, cu iarba din afara coridoarelor etichetată explicit „rest”. Rafinează masca de vegetație (elimină iarba, buruienile și coroanele de pomi). Intră în export doar dacă bate varianta clasică pe **ambele** exemple (ablația A–F din §4.6).
7. **Waste: precizie înainte de toate.** Pre-adnotăm automat doar candidații pe care îi confirmă două modele (linear probe + SAM 3/CLIP). Restul se confirmă de un om, pe o listă scurtă de candidați ordonată după scor, iar zonele fără candidați se caută manual în Marcaj.

### Strategia de scor (estimări realiste)

| Criteriu | Pondere | Mecanism | Țintă |
|---|---|---|---|
| Canopy | 25% | a\* ∩ coridor, NN opțional, fără canopy fără rând | 0,75–0,85 din criteriu |
| Waste | 10% | candidați + filtre + linear probe (+ SAM 3) + confirmare umană | 0,3–0,6 (incert: nu avem exemple) |
| Axe F1 | 8% | detector + legare + QA pre-publish | ≥ 0,9 |
| Atribute | 5% | reguli măsurate (gol ≥ 5 m, fracție vegetație) | ≥ 0,8 |
| Grupare vineyard_id | 2% | blocuri pe graful rândurilor, tăiate de pasaje | ≥ 0,9 |
| Numărători + măsurători | 10% | din exportul Marcaj, cu aceleași formule | 0,7–0,9 |
| Traseu | 25% | ținte = goluri + waste, graf pe inter-rânduri, TSP | acoperire ≥ 90%, 0% risc de eliminare |
| Inginerie | 15% | DAG clar, teste, Docker, benchmark, UI | 10–13 / 15 |

---

## 1. Maparea cerințelor pe componente

| Cerință (Solution scope) | Modul | Algoritm | Ieșire | Scor | Test de acceptanță |
|---|---|---|---|---|---|
| Canopies (`vineyard`, poligoane, plante separate) | `canopy/` | a\* (σ 2,5, prag 4) ∩ coridor ±0,30 m → componente ≥ 0,19 m² → simplificare | `canopies.gpkg` + CVAT | 25% | pe exemple: 0,6·IoU + 0,4·F1 ≥ 0,80 |
| Waste (`waste`, bbox) | `waste/` | candidați de culoare/formă → filtre geometrice și de periodicitate → linear probe CLIP/DINOv2 (+ SAM 3 pe crop) → auto-accept dublu sau confirmare umană | `waste.gpkg` + CVAT | 10% | 0 FP pe cele 2 exemple (care au 0 waste și sute de tuburi albe) |
| Row axes (`row`, `row_id`) | `rows/` | orientare + profil perpendicular + Huber fitLine per tile, apoi legare în UTM | `rows.gpkg` + CVAT | 8% | F1 axe ≥ 0,95 pe exemple (prototip: 0,98) |
| Inter-row areas (`interrow_area`) | `interrow/` | bandă între axa_i + 0,30 m și axa_{i+1} − 0,30 m, tăiată la rândul mai scurt, minus canopy, minus găuri | `interrows.gpkg` + CVAT | 2% arie + traseu | IoU ≥ 0,95 pe exemple (prototip: 0,955) |
| `row_structure` | `attributes/` | gol maxim pe axă (inclusiv capetele) ≥ 5 m → disrupted; umbră/iarbă → unassessable | atribut | parte din 5% | 51/51 corect pe exemple cu canopy de referință |
| `interrow_cover` | `attributes/` | fracția de vegetație: < 0,25 bare_soil, 0,25–0,75 mixed, > 0,75 vegetation; umbră → unassessable | atribut | parte din 5% | 49/49 corect pe exemple |
| Blocks (`vineyard_id`) | `blocks/` | componente conexe pe graful rândurilor (vecini ≤ 4 m), tăiate de pasaje | ID pe toate obiectele | 2% + 2% | fiecare bloc vizibil = un ID pe overview |
| Inspection locations | `targets/` | goluri ≥ 5 m pe rândul global + waste | `targets.geojson` | 15% (acoperire) | fiecare gol ≥ 5 m din exemple are o țintă |
| Counts & measurements | `measure/` | din export Marcaj: ID-uri distincte, lungimi, uniunea canopy, suma inter-rândurilor | `measurements.csv` | 10% | valorile pe exemple = calcul independent ± 0,1% |
| Annotation in Marcaj | `export/`, proces | CVAT 1.1 writer + validator + 7 ZIP-uri (≤ 85 000 000 B fiecare); corectură pe petice vecine | 7 ZIP-uri | condiție de admitere | round-trip exemple identic geometric și pe atribute; 311 fișiere în Marcaj |
| Walking route | `route/` | domeniu = inter-rânduri ∪ pasaje − interzis; graf pe linii mediane; Dijkstra + TSP generalizat (GTSP, câte un set de candidați per țintă) | `route.geojson` | 25% | în afara domeniului < 0,5%; start/final ≤ 5 m; 100% din țintele reachable acoperite |
| Web interface | `web/` | MapLibre static: fond XYZ (gdal2tiles), canopy MVT (tippecanoe), GeoJSON global pentru rest, tabele vanilla | site static + link | admitere + pitch | toate elementele cerute vizibile pe hartă |
| Submission | repo | README, Docker, lock, weights, timp procesare | repo | admitere + 15% | `make all` reproduce `route.geojson` și `measurements.csv` |
| Neural network | `nn/` | U-Net ResNet18 pe pseudo-etichete (+ opțional detector waste) | weights + script de antrenare | condiție din brief | ablație pe exemple în README |

---

## 2. Principii de arhitectură

1. **O singură sursă de adevăr geometrică:** straturile globale în UTM (GeoPackage). Tot restul (CVAT, UI, CSV) e derivat.
2. **Etape idempotente cu cache:** fiecare etapă scrie în `work/<etapă>/` și e rerulată doar dacă intrările sau parametrii s-au schimbat (hash de config + versiune de cod).
3. **Parametrii într-un singur YAML** (`configs/default.yaml`), cu valorile măsurate pe exemple. Nicio constantă magică în cod.
4. **Override-uri manuale înainte de Publish** (`configs/overrides.yaml`): rânduri de adăugat, șters sau prelungit și tile-uri forțat goale, în coordonate UTM. Se aplică deterministic înainte de export. Așa corecturile pre-publish sunt reproductibile, nu făcute de mână în XML.
5. **Același cod pentru „predicție” și „adevăr corectat”:** etapele 13–16 primesc la intrare fie straturile noastre, fie exportul din Marcaj.
6. **Imutabil:** funcțiile primesc geometrii și întorc geometrii noi. Nimic nu se modifică pe loc (regula de stil a echipei).

---

## 3. Contracte comune

### 3.1 Georeferențiere (verificată pe exemple)

- Tile-urile au `ModelPixelScale = (0,025; 0,025)` și `ModelTiepoint` = colțul stânga-sus.
- Grila e regulată: **X_stânga(c) = 628 992,0 + 51,2·c**, **Y_sus(r) = 5 221 222,4 − 51,2·r** (verificat pe `r006_c004` și `r021_c012`).
- Pixel → UTM: `X = X_stânga + px·0,025`, `Y = Y_sus − py·0,025`, cu coordonatele de pixel CVAT float în intervalul 0…2048 (convenția colțului, ca în exemplu).
- La export, se taie la cutia tile-ului [0, 2048] și se rotunjește la 0,1 px.
- Zona neagră (nodata) de pe tile-urile de margine se exclude din măsti (`R+G+B == 0`).

### 3.2 Straturi globale (`work/layers.gpkg`, EPSG:32635)

| Strat | Geometrie | Câmpuri |
|---|---|---|
| `tiles` | Polygon | `tile` (nume fișier), `r`, `c`, `part` (1–7, partea de upload; vezi §4.13), `has_nodata`, `is_empty` |
| `rows` | LineString | `row_id`, `vineyard_id`, `length_m`, `n_tiles`, `confidence`, `source` (model/override/marcaj) |
| `row_pieces` | LineString | `row_id`, `vineyard_id`, `tile`, `row_structure`, `max_gap_m` |
| `canopies` | Polygon | `canopy_id`, `vineyard_id`, `row_id`, `tile`, `area_m2`, `source` |
| `interrows` | Polygon | `interrow_id`, `vineyard_id`, `row_left`, `row_right`, `tile`, `interrow_cover`, `veg_frac`, `shadow_frac`, `area_m2` |
| `blocks` | Polygon | `vineyard_id`, `n_rows`, `area_m2`, `mean_spacing_m`, `azimuth_deg` |
| `waste` | Polygon (bbox) | `waste_id`, `vineyard_id` (sau gol), `tile`, `score`, `source` |
| `targets` | Point | `target_id` (T001…), `type` (gap/missing/sparse/waste), `vineyard_id`, `row_id`, `gap_length_m`, `reachable` |
| `domain` | MultiPolygon | `kind` (interrow/passage/connector) |
| `route` | LineString | `length_m`, `n_targets`, `n_visited`, `outside_share` |

### 3.3 ID-uri

- `vineyard_id`: `V01`, `V02`… numerotate de la nord-vest spre sud-est după centroid (stabil între rulări).
- `row_id`: `V03-R001`… în ordinea offset-ului perpendicular în bloc, cu 3 cifre (blocurile mari pot trece de 99 de rânduri).
- Rândurile adăugate manual în Marcaj primesc numere de la `R900` în sus, din registrul echipei, ca să nu se ciocnească.
- `target_id`: `T001`…, iar waste: `W001`….

### 3.4 DAG-ul etapelor

| # | Etapă | Nivel | Intrare | Ieșire |
|---|---|---|---|---|
| 1 | `index` | global | tile-uri | `tiles`, grafului vecinilor |
| 2 | `vegmask` | per tile (paralel) | tile | `work/vegmask/<tile>.png` (1 bit) + mască umbră |
| 2b | `nn_infer` (opțional) | per tile | tile | mască rafinată |
| 3 | `rows_tile` | per tile | mască | segmente de axă per tile + scor |
| 4 | `rows_link` | global | segmente | `rows` globale (lanțuri) |
| 5 | `blocks` | global | `rows`, pasaje | `blocks`, `vineyard_id`, `row_id` |
| 6 | `canopy` | per tile | mască, `row_pieces` | `canopies` |
| 7 | `interrow` | per bloc/tile | `rows`, `canopies` | `interrows` + cover |
| 8 | `structure` | per tile | `canopies`, `row_pieces` | `row_structure` |
| 9 | `waste` | per tile | tile, `rows` | `waste` |
| 10 | `qa_sheets` | per tile | toate | JPEG-uri de control + `qa_report.csv` |
| 11 | `export_cvat` | per parte | straturi + override | 7 ZIP-uri + raport validator + `empty_tiles.csv` |
| 12 | `import_marcaj` | global | export Marcaj | straturi `source=marcaj` |
| 13 | `measure` | global | straturi | `measurements.csv` |
| 14 | `targets` | global | straturi | `targets.geojson` |
| 15 | `route` | global | domeniu, ținte, start | `route.geojson` + raport validare |
| 16 | `web_build` | global | toate | `web/ortho/` (XYZ), `web/vt/` (MVT), `web/data/*` (GeoJSON + JSON pentru tabele) |

### 3.5 Structura codului

```
vineyard-siret3/
├── README.md  route.geojson  measurements.csv  LICENSE  Dockerfile  .dockerignore  Makefile  pyproject.toml  uv.lock  requirements.lock
├── configs/  default.yaml  overrides.yaml
├── src/vineyard/
│   ├── cli.py                 # typer: index, run, export-cvat, from-marcaj, measure, route, web, benchmark, all
│   ├── config.py  logging.py  parallel.py (parallel_map cu spawn, vezi §4.1)
│   ├── io/        tiles.py (citire, georef), cvat_write.py, cvat_read.py, gpkg.py, zipper.py
│   ├── vision/    vegmask.py (a*, umbră), morphology.py
│   ├── rows/      orientation.py, detect_tile.py, link.py, structure.py
│   ├── blocks/    segment.py, ids.py
│   ├── canopy/    corridor.py, instances.py, simplify.py, trees.py
│   ├── interrow/  bands.py, cover.py
│   ├── waste/     candidates.py, filters.py, classify.py
│   ├── nn/        dataset.py, unet.py, train.py, infer.py
│   ├── targets/   gaps.py
│   ├── route/     domain.py, graph.py, tsp.py, validate.py
│   ├── measure/   measurements.py
│   ├── qa/        sheets.py, metrics.py (reimplementarea scorului oficial)
│   └── web/       build.py (gdal2tiles, tippecanoe, GeoJSON WGS84 cu 6 zecimale, rows/blocks/route.json)
├── web/  index.html  app.js  style.css  vendor/ (maplibre-gl local)  ortho/ vt/ data/ (generate)
├── tests/  test_georef.py test_cvat_roundtrip.py test_metrics.py test_route_validate.py test_rows.py
└── work/ (gitignored)
```

### 3.6 Config (valori implicite, măsurate)

```yaml
vegmask:   {index: lab_a, blur_sigma_px: 2.5, threshold: 4.0,
            otsu_fallback_veg_frac: [0.10, 0.90], texture_fallback: v_std_0.25m}
rows:      {spacing_min_m: 1.8, spacing_max_m: 3.8, band_m: 0.35, angle_gate_deg: 3,
            snap_to_edge_m: 3.0, min_row_len_m: 1.0, min_rows_per_block: 3,
            spacing_estimate: autocorr, peak_min_dist_factor: 0.6, periodicity_min_snr: 3.0,
            onlattice_tol_factor: 0.25, residual_split_m: 0.15, track_window_px: 256,
            link_angle_max_deg: 3, link_offset_max_m: 0.3, link_gap_max_tiles: 1}
orchard:   {width_spacing_ratio_max: 0.45, along_period_m: [3, 6], along_duty_max: 0.60,
            tree_blob_area_m2: 2.0, tree_blob_axis_ratio_max: 1.5}
canopy:    {corridor_half_m: 0.30, min_area_m2: 0.19, simplify_px: 2.5, max_perp_width_tree_m: 1.5}
interrow:  {offset_m: 0.30, cover_bare_max: 0.25, cover_veg_min: 0.75, shadow_unassessable: 0.5}
structure: {gap_disrupted_m: 5.0, unassessable_visible_min: 0.5, occ_close_m: 0.4,
            occ_window_m: 0.5, occ_min: 0.20, gap_fill_max_width_m: 1.2}
blocks:    {neighbour_max_m: 4.0, collinear_gap_max_m: 5.0, spacing_jump_max_m: 0.2,
            phase_tol_factor: 0.3, headland_min_rows: 3}
waste:     {min_area_m2: 0.015, white_min_area_m2: 0.03, max_area_m2: 6.0, min_dist_axis_m: 0.6,
            plant_phase_tol: 0.25, hose_min_len_m: 3.0, hose_max_width_m: 0.05, nms_iou: 0.3,
            box_pad: 0.10, candidate_min: 0.3, auto_probe_min: 0.9, auto_sam3_min: 0.5,
            auto_clip_margin_min: 0.2, sam3_crop_px: 512, block_radius_m: 10}
nn:        {encoder: resnet18, patch_px: 512, gsd_m: 0.05, batch: 8, lr: 3.0e-4, epochs: 20,
            ignore_band_px: 3, label_smoothing: 0.05, patience: 3, threshold: 0.5}
export:    {n_parts: 7, max_zip_bytes: 85000000, clip_px: [0, 2048], min_poly_px2: 16, simplify_px: 0.5}
runtime:   {workers: auto, workers_sweep: [5, 8, 10], maxtasksperchild: 50, chunksize: 2}
targets:   {gap_min_m: 5.0, gap_step_m: 15.0, missing_min_m: 2.0, include_missing: true}  # include_missing: decizie după Slack §8.3
route:     {start_tolerance_m: 5, visit_radius_m: 2.0, candidate_radius_m: 1.9,
            max_candidates_per_side: 3, max_outside_share: 0.005, domain_grid_size_m: 0.001,
            domain_inner_buffer_m: 0.05, string_pull_erode_m: 0.3, skeleton_res_m: 0.25,
            spur_min_m: 2.0, cover_iterations: 3, tsp_time_s: 30, tsp_time_final_s: 60,
            tsp_first: PATH_CHEAPEST_ARC, tsp_meta: GUIDED_LOCAL_SEARCH, cost_unit: cm}
```

Valorile noi (`periodicity_min_snr`, `orchard.*`, `occ_*`, `waste.auto_*`) sunt estimări inginerești din cercetare și se calibrează pe cele 2 tile-uri exemplu înainte de rularea pe 311.

---

## 4. Subsistemele în detaliu

### 4.1 Date, georef și performanță

- **Mediu: Python 3.12 cu uv, fără conda.** Planul inițial cerea 3.11, dar [rasterio 1.5.1](https://pypi.org/pypi/rasterio/json), [pyproj 3.8.0](https://pypi.org/pypi/pyproj/json), [numpy 2.5](https://pypi.org/pypi/numpy/json) și scipy 1.18 cer Python ≥ 3.12. Pe 3.11 am rămâne pe versiuni vechi. Nu trecem pe 3.13 sau 3.14, unde riscul de wheel-uri lipsă e mai mare.
  - Conda base de pe Mac exportă `PROJ_DATA`/`GDAL_DATA` și poate strica wheel-urile rasterio/pyproj ([rasterio install](https://rasterio.readthedocs.io/en/stable/installation.html)).
  - Pașii de instalare, în ordine ([uv + PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)):
    1. `conda deactivate` (sau `unset PROJ_DATA PROJ_LIB GDAL_DATA`)
    2. `uv python pin 3.12`
    3. `uv add …`
    4. `uv lock`
    5. `uv sync --locked`
    6. `uv export --format requirements-txt --no-hashes -o requirements.lock`
  - Se comit și `uv.lock`, și `requirements.lock`.
  - **Versiuni fixate (`==` în pyproject):**
    - geo: `rasterio==1.5.1`, `shapely==2.1.2`, `geopandas==1.1.4`, `pyproj==3.8.0`, `pyogrio==0.13.0`;
    - imagine și calcul: `opencv-python-headless==4.14.0.94` (rămânem pe 4.x, fără majorul 5.0 din 07.2026), `scikit-image==0.26.0`, `scipy==1.18.1`;
    - traseu: `ortools==9.15.6755`, `python-tsp==0.5.0`;
    - ML: `torch==2.14.0`, `torchvision==0.29.0`, `segmentation-models-pytorch==0.5.0`;
    - utilitare: `typer`, `pyyaml`, `lxml`.

    Restul (numpy 2.x, timm, huggingface-hub, pandas) îl rezolvă lock-ul.
  - **torch pe Linux:** în pyproject intră `[tool.uv.sources] torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]` (la fel pentru torchvision), plus `[[tool.uv.index]] name="pytorch-cpu" url="https://download.pytorch.org/whl/cpu" explicit=true`. Pe macOS torch vine de pe PyPI, cu MPS.
  - **Test imediat după lock:** `uv run python -c "import rasterio,pyogrio,pyproj,geopandas,cv2,torch;print(rasterio.__gdal_version__,pyproj.proj_version_str,torch.backends.mps.is_available())"`, plus un round-trip GPKG. Dacă numpy 2.5 nu se rezolvă cu scikit-image 0.26, fixăm `numpy<2.5`.
  - Rasterio aduce GDAL în wheel, deci pipeline-ul Python nu are nevoie de GDAL separat. Excepție: `brew install gdal tippecanoe`, necesare doar pentru `make web` (§4.14).
- **Disc (18 GB liberi):** tile-urile dezarhivate ~1,2 GB, măștile 1-bit PNG ~50 MB, JPEG-urile de QA ~300 MB, date web ~200 MB. Nu salvăm rastere float intermediare.
- **Memorie:** un tile RGB = 12,6 MB; per proces ~150 MB cu indicii. 8 procese în paralel = ~1,2 GB. Mozaicul întreg la rezoluție completă (~10 GB) **nu** se construiește: legarea rândurilor lucrează pe vectori.
- **Paralelism:** un singur helper, `parallel_map(fn, items, workers)`, folosit pe tile-uri la etapele 2, 3, 6, 8, 9 și 10.
  - Implementare: `multiprocessing.get_context('spawn').Pool(workers, initializer=_init_worker, maxtasksperchild=50)` + `imap_unordered(fn, tile_paths, chunksize=2)`.
  - De ce spawn explicit: pe macOS implicitul e spawn, iar pe Linux/Docker cu 3.12 e încă fork ([multiprocessing](https://docs.python.org/3/library/multiprocessing.html)). Fixat explicit, codul se comportă la fel pe ambele platforme.
  - Workerii primesc doar căi și config, iar `rasterio.open` se apelează în worker.
  - Funcțiile sunt top-level, în `vineyard.stages.*`. Orice script din `scripts/` are guard `if __name__ == '__main__'` ([pytorch #162612](https://github.com/pytorch/pytorch/issues/162612)).
  - `_init_worker()` apelează `cv2.setNumThreads(1)` și `cv2.ocl.setUseOpenCL(False)`. Workerii CPU nu importă torch.
  - CLI-ul setează variabilele de mediu înaintea importului numpy/cv2: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 GDAL_NUM_THREADS=1 GDAL_CACHEMAX=256`. Fără ele apare oversubscription de thread-uri ([opencv #15277](https://github.com/opencv/opencv/issues/15277)).
- **NN pe MPS:** etapă separată, rulată în procesul principal pe `device='mps'` (fallback `cpu`), niciodată concurent cu Pool-ul CPU.
  - DataLoader cu `num_workers` 0–2, batch 4–8 × 512², `torch.inference_mode()`.
  - Variabile de mediu: `PYTORCH_ENABLE_MPS_FALLBACK=1` și `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8`, pentru că memoria unificată de 18 GB e împărțită cu procesele CPU ([MPS env](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)).
  - Weights-urile smp se salvează în `weights/`, iar la inferență `encoder_weights=None`, ca rularea să meargă offline.
- **Timp estimat pe M3 Pro:** mască ~0,3 s/tile; axe ~2–4 s/tile; canopy + inter-rânduri ~1 s/tile; inferența NN ~5–10 min pe MPS.
  - M3 Pro are **11 nuclee (5 P + 6 E)**, nu 10 egale. `Pool(cores−2)` ar pune cel puțin 4 procese pe E-cores, deci scalarea nu e liniară.
  - Estimarea de ~5–10 min pentru 311 tile-uri se **măsoară**, nu se presupune. Numărul de workeri se alege printr-un sweep `--workers 5,8,10` pe 30 de tile-uri.
- **`vineyard benchmark`** scrie `reports/timing.json`:
  - `perf_counter` per etapă și total, 3 rulări (mediană + min), cold separat de warm;
  - pentru NN, `torch.mps.synchronize()` înainte de oprirea cronometrului;
  - s/tile per etapă și peak RSS (`/usr/bin/time -l make final`);
  - hardware: `sysctl -n machdep.cpu.brand_string`, nucleele P/E, `hw.memsize`, `sw_vers -productVersion`;
  - software și rulare: versiunile Python, torch, rasterio și `rasterio.__gdal_version__`, device, n_workers, n_tiles = 311.

  În README: un tabel per etapă și o cifră end-to-end (laptop pe alimentare, fără alte aplicații grele). Regulamentul judecă performanța după timpul și hardware-ul declarate, iar juriul poate cere re-rulare.

### 4.2 Unde există vie (tile-uri fără vie = penalizare)

Canopy-uri apar **doar** în coridoarele rândurilor validate, deci filtrul real e la validarea rândurilor:

1. Masca interzisă (`forbidden.geojson`) și nodata se scot din orice tile.
2. Un rând candidat e acceptat doar dacă:
   - lățimea vegetației perpendicular pe axă (percentila 80) e < 1,2 m (pomii au coroane de 2–4 m);
   - ocuparea coridorului de-a lungul axei e ≥ 15%;
   - are cel puțin 2 vecini paraleli la 1,8–3,8 m (≥ 3 rânduri per bloc, ca la regula viei de grădină).
3. **Livezi:** distanța între rânduri de 4–6 m și coroanele rotunde înseamnă respingere. Spectrul singur nu separă livada de vie, pe când structura spațială le separă ([Warner](https://www.tandfonline.com/doi/full/10.1080/13658816.2010.510839), [Delenne](https://hal.inrae.fr/hal-02587901)). Filtrul p80 < 1,2 m poate lăsa să treacă o livadă tânără, așa că adăugăm patru verificări ieftine, per rând sau per bloc:
   - **raportul lățime canopy / spacing:** via are ~0,15–0,45, livada ~0,5–0,8, deci peste 0,45 se respinge;
   - **periodicitatea în lungul rândului:** FFT pe ocupare. Un vârf clar la 3–6 m cu duty cycle < 60% înseamnă pomi discreți, deci respingere. Via matură are ocupare aproape continuă, via tânără pete la 1–1,5 m;
   - **circularitatea:** o componentă din coridor cu arie > 2 m² și raport al axelor < 1,5 e un pom. Același filtru prinde pomii izolați din vie (regula NOT A CANOPY);
   - **armonica:** o livadă cu rânduri la 5 m are armonica a 2-a la 2,5 m, adică în intervalul de căutare. Dacă vârful la `2s` e mai puternic decât cel la `s`, s-a prins armonica: spacing-ul real e `2s`, probabil o livadă.

   Pragurile sunt estimări și se calibrează pe exemple.
4. **Pajiști/culturi:** un tile fără periodicitate clară e gol. Criteriul „prominența vârfului < 5%” se înlocuiește cu un SNR spectral: energia vârfului FFT la `1/s`, împărțită la mediana spectrului radial. Sub ~3, tile-ul e „fără vie” (calibrat pe exemple). Prominența de 5% e arbitrară: pe inter-rândurile înierbate dă vârfuri false la jumătatea distanței ([Delenne](https://hal.inrae.fr/hal-02587805)).
5. Tile-urile fără niciun obiect intră în `empty_tiles.txt`, iar în Marcaj se bifează „No objects in this frame”.

### 4.3 Axele rândurilor

**Per tile** (prototip validat, F1 0,98 pe exemple):

1. Masca a\*, eșantion de ≤ 200 000 de pixeli de vegetație.
   - **Profil de rezervă** pentru inter-rândurile complet înierbate, unde profilul a\* devine aproape plat: deviația standard locală a lui V (fereastră de 0,25 m) sau V inversat. Canopy-ul viței e mai închis la culoare și mai texturat decât iarba.
   - Pe exemplele cu `interrow_cover = vegetation` verificăm dacă a\* își pierde vârfurile. Dacă da, perechea (a\*, textură) devine intrarea standard.
   - Masca NN (§4.6), odată antrenată vie-vs-iarbă, e intrarea cea mai robustă.
   - Fără DSM, literatura nu are altă separare vie/iarbă ([Versatile 2024](https://www.sciencedirect.com/science/article/abs/pii/S0168169924007634)).
2. Unghiul dominant = unghiul care maximizează varianța histogramei offset-ului perpendicular (pas 0,5°, bin 0,1 m). Metoda e echivalentă cu o transformată Radon și e confirmată de literatură: 2° eroare la orientare și 6 cm la inter-rând ([Delenne](https://hal.inrae.fr/hal-02587901)). Se poate accelera cu FFT 2D pe masca la 0,1 m/px.
3. **Spacing-ul `s` per tile** se estimează din autocorelația profilului perpendicular: primul vârf în intervalul 1,8–3,8 m.
   - Profilul se netezește cu σ = 3 bin, iar vârfurile se caută cu distanța minimă **`0,6·s`**, în loc de 0,9 m fix. Prominența de 5% dispare (vezi §4.2.4).
   - **Verificarea on-lattice:** un vârf al cărui offset se abate cu mai mult de `0,25·s` de grila `o0 + i·s` se marchează suspect (smoc de iarbă, urmă de roți). Se păstrează doar dacă trece testele de lățime și ocupare.
4. Per vârf: `cv2.fitLine` Huber pe pixelii din banda ±0,35 m, iterat de 3 ori cu restrângere la ±0,35 m. Pasul face deja ce face Hough-clustering + TLS ([Comba 2015](https://dl.acm.org/doi/abs/10.1016/j.compag.2015.03.011)), deci nu trecem pe Hough. Unghiul poate varia față de cel dominant cu cel mult 3°, ceea ce acoperă blocurile în evantai (46–53°).
   - **Rezidual:** dacă p95 al distanței pixelilor din bandă față de linie e > 0,15 m, rândul e curb.
   - O linie dreaptă pe un tile de 51,2 m trece metrica (0,4 m) doar până la R ≈ 800 m.
   - Pe aceste benzi se face **tracking**: ferestre de 256 px (6,4 m) de-a lungul rândului, centroidul vegetației din banda ±0,35 m, fiecare fereastră pornită de la cea precedentă.
   - Ieșirea e o polilinie cu vertex la 5–10 m, simplificată Douglas-Peucker cu 0,1 m ([Nolan 2015](https://research.monash.edu/en/publications/automated-detection-and-segmentation-of-vine-rows-using-high-reso/)). În CVAT rămâne tot polyline.
5. Capete: percentilele 0,5 și 99,5 ale proiecției pe linie; dacă un capăt e la < 3 m de marginea tile-ului, se prelungește până la margine (referința trasează prin goluri până la margine).
6. Tile-urile cu mai multe blocuri cu orientări diferite: după primul bloc, pixelii explicați se scot și se repetă pasul 2 pe rest (max 3 orientări).

**Global (legarea):**

1. **Legare pe dreapta-suport, nu pe capete** (union-find). Două piese din tile-uri adiacente se leagă dacă `|Δunghi| ≤ 3°` și offset-ul perpendicular al piesei B față de dreapta-suport prelungită a piesei A e ≤ 0,3 m. Se permite un gol coliniar de ≤ 1 tile.
   - Condiția veche („capete la ≤ 0,4 m”) rupea lanțul când rândul avea un gol lângă margine: capătul se oprea la > 3 m de margine și nu se mai prelungea, rezultând un `row_id` dublu.
   - Toate tile-urile vin din același ortofoto, deci la margini nu există decalaj de mozaicare ([Comba 2015](https://dl.acm.org/doi/abs/10.1016/j.compag.2015.03.011)).
2. Un lanț = un rând fizic. Se refit-uiește ca polilinie în UTM: dreaptă dacă abaterea maximă e ≤ 0,15 m, altfel cu aceeași funcție de tracking de la pasul 4 per tile (vertex la 5–10 m, DP 0,1 m).
3. Decuparea pe tile dă `row_pieces` cu 2 puncte (sau n pentru curbe).
4. **Cioturi de colț** (< 1 m în tile): se păstrează doar dacă rândul global continuă în tile-ul vecin. Referința le are, iar F1 le cere.
5. Rândurile extreme ale blocului se păstrează (regula 4.1).

**Cod extern:** păstrăm implementarea proprie. Repo-urile open source pentru rânduri ([crop-row-detector](https://github.com/ibaldoncini/crop-row-detector), [crop_row_detection](https://github.com/petern3/crop_row_detection), [plant-line-detection](https://github.com/maikbasso/plant-line-detection)) sunt generice. Niciunul nu tratează legarea între tile-uri, blocurile sau rândurile curbe pe ortofoto georeferențiat.

### 4.4 Blocuri și ID-uri

1. Graf: nod = rând global; muchie dacă două rânduri sunt paralele (≤ 5°), la 1,8–4,0 m, cu suprapunere de-a lungul ≥ 20%, **sau** coliniare cu gol ≤ 5 m.
2. Muchia se taie în trei cazuri:
   - segmentul care le leagă intersectează un pasaj (`passages.geojson`), pentru că un drum separă întotdeauna blocurile;
   - **salt de spacing sau de fază:** distanța dintre cele două rânduri diferă cu > 0,2 m (> 8%) de mediana locală `s`, sau offset-ul nu e multiplu de `s` (abatere > `0,3·s`). Parcelele vecine diferă tocmai prin (orientare, inter-rând) ([Delenne](https://hal.inrae.fr/hal-02587901), [HAL 02596316](https://hal.inrae.fr/hal-02596316)). Fără această regulă, două parcele cu aceeași orientare, separate de o întoarcere de 4–6 m care nu apare în `passages.geojson`, s-ar uni într-un bloc;
   - **golul coliniar ≤ 5 m e o bandă transversală:** zona fără vegetație continuă apare simultan pe ≥ 3 rânduri vecine, deci e o zonă de întoarcere sau un drum. Rândurile respective formează blocuri noi.
3. Componente conexe → blocuri; blocurile cu < 3 rânduri se elimină.
4. `vineyard_id` se dă în ordinea centroidului (N→S, apoi V→E).
   - **`row_id` vine din indexarea pe grilă per bloc:** se estimează `o0` și `s` (mediane), iar `row_index = round((offset − o0)/s)`. Piesele din tile-uri diferite cu același index și `|Δoffset| < 0,3·s` primesc același `row_id`. Așa rezolvăm și golurile de la marginea tile-ului.
   - La blocurile în evantai (46–53°), grila se estimează local, pe o fereastră de 2×2 tile-uri.
   - La blocurile mici sau neregulate (vii de grădină), unde grila nu se poate estima, fallback pe ordinea offset-ului din legarea simplă.
5. Poligonul blocului = înfășurătoarea concavă a coridoarelor (pentru UI, pentru atribuirea waste-ului și pentru regula de 10 m).

### 4.5 Canopy

1. `mask ∩ coridor(row_piece, ±0,30 m)` → componente conexe 8 → păstrează ≥ 0,19 m².
   - Masca a\* cu prag fix rămâne principală: „indice + prag” e printre cele mai bune metode pe vie ([Poblete-Echeverría 2017](https://doi.org/10.3390/rs9030268)).
   - **Fallback Otsu per tile:** dacă fracția de vegetație din coridoare e anormală (< 10% sau > 90%), pragul a\* se înlocuiește cu Otsu pe a\*, calculat doar pe pixelii din coridoare.
   - **ExGR** = (2G−R−B) − (1,4R−G), pe RGB normalizat, stabil la iluminare variabilă ([Sci. Rep. 2025](https://www.nature.com/articles/s41598-025-23868-1)). Îl calculăm doar ca a doua opinie în foile QA, nu în export.
   - Niciun indice nu separă via de iarba verde; asta face coridorul sau NN-ul.
2. **Pomi în rând:** dacă componenta de vegetație originală (înainte de coridor) are lățimea perpendiculară > 1,5 m, partea din coridor se șterge (nu e viță). Dacă obstacolul produce un gol ≥ 5 m, rândul devine disrupted, iar în inter-rânduri se face o gaură.
3. Contur: `cv2.findContours` + `approxPolyDP(ε = 2,5 px)`, țintă ~15 vârfuri. Poligoanele invalide se repară (`make_valid`), iar bucățile < 0,19 m² după decupare se elimină.
4. **Nu** împărțim forțat la 1–1,5 m. Referința păstrează pâlcurile continue, iar split-ul ar scădea F1@0,5. Excepție: via matură cu coroane unite pe > 3 m, unde tăiem la minimele locale ale lățimii canopy-ului de-a lungul axei (îngustare vizibilă), cum cer regulile. Decizia finală vine după întrebarea pe Slack (§8).
5. Decupare la marginea tile-ului; fiecare parte e un poligon separat în tile-ul ei.

### 4.6 Rețeaua neuronală (livrabil obligatoriu)

- **Model:** U-Net cu encoder ResNet18 pre-antrenat ImageNet (`segmentation_models_pytorch`, alternativă ResNet34), 2 clase: viță (canopy în coridor) și rest. **Fără SegFormer/MiT:** e mai lent pe MPS (attention cu operații în fallback), iar câștigul la 2 clase e mic.
- **Date:** pseudo-etichete pe **tot tile-ul**, nu doar în coridor, pe tile-urile unde QA-ul a validat rândurile (~100+ tile-uri).
  - viță = a\* ∩ coridor, după filtrul de pomi → 1;
  - **toată vegetația din afara coridoarelor (iarbă, pomi) → 0, explicit.**

  Așa rețeaua învață viță vs. iarbă, ceea ce a\* nu poate. Un U-Net pe RGB bate indicii + Otsu tocmai prin textură și context ([PMC11217331](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11217331/), [RS 11(9):1023](https://doi.org/10.3390/rs11091023)). Antrenat doar în coridor, NN-ul ar copia masca a\*.

  Validare: cele 2 tile-uri cu referință, nefolosite la antrenare.
- **Antrenare:**
  - patch-uri 512×512 la 0,05 m/px (downsample 2×), pre-extrase în `.npy`/uint8;
  - augmentări: flip, rotație 90°, jitter de culoare ±10%;
  - **anti-zgomot** ([Self-Filtered Learning](https://impact.ornl.gov/en/publications/self-filtered-learning-for-semantic-segmentation-of-buildings-in-/), [arXiv 2402.16164](https://arxiv.org/html/2402.16164), [NCE+RCE](http://proceedings.mlr.press/v119/ma20c/ma20c.pdf)):
    - bandă ignore de 3 px (`ignore_index=255`) la marginile pseudo-măștii și ale coridorului;
    - loss Dice + BCE cu label smoothing 0,05;
    - early stopping pe cele 2 tile-uri de referință (patience 3);
    - opțional: după epoca 5 se aruncă cele mai zgomotoase 10% din patch-uri (după loss) și antrenarea se reia;
    - fără self-training iterativ (risc de confirmation bias);
  - **setări MPS:** fp32 fără autocast, `export PYTORCH_ENABLE_MPS_FALLBACK=1`, batch 8 (max. 16 pe 18 GB), AdamW cu lr 3e-4 și cosine decay, 15–20 de epoci, DataLoader `num_workers=4` cu guard `__main__`. AMP/bf16 pe MPS e instabil ([HF perf](https://huggingface.co/docs/transformers/v4.49.0/perf_train_special), [ultralytics #26237](https://github.com/ultralytics/ultralytics/issues/26237)).
  - **Timp:** estimat **~15–30 min** pentru 15 epoci (~400 de patch-uri/epocă, ~10–20 img/s, estimare proprie), nu 1–2 h. Se confirmă vineri printr-un **smoke-run de 1 epocă** pe MPS. La erori sau NaN se trece pe `device='cpu'` pentru debug. Timpul câștigat merge în ablație.
- **Utilizare:** `mask_final = (nn_prob > 0,5) ∩ coridor`, sau vot cu masca a\*. Pragul rămâne fix la 0,5, fără tuning pe referință.
- **Ablație** (metrica challenge-ului, 0,6·IoU + 0,4·F1@0,5, raportată per tile și în medie):

  | Variantă | Ce testează |
  |---|---|
  | A | a\* ∩ coridor (baseline) |
  | B | NN > 0,5 ∩ coridor |
  | C | a\* AND NN |
  | D | a\* OR NN |
  | E | NN fără coridor (a învățat singur viță vs. iarbă?) |
  | F | NN antrenat fără banda ignore și fără label smoothing |

  - Pe lângă tabel: 3–4 panouri pe tile-uri grele (iarbă verde între rânduri, umbră, pomi în rând), cu RGB | a\* | NN | diferență, plus timpul de inferență per tile.
  - B, C sau D intră în export **doar** dacă varianta câștigă pe **ambele** tile-uri. Cu 2 tile-uri, o câștigare pe unul singur e zgomot.
  - Altfel NN-ul rămâne livrabil, cu ablația documentată în README și în pitch.
- **Seturi externe evaluate și nefolosite** (se trec în README):
  - [Riseholme](https://zenodo.org/records/19234907): CC-BY-4.0, dar cadre UAV brute, GSD necunoscut și un sezon fără frunze;
  - [Brescia groundcover](https://zenodo.org/records/17701564): CC-BY-4.0, dar acces restricționat și GSD de ordinul milimetrilor.
- **Weights:** Hugging Face Hub sau GitHub Release (~50 MB), link în README; `vineyard nn-train` reproduce antrenarea.
- **Al doilea model:** linear probe pe embedding-uri OpenCLIP/DINOv2 pentru waste (vezi 4.9).

### 4.7 Inter-rânduri și `interrow_cover`

1. Pentru fiecare pereche de rânduri vecine din același bloc: bandă între axa_i deplasată +0,30 m și axa_{i+1} deplasată −0,30 m.
2. Capete: tăiate la proiecția capătului rândului mai scurt (regula 5.1). Headland-ul și drumul nu intră.
3. Minus uniunea canopy-urilor (garantează zero suprapunere), minus găuri pentru pomi/clădiri (vegetație lată > 1,5 m sau poligon interzis).
4. Decupare pe tile: un poligon per inter-rând per tile. Nu se face inter-rând în afara rândurilor extreme.
5. **Cover:** `veg_frac` = fracția măștii a\* în poligon; `shadow_frac` = fracția de pixeli cu V < 50 (HSV).
   - `shadow_frac > 0,5`, sau lățime < 0,4 m (canopy închis) → `unassessable`
   - `veg_frac < 0,25` → `bare_soil`
   - `veg_frac > 0,75` → `vegetation`
   - altfel → `mixed`

   Pe exemple: bare_soil 0,00–0,16, mixed 0,51–0,68, deci toate 49 corecte.

### 4.8 `row_structure`

- Pe fiecare `row_piece`: golul maxim de-a lungul axei în coridorul ±0,30 m, **inclusiv de la capetele piesei până la primul/ultimul canopy** (așa reproduce referința toate 51 de etichete).
- **Profilul de ocupare, înainte de măsurare** (metoda standard pe ortofoto, [Primicerio 2017](https://www.tandfonline.com/doi/full/10.1080/22797254.2017.1308234)):
  - închidere morfologică 1D de 0,4 m, care acoperă găurile din frunziș și umbrele mici care ar fragmenta golurile;
  - o celulă e ocupată doar dacă ocuparea pe ±0,30 m e ≥ 20%, pe o fereastră de 0,5 m;
  - vegetația care umple golul contează doar dacă e mai îngustă de 1,2 m și nu e iarbă (aceeași trăsătură de textură ca la §4.3 pasul 1). Altfel iarba înaltă din inter-rând ar umple fals golul.
- Gol ≥ 5,0 m → `disrupted`, altfel `regular`.
- `unassessable`: < 50% din lungimea piesei e vizibilă (umbră adâncă, supraexpunere cu V > 245, sau vegetație lată peste rând).
- Rândurile cu gol între 4,5 și 7 m intră pe lista de verificare manuală (zona în care masca poate greși).
- Același profil produce și golurile de ≥ ~2 m (1,5 × distanța dintre plante), care devin ținte `missing` în §4.10.

### 4.9 Waste

Context: exemplele au 0 gunoaie și sute de tuburi albe. Un fals pozitiv costă cât un ratat.

1. **Candidați (recall mare):** pete cu saturație mare sau culori ne-naturale (albastru, roșu, galben intens), plus pete foarte luminoase (V > 200, S < 45).
   - Aria acceptată e 0,015–6 m². O sticlă de 25 cm are ~0,018 m²; pragul vechi, de 0,05 m², ar fi ratat-o.
   - Petele albe mici sunt tratate de filtrul 2b.
2. **Filtre dure** (regulile listează explicit ca NOT WASTE tuburile, țărușii, stâlpii și sârmele, furtunurile, pietrele, solul, arbuștii înfloriți, resturile de tăiere și vehiculele; „when in doubt, leave it out”):
   - elimini tot ce e la < 0,6 m de o axă (tuburi, țăruși);
   - **2a. periodicitate:** elimini orice candidat alb la ≤ 0,6 m de axă **și** la ±25% de o poziție de plantă prezisă (proiecția pe axă, modulo pasul de plantare). Tuburile stau regulat, cu pasul de plantare;
   - **2b.** elimini petele albe și nesaturate < 0,03 m² (un tub din nadir are ~0,01–0,02 m²);
   - elimini ce e alungit și subțire, cu raport laturi > 2,5 și arie < 0,12 m² (tub);
   - **2c. furtun:** elimini candidații liniari > 3 m cu lățimea < 5 cm;
   - elimini ce se suprapune cu clădiri sau cu zona interzisă;
   - elimini vehiculele (> 6 m²);
   - **2d.** NMS între candidați la IoU > 0,3, cu o singură cutie pe cluster. Duplicatele sunt FP la scorarea F1@IoU 0,3.
3. **Verificare cu două modele:**
   - **Linear probe** (logistic regression) pe embedding-uri OpenCLIP ViT-B/32 sau DINOv2-S.
     - Crop strâns: box + 50% context, minim 64 px, mărit la 224 px.
     - Pozitivele vin din [UAVVaste](https://github.com/PUTvision/UAVVaste) (Apache-2.0, 772 de imagini UAV, 3 718 adnotări `rubbish` pe iarbă și sol, [Zenodo 8214061](https://zenodo.org/record/8214061)): câteva sute de crop-uri de 64–128 px, rescalate empiric la ~2,5 cm/px.
     - Negativele sunt sute de tuburi, țăruși, pietre și bucăți de sol din cele 2 exemple.
     - Pragul se calibrează pentru **0 FP pe exemple**.
     - Motivul schimbării: CLIP zero-shot e evaluat pe scene întregi, nu pe obiecte de 10–40 px. Pe crop-uri dominate de sol, un prag absolut de 0,85 e nefundamentat ([RemoteCLIP](https://arxiv.org/html/2306.11029v4)).
   - **CLIP zero-shot** rămâne doar pentru ordonarea listei. Semnalul e marja softmax (pozitiv max − negativ max), nu un prag absolut.
   - **SAM 3** ([docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3)), dacă primim acces:
     - rulează cu prompt text pentru fiecare dintre `trash`, `plastic bag`, `plastic bottle`, `tire`, `rubbish heap`, separat;
     - în același crop primește 1–3 box-uri **negative** (`input_boxes_labels` = 0) pe tuburile albe detectate pe axă;
     - exemplarele trebuie să fie din aceeași imagine. N-avem exemplare pozitive de waste, deci varianta din planul inițial cu „exemplare pozitive/negative” nu e realizabilă;
     - rulează pe **crop-uri de 512 px centrate pe candidați**, nu pe tile întreg. SAM 3 redimensionează intern la 1008 px, deci un tile de 2048 px ar ajunge la ~5 cm/px ([sam3-mac](https://github.com/benreichman/sam3-mac));
     - rulează pe **CPU** (transformers `Sam3Model`, `device='cpu'`), cu buget ≤ 30 min. Pe MPS e mai lent, iar oficial nu e suportat ([sam3 #164](https://github.com/facebookresearch/sam3/issues/164)). Fallback: portul [MLX](https://huggingface.co/mlx-community/sam3-image), care acceptă box-uri include/exclude;
     - `post_process_instance_segmentation(threshold=0.4)` pentru lista de candidați; masca trebuie să se suprapună pe candidat.
   - **Fără YOLO pe DroneWaste.** [DroneWaste](https://zenodo.org/records/17045559) e deschis (CC BY 4.0), dar e despre grămezi în depozite ilegale, atinge doar mAP@50 ≈ 38% chiar in-domain, nu are greutăți publicate și are 3,9 GB. Cel mult devine sursă suplimentară de crop-uri pozitive, dacă rămâne timp.
4. **Pre-adnotare automată** doar dacă linear probe ≥ 0,9 **și** (SAM 3 ≥ 0,5 **sau** marjă CLIP > 0,2).
   - Criteriul vechi, „scor ≥ 0,85”, dispare.
   - Box-ul e axis-aligned, cu padding de 10% (IoU ≥ 0,3 e tolerant). Obiectele suprapuse primesc o singură cutie.
5. **`vineyard_id`:** blocul care conține cutia, altfel blocul cel mai apropiat la ≤ 10 m, altfel gol.
6. **Confirmare umană:**
   - toți candidații cu scor ≥ 0,3 intră în `waste_candidates.csv` (tile, coordonate, crop, scoruri), ordonați după scor;
   - în export ajung doar cei confirmați de om, plus cei auto de la pasul 4;
   - timp estimat: ~1 min pe 20 de candidați. Waste valorează 10%, iar un FP anulează un TP.
7. **README:** atribuire Apache-2.0 pentru UAVVaste și CC BY 4.0 pentru DroneWaste, dacă sunt folosite.

### 4.10 Ținte de inspecție

- **Goluri:** pe rândul **global** (nu per tile), fiecare gol ≥ 5 m între canopy-uri produce o țintă în centrul golului, pe axă. Golurile > 20 m primesc câte o țintă la fiecare ~15 m, ca să acopere varianta „o țintă per plantă lipsă”.
- **Plante lipsă (`missing`):** golurile de ~2–5 m (≥ 1,5 × distanța dintre plante) din profilul de ocupare din §4.8 produc o țintă în centrul golului ([Primicerio 2017](https://www.tandfonline.com/doi/full/10.1080/22797254.2017.1308234)). Ele acoperă varianta „potentially missing vines” din brief.
  - Sunt controlate de flag-ul `targets.include_missing`.
  - Includerea lor în traseu se decide după răspunsul la întrebarea 3 de pe Slack (§8) și după costul în lungime, măsurat cu bucla de acoperire din §4.11.
- **Plantare rară:** segmente de 20 m cu ocupare a coridorului < 10%, dar fără gol ≥ 5 m, produc o țintă de tip `sparse` (prioritate mai mică).
- **Waste:** centrul fiecărei cutii (din exportul Marcaj corectat).
- **Accesibilitate:** o țintă pe axă e la 1,25–1,40 m de linia mediană a inter-rândului vecin, deci în raza de 2 m. Raza e confirmată în descrierea challenge-ului: o țintă e vizitată dacă traseul trece la ≤ 2 m de ea. Trecerea prin **oricare** dintre cele două inter-rânduri vecine ajunge (vezi GTSP în §4.11). Waste-ul din afara domeniului e accesibil dacă e la ≤ 2 m de domeniu; altfel `reachable = false` și se raportează.
- Fiecare țintă are `target_id`, coordonate UTM, `vineyard_id`, `row_id` și `type`.

### 4.11 Traseul

Suprafețe măsurate: pasajele au ~40 000 m² (2 poligoane cu 13 găuri), iar zona de studiu 81,5 ha.

1. **Domeniu permis** = uniunea inter-rândurilor (din exportul Marcaj) și a pasajelor, minus zona interzisă. Se construiește cu `shapely.union_all(polys, grid_size=0.001)` + `shapely.make_valid`. Pentru validare se folosește `domain_in = domain.buffer(-0.05)`, iar pentru scurtături `domain_eroded = domain.buffer(-0.3)`. Ambele se pregătesc cu `shapely.prepare`.
2. **Graf:**
   - **inter-rânduri:** linia mediană = media geometrică a celor două axe consecutive din §4.3, tăiată cu `intersection(domain_eroded)` și eșantionată la 2 m (noduri). Nu folosim Voronoi, pentru că axele există deja. Pachetul [`centerline`](https://pypi.org/project/centerline/) (`interpolation_distance=0.5`) rămâne doar fallback, pentru că pe poligoane cu găuri produce ramuri parazite;
   - **pasaje:** `buffer(-0.3)` → `skimage.morphology.skeletonize` pe raster la 0,25 m, doar pe bbox-ul pasajelor.
     - Graful se construiește direct cu [`sknw.build_sknw(ske, multi=False, iso=False)`](https://github.com/Image-Py/sknw); pixelii trec în UTM prin transformul affine.
     - Ramurile terminale < 2 m se elimină, iar nodurile se pun la fiecare 5 m.
     - Alternativa vectorială, doar dacă scheletul dă probleme: [`shapely.voronoi_polygons(segmentize(boundary, 0.5), only_edges=True)`](https://shapely.readthedocs.io/en/stable/reference/shapely.voronoi_polygons.html) + `covered_by(domain)`;
   - muchiile se simplifică cu `shapely.simplify(edge, 0.25)` și se verifică cu `covered_by(domain_eroded)`;
   - conectori capăt de inter-rând → cel mai apropiat punct al scheletului pasajelor, cu lungimea în afara domeniului înregistrată;
   - greutate muchie = lungime; conectorii au o penalizare suplimentară, ca să fie folosiți doar când e nevoie.
3. **Ținte → seturi de candidați (Close-Enough TSP ca GTSP)** ([Carrabs](https://www.sciencedirect.com/science/article/abs/pii/S0305054816302179), [arXiv 2507.03775](https://arxiv.org/pdf/2507.03775)):
   - candidații unei ținte sunt nodurile grafului la ≤ 1,9 m (marjă de 0,1 m sub raza de 2 m), cel mult 2–3 pe fiecare parte, din **ambele** inter-rânduri vecine;
   - fiecare set intră în OR-Tools ca `routing.AddDisjunction([idx_candidati])`, fără penalitate, deci se vizitează exact un nod din set;
   - țintele de pe același nod se deduplică; țintele cu `reachable=false` nu intră în model;
   - totalul de noduri rămâne ≤ ~1 500 (la nevoie, 2 candidați pe parte sau ținte sparse grupate).

   Planul inițial lega fiecare țintă de un singur nod, cel mai apropiat. O țintă pe axă are însă doi vecini reali, iar alegerea forțată a unuia produce ocoluri.
4. **Distanțe:** Dijkstra multi-sursă cu `scipy.sparse.csgraph.dijkstra` pe indici, din toți candidații și din START.
5. **Ordinea:** OR-Tools ([`ortools==9.15.6755`](https://pypi.org/project/ortools/), wheel arm64 cp311–cp314).
   - Model: `RoutingIndexManager(n, 1, 0)`, cu depot = START la indexul 0. Costurile sunt **întregi în cm**, `int(round(d_m*100))`, pentru că callback-ul acceptă doar int.
   - Căutare ([opțiuni](https://developers.google.com/optimization/routing/routing_options)): `PATH_CHEAPEST_ARC` + `GUIDED_LOCAL_SEARCH`, `time_limit` 30 s (60 s la rularea finală). GLS nu se oprește singur, deci limita e obligatorie.
   - Benchmark scurt: 10/30/60 s × {PATH_CHEAPEST_ARC, SAVINGS}. Rămânem la 30 s dacă diferența e < 0,5%.
   - A doua opinie, opțională: [`elkai==2.0.1`](https://github.com/fikisipi/elkai) (LKH-3), `solve_tsp(runs=10)` pe aceeași matrice, doar ca benchmark. Licența LKH e necomercială, deci elkai nu e dependență obligatorie. Se păstrează turul mai scurt, iar sursa se notează în metadate.
   - Fallback, dacă OR-Tools aruncă excepție sau nu întoarce soluție: [`python-tsp==0.5.0`](https://github.com/fillipe-gsm/python-tsp) cu `solve_tsp_local_search(D, x0=nn_tour, perturbation_scheme='two_opt', max_processing_time=20)`, apoi `solve_tsp_lin_kernighan`. Înlocuiește 2-opt-ul scris de noi.
   - Rezultatul se desfășoară în LineString prin căile Dijkstra, iar START se adaugă explicit ca prim și ultim vârf.
   - **Buclă de acoperire** (2–3 iterații, [CETSP](https://arxiv.org/html/2310.04257v2)): `shapely.dwithin(targets, route, 2.0)` găsește țintele acoperite deja de căi, acestea se scot din model și se re-rezolvă, până nu mai scade lungimea.
   - **String pulling:** subsecvența de vârfuri `i..j` se înlocuiește cu segmentul `[p_i, p_j]` dacă `shapely.covered_by(LineString([p_i, p_j]), domain_eroded)`. Scheletul raster e în zig-zag și supraestimează lungimea.
6. **Validare automată (blochează commit-ul dacă eșuează):**
   - **rapid:** `shapely.covered_by(route, domain_in)` pe domeniul pregătit. `prepare` accelerează doar predicatele, nu overlay-urile ([shapely.prepare](https://shapely.readthedocs.io/en/stable/reference/shapely.prepare.html));
   - **exact, la final:** `outside_share = shapely.difference(route, domain_in, grid_size=0.001).length / route.length`. Trebuie < 0,5% (pragul de eliminare e 2%). `grid_size` elimină artefactele de virgulă mobilă;
   - start și final la ≤ 5 m de START (ideal exact START);
   - `length_m = round(route.length, 2)`, recalculat din GeoJSON-ul recitit;
   - un singur LineString (nu Multi), EPSG:32635, `is_valid`, fără segmente de lungime zero. **Nu** se testează `is_simple`: un tur pe coridoare trece inevitabil de două ori prin inter-rândurile fundătură, deci un astfel de test ar bloca greșit commit-ul;
   - acoperire: **100%** din țintele reachable la ≤ 2 m de traseu (blocant), plus procentul raportat pe toate țintele.
7. **Robustețe față de inter-rândurile de referință:** mergem pe linia mediană, la ≥ 0,6 m de marginile benzii, deci chiar dacă poligoanele lor diferă cu câțiva decimetri rămânem înăuntru.
8. **Economie pentru pitch:** lungimea traseului vs. un traseu naiv (toate inter-rândurile în serpentină, sau țintele în ordinea ID-urilor). Se raportează km, minute la 4 km/h și procentul economisit.

### 4.12 Măsurători (`measurements.csv`)

Calculate din exportul Marcaj corectat, cu aceleași formule pe care le folosesc probabil organizatorii:

| Coloană | Definiție |
|---|---|
| `level` | `total` / `block` / `row` |
| `vineyard_id`, `row_id` | gol la `total` |
| `n_blocks`, `n_rows` | ID-uri distincte |
| `row_length_m` | suma lungimilor pieselor per `row_id` (piesele nu se suprapun între tile-uri) |
| `total_row_length_m` | suma pe bloc / total |
| `canopy_area_m2`, `canopy_area_ha` | aria **uniunii** poligoanelor canopy |
| `interrow_area_m2`, `interrow_area_ha` | suma inter-rândurilor |
| `n_canopies`, `n_tiles` | informativ |

Rânduri de exemplu: `total,,,12,340,...`; `block,V03,,,24,1520.4,...`; `row,V03,V03-R017,,,64.5,...`. Totul în metri, orizontal, EPSG:32635.

### 4.13 Export CVAT, Marcaj și round-trip

1. **Writer:**
   - Blocul `<meta>` se copiază byte-cu-byte din exemplu (cu `<version>1.1</version>`); etichetele se leagă de proiect după nume.
   - `<image id name width=2048 height=2048>`, cu `id` = 0..n-1 **per ZIP**, în ordinea sortată a numelor, și `name` = **doar basename-ul** (`siret3_r021_c012.tif`, fără `images/`).
   - Tile-urile fără obiecte apar ca `<image .../>` gol, nu se omit. Nu adăugăm `<tag>`.
   - Poligoanele și poliliniile au `source="auto"`, `occluded="0"` și `z_order="0"`. CVAT oricum suprascrie `source` cu `"file"` la import ([cvat.py](https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/formats/cvat.py)).
   - Waste se scrie ca `<box label="waste" source="auto" occluded="0" xtl ytl xbr ybr z_order="0"><attribute name="vineyard_id">…</attribute></box>`, cu `xtl<xbr` și `ytl<ybr`.
   - Atributele sunt toate prezente, în ordinea din exemplu: `vineyard_id, row_id, row_structure` pentru row și `vineyard_id, interrow_cover` pentru interrow_area. Coordonatele au 1 zecimală. Valorile se escapează XML (lxml).
   - **Geometria se curăță înainte de serializare**, pentru că CVAT stochează ce primește și nu repară nimic ([cvat #2990](https://github.com/cvat-ai/cvat/issues/2990)):
     - clip la [0,2048]² (`intersection(box(0,0,2048,2048))`), apoi `make_valid`;
     - MultiPolygon se sparge în poligoane cu aceleași atribute, iar bucățile < 16 px² se elimină;
     - `simplify(0.5 px, preserve_topology=True)`;
     - fără vârf de închidere duplicat;
     - polyline-urile se clipează cu aceeași cutie.
2. **Validator** (blochează exportul). Pe calea de import, CVAT **nu** verifică numărul de puncte și valorile select, aruncă în tăcere atributele cu nume necunoscut și pică la o etichetă necunoscută ([serializers.py](https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/engine/serializers.py), [bindings.py](https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/bindings.py)). Validatorul nostru e singura plasă de siguranță și verifică:
   - etichete ∈ {vineyard, waste, row, interrow_area}, cu tipul geometric corect: polygon / box / polyline / polygon;
   - setul **exact** de atribute per etichetă, fără atribute în plus; valorile exact cu litere mici, fără spații, din listă: `row_structure` ∈ {regular, disrupted, unassessable}, `interrow_cover` ∈ {bare_soil, vegetation, mixed, unassessable};
   - fiecare obiect are `vineyard_id`; waste-ul poate avea gol doar dacă e la > 10 m de un bloc;
   - rândurile au `row_id` și `row_structure` nevide, iar inter-rândurile au `interrow_cover` nevid;
   - geometrie:
     - polygon cu ≥ 3 vârfuri distincte, `is_valid` și arie ≥ 16 px² (0,01 m²);
     - polyline cu ≥ 2 puncte distincte și lungime ≥ 4 px (0,1 m);
     - box cu `xtl<xbr` și `ytl<ybr`;
     - toate coordonatele în [0, 2048];
   - nicio suprapunere canopy ∩ inter-rând > 0,05 m² per tile;
   - **bijecție nume:** set(`<image name>`) == set(`images/*.tif`) în fiecare ZIP, cu potrivire exactă, case-sensitive, NFC și extensia `.tif`. Reuniunea ZIP-urilor = cele 311 fișiere, fără duplicate. Dacă numele nu se potrivește, CVAT face fallback pe `id` și poate atașa **în tăcere** adnotările altui tile (`match_dm_item` în [bindings.py](https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/bindings.py));
   - sha256-ul fiecărui tile e identic cu sursa;
   - fiecare ZIP ≤ 85 000 000 B, iar conținutul e strict pe whitelist (vezi pasul 4).
3. **Round-trip:** citim exemplul, îl rescriem și comparăm **geometric** (IoU ≥ 0,999, Hausdorff < 0,1 px) și pe atribute, nu textual. Exportul real din Marcaj are 2 zecimale, `subset`/`task_id` și `source="file"`.
4. **ZIP-uri: 7 părți de ~45 de tile-uri**, nu 5.
   - Cele 5 părți originale au deja 93,2–94,1 MB (part4 = 94 108 721 B = 89,75 MiB), iar tile-urile sunt stocate fără compresie. `annotations.xml` adaugă ~26 KB comprimat pe tile (măsurat pe exemplu), deci ~2 MB pe parte. O parte ar ajunge astfel la ~96 MB, peste 90 MB și peste 90 MiB (măsurat local cu `ls -la 01_tiles/`, vezi [cercetarea §6.1](research/CERCETARE_Siret3.md)).
   - Împărțirea: ordinea sortată după nume, cu vecinii împreună.
   - Layout exact: `annotations.xml` (deflate -9) + `images/<nume original>.tif` (stored, `zip -0`), creat cu `zip -X` sau cu `zipfile` din Python, nu cu Finder.
   - Fără `__MACOSX/`, `.DS_Store` sau alte `.xml`: CVAT încarcă **toate** `**/*.xml` din arhivă.
   - Fiecare parte are `annotations.xml` doar cu tile-urile ei. Limita e ≤ 85 000 000 B per ZIP, cu marjă față de ambiguitatea MB/MiB.
   - Numărul de părți se fixează după ce măsurăm XML-ul real (canopy mai detaliat = XML mai mare).
5. **Test Marcaj vineri noaptea:** urcăm un ZIP cu exemplele, cel puțin un `<box>` waste și un tile cu `<image/>` gol. Verificăm:
   - raportul de import (frame-uri, obiecte, „skipped/dropped”);
   - în editor, pe 1–2 tile-uri, că poligoanele stau pe imaginea corectă;
   - dacă tile-ul gol apare cu „No objects” bifat (improbabil).

   Apoi „Remove all”. Așa formatul e validat înainte de ziua Publish.
6. **Publish (sâmbătă ~12:00):** se urcă **toate cele 7 părți**, secvențial, cu raportul fiecărei părți salvat (captură) în tracker. Se verifică **311 fișiere** în cardul Data, și abia apoi Publish. Checklist-ul verifică suma = 311, nu numărul de părți.
7. **Corectură:** ~63 de job-uri a câte 5 tile-uri, împărțite pe petice vecine per om, cu un tabel de urmărire (job, om, stare, Submit). Prioritățile, în ordinea scorului:
   1. șterge canopy și rânduri false pe tile-uri fără vie; bifează „No objects” pe toate tile-urile din `empty_tiles.csv`. Flag-ul nu poate fi setat prin import, iar fără el job-ul nu se poate trimite. Tracker-ul are o coloană „No objects bifat”;
   2. rânduri lipsă sau greșite; `row_id` identic peste margini (se copiază din vecin);
   3. `row_structure` și `interrow_cover`;
   4. sweep de waste;
   5. canopy-uri grosier greșite (ștergere sau adăugare pâlc; nu se retrasează plantă cu plantă);
   6. **Submit** la fiecare job imediat ce e gata.
8. **Export înapoi:** CVAT 1.1 din Marcaj → `vineyard from-marcaj` → straturi în UTM (`source=marcaj`), piesele unite per `row_id`, dedup la margini. Apoi se rulează etapele 13–16.
   - Parser-ul potrivește după `basename(name)`, nu după `id`, pentru că `id` e frame-ul global al proiectului, iar `name` poate include o cale.
   - Ignoră `subset`, `task_id`, `group_id` și `source`, acceptă `<tag>` și `<image>` goale și citește coordonatele ca float.
   - Verifică 311 `<image>` unice și zero etichete sau valori în afara listei.

### 4.14 Web UI

- **Stack:** site static cu **`maplibre-gl@5.24.0` (UMD)**, pus local în `web/vendor/` (demo-ul merge offline, fără CDN), plus un `app.js` vanilla de ~400 de linii, fără framework și fără build step.
  - v6 (6.11.2) e doar ESM și cere WebGL2, iar exemplele cu `<script src=maplibre-gl.js>` nu mai merg pe el ([ghid v5→v6](https://maplibre.org/maplibre-gl-js/docs/guides/v5-to-v6-migration-guide/)).
  - Deploy pe GitHub Pages prin Actions (`actions/upload-pages-artifact` + `actions/deploy-pages`), din folderul `web/` generat.
  - Fallback de pe laptop: `python -m http.server -d web`. Merge pentru că nu folosim HTTP Range.
  - Cloudflare Pages doar ca backup (`npx wrangler pages deploy web`).
- **Imagine de fond: piramidă XYZ statică**, nu 311 surse `image`. Fiecare sursă înseamnă o textură și un draw call separat, fără piramidă pentru zoom mic ([MapLibre large data](https://maplibre.org/maplibre-gl-js/docs/guides/large-data/)).
  - Comenzi ([gdal2tiles](https://gdal.org/en/stable/programs/gdal2tiles.html)):
    - `gdalbuildvrt siret3.vrt tiles/*.tif` (cu `-addalpha` sau `-srcnodata 0` dacă marginile negre nu ies transparente; se testează pe 5 tile-uri)
    - `gdal2tiles.py --xyz -z 14-20 -r average --tiledriver=WEBP --webp-quality=75 --processes=10 -x -w none siret3.vrt web/ortho`
  - Rezultat estimat: ~2 000 de fișiere, ~40–60 MB. Sursa MapLibre: `{type:'raster', tiles:['ortho/{z}/{x}/{y}.webp'], tileSize:256, minzoom:14, maxzoom:20}`, cu overzoom peste z20.
  - Sursa `image` cu 4 colțuri rămâne doar fallback, dacă GDAL nu se instalează.
- **Straturi vectoriale:**
  - **canopy și waste** devin tile-uri MVT într-un director, generate cu `tippecanoe -e web/vt -l canopy -Z16 -z20 --no-tile-compression --no-feature-limit --no-tile-size-limit --drop-densest-as-needed -y id -y vineyard_id -y row_id canopy_wgs84.geojson` ([tippecanoe](https://github.com/felt/tippecanoe)). `--no-tile-compression` e obligatoriu, pentru că Pages și python nu trimit `Content-Encoding: gzip` pentru `.pbf`;
  - sursa MVT în MapLibre: `{type:'vector', tiles:['vt/{z}/{x}/{y}.pbf'], minzoom:16, maxzoom:20, promoteId:{canopy:'id'}}`;
  - rândurile, inter-rândurile, blocurile, țintele și traseul rămân câte un GeoJSON global (WGS84, 6 zecimale, `promoteId`);
  - dispare logica „câte un GeoJSON per tile, încărcat dinamic”;
  - fallback fără tippecanoe: un singur `canopy.geojson` cu 6 zecimale, doar `id` și `row_id`, `minzoom:17`.
- **Fără PMTiles la demo.** Driverul GDAL scrie doar MVT, nu raster. HTTP Range a fost fragil pe GitHub Pages ([discuția 178318](https://github.com/orgs/community/discussions/178318), [PMTiles #584](https://github.com/protomaps/PMTiles/issues/584)) și e corupt pe Cloudflare ([demotiles #35](https://github.com/maplibre/demotiles/issues/35)). În plus, `python -m http.server` nu suportă deloc Range.
- **Buget:** `web/` < 300 MB, niciun fișier > 25 MB, < 20 000 de fișiere, fără Git LFS. Aceste limite se potrivesc cu [GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits) (≤ 1 GB) și cu [Cloudflare Pages Free](https://developers.cloudflare.com/pages/platform/limits/) (20k fișiere, 25 MiB per fișier). z21 pentru raster (~6–8k fișiere) doar dacă rămâne timp.
- **Panouri:** build-ul exportă `rows.json`, `blocks.json` și `route.json` cu `id`, `vineyard_id`, `row_id`, `length_m`, `area_m2`, `row_structure` și `bbox:[w,s,e,n]` precalculat.
  1. Sumar: număr de blocuri și rânduri, lungimea totală a rândurilor, aria canopy și inter-rând (m² și ha).
  2. Tabel blocuri și tabel rânduri. Tabelele sunt vanilla: sortare cu `[...rows].sort()`, fără mutație. Peste 5k rânduri se randează doar primele 500, plus un filtru text.
     - Click pe un rând din tabel: `map.fitBounds(bbox)` + `setFeatureState({sel:true})`.
     - Click pe hartă: `scrollIntoView` în tabel.
     - Detaliile unui rând: `vineyard_id`, `row_id`, lungimea, `row_structure` per tile.
  3. Traseu: lungime, ținte vizitate/total, timp la 4 km/h, economie vs. traseul naiv, lista ordonată a țintelor.
  4. Procesare: timpii per etapă din `timing.json`, hardware, versiunea modelului.
- **Legenda culorilor** ca în preview-urile oficiale: axe roșii (disrupted magenta), canopy verde, inter-rânduri cyan.

### 4.15 Inginerie și livrare

- **CLI:** `vineyard all` (index → export), `vineyard from-marcaj <zip>`, `vineyard measure`, `vineyard route`, `vineyard web`, `vineyard benchmark`, `vineyard nn-train`.
- **Makefile:**
  - `make setup`: `uv sync --locked`, plus `brew install gdal tippecanoe` pentru web;
  - `make preannotate`;
  - `make final MARCAJ=export.zip`: produce `route.geojson` + `measurements.csv` în rădăcină;
  - `make web`: gdal2tiles + tippecanoe + exportul JSON;
  - `make test`;
  - `make docker-build`: `docker buildx build --platform linux/arm64 -t vineyard:arm64 --load .`, nativ și rapid pentru test local;
  - `make docker-push`: `docker buildx build --platform linux/amd64,linux/arm64 -t <user>/vineyard:1.0 --push .`, rulat **o singură dată**, sâmbătă noaptea. amd64 e emulat și lent ([docker/for-mac #7075](https://github.com/docker/for-mac/issues/7075)); alternativa e build-ul în GitHub Actions.
- **Docker:** `python:3.12-slim`, în **două etape**, după pattern-ul [uv Docker](https://docs.astral.sh/uv/guides/integration/docker/).
  - Etapa builder:
    - `COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /bin/` (versiune fixată, nu `latest`);
    - `UV_COMPILE_BYTECODE=1` și `UV_LINK_MODE=copy`;
    - `uv sync --locked --no-install-project` cu cache mount, apoi `COPY . /app` și `uv sync --locked`.
  - Etapa finală copiază doar `/app/.venv` și setează `PATH=/app/.venv/bin:$PATH`, `OMP_NUM_THREADS=1`, `PYTHONUNBUFFERED=1` și celelalte variabile de thread-uri din §4.1. Rulează `CMD ["make","final"]`.
  - Nu e nevoie de apt pentru GDAL, PROJ sau build-essential: toate wheel-urile există pentru manylinux aarch64 și x86_64.
  - **torch trebuie să fie `+cpu`.** Pe Linux, torch de pe PyPI trage CUDA 13 (mai mulți GB), pe când wheel-ul `+cpu` are ~160–200 MB ([torch 2.14.0](https://pypi.org/pypi/torch/2.14.0/json)). Sursa uv cu marker linux din §4.1 rezolvă asta. Verificare: `python -c "import torch;print(torch.__version__)"` → `2.14.0+cpu`. Imaginea estimată: ~1,2–1,6 GB.
  - Tile-urile se montează ca volum (`-v $PWD/data:/app/data:ro`), iar `.dockerignore` exclude `data/`, `work/`, `.venv/` și `*.tif`.
  - Înainte de livrare: `docker run --network none`, ca să confirmăm că nu se descarcă weights la runtime.
  - Timpul se raportează în README pe **două rânduri**:
    - (a) macOS nativ, M3 Pro + MPS;
    - (b) Docker arm64 CPU, cu limitele VM din Docker Desktop declarate (de ex. 8 CPU / 10 GB, `--workers 6`).

    Nu raportăm timpi din amd64 emulat.
- **Teste (pytest):**
  - georef (colțuri de tile cunoscute);
  - round-trip CVAT pe exemple (comparație geometrică și pe atribute);
  - validatorul de export pe cazuri stricate: nume de atribut greșit, „Regular”, poligon bowtie, coordonate > 2048, ZIP cu `.DS_Store`/XML în plus, ZIP > 85 MB;
  - metrici oficiale reimplementate (canopy IoU/F1, F1 axe cu 0,4 m / 80%, IoU inter-rânduri, atribute acuratețe + macro-F1);
  - validatorul de traseu pe cazuri sintetice: traseu care iese din domeniu, start greșit, plus un tur dus-întors valid, care nu trebuie respins;
  - regresie pe exemple (scor canopy ≥ 0,80, F1 axe ≥ 0,95).

  GitHub Actions rulează testele.
- **README:** conține:
  - instalarea (uv, Python 3.12) și rularea de la tile-uri la `route.geojson` și `measurements.csv`;
  - dependențele fixate și link-ul la weights;
  - timpul de procesare pentru toate cele 311 tile-uri, cu hardware-ul (M3 Pro, 11 nuclee 5P + 6E, 18 GB) și tabelul din `timing.json`;
  - API-urile plătite sau LLM-urile folosite, declarate explicit;
  - link-ul la UI;
  - licențe și atribuiri: CC BY 4.0 (3DATA COLLECT / OpenAerialMap), ODbL pentru OSM, plus Apache-2.0 (UAVVaste) și CC BY 4.0 (DroneWaste), dacă sunt folosite;
  - seturile externe evaluate și nefolosite (Riseholme, Brescia), cu motivul.
- **Maparea pe grila de inginerie:**
  - arhitectură (5) = DAG + contracte + un singur cod pentru predicție și corectură;
  - robustețe (4) = validatoare, override-uri, fallback-uri (NN opțional, TSP fallback python-tsp, fallback Otsu, profil de textură);
  - scalabilitate (3) = procesare per tile paralelă, lucru vectorial global, liniar în numărul de tile-uri;
  - performanță măsurată (3) = benchmark reproductibil.

---

## 5. Validare și auto-verificare

| Moment | Ce verificăm | Prag |
|---|---|---|
| Vineri noaptea | round-trip CVAT pe exemple; import de test în Marcaj (exemple + un `<box>` waste + un `<image/>` gol) | identic geometric; raport fără skipped/dropped; poligoanele pe tile-ul corect |
| Vineri noaptea | `uv lock` + testul de import (rasterio, pyogrio, pyproj, cv2, torch MPS); smoke-run NN de 1 epocă | fără erori; timpul pe epocă măsurat |
| După pipeline pe 311 | scor pe exemple (canopy, axe, inter-rând, atribute) | ≥ 0,80 / ≥ 0,95 / ≥ 0,95 / ≥ 0,9 |
| QA pre-publish | foi de control cu toate cele 311 tile-uri (JPEG cu overlay: axe cu `row_id`, canopy, inter-rânduri, waste) | fiecare tile bifat OK sau cu override |
| QA pre-publish | tile-uri fără vie cu obiecte, livezi detectate ca vie | 0 |
| QA pre-publish | aceleași `row_id` pe ambele părți ale fiecărei margini | 100% |
| Înainte de Publish | validator export; 7 ZIP-uri ≤ 85 000 000 B, conținut pe whitelist; 311 fișiere | toate trec |
| Duminică | `measurements.csv` recalculat din export; traseu validat | în afara domeniului < 0,5%; start ≤ 5 m; 100% ținte reachable acoperite |
| Duminică 14:00 | Marcaj: toate job-urile Submitted, 0 în lucru | 100% |

---

## 6. Planul de lucru, oră cu oră

Roluri (pentru 5 oameni; la 3 sau 4 se comasează ca în tabelul următor):

- **A, lead pipeline / ML:** vegmask, rânduri, legare, canopy, NN.
- **B, geo și export:** index, georef, CVAT writer/validator, ZIP-uri, Marcaj, from-marcaj, măsurători.
- **C, traseu:** domeniu, graf, TSP, validator, ținte.
- **D, web UI și livrare:** UI, README, Docker, pitch.
- **E, QA și adnotare:** foi de control, override-uri, waste sweep, coordonarea corecturii în Marcaj.

| Oameni | Comasare |
|---|---|
| 3 | A = A; B = B + C; C = D + E |
| 4 | A; B; C + D (UI după traseu); E |
| 5 | ca mai sus |

| Interval | A | B | C | D | E | Gate |
|---|---|---|---|---|---|---|
| **Vin 21–23** | mediu (uv, Python 3.12, lock + test import), vegmask + axe per tile pe 311 (prototipul existent); smoke-run NN 1 epocă | mediu (`uv sync --locked`), index tile-uri, georef, CVAT writer + round-trip | citește GeoJSON-urile, schelet pasaje, prototip graf | schelet repo, CLI, README, schelet UI; `brew install gdal tippecanoe` | cerere acces `facebook/sam3` de pe 2 conturi HF; import de test în Marcaj (când sosesc conturile) | round-trip OK; lock OK |
| **Vin 23 – Sâm 02** | legare rânduri, blocuri, ID-uri | validator + ZIP-uri; canopy + inter-rânduri pe straturi | ținte din goluri; TSP pe date provizorii | date web din straturile provizorii | foi de control v1, prima trecere QA | pipeline complet pe 311 |
| **Sâm 02–07** | pornește antrenarea NN (estimat ~15–30 min, rulează singură); somn | somn | somn | somn | somn | — |
| **Sâm 07–10** | ablație NN A–F pe exemple; fix-uri de parametri | row_structure, cover, waste candidați + linear probe | validator traseu | UI tabele | QA complet 311 + `overrides.yaml`; confirmare candidați waste | toate tile-urile bifate |
| **Sâm 10–11:30** | rulare finală | export final, validator, 7 ZIP-uri, upload secvențial | — | — | verifică rapoartele de import, 311 fișiere | **PUBLISH ~12:00** |
| **Sâm 12–20** | corectură Marcaj | corectură + registru ID | graf pe exportul intermediar | UI + deploy v1 | coordonare corectură, waste sweep | 50% job-uri Submitted |
| **Sâm 20 – Dum 02** | corectură | from-marcaj + măsurători pe export intermediar | traseu pe export intermediar | README, Docker (`docker-push` multi-arch, o singură dată), benchmark | corectură | traseu valid pe date provizorii |
| **Dum 02–08** | somn în ture (câte 2 treji) | | | | | — |
| **Dum 08–11** | ultimele job-uri | — | — | pitch | toate job-urile Submitted | **100% Submitted ~11:00** |
| **Dum 11–13** | — | export final Marcaj → măsurători | ținte + traseu final + validare | UI cu datele finale, deploy | verificare încrucișată | fișierele din rădăcină finale |
| **Dum 13–14** | freeze cod | commit final | — | repetiție pitch | — | **freeze 14:00** |
| **Dum 14–15** | buffer | | | | | **deadline 15:00** |

**Ce tăiem dacă întârziem** (în ordinea inversă a valorii pe oră):

1. SAM 3 la waste → doar linear probe + confirmare umană; apoi linear probe → doar sweep manual pe candidații de culoare;
2. NN de canopy în export → rămâne livrabil cu ablație;
3. `sparse` (și, dacă e nevoie, `missing`) din ținte;
4. tracking pe rânduri curbe (rămân drepte);
5. elkai și benchmark-ul TSP; z21 în piramida raster;
6. panoul „procesare” din UI.

**Nu se taie niciodată:** QA pre-publish, validatorul de export, Publish la timp, Submit la toate job-urile, validatorul de traseu.

---

## 7. Registru de riscuri

| Risc | Prob. | Impact | Mitigare | Owner |
|---|---|---|---|---|
| Publish fără toate cele 7 părți sau înainte de QA | mică | foarte mare | checklist de 311 fișiere (suma pe părți, nu numărul de părți); Publish doar de B cu confirmarea lui E | B |
| ZIP de upload > 90 MB dacă păstrăm cele 5 părți originale (~96 MB cu XML) | sigură la 5 părți | foarte mare (upload respins la Publish) | 7 părți, limită validator ≤ 85 000 000 B, măsurare pe XML-ul real; întrebare pe Slack | B |
| Nume de tile nepotrivit în XML → CVAT atașează în tăcere adnotările altui tile (fallback pe `id`) | mică | mare | bijecție nume ↔ `images/` în validator; verificare în editor la testul de vineri | B |
| Import CVAT acceptă tacit geometrie invalidă, valori select greșite; atributele cu typo dispar | medie | mare | validator strict (tip, set exact de atribute, valori, `is_valid`, clip [0,2048]); teste pe cazuri stricate | B |
| „No objects in this frame” nu se poate seta la import → job-uri blocate la Submit | sigură | mediu | `empty_tiles.csv`, coloană în tracker, bifare manuală în prioritatea 1 | E |
| Axe greșite → canopy greșite după Publish (nu mai putem regenera) | medie | mare | QA vizual al tuturor tile-urilor + override-uri înainte de Publish | E |
| Livezi, grădini sau pajiști detectate ca vie | medie | mare (penalizare) | filtre 4.2 (raport lățime/spacing, periodicitate în lungul rândului, circularitate, verificarea armonicii) + QA; tile gol la dubiu | A |
| Livadă tânără (coroane < 1,2 m, rânduri la 3,5–4 m), la limita intervalului 1,8–3,8 m | mică | mare | verificare manuală în QA dacă există în zonă | E |
| Inter-rânduri complet înierbate: profilul a\* plat, axe ratate, `interrow_cover` greșit (fără DSM) | medie | mare | profil de rezervă pe textură (std V); masca NN vie-vs-iarbă ca intrare; QA | A |
| Rânduri legate greșit la margini → `row_id` dublat sau două parcele unite | medie | mediu | legare pe dreapta-suport, grilă `round((offset−o0)/s)`, tăiere la salt de spacing/fază; fallback pe legarea simplă la blocuri mici | A |
| Parametri noi (SNR ≥ 3, duty < 60%, raport ≤ 0,45, praguri waste) nefundamentați pe datele noastre | medie | mediu | calibrare pe cele 2 exemple înainte de rularea pe 311; valorile în config | A |
| ID-uri inconsistente peste margini după corectură | medie | mediu | ID-uri globale din model; registru R900+ pentru rânduri noi; verificare automată în from-marcaj | B |
| Traseu > 2% în afara domeniului | mică | 25% pierdut | linii mediane, conectori minimi, string pulling doar pe `domain.buffer(-0.3)`, validator blocant în 2 niveluri | C |
| Mediana calculată din axe iese din inter-rândurile corectate în Marcaj | medie | mediu | tăiere cu `domain_eroded`, reconectarea capetelor, validare `covered_by` | C |
| Prea multe noduri GTSP (ținte × candidați > ~1 500) | mică | mic | 2 candidați pe parte, gruparea țintelor sparse, buclă de acoperire | C |
| Nu reacționăm la ținte ascunse definite altfel | medie | până la 15% | goluri + sparse + waste; întrebare pe Slack | C |
| Acces SAM 3 întârziat (gated, aprobare uneori manuală la Meta) | mare | mic | cerere ACUM de pe 2 conturi HF (`hf auth login`); fără acces în 12 h → linear probe + sweep manual; SAM 3 doar bonus | E |
| SAM 3 fără suport oficial CPU/MPS; lent (~3–4 s/crop pe CPU) | medie | mic | doar pe crop-uri de 512 px centrate pe candidați, buget ≤ 30 min; fallback portul MLX (licența declarată apache-2.0 e ambiguă față de SAM License) | A |
| FP la waste pe tuburi, furtunuri, pietre; pragul nu poate fi calibrat pentru recall (0 pozitive în exemple) | medie | mediu | filtre de periodicitate/furtun/NMS, auto-accept dublu, confirmare umană pentru restul | E |
| GSD necunoscut la UAVVaste/DroneWaste → crop-uri pozitive la altă scară | medie | mic | rescalare estimată empiric după dimensiunea obiectelor; pragul calibrat doar pe FP | A |
| NN nu bate clasicul (sau diferențele pe 2 tile-uri sunt zgomot) | medie | mic | rămâne livrabil cu ablația A–F per tile; fuziune doar dacă câștigă pe ambele tile-uri; exportul folosește clasicul | A |
| Instabilitate MPS (ops lipsă, NaN, regresii torch 2.14) | medie | mic | fp32, `PYTORCH_ENABLE_MPS_FALLBACK=1`, smoke-run vineri; fallback CPU sau torch 2.13.x | A |
| Instalări grele (ortools, torch) pe arm64 | mică | mediu | wheel-uri oficiale cp312; fallback python-tsp fără ortools | B |
| Conflicte între versiuni (numpy 2.5 / scikit-image 0.26 / opencv 4.14) sau cu conda base | medie | mediu | `uv lock` + test de import vineri; `conda deactivate`; la nevoie `numpy<2.5` | B |
| Imagine Docker de câțiva GB (torch cu CUDA pe Linux) sau build amd64 blocat | medie | mediu | torch `+cpu` prin index uv cu marker linux; arm64 local, amd64 o singură dată sau în Actions | D |
| OOM în Docker Desktop (18 GB împărțiți cu VM-ul) | medie | mic | `--workers 6` în container; limite VM declarate în README | D |
| Demo web lent sau rupt (311 surse `image`, PMTiles/Range pe Pages sau `http.server`) | medie | mare la pitch | piramidă XYZ + MVT fără compresie, fără PMTiles, maplibre local în `vendor/` | D |
| Licența necomercială LKH (elkai) în repo | mică | mic | elkai doar în scripturile de benchmark, nu dependență; întrebare pe Slack | C |
| Disc plin (18 GB) | mică | mediu | fără rastere float, curățare `work/` | B |
| Oboseala echipei | mare | mediu | ture de somn în plan; freeze duminică 14:00 | toți |

---

## 8. Întrebări pentru Slack-ul organizatorilor (vineri seara)

1. Pentru regula de 2% a traseului, „passable inter-row areas” sunt inter-rândurile **din referința** voastră sau cele din proiectul nostru Marcaj?
2. În exemplul `r006_c004`, pe rândurile cu iarbă, referința are poligoane canopy de până la 55 m (tot rândul), deși regula spune „one canopy = one plant”. Referința ascunsă urmează aceeași convenție?
3. Țintele de inspecție ascunse sunt centrele golurilor ≥ 5 m? Un gol lung are una sau mai multe ținte? Intră și golurile mai scurte, de o plantă lipsă (~1,5–5 m), ca „potentially missing vines”? De răspuns depinde `targets.include_missing`.
4. Limita de 90 MB e în MB sau MiB? Părțile originale au deja 93,2–94,1 MB (part4 = 94 108 721 B = 89,75 MiB), iar cu `annotations.xml` ajung la ~96 MB.
5. „No objects in this frame” se poate seta din `annotations.xml` la import, sau doar manual?
6. Putem urca pre-adnotările în **mai mult de 5 ZIP-uri** (de ex. 7 părți de ~45 de tile-uri, fiecare < 85 MB), cu condiția ca Data să arate 311 fișiere? Ghidul spune „all five parts”, dar cele 5 părți nu mai încap sub limită după adăugarea XML-ului.
7. Gunoiul aflat chiar pe rând (sub vie, la < 0,6 m de axă) apare în referința ascunsă? Filtrul nostru împotriva tuburilor albe îl elimină.
8. E permisă folosirea seturilor externe deschise (UAVVaste Apache-2.0, DroneWaste CC BY 4.0) și a modelelor gated (SAM 3, SAM License) pentru pre-adnotare, cu declarare în README?
9. Codul trimis trebuie să aibă o licență deschisă, fără componente necomerciale? (LKH/elkai ar rămâne doar ca benchmark opțional.)
10. La o eventuală re-rulare, juriul folosește o mașină amd64 sau arm64, cu sau fără GPU? De asta depinde imaginea Docker pe care o publicăm.

---

## 9. Pitch (5 minute)

1. **Problema** (30 s): mentenanța costă 52–80 mii MDL/ha; auditul pentru subvenții și inspecția înseamnă timp de mers.
2. **Pipeline-ul** (60 s): diagrama din §0; hibrid NN + viziune clasică; timpul pentru 311 tile-uri pe un laptop.
3. **Demo live** (2 min 30 s): harta cu blocuri și rânduri, click pe un rând (ID, lungime, disrupted), canopy și inter-rânduri, ținte și waste, traseul cu lungimea și economia în km, minute și procente față de traseul naiv.
4. **Calitate** (40 s): scorul pe exemple cu metricile oficiale; ablația NN; validatorul de traseu (0% risc de eliminare).
5. **Reproductibilitate** (20 s): `make final`, Docker, weights, README.

---

## 10. Surse interne

- Analiza exemplelor și rețeta validată: `../analiza_exemple/RAPORT_exemple_si_reteta.md`, cu scripturile prototip în `../analiza_exemple/scripts/` (`axes.py` = detectorul de axe, `attrs.py` = inter-rânduri și atribute).
- Planul inițial (inventar, ore): `../PLAN_GigaHack_Siret3.md`.
- Anexa de contracte detaliate (câmp cu câmp, CVAT, config complet): `00_contracte.md`. Are câteva verificări utile: START cade în interiorul pasajelor; 28 din 145 de tile-uri analizate au peste 1% zonă neagră; ordinea `R001` = rândul cel mai nordic reproduce numerotarea din exemple. La conflicte, planul de față are prioritate.
- Regulile și descrierea challenge-ului: `../data & info/03_docs/`.
- Cercetarea de vineri seara (7 fire în paralel, toate sursele cu link, riscuri și constatări cu încredere mică): `research/CERCETARE_Siret3.md`.
- Seturi externe evaluate și nefolosite: [Riseholme](https://zenodo.org/records/19234907) (cadre brute, GSD necunoscut, un sezon fără frunze), [Brescia groundcover](https://zenodo.org/records/17701564) (acces restricționat, GSD de ordinul mm), [DroneWaste](https://zenodo.org/records/17045559) pentru antrenarea YOLO (depozite, mAP@50 ≈ 38%). Folosit ca pozitive pentru waste: [UAVVaste](https://github.com/PUTvision/UAVVaste).

---

## 11. Ce s-a schimbat după cercetare

1. **Export: 7 ZIP-uri, nu 5** (§4.13, §0, §5, §6, §7). Cele 5 părți originale au deja 93,2–94,1 MB, iar cu XML-ul ar ajunge la ~96 MB. Limita validatorului e ≤ 85 000 000 B per ZIP, cu layout strict `annotations.xml` + `images/`. Sursă: măsurare locală a `01_tiles/*.zip` și exemplului, [cercetare §6.1](research/CERCETARE_Siret3.md).
2. **Validator CVAT mult mai strict** (§4.13): bijecția nume ↔ fișiere, setul exact de atribute, valorile din listă, geometrie validă și clip la [0,2048]. CVAT nu verifică nimic din acestea la import și poate atașa adnotări pe alt tile ([cvat bindings.py](https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/dataset_manager/bindings.py), [serializers.py](https://raw.githubusercontent.com/cvat-ai/cvat/develop/cvat/apps/engine/serializers.py)).
3. **Waste** (§4.9): SAM 3 folosește doar prompt text + box-uri **negative** din același crop, pe crop-uri de 512 px, pe CPU ([SAM 3 docs](https://huggingface.co/docs/transformers/main/en/model_doc/sam3), [sam3-mac](https://github.com/benreichman/sam3-mac)). CLIP zero-shot nu mai decide; decide un linear probe cu pozitive din [UAVVaste](https://github.com/PUTvision/UAVVaste). Auto-accept doar cu acordul a două modele, restul trece prin confirmare umană.
4. **Fără YOLO pe DroneWaste** (§4.9): domeniu de depozite, mAP@50 ≈ 38% chiar in-domain, fără greutăți publicate ([Zenodo 17045559](https://zenodo.org/records/17045559)).
5. **Legarea rândurilor pe dreapta-suport și `row_id` din grila per bloc** (§4.3, §4.4). Legarea prin capete rupea lanțurile la golurile de lângă margine. Blocurile se taie și la salt de spacing sau de fază ([Delenne](https://hal.inrae.fr/hal-02587901), [Comba 2015](https://dl.acm.org/doi/abs/10.1016/j.compag.2015.03.011)).
6. **Detecția rândurilor** (§4.2, §4.3): spacing estimat din autocorelație, SNR spectral în loc de prominența de 5%, verificare on-lattice, patru filtre vie-vs-livadă (inclusiv armonica `2s`), profil de rezervă pe textură, tracking pentru rândurile curbe ([Delenne](https://hal.inrae.fr/hal-02587805), [Warner](https://www.tandfonline.com/doi/full/10.1080/13658816.2010.510839), [Versatile 2024](https://www.sciencedirect.com/science/article/abs/pii/S0168169924007634)).
7. **NN antrenat vie-vs-iarbă pe tot tile-ul**, cu bandă ignore, label smoothing, early stopping și ablația A–F. Timpul estimat scade de la 1–2 h la ~15–30 min, de confirmat cu smoke-run (§4.6; [PMC11217331](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11217331/), [NCE+RCE](http://proceedings.mlr.press/v119/ma20c/ma20c.pdf), [HF perf MPS](https://huggingface.co/docs/transformers/v4.49.0/perf_train_special)).
8. **Traseu ca GTSP** (§4.11): fiecare țintă are candidați în ambele inter-rânduri vecine (`AddDisjunction`), plus buclă de acoperire, string pulling, costuri în cm și `ortools==9.15.6755` ([OR-Tools](https://developers.google.com/optimization/routing/routing_options), [Carrabs](https://www.sciencedirect.com/science/article/abs/pii/S0305054816302179)). Fallback python-tsp, iar elkai doar ca benchmark ([python-tsp](https://github.com/fillipe-gsm/python-tsp), [elkai](https://github.com/fikisipi/elkai)).
9. **Validatorul de traseu** (§4.11): nu mai cere `is_simple`, care ar fi blocat orice tur dus-întors valid. Verifică în 2 niveluri, cu `covered_by` pe domeniul pregătit și `difference(grid_size=0.001)` ([shapely.prepare](https://shapely.readthedocs.io/en/stable/reference/shapely.prepare.html)).
10. **Python 3.12 + uv, fără conda, versiuni fixate** (§4.1, §4.15). Pachetele cheie cer ≥ 3.12 ([rasterio](https://pypi.org/pypi/rasterio/json), [pyproj](https://pypi.org/pypi/pyproj/json)). Docker în două etape, cu torch `+cpu` ([uv Docker](https://docs.astral.sh/uv/guides/integration/docker/), [uv PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)).
11. **Paralelism și benchmark** (§4.1): pool spawn explicit, 1 thread OpenCV/BLAS per worker, NN pe MPS doar în procesul principal. M3 Pro are 5P + 6E nuclee, deci numărul de workeri se alege prin sweep ([multiprocessing](https://docs.python.org/3/library/multiprocessing.html), [opencv #15277](https://github.com/opencv/opencv/issues/15277), [MPS env](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html)).
12. **Web** (§4.14): piramidă XYZ cu gdal2tiles în loc de 311 surse `image`, canopy ca MVT cu tippecanoe, fără PMTiles, `maplibre-gl@5.24.0` local ([MapLibre large data](https://maplibre.org/maplibre-gl-js/docs/guides/large-data/), [gdal2tiles](https://gdal.org/en/stable/programs/gdal2tiles.html), [PMTiles #584](https://github.com/protomaps/PMTiles/issues/584)).
13. **Goluri și ținte** (§4.8, §4.10): închidere 1D de 0,4 m pe profilul de ocupare și ținte noi `missing` pentru golurile de ~2–5 m, sub flag ([Primicerio 2017](https://www.tandfonline.com/doi/full/10.1080/22797254.2017.1308234)). Raza de vizitare de 2 m e confirmată în descrierea challenge-ului.
14. **Riscuri și întrebări** (§7, §8): 19 riscuri noi și 5 întrebări noi pe Slack (numărul de ZIP-uri, gunoiul de pe rând, seturile și modelele externe, licența codului, arhitectura pentru re-rulare).
