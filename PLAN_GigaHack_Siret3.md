# Vineyard AI Field Challenge (Sireț3): inventar și plan

Deadline: **duminică 27.09, ora 15:00** (repo + proiect Marcaj). Acum: vineri seara, ~44 h rămase.

## 1. Ce avem în folder (`Vin Gigahack/data & info`, 1 GB)

| Folder | Conținut | Observații |
|---|---|---|
| `01_tiles/` | 5 ZIP-uri (74+71+78+76+12 = **311 tile-uri** GeoTIFF, 2048×2048, 2.5 cm/px, EPSG:32635) + `overview.png` | Numele: `siret3_rXXX_cYYY.tif`. Se urcă în Marcaj **neschimbate**. |
| `02_route/` | `start.geojson` (Point 629504.70, 5220250.75, tile r018_c010), `passages.geojson` (1 MultiPolygon), `forbidden.geojson` (1 MultiPolygon, satul), `study_area.geojson` | Toate în EPSG:32635. |
| `03_docs/` | Descrierea challenge-ului, regulile de adnotare, ghidul Marcaj | Rezumat mai jos. |
| `04_source/` | Ortomozaicul complet (659 MB, EPSG:4326) | Doar pentru antrenare, nu e obligatoriu. |
| `05_examples/` | ZIP CVAT cu 2 tile-uri adnotate (650 canopy, 51 rânduri, 49 interrow, 0 waste) + 2 preview-uri | Singurul „ground truth”; e și șablonul exact de upload. |

Mașina: Apple M3 Pro, 18 GB RAM, **doar 18 GB liberi pe disc**. Python 3 (anaconda), `pdftotext`; lipsesc rasterio / shapely / geopandas / torch, trebuie instalate.

## 2. Ce contează din PDF-uri

**Scoring (85% automat pe un subset ascuns de tile-uri, 15% juriu):**
- 25% canopy: 0.6 × IoU + 0.4 × F1 per plantă (IoU ≥ 0.5). Canopy pe tile-uri fără vie = penalizare.
- 25% traseu: 15% acoperire (țintă vizitată dacă treci la ≤ 2 m), 10% eficiență (acordată de la 90% acoperire). **Scor 0** dacă > 2% din traseu iese din inter-rânduri/pasaje sau nu revine la start (5 m).
- 15% axe + atribute: 8% F1 axe (80% din lungime la ≤ 0.4 m), 5% atribute, 2% gruparea pe `vineyard_id`.
- 15% inginerie: arhitectură, robustețe, scalabilitate, timp de procesare declarat în README.
- 10% waste: F1 bbox la IoU ≥ 0.3; un fals pozitiv costă cât un ratat.
- 10% numărători și măsurători (blocuri, rânduri, arie canopy, arie inter-rând, lungime totală rânduri), **calculate de organizatori din adnotările din Marcaj**.

**Reguli care ne dictează planul:**
1. Pre-adnotările se importă **o singură dată**, înainte de Publish, pentru toate cele 311 tile-uri. După Publish se corectează doar manual. Până la Publish se poate face „Remove all” și reimport.
2. ID-urile (`vineyard_id`, `row_id`) trebuie să fie aceleași peste marginile tile-urilor, deci le generăm global pe tot mozaicul, nu per tile.
3. Un tile gol se bifează „No objects in this frame”. Doar job-urile **Submitted** se punctează (63 de job-uri a câte 5 tile-uri).
4. O plantă = un poligon. Un rând = o polilinie per tile, trasă prin goluri. `disrupted` = gol ≥ 5 m în acel tile.
5. Inter-rândul merge de la marginea canopy-ului la marginea canopy-ului (în exemplu sunt simple patrulatere), niciodată în afara rândurilor extreme.
6. Bloc = plantație conexă (goluri < 5 m); un drum separă întotdeauna blocurile.
7. Repo-ul trebuie să conțină în rădăcină: `route.geojson` (un LineString cu `length_m`), `measurements.csv`, `README.md` (pași, dependențe fixate, timp și hardware, link weights, link UI), cod. Dockerfile e un plus.

## 3. Structura propusă a proiectului

```
vineyard-siret3/
├── README.md  route.geojson  measurements.csv
├── pyproject.toml / requirements.lock   Dockerfile   Makefile
├── configs/default.yaml          # praguri: ExG, spațiere rânduri, 5 m gol, buffer-e
├── data/  (gitignored)           # symlink spre „data & info”, tile-uri dezarhivate
├── src/vineyard/
│   ├── io/        tiles.py (citire, pixel↔UTM), cvat_write.py, cvat_read.py (export Marcaj)
│   ├── canopy/    vegetation mask (ExG/ML) + împărțire pe plante
│   ├── rows/      orientare + detecție axe per tile, legare globală și row_id
│   ├── blocks/    componente conexe (închidere 5 m, tăiate de pasaje) → vineyard_id
│   ├── interrow/  benzi între axe vecine minus canopy, clasificare cover
│   ├── waste/     detector (conservator)
│   ├── targets/   puncte de inspecție: goluri ≥ 5 m în rânduri + waste
│   ├── measure/   measurements.csv din adnotări (ale noastre sau exportul Marcaj)
│   └── route/     graf de mers (pasaje + axe inter-rând), matrice distanțe, TSP, validare 2%
├── scripts/   run_pipeline.py, build_upload_zips.py, from_marcaj_export.py, benchmark.py
├── web/       hartă statică MapLibre/Leaflet + GeoJSON (deploy GitHub Pages/Vercel)
└── tests/     round-trip CVAT pe cele 2 exemple, metrici de scoring re-implementate
```

Principiul: toată geometria se lucrează în UTM (EPSG:32635) pe mozaic, apoi se taie per tile și se convertește în pixeli doar la scrierea CVAT. Aceleași module de măsurare și traseu rulează la final pe exportul corectat din Marcaj.

## 4. Abordarea tehnică (pragmatică, pentru 44 h)

- **Canopy:** index de vegetație (ExG) + masca rândurilor ca să elimine iarba dintre rânduri, apoi împărțire pe plante prin minime de-a lungul axei (sau la 1.0–1.5 m). Opțional SAM/U-Net fin-tunat pe cele 2 tile-uri exemplu, dacă baseline-ul e slab. Tile-urile fără vie trebuie să iasă goale (filtru: fără periodicitate de rânduri = fără canopy).
- **Rânduri:** orientare per tile din FFT/Hough pe masca de vegetație, profile perpendiculare → axe; legare între tile-uri în UTM ca să rezulte `row_id` global. `row_structure` din lungimea maximă a golului de canopy de-a lungul axei.
- **Inter-rânduri:** poligon între două axe vecine, restrâns la marginile canopy, tăiat la capetele rândului mai scurt. `interrow_cover` din fracția de vegetație (< 25% bare_soil, > 75% vegetation, altfel mixed).
- **Waste:** detector conservator (pete albe/colorate care nu sunt tuburi aliniate pe rând) + completare manuală; valorează doar 10%, nu merită mult timp.
- **Traseu:** graf din scheletul pasajelor + axele inter-rândurilor (nu marginile, ca să rămânem sub 2% în afara zonei), legături capăt-de-rând ↔ pasaj; distanțe Dijkstra între ținte; TSP cu OR-Tools; întoarcere la start; validare automată a procentului în afara zonei permise.
- **Web UI:** hartă cu tile-uri/overview, straturi canopy/rânduri/inter-rânduri/waste/ținte, traseul cu lungimea, tabele cu blocuri, rânduri, lungimi și arii.

## 5. Planul pe ore

| Când | Ce | Poarta |
|---|---|---|
| **Vin 19–23** | Mediu Python (rasterio, shapely, geopandas, opencv, ortools, torch MPS), dezarhivare tile-uri într-un subfolder nou, index spațial al tile-urilor. Writer CVAT testat: re-scriem exemplul și verificăm că e identic. Primul baseline canopy + rânduri pe cele 2 exemple, cu metricile din PDF. | Round-trip CVAT OK |
| **Vin 23 – Sâm 10** | Rânduri globale + ID-uri, blocuri, inter-rânduri, atribute, waste. Rulare pe toate 311, vizualizare pe overview, corecturi de praguri. | Scor decent pe exemple, fără canopy în sat |
| **Sâm 10–12** | Construim cele 5 ZIP-uri (< 90 MB), le urcăm, verificăm rapoartele de import și **311 fișiere**, apoi **Publish**. | **Hard gate: Publish până sâmbătă ~12:00** |
| **Sâm 12 – Dum 11** | Echipa corectează în Marcaj pe petice vecine (ID-urile deja globale). În paralel: traseu, măsurători, Web UI, README, Dockerfile, benchmark de timp. | Toate job-urile trimise treptat |
| **Dum 11–13** | Export din Marcaj → recalculăm `measurements.csv`, țintele și `route.geojson` pe adnotările corectate; deploy UI. | Validatorul de traseu trece |
| **Dum 13–15** | Buffer. In progress = 0 în Marcaj, repo final, link UI și weights în README, pregătire pitch. | Submit |

## 6. Riscuri

- Publish prea devreme sau fără toate 5 părțile: pierdem pre-adnotările. Nu publicăm fără verificarea „311 files”.
- Traseu cu > 2% în afara zonei = 0 pe 25%. Validăm automat înainte de commit.
- Disc: 18 GB liberi ajung pentru tile-uri (~1 GB) și modele mici, dar nu pentru multe intermediare pe mozaicul complet. Salvăm vectori, nu rastere mari.
- Canopy lipite (vie matură): împărțirea la distanța de plantare e esențială pentru F1-ul per plantă.
