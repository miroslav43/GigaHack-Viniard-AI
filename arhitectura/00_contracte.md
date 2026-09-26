# 00 · Contracte comune (v1.0)

Vineyard AI Field Challenge (Sireț3), Deeptech GigaHack 2026. Documentul fixează interfețele pe care le folosesc toate subsistemele: coordonatele, straturile vectoriale, ID-urile, graful de stagii, formatul CVAT, livrabilele, configul și logurile. Celelalte documente din `arhitectura/` pornesc de aici și nu redefinesc nimic din ce e mai jos.

- **Stare:** înghețat pentru implementare vineri 25.09, ora 23:00. O schimbare după această oră înseamnă `CONTRACT_VERSION` mărit în `src/vineyard/contracts/__init__.py`, o linie în secțiunea 13 și un mesaj către echipă.
- **Ce e verificat:** tag-urile GeoTIFF, formula grilei (pe 147 de tile-uri, apoi la ingest pe toate 311), convențiile XML CVAT din exemple, numerotarea rândurilor, zona neagră și dimensiunile ZIP-urilor. Tot ce e marcat „(măsurat)” a fost rulat pe date. Scripturile de verificare sunt în scratchpad (`geotags.py`, `grid.py`, `nodata.py`).

---

## 0. Cele 12 reguli, pe scurt

1. **Toată geometria se ține în EPSG:32635, în metri, float64.** Pixelii apar doar la granița CVAT (export și import) și în rasterele per tile.
2. **Pixelii au convenția de colț:** `PixelIsArea`. Coordonata CVAT continuă `(u, v)` merge de la 0 la 2048. Centrul pixelului cu indexul `(i, j)` e la `(i + 0.5, j + 0.5)`.
3. **`X = x0 + 0.025·u`, `Y = y0 − 0.025·v`**, cu `x0 = 628992.0 + 51.2·col` și `y0 = 5221222.4 − 51.2·row`.
4. **Format canonic: GeoParquet, un fișier per strat.** Pentru QGIS și livrabile exportăm GeoJSON în EPSG:32635, cu membrul `crs` ca la organizatori. Web-ul primește EPSG:4326.
5. **AnnSet-ul e singura interfață dintre percepție și restul lanțului.** Are 4 straturi per tile: `canopies`, `row_pieces`, `interrow_pieces`, `waste`. Importul din Marcaj produce un AnnSet identic ca schemă, iar țintele, domeniul de mers, traseul, măsurătorile și web-ul consumă numai AnnSet-uri.
6. **ID-uri:** `V01`, `V01-R001`, `V01-I001`, `V01-R001@siret3_r021_c012`, `siret3_r021_c012:C0001`, `W0001`, `T-GAP-0001`.
7. **Orice obiect poartă `source`, `run_id`, `model_version`, `confidence` și `qa_flags`.**
8. **Stagiile sunt idempotente.** Cheia de cache e `sha1(STAGE_VERSION ‖ config-ul stagiului ‖ cheile intrărilor)`, iar scrierile sunt atomice.
9. **ZIP-urile de upload se reîmpachetează la cel mult 85 MiB.** Cele 5 părți originale au deja 88,9–89,75 MiB fără adnotări (măsurat).
10. **Poligoanele exportate în CVAT n-au găuri și nu repetă primul vârf.** Exteriorul e CCW în UTM, ca în exemple (măsurat).
11. **Există un singur config YAML**, validat cu pydantic. Valorile implicite sunt faptele măsurate pe exemple.
12. **Logurile sunt JSONL cu `run_id`, `stage` și `tile_id`.** Timpii per stagiu ajung în `metrics/timings.json`, iar README-ul îi preia de acolo.

---

## 1. Convenții de coordonate (verificate pe date)

### 1.1 Tag-urile GeoTIFF (măsurat cu `tifffile`)

| Tag | Valoare (ex. `siret3_r006_c004.tif`) | Ce înseamnă |
|---|---|---|
| 33550 ModelPixelScale | `(0.025, 0.025, 0.0)` | 2,5 cm/px pe X și pe Y |
| 33922 ModelTiepoint | `(0, 0, 0, 629196.8, 5220915.2, 0)` | punctul raster (0,0) corespunde lui (X, Y) = (629196.8, 5220915.2) |
| GTRasterTypeGeoKey | `RasterPixelIsArea (1)` | raster (0,0) e **colțul** stânga-sus al primului pixel, nu centrul lui |
| ProjectedCSTypeGeoKey | 32635, unități 9001 (metru) | WGS 84 / UTM 35N |
| Imagine | 2048×2048×3 uint8, JPEG (compression 7, YCbCr), bucăți interne 256×256, o singură pagină | fără canal alfa, fără tag nodata (42113 lipsește) |

### 1.2 Formula pixel ↔ UTM (contract)

```
X = x0 + GSD · u          u = (X − x0) / GSD
Y = y0 − GSD · v          v = (y0 − Y) / GSD          GSD = 0.025 m
```

`(u, v)` e coordonata CVAT continuă, în intervalul [0, 2048]. Exemplele confirmă că e aceeași convenție ca GeoTIFF-ul: toate cele 51 de axe de rând au capete exact la `0.0` sau `2048.0`, iar canopy-urile tăiate de margine au vârfuri exact pe `0.0`/`2048.0` (măsurat). Nu e nevoie de niciun offset de ½ pixel între GeoTIFF și CVAT.

### 1.3 Grila globală (verificată)

```
x0(col) = 628992.0 + 51.2 · col       y0(row) = 5221222.4 − 51.2 · row
tile box = (x0, y0 − 51.2, x0 + 51.2, y0)       # (minx, miny, maxx, maxy)
```

- Formula dă exact tiepoint-ul (|Δ| < 1e-6 m) pe toate cele 145 de tile-uri din părțile 1–2 și pe cele 2 exemple. `ingest` o re-verifică pe toate 311 și se oprește cu eroare la orice abatere.
- Grila are rândurile `r005..r039` (35) și coloanele `c000..c033` (34), cu 311 tile-uri ocupate. `row` crește spre sud, `col` spre est.
- Tile-urile sunt adiacente exact, fără suprapunere. Suma ariilor dă exact `study_area.geojson` (81,53 ha = 311 × 2621,44 m²). Vârfurile lui `study_area` cad exact pe grilă.
- **Coordonatele de mozaic** (pentru operații raster pe blocuri): `U = 2048·col + u`, `V = 2048·row + v`. Mozaicul întreg ar avea 69632 × 71680 px la 2,5 cm; la 0,2 m/px (factor 8) are 8704 × 8960 px, circa 78 MB pentru o mască uint8.
- START (629504.70, 5220250.75) cade în `siret3_r018_c010` la `(u, v) = (28.0, 2002.0)`, în colțul SV al tile-ului, la granița cu `r019` și `c009`. Punctul e în interiorul `passages.geojson` (verificat).
- Convergența grilei UTM față de nordul geografic e γ ≈ (28,707° − 27°)·sin 47,12° ≈ **1,25°**. Nu contează în UTM. Contează doar când web-ul pune ortofotografia peste o hartă în Web Mercator (secțiunea 7).

### 1.4 Trei sisteme de pixeli (sursa clasică de erori de ½ pixel)

| Sistem | Unde apare | Conversie spre CVAT `(u, v)` |
|---|---|---|
| Index numpy `arr[j, i]` (rând j, coloană i) | măști, `np.nonzero` | `u = i + 0.5`, `v = j + 0.5` (centrul pixelului) |
| Contur OpenCV `(x, y)` = `(i, j)` | `cv2.findContours`, `cv2.fitLine` pe `np.nonzero` | `+0.5` pe ambele axe |
| CVAT continuu `(u, v)` | XML, `utm_to_px` | identitate |

Reguli obligatorii:
- **Vectorizare (mască → poligon):** `findContours` trece prin centrele pixelilor de margine și pierde circa ½ px pe contur. La un canopy median (0,47 m² ≈ 752 px, perimetru ≈ 120 px) pierderea e de ~8% din arie. De aceea se face `+0.5` și apoi `buffer(canopy.vector_outset_px · GSD)`. Implicit e 0,5 px, iar valoarea se reglează pe exemple (vezi `vector_outset_px`). După outset, canopy-ul se intersectează din nou cu coridorul ca să nu atingă inter-rândul.
- **Rasterizare (poligon CVAT → mască):** `cv2.fillPoly(mask, [np.round((uv − 0.5) * 16).astype(np.int32)], 1, shift=4)`. Nu se folosește `np.round(uv)` direct, cum făcea prototipul (eroare de 1,25 cm).
- **Metricile oficiale din `eval` se calculează vectorial, cu shapely în UTM:** IoU pe uniuni, IoU pe perechi, acoperirea axelor la 0,4 m. Rasterul e permis doar ca aproximare rapidă și e marcat ca atare.

### 1.5 Unghiuri, normale, orientare

- `angle_deg` se măsoară **în UTM**: `atan2(dY, dX)` în grade, mod 180, în intervalul [0, 180). Estul are 0°, iar unghiul crește în sens trigonometric. În pixeli axa v e inversată, deci `angle_px = −angle_utm` (mod 180).
- **Normala canonică a unui bloc** e `n = (−sin θ, cos θ)`, întoarsă astfel încât `n_y > 0` (spre nord). Dacă `|n_y| < 0,01` (rânduri exact N–S), se alege `n_x > 0`.
- **Numerotarea rândurilor:** R001 este rândul cu `n·c` maxim, adică cel mai nordic pe normală (c = centroidul axei). Regula reproduce exact ordinea `R01..R25` și `R01..R26` din ambele exemple (verificat; unghiuri UTM 133,3° și 113,2°).
- **Poligoane:** exteriorul e CCW în UTM (`shapely.orient(p, 1.0)`), iar găurile sunt CW. În pixeli asta înseamnă arie shoelace negativă, exact ca toate cele 699 de poligoane din exemple (măsurat).

### 1.6 Tăierea la tile și zona validă

- Fiecare tile are două decupaje: `tile_box` (pătratul de 51,2 m) și `tile_valid` (partea cu imagine, stratul 2.5.2). **Decupajul efectiv este `clip = tile_box ∩ tile_valid`.**
- **Poligoanele** (canopy, inter-rând) se intersectează cu `clip`. Un MultiPolygon rezultat devine mai multe poligoane CVAT cu aceleași atribute.
- **Axele** se intersectează cu `tile_box`. Dacă zona neagră le taie, se păstrează o singură polilinie, de la primul la ultimul vârf valid de-a lungul axei. Regula „o polilinie per rând per tile” are prioritate față de o porțiune scurtă peste negru.
- **Waste:** cutia se taie la `tile_box`. Un obiect tăiat de marginea tile-ului dă două cutii, câte una în fiecare tile, cu același `waste_id` și sufixele `a`/`b`.
- **Prelungirea axelor „până la margine”** (regula de 3 m din prototip) se face până la marginea lui `clip`, nu până la marginea pătratului.

### 1.7 Zona neagră (nodata) pe tile-urile de margine (măsurat)

- 28 din 145 de tile-uri analizate au peste 1% negru. Cele mai rele: `r015_c004` 83,7%, `r006_c002` 82,6%, `r023_c010` 80,2%.
- Negrul e aproape tot `(0,0,0)` exact. Fracția cu `max(R,G,B) ≤ 8` depășește fracția de zerouri exacte cu cel mult 0,4 puncte procentuale. Diferența vine din „ringing”-ul JPEG, lat de câțiva pixeli la graniță (valori 1–30 pe maxim de canal).
- **Rețeta (`tile_prep`):**
  1. `nd = max(R,G,B) ≤ nodata.max_rgb` (10).
  2. Se păstrează componentele conexe care ating marginea tile-ului și au aria ≥ `nodata.min_area_m2` (0,5 m²). Umbrele adânci din interior rămân astfel date valide.
  3. Închidere morfologică cu 5 px, apoi dilatare cu `nodata.dilate_px` (4 px, pentru ringing).
  4. `valid = ¬nd`, vectorizat cu `approxPolyDP` la 2 px și convertit în UTM ca `tile_valid`.
- Masca de vegetație se înmulțește cu `valid` erodat cu încă 4 px, ca franjurile de crominanță JPEG să nu devină „verde”.
- Un tile cu `valid_frac < 0,02` primește statusul `empty_nodata` și nu trece prin percepție. Rămâne totuși în ZIP, cu `<image>` gol.
- În calculul golurilor de rând, porțiunile peste negru sunt „necunoscute”, nu „lipsă”. Nicio țintă nu se plasează în negru.

### 1.8 Precizie și rotunjiri

| Unde | Rotunjire |
|---|---|
| GeoParquet | float64, fără rotunjire |
| GeoJSON 32635 (exporturi, QA) | 3 zecimale (mm) |
| `route.geojson` | 2 zecimale (cm). Primul și ultimul vârf sunt **exact** coordonatele START. |
| CVAT XML | 1 zecimală de pixel (0,1 px = 2,5 mm), ca în exemple. Vârfurile consecutive identice după rotunjire se elimină. |
| GeoJSON 4326 (web) | 7 zecimale (~1 cm) |

---

## 2. Modelul de date canonic

### 2.1 Format și locație

- **Canonic: GeoParquet**, câte un fișier per strat: `work/runs/<run_id>/layers/<strat>.parquet`, geometrie WKB, CRS EPSG:32635 în metadatele `geo`. Motivele:
  - `pyarrow` 14 e deja instalat, iar `geopandas.read_parquet`/`to_parquet` cer doar pyarrow, fără GDAL ([doc read_parquet](https://geopandas.org/en/stable/docs/reference/api/geopandas.read_parquet.html)).
  - Tipurile coloanelor (int, bool, enum) se păstrează.
  - Un fișier per strat înseamnă un singur stagiu proprietar per fișier, deci cache simplu și fără lock-uri SQLite între procese.
- **De instalat vineri seara** (roțile pip au GEOS și GDAL incluse): `shapely>=2.0`, `pyproj`, `pyogrio`, `geopandas>=1.0`. `pyogrio` e dependență obligatorie a geopandas 1.x ([install](https://geopandas.org/en/stable/getting_started/install.html)). Tot atunci: `ortools` pentru TSP, opțional.
- **Exporturi derivate:**
  - `exports/geojson/<strat>.geojson` în EPSG:32635, cu `"crs": {"type":"name","properties":{"name":"urn:ogc:def:crs:EPSG::32635"}}`, identic cu fișierele organizatorilor. `id`-ul feature-ului este cheia primară a stratului.
  - Opțional, `exports/qgis/bundle.gpkg` (prin pyogrio), doar pentru inspecție.
- **Rezervă dacă shapely/geopandas nu se instalează:** drumul critic al exportului CVAT (termen: sâmbătă 12:00) nu depinde de geopandas. `cvat/writer.py` primește numpy, iar tăierea la dreptunghi se face cu Sutherland–Hodgman în numpy. Straturile se pot scrie ca GeoJSON cu `json` simplu, prin `geo/vector_io.py`, care ascunde implementarea.

### 2.2 Coloane comune (proveniență), prezente în orice strat produs de noi

| Câmp | Tip | Valori |
|---|---|---|
| `source` | str (enum) | `model` · `marcaj` · `reference` (exemplele organizatorilor) |
| `run_id` | str | `YYYYMMDDTHHMM-<source>-<hash6>`, de ex. `20260926T0310-model-a1b2c3` |
| `model_version` | str | `pipe@<pkg_ver>+<git7>;nn=<name>@<ver>+<sha8>;waste=<det>@<ver>`. Pentru Marcaj: `marcaj-export@<sha8 al fișierului>`. |
| `confidence` | float32 [0,1] | scorul modelului. `1.0` pentru `marcaj` și `reference`. |
| `qa_flags` | str | coduri separate prin `;` (secțiunea 2.7), gol dacă nu e nimic |

### 2.3 Enumerări (`contracts/enums.py`, scrise exact așa, cu litere mici)

```
Label          = vineyard | waste | row | interrow_area
RowStructure   = regular | disrupted | unassessable
InterrowCover  = bare_soil | vegetation | mixed | unassessable
Source         = model | marcaj | reference
TargetKind     = row_gap | row_end_short | missing_row | waste | other
TileStatus     = ok | empty_nodata | no_vineyard | failed
Severity       = error | warning | info
EdgeKind       = interrow_centerline | passage_centerline | connector | target_spur
```

### 2.4 ID-uri (`contracts/ids.py`: formatare, parsare, regex)

| Obiect | Format | Regex (ieșirea modelului) | Regula de atribuire, deterministă |
|---|---|---|---|
| tile | `siret3_r021_c012` | `^siret3_r\d{3}_c\d{3}$` | numele fișierului fără `.tif` |
| bloc | `V01` | `^V\d{2,3}$` | sortare după `(−round(cy), round(cx))` a `representative_point()`: V01 e cel mai nordic. Lățimea e de 2 cifre, 3 dacă sunt peste 99 de blocuri. |
| rând (fizic) | `V01-R001` | `^V\d{2,3}-R\d{3}$` | în bloc, `n·c` descrescător (secțiunea 1.5), R001 cel mai nordic. Mereu 3 cifre (un bloc poate avea peste 100 de rânduri). |
| inter-rând | `V01-I001` | `^V\d{2,3}-I\d{3}$` | `I k` e între `R k` și `R k+1` |
| bucată de rând (per tile) | `V01-R001@siret3_r021_c012` | `<row_id>@<tile_id>` | dacă în Marcaj apar duplicate: sufix `#2`, `#3` |
| bucată de inter-rând | `V01-I001@siret3_r021_c012` | `<interrow_id>@<tile_id>` | din Marcaj, până la legare: `siret3_r021_c012:I001` |
| canopy | `siret3_r021_c012:C0001` | `^<tile_id>:C\d{4}$` | în tile, sortare după `(row_index, poziția de-a lungul axei)`. Din Marcaj: ordinea din XML. |
| candidat de rând | `siret3_r021_c012:K07` | `^<tile_id>:K\d{2}$` | ordinea vârfurilor din profil |
| waste | `W0001` | `^W\d{4}[ab]?$` | global, sortare după `(tile_id, ytl, xtl)` |
| țintă | `T-GAP-0001` | `^T-(GAP|END|MRW|WST|OTH)-\d{4}$` | per tip, sortare după `(vineyard_id, row_id, poziție)`. `WST` păstrează ordinea `waste_id`. |
| nod / muchie de graf | int64 | — | ordinea de creare, stabilă la aceeași intrare |

- **Valorile ID-urilor nu se punctează, doar gruparea.** Prefixul `V01-` din `row_id` e o convenție. Importul din Marcaj acceptă orice text nevid și **nu redenumește nimic**, pentru că organizatorii numără valorile brute. QA doar semnalează problemele.
- **După Publish, ID-urile sunt înghețate.** `export_cvat` scrie `id_registry.json` cu ultimul `V` folosit și ultimul `R` per bloc. Un bloc sau un rând adăugat manual în Marcaj primește următorul număr liber (de ex. `V14`, `V03-R118`).

### 2.5 Registrul straturilor

Cheie: **T** = per tile, **B** = per bloc, **G** = global. „AnnSet” = stratul face parte din setul de adnotări (secțiunea 2.6).

| Strat | Geometrie | PK | Produs de | Consumat de |
|---|---|---|---|---|
| `tile_index` | Polygon (tile_box) | `tile_id` | `ingest` | toate |
| `tile_valid` | MultiPolygon | `tile_id` | `tile_prep` | percepție, export, țintă, web |
| `tile_status` | Polygon (tile_box) | `tile_id` | `assemble` / `import_marcaj` | QA, web, `review_queue` |
| `in_passages`, `in_forbidden`, `in_study_area`, `in_start` | ca la sursă | `fid` | `ingest` | `blocks`, `passable`, `route`, web |
| `row_candidates` | LineString | `cand_id` | `rows_detect` (T) | `rows_link` |
| `rows` | LineString | `row_id` | `blocks` (model) / `derive` (orice AnnSet) | canopy, inter-rând, țintă, măsurători, web |
| `blocks` | MultiPolygon | `vineyard_id` | `blocks` / `derive` | waste (vineyard_id), țintă, web, măsurători |
| **`canopies`** (AnnSet) | Polygon | `canopy_id` | `canopy` (T) / import | măsurători, țintă, passable, web |
| **`row_pieces`** (AnnSet) | LineString | `piece_id` | `row_attrs` (T) / import | `derive`, măsurători, export |
| `interrows` | Polygon | `interrow_id` | `interrow` (B) / `derive` | passable, web |
| **`interrow_pieces`** (AnnSet) | Polygon | `piece_id` | `interrow` (B→T) / import | măsurători, passable, export |
| `waste_candidates` | Polygon (cutie) | `waste_id` | `waste` (T) | revizie, prag de export |
| **`waste`** (AnnSet) | Polygon (cutie) | `waste_id` | `waste` / import | țintă, export, web |
| `targets` | Point | `target_id` | `targets` | route, web |
| `target_extents` | LineString | `target_id` | `targets` | route (vizitarea unui gol lung) |
| `passable_parts` | Polygon | `part_id` | `passable` | validare, web |
| `passable_domain` | MultiPolygon (1 rând) | `domain_id` | `passable` | route, validare |
| `walk_nodes` | Point | `node_id` | `passable` | route |
| `walk_edges` | LineString | `edge_id` | `passable` | route |
| `route` | LineString (1 rând) | `route_id` | `route` | publish, web |
| `route_stops` | Point | (`route_id`, `seq`) | `route` | web |
| `qa_issues` | Point | `issue_id` | toate stagiile / `qa` | echipa de corectare, web |

#### 2.5.1 `tile_index` (static, un singur exemplar, în `work/tile_index.parquet`)

| Câmp | Tip | Descriere |
|---|---|---|
| `tile_id` | str | PK |
| `file_name` | str | `siret3_r021_c012.tif` |
| `grid_row`, `grid_col` | int16 | din nume, verificate cu tag-urile |
| `x0`, `y0`, `x1`, `y1` | float64 | UL = tiepoint, LR = UL + (51,2; −51,2) |
| `gsd_m` | float64 | 0.025 |
| `width_px`, `height_px` | int16 | 2048 |
| `src_zip` | str | partea originală (`…part1of5.zip`) |
| `path` | str | calea locală absolută (`work/tiles/<file>`) |
| `sha256` | str | hash-ul octeților fișierului. Garantează că tile-ul ajunge neschimbat în upload. |
| `file_size` | int64 | octeți |
| `nodata_frac` | float32 | din `tile_prep` |
| `valid_area_m2` | float64 | aria lui `tile_valid` |

#### 2.5.2 `tile_valid`: `tile_id`, `valid_frac` (float32), geometria = partea cu imagine a tile-ului.

#### 2.5.3 `tile_status` (per rulare)

`tile_id`, `status` (TileStatus), `has_vineyard` (bool), `n_canopies`, `n_row_pieces`, `n_interrow_pieces`, `n_waste` (int32), `veg_frac` (float32), `vineyard_score` (float32, scorul NN la nivel de tile), `upload_zip` (str), `image_id` (int32, indexul în ZIP), `review_priority` (int8, 1 = urgent … 3), `issues` (str), plus coloanele comune.

#### 2.5.4 `row_candidates` (intern, doar din model)

| Câmp | Tip | Descriere |
|---|---|---|
| `cand_id`, `tile_id` | str | |
| `angle_deg` | float32 | convenția UTM (secțiunea 1.5) |
| `offset_m` | float32 | offset perpendicular cu semn față de centrul tile-ului |
| `length_m` | float32 | |
| `support_frac` | float32 | cât din axă are vegetație în coridorul ±0,30 m |
| `width_med_m` | float32 | lățimea mediană a vegetației, transversal (viță < 1 m, livadă 2–4 m) |
| `local_spacing_m` | float32 | distanța până la vecini (viță 2,5–2,8 m) |
| `vine_score` | float32 | media `canopy_prob` din NN pe coridor |
| `rejected_reason` | str \| null | `angle_gate`, `orchard_width`, `spacing`, `short`, `nodata` … |

#### 2.5.5 `rows` (un rând fizic = un `row_id`)

| Câmp | Tip | Descriere |
|---|---|---|
| `row_id` | str | PK |
| `vineyard_id` | str | |
| `row_index` | int16 | 1…n în bloc (secțiunea 1.5) |
| `length_m` | float64 | **Σ lungimilor bucăților** (așa calculează organizatorii din adnotări) |
| `extent_m` | float64 | lungimea liniei contopite |
| `n_pieces` | int16 | |
| `tile_ids` | str | tile-urile atinse, separate prin virgulă, în ordinea axei |
| `angle_deg` | float32 | |
| `spacing_prev_m`, `spacing_next_m` | float32 | distanța până la R(k−1) și R(k+1) (NaN la margine) |
| `max_gap_m` | float32 | cel mai mare gol pe rândul global, fără capete |
| `n_gaps_ge5` | int16 | numărul de goluri de cel puțin 5 m |
| `structure_any` | str | `disrupted` dacă o bucată e disrupted, `unassessable` dacă toate sunt, altfel `regular` |

#### 2.5.6 `blocks`

`vineyard_id` (PK), `n_rows`, `n_row_pieces`, `n_canopies`, `n_tiles` (int32), `row_length_m`, `canopy_area_m2`, `interrow_area_m2`, `outline_area_m2` (float64), `angle_deg`, `spacing_med_m` (float32), `is_garden` (bool: parcelă din sat, cel puțin 3 rânduri), plus coloanele comune. Geometria e conturul: `unary_union(buffer(obiecte, 2.5)).buffer(−2.5)`.

#### 2.5.7 `canopies` (AnnSet)

| Câmp | Tip | Descriere |
|---|---|---|
| `canopy_id` | str | PK |
| `tile_id`, `vineyard_id` | str | |
| `row_id` | str \| null | rândul asociat, **intern**. Nu se exportă în CVAT, unde canopy-ul are doar `vineyard_id`. La import se recalculează: cea mai apropiată bucată de rând din tile, la cel mult 0,5 m. |
| `area_m2` | float64 | |
| `n_vertices` | int16 | țintă ~15 (mediana din referință) |
| `along_m` | float32 | întinderea de-a lungul axei |
| `is_clump` | bool | `area_m2 > 2` (pâlc lung de vegetație, ca în rândurile cu iarbă din r006) |
| `touches_edge` | bool | atinge marginea lui `clip` |

#### 2.5.8 `row_pieces` (AnnSet), echivalentul 1:1 al poliliniei CVAT `row`

| Câmp | Tip | Descriere |
|---|---|---|
| `piece_id` | str | PK |
| `row_id`, `vineyard_id`, `tile_id` | str | |
| `row_structure` | str (RowStructure) | per tile |
| `length_m` | float64 | |
| `max_gap_m` | float32 \| NaN | golul maxim în coridorul ±0,30 m, **cu capetele incluse** până la capătul bucății (marginea tile-ului sau prima/ultima viță). Porțiunile peste negru sunt excluse. |
| `n_vertices` | int16 | 2 pentru rânduri drepte |

#### 2.5.9 `interrows` (global) și `interrow_pieces` (AnnSet)

`interrows`: `interrow_id` (PK), `vineyard_id`, `row_left_id`, `row_right_id`, `area_m2`, `width_mean_m`, `length_m`. Banda merge de la `axa_k + 0,30 m` la `axa_{k+1} − 0,30 m` și se taie la capătul rândului mai scurt. Găurile vin din `in_forbidden` și din copacii din inter-rând.

`interrow_pieces`:

| Câmp | Tip | Descriere |
|---|---|---|
| `piece_id` | str | PK |
| `interrow_id` | str \| null | null pentru Marcaj până la legare |
| `vineyard_id`, `tile_id` | str | |
| `row_left_id`, `row_right_id` | str \| null | rândurile vecine (din model sau din legare) |
| `interrow_cover` | str (InterrowCover) | per tile |
| `veg_frac` | float32 | fracția din masca a\* (< 0,25 `bare_soil`, 0,25–0,75 `mixed`, > 0,75 `vegetation`) |
| `shadow_frac` | float32 | fracția de umbră adâncă (peste `interrow.unassessable_shadow_frac` → `unassessable`) |
| `area_m2`, `width_mean_m` | float64 / float32 | |
| `n_notches` | int8 | găuri transformate în „crestături” la export (secțiunea 4.1) |

#### 2.5.10 `waste_candidates` și `waste` (AnnSet)

| Câmp | Tip | Descriere |
|---|---|---|
| `waste_id` | str | PK |
| `tile_id` | str | |
| `vineyard_id` | str | blocul în care stă obiectul sau cel mai apropiat bloc la cel mult 10 m. Altfel `""`, adică atribut prezent, dar gol, cum cer regulile. |
| `dist_block_m` | float32 | |
| `px_xtl`, `px_ytl`, `px_xbr`, `px_ybr` | float32 | cutia în coordonate CVAT ale tile-ului |
| `area_m2` | float32 | aria cutiei (orientativă, nu e aria deșeului) |
| `category` | str | `bag`, `bottle`, `tyre`, `debris`, `heap`, `unknown` |
| `detector` | str | `sam3`, `nn`, `rule`, `manual` |
| `exported` | bool | `confidence ≥ waste.export_min_confidence` și trece filtrele |
| `reject_reason` | str \| null | doar în `waste_candidates`: `near_axis`, `tube_shape`, `too_small` … |

Cutia e aliniată la axe și în UTM, pentru că grila tile-ului nu e rotită față de UTM (geotransform fără termeni de rotație).

#### 2.5.11 `targets` și `target_extents`

| Câmp | Tip | Descriere |
|---|---|---|
| `target_id` | str | PK |
| `kind` | str (TargetKind) | `row_gap` = gol interior ≥ 5 m pe rândul global · `row_end_short` = capăt de rând retras cu cel puțin 5 m față de vecini · `missing_row` = spațiere de peste 1,8× mediana · `waste` · `other` |
| `vineyard_id`, `tile_id` | str | |
| `row_id`, `interrow_id`, `waste_id` | str \| null | legăturile cerute de brief |
| `x`, `y` | float64 | coordonatele UTM (duplicat al geometriei, pentru CSV și UI) |
| `gap_length_m` | float32 \| NaN | |
| `priority` | int8 | 1 = waste și goluri > 10 m, 2 = restul golurilor, 3 = altele |
| `reachable` | bool | `false` dacă e în `in_forbidden` sau la peste `route.max_snap_m` de graf |
| `reach_note` | str | |
| `snap_dist_m` | float32 | distanța până la cea mai apropiată muchie a grafului |

`target_extents` conține segmentul golului pe axă (`target_id`, `kind`, `length_m`). Punctul ascuns al organizatorilor poate fi oriunde în gol. Traseul trebuie să treacă la cel mult 2 m de **tot** segmentul sau să-l eșantioneze la fiecare `targets.gap_sample_step_m`.

#### 2.5.12 Domeniul de mers și graful

- `passable_parts`: `part_id` (str), `kind` (`interrow` | `passage` | `connector`), `ref_id` (bucata de inter-rând sau indexul pasajului), `area_m2`, `erosion_m`.
- `passable_domain`: un singur rând. `domain_id = "D1"`, `area_m2`, `erosion_m`, `n_components`, geometria = `union(inter-rânduri AnnSet, in_passages) − in_forbidden − buffer(canopies, 0)`.
- `walk_nodes`: `node_id` (int64), `kind` (`start` | `row_end` | `passage` | `target` | `junction`), `ref_id` (str).
- `walk_edges`: `edge_id` (int64), `u`, `v` (int64), `length_m` (float64), `kind` (EdgeKind), `ref_id` (str), `inside_frac` (float32: cât din muchie e în `passable_domain` erodat), `cost` (float64).

#### 2.5.13 `route` și `route_stops`

- `route`: `route_id` (`RT-<run_id>`), `length_m` (float64), `n_targets`, `n_visited_est` (int32), `coverage_est`, `outside_frac_est` (float32), `closure_m` (float32, trebuie să fie 0), `walking_time_min` (float32, la 4 km/h), `solver` (str), `solve_time_s` (float32), plus coloanele comune.
- `route_stops`: `route_id`, `seq` (int32), `target_id`, `cum_dist_m`, `leg_m` (float64).

#### 2.5.14 `qa_issues`

`issue_id` (str `Q00001`), `severity` (Severity), `code` (str), `tile_id`, `object_id` (str), `message` (str, în română), geometria = punctul de verificat. Codurile standard:

`missing_attr`, `bad_enum`, `id_case_collision`, `row_multi_block`, `row_piece_misaligned`, `row_missing_in_tile`, `dup_row_in_tile`, `canopy_interrow_overlap`, `canopy_on_non_vineyard_tile`, `waste_block_unknown`, `blocks_should_merge`, `block_split`, `structure_borderline` (gol între 4,5 și 7 m), `cover_borderline`, `empty_tile_confirm`, `tile_failed`.

### 2.6 AnnSet: interfața care face posibil round-trip-ul

```
work/runs/<run_id>/annset/
  canopies.parquet  row_pieces.parquet  interrow_pieces.parquet  waste.parquet
  annset.json   # {contract_version, source, run_id, model_version, created_at, n_tiles, counts{...}, inputs[...]}
```

- Un AnnSet corespunde 1:1 cu ce se vede în Marcaj: aceleași obiecte, per tile, în UTM, cu atributele CVAT și câmpurile calculate.
- Se construiește în trei feluri:
  - `assemble`, din model;
  - `import_marcaj`, din exportul Marcaj;
  - `import_reference`, din `05_examples`, pentru evaluare.
- Aval, stagiile `derive`, `targets`, `passable`, `route`, `measure`, `web_bundle` și `publish` primesc **numai** `--annset <run_id>` și nu știu dacă datele vin din model sau din Marcaj.

### 2.7 Validare la granița fiecărui stagiu (`contracts/schemas.py`)

- `validate_layer(gdf, name)` verifică:
  - coloanele obligatorii și tipurile;
  - enum-urile;
  - regex-urile de ID (strict pentru `source=model`, relaxat pentru `marcaj`);
  - PK unic;
  - CRS = EPSG:32635;
  - geometrie nevidă, validă (`shapely.is_valid`), 2D;
  - LineString cu cel puțin 2 vârfuri distincte și lungime ≥ 0,05 m.
- Invarianți topologici, verificați de `qa`, iar pentru model și de `assemble`:
  - canopy ∩ inter-rând < 0,01 m² per tile;
  - fiecare `row_id` are un singur `vineyard_id`;
  - fiecare `vineyard_id` din `waste` există și pe cel puțin un canopy sau rând;
  - toate obiectele sunt în `clip`-ul tile-ului lor (toleranță 1 mm);
  - nicio bucată de inter-rând în afara rândurilor extreme.
- O eroare de schemă oprește stagiul. O eroare topologică produce un `qa_issue` cu `severity=error`, iar exportul CVAT refuză să ruleze dacă există erori de acest fel în AnnSet-ul modelului.

---

## 3. Graful de stagii (DAG)

### 3.1 Diagrama

```
 INTRĂRI: 01_tiles/*.zip · 02_route/*.geojson · 05_examples · models/*
   │
   ▼
[ingest] G ─► work/tiles/*.tif (copii identice), tile_index, in_* (route inputs)
   ▼
[tile_prep] T ─► valid mask, veg mask (Lab a*), stats ─► tile_valid
   ▼
[nn_infer] T (MPS, un proces) ─► canopy_prob, axis_prob (uint8 1024²)
   ▼
[rows_detect] T ─► row_candidates
   ▼
[rows_link] G (pe clustere) ─► rows_raw (linii globale, fără ID)
   ▼
[blocks] G ─► blocks, rows (vineyard_id, row_id, row_index)
   ├──► [canopy] T ─────────► canopies ──┐
   ├──► [interrow] B→T ─────► interrows, interrow_pieces (cover per tile)
   ├──► [row_attrs] T (după canopy) ─► row_pieces (row_structure, max_gap)
   └──► [waste] T ──────────► waste_candidates ─► waste (W-ids, vineyard_id)
                                            ▼
                               [assemble] G ─► AnnSet(model) + tile_status + qa
                                            ▼
                               [export_cvat] G ─► upload ZIP-uri ≤85 MiB ─► MARCAJ (import, Publish)
                                                                               │ corecturi manuale
                               [import_marcaj] G ◄── export CVAT XML ◄──────────┘
                                            ▼
                                     AnnSet(marcaj)          [import_reference] ─► AnnSet(reference)
 ┌───────────────── orice AnnSet ───────────┘                          │
 ▼                                                                     ▼
[derive] G ─► rows (contopite pe row_id), blocks, legare inter-rânduri, qa_issues
[targets] G ─► targets, target_extents
[passable] G ─► passable_parts, passable_domain, walk_nodes, walk_edges
[route] G ─► route, route_stops, metrics/route_validation.json
[measure] G ─► measurements.csv, measurements.json
[web_bundle] G ─► web/data/**
[publish] G ─► ./route.geojson, ./measurements.csv (rădăcina repo-ului)
[eval] ─► metrics/eval_examples.json (AnnSet(model) vs AnnSet(reference) pe cele 2 tile-uri)
[bench] ─► metrics/timings.json (agregat din loguri)
```

### 3.2 Tabelul stagiilor

Scope: T = per tile, paralel pe `runtime.n_workers` procese · B = per bloc · G = global.

| # | Stagiu | Scope | Intrări | Ieșiri | Chei de config | Cheie de cache per item |
|---|---|---|---|---|---|---|
| 1 | `ingest` | G | ZIP-uri, `02_route/*` | `work/tiles/*.tif`, `tile_index`, `in_*` | `paths`, `grid` | sha1(numele + CRC-urile din directorul central al ZIP-urilor) |
| 2 | `tile_prep` | T | tif | `cache/valid/<t>.png`, `cache/veg/<t>.png`, `cache/stats/<t>.json` | `nodata`, `veg` | sha256 tile |
| 3 | `nn_infer` | T | tif, valid | `cache/nn/<mv>/canopy_prob/<t>.png`, `axis_prob/<t>.png` | `nn` | sha256 tile + `weights_sha256` |
| 4 | `rows_detect` | T | veg, nn, valid | `cache/rows_detect/<t>.parquet` | `rows.detect` | cheile 2 și 3 ale tile-ului |
| 5 | `rows_link` | G | toți candidații, `in_passages` | `layers/rows_raw.parquet` | `rows.link` | sha1(cheile 4 sortate) |
| 6 | `blocks` | G | rows_raw, `in_passages`, `in_forbidden` | `blocks`, `rows` | `blocks` | cheia 5 |
| 7 | `canopy` | T | veg, nn, `rows` ∩ tile ⊕ 1 m, valid | `cache/canopy/<t>.parquet` | `canopy` | cheile 2 și 3 + sha1(WKB-ul rândurilor care ating tile-ul, sortate după row_id) |
| 8 | `interrow` | B→T | `rows`, veg, `in_forbidden`, nn | `interrows`, `cache/interrow/<t>.parquet` | `interrow` | per tile: cheia 2 + sha1(rândurile care ating tile-ul) |
| 9 | `row_attrs` | T | `rows`, canopy tile, valid | `cache/row_attrs/<t>.parquet` | `row_structure` | cheia 7 |
| 10 | `waste` | T | tif, valid, `rows`, `blocks` | `cache/waste/<t>.parquet` | `waste` | sha256 tile + versiunea detectorului + sha1(rândurile din tile) |
| 11 | `assemble` | G | 7–10, `tile_index` | `annset/*`, `tile_status`, `qa_issues` | `export` (praguri) | sha1(cheile 7–10) |
| 12 | `export_cvat` | G | AnnSet, `tile_index`, tif-uri | `exports/marcaj_upload/*.zip`, `upload_manifest.csv`, `id_registry.json` | `export.cvat` | sha1(annset.json + config) |
| 13 | `import_marcaj` | G | XML/ZIP export, `tile_index` | `annset/*` (source=marcaj), `qa_issues` | `import` | sha256 al fișierelor de export |
| 14 | `derive` | G | AnnSet | `rows`, `blocks`, legare inter-rânduri, `qa_issues` | `derive`, `blocks` | sha1(annset.json) |
| 15 | `targets` | G | AnnSet, derive, `tile_valid` | `targets`, `target_extents` | `targets` | cheia 14 |
| 16 | `passable` | G | AnnSet, derive, `in_*` | `passable_*`, `walk_*` | `route.domain` | cheia 14 |
| 17 | `route` | G | 15, 16, `in_start` | `route`, `route_stops`, validare | `route` | cheile 15 și 16 + `route.solver` |
| 18 | `measure` | G | AnnSet, derive | `measurements.csv`, `.json` | `measure` | cheia 14 |
| 19 | `web_bundle` | G | toate straturile, tif-uri | `web/data/**` | `web` | cheile 14–18 |
| 20 | `publish` | G | 17, 18 | `./route.geojson`, `./measurements.csv` | — | fără cache (copie + validare) |

Note despre ordine:
- `canopy` depinde de rândurile globale, deci rulează după `blocks`.
- `row_attrs` depinde de `canopy`.
- `waste` folosește axele (excluderea tuburilor de lângă axă) și blocurile (`vineyard_id` la cel mult 10 m).
- `interrow` calculează banda per bloc, apoi o taie per tile și calculează `interrow_cover` per bucată, pe rasterul `veg` al tile-ului.

### 3.3 Cache, idempotență, determinism

- Cheia unui item e `sha1(f"{stage}:{STAGE_VERSION}" ‖ json_canonic(cfg[stage.cfg_keys]) ‖ sorted(input_keys))`. `STAGE_VERSION` e o constantă în modulul stagiului și se mărește la orice schimbare de logică.
- Fiecare artefact `X` are alături `X.key` (text). Stagiul sare peste item dacă `X` și `X.key` există și cheile coincid. Cu `--force <stage>` sau `--force-all` se recalculează.
- **Scrierea e atomică:** `X.tmp-<pid>`, apoi `os.replace`. Cheia se scrie ultima.
- Seed global `runtime.seed = 0`. Sortările sunt stabile după ID și nu există iterații peste `set` în codul care produce ieșiri. Două rulări cu același config dau fișiere identice octet cu octet, fără câmpurile `created_at`.
- **Scalabilitate:** o corectură de rând invalidează doar tile-urile atinse de acel rând (cheia 7 include WKB-ul local), nu toate cele 311.

### 3.4 Structura directorului de lucru

```
work/                                  (în .gitignore)
  tiles/<file_name>.tif                # 311 copii identice octet cu octet (sha256 în tile_index), ~376 MB
  tile_index.parquet
  cache/<stage>/<tile_id>.{png,parquet,json} + .key
  runs/<run_id>/
    run.json                           # config rezolvat, git sha, host, contract_version, stagii, timpi
    layers/*.parquet                   # straturi globale
    annset/*.parquet + annset.json
    exports/geojson/*.geojson          # EPSG:32635
    exports/marcaj_upload/siret3_upload_01of05.zip … + upload_manifest.csv + id_registry.json
    qa/qa_issues.parquet  qa/review_queue.csv  qa/tile_status.csv  qa/previews/<tile_id>.jpg
    metrics/timings.json  metrics/eval_examples.json  metrics/route_validation.json
    logs/pipeline.jsonl
  runs/LATEST_MODEL, runs/LATEST_MARCAJ  # symlink-uri
```

Bugetul de disc (liber acum: ~15 GB):
- tile-uri: 0,38 GB;
- măștile 1-bit PNG: ~0,1 GB;
- NN 2×1024² uint8 PNG: ~0,2 GB;
- rulări: cel mult ~0,2 GB fiecare.

Nu se salvează rastere de mozaic la rezoluție completă.

### 3.5 Erori și reluare

- O eroare pe un tile nu oprește rularea. Tile-ul primește `status=failed` și un `qa_issue` `tile_failed` cu stack trace-ul în log, iar ieșirile lui rămân goale. Codul de ieșire e ≠ 0 dacă există tile-uri eșuate, afară de cazul în care se dă `--allow-failures`.
- Stagiile globale se opresc la prima eroare, cu un mesaj care numește stagiul, stratul și obiectul.
- Pentru reluare: `vineyard run --from <stage>` sau `--tiles siret3_r02*`. Cache-ul face reluarea ieftină.

### 3.6 Round-trip Marcaj (pașii 12 → 13 → 14…)

1. **Sâmbătă, până la 12:00:** `export_cvat` pe AnnSet(model), apoi upload-ul tuturor ZIP-urilor. Se verifică 311 fișiere și abia apoi se dă **Publish** (poarta fermă).
2. **Sâmbătă seara** (repetiție) și **duminică la ~11:00** (final): export „CVAT for images 1.1” din Marcaj, la nivel de proiect sau pe job-uri, apoi `vineyard import-marcaj <fișiere>`. Rezultă `runs/<ts>-marcaj-<h>/annset`.
3. `derive` → `targets` → `passable` → `route` → `measure` → `web_bundle`, cu **același cod ca pe AnnSet(model)**.
4. `publish --annset LATEST_MARCAJ` copiază `route.geojson` și `measurements.csv` în rădăcina repo-ului.
5. `qa_issues` din importul Marcaj (ID-uri inconsecvente, atribute lipsă) ajung la echipă ca listă de corecturi înainte de 15:00.

---

## 4. Contractul CVAT (export și import)

### 4.1 Export (`cvat/writer.py`)

- **Antetul** e `<?xml version="1.0" encoding="utf-8"?>`, apoi `<annotations><version>1.1</version>`. Blocul `<meta><task><name>…</name><labels>…</labels></task></meta>` se **copiază verbatim** din `05_examples/.../annotations.xml`, cu `mutable=False` și `default_value` exact ca acolo. E formatul pe care Marcaj îl acceptă sigur.
- Fiecare ZIP conține **toate** tile-urile lui, inclusiv pe cele goale. Ordinea e lexicală după `file_name`, cu `<image id="k" name="siret3_r021_c012.tif" width="2048" height="2048">` și `k = 0…n−1` în ZIP. Numele nu are prefixul `images/`, ca în exemple.
- Ordinea formelor într-o imagine: `row` (polyline), `interrow_area`, `vineyard`, `waste` (box). Fiecare formă are `source="<export.cvat.shape_source>"` (implicit `manual`, ca în exemple), `occluded="0"`, `z_order="0"`.
- Atributele, în această ordine:
  - `vineyard`: `vineyard_id`
  - `row`: `vineyard_id`, `row_id`, `row_structure`
  - `interrow_area`: `vineyard_id`, `interrow_cover`
  - `waste`: `vineyard_id`, eventual gol (`<attribute name="vineyard_id"></attribute>`)
- Coordonatele: `utm_to_px` → `clip` la [0, 2048] → `"%.1f,%.1f"` separate prin `;`, apoi se elimină duplicatele consecutive.
- Criterii minime:
  - poligon: cel puțin 3 puncte distincte și aria ≥ 1 px²;
  - polilinie: lungime ≥ `export.min_row_piece_m` (0,5 m);
  - canopy: aria ≥ `canopy.min_area_m2`, aplicat per tile după tăiere;
  - bucată de inter-rând: aria ≥ `export.min_interrow_piece_m2` (0,25 m²).
- **Poligoanele au exteriorul CCW în UTM** (arie shoelace negativă în pixeli) și nu repetă primul vârf.
- **Găurile devin crestături.** CVAT nu suportă găuri, așa că fiecare gaură se unește cu cea mai apropiată latură lungă, printr-o tăietură de lățime 0 transformată în crestătură. Rămâne un singur poligon per inter-rând per tile. Dacă gaura taie toată lățimea, rezultă 2 poligoane cu aceleași atribute.
- Un MultiPolygon rezultat din tăiere devine mai multe `<polygon>` cu aceleași atribute.
- **Box:** `xtl = min u`, `ytl = min v`, `xbr = max u`, `ybr = max v`, cu 1 zecimală.
- Tile-urile goale apar ca `<image …></image>` fără copii. „No objects in this frame” se bifează manual în Marcaj, iar `review_queue.csv` le listează cu `empty_tile_confirm`.
- **Autoverificare obligatorie după scriere:**
  1. Se citește fiecare ZIP cu `cvat/reader.py` și se compară cu AnnSet-ul sursă: aceleași numărători per tile și etichetă, aceleași atribute, abatere maximă a vârfurilor ≤ 0,06 px.
  2. sha256-ul fiecărui `images/*.tif` e egal cu cel din `tile_index`.
  3. Totalul pe toate ZIP-urile e de 311 imagini unice.
  4. Fiecare ZIP are ≤ `export.cvat.max_zip_mib`.

### 4.2 Împachetarea ZIP (măsurat)

- ZIP-urile originale sunt `STORED` (fără compresie) și au 89,48 / 89,29 / 88,89 / 89,75 / 9,93 MiB. XML-ul din exemple are 25,5 KB comprimat per tile. **Cu adnotările, părțile 1–4 ar trece de 90 MiB**, deci nu se refolosesc ca atare.
- Tile-urile se sortează lexical și se împachetează greedy cu bugetul `max_zip_mib = 85` (85 MiB = 89,1 MB, sub 90 în ambele interpretări ale lui „MB”). `images/*.tif` se adaugă ca `ZIP_STORED` (JPEG-ul nu se mai comprimă), iar `annotations.xml` cu `ZIP_DEFLATED`.
- Estimarea pesimistă (45 KB XML per tile) dă 5 ZIP-uri, cu 68 / 64 / 71 / 69 / 39 de tile-uri (măsurat pe dimensiunile reale).
- Numele: `siret3_upload_01of05.zip`, … Structura: `annotations.xml` + `images/<file_name>`.
- `upload_manifest.csv` are coloanele `zip_name, image_id, tile_id, sha256, n_vineyard, n_row, n_interrow_area, n_waste, xml_bytes`.

### 4.3 Import (`cvat/reader.py`), folosit și pentru exemple

- **Intrare:** unul sau mai multe fișiere `.zip` (caută `annotations.xml` la orice adâncime) sau `.xml`. Formatul suportat e „CVAT for images 1.1”. `json_simple` se implementează doar după ce vedem un export real.
- **Imagini:**
  - numele se normalizează la basename și trebuie să existe în `tile_index` (altfel eroare);
  - dimensiunea trebuie să fie 2048×2048;
  - dacă aceeași imagine apare în mai multe fișiere, câștigă ultimul fișier dat în CLI, cu warning.
- **Forme:**
  - `polygon`/`polyline`/`box` se mapează pe etichete și straturi: `vineyard` → `canopies`, `row` → `row_pieces`, `interrow_area` → `interrow_pieces`, `waste` → `waste`;
  - `mask` (pensulă): RLE decodat → contur → poligon, cu warning;
  - `points` și `tag`: ignorate, cu info;
  - `track`: eroare.
  - O etichetă necunoscută sau o formă greșită pentru etichetă (de ex. `vineyard` ca box) produce `qa_issue` de tip error, iar obiectul se sare.
- Conversia se face cu `px_to_utm(tile, uv)`, apoi `make_valid`, apoi orientarea CCW.
- `source=marcaj`, `confidence=1.0`, `model_version=marcaj-export@<sha8>`.

### 4.4 Normalizarea atributelor la import

- **Enum-uri:** `strip`, `lower`, spații și `-` → `_`. Sinonime acceptate cu warning: `bare soil`, `bare` → `bare_soil`; `grass`, `veg` → `vegetation`; `unknown` → `unassessable`. Orice altă valoare dă `bad_enum` (error), iar valoarea brută se păstrează.
- **ID-uri:** doar `strip`. Nu se schimbă litere mari sau mici și nu se renumerotează, pentru că organizatorii numără valorile brute.
  - `V03` și `v03` în același set → `id_case_collision`.
  - Un obiect fără `vineyard_id` → `missing_attr` (error).
  - Un `row` fără `row_id` sau `row_structure` → `missing_attr`.

### 4.5 Contopirea bucăților per `row_id` (`derive`)

1. Bucățile se grupează după `row_id`. Dacă au `vineyard_id` diferite → `row_multi_block` (error), iar pentru geometrie se folosește `vineyard_id` majoritar.
2. Direcția principală se obține prin PCA pe toate vârfurile. Bucățile se sortează după proiecția mijlocului.
3. Linia contopită e LineString-ul prin toate vârfurile ordonate după proiecție (rânduri drepte sau ușor curbe).
4. La fiecare joncțiune, abaterea laterală dintre bucăți consecutive trebuie să fie ≤ 0,4 m, altfel `row_piece_misaligned`.
5. Dacă linia trece printr-un tile valid fără bucată cu acel `row_id` → `row_missing_in_tile`. Două bucăți cu același `row_id` în același tile → `dup_row_in_tile`.
6. `length_m = Σ lungimile bucăților`, `extent_m = lungimea liniei contopite`.
7. **Legarea inter-rândurilor:** pentru fiecare bucată de inter-rând se caută cele două bucăți de rând din același tile și bloc, cele mai apropiate de laturile lungi (distanța perpendiculară de la centroid, pe normala blocului). Rezultă `row_left_id`/`row_right_id`, iar `interrow_id = <vineyard_id>-I<min(k)>` când ID-urile rândurilor respectă `-R\d+`, altfel după ordinea `n·c`.
8. **Blocurile din Marcaj:** geometria e `unary_union(buffer(obiectele blocului, 2.5)).buffer(−2.5)`.
   - Două ID-uri diferite la mai puțin de 5 m, fără pasaj între ele → `blocks_should_merge` (warning).
   - Un ID cu componente la peste 5 m una de alta → `block_split` (warning).

---

## 5. Livrabile în rădăcina repo-ului

### 5.1 `route.geojson`

- Un `FeatureCollection` cu **un singur** Feature `LineString`. Membrul `crs` e `urn:ogc:def:crs:EPSG::32635`, identic cu `start.geojson`.
- Coordonatele au 2 zecimale. Primul și ultimul vârf sunt START (629504.70, 5220250.75), exact.
- Proprietăți obligatorii: `length_m` (float, 2 zecimale, lungimea planară a LineString-ului). Proprietăți informative: `route_id`, `run_id`, `source`, `n_targets`, `n_visited_est`, `coverage_est`, `outside_frac_est`, `closure_m`, `walking_time_min`, `created_at` (ISO 8601 cu +03:00).
- `publish` refuză fișierul dacă:
  - `outside_frac_est > route.max_outside_frac_publish` (0,005, adică de 4 ori sub pragul oficial de 2%);
  - `closure_m > 0,01`;
  - geometria nu e simplu LineString;
  - `length_m` nu coincide cu lungimea recalculată (±0,01 m).

### 5.2 `measurements.csv` (UTF-8, virgulă ca separator, punct zecimal, antet)

Coloane, în ordine:

```
level,vineyard_id,row_id,n_blocks,n_rows,n_row_pieces,n_canopies,row_length_m,canopy_area_m2,canopy_area_ha,interrow_area_m2,interrow_area_ha,n_waste,n_targets,max_gap_m,structure_any,source,run_id
```

- `level=total`: un singur rând, cu `vineyard_id=ALL`, `row_id` gol.
- `level=block`: câte un rând per `vineyard_id`, sortate.
- `level=row`: câte un rând per `row_id`, sortate.
- Celulele care nu se aplică rămân goale.
- Rotunjiri: lungimi și arii în m² la 0,01; ha la 0,0001.

Calcul, identic cu cel al organizatorilor din adnotările Marcaj (planar, UTM):

| Mărime | Formulă |
|---|---|
| `n_blocks` | numărul de valori `vineyard_id` distincte și nevide, pe toate cele 4 etichete (vezi întrebarea Q3) |
| `n_rows` | numărul de valori `row_id` distincte |
| `row_length_m` (rând) | Σ lungimilor bucăților cu acel `row_id` |
| `row_length_m` (total sau bloc) | Σ pe toate bucățile din total sau din bloc |
| `canopy_area_m2` | `area(unary_union(canopies))`. Tile-urile sunt disjuncte, deci uniunea se face per tile și apoi se însumează. |
| `interrow_area_m2` | `area(unary_union(interrow_pieces))` |
| `*_ha` | m² / 10 000 |

În plus, `measurements.json` conține aceleași date, structurat: `{total, blocks[], rows[], meta{run_id, source, contract_version, created_at}}`. Web-ul îl citește direct.

---

## 6. Contractul rețelei neuronale

- **Intrare:** tile-ul RGB uint8 cu masca `valid`, redimensionat la `nn.in_gsd_m` (implicit 0,05 m, adică 1024², cu `cv2.INTER_AREA`).
- **Ieșiri per tile:** PNG cu un singur canal, uint8 (`p·255`), 1024², aliniate la colțul UL. Pixelul `k` acoperă `u ∈ [2k, 2k+2)`.
  - `canopy_prob`: probabilitatea „canopy de viță” față de sol, iarbă, pomi, clădiri și altele. Acest canal e obligatoriu.
  - `axis_prob`: heatmap-ul axei de rând (gaussian cu σ = 0,15 m). Opțional.
  - Upsample cu `INTER_LINEAR`. OpenCV folosește convenția centrului de jumătate de pixel, deci alinierea e corectă pentru factori întregi.
- **Consumatori:**
  - `rows_detect` ponderează profilul și respinge livezile și culturile în rânduri (`vine_score`);
  - `canopy` folosește `veg ∧ canopy_prob ≥ nn.prob_threshold` în coridor;
  - `tile_status.vineyard_score` este p95 din `canopy_prob` pe coridoare;
  - `interrow` folosește NN pentru găurile din copaci.
- **Greutăți:** `models/<name>/<version>/weights.pt` (state_dict) + `model_card.json`, cu câmpurile `{name, version, arch, encoder, in_gsd_m, classes, train_data[{name, url, licence}], metrics, sha256, created_at, train_cmd}`.
  - Distribuție: GitHub Release sau Hugging Face Hub.
  - `vineyard nn fetch` verifică sha256.
  - `vineyard nn train --config …` reproduce antrenarea.
  - `model_version = <name>@<version>+<sha256[:8]>`.
- **Degradare controlată:** cu `nn.enabled=false` sau fără greutăți, stagiile primesc `canopy_prob=None` și merg pe rețeta clasică. Asta e o cerință de robustețe. `model_version` o notează (`nn=none`).
- **Inferența** rulează în procesul principal pe `nn.device` (`mps`) în batch-uri de `nn.batch_size`. Nu se face în workerii paraleli, pentru că GPU-ul trebuie să aibă un singur consumator.

---

## 7. Contractul pachetului web (`web/data/`)

- `manifest.json` conține `{contract_version, run_id, source, created_at, crs: "EPSG:4326", layers:[{name, file, geom, count, minzoom, style_hint}], kpis}`.
- Straturile se exportă în **EPSG:4326** (RFC 7946, lon/lat, 7 zecimale), prin `pyproj` la export, pentru Leaflet sau MapLibre: `rows`, `row_pieces`, `interrow_pieces`, `blocks`, `waste`, `targets`, `route`, `route_stops`, `in_passages`, `in_forbidden`, `in_study_area`, `tile_index`. Copiile în EPSG:32635 stau în `web/data/utm/`, pentru descărcare.
- Canopy-urile se fragmentează per tile, în `web/data/canopies/<tile_id>.geojson`, și se încarcă la `zoom ≥ 18` sau pentru tile-urile din viewport. Sunt ~100k poligoane în total.
- **Ortofotografia:** `web/data/ortho/<tile_id>.jpg` (512×512, 0,1 m/px, calitate 80, ~16 MB în total) plus `ortho_index.json` cu **4 colțuri lon/lat** per tile. Cu sursa `image` din MapLibre (4 colțuri), rotația de ~1,25° a grilei UTM se reproduce corect. Un simplu `imageOverlay` pe bounding box ar decala colțurile cu ~1,1 m.
- `summary.json` (KPI): blocuri, rânduri, lungime totală, arie canopy și inter-rând (m² și ha), lungimea traseului, timpul de mers, numărul de ținte.
- `measurements.json` e copiat din secțiunea 5.2.

---

## 8. Layout-ul pachetului Python și CLI

```
vineyard-siret3/                      # repo (rădăcina = livrabil)
├── README.md  route.geojson  measurements.csv  LICENSE  CITATION (CC BY 4.0 Sireț3, ODbL OSM)
├── pyproject.toml  requirements.lock  Dockerfile  Makefile
├── configs/default.yaml  configs/local.example.yaml
├── data/raw -> "../data & info"      # symlink, read-only
├── models/<name>/<version>/{weights.pt,model_card.json}   (în .gitignore, descărcate prin `nn fetch`)
├── src/vineyard/
│   ├── __init__.py                   # __version__
│   ├── cli.py                        # aplicația typer `vineyard`
│   ├── config.py                     # modele pydantic, încărcare YAML + suprascrieri --set, hash per subarbore
│   ├── logging_setup.py              # consolă rich + handler JSONL
│   ├── contracts/
│   │   ├── __init__.py               # CONTRACT_VERSION = "1.0"
│   │   ├── enums.py  ids.py  schemas.py (LAYER_SCHEMAS, validate_layer)
│   ├── geo/
│   │   ├── tiling.py                 # TileRef, px_to_utm, utm_to_px, index_to_px, tiles_for_bounds, mosaic_px
│   │   ├── raster.py                 # read_tile, valid_mask, mask_to_polygons, rasterize_px, resize_aligned
│   │   ├── vector_io.py              # read/write parquet și geojson (crs, rotunjiri), fallback json
│   │   └── ops.py                    # clip_to_tile, orient, make_valid, notch_holes, split_multi
│   ├── pipeline/
│   │   ├── dag.py                    # registrul de stagii, cheile de cache, runner (ProcessPool per tile)
│   │   ├── context.py                # RunContext (run_id, căi, cfg, logger, timere)
│   │   └── stages/                   # un modul per stagiu (secțiunea 3.2): ingest.py, tile_prep.py, nn_infer.py,
│   │                                 #   rows_detect.py, rows_link.py, blocks.py, canopy.py, interrow.py,
│   │                                 #   row_attrs.py, waste.py, assemble.py, export_cvat.py, import_marcaj.py,
│   │                                 #   derive.py, targets.py, passable.py, route.py, measure.py,
│   │                                 #   web_bundle.py, publish.py, evaluate.py
│   ├── perception/                   # logică pură (numpy, cv2), fără I/O
│   │   ├── vegmask.py  rows_detect.py  rows_link.py  blocks.py  canopy.py  interrow.py  attrs.py
│   │   └── waste/  {candidates.py, sam3_adapter.py, filters.py}
│   ├── nn/  {model.py, dataset.py, pseudolabels.py, train.py, infer.py, weights.py}
│   ├── cvat/ {template.py (bloc <meta> verbatim), writer.py, reader.py, packer.py, normalize.py}
│   ├── annset/ {assemble.py, derive.py, merge_rows.py, link_interrows.py, qa.py}
│   ├── route/ {domain.py, graph.py, targets.py, solver.py (NN+2-opt/Or-opt; ortools opțional), validate.py}
│   ├── measure/ {measurements.py}
│   ├── eval/ {metrics.py (canopy 0.6·IoU+0.4·F1, row F1 0.4 m/80%, atribute acc+macroF1, IoU inter-rând, numărători), examples.py}
│   └── web/ {bundle.py, ortho.py}
├── web/  index.html  app.js  style.css   # static (MapLibre sau Leaflet), citește web/data/
├── server/app.py                          # FastAPI opțional: servește web/, /api/summary, /api/rerun-route
├── scripts/  benchmark.sh  make_release.sh
└── tests/  test_tiling.py  test_ids.py  test_schemas.py  test_cvat_roundtrip.py  test_merge_rows.py
           test_measure_examples.py  test_route_validate.py  test_packer.py
```

Funcțiile din `geo/tiling.py` sunt contract (semnăturile nu se schimbă):

```python
GSD_M = 0.025; TILE_PX = 2048; TILE_M = 51.2
GRID_ORIGIN_X = 628992.0; GRID_ORIGIN_Y = 5221222.4

@dataclass(frozen=True)
class TileRef:
    tile_id: str; grid_row: int; grid_col: int; x0: float; y0: float   # (x0, y0) = colțul UL = ModelTiepoint
    @property
    def bounds(self) -> tuple[float, float, float, float]:              # (minx, miny, maxx, maxy)
        return (self.x0, self.y0 - TILE_M, self.x0 + TILE_M, self.y0)

def tile_ref_from_grid(row: int, col: int) -> TileRef          # x0 = ORIGIN_X + 51.2*col ; y0 = ORIGIN_Y - 51.2*row
def tile_ref_from_tags(path: str | Path) -> TileRef             # citește 33922/33550, verifică vs grilă (|Δ| < 1e-6)
def px_to_utm(t: TileRef, uv: np.ndarray) -> np.ndarray         # (N,2) CVAT continuu -> (N,2) UTM
def utm_to_px(t: TileRef, xy: np.ndarray) -> np.ndarray         # invers, fără clip
def index_to_px(ij: np.ndarray) -> np.ndarray                   # (i=col, j=rând) index/contur -> centru pixel (+0.5)
def tiles_for_bounds(minx, miny, maxx, maxy) -> list[str]       # tile-urile existente atinse (O(1) pe grilă)
def mosaic_px(t: TileRef, uv: np.ndarray) -> np.ndarray          # U = 2048*col + u ; V = 2048*row + v
```

**CLI** (`[project.scripts] vineyard = "vineyard.cli:app"`, cu opțiunile comune `--config`, `--set k=v`, `--run-id`, `--tiles <glob>`, `--workers`, `--force <stage>`):

| Comandă | Ce face |
|---|---|
| `vineyard ingest` | verifică și copiază tile-urile, construiește `tile_index`, normalizează `02_route` |
| `vineyard run [--until assemble] [--from <stage>]` | percepția completă → AnnSet(model) |
| `vineyard export-cvat --annset <run>` | ZIP-urile de upload + autoverificare |
| `vineyard import-marcaj <fișiere…>` | AnnSet(marcaj) + `qa_issues` |
| `vineyard import-reference` | AnnSet(reference) din `05_examples` |
| `vineyard derive / targets / route / measure --annset <run>` | stagii individuale |
| `vineyard post --annset <run>` | derive → targets → passable → route → measure → web_bundle |
| `vineyard publish --annset <run>` | copiază și validează `route.geojson` și `measurements.csv` în rădăcină |
| `vineyard eval-examples --annset <run>` | metricile oficiale pe cele 2 exemple |
| `vineyard qa --annset <run>` | `qa_issues` + `review_queue.csv` |
| `vineyard nn fetch / train / infer` | greutăți |
| `vineyard web build / serve` | pachetul web, server local |
| `vineyard bench` | rulare curată, cronometrată, pe toate 311 → `metrics/timings.json` + tabelul pentru README |

`Makefile`: `make setup`, `make all` (ingest → run → export-cvat), `make post ANNSET=LATEST_MARCAJ`, `make test`, `make bench`.

---

## 9. Config YAML (`configs/default.yaml`), cu valorile implicite măsurate

```yaml
contract_version: "1.0"
project: {name: siret3, crs: "EPSG:32635"}
paths:
  data_root: data/raw                      # symlink spre "data & info" (read-only)
  tiles_zip_glob: "01_tiles/siret3_challenge_tiles_part*of5.zip"
  route_dir: 02_route
  examples_dir: 05_examples/siret3_examples_cvat
  work_dir: work
  models_dir: models
grid: {gsd_m: 0.025, tile_px: 2048, tile_m: 51.2, origin_x: 628992.0, origin_y: 5221222.4, expected_tiles: 311}
runtime: {n_workers: 8, seed: 0, allow_failures: false}
nodata: {max_rgb: 10, min_area_m2: 0.5, close_px: 5, dilate_px: 4, veg_erode_px: 4, min_valid_frac: 0.02}
veg:                                        # Lab a*: na = -(a* - 128); bate ExG (IoU 0.49 → 0.8+)
  method: lab_a
  blur_sigma_px: 2.5
  threshold: 4.0
  morph_close_px: 0                         # închiderea unește plante și scade F1 (măsurat)
nn:
  enabled: true
  name: vine-unet
  version: v1
  arch: unet
  encoder: resnet18                         # timm, features_only
  in_gsd_m: 0.05
  device: mps
  batch_size: 8
  prob_threshold: 0.5
  weights_sha256: null                      # se completează la release
rows:
  detect:
    angle_step_deg: 0.5
    angle_bin_m: 0.1
    profile_bin_m: 0.05
    smooth_sigma_bins: 3
    peak_min_distance_m: 0.9
    peak_prominence_frac: 0.05
    band_init_m: 0.45
    band_fit_m: 0.35
    fit_iterations: 3
    angle_gate_deg: 3.0
    end_percentiles: [0.5, 99.5]
    extend_to_edge_m: 3.0
    min_row_length_m: 1.0
  link:
    perp_tol_m: 0.5
    angle_tol_deg: 3.0
    max_join_gap_m: 6.0
  filter:
    spacing_range_m: [2.2, 3.8]             # măsurat 2.49–3.73, mediane 2.53 / 2.78
    max_canopy_width_m: 1.0                 # livezi: coroane 2–4 m, la 4–6 m
    min_vine_score: 0.35
blocks:
  gap_close_m: 5.0                          # două plantații la < 5 m = același bloc
  corridor_dilate_m: 2.5
  cut_by_passages: true
  min_rows: 3                               # vii de grădină: cel puțin 3 rânduri
  id_prefix: V
  id_width: 2
  row_id_width: 3
canopy:
  corridor_half_m: 0.30
  min_area_m2: 0.19
  connectivity: 8
  approx_eps_px: 2.5                        # ~15 vârfuri per plantă
  vector_outset_px: 0.5                     # de reglat pe exemple (secțiunea 1.4)
  clump_area_m2: 2.0
  split_clumps: false                       # referința NU le taie (poligoane de până la 55 m)
interrow:
  offset_from_axis_m: 0.30
  cover_thresholds: [0.25, 0.75]            # bare_soil 0.00–0.16, mixed 0.51–0.68 (măsurat)
  shadow_v_max: 35
  unassessable_shadow_frac: 0.5
  hole_min_area_m2: 1.0
  hole_sources: [forbidden, tree_mask]
row_structure:
  gap_disrupted_m: 5.0
  include_end_gaps: true                    # reproduce toate cele 51 de etichete
  borderline_m: [4.5, 7.0]                  # → qa structure_borderline
waste:
  enabled: true
  detector: sam3
  export_min_confidence: 0.6                # întâi precizia: un FP costă cât un FN
  axis_exclusion_m: 0.6
  tube_max_aspect: 2.5
  tube_max_area_m2: 0.12
  min_area_m2: 0.05
  block_assign_max_m: 10.0
targets:
  gap_min_m: 5.0
  gap_sample_step_m: 3.0                    # eșantionarea golurilor lungi (vizitare ≤ 2 m)
  end_short_min_m: 5.0
  missing_row_spacing_factor: 1.8
  include_waste: true
route:
  start_file: 02_route/start.geojson
  visit_radius_m: 2.0                       # oficial
  plan_radius_m: 1.5                        # marjă de siguranță
  domain:
    erosion_m: 0.25
    snap_tol_m: 0.05
  max_snap_m: 5.0
  max_outside_frac_official: 0.02
  max_outside_frac_publish: 0.005
  return_tol_m: 5.0                         # noi închidem exact (0 m)
  solver: {method: auto, time_limit_s: 60}  # auto = ortools dacă e instalat, altfel NN + 2-opt + Or-opt
  walking_speed_kmh: 4.0
export:
  min_row_piece_m: 0.5
  min_interrow_piece_m2: 0.25
  cvat:
    coord_decimals: 1
    shape_source: manual
    max_zip_mib: 85
    zip_name: "siret3_upload_{i:02d}of{n:02d}.zip"
    verify_roundtrip: true
import: {accept_enum_synonyms: true, duplicate_policy: last_wins}
measure: {round_m: 0.01, round_ha: 0.0001}
web: {ortho_px: 512, ortho_quality: 80, canopy_minzoom: 18, geojson_decimals_4326: 7}
logging: {level: INFO, jsonl: true, console: rich}
```

- Config-ul se încarcă cu `load_config(paths…, overrides)` și se validează cu pydantic (`extra="forbid"`: o cheie greșită e eroare, nu se ignoră în tăcere).
- `cfg_hash(stage)` = sha1 pe JSON-ul canonic al subarborilor citiți de stagiu.
- `run.json` salvează config-ul rezolvat complet.

---

## 10. Logging, metrici, timpi

- Logger-ele se numesc `vineyard.<modul>`. Consola folosește `rich` (INFO). Fișierul `runs/<id>/logs/pipeline.jsonl` are câte un eveniment JSON pe linie:

```json
{"ts":"2026-09-26T03:10:04.512+03:00","level":"INFO","run_id":"20260926T0310-model-a1b2c3",
 "stage":"canopy","tile_id":"siret3_r021_c012","event":"tile.done","duration_s":0.84,
 "n_canopies":402,"veg_frac":0.071}
```

- **Evenimente standard:** `run.start`, `run.end`, `stage.start`, `stage.end` (cu `n_items`, `n_cached`, `n_failed`, `wall_s`, `cpu_s`, `peak_rss_mb`), `tile.done`, `tile.cached`, `tile.failed` (cu `error`, `traceback`), `qa.issue`, `metric`.
- **Denumirea metricilor:** snake_case cu unitatea ca sufix: `_m`, `_m2`, `_ha`, `_s`, `_frac`, `_px`, `_mib`. Exemple: `route.length_m`, `canopy.iou_frac`.
- Erorile nu se înghit niciodată. Orice `except` loghează `tile.failed` sau `stage.failed` cu contextul, iar mesajele către utilizator (CLI, web) sunt în română, fără trace.
- `metrics/timings.json` are forma `{host:{cpu:"Apple M3 Pro", cores:11, ram_gb:18, os, python:"3.11.7", torch:"2.5.1", device:"mps"}, total_wall_s, stages:{<stage>:{wall_s, cpu_s, n_items, n_cached, s_per_item, peak_rss_mb}}}`. `vineyard bench` rulează fără cache (`--force-all`) și scrie tabelul Markdown în README (cerință oficială: timp + hardware).
- `metrics/eval_examples.json` conține, per tile și ca medie:
  - canopy: `iou`, `f1_at_0_5`, `score = 0.6·iou + 0.4·f1`;
  - axe: `row_f1`;
  - atribute: `accuracy` și `macro_f1` (pentru `row_structure` și `interrow_cover`);
  - `interrow_iou`;
  - erorile relative ale numărătorilor și ariilor față de referință.

  Aceasta e poarta de regresie: o schimbare nu se integrează dacă `score` scade cu peste 0,01.
- `metrics/route_validation.json` conține `{length_m, closure_m, outside_len_m, outside_frac, n_targets, n_reachable, n_visited_est, coverage_est, legs_outside:[{seq, len_m}]}`.

---

## 11. Teste de contract (minim, rulate în `make test`)

| Test | Ce fixează |
|---|---|
| `test_tiling.py` | formula grilei = tiepoint pe toate tile-urile · `px_to_utm(utm_to_px(x)) == x` (1e-9) · START în `r018_c010` la (28,0; 2002,0) |
| `test_cvat_roundtrip.py` | exemple XML → AnnSet(reference) → XML: aceleași numărători (399/251 canopy, 25/26 rânduri, 24/25 inter-rânduri), aceleași atribute, vârfuri la ≤ 0,06 px, orientare CCW în UTM, fără vârf de închidere |
| `test_ids.py` | regex-urile, formatarea, ordinea rândurilor (R01 = `n·c` maxim) pe exemple |
| `test_schemas.py` | fiecare strat produs trece `validate_layer` · o coloană lipsă sau un enum greșit cad |
| `test_merge_rows.py` | 2 bucăți coliniare în tile-uri vecine → 1 rând, `length_m` = sumă · bucăți decalate cu 0,6 m → `row_piece_misaligned` |
| `test_measure_examples.py` | pe AnnSet(reference): blocuri = 2, rânduri = 51 · ariile egale cu cele calculate independent cu shapely |
| `test_packer.py` | niciun ZIP > 85 MiB · 311 imagini unice · sha256 neschimbat |
| `test_route_validate.py` | un traseu sintetic care iese 3% din domeniu e respins · închiderea la START |

---

## 12. Întrebări deschise și riscuri de contract

**Întrebări (de pus pe Slack la Marcaj, 09:00–23:00):**
- **Q1.** Acceptă importul `source="auto"`? Până la răspuns folosim `manual`, ca în exemple. Se poate testa pe proiectul Draft, apoi „Remove all”.
- **Q2.** Cum apare „No objects in this frame” în exportul CVAT? Se poate seta la import? Presupunem că nu: le bifează echipa, iar `review_queue.csv` le listează.
- **Q3.** Un `vineyard_id` care apare doar pe `waste`, sau un `vineyard_id` gol pe waste, intră în numărul de blocuri? Până la răspuns, QA interzice primul caz (`waste_block_unknown`).
- **Q4.** Criteriul de 2% se verifică față de inter-rândurile referinței sau față de ale noastre? Presupunem varianta cea mai strictă: mergem pe liniile mediane ale inter-rândurilor și prin interiorul pasajelor, cu eroziunea `route.domain.erosion_m`.
- **Q5.** `route.geojson`: e bun un FeatureCollection cu un singur Feature? Presupunem că da, ca la fișierele lor.
- **Q6.** Referința ascunsă păstrează poligoanele-pâlc lungi (iarbă în coridor, până la 55 m) și pe alte tile-uri? Presupunem că da (`split_clumps: false`).
- **Q7.** Cum sunt reprezentate găurile inter-rândurilor în referință (crestătură sau două poligoane)? Presupunem crestătură.

**Riscuri:**
- Dacă instalarea shapely/geopandas eșuează, drumul CVAT rămâne pe numpy (secțiunea 2.1). Instalarea se face vineri seara, primul lucru.
- Deriva de contract între oamenii care lucrează în paralel: `validate_layer` la fiecare graniță plus testele de contract.
- Erorile de ½ pixel: sunt fixate în secțiunea 1.4 și testate.
- Depășirea de 90 MB: ZIP-urile se reîmpachetează la 85 MiB, cu verificare automată.
- Un ID schimbat după Publish: ID-urile sunt înghețate, iar `id_registry.json` dă următoarele numere libere.

---

## 13. Istoric

| Versiune | Data | Schimbare |
|---|---|---|
| 1.0 | 25.09.2026, 20:45 | prima versiune: coordonate verificate, straturi, DAG, CVAT, config, loguri |

## Surse

- Date locale (citite): `01_tiles/*.zip` (tag-uri și dimensiuni), `05_examples/siret3_examples_cvat/annotations.xml`, `02_route/*.geojson`, `README.md`, regulile de adnotare v1.0, descrierea challenge-ului, ghidul Marcaj, `analiza_exemple/RAPORT_exemple_si_reteta.md`.
- [GeoPandas: dependențe](https://geopandas.org/en/stable/getting_started/install.html) (pyogrio e obligatoriu în 1.x).
- [GeoPandas: `read_parquet`](https://geopandas.org/en/stable/docs/reference/api/geopandas.read_parquet.html) (cere doar pyarrow).
- [Formatul CVAT for images 1.1](https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/) (atributele `box`, `polygon`, `mask`, `source`).
