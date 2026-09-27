# Analiză TIF: modelul rulat live pe un tile încărcat (spec, 27.09.2026)

## Scop

Demo la pitch: juriul alege un tile Sireț3, îl încărcăm pe pagina `/analiza`, iar platforma rulează **pipeline-ul real**
(același cod ca pentru livrabile) doar pe acel tile și afișează peste imagine: masca de vegetație, coroanele, axele de
rând (`row_structure`), inter-rândurile (`interrow_cover`) și candidații de deșeuri.

Nu e o funcție de produs: acceptăm doar tile-uri pe grila Sireț3.

## Decizii

- **Input:** GeoTIFF RGB uint8, EPSG:32635, 2048 × 2048 px, 0,025 m/px, colțul NV pe grila §6.6. `tile_id` se deduce
  din coordonate, nu din numele fișierului. Orice altceva primește o eroare cu cod.
- **Doar modelul:** fără `overrides.yaml`, `row_seeds.csv` sau confirmări de deșeuri (fișiere goale / `rows_seeded.enabled=false`).
  Deșeurile afișate sunt candidații modelului care au trecut de filtre, marcați „nevalidați”.
- **Izolare:** fiecare job are propriul `data_root` (un ZIP cu un tile + `02_route` din pachetul organizatorilor) și propriul
  `work_dir`, cu `grid.expected_tiles=1`. Nu atinge `src/AI/work` și nici cache-ul rulărilor principale.
- **Un singur job odată** (CPU). Un al doilea upload în timpul rulării primește 409.
- **Durată măsurată:** ~24 s pe `siret3_r021_c012` (M3 Pro, 1 worker), din care ~11 s încărcarea CLIP pentru deșeuri.

## Arhitectură

```
browser /analiza ──POST multipart──▶ Next API /api/analiza            (Node runtime)
                                      ├─ salvează <jobs>/<id>/input.tif, status.json {state: queued}
                                      └─ spawn: uv run --no-sync vineyard demo tile <input.tif> --out <jobs>/<id>
browser ──poll 1 s──▶ GET /api/analiza/<id>           → status.json (+ result.json când e gata)
browser ──────────▶ GET /api/analiza/<id>/<fișier>    → preview.jpg, veg_mask.png, *.geojson (listă albă)
```

- `<jobs>` = `ANALIZA_JOBS_DIR` sau `<os.tmpdir>/vineyard-analiza`; `src/AI` = `VINEYARD_AI_DIR` sau `../../AI` față de `frontend/`.
- Python: pachetul nou `vineyard.demo` (sub-comanda `vineyard demo tile`), fără dependențe noi.

## Contract: dosarul unui job (scris de `vineyard demo tile`)

| Fișier | Conținut |
|---|---|
| `status.json` | `{"state": "queued"\|"running"\|"done"\|"error", "stage": str\|null, "stage_index": int, "n_stages": int, "error": {"code": str, "message": str}\|null, "updated_at": ISO}` |
| `result.json` | `{"tile_id", "corners": [[lon,lat]×4 TL,TR,BR,BL], "counts": {"canopies","rows","interrows","waste"}, "canopy_area_m2", "interrow_area_m2", "row_length_m", "timings_s": {stage: s}, "total_s", "model_version", "waste_verify_level"}` |
| `preview.jpg` | imaginea tile-ului, 1024 px |
| `veg_mask.png` | masca de vegetație, 1024 px, RGBA (culoarea `map.vegMask` din `rasterColors.json`, transparent în rest) |
| `canopies.geojson` | Polygon, EPSG:4326: `canopy_id`, `vineyard_id`, `row_id`, `area_m2` |
| `rows.geojson` | LineString: `row_id`, `vineyard_id`, `row_structure`, `length_m`, `max_gap_m` |
| `interrows.geojson` | Polygon: `interrow_id`, `vineyard_id`, `interrow_cover`, `area_m2` |
| `waste.geojson` | Polygon (cutie): `waste_id`, `rank_score`, `probe_p`, `area_m2`, `category`: candidații pe care regula de decizie îi pune în lista de revizie (scor ≥ `waste.decide.candidate_min`), cel mai bun primul |

Etapele (11, `n_stages`): `ingest`, `tile_prep`, `nn_infer`, `rows_detect`, `rows_link`, `blocks`, `canopy`, `interrow`,
`row_attrs`, `waste`, `assemble`. Măsurat cu `vineyard demo tile`: 17–18 s în pipeline, ~22 s cu pornirea Python
(`siret3_r021_c012`: 395 coroane, 25 rânduri, 24 inter-rânduri; `siret3_r005_c004`: 1 candidat de deșeu).

Coduri de eroare: `not_geotiff`, `wrong_crs`, `wrong_bands`, `wrong_size`, `wrong_gsd`, `off_grid`, `pipeline_failed`, `busy`, `too_large`.

## Pagina `/analiza`

Zona de upload (drag & drop, `.tif`, ≤ 100 MB) → bară de progres cu etapa curentă (ingest … assemble) → hartă MapLibre
centrată pe tile, cu previzualizarea ca imagine și straturile de mai sus (aceleași culori ca în `/harta`, comutatoare,
click → atribute) + panou cu numărători, arii, lungimi și timpul pe etape. Mesajele de eroare sunt traduse din coduri.

## Teste

- Python: validarea inputului și deducerea `tile_id` (tile bun, CRS greșit, dimensiune greșită, în afara grilei); exportul
  GeoJSON pe AnnSet-ul probei.
- Web: unitare pentru validarea numelui de fișier / listei albe și pentru maparea statusului; e2e opțional.
