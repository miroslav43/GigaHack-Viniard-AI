# Vineyard AI Field Challenge — data package

Deeptech GigaHack 2026 · challenge by Marcaj · 25–27 September 2026

## Contents

| Folder | File | What |
|---|---|---|
| `01_tiles/` | `siret3_challenge_tiles_part1of5.zip` … `part5of5.zip` | the **311 challenge tiles** — GeoTIFF, EPSG:32635, 0.025 m/px, 2048 × 2048 px (51.2 m); five ZIPs of at most 94 MB |
| | `overview.png` | 1 m/px overview: the challenge tiles outlined in yellow, the route START in red |
| `02_route/` | `start.geojson` | route start and finish — Point |
| | `passages.geojson` | authorised passages (roads, tracks, headlands) — MultiPolygon, `type` = `passage` |
| | `forbidden.geojson` | forbidden zones (village core, buildings, compounds) — MultiPolygon, `type` = `forbidden` |
| | `study_area.geojson` | outline of the 311 tiles |
| | `preview_passages_forbidden.png` | preview of the above |
| `03_docs/` | `Vineyard_AI_Field_Challenge_description.pdf` | the challenge: tasks, submission, rules, judging criteria |
| | `Vineyard_AI_annotation_rules.pdf` | how to annotate — labels, attributes, every case with examples |
| | `Marcaj_quick_start_for_teams.pdf` | step by step in Marcaj: sign in, build and upload the ZIPs, publish, correct, submit |
| `05_examples/` | `siret3_examples_cvat.zip` | **two example tiles annotated by the rules**, in the CVAT for images 1.1 format — the same format you upload; import them into Marcaj to see a finished tile. Not scored. |
| | `preview_siret3_r021_c012.jpg`, `preview_siret3_r006_c004.jpg` | previews: row axes red (disrupted magenta), canopies green, inter-row areas cyan |
| `04_source/` | `siret3_source_orthomosaic_EPSG4326.tif` | the full original orthomosaic (659 MB, **EPSG:4326**, as published on OpenAerialMap) — only for training on the whole survey |

All GeoJSON files are in **EPSG:32635** (WGS 84 / UTM 35N, metres); each declares it in its `crs` member. Route START: X = 629504.70, Y = 5220250.75 (47.1230335 N, 28.7073776 E).

## How to use the tiles

1. Run your model over the tiles and pack the tiles with its pre-annotations into ZIPs in the **CVAT for images 1.1** format (`annotations.xml` + `images/` with the tiles **unchanged, original names**) — at most 90 MB each; one ZIP per part works.
2. Upload **all five parts** to your team's Marcaj project, check it has **311 files**, then publish. Pre-annotations cannot be imported after publishing.
3. Correct in Marcaj, submit every job before the deadline — **15:00, Sunday 27 September**.

The annotation rules explain labels, attributes and IDs. The examples:

| Tile | Block | Rows | Canopies | Inter-row areas | What it shows |
|---|---|---|---|---|---|
| `siret3_r021_c012` | `V01` | 25, all `regular` | 399 | 24, all `bare_soil` | young vines on tilled soil — one polygon per plant |
| `siret3_r006_c004` | `V02` | 26, 5 `disrupted` | 251 | 25: 21 `bare_soil`, 4 `mixed` | sparse rows with long gaps; grass strips in the inter-rows; white vine tubes and stakes are not waste |

The annotation rules explain labels, attributes and IDs. Scoring uses a hidden subset of the 311 tiles: annotate all of them.

## Licence

Sireț3 imagery: **CC BY 4.0** — credit 3DATA COLLECT / OpenAerialMap, contributors to the Open Imagery Network. The challenge tiles are re-projected to EPSG:32635 and cut from it. Retain the attribution when you reuse the imagery.
Route data contains information from OpenStreetMap © OpenStreetMap contributors, ODbL.
