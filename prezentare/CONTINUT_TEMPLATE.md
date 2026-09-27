# Slide content for the organisers' template

The organisers' template is mandatory. Paste this text slide by slide, following their plan:
Title → Problem → Solution → Demo → Impact → Scaling.

- **Numbers:** every number is the same as in `pitch.pdf` (run rc10f / post run `20260927T0452-post-rc10f`; README §6 and §10).
- **Estimates:** deployment and human-check hours are team estimates, and the slide says so.
- **Diagrams:** `export/slide-NN.png` holds each main Beamer slide at 200 dpi. Crop a diagram from it, or paste a whole slide as an image if time runs out.
- **Speaker notes:** see `SCRIPT.md`.

---

## 1 · Title
- **From 311 drone tiles to a 24 km inspection walk**
- AI vineyard inventory and inspection routes from one drone flight
- Team **Solemtrix Hardware & Software**
- Team lead: **TODO**
- Members: Miroslav Maletici, Razvan Pervulescu, **TODO: other members**
- GigaHack 2026 · Vineyard AI Field Challenge · Sireț3 · 27 September 2026

## 2 · The problem (business analysis): who checks every vine?
- The national vine registry lists **45,500 ha**. The official statistics list **66,400 ha** of wine varieties (2024). That is a **20,900 ha gap**.
- Planting subsidies reach up to 300,000 lei/ha, so every declared hectare needs evidence.
- **Roles and user stories:**
  1. **State inspector** (vine registry, subsidy control): I want every block and row measured from one drone flight, so that I can verify declared areas and plantings without walking every parcel.
  2. **Farmer / vineyard owner:** I want a map of missing plants, gaps and waste per row, so that I can replant and clean where it matters, before the harvest.
  3. **Agronomist / field inspector:** I want the shortest closed walk through every problem spot, as GPX on my phone, so that one inspection takes 6.0 h instead of 17 h.
  4. **Municipality (primăria):** I want farms, roads, cadastre parcels and dumped waste on one map, so that I can plan land use and clean-ups.
- Sources, small print: agroexpert.md (registry vs statistics, 2024); telegraph.md (subsidy); maia.gov.md (vine registry).

## 3 · The business case: one flight, one map, one walk
- **Flow:**
  - Drone flight (2.5 cm/px)
  - → AI pipeline (17 min)
  - → human check in Marcaj
  - → map and reports per farm / block / row ID
  - → walk or robot (shortest route, GPX)
  - → decision (subsidy check, replanting, clean-up)
  - → next season: fly again and compare.
- **Sireț3:** 81.5 ha · 311 GeoTIFF tiles · 27 farms · 44 vineyard blocks · 674 rows (47.7 km) · 13,702 plants · 1,215 inspection targets.
- Image: the diagram in `export/slide-03.png`.

## 4 · Solution: architecture (CV/ML algorithms + rule logic)
1. **Row axes as a 1-D signal:**
   - dominant angle, then projection on the row normal;
   - autocorrelation gives the spacing (1.8–3.8 m);
   - SNR ≥ 3 rejects orchards;
   - Huber line fit, then union-find links rows across tiles.
2. **Canopies, one polygon per plant:**
   - Lab a* vegetation threshold, kept only inside the row corridor, then connected components;
   - a U-Net runs only where CV is blind (old vines in grass: 88 tiles, 13 s).
3. **Attributes, explicit rules:**
   - a gap ≥ 5 m marks the row *disrupted*;
   - inter-row cover comes from the vegetation fraction;
   - an unclear image gives *unassessable*.
4. **Waste, a cascade:** colour/shape rules → OpenCLIP probe → SAM 3 on the top crops → a human approves every box.
5. **Blocks, farms, route:**
   - cadastre parcels as a prior, plus road cuts;
   - a walking graph on the inter-rows and the authorised passages;
   - a generalised TSP (OR-Tools, 2-opt), closed at START, ≤ 2 % outside.
- **Key numbers:**
  - CV on all 311 tiles takes **≈ 3 min** on a laptop CPU (measured), with **0 training labels**;
  - SAM 3 on every tile would take **≈ 18 h** (extrapolated);
  - canopy score on the 2 labelled tiles: CV 0.816, U-Net alone 0.641.
- Images: `img/rows_preview.jpg`, `img/canopy_zoom.jpg`.

## 5 · Solution: data pipeline, GeoTIFF → route and reports
- **Steps:**
  1. GeoTIFF: 311 tiles, EPSG:32635, 5 ZIPs, 385 MB. Ingest and sha256 take 2.5 s.
  2. Perception (rows, canopies, inter-rows, attributes, waste, QA previews): 384 s.
  3. Marcaj: 5 validated CVAT ZIPs, human correction, export.
  4. Post (targets, walking graph, route, measurements, farms, roads): 603 s.
  5. Deliverables: `route.geojson` / GPX, `measurements.csv`, `targets.geojson`, the web map and reports.
- **≈ 17 min** from tiles to deliverables on one laptop (Apple M3 Pro), CPU only.
- **Engineering:**
  - caches keyed by the config hash;
  - one config file;
  - 2,378 tests and CI;
  - pinned `uv.lock`;
  - offline Docker (`--network none`);
  - no paid API and no LLM at runtime.
- Image: `export/slide-05.png` (the diagram).

## 6 · Result: a route per farm, inter-rows only
- **Farm F09 (blocks V13 + V16):**
  - **9.7 km** closed walk;
  - **−65 %** vs a 27.5 km serpentine;
  - **455/455** targets visited;
  - **2 h 26 min** at 4 km/h, solved in 1.5 s in the browser.
- **Whole survey, official `route.geojson`:**
  - **24.2 km** vs a 68.2 km serpentine (−65 %);
  - **145/145** reachable must-visit targets, +650 optional (795 on the walk);
  - **1.29 %** outside the allowed area (limit 2 %), closed at the organizers' pre-test START.
- **Organisers' 2 example tiles:** row F1 1.00 (51/51), canopy score 0.876, inter-row IoU 0.96, counts score 0.973.
- Image: `img/farm_ortho.jpg` (F09 orthophoto), or the map from `export/slide-06.png`.

## 7 · Demo
- Screencast: **TODO: YouTube link**
- Code: https://github.com/miroslav43/Solemtrix_Hardware_Software__Vineyard_AI_Field_Challenge
- Live demo from the team laptop: `cd src/Web/frontend && pnpm start` → http://localhost:3000/harta
- **What it shows:**
  - the map with every layer, and the `vineyard_id` / `row_id` on click;
  - blocks and rows with counts, lengths and areas;
  - the official route and per-farm routes, as GPX and GeoJSON;
  - farms, roads and live AGCC cadastre parcels;
  - RO / EN / RU.
- **Pre-test point 47.1225 N, 28.7095 E (UTM 629663.8, 5220195.3):**
  - farm F09, block V13;
  - inter-row V13-I020 (*mixed*), next to row V13-R020 (*disrupted*, 228.9 m, 70 plants);
  - the route now starts and ends exactly at this point (0.00 m closure), as the organisers asked.
- Image: a screenshot of the web map.

## 8 · Impact: economics and deployment time
- **Economic effect** (Sireț3, 81.5 ha, walking at 4 km/h):
  - one full inspection walk takes **6.0 h instead of 17 h** (24.2 km vs 68.2 km);
  - **≈ 11.0 h** of inspector time saved per survey (−65 %); farm F09 alone takes 2 h 26 instead of 6 h 52;
  - marginal compute cost is **≈ 0**: 17 min on a laptop CPU, no paid API, no GPU, open weights.
- **Deployment time: ≈ 11–20 h from zero to the first inspected region.**
  - One-time setup:

    | Step | Time |
    |---|---|
    | Server or laptop: Docker / `uv sync`, `pnpm install`, model weights from the GitHub release | 1–2 h |
    | Web app for inspectors: build, hosting, TLS, logins | 2–3 h |
    | Calibrate a new region: label 2 reference tiles, check the thresholds | 2–4 h |
    | Train the operators (inspector, agronomist) | 1–2 h |

  - Per new survey (≈ 80 ha):

    | Step | Time |
    |---|---|
    | Ingest and AI pipeline (measured: 2.5 s + ≈ 17 min) | 0.5 h |
    | Human check in Marcaj | 4–8 h |
    | Final run: route, measurements, reports, publish | 0.5 h |

- Compute times are measured (README §6). Set-up and human-check times are team estimates.

## 9 · Scaling to industry
- **≈ 4.7 s/ha** for perception on one laptop (measured on 81.5 ha).
  - All ≈ 100,000 ha of Moldovan vineyards: **≈ 130 h** on one laptop, or ≈ 16 h on a 64-core server.
  - This is a linear extrapolation. Tiles are independent, so the work parallelises.
- **Features for the industry:**
  1. **Ground robot, "Street View" for every row:**
     - it drives our route (24 km instead of ≥ 41 km);
     - side cameras image two rows per pass;
     - on-board ML flags disease symptoms;
     - every image is tied to its `row_id`.
  2. **Season-to-season change:** new gaps, replanting and abandoned plots, per row.
  3. **Registry and subsidy integration:** export per parcel to the vine registry (ONVV) and to AIPA field checks. The AGCC cadastre parcels are already on the map.
  4. **Per-farm reports and routes:** GPX for phones and robots, CSV by block and row.
  5. **On government infrastructure:**
     - an offline Docker image with no external API;
     - a multi-user web app in RO / EN / RU;
     - other row crops (orchards, berries) reuse the same 1-D row model.
- Image: `export/slide-09.png`. Robot photos go in `robot/robot.jpg` when available.

## 10 · Summary / Thank you
| Claim | Number |
|---|---|
| Efficient | 17 min from tiles to deliverables, one laptop, CPU only |
| Accurate | row F1 1.00, canopy 0.876, inter-row IoU 0.96 (organisers' 2 example tiles) |
| Short routes | −65 % vs a serpentine, ≈ 11.0 h saved per walk |
| Deployable | 11–20 h from zero to the first inspected region (estimate) |

**Thank you! Questions?**
