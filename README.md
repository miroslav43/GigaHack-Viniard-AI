# Sireț3 Vineyard AI · team Solemtrix

GigaHack 2026, Vineyard AI Field Challenge (Marcaj). This project turns the 311 supplied Sireț3 GeoTIFF tiles into:

- vineyard annotations (canopies, row axes, inter-rows, attributes, blocks, waste) for Marcaj;
- measurements;
- a walking route;
- a web interface.

| Deliverable | Where |
|---|---|
| `route.geojson` (one LineString, EPSG:32635, `length_m`) | repo root, written by `vineyard publish` / `vineyard final` |
| `measurements.csv` (blocks, rows, lengths, areas by `vineyard_id` / `row_id`) | repo root, same commands |
| Processing code | [`src/AI/`](src/AI) (Python package `vineyard`) |
| Web application | [`src/Web/`](src/Web) (Next.js) |
| Deployed interface | **TODO before submission: deployed URL.** SUBMISSION requires a link to the working web interface. The pitch demo from the team laptop is in addition to this link, not a replacement for it |

## Contents

1. [What the solution does](#1-what-the-solution-does)
2. [Repository layout](#2-repository-layout)
3. [Install](#3-install)
4. [Run: from the tiles to route.geojson and measurements.csv](#4-run-from-the-tiles-to-routegeojson-and-measurementscsv)
5. [Model weights](#5-model-weights)
6. [Processing time and hardware](#6-processing-time-and-hardware)
7. [Paid APIs and LLMs](#7-paid-apis-and-llms)
8. [Web interface](#8-web-interface)
9. [Outputs](#9-outputs)
10. [Current results](#10-current-results-model-run-before-marcaj-corrections)
11. [Licences and attribution](#11-licences-and-attribution)

## 1. What the solution does

```
5 tile ZIPs (311 GeoTIFF, EPSG:32635, 2.5 cm/px)
 │  vineyard run   ingest → tile_prep → nn_infer* → rows_detect → rows_link → blocks → canopy
 │                 → interrow → row_attrs → waste → assemble → qa_previews
 ▼
AnnSet(model): canopies, row axes, inter-rows, row_structure, interrow_cover, vineyard_id / row_id, waste
 │  vineyard export-cvat      → 5 validated CVAT-for-images-1.1 ZIPs (≤ 85,000,000 B each)
 ▼
Marcaj: upload the 5 ZIPs → publish → manual correction → export CVAT 1.1
 │  vineyard from-marcaj      → AnnSet(marcaj) + QA issues
 │  vineyard post             derive → passable → targets → route → measure → web_bundle
 │  vineyard publish          → validated copy to the repo root
 ▼
route.geojson + measurements.csv (repo root) · web bundle → src/Web
```

`*` `nn_infer` runs only with `nn.enabled=true` (off by default, see §5).

The perception is classical computer vision, ported from the prototype that we validated on the 2 example tiles:

- **Vegetation:** a threshold on the Lab a* channel, restricted to valid (non-nodata) pixels.
- **Row axes:** per tile we find the dominant row orientation, the row spacing (autocorrelation), an SNR gate and a Huber line fit. Rows are linked across tiles in UTM with union-find. Each connected group of rows becomes a block (`V01`…), with rows numbered `R001`…. A row keeps one `vineyard_id` / `row_id` across tile edges.
- **Canopies:** the vegetation mask inside each row corridor, split into connected components, so each separable plant is its own polygon.
- **Inter-rows:** the band between two adjacent row axes, inset 0.30 m from each axis and trimmed to the shorter row. Forbidden zones are notched out, and the band is cut per tile.
- **Attributes:**
  - `row_structure` comes from a single gap engine that measures gaps on canopy pixels (a gap ≥ 5 m means `disrupted`). It reproduces all 51 reference labels.
  - `interrow_cover` comes from the vegetation fraction.
- **Waste:**
  - Candidates are found by rule: HSV colour/brightness blobs, filtered by shape and lattice (vine tubes, stakes).
  - Optionally, an OpenCLIP linear probe and SAM 3 verification rank the candidates for review.
  - Nothing is exported automatically. Only the boxes the team accepted in `src/AI/configs/waste_confirmed.csv` are exported. SAM 3 and the probe pre-screen them, then the team approves them. The current file accepts 3 boxes.
- **Route:**
  - The walking domain is (inter-rows ∪ authorised passages) − forbidden zones − canopies.
  - A graph is built from the inter-row centrelines and the passage skeletons.
  - The targets are row gaps, missing rows, missing plants and waste. They are solved as a GTSP with OR-Tools, with python-tsp as a fallback.
  - The route is validated for closure at START, its length, and the share outside the allowed area (≤ 1.5%, stricter than the 2% rule).

Every threshold lives in one file: `src/AI/configs/default.yaml`. Every run writes the following under `work/runs/<run_id>/`:

- `run.json` and `config.json`;
- `logs/pipeline.jsonl`;
- `metrics/timings.json`.

Per-tile caches are keyed by the config hash.

## 2. Repository layout

```
.
├── route.geojson, measurements.csv   deliverables (after `vineyard publish` / `vineyard final`)
├── src/AI/                  processing code (uv project)
│   ├── vineyard/            package: pipeline stages, perception, cvat, annset, route, measure, nn, web
│   ├── configs/             default.yaml (all thresholds), overrides.yaml (QA edits), waste_confirmed.csv
│   ├── tests/               pytest suite (2,378 tests collected)
│   ├── docs/design/         design documents and the execution log (00_PLAN.md)
│   ├── Makefile, pyproject.toml, uv.lock, requirements.lock, .python-version (3.12)
│   ├── models/              model weights (git-ignored, see §5)
│   └── work/                extracted tiles, caches, runs (git-ignored)
├── src/Web/                 web app: frontend/ (Next.js), supabase/ (migrations), docs/
├── data & info/             organizer package: route inputs, docs and examples are in git;
│                            the tile ZIPs and the source orthomosaic are NOT (too large)
├── analiza_exemple/         prototype scripts on the 2 example tiles
└── .github/workflows/       ai-tests.yml (ruff + pytest), web.yml (web checks)
```

## 3. Install

### 3.1 Requirements

| Part | Tools |
|---|---|
| Processing (`src/AI`) | [uv](https://docs.astral.sh/uv/) 0.11.1 (the version used locally and in CI), Python 3.12 (installed by uv), ~5 GB free disk for `work/` |
| Web (`src/Web/frontend`) | Node.js 22 LTS, pnpm 10.33.0 (`corepack enable` picks it up from `package.json`) |
| Platforms | macOS on Apple silicon (arm64) and Linux (torch comes from the PyTorch CPU index). Declared in `pyproject.toml` → `[tool.uv] environments`. |

### 3.2 Python environment (pinned)

```bash
cd src/AI
uv python install 3.12
uv sync --locked --all-extras        # = make setup; exact versions from uv.lock
unset PROJ_DATA PROJ_LIB GDAL_DATA   # an active conda/GDAL install would shadow the wheels' PROJ/GDAL data
uv run --no-sync vineyard doctor     # checks libraries, GDAL/PROJ, torch/MPS, data, disk
```

Dependencies are pinned in `uv.lock`. `requirements.lock` is the pip-format export of the same lock (`make lock`). The core install (`uv sync --locked`, no extras) covers perception, CVAT export/import and measurements.

| Extra | Pins | Needed for |
|---|---|---|
| `route` | ortools 9.15.6755, python-tsp 0.5.0, sknw 0.15 | the route stage (`vineyard post`, `vineyard final`) |
| `nn` | torch 2.14.0, torchvision 0.29.0, segmentation-models-pytorch 0.5.0 | U-Net training and inference (MPS on Apple silicon, CPU elsewhere) |
| `waste-ml` | transformers ≥ 5.0 (locked 5.17.0), open-clip-torch 3.3.0, huggingface-hub (locked 1.33.0) | waste probe (OpenCLIP) and SAM 3 verification |

### 3.3 Docker (CPU, optional)

`src/AI/Dockerfile` builds from `python:3.12-slim` with uv 0.11.1 and runs `uv sync --frozen` from `uv.lock`. The image runs as a non-root user and downloads nothing at run time. Weights are not baked in; mount them on `/opt/vineyard/src/AI/models` when you need them.

```bash
cd src/AI
docker build -t vineyard .     # EXTRAS="route" (default): pipeline + route + measurements, no torch
# docker build --build-arg EXTRAS="route nn waste-ml" -t vineyard .   # adds torch 2.14.0+cpu
docker run --rm --network none -v "$PWD/../../data & info":/data:ro -v vineyard-work:/work vineyard doctor
docker run --rm --network none -v "$PWD/../../data & info":/data:ro -v vineyard-work:/work vineyard all --workers 8
docker run --rm --network none -v "$PWD/../../data & info":/data:ro -v vineyard-work:/work \
  -v "$PWD/marcaj":/marcaj:ro vineyard final /marcaj/<export>.zip \
  --set paths.publish_dir=/work/publish --set web.out_dir=/work/web
```

`make docker-build` builds the native arm64 image. `DOCKER_EXTRAS="route nn waste-ml"` adds torch.

### 3.4 Data

The organizer package lives in `data & info/` at the repo root. Only the tile ZIPs must be added by hand:

```
data & info/
├── 01_tiles/siret3_challenge_tiles_part1of5.zip … part5of5.zip   ← the 5 supplied ZIPs (311 tiles), not in git
├── 02_route/start.geojson, passages.geojson, forbidden.geojson, study_area.geojson   (in git)
└── 05_examples/siret3_examples_cvat/annotations.xml + images/   (in git; reference for eval and NN validation)
```

The pipeline reads the ZIPs directly. `ingest` checks every name against the 311-tile grid and every CRC, stores a sha256 and extracts the tiles to `work/tiles/`. The web orthophoto instead needs each ZIP extracted into a folder of the same name:

```bash
cd "data & info/01_tiles"
for z in siret3_challenge_tiles_part*of5.zip; do unzip -q -n "$z" -d "${z%.zip}"; done
```

Environment variables (both optional):

| Variable | Default | Meaning |
|---|---|---|
| `VINEYARD_DATA_ROOT` | `../../data & info` (relative to `src/AI`) | organizer data folder |
| `VINEYARD_WORK_DIR` | `src/AI/work` | tiles, caches and runs |

Any config key can be overridden per command with `--set a.b=value` or `--config overlay.yaml` (see `configs/local.example.yaml`). `vineyard config show --key route` prints the resolved values.

## 4. Run: from the tiles to route.geojson and measurements.csv

All commands run from `src/AI` with `unset PROJ_DATA PROJ_LIB GDAL_DATA` done once in the shell. `vineyard <command> --help` lists every option.

### Step 1: pre-annotation (AI model, 311 tiles)

```bash
uv run --no-sync vineyard run --until qa_previews --workers 8   # perception → AnnSet(model), link runs/LATEST_MODEL
uv run --no-sync vineyard export-cvat --annset LATEST_MODEL     # 5 validated upload ZIPs
# or both in one command:
uv run --no-sync vineyard all --workers 8
```

This produces:

- `work/runs/<run_id>/annset/`, the model annotations;
- `qa/`, with per-tile previews, `review_queue.csv` and `waste_review.html`;
- `exports/marcaj_upload/siret3_upload_0Nof05.zip`, plus `validation_report.json` and `id_registry.json`.

To re-check the ZIPs independently:

```bash
uv run --no-sync vineyard cvat validate work/runs/LATEST_MODEL/exports/marcaj_upload/*.zip --full-set
```

Optional quality check on the 2 reference tiles (official metrics):

```bash
uv run --no-sync vineyard import-reference
uv run --no-sync vineyard eval-examples --annset LATEST_MODEL --gates
```

### Step 2: Marcaj (manual)

1. Upload the 5 ZIPs to the team project and check that it holds 311 files.
2. Publish the project.
3. Correct the geometry and attributes in Marcaj, then submit every job.
4. Export the project as **CVAT for images 1.1**.

### Step 3: import the Marcaj export, post-process, publish

```bash
uv run --no-sync vineyard final /path/to/marcaj_export.zip
```

`final` runs `from-marcaj`, then `post` with the final solver time limit, then `publish` with `require_source=marcaj`. The same steps one by one:

```bash
uv run --no-sync vineyard from-marcaj /path/to/marcaj_export.zip   # alias of import-marcaj; --partial for an intermediate export
uv run --no-sync vineyard post --annset LATEST_MARCAJ --final      # derive → passable → targets → route → measure → web_bundle
uv run --no-sync vineyard publish --annset LATEST_MARCAJ --require-source marcaj
```

To publish directly from the model annotations, without Marcaj (the §10 numbers come from this `post` step; `publish` then copies its files to the repo root):

```bash
uv run --no-sync vineyard post --annset LATEST_MODEL
uv run --no-sync vineyard publish --annset LATEST_MODEL
```

`publish` opens a new run and looks for the newest post run of the same AnnSet whose id contains `-post-`, which every default id does. A post run with a custom `--run-id` does not match, so publish it in that same run instead. For the §10 run that is `vineyard publish --annset full-v2 --run-id post-full-v2`.

`post` writes:

- `work/runs/<post run>/exports/{route.geojson, measurements.csv, measurements.json}`;
- `metrics/*.json` (targets, route validation, measurements, timings);
- the web bundle in `src/Web/data/surveys/siret3/pipeline/` (EPSG:32635).

`publish` recomputes everything from the files before it copies them to the repo root. If any check fails, it refuses and leaves the root untouched. The checks are:

- one LineString with the EPSG:32635 `crs` member;
- first and last vertex equal to START;
- `length_m` equal to the recomputed length;
- outside share ≤ 1.5%;
- CSV format and sums consistent.

The single stages are also commands: `derive`, `passable`, `targets`, `route`, `measure`, `web build`.

### Makefile shortcuts (from `src/AI`)

| Target | Runs |
|---|---|
| `make setup` | `uv sync --locked --all-extras` |
| `make doctor` · `make test` · `make lint` | environment check · fast tests · ruff |
| `make preannotate` | `vineyard run --until qa_previews --workers 8` + `vineyard export-cvat --annset LATEST_MODEL` |
| `make validate-zips` | `vineyard cvat validate work/runs/LATEST_MODEL/exports/marcaj_upload/*.zip` |
| `make eval` | `vineyard eval-examples --annset LATEST_MODEL --gates` |
| `make post` · `make publish` | `vineyard post` · `vineyard publish` (`ANNSET=LATEST_MODEL` by default) |
| `make final MARCAJ=<export.zip>` | `from-marcaj` → `post --set route.solver.final=true` → `publish` (use a path without spaces, or the CLI above) |
| `make web` | `vineyard web build` (web bundle only) |
| `make bench` | `vineyard bench`: times perception only (ingest → qa_previews on all 311 tiles, cache ignored) → `metrics/timings.json`. It runs no CVAT export and no post chain (see §6) |
| `make nn-train` | `vineyard nn train` |

## 5. Model weights

The **submitted annotations need no weights.** The classical pipeline is the default: `nn.enabled=false`, `nn.fusion=A`. The U-Net ablation (`vineyard nn ablate`) kept variant A, because no NN variant beat it on both reference tiles. The mean canopy score was 0.816 for A and 0.773 for the best NN variant, D.

Weights live under `src/AI/models/` (git-ignored):

| Model | Path | Used for | How to get it |
|---|---|---|---|
| **vine-unet v1** (U-Net, ResNet-18 encoder, canopy probability at 5 cm/px) | `src/AI/models/vine-unet/v1/{weights.pt, model_card.json}` (57,414,739 B, sha256 `8b17fea024a3a0ffea13da03a660ef55439ed2209404100ca7c0c866de95ca19`) | optional canopy fusion (`--set nn.enabled=true --set nn.fusion=B\|C\|D`) | **WEIGHTS LINK: TODO** (host `weights.pt` and `model_card.json` together), or reproduce (below) |
| **SAM 3** (`facebook/sam3`, transformers `Sam3Model`) | `src/AI/models/hf/sam3/` (~3.4 GB) | waste verification (`waste.sam3.enabled=true`) | Hugging Face, gated: request access and accept the SAM License |
| **OpenCLIP ViT-B-32 `laion2b_s34b_b79k`** | Hugging Face cache | waste probe embeddings | downloaded automatically by `open_clip` on first use |
| **waste-probe v1** (logistic regression on CLIP embeddings) | `src/AI/models/waste-probe/v1/{probe.npz, model_card.json}` (12 KB) | waste ranking | reproduce from UAVVaste (below) |
| ResNet-18 ImageNet encoder init | Hugging Face cache (`smp-hub/resnet18.imagenet`) | U-Net training only | downloaded automatically by segmentation-models-pytorch |

**Install vine-unet v1 from a link.** `nn fetch` downloads two files: `weights.pt` and its `model_card.json`. It installs them only if the sha256 of `weights.pt` and the sha256 recorded in the card both equal `--sha256`. By default the card URL is the sibling `<same folder>/model_card.json`, so host both files side by side:

```bash
uv run --no-sync vineyard nn fetch --url <WEIGHTS LINK>/weights.pt \
  --sha256 8b17fea024a3a0ffea13da03a660ef55439ed2209404100ca7c0c866de95ca19 --version v1
```

Some share links have no sibling path, for example Google Drive or a Hugging Face `resolve` URL. With those, pass the card's own link:

```bash
uv run --no-sync vineyard nn fetch --url <link to weights.pt> --card-url <link to model_card.json> \
  --sha256 8b17fea024a3a0ffea13da03a660ef55439ed2209404100ca7c0c866de95ca19 --version v1
```

**Reproduce vine-unet v1.** It is trained on pseudo-labels from the model run, and the 2 reference tiles are held out for validation and early stopping:

```bash
uv run --no-sync vineyard import-reference                      # AnnSet(reference) from 05_examples
uv run --no-sync vineyard nn pseudolabels --run LATEST_MODEL --workers 6
uv run --no-sync vineyard nn train --run LATEST_MODEL           # MPS if available, else CPU; seed 0
uv run --no-sync vineyard nn ablate --run LATEST_MODEL          # variants A–F on the 2 reference tiles
uv run --no-sync vineyard nn infer --tiles 'siret3_r0*'         # optional: cache canopy probabilities
```

The recorded training command was `vineyard nn train --run full-v2 --reference 20260926T0031-reference-0c1714` (from `model_card.json`). Training on MPS is not bit-for-bit deterministic, so a rerun gives close but not identical weights.

**SAM 3:**

```bash
uv run --no-sync hf auth login
uv run --no-sync hf download facebook/sam3 --local-dir models/hf/sam3
uv run --no-sync vineyard waste sam3-check --n 3 --set waste.sam3.enabled=true   # loads it on CPU, measures seconds per crop
uv run --no-sync vineyard run --run-id <model run id> --from waste --until qa_previews --set waste.sam3.enabled=true
```

SAM 3 is off by default (`waste.sam3.enabled: false` in `default.yaml`). Without the `--set`, `sam3-check` prints `disabled` and gives no timing.

**Waste probe.** It is trained on [UAVVaste](https://zenodo.org/records/8214061) (Zenodo record 8214061, CC BY 4.0, 3.0 GB, md5 `1575c32c04bdf944047563e4a1786c2a`):

```bash
mkdir -p work/external/uavvaste
curl -L -o work/external/uavvaste/UAVVasteDataset.zip \
  "https://zenodo.org/api/records/8214061/files/UAVVasteDataset.zip/content"
uv run --no-sync vineyard waste probe-data --zip work/external/uavvaste/UAVVasteDataset.zip   # positives + Sireț3 negatives
uv run --no-sync vineyard waste probe-train                                                     # → models/waste-probe/v1
```

## 6. Processing time and hardware

**Hardware.** All timings below were measured on this machine:

- MacBook Pro (Mac15,6), Apple M3 Pro: 11 CPU cores (5 performance + 6 efficiency), 18 GB unified memory;
- macOS 26.2 (25C56), Python 3.12.13, uv 0.11.1.

The pipeline runs on CPU with 8 worker processes (`runtime.n_workers`), each single-threaded. Only U-Net training and inference use the Apple GPU (MPS). SAM 3 runs on CPU.

**Full tile set, tiles → route.geojson + measurements.csv (model annotations): ≈ 14 min without the waste probe, ≈ 16 min with it.** The total is the sum of the measured stages below. Each row names its run. The times come from that run's `run.end` / `stage.end` events in `work/runs/<run>/logs/pipeline.jsonl`. A later partial re-run of the same run id overwrites `run.json` and `metrics/timings.json`, but it only appends to the log.

| Step (311 tiles) | Command | Wall time |
|---|---|---|
| Ingest: read 5 ZIPs, extract + sha256 of 311 tiles (385 MB) | `vineyard ingest` (first run, nothing cached) | 2.5 s |
| Tile prep: nodata and vegetation masks, uncached | `vineyard run --run-id p1pipe-prep --until tile_prep --workers 8` | 14.2 s |
| Perception, rows → QA previews, measured **before the waste probe was installed** (waste rules only, verify level L3) | `vineyard run --run-id full-v2 --until qa_previews` (8 workers). Ingest and tile_prep came from cache, and 10–13 tiles each of canopy, interrow, row_attrs and waste were reused from cache. Source: the first `run.end` of `full-v2`, at 04:08:51 | **157.4 s** |
| &nbsp;&nbsp;of which | rows_detect 66.0 · rows_link 3.1 · blocks 1.1 · canopy 16.2 · interrow 7.1 · row_attrs 8.2 · waste (rules only) 35.4 · assemble 11.5 · qa_previews 5.8 | |
| CVAT export: 5 ZIPs, validated | `vineyard export-cvat --annset LATEST_MODEL` (run `20260926T0414-post-581775`) | 16.3 s |
| Post chain | `vineyard post --annset full-v2 --run-id post-full-v2` (8 workers). Source: the `run.end` of `post-full-v2` at 06:04:45 | **655.4 s** |
| &nbsp;&nbsp;of which | derive 28.9 · passable 79.6 · targets 31.9 · route 508.7 · measure 4.0 · web_bundle 1.7 | |
| **Total** | | **≈ 846 s ≈ 14 min** |

**The waste probe adds time.** The default config has `waste.probe.enabled: true`. When `models/waste-probe/v1` is installed (§5), the waste stage also runs OpenCLIP and the linear probe (verify level L1), not the rules alone. The `full-v2` re-run at 08:48 used 6 workers and measured this stage at **166.5 s** instead of 35.4 s. With the probe, perception takes ≈ 288 s and the total is ≈ 977 s ≈ 16 min.

The route stage dominates the total. It probes several outside-share policies, each with an OR-Tools solve limited to 30 s (`route.solver.time_limit_s`). `vineyard final` / `--final` raises that limit to 60 s, so expect a longer route stage on the final Marcaj run.

`make bench` (`vineyard bench`) re-times **perception only**: ingest → qa_previews with the cache ignored. It runs neither the CVAT export nor the post chain, so it writes no `route.geojson` or `measurements.csv`. The full timed re-run, from the tiles to both deliverables, is:

```bash
uv run --no-sync vineyard bench --workers 8                    # perception, cache ignored; moves runs/LATEST_MODEL
uv run --no-sync vineyard export-cvat --annset LATEST_MODEL    # 5 upload ZIPs
uv run --no-sync vineyard post --annset LATEST_MODEL           # derive → … → route → measure → web_bundle
```

Each command writes its stage times to its own run's `metrics/timings.json`, and a `run.end` event to `logs/pipeline.jsonl`.

Optional ML steps (not needed for the deliverables), same machine:

| Step | Command | Measured |
|---|---|---|
| U-Net pseudo-labels (90 tiles, 339 patches) | `vineyard nn pseudolabels --run full-v2 --workers 6` | ~10 s |
| U-Net training, MPS (11 epochs, early stop, best epoch 8) | `vineyard nn train --run full-v2 --reference …` | 260 s of epochs, 15 img/s |
| U-Net inference, MPS | `nn_infer` stage | 0.30 s per tile (≈ 1.5 min for 311 tiles, extrapolated) |
| Waste probe data (3,000 UAVVaste positives, 8,292 Sireț3 negatives) | `vineyard waste probe-data` | ~100 s |
| Waste probe training (OpenCLIP embeddings + logistic regression) | `vineyard waste probe-train` | ~67 s |
| SAM 3 verification, CPU | `vineyard run --run-id full-v2 --from waste --until qa_previews --workers 6 --set waste.sam3.enabled=true` | 13.1 s per 512 px crop (137 crops in 1,788 s, then the 1,800 s budget stopped it); waste stage 1,960 s |

## 7. Paid APIs and LLMs

- **No paid API and no LLM is called at runtime.** Neither the pipeline nor the web app calls one, and none is needed to reproduce the results.
- All models are open weights (see §5), downloaded for free from Hugging Face or Zenodo.
- **Development:** the code was written with **Claude (Anthropic)** as a coding assistant (Claude Code). Claude is a development tool only; it is not part of the pipeline.
- External services used by the web app only, and not needed for the jury view or the deliverables:
  - Supabase, for the optional login (without keys the app runs in open demo mode);
  - OpenStreetMap Nominatim, in the super-admin page, to enrol a new municipality.

## 8. Web interface

**Deployed interface: TODO before submission (URL).** SUBMISSION requires a link to the working web interface. The pitch demo on the team laptop is in addition to that link. The commands below build the same site locally.

It shows:

- the orthophoto;
- canopies, row axes (`row_id`), inter-rows, waste and inspection targets;
- the route polyline with `length_m` and START/FINISH;
- the `vineyard_id` / `row_id` of an object on click;
- canopy and inter-row areas (m² and ha);
- block and row counts;
- the length of each row and the total row length.

```bash
cd src/Web/frontend
corepack enable                          # pnpm 10.33.0 from package.json
pnpm install --frozen-lockfile           # exact versions from pnpm-lock.yaml
pnpm data                                # orthophoto from the extracted GeoTIFFs (§3.4, ~33 s) + demo survey
pnpm data:survey --survey siret3         # AI bundle src/Web/data/surveys/siret3/pipeline → public/data/siret3
NEXT_PUBLIC_SURVEY_ID=siret3 pnpm build  # the survey id is fixed at build time
pnpm start                               # → http://localhost:3000
```

- `pnpm data:survey` needs the bundle that `vineyard post` writes (§4, step 3).
- Pages: `/` (overview), `/harta` (map), `/blocuri` (blocks and rows), `/ruta` (route, GPX and GeoJSON download).
- Deep links: `/harta?rand=V02-R16`, `/harta?bloc=V01`.
- Login is optional: copy `.env.example` to `.env.local` and add the Supabase URL and key. Without it, the app runs in open demo mode.
- Details: [`src/Web/CLAUDE.md`](src/Web/CLAUDE.md).

## 9. Outputs

All geometry is in **EPSG:32635** (WGS 84 / UTM 35N, metres). Measurements are planar, with no terrain correction.

**`route.geojson`** is a FeatureCollection with a single LineString Feature and the `crs` member `urn:ogc:def:crs:EPSG::32635`. Coordinates have 2 decimals.

- The route starts and ends exactly at START, X = 629504.70, Y = 5220250.75. The rules allow 5 m; `publish` requires ≤ 0.01 m.
- Properties:
  - `length_m`;
  - `duration_min` (at 4 km/h);
  - `outside_share` (fraction of the length outside inter-rows and passages);
  - `n_targets`, `n_visited_est`;
  - `baseline_length_m`;
  - `source` (`model` | `marcaj`), `solver`, `policy`.

**`measurements.csv`** is UTF-8 with a header. Metres and m² have 2 decimals, hectares have 4. It has one `survey` line, one `block` line per `vineyard_id` and one `row` line per `row_id`:

```
level,vineyard_id,row_id,block_count,row_count,row_length_m,canopy_area_m2,canopy_area_ha,interrow_area_m2,interrow_area_ha,plant_count,row_structure
```

| Column | Meaning |
|---|---|
| `block_count` · `row_count` | distinct `vineyard_id` / `row_id` |
| `row_length_m` | a row is the sum of its pieces; block and survey values are the sums of their rows |
| `canopy_area_m2` / `_ha` | area of the **union** of the canopy polygons (overlaps across tile edges are counted once) |
| `interrow_area_m2` / `_ha` | area of the **union** of the inter-row polygons |
| `plant_count` | number of canopy polygons |
| `row_structure` | row lines only: `disrupted` if any piece is disrupted, `unassessable` if all are, else `regular` |

## 10. Current results (model run, before Marcaj corrections)

These values come from the model annotations: run `full-v2` → `post-full-v2`, whose `run.end` is at 06:04:45 on 26 Sep 2026. **They are the model run before Marcaj corrections, and the final run refreshes them.** The final `route.geojson` and `measurements.csv` are produced from the Marcaj export with `vineyard final`, so their values will differ.

`post-full-v2` ran before the `full-v2` re-run at 08:51, which exports the 3 accepted waste boxes. That re-run left the canopies, rows and inter-rows unchanged (the same parquet contents apart from `model_version`), so the rows from Tiles processed through Plants still hold. The waste, target and route rows change with the post re-run. **TODO before submission:** re-run `post-full-v2` on the current `full-v2` and publish it (§4). Then refresh the waste, target and route rows from that run's `exports/route.geojson`, `metrics/targets.json` and `metrics/route_validation.json`, and from the published root files.

| Quantity | Value |
|---|---|
| Tiles processed | 311 |
| Vineyard blocks | 53 |
| Rows | 695 |
| Total row length | 43,962.50 m |
| Canopy area (union) | 13,575.82 m² (1.3576 ha) |
| Inter-row area (union) | 86,201.00 m² (8.6201 ha) |
| Plants (canopy polygons) | 11,153 |
| Inspection targets | 1,289 (378 must-visit, 911 optional), with no waste target in this run |
| Waste boxes | 0 in this run. 3 accepted waste boxes (SAM 3-verified, ≤ 10 m from a block) are included from the next post run |
| Route length | 17,451.45 m (≈ 262 min at 4 km/h) |
| Route share outside the allowed area | 1.14% (198.7 m; the rule limit is 2%) |
| Route closure at START | 0.00 m |

## 11. Licences and attribution

| Asset | Licence |
|---|---|
| Sireț3 orthomosaic / tiles | CC BY 4.0. Credit 3DATA COLLECT / OpenAerialMap, contributors to the Open Imagery Network (re-projected to EPSG:32635 and tiled by the organizers) |
| Route inputs (passages, forbidden zones), UAT boundary in the web app | contain OpenStreetMap data © OpenStreetMap contributors, ODbL |
| UAVVaste dataset (waste probe positives) | CC BY 4.0, Zenodo record 8214061, doi:10.5281/zenodo.8214061 |
| SAM 3 (`facebook/sam3`) | SAM License (Meta; gated on Hugging Face; not redistributed here) |
| OpenCLIP ViT-B-32 `laion2b_s34b_b79k` weights · `open_clip` | MIT · MIT |
| segmentation-models-pytorch | MIT |
| ResNet-18 ImageNet encoder weights | torchvision `resnet18-5c106cde.pth` (torchvision: BSD-3-Clause), mirrored as `smp-hub/resnet18.imagenet` (licence tag "other"); trained on ImageNet |
| Main libraries | PyTorch and torchvision (BSD-style), transformers (Apache-2.0), OR-Tools (Apache-2.0), python-tsp (MIT), rasterio / shapely / geopandas (BSD-3-Clause) |
| **This repository's code** | **TODO: choose a licence.** There is no LICENSE file yet; `src/AI/pyproject.toml` declares MIT. |
