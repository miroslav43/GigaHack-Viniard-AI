# Plan: implement the Sireț3 vineyard AI pipeline in `src/AI`

## Context

`arhitectura/ARHITECTURA_Siret3.md` (v1.1) and `arhitectura/00_contracte.md` (v1.0) describe a pipeline of about 20 stages for the GigaHack "Vineyard AI Field Challenge":

1. 311 GeoTIFF tiles go in.
2. The AI produces pre-annotations: canopy, rows, inter-rows, attributes, blocks and waste.
3. These are packed as CVAT 1.1 ZIPs and uploaded to Marcaj.
4. The team corrects them by hand in Marcaj.
5. The corrected set is imported back and used to produce targets, the walking route, the measurements and the web data.

`src/AI` holds only `readme.md` today.

**Hard gate:** every tile plus its pre-annotations must be uploaded and **Published in Marcaj by Saturday ~12:00**. It is now Friday ~22:50, so validated ZIPs must exist by ~Sat 07:00 to leave time for QA. The final deadline is Sunday 15:00.

The prototype in `analiza_exemple/scripts/{axes,attrs,fit}.py` is already validated on the 2 reference tiles: row F1 0.98 and canopy score 0.82 end to end. **We port it; we don't rewrite it.**

## User decisions

- **Scope:** all four parts.
  - pre-annotation, ending in ZIPs;
  - the U-Net NN;
  - waste (SAM 3 + probe);
  - post-Marcaj: import, derive, targets, route, measurements, publish, and the web **data** bundle into `src/Web/data/`.

  The `src/Web` frontend belongs to teammates.
- **Layout:** a self-contained project in `src/AI/`.
  - `pyproject.toml`, `uv.lock`, `requirements.lock`, `Makefile`, `Dockerfile`;
  - `configs/`, the package `vineyard/`, `tests/`, and `work/` (gitignored).

  `publish` writes `route.geojson` and `measurements.csv` to the **repo root**. The main `README.md` is also at the repo root.
- **Git:** commit and push to `origin main` after every phase gate. Stage only `src/AI/**`, `.github/**`, and the root deliverables and README. Never stage `arhitectura/` or stray JPEGs unless asked.

## Facts measured by the design agents that change the docs

| Fact | Consequence |
|---|---|
| Reference canopies have **integer vertices only** (max 2047), i.e. raw `findContours`. Vector area is 237.12 / 299.06 m²; the pixel set is 256.95 / 319.36 m². | Canopy vectorization: **no +0.5, no outset**. The min-area filter (0.19 m²) runs on the vector area. Rows and interrows keep the continuous convention (ends exactly on 0 and 2048). The eval sweep {raw, raw∩corridor, contract +0.5/0.5} confirms the choice before RC0. Contract erratum in §13. |
| Greedy packing at ≤ 85 000 000 B yields **5 parts**, for any XML size from 26 to 80 KB per tile. | The part count is derived, not fixed at 7. The Marcaj guide's "5 parts" still holds. |
| Reference canopy∩interrow overlap is 0.033–0.037 m² per tile. Interrows are clean 4–5 vertex quads. | One key, `export.cvat.max_canopy_interrow_overlap_m2 = 0.05`, checked after conversion to px. Subtract canopy from interrows only when a tile exceeds it. Canopies get no second simplify in the writer. |
| Gap edge cases: R15 = 5.009 (`regular`), R06 = 5.054 (`disrupted`). Measured on the pixel set, all **51/51** are correct. | A single gap engine measures on canopy **pixel sets**, threshold 5.0. The test requires 51/51. |
| Example row IDs have 2 digits (`V01-R01`). | Model output emits `R001`. Reference and Marcaj imports use relaxed regexes and never rename. |
| `facebook/sam3.1` (already downloaded to `models/sam3.1/`, 3 502 755 717 B, still named `.incomplete`) is a raw `.pt` with a `Sam3VideoModel` config and **no transformers weights**. | SAM adapter backends, in order: (1) transformers `Sam3Model` from `facebook/sam3`, which needs its own access request; (2) the official `sam3` package with the local 3.1 checkpoint, CPU, unverified on Mac; (3) disabled. |
| START is inside passages polygon 1, 0.868 m from the edge. Passages have 2 disconnected components (one of 2 137 m²) and CW exteriors. | The route runs on the main component. The web export re-orients rings (RFC 7946). |
| brew python 3.14 has a broken `pyexpat`. | Use the uv-managed 3.12 and parse XML with `lxml`. |

## Conflict resolutions (ARHITECTURA wins over the contract, per its §10; integration fixes from the critics)

**Between the two docs**

- **Storage:** GeoParquet, one file per layer, plus AnnSet (contract §2).
- **Environment:** Python 3.12 + uv with the arch §4.1 pins, all verified on PyPI. Also pin `lxml`, `scikit-learn` and `sknw`.
- **Rows and IDs:**
  - row spacing gate 1.8–3.8 m;
  - IDs per contract §2.4;
  - R001 is the northernmost row along the block normal;
  - V01 is sorted by `(−round(cy), round(cx))`.
- **CVAT shape `source`:** `manual`, via config.
- **Route:** arch §4.11. GTSP via OR-Tools `AddDisjunction`, python-tsp fallback, a 2-level validator, and **no `is_simple`** check.

**Design integration (single owners, from the critics)**

1. **Stage protocol:** `pipeline/registry.StageSpec` + `STAGE = StageSpec(...)` + `run_tile_stage` for every stage. There is no `dag.py`.
   - `PRE = ingest, tile_prep, nn_infer, rows_detect, rows_link, blocks, canopy, interrow, row_attrs, waste, assemble, qa_previews`
   - `POST = import_marcaj, derive, passable, targets, route, measure, web_bundle`
   - `export_cvat`, `publish`, `eval` and `import_reference` are standalone commands.
2. **Config:** a single `vineyard/config/` package with `sections_{core,io,perception,nn,waste,post}.py`.
   - `AppConfig` lists every section, including `qa`, `derive` and `publish`.
   - One key per concept: `nn.fusion` A–E; `nn.use_in_rows`; `veg.shadow_v_max`; `runtime.n_workers`; `waste.decide.*`; `route.solver.*`.
   - Only the orchestrator edits `default.yaml`.
3. **CVAT and AnnSet:** `cvat/*`, `annset/{model,io}` and `cvat/to_annset.py` are owned once, by the foundation.
   - `AnnSetMeta` gains `tile_ids`.
   - Post keeps only `annset/import_checks.py` plus the `import_marcaj` stage.
4. **One owner for each shared rule:**
   - `perception/vegmask.py` (perception);
   - `nn/{probs,fusion}.py` (ML; torch-free);
   - `contracts/ordering.py`;
   - `contracts/qa.py` (`QaIssue`, `issues_to_gdf`, Q00001);
   - a single gap engine in `perception/attrs.py`, reused by `targets` and `import`.
5. **IDs:**
   - row candidates use `K\d{2,3}`;
   - piece IDs take a `#k` suffix after `explode`;
   - `TargetKind` gains `missing_plant` (T-MSP) and `sparse` (T-SPR);
   - interrows are numbered by geometric `row_index`, not `min(k)`, so R900 rows don't collide;
   - `CONTRACT_VERSION = "1.1"`.
6. **Waste:** the `waste` stage writes the final `layers/waste.parquet`, merging `configs/waste_confirmed.csv`. `assemble` only reads it.
7. **Post runs** write to their own `runs/<ts>-post-<h6>/` (`run.json` records `annset_ref`). They never overwrite model-run layers.
8. **`tile_valid`** is static at `work/layers/tile_valid.parquet`. Post fails loudly if it is missing.
9. **Enum validation** is relaxed for `source ∈ {marcaj, reference}`: a qa warning, not an error. The raw value is kept.
10. **`measurements.csv`:** interrow area is written as a sum (arch §4.12); the JSON also carries the union.
11. **Score-protecting fixes added to perception:**
    - "skip-one" block edges at [1.6, 2.3]×s, so a missing row doesn't split a block (flag `missing_row_suspect`);
    - chains split at unsupported holes ≥ 5 m;
    - pieces with no candidate support are flagged `row_interpolated` and need vegetation evidence before canopies are drawn on them;
    - `canopy_on_non_vineyard_tile` is flagged in `assemble`.

## Package layout (`src/AI/`, flat: `vineyard/` next to `pyproject.toml`)

```
_env.py errors.py logging_setup.py doctor.py cli.py cli_options.py cli_post.py (post sub-app, mounted)
config/      loader, hashing, model, sections_*          contracts/  enums ids ordering qa schema_defs schemas tile_grid.txt
geo/         tiling(contract §8) raster polygonize vector_io ops
pipeline/    registry context runner cache atomic parallel(spawn) timings tile_index tile_cache stages/*
ingest/      tiles route_inputs                           cvat/  template model writer reader rle normalize to_cvat to_annset
                                                                 report validator_doc validator_zip packer selfcheck export testzip cli
perception/  vegmask profile linefit row_features rows_detect rows_link blocks_graph blocks overrides corridor canopy trees
             interrow attrs topology types  waste/{candidates lattice filters nms crops decide assign confirm review verify
             uavvaste positives negatives clip_embed probe sam3_adapter cli}
nn/          probs fusion pseudolabels patch_store dataset model losses schedule train validate infer weights ablation panels cli
annset/      model io assemble import_checks merge_rows row_order blocks link_interrows derive
route/       domain skeleton centerlines graph_types graph connectors graph_io targets target_rules candidates distances
             solver solver_fallback unroll planner baseline validate geojson
measure/     measurements             web/  geojson4326 panels ortho bundle          eval/  matching metrics report
qa/          render review overview   tests/<area>/…  (pytest --import-mode=importlib, markers needs_tiles/slow/examples/mps/network)
```

- Every file stays ≤ 400 lines (800 hard) and every function < 50 lines.
- Dataclasses are frozen and every threshold lives in config (team style).
- **Detailed specs:** the 4 design documents and 2 critiques from the design workflow. Step 0 of implementation saves them to `src/AI/docs/design/` and commits them. Where they conflict, this plan wins.

## Orchestration mechanics

- **Implementation runs as Workflow fan-outs.**
  - Each work package (WP) gets one agent in its own git worktree (`isolation: 'worktree'`), on branch `wp/<id>`.
  - Each agent exclusively owns the files listed for its WP.
  - The orchestrator (me) merges with `--no-ff` after the WP gate passes, then runs `make test` on `main`.
- **Shared data and work directories:**
  - `VINEYARD_DATA_ROOT="/Users/maleticimiroslav/Vin Gigahack/data & info"`.
  - `VINEYARD_WORK_DIR` is the main checkout's `src/AI/work`, so tiles and cache are shared.
  - At every gate, `VINEYARD_REQUIRE_TILES=1` turns `needs_tiles` skips into failures (worktrees have no tile ZIPs).
- **Orchestrator-only files after G0:**
  - `pyproject.toml`, the locks, `Makefile`, `Dockerfile`;
  - `configs/*`, `vineyard/config/**`, `cli*.py`;
  - `pipeline/{registry,context}.py`, `contracts/**`, `annset/{model,io}.py`;
  - `geo/{tiling,raster,vector_io}.py`, `perception/{vegmask,types}.py`, `nn/{probs,fusion}.py`, `cvat/{model,template}.py`;
  - `tests/conftest.py`, `tests/helpers/**`.

  An agent that needs a new key adds a `# CONFIG-REQUEST` constant instead. `grep` for that marker must return nothing at each gate.
- **WP done criteria:**
  - its tests are green;
  - `ruff check` is clean;
  - coverage is ≥ 80% on its modules;
  - `git diff --name-only main...wp/<id>` lists only owned files.
- **Resources:**
  - Only the orchestrator runs 311-tile jobs (`--workers 8`). Agents use the 2 examples with `--workers ≤ 2`.
  - Nothing imports torch while a CPU pool runs.
  - NN training and SAM never overlap each other or a full run.

## Phases, work packages, gates (times Fri/Sat, EEST)

### Phase 0: environment and frozen contracts (23:00 → G0a 23:55, G0b 00:45)

**P0.1 (orchestrator alone)**
- Save the design docs.
- Create `pyproject.toml` with `.python-version` 3.12 and three extras: `nn`, `waste-ml`, `route`.
- Run `uv lock` + `uv sync`: the core first, then the extras. Timebox 25 min. Fallbacks: `numpy<2.5`, torch 2.13.x, drop `sknw`.
- Also create `.gitignore`, `.dockerignore`, the Makefile skeleton (unexport `PROJ_DATA`/`PROJ_LIB`/`GDAL_DATA`, thread env vars, quoted paths), the pytest config and CI `.github/workflows/ai-tests.yml`.
- **Gate G0-env:**
  ```
  uv sync --locked
  uv run python -c "import rasterio,shapely,geopandas,pyogrio,pyproj,cv2,skimage,scipy,lxml,torch; print(torch.backends.mps.is_available())"
  ```

**P0.2: 3 agents on disjoint files**

| WP | Owns |
|---|---|
| **P0-CFG** | `_env`, `errors`, `logging_setup`, `doctor`, `cli`, `cli_options`, `config/*`, `default.yaml` (every key from all 4 designs), `pipeline/{registry,context}` |
| **P0-CON** | `contracts/*` (incl. `tile_grid.txt` with the 311 IDs), `annset/{model,io}`, `cvat/{model,template}`, torch-free `nn/{probs,fusion}` |
| **P0-GEO** | `geo/{tiling,raster,vector_io}` (+ `ops` and `polygonize` stubs), `perception/{vegmask,types}` (port of `veg_mask`), `pipeline/tile_cache.py`, `tests/helpers/*`, and a `perception/corridor.py` stub with signatures |

- **G0a:** every foundation module imports. Commit and push. **Phase 1 fans out from here.**
- **G0b:**
  - `pytest tests/{contracts,geo,config,cli}` passes with coverage ≥ 85%;
  - `vineyard doctor` passes;
  - tiling oracles hold: r006_c004 has x0 = 629196.8, y0 = 5220915.2, and START maps to r018_c010 at (28.0, 2002.0);
  - an import-hygiene test confirms no torch is loaded.

  Commit and push `phase0`.

### Phase 1: parallel build of the pre-annotation path (00:00 → G1 03:30)

| WP | Exclusive files | Milestone / acceptance |
|---|---|---|
| **P1-PIPE** | `pipeline/{runner,cache,atomic,parallel,timings,tile_index}`, `ingest/*`, stages `ingest`, `tile_prep` | Ingest of 311 tiles in < 60 s, tags vs grid, sha256 (00:45). `tile_prep` on all 311 tiles with 0 failures; rerun fully cached (01:30). |
| **P1-GEO** | `geo/{ops,polygonize}` (parametric `offset`/`outset`, default raw), then continues as P1-WST | Clip, orient CCW, `make_valid`, notch holes (≥ 1 px), split multi (01:15) |
| **P1-CVIO** | `cvat/{writer,reader,rle,normalize,to_annset,testzip,cli}`, stage `import_reference` | **M1 01:00:** byte-identical example round-trip plus `make-test-zip` (examples + 1 waste box + 1 empty tile) |
| **P1-CVEX** | `cvat/{to_cvat,report,validator_doc,validator_zip,packer,selfcheck,export}`, stage `export_cvat` | **M2 02:30:** empty-AnnSet export of 311 tiles → 5 ZIPs, each ≤ 85 000 000 B, validated. Name bijection, sha256, whitelist, `id_registry.json`, `empty_tiles.csv`, manifest. |
| **P1-EVAL** | `eval/{matching,metrics,report}`, stage `evaluate` | Official vector metrics; reference vs reference = 1.0; baseline/gates CLI |
| **P1-ROWS** | `perception/{profile,linefit,row_features,rows_detect}`, stage `rows_detect` | Port of `dominant_angle` / `detect_rows`: autocorrelation spacing, SNR gate, on-lattice, Huber fit, multi-orientation. F1 ≥ 0.95 per example tile; all 51 reference rows survive the filters; ≤ 4 s/tile. |
| **P1-BLK** | `perception/{rows_link,blocks_graph,blocks,overrides}`, stages `rows_link`, `blocks`, empty `configs/overrides.yaml` | Support-line union-find; grid `row_id`; skip-one edges; passage cuts; one block per example, deterministic |
| **P1-CAN** | `perception/{corridor,canopy,interrow,trees}`, stages `canopy`, `interrow` | `corridor.py` real within 45 min. Canopy ≥ 0.84 with reference axes (raw convention). Interrow IoU ≥ 0.98, cover 49/49. |
| **P1-ATT** | `perception/{attrs,topology}`, `annset/assemble`, `qa/*`, stages `row_attrs`, `assemble`, `qa_previews` | row_structure **51/51** (pixel-set gaps); 0 errors on the reference AnnSet; previews in < 1 s; `review_queue.csv` |
| **P1-WST** (from 01:15) | `perception/waste/{types,candidates,lattice,filters,nms,crops,decide,assign,confirm,review,verify}`, stage `waste`, `configs/waste_confirmed.csv` header | Rule-based candidates plus review HTML; **0 auto-accepted boxes** (export only rows in `waste_confirmed.csv`); 0 candidates exported on the examples |

**G1 (on `main`):**
```
VINEYARD_REQUIRE_TILES=1 pytest -m "not slow"
VINEYARD_REQUIRE_TILES=1 pytest -m slow tests/perception tests/cvat
pytest --cov=vineyard   # ≥ 80% on logic modules
ruff check
```
Also: the CONFIG-REQUEST grep is empty and the M1/M2 artefacts exist. Commit and push `phase1`.

### Phase 2: integration, RC0, RC1 (03:30 → 07:00)

1. **Integration on the 2 examples:**
   ```
   vineyard run --run-id i1-ex --tiles siret3_r006_c004 --tiles siret3_r021_c012 --until assemble
   vineyard eval-examples --gates
   ```
   Also run the canopy convention sweep and write the baseline.
   **G2:** canopy ≥ 0.80, row F1 ≥ 0.95 per tile, interrow IoU ≥ 0.95, attributes ≥ 0.90, one block per example. A gate may be waived only for a miss of ≤ 0.01, documented.
2. **Full run and RC0 (~04:30):**
   ```
   /usr/bin/time -l vineyard run --until qa_previews --workers 8
   vineyard export-cvat --annset LATEST_MODEL
   ```
   Re-run the eval gates on the full-run AnnSet restricted to the 2 example tiles. Commit and push tag `rc0`. **From this point a valid upload always exists.**
3. **Hardening until 06:15.** Each item is behind a flag, owned by the author of that file, and merged only if it is a no-op on the examples (±0.005):
   - triage `overview.jpg` and priority-1 `review_queue`;
   - calibrate SNR on the 311-tile distribution;
   - orchard rejection, tree filter, texture/Otsu fallback, curved-row tracking;
   - measure the route-feasibility proxy: the share of interrow ends within 0.5 m of passages (critic S5).
4. **RC1 (06:15–07:00):** re-run, `export-cvat`, the independent ZIP checks (§Verification) and the baseline regression (max drop 0.01). Commit and push tag `rc1`. RC1 is read-only from here.

**Decision points**
- **01:30:** if `tile_prep` is not done, the orchestrator takes it over.
- **04:15:** if G2 still fails, ship the best config with a waiver. Repair in this order: rows, canopy, IDs, interrow, attributes.
- **05:30:** if the export is not clean, use the degradation ladder.
- **10:40:** if RC2 fails, upload RC1.

**Degradation ladder**
- **L1:** flags off.
- **L2:** empty waste layer.
- **L3:** `force_empty` for failing tiles.
- **L4:** upload the M2 empty-annotation ZIPs, so Publish still happens.

### Phase 3: human QA, RC2, upload, Publish (07:00 → 12:00)

- **QA, 07:00–10:00:**
  - fill `configs/overrides.yaml` (force-empty, add/delete/extend rows);
  - fill `waste_confirmed.csv` from the review HTML.
- **Off-path branches in the meantime, merged only after 10:30:**
  - **NN** WP-N1/N2: pseudo-labels, train on MPS 07:15–09:45 (after a 1-epoch smoke run), ablation A–F. It enters RC2 **only** if it wins on **both** example tiles by 09:30; otherwise `nn.fusion=A`.
  - **Post** WP2 (derive), WP4 (passable) and WP5 (route), on synthetic data.
- **RC2 (10:00–10:30, config and data only):**
  ```
  vineyard run --until qa_previews
  vineyard export-cvat
  ```
  Then the checks. Commit and push tag `upload-final`.
- **Upload and Publish (humans), 10:30–12:00.**

### Phase 4: after Publish (Sat 12:00 → Sun 02:00)

- **Merge the off-path branches.**
  - **Post:** `import_marcaj` + `import_checks`; WP3 (targets, measure, publish); WP6 (web bundle into `../Web/data`: 4326 GeoJSON at 7 decimals, `rows.json`/`blocks.json`/`route.json`, `summary.json`, `manifest.json`, and a per-tile 512 px JPEG + `ortho_index.json`, since gdal and tippecanoe are absent).
  - **ML:** N3 ablation and panels (a deliverable).
  - **Waste:** W2 (UAVVaste positives, CLIP probe) and W3b (SAM adapter). Both only feed a **ranked list for manual boxes in Marcaj**.
  - **Delivery:** Docker (2-stage uv, torch `+cpu`) and the root README.
- **Headland strategy:** decide it from the proxy measured in Phase 2. Options: enter only from interrow ends that touch passages, out-and-back into dead ends, or mark headland-only targets `reachable=false`.
- **Gates:**
  - `vineyard post --annset LATEST_MODEL` runs end to end by ~18:00, with `outside_frac ≤ 0.005`, closure ≤ 0.01 m, and 100% of reachable targets within 2 m;
  - at ~20:00, the intermediate Marcaj export goes through `from-marcaj` and the review list goes to the annotators;
  - Docker is built and pushed once, then checked with `docker run --network none`.

### Phase 5: Sunday final (08:00 → 14:00)

1. At 11:00, every job must be Submitted. Take the final export.
2. Run:
   ```
   vineyard from-marcaj <zips>
   vineyard post --annset LATEST_MARCAJ --set route.solver.final=true
   vineyard publish --set publish.require_source=marcaj
   ```
3. Commit the final Marcaj export under `src/AI/data/marcaj_final/`, pinned by sha256 so `make final` is reproducible.
4. Take the README timings from `vineyard bench`.
5. Commit and push by 13:00. Code freeze at 14:00.

## Human actions (not automatable)

| When | Action |
|---|---|
| **Now, before Slack closes at 23:00** | Send arch §8 questions Q1, Q3, Q5, Q7, Q8 (Q4/Q6 are moot: 5 parts fit). |
| Now | 1. `hf auth login`. 2. Request access to **`facebook/sam3`**, the repo transformers can load. 3. Re-run `hf download facebook/sam3.1 --local-dir models/sam3.1` so it finalizes the `.incomplete` file (the bytes are complete). |
| ~01:15 (M1) | Upload the test ZIP to Marcaj: examples + 1 box + 1 empty tile. Screenshot the import report. Check placement. **Export CVAT 1.1** to `src/AI/tests/fixtures/marcaj/`. Then Remove all. |
| ~02:45 (M2) | Upload one real ~80 MB part and time it. Remove all. |
| 07:00–10:00 | QA previews by priority; fill in `overrides.yaml`, `waste_confirmed.csv` and the empty-tiles list. |
| 10:30–12:00 | Upload the 5 parts one at a time with screenshots. Check **311 files**, then **Publish**. |
| Sat 12:00 → Sun 11:00 | Correct the 63 jobs, tick "No objects" on the empty tiles, Submit each job. |

## Verification (end to end)

- **P0:** G0-env, G0b, `vineyard doctor`, the tiling oracles, and `tile_grid.txt` == the 311 ZIP names.
- **Per WP:** its TDD list from `docs/design/*`.
  - byte-identical round-trip;
  - the packer yields 5 parts ≤ 85 000 000 B;
  - reference vs reference = 1.0;
  - 51/51 row_structure;
  - interrow IoU ≥ 0.98 with reference axes;
  - 0 waste exported on the examples;
  - post: 8 GAP targets / 0 END on the reference, measurements 1941.55 m and 51 rows, synthetic route-validator cases (3% outside → rejected; a valid out-and-back → accepted).
- **G2/RC:** `eval-examples --gates` on the full 311-tile run, and `timings.json` total ≤ 20 min.
- **Independent ZIP checks:**
  - `vineyard cvat validate *.zip`;
  - `stat -f %z` ≤ 85 000 000;
  - `unzip -Z1` lists only `annotations.xml` and `images/siret3_rNNN_cNNN.tif`;
  - the union is 311 unique tiles;
  - self-check IoU ≥ 0.98;
  - 5 random previews checked by eye.
- **Marcaj round-trip:** the test-export fixture parses with no structural errors, IoU ≥ 0.999 against what was uploaded.
- **Post and Sunday:** `vineyard post` end to end; `publish` refuses bad cases; the web `manifest.json` loads in `src/Web`; `docker run --network none`.
