# Execution plan: Siret3 `src/AI` (critic's report)

This plan was written at 22:33 EEST (checked with `date`). It assumes implementation starts at about 23:00. All times are Chisinau time. The project root is `/Users/maleticimiroslav/Vin Gigahack/src/AI`.

## 0. Things checked while planning (read-only)

**Repository and machine**
- `src/AI` contains only `readme.md`. The git history has a single commit. `origin` is `github.com/miroslav43/GigaHack-Viniard-AI-`.
- `arhitectura/` and a few JPEGs in the repo root are untracked. Phase commits should add only `src/AI/**` and `.github/**`.
- The uv cache is empty, so the first `uv sync` downloads torch, OpenCV and the rest.
- Free disk is 34 GiB. `models/` already holds 3.0 GB of the SAM 3.1 download.
- Available tools: `pdftotext`, `docker` and `/opt/anaconda3/bin/hf`. Not installed: `gdal2tiles` and `tippecanoe`.

**Rules from the brief and the data README**
- Pre-annotations can be imported once, before Publish, and only together with the tiles.
- Before Publish the uploaded files can be deleted with "Remove all" and imported again.
- Slack support answers 09:00–23:00. Anything Marcaj does unexpectedly tonight can only be diagnosed by testing it ourselves.
- Only submitted jobs are scored. The project has 63 jobs of 5 tiles each.

**Worktree pitfall**
- Tile ZIPs are gitignored, but the examples and `02_route` are committed.
- A git worktree placed anywhere else will therefore have the examples but no tiles. Any `needs_tiles` test there would **skip silently and report green**.
- This is handled in §2 and risk R3.

---

## 1. Conflicts between the four designs, resolved in Phase 0

These must be resolved and frozen before fan-out. Without that, the designs cannot be built in parallel. In each row the foundation design wins unless the row says otherwise.

| # | Conflict | Frozen decision |
|---|---|---|
| X1 | Stage protocol. Foundation uses `StageSpec` plus `STAGE`. Perception assumes `pipeline/dag.py` with `run_tile(ctx, t)`. ML and waste use `run(ctx, tile_ids)`. | Use foundation's `StageSpec(run: Callable[[RunContext], StageResult])` plus `run_tile_stage`. Multi-step stages (interrow = global bands then per tile; waste = pool, then main-process verify, then finalize) do all their steps inside `run`. There is no `pipeline/dag.py`. |
| X2 | Stage order in the registry | `PRE_STAGES = ingest, tile_prep, nn_infer, rows_detect, rows_link, blocks, canopy, interrow, row_attrs, waste, assemble, qa_previews`. `POST_STAGES = import_marcaj, derive, passable, targets, route, measure, web_bundle`. This applies post's requested edges passable→targets and targets→measure. `export_cvat`, `publish`, `eval` and `import_reference` are standalone commands. `STAGE_MODULES` is pre-filled with every name. |
| X3 | Config lives in four places: `config/sections_*`, `perception/settings.py`, `nn/config.py`, `waste/config.py`, `config/post_*.py` | One package, `vineyard/config/`, with `sections_core`, `sections_io`, `sections_perception`, `sections_nn`, `sections_waste` and `sections_post`. `AppConfig` gets every section: contract_version, project, paths, grid, runtime, nodata, veg, nn, rows, orchard, blocks, canopy, interrow, row_structure, waste, qa, derive, targets, route, export, import, measure, publish, web, eval, logging. P0-CFG writes **every key from all four design tables**. `perception/settings.py`, `nn/config.py`, `waste/config.py` and `cli_post.py` are **not** created. |
| X4 | Duplicate keys | `nn.fusion` (A–E, default A) is the only fusion key; `canopy.nn_fusion` is dropped. `nn.use_in_rows` replaces `rows.detect.use_nn_weight`. `veg.shadow_v_max` = 50 lives under `veg` (C's `interrow.shadow_v_max` is dropped). `canopy.vector_offset_px` and `canopy.vector_outset_px` both exist; their defaults are decided by the G2 sweep (R5). `ImportConfig` is defined once, in `sections_io`, and also includes post's `require_all_tiles`, `min_line_length_m` and `accept_masks`. `nn.enabled: false` until after Publish. |
| X5 | Tile cache paths | Use the contract paths: `cache/valid/`, `cache/veg/`, `cache/stats/<t>.json` (not `tile_stats`), plus a new `cache/vis/<t>.png` (uint8 codes 0–3). `tile_prep` writes all four. |
| X6 | `veg_mask`, `fuse_masks` and `upsample_prob` are each defined twice | Exactly one `perception/vegmask.py` (neg_a, veg_mask, vis_codes, otsu_threshold, `compute_tile_masks`). `fuse_masks`, `upsample_prob` and `load_canopy_prob` exist only in the torch-free `nn/fusion.py` and `nn/probs.py`. `FusionVariant` (A–E) goes in `contracts/enums.py`. |
| X7 | QA issue type is defined three ways | `contracts/qa.py` holds `QaIssue(severity, code, tile_id, object_id, message, x, y)`, `issues_to_gdf` and Q00001 numbering. `QA_CODES` is the union of the codes in all four designs. `cvat/report.Issue` stays as a separate type for ZIP validation. |
| X8 | CVAT reader and import are designed twice (foundation WP-D and post WP1) | Foundation owns `cvat/{reader,rle,normalize,to_annset}` and `stages/import_reference.py`. Post's `RawDoc` API and its `annset/{from_cvat,canopy_rows,io}.py` are dropped. Post keeps `annset/import_checks.py` and `stages/import_marcaj.py`. **Normalization policy follows post:** any change to a raw value emits a warning, because organizers score raw Marcaj values. Foundation's `" Regular "` → no-issue test is changed to expect a warning. |
| X9 | AnnSet metadata | Foundation's frozen `AnnSetMeta`, plus `tile_ids: tuple[str, ...]`. This lists every imported image, including empty ones, and post needs it for coverage. |
| X10 | Ordering is implemented three times | Only `contracts/ordering.py`. `perception/ordering.py` is not created. Post's `lines.py` imports `canonical_normal` and `order_by_normal`. |
| X11 | Gap engine is implemented twice | `perception/attrs.py`, owned by perception, is the only one. Its API must support `unknown` intervals, `include_ends`, and `Gap(start_m, end_m, kind ∈ {interior, head, tail}, censored)`. Post's `annset/gaps.py` is dropped. Post's measured values (R09 12.9, R07 11.33, R15 5.0) become cross-check tests in `test_attrs`. |
| X12 | IDs | Row candidates: `K\d{2,3}`. Waste candidates: `<tile>:W0001`. `TargetKind` adds `missing_plant` (`T-MSP`) and `sparse` (`T-SPR`), using post's codes, not foundation's MIS/SPA. |
| X13 | Layers added by several designs | `schema_defs` includes: `rows_raw`, `row_pairs`, `rows_rejected`, `interrow_pieces_linked`, `target_visits`, `waste_candidates` (extra columns allowed), `row_candidates` extra columns, relaxed enums when source ≠ model. `CONTRACT_VERSION = "1.1"` is set once in P0; later changes must be additive. |
| X14 | Waste → assemble coupling | The `waste` stage writes the final `layers/waste.parquet` in the AnnSet schema itself, calling `merge_confirmed` with `configs/waste_confirmed.csv`. `assemble` only reads that file. There is no function dependency between the two. |
| X15 | Per-stage CLI flags | P0 `cli.py` wires every command generically. Stage-specific flags map to `--set` overrides; for example `--partial` means `import.require_all_tiles=false`, and `--final` means `route.solver.final=true`. Package owners never edit `cli.py`. |
| X16 | ML's names for canopy and eval functions | `corridor_mask` → `perception.corridor.corridor_labels`. `label_components` / `mask_to_canopies` → `perception.canopy.extract_canopies` (pass the fused mask as `veg`, variant A). `canopy_score` → `eval.metrics.canopy_metrics`. ML adapts to these after Publish. |
| X17 | `source` attribute on shapes | Default is `manual` (user decision), overriding arch "auto". |

---

## 2. Orchestration rules

1. **Worktrees.** Each Phase 1+ package gets its own branch `wp/<id>` in its own worktree. Each worktree runs `uv sync --locked`; on APFS uv clones files, so this is cheap. **Never set `UV_LINK_MODE=copy` locally.**
2. **Data paths are absolute and shared.** The config loader honours `VINEYARD_DATA_ROOT` and `VINEYARD_WORK_DIR`.
   - Every worktree exports `VINEYARD_DATA_ROOT="/Users/maleticimiroslav/Vin Gigahack/data & info"`.
   - Every worktree exports `VINEYARD_WORK_DIR` pointing at the main checkout's `src/AI/work`, where `work/tiles` and `cache/` are shared.
   - Per-worktree runs use distinct `--run-id` values.
   - At every gate, `VINEYARD_REQUIRE_TILES=1` turns a `needs_tiles` skip into a failure.
3. **pytest settings, set in P0.**
   - `addopts = "--import-mode=importlib --strict-markers"`, `pythonpath = ["."]`, `tests/helpers/__init__.py`.
   - Markers: `needs_tiles`, `slow`, `examples`, `mps`, `network`.
   - importlib mode is required because the designs reuse test basenames: `test_blocks.py`, `test_ordering.py` and `test_canopy`.
4. **Files only the orchestrator edits after G0:**
   - `pyproject.toml`, `uv.lock`, `requirements.lock`, `Makefile`, `Dockerfile`
   - `configs/default.yaml` and `vineyard/config/**`
   - `cli.py`, `cli_options.py`, `pipeline/{registry,context}.py`
   - `contracts/**`, `annset/{model,io}.py`
   - `geo/{tiling,raster,vector_io}.py`, `perception/{vegmask,types}.py`
   - `pipeline/tile_cache.py`, `pipeline/stages/_raster_source.py`
   - `nn/{probs,fusion}.py`, `cvat/{model,template}.py`
   - `tests/conftest.py`, `tests/helpers/**`

   A package that needs a new config key adds a module-level `Final` constant tagged `# CONFIG-REQUEST`. The orchestrator moves it into config when merging. At a gate, `grep -rn CONFIG-REQUEST vineyard` must return nothing. `perception/types.py` holds cross-package types only; types used inside one package stay in that package's modules.
5. **Merge protocol.** A package is done when:
   - its tests are green in its worktree;
   - `uv run ruff check`;
   - coverage is at least 80% on its modules;
   - `git diff --name-only main...wp/<id>` lists only files it owns.

   The orchestrator then merges with `--no-ff`, runs `make test` on main, and pushes when the phase gate passes.
6. **Main freeze windows.**
   - 06:00–07:00 and 09:45–10:45 Sat: main accepts only fixes to the pre-annotation path.
   - Off-path branches (NN, post, waste ML) merge only after the RC2 export (~10:30).
7. **Machine resources.**
   - Only the orchestrator runs 311-tile jobs (`workers=8`).
   - Agents test on the examples with `--workers 2`.
   - Nothing that imports torch runs while a pool is active. NN training and SAM never run alongside each other or alongside a full run.

---

## 3. Phases, work packages, gates

### Phase 0: environment and frozen contracts (23:00–00:45, feature work does not fan out before G0a)

**P0.1, run by the orchestrator alone (23:00–23:25)**
- Owns: `pyproject.toml`, `.python-version` (3.12), `uv.lock`, `requirements.lock`, `.gitignore`, `.dockerignore`, a skeleton `Makefile`, every package `__init__.py`, `tests/conftest.py`, pytest config, and `.github/workflows/ai-tests.yml`.
- Lock order:
  1. Core pins first: arch §4.1 plus `lxml`, `pyyaml`, `rich`, `scikit-learn`, `pytest`, `pytest-cov`, `ruff`.
  2. Then the extras: `nn` (torch, torchvision, smp), `waste-ml` (transformers, open-clip-torch, huggingface_hub), `route` (ortools, python-tsp, sknw).
- If an extra blocks resolution for more than 10 minutes, drop it and log it. It is re-added in Phase 3 or 4.
- **Gate G0-env:**
  ```
  cd "/Users/maleticimiroslav/Vin Gigahack/src/AI"
  unset PROJ_DATA PROJ_LIB GDAL_DATA
  uv sync --locked
  uv run python -c "import rasterio,shapely,geopandas,pyogrio,pyproj,cv2,skimage,scipy,lxml,pydantic,typer; import torch; print(torch.backends.mps.is_available())"
  git push --dry-run origin main
  ```

**P0.2, three agents on disjoint files (23:25–00:45)**

At **23:55 (G0a)** the interfaces are frozen: signatures and dataclasses are committed, and the modules that must be real are implemented. The rest is implemented and tested by **00:45 (G0b)**.

| Package | Exclusive files | Must be real at G0a (the rest may be stubs) |
|---|---|---|
| **P0-CFG** | `_env.py`, `errors.py`, `logging_setup.py`, `cli.py`, `cli_options.py`, `doctor.py`, `config/*` (all sections per X3/X4), `configs/default.yaml`, `configs/local.example.yaml`, `pipeline/registry.py`, `pipeline/context.py`, tests `test_config/test_logging/test_cli` | `load_config`, `cfg_hash`, full `AppConfig`, `RunContext`, `RunPaths`, `new_run_context`, `StageSpec`, `STAGE_MODULES`, `load_stage` |
| **P0-CON** | `contracts/{__init__,enums,ids,ordering,qa,schema_defs,schemas}.py`, `contracts/tile_grid.txt`, `annset/{model,io}.py`, `cvat/{model,template}.py`, `nn/{probs,fusion}.py` (torch-free), tests | enums, ids, ordering, `LAYER_SCHEMAS` (every layer from X13), `QaIssue`, `AnnSet`, `CvatShape/Image/Document`, `META_BLOCK`. `validate_layer` may be lenient at G0a. |
| **P0-GEO** | `geo/{tiling,raster,vector_io}.py`, stub `geo/{ops,polygonize}.py` (`mask_to_polygons`/`valid_polygon` moved out of `raster.py` so each file has one owner), `perception/{vegmask,types}.py`, `pipeline/tile_cache.py` (load_veg/valid/vis/stats, `tile_prep_key`), `pipeline/stages/_raster_source.py`, `tests/helpers/{synth,examples,proto_metrics}.py`, stub `perception/corridor.py` (signatures of `clip_rows_to_tile`, `corridor_polygon`, `corridor_labels`, owned by P1-CAN) | tiling (all contract functions), `read_tile`, `rasterize_px`, mask PNG I/O, `write_layer`/`read_layer`/`write_geojson`, `compute_tile_masks` |

- **G0a (23:55).** `uv run python -c "import vineyard.config, vineyard.contracts.schemas, vineyard.geo.tiling, vineyard.annset.model, vineyard.pipeline.registry, vineyard.perception.vegmask"` succeeds, then commit and push. **Phase 1 fans out from this commit.**
- **G0b (00:45).**
  ```
  VINEYARD_REQUIRE_TILES=1 uv run pytest tests/contracts tests/geo tests/config tests/cli -q \
    --cov=vineyard.contracts --cov=vineyard.geo --cov=vineyard.config --cov-fail-under=85
  uv run vineyard doctor
  uv run vineyard --help        # lists every contract §8 command
  ```
  Also required:
  - the import-hygiene test passes (`"torch" not in sys.modules` after importing registry, stages and nn.probs);
  - `test_tiling` known corners pass;
  - START maps to r018_c010 at (28.0, 2002.0).

  Then commit and push `phase0`. Phase 1 packages merge main again.

### Phase 1: parallel build of the pre-annotation path (00:00–03:15)

| Package | Exclusive files (under `vineyard/`, tests in `tests/<area>/`) | Consumes | Milestone |
|---|---|---|---|
| **P1-PIPE** (fnd WP-C) | `pipeline/{runner,cache,atomic,parallel,timings,tile_index}.py`, `ingest/*`, `stages/{ingest,tile_prep}.py` | P0 | Ingest of 311 tiles in under 60 s by **00:45**; tile_prep on 311 tiles by **01:30** |
| **P1-GEO** (P0-GEO agent continues) | `geo/{ops,polygonize}.py` | P0 | Done **01:15**, then the agent becomes P1-WST |
| **P1-CVIO** (fnd WP-D, part 1) | `cvat/{writer,reader,rle,normalize,to_annset,testzip,cli}.py`, `stages/import_reference.py` | P0 | **M1 01:00**: byte-identical round-trip plus `make-test-zip` |
| **P1-CVEX** (fnd WP-D, part 2) | `cvat/{to_cvat,report,validator_doc,validator_zip,packer,selfcheck,export}.py`, `stages/export_cvat.py` | `cvat/model`, `geo/ops` (01:15), reader (01:00) | **M2 02:30**: empty-AnnSet export of 311 tiles is validated |
| **P1-EVAL** (fnd WP-E) | `eval/{matching,metrics,report}.py`, `stages/evaluate.py` | reader (01:00) | Reference vs reference scores 1.0 by **02:30** |
| **P1-ROWS** (perc WP1) | `perception/{profile,linefit,row_features,rows_detect}.py`, `stages/rows_detect.py` | vegmask | Per-tile F1 of at least 0.95 on the examples by **03:00** |
| **P1-BLK** (perc WP2) | `perception/{rows_link,blocks_graph,blocks,overrides}.py`, `stages/{rows_link,blocks}.py`, `configs/overrides.yaml` (empty template) | row_candidates schema | One block per example, deterministic, by **03:00** |
| **P1-CAN** (perc WP3) | `perception/{corridor,canopy,interrow,trees}.py`, `stages/{canopy,interrow}.py` | rows schema | `corridor.py` real within 45 min (P1-ATT and P1-WST import it); canopy ≥ 0.84 with reference axes by **03:00** |
| **P1-ATT** (perc WP4) | `perception/{attrs,topology}.py`, `annset/assemble.py`, `qa/{render,review,overview}.py`, `stages/{row_attrs,assemble,qa_previews}.py` | corridor, `layers/waste.parquet` (file only) | At least 50/51 attributes; 0 errors on the reference AnnSet by **03:15** |
| **P1-WST** (ML W1 + rule path of W3a; starts **01:15**) | `perception/waste/{types,candidates,lattice,filters,nms,crops,decide,assign,confirm,review,verify}.py`, `stages/waste.py`, `configs/waste_confirmed.csv` (header only) | rows, blocks, `corridor.clip_rows_to_tile` | L3 rule-only stage: 0 auto on the examples; review HTML; by **03:30** |

- **Peak concurrency** is 9 agents between 00:00 and 01:15. The ML (N1–N3, W2) and post packages do **not** start in Phase 1.
- **Gate G1 (03:15–03:30), run on main after all merges:**
  ```
  VINEYARD_REQUIRE_TILES=1 uv run pytest -m "not slow" -q
  VINEYARD_REQUIRE_TILES=1 uv run pytest -m slow tests/perception tests/cvat -q
  uv run pytest --cov=vineyard --cov-report=term-missing   # >= 80% on logic modules
  uv run ruff check vineyard tests
  grep -rn CONFIG-REQUEST vineyard                        # must be empty
  ```
  Also required: M1 and M2 artifacts exist. Then commit and push `phase1`.

### Phase 2: integration, then RC0 and RC1 (03:15–07:00)

1. **I1, integration on the two examples (03:15–04:15).**
   ```
   uv run vineyard run --run-id i1-ex --tiles 'siret3_r006_c004' --tiles 'siret3_r021_c012' --until assemble
   uv run vineyard eval-examples --annset i1-ex --gates
   ```
   Use an explicit run id: `LATEST_*` is not updated when `--tiles` is used. Package owners stay available to fix their own files.

   **Canopy convention sweep (R5):** re-run from canopy to assemble with `--set canopy.vector_offset_px` / `vector_outset_px` set to {0/0, 0.5/0.5}. Keep the winner in `default.yaml` (orchestrator commit). Then run `eval-examples --write-baseline` to create the regression baseline.
2. **Gate G2.**
   - canopy score ≥ 0.80 (mean, vector metric);
   - row F1 ≥ 0.95 on each tile;
   - interrow IoU ≥ 0.95;
   - attributes ≥ 0.90;
   - one block per example.

   **Waiver rule:** a gate may be waived only if it misses by ≤ 0.01 for a documented reason (for example V02-R15). The waiver is written to `run.json` and the commit message. Missing the upload is always worse than a 0.01 shortfall.
3. **F1 full run and RC0 (04:15–04:45).**
   ```
   /usr/bin/time -l uv run vineyard run --until qa_previews --workers 8
   uv run vineyard export-cvat --annset LATEST_MODEL
   ```
   `export-cvat` runs validation, self-check and the atomic publish. Record `metrics/timings.json` and peak RSS. Expected ≤ 20 min in total.
   - Run the eval gates **on the full-run AnnSet** restricted to the 2 example tiles (neighbour context matters for linking and IDs).
   - Commit and push the `rc0` tag. **From here a valid, non-empty upload set exists.**
4. **Hardening (04:45–06:15).**
   - Triage `qa/overview.jpg` and the priority-1 part of `review_queue.csv`.
   - Calibrate the SNR threshold from the distribution across all 311 tiles.
   - Fix false vineyards on non-vineyard tiles.
   - Arch §6 lists robustness features that are behind flags (default off). The owner of each file implements its own item:
     - curved-row tracking: P1-ROWS;
     - orchard block rejection: P1-BLK;
     - tree filter: P1-CAN;
     - texture and Otsu fallbacks in `vegmask.py`: ownership moves temporarily to the P1-ROWS agent.
   - Each flag is merged only if it is a no-op on the examples (±0.005 with rows unchanged) and has passed its synthetic test. **Code merges for the pre-annotation path close at 06:15.**
5. **RC1 (06:15–07:00).**
   - Full re-run (cache hits except for the changed stages), then `export-cvat`, then the independent checks in §5.
   - `eval-examples --gates --baseline` with `max_drop` 0.01.
   - Commit and push the `rc1` tag. **RC1 ZIPs are then treated as read-only.**

### Phase 3: human QA, RC2, upload, Publish (07:00–12:00)

- **Human QA, 07:00–10:00:**
  - `configs/overrides.yaml` (force-empty tiles, delete/extend/add rows, excluded areas);
  - `configs/waste_confirmed.csv` from `waste_review.html`.
- **RC2, 10:00–10:30, config and data changes only (no code):**
  ```
  uv run vineyard run --until qa_previews
  uv run vineyard export-cvat --annset LATEST_MODEL
  ```
  Then rerun the §5 checks and the eval regression. Commit and push the `upload-final` tag.
  - If RC2 fails any check by 10:40, **upload RC1** instead.
- **Off-path work on branches, 07:00–10:30.** These merge to main only after 10:30:
  - NN N1/N2: MPS training runs 07:15–09:45 and must end before the RC2 run;
  - post WP2 (derive), WP4 (passable) and WP5 (route, on synthetic graphs), each an independent branch.
- **Upload and Publish, 10:30–12:00** (human, §6).

### Phase 4: after Publish (Sat 12:00 – Sun 02:00)

- **Merge the off-path branches.** Then build:
  - post: `import_marcaj` stage plus `import_checks`, WP3 (targets, measure, publish), WP6 (web bundle into `../Web/data`);
  - ML: N3 ablation (a deliverable only, never promoted into the upload);
  - waste: W2 probe and W3b SAM adapter, used only as a **ranked candidate list for manual waste boxes in Marcaj**, and only if ready by about 18:00;
  - Docker and README.
- **Gates:**
  - `uv run vineyard post --annset LATEST_MODEL` runs end to end by about 18:00, with the route validator passing (`outside_frac ≤ 0.005`, closure ≤ 0.01 m, 100% of must-visit targets covered);
  - 20:00: intermediate Marcaj export → `vineyard from-marcaj` → `review_queue.csv` goes to the annotators by 21:00;
  - Docker is built and pushed once, around 00:00, then checked with `docker run --network none`.

### Phase 5: Sunday final (08:00–14:00)

- 11:00: all jobs are Submitted; final Marcaj export.
- Run the full post chain on the Marcaj export with the `route.solver.final` override (per X15):
  ```
  uv run vineyard from-marcaj <files> && uv run vineyard post --annset LATEST_MARCAJ --set route.solver.final=true
  ```
  Then `publish` with `--set publish.require_source=marcaj` (refuses if the source isn't marcaj).
- Commit and push the root `route.geojson` and `measurements.csv` by 13:00. README timings come from `vineyard bench`. Code freeze at 14:00.

---

## 4. Critical path to validated ZIPs by Saturday 07:00

| Step | Start | End | Duration | Slack |
|---|---|---|---|---|
| P0.1 env, lock, skeleton | 23:00 | 23:25 | 0:25 | 0 |
| P0.2 interface freeze (G0a) | 23:25 | 23:55 | 0:30 | 0 |
| P0 contract implementation (G0b), overlaps Phase 1 | 00:00 | 00:45 | 0:45 | 0:30 |
| Ingest and tile_prep on 311 tiles | 00:00 | 01:30 | 1:30 | 1:30 (needed at I1) |
| Perception packages P1-ROWS, BLK, CAN, ATT (**longest chain**) | 00:00 | 03:15 | 3:15 | 0 |
| CVAT export M2 (empty set, validated) | 00:00 | 02:30 | 2:30 | 0:45 |
| I1 integration on examples plus G2 | 03:15 | 04:15 | 1:00 | 0 |
| F1 full run plus RC0 export | 04:15 | 04:45 | 0:30 | 0 |
| Hardening and triage (**this is the buffer**) | 04:45 | 06:15 | 1:30 | 1:30 |
| RC1 | 06:15 | 07:00 | 0:45 | 0 |

**Honest assessment.** A validated RC0 by about 05:00 is likely. RC1 at 07:00 holds only if the perception packages land by about 03:30, and I1 is where the schedule will slip. Four decision points bound the damage:

- **DP1, 01:30.** If `tile_prep` has not run on all 311 tiles, the orchestrator takes over P1-PIPE and runs it inline.
- **DP2, 04:15.** If G2 fails with no fix in sight, ship with the best config and a waiver. Repair in this order: rows F1, canopy, IDs/blocks, interrow, attributes.
- **DP3, 05:30.** If the full run does not export cleanly, drop to the degradation ladder below.
- **DP4, 10:40.** If RC2 fails, upload RC1.

**Degradation ladder**

| Level | What changes | Effect |
|---|---|---|
| L0 | Nothing | Full path |
| L1 | P1 flags off; `rows_link` simple endpoint linking | Weaker cross-tile linking |
| L2 | Waste layer empty | 0 waste false positives guaranteed |
| L3 | `force_empty` on failing or dubious tiles | Those tiles go up blank for manual work |
| L4 | Upload the M2 empty-annotation ZIPs | A valid Publish is still guaranteed |

**Must NOT be deferred.** All of this changes what gets uploaded, so the code is due by 06:15:
- canopy, including the offset/outset convention;
- rows and linking;
- blocks and ID ordering (R001 northernmost, V01 sort);
- interrows and `interrow_cover`;
- `row_structure`;
- waste policy for the upload. Set `waste.decide` so no box is auto-accepted before Publish; only rows in `waste_confirmed.csv` are exported;
- `tile_status` and `empty_tiles.csv`;
- `id_registry.json` (R900+);
- QA previews and `review_queue`;
- validators, packer and self-check;
- parsing the real Marcaj test export fixture.

**Deferred until after Publish:**
- NN training, ablation and any promotion (the NN is a deliverable, but `nn.enabled: false` for RC1/RC2);
- the SAM 3 adapter and the CLIP linear probe;
- the UAVVaste fetch;
- the post chain (derive, passable, targets, route, measure, publish);
- the web bundle and ortho;
- Docker, README and bench.

---

## 5. Verification per phase

| Phase | Checks |
|---|---|
| P0 | G0-env commands; G0b pytest with ≥ 85% coverage on contracts, geo and config; `vineyard doctor`; import hygiene; tiling oracles (r006_c004 x0 = 629196.8, y0 = 5220915.2; START at (28.0, 2002.0)); `tile_grid.txt` equals the 311 ZIP names |
| P1 (per package) | Each design's TDD list. For example: byte-identical example round-trip; packer gives n = 5 with every ZIP ≤ 85,000,000 B; reference-vs-reference eval = 1.0; a hard-rejected candidate matches no reference row (51/51); interrow IoU ≥ 0.98 with reference axes; the r006 partial band does not split the block; waste auto = 0 on the examples. Coverage ≥ 80% on logic modules. |
| P1 milestones | `uv run vineyard cvat roundtrip-examples`; `uv run vineyard cvat make-test-zip` + `uv run vineyard cvat validate <zip>`; `uv run vineyard export-cvat --annset EMPTY` (P1-CVEX provides this alias) → 5 validated ZIPs; `uv run vineyard ingest` (under 60 s, 311 rows) and `uv run vineyard run --until tile_prep` (0 failures; the rerun reports all cached) |
| G2 / RC | `eval-examples --gates` (canopy ≥ 0.80, row F1 ≥ 0.95, interrow IoU ≥ 0.95, attributes ≥ 0.90) on the **full 311-tile run**; `--baseline` drop ≤ 0.01; `timings.json` with a total ≤ 20 min; 0 failed tiles (or failed tiles listed as empty with reason `failed`) |
| RC ZIPs (independent of our validator) | `uv run vineyard cvat validate work/runs/<id>/exports/marcaj_upload/*.zip` (bijection, sha256, whitelist, meta, image ids, the set of 311 tiles); `stat -f %z *.zip` each ≤ 85,000,000; `unzip -Z1` shows only `annotations.xml` and `images/siret3_rNNN_cNNN.tif`; the union has 311 unique names; self-check union IoU ≥ 0.98; the manifest's non-empty counts look plausible; spot-check 5 random previews by eye |
| Marcaj round-trip | The test export captured in H3 parses with `read_cvat_files` + `document_to_annset` with no structural errors; geometry IoU ≥ 0.999 against what was uploaded |
| P4 / P5 | Post TDD (targets give 8 GAP / 0 END on the reference; measurements give 1941.55 m / 51 rows; route validator synthetic cases); `vineyard post` end to end; publish refuses on bad cases; web manifest loads in `src/Web`; `docker run --network none` |

---

## 6. What the humans must do, and when

| When | Who | Action |
|---|---|---|
| **Now, before 23:00** | lead | Send the arch §8 Slack questions; Slack closes at 23:00. Q4/Q6 are now moot (5 parts ≤ 85,000,000 B). Q1, Q3, Q5, Q7 and Q8 still matter. |
| Now | lead | Confirm the Marcaj accounts email; log in; check the project labels match the example `<meta>`. |
| Now | ML owner | `hf auth login` (read token). **Request access to `facebook/sam3`**, the transformers-loadable repo; `sam3.1` has no transformers weights. Decide whether to stop or delete the 3.5 GB `sam3.1` download (34 GiB free disk). |
| Now | lead | Check `gh auth status` / git push credentials, so pushes after each phase don't stall. |
| ~01:15 (M1) | uploader | **Format test.** Upload the test ZIP (2 example tiles, 1 waste box, 1 empty tile). Screenshot the import report (frames, objects, skipped/dropped). In the editor check that polygons sit on the right tiles, the box is present, and whether the empty tile shows "No objects". **Export the task as CVAT for images 1.1, annotations only**, and save it to `src/AI/tests/fixtures/marcaj/marcaj_export_sample.zip`. Then **Remove all**. |
| ~02:45 (M2) | uploader | **Size and time test.** Upload one real ~80–85 MB part from M2. Check it is accepted and time the upload plus import. Remove all. This number sizes the 10:30 upload window. |
| 07:00–10:00 | QA team (E) | Review previews in `review_queue` priority order. Fill `overrides.yaml` (force-empty non-vineyard tiles first, then rows and cross-edge `row_id`s), `waste_confirmed.csv` and `empty_tiles.csv`. |
| 10:30–11:45 | B, confirmed by E | Check nothing is left from the tests. Upload the 5 parts one after another, screenshotting each import report. Check the Data card shows **311 files**. Spot-check 2 tiles. |
| **≤ 12:00** | B, confirmed by E | **Publish.** It cannot be undone. |
| Sat 12:00 – Sun 11:00 | everyone | Correct the 63 jobs. Tick "No objects" on `empty_tiles.csv` tiles. Submit each job when done. Add waste boxes manually from the ranked list. |
| Sat ~20:00 | lead | Intermediate Marcaj export, then run `from-marcaj` and send the review list to the annotators. |
| Sun 11:00 | lead | Final export after 100% of jobs are Submitted. Commit and push the deliverables by 13:00; freeze at 14:00. |

---

## 7. Top 10 execution risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | The four designs disagree on stage protocol, config location, duplicated modules and ID codes, so parallel agents build incompatible pieces | Freeze the §1 X1–X17 decisions in P0. Orchestrator-owned shared files. Pre-filled registry. Additive-only contracts after G0 via `CONFIG-REQUEST` and change requests. Ownership check at every merge. |
| R2 | uv lock or environment fails (numpy 2.5 vs OpenCV/skimage/numba via sknw, torch 2.14 on macOS, conda PROJ variables, broken brew 3.14 pyexpat) | 25-minute timebox. Core lock first, extras separately. Fallbacks: `numpy<2.5`, torch 2.13.x, drop sknw. `unset PROJ_DATA PROJ_LIB GDAL_DATA` in the Makefile and `_env`. `doctor` is the G0 gate. |
| R3 | False-green tests: worktrees lack the tile ZIPs so `needs_tiles` skips; duplicate test basenames break collection | Absolute `VINEYARD_DATA_ROOT` / `VINEYARD_WORK_DIR`. `VINEYARD_REQUIRE_TILES=1` at every gate. `--import-mode=importlib --strict-markers`. |
| R4 | Perception integration slips past 04:15 (four 3-hour packages meet for the first time at I1) | Schemas frozen at G0a so packages test against reference-derived fixtures. Owners stay on call through I1. Decision points DP1–DP3. Degradation ladder down to M2 empty ZIPs. |
| R5 | Canopy vertex convention: reference is raw integer contours, contract is +0.5 with 0.5 px outset. Also vector vs raster metric shift. Either can move canopy IoU by several points (canopy is 25% of the score and is effectively final after upload, since humans cannot redraw plants) | Both parameters are in config. Sweep at G2 and pick by eval. Gates re-baselined on the vector metric. Decided before RC0, never after 06:15. |
| R6 | Export blocked late by qa errors, invalid-after-rounding geometry, or oversize parts | Full export exercised at RC0 (04:45), 2+ hours before RC1. Per-object drop plus qa warning. A failed tile becomes an empty `<image>` with reason. Automatic n+1 repack. `--allow-qa-errors` logged. Immutable RC1 fallback. |
| R7 | Marcaj behaviour is unknown (`source` attribute, name matching, box import, empty frames, size limit, import time) and there is no Slack support until 09:00 | Empirical tests H3/H4 tonight. The Marcaj export fixture is parsed before Publish. Upload starts by 10:30 with measured timing. |
| R8 | Resource contention: 18 GB RAM, 11 cores, 34 GiB disk; 9 agents plus 8-worker pools plus MPS training plus SAM | Only the orchestrator runs 311-tile jobs. Agents use examples and `--workers ≤ 2`. No torch before RC1. NN training 07:15–09:45 only. SAM after Publish only. Shared `work/`. APFS-clone venvs. Decide on the SAM 3.1 download. |
| R9 | False vineyards on orchard, garden or grass tiles (canopy penalty, wrong counts) and inconsistent `row_id`s across tile edges | Soft SNR plus rescue linking. `overview.jpg` coloured by block. Priority-1 queue: SNR, bands, orchard, overrides. `force_empty_tiles` and row overrides applied in RC2. `empty_tiles.csv` for "No objects". |
| R10 | Route (25% of the score) gets too little time because everything focuses on the upload | Post WP2/WP4/WP5 start at 07:00 on synthetic data and RC1 (branches merged after 10:30). `vineyard post` end to end by 18:00 on AnnSet(model). Rehearsal on the intermediate Marcaj export at 20:00. Two-level route validator; publish refuses on failure. |

### Critical Files for Implementation
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/contracts/schema_defs.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/config/model.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/pipeline/registry.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/cvat/export.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/annset/assemble.py