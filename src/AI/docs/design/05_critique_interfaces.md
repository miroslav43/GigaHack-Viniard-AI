# Critic report: cross-check of the foundation, perception, ML and post designs

**Overall:** the four designs cannot be integrated as they stand. I found 4 integration blockers where two designs build the same files with incompatible types. The biggest score risk is the canopy pixel convention, which I measured on the reference. Everything below comes from reading all four designs, `00_contracte.md` §1–§13, ARHITECTURA §3–§11, the brief, the prototype scripts, and measurements on `05_examples`.

## 0. New facts measured during this review

These change decisions, so they come first.

| # | Fact (measured on `05_examples/.../annotations.xml`) | Consequence |
|---|---|---|
| F1 | All 650 reference canopies have **integer vertices only** (0 non-integer coordinates), with min 0.0 and max **2047.0**. | Canopies are raw `findContours` indices: no +0.5, no outset, and no vector clipping to the corridor (a vector clip would produce fractional vertices). Contract §1.2 is wrong on this point. |
| F2 | Canopy area: vector (shoelace) = **237.12 / 299.06 m²**. The pixel set, i.e. `fillPoly(int(p))` with the boundary included, = **256.95 / 319.36 m²**. The ratio is 0.923 / 0.936. | Contract §1.4 (+0.5 px, then a 0.5 px outset) rebuilds the pixel set. Against the reference this gives a class-IoU ceiling of about 0.92 and about +8% canopy area. |
| F3 | Exact canopy ∩ interrow overlap in the reference, computed by polygon clipping: **0.0366 m² (r021) and 0.0333 m² (r006) per tile**. Worst single overlap is 0.011 m². | That is above the contract §2.7 error limit of 0.01 m²/tile and below the arch §4.13 validator limit of 0.05 m²/tile. |
| F4 | Reference interrows have **4 or 5 vertices**. The smallest are 7.46 / 15.72 m². | They are clean quads. No canopy subtraction and no notches appear in the reference. |
| F5 | Smallest reference canopies measure 0.189–0.19 m² as vectors. | The reference's min-area filter is on the **contour (vector) area**, not on pixel count. |
| F6 | Shortest reference row pieces: 1.10 m (r021) and 4.99 m (r006). | There is no evidence of stubs under 0.5 m in the examples. |
| F7 | `models/sam3.1/README.md` line 39 says "there is no Hugging Face Transformers integration". | This confirms the ML design's pre-check that skips any model id without transformers weights. |

---

## 1. Interface mismatches between designs

### I1 [BLOCKER] Three incompatible stage protocols
- **Foundation:** `pipeline/registry.py` has `StageSpec(name, version, scope, cfg_keys, requires, run)`. Each module exports `STAGE: StageSpec`. Tile stages go through `run_tile_stage(ctx, spec, make_task, tile_fn(TileTask), input_keys)`.
- **Perception:** assumes a `pipeline/dag.py` with module constants `STAGE_NAME/STAGE_VERSION/SCOPE/CFG_KEYS`, `item_key(ctx, tile_id)`, and `run_tile(ctx, tile_id) -> TileOutcome` or `run(ctx) -> StageOutcome`.
- **ML:** `STAGE_NAME/STAGE_VERSION/CFG_KEYS` and `run(ctx, tile_ids) -> StageReport`.
- **Post:** says it uses `STAGE: Stage` (the foundation API) with fields `name, STAGE_VERSION, cfg_keys, inputs`.
- **Fix:** the foundation's `StageSpec` + `StageResult` + `run_tile_stage` is the canonical protocol.
  - Every stage module exports `STAGE = StageSpec(...)`.
  - The tile function is a top-level `fn(TileTask) -> Mapping`.
  - `dag.py`, `TileOutcome`, `StageOutcome` and `StageReport` are removed from the other designs.
  - Freeze it at 23:15.

### I2 [BLOCKER] The CVAT reader, the AnnSet types and the CVAT→AnnSet converter are each built twice, with incompatible types

| Item | Foundation | Post |
|---|---|---|
| Reader | `cvat/{reader,rle,normalize}.py` → `CvatDocument/CvatImage/CvatShape`, `parse_xml`, `read_cvat_files` | same paths → `RawDoc/RawImage/RawShape`, `read_cvat`, `read_cvat_many` |
| `normalize_enum` | `(attr, raw, *, synonyms, accept_synonyms) -> (str, Issue\|None)` | `(raw, allowed, synonyms, accept_synonyms) -> Normalized` |
| AnnSet | `annset/model.py`: `AnnSet(meta: AnnSetMeta, …)`; `AnnSetMeta` has no `tile_ids` | `AnnSet(…, meta: Mapping)`; relies on `meta["tile_ids"]` to know coverage |
| Converter | `cvat/to_annset.document_to_annset(...) -> (AnnSet, qa gdf)`; canopy-row tie goes to the lowest `piece_id` | `annset/from_cvat.annset_from_cvat(...) -> ImportResult` + `canopy_rows.py`; tie goes to the smaller perpendicular distance |
| `pipeline/stages/import_reference.py` | 100 lines | 70 lines |

- **Fix:**
  - The foundation owns `cvat/*`, `annset/model.py`, `annset/io.py`, `document_to_annset` and the `import_reference` stage. They are on the 07:00 critical path through the export self-check.
  - Add `tile_ids: tuple[str, ...]` (all imported images, including empty ones) to `AnnSetMeta`.
  - Post keeps only `import_checks.py` and wraps the foundation calls in its `import_marcaj` stage.
  - Use one tie-break rule: perpendicular distance first, then `piece_id`.

### I3 [BLOCKER] Config is defined in 4 places with conflicting keys and types
With `extra=forbid`, loading `default.yaml` will fail.
- **Section models are defined in several places:**
  - foundation `config/sections_{core,io,perception,ml,post}.py`;
  - perception `perception/settings.py`, which also redefines `veg` (owned by the foundation), adds `qa` and adds `paths.overrides`;
  - ML `nn/config.py` and `perception/waste/config.py`;
  - post `config/post_{import,route,outputs}.py`, which also redefines `import`.
- **The foundation's `AppConfig` has no `derive`, `publish` or `qa` fields.**
- **Type and name clashes:**
  - `veg.texture_fallback` is the string `v_std_0.25m` in the foundation but an object `{enabled, window_m, top_frac, trigger_min_veg_frac}` in perception.
  - The foundation seeds `rows.filter.spacing_range_m`, `rows.link.perp_tol_m` and `rows.link.gap_max_tiles`, and its test asserts `rows.filter.spacing_range_m == [1.8, 3.8]`. Perception uses `rows.detect.spacing_min_m/spacing_max_m`, `rows.link.link_offset_max_m`, `link_gap_max_tiles` and `link_angle_max_deg`.
  - Shadow threshold: foundation `interrow.shadow_v_max` vs perception `veg.shadow_v_max`.
  - Canopy offset: foundation `canopy.vector_offset_px` vs perception `canopy.vector_outset_px`.
  - Waste: the foundation seeds flat `waste.auto_*` keys; ML uses nested `waste.decide.*`.
  - Route: the foundation seeds `route.tsp_time_s` and `route.max_outside_share`; post uses `route.solver.time_limit_s`, `route.max_outside_frac_publish` and `route.domain.*`.
  - Workers: the foundation has `runtime.workers`; perception reads `runtime.n_workers`.
- **Fix:**
  - Section models live only under `vineyard/config/`, one module per owner: core/io = foundation; perception = perception; nn and waste = ML; import_extras, derive, targets, route, measure, publish and web = post.
  - Delete `perception/settings.py`, `nn/config.py`, `perception/waste/config.py` and `config/post_*.py`, or turn them into re-exports.
  - The root `AppConfig` fields, in order: `contract_version, project, paths, grid, runtime, nodata, veg, nn, rows, orchard, blocks, canopy, interrow, row_structure, waste, targets, derive, route, export, import, measure, publish, web, eval, qa, logging`.
  - For each section, the owner's key names win. Delete the foundation's "seeded" versions.
  - Only one integrator edits `default.yaml`; the others send YAML blocks.

### I4 [BLOCKER] `perception/vegmask.py` is owned twice, and the fusion/upsample code exists twice
- **vegmask.py:** the foundation writes a 70-line version (`neg_a_star`, `veg_mask_lab_a`). Perception writes a 230-line version (`compute_tile_masks`, `neg_a`, `veg_mask`, `vis_codes`, `fuse_masks`, `upsample_prob`).
- **Fusion:**
  - Perception: `fuse_masks(veg, prob, variant, thr)` with `FusionVariant{NONE,NN,AND,OR,NN_NO_CORRIDOR}` and key `canopy.nn_fusion: none`.
  - ML: `nn/fusion.fuse_masks(veg, prob, corridor, variant, threshold) -> FusedMask` with `FusionVariant{A..E}` and key `nn.fusion: A`.
- **Upsample:** `upsample_prob` is in both perception `vegmask.py` and ML `nn/probs.py`.
- **NN in rows:** perception `rows.detect.use_nn_weight` vs ML `nn.use_in_rows/use_in_interrow`.
- **Fix:**
  - Perception owns `vegmask.py`; the foundation's `tile_prep` calls `compute_tile_masks`.
  - ML owns `nn/fusion.py` (A–E) and `nn/probs.py` (upsample and loading).
  - Perception imports from ML and drops its own copies.
  - Keep one set of keys: `nn.fusion` and `nn.use_in_rows/use_in_interrow`.

### I5 [MAJOR] `QaIssue` is defined three times
- Foundation: `cvat/report.Issue(severity, code, message, zip_name, tile_id, object_ref)`, with no point.
- Perception: `QaIssue` in `perception/types.py`, "re-exported by contracts".
- Post: `annset/qa.QaIssue(severity, code, tile_id, object_id, message, x, y)` plus `issues_to_gdf`.
- **Fix:** create `contracts/qa.py`, owned by the foundation, holding the post definition: `QaIssue`, `issues_to_gdf`, and one deterministic `Q00001` numbering (sort by severity, code, tile_id, object_id). Export-validation `Issue` values convert into it. Also add a single `QA_CODES` set covering all proposed codes from perception, post and the foundation.

### I6 [MAJOR] Three implementations of the contract's ordering rules
- Foundation `contracts/ordering.py`, perception `perception/ordering.py`, and post `annset/lines.py` + `row_order.py` all implement the canonical normal, R001 as maximum n·c, and the block key `(−round(cy), round(cx))`.
- **Fix:** keep only `contracts/ordering.py`. The others call it.

### I7 [MAJOR] Occupancy and gap logic is written twice, and each design claims the other reuses it
- Perception `perception/attrs.py`: 0.05 m bins, a 0.5 m window with occupancy ≥ 0.20, then a 0.4 m 1D closing.
- Post `annset/gaps.py`: clipped canopy vertices projected onto the axis as intervals, with no window and no closing.
- The two give different `max_gap_m` on the same row, so the model's `row_structure` and the targets/import `max_gap_m` diverge. The difference matters exactly at the 5.0 m edge (R15 = 5.009, R06 = 5.054).
- **Fix:** keep one `vineyard/annset/gaps.py`, owned by post, with the convention from C3 below. Perception's `row_attrs` and `targets` both call it.

### I8 [MAJOR] Nobody owns the waste handoff into the AnnSet
- Perception's `assemble` reads `layers/waste.parquet` (AnnSet schema).
- ML writes `layers/waste_candidates.parquet` and says "`assemble` calls `merge_confirmed`". Perception's `assemble_annset(...)` does not take waste confirmations.
- **Fix:**
  - The finalize step of the ML waste stage calls `merge_confirmed` and writes `layers/waste.parquet` (AnnSet `waste` schema, exported only, W-ids, a/b pairs).
  - `assemble` only reads it.
  - Document the re-export command after confirmations at ~10:00: `run --from waste --until assemble` → `export-cvat`.

### I9 [MAJOR] Post stages write into the model run directory
When post runs on the model AnnSet (`vineyard post --annset <model run>`), it overwrites files that other stages own:
- `derive` writes `layers/rows.parquet`, `blocks.parquet` and `interrows.parquet`. Perception already wrote these, and the WKB of the `rows` feeds the canopy/interrow cache keys (C§3.2).
- The post `qa` stage overwrites `qa/qa_issues.parquet` and `qa/review_queue.csv` written by `assemble`.
- **Fix:** every post stage writes to a new run directory, `runs/<ts>-post-<h6>/`, whose `run.json` records `annset_ref`. `LATEST_*` links are never updated by post runs.

### I10 [MAJOR] Stage lists and order in the foundation registry
- `POST_STAGES=(derive, targets, passable, route, …)` runs `targets` before `passable`, but post's targets read `passable_domain`, and `measure` needs `targets` for `n_targets`.
- `PRE_STAGES` lacks `interrow_bands` (perception's two-step interrow), `qa_previews` and `export_cvat`.
- **Fix:**
  - `PRE_STAGES = (ingest, tile_prep, nn_infer, rows_detect, rows_link, blocks, canopy, interrow_bands, interrow, row_attrs, waste, assemble, qa_previews, export_cvat)`.
  - `POST_STAGES = (import_marcaj, derive, passable, targets, route, measure, web_bundle, publish)`.

### I11 [MAJOR] CLI wiring
- The foundation loads sub-apps from `vineyard.{nn,waste,web}.cli`. But ML's waste CLI is at `vineyard/perception/waste/cli.py`, and post's `web build` lives in `vineyard/cli_post.py`. Both would fall through to the "neimplementat" placeholder (exit 3).
- `cli.py` (foundation) and `cli_post.py` (post) both define `import-marcaj`, `derive`, `route`, `measure`, `publish`, `post` and `qa`, which is a typer name collision.
- **Fix:** mount `vineyard.perception.waste.cli:app` as `waste` and `vineyard.cli_post:app` for the post commands. Remove the duplicates from the foundation's `cli.py`.

### I12 [MINOR] `tile_prep` outputs vs what perception reads
- Perception reads `cache/{valid,veg,vis,stats}/<t>`. The foundation writes `cache/tile_stats/<t>.json` and no `vis` at all.
- `write_mask_png` is 1-bit, but vis codes {0,1,2,3} need uint8.
- **Fix:** consumers use only the accessor functions (`load_*`). `tile_prep` also writes `cache/vis/<t>.png` as uint8, using perception's `vis_codes`.

### I13 [MINOR] Functions ML expects under names nobody provides
- ML expects `corridor_mask`, `label_components`, `mask_to_canopies(mask, rows_in_tile, tile, cfg)`, `eval.canopy_score(pred_gdf, ref_gdf)` and `geo.ops.clip_to_tile`.
- Perception offers `corridor_labels`, `vine_mask` and `extract_canopies(veg, …, prob, variant)`, which fuses internally. The foundation offers `canopy_metrics(...) -> CanopyScore` and `clip_polygonal/clip_box`.
- **Fix:** `extract_canopies` takes an already-fused mask. Ablation variant E needs a `corridor=None` path. ML adapts to the foundation and perception names.

### I14 [MINOR] Keys duplicated for the same number
- The 10 m waste rule: `waste.block_assign_max_m` and `export.cvat.waste_empty_vid_min_dist_m`.
- The 0.30 m corridor: `canopy.corridor_half_m`, `derive.gap_corridor_half_m` and `targets.corridor_half_m`.
- Output paths: `paths.repo_root/web_data_dir` and `publish.root_dir/web.out_dir`.
- Notch width: `export.cvat.notch_width_px` is in px, but perception applies `notch_holes` in UTM metres.
- **Fix:** one key each. Consumers read that key and convert units explicitly.

### I15 [MINOR] Several writers for the same output file
- `empty_tiles.csv` is written by perception (`qa/`) and by the foundation (`exports/marcaj_upload/`, with `zip_name` and `image_id`).
- `tile_status` is written by `assemble`, and perception expects "export fills `upload_zip`/`image_id`".
- **Fix:** only the export's `empty_tiles.csv` exists. Export never rewrites `tile_status`; `upload_manifest.csv` is joined to it when needed.

### I16 [MINOR] Target kind codes disagree
- The foundation proposes MIS/SPA; post proposes MSP/SPR.
- **Fix:** add `missing_plant`→MSP and `sparse`→SPR to `TargetKind` and `TARGET_RE`, in `CONTRACT_VERSION` 1.1.

---

## 2. Contract violations and contract bugs

### C1 [MAJOR] ID regexes will reject valid model output
- Perception can emit more than 99 row candidates per tile: `find_peaks` with no height criterion × 3 orientations, up to about 34 × 3. That breaks the contract regex `K\d{2}`, so `validate_layer` raises and the tile fails.
- Interrow bands that split in a tile (a full-width forbidden hole or a nodata cut, then `explode`) produce **duplicate `piece_id`s** → `SchemaError` in `assemble` at about 06:45.
- **Fix:**
  - Change the row-candidate regex to `K\d{2,3}`.
  - Perception must use `format_interrow_piece_id(..., dup=k)` (`#2`, …) after `explode`.
  - Extend the piece regexes to allow `#\d+`. The foundation test already accepts this for row pieces.

### C2 [MAJOR] Canopy ∩ interrow overlap limits are inconsistent
- Contract §2.7 uses 0.01 m²/tile as an **error** in `assemble`/`qa`. Arch §4.13 and the foundation validator use 0.05 m²/tile. The reference itself measures 0.033–0.037 (F3).
- As a result:
  - perception's test "reference AnnSet → 0 errors" fails;
  - Marcaj and reference imports get error floods;
  - on dense model tiles, rounding to 0.1 px plus the writer's 0.5 px simplify can push past the limit and block the export.
- **Fix:**
  - One key, `export.cvat.max_canopy_interrow_overlap_m2: 0.05`, checked **after** conversion to px and rounding.
  - The writer does not simplify canopies a second time (they are already DP'd at 2.5 px).
  - `assemble` subtracts canopies from interrows only when the per-tile overlap exceeds the limit. Unconditional subtraction adds slivers and vertices; reference interrows have 4–5 vertices (F4).

### C3 [MAJOR] Pixel convention for canopies: contract §1.2/§1.4 contradicts the reference (F1, F2)
- **Fix:** add a contract erratum in §13 and apply it to canopies only:
  - vectorize with `findContours` → `approxPolyDP(2.5)` → integer vertices → `px_to_utm` with **no +0.5 and no outset**;
  - apply the min-area filter to the vector area (F5);
  - the inverse rasterization for this convention is `fillPoly(int(p))` with the boundary included, which gives back the pixel set;
  - rows and interrows keep the continuous convention (their reference ends are exactly on 0.0 and 2048.0).
- Details and score impact are in S1.

### C4 [MAJOR] Interrow ID rule `I<min(k)>` (contract §4.5.7, copied by post) collides
- A manual row `V01-R900` (arch §3.3) inserted between R004 and R005 gives interrows R004|R900 = I004 and R900|R005 = I005. The second collides with R005|R006 = I005. The duplicate PK in `interrows` raises `SchemaError`, and the Sunday chain stops.
- **Fix:** always number interrows by the geometric `row_index` order (n·c) inside the block. Keep `min(k)` only as a debug column.

### C5 [MINOR] Enum validation is strict even for Marcaj and reference data
- The foundation's `validate_layer` treats "Regular" as an error for any source. Contract §4.4 says keep the raw value and flag `bad_enum`.
- **Fix:** relax enum checks for `source ∈ {marcaj, reference}` (issue a qa warning instead of `SchemaError`). The risk is low because these attributes are select fields in CVAT.

### C6 [MINOR] Arch and contract disagree on a few smaller points
- **`measurements.csv`:** arch §4.12 has `interrow_area` as a sum, and its example leaves `vineyard_id` empty on the total row. The contract has a union and `ALL`. Post chose union/ALL, which violates "arch wins". These only differ when interrows overlap, and the CSV is shown to the jury, not scored. **Fix:** write the sum in the CSV, keep the union in the JSON, and follow the arch on the total row.
- **Row-stub minimum:** `export.min_row_piece_m: 0.5` conflicts with arch §4.3.4, which keeps corner stubs when the row continues in the neighbouring tile. **Fix:** allow a 0.1 m minimum for stubs whose row continues.

### C7 [MINOR] A 0.2 px notch width can collapse when rounded to 0.1 px
- Vertex rounding can make the two sides of the notch cross, which triggers `invalid_after_rounding`. `make_valid` then brings the hole back.
- **Fix:** use a notch width of at least 1 px (0.025 m).

### C8 [MINOR] `tile_valid` is stored per run
- The foundation writes it to `runs/<id>/layers/`, but it depends only on the tile bytes and the `nodata` config.
- **Fix:** make it a static file, `work/layers/tile_valid.parquet` (see S7).

---

## 3. Correctness bugs likely to cost score

### S1 [MAJOR, about 2 points] Canopy area inflation from the +0.5 / outset convention
- **Why:** perception applies `+0.5 → buffer(0.5 px, mitre)`, and its tests pin a 3×3 block to `[10,13]` with a union of 256.9 / 319.4 m². Those are the pixel-set numbers from F2, not the reference numbers (237.1 / 299.1).
- **Cost:**
  - class IoU is capped at about 0.92: 0.6 × 0.077 × 25% ≈ 1.2 points;
  - canopy area is about +8%, which scores 1 − 0.084/0.15 ≈ 0.44 on the 2% area criterion (about −1.1 points);
  - the prototype's 0.82 / 0.855 were apples-to-apples raster comparisons, and only the raw convention reproduces them in vector form.
- **Fix:**
  - Default to offset 0 and outset 0, with the min-area filter on the vector area.
  - Merge the three vectorizers (perception `canopy.py`, foundation `geo.raster.mask_to_polygons`, ML `mask_to_canopies`) into one `geo.raster.mask_to_polygons(offset, outset)`.
  - `eval-examples` sweeps three variants: (a) raw, (b) raw ∩ vector corridor, (c) contract +0.5/0.5. Pick the best on the vector metric; (a) is expected.
  - Update the perception tests to 237.1 / 299.1 m² (±5%).
  - ML's NN validation must rasterize the reference with `fillPoly(int(p))`, not `(uv−0.5)*16`.

### S2 [MAJOR] Gap measurement decides `row_structure` at the 5.0 m edge
- RAPORT line 37 reproduced **51/51** labels with the pixel method (1 px bins on the pixel set). Both designs accept 50/51 because the vector gap on R15 is 5.009.
- The pixel gap is roughly the vector gap minus about 1 px: R15 ≈ 4.98 (regular) and R06 ≈ 5.03 (disrupted).
- **Fix:** the single `gaps` module (I7) measures on canopy pixel sets, using the inverse rasterization from C3, and keeps the threshold at 5.0. The test requires 51/51. The occupancy window and closing stay only for model canopies, behind a flag, and are re-validated for 51/51.

### S3 [MAJOR] A missing row splits a block
- Perception's block graph links rows only at 1.8–4.0 m spacing (plus collinear links). A grubbed row in the middle of a block gives about 5 m spacing: no edge, **two `vineyard_id`s**, and no interrow across that gap. The result:
  - grouping and block counts are wrong;
  - the walking domain has a hole;
  - post's `missing_row` targets can never appear on the model.
- **Fix:** add "skip-one" parallel edges when the spacing is within [1.6, 2.3] × local s. Flag them `missing_row_suspect` (QA priority 1). Build the interrow band between the two neighbouring rows.

### S4 [MAJOR] Chain interpolation over a gap tile creates false rows and canopies
- `rows_link` allows collinear gaps of up to one tile (51.2 m). `clip_rows_to_tile` then emits a row piece, and canopies from grass inside the corridor, on a tile with **zero candidates**, such as a road or field between two collinear vineyards.
- That is a false positive on a non-vineyard tile, which the brief penalises at 0.5 × the share of the tile covered. The full-width band cut only helps when the band is at least 4 m wide and spans every row.
- **Fix:**
  - Split chains at unsupported holes of at least `blocks.collinear_gap_max_m` (5 m) unless the neighbouring rows are also supported there.
  - Mark row pieces with no candidate support as `row_interpolated` (QA priority 1).
  - Canopy extraction on such pieces requires `vine_score` or `corridor_veg_frac` evidence.

### S5 [MAJOR, route criterion (25%) at risk of scoring 0] Headlands are outside the walking domain
- Interrows stop at the shorter row's end, and headlands are neither interrow nor passage. Every entry into an interrow may therefore need an outside connector.
- Rough estimate: 300 interrow visits × 2 × 3 m ≈ 1.8 km on a route of about 20 km, i.e. about 9%, which is over the 2% elimination limit.
- Post only reports this (`domain_disconnected`, penalty ×10).
- **Fix (not covered by any design):**
  1. On Saturday at about 08:00, run `passable` + `route` on AnnSet(model) and **measure `outside_frac`** and the share of interrow ends within 0.5 m of `passages`.
  2. If it is over 0.5%, choose a strategy: enter each interrow only from an end that touches a passage (out-and-back dead-ends); drop targets that are only reachable through headlands (`reachable=false`); and send Slack Q1/Q4.
- This must be known before Sunday.

### S6 [MAJOR] Post-run clobbering (I9) and the POST_STAGES order (I10) break Sunday reproducibility and the cache
See I9 and I10 for the fixes.

### S7 [MAJOR] No `tile_valid` in Marcaj runs
- Post falls back to "full tiles" when `tile_valid` is missing. Nodata spans then count as gaps, which gives false GAP/END targets on edge tiles and breaks C§1.7 ("no target in nodata").
- **Fix:** make `tile_valid` static (C8). Post fails loudly if it is absent.

### S8 [MINOR] `canopy_on_non_vineyard_tile` is in the contract, but no design computes it
- **Fix:** perception's `assemble` flags tiles where canopies exist but the maximum `vine_score` or SNR is low.

---

## 4. Required by the docs, but not covered by any design

1. **[MAJOR] Root `README.md` has no owner.** The brief requires:
   - install and run steps from the tiles to both deliverables;
   - pinned dependencies and the weights link;
   - processing time for all 311 tiles plus the hardware (the table from `metrics/timings.json`);
   - a statement of paid APIs and LLMs used;
   - the UI link;
   - licences: CC BY 4.0 for Sireț3, ODbL for OSM, plus UAVVaste, SAM and OpenCLIP.

   ML covers only the third-party paragraph. Also add `LICENSE`/`CITATION`. Assign an owner on Saturday.
2. **[MAJOR] Reproducing `make final` requires the Marcaj export in the repo.** The Docker `CMD make final` and "run from supplied tiles to `route.geojson`" only work if the final Marcaj CVAT export is committed (for example `data/marcaj_final/*.zip`, a few MB) and pinned by `sha256` in the README. Also run `docker run --network none` (arch §4.15).
3. **[MAJOR] Early route-feasibility gate on the model AnnSet (S5).**
4. **[MAJOR] Integration ownership of shared files.** Four parallel agents pushing to `main` will conflict on `pyproject.toml`, `default.yaml`, `cli.py`, `registry.py`, `contracts/*` and `Makefile`. Give the foundation single-writer ownership of these; the others open small PRs or send patch fragments. Per-phase commit + push stays.
5. **[MINOR] Nothing produces `tile_status` for Marcaj imports** (contract §2.5), and nothing computes `canopy_on_non_vineyard_tile` (S8).
6. **[MINOR] Dependency pins are missing.**
   - Core: `lxml` (the foundation relies on it, and brew python3.14 pyexpat is broken).
   - Waste training: `scikit-learn`.
   - Extras: `transformers` and `open-clip-torch` in a `waste-ml` extra.
   - Route: `sknw` (numba 0.67 was checked by post as OK for numpy 2.5).
7. **[MINOR] The schedule doesn't line up.**
   - ML's overnight NN chain "starts ~02:00", but pseudo-labels need perception's full run (M2, 03:00–03:30). Realistic start is about 03:45 with auto-selected tiles.
   - Arch §4.1 says MPS training must not overlap CPU-pool runs, but that window overlaps perception's QA re-runs (M3, 03:30–06:00).
   - **Fix:** train either 03:45–05:15 with the QA re-runs at `--workers 4`, or 07:00–09:00 after the ZIPs exist, with promotion by re-export at about 10:00. Either way it stays off the 07:00 critical path.
8. **[MINOR] Web data contract with the frontend teammates.** Freeze the `manifest.json` schema and the ortho mode (XYZ or `ortho_index.json`, since gdal2tiles and tippecanoe are not installed) together with the `src/Web` team before WP6.

---

## Resolved ownership, to put in the final plan

| Area | Single owner |
|---|---|
| `contracts/*` (incl. `qa.py`, `ordering.py`), `geo/*`, `annset/{model,io}`, `pipeline/*`, `config/` root + `default.yaml`, `cli.py`, `cvat/*`, the `import_reference` / `export_cvat` / `eval` stages, `tile_prep` (calls perception `vegmask`) | foundation |
| `perception/*` (incl. `vegmask`), `annset/assemble.py`, `qa/*`, stages `rows_detect … assemble`, `interrow_bands`, `qa_previews` | perception |
| `nn/*` (incl. `fusion`, `probs`), `perception/waste/*`, stages `nn_infer` and `waste` (writes `layers/waste.parquet`) | ML |
| `annset/{gaps,import_checks,merge_rows,row_order,blocks,link_interrows,derive}`, `route/*`, `measure/*`, `web/*`, `cli_post.py`, post stages running in their own run directory | post |

### Critical Files for Implementation
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/pipeline/registry.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/config/model.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/geo/raster.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/annset/model.py
- /Users/maleticimiroslav/Vin Gigahack/src/AI/vineyard/annset/gaps.py